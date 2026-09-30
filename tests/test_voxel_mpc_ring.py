from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import amortized_opt.train_cost as train_cost
from amortized_opt.train_cost import (
    Phase2Config,
    _VoxelMPCParentRing,
    _VoxelTrainingState,
    _index_batch,
    _index_batch_weights,
    _mpc_aux_rows,
    effective_hits,
    per_element_lengths,
)
from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.tokenizer import build_waypoint_masks
from amortized_opt.voxel_pool import (
    ExpansionProposal,
    ProbeBuffer,
    ProbeCandidateKey,
    VoxelKey,
)


@dataclass
class _Masks:
    elem_mask: torch.Tensor


@dataclass
class _Context:
    token: torch.Tensor
    optional: torch.Tensor | None = None


def _batch(n=3, horizon=4, dof=2):
    q0 = torch.arange(n * dof, dtype=torch.float32).reshape(n, dof)
    return {
        "bucket": horizon,
        "q0": q0,
        "v0": -q0,
        "state": torch.cat([q0, -q0], dim=-1),
        "init": torch.zeros(n, horizon, dof),
        "seed": torch.zeros(n, horizon, dof),
        "masks": _Masks(torch.ones(n, horizon, dof)),
        "context": _Context(torch.arange(n, dtype=torch.float32).unsqueeze(1)),
        "mpc_init": torch.zeros(n, dtype=torch.bool),
        "al_ok": torch.tensor([True, False, True]),
    }


def test_probe_retry_choice_visits_least_attempted_candidates_first():
    state = object.__new__(_VoxelTrainingState)
    records = [
        SimpleNamespace(candidate=f"candidate-{i}", attempts=attempts)
        for i, attempts in enumerate((3, 0, 2, 0, 1))
    ]
    chosen = state._probe_retry_choice(
        records, 3, np.random.default_rng(7),
    )
    attempts_by_candidate = {
        record.candidate: record.attempts for record in records
    }
    assert sorted(attempts_by_candidate[candidate] for candidate in chosen) == [0, 0, 1]
    assert len(set(chosen)) == len(chosen)


def test_parent_ring_shifts_decode_and_preserves_problem_state():
    batch = _batch()
    traj = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)
    ring = _VoxelMPCParentRing(device="cpu")

    assert ring.refresh(
        4, batch, traj, hard_ok=torch.tensor([True, False, True]),
    ) == 2
    assert ring.size(4) == 2
    stored = ring.data[4]
    expected = torch.cat([traj[[0, 2], 1:], traj[[0, 2], -1:]], dim=1)
    torch.testing.assert_close(stored["seed"], expected)
    torch.testing.assert_close(stored["init"], expected)
    torch.testing.assert_close(stored["q0"], batch["q0"][[0, 2]])
    torch.testing.assert_close(stored["state"], batch["state"][[0, 2]])
    torch.testing.assert_close(stored["context"].token, batch["context"].token[[0, 2]])
    assert stored["mpc_init"].all()
    assert stored["al_ok"].all()


def test_parent_ring_replaces_generation_and_round_trips():
    ring = _VoxelMPCParentRing(device="cpu")
    ring.refresh(
        4, _batch(), torch.ones(3, 4, 2),
        hard_ok=torch.ones(3, dtype=torch.bool),
    )
    one = _batch(n=1)
    one["al_ok"] = torch.ones(1, dtype=torch.bool)
    ring.refresh(
        4, one, torch.full((1, 4, 2), 7.0),
        hard_ok=torch.ones(1, dtype=torch.bool),
    )
    assert ring.size(4) == 1
    assert ring.generation[4] == 2

    restored = _VoxelMPCParentRing(device="cpu")
    restored.load_state_dict(ring.state_dict())
    assert restored.distribution() == {4: 1}
    assert restored.generation == {4: 2}
    torch.testing.assert_close(restored.data[4]["seed"], ring.data[4]["seed"])

    sample = restored.batch(4, torch.tensor([0, 0]))
    sample["q0"].zero_()
    assert not torch.equal(sample["q0"], restored.data[4]["q0"].expand_as(sample["q0"]))


def test_parent_ring_rejects_uncertified_ingress_and_drops_stale_generation():
    ring = _VoxelMPCParentRing(device="cpu")
    batch = _batch()
    traj = torch.ones(3, 4, 2)
    assert ring.refresh(
        4, batch, traj, hard_ok=torch.ones(3, dtype=torch.bool),
    ) == 3
    assert ring.refresh(
        4, batch, traj, hard_ok=torch.tensor([False, False, False]),
    ) == 0
    assert ring.size(4) == 0
    assert ring.generation[4] == 2

    with pytest.raises(ValueError, match="hard_ok"):
        ring.refresh(4, batch, traj, hard_ok=torch.ones(2, dtype=torch.bool))


def test_parent_ring_priming_hard_checks_current_decode(monkeypatch):
    batch = _batch(n=2)
    batch.update({
        "hit": torch.tensor([[2, 3], [2, 3]]),
        "active": torch.ones(2, 2, dtype=torch.bool),
        "al_ok": torch.ones(2, dtype=torch.bool),
        "strike": SimpleNamespace(hit_index=torch.tensor([[2, 3], [2, 3]])),
    })
    # dataclasses.replace is used on the strike context in production.
    @dataclass
    class _Strike:
        hit_index: torch.Tensor
    batch["strike"] = _Strike(batch["hit"])

    state = object.__new__(_VoxelTrainingState)
    state.source = SimpleNamespace(buckets=(4,))
    state.cfg = SimpleNamespace(
        voxel_frac_mpc=0.5,
        voxel_frac_pool=0.3,
        voxel_frac_boundary=0.15,
        voxel_frac_probe=0.05,
        batch_for=lambda _bucket: 4,
    )
    state.pool = SimpleNamespace(keys=lambda **_kwargs: ["known"])
    state._materialize = lambda *_args, **_kwargs: (batch, None)
    monkeypatch.setattr(train_cost, "encode", lambda *_args: None)
    monkeypatch.setattr(train_cost, "_voxel_current_goal_only", lambda value: value)

    class _Model:
        def __call__(self, init, *_args):
            return SimpleNamespace(traj=torch.ones_like(init))

    class _Cost:
        def forward(self, q0, *_args, **_kwargs):
            n = q0.shape[0]
            zeros = torch.zeros(n)
            # Row 1 misses the terminal gate and must not seed the ring.
            return SimpleNamespace(diagnostics={
                "terminal_region_dist": torch.tensor([0.0, 0.1]),
                "collision_min_sdf": zeros,
                "joint_box_excess_max": zeros,
                "torque_excess_max": zeros,
            })

    ring = _VoxelMPCParentRing(device="cpu")
    primed = state.prime_mpc_ring(
        ring, _Model(), {4: _Cost()}, {},
        generator=torch.Generator().manual_seed(0),
        rng=np.random.default_rng(0), yaw_i=0,
    )
    assert primed == {4: 1}
    assert ring.size(4) == 1


def test_template_row_alternate_pair_retimes_to_original_task_terminal():
    """A new pair must not turn an old in-flight knot into the goal pose."""
    from tests.test_voxel_pool import _template_batch

    source_batch = _template_batch(chart=9, bucket=50)
    source_batch["state"] = torch.cat(
        [source_batch["q0"], source_batch["v0"]], dim=-1,
    )
    source_batch["masks"] = build_waypoint_masks(
        torch.tensor([0.3]), torch.tensor([0.8]), max_horizon=50,
        active_left=torch.tensor([True]), active_right=torch.tensor([True]),
        joint_names=G1_UPPER_BODY_JOINTS,
    )

    class _Source:
        device = torch.device("cpu")
        dtype = torch.float32
        q_lo = torch.full((N_DOF,), -10.0)
        q_hi = torch.full((N_DOF,), 10.0)
        cfg = SimpleNamespace(dt=0.02, joint_names=G1_UPPER_BODY_JOINTS)

        def batch(self, bucket, idx, **_kwargs):
            assert bucket == 50
            return _index_batch(source_batch, idx)

        def masks_for(self, hit, active, bucket):
            tth = hit.clamp_min(0).float() * self.cfg.dt
            return build_waypoint_masks(
                tth[:, 0], tth[:, 1], max_horizon=bucket, dt=self.cfg.dt,
                active_left=active[:, 0], active_right=active[:, 1],
                joint_names=self.cfg.joint_names,
            )

    state = _VoxelTrainingState(_Source(), Phase2Config(voxel_pool=True))
    key = VoxelKey(9, 50, (0,) * N_DOF, (2,) * N_DOF)
    state.pool.ingest(key, hit_pair=(40, 15))
    state.templates[key] = [(0, (40, 15))]
    materialized, _ = state._materialize(
        50, [key], probe=False, generator=torch.Generator().manual_seed(1),
        rng=np.random.default_rng(1), yaw_i=0,
    )

    source_eff = effective_hits(source_batch["hit"], source_batch["active"], 50)
    source_lengths = per_element_lengths(source_eff, G1_UPPER_BODY_JOINTS)
    old_terminal = source_batch["seed"].gather(
        1, (source_lengths - 1).unsqueeze(1),
    ).squeeze(1)
    new_eff = effective_hits(materialized["hit"], materialized["active"], 50)
    new_lengths = per_element_lengths(new_eff, G1_UPPER_BODY_JOINTS)
    new_terminal = materialized["seed"].gather(
        1, (new_lengths - 1).unsqueeze(1),
    ).squeeze(1)
    torch.testing.assert_close(new_terminal, old_terminal)
    assert materialized["hit"].tolist() == [[40, 15]]
    torch.testing.assert_close(
        materialized["context"].goal_tth, torch.tensor([[0.8, 0.3]]),
    )
    assert torch.equal(materialized["qd0"], materialized["v0"])
    assert torch.equal(materialized["state"][:, N_DOF:], materialized["qd0"])
    assert bool(((materialized["qd0"] >= 0.0) & (materialized["qd0"] < 0.5)).all())
    torch.testing.assert_close(
        materialized["context"].goal_vel, materialized["strike"].v_star,
    )
    speed = torch.linalg.vector_norm(materialized["strike"].v_star, dim=-1)
    assert bool(((speed >= 1.0) & (speed <= 3.0)).all())


def test_aux_rows_are_mpc_only_and_batch_weights_follow_them():
    marker = torch.tensor([False, True, False, True, True, False])
    rows = _mpc_aux_rows(marker, 0.5)
    assert rows.tolist() == [1, 3, 4]
    weights = {
        "terminal": torch.arange(6, dtype=torch.float32),
        "limit_alpha": torch.arange(6, dtype=torch.float32),
        "smooth": torch.tensor(2.0),
    }
    out = _index_batch_weights(weights, rows, batch_size=6)
    assert out["terminal"].tolist() == [1.0, 3.0, 4.0]
    assert out["limit_alpha"] is weights["limit_alpha"]
    assert out["smooth"] is weights["smooth"]


def test_voxel_mode_rejects_unmasked_gradient_surgery():
    try:
        Phase2Config(voxel_pool=True, grad_balance="al_pcgrad")
    except ValueError as exc:
        assert "probe rows" in str(exc)
    else:
        raise AssertionError("voxel gradient surgery must fail closed")


def test_voxel_mode_rejects_invalid_materialized_strike_speed_range():
    with pytest.raises(ValueError, match="strike-speed"):
        Phase2Config(
            voxel_pool=True,
            voxel_strike_speed_min=3.0,
            voxel_strike_speed_max=1.0,
        )


def test_new_probe_does_not_report_an_existing_failed_probe_as_fresh():
    parent = VoxelKey(9, 50, (0,) * N_DOF, (1,) * N_DOF)
    child = VoxelKey(9, 50, (1,) + (0,) * (N_DOF - 1), (1,) * N_DOF)
    proposal = ExpansionProposal(
        parent=parent, key=child, parent_hit_pair=(15, 40),
        hit_pair=(15, 39), finishing_hand=1,
    )
    state = object.__new__(_VoxelTrainingState)
    state.pool = SimpleNamespace(
        keys=lambda **_kwargs: (parent,),
        expand_at=lambda _parent, **_kwargs: (proposal,),
    )
    state.templates = {parent: [(7, (15, 40))]}
    state.probes = ProbeBuffer(capacity=4)
    candidate = ProbeCandidateKey(child, (15, 39))
    state.probes.add(candidate, step=0)
    state.probe_templates = {candidate: (7, (15, 39))}
    state.probe_finishing_hands = {candidate: 1}

    assert state._new_probe(50, np.random.default_rng(1), step=100) is None
    assert len(state.probes) == 1


def test_new_probe_skips_position_cells_outside_joint_box():
    parent = VoxelKey(9, 50, (0,) * N_DOF, (1,) * N_DOF)
    child = VoxelKey(9, 50, (1,) + (0,) * (N_DOF - 1), (1,) * N_DOF)
    proposal = ExpansionProposal(
        parent=parent, key=child, parent_hit_pair=(15, 40),
        hit_pair=(15, 40), finishing_hand=1,
    )
    state = object.__new__(_VoxelTrainingState)
    from amortized_opt.voxel_pool import VoxelPool
    state.pool = VoxelPool(q_cell_rad=0.02)
    state.pool.ingest(parent, hit_pair=(15, 40), finishing_hand=1)
    state.pool.expand_at = lambda _parent, **_kwargs: (proposal,)
    state.source = SimpleNamespace(
        q_lo=torch.full((N_DOF,), -0.5),
        q_hi=torch.tensor([0.0] + [0.5] * (N_DOF - 1)),
    )
    state.templates = {parent: [(7, (15, 40))]}
    state.probes = ProbeBuffer(capacity=4)
    state.probe_templates = {}
    state.probe_finishing_hands = {}

    assert state._new_probe(50, np.random.default_rng(1), step=100) is None
    assert len(state.probes) == 0
