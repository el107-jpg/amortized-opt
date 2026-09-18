from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import N_DOF, PELVIS_IN_ENV
from amortized_opt.cost import StrikeContext
from amortized_opt.tokenizer import ContextInputs
from amortized_opt.voxel_pool import (
    BootstrapCandidate,
    ChartTemplate,
    ProbeBuffer,
    ProbeCandidateKey,
    VoxelKey,
    VoxelPool,
    ingest_certified_candidates,
    ordinary_bootstrap_candidate,
    ordinary_slice_batches,
    post_hit_bootstrap_candidate,
    rest_bootstrap_candidate,
)


def _key(*, chart: int = 9, q_joint0: int = 0) -> VoxelKey:
    q = [0] * N_DOF
    q[0] = q_joint0
    return VoxelKey(chart, 50, tuple(q), (1,) * N_DOF)


def _template_batch(chart: int = 9, bucket: int = 50) -> dict:
    modes = divmod(chart, 8)
    active = torch.tensor([[mode != 0 for mode in modes]])
    drums = torch.tensor([[max(mode - 1, 0) for mode in modes]])
    ctx = ContextInputs.zeros(1, num_drums=7, num_future=1)
    ctx.drum_valid[:] = True
    ctx.drum_pos[0, :, 0] = 0.5
    ctx.drum_pos[0, :, 1] = torch.linspace(0.3, -0.3, 7)
    ctx.drum_pos[0, :, 2] = 1.0
    ctx.drum_axis[..., 2] = 1.0
    ctx.drum_radius[:] = 0.1
    ctx.drum_height[:] = 0.1
    ctx.goal_valid = active
    ctx.goal_drum = drums
    for hand in range(2):
        if active[0, hand]:
            ctx.goal_pos[0, hand] = ctx.drum_pos[0, drums[0, hand]]
    ctx.goal_vel[..., 2] = -2.0 * active
    ctx.goal_tth[:] = torch.tensor([[0.3, 0.8]])
    ctx.future_valid[:] = True
    ctx.future_pos[:] = 9.0
    ctx.idle_cyl[:] = 1.0
    ctx.idle_valid = ~active

    f = torch.zeros(1, 2)
    f3 = torch.zeros(1, 2, 3)
    n_out = f3.clone()
    n_out[..., 2] = 1.0
    target = torch.where(active, drums, torch.full_like(drums, -1))
    strike = StrikeContext(
        hit_index=torch.tensor([[15, 40]]), active=active.float(),
        target_drum=target, start_drum=target.clone(),
        p_star=ctx.goal_pos - torch.as_tensor(PELVIS_IN_ENV),
        v_star=-2.0 * n_out * active.unsqueeze(-1), n_out=n_out,
        hit_radius=f + 0.08, disk_offset=f3.clone(),
        half_plane_dir=f3.clone(), has_half_plane=f.clone(),
        contact_sigma=f + 0.02, shaft_allowed=f.clone(),
        start_shaft_allowed=f.clone(), elev_required=f.clone(),
        drum_head_centers=ctx.drum_pos - torch.as_tensor(PELVIS_IN_ENV),
        drum_radii=ctx.drum_radius,
    )
    seed = torch.linspace(0.0, 0.3, bucket).view(1, bucket, 1).expand(1, bucket, N_DOF)
    return {
        "bucket": bucket, "q0": torch.zeros(1, N_DOF),
        "v0": torch.zeros(1, N_DOF), "seed": seed, "init": seed,
        "context": ctx, "strike": strike, "active": active,
        "hit": torch.tensor([[15 if active[0, 0] else -1,
                              40 if active[0, 1] else -1]]),
        "kind": torch.tensor([0]),
    }


def test_key_quantizes_20_mrad_cells_and_four_velocity_bins():
    q = np.zeros(N_DOF)
    q[0] = 0.0199
    q[1] = -0.0001
    qd = np.zeros(N_DOF)
    qd[:4] = [-1.0, -0.1, 0.1, 1.0]
    key = VoxelKey.from_state(chart_id=9, bucket=50, q0=q, qd0=qd)
    assert key.q0_cell[:2] == (0, -1)
    assert key.qd_bin[:4] == (0, 1, 2, 3)


def test_dual_hit_pairs_are_atomic_not_componentwise_minimized():
    pool = VoxelPool()
    key = _key()
    assert pool.ingest(key, hit_pair=(15, 40), source="bootstrap")
    assert not pool.ingest(key, hit_pair=(40, 15), source="mpc")
    cell = pool.cell(key)
    assert cell.hit_pairs == [(15, 40), (40, 15)]
    assert (15, 15) not in cell.hit_pairs
    assert cell.sources == {"bootstrap", "mpc"}


def test_hit_pair_must_match_idle_modes():
    both_idle = _key(chart=0)
    pool = VoxelPool()
    pool.ingest(both_idle, hit_pair=(-1, -1))
    with pytest.raises(ValueError, match="idle hand"):
        pool.ingest(both_idle, hit_pair=(5, -1))


def test_ingest_tracks_face_boundary():
    pool = VoxelPool()
    center = _key()
    pool.ingest(center, hit_pair=(10, 10))
    assert pool.is_boundary(center)
    neighbors = list(pool.face_neighbors(center))
    assert len(neighbors) == 2 * N_DOF
    for neighbor in neighbors:
        pool.ingest(neighbor, hit_pair=(10, 10))
    assert not pool.is_boundary(center)
    assert all(pool.is_boundary(neighbor) for neighbor in neighbors)


def test_uniform_materialization_stays_inside_cell():
    pool = VoxelPool(q_cell_rad=0.02)
    key = _key(q_joint0=-2)
    q = pool.sample_q0([key] * 64, rng=np.random.default_rng(3))
    assert np.all(q[:, 0] >= -0.04)
    assert np.all(q[:, 0] < -0.02)
    assert np.all(q[:, 1:] >= 0.0)
    assert np.all(q[:, 1:] < 0.02)


def test_position_materialization_intersects_hard_joint_bounds():
    pool = VoxelPool(q_cell_rad=0.02)
    key = _key(q_joint0=0)
    q = pool.sample_q0(
        [key] * 64, rng=np.random.default_rng(9),
        q_lo=np.full(N_DOF, 0.01), q_hi=np.full(N_DOF, 0.5),
    )
    assert np.all(q >= 0.01)
    assert np.all(q < 0.02)


def test_failed_probe_remains_unknown_and_success_promotes():
    pool = VoxelPool()
    probes = ProbeBuffer(capacity=2)
    key = _key()
    candidate = ProbeCandidateKey(key, (20, 25))
    probes.add(candidate, step=7)
    probes.record_failure(candidate)
    assert candidate in probes
    assert key not in pool
    assert probes.records()[0].attempts == 1

    assert probes.promote(
        candidate, pool, source="mpc", hard_valid=True,
    )
    assert candidate not in probes
    assert key in pool


def test_probe_capacity_evicts_oldest_unknown():
    probes = ProbeBuffer(capacity=2)
    keys = [_key(q_joint0=i) for i in range(3)]
    candidates = [ProbeCandidateKey(key, (15, 40)) for key in keys]
    for step, candidate in enumerate(candidates):
        probes.add(candidate, step=step)
    assert candidates[0] not in probes
    assert candidates[1] in probes and candidates[2] in probes


def test_probe_buffer_distinguishes_atomic_timings_for_same_voxel():
    probes = ProbeBuffer(capacity=8)
    key = _key()
    candidates = [
        ProbeCandidateKey(key, pair)
        for pair in ((14, 39), (16, 41), (17, 42), (18, 43), (19, 44))
    ]
    assert all(probes.add(candidate, step=0) for candidate in candidates)
    assert len(probes) == 5


def test_successful_timing_probe_augments_existing_voxel_atomically():
    pool = VoxelPool()
    key = _key()
    pool.ingest(key, hit_pair=(15, 40))
    probes = ProbeBuffer(capacity=8)
    candidate = ProbeCandidateKey(key, (16, 41))
    probes.add(candidate, step=3)
    assert not probes.promote(candidate, pool, source="mpc", hard_valid=True)
    assert candidate not in probes
    assert pool.cell(key).hit_pairs == [(15, 40), (16, 41)]


def test_velocity_materialization_samples_inside_all_four_bins():
    pool = VoxelPool()
    keys = [
        VoxelKey(9, 50, (0,) * N_DOF, (velocity_bin,) * N_DOF)
        for velocity_bin in range(4)
    ]
    qd = pool.sample_qd0(keys, rng=np.random.default_rng(4), qd_lo=-2.0, qd_hi=2.0)
    bounds = [(-2.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 2.0)]
    for row, (lo, hi) in enumerate(bounds):
        assert np.all(qd[row] >= lo)
        assert np.all(qd[row] < hi)


def test_materialize_mirrors_phase2_schema_and_clears_future_context():
    source = _template_batch(chart=8)
    # Legacy source taxonomy must not survive in the unified voxel pool.
    source["kind"].fill_(2)
    template = ChartTemplate.from_batch(source, 0, chart_id=8)
    pool = VoxelPool()
    pool.register_template(template)
    key = VoxelKey(8, 50, (0,) * N_DOF, (2,) * N_DOF)
    pool.ingest(key, hit_pair=(15, -1), finishing_hand=0)

    batch, pairs = pool.materialize_batch(
        [key] * 16, rng=np.random.default_rng(8),
        strike_speed_min=1.25, strike_speed_max=1.75,
    )
    assert pairs == [(15, -1)] * 16
    assert batch["q0"].shape == (16, N_DOF)
    assert batch["qd0"].shape == batch["v0"].shape == (16, N_DOF)
    assert batch["init"].shape == batch["seed"].shape == (16, 50, N_DOF)
    assert torch.all((batch["q0"] >= 0.0) & (batch["q0"] < 0.02))
    assert torch.all((batch["qd0"] >= 0.0) & (batch["qd0"] < 0.5))
    assert torch.equal(batch["active"], torch.tensor([[True, False]]).expand(16, 2))
    assert torch.allclose(batch["context"].goal_tth[:, 0], torch.full((16,), 0.3))
    assert torch.count_nonzero(batch["context"].goal_tth[:, 1]) == 0
    assert not bool(batch["context"].future_valid.any())
    assert torch.equal(batch["context"].idle_valid, ~batch["context"].goal_valid)
    assert torch.count_nonzero(batch["context"].goal_vel[:, 1]) == 0
    assert torch.allclose(batch["context"].goal_vel, batch["strike"].v_star)
    speed = torch.linalg.vector_norm(batch["strike"].v_star[:, 0], dim=-1)
    assert torch.all((speed >= 1.25) & (speed <= 1.75))
    assert torch.equal(batch["state"][:, :N_DOF], batch["q0"])
    assert torch.equal(batch["state"][:, N_DOF:], batch["qd0"])
    assert torch.count_nonzero(batch["kind"]) == 0
    assert torch.isfinite(batch["seed"]).all()


def test_materialize_uses_requested_complete_witness_pair_for_all_timing_fields():
    source = _template_batch(chart=9)
    template = ChartTemplate.from_batch(source, 0, chart_id=9)
    pool = VoxelPool()
    pool.register_template(template)
    key = _key()
    pool.ingest(key, hit_pair=(15, 40))
    pool.ingest(key, hit_pair=(40, 15))
    batch, pairs = pool.materialize_batch(
        [key], hit_pairs=[(40, 15)], rng=np.random.default_rng(12),
    )
    assert pairs == [(40, 15)]
    assert batch["hit"].tolist() == [[40, 15]]
    torch.testing.assert_close(
        batch["context"].goal_tth, torch.tensor([[0.8, 0.3]]),
    )
    torch.testing.assert_close(batch["masks"].hit_tth, torch.tensor([[0.8, 0.3]]))


def test_post_hit_anchor_and_tightening_change_only_finishing_hand():
    source = _template_batch(chart=9)
    template = ChartTemplate.from_batch(source, 0, chart_id=9)
    candidate = post_hit_bootstrap_candidate(
        q0=np.zeros(N_DOF), qd0=np.zeros(N_DOF), template=template,
        hit_pair=(12, 17), finishing_hand=0,
        witness_trajectory=np.zeros((50, N_DOF)), anchor_tth_s=1.0,
    )
    assert candidate.hit_pair == (50, 17)
    pool = VoxelPool()
    assert ingest_certified_candidates(pool, [candidate], [True]) == 1
    assert pool.tightened_hit_pair(candidate.key, candidate.hit_pair) == (49, 17)
    # A proposal alone is not evidence and cannot mutate feasibility.
    assert pool.cell(candidate.key).hit_pairs == [(50, 17)]


def test_expand_at_returns_unknown_uncertified_atomic_timing_proposals():
    pool = VoxelPool()
    key = _key()
    pool.ingest(key, hit_pair=(15, 40), finishing_hand=1)
    proposals = pool.expand_at(key)
    spatial = [proposal for proposal in proposals if proposal.key != key]
    timing = [proposal for proposal in proposals if proposal.key == key]
    assert len(spatial) == 2 * N_DOF + 4 * (N_DOF * (N_DOF - 1) // 2)
    assert len(timing) == 5
    assert all(p.parent_hit_pair == (15, 40) for p in proposals)
    assert {p.hit_pair for p in spatial} == {(15, 40)}
    assert {p.hit_pair for p in timing} == {
        (15, 39), (15, 41), (15, 42), (15, 43), (15, 44),
    }
    assert all(p.finishing_hand == 1 for p in proposals)
    assert all(p.key not in pool for p in spatial)
    assert pool.cell(key).hit_pairs == [(15, 40)]


def test_pairwise_diagonal_neighbors_change_exactly_two_joints():
    pool = VoxelPool()
    key = _key()
    faces = set(pool.face_neighbors(key))
    neighbors = set(pool.positional_neighbors(key))
    diagonals = neighbors - faces
    assert len(diagonals) == 4 * (N_DOF * (N_DOF - 1) // 2)
    for neighbor in diagonals:
        delta = np.asarray(neighbor.q0_cell) - np.asarray(key.q0_cell)
        assert np.count_nonzero(delta) == 2
        assert set(np.abs(delta[delta != 0])) == {1}


def test_ordinary_dual_timing_stencil_shifts_complete_pair_together():
    pool = VoxelPool()
    key = _key()
    pool.ingest(key, hit_pair=(15, 40))
    assert pool.timing_probe_pairs(key, (15, 40)) == (
        (14, 39), (16, 41), (17, 42), (18, 43), (19, 44),
    )


def test_certified_longer_timing_shell_is_retained_but_bounded():
    pool = VoxelPool()
    key = _key()
    for right_hit in range(40, 46):
        pool.ingest(key, hit_pair=(15, right_hit), finishing_hand=1)
    assert pool.cell(key).hit_pairs == [
        (15, 40), (15, 41), (15, 42), (15, 43), (15, 44),
    ]
    timing = [proposal for proposal in pool.expand_at(key) if proposal.key == key]
    assert {proposal.hit_pair for proposal in timing} == {(15, 39)}


def test_uncertified_candidate_remains_unknown_and_cannot_replace_positive_cell():
    source = _template_batch(chart=9)
    template = ChartTemplate.from_batch(source, 0, chart_id=9)
    key = _key()
    candidate = BootstrapCandidate(
        key=key, hit_pair=(15, 40), template=template,
        witness_trajectory=np.zeros((50, N_DOF)),
    )
    pool = VoxelPool()
    assert ingest_certified_candidates(pool, [candidate], [False]) == 0
    assert key not in pool
    ingest_certified_candidates(pool, [candidate], [True])
    failed_later = BootstrapCandidate(
        key=key, hit_pair=(40, 15), template=template,
        witness_trajectory=np.zeros((50, N_DOF)),
    )
    ingest_certified_candidates(pool, [failed_later], [False])
    assert pool.cell(key).hit_pairs == [(15, 40)]


def test_certified_ingest_requires_boolean_gate_not_truthy_scores():
    source = _template_batch(chart=9)
    template = ChartTemplate.from_batch(source, 0, chart_id=9)
    candidate = ordinary_bootstrap_candidate(
        q0=np.zeros(N_DOF), qd0=np.zeros(N_DOF), template=template,
        hit_pair=(15, 40), witness_trajectory=np.zeros((50, N_DOF)),
    )
    with pytest.raises(TypeError, match="boolean hard-gate"):
        ingest_certified_candidates(VoxelPool(), [candidate], [1.0])
    with pytest.raises(TypeError, match="boolean hard-gate"):
        ingest_certified_candidates(VoxelPool(), [candidate], [np.nan])


def test_rest_helper_requires_idle_chart_and_long_gap():
    source = _template_batch(chart=8)
    template = ChartTemplate.from_batch(source, 0, chart_id=8)
    with pytest.raises(ValueError, match="gap_s"):
        rest_bootstrap_candidate(
            q0=np.zeros(N_DOF), qd0=np.zeros(N_DOF), template=template,
            hit_pair=(15, -1), witness_trajectory=np.zeros((50, N_DOF)),
            gap_s=0.5, min_gap_s=1.0,
        )
    candidate = rest_bootstrap_candidate(
        q0=np.zeros(N_DOF), qd0=np.zeros(N_DOF), template=template,
        hit_pair=(15, -1), witness_trajectory=np.zeros((50, N_DOF)),
        gap_s=1.2,
    )
    assert candidate.key.chart_id == 8


def test_cache_round_trip_and_gate_fingerprint_rejection(tmp_path):
    source = _template_batch(chart=9)
    template = ChartTemplate.from_batch(source, 0, chart_id=9)
    pool = VoxelPool()
    pool.register_template(template)
    key = _key()
    pool.ingest(
        key, hit_pair=(15, 40), source="bootstrap", feas_margin=0.2,
        finishing_hand=1,
    )
    path = tmp_path / "voxel_pool.npz"
    gate = {"terminal_eps": 1e-6, "collision_clearance": 0.0, "robot": "g1"}
    pool.save_npz(path, gate_definition=gate)
    restored = VoxelPool.load_npz(path, expected_gate_definition=gate)
    assert restored.keys() == pool.keys()
    assert restored.cell(key).hit_pairs == [(15, 40)]
    assert restored.cell(key).finishing_hand_for((15, 40)) == 1
    assert restored.template(9, 50).context["goal_pos"].shape == (2, 3)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        VoxelPool.load_npz(
            path, expected_gate_definition={**gate, "collision_clearance": 0.01},
        )


def test_probe_buffer_round_trip_preserves_unknown_attempts():
    probes = ProbeBuffer(capacity=3)
    key = _key()
    candidate = ProbeCandidateKey(key, (15, 40))
    probes.add(candidate, step=11)
    probes.record_failure(candidate)
    restored = ProbeBuffer.load_state_dict(probes.state_dict())
    assert candidate in restored
    assert restored.records()[0].hit_pair == (15, 40)
    assert restored.records()[0].created_step == 11
    assert restored.records()[0].attempts == 1


def test_probe_cache_is_gate_fingerprinted(tmp_path):
    probes = ProbeBuffer(capacity=3)
    probes.add(ProbeCandidateKey(_key(), (15, 40)), step=4)
    gate = {"robot": "g1", "collision": "v1"}
    path = tmp_path / "probes.npz"
    probes.save_npz(path, gate_definition=gate)
    restored = ProbeBuffer.load_npz(path, expected_gate_definition=gate)
    assert restored.records() == probes.records()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ProbeBuffer.load_npz(
            path, expected_gate_definition={"robot": "g1", "collision": "v2"},
        )


def test_ordinary_slice_helper_never_ingests_and_rebuilds_collocations():
    horizon = 5
    cols = {
        "q0": torch.zeros(1, N_DOF),
        "qd0": torch.zeros(1, N_DOF),
        "init": torch.arange(1, horizon + 1).float().view(1, horizon, 1).expand(1, horizon, N_DOF),
        "active": torch.tensor([[True, False]]),
        "hit": torch.tensor([[5, -1]]),
        "goal_tth": torch.tensor([[0.1, 0.0]]),
        "yaw_now": torch.zeros(1),
    }
    sliced = ordinary_slice_batches(cols, stride=2, buckets=(5,))
    assert [bucket for bucket, _ in sliced] == [5, 5, 5]
    assert torch.equal(sliced[1][1]["hit"], torch.tensor([[3, -1]]))
    assert torch.equal(sliced[2][1]["hit"], torch.tensor([[1, -1]]))


def test_materialize_exact_state_override_must_match_key():
    source = _template_batch(chart=9)
    template = ChartTemplate.from_batch(source, 0, chart_id=9)
    pool = VoxelPool()
    pool.register_template(template)
    q0 = np.full((1, N_DOF), 0.011, dtype=np.float32)
    qd0 = np.full((1, N_DOF), 0.1, dtype=np.float32)
    key = VoxelKey.from_state(chart_id=9, bucket=50, q0=q0[0], qd0=qd0[0])
    batch, _ = pool.materialize_batch(
        [key], rng=np.random.default_rng(0), hit_pairs=[(15, 40)],
        q0_values=q0, qd0_values=qd0, probe=True,
    )
    torch.testing.assert_close(batch["q0"], torch.from_numpy(q0))
    torch.testing.assert_close(batch["qd0"], torch.from_numpy(qd0))
    with pytest.raises(ValueError, match="does not match VoxelKey"):
        pool.materialize_batch(
            [key], rng=np.random.default_rng(0), hit_pairs=[(15, 40)],
            q0_values=q0 + 0.03, qd0_values=qd0, probe=True,
        )
