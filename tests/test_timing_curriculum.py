import pytest
import torch
from types import SimpleNamespace
from dataclasses import dataclass

from amortized_opt.model import ContextEncoding
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
from amortized_opt.train_cost import (
    Phase2Config,
    _expand_timing_base_tasks,
    _bellman_parent_ingress_ok,
    _suffix_rows,
    _timing_assist_batch,
    _timing_mpc_parent_ingress_ok,
    _timing_primary_layout,
    effective_hits,
    per_element_lengths,
    retime_batch_timing,
    vel_ramp_seed,
)
from amortized_opt.violation_critic import (
    ViolationCriticConfig,
    ViolationCriticTrainer,
    critic_feature_dim,
)
from amortized_opt.timing_curriculum import (
    TimingPair,
    anchor_connected_component,
    directional_pairs,
    first_exit_indices,
    interval_boundary_loss,
    pareto_frontier_mask,
    sample_directions,
)


def test_timing_pair_is_atomic_and_idle_hand_uses_sentinel():
    assert TimingPair(7, 9).validate((True, True), 10).as_tuple() == (7, 9)
    assert TimingPair(7, -1).validate((True, False), 10).as_tuple() == (7, -1)
    with pytest.raises(ValueError, match="inactive right"):
        TimingPair(7, 9).validate((True, False), 10)
    with pytest.raises(ValueError, match="active left"):
        TimingPair(0, -1).validate((True, False), 10)


def test_primary_layout_groups_timing_tables_by_physical_task():
    cfg = ViolationCriticConfig()
    assert _timing_primary_layout(512, cfg) == (40, 40, 40, 160, 272)
    assert _timing_primary_layout(64, cfg) == (5, 5, 5, 20, 34)

    batch = {
        "q0": torch.tensor([[1.0], [2.0]]),
        "v0": torch.tensor([[3.0], [4.0]]),
    }
    expanded = _expand_timing_base_tasks(
        batch, anchor_per_task=1, boundary_per_task=1, interior_per_task=4,
    )
    assert expanded["q0"].shape == (12, 1)
    torch.testing.assert_close(
        expanded["q0"][:, 0],
        torch.tensor([1.0] * 6 + [2.0] * 6),
    )
    assert expanded["timing_primary_kind"].tolist() == (
        [0, 2, 3, 3, 3, 3] * 2
    )
    assert expanded["timing_primary_group"].tolist() == (
        [0] * 6 + [1] * 6
    )


def test_bellman_ingress_uses_half_former_mpc_physical_limits():
    cfg = ViolationCriticConfig()
    slack = torch.sinh(torch.tensor(cfg.acceptable_normalized_violation))
    joint_max = 0.5 * (cfg.joint_limit + float(slack) * cfg.joint_scale)
    torque_max = 0.5 * (cfg.torque_limit + float(slack) * cfg.torque_scale)
    gate = torch.zeros(5, 8)
    gate[:, 4:8] = torch.tensor(
        [0.020, -0.005, joint_max, torque_max]
    )
    gate[1, 4] = 0.0201
    gate[2, 5] = -0.0051
    gate[3, 6] = joint_max + 1.0e-4
    gate[4, 7] = torque_max + 1.0e-3

    assert _bellman_parent_ingress_ok(gate, cfg).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_critic_mpc_parent_ingress_accepts_every_actor_row():
    gate = torch.tensor([
        [0.0] * 8,
        [0.0, 0.0, 0.0, 0.0, 10.0, -2.0, 3.0, 1_000.0],
    ])
    assert _timing_mpc_parent_ingress_ok(gate).tolist() == [True, True]


def test_suffix_rows_include_every_primary_stratum_proportionally():
    batch = {
        "q0": torch.zeros(20, 1),
        "timing_primary_kind": torch.tensor([0] * 8 + [1] * 6 + [2] * 2 + [3] * 4),
    }
    rows = _suffix_rows(batch, 0.5)
    assert rows.numel() == 10
    selected = batch["timing_primary_kind"].index_select(0, rows)
    assert set(selected.tolist()) == {0, 1, 2, 3}
    assert [(selected == kind).sum().item() for kind in range(4)] == [4, 3, 1, 2]


def test_directional_pairs_keep_complete_pairs_and_one_hand_special_case():
    active = torch.tensor([[True, True], [True, False], [False, True]])
    directions = torch.tensor([[1.0, 0.5], [1.0, 0.0], [0.0, 1.0]])
    pairs, alpha = directional_pairs(active, directions, horizon=10, step=3)
    assert pairs.shape == (3, len(alpha), 2)
    assert pairs[:, 0].tolist() == [[10, 10], [10, -1], [-1, 10]]
    assert pairs[0, -1].tolist() == [1, 6]
    assert torch.all(pairs[1, :, 1] == -1)
    assert torch.all(pairs[2, :, 0] == -1)


def test_sampled_directions_cover_axes_balance_and_idle():
    active = torch.tensor([[True, True]] * 100 + [[True, False], [False, True], [False, False]])
    direction = sample_directions(
        active, generator=torch.Generator().manual_seed(3), axis_probability=0.2,
    )
    assert torch.any(direction[:100, 1] == 0.0)
    assert torch.any(direction[:100, 0] == 0.0)
    assert torch.any((direction[:100] > 0.0).all(dim=1))
    torch.testing.assert_close(direction[-3], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(direction[-2], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(direction[-1], torch.tensor([0.0, 0.0]))


def test_first_exit_uses_anchor_connected_failure_not_later_recovery():
    manageable = torch.tensor(
        [
            [True, True, False, True, True],
            [True, True, True, True, True],
            [False, True, True, True, True],
        ]
    )
    selected, last_pass, first_fail = first_exit_indices(manageable)
    assert selected.tolist() == [1, 4, 0]
    assert last_pass.tolist() == [1, 4, 0]
    assert first_fail.tolist() == [2, 5, 0]
    outside, _, _ = first_exit_indices(manageable, collar=1)
    assert outside.tolist() == [2, 4, 0]


def test_connected_component_drops_disconnected_false_positive_island():
    feasible = torch.tensor(
        [
            [True, False, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ]
    )
    component = anchor_connected_component(feasible, (2, 3))
    assert not bool(component[0, 0])
    assert int(component.sum()) == 4
    assert not bool(anchor_connected_component(feasible, (0, 1)).any())


def test_pareto_frontier_handles_nonrectangular_joint_feasibility():
    # Individually short values exist in the feasible set, but (0, 0) is not
    # feasible: the hands' times cannot be combined independently.
    feasible = torch.tensor(
        [
            [False, False, True, True],
            [False, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    frontier = pareto_frontier_mask(feasible)
    expected = torch.tensor(
        [
            [False, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )
    assert torch.equal(frontier, expected)


def test_interval_boundary_loss_accepts_any_point_inside_exact_bracket():
    lo = torch.tensor([2.0, 5.0])
    hi = torch.tensor([4.0, 7.0])
    assert interval_boundary_loss(torch.tensor([3.0, 6.0]), lo, hi) == 0.0
    assert interval_boundary_loss(torch.tensor([1.0, 9.0]), lo, hi) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="must not exceed"):
        interval_boundary_loss(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0]))


def test_retime_batch_rebuilds_every_timing_field_and_retains_terminal():
    names = (
        "waist_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    )
    dt, horizon = 0.02, 6

    class Pool:
        cfg = SimpleNamespace(dt=dt, joint_names=names)

        @staticmethod
        def masks_for(hit, active, h):
            t = hit.clamp_min(0).float() * dt
            return build_waypoint_masks(
                t[:, 0], t[:, 1], max_horizon=h, dt=dt,
                active_left=active[:, 0], active_right=active[:, 1],
                joint_names=names,
            )

    active = torch.tensor([[True, True]])
    old_hit = torch.tensor([[4, 5]])
    q0 = torch.zeros(1, 3)
    qd0 = torch.zeros_like(q0)
    terminal = torch.tensor([[0.3, 0.5, -0.4]])
    old_eff = effective_hits(old_hit, active, horizon)
    old_len = per_element_lengths(old_eff, names)
    seed = vel_ramp_seed(q0, qd0, terminal, old_len, horizon, dt)
    context = ContextInputs.zeros(1, num_drums=1, num_future=0)
    batch = {
        "bucket": horizon,
        "q0": q0,
        "v0": qd0,
        "state": torch.cat([q0, qd0], dim=-1),
        "seed": seed,
        "init": seed.clone(),
        "hit": old_hit,
        "active": active,
        "masks": Pool.masks_for(old_hit, active, horizon),
        "context": context,
    }

    new_hit = torch.tensor([[2, 6]])
    retained = retime_batch_timing(batch, new_hit, Pool())
    torch.testing.assert_close(retained, terminal)
    assert torch.equal(batch["hit"], new_hit)
    torch.testing.assert_close(batch["context"].goal_tth, new_hit.float() * dt)
    # Waist follows the longer hand, while each arm reaches the same retained
    # terminal at its own new hit.
    torch.testing.assert_close(batch["seed"][0, 5, 0], terminal[0, 0])
    torch.testing.assert_close(batch["seed"][0, 1, 1], terminal[0, 1])
    torch.testing.assert_close(batch["seed"][0, 5, 2], terminal[0, 2])
    torch.testing.assert_close(batch["init"], batch["seed"] * batch["masks"].elem_mask)


def test_retime_batch_rejects_non_atomic_idle_component():
    class Pool:
        cfg = SimpleNamespace(dt=0.02, joint_names=("waist_yaw_joint",))

        @staticmethod
        def masks_for(hit, active, h):
            return build_waypoint_masks(
                hit[:, 0].clamp_min(0).float() * 0.02,
                hit[:, 1].clamp_min(0).float() * 0.02,
                max_horizon=h, active_left=active[:, 0], active_right=active[:, 1],
                joint_names=("waist_yaw_joint",),
            )

    q = torch.zeros(1, 1)
    active = torch.tensor([[True, False]])
    hit = torch.tensor([[2, -1]])
    masks = Pool.masks_for(hit, active, 3)
    batch = {
        "bucket": 3, "q0": q, "v0": q, "state": torch.cat([q, q], -1),
        "seed": torch.zeros(1, 3, 1), "init": torch.zeros(1, 3, 1),
        "hit": hit, "active": active, "masks": masks,
        "context": ContextInputs.zeros(1, num_drums=1, num_future=0),
    }
    with pytest.raises(ValueError, match="inactive"):
        retime_batch_timing(batch, torch.tensor([[2, 2]]), Pool())


@pytest.mark.parametrize(
    ("score_pattern", "expected_pred"),
    [([-1.0, -1.0, 1.0, 1.0], 2), ([1.0, 1.0, 1.0, 1.0], 0)],
)
def test_assist_directly_admits_without_actor_prevalidation(
    monkeypatch, score_pattern, expected_pred,
):
    names = ("waist_yaw_joint", "left_elbow_joint", "right_elbow_joint")
    dt, horizon, batch_size, d_ctx = 0.02, 6, 2, 4

    class Pool:
        cfg = SimpleNamespace(dt=dt, joint_names=names)
        q_lo = torch.full((3,), -3.0)
        q_hi = torch.full((3,), 3.0)

        @staticmethod
        def masks_for(hit, active, h):
            t = hit.clamp_min(0).float() * dt
            return build_waypoint_masks(
                t[:, 0], t[:, 1], max_horizon=h, dt=dt,
                active_left=active[:, 0], active_right=active[:, 1],
                joint_names=names,
            )

    @dataclass
    class Strike:
        hit_index: torch.Tensor

    class Actor(torch.nn.Module):
        def forward(self, init, state, encoding, masks):
            raise AssertionError("timing admission must not run the actor")

    class Cost:
        joint_names = names
        fk = object()

        def forward(self, q0, traj, v0, strike, lengths, weights):
            raise AssertionError("timing admission must not run physics")

    monkeypatch.setattr(
        "amortized_opt.train_cost.correct_quintic_init_arms",
        lambda _fk, init, *_args, **_kwargs: init,
    )
    active = torch.ones(batch_size, 2, dtype=torch.bool)
    old_hit = torch.tensor([[4, 5], [5, 4]])
    q0 = torch.zeros(batch_size, 3)
    qd0 = torch.zeros_like(q0)
    terminal = torch.tensor([[0.2, 0.4, -0.3], [-0.1, 0.2, 0.5]])
    old_len = per_element_lengths(effective_hits(old_hit, active, horizon), names)
    seed = vel_ramp_seed(q0, qd0, terminal, old_len, horizon, dt)
    context = ContextInputs.zeros(batch_size, num_drums=1, num_future=0)
    context.goal_pos = torch.zeros(batch_size, 2, 3)
    batch = {
        "bucket": horizon, "q0": q0, "v0": qd0,
        "state": torch.cat([q0, qd0], -1), "seed": seed,
        "init": seed.clone(), "hit": old_hit.clone(), "active": active,
        "masks": Pool.masks_for(old_hit, active, horizon), "context": context,
        "kind": torch.zeros(batch_size, dtype=torch.long),
        "mpc_init": torch.zeros(batch_size, dtype=torch.bool),
    }
    encoding = ContextEncoding(
        memory=torch.randn(batch_size, 3, d_ctx),
        mask=torch.ones(batch_size, 3, dtype=torch.bool),
    )
    critic_cfg = ViolationCriticConfig(
        mode="assist", hidden_dim=8, depth=1, batch_size=2,
        recent_capacity=8, audit_capacity=2, assist_fraction=1.0,
        min_assist_samples=1, coarse_step_knots=2,
        collar_inside_probability=0.0,
        collar_boundary_probability=1.0,
        collar_outside_probability=0.0,
    )
    critic = ViolationCriticTrainer(
        critic_feature_dim(3), critic_cfg, device="cpu", seed=5,
    )
    # Predict an exit after the second ray point for every row.
    critic.conservative_score = lambda x: torch.tensor(
        score_pattern * batch_size, device=x.device,
    )
    cfg = Phase2Config(
        device="cpu", compile_mode=None, timing_critic=critic_cfg,
    )
    original_hit = batch["hit"].clone()
    observation, stats = _timing_assist_batch(
        batch=batch, encoding=encoding,
        strike=Strike(effective_hits(old_hit, active, horizon)),
        model=Actor(), cost=Cost(), weights={}, pool=Pool(),
        critic=critic, cfg=cfg,
    )
    assert observation is None
    assert stats["timing_assist_proposed"] == batch_size
    assert stats["timing_selected_pred_manageable"] == expected_pred
    assert stats["timing_advanced"] == batch_size
    assert stats["timing_selected_tth_mean_s"] < horizon * dt
    assert stats["timing_selected_tth_p50_s"] < horizon * dt
    assert batch["timing_selected_pred_manageable"].tolist() == (
        [expected_pred > 0] * batch_size
    )
    assert batch["timing_advanced"].tolist() == [True, True]
    assert not torch.equal(batch["hit"], original_hit)
    torch.testing.assert_close(
        batch["masks"].hit_tth, batch["hit"].float() * dt,
    )
