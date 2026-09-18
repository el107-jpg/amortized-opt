"""Always-project PPO onto the analytic cost's nullspace at waypoint 0.

Cost is rescaled per sample to the PPO gradient's magnitude so FusedCost
cannot swallow the 17-dim KL budget.  The projected PPO leftover is added at
waypoint 0 at PPO's native scale.
"""

from __future__ import annotations

import pytest
import torch

from isaac_rl_ppo.ppo_loop import combine_wp0_orth


def _rand_traj_grad(B=4, H=8, n=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, n, generator=g)


def test_parallel_rl_is_dropped():
    """If PPO already points along cost, option A leaves scaled cost only."""
    g_cost = _rand_traj_grad(seed=1)
    g_rl = torch.zeros_like(g_cost)
    g_rl[:, 0] = 3.0 * g_cost[:, 0]

    out, stats = combine_wp0_orth(g_rl, g_cost)
    assert stats["orth_remain"] < 1e-5
    assert abs(stats["orth_cos"]) > 0.99
    c0 = g_cost[:, 0]
    o0 = out[:, 0]
    cos = (o0 * c0).sum(-1) / (o0.norm(dim=-1) * c0.norm(dim=-1)).clamp_min(1e-8)
    assert torch.all(cos > 0.99)


def test_perpendicular_rl_is_added_at_native_scale():
    B, H, n = 3, 6, 17
    g_cost = torch.zeros(B, H, n)
    g_rl = torch.zeros(B, H, n)
    g_cost[:, 0, 0] = 4.0
    g_cost[:, 3, 5] = 9.0
    g_rl[:, 0, 1] = 2.0

    out, stats = combine_wp0_orth(g_rl, g_cost)
    assert torch.allclose(out[:, 0, 1], torch.full((B,), 2.0), atol=1e-5)
    assert stats["orth_cos"] == pytest.approx(0.0, abs=1e-5)
    assert stats["orth_remain"] == pytest.approx(1.0, abs=1e-5)
    cost_part = out.clone()
    cost_part[:, 0, 1] = 0.0
    assert torch.allclose(
        cost_part.flatten(1).norm(dim=-1), torch.full((B,), 2.0), atol=1e-4)


def test_tail_of_rl_is_ignored():
    """A leaked tail PPO gradient must not rewrite the next-tick warm start."""
    g_cost = _rand_traj_grad(seed=2)
    g_rl = _rand_traj_grad(seed=3)
    out, _ = combine_wp0_orth(g_rl, g_cost)
    gc_t = g_cost[:, 1:].flatten(1)
    out_t = out[:, 1:].flatten(1)
    cos = (out_t * gc_t).sum(-1) / (
        out_t.norm(dim=-1) * gc_t.norm(dim=-1)).clamp_min(1e-8)
    assert torch.all(cos > 0.99)


def test_huge_cost_is_matched_to_rl_magnitude():
    """The failure mode of the first Isaac run: aux gradient 1000x PPO."""
    B, H, n = 2, 5, 8
    g_rl = torch.zeros(B, H, n)
    g_cost = torch.zeros(B, H, n)
    g_rl[:, 0, 0] = 1.0
    g_cost[:] = 100.0
    out, stats = combine_wp0_orth(g_rl, g_cost)
    assert stats["orth_cost_scale"] < 0.01
    assert float(out.flatten(1).norm(dim=-1).max()) < 10.0


def test_zero_cost_keeps_raw_rl_at_waypoint_0():
    g_rl = _rand_traj_grad(seed=4)
    g_cost = torch.zeros_like(g_rl)
    out, _ = combine_wp0_orth(g_rl, g_cost)
    assert torch.equal(out[:, 0], g_rl[:, 0])
    assert torch.equal(out[:, 1:], g_cost[:, 1:])


def test_zero_rl_scales_cost_to_unit():
    g_cost = _rand_traj_grad(seed=5)
    out, _ = combine_wp0_orth(torch.zeros_like(g_cost), g_cost)
    norms = out.flatten(1).norm(dim=-1)
    assert torch.allclose(norms, torch.ones(g_cost.shape[0]), atol=1e-5)


def test_traj_grad_scale_shrinks_the_combined_grad():
    g_rl = _rand_traj_grad(seed=6)
    g_cost = _rand_traj_grad(seed=7)
    full, _ = combine_wp0_orth(g_rl, g_cost)
    shrunk, stats = combine_wp0_orth(g_rl, g_cost, traj_grad_scale=0.2)
    assert stats["orth_traj_scale"] == pytest.approx(0.2)
    assert torch.allclose(shrunk, 0.2 * full, atol=1e-5)


def test_per_sample_not_batched_flatten():
    """Env 0's cost direction must not project env 1's RL."""
    B, H, n = 2, 4, 5
    g_cost = torch.zeros(B, H, n)
    g_rl = torch.zeros(B, H, n)
    g_cost[0, 0, 0] = 1.0
    g_cost[1, 0, 1] = 1.0
    g_rl[0, 0, 0] = 1.0
    g_rl[1, 0, 0] = 1.0

    out, _ = combine_wp0_orth(g_rl, g_cost)
    assert out[0, 0, 1].abs() < 1e-5
    assert torch.allclose(out[1, 0, 0], torch.tensor(1.0), atol=1e-5)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="g_rl"):
        combine_wp0_orth(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5))


def test_zero_leak_matches_hard_orth():
    g_rl = _rand_traj_grad(seed=8)
    g_cost = _rand_traj_grad(seed=9)
    hard, _ = combine_wp0_orth(g_rl, g_cost)
    leaked, stats = combine_wp0_orth(g_rl, g_cost, wp0_leak=0.0)
    assert stats["orth_leak"] == 0.0
    assert torch.equal(hard, leaked)


def test_leak_reinjects_the_parallel_slice_at_wp0():
    """PPO fully along cost: leak α keeps α · g_rl0 at waypoint 0."""
    g_cost = _rand_traj_grad(seed=10)
    g_rl = torch.zeros_like(g_cost)
    g_rl[:, 0] = 3.0 * g_cost[:, 0]
    leak = 0.25
    hard, _ = combine_wp0_orth(g_rl, g_cost, wp0_leak=0.0)
    leaked, stats = combine_wp0_orth(g_rl, g_cost, wp0_leak=leak)
    assert stats["orth_leak"] == pytest.approx(leak)
    assert stats["orth_para_frac"] == pytest.approx(1.0, abs=1e-4)
    delta = leaked[:, 0] - hard[:, 0]
    assert torch.allclose(delta, leak * g_rl[:, 0], atol=1e-5)
    assert torch.allclose(leaked[:, 1:], hard[:, 1:], atol=1e-6)


def test_full_leak_adds_raw_rl_at_waypoint_0():
    B, H, n = 3, 6, 17
    g_cost = torch.zeros(B, H, n)
    g_rl = torch.zeros(B, H, n)
    g_cost[:, 0, 0] = 4.0
    g_cost[:, 3, 5] = 9.0
    g_rl[:, 0, 0] = 2.0          # parallel
    g_rl[:, 0, 1] = 3.0          # orthogonal
    out, _ = combine_wp0_orth(g_rl, g_cost, wp0_leak=1.0)
    assert torch.allclose(out[:, 0, 1], torch.full((B,), 3.0), atol=1e-5)
    # scaled cost at wp0 dim 0 plus the full parallel PPO slice
    hard, _ = combine_wp0_orth(g_rl, g_cost, wp0_leak=0.0)
    assert torch.allclose(out[:, 0, 0], hard[:, 0, 0] + 2.0, atol=1e-5)


def test_leak_does_not_touch_the_tail():
    g_cost = _rand_traj_grad(seed=11)
    g_rl = _rand_traj_grad(seed=12)
    out, _ = combine_wp0_orth(g_rl, g_cost, wp0_leak=0.4)
    gc_t = g_cost[:, 1:].flatten(1)
    out_t = out[:, 1:].flatten(1)
    cos = (out_t * gc_t).sum(-1) / (
        out_t.norm(dim=-1) * gc_t.norm(dim=-1)).clamp_min(1e-8)
    assert torch.all(cos > 0.99)
