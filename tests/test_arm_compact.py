"""Active-arm compactness posture objective."""

from __future__ import annotations

import torch

from amortized_opt.constants import N_DOF, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.train_cost import phase2_cost_config
from test_cost import _synthetic_ctx


def test_compact_posture_is_opt_in_and_calibrated_for_phase2():
    assert CostConfig().w_arm_compact == 0.0
    cfg = phase2_cost_config()
    assert cfg.w_arm_compact == 15.0
    assert cfg.arm_compact_mode == "linear"
    assert cfg.arm_compact_height_max == 0.28
    assert cfg.arm_compact_radius_max == 0.28


def test_compact_posture_prices_only_active_arm_and_excludes_q0():
    torch.manual_seed(73)
    H, B = 5, 2
    cfg = CostConfig(
        w_arm_compact=7.0,
        arm_compact_height_max=0.0,
        arm_compact_radius_max=0.01,
    )
    cost = FusedCost(cfg, max_horizon=H, dtype=torch.float64)
    q0 = torch.as_tensor(ready_joint_pos(), dtype=torch.float64).expand(B, -1).clone()
    wp = q0[:, None].expand(-1, H, -1).clone().requires_grad_(True)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    hit = torch.full((B,), H, dtype=torch.long)
    ctx = _synthetic_ctx(
        B, cost.n_drums, "cpu", torch.float64,
        kl=hit, kr=hit, active=(1.0, 0.0),
    )

    out = cost.forward(q0, wp, v0, ctx)
    raw = out.raw["posture_arm_compact"]
    assert torch.all(raw > 0.0)
    assert torch.all(out.counts["posture_arm_compact"] == H)
    assert torch.allclose(
        out.terms["posture"],
        cost.with_config(w_arm_compact=0.0).forward(q0, wp, v0, ctx).terms["posture"]
        + cfg.w_arm_compact * raw,
    )

    grad = torch.autograd.grad(raw.sum(), wp)[0]
    left = cost._gcols["left"]
    right = cost._gcols["right"]
    assert grad.index_select(2, left).abs().sum() > 0.0
    assert grad.index_select(2, right).abs().max() == 0.0

    # Diagnostics are available even with the price disabled, which is what
    # permits threshold calibration before changing an existing objective.
    off = cost.with_config(w_arm_compact=0.0).forward(q0, wp.detach(), v0, ctx)
    for key in (
        "arm_com_height_max", "arm_com_radius_max",
        "arm_com_height_mean", "arm_com_radius_mean",
        "arm_com_height_excess_max", "arm_com_radius_excess_max",
    ):
        assert key in off.diagnostics


def test_linear_mode_has_no_deadband_flat_spot():
    H, B = 4, 1
    cost = FusedCost(
        CostConfig(w_arm_compact=1.0, arm_compact_mode="linear"),
        max_horizon=H, dtype=torch.float64,
    )
    q0 = torch.as_tensor(ready_joint_pos(), dtype=torch.float64).view(1, -1)
    wp = q0[:, None].expand(B, H, -1).clone().requires_grad_(True)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    hit = torch.full((B,), H, dtype=torch.long)
    ctx = _synthetic_ctx(
        B, cost.n_drums, "cpu", torch.float64,
        kl=hit, kr=hit, active=(1.0, 0.0),
    )
    out = cost.forward(q0, wp, v0, ctx)
    expected = (
        out.diagnostics["arm_com_height_mean"]
        + out.diagnostics["arm_com_radius_mean"]
    )
    assert torch.allclose(out.raw["posture_arm_compact"].detach(), expected)
    grad = torch.autograd.grad(out.raw["posture_arm_compact"].sum(), wp)[0]
    assert grad.abs().sum() > 0.0
