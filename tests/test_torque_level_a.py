"""Level-A torque physics: diag(M)+gravity proxy and blend into full RNEA."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.train_cost import WeightSchedule, _wtensors, phase2_al_config


def _synthetic_ctx(B, device, dtype):
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    kl = torch.full((B,), 40, device=device, dtype=torch.long)
    kr = torch.full((B,), 45, device=device, dtype=torch.long)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=ones,
        target_drum=torch.zeros(B, 2, device=device, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, device=device, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, device=device, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, device=device, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, device=device, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, device=device, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, device=device, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        elev_required=ones,
    )


def _random_traj(B, H, device, dtype):
    q0 = torch.randn(B, N_DOF, device=device, dtype=dtype) * 0.1
    wp = torch.randn(B, H, N_DOF, device=device, dtype=dtype) * 0.2
    v0 = torch.zeros(B, N_DOF, device=device, dtype=dtype)
    lengths = torch.full((B, 3), float(H), device=device, dtype=dtype)
    return q0, wp, v0, lengths


@pytest.mark.parametrize("device", ["cpu"])
def test_torque_full_blend_schedule(device):
    s = WeightSchedule(torque_full_start=1000, torque_full_ramp=500)
    assert s.torque_full_blend(0) == 0.0
    assert s.torque_full_blend(999) == 0.0
    assert 0.0 < s.torque_full_blend(1250) < 1.0
    assert s.torque_full_blend(2000) == 1.0
    default = WeightSchedule()
    assert default.torque_full_blend(0) == 1.0


def test_level_a_torque_differs_from_full():
    dtype = torch.float64
    device = "cpu"
    B, H = 2, 12
    cost_full = FusedCost(
        CostConfig(torque_physics="full", n_sub=1), max_horizon=H, dtype=dtype,
    )
    cost_a = FusedCost(
        CostConfig(torque_physics="diag_grav_frozen", n_sub=1), max_horizon=H,
        dtype=dtype,
    )
    q0, wp, v0, lengths = _random_traj(B, H, device, dtype)
    ctx = _synthetic_ctx(B, device, dtype)
    w0 = _wtensors(
        WeightSchedule().multipliers(0),
        AugmentedLagrangian(phase2_al_config()).weights(),
        device, dtype,
        sched=WeightSchedule(torque_full_start=-1),
        joint_names=cost_full.joint_names,
        step=0,
    )
    w1 = _wtensors(
        WeightSchedule().multipliers(0),
        AugmentedLagrangian(phase2_al_config()).weights(),
        device, dtype,
        sched=WeightSchedule(),
        joint_names=cost_full.joint_names,
        step=0,
    )
    ct_full = cost_full.forward(q0, wp, v0, ctx, lengths=lengths, weights=w1)
    ct_a = cost_a.forward(q0, wp, v0, ctx, lengths=lengths, weights=w0)
    assert torch.isfinite(ct_a.terms["torque"]).all()
    assert torch.isfinite(ct_full.terms["torque"]).all()
    assert not torch.allclose(ct_a.terms["torque"], ct_full.terms["torque"])


def test_blend_interpolates_torque_term():
    dtype = torch.float64
    device = "cpu"
    B, H = 2, 12
    cost = FusedCost(
        CostConfig(torque_physics="diag_grav_frozen", n_sub=1), max_horizon=H,
        dtype=dtype,
    )
    q0, wp, v0, lengths = _random_traj(B, H, device, dtype)
    ctx = _synthetic_ctx(B, device, dtype)
    sched = WeightSchedule(torque_full_start=0, torque_full_ramp=1)
    al = AugmentedLagrangian(phase2_al_config())
    w_lo = _wtensors(
        sched.multipliers(0), al.weights(), device, dtype,
        sched=WeightSchedule(torque_full_start=-1, torque_full_ramp=1),
        joint_names=cost.joint_names, step=0,
    )
    w_hi = _wtensors(
        sched.multipliers(0), al.weights(), device, dtype,
        sched=WeightSchedule(torque_full_start=0, torque_full_ramp=1),
        joint_names=cost.joint_names, step=1,
    )
    t_lo = cost.forward(q0, wp, v0, ctx, lengths=lengths, weights=w_lo).terms["torque"]
    t_hi = cost.forward(q0, wp, v0, ctx, lengths=lengths, weights=w_hi).terms["torque"]
    assert not torch.allclose(t_lo, t_hi)
