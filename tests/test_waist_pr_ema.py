"""Leaky EMA of clipped waist P/R as the upright-well reference.

Decoder owns pitch/roll.  Snapshot / 81-grid keep ``waist_pr_ref=None``
(dense well vs 0).  Closed-loop harvest fills a stopgrad leaked EMA of
the box-clamped command.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.train_cost import phase2_cost_config
from amortized_opt.warmstart import update_waist_pr_ema, waist_pr_slice

I_PITCH, I_ROLL = waist_pr_slice()
H = 10


def test_update_is_convex_combination_then_leak():
    ema = np.array([0.4, -0.2])
    cmd = np.array([0.0, 0.0])
    out = update_waist_pr_ema(ema, cmd, alpha=0.25, leak=0.1)
    mixed = 0.75 * ema + 0.25 * cmd
    expect = 0.9 * mixed
    np.testing.assert_allclose(out, expect)


def test_leak_alone_decays_to_zero():
    cmd = torch.tensor([0.5, -0.5])
    ema = cmd.clone()
    a, b = 0.1, 0.05
    for _ in range(200):
        ema = update_waist_pr_ema(ema, cmd, alpha=a, leak=b)
    # leaky integrator of a constant c → α(1−β) / (1 − (1−β)(1−α)) · c
    gain = a * (1.0 - b) / (1.0 - (1.0 - b) * (1.0 - a))
    assert torch.allclose(ema, gain * cmd, atol=1e-5)
    assert float(gain) < 1.0


def test_well_tracks_stopgrad_ema_ref():
    B = 2
    cost = FusedCost(CostConfig(n_sub=1, w_waist_upright=1.0),
                     max_horizon=H, device="cpu", dtype=torch.float64)
    ref = torch.tensor([[0.10, -0.05], [0.00, 0.20]], dtype=torch.float64)
    pr_ref = ref.unsqueeze(1).expand(-1, H, -1).contiguous()
    pr_ref = pr_ref.clone().requires_grad_(True)
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    on = q0.unsqueeze(1).repeat(1, H, 1)
    on[:, :, I_PITCH] = ref[:, 0].unsqueeze(1)
    on[:, :, I_ROLL] = ref[:, 1].unsqueeze(1)
    ct_on = cost.forward(q0, on, v0, _strike(B, pr_ref))
    assert float(ct_on.raw["posture_upright"].abs().max()) < 1e-12

    off = on.detach().clone()
    off[:, :, I_PITCH] = off[:, :, I_PITCH] + 0.3
    off.requires_grad_(True)
    ct_off = cost.forward(q0, off, v0, _strike(B, pr_ref))
    ct_off.raw["posture_upright"].sum().backward()
    assert pr_ref.grad is None or float(pr_ref.grad.abs().sum()) == 0.0
    assert float(off.grad[:, :, I_PITCH].abs().max()) > 0


def test_alpha_one_leak_zero_is_lag1():
    ema = torch.tensor([0.3, 0.1])
    cmd = torch.tensor([-0.2, 0.4])
    out = update_waist_pr_ema(ema, cmd, alpha=1.0, leak=0.0)
    assert torch.equal(out, cmd)


def test_rejects_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        update_waist_pr_ema(np.zeros(2), np.zeros(2), alpha=1.5, leak=0.0)
    with pytest.raises(ValueError, match="leak"):
        update_waist_pr_ema(np.zeros(2), np.zeros(2), alpha=0.1, leak=-0.01)


def _strike(B, pr_ref):
    kk = torch.full((B, 2), H, dtype=torch.long)
    ones = torch.ones(B, 2, dtype=torch.float64)
    return StrikeContext(
        hit_index=kk, active=ones.clone(),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=torch.float64),
        v_star=torch.full((B, 2, 3), -1.0, dtype=torch.float64),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=torch.float64),
        disk_offset=torch.zeros(B, 2, 3, dtype=torch.float64),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=torch.float64),
        has_half_plane=torch.zeros(B, 2, dtype=torch.float64),
        contact_sigma=torch.full((B, 2), 0.20, dtype=torch.float64),
        shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        start_shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        elev_required=ones.clone(),
        waist_pr_ref=pr_ref,
    )


def test_snapshot_none_is_upright_vs_zero():
    B = 2
    cost = FusedCost(CostConfig(n_sub=1, w_waist_upright=1.0),
                     max_horizon=H, device="cpu", dtype=torch.float64)
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    q0[:, I_PITCH] = 0.2
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    ct = cost.forward(q0, wp, v0, _strike(B, None))
    # constant 0.2 on pitch, 0 on roll → mean over 2 lean rows of 0.2²
    assert torch.allclose(ct.raw["posture_upright"],
                          torch.full((B,), 0.5 * 0.04, dtype=torch.float64),
                          atol=1e-12)


def test_phase2_limit_weights_share_a_characteristic_unit():
    """1 N·m, 0.5 rad/s, 0.05 rad cost the same at λ=1."""
    cfg = phase2_cost_config()
    unit = cfg.w_torque * 1.0
    assert cfg.w_velocity * 0.5 == pytest.approx(unit)
    assert cfg.w_joint_box * 0.05 == pytest.approx(unit)
