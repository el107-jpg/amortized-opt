"""Gravity-only static torque cost term and torque_static_only AL bypass."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import (
    N_DOF,
    static_torque_tighten_divisors,
    torque_limit_eta_scale,
)
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import AugmentedLagrangian, CONSTRAINT_GROUPS
from amortized_opt.train_cost import TERM_KEYS, phase2_al_config


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


def test_torque_static_term_key_present():
    assert "torque_static" in TERM_KEYS


@pytest.mark.parametrize("device", ["cpu"])
def test_static_differs_from_full_torque(device):
    dtype = torch.float64
    B, H = 2, 12
    cost_full = FusedCost(CostConfig(n_sub=1), max_horizon=H, dtype=dtype)
    cost_static = FusedCost(
        CostConfig(
            n_sub=1, w_torque=0.0, w_torque_peak=0.0,
            w_torque_static=20.0, torque_static_only=True,
        ),
        max_horizon=H, dtype=dtype,
    )
    q0, wp, v0, lengths = _random_traj(B, H, device, dtype)
    ctx = _synthetic_ctx(B, device, dtype)
    ct_full = cost_full.forward(q0, wp, v0, ctx, lengths=lengths)
    ct_static = cost_static.forward(q0, wp, v0, ctx, lengths=lengths)
    assert torch.isfinite(ct_static.terms["torque_static"]).all()
    assert ct_static.terms["torque"].abs().max() == 0.0
    assert not torch.allclose(
        ct_static.terms["torque_static"], ct_full.terms["torque"],
    )


@pytest.mark.parametrize("device", ["cpu"])
def test_static_al_populates_torque_violations(device):
    dtype = torch.float64
    B, H = 2, 12
    cost = FusedCost(
        CostConfig(
            n_sub=1, w_torque_static=1.0, torque_static_only=True,
            torque_static_al=True,
        ),
        max_horizon=H, dtype=dtype,
    )
    q0, wp, v0, lengths = _random_traj(B, H, device, dtype)
    ctx = _synthetic_ctx(B, device, dtype)
    ct = cost.forward(q0, wp, v0, ctx, lengths=lengths)
    assert "torque_waist" in ct.violations
    assert "torque_peak" in ct.violations
    assert torch.isfinite(ct.terms["torque_static"]).all()
    assert float(ct.raw["torque_waist"].sum()) >= 0.0
    off = FusedCost(
        CostConfig(
            n_sub=1, w_torque_static=1.0, torque_static_only=True,
            torque_static_al=False,
        ),
        max_horizon=H, dtype=dtype,
    )
    ct_off = off.forward(q0, wp, v0, ctx, lengths=lengths)
    assert float(ct_off.violations["torque_waist"].max()) == 0.0


@pytest.mark.parametrize("device", ["cpu"])
def test_static_only_zeros_torque_violations(device):
    dtype = torch.float64
    B, H = 2, 12
    cost = FusedCost(
        CostConfig(
            n_sub=1, w_torque_static=1.0, torque_static_only=True,
        ),
        max_horizon=H, dtype=dtype,
    )
    q0, wp, v0, lengths = _random_traj(B, H, device, dtype)
    ctx = _synthetic_ctx(B, device, dtype)
    ct = cost.forward(q0, wp, v0, ctx, lengths=lengths)
    for g in CONSTRAINT_GROUPS:
        if g.startswith("torque"):
            assert float(ct.violations[g].max()) == 0.0


@pytest.mark.parametrize("device", ["cpu"])
def test_static_torque_tighten_increases_hinge(device):
    dtype = torch.float64
    B, H = 2, 12
    q0, wp, v0, lengths = _random_traj(B, H, device, dtype)
    ctx = _synthetic_ctx(B, device, dtype)
    loose = FusedCost(
        CostConfig(
            n_sub=1, w_torque_static=1.0, torque_static_only=True,
        ),
        max_horizon=H, dtype=dtype,
    )
    tight = FusedCost(
        CostConfig(
            n_sub=1, w_torque_static=1.0, torque_static_only=True,
            static_torque_tighten="decoder_measured",
        ),
        max_horizon=H, dtype=dtype,
    )
    ct_loose = loose.forward(q0, wp, v0, ctx, lengths=lengths)
    ct_tight = tight.forward(q0, wp, v0, ctx, lengths=lengths)
    assert float(ct_tight.terms["torque_static"].mean()) >= float(
        ct_loose.terms["torque_static"].mean()
    )
    assert not torch.allclose(
        tight.c_static_torque_tighten, loose.c_static_torque_tighten,
    )


def test_static_torque_tighten_divisors_arm_range():
    div = static_torque_tighten_divisors(mode="decoder_measured")
    assert div.shape == (N_DOF,)
    assert div.min() >= 1.0
    assert div.max() <= 10.0
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS
    idx = {n: i for i, n in enumerate(G1_UPPER_BODY_JOINTS)}
    assert div[idx["waist_roll_joint"]] > 1.0
    assert div[idx["left_shoulder_roll_joint"]] > div[idx["left_elbow_joint"]]
    assert div[idx["left_wrist_yaw_joint"]] > div[idx["left_wrist_pitch_joint"]]


def test_static_waist_tightened_dynamic_waist_full_hard():
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS
    idx = {n: i for i, n in enumerate(G1_UPPER_BODY_JOINTS)}
    eta = torque_limit_eta_scale(G1_UPPER_BODY_JOINTS, eta_torque=0.8)
    assert eta[idx["waist_pitch_joint"]] == 1.0
    assert eta[idx["left_elbow_joint"]] == 0.8
    static_cost = FusedCost(
        CostConfig(
            n_sub=1, w_torque_static=1.0, torque_static_only=True,
            static_torque_tighten="decoder_measured", eta_torque=0.8,
        ),
        max_horizon=8, dtype=torch.float64,
    )
    dynamic_cost = FusedCost(
        CostConfig(n_sub=1, w_torque=1.0, eta_torque=0.8),
        max_horizon=8, dtype=torch.float64,
    )
    wr = idx["waist_roll_joint"]
    el = idx["left_elbow_joint"]
    assert static_cost.c_static_torque_tighten[wr] > 1.0
    assert static_cost.c_torque_eta[wr] == 1.0
    assert static_cost.c_torque_eta[el] == 0.8
    assert dynamic_cost.c_torque_eta[wr] == 1.0
    assert static_cost.c_static_torque_tighten[el] > 1.0


def test_al_update_skips_torque_groups_when_static_only():
    al = AugmentedLagrangian(phase2_al_config())
    lam_before = dict(al.lam)
    viol = {g: 10.0 for g in CONSTRAINT_GROUPS}
    viol_mean = {
        g: v for g, v in viol.items()
        if not g.startswith("torque")
    }
    al.update(viol_mean)
    for g in CONSTRAINT_GROUPS:
        if g.startswith("torque"):
            assert al.lam[g] == lam_before[g]
        else:
            assert al.lam[g] != lam_before[g] or viol[g] == 0.0
