"""``joint_box_peak`` -- smooth-max ReLU on the decoder plan."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import CONSTRAINT_GROUPS, ALConfig, AugmentedLagrangian
from amortized_opt.train_cost import (
    METRIC_NAMES,
    VIOL_KEYS,
    VIOL_OFFSET,
    WeightSchedule,
    migrate_al_state,
    phase2_al_config,
)

DT = 0.02
H = 12
WAIST_PITCH = G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")


def _ctx(B, *, kl, kr):
    ones = torch.ones(B, 2, dtype=torch.float64)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=ones.clone(),
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
    )


def _cost(**kw):
    return FusedCost(
        CostConfig(waist_pr_box=0.20, **kw),
        max_horizon=H, device="cpu", dtype=torch.float64, dt=DT,
    )


def _batch(*, oob: bool, B: int = 2):
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    if oob:
        wp[:, :, WAIST_PITCH] = 0.35
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    return q0, wp, v0, kl, kr


def _fwd(cost, q0, wp, v0, kl, kr, **kw):
    return cost.forward(q0, wp, v0, _ctx(q0.shape[0], kl=kl, kr=kr), **kw)


@pytest.fixture
def violating():
    cost = _cost(w_joint_box_peak=1.0)
    q0, wp, v0, kl, kr = _batch(oob=True)
    out = _fwd(cost, q0, wp, v0, kl, kr)
    assert float(out.diagnostics["joint_box_excess_max"].max()) > 0.0
    return cost, out


def test_group_exists_and_phase2_al_has_it():
    assert "joint_box_peak" in CONSTRAINT_GROUPS
    assert CONSTRAINT_GROUPS[-5:-1] == (
        "torque_peak", "collision_peak", "joint_box_peak", "waist_pr_band",
    )
    assert CONSTRAINT_GROUPS[-1] == "velocity_peak"
    cfg = phase2_al_config()
    for name in ("rho", "target", "pen", "lam_max"):
        assert "joint_box_peak" in getattr(cfg, name)
    assert cfg.target["joint_box_peak"] == 0.0


def test_violation_matches_excess_max_diagnostic(violating):
    _, out = violating
    assert torch.equal(
        out.violations["joint_box_peak"],
        out.diagnostics["joint_box_excess_max"],
    )


def test_peak_raw_is_smooth_max_of_waypoint_relu(violating):
    """All H waypoints at +0.15 rad over the tightened waist box (one joint)."""
    _, out = violating
    excess = 0.15
    # p-norm over H equal entries: excess * H**(1/p)
    expect = excess * (H ** (1.0 / 12.0))
    assert float(out.raw["joint_box_peak"].mean()) == pytest.approx(expect, rel=1e-5)
    assert float(out.violations["joint_box_peak"].mean()) == pytest.approx(
        excess, rel=1e-6,
    )


def test_gradient_hits_every_oob_waypoint():
    cost = _cost(w_joint_box_peak=1.0)
    q0, wp, v0, kl, kr = _batch(oob=True)
    wp = wp.clone().requires_grad_(True)
    out = _fwd(cost, q0, wp, v0, kl, kr)
    out.raw["joint_box_peak"].sum().backward()
    g = wp.grad[:, :, WAIST_PITCH]
    assert g is not None
    assert bool((g.abs() > 0).all())


def test_zero_weight_is_a_noop_on_peak_raw():
    cost = _cost(w_joint_box_peak=0.0)
    q0, wp, v0, kl, kr = _batch(oob=True)
    out = _fwd(cost, q0, wp, v0, kl, kr)
    assert float(out.raw["joint_box_peak"].abs().max()) == 0.0
    assert out.raw["joint_box_peak"].requires_grad is False


def test_multipliers_move_independently():
    al = AugmentedLagrangian(ALConfig(rho=1.0, decay=1.0))
    base = dict(al.lam)
    al.update({"joint_box_waist": 0.0, "joint_box_peak": 0.08})
    assert al.lam["joint_box_peak"] == pytest.approx(base["joint_box_peak"] + 0.08)
    assert al.lam["joint_box_waist"] == pytest.approx(base["joint_box_waist"])


def test_resume_appends_without_seeding_from_joint_box():
    old = {
        "lam": {"torque": 7.2, "velocity": 1.0, "joint_box": 3.0,
                "collision": 4.8, "terminal": 2.0},
        "groups": ["torque", "velocity", "joint_box", "collision", "terminal"],
    }
    new = migrate_al_state(old)
    assert "joint_box_peak" in new["groups"]
    assert "joint_box_peak" not in new["lam"]


def test_schedule_ramps_independently():
    s = WeightSchedule(joint_box_peak_start=0, joint_box_peak_ramp=1000)
    assert s.multipliers(0)["joint_box_peak"] == pytest.approx(0.0)
    assert s.multipliers(1000)["joint_box_peak"] == pytest.approx(1.0)


def test_metrics_block_names_peak():
    i = VIOL_KEYS.index("joint_box_peak")
    assert METRIC_NAMES[VIOL_OFFSET + i] == "v_joint_box_peak"
