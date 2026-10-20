"""Tip lateral-ordering hinge (plain objective, not AL).

Distinct from ``w_stick_stick`` (capsule clearance): sticks can swap sides while
staying physically separated.  This term prices torso-frame
``tip_L.y - tip_R.y`` going negative.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.fk import ChainFK
from amortized_opt.idle_cylinder import tip_in_torso
from amortized_opt.idle_home import default_idle_home_path, load_idle_home_table
from amortized_opt.train_cost import (
    DIAG_KEYS,
    DIAG_NAMES,
    METRIC_NAMES,
    TERM_KEYS,
    WeightSchedule,
)

DT = 0.02
H = 12
JOINT = {n: i for i, n in enumerate(G1_UPPER_BODY_JOINTS)}


def _ctx(B: int, *, dtype=torch.float64) -> StrikeContext:
    ones = torch.ones(B, 2, dtype=dtype)
    return StrikeContext(
        hit_index=torch.full((B, 2), H, dtype=torch.long),
        active=ones.clone(),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.20, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, dtype=dtype),
        elev_required=ones.clone(),
    )


def _cost(**kw) -> FusedCost:
    return FusedCost(
        CostConfig(**kw), max_horizon=H, device="cpu", dtype=torch.float64, dt=DT
    )


def _fwd(cost: FusedCost, q0: torch.Tensor, wp: torch.Tensor):
    return cost.forward(q0, wp, torch.zeros_like(q0), _ctx(q0.shape[0]))


def _held(q0: torch.Tensor) -> torch.Tensor:
    return q0.unsqueeze(1).repeat(1, H, 1)


def test_ready_pose_has_near_zero_crossover_raw():
    q = torch.as_tensor(ready_joint_pos(G1_UPPER_BODY_JOINTS), dtype=torch.float64).view(1, -1)
    out = _fwd(_cost(w_tip_crossover=50.0), q, _held(q))
    assert float(out.diagnostics["tip_lateral_min"].min()) > 0.2
    assert float(out.raw["tip_crossover"].max()) < 1e-4


def test_idle_home_pose_prices_crossover():
    table = load_idle_home_table(default_idle_home_path(), joint_names=G1_UPPER_BODY_JOINTS)
    q = torch.as_tensor(table.full_pose(yaw=0.0), dtype=torch.float64).view(1, -1)
    out = _fwd(_cost(w_tip_crossover=50.0), q, _held(q))
    assert float(out.diagnostics["tip_lateral_min"].min()) < 0.0
    assert float(out.raw["tip_crossover"].min()) > 1e-4
    assert float(out.terms["tip_crossover"].min()) > 0.0


def test_weight_zero_is_bit_identical_on_total():
    table = load_idle_home_table(default_idle_home_path(), joint_names=G1_UPPER_BODY_JOINTS)
    q = torch.as_tensor(table.full_pose(yaw=0.0), dtype=torch.float64).view(1, -1)
    wp = _held(q)
    off = _fwd(_cost(w_tip_crossover=0.0), q, wp)
    on = _fwd(_cost(w_tip_crossover=50.0), q, wp)
    # diagnostics still report ordering with the weight off
    assert float(off.diagnostics["tip_lateral_min"].min()) < 0.0
    assert float(off.terms["tip_crossover"].sum()) == 0.0
    assert float(on.terms["tip_crossover"].sum()) > 0.0
    # other terms unchanged when weight is zero vs a fresh cost without the field
    base = _fwd(_cost(), q, wp)
    assert torch.allclose(off.total, base.total)


def test_torso_yaw_invariance_of_lateral_ordering():
    table = load_idle_home_table(default_idle_home_path(), joint_names=G1_UPPER_BODY_JOINTS)
    fk = ChainFK(device="cpu", dtype=torch.float64)
    d0 = None
    for yaw in (0.0, 0.7, -1.2):
        q = torch.as_tensor(table.full_pose(yaw=yaw), dtype=torch.float64).view(1, -1)
        yl = float(tip_in_torso(fk, q, "left")[0, 1])
        yr = float(tip_in_torso(fk, q, "right")[0, 1])
        d = yl - yr
        if d0 is None:
            d0 = d
        assert abs(d - d0) < 1e-5


def test_gradient_increases_lateral_separation():
    table = load_idle_home_table(default_idle_home_path(), joint_names=G1_UPPER_BODY_JOINTS)
    cost = _cost(w_tip_crossover=50.0)
    q0 = torch.as_tensor(table.full_pose(yaw=0.0), dtype=torch.float64).view(1, -1)
    wp = _held(q0).clone().requires_grad_(True)
    out = _fwd(cost, q0, wp)
    before = float(out.diagnostics["tip_lateral_min"].min())
    assert before < 0.0
    out.raw["tip_crossover"].sum().backward()
    g = wp.grad
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 1e-6
    with torch.no_grad():
        wp2 = wp - (0.05 / float(g.abs().max())) * g
    after = float(_fwd(cost, q0, wp2).diagnostics["tip_lateral_min"].min())
    assert after > before


def test_sample_zero_excluded_from_priced_mask():
    """q0 is measured; a crossed q0 with uncrossed waypoints must not floor the term."""
    table = load_idle_home_table(default_idle_home_path(), joint_names=G1_UPPER_BODY_JOINTS)
    q_cross = torch.as_tensor(table.full_pose(yaw=0.0), dtype=torch.float64).view(1, -1)
    q_ready = torch.as_tensor(
        ready_joint_pos(G1_UPPER_BODY_JOINTS), dtype=torch.float64
    ).view(1, -1)
    # Hold ready for all waypoints; only q0 is crossed.
    wp = _held(q_ready)
    out = _fwd(_cost(w_tip_crossover=50.0), q_cross, wp)
    assert float(out.raw["tip_crossover"].max()) < 1e-3


def test_validate_rejects_bad_knobs():
    with pytest.raises(ValueError):
        CostConfig(w_tip_crossover=-1.0).validate()
    with pytest.raises(ValueError):
        CostConfig(tip_crossover_margin=-0.01).validate()
    with pytest.raises(ValueError):
        CostConfig(beta_tip_crossover=0.0).validate()


def test_term_and_diag_keys_append_only():
    assert TERM_KEYS[-5] == "terminal_aim"
    assert TERM_KEYS[-4] == "waist_pr_band"
    assert TERM_KEYS[-3] == "waist_pr_terminal"
    assert TERM_KEYS[-2] == "waist_upright"
    assert TERM_KEYS[-1] == "tip_up"
    assert "tip_crossover" in TERM_KEYS
    assert "tip_lateral_min" in DIAG_KEYS
    assert "tip_crossover_max" in DIAG_KEYS
    assert "waist_pr_N" in DIAG_KEYS
    assert len(DIAG_KEYS) == len(DIAG_NAMES)
    assert "c_tip_crossover" in METRIC_NAMES
    assert "c_terminal_aim" in METRIC_NAMES
    assert "c_waist_pr_terminal" in METRIC_NAMES
    assert "c_waist_upright" in METRIC_NAMES
    assert METRIC_NAMES.count("c_tip_crossover") == 1


def test_tip_crossover_schedule_ramps():
    sched = WeightSchedule(tip_crossover_start=0, tip_crossover_ramp=100)
    assert sched.tip_crossover_alpha(0) == pytest.approx(0.0, abs=1e-6)
    assert sched.tip_crossover_alpha(100) == pytest.approx(1.0, abs=1e-6)
    assert sched.tip_crossover_alpha(50) == pytest.approx(0.5, abs=0.05)
    assert sched.multipliers(100)["tip_crossover"] == pytest.approx(1.0)
