"""AL λ multiplies ReLU live, never the softplus warning.

Pins the split for every constraint family: a feasible row is invariant to
λ, a one-hot over-limit wrist knot scales linearly in the hot segment's λ,
and the torque-peak ReLU matches ``diag["torque_excess_max"]``.
"""

from __future__ import annotations

import torch

from amortized_opt.constants import N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import (
    ACTUATOR_SEGMENT_NAMES,
    CONSTRAINT_GROUPS,
    TERM_FOR_GROUP,
    limit_group_name,
)

DT = 0.02
H = 12
WRIST_COLS = (8, 9, 15, 16)
ONE_ELEMENT_AMP = 0.10
#: Shoulder pitch / roll / elbow that park both arms clear of every drum
#: (same pose ``tests/test_collision_peak.py`` uses).
CLEAR_POSE = {3: 0.3, 4: 0.3, 6: 1.2, 10: 0.3, 11: -0.3, 13: 1.2}


def _ctx(B, *, kl, kr, pr_ref=None):
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
        waist_pr_ref=pr_ref,
    )


def _cost(**kw):
    return FusedCost(
        CostConfig(w_waist_pr_band=1.0, waist_pr_band_rad=0.15, **kw),
        max_horizon=H, device="cpu", dtype=torch.float64, dt=DT,
    )


def _hold(*, clear=False, B=2):
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    if clear:
        for j, v in CLEAR_POSE.items():
            q0[:, j] = v
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    return q0, wp, v0, kl, kr


def _fwd(cost, q0, wp, v0, kl, kr, *, pr_ref=None, **kw):
    return cost.forward(
        q0, wp, v0, _ctx(q0.shape[0], kl=kl, kr=kr, pr_ref=pr_ref), **kw,
    )


def _lambda50(family_term: str) -> dict[str, float]:
    """λ=50 on every AL group that lands on ``family_term``."""
    out = {}
    for g, term in TERM_FOR_GROUP.items():
        if term == family_term:
            out[g] = 50.0
    return out


# ============================================================ 1. feasible ==
def test_feasible_hold_is_invariant_to_lambda():
    """Gravity-only hold: ``violations[g]==0`` ⇒ λ 1 vs 50 does not change terms."""
    cost = _cost()
    q0, wp, v0, kl, kr = _hold(clear=True)
    pr = torch.zeros(q0.shape[0], H, 2, dtype=torch.float64)
    one = _fwd(cost, q0, wp, v0, kl, kr, pr_ref=pr)
    families = {
        "torque": "torque",
        "velocity": "velocity",
        "joint_box": "joint_box",
        "collision": "collision",
        "terminal": "terminal",
        "waist_pr_band": "waist_pr_band",
    }
    for fam, term in families.items():
        groups = [g for g, t in TERM_FOR_GROUP.items() if t == term
                  and g not in ("torque_peak", "collision_peak", "joint_box_peak",
                                "velocity_peak")]
        feasible = all(float(one.violations[g].abs().max()) == 0.0 for g in groups)
        if not feasible:
            continue
        hot = _fwd(cost, q0, wp, v0, kl, kr, pr_ref=pr, weights=_lambda50(term))
        assert torch.allclose(hot.terms[term], one.terms[term], atol=1e-12), fam
        if fam == "torque":
            assert torch.allclose(hot.raw["torque"], one.raw["torque"])
        if fam == "velocity":
            assert torch.allclose(hot.raw["velocity"], one.raw["velocity"])
        if fam == "joint_box":
            assert torch.allclose(hot.raw["joint_box"], one.raw["joint_box"])


def test_gravity_hold_limit_families_are_feasible():
    """Zeros + clear arms: torque / velocity / joint_box live is identically 0."""
    cost = _cost()
    q0, wp, v0, kl, kr = _hold(clear=True)
    out = _fwd(cost, q0, wp, v0, kl, kr)
    for fam in ("torque", "velocity", "joint_box"):
        for seg in ACTUATOR_SEGMENT_NAMES:
            g = limit_group_name(fam, seg)
            assert float(out.violations[g].abs().max()) == 0.0, g
        assert float(out.raw[fam].min()) > 0.0, fam  # softplus floor


# ============================================== 2. one-hot wrist knot =====
def test_one_hot_wrist_scales_linearly_in_segment_lambda():
    B = 1
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    wp[0, 3, WRIST_COLS[0]] = ONE_ELEMENT_AMP
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    cost = _cost(w_torque_peak=0.0)
    ref = _fwd(cost, q0, wp, v0, kl, kr)
    hot = _fwd(cost, q0, wp, v0, kl, kr, weights={"torque_left_wrist": 5.0})
    live = ref.violations["torque_left_wrist"]
    assert float(live.min()) > 0.0
    assert torch.allclose(hot.raw["torque"], ref.raw["torque"])
    delta = hot.terms["torque"] - ref.terms["torque"]
    assert torch.allclose(delta, (5.0 - 1.0) * live, atol=1e-12)


# ====================================================== 3. ok-style peak ==
def test_torque_peak_live_matches_excess_max_diagnostic():
    B = 1
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    wp[0, 3, WRIST_COLS[0]] = ONE_ELEMENT_AMP
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    cost = _cost(w_torque_peak=1.0)
    out = _fwd(cost, q0, wp, v0, kl, kr)
    assert torch.equal(out.violations["torque_peak"],
                       out.diagnostics["torque_excess_max"])
    assert set(CONSTRAINT_GROUPS) <= set(out.violations)
