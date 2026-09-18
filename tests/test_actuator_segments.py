"""Actuator-class split of torque / velocity / joint_box, hierarchical dual."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.lagrangian import (
    ACTUATOR_SEGMENTS,
    ACTUATOR_SEGMENT_NAMES,
    CONSTRAINT_GROUPS,
    LIMIT_FAMILIES,
    LIMIT_GROUP_KEYS,
    TERM_FOR_GROUP,
    limit_group_name,
)

DT = 0.02
WRIST_COLS = (8, 9, 15, 16)
ONE_ELEMENT_AMP = 0.10


def test_segments_partition_the_seventeen_joints():
    seen = []
    for _name, joints in ACTUATOR_SEGMENTS:
        seen.extend(joints)
    assert len(seen) == len(set(seen)) == N_DOF == 17
    assert set(seen) == set(G1_UPPER_BODY_JOINTS)
    assert set(LIMIT_GROUP_KEYS) <= set(CONSTRAINT_GROUPS)
    assert CONSTRAINT_GROUPS[-5:] == (
        "torque_peak", "collision_peak", "joint_box_peak", "waist_pr_band",
        "velocity_peak",
    )
    assert TERM_FOR_GROUP["joint_box_waist"] == "joint_box"
    assert TERM_FOR_GROUP["torque_left_wrist"] == "torque"
    for fam in LIMIT_FAMILIES:
        for seg in ACTUATOR_SEGMENT_NAMES:
            assert limit_group_name(fam, seg) in CONSTRAINT_GROUPS


def _ctx(B, *, kl, kr):
    ones = torch.ones(B, 2, dtype=torch.float64)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=ones.clone(),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=torch.float64),
        v_star=torch.full((B, 2, 3), -1.0, dtype=torch.float64),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=torch.float64),
        disk_offset=torch.zeros(B, 2, 3, dtype=torch.float64),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=torch.float64),
        has_half_plane=torch.zeros(B, 2, dtype=torch.float64),
        contact_sigma=torch.full((B, 2), 0.20, dtype=torch.float64),
        shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        start_shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        elev_required=ones.clone(),
    )


def _batch(*, amp: float, H: int = 12, B: int = 4):
    q0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    v0 = torch.zeros(B, N_DOF, dtype=torch.float64)
    wp = q0.unsqueeze(1).repeat(1, H, 1)
    if amp:
        for b in range(B):
            j = WRIST_COLS[b % len(WRIST_COLS)]
            k = 3 + b
            wp[b, k, j] = amp * (-1.0) ** b
    kl = torch.full((B,), H, dtype=torch.long)
    kr = torch.full((B,), H, dtype=torch.long)
    return dict(q0=q0, wp=wp, v0=v0, kl=kl, kr=kr, H=H, B=B)


def _fwd(b, cost, **kw):
    return cost.forward(
        b["q0"], b["wp"], b["v0"],
        _ctx(b["B"], kl=b["kl"], kr=b["kr"]), **kw,
    )


def test_segment_hinge_shares_sum_to_the_global_mean():
    """All-ones segment weights reproduce the pre-split mean half exactly."""
    b = _batch(amp=ONE_ELEMENT_AMP)
    cost = FusedCost(CostConfig(w_torque_peak=1.0), max_horizon=12,
                     device="cpu", dtype=torch.float64, dt=DT)
    out = _fwd(b, cost)
    shares = sum(out.raw[limit_group_name("torque", s)] for s in ACTUATOR_SEGMENT_NAMES)
    assert torch.allclose(shares, out.raw["torque"], atol=1e-12)
    hard = sum(
        out.violations[limit_group_name("torque", s)] for s in ACTUATOR_SEGMENT_NAMES
    )
    # default λ=1: warning + peak product + sum_seg live.  Peak is priced
    # only as λ_peak * w_peak * raw[peak]; do not also add live["torque_peak"].
    assert torch.allclose(
        out.terms["torque"],
        cost.cfg.w_torque * out.raw["torque"]
        + cost.cfg.w_torque_peak * out.raw["torque_peak"]
        + hard,
        atol=1e-12,
    )


def test_hierarchical_residual_is_class_max():
    """One hot wrist element: dual sees the full peak, not peak/3."""
    b = _batch(amp=ONE_ELEMENT_AMP, B=1)
    cost = FusedCost(CostConfig(w_torque_peak=1.0), max_horizon=12,
                     device="cpu", dtype=torch.float64, dt=DT)
    out = _fwd(b, cost)
    peak = out.violations["torque_peak"]
    hier = out.violations["torque_left_wrist"]
    mean = out.violations["torque"]
    assert torch.allclose(hier, peak, rtol=0, atol=1e-12)
    n_valid = out.counts["torque"]
    assert torch.allclose(mean, peak / n_valid.clamp_min(1), rtol=0, atol=1e-12)
    assert float(hier.min()) > 10.0 * float(mean.max())


def test_raising_one_class_lambda_does_not_scale_the_others():
    b = _batch(amp=ONE_ELEMENT_AMP, B=1)
    cost = FusedCost(CostConfig(w_torque_peak=0.0), max_horizon=12,
                     device="cpu", dtype=torch.float64, dt=DT)
    ref = _fwd(b, cost)
    hot = _fwd(b, cost, weights={"torque_left_wrist": 5.0})
    delta = hot.terms["torque"] - ref.terms["torque"]
    expect = (5.0 - 1.0) * ref.violations["torque_left_wrist"]
    assert torch.allclose(delta, expect, atol=1e-12)
    # a clean class's softplus share is untouched
    assert torch.equal(hot.raw["torque_waist"], ref.raw["torque_waist"])
    assert torch.equal(hot.raw["torque"], ref.raw["torque"])
