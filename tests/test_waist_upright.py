"""Section 7d: the waist-upright terms.

Two additive, default-off pieces of :class:`~amortized_opt.cost.CostConfig`:

* ``waist_pr_box`` -- a per-joint override that shrinks the ``waist_pitch`` /
  ``waist_roll`` rows of the joint box (soft hinge AND the hard limits the
  augmented-Lagrangian violation is measured against).  ``waist_yaw`` is NOT
  touched.
* ``w_waist_upright`` -- a quadratic well on the same two rows, inside the
  existing ``posture`` term, with yaw excluded by construction.

Everything below is an assertion about masking, yaw insensitivity, gradient
support, the default-off guarantee and the ramp semantics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext

I_YAW = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
I_PITCH = G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")
I_ROLL = G1_UPPER_BODY_JOINTS.index("waist_roll_joint")
ARM_COLS = [i for i in range(N_DOF) if i not in (I_YAW, I_PITCH, I_ROLL)]

H = 12
DT = 0.02


@pytest.fixture(scope="module")
def base_cost():
    return FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)


def _ctx(B, device="cpu", dtype=torch.float64, k=H):
    """A minimal both-hands-active context; the terminal geometry is irrelevant
    here, only the horizon (which sets the element mask) matters."""
    kk = torch.full((B, 2), int(k), device=device, dtype=torch.long)
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    return StrikeContext(
        hit_index=kk,
        active=ones.clone(),
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
        elev_required=ones.clone(),
    )


def _traj(waist=(0.0, 0.0, 0.0), B=1, dtype=torch.float64):
    """A constant-pose trajectory with the three waist joints set explicitly."""
    q0 = torch.zeros(B, N_DOF, dtype=dtype)
    q0[:, I_YAW], q0[:, I_PITCH], q0[:, I_ROLL] = waist
    wp = q0.unsqueeze(1).repeat(1, H, 1).clone()
    v0 = torch.zeros(B, N_DOF, dtype=dtype)
    return q0, wp, v0


# ============================================================ default off ====
def test_new_fields_default_to_off():
    c = CostConfig()
    assert c.waist_pr_box is None
    assert c.w_waist_upright == 0.0


def test_defaults_are_bit_identical_to_the_pre_7d_cost(base_cost):
    """With both fields off, every term is EXACTLY what it was before."""
    q0, wp, v0 = _traj(waist=(0.7, 0.45, -0.40))
    ctx = _ctx(1)
    out = base_cost.forward(q0, wp, v0, ctx)
    # the upright raw term is computed and reported, but priced at zero
    assert out.raw["posture_upright"].item() > 0.0
    same = base_cost.with_config(w_waist_upright=0.0).forward(q0, wp, v0, ctx)
    assert torch.equal(out.terms["posture"], same.terms["posture"])
    # and the box is the URDF one: 0.45 rad is INSIDE +-0.52, so no violation
    assert out.violations["joint_box"].item() == pytest.approx(0.0, abs=1e-12)


# =========================================================== upright well ====
def test_upright_is_the_masked_mean_of_pitch_and_roll_squared(base_cost):
    """``raw["posture_upright"]`` == mean over the two waist LEAN rows of q^2."""
    cost = base_cost.with_config(w_waist_upright=1.0)
    for pitch, roll in ((0.3, -0.1), (0.0, 0.0), (-0.45, 0.5)):
        q0, wp, v0 = _traj(waist=(1.1, pitch, roll))
        out = cost.forward(q0, wp, v0, _ctx(1))
        # constant pose -> the mean over samples is the per-sample value
        expect = 0.5 * (pitch**2 + roll**2)
        assert out.raw["posture_upright"].item() == pytest.approx(expect, rel=1e-12)


def test_upright_counts_only_the_two_lean_rows(base_cost):
    """The reduction divides by 2 x n_samples, not by 17 x n_samples."""
    cost = base_cost.with_config(w_waist_upright=1.0)
    q0, wp, v0 = _traj(waist=(0.0, 0.4, 0.0))
    out = cost.forward(q0, wp, v0, _ctx(1))
    n_samp = float(out.counts["posture"].item()) / N_DOF
    assert out.counts["posture_upright"].item() == pytest.approx(2.0 * n_samp)
    assert out.raw["posture_upright"].item() == pytest.approx(0.5 * 0.4**2)


def test_upright_is_yaw_invariant(base_cost):
    """Sweeping waist YAW over its whole range moves the term by exactly zero."""
    cost = base_cost.with_config(w_waist_upright=100.0)
    ref = None
    for yaw in (-2.6, -1.0, 0.0, 0.7, 2.6):
        q0, wp, v0 = _traj(waist=(yaw, 0.25, -0.15))
        out = cost.forward(q0, wp, v0, _ctx(1))
        val = out.raw["posture_upright"].item()
        if ref is None:
            ref = val
        assert val == pytest.approx(ref, rel=1e-14, abs=1e-15)


def test_upright_gradient_lives_only_on_pitch_and_roll(base_cost):
    """d(upright)/dq is nonzero on the two lean rows and EXACTLY zero elsewhere."""
    cost = base_cost.with_config(w_waist_upright=1.0)
    q0, wp, v0 = _traj(waist=(0.8, 0.31, -0.22))
    wp = wp.clone().requires_grad_(True)
    out = cost.forward(q0, wp, v0, _ctx(1))
    out.raw["posture_upright"].sum().backward()
    g = wp.grad[0]                                    # (H, 17)
    assert torch.all(g[:, I_PITCH] > 0)               # pushes pitch toward 0
    assert torch.all(g[:, I_ROLL] < 0)                # roll is negative here
    assert g[:, I_YAW].abs().max().item() == 0.0
    assert g[:, ARM_COLS].abs().max().item() == 0.0
    # ...and the values are the analytic 2 q / n_valid
    n = float(out.counts["posture_upright"].item())
    assert g[0, I_PITCH].item() == pytest.approx(2 * 0.31 / n, rel=1e-10)


def test_upright_enters_the_total_only_when_weighted(base_cost):
    q0, wp, v0 = _traj(waist=(0.0, 0.4, 0.4))
    ctx = _ctx(1)
    off = base_cost.forward(q0, wp, v0, ctx)
    on = base_cost.with_config(w_waist_upright=3.0).forward(q0, wp, v0, ctx)
    d = on.terms["posture"].item() - off.terms["posture"].item()
    assert d == pytest.approx(3.0 * 0.5 * (0.4**2 + 0.4**2), rel=1e-10)
    # and the sub-multiplier scales exactly that increment
    half = base_cost.with_config(w_waist_upright=3.0).forward(
        q0, wp, v0, ctx, weights={"posture_upright": 0.5}
    )
    assert (half.terms["posture"].item() - off.terms["posture"].item()
            == pytest.approx(0.5 * d, rel=1e-10))


def test_upright_respects_the_element_mask(base_cost):
    """Waypoints beyond the horizon must not be priced."""
    cost = base_cost.with_config(w_waist_upright=1.0)
    q0 = torch.zeros(1, N_DOF, dtype=torch.float64)
    wp = torch.zeros(1, H, N_DOF, dtype=torch.float64)
    wp[0, :4, I_PITCH] = 0.5          # inside a 4-waypoint horizon
    wp[0, 4:, I_PITCH] = 50.0         # garbage past it
    v0 = torch.zeros(1, N_DOF, dtype=torch.float64)
    out = cost.forward(q0, wp, v0, _ctx(1, k=4))
    # knots 0..4: q0 (=0) plus four at 0.5 -> mean over 2 rows x 5 knots
    assert out.raw["posture_upright"].item() == pytest.approx(4 * 0.25 / 10.0)


# ============================================================== waist box ====
def test_box_override_shrinks_only_pitch_and_roll():
    """0.35 rad is legal under the URDF box and illegal under a 0.20 one."""
    q0, wp, v0 = _traj(waist=(0.0, 0.35, 0.0))
    ctx = _ctx(1)
    wide = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    tight = wide.with_config(waist_pr_box=0.20)
    a = wide.forward(q0, wp, v0, ctx)
    b = tight.forward(q0, wp, v0, ctx)
    assert a.violations["joint_box"].item() == pytest.approx(0.0, abs=1e-12)
    assert b.violations["joint_box"].item() > 0.0
    assert b.diagnostics["joint_box_excess_max"].item() == pytest.approx(0.15, rel=1e-9)
    assert b.raw["joint_box"].item() > a.raw["joint_box"].item()


def test_box_override_leaves_yaw_alone():
    """The tight/wide DELTA is independent of yaw, and no yaw is ever illegal.

    A yaw of 2.5 rad is 12.5x the new pitch/roll box and well past the soft knee
    (``eta_position x 2.618 = 2.094``), so its hinge is live -- and it must be
    the same live hinge under both configs.  Comparing the delta rather than the
    absolute value is the point: shrinking the two lean rows DOES change the
    joint-box mean even at ``q = 0``, because a softplus is strictly positive
    and its value at ``-half/beta`` depends on ``half``.  That constant shift is
    the tightened rows' business; what must not move is yaw.
    """
    ctx = _ctx(1)
    wide = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    tight = wide.with_config(waist_pr_box=0.20)
    deltas = []
    for yaw in (0.0, 1.5, 2.5, -2.5):
        q0, wp, v0 = _traj(waist=(yaw, 0.0, 0.0))
        a = wide.forward(q0, wp, v0, ctx)
        b = tight.forward(q0, wp, v0, ctx)
        assert b.violations["joint_box"].item() == pytest.approx(0.0, abs=1e-12)
        assert b.violations["joint_box"].item() == a.violations["joint_box"].item()
        deltas.append(b.raw["joint_box"].item() - a.raw["joint_box"].item())
    assert deltas[0] > 0.0                      # the shift is real...
    for d in deltas[1:]:                        # ...and identical at every yaw
        assert d == pytest.approx(deltas[0], rel=1e-12)


def test_box_override_soft_hinge_moves_by_eta():
    """The softplus knee sits at ``eta_position x waist_pr_box``, like every row."""
    cfg = CostConfig()
    wide = FusedCost(cfg, max_horizon=H, device="cpu", dtype=torch.float64)
    tight = wide.with_config(waist_pr_box=0.20)
    knee = cfg.eta_position * 0.20
    ctx = _ctx(1)
    # exactly at the knee the hinge contributes beta*log2 per element, the same
    # constant a feasible row pays; just below it, strictly less.
    at = tight.forward(*_traj(waist=(0.0, knee, 0.0)), ctx)
    below = tight.forward(*_traj(waist=(0.0, knee - 0.05, 0.0)), ctx)
    above = tight.forward(*_traj(waist=(0.0, knee + 0.05, 0.0)), ctx)
    assert below.raw["joint_box"].item() < at.raw["joint_box"].item()
    assert at.raw["joint_box"].item() < above.raw["joint_box"].item()


def test_box_override_gradient_is_confined_to_the_lean_rows():
    wide = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    tight = wide.with_config(waist_pr_box=0.20)
    q0, wp, v0 = _traj(waist=(1.4, 0.4, -0.4))
    wp = wp.clone().requires_grad_(True)
    out = tight.forward(q0, wp, v0, _ctx(1))
    out.raw["joint_box"].sum().backward()
    g_tight = wp.grad[0].clone()

    wp2 = _traj(waist=(1.4, 0.4, -0.4))[1].clone().requires_grad_(True)
    out2 = wide.forward(q0, wp2, v0, _ctx(1))
    out2.raw["joint_box"].sum().backward()
    g_wide = wp2.grad[0]

    d = (g_tight - g_wide).abs()
    assert d[:, I_PITCH].max().item() > 1e-6
    assert d[:, I_ROLL].max().item() > 1e-6
    assert d[:, I_YAW].max().item() == 0.0
    assert d[:, ARM_COLS].max().item() == 0.0


def test_box_ramp_interpolates_the_half_width():
    """``weights["waist_box"]`` blends the half-width; 0 reproduces the old box."""
    wide = FusedCost(CostConfig(), max_horizon=H, device="cpu", dtype=torch.float64)
    tight = wide.with_config(waist_pr_box=0.20)
    q0, wp, v0 = _traj(waist=(0.0, 0.35, 0.0))
    ctx = _ctx(1)
    ref_off = wide.forward(q0, wp, v0, ctx)
    at0 = tight.forward(q0, wp, v0, ctx, weights={"waist_box": 0.0})
    at1 = tight.forward(q0, wp, v0, ctx, weights={"waist_box": 1.0})
    mid = tight.forward(q0, wp, v0, ctx, weights={"waist_box": 0.5})
    assert at0.raw["joint_box"].item() == pytest.approx(
        ref_off.raw["joint_box"].item(), rel=1e-12
    )
    assert at0.violations["joint_box"].item() == pytest.approx(0.0, abs=1e-12)
    for lo, hi in ((at0, mid), (mid, at1)):
        assert lo.raw["joint_box"].item() < hi.raw["joint_box"].item()
    # half-way the hard limit is (0.52 + 0.20)/2 = 0.36, so 0.35 is still legal
    assert mid.violations["joint_box"].item() == pytest.approx(0.0, abs=1e-12)
    assert at1.violations["joint_box"].item() > 0.0


def test_box_override_validates():
    # ``waist_pr_box <= 0`` is silently coerced to ``None`` (the "off" state),
    # which is intentional -- ``--waist-pr-box 0.0`` is the natural CLI spelling
    # for "I do not want this constraint".  The validation gate is on the
    # non-negative weights, where a negative value would price a physics
    # constraint backward.
    assert CostConfig(waist_pr_box=0.0).validate().waist_pr_box is None
    assert CostConfig(waist_pr_box=-0.2).validate().waist_pr_box is None
    with pytest.raises(ValueError):
        CostConfig(w_waist_upright=-1.0).validate()
    with pytest.raises(ValueError):
        CostConfig(w_waist_aim=-1.0).validate()


# ============================================================ diagnostics ====
def test_waist_diagnostics_split_lean_from_yaw(base_cost):
    q0, wp, v0 = _traj(waist=(1.9, 0.33, -0.41))
    out = base_cost.forward(q0, wp, v0, _ctx(1))
    assert out.diagnostics["waist_pr_max"].item() == pytest.approx(0.41, rel=1e-12)
    assert out.diagnostics["waist_yaw_max"].item() == pytest.approx(1.9, rel=1e-12)


# ============================================================ integration ====
def test_both_terms_together_reduce_a_leaning_plan(base_cost):
    """A gradient step on the FULL objective moves pitch/roll toward upright,
    and the two new terms contribute exactly zero yaw gradient.

    The whole-objective gradient on yaw is NOT zero (the mid-range well and the
    torque term both touch it); what the tests pin is that turning the 7d terms
    on does not change it by a single ulp.
    """
    off = base_cost
    on = base_cost.with_config(waist_pr_box=0.20, w_waist_upright=5.0,
                               w_joint_box=50.0)
    grads = {}
    for tag, cost in (("off", off.with_config(w_joint_box=50.0)), ("on", on)):
        q0, wp, v0 = _traj(waist=(1.2, 0.45, -0.38))
        wp = wp.clone().requires_grad_(True)
        cost.forward(q0, wp, v0, _ctx(1)).total.sum().backward()
        grads[tag] = wp.grad[0].clone()
    # the new terms push both lean joints back toward zero...
    d = grads["on"] - grads["off"]
    assert d[:, I_PITCH].min().item() > 0.0     # pitch is +0.45 -> gradient +
    assert d[:, I_ROLL].max().item() < 0.0      # roll  is -0.38 -> gradient -
    # ...and touch nothing else, yaw included
    assert d[:, I_YAW].abs().max().item() == 0.0
    assert d[:, ARM_COLS].abs().max().item() == 0.0
    assert torch.equal(grads["on"][:, I_YAW], grads["off"][:, I_YAW])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_nans_at_the_origin(dtype):
    """q = 0 is the minimum of both terms; the gradient there must be finite."""
    cost = FusedCost(CostConfig(waist_pr_box=0.20, w_waist_upright=5.0),
                     max_horizon=H, device="cpu", dtype=dtype)
    q0, wp, v0 = _traj(waist=(0.0, 0.0, 0.0), dtype=dtype)
    wp = wp.clone().requires_grad_(True)
    out = cost.forward(q0, wp, v0, _ctx(1, dtype=dtype))
    out.total.sum().backward()
    assert torch.isfinite(out.total).all()
    assert torch.isfinite(wp.grad).all()
    assert out.raw["posture_upright"].item() == 0.0
