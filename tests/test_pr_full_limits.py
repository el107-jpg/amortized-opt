"""Per-JOINT exemption from the limit-hinge ramp (waist pitch/roll).

The phase-2 curriculum ramps the ``torque`` / ``velocity`` / ``joint_box``
hinges from 5 % to 100 % so the decoder learns to REACH before the limits
close.  Measured on ``runs/phase2_19m_ctxv2``, the model spends that window
pumping the WAIST PITCH/ROLL: excursion 0.24 -> 0.64 rad and rate 1.8 -> 3.0
rad/s over steps 1 k -> 10 k, while the PRICED yaw rate fell 2.5 -> 1.1 -- the
momentum migrated into the unpriced degrees of freedom.  Static RNEA says the
discount is a lie for those two joints: against the 25 N*m waist-pitch rating,
upright already costs 13.1 N*m, a legal 0.25 rad lean 20.1 N*m, and the observed
0.64 rad 28.4 N*m -- over rating before any acceleration.

So :attr:`WeightSchedule.full_limit_joints` pins those two joints' hinge alpha
at 1.0 from step 0 while everything else ramps, via a ``(D,)``
``weights["limit_alpha"]`` vector the cost applies PER ELEMENT before each
hinge's reduction.

What these tests pin:

1. ``limit_alpha=None`` is bit-exact with the pre-exemption cost, on all three
   groups (and equal to an explicit all-ones vector).
2. The vector prices per JOINT: a waist-pitch hinge at 1.0, an
   identical-magnitude arm hinge at the ramp's alpha -- checked against a
   hand-computed ``joint_box`` term and against the exact per-joint
   decomposition of the ``torque`` term.
3. Every RAW violation and diagnostic is bit-identical on or off, which is what
   makes the augmented Lagrangian blind to the feature.
4. The schedule builds the vector correctly at step 0 and at ``ramp_end``, and
   ``_wtensors`` moves the ramp into it WITHOUT leaving a copy in the scalar.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.train_cost import WeightSchedule, _wtensors

DT = 0.02
H = 12
DTYPE = torch.float64

PITCH = G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")
ROLL = G1_UPPER_BODY_JOINTS.index("waist_roll_joint")
ELBOW = G1_UPPER_BODY_JOINTS.index("left_elbow_joint")
#: The three groups the exemption touches.  ``torque_peak`` / ``collision_peak``
#: are deliberately NOT here -- they keep their own group schedules.
GROUPS = ("torque", "velocity", "joint_box")


# ------------------------------------------------------------------ setup --
def _ctx(B, *, kl, kr):
    """The minimal synthetic context ``tests/test_torque_peak_al.py`` uses."""
    ones = torch.ones(B, 2, dtype=DTYPE)
    return StrikeContext(
        hit_index=torch.stack([kl, kr], dim=1),
        active=ones.clone(),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=DTYPE),
        v_star=torch.full((B, 2, 3), -1.0, dtype=DTYPE),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE).expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=DTYPE),
        disk_offset=torch.zeros(B, 2, 3, dtype=DTYPE),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=DTYPE),
        has_half_plane=torch.zeros(B, 2, dtype=DTYPE),
        contact_sigma=torch.full((B, 2), 0.20, dtype=DTYPE),
        shaft_allowed=torch.zeros(B, 2, dtype=DTYPE),
        start_shaft_allowed=torch.zeros(B, 2, dtype=DTYPE),
        elev_required=ones.clone(),
    )


def _cost(**kw):
    return FusedCost(CostConfig(**kw), max_horizon=H, device="cpu",
                     dtype=DTYPE, dt=DT)


def _fwd(cost, q0, wp, v0, **kw):
    B = q0.shape[0]
    k = torch.full((B,), H, dtype=torch.long)
    return cost.forward(q0, wp, v0, _ctx(B, kl=k, kr=k), **kw)


def _random_batch(B=3, seed=0):
    """Random plans big enough to violate all three limit groups at once."""
    g = torch.Generator().manual_seed(seed)
    q0 = 0.20 * torch.randn(B, N_DOF, generator=g, dtype=DTYPE)
    v0 = 0.50 * torch.randn(B, N_DOF, generator=g, dtype=DTYPE)
    wp = q0.unsqueeze(1) + 0.60 * torch.randn(B, H, N_DOF, generator=g, dtype=DTYPE)
    return q0, wp, v0


def _hold(q):
    """A motionless plan holding ``q`` -- ``joint_box`` alone, hand-computable."""
    q0 = q.clone()
    return q0, q0.unsqueeze(1).repeat(1, H, 1), torch.zeros_like(q0)


# ============================================ 1. the None path is bit-exact ==
def test_none_is_bit_exact_with_an_all_ones_vector():
    """The default must not merely agree with today -- it must be the same bits.

    ``limit_alpha=None`` skips the multiply entirely rather than multiplying by
    ones, so this also proves the skip is what the ones-vector would have done.
    """
    cost = _cost(w_torque_peak=1.0, w_collision_peak=1.0)
    q0, wp, v0 = _random_batch()
    off = _fwd(cost, q0, wp, v0)
    ones = _fwd(cost, q0, wp, v0, weights={"limit_alpha": torch.ones(N_DOF, dtype=DTYPE)})
    for g in GROUPS:
        assert torch.equal(off.terms[g], ones.terms[g]), g
        assert torch.equal(off.raw[g], ones.raw[g]), g
    assert torch.equal(off.total, ones.total)


def test_none_path_unchanged_by_an_unrelated_weights_dict():
    """A ``weights`` dict WITHOUT ``limit_alpha`` still takes the skip path."""
    cost = _cost()
    q0, wp, v0 = _random_batch(seed=7)
    a = _fwd(cost, q0, wp, v0, weights=None)
    b = _fwd(cost, q0, wp, v0, weights={"torque": 1.0, "velocity": 1.0})
    for g in GROUPS:
        assert torch.equal(a.terms[g], b.terms[g]), g


# ==================================== 2. the vector prices per JOINT, exactly ==
def test_joint_box_hand_computed_two_joints_one_ramp():
    """Identical-magnitude box excess on waist_pitch and on an arm joint.

    Both joints are placed the same 0.30 rad outside their OWN soft half-width,
    so their hinges agree to 1e-14 relative (the two half-widths differ, so the
    sums round differently); the only thing that can separate their
    contributions to the term is the alpha vector.  The whole
    term is then recomputed here from ``q``, ``q_mid``, ``q_half`` and the
    softplus -- no reuse of the cost's own reduction.
    """
    alpha = 0.05
    cost = _cost(waist_pr_box=None)
    cfg = cost.cfg
    half = cfg.eta_position * cost.q_half            # (D,)
    mid = cost.q_mid
    excess = 0.30                                    # rad outside the soft box

    q = mid.clone().unsqueeze(0)                     # (1, D), centred
    q[0, PITCH] = mid[PITCH] + half[PITCH] + excess
    q[0, ELBOW] = mid[ELBOW] + half[ELBOW] + excess
    q0, wp, v0 = _hold(q)

    vec = torch.full((N_DOF,), alpha, dtype=DTYPE)
    vec[PITCH] = vec[ROLL] = 1.0
    out = _fwd(cost, q0, wp, v0, weights={"limit_alpha": vec})

    # the hinge, per joint, from q alone (constant over knots by construction)
    dev = (q[0] - mid).abs()
    h = F.softplus((dev - half) / cfg.beta_position) * cfg.beta_position
    # ... and the identical-magnitude claim, before anything is priced
    assert h[PITCH].item() == pytest.approx(h[ELBOW].item(), rel=1e-14)

    cnt = out.counts["joint_box"][0]
    n_wp = float(cnt.item()) / N_DOF
    assert n_wp == int(n_wp), "expected a full, rectangular mask"
    expect_raw = float((vec * h).sum().item() / float(N_DOF))
    assert out.raw["joint_box"][0].item() == pytest.approx(expect_raw, rel=1e-12)
    from amortized_opt.lagrangian import ACTUATOR_SEGMENT_NAMES, limit_group_name
    hard = sum(
        out.violations[limit_group_name("joint_box", s)][0].item()
        for s in ACTUATOR_SEGMENT_NAMES
    )
    assert out.terms["joint_box"][0].item() == pytest.approx(
        cfg.w_joint_box * expect_raw + hard, rel=1e-12
    )

    # the point of the exercise: same hinge, 1/alpha apart in price
    priced_pitch = 1.0 * h[PITCH].item()
    priced_elbow = alpha * h[ELBOW].item()
    assert priced_pitch == pytest.approx(priced_elbow / alpha, rel=1e-12)


def test_torque_term_is_the_exact_per_joint_sum():
    """``warn(v) == sum_j v_j * warn(e_j)``: the alpha really is PER ELEMENT.

    Linearity holds on the WARNING (softplus) half.  The hard λ×ReLU wall
    is invariant to ``limit_alpha``, so it is subtracted before the identity.
    """
    from amortized_opt.lagrangian import ACTUATOR_SEGMENT_NAMES, limit_group_name

    def _warn(out):
        hard = sum(
            out.violations[limit_group_name("torque", s)]
            for s in ACTUATOR_SEGMENT_NAMES
        )
        return out.terms["torque"] - hard

    alpha = 0.05
    cost = _cost()                       # w_torque_peak = 0: the mean half only
    q0, wp, v0 = _random_batch(seed=3, B=2)
    vec = torch.full((N_DOF,), alpha, dtype=DTYPE)
    vec[PITCH] = vec[ROLL] = 1.0

    per_joint = []
    for j in range(N_DOF):
        e = torch.zeros(N_DOF, dtype=DTYPE)
        e[j] = 1.0
        per_joint.append(_warn(_fwd(cost, q0, wp, v0, weights={"limit_alpha": e})))
    stack = torch.stack(per_joint, dim=0)                 # (D, B)
    expect = (vec.view(-1, 1) * stack).sum(dim=0)

    got_out = _fwd(cost, q0, wp, v0, weights={"limit_alpha": vec})
    got = _warn(got_out)
    assert torch.allclose(got, expect, rtol=1e-11, atol=1e-13)

    # and the discriminating claim: the waist-pitch column enters at 1.0 while
    # the elbow column enters at the ramp's alpha, per unit of hinge.
    ramped = _warn(_fwd(
        cost, q0, wp, v0,
        weights={"limit_alpha": torch.full((N_DOF,), alpha, dtype=DTYPE)},
    ))
    lift = got - ramped
    assert torch.allclose(lift, (1.0 - alpha) * (stack[PITCH] + stack[ROLL]),
                          rtol=1e-11, atol=1e-13)
    assert bool((stack[ELBOW] > 0).all())        # the arm column is priced...
    assert bool((lift > 0).all())                # ...and it is NOT in the lift


def test_all_three_groups_respond_to_the_vector():
    """Velocity and joint_box take the same per-element alpha as torque."""
    alpha = 0.05
    cost = _cost()
    q0, wp, v0 = _random_batch(seed=11)
    lo = torch.full((N_DOF,), alpha, dtype=DTYPE)
    vec = lo.clone()
    vec[PITCH] = vec[ROLL] = 1.0
    a = _fwd(cost, q0, wp, v0, weights={"limit_alpha": lo})
    b = _fwd(cost, q0, wp, v0, weights={"limit_alpha": vec})
    for g in GROUPS:
        assert bool((b.terms[g] > a.terms[g]).all()), g


def test_peak_subterms_ignore_the_vector():
    """``torque_peak`` / ``collision_peak`` keep their OWN group schedules."""
    cost = _cost(w_torque_peak=1.0, w_collision_peak=1.0)
    q0, wp, v0 = _random_batch(seed=5)
    off = _fwd(cost, q0, wp, v0)
    vec = torch.full((N_DOF,), 0.05, dtype=DTYPE)
    vec[PITCH] = vec[ROLL] = 1.0
    on = _fwd(cost, q0, wp, v0, weights={"limit_alpha": vec})
    assert torch.equal(off.raw["torque_peak"], on.raw["torque_peak"])
    assert torch.equal(off.raw["collision_peak"], on.raw["collision_peak"])


# ================== 3. the augmented-Lagrangian path cannot see the feature ==
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_violations_and_diagnostics_bit_identical(seed):
    """The AL integrates RAW violations; the ramp only ever scaled the PENALTY.

    Every ``v_*`` metric and every diagnostic must be the same bits with the
    exemption on or off -- otherwise the multipliers would drift between the two
    configurations and an A/B would not be an A/B.
    """
    cost = _cost(w_torque_peak=1.0, w_collision_peak=1.0)
    q0, wp, v0 = _random_batch(seed=seed)
    off = _fwd(cost, q0, wp, v0)
    vec = torch.full((N_DOF,), 0.05, dtype=DTYPE)
    vec[PITCH] = vec[ROLL] = 1.0
    on = _fwd(cost, q0, wp, v0, weights={"limit_alpha": vec})
    assert set(off.violations) == set(on.violations)
    for k in off.violations:
        assert torch.equal(off.violations[k], on.violations[k]), f"violations[{k}]"
    assert set(off.diagnostics) == set(on.diagnostics)
    for k in off.diagnostics:
        assert torch.equal(off.diagnostics[k], on.diagnostics[k]), f"diagnostics[{k}]"
    for k in off.counts:
        assert torch.equal(off.counts[k], on.counts[k]), f"counts[{k}]"


# ============================================== 4. the schedule and _wtensors ==
def test_schedule_vector_step0_and_end():
    s = WeightSchedule(ramp_start=0, ramp_end=1_000, lo_limits=0.05)
    v0 = s.limit_alpha_vector(0, G1_UPPER_BODY_JOINTS)
    assert v0.shape == (N_DOF,)
    ones = {i for i, x in enumerate(v0) if x == 1.0}
    assert ones == {PITCH, ROLL}, "exactly the named joints, nobody else"
    others = np.delete(v0, [PITCH, ROLL])
    assert np.allclose(others, s.limit_alpha(0))
    assert others[0] == pytest.approx(0.05)

    end = s.limit_alpha_vector(1_000, G1_UPPER_BODY_JOINTS)
    assert np.allclose(end, 1.0), "converges to all-ones at ramp_end"


def test_schedule_vector_is_none_when_nothing_is_exempt():
    """``()`` is the escape hatch, and it must reach the BIT-EXACT cost path."""
    s = WeightSchedule(full_limit_joints=())
    assert s.limit_alpha_vector(0, G1_UPPER_BODY_JOINTS) is None
    # a name the model does not have is not an error, just nothing to exempt
    s2 = WeightSchedule(full_limit_joints=("no_such_joint",))
    assert s2.limit_alpha_vector(0, G1_UPPER_BODY_JOINTS) is None


def test_wtensors_moves_the_ramp_into_the_vector_without_copying_it():
    """The ramp lives in the vector; constraint scalars are λ only.

    ``mult["torque"]`` still records ``lim`` (the warning ramp), but
    :func:`_wtensors` does not fold it into λ.  The product
    ``λ * vec[non-exempt]`` is the warning scale at that joint; λ itself
    is the hard-wall price.
    """
    s = WeightSchedule(ramp_start=0, ramp_end=1_000, lo_limits=0.05)
    lam = {"torque": 3.0, "velocity": 2.0, "joint_box": 5.0}
    mult = s.multipliers(0)
    plain = _wtensors(mult, lam, "cpu", torch.float64)
    fancy = _wtensors(mult, lam, "cpu", torch.float64, sched=s,
                      joint_names=G1_UPPER_BODY_JOINTS, step=0)
    assert "limit_alpha" not in plain
    vec = fancy["limit_alpha"]
    assert vec.shape == (N_DOF,)
    for g in GROUPS:
        assert fancy[g].item() == pytest.approx(lam[g])
        assert plain[g].item() == pytest.approx(lam[g])
        assert vec[ELBOW].item() == pytest.approx(s.limit_alpha(0), rel=1e-12)
        assert vec[PITCH].item() == pytest.approx(1.0)

    # groups OUTSIDE the three: peaks keep alpha * λ; others stay as they were
    for k in ("collision", "smooth", "terminal", "torque_peak", "collision_peak"):
        assert torch.equal(plain[k], fancy[k]), k


def test_wtensors_without_a_schedule_is_unchanged():
    s = WeightSchedule()
    lam = {"torque": 3.0}
    a = _wtensors(s.multipliers(0), lam, "cpu", torch.float64)
    b = _wtensors(s.multipliers(0), lam, "cpu", torch.float64,
                  sched=WeightSchedule(full_limit_joints=()),
                  joint_names=G1_UPPER_BODY_JOINTS, step=0)
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k
