"""The fused differentiable trajectory cost (section 3 of the plan).

One :class:`~amortized_opt.fk.ChainFK` pass over the sub-sampled spline feeds
every geometric term; :class:`~amortized_opt.rnea.BatchedRNEA` runs once on the
same states for the torque term; the spline coefficients give the velocity
maxima and the smoothness integrals in closed form.  Everything is batched,
differentiable, mask-aware and free of device->host syncs.

Term inventory
--------------

============== ====================================================
group          what it prices
============== ====================================================
``terminal``   contact point inside the legal hit region, impact
               velocity against the commanded one, stick elevation
               band.  Per hand, at that hand's own hit index.
``collision``  drum bodies, under-drum volumes and self (torso
               capsule + head sphere) against tip / shaft / wrist
               spheres, with the harness's target-drum and
               departure exemptions.
``torque``     RNEA (armature ON) at the sub-sampled states against
               ``eta_torque`` x the Isaac effort limits.  Priced twice: a MEAN
               over the (knot, joint) hinge elements, plus an optional
               SMOOTH-MAX over the same elements (``w_torque_peak``) because the
               harness gate is a zero-tolerance max -- see :meth:`FusedCost._torque`.
               The two halves are SEPARATE constraint groups (``torque`` and
               ``torque_peak``) with independent multipliers.
``effort``     optional ``w_effort`` x mean ``tau^2`` on the same RNEA
               torques -- an energy proxy, faded to zero over the
               last ``effort_fade_wp`` waypoints before each hand's
               hit so it cannot fight the impact-speed term.  A plain
               objective: no multiplier.
``velocity``   per-segment max ``|qd|`` against ``eta_velocity`` x
               the speed ratings.
``joint_box``  sub-sampled ``q`` against ``eta_position`` x the
               posture-narrowed URDF box, optionally with the waist
               PITCH/ROLL rows shrunk to ``+-waist_pr_box`` (yaw
               untouched -- the torso must spin to reach the kit).
``smooth``     closed-form ``int v^2 / a^2 / j^2`` per segment.
``posture``    mid-range well, elbow-down preference, anti-axial
               twist, and an optional UPRIGHT well on waist
               pitch/roll (``w_waist_upright``; yaw excluded).
============== ====================================================

The first five are *constraint* groups: they carry an augmented-Lagrangian
multiplier (see :mod:`amortized_opt.lagrangian`) and report a raw violation
measure alongside the softplus penalty -- six multipliers in all, since the
torque group's max-shaped half carries its own (``torque_peak``).  The last two
are plain objective terms.

Hinges
------

Every limit is ``softplus((x - lim) / beta) * beta`` as the plan specifies, with
``lim`` a *conservative* fraction of the hard limit (torque 80 %, velocity 70 %,
joint box 80 %).  Softplus is strictly positive, so a perfectly feasible sample
still pays ``beta * log 2`` per element; that constant is what buys gradient on
the safe side of the limit, and it is why the diagnostics also report the raw
``relu(x - hard_limit)`` excess, which *is* zero when feasible and is what the
Lagrange multipliers are driven by.

Normalization
-------------

Every term reduces to a **mean over valid elements**, and the element counts are
returned in ``CostTerms.counts`` so a training loop can renormalize across
horizon lengths (the plan's "loss normalization across horizon lengths").

The one place that normalization actively *hurts* is the torque group, whose
acceptance gate is a max and not a mean; :func:`smooth_max` and the
``w_torque_peak`` weight exist to put a max-shaped signal back into the loss.
See :meth:`FusedCost._torque`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    CONTROL_DT,
    G1_UPPER_BODY_JOINTS,
    HEAD_SPHERE,
    PELVIS_IN_ENV,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
    WRIST_RADIUS,
    ready_joint_pos,
)
from .fk import (
    ChainFK,
    stick_axis_local,
    stick_mount_transform,
    stick_shaft_offsets,
    stick_tip_offset,
)
from .limits import (
    ETA_POSITION,
    ETA_TORQUE,
    ETA_VELOCITY,
    isaac_effort_limits,
    isaac_velocity_limits,
    urdf_position_limits,
)
from .rnea import BatchedRNEA
from .sdf import (
    DrumSceneSpec,
    point_capsule_sdf,
    point_cylinder_sdf,
    point_sphere_sdf,
)
from .strike import (
    STICK_ELEV_MAX_DEG,
    STICK_ELEV_MIN_DEG,
    STICK_TIP_SIGMA,
    contact_sigma_candidates,
    shaft_hit_allowed,
    strike_elevation_required,
)
from .tspline import TrajectorySpline, TrajectorySplineBuilder

__all__ = [
    "CostConfig",
    "CostTerms",
    "FusedCost",
    "StrikeContext",
    "axial_twist_directions",
    "drum_heads_from_context",
    "smooth_max",
    "strike_context_from_cases",
]

_HANDS = ("left", "right")
#: Contact arc coordinates probed when the drum may be struck by the shaft.
#: ``contact_sigma_candidates("cymbal_left")`` -> (0.10, 0.15, 0.20, 0.05), i.e.
#: the bead plus the 15 cm of shaft behind it the plan calls out.
SHAFT_SIGMAS: tuple[float, ...] = contact_sigma_candidates("cymbal_left")


# ============================================================== config ======
@dataclass
class CostConfig:
    """Weights, margins and hinge temperatures.

    Defaults are informed by what ``TrajOptConfig.validate()`` actually *leaves
    enabled* under ``simplified_cost`` (which is the repo default), not by the
    declared dataclass fields -- see the per-field notes.  The absolute
    magnitudes are NOT trajopt's: trajopt prices hard constraints through
    ``mu_*`` slack penalties in the 1e5..1e9 range because they must outbid a
    jerk Hessian near 3e8.  Here the constraint groups are balanced by the
    augmented-Lagrangian multipliers instead, so the weights below are the
    *relative* prices at lambda = 1 and are O(1)-ish.
    """

    # ---------------------------------------------------------- sampling --
    #: Sub-samples per spline segment for the FK / SDF / RNEA pass.  1 = the
    #: waypoint knots themselves, which is the plan's "the waypoints are dense
    #: enough" reading and the only setting that fits the 20 ms budget at
    #: batch 256 x 50 (see ``tests/test_cost_speed.py``).
    n_sub: int = 1
    #: Probes per segment for the velocity maximum when
    #: ``velocity_max_mode == "subsample"``.
    n_vel_probe: int = 5
    #: States the RNEA torque term is evaluated at.
    #:
    #: ``"knot_fd"`` (default) uses the knots with the SCORER's finite-difference
    #: rates -- which is also what the first-order-hold PD plant is commanded to
    #: do between waypoints, so cost and scorer agree by construction.
    #: ``"spline"`` uses the C2 spline's own qd/qdd at the sub-samples.  The two
    #: disagree materially: on the recorded trajopt baseline the spline's |qdd|
    #: is ~2.3x the finite difference wherever the QP left a one-knot zig-zag,
    #: and the spline convention reports a median 1.1 N*m torque overshoot on
    #: trajectories the scorer calls clean.  See DECISIONS.
    physics_mode: str = "knot_fd"
    #: ``"analytic"`` locates the per-segment ``|qd|`` maximum exactly (velocity
    #: is quadratic in the segment parameter, so it is one clamped division);
    #: ``"subsample"`` is the GPU review's recommendation and under-estimates by
    #: up to 7.4 % at 5 probes.  Analytic is both cheaper and exact HERE -- this
    #: is not the shelved torque locator, which needs an RNEA per candidate.
    velocity_max_mode: str = "analytic"

    # ------------------------------------------------------ conservatism --
    eta_torque: float = ETA_TORQUE      # 0.80
    eta_velocity: float = ETA_VELOCITY  # 0.70
    eta_position: float = ETA_POSITION  # 0.80
    #: ``"posture"`` uses ``G1_DRUM_POSTURE_LIMITS_RAD``-narrowed URDF limits --
    #: the box trajopt is actually constrained to and the one the harness's
    #: ``ok_joint_limits`` checks.  ``"hard"`` uses the raw URDF box.
    box_source: str = "posture"
    #: Optional DC-motor style ceiling ``tau_max(qd) = sat * (1 - |qd|/v_lim)``
    #: clipped to the effort limit.  OFF by default: the G1's actuator groups are
    #: ``ImplicitActuatorCfg`` / ``IdealPDActuatorCfg``, neither of which models
    #: back-EMF derating, and no ``saturation_effort`` exists anywhere in the
    #: config to calibrate one.  See DECISIONS.
    torque_derate: str = "none"  # "none" | "linear"
    torque_saturation_scale: float = 1.0  # sat = scale * effort_limit

    # ----------------------------------------------------------- hinges ---
    beta_torque: float = 1.0        # N*m
    beta_velocity: float = 0.5      # rad/s
    beta_position: float = 0.02     # rad
    beta_collision: float = 0.005   # m
    beta_terminal: float = 0.005    # m
    beta_elev: float = 0.05         # rad
    collision_margin: float = 0.005  # m, matches the harness's margin count

    # -------------------------------------------------------- terminal ----
    #: Distance from the contact point to the legal hit region.  This is the
    #: plan's "distance from the closest point within the drum head ring".
    w_terminal_region: float = 200.0
    #: Soft pull to the AIM point once inside the region.  ``validate()`` zeroes
    #: trajopt's ``w_strike_position``; kept at 0 here for the same reason (the
    #: disk is what the RL detector scores) but exposed.
    w_terminal_aim: float = 0.0
    #: Impact velocity, split normal / tangential to the commanded direction.
    w_terminal_velocity: float = 20.0
    terminal_a_normal: float = 1.0
    #: ``validate()`` zeroes the tangential component.  Kept at the DECLARED
    #: ratio (7 / 50 = 0.14) instead: the harness scores speed MAGNITUDE and
    #: direction cosine, both of which a zero tangential weight leaves unpriced.
    terminal_a_tangential: float = 0.14
    w_terminal_elev: float = 5.0
    terminal_axial_tol: float = 0.02   # m, harness's on-head axial band
    elev_min_deg: float = STICK_ELEV_MIN_DEG
    elev_max_deg: float = STICK_ELEV_MAX_DEG
    #: ``"softmin"`` scores the best contact point along the legal stick span
    #: for shaft-hittable drums (the plan's "the entire stick up to ~15 cm from
    #: the tip will work"); ``"spec"`` uses the strike's own ``contact_sigma``,
    #: which is what the harness scores.
    contact_mode: str = "softmin"
    contact_softmin_temp: float = 0.01  # m
    #: ``"fd_coarse"`` = ``(p(q_k) - p(q_{k-1})) / dt`` -- the harness /
    #: FOH-plant convention.  ``"fd_fine"`` uses the spline sub-sample spacing.
    impact_velocity_mode: str = "fd_coarse"

    # ------------------------------------------------------- collision ----
    w_collision_drum: float = 100.0
    w_collision_under: float = 100.0
    w_collision_self: float = 100.0
    #: Price on the SMOOTH-MAX of the collision hinge, alongside the three
    #: mean-reduced sub-costs above.  The exact counterpart of
    #: ``w_torque_peak``, and for the same reason: ``ok_collision`` is a
    #: zero-tolerance check that no contact point penetrates any drum, while
    #: the mean is diluted by ``n_seen`` = ``2 x S x P x D`` -- ~2 100 elements
    #: at bucket 25 and ~4 200 at bucket 50 -- of which the overwhelming
    #: majority are stick points nowhere near the drum they are being scored
    #: against.  A single point 30 mm inside a drum head therefore moves the
    #: mean by ~1e-5 while failing the case outright.
    #:
    #: UNLIKE the three means above, this half reduces the raw PENETRATION, not
    #: the margin-softplus -- ``smooth_max`` over ``max(-sdf, 0)``.  Peaking the
    #: hinge was tried and measured worse than the mean it was meant to fix
    #: (2.9x gate separation against 3.9x): the softplus floors at 0.0066 m for
    #: an element at zero clearance, every clean sample pays it, and a max over
    #: a floor is mostly floor.  The penetration is exactly zero on a clear
    #: element and separates 5.0x.  It is also what ``ok_collision`` thresholds
    #: -- ``pen_count == 0``, no margin, no smoothing.  The margin and the
    #: smoothing belong to the mean, whose job is dense shaping on the safe
    #: side; this half's job is the gate.  See :meth:`FusedCost._collision`.
    #:
    #: The analogy to ``torque_peak`` is REAL but WEAKER, and the difference is
    #: worth stating: torque violations are genuinely sparse (~0.27 % of ~850
    #: elements at the median), so a mean truly cannot see them, and the peak
    #: separates that gate 26x against the mean's 3.4x.  Collision violations
    #: are NOT sparse -- measured on the champion's 243-case grid, a failing
    #: case has a median of 10 penetrating elements at a median 35 mm depth --
    #: so the mean does partially see them.  Expect a smaller win.
    #:
    #: ZERO by default, so every existing checkpoint and the section-3
    #: validation weights stay bit-identical, and the smooth-max is not even
    #: computed.  Scale: ``raw`` here is METRES OF PENETRATION -- 0 on a clear
    #: sample, ~0.027 at the median failing case -- so a weight around 500 puts
    #: the failing half around 14 against a live ``terminal`` of ~65 and a
    #: collision mean of ~1, while a clean sample pays exactly nothing.  Priced
    #: by its OWN augmented-Lagrangian group (``collision_peak``), NOT by
    #: ``weights["collision"]``.
    w_collision_peak: float = 0.0
    #: ``"pnorm"`` / ``"lse"``, exactly as ``torque_peak_mode``.
    collision_peak_mode: str = "pnorm"
    #: p-norm exponent.  The reduction runs on ``h / max(h)`` so the exponent
    #: never sees a value above 1 and cannot overflow fp32.
    collision_peak_p: float = 12.0
    #: LSE temperature, METRES.  Only read when ``collision_peak_mode == "lse"``.
    #: Two decades below ``torque_peak_temp`` because this hinge is in metres,
    #: not newton-metres.
    collision_peak_temp: float = 5.0e-4
    #: Waypoints at the end of a hand's horizon whose contact geoms are exempt
    #: from the target drum, and at the start exempt from the drum the pose is
    #: departing.  Both mirror ``TrajectoryScorer``.
    strike_exempt_waypoints: int = 2
    start_exempt_waypoints: int = 2
    shaft_samples: int = 4

    # ---------------------------------------------------------- limits ----
    w_torque: float = 1.0
    #: Price on the SMOOTH-MAX of the torque hinge, alongside ``w_torque``'s
    #: mean.  ``ok_torque`` is a zero-tolerance max against the hard effort
    #: limits, but the mean is diluted by ~850 elements: on the recorded eval
    #: CSVs of the phase-2 failures the median case is over on ONE (knot, joint)
    #: element out of ~850 (0.27 %; p90 0.51 %, worst 1.5 %) by a median
    #: 0.87 N*m -- which moves the mean by ~0.3 % and fails the whole case.  The
    #: mean literally cannot see what the gate checks.  ZERO by default so the
    #: section-3 validation weights (pinned by the cost-validation tests) and
    #: every existing checkpoint keep bit-identical behaviour; a training config
    #: turns it on.  This half is scaled by its OWN group multiplier
    #: ``weights["torque_peak"]`` -- an augmented-Lagrangian group driven by the
    #: MAX raw violation, NOT by ``weights["torque"]``, which integrates a mean
    #: ~n_valid times smaller and has no idea the peak exists.
    #: See :meth:`FusedCost._torque`.
    w_torque_peak: float = 0.0
    #: ``"pnorm"`` -- ``(sum h^p)^(1/p)``, scale-equivariant, always an
    #: OVER-estimate of the max by at most ``n_valid^(1/p)`` (1.9x at p = 12,
    #: n = 850, and only when every element is at the max).
    #: ``"lse"`` -- ``T * logsumexp(h / T)``, over-estimates by at most
    #: ``T * log(n_valid)`` (0.34 N*m at T = 0.05, n = 850) but that bias is
    #: ADDITIVE, so it does not vanish as the peak shrinks.  ``pnorm`` is the
    #: default for exactly that reason.
    torque_peak_mode: str = "pnorm"
    #: p-norm exponent.  12 keeps the single-violator estimate within ~1 % of
    #: the true max while staying far from fp32 overflow (the reduction is
    #: computed on ``h / max(h)``, so the exponent never sees a number > 1).
    torque_peak_p: float = 12.0
    #: LSE temperature, N*m.  Only read when ``torque_peak_mode == "lse"``.
    torque_peak_temp: float = 0.05
    #: STRIKE-MASKED EFFORT.  Price on the mean of ``tau^2`` (N^2 m^2) over the
    #: valid ``(knot, joint)`` elements -- an energy proxy, NOT a limit hinge, so
    #: it lives in the ``posture`` (objective) block and carries no
    #: augmented-Lagrangian multiplier.  Zero by default.
    #:
    #: The weight is FADED OUT over the last ``effort_fade_wp`` waypoints before
    #: each hand's own hit, because a strike IS kinetic energy: an unmasked
    #: ``tau^2`` term is in direct opposition to ``w_terminal_velocity``, which
    #: asks for 2 m/s at the tip, and the cheapest way to satisfy both is not to
    #: strike.  Inside the fade window the term is exactly zero and carries
    #: exactly zero gradient.  The waist columns fade near EITHER hand's hit
    #: (they carry both arms); an arm's columns fade only near its own.  An
    #: INACTIVE hand has no strike to protect, so its arm is priced everywhere.
    #:
    #: Raw ``tau^2``, deliberately un-normalized by the effort limits: the point
    #: is that a waist-yaw newton-metre is physically more expensive than a
    #: wrist one, and normalizing by the 88 / 5 N*m limits would erase exactly
    #: that.  Expect ``w_effort`` in the 1e-4..1e-2 range -- ``raw["effort"]`` is
    #: ~10-100 on real trajectories.
    w_effort: float = 0.0
    #: Waypoints before a hit over which the effort weight is EXACTLY ZERO --
    #: not merely small.  A linear ramp alone would leave the last waypoint
    #: before the strike at ``1/W`` of full price, and it is precisely those
    #: waypoints that have to carry the 2 m/s; a hard zero means the term
    #: contributes no gradient at all to the approach it must not shape.  The
    #: weight then ramps linearly back to 1 over the ``effort_fade_wp``
    #: waypoints BEFORE the off-window (so the total protected span is 2 W, of
    #: which the inner W is free), and is full again once the hit is past.
    #: 0 disables the fade entirely and prices the strike like anything else.
    effort_fade_wp: int = 5
    w_velocity: float = 1.0
    w_joint_box: float = 10.0
    #: PER-JOINT override of the joint box for the two waist LEAN joints
    #: (``waist_pitch``, ``waist_roll``), in radians of half-width.  ``None``
    #: (default) leaves the box exactly as ``box_source`` builds it -- +-0.52 rad
    #: from the URDF -- so every existing checkpoint and the section-3 validation
    #: weights are bit-identical.  A float shrinks BOTH rows to
    #: ``+-waist_pr_box``: the softplus hinge moves to ``eta_position`` x that
    #: (the same conservatism every other row gets) and the RAW violation moves
    #: to the value itself, so the tightened rows are priced by the EXISTING
    #: ``joint_box`` augmented-Lagrangian group rather than by a new multiplier.
    #: ``waist_yaw`` is deliberately untouched -- the torso has to spin to reach
    #: the outer drums, and constraining it would fight the reach directly.
    #: Ramp it in with ``weights["waist_box"]`` (0 = the original box, 1 = fully
    #: shrunk, linear in the half-width) so a continuation does not step the AL.
    waist_pr_box: float | None = None

    # ------------------------------------------------------ smoothness ----
    #: trajopt under ``simplified_cost``: ``w_v = 0`` (``simplified_w_v``),
    #: ``w_a = 1e-4``, ``w_j = 1e-5``.  Rescaled here so the three land within
    #: two decades of the limit terms at lambda = 1; the RATIO w_a : w_j = 10:1
    #: is trajopt's.  ``w_smooth_vel`` stays 0, matching ``simplified_w_v``.
    w_smooth_vel: float = 0.0
    w_smooth_acc: float = 1.0e-3
    w_smooth_jerk: float = 1.0e-4

    # --------------------------------------------------------- posture ----
    w_posture_mid: float = 1.0e-2
    #: UPRIGHT WELL on the waist LEAN joints:
    #: ``w_waist_upright * mean_valid(q_waist_pitch^2 + q_waist_roll^2)``, in
    #: rad^2, reduced over the two rows only (NOT diluted by the other 15
    #: joints, so the weight is interpretable against a target excursion --
    #: at 0.15 rad the raw term is 0.0225).  ``waist_yaw`` is excluded by
    #: construction: the mask has a hard zero in that column, so the term is
    #: exactly yaw-invariant and carries no yaw gradient.  Zero by default, as
    #: every new term in this package is.  Ramp it with
    #: ``weights["posture_upright"]``.
    w_waist_upright: float = 0.0
    #: Waist-aim IL penalty weight (rad^-2).  When > 0 and ``StrikeContext``
    #: carries ``waist_yaw_ref``, penalises ``(q_yaw - ref)^2`` over the horizon.
    #: Zero by default (no IL).  Ramp via ``weights["waist_aim"]``.
    w_waist_aim: float = 0.0
    #: WAIST-YAW RATE penalty, ``w_waist_yaw_vel * mean_valid(qd_waist_yaw^2)``,
    #: in (rad/s)^-2.  Yaw ONLY -- pitch and roll are excluded by the same
    #: ``c_waist_yaw`` selector the diagnostics use.
    #:
    #: Why this exists: the terminal term prices the stick's speed at contact,
    #: and the torso is by far the cheapest place to buy that speed.  It carries
    #: both arms, so a yaw sweep adds tip speed to the striking hand at no cost
    #: in arm torque -- the planner is free to WIND UP the waist and let the arm
    #: ride it.  Nothing else in the objective notices: ``w_waist_upright``
    #: prices pitch/roll only, ``waist_aim`` prices yaw POSITION against a
    #: reference the swing can pass straight through at speed, and the velocity
    #: hinge is a limit, not a preference, so it is silent until 32 rad/s.  This
    #: is the term that makes torso momentum cost something.
    #:
    #: Scale: the mean is over the yaw row alone, so the weight is directly
    #: comparable to a target rate in (rad/s)^2 -- a torso turning at a steady
    #: 1 rad/s pays ``w`` exactly.  Ramped by ``weights["waist_yaw_vel"]``.
    w_waist_yaw_vel: float = 0.0
    #: IDLE-ARM TERMINAL.  **The idle arm's goal, priced exactly like a strike.**
    #:
    #: A hand with a scheduled hit gets a terminal region + a terminal velocity
    #: (``w_terminal_region`` / ``w_terminal_velocity``).  A hand with no hit gets
    #: the same two things, aimed at a resting posture instead of a drum head:
    #: this weight prices ``|q_arm(K) - home|^2`` at its terminal knot, and
    #: ``w_idle_vel`` asks that knot to be stopped.
    #:
    #: Everything else is already handled and is *identical* for both hands --
    #: collision, self-collision, torque, velocity and joint-box all run over the
    #: idle arm's whole horizon with no special case.  So this term does not have
    #: to keep the arm out of the kit; it only has to say where the arm should
    #: end up, and the rest of the objective shapes how it gets there.
    #:
    #: ``home`` is ONE posture per hand (see :mod:`~amortized_opt.idle_home`), not
    #: a function of waist yaw: arm joint angles are already torso-relative, so a
    #: fixed vector keeps its posture relative to the body however the torso
    #: turns, and a parked arm genuinely sits still.
    w_idle_home: float = 0.0
    #:
    #: A **static weight, never an augmented-Lagrangian group**: on closed-loop
    #: states an arm mid-flight from a strike cannot be at home, and the repo has
    #: already paid once for pricing an unsatisfiable condition with a multiplier
    #: (the shortfall Lagrangian pinned at ``lam_max`` starved a 2 197-iteration
    #: run).  Keep it near ``w_posture_mid``'s scale so a mid-horizon
    #: re-activation pulls the arm out of home smoothly.
    #:
    #: IDLE-ARM TERMINAL VELOCITY: the parked arm arrives **stopped**.
    #: ``|qd_arm|^2`` at that hand's terminal knot, rad^2/s^2.  Directly the
    #: counterpart of ``w_terminal_velocity``, which asks a striking hand for
    #: 2 m/s at its terminal knot.
    #:
    #: Both idle terms are exactly zero, with exactly zero gradient, until a
    #: posture is installed with :meth:`FusedCost.set_idle_home`.
    w_idle_vel: float = 0.0
    w_elbow_down: float = 1.0
    #: Metres the elbow may sit ABOVE the shoulder->wrist chord midpoint before
    #: the hinge bites.  0.05 m fires on 4.9 % (left) / 2.1 % (right) of the
    #: recorded trajopt baseline poses.
    elbow_down_margin: float = 0.05
    beta_elbow: float = 0.02
    #: Port of trajopt's ``w_axial_twist``: a rank-1 quadratic on the net roll of
    #: each limb about its own long axis, measured from the ready pose.  Zero by
    #: default, exactly as ``TrajOptConfig.w_axial_twist`` is.
    w_axial_twist: float = 0.0
    twist_axis_floor: float = 0.17

    # ------------------------------------------------------ reductions ----
    #: ``"mean"`` divides each term by its valid-element count (the plan's
    #: per-horizon-length normalization); ``"sum"`` leaves it unnormalized.
    reduction: str = "mean"

    def validate(self) -> "CostConfig":
        if self.physics_mode not in ("knot_fd", "spline"):
            raise ValueError(f"physics_mode={self.physics_mode!r}")
        if self.velocity_max_mode not in ("analytic", "subsample"):
            raise ValueError(f"velocity_max_mode={self.velocity_max_mode!r}")
        if self.contact_mode not in ("softmin", "spec"):
            raise ValueError(f"contact_mode={self.contact_mode!r}")
        if self.impact_velocity_mode not in ("fd_coarse", "fd_fine"):
            raise ValueError(f"impact_velocity_mode={self.impact_velocity_mode!r}")
        if self.box_source not in ("posture", "hard"):
            raise ValueError(f"box_source={self.box_source!r}")
        if self.torque_derate not in ("none", "linear"):
            raise ValueError(f"torque_derate={self.torque_derate!r}")
        if self.torque_peak_mode not in ("pnorm", "lse"):
            raise ValueError(f"torque_peak_mode={self.torque_peak_mode!r}")
        if self.torque_peak_p < 1.0:
            raise ValueError("torque_peak_p must be >= 1")
        if self.torque_peak_temp <= 0.0:
            raise ValueError("torque_peak_temp must be > 0")
        if self.collision_peak_mode not in ("pnorm", "lse"):
            raise ValueError(f"collision_peak_mode={self.collision_peak_mode!r}")
        if self.collision_peak_p < 1.0:
            raise ValueError("collision_peak_p must be >= 1")
        if self.collision_peak_temp <= 0.0:
            raise ValueError("collision_peak_temp must be > 0")
        if self.w_collision_peak < 0.0:
            raise ValueError("w_collision_peak must be >= 0")
        if self.reduction not in ("mean", "sum"):
            raise ValueError(f"reduction={self.reduction!r}")
        if self.n_sub < 1:
            raise ValueError("n_sub must be >= 1")
        if self.waist_pr_box is not None and self.waist_pr_box <= 0.0:
            object.__setattr__(self, "waist_pr_box", None)
        if self.w_effort < 0.0:
            raise ValueError("w_effort must be >= 0")
        if self.effort_fade_wp < 0:
            raise ValueError("effort_fade_wp must be >= 0")
        if self.w_waist_upright < 0.0:
            raise ValueError("w_waist_upright must be >= 0")
        if self.w_waist_aim < 0.0:
            raise ValueError("w_waist_aim must be >= 0")
        if self.w_waist_yaw_vel < 0.0:
            raise ValueError("w_waist_yaw_vel must be >= 0")
        if self.w_idle_home < 0.0:
            raise ValueError("w_idle_home must be >= 0")
        if self.w_idle_vel < 0.0:
            raise ValueError("w_idle_vel must be >= 0")
        return self


# ============================================================= context ======
@dataclass
class StrikeContext:
    """Everything the terminal + exemption logic needs, batched, hand-major.

    Index 0 is the LEFT hand, index 1 the RIGHT, throughout.  ``active`` marks a
    hand that has a scheduled strike; an inactive hand contributes nothing to
    the terminal cost (its horizon still exists, so its joints are still priced
    by the limit / smoothness / collision terms).
    """

    hit_index: torch.Tensor       # (B, 2) long -- knot index of the strike
    active: torch.Tensor          # (B, 2) float 0/1
    target_drum: torch.Tensor     # (B, 2) long, -1 = none
    start_drum: torch.Tensor      # (B, 2) long, -1 = none
    p_star: torch.Tensor          # (B, 2, 3) aim point, pelvis frame
    v_star: torch.Tensor          # (B, 2, 3) commanded impact velocity
    n_out: torch.Tensor           # (B, 2, 3) outward head normal
    hit_radius: torch.Tensor      # (B, 2)
    disk_offset: torch.Tensor     # (B, 2, 3) aim -> head centre, in plane
    half_plane_dir: torch.Tensor  # (B, 2, 3)
    has_half_plane: torch.Tensor  # (B, 2) float 0/1
    contact_sigma: torch.Tensor   # (B, 2)
    shaft_allowed: torch.Tensor   # (B, 2) float 0/1 -- target drum
    start_shaft_allowed: torch.Tensor  # (B, 2) float 0/1 -- departure drum
    elev_required: torch.Tensor   # (B, 2) float 0/1
    #: ``(B, D, 3)`` per-sample drum head centres in the COST's frame (pelvis).
    #: Set this whenever the kit is domain-randomized: without it the collision
    #: term prices the nominal layout while ``p_star`` / the goal tokens describe
    #: the randomized one, and the two disagree by up to ~11 cm.  Build it from
    #: the context tokens with :func:`drum_heads_from_context` so "the scene the
    #: cost sees" and "the scene the model was told about" are the same array.
    drum_head_centers: torch.Tensor | None = None
    #: ``(B, H)`` waist-yaw reference trajectory at knot resolution, computed by
    #: :mod:`~amortized_opt.waist_aim`.  ``None`` means the IL term is disabled.
    waist_yaw_ref: torch.Tensor | None = None

    @property
    def batch(self) -> int:
        return int(self.hit_index.shape[0])

    def to(self, device, dtype: torch.dtype | None = None) -> "StrikeContext":
        out = {}
        for k, v in self.__dict__.items():
            if v is None:
                out[k] = None
            elif v.dtype == torch.long:
                out[k] = v.to(device)
            else:
                out[k] = v.to(device=device, dtype=dtype or v.dtype)
        return StrikeContext(**out)


def drum_heads_from_context(drum_pos_env: torch.Tensor) -> torch.Tensor:
    """``(B, D, 3)`` env-frame drum centres -> the cost's pelvis frame.

    :func:`~amortized_opt.rhythm.build_context_inputs` writes ``drum_pos`` in the
    env frame (the :mod:`~amortized_opt.mpc_gen` default, so phase-2 batches
    collate identically); :class:`FusedCost` is pelvis-rooted.  Deriving the
    collision scene from the *tokens* rather than from a parallel copy of the
    layout is what makes "the goal context matches the scene exactly" true by
    construction instead of by convention.
    """
    off = torch.as_tensor(
        PELVIS_IN_ENV, device=drum_pos_env.device, dtype=drum_pos_env.dtype
    )
    return drum_pos_env - off


def strike_context_from_cases(
    cases: Sequence,
    *,
    drum_index: dict[str, int],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> StrikeContext:
    """Build a :class:`StrikeContext` from a list of ``EvalCase``.

    The harness's grid is single-strike (one hand hits, the other holds), so the
    idle hand comes back ``active = 0`` with its horizon equal to the striking
    hand's.  Dual-arm contexts are built by filling both columns.
    """
    B = len(cases)
    z = lambda *s: np.zeros(s, dtype=float)
    hit = np.zeros((B, 2), dtype=np.int64)
    active = z(B, 2)
    tgt = -np.ones((B, 2), dtype=np.int64)
    src = -np.ones((B, 2), dtype=np.int64)
    p_star, v_star, n_out = z(B, 2, 3), z(B, 2, 3), z(B, 2, 3)
    hit_r = np.full((B, 2), 0.08)
    disk_off, hp_dir = z(B, 2, 3), z(B, 2, 3)
    has_hp, sigma = z(B, 2), np.full((B, 2), STICK_TIP_SIGMA)
    shaft, start_shaft, elev = z(B, 2), z(B, 2), z(B, 2)
    n_out[..., 2] = 1.0

    for b, case in enumerate(cases):
        j = _HANDS.index(case.hand)
        s = case.strike
        hit[b, :] = int(case.N)
        active[b, j] = 1.0
        tgt[b, j] = drum_index.get(str(s.drum), -1)
        if case.start_drum is not None:
            src[b, j] = drum_index.get(str(case.start_drum), -1)
            start_shaft[b, j] = float(shaft_hit_allowed(case.start_drum))
        p_star[b, j] = np.asarray(s.position, float)
        v = np.asarray(s.velocity, float)
        v_star[b, j] = v
        nv = float(np.linalg.norm(v))
        n_out[b, j] = -v / nv if nv > 1e-9 else np.array([0.0, 0.0, 1.0])
        hit_r[b, j] = s.resolved_hit_radius()
        if s.disk_center_offset is not None:
            disk_off[b, j] = np.asarray(s.disk_center_offset, float)
        if s.half_plane_dir is not None:
            hp_dir[b, j] = np.asarray(s.half_plane_dir, float)
            has_hp[b, j] = 1.0
        sigma[b, j] = s.resolved_contact_sigma()
        shaft[b, j] = float(shaft_hit_allowed(s.drum))
        elev[b, j] = float(strike_elevation_required(s.drum))

    t = lambda a: torch.as_tensor(a, device=device, dtype=dtype)
    ti = lambda a: torch.as_tensor(a, device=device, dtype=torch.long)
    return StrikeContext(
        hit_index=ti(hit), active=t(active), target_drum=ti(tgt), start_drum=ti(src),
        p_star=t(p_star), v_star=t(v_star), n_out=t(n_out), hit_radius=t(hit_r),
        disk_offset=t(disk_off), half_plane_dir=t(hp_dir), has_half_plane=t(has_hp),
        contact_sigma=t(sigma), shaft_allowed=t(shaft),
        start_shaft_allowed=t(start_shaft), elev_required=t(elev),
    )


# ============================================================== output ======
@dataclass
class CostTerms:
    """Per-sample cost decomposition.  Every tensor is ``(B,)``."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    raw: dict[str, torch.Tensor] = field(default_factory=dict)
    violations: dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    counts: dict[str, torch.Tensor] = field(default_factory=dict)

    def detach_numpy(self) -> dict[str, np.ndarray]:
        out = {"total": self.total.detach().cpu().numpy()}
        for grp in ("terms", "raw", "violations", "diagnostics", "counts"):
            for k, v in getattr(self, grp).items():
                out[f"{grp[:4]}.{k}"] = v.detach().cpu().numpy()
        return out


# =============================================================== helpers ====
def _softplus_hinge(x: torch.Tensor, beta: float) -> torch.Tensor:
    """``softplus(x / beta) * beta`` -- the plan's limit hinge, in x's units."""
    return F.softplus(x / beta) * beta


def smooth_max(
    h: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "pnorm",
    p: float = 12.0,
    temp: float = 0.05,
    *,
    dims: tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable, mask-aware smooth maximum of a NON-NEGATIVE ``h``.

    The counterpart to :meth:`FusedCost._reduce`: same array, same mask, but a
    max-shaped reduction instead of a mean.  ``h`` is reduced over ``dims``
    (default: every dim after the batch dim), leaving ``(B,)``.  Masked-out
    elements contribute EXACTLY zero in both modes -- ``h * mask`` for
    ``"pnorm"``, a ``-1e9`` fill for ``"lse"`` -- so a garbage value beyond a
    sample's horizon cannot leak in.

    ``mode="pnorm"`` returns ``(sum_i h_i^p)^(1/p)``, computed in the
    normalized form ``m * (sum_i (h_i / m)^p)^(1/p)`` with ``m = max_i h_i``
    held as a DETACHED constant.  That is algebraically the same number and the
    same gradient, but the exponent only ever sees values in ``[0, 1]``, so
    ``p = 12`` cannot overflow the way ``h^12`` would at ``h = 100`` in fp32
    (1e24 is fine, ``h = 1e4`` is not).  Rows whose masked max is zero return a
    hard zero with zero gradient rather than the ``0 ** (1/p)`` infinity.

    ``mode="lse"`` returns ``T * logsumexp(h / T)``.  No ``T * log(n)`` bias is
    subtracted: with the masked-out entries pushed to ``-1e9`` the sum is
    dominated by the max whenever the peak is large against ``T``, and
    subtracting the bias would drive a feasible row NEGATIVE.  The price is that
    the estimate OVER-ESTIMATES the true max by at most ``T * log(n_valid)``
    (0.34 N*m at ``T = 0.05``, ``n_valid = 850``), additively.

    Both modes are computed in fp32 or better: a bf16 ``h`` (autocast) is
    upcast, because a 12th power and a ``logsumexp`` at ``T = 0.05`` both burn
    far more mantissa than bf16's 8 bits have.
    """
    if dims is None:
        dims = tuple(range(1, h.dim()))
    else:
        dims = tuple(dims)
    if dims != tuple(range(1, h.dim())):
        raise ValueError(
            f"smooth_max reduces the trailing dims only; got dims={dims} "
            f"for a {h.dim()}-d tensor"
        )
    work = h if h.dtype in (torch.float32, torch.float64) else h.to(torch.float32)
    m = mask.to(work.dtype)
    hm = work * m

    if mode == "pnorm":
        scale = hm.amax(dim=dims, keepdim=True).detach().clamp_min(eps)
        s = ((hm / scale).clamp_min(0.0) ** p).sum(dim=dims)          # (B,)
        flat = scale.reshape(s.shape)
        live = flat > eps
        # A dead row's sum is 0 and d/ds s**(1/p) is +inf there; swap in a 1
        # BEFORE the power so the row's gradient is a clean zero, not a NaN.
        out = flat * torch.where(live, s, torch.ones_like(s)).pow(1.0 / p)
        return torch.where(live, out, torch.zeros_like(out))

    if mode == "lse":
        B = hm.shape[0]
        big = torch.full((), -1e9, device=hm.device, dtype=hm.dtype)
        x = torch.where(m > 0, hm, big).reshape(B, -1)
        live = (m.reshape(B, -1) > 0).any(dim=1)
        out = temp * torch.logsumexp(x / temp, dim=1)
        return torch.where(live, out, torch.zeros_like(out))

    raise ValueError(f"smooth_max mode={mode!r}")


@dataclass
class _Rates:
    """Minimal (q, qd, qdd, mask) bundle, so the torque term can reduce over
    either the spline sub-samples or the knot finite differences."""

    q: torch.Tensor
    qd: torch.Tensor
    qdd: torch.Tensor
    mask: torch.Tensor


def axial_twist_directions(
    fk: ChainFK,
    *,
    q_ref: np.ndarray | None = None,
    twist_axis_floor: float = 0.17,
    joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
) -> np.ndarray:
    """``(2, n)`` accumulated-axial-twist directions, left then right.

    A port of the *concept* behind ``trajopt.robot.axial_twist_direction``:
    ``c_j = omega_j . L``, the world rotation axis of joint ``j`` at the ready
    posture projected onto the limb's long axis (shoulder -> stick tip).  The
    scalar ``c' (q - q_ready)`` is the net roll of the hand about that axis, so
    penalizing its square prices twist that COMPOUNDS down the chain and leaves
    twist that CANCELS free -- something no diagonal weight can express.

    Axes within ``twist_axis_floor`` of perpendicular are dropped, and the waist
    columns are zeroed (the waist turns the torso, it does not twist the arm).
    Requires ``fk`` to carry ``{hand}_wrist`` and ``{hand}_shoulder`` frames.
    """
    q = ready_joint_pos(joint_names) if q_ref is None else np.asarray(q_ref, float)
    qt = torch.as_tensor(q, dtype=torch.float64, device=fk.device).view(1, -1)
    out = np.zeros((2, len(joint_names)))
    waist = [i for i, n in enumerate(joint_names) if n.startswith("waist_")]
    for hi, hand in enumerate(_HANDS):
        wname, sname = f"{hand}_wrist", f"{hand}_shoulder"
        if wname not in fk.frame_names or sname not in fk.frame_names:
            continue
        ch = fk._chains[wname]
        R = np.eye(3)
        axes: dict[int, np.ndarray] = {}
        for k, ji in enumerate(ch["qidx_np"].tolist()):
            R_pre = R @ ch["R_np"][k]
            a = ch["axis_np"][k]
            axes[int(ji)] = R_pre @ a
            th = float(q[int(ji)])
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            R = R_pre @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K))
        with torch.no_grad():
            Rw, pw = fk.frame(qt, wname)
            _Rs, ps = fk.frame(qt, sname)
            tip = (
                pw + (Rw @ torch.as_tensor(
                    stick_tip_offset(hand), dtype=torch.float64, device=fk.device
                ).view(1, 3, 1)).squeeze(-1)
            )[0].cpu().numpy()
        L = tip - ps[0].cpu().numpy()
        nrm = float(np.linalg.norm(L))
        if nrm < 1e-6:
            continue
        L = L / nrm
        c = np.zeros(len(joint_names))
        for ji, ax in axes.items():
            c[ji] = float(ax @ L)
        c[np.abs(c) < float(twist_axis_floor)] = 0.0
        c[waist] = 0.0
        out[hi] = c
    return out


# ============================================================ the cost ======
class FusedCost:
    """Batched differentiable trajectory cost over the 17-DoF upper body.

    ``forward(q0, waypoints, v0, ctx, lengths=...)`` takes the start pose, the
    ``(B, H, 17)`` waypoint block on the ``CONTROL_DT`` grid, the start velocity
    and a :class:`StrikeContext`, and returns :class:`CostTerms`.

    The instance owns the FK / RNEA / drum-scene tensors and the spline gain
    tables; build one per (device, max_horizon) and reuse it.
    """

    #: FK frames the cost needs: wrists (sticks), elbows + shoulders (posture),
    #: torso (self-collision).
    FRAMES: dict[str, str] = {
        "left_wrist": "left_wrist_yaw_joint",
        "right_wrist": "right_wrist_yaw_joint",
        "left_elbow": "left_elbow_joint",
        "right_elbow": "right_elbow_joint",
        "left_shoulder": "left_shoulder_pitch_joint",
        "right_shoulder": "right_shoulder_pitch_joint",
        "torso": "waist_pitch_joint",
    }

    def __init__(
        self,
        cfg: CostConfig | None = None,
        *,
        max_horizon: int = 100,
        dt: float = CONTROL_DT,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        layout_path: str | Path | None = None,
        urdf_path: str | Path | None = None,
        scene: DrumSceneSpec | None = None,
        fk: ChainFK | None = None,
        rnea: BatchedRNEA | None = None,
        under_depth: float = 0.08,
        under_clearance: float = 0.01,
        bc_start: str = "clamped",
        bc_end: str = "natural",
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
    ):
        self.cfg = (cfg or CostConfig()).validate()
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = float(dt)
        self.max_horizon = int(max_horizon)
        self.joint_names = tuple(joint_names)
        self.n_dof = len(self.joint_names)

        self.fk = fk if fk is not None else ChainFK(
            urdf_path, joint_names=self.joint_names, frames=dict(self.FRAMES),
            device=self.device, dtype=dtype,
        )
        self.rnea = rnea if rnea is not None else BatchedRNEA.from_urdf(
            urdf_path, joint_names=self.joint_names, with_armature=True,
            device=self.device, dtype=dtype,
        )
        self.scene = scene if scene is not None else DrumSceneSpec.from_layout(
            layout_path, frame="pelvis",
            under_depth=under_depth, under_clearance=under_clearance,
        )
        self.drum_index = {n: i for i, n in enumerate(self.scene.names)}
        self.spline = TrajectorySplineBuilder(
            max_segments=self.max_horizon, dt=self.dt,
            bc_start=bc_start, bc_end=bc_end, joint_names=self.joint_names,
            device=self.device, dtype=dtype,
        )
        #: Idle-arm park posture ``(2, n_arm)``, installed by
        #: :meth:`set_idle_home`.  ``None`` (the default) makes ``w_idle_home`` /
        #: ``w_idle_vel`` exact no-ops.
        self.c_idle_q: torch.Tensor | None = None
        self._build_constants(under_depth, under_clearance)

    # ------------------------------------------------------------ build ---
    def _build_constants(self, under_depth: float, under_clearance: float) -> None:
        cfg = self.cfg
        dev, dt_ = self.device, self.dtype
        t = lambda a: torch.as_tensor(np.asarray(a, float), device=dev, dtype=dt_)

        sc = self.scene
        heads = t(sc.head_centers)
        axes = t(sc.axes)
        half_h = t(sc.heights / 2.0)
        self.c_heads, self.c_axes, self.c_radii = heads, axes, t(sc.radii)
        self.c_half_h = half_h
        self.c_body_c = heads - half_h.unsqueeze(-1) * axes
        self.c_heights_uc = t(sc.heights + under_clearance)   # head -> under top
        self.c_under_off = float(under_depth) / 2.0
        under_top = heads - self.c_heights_uc.unsqueeze(-1) * axes
        self.c_under_c = under_top - self.c_under_off * axes
        self.c_under_half = t(np.full(sc.num_drums, under_depth / 2.0))
        self.n_drums = sc.num_drums

        # Collision sample points, wrist-local: [tip, shaft*S, wrist].
        S = int(cfg.shaft_samples)
        self.c_points = {}
        for hand in _HANDS:
            local = np.concatenate(
                [stick_tip_offset(hand)[None, :], stick_shaft_offsets(hand, S),
                 np.zeros((1, 3))], axis=0
            )
            self.c_points[hand] = t(local)
        self.n_points = 1 + S + 1
        self.tip_index = 0
        self.c_pt_radii = t(
            np.concatenate([[TIP_RADIUS], np.full(S, STICK_RADIUS), [WRIST_RADIUS]])
        )
        row_tip = np.zeros(self.n_points)
        row_tip[0] = 1.0
        row_shaft = np.zeros(self.n_points)
        row_shaft[1 : 1 + S] = 1.0
        self.c_row_tip, self.c_row_shaft = t(row_tip), t(row_shaft)

        a, b, r = TORSO_CAPSULE
        self.c_torso_a, self.c_torso_b, self.c_torso_r = t(a), t(b), float(r)
        hc, hr = HEAD_SPHERE
        self.c_head_c, self.c_head_r = t(hc), float(hr)

        # Stick mount, per hand: contact(sigma) = p + R @ (mount_p + sigma*axis)
        self.c_mount_p, self.c_mount_axis = {}, {}
        for hand in _HANDS:
            Rm, pm = stick_mount_transform(hand)
            self.c_mount_p[hand] = t(pm)
            self.c_mount_axis[hand] = t(stick_axis_local(hand))
        self.c_sigmas = t(np.asarray(SHAFT_SIGMAS, float))  # (Sc,)
        self.n_sigmas = int(self.c_sigmas.shape[0])
        tip_first = np.zeros(self.n_sigmas)
        tip_first[int(np.argmax(np.asarray(SHAFT_SIGMAS)))] = 1.0
        self.c_sigma_is_tip = t(tip_first)  # which candidate is the bead

        # Limits.
        self.tau_hard = t(isaac_effort_limits(self.joint_names))
        self.v_hard = t(isaac_velocity_limits(self.joint_names))
        lo, hi = urdf_position_limits(
            self.joint_names, posture_limits=(cfg.box_source == "posture")
        )
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.q_mid, self.q_half = t(mid), t(half)
        self.q_lo_hard, self.q_hi_hard = t(lo), t(hi)

        # Row selectors for the two waist LEAN joints and for waist YAW.  Built
        # UNCONDITIONALLY (they cost 2 x 17 floats) so ``with_config`` can switch
        # ``waist_pr_box`` / ``w_waist_upright`` on without rebuilding the
        # constants, and so the yaw exclusion is one explicit array rather than
        # an implicit index.
        pr = np.zeros(self.n_dof)
        yaw = np.zeros(self.n_dof)
        for nm, row in (("waist_pitch_joint", pr), ("waist_roll_joint", pr),
                        ("waist_yaw_joint", yaw)):
            if nm in self.joint_names:
                row[self.joint_names.index(nm)] = 1.0
        self.c_waist_pr, self.c_waist_yaw = t(pr), t(yaw)

        # Posture: twist rows + reference pose.
        self.c_twist = t(
            axial_twist_directions(self.fk, twist_axis_floor=cfg.twist_axis_floor,
                                   joint_names=self.joint_names)
        )  # (2, n)
        self.q_ready = t(ready_joint_pos(self.joint_names))

        # Group column indices (waist / left / right), for per-hand masks.
        # Warm BOTH spline caches here: they are built from Python lists, so a
        # first call inside a compiled region is a data-dependent graph break
        # (dynamo traces `self._group_np.tolist()` as 17 `.item()` calls).
        self._cols = self.spline._col_index(dev)
        self.spline._group_row(dev)
        self._gcols = {"waist": self._cols[0], "left": self._cols[1],
                       "right": self._cols[2]}
        # ...and the inverse map, joint -> group id (0 waist / 1 left / 2 right),
        # which is what turns a per-GROUP quantity into a per-JOINT one in a
        # single ``index_select`` (see ``_effort_fade``).
        self.c_joint_group = torch.as_tensor(
            [1 if nm.startswith("left_") else (2 if nm.startswith("right_") else 0)
             for nm in self.joint_names],
            device=dev, dtype=torch.long,
        )

    def to(self, device, dtype: torch.dtype | None = None) -> "FusedCost":
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.fk = self.fk.to(self.device, self.dtype)
        self.rnea = self.rnea.to(self.device, self.dtype)
        self.spline = self.spline.to(self.device, self.dtype)
        self._build_constants(self.scene.under_depth, self.scene.under_clearance)
        if self.c_idle_q is not None:
            self.c_idle_q = self.c_idle_q.to(device=self.device, dtype=self.dtype)
        return self

    # ----------------------------------------------------------- idle arm ---
    def set_idle_home(self, table) -> "FusedCost":
        """Install the park posture the idle-arm terms are priced against.

        ``table`` is an :class:`~amortized_opt.idle_home.IdleHomeTable` or a
        ``(2, n_arm)`` array: ONE arm posture per hand.  It is torso-relative --
        the arm is mounted on the torso, so a fixed vector of joint angles keeps
        its posture relative to the body at every waist yaw, and the term below
        is therefore exactly yaw-independent by construction rather than by a
        detached index.

        Pass ``None`` to uninstall (the terms go back to exact zeros).
        """
        if table is None:
            self.c_idle_q = None
            return self
        if hasattr(table, "q_arm"):
            if tuple(table.joint_names) != self.joint_names:
                raise ValueError(
                    f"idle-home pose is for {tuple(table.joint_names)}, "
                    f"cost is on {self.joint_names}"
                )
            table = table.q_arm
        q = torch.as_tensor(
            np.asarray(table, float), device=self.device, dtype=self.dtype
        )
        n_arm = int(self._gcols["left"].numel())
        if q.shape != (2, n_arm):
            raise ValueError(
                f"idle-home pose must be (2, {n_arm}), got {tuple(q.shape)}"
            )
        self.c_idle_q = q
        return self

    def with_config(self, **kw) -> "FusedCost":
        """A shallow clone sharing every tensor, with an amended config.

        Only fields that the *forward pass* reads take effect -- weights,
        betas, ``n_sub``, ``physics_mode``, ``contact_mode``, ``reduction``,
        ``eta_*``.  Fields baked into the constants at construction
        (``box_source``, ``shaft_samples``, ``twist_axis_floor``) need a fresh
        :class:`FusedCost`.
        """
        out = object.__new__(FusedCost)
        out.__dict__.update(self.__dict__)
        out.cfg = replace(self.cfg, **kw).validate()
        return out

    # -------------------------------------------------------- reductions --
    def _reduce(self, x: torch.Tensor, mask: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor]:
        """(value, count) reduced over ``dims`` with ``mask`` applied."""
        num = (x * mask).sum(dim=dims)
        cnt = mask.sum(dim=dims)
        if self.cfg.reduction == "sum":
            return num, cnt
        return num / cnt.clamp_min(1.0), cnt

    # ------------------------------------------------------------ forward --
    def forward(
        self,
        q0: torch.Tensor,           # (B, 17)
        waypoints: torch.Tensor,    # (B, H, 17)
        v0: torch.Tensor,           # (B, 17)
        ctx: StrikeContext,
        *,
        lengths: torch.Tensor | None = None,   # (B, 3) segments per group
        elem_mask: torch.Tensor | None = None,  # (B, H, 17)
        weights: dict[str, torch.Tensor | float] | None = None,
    ) -> CostTerms:
        cfg = self.cfg
        if lengths is None:
            if elem_mask is not None:
                lengths = self.spline.lengths_from_mask(elem_mask)
            else:
                lengths = self._lengths_from_context(ctx)
        sp = self.spline.build(q0, waypoints, v0, lengths)
        ss = sp.subsample(cfg.n_sub, include_end=True)
        B, S, D = ss.q.shape

        # Per-hand sample validity: a sample counts for a hand if that hand's
        # joints are inside its horizon at that sample.
        m_hand = {
            h: ss.mask.index_select(2, self._gcols[h]).amax(dim=2) for h in _HANDS
        }  # (B, S)
        m_any = torch.maximum(m_hand["left"], m_hand["right"])

        qf = ss.q.reshape(B * S, D)
        frames = self.fk.forward(qf)  # ONE pass

        terms: dict[str, torch.Tensor] = {}
        raw: dict[str, torch.Tensor] = {}
        viol: dict[str, torch.Tensor] = {}
        diag: dict[str, torch.Tensor] = {}
        counts: dict[str, torch.Tensor] = {}

        # Group multipliers (the AL lambdas) plus optional SUB-term multipliers
        # so a curriculum can ramp e.g. the impact-speed and elevation prices
        # without rebuilding the config (which would retrigger a compile).  Pass
        # 0-dim tensors, not floats, to keep the compiled graph shape-stable.
        # Built BEFORE the terms because ``waist_box`` is not a price on a
        # penalty, it is a blend on the joint box's own HALF-WIDTH, so
        # ``_joint_box`` has to see it.
        w = dict(
            terminal=1.0, collision=1.0, torque=1.0, velocity=1.0, joint_box=1.0,
            smooth=1.0, posture=1.0,
            terminal_region=1.0, terminal_velocity=1.0, terminal_elev=1.0,
            posture_mid=1.0, posture_elbow=1.0, torque_peak=1.0,
            collision_peak=1.0,
            posture_upright=1.0, waist_box=1.0, effort=1.0, waist_aim=1.0,
            posture_idle=1.0, waist_yaw_vel=1.0,
        )
        if weights:
            w.update({k: v for k, v in weights.items() if k in w})

        self._collision(frames, B, S, ctx, m_hand, raw, viol, diag, counts, sp)
        self._torque(sp, ss, v0, B, S, D, ctx, raw, viol, diag, counts)
        self._velocity(sp, raw, viol, diag, counts)
        self._joint_box(ss, raw, viol, diag, counts, w["waist_box"])
        self._smoothness(sp, raw, diag, counts)
        self._posture(frames, ss, B, S, ctx, raw, diag, counts)
        self._terminal(sp, ctx, raw, viol, diag, counts)

        terms["terminal"] = w["terminal"] * (
            (cfg.w_terminal_region * w["terminal_region"]) * raw["terminal_region"]
            + cfg.w_terminal_aim * raw["terminal_aim"]
            + (cfg.w_terminal_velocity * w["terminal_velocity"]) * raw["terminal_velocity"]
            + (cfg.w_terminal_elev * w["terminal_elev"]) * raw["terminal_elev"]
        )
        # As with ``torque``, the two reductions of the collision hinge carry
        # INDEPENDENT multipliers: ``w["collision"]`` prices the three
        # mean-reduced sub-costs, ``w["collision_peak"]`` the smooth max over
        # the same hinges.  Separate augmented-Lagrangian groups so the peak
        # price can rise while the mean price relaxes -- raising a weight
        # against a multiplier calibrated on the other reduction is what
        # destabilized the section-7b continuations.  At
        # ``cfg.w_collision_peak = 0`` (the default) the second product is
        # exactly zero and the term is bit-identical to the mean-only form.
        terms["collision"] = w["collision"] * (
            cfg.w_collision_drum * raw["collision_drum"]
            + cfg.w_collision_under * raw["collision_under"]
            + cfg.w_collision_self * raw["collision_self"]
        ) + w["collision_peak"] * cfg.w_collision_peak * raw["collision_peak"]
        # The two reductions of the torque hinge carry INDEPENDENT multipliers:
        # ``w["torque"]`` prices the mean, ``w["torque_peak"]`` the smooth max.
        # They are separate augmented-Lagrangian groups (see
        # :data:`~amortized_opt.lagrangian.CONSTRAINT_GROUPS`) because their
        # violations differ by ~n_valid in scale, and ``w["torque_peak"]`` also
        # carries the curriculum ramp on the max-shaped half.  At
        # ``cfg.w_torque_peak = 0`` (the default) the second product is exactly
        # zero and the term is bit-identical to the mean-only form.
        terms["torque"] = (
            w["torque"] * cfg.w_torque * raw["torque"]
            + w["torque_peak"] * cfg.w_torque_peak * raw["torque_peak"]
        )
        # Effort is an OBJECTIVE, not a constraint: no augmented-Lagrangian
        # multiplier, only its config weight and a curriculum ramp.  Its own
        # term key rather than a share of ``torque`` precisely because the
        # torque key carries ``lam_torque``, and multiplying an energy proxy by
        # a limit-violation multiplier is the mistake section 7b paid for.
        # Exactly zero at the default ``cfg.w_effort = 0``.
        terms["effort"] = (cfg.w_effort * w["effort"]) * raw["effort"]
        terms["velocity"] = w["velocity"] * cfg.w_velocity * raw["velocity"]
        terms["joint_box"] = w["joint_box"] * cfg.w_joint_box * raw["joint_box"]
        terms["smooth"] = w["smooth"] * (
            cfg.w_smooth_vel * raw["smooth_vel"]
            + cfg.w_smooth_acc * raw["smooth_acc"]
            + cfg.w_smooth_jerk * raw["smooth_jerk"]
        )
        terms["posture"] = w["posture"] * (
            (cfg.w_posture_mid * w["posture_mid"]) * raw["posture_mid"]
            + (cfg.w_elbow_down * w["posture_elbow"]) * raw["posture_elbow"]
            + cfg.w_axial_twist * raw["posture_twist"]
            # At ``cfg.w_waist_upright = 0`` (the default) this product is
            # exactly zero and the term is bit-identical to the pre-7d form.
            + (cfg.w_waist_upright * w["posture_upright"]) * raw["posture_upright"]
            # The IDLE ARM pair, both exactly zero without an installed home
            # table.  They ride the ``posture`` group multiplier (an objective
            # block, no augmented-Lagrangian multiplier of their own) and share
            # one curriculum ramp, ``weights["posture_idle"]``.
            + (cfg.w_idle_home * w["posture_idle"]) * raw["posture_idle_home"]
            + (cfg.w_idle_vel * w["posture_idle"]) * raw["posture_idle_vel"]
        )
        # Waist-aim IL: an OBJECTIVE (no multiplier) — penalise deviation from
        # the reference yaw trajectory computed by ``waist_aim.yaw_reference``.
        # Exactly zero at ``cfg.w_waist_aim = 0`` (the default).
        terms["waist_aim"] = (cfg.w_waist_aim * w["waist_aim"]) * raw["waist_aim"]
        # Waist-yaw RATE: also an objective, and deliberately its own term rather
        # than a line in the posture block -- it is the only thing in the cost
        # that prices torso MOMENTUM, and it has to be readable on its own in
        # ``metrics.jsonl`` to tell "the waist stopped moving" from "the waist
        # stopped being useful".  Exactly zero at ``w_waist_yaw_vel = 0``.
        terms["waist_yaw_vel"] = (
            cfg.w_waist_yaw_vel * w["waist_yaw_vel"]
        ) * raw["waist_yaw_vel"]
        total = sum(terms.values())
        return CostTerms(total=total, terms=terms, raw=raw, violations=viol,
                         diagnostics=diag, counts=counts)

    __call__ = forward

    def _lengths_from_context(self, ctx: StrikeContext) -> torch.Tensor:
        """(B, 3) group horizons from the two hit indices (waist = the longer)."""
        left, right = ctx.hit_index[:, 0], ctx.hit_index[:, 1]
        both = torch.maximum(left, right)
        return torch.stack([both, left, right], dim=1)

    # ---------------------------------------------------------- collision --
    def _points_world(self, frames, hand: str) -> torch.Tensor:
        R, p = frames[f"{hand}_wrist"]
        return ChainFK.points_world(R, p, self.c_points[hand].to(p.dtype))

    def _collision(self, frames, B, S, ctx, m_hand, raw, viol, diag, counts, sp) -> None:
        cfg = self.cfg
        dtype = self.c_heads.dtype
        D = self.n_drums
        P = self.n_points
        n_sub = cfg.n_sub
        s_idx = torch.arange(S, device=self.device, dtype=torch.long).view(1, -1)

        pen_d = 0.0
        pen_u = 0.0
        pen_s = 0.0
        v_max = None
        min_sdf = None
        n_seen = 0.0
        # The smooth-max half is built from the SAME hinges the means reduce,
        # collected flat and concatenated so ONE :func:`smooth_max` sees every
        # (knot, point, drum) pair of both hands and all three collision types
        # -- a per-hand max combined afterwards would be a max of p-norms, not
        # the p-norm of the union.  Only assembled when the term is priced: the
        # concatenation is ~9 k floats per sample and the default weight is 0.
        want_peak = cfg.w_collision_peak > 0.0
        peak_h: list[torch.Tensor] = []
        peak_m: list[torch.Tensor] = []
        # Domain randomization moves the drums; when the caller supplies the
        # per-sample head centres (fix B) the collision scene is rebuilt from
        # them, so it agrees with the terminal cost and the goal tokens.
        if ctx.drum_head_centers is None:
            body_c = self.c_body_c.view(1, 1, 1, D, 3)
            under_c = self.c_under_c.view(1, 1, 1, D, 3)
        else:
            # The context tokens are padded to ``num_drums`` slots; the scene
            # has however many the layout defines. Row i is drum id i in both
            # (both are sorted by the layout's ``id``), so a prefix slice lines
            # them up.
            heads_b = ctx.drum_head_centers[:, :D].to(dtype)   # (B, D, 3)
            axb = self.c_axes.view(1, D, 3)
            body_c = (heads_b - self.c_half_h.view(1, D, 1) * axb).view(B, 1, 1, D, 3)
            under_top_b = heads_b - self.c_heights_uc.view(1, D, 1) * axb
            under_c = (under_top_b - self.c_under_off * axb).view(B, 1, 1, D, 3)

        for hi, hand in enumerate(_HANDS):
            pts = self._points_world(frames, hand).reshape(B, S, P, 3)
            x = pts.unsqueeze(3)  # (B,S,P,1,3)
            ax = self.c_axes.view(1, 1, 1, D, 3)
            rad = self.c_radii.view(1, 1, 1, D)
            sdf_b = point_cylinder_sdf(
                x, body_c, ax, rad, self.c_half_h.view(1, 1, 1, D),
            ) - self.c_pt_radii.view(1, 1, P, 1)
            sdf_u = point_cylinder_sdf(
                x, under_c, ax, rad, self.c_under_half.view(1, 1, 1, D),
            ) - self.c_pt_radii.view(1, 1, P, 1)

            # --- exemption windows (mirror TrajectoryScorer) -----------------
            k = ctx.hit_index[:, hi]
            first_exempt = (k - cfg.strike_exempt_waypoints).clamp_min(0) * n_sub
            win_strike = (s_idx >= first_exempt.view(-1, 1)).to(dtype)  # (B,S)
            last_start = cfg.start_exempt_waypoints * n_sub
            win_start = (s_idx <= last_start).to(dtype).expand(B, S)
            if cfg.start_exempt_waypoints <= 0:
                win_start = torch.zeros_like(win_start)

            rows_t = (
                self.c_row_tip.view(1, P)
                + self.c_row_shaft.view(1, P) * ctx.shaft_allowed[:, hi : hi + 1]
            ).clamp_max(1.0)
            rows_s = (
                self.c_row_tip.view(1, P)
                + self.c_row_shaft.view(1, P) * ctx.start_shaft_allowed[:, hi : hi + 1]
            ).clamp_max(1.0)
            tgt = ctx.target_drum[:, hi]
            src = ctx.start_drum[:, hi]
            oh_t = F.one_hot(tgt.clamp(min=0), D).to(dtype) * (tgt >= 0).to(dtype).unsqueeze(-1)
            oh_s = F.one_hot(src.clamp(min=0), D).to(dtype) * (src >= 0).to(dtype).unsqueeze(-1)
            ex = (
                win_strike.view(B, S, 1, 1) * rows_t.view(B, 1, P, 1) * oh_t.view(B, 1, 1, D)
                + win_start.view(B, S, 1, 1) * rows_s.view(B, 1, P, 1) * oh_s.view(B, 1, 1, D)
            ).clamp_max(1.0)
            keep = 1.0 - ex  # (B,S,P,D)
            smask = m_hand[hand].view(B, S, 1, 1)

            hd = _softplus_hinge(cfg.collision_margin - sdf_b, cfg.beta_collision)
            hu = _softplus_hinge(cfg.collision_margin - sdf_u, cfg.beta_collision)
            w = keep * smask
            wf = w.reshape(B, -1) if want_peak else None

            # --- time-bucketed drum collision (diagnostic only) ---------
            # Per-timestep hinge, mean over (points, drums) for this hand
            hsum = ((hd + hu) * w).sum(dim=(2, 3))        # (B, S)
            hcnt = w.sum(dim=(2, 3)).clamp_min(1.0)        # (B, S)
            hps = hsum / hcnt                               # (B, S)
            # Distance to hit in waypoints (positive = before hit)
            k_wp = k.float().view(B, 1)                     # (B, 1)
            dist_wp = k_wp - s_idx.float() / float(n_sub)   # (B, S)
            exempt_start = float(cfg.strike_exempt_waypoints)
            nohit = (k <= 0).view(B, 1)                     # (B, 1): idle hand
            b_far  = (dist_wp > 10.0) & ~nohit
            b_mid  = (dist_wp > 3.0) & (dist_wp <= 10.0) & ~nohit
            b_near = (dist_wp > exempt_start) & (dist_wp <= 3.0) & ~nohit
            b_xmpt = (dist_wp <= exempt_start) & ~nohit
            b_no   = nohit.expand(B, S)
            for tag, bm in [("far", b_far), ("mid", b_mid), ("near", b_near),
                            ("exempt", b_xmpt), ("nohit", b_no)]:
                bc = bm.float().sum(dim=1).clamp_min(1.0)
                diag[f"t_col_{tag}_{hand}"] = (
                    (hps * bm.float()).sum(dim=1) / bc
                ).detach()

            hand_pen_d = (hd * w).sum(dim=(1, 2, 3))
            hand_pen_u = (hu * w).sum(dim=(1, 2, 3))
            hand_n = w.sum(dim=(1, 2, 3))
            pen_d = pen_d + hand_pen_d
            pen_u = pen_u + hand_pen_u
            n_seen = n_seen + hand_n

            # --- self collision ---------------------------------------------
            Rt, pt = frames["torso"]
            off = lambda loc: (pt + (Rt @ loc.to(pt.dtype).view(1, 3, 1)).squeeze(-1)
                               ).reshape(B, S, 3).unsqueeze(2)
            a = off(self.c_torso_a)
            b = off(self.c_torso_b)
            hc = off(self.c_head_c)
            s_t = point_capsule_sdf(pts, a, b, self.c_torso_r)
            s_h = point_sphere_sdf(pts, hc, self.c_head_r)
            sdf_self = torch.minimum(s_t, s_h) - self.c_pt_radii.view(1, 1, P)
            ws = m_hand[hand].view(B, S, 1)
            hs = _softplus_hinge(cfg.collision_margin - sdf_self, cfg.beta_collision)
            pen_s = pen_s + (hs * ws).sum(dim=(1, 2))

            # --- raw violation (penetration depth), for the multipliers -------
            big = torch.full_like(sdf_b, 1e3)
            vd = torch.where(w > 0.5, (-sdf_b).clamp_min(0.0), torch.zeros_like(sdf_b))
            vu = torch.where(w > 0.5, (-sdf_u).clamp_min(0.0), torch.zeros_like(sdf_u))
            vs = (-sdf_self).clamp_min(0.0) * ws
            if want_peak:
                # The peak half reduces the PENETRATION, not the margin-softplus
                # the means reduce.  Measured on the champion's 243-case grid,
                # peaking the hinge separates the acceptance gate 2.9x -- WORSE
                # than the mean's 3.9x -- because the softplus has a floor
                # (0.0066 m at zero clearance, 0.0035 m at the 5 mm margin) that
                # every clean sample pays, and a max over a floor is mostly
                # floor.  The penetration is exactly zero on a clear element, so
                # the same reduction separates 5.0x.  It is also the quantity
                # ``ok_collision`` actually thresholds: the gate is
                # ``pen_count == 0``, with no margin and no smoothing.  The
                # margin and the softplus belong to the MEAN, whose job is dense
                # shaping on the safe side; the peak's job is the gate.
                peak_h += [vd.reshape(B, -1), vu.reshape(B, -1),
                           vs.reshape(B, -1)]
                peak_m += [wf, wf, ws.expand(B, S, P).reshape(B, -1)]
            hand_max = torch.maximum(
                torch.maximum(vd.amax(dim=(1, 2, 3)), vu.amax(dim=(1, 2, 3))),
                vs.amax(dim=(1, 2)),
            )
            v_max = hand_max if v_max is None else torch.maximum(v_max, hand_max)
            msd = torch.minimum(
                torch.where(w > 0.5, sdf_b, big).amin(dim=(1, 2, 3)),
                torch.where(w > 0.5, sdf_u, big).amin(dim=(1, 2, 3)),
            )
            msd = torch.minimum(
                msd, torch.where(ws > 0.5, sdf_self, big[..., 0]).amin(dim=(1, 2))
            )
            min_sdf = msd if min_sdf is None else torch.minimum(min_sdf, msd)

            # Per-hand breakdown.  The recorded trajopt baseline freezes its idle
            # arm at URDF neutral, which is itself inside a drum's under-volume;
            # its ``ok_collision`` therefore looks at the STRIKING hand only
            # (``coll_min_sdf_m``), while ``coll_min_sdf_all_m`` matches the
            # both-hands number this cost actually optimizes.
            diag[f"collision_hinge_{hand}"] = (
                (hand_pen_d + hand_pen_u) / hand_n.clamp_min(1.0)
            ).detach()
            diag[f"collision_min_sdf_{hand}"] = msd.detach()
            diag[f"collision_pen_max_{hand}"] = hand_max.detach()

        denom = n_seen.clamp_min(1.0) if cfg.reduction == "mean" else 1.0
        raw["collision_drum"] = pen_d / denom
        raw["collision_under"] = pen_u / denom
        raw["collision_self"] = pen_s / denom
        if want_peak:
            raw["collision_peak"] = smooth_max(
                torch.cat(peak_h, dim=1), torch.cat(peak_m, dim=1),
                cfg.collision_peak_mode, cfg.collision_peak_p,
                cfg.collision_peak_temp, dims=(1,),
            )
        else:
            # Exactly zero, with exactly zero gradient -- the hot path never
            # builds the concatenation and the term is bit-identical to the
            # pre-peak form.
            raw["collision_peak"] = torch.zeros_like(pen_d)
        viol["collision"] = v_max.detach() if v_max is not None else pen_d.detach() * 0
        # The peak group integrates the SAME per-sample max penetration as
        # ``collision``.  That is not an oversight: ``viol["collision"]`` has
        # always been a max (unlike ``viol["torque"]``, which is a mean), so
        # there is no second quantity to hand the new group, and making the
        # mean group's violation a mean would change the multiplier dynamics of
        # every existing run.  The two groups are still worth separating -- the
        # PENALTIES they scale differ by ``n_seen``, and each carries its own
        # ``rho`` -- but they will track the same signal.  See DECISIONS.
        viol["collision_peak"] = viol["collision"]
        diag["collision_min_sdf"] = min_sdf.detach()
        diag["collision_pen_max"] = v_max.detach()
        # Per-type breakdown (the three sub-costs before weighting)
        diag["collision_drum"] = raw["collision_drum"].detach()
        diag["collision_under"] = raw["collision_under"].detach()
        diag["collision_self"] = raw["collision_self"].detach()
        counts["collision"] = n_seen.detach()

    # ------------------------------------------------------------- torque --
    def _effort_fade(self, ctx: StrikeContext, n: int, dtype) -> torch.Tensor:
        """``(B, n, D)`` per-element weight for the ``tau^2`` effort term.

        1 far from every relevant strike, ramping linearly to 0 at the strike
        knot itself over ``cfg.effort_fade_wp`` knots.  An ARM's columns see
        only their own hand's hit; the WAIST columns see whichever hit is
        nearer, because the torso carries both arms and stiffening it inside
        either approach is the same mistake.  A hand with ``active = 0`` has no
        strike to protect and therefore no fade.

        Index convention: ``i`` runs over the same axis ``tau`` does, so it is
        knots in ``physics_mode="knot_fd"`` and sub-samples otherwise; the
        sub-sample spacing is folded in so ``effort_fade_wp`` always means
        WAYPOINTS.
        """
        cfg = self.cfg
        W = float(cfg.effort_fade_wp)
        dev = self.device
        step = 1.0 if cfg.physics_mode == "knot_fd" else 1.0 / float(cfg.n_sub)
        pos = torch.arange(n, device=dev, dtype=dtype).view(1, n, 1) * step
        if W <= 0.0:
            return torch.ones((), device=dev, dtype=dtype)
        f = []
        for hi in range(2):
            k = ctx.hit_index[:, hi].to(dtype).view(-1, 1, 1)
            act = ctx.active[:, hi].to(dtype).view(-1, 1, 1)
            d = k - pos                          # waypoints still to go, this hand
            # OFF for the last W waypoints (0 <= d <= W), then linearly back to
            # full over the W before that, and full again once the hit is PAST
            # (d < 0) -- a hand that has already struck has nothing left to
            # protect, and without that branch a first strike at knot 4 would
            # zero the waist for the whole rest of the horizon.
            ramp = ((d - W) / W).clamp(0.0, 1.0)
            fh = torch.where(d < 0, torch.ones_like(ramp), ramp)
            f.append(act * fh + (1.0 - act))     # inactive hand -> no fade at all
        f_left, f_right = f
        f_waist = torch.minimum(f_left, f_right)
        # (B, n, 3) -> (B, n, n_dof) by the joint's group, one index_select
        return torch.cat([f_waist, f_left, f_right], dim=2).index_select(
            2, self.c_joint_group
        )

    def _torque(self, sp, ss, v0, B, S, D, ctx, raw, viol, diag, counts) -> None:
        """RNEA torque against ``eta_torque`` x the effort limits.

        Two reductions of the SAME hinge array ``h``, both returned:

        * ``raw["torque"]`` -- the mean over valid ``(knot, joint)`` elements,
          the file-wide convention.
        * ``raw["torque_peak"]`` -- a differentiable smooth max over those same
          elements (:func:`smooth_max`), priced by ``cfg.w_torque_peak`` and by
          its OWN augmented-Lagrangian multiplier ``w["torque_peak"]``.

        Each half reports its own raw violation, because the two are the same
        physical quantity reduced differently and are ~``n_valid`` apart in
        magnitude:

        * ``viol["torque"]`` -- the per-element MEAN excess over the hard
          limits;
        * ``viol["torque_peak"]`` -- the per-sample MAX excess over the hard
          limits (identical to ``diag["torque_excess_max"]``), which is the
          quantity the acceptance gate thresholds.

        Both are detached, both are exactly zero on a feasible sample.

        The peak exists because the mean is the wrong SHAPE for the gate it is
        supposed to move.  ``ok_torque`` in the eval harness is a zero-tolerance
        max against the hard limits, and torque violations are extremely sparse:
        on the recorded eval CSVs of the phase-2 model's ``ok_torque`` failures
        the median case is over on ONE element out of ~30 knots x 17 joints
        (0.27 % of valid elements; p90 0.51 %, worst case 1.5 %) by a median
        0.87 N*m.  One such element moves the mean by ~0.3 % of its own size --
        roughly 1e-3 N*m of loss -- while failing the case outright.  Averaging
        is exactly the operation that destroys the signal, so the peak term puts
        a max-shaped one back alongside it (the mean is kept: it is what gives
        every element gradient on the safe side of the limit, and what keeps the
        term from collapsing onto whichever single element happens to be worst).
        """
        cfg = self.cfg
        if cfg.physics_mode == "knot_fd":
            q, qd, qdd = sp.knot_finite_differences(v0)
            idx = torch.arange(
                sp.max_segments + 1, device=q.device, dtype=torch.long
            ).view(1, -1, 1)
            mask = (idx <= sp.lengths.unsqueeze(1)).to(q.dtype)
        else:
            q, qd, qdd, mask = ss.q, ss.qd, ss.qdd, ss.mask
        n = q.shape[1]
        tau = self.rnea.rnea(
            q.reshape(B * n, D), qd.reshape(B * n, D), qdd.reshape(B * n, D)
        ).reshape(B, n, D)
        lim_hard = self.tau_hard.view(1, 1, D)
        if cfg.torque_derate == "linear":
            sat = cfg.torque_saturation_scale * lim_hard
            derate = sat * (1.0 - qd.abs() / self.v_hard.view(1, 1, D)).clamp(0.0, 1.0)
            lim_hard = torch.minimum(lim_hard.expand_as(derate), derate)
        lim = cfg.eta_torque * lim_hard
        excess = tau.abs() - lim
        h = _softplus_hinge(excess, cfg.beta_torque)
        ss = _Rates(q, qd, qdd, mask)  # local alias for the reductions below
        val, cnt = self._reduce(h, ss.mask, (1, 2))
        raw["torque"] = val
        raw["torque_peak"] = smooth_max(
            h, ss.mask, cfg.torque_peak_mode, cfg.torque_peak_p, cfg.torque_peak_temp,
            dims=(1, 2),
        )
        diag["torque_hinge_max"] = (h * ss.mask).amax(dim=(1, 2)).detach()
        counts["torque"] = cnt.detach()
        over = (tau.abs() - lim_hard).clamp_min(0.0) * ss.mask
        viol["torque"] = (over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)).detach()
        excess_max = over.amax(dim=(1, 2)).detach()
        # The peak group's AL multiplier is driven by the MAX excess -- the same
        # tensor the diagnostics report, exposed as a violation so
        # ``AugmentedLagrangian`` can integrate it.
        viol["torque_peak"] = excess_max
        diag["torque_excess_max"] = excess_max
        diag["torque_max"] = (tau.abs() * ss.mask).amax(dim=(1, 2)).detach()
        diag["torque_frac_over"] = (
            (over > 0).to(tau.dtype).sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        ).detach()

        # ---- strike-masked effort (an ENERGY proxy, not a limit) -------------
        # ``fade`` is 0 inside the last ``effort_fade_wp`` waypoints before each
        # hand's own hit, so the term cannot fight the impact-speed objective;
        # everywhere else it is a plain mean of tau^2 over the same valid
        # elements the hinge uses.  Always computed (it is also the diagnostic
        # the energy claim is measured with); priced only when ``w_effort > 0``.
        fade = self._effort_fade(ctx, n, tau.dtype)
        eff_mask = ss.mask * fade
        raw["effort"], cnt_eff = self._reduce(tau * tau, eff_mask, (1, 2))
        counts["effort"] = cnt_eff.detach()
        tau_sq = tau * tau
        diag["torque_sq_mean"] = (
            (tau_sq * ss.mask).sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        ).detach()
        diag["torque_abs_mean"] = (
            (tau.abs() * ss.mask).sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        ).detach()

    # ----------------------------------------------------------- velocity --
    def _velocity(self, sp: TrajectorySpline, raw, viol, diag, counts) -> None:
        cfg = self.cfg
        vmax = (
            sp.segment_velocity_max_analytic()
            if cfg.velocity_max_mode == "analytic"
            else sp.segment_velocity_max(cfg.n_vel_probe)
        )  # (B, H, n)
        lim_hard = self.v_hard.view(1, 1, -1)
        h = _softplus_hinge(vmax - cfg.eta_velocity * lim_hard, cfg.beta_velocity)
        val, cnt = self._reduce(h, sp.seg_mask, (1, 2))
        raw["velocity"] = val
        counts["velocity"] = cnt.detach()
        over = (vmax - lim_hard).clamp_min(0.0) * sp.seg_mask
        viol["velocity"] = (over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)).detach()
        diag["velocity_excess_max"] = over.amax(dim=(1, 2)).detach()
        diag["velocity_max"] = vmax.amax(dim=(1, 2)).detach()

    # ---------------------------------------------------------- joint box --
    def _joint_box(self, ss, raw, viol, diag, counts, w_waist_box=1.0) -> None:
        """Sub-sampled ``q`` against the (optionally waist-narrowed) box.

        ``cfg.waist_pr_box`` shrinks the ``waist_pitch`` / ``waist_roll`` rows
        only.  It moves the SOFT half-width (to ``eta_position x waist_pr_box``,
        so the tightened rows are held to the same 80 % conservatism as every
        other joint) AND the HARD limits the raw violation is measured against
        (to ``+-waist_pr_box``), which is what makes the new constraint flow
        through the existing ``joint_box`` multiplier instead of needing one of
        its own.  ``w_waist_box`` linearly blends the half-width between the
        original box (0) and the shrunk one (1) so a continuation can ramp the
        constraint in rather than stepping the augmented Lagrangian.
        """
        cfg = self.cfg
        half = cfg.eta_position * self.q_half.view(1, 1, -1)
        lo_hard = self.q_lo_hard.view(1, 1, -1)
        hi_hard = self.q_hi_hard.view(1, 1, -1)
        if cfg.waist_pr_box is not None:
            m = self.c_waist_pr.view(1, 1, -1)
            tight = float(cfg.waist_pr_box)
            half = half + (w_waist_box * m) * (cfg.eta_position * tight - half)
            lo_hard = lo_hard + (w_waist_box * m) * (-tight - lo_hard)
            hi_hard = hi_hard + (w_waist_box * m) * (tight - hi_hard)
        dev = (ss.q - self.q_mid.view(1, 1, -1)).abs()
        h = _softplus_hinge(dev - half, cfg.beta_position)
        val, cnt = self._reduce(h, ss.mask, (1, 2))
        raw["joint_box"] = val
        counts["joint_box"] = cnt.detach()
        over = torch.maximum(lo_hard - ss.q, ss.q - hi_hard).clamp_min(0.0) * ss.mask
        viol["joint_box"] = (over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)).detach()
        diag["joint_box_excess_max"] = over.amax(dim=(1, 2)).detach()

    # -------------------------------------------------------- smoothness ---
    def _smoothness(self, sp: TrajectorySpline, raw, diag, counts) -> None:
        ints = sp.segment_integrals()
        cnt = sp.seg_mask.sum(dim=(1, 2))
        d = cnt.clamp_min(1.0) if self.cfg.reduction == "mean" else 1.0
        for key, name in (("vel2", "smooth_vel"), ("acc2", "smooth_acc"),
                          ("jerk2", "smooth_jerk")):
            raw[name] = ints[key].sum(dim=(1, 2)) / d
            diag[name + "_integral"] = ints[key].sum(dim=(1, 2)).detach()
        counts["smooth"] = cnt.detach()

    # ----------------------------------------------------------- posture ---
    def _posture(self, frames, ss, B, S, ctx, raw, diag, counts) -> None:
        cfg = self.cfg
        # mid-range well, normalized by half-range so every joint counts equally
        u = (ss.q - self.q_mid.view(1, 1, -1)) / self.q_half.view(1, 1, -1).clamp_min(1e-6)
        val, cnt = self._reduce(u * u, ss.mask, (1, 2))
        raw["posture_mid"] = val
        counts["posture"] = cnt.detach()
        diag["posture_mid_max"] = (u.abs() * ss.mask).amax(dim=(1, 2)).detach()

        # elbow-down: the elbow must not rise above the shoulder->wrist chord
        m_hand = {h: ss.mask.index_select(2, self._gcols[h]).amax(dim=2) for h in _HANDS}
        pen = 0.0
        n = 0.0
        worst = None
        for hand in _HANDS:
            ze = frames[f"{hand}_elbow"][1].reshape(B, S, 3)[..., 2]
            zw = frames[f"{hand}_wrist"][1].reshape(B, S, 3)[..., 2]
            zs = frames[f"{hand}_shoulder"][1].reshape(B, S, 3)[..., 2]
            drop = 0.5 * (zs + zw) - ze  # > 0 means elbow below the chord
            h = _softplus_hinge(-(drop + cfg.elbow_down_margin), cfg.beta_elbow)
            pen = pen + (h * m_hand[hand]).sum(dim=1)
            n = n + m_hand[hand].sum(dim=1)
            w = (-(drop + cfg.elbow_down_margin)).clamp_min(0.0) * m_hand[hand]
            worst = w.amax(dim=1) if worst is None else torch.maximum(worst, w.amax(dim=1))
        raw["posture_elbow"] = pen / (n.clamp_min(1.0) if cfg.reduction == "mean" else 1.0)
        diag["posture_elbow_max"] = worst.detach()

        # anti-axial twist: rank-1 quadratic per hand about the limb long axis
        dq = ss.q - self.q_ready.view(1, 1, -1)
        tw = torch.einsum("bsn,hn->bsh", dq, self.c_twist)  # (B, S, 2)
        hm = torch.stack([m_hand["left"], m_hand["right"]], dim=-1)
        val, cnt2 = self._reduce(tw * tw, hm, (1, 2))
        raw["posture_twist"] = val
        diag["posture_twist_max"] = (tw.abs() * hm).amax(dim=(1, 2)).detach()
        counts["posture_twist"] = cnt2.detach()

        # UPRIGHT WELL: mean q^2 over the waist PITCH/ROLL rows only.  The mask
        # is ``ss.mask * c_waist_pr``, so (a) the mean is over 2 rows rather than
        # 17 -- the weight is then directly comparable to a target excursion in
        # rad^2 -- and (b) waist YAW is multiplied by a hard zero, so the term is
        # exactly yaw-invariant and no yaw gradient exists to leak.
        m_pr = ss.mask * self.c_waist_pr.view(1, 1, -1)
        val_up, cnt_up = self._reduce(ss.q * ss.q, m_pr, (1, 2))
        raw["posture_upright"] = val_up
        counts["posture_upright"] = cnt_up.detach()
        # Waist usage diagnostics, always reported: the point of the upright
        # term is to move these, and yaw is here to prove it did NOT move.
        qa = ss.q.abs() * ss.mask
        diag["waist_pr_max"] = (qa * self.c_waist_pr.view(1, 1, -1)).amax(dim=(1, 2)).detach()
        diag["waist_yaw_max"] = (qa * self.c_waist_yaw.view(1, 1, -1)).amax(dim=(1, 2)).detach()

        # WAIST-YAW RATE well.  Same shape as the upright well one block up, on
        # ``qd`` instead of ``q`` and on the YAW row instead of pitch/roll, so
        # the two are exact complements: that one says where the torso may lean,
        # this one says how fast it may turn.  Reduced over the yaw row alone,
        # which is what makes the weight readable as a target rate in
        # (rad/s)^2.  The peak rate is reported unconditionally -- it is the
        # number the ablation is actually about, and a run with the weight at
        # zero still has to be able to show what its waist was doing.
        m_yaw = ss.mask * self.c_waist_yaw.view(1, 1, -1)
        val_yv, cnt_yv = self._reduce(ss.qd * ss.qd, m_yaw, (1, 2))
        raw["waist_yaw_vel"] = val_yv
        counts["waist_yaw_vel"] = cnt_yv.detach()
        diag["waist_yawd_max"] = (
            ss.qd.abs() * m_yaw
        ).amax(dim=(1, 2)).detach()
        diag["waist_prd_max"] = (
            ss.qd.abs() * ss.mask * self.c_waist_pr.view(1, 1, -1)
        ).amax(dim=(1, 2)).detach()

        # IDLE ARM: park at the home posture, and arrive stopped.
        #
        # The mask is ``m_hand[h] * (1 - active[h])``, so (a) only a hand with no
        # scheduled strike is priced -- a striking hand's terminal objective is
        # never contested -- and (b) the reduction is over that arm's 7 rows
        # only.  ``home`` is a CONSTANT per hand: arm joint angles are already
        # torso-relative, so one posture holds its relationship to the body at
        # every waist yaw.  The term is therefore exactly yaw-independent -- no
        # waist column appears in it at all, so no yaw gradient can leak and the
        # target the arm is asked to hold does not move when the torso turns.
        idle_pen = ss.q.new_zeros(B)
        idle_vel = ss.q.new_zeros(B)
        idle_n = ss.q.new_zeros(B)
        idle_worst = ss.q.new_zeros(B)
        if self.c_idle_q is not None and (cfg.w_idle_home > 0.0 or cfg.w_idle_vel > 0.0):
            S_ = ss.q.shape[1]
            # "final velocity zero": a (k / K)^2 profile over the hand's OWN
            # horizon -- ~0 where the arm is still travelling into home, 1 at the
            # terminal knot.  ``lengths`` is in segments, the sub-sample grid is
            # ``n_sub`` finer, and sample K*n_sub is the terminal knot.
            for hi, hand in enumerate(_HANDS):
                cols = self._gcols[hand]
                idle = 1.0 - ctx.active[:, hi]                          # (B,)
                # THIS hand's terminal sample.  ``m_hand`` is 1 up to and
                # including it (``SplineSamples.mask`` includes the terminal
                # knot), so the count is that index + 1.
                k = (m_hand[hand].sum(dim=1) - 1.0).clamp(0.0, S_ - 1).long()
                g = k.view(-1, 1, 1).expand(-1, 1, cols.numel())
                q_k = ss.q.index_select(2, cols).gather(1, g).squeeze(1)   # (B, n_arm)
                qd_k = ss.qd.index_select(2, cols).gather(1, g).squeeze(1)
                # ---- position: be at home at the terminal knot.
                e = q_k - self.c_idle_q[hi]
                sq = (e * e).sum(-1)                                    # rad^2
                idle_pen = idle_pen + idle * sq
                d = torch.sqrt(sq + 1e-12)
                # ---- velocity: arrive stopped.  The idle arm's counterpart of
                # ``w_terminal_velocity`` -- the striking hand is asked for
                # 2 m/s, the parked one for zero.
                idle_vel = idle_vel + idle * (qd_k * qd_k).sum(-1)
                idle_n = idle_n + idle
                idle_worst = torch.maximum(idle_worst, idle * d)
        den_idle = idle_n.clamp_min(1.0)
        raw["posture_idle_home"] = idle_pen / den_idle
        raw["posture_idle_vel"] = idle_vel / den_idle
        counts["posture_idle"] = idle_n.detach()
        diag["idle_home_dist"] = idle_worst.detach()

        # Waist-aim IL: penalise yaw, pitch, and roll deviation from the
        # reference trajectory.  Yaw follows waist_aim; pitch and roll are 0.
        # The reference is at knot resolution (H); expand to sub-sample grid (S).
        if ctx.waist_yaw_ref is not None:
            yaw_ref = ctx.waist_yaw_ref  # (B, H)
            H_ref = yaw_ref.shape[1]
            n_sub = int(cfg.n_sub)
            S = ss.q.shape[1]
            # Map sub-sample index → knot index
            knot_idx = (torch.arange(S, device=ss.q.device) // n_sub).clamp(0, H_ref - 1)
            yaw_ref_ss = yaw_ref[:, knot_idx]  # (B, S)
            # Full 3-DOF reference: yaw, pitch=0, roll=0
            ref_ss = torch.stack([
                yaw_ref_ss,
                torch.zeros_like(yaw_ref_ss),
                torch.zeros_like(yaw_ref_ss),
            ], dim=-1)  # (B, S, 3)
            # Waist column indices in joint order
            yaw_i = int(self.c_waist_yaw.nonzero(as_tuple=True)[0][0])
            pr_idx = self.c_waist_pr.nonzero(as_tuple=True)[0]  # (2,)
            # Reorder to [yaw, pitch, roll] matching waist_aim convention
            names = self.joint_names
            waist_order = []
            for nm in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"):
                if nm in names:
                    waist_order.append(names.index(nm))
            q_waist = ss.q[:, :, waist_order]  # (B, S, 3)
            sq_err = ((q_waist - ref_ss) ** 2).sum(dim=-1)  # (B, S)
            # Reduce over time, masked by validity
            m_any = ss.mask.amax(dim=2)  # (B, S)
            val_aim = (sq_err * m_any).sum(dim=1) / m_any.sum(dim=1).clamp_min(1.0)
            raw["waist_aim"] = val_aim
        else:
            raw["waist_aim"] = ss.q.new_zeros(B)

    # ---------------------------------------------------------- terminal ---
    def _terminal(self, sp: TrajectorySpline, ctx: StrikeContext,
                  raw, viol, diag, counts) -> None:
        cfg = self.cfg
        B = sp.x.shape[0]
        dtype = sp.x.dtype
        dt = self.dt if cfg.impact_velocity_mode == "fd_coarse" else self.dt / cfg.n_sub

        reg = sp.x.new_zeros(B)
        reg_raw = sp.x.new_zeros(B)
        aim = sp.x.new_zeros(B)
        vel = sp.x.new_zeros(B)
        elev_pen = sp.x.new_zeros(B)
        n_act = sp.x.new_zeros(B)
        d_pos, d_speed, d_elev, d_cos = [], [], [], []

        for hi, hand in enumerate(_HANDS):
            k = ctx.hit_index[:, hi].clamp(1, sp.max_segments)
            q_k = sp.knot(k)
            if cfg.impact_velocity_mode == "fd_coarse":
                q_p = sp.knot(k - 1)
            else:
                # one sub-sample step back along the spline
                s_back = torch.full((1,), self.dt - dt, device=sp.x.device, dtype=dtype)
                qs, _v, _a, _j = sp.eval_offsets(s_back)  # (B, H, 1, n)
                idx = (k - 1).view(-1, 1, 1, 1).expand(-1, 1, 1, self.n_dof)
                q_p = qs.gather(1, idx).reshape(B, self.n_dof)
            qq = torch.cat([q_k, q_p], dim=0)
            R, p = self.fk.frame(qq, f"{hand}_wrist")
            base = p + (R @ self.c_mount_p[hand].to(dtype).view(1, 3, 1)).squeeze(-1)
            u = (R @ self.c_mount_axis[hand].to(dtype).view(1, 3, 1)).squeeze(-1)
            base_k, base_p = base[:B], base[B:]
            u_k, u_p = u[:B], u[B:]

            # contact candidates
            if cfg.contact_mode == "spec":
                sig = ctx.contact_sigma[:, hi].view(B, 1, 1)
                valid = sp.x.new_ones(B, 1)
            else:
                sig = self.c_sigmas.to(dtype).view(1, -1, 1)
                valid = (
                    self.c_sigma_is_tip.to(dtype).view(1, -1)
                    + ctx.shaft_allowed[:, hi : hi + 1]
                ).clamp_max(1.0)  # (B, Sc)
            pc_k = base_k.unsqueeze(1) + sig * u_k.unsqueeze(1)  # (B, Sc, 3)
            pc_p = base_p.unsqueeze(1) + sig * u_p.unsqueeze(1)
            v_c = (pc_k - pc_p) / dt

            n_out = ctx.n_out[:, hi].unsqueeze(1)
            e = pc_k - ctx.p_star[:, hi].unsqueeze(1)
            axial = (e * n_out).sum(-1)
            d_in = e - axial.unsqueeze(-1) * n_out - ctx.disk_offset[:, hi].unsqueeze(1)
            r_disk = (
                torch.sqrt((d_in * d_in).sum(-1) + 1e-12) - ctx.hit_radius[:, hi : hi + 1]
            ).clamp_min(0.0)
            hp = (d_in * ctx.half_plane_dir[:, hi].unsqueeze(1)).sum(-1)
            r_hp = (-hp).clamp_min(0.0) * ctx.has_half_plane[:, hi : hi + 1]
            r_ax = (axial.abs() - cfg.terminal_axial_tol).clamp_min(0.0)
            dist = torch.sqrt(r_disk**2 + r_hp**2 + r_ax**2 + 1e-12)  # (B, Sc)

            # soft-min over the legal contact span
            big = 1e3 * (1.0 - valid)
            logits = -(dist + big) / cfg.contact_softmin_temp
            wgt = torch.softmax(logits, dim=1)
            act = ctx.active[:, hi]

            reg = reg + act * (wgt * _softplus_hinge(dist, cfg.beta_terminal)).sum(1)
            reg_raw = reg_raw + act * (wgt * dist).sum(1)
            aim_d = torch.sqrt((e * e).sum(-1) + 1e-12)
            aim = aim + act * (wgt * aim_d).sum(1)

            # impact velocity, at the same (soft-selected) contact point
            vs = ctx.v_star[:, hi].unsqueeze(1)
            speed_star = torch.sqrt((vs * vs).sum(-1) + 1e-12)
            nv = vs / speed_star.unsqueeze(-1)
            dv = v_c - vs
            along = (dv * nv).sum(-1)
            perp2 = (dv * dv).sum(-1) - along * along
            vcost = (
                cfg.terminal_a_normal * along * along
                + cfg.terminal_a_tangential * perp2.clamp_min(0.0)
            )
            vel = vel + act * (wgt * vcost).sum(1)

            # stick elevation band on |elev|
            cosang = -(u_k * ctx.n_out[:, hi]).sum(-1).clamp(-1.0, 1.0)
            el = torch.asin(cosang).abs()
            lo = float(np.deg2rad(cfg.elev_min_deg))
            hi_ = float(np.deg2rad(cfg.elev_max_deg))
            ep = (
                _softplus_hinge(lo - el, cfg.beta_elev)
                + _softplus_hinge(el - hi_, cfg.beta_elev)
            )
            elev_pen = elev_pen + act * ctx.elev_required[:, hi] * ep
            n_act = n_act + act

            # diagnostics at the SPEC contact point (what the harness scores)
            sig_spec = ctx.contact_sigma[:, hi].view(B, 1)
            pk = base_k + sig_spec * u_k
            pp = base_p + sig_spec * u_p
            vspec = (pk - pp) / dt
            d_pos.append(
                act * torch.sqrt(((pk - ctx.p_star[:, hi]) ** 2).sum(-1) + 1e-12)
            )
            sp_ = torch.sqrt((vspec * vspec).sum(-1) + 1e-12)
            d_speed.append(act * (sp_ - speed_star.squeeze(1)).abs())
            d_cos.append(act * (vspec * nv.squeeze(1)).sum(-1) / sp_.clamp_min(1e-9))
            d_elev.append(act * torch.rad2deg(el))

        den = n_act.clamp_min(1.0)
        raw["terminal_region"] = reg / den
        raw["terminal_aim"] = aim / den
        raw["terminal_velocity"] = vel / den
        raw["terminal_elev"] = elev_pen / den
        # Raw violation = distance OUTSIDE the legal hit region (0 when on head).
        viol["terminal"] = (reg_raw / den).detach()
        diag["terminal_region_dist"] = (reg_raw / den).detach()
        diag["contact_pos_err"] = (sum(d_pos) / den).detach()
        diag["contact_speed_err"] = (sum(d_speed) / den).detach()
        diag["contact_dir_cos"] = (sum(d_cos) / den).detach()
        diag["stick_elev_deg"] = (sum(d_elev) / den).detach()
        counts["terminal"] = n_act.detach()
