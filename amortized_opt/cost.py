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
``collision``  stick capsule (butt→tip) and wrist sphere against
               every drum body / under-volume, plus the same
               capsule and wrist against the torso capsule and
               head sphere.  Target-drum and departure exemptions
               cover the stick capsule (you have to touch a drum
               to hit it); the wrist is never exempted.
``torque``     RNEA (armature ON) at the sub-sampled states against
               ``eta_torque`` x the Isaac effort limits.  Priced twice: a MEAN
               softplus warning over the (knot, joint) hinge elements (fixed
               ``cfg.w_torque``, never × λ), plus an optional SMOOTH-MAX of the
               ReLU excess (``w_torque_peak`` × ``λ_peak``) because the harness
               gate is a zero-tolerance max -- see :meth:`FusedCost._torque`.
               Per-actuator-class AL groups (``torque_*``) multiply the ReLU
               residual (``live``), which is already zero when feasible.
               TRACKING half (off by default, ``w_track_torque`` /
               ``w_track_torque_peak``): the same softplus warning on
               ``tau_rnea + tau_fb``, the torque the PD servo DEMANDS while
               tracking the plan.  It shares the ``torque`` term key and, when
               priced, supplies the ``torque`` group's raw violation -- see
               :meth:`FusedCost._track_torque` and the note at the top of
               :mod:`amortized_opt.pd_sim`.
``effort``     optional ``w_effort`` x mean ``tau^2`` on the same RNEA
               torques -- an energy proxy, faded to zero over the
               last ``effort_fade_wp`` waypoints before each hand's
               hit so it cannot fight the impact-speed term.  A plain
               objective: no multiplier.
``torque_static``  optional ``w_torque_static`` x mean softplus on
               ``|RNEA(q,0,0)|`` against the effort limits -- a
               gravity-only static proxy with no augmented-Lagrangian
               multiplier.  With ``torque_static_only`` the coupled
               RNEA hinge and its torque AL groups are skipped.
``velocity``   per-segment max ``|qd|`` against ``eta_velocity`` x
               the speed ratings.
``joint_box``  decoder **waypoints** against ``eta_position`` x the
               posture-narrowed URDF box, optionally with the waist
               PITCH/ROLL rows shrunk to ``+-waist_pr_box`` (yaw
               untouched -- the torso must spin to reach the kit).
               The warning is a mean softplus; the peak half (when
               enabled) is a **smooth max** of per-(waypoint, joint)
               ReLU hard excess (max-shaped dense grad).
``smooth``     closed-form ``int v^2 / a^2 / j^2`` per segment.
``posture``    mid-range well, elbow-down preference, anti-axial
               twist, a compact arm-mass proxy, and an optional UPRIGHT well on waist
               pitch/roll (``w_waist_upright``; yaw excluded).
               With ``waist_pr_ref`` set the well is
               ``mean((q_pr - stopgrad(sp))²)`` -- a decoder tax,
               not an encoder teacher.
``waist_pr_terminal``
               V_f on P/R at the NEARER active hit knot (min of the
               two hands' hit_index), always vs 0.  The later hit is
               only context; this is the strike the committed prefix
               is for.  Own objective term; not mixed into ``posture``.
``waist_pr_band``
               decoder waist P/R vs ``stopgrad(encoder sp)``, peak
               rad.  Own AL group (not mixed into ``posture``).
=============== ====================================================

Limit families (torque / velocity / joint_box, split by actuator class)
plus collision, terminal, ``torque_peak``, ``collision_peak`` and
``waist_pr_band`` are *constraint* groups: they carry an
augmented-Lagrangian multiplier (see :mod:`amortized_opt.lagrangian`) that
multiplies the ReLU residual (``CostTerms`` live / violations), which is
already zero when feasible.  The softplus warning sits at a fixed
``CostConfig.w_*`` (plus non-λ curriculum: ``limit_alpha``, terminal
sub-ramps, waist box, peak alphas).  Smooth / posture / aim / rate wells
are plain objective terms.

Hinges
------

Every limit is ``softplus((x - lim) / beta) * beta`` as the plan specifies, with
``lim`` a *conservative* fraction of the hard limit (torque 80 %, velocity 70 %,
joint box 80 %).  Softplus is strictly positive, so a perfectly feasible sample
still pays ``beta * log 2`` per element; that constant is the warning that
buys gradient on the safe side of the limit, priced at ``cfg.w_*`` and never
by λ.  The hard wall is standard PHR: ``λ [g]₊ + (pen / 2) [g]₊²`` on the
raw ``relu(x - hard_limit)`` excess, which *is* zero when feasible.

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

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    CONTROL_DT,
    FIXED_LOWER_BODY_CAPSULES,
    G1_UPPER_BODY_JOINTS,
    HEAD_SPHERE,
    PD_SUBSTEPS,
    PELVIS_IN_ENV,
    SHAFT_TIP_CLEAR,
    STICK_LENGTH,
    STICK_MOUNT_POS_LEFT,
    STICK_MOUNT_POS_RIGHT,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
    WRIST_RADIUS,
    ready_joint_pos,
    static_torque_tighten_divisors,
    torque_limit_eta_scale,
)
from .lagrangian import (
    ACTUATOR_SEGMENTS,
    ACTUATOR_SEGMENT_NAMES,
    LIMIT_FAMILIES,
    TERM_FOR_GROUP,
    limit_group_name,
)
from .fk import (
    ChainFK,
    stick_axis_local,
    stick_mount_transform,
    stick_shaft_offsets,
    stick_tip_offset,
)
from .idle_cylinder import tip_in_torso
from .limits import (
    ETA_POSITION,
    ETA_TORQUE,
    ETA_VELOCITY,
    available_torque_torch,
    isaac_effort_limits,
    scale_soft_wrist_effort,
    isaac_velocity_limits,
    urdf_position_limits,
)
from .rnea import BatchedRNEA
from .sdf import (
    DrumSceneSpec,
    fixed_capsule_self_sdf,
    point_capsule_sdf,
    point_cylinder_sdf,
    point_sphere_sdf,
    segment_cylinder_sdf,
    segment_hit_region,
    segment_segment_sdf,
)
# Bound separately from ``segment_segment_sdf`` so the stick-stick "default
# off" sabotage test can patch the CROSS-HAND pair without also shifting
# stick-vs-torso / stick-vs-head, which use the same primitive.
cross_hand_sdf = segment_segment_sdf
from .strike import (
    CYMBAL_CONTACT_SIGMA_FALLBACK,
    STICK_ELEV_MAX_DEG,
    STICK_ELEV_MIN_DEG,
    STICK_TIP_SIGMA,
    contact_sigma_candidates,
    shaft_hit_allowed,
    strike_axial_limits,
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
    "stick_cost_from_context",
    "smooth_max",
    "strike_context_from_cases",
]

_HANDS = ("left", "right")
#: Contact arc coordinates probed when the drum may be struck by the shaft.
#: ``contact_sigma_candidates("cymbal_left")`` -> (0.10, 0.15, 0.20, 0.05), i.e.
#: the bead plus the 15 cm of shaft behind it the plan calls out.
SHAFT_SIGMAS: tuple[float, ...] = contact_sigma_candidates("cymbal_left")

#: Peak groups whose differentiable price is already
#: ``λ * cfg.w_*_peak * smooth_max(ReLU)``.  The linear ``λ * live`` add
#: in :meth:`FusedCost.forward` skips these so the peak is not counted
#: twice.  The PHR quadratic squares that same smooth max (``raw``), not
#: the horizon hard max: ``live`` stays the detached gate statistic the
#: dual integrates, and a one-hot ``amax`` in the backward is what made
#: the penalty swap joints between steps.
_PEAK_AL_GROUPS = frozenset({
    "torque_peak", "velocity_peak", "collision_peak", "joint_box_peak",
})


# ============================================================== config ======
@dataclass
class CostConfig:
    """Weights, margins and hinge temperatures.

    Defaults are informed by what ``TrajOptConfig.validate()`` actually *leaves
    enabled* under ``simplified_cost`` (which is the repo default), not by the
    declared dataclass fields -- see the per-field notes.  The absolute
    magnitudes are NOT trajopt's: trajopt prices hard constraints through
    ``mu_*`` slack penalties in the 1e5..1e9 range because they must outbid a
    jerk Hessian near 3e8.  Here the hard wall is ``λ [g]₊ + (pen/2)[g]₊²``
    and the softplus warning sits at a fixed ``w_*`` (never × λ), so the
    weights below are the *relative* warning prices and are O(1)-ish.
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
    #: Soft in-plane pull toward the AIM (not the geometric head centre) after
    #: a deadband.  ``CostConfig`` keeps this at 0 so ``validate()`` stays
    #: bit-identical with trajopt (which zeroes ``w_strike_position``);
    #: :func:`~amortized_opt.train_cost.phase2_cost_config` turns on a weak
    #: weight so training does not park on the rim.  Priced as a plain
    #: objective -- it does **not** ride ``lam_terminal``.
    w_terminal_aim: float = 0.0
    #: In-plane metres inside which ``w_terminal_aim`` is idle.  Matches the
    #: eval harness's 2 cm hit slack: closer than this, MPC can finish the
    #: strike.
    terminal_aim_deadband_m: float = 0.02
    #: Impact velocity, split normal / tangential to the commanded direction.
    w_terminal_velocity: float = 20.0
    terminal_a_normal: float = 1.0
    #: ``validate()`` zeroes the tangential component.  Kept at the DECLARED
    #: ratio (7 / 50 = 0.14) instead: the harness scores speed MAGNITUDE and
    #: direction cosine, both of which a zero tangential weight leaves unpriced.
    terminal_a_tangential: float = 0.14
    w_terminal_elev: float = 5.0
    #: Aim-frame axial band of the legal hit cylinder (outward normal).  Defaults
    #: to the interior 1 cm slab from the aim to the head surface; see
    #: :func:`~amortized_opt.strike.strike_axial_limits`.
    terminal_axial_lo: float = field(default_factory=lambda: strike_axial_limits()[0])
    terminal_axial_hi: float = field(default_factory=lambda: strike_axial_limits()[1])
    elev_min_deg: float = STICK_ELEV_MIN_DEG
    elev_max_deg: float = STICK_ELEV_MAX_DEG
    #: ``"softmin"`` (default) scores the playing-shaft capsule on cymbals /
    #: hi-hat -- the exact min of the legal-region hypot over
    #: :func:`~amortized_opt.strike.shaft_hit_sigma_range`, softened over the
    #: same critical points -- and the bead on pads.  ``"spec"`` collapses the
    #: capsule to the strike's own ``contact_sigma`` (ablation).  Eval, probe
    #: and the RL scorer always use the capsule, matching ``"softmin"``.
    contact_mode: str = "softmin"
    contact_softmin_temp: float = 0.01  # m
    #: ``"fd_coarse"`` = ``(p(q_k) - p(q_{k-1})) / dt`` -- the harness /
    #: FOH-plant convention.  ``"fd_fine"`` uses the spline sub-sample spacing.
    impact_velocity_mode: str = "fd_coarse"

    # ------------------------------------------------------- collision ----
    w_collision_drum: float = 100.0
    w_collision_under: float = 100.0
    w_collision_self: float = 100.0
    #: Price on the CROSS-HAND (stick vs stick) collision mean.  The three
    #: weights above score one hand's stick capsule and wrist against the DRUMS
    #: and against the torso/head; the two hands' own geometry was never
    #: compared, so nothing in the cost ever forbade the two sticks from passing
    #: through each other.  This is that term.
    #:
    #: A capsule pair, same geometry the per-hand block uses against the drums:
    #: the full 0.40 m stick plus the two wrist balls -- see
    #: :func:`~amortized_opt.sdf.segment_segment_sdf`.
    #:
    #: ZERO by default, so every existing checkpoint and every live run is
    #: bit-identical: at zero the term contributes an exact ``0.0 *`` product,
    #: and the cross-hand hinge is kept OUT of the shared ``collision_peak`` /
    #: ``viol["collision"]`` / ``collision_min_sdf`` reductions (it joins them
    #: only once the weight is positive).  The diagnostics ``collision_stick``
    #: and ``collision_stick_min_sdf`` are computed either way, so a run can
    #: measure the cross-hand clearance before paying for it.  Scale: the raw is
    #: the same margin-softplus, in metres, that ``w_collision_self`` prices.
    w_stick_stick: float = 0.0
    #: TIP LATERAL ORDERING (plain objective, not AL).  Softplus hinge on
    #: ``margin - (tip_L.y - tip_R.y)`` in the torso frame, so reversed tips
    #: (left tip more rightward than right tip) pay.  Distinct from
    #: ``w_stick_stick``, which prices capsule clearance -- sticks can swap
    #: sides while staying physically separated.  ZERO by default (bit-
    #: identical totals); diagnostics ``tip_lateral_min`` /
    #: ``tip_crossover_max`` are computed either way.
    w_tip_crossover: float = 0.0
    #: Metres of required ``tip_L.y - tip_R.y`` before the hinge is zero.
    #: 0 means any reversed ordering is penalized.
    tip_crossover_margin: float = 0.0
    #: Softplus temperature (metres) for the tip-ordering hinge.
    beta_tip_crossover: float = 0.01
    #: Price on a smooth max of ReLU penetration over scored pairs, alongside
    #: the three mean softplus sub-costs.  AL ``live`` stays the per-sample
    #: **hard max** penetration (``ok_collision``).  ZERO by default.
    #: See :meth:`FusedCost._collision`.
    w_collision_peak: float = 0.0
    #: Differentiable peak shape for ``raw["collision_peak"]``.
    collision_peak_mode: str = "pnorm"
    collision_peak_p: float = 12.0
    #: LSE temperature, METRES.  Only read when ``collision_peak_mode == "lse"``.
    collision_peak_temp: float = 5.0e-4
    #: Waypoints at the end of a hand's horizon whose contact geoms are exempt
    #: from the target drum, and at the start exempt from the drum the pose is
    #: departing.  Both mirror ``TrajectoryScorer``.
    strike_exempt_waypoints: int = 2
    start_exempt_waypoints: int = 2
    shaft_samples: int = 4

    # ---- torque physics curriculum (level A -> full RNEA) -----------------
    #: ``"full"`` -- standard coupled RNEA in the torque hinge.  ``"diag_grav"``
    #: -- per-joint ``diag(M(q)) @ qdd + g(q)`` (no Coriolis, no off-diagonal
    #: inertia).  ``"diag_grav_frozen"`` -- same with ``diag(M(q_ready))`` cached
    #: at build (cheap).  Blend to full RNEA via ``weights["torque_full_blend"]``.
    torque_physics: str = "full"

    # ---------------------------------------------------------- limits ----
    w_torque: float = 1.0
    #: Price on a smooth max of ReLU hard torque excess, alongside ``w_torque``'s
    #: mean softplus.  ``ok_torque`` is a zero-tolerance max; the priced half
    #: is ``λ_peak * w_torque_peak * smooth_max(ReLU(|τ|-lim))`` so the worst
    #: cells share gradient without one-hot active-set jumps.  AL ``live`` for
    #: this group stays the horizon **hard max**.  ZERO by default.
    #: See :meth:`FusedCost._torque`.
    w_torque_peak: float = 0.0
    #: Differentiable peak shape for ``raw["torque_peak"]``.
    torque_peak_mode: str = "pnorm"
    torque_peak_p: float = 12.0
    #: LSE temperature, N*m.  Only read when ``torque_peak_mode == "lse"``.
    torque_peak_temp: float = 0.05

    # ---- TRACKING TORQUE (the demand the PD controller actually makes) ----
    #: Price on the MEAN of the tracking-torque hinge -- the same softplus, at
    #: the same ``beta_torque``, on ``|tau_rnea + tau_fb|`` instead of
    #: ``|tau_rnea|``.  ``tau_fb`` is the estimated PD feedback demand
    #: (:func:`~amortized_opt.pd_sim.feedback_torque`), which is what actually
    #: saturates at deployment: the plan's own RNEA can be clean on every joint
    #: while the servo tracking it asks for 6x the wrist rating.
    #:
    #: When either tracking weight is positive the ``torque`` constraint group's
    #: RAW VIOLATION switches to the tracking excess -- ONE multiplier on ONE
    #: constraint, the one that describes the hardware.  ``w_torque`` /
    #: ``w_torque_peak`` keep working unchanged (they are the rollback), and at
    #: the default 0.0 for both weights below nothing here is computed at all:
    #: the whole block is behind a python branch, so the cost is bit-for-bit and
    #: cost-free.
    w_track_torque: float = 0.0
    #: Price on the SMOOTH-MAX of the same hinge, by the same argument that
    #: created ``w_torque_peak`` (the gate is a max, the mean cannot see it).
    #: Unlike ``w_torque_peak`` this is a FIXED weight with no
    #: augmented-Lagrangian multiplier of its own -- it rides the per-joint
    #: ``limit_alpha`` curriculum ramp the mean hinges use, and nothing else.
    w_track_torque_peak: float = 0.0
    #: Kernel identification (:func:`~amortized_opt.pd_sim.extract_torque_kernels`):
    #: maximum FIR length, and the fraction of a joint's peak tap below which
    #: the tail is truncated.
    track_torque_taps: int = 24
    track_torque_tol: float = 1.0e-3
    #: Which substep of a control tick the estimate reports.  ``"max"`` -- the
    #: worst substep, which is what an effort limit limits.  ``"last"`` is
    #: exactly LTI but reports ~0 on the wrists (see ``simulate_pd``).
    track_torque_reduce: str = "max"
    #: Scoring-only multiplier on the four soft wrist pitch/yaw effort ratings
    #: (Isaac 5 N*m).  ``1.0`` = bit-identical to the hardware table.  ``2.0``
    #: allows brief reflected-Kd demand spikes in the cost hinge and (when the
    #: eval scorer is built with the same value) ``ok_torque``, without
    #: changing IdealPD's real clamp.  See :func:`scale_soft_wrist_effort`.
    soft_wrist_effort_scale: float = 1.0
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
    #: STATIC (gravity-only) TORQUE.  Mean softplus on ``|RNEA(q,0,0)|`` against
    #: ``eta_torque`` x the effort limits -- a cheap pose feasibility proxy with
    #: no AL multiplier.  Pair with ``torque_static_only`` to skip the coupled
    #: RNEA hinge and its per-segment duals during early training experiments.
    w_torque_static: float = 0.0
    #: When True, skip :meth:`FusedCost._torque` entirely (zero viol/live for
    #: all torque AL groups).  Training should also set ``w_torque`` /
    #: ``w_torque_peak`` to 0 and omit torque groups from dual updates.
    torque_static_only: bool = False
    #: When True, keep :meth:`FusedCost._torque` (e.g. level-A physics) but
    #: omit all ``torque*`` groups from dual updates.  Pair with ``w_torque_peak``
    #: at 0 for simplified-torque experiments without augmented-Lagrangian blow-up.
    torque_al_off: bool = False
    #: When True (requires ``torque_static_only``), :meth:`FusedCost._torque_static`
    #: fills the existing ``torque_*`` / ``torque_peak`` AL groups from
    #: ``|RNEA(q,0,0)|`` excess and the hard wall ``λ [g]₊`` prices ``terms["torque"]``.
    torque_static_al: bool = False
    #: Per-joint static limit tightening.  ``None`` / ``"off"`` = uniform limits.
    #: ``"decoder_measured"`` divides each joint's hard limit by the measured
    #: dyn/static RMS ratio for the gravity-only static proxy
    #: (``constants.static_torque_tighten_divisors``).  Coupled RNEA uses
    #: ``constants.torque_limit_eta_scale`` (waist at full hard).
    static_torque_tighten: str | None = None
    w_velocity: float = 1.0
    #: Price on a smooth max of ReLU hard velocity excess over segments,
    #: alongside ``w_velocity``'s mean softplus.  AL ``live`` stays the horizon
    #: **hard max**.  ZERO by default.  See :meth:`FusedCost._velocity`.
    w_velocity_peak: float = 0.0
    #: Differentiable peak shape for ``raw["velocity_peak"]``.
    velocity_peak_mode: str = "pnorm"
    velocity_peak_p: float = 12.0
    #: LSE temperature, rad/s.  Only read when ``velocity_peak_mode == "lse"``.
    velocity_peak_temp: float = 0.05
    w_joint_box: float = 10.0
    #: Price on a smooth max of per-(waypoint, joint) ReLU hard-box excess,
    #: alongside ``w_joint_box``'s waypoint mean softplus warning.
    #: ``ok_joint_limits`` is a zero-tolerance max over the plan; this half
    #: prices ``λ_peak * w_joint_box_peak * smooth_max(ReLU(q_wp - box))``.
    #: The AL group still integrates the horizon hard max.  ZERO by default.
    #: See :meth:`FusedCost._joint_box`.
    w_joint_box_peak: float = 0.0
    #: Differentiable peak shape for ``raw["joint_box_peak"]``.
    joint_box_peak_mode: str = "pnorm"
    joint_box_peak_p: float = 12.0
    #: LSE temperature, rad.  Only read when ``joint_box_peak_mode == "lse"``.
    joint_box_peak_temp: float = 0.02
    #: PER-JOINT override of the joint box for the two waist LEAN joints
    #: (``waist_pitch``, ``waist_roll``), in radians of half-width.  ``None``
    #: (default) leaves the box exactly as ``box_source`` builds it -- +-0.52 rad
    #: from the URDF -- so every existing checkpoint and the section-3 validation
    #: weights are bit-identical.  A float shrinks BOTH rows to
    #: ``+-waist_pr_box``: the softplus hinge moves to ``eta_position`` x that
    #: (the same conservatism every other row gets) and the RAW violation moves
    #: to the value itself, so the tightened rows are priced by the EXISTING
    #: ``joint_box`` augmented-Lagrangian group rather than by a new multiplier.
    #: In the direct-movement recipe this also tightens ``waist_yaw`` to a fixed
    #: ±80 deg guardrail inside the existing joint-box path so torso spin stays
    #: stable without introducing a separate yaw-specific knob.
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
    #: Compact striking posture.  This is deliberately an ARM-mass proxy, not
    #: whole-body COM: phase 2 controls only the waist and arms, so a full-body
    #: COM would be dominated by the fixed pelvis/legs and give a weak or
    #: misleading gradient.  For each active hand we combine approximate
    #: upper-arm, forearm, wrist/hand and stick centres, transform the result
    #: into the torso frame, and square only the excess above the height and
    #: horizontal-radius deadbands below.  The measured q0 sample is excluded.
    #:
    #: Raw units are metres in ``linear`` mode and m^2 in ``hinge_sq`` mode.
    #: Zero by default so old objectives are bit-identical; phase-2 training
    #: opts in after gradient calibration.
    w_arm_compact: float = 0.0
    #: ``"hinge_sq"`` preserves the original deadbanded ReLU-squared cost.
    #: ``"linear"`` continuously prices torso-frame height and radius, so
    #: every active-arm pose has a downward/inward gradient rather than a flat
    #: region below an arbitrary threshold.
    arm_compact_mode: str = "hinge_sq"
    arm_compact_height_max: float = 0.28
    arm_compact_radius_max: float = 0.28
    #: Relative prices inside the unit-weight compactness term.  Keeping these
    #: separate lets height/reach be balanced without changing its outer
    #: gradient-budget weight.
    arm_compact_height_scale: float = 1.0
    arm_compact_radius_scale: float = 1.0
    #: UPRIGHT WELL on the waist LEAN joints:
    #: ``w_waist_upright * mean_valid(q_waist_pitch^2 + q_waist_roll^2)``, in
    #: rad^2, reduced over the two rows only (NOT diluted by the other 15
    #: joints, so the weight is interpretable against a target excursion --
    #: at 0.15 rad the raw term is 0.0225).  ``waist_yaw`` is excluded by
    #: construction: the mask has a hard zero in that column, so the term is
    #: exactly yaw-invariant and carries no yaw gradient.  When
    #: ``StrikeContext.waist_pr_ref`` is set the well tracks that set-point
    #: instead of 0, with ``stopgrad(sp)`` so this is a decoder tax on
    #: leaving the announcement (the 0.15 band is the free window; this
    #: well is the price of using it).  The encoder teacher for lean is
    #: gated strike-knot distill, not this term.  Zero by default, as
    #: every new term in this package is.  Ramp it with
    #: ``weights["posture_upright"]``.
    w_waist_upright: float = 0.0
    #: TERMINAL P/R COST (economic-MPC V_f).  ``mean_pr(q_hit^2)`` at the
    #: NEARER active hit (``min`` of the two hands' ``hit_index``), always
    #: vs 0 -- even when ``waist_pr_ref`` walks the dense well.  Waist
    #: *length* is the later hit (the torso is shared); this term is the
    #: strike the committed prefix is actually for.  Same units as
    #: ``w_waist_upright`` (rad^-2).  Zero by default (bit-identical);
    #: ramped by ``weights["waist_pr_terminal"]``.
    w_waist_pr_terminal: float = 0.0
    #: Waist-aim IL penalty weight (rad^-2).  When > 0 and ``StrikeContext``
    #: carries ``waist_yaw_ref``, penalises ``(q_yaw - ref)^2`` over the horizon.
    #: Zero by default (no IL).  Ramp via ``weights["waist_aim"]``.
    #:
    #: Two reference sources share this term: the analytic
    #: ``waist_aim.yaw_reference`` ramp the pool bakes per sample, and -- under
    #: ``Phase2Config.waist_sp_from_head`` -- the encoder's LEARNED set-point
    #: broadcast over the horizon (``docs/waist_sp_head_plan.md``), through
    #: which gradient reaches the head.
    w_waist_aim: float = 0.0
    #: DEADBAND on the yaw error of the ``waist_aim`` term, rad.  Yaw deviation
    #: inside ``|q_yaw - ref| <= waist_aim_deadband`` is free; only the excess
    #: is squared.  0.0 (the default) is the original quadratic-everywhere
    #: behaviour.  The point, with the learned set-point: the anchor kills
    #: drift and wobble without micro-tugging the fine azimuth adjustments the
    #: arms may want, and travel across a segment boundary is not priced until
    #: it strays from the new set-point by more than the band.
    waist_aim_deadband: float = 0.0
    #: Restrict the ``waist_aim`` term to the YAW row.  False (the default)
    #: keeps the original behaviour of also pricing pitch/roll against 0 --
    #: redundant with ``w_waist_upright`` when both are on, which is exactly
    #: the situation the learned set-point runs in, so head runs set True.
    waist_aim_yaw_only: bool = False
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
    #: Target-directed monotonic yaw preference.  Let ``d`` be the sign from
    #: measured yaw to the analytic reference endpoint.  Every waypoint step
    #: pays ``relu(-d * (q[k] - q[k-1]))``.  This forbids the cheap pattern
    #: "move slightly the wrong way now, turn toward the target later" while
    #: still allowing acceleration and deceleration without position reversal.
    #: Requires ``StrikeContext.waist_yaw_ref``; zero keeps legacy behavior.
    w_waist_yaw_monotonic: float = 0.0
    #: Segment-wise acceleration-direction preference. Position and velocity
    #: may initially point the wrong way, but acceleration must point from the
    #: current yaw toward the nearest active hit's analytic yaw. Recomputed
    #: after every re-goal, so a following hit may reverse direction.
    w_waist_yaw_accel_direction: float = 0.0
    #: Strong boundary-only version: price a first-sample acceleration whose
    #: sign points away from the nearest active hit.  Later acceleration is
    #: unconstrained so the waist can brake normally near the target.
    w_waist_yaw_initial_accel_direction: float = 0.0
    #: Ignore an analytic endpoint displacement smaller than this many radians
    #: when selecting the desired direction.
    waist_yaw_monotonic_target_deadband: float = 1.0e-3
    #: WAIST PITCH/ROLL RATE penalty, ``w_waist_pr_vel * mean_valid(qd_pr^2)``,
    #: in (rad/s)^-2.  The exact complement of ``w_waist_yaw_vel``: that one
    #: says how fast the torso may TURN, this one how fast it may LEAN.
    #:
    #: Why this exists.  ``w_waist_upright`` prices lean POSITION and the joint
    #: box bounds it, but neither notices how the torso GETS there -- a plan can
    #: whip pitch/roll across its range and pay only for where it ends up.  The
    #: reach measurement of 2026-08-13 (arm-only IK to the randomized kit) found
    #: the reach benefit of lean saturates at the URDF limit, so lean is worth
    #: paying for as a POSTURE and worth nothing as MOMENTUM: the torso should
    #: settle into a lean and strike from it, not swing through one.  Reduced
    #: over the two pitch/roll rows alone, so like its yaw twin the weight reads
    #: directly as a target rate -- a torso leaning at a steady 1 rad/s pays
    #: ``w`` exactly.  Ramped by ``weights["waist_pr_vel"]``, which shares the
    #: waist ramp with every other waist term.
    w_waist_pr_vel: float = 0.0
    #: DECODER P/R BAND around the encoder set-point.  Priced as the **sum**
    #: of per-waypoint ReLU ``relu(|q_pr - stopgrad(sp)| - band)`` so every
    #: plan row past the band gets gradient; AL ``live`` is the horizon
    #: **max**.  ``sp`` is ``StrikeContext.waist_pr_ref``; ``stopgrad`` is
    #: load-bearing.  Idle when the ref is None.  ZERO by default;
    #: ``phase2_cost_config`` turns it on.
    w_waist_pr_band: float = 0.0
    waist_pr_band_rad: float = 0.15
    #: IDLE-ARM SAFETY CYLINDER.  **The idle arm's goal, priced over its horizon.**
    #:
    #: A hand with a scheduled hit gets a terminal region + a terminal velocity
    #: (``w_terminal_region`` / ``w_terminal_velocity``).  A hand with no hit gets
    #: the cylinder hinge instead: stick tip outside the torso-frame idle
    #: cylinder (see :mod:`~amortized_opt.idle_cylinder`) is softplus-hinged at
    #: every masked knot on that hand's horizon, and ``w_idle_vel`` asks the
    #: terminal knot to be stopped.
    #:
    #: Everything else is already handled and is *identical* for both hands --
    #: collision, self-collision, torque, velocity and joint-box all run over the
    #: idle arm's whole horizon with no special case.  The cylinder only says
    #: the tip must stay in a torso-fixed park region; the rest of the objective
    #: shapes how it gets there.
    #:
    #: A **static weight, never an augmented-Lagrangian group**: on closed-loop
    #: states an arm mid-flight from a strike cannot be inside the cylinder, and
    #: the repo has already paid once for pricing an unsatisfiable condition with
    #: a multiplier (the shortfall Lagrangian pinned at ``lam_max`` starved a
    #: 2 197-iteration run).
    w_idle_cylinder: float = 0.0
    #: Softplus temperature (metres) for the tip-outside-cylinder hinge.
    beta_idle_cyl: float = 0.01
    #:
    #: IDLE-ARM TERMINAL VELOCITY: the parked arm arrives **stopped**.
    #: ``|qd_arm|^2`` at that hand's terminal knot, rad^2/s^2.  Directly the
    #: counterpart of ``w_terminal_velocity``, which asks a striking hand for
    #: 2 m/s at its terminal knot.
    #:
    #: Both idle terms are exactly zero, with exactly zero gradient, until a
    #: cylinder is installed with :meth:`FusedCost.set_idle_cylinder`.
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
        if self.torque_physics not in ("full", "diag_grav", "diag_grav_frozen"):
            raise ValueError(f"torque_physics={self.torque_physics!r}")
        if self.torque_peak_mode not in ("pnorm", "lse"):
            raise ValueError(f"torque_peak_mode={self.torque_peak_mode!r}")
        if self.torque_peak_p < 1.0:
            raise ValueError("torque_peak_p must be >= 1")
        if self.torque_peak_temp <= 0.0:
            raise ValueError("torque_peak_temp must be > 0")
        if self.w_track_torque < 0.0:
            raise ValueError("w_track_torque must be >= 0")
        if self.w_track_torque_peak < 0.0:
            raise ValueError("w_track_torque_peak must be >= 0")
        if int(self.track_torque_taps) < 1:
            raise ValueError("track_torque_taps must be >= 1")
        if self.track_torque_reduce not in ("last", "max"):
            raise ValueError(f"track_torque_reduce={self.track_torque_reduce!r}")
        if not (self.soft_wrist_effort_scale > 0.0):
            raise ValueError("soft_wrist_effort_scale must be > 0")
        if self.collision_peak_mode not in ("pnorm", "lse"):
            raise ValueError(f"collision_peak_mode={self.collision_peak_mode!r}")
        if self.collision_peak_p < 1.0:
            raise ValueError("collision_peak_p must be >= 1")
        if self.collision_peak_temp <= 0.0:
            raise ValueError("collision_peak_temp must be > 0")
        if self.w_collision_peak < 0.0:
            raise ValueError("w_collision_peak must be >= 0")
        if self.w_velocity_peak < 0.0:
            raise ValueError("w_velocity_peak must be >= 0")
        if self.velocity_peak_mode not in ("pnorm", "lse"):
            raise ValueError(f"velocity_peak_mode={self.velocity_peak_mode!r}")
        if self.velocity_peak_p < 1.0:
            raise ValueError("velocity_peak_p must be >= 1")
        if self.velocity_peak_temp <= 0.0:
            raise ValueError("velocity_peak_temp must be > 0")
        if self.joint_box_peak_mode not in ("pnorm", "lse"):
            raise ValueError(f"joint_box_peak_mode={self.joint_box_peak_mode!r}")
        if self.joint_box_peak_p < 1.0:
            raise ValueError("joint_box_peak_p must be >= 1")
        if self.joint_box_peak_temp <= 0.0:
            raise ValueError("joint_box_peak_temp must be > 0")
        if self.w_joint_box_peak < 0.0:
            raise ValueError("w_joint_box_peak must be >= 0")
        if self.w_stick_stick < 0.0:
            raise ValueError("w_stick_stick must be >= 0")
        if self.w_terminal_aim < 0.0:
            raise ValueError("w_terminal_aim must be >= 0")
        if self.terminal_aim_deadband_m < 0.0:
            raise ValueError("terminal_aim_deadband_m must be >= 0")
        if self.w_tip_crossover < 0.0:
            raise ValueError("w_tip_crossover must be >= 0")
        if self.tip_crossover_margin < 0.0:
            raise ValueError("tip_crossover_margin must be >= 0")
        if self.beta_tip_crossover <= 0.0:
            raise ValueError("beta_tip_crossover must be > 0")
        if self.reduction not in ("mean", "sum"):
            raise ValueError(f"reduction={self.reduction!r}")
        if self.n_sub < 1:
            raise ValueError("n_sub must be >= 1")
        if self.waist_pr_box is not None and self.waist_pr_box <= 0.0:
            object.__setattr__(self, "waist_pr_box", None)
        if self.w_effort < 0.0:
            raise ValueError("w_effort must be >= 0")
        if self.w_torque_static < 0.0:
            raise ValueError("w_torque_static must be >= 0")
        if self.torque_static_al and not self.torque_static_only:
            raise ValueError("torque_static_al requires torque_static_only")
        if self.torque_static_al and self.w_torque_static <= 0.0:
            raise ValueError("torque_static_al requires w_torque_static > 0")
        if self.static_torque_tighten not in (None, "off", "none", ""):
            static_torque_tighten_divisors(
                G1_UPPER_BODY_JOINTS, mode=self.static_torque_tighten,
            )
        if self.effort_fade_wp < 0:
            raise ValueError("effort_fade_wp must be >= 0")
        if self.w_waist_upright < 0.0:
            raise ValueError("w_waist_upright must be >= 0")
        if self.w_waist_pr_terminal < 0.0:
            raise ValueError("w_waist_pr_terminal must be >= 0")
        if self.w_waist_aim < 0.0:
            raise ValueError("w_waist_aim must be >= 0")
        if self.w_waist_yaw_vel < 0.0:
            raise ValueError("w_waist_yaw_vel must be >= 0")
        if self.w_waist_yaw_monotonic < 0.0:
            raise ValueError("w_waist_yaw_monotonic must be >= 0")
        if self.w_waist_yaw_accel_direction < 0.0:
            raise ValueError("w_waist_yaw_accel_direction must be >= 0")
        if self.w_waist_yaw_initial_accel_direction < 0.0:
            raise ValueError("w_waist_yaw_initial_accel_direction must be >= 0")
        if self.waist_yaw_monotonic_target_deadband < 0.0:
            raise ValueError("waist_yaw_monotonic_target_deadband must be >= 0")
        if self.w_waist_pr_vel < 0.0:
            raise ValueError("w_waist_pr_vel must be >= 0")
        if self.w_waist_pr_band < 0.0:
            raise ValueError("w_waist_pr_band must be >= 0")
        if self.waist_pr_band_rad < 0.0:
            raise ValueError("waist_pr_band_rad must be >= 0")
        if self.w_idle_cylinder < 0.0:
            raise ValueError("w_idle_cylinder must be >= 0")
        if self.beta_idle_cyl <= 0.0:
            raise ValueError("beta_idle_cyl must be > 0")
        if self.w_idle_vel < 0.0:
            raise ValueError("w_idle_vel must be >= 0")
        if self.w_arm_compact < 0.0:
            raise ValueError("w_arm_compact must be >= 0")
        if self.arm_compact_mode not in ("hinge_sq", "linear"):
            raise ValueError(f"arm_compact_mode={self.arm_compact_mode!r}")
        if self.arm_compact_height_max < 0.0:
            raise ValueError("arm_compact_height_max must be >= 0")
        if self.arm_compact_radius_max <= 0.0:
            raise ValueError("arm_compact_radius_max must be > 0")
        if self.arm_compact_height_scale < 0.0:
            raise ValueError("arm_compact_height_scale must be >= 0")
        if self.arm_compact_radius_scale < 0.0:
            raise ValueError("arm_compact_radius_scale must be >= 0")
        if self.terminal_axial_lo > self.terminal_axial_hi:
            raise ValueError("terminal_axial_lo must be <= terminal_axial_hi")
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
    #: ``(B, D)`` per-sample drum head RADII, rows aligned with
    #: ``drum_head_centers``.  Set it whenever
    #: :attr:`~amortized_opt.rhythm.KitRandomization.per_drum_radius` is on, for
    #: exactly the reason the centres are per-sample: without it the drum token
    #: describes a resized head while the collision cylinder keeps the nominal
    #: one, so the objective and the scene disagree.  Radius is
    #: frame-independent, so this is ``ContextInputs.drum_radius`` verbatim --
    #: one array, and the two cannot drift.
    drum_radii: torch.Tensor | None = None
    #: ``(B, 2, P, 3)`` per-sample stick contact points in the WRIST frame, hand
    #: major (0 = left), ``P = 1 tip + n_shaft + 1 wrist`` in ``c_points`` order.
    #: Set this whenever the STICK is domain-randomized, for exactly the reason
    #: ``drum_head_centers`` exists: without it the collision and terminal terms
    #: price a nominal stick while the data was generated with a randomized one,
    #: so the objective and the geometry disagree.  Build it with
    #: :func:`stick_points_from_specs` so the two cannot drift apart.
    stick_points_local: torch.Tensor | None = None
    #: ``(B, 2, 3)`` per-sample stick AXIS (handle->tip unit) in the wrist frame.
    #: The terminal term builds its contact candidates as ``mount + sigma *
    #: axis``, so a tilted stick needs this as well as the collision points.
    stick_axis_local: torch.Tensor | None = None
    #: ``(B, 2)`` per-sample arc coordinate of the TIP.  Length randomization is
    #: handle-fixed, so a longer stick moves the tip out along the axis and this
    #: is where the bead actually is; the shaft candidates keep their nominal
    #: sigmas, which stay on the shaft by construction.
    stick_tip_sigma: torch.Tensor | None = None
    #: ``(B, H)`` waist-yaw reference trajectory at knot resolution, computed by
    #: :mod:`~amortized_opt.waist_aim`.  ``None`` means the IL term is disabled.
    #:
    #: KNOT CONVENTION (the one place it is defined): ``waist_yaw_ref[:, k]``
    #: is the reference for **knot k+1** -- the same rows-1..H layout as the
    #: decoder's waypoint tokens (``WarmStart.init_traj``, ``MPCRoller``'s
    #: ``init``), which never include the measured ``q0`` (knot 0).  Every
    #: producer (the fresh pool's seed yaw column, ``MPCRoller``'s harvested
    #: reference, the learned set-point broadcast) and the one consumer
    #: (:meth:`FusedCost._posture`'s ``waist_aim`` term) must follow it; a
    #: trajectory equal to its own seed then prices the term at exactly zero.
    waist_yaw_ref: torch.Tensor | None = None
    #: ``(B, H, 2)`` pitch/roll reference at knot resolution, ``[pitch, roll]``.
    #: Same knot convention as ``waist_yaw_ref`` (row ``k`` is knot ``k+1``).
    #: ``None`` keeps the upright well pricing ``q_pr²`` against 0 (snapshot
    #: / 81-grid).  When set it prices ``‖q_pr − stopgrad(ref)‖²``: the
    #: encoder P/R head used to fill this, now a leaky EMA of the clipped
    #: P/R command harvested by the roller.  Sample 0 is then masked out of
    #: the reduction, matching ``waist_aim``.
    waist_pr_ref: torch.Tensor | None = None

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


def stick_cost_from_context(
    stick_axis: torch.Tensor,
    stick_tip: torch.Tensor,
    stick_length: torch.Tensor,
    stick_valid: torch.Tensor,
    n_shaft: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    """The cost's three stick tensors, built from the CONTEXT columns.

    The stick counterpart of :func:`drum_heads_from_context`, and it exists for
    the same reason: "the stick the cost prices" and "the stick the model was
    told about" must be one array, or they drift.  Takes the
    ``(B, 2, ...)`` blocks a pool stores -- axis, tip offset, length, valid --
    and returns ``(points, axis, tip_sigma)`` shaped for
    :class:`StrikeContext`.

    Returns ``(None, None, None)`` when no row carries a stick, so a caller can
    pass the result straight through and get the nominal, baked-in geometry.

    The shaft samples are rebuilt here rather than stored: they are a
    deterministic ``linspace`` along the axis between the (fixed) butt and the
    tip, so storing them would be four more columns that can only disagree.
    """
    if stick_valid is None or not bool(stick_valid.any()):
        return None, None, None
    dtype, device = stick_axis.dtype, stick_axis.device
    B = stick_axis.shape[0]
    m = stick_valid.unsqueeze(-1).to(dtype)                      # (B, 2, 1)
    butt = -STICK_LENGTH / 2.0
    mounts = torch.stack([
        torch.as_tensor(STICK_MOUNT_POS_LEFT, device=device, dtype=dtype),
        torch.as_tensor(STICK_MOUNT_POS_RIGHT, device=device, dtype=dtype),
    ]).view(1, 2, 1, 3)                                          # mount is NOT randomized

    # Rows without a stick fall back to the NOMINAL geometry -- axis, tip
    # sigma AND the tip point are blended toward the shipped stick BEFORE the
    # point cloud is built, so an invalid row's points come out exactly the
    # nominal stick's rather than a zero-length stub at the wrist.  A single
    # valid row must not poison the rest of the batch.
    nom_axis = torch.stack([
        torch.as_tensor(stick_axis_local("left"), device=device, dtype=dtype),
        torch.as_tensor(stick_axis_local("right"), device=device, dtype=dtype),
    ]).view(1, 2, 3)
    axis = stick_axis * m + nom_axis * (1.0 - m)                 # (B, 2, 3)
    tip_sigma = (stick_length - STICK_LENGTH / 2.0)              # (B, 2)
    tip_sigma = tip_sigma * m.squeeze(-1) + (STICK_LENGTH / 2.0) * (1.0 - m.squeeze(-1))
    nom_tip = mounts.squeeze(2) + (STICK_LENGTH / 2.0) * nom_axis  # (1, 2, 3)
    tip_pt = stick_tip * m + nom_tip * (1.0 - m)                 # (B, 2, 3)

    n = max(int(n_shaft), 2)
    u = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).view(1, 1, n)
    end = (tip_sigma - SHAFT_TIP_CLEAR).unsqueeze(-1)            # (B, 2, 1)
    sig = butt + u * (end - butt)                                # (B, 2, n)
    shaft = mounts + sig.unsqueeze(-1) * axis.unsqueeze(2)       # (B, 2, n, 3)

    tip = tip_pt.unsqueeze(2)                                    # (B, 2, 1, 3)
    wrist = torch.zeros(B, 2, 1, 3, device=device, dtype=dtype)
    pts = torch.cat([tip, shaft, wrist], dim=2)                  # (B, 2, P, 3)
    return pts, axis, tip_sigma


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
    balance_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Detached aux-head labels (vel/acc/torque/fk_*), when
    #: :meth:`FusedCost.forward` was asked for ``aux_mode``.  Built from the
    #: same spline / knot RNEA / FK plant the cost already paid for, so the
    #: train step does not rebuild them in a second ``aux_targets`` pass.
    aux_labels: dict[str, torch.Tensor] | None = None

    def detach_numpy(self) -> dict[str, np.ndarray]:
        out = {"total": self.total.detach().cpu().numpy()}
        for grp in (
            "terms", "raw", "violations", "diagnostics", "counts",
            "balance_terms",
        ):
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
        #: Idle-arm safety cylinder ``(8,)`` = center(3)+axis(3)+radius+half_height,
        #: installed by :meth:`set_idle_cylinder`.  ``None`` (the default) makes
        #: ``w_idle_cylinder`` / ``w_idle_vel`` exact no-ops.
        self.c_idle_cyl: torch.Tensor | None = None
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

        # Collision endpoints, wrist-local: [tip, shaft*S, wrist].  Row 0 is
        # the bead and row 1 is the first shaft sample, which
        # ``stick_shaft_offsets`` places at the butt, so [row 1 -> row 0] is
        # the whole stick capsule.  The wrist is a separate sphere at the
        # origin.  Shaft *samples* are no longer scored against drums -- they
        # exist so the same cloud still yields the capsule endpoints under
        # stick-length randomization.
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
        # Stick capsule swallows the bead as well as the shaft (the bead is
        # the fatter of the two); the wrist keeps its own ball.
        self.c_stick_r = float(max(TIP_RADIUS, STICK_RADIUS))
        self.c_wrist_r = float(WRIST_RADIUS)

        a, b, r = TORSO_CAPSULE
        self.c_torso_a, self.c_torso_b, self.c_torso_r = t(a), t(b), float(r)
        hc, hr = HEAD_SPHERE
        self.c_head_c, self.c_head_r = t(hc), float(hr)
        # The lower body is welded at the upstream G1 standing pose while this
        # optimizer commands the waist/arms only.  Keep its collision proxies
        # in the pelvis frame; they intentionally do not follow ``torso``.
        self.c_lower_a = t([x[1] for x in FIXED_LOWER_BODY_CAPSULES])
        self.c_lower_b = t([x[2] for x in FIXED_LOWER_BODY_CAPSULES])
        self.c_lower_r = t([x[3] for x in FIXED_LOWER_BODY_CAPSULES])

        # Stick mount, per hand: contact(sigma) = p + R @ (mount_p + sigma*axis)
        self.c_mount_p, self.c_mount_axis = {}, {}
        for hand in _HANDS:
            Rm, pm = stick_mount_transform(hand)
            self.c_mount_p[hand] = t(pm)
            self.c_mount_axis[hand] = t(stick_axis_local(hand))
        # Approximate mass split used by the compact active-arm proxy:
        # shoulder-group, elbow/forearm, wrist+hand, drumstick.
        self.c_arm_compact_masses = t((2.095, 0.600, 0.824, 0.100))
        self.c_sigmas = t(np.asarray(SHAFT_SIGMAS, float))  # (Sc,)
        self.n_sigmas = int(self.c_sigmas.shape[0])
        tip_first = np.zeros(self.n_sigmas)
        tip_first[int(np.argmax(np.asarray(SHAFT_SIGMAS)))] = 1.0
        self.c_sigma_is_tip = t(tip_first)  # which candidate is the bead

        # Limits.  Soft-wrist scale is scoring-only (cost / ok_torque); the
        # plant clamp still uses the unscaled Isaac table.
        self.tau_hard = t(scale_soft_wrist_effort(
            isaac_effort_limits(self.joint_names), self.joint_names,
            scale=cfg.soft_wrist_effort_scale,
        ))
        mode = cfg.static_torque_tighten
        if mode in (None, "off", "none", ""):
            tighten = np.ones(self.n_dof, dtype=float)
        else:
            tighten = static_torque_tighten_divisors(
                self.joint_names, mode=mode,
            )
        self.c_static_torque_tighten = t(tighten)
        self.c_torque_eta = t(
            torque_limit_eta_scale(self.joint_names, eta_torque=cfg.eta_torque)
        )
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

        # Actuator-class column indices for the split limit duals.  Built as
        # long tensors so ``index_select`` in the compiled cost is a gather.
        self.c_seg: list[tuple[str, torch.Tensor]] = []
        for name, joints in ACTUATOR_SEGMENTS:
            idx = [self.joint_names.index(j) for j in joints]
            self.c_seg.append(
                (name, torch.as_tensor(idx, device=dev, dtype=torch.long))
            )

        # Posture: twist rows + reference pose.
        self.c_twist = t(
            axial_twist_directions(self.fk, twist_axis_floor=cfg.twist_axis_floor,
                                   joint_names=self.joint_names)
        )  # (2, n)
        self.q_ready = t(ready_joint_pos(self.joint_names))
        from .pd_sim import mass_matrix

        M0 = mass_matrix(self.rnea, self.q_ready.view(1, -1))
        self._m_diag_ready = torch.diagonal(M0, dim1=-2, dim2=-1).detach().squeeze(0)

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

        # TRACKING-TORQUE FIR bank, ``(n, L)``.  Built here (so it follows the
        # object across ``.to()``) but ONLY when a tracking weight is on: it
        # costs one RNEA mass matrix and one ``n``-sample PD rollout, and a run
        # that does not price the term should not pay for either.
        self.c_tau_kernel: torch.Tensor | None = None
        if self.track_torque_on:
            self.c_tau_kernel = self.build_torque_kernels()

    # ------------------------------------------------- tracking-torque FIR ---
    @property
    def track_torque_on(self) -> bool:
        """Is the tracking-torque estimator priced at all?

        A python bool read at TRACE time, so the whole block below is compiled
        away (and is bit-for-bit absent) whenever both weights are zero.
        """
        return bool(self.cfg.w_track_torque > 0.0 or self.cfg.w_track_torque_peak > 0.0)

    def build_torque_kernels(self) -> torch.Tensor:
        """Identify the per-joint ``command -> demanded torque`` FIR bank.

        Delegates to :func:`~amortized_opt.pd_sim.nominal_torque_kernels` --
        THE one identification, shared with the phase-1/2 auxiliary torque
        label -- at the NOMINAL runtime gains (the plant's ``kp_nominal`` /
        ``kd_nominal``, i.e. what ``BatchedPDSim`` uses before gain
        randomization) and a frozen diagonal mobility taken at the drummer's
        ready pose, ``diag(inv(M(q_ready) + armature))``.  Identified in
        float64 whatever the cost's own dtype -- it is 24 taps of a stiff
        recursion, done once.
        """
        from .pd_sim import nominal_torque_kernels

        cfg = self.cfg
        return nominal_torque_kernels(
            self.rnea, self.q_ready,
            joint_names=self.joint_names, dt=self.dt, substeps=PD_SUBSTEPS,
            taps=int(cfg.track_torque_taps), tol=float(cfg.track_torque_tol),
            torque_reduce=cfg.track_torque_reduce,
            device=self.device, dtype=self.dtype,
        )

    def to(self, device, dtype: torch.dtype | None = None) -> "FusedCost":
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.fk = self.fk.to(self.device, self.dtype)
        self.rnea = self.rnea.to(self.device, self.dtype)
        self.spline = self.spline.to(self.device, self.dtype)
        self._build_constants(self.scene.under_depth, self.scene.under_clearance)
        if self.c_idle_cyl is not None:
            self.c_idle_cyl = self.c_idle_cyl.to(device=self.device, dtype=self.dtype)
        return self

    # ----------------------------------------------------------- idle arm ---
    def set_idle_cylinder(self, cyl) -> "FusedCost":
        """Install the torso-frame cylinder the idle-arm tip hinge prices against.

        ``cyl`` is an :class:`~amortized_opt.idle_cylinder.IdleCylinder`, an
        :class:`~amortized_opt.idle_cylinder.IdleCylinderLibrary` (uses its
        ``.cylinder``), or an ``(8,)`` packed vector
        (center + axis + radius + half_height).  Pass ``None`` to uninstall
        (the terms go back to exact zeros).
        """
        if cyl is None:
            self.c_idle_cyl = None
            return self
        if hasattr(cyl, "cylinder"):
            cyl = cyl.cylinder
        if hasattr(cyl, "pack"):
            v = np.asarray(cyl.pack(), float).reshape(8)
        else:
            v = np.asarray(cyl, float).reshape(8)
        self.c_idle_cyl = torch.as_tensor(v, device=self.device, dtype=self.dtype)
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
        # ``tau_hard`` is baked at construction; refresh when the soft-wrist
        # scoring scale changes so ``with_config`` stays honest.
        if "soft_wrist_effort_scale" in kw:
            t = lambda a: torch.as_tensor(
                np.asarray(a, float), device=out.tau_hard.device, dtype=out.tau_hard.dtype
            )
            out.tau_hard = t(scale_soft_wrist_effort(
                isaac_effort_limits(out.joint_names), out.joint_names,
                scale=out.cfg.soft_wrist_effort_scale,
            ))
        if "static_torque_tighten" in kw:
            mode = out.cfg.static_torque_tighten
            if mode in (None, "off", "none", ""):
                div = np.ones(out.n_dof, dtype=float)
            else:
                div = static_torque_tighten_divisors(out.joint_names, mode=mode)
            out.c_static_torque_tighten = torch.as_tensor(
                div, device=out.tau_hard.device, dtype=out.tau_hard.dtype,
            )
        if "static_torque_tighten" in kw or "eta_torque" in kw:
            out.c_torque_eta = torch.as_tensor(
                torque_limit_eta_scale(
                    out.joint_names, eta_torque=out.cfg.eta_torque,
                ),
                device=out.tau_hard.device, dtype=out.tau_hard.dtype,
            )
        return out

    # -------------------------------------------------------- reductions --
    def _reduce(self, x: torch.Tensor, mask: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor]:
        """(value, count) reduced over ``dims`` with ``mask`` applied."""
        num = (x * mask).sum(dim=dims)
        cnt = mask.sum(dim=dims)
        if self.cfg.reduction == "sum":
            return num, cnt
        return num / cnt.clamp_min(1.0), cnt

    def _emit_limit_segments(
        self, family: str, over: torch.Tensor, hinge: torch.Tensor,
        mask: torch.Tensor, cnt: torch.Tensor, raw, viol, live,
        *, write_raw: bool = True,
    ) -> None:
        """Per-actuator-class hinge share + hierarchical dual residual.

        Hinge share is ``sum(h in G) / n_valid`` so the seven shares sum to the
        global mean (diagnostics / optional; they do not enter ``terms`` × λ).
        Dual residual is ``max_{k,j in G} [g]₊`` -- a one-hot joint is the
        full peak inside its class, not peak/|G|.  Warning uses the global
        mean at ``cfg.w_*`` with no λ.  Hard is ``w[seg] * live[seg]`` in
        :meth:`forward`.
        """
        denom = cnt.clamp_min(1.0)
        for name, idx in self.c_seg:
            o = over.index_select(-1, idx)
            h = hinge.index_select(-1, idx)
            m = mask.index_select(-1, idx)
            hier = (o * m).amax(dim=(1, 2))
            key = f"{family}_{name}"
            viol[key] = hier.detach()
            live[key] = hier
            if write_raw:
                raw[key] = (h * m).sum(dim=(1, 2)) / denom

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
        aux_mode: str | None = None,
        aux_tau_kernel: torch.Tensor | None = None,
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
        # Live (undettached) copy of the same ReLU residuals the dual reads.
        # Public ``violations`` stay detached; consumers are the PHR quadratic
        # ``(pen / 2) [g]₊²`` and the hard wall ``λ [g]₊``.
        live: dict[str, torch.Tensor] = {}
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
            collision_peak=1.0, joint_box_peak=1.0, velocity_peak=1.0,
            posture_upright=1.0, waist_box=1.0, effort=1.0, waist_aim=1.0,
            posture_idle=1.0, waist_yaw_vel=1.0, waist_pr_vel=1.0,
            tip_crossover=1.0, waist_pr_band=1.0, waist_pr_terminal=1.0,
        )
        for fam in LIMIT_FAMILIES:
            for seg in ACTUATOR_SEGMENT_NAMES:
                w[limit_group_name(fam, seg)] = w[fam]
        if weights:
            w.update({k: v for k, v in weights.items() if k in w})
            # Family key without per-segment keys still broadcasts onto every
            # ``torque_*`` / ``velocity_*`` / ``joint_box_*`` λ (the hard
            # ReLU wall).  Keys that WERE passed win, so a live AL dict of
            # ``torque_waist`` etc. is not overwritten by a leftover family
            # entry.
            for fam in LIMIT_FAMILIES:
                for seg in ACTUATOR_SEGMENT_NAMES:
                    key = limit_group_name(fam, seg)
                    if key not in weights:
                        w[key] = w[fam]

        # ---- per-JOINT price on the three limit hinges -----------------------
        # ``weights["limit_alpha"]`` is a ``(D,)`` vector, deliberately NOT a
        # member of ``w`` above: ``w``'s entries are scalar prices on an already
        # reduced term, this one multiplies the hinge PER ELEMENT *before* its
        # reduction, so a curriculum can hold one joint at full strength while
        # the rest ramp (see ``WeightSchedule.full_limit_joints``).  It touches
        # exactly the mean halves of ``torque`` / ``velocity`` / ``joint_box``;
        # the max-shaped ``torque_peak`` keeps its own group schedule, and every
        # RAW violation is computed from the unscaled excess, so the augmented
        # Lagrangian cannot see this at all.
        #
        # ``None`` is all-ones and is bit-identical to the pre-exemption form
        # (the multiply is skipped, not performed against ones).  The branch is
        # on the OBJECT, resolved at trace time exactly like
        # ``ctx.waist_yaw_ref`` below, so nothing here is data-dependent.
        lim_a = weights.get("limit_alpha") if weights else None
        if lim_a is not None:
            lim_a = torch.as_tensor(
                lim_a, device=ss.q.device, dtype=ss.q.dtype
            ).view(1, 1, D)

        self._collision(frames, B, S, ctx, m_hand, raw, viol, live, diag, counts, sp)
        tau = None
        if cfg.torque_static_only:
            self._torque_off(
                B, ss, raw, viol, live, diag, counts,
                zero_viol=not cfg.torque_static_al,
            )
        else:
            torque_full_blend = 1.0
            if cfg.torque_physics != "full":
                if weights is not None and "torque_full_blend" in weights:
                    tb = weights["torque_full_blend"]
                    torque_full_blend = float(
                        tb.item() if isinstance(tb, torch.Tensor) else tb
                    )
                else:
                    torque_full_blend = 0.0
            tau = self._torque(
                sp, ss, v0, B, S, D, ctx, raw, viol, live, diag, counts, lim_a,
                torque_full_blend=torque_full_blend,
            )
        if cfg.w_torque_static > 0.0 or (
            cfg.torque_static_only and cfg.torque_static_al
        ):
            self._torque_static(
                sp, ss, v0, B, S, D, raw, viol, live, diag, counts, lim_a,
            )
        # TRACKING TORQUE.  Entirely behind a trace-time python branch: with
        # both weights at 0 (the default) not one tensor of it is created, and
        # ``viol["torque"]`` keeps its RNEA meaning.  The three diagnostics are
        # emitted either way so ``DIAG_KEYS`` has a fixed width.
        tau_track = None
        if self.track_torque_on:
            tau_track = self._track_torque(
                sp, v0, B, D, raw, viol, live, diag, counts, lim_a,
            )
        else:
            z = ss.q.new_zeros(B)
            diag["track_torque_excess_mean"] = z
            diag["track_torque_excess_max"] = z
            diag["track_torque_max"] = z
        self._velocity(sp, raw, viol, live, diag, counts, lim_a)
        wp_mask = elem_mask if elem_mask is not None else sp.seg_mask
        self._joint_box(
            waypoints, wp_mask, raw, viol, live, diag, counts, w["waist_box"], lim_a,
        )
        self._smoothness(sp, raw, diag, counts)
        self._posture(sp, frames, ss, B, S, ctx, raw, diag, counts)
        self._waist_pr_band(waypoints, wp_mask, B, ctx, raw, viol, live)
        self._terminal(sp, ctx, raw, viol, live, diag, counts)

        # Warning (no λ): softplus / curriculum at ``cfg.w_*``.  Hard wall
        # ``λ [g]₊`` is added below over ``TERM_FOR_GROUP``.  Default λ=1 when
        # the caller did not pass AL, matching today's missing-key default,
        # but it multiplies ReLU live, not the softplus.
        #
        # Terminal sub-hinges keep their curriculum ramps; ``w["terminal"]``
        # (λ) does NOT scale the warning.  The in-plane aim pull is a plain
        # objective.  Exactly zero at ``cfg.w_terminal_aim = 0``.
        terms["terminal"] = (
            (cfg.w_terminal_region * w["terminal_region"]) * raw["terminal_region"]
            + (cfg.w_terminal_velocity * w["terminal_velocity"]) * raw["terminal_velocity"]
            + (cfg.w_terminal_elev * w["terminal_elev"]) * raw["terminal_elev"]
        )
        terms["terminal_aim"] = cfg.w_terminal_aim * raw["terminal_aim"]
        # Mean collision is a margin-softplus warning at ``cfg.w_collision_*``;
        # ``w["collision"]`` (λ) does not scale it.  The peak is already a
        # ReLU reduction, priced as ``λ_peak * w_collision_peak * raw_peak``
        # (hard).  At ``cfg.w_collision_peak = 0`` that product is exactly
        # zero.  Cross-hand mean joins the warning block; at
        # ``cfg.w_stick_stick = 0`` it is an exact zero.
        terms["collision"] = (
            cfg.w_collision_drum * raw["collision_drum"]
            + cfg.w_collision_under * raw["collision_under"]
            + cfg.w_collision_self * raw["collision_self"]
            + cfg.w_stick_stick * raw["collision_stick"]
        ) + w["collision_peak"] * cfg.w_collision_peak * raw["collision_peak"]
        # Limit-family warning is the global mean softplus at ``cfg.w_*``.
        # Per-segment λ hits ``live[seg]`` only (loop below), never the
        # diagnostic shares.  Peak halves are ``λ_peak * w_peak * smooth_max(ReLU)``;
        # skipped in the linear live loop so they are not double-counted.
        terms["torque"] = (
            w["torque_peak"] * cfg.w_torque_peak * raw["torque_peak"]
            + cfg.w_torque * raw["torque"]
        )
        terms["velocity"] = (
            w["velocity_peak"] * cfg.w_velocity_peak * raw["velocity_peak"]
            + cfg.w_velocity * raw["velocity"]
        )
        terms["joint_box"] = (
            w["joint_box_peak"] * cfg.w_joint_box_peak * raw["joint_box_peak"]
            + cfg.w_joint_box * raw["joint_box"]
        )
        # Peaks are smooth-max ReLU (dense, max-shaped); AL max is separate.
        # Tracking hinge is a warning (``cfg.w_track_* × raw``), still under
        # ``limit_alpha``.  Not multiplied by λ.  The torque group's ``live``
        # is already whatever ``_track_torque`` wrote.
        if self.track_torque_on:
            terms["torque"] = terms["torque"] + (
                cfg.w_track_torque * raw["track_torque"]
                + cfg.w_track_torque_peak * raw["track_torque_peak"]
            )
        # Effort is an OBJECTIVE, not a constraint: no augmented-Lagrangian
        # multiplier, only its config weight and a curriculum ramp.  Its own
        # term key rather than a share of ``torque`` precisely because the
        # torque key carries ``lam_torque``, and multiplying an energy proxy by
        # a limit-violation multiplier is the mistake section 7b paid for.
        # Exactly zero at the default ``cfg.w_effort = 0``.
        terms["effort"] = (cfg.w_effort * w["effort"]) * raw["effort"]
        # Gravity-only static torque: plain objective, no λ.
        terms["torque_static"] = (
            cfg.w_torque_static * raw["torque_static"]
            if cfg.w_torque_static > 0.0
            else ss.q.new_zeros(B)
        )
        terms["smooth"] = w["smooth"] * (
            cfg.w_smooth_vel * raw["smooth_vel"]
            + cfg.w_smooth_acc * raw["smooth_acc"]
            + cfg.w_smooth_jerk * raw["smooth_jerk"]
        )
        terms["posture"] = w["posture"] * (
            (cfg.w_posture_mid * w["posture_mid"]) * raw["posture_mid"]
            + (cfg.w_elbow_down * w["posture_elbow"]) * raw["posture_elbow"]
            + cfg.w_axial_twist * raw["posture_twist"]
            + cfg.w_arm_compact * raw["posture_arm_compact"]
            # The IDLE ARM pair, both exactly zero without an installed cylinder.
            # They ride the ``posture`` group multiplier (an objective block, no
            # augmented-Lagrangian multiplier of their own) and share one
            # curriculum ramp, ``weights["posture_idle"]``.
            + (cfg.w_idle_cylinder * w["posture_idle"]) * raw["posture_idle_cylinder"]
            + (cfg.w_idle_vel * w["posture_idle"]) * raw["posture_idle_vel"]
        )
        # Waist-upright POSITION: own term (grad-balance ``waist`` group with the
        # rate terms).  At ``cfg.w_waist_upright = 0`` this is exactly zero.
        terms["waist_upright"] = (
            (cfg.w_waist_upright * w["posture_upright"]) * raw["posture_upright"]
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
        terms["waist_yaw_monotonic"] = (
            cfg.w_waist_yaw_monotonic * w["waist_yaw_vel"]
        ) * raw["waist_yaw_monotonic"]
        terms["waist_yaw_accel_direction"] = (
            cfg.w_waist_yaw_accel_direction * w["waist_yaw_vel"]
        ) * raw["waist_yaw_accel_direction"]
        terms["waist_yaw_initial_accel_direction"] = (
            cfg.w_waist_yaw_initial_accel_direction * w["waist_yaw_vel"]
        ) * raw["waist_yaw_initial_accel_direction"]
        # Waist pitch/roll RATE, the lean half of the same idea: reach saturates
        # at the URDF limit, so a lean is worth paying for as a posture and
        # worth nothing as momentum.  Exactly zero at ``w_waist_pr_vel = 0``.
        terms["waist_pr_vel"] = (
            cfg.w_waist_pr_vel * w["waist_pr_vel"]
        ) * raw["waist_pr_vel"]
        # Economic-MPC V_f on P/R at the nearer active hit.  Own term so it
        # shows up as ``c_waist_pr_terminal`` rather than hiding inside
        # posture.  Exact zero at ``w_waist_pr_terminal = 0``.
        terms["waist_pr_terminal"] = (
            cfg.w_waist_pr_terminal * w["waist_pr_terminal"]
        ) * raw["waist_pr_terminal"]
        # Decoder P/R vs stopgrad(encoder SP).  Warning is sum of per-waypoint
        # ReLU at ``cfg.w_waist_pr_band`` (no λ).  Hard is ``λ * live`` (max)
        # below.  Exact zero at the default ``cfg.w_waist_pr_band = 0``.
        terms["waist_pr_band"] = cfg.w_waist_pr_band * raw["waist_pr_band"]
        # Tip lateral ordering: plain objective (no AL).  Exactly zero at
        # ``cfg.w_tip_crossover = 0``.
        terms["tip_crossover"] = (
            cfg.w_tip_crossover * w["tip_crossover"]
        ) * raw["tip_crossover"]
        # Hard wall: λ × ReLU live.  Peak groups already carry λ in the
        # ``w_peak * λ * smooth_max(ReLU)`` product above.  Runs even when
        # ``weights is None`` so default λ=1 still applies.
        for g, term in TERM_FOR_GROUP.items():
            if g in _PEAK_AL_GROUPS:
                continue
            v = live.get(g)
            if v is None:
                continue
            terms[term] = terms[term] + w[g] * v
        # PHR quadratic.  Keys ``pen_<group>`` are absent when the coefficient
        # is zero, so a caller that does not pass them is bit-identical (not
        # "plus a zero").  Segment groups square ``live`` (the ReLU residual).
        # Peak groups square ``raw`` (the smooth max already priced above);
        # ``live`` for those groups stays the detached hard max the dual reads.
        if weights:
            for g, term in TERM_FOR_GROUP.items():
                p = weights.get(f"pen_{g}")
                if p is None:
                    continue
                v = raw.get(g) if g in _PEAK_AL_GROUPS else live.get(g)
                if v is None:
                    continue
                terms[term] = terms[term] + 0.5 * p * v.clamp_min(0.0).square()
        total = sum(terms.values())
        # Balance-only reference: the exact warning + AL wall attributable to
        # yaw/pitch/roll inside the global joint-box term.  It is NOT added to
        # ``total`` again; one-sided PCGrad uses only its gradient direction.
        box_waist_key = limit_group_name("joint_box", "waist")
        box_waist = (
            cfg.w_joint_box * raw[box_waist_key]
            + w[box_waist_key] * live[box_waist_key]
        )
        if weights:
            p = weights.get(f"pen_{box_waist_key}")
            if p is not None:
                box_waist = (
                    box_waist
                    + 0.5 * p * live[box_waist_key].clamp_min(0.0).square()
                )
        return CostTerms(
            total=total, terms=terms, raw=raw, violations=viol,
            diagnostics=diag, counts=counts,
            balance_terms={box_waist_key: box_waist},
            aux_labels=(
                self._aux_labels_from_plant(
                    sp, ss, tau, v0,
                    torque_mode=aux_mode,
                    tau_kernel=aux_tau_kernel,
                    tau_track=tau_track,
                    B=B, D=D,
                )
                if aux_mode is not None else None
            ),
        )

    __call__ = forward

    def _lengths_from_context(self, ctx: StrikeContext) -> torch.Tensor:
        """(B, 3) group horizons from the two hit indices (waist = the longer)."""
        left, right = ctx.hit_index[:, 0], ctx.hit_index[:, 1]
        both = torch.maximum(left, right)
        return torch.stack([both, left, right], dim=1)

    # ---------------------------------------------------------- collision --
    def _points_world(self, frames, hand: str, ctx=None) -> torch.Tensor:
        R, p = frames[f"{hand}_wrist"]
        local = self.c_points[hand]
        if ctx is not None and getattr(ctx, "stick_points_local", None) is not None:
            # (B, 2, P, 3) -> (B, P, 3) for this hand, then repeated over the S
            # spline samples the frames were flattened to.
            hi = 0 if hand == "left" else 1
            per = ctx.stick_points_local[:, hi].to(p.dtype)          # (B, P, 3)
            reps = p.shape[0] // per.shape[0]
            local = per.repeat_interleave(reps, dim=0) if reps > 1 else per
        return ChainFK.points_world(R, p, local.to(p.dtype))

    def _collision(self, frames, B, S, ctx, m_hand, raw, viol, live, diag, counts, sp) -> None:
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
        # (knot, capsule/wrist, drum) pair of both hands and all three collision
        # types -- a per-hand max combined afterwards would be a max of
        # p-norms, not the p-norm of the union.  Only assembled when the term
        # is priced.
        want_peak = cfg.w_collision_peak > 0.0
        peak_h: list[torch.Tensor] = []
        peak_m: list[torch.Tensor] = []
        #: Each hand's world sample points, kept for the CROSS-HAND block after
        #: the loop -- the per-hand pass already pays for the transform.
        pts_by_hand: dict[str, torch.Tensor] = {}
        # Domain randomization moves the drums; when the caller supplies the
        # per-sample head centres (fix B) the collision scene is rebuilt from
        # them, so it agrees with the terminal cost and the goal tokens.
        if ctx.drum_head_centers is None:
            body_c = self.c_body_c.view(1, 1, D, 3)
            under_c = self.c_under_c.view(1, 1, D, 3)
        else:
            # The context tokens are padded to ``num_drums`` slots; the scene
            # has however many the layout defines. Row i is drum id i in both
            # (both are sorted by the layout's ``id``), so a prefix slice lines
            # them up.
            heads_b = ctx.drum_head_centers[:, :D].to(dtype)   # (B, D, 3)
            axb = self.c_axes.view(1, D, 3)
            body_c = (heads_b - self.c_half_h.view(1, D, 1) * axb).view(B, 1, D, 3)
            under_top_b = heads_b - self.c_heights_uc.view(1, D, 1) * axb
            under_c = (under_top_b - self.c_under_off * axb).view(B, 1, D, 3)
        # The RADII are per-sample for the same reason (``per_drum_radius``);
        # they are frame-free, so the context column is used verbatim.  Sliced
        # to the scene's drum count exactly as the centres are: the tokens are
        # padded to ``num_drums`` and row i is drum id i in both.
        if ctx.drum_radii is None:
            rad = self.c_radii.view(1, 1, D)
        else:
            rad = ctx.drum_radii[:, :D].to(dtype).view(B, 1, D)
        ax = self.c_axes.view(1, 1, D, 3)
        half_h = self.c_half_h.view(1, 1, D)
        under_half = self.c_under_half.view(1, 1, D)

        for hi, hand in enumerate(_HANDS):
            pts = self._points_world(frames, hand, ctx).reshape(B, S, P, 3)
            pts_by_hand[hand] = pts
            tip_w, butt_w, wrist_w = pts[:, :, 0], pts[:, :, 1], pts[:, :, -1]

            # --- exemption windows (mirror TrajectoryScorer) -----------------
            k = ctx.hit_index[:, hi]
            first_exempt = (k - cfg.strike_exempt_waypoints).clamp_min(0) * n_sub
            win_strike = (s_idx >= first_exempt.view(-1, 1)).to(dtype)  # (B,S)
            last_start = cfg.start_exempt_waypoints * n_sub
            win_start = (s_idx <= last_start).to(dtype).expand(B, S)
            if cfg.start_exempt_waypoints <= 0:
                win_start = torch.zeros_like(win_start)

            tgt = ctx.target_drum[:, hi]
            src = ctx.start_drum[:, hi]
            oh_t = F.one_hot(tgt.clamp(min=0), D).to(dtype) * (tgt >= 0).to(dtype).unsqueeze(-1)
            oh_s = F.one_hot(src.clamp(min=0), D).to(dtype) * (src >= 0).to(dtype).unsqueeze(-1)
            # Stick capsule vs the TARGET (and, at departure, the START) drum
            # is exempted: a hit is contact, so those waypoints cannot also be
            # a collision.  Pads and cymbals share this now that both are
            # capsules; HIT membership is still tip-only on pads.  The wrist
            # sphere is never exempted.
            ex_cap = (
                win_strike.unsqueeze(-1) * oh_t.unsqueeze(1)
                + win_start.unsqueeze(-1) * oh_s.unsqueeze(1)
            ).clamp_max(1.0)
            smask = m_hand[hand]  # (B, S)
            w_cap = (1.0 - ex_cap) * smask.unsqueeze(-1)  # (B,S,D)
            w_wr = smask.unsqueeze(-1).expand(B, S, D)

            cap_b = segment_cylinder_sdf(
                butt_w.unsqueeze(2), tip_w.unsqueeze(2),
                body_c, ax, rad, half_h, ra=self.c_stick_r,
            )
            cap_u = segment_cylinder_sdf(
                butt_w.unsqueeze(2), tip_w.unsqueeze(2),
                under_c, ax, rad, under_half, ra=self.c_stick_r,
            )
            wr_x = wrist_w.unsqueeze(2)  # (B,S,1,3) broadcasts vs (.,1,D,3)
            sdf_wb = point_cylinder_sdf(wr_x, body_c, ax, rad, half_h) - self.c_wrist_r
            sdf_wu = point_cylinder_sdf(wr_x, under_c, ax, rad, under_half) - self.c_wrist_r
            hdc = _softplus_hinge(cfg.collision_margin - cap_b, cfg.beta_collision)
            huc = _softplus_hinge(cfg.collision_margin - cap_u, cfg.beta_collision)
            hdw = _softplus_hinge(cfg.collision_margin - sdf_wb, cfg.beta_collision)
            huw = _softplus_hinge(cfg.collision_margin - sdf_wu, cfg.beta_collision)

            # --- time-bucketed drum collision (diagnostic only) ---------
            hsum = ((hdc + huc) * w_cap + (hdw + huw) * w_wr).sum(dim=2)
            hcnt = (w_cap + w_wr).sum(dim=2).clamp_min(1.0)
            hps = hsum / hcnt                               # (B, S)
            k_wp = k.float().view(B, 1)
            dist_wp = k_wp - s_idx.float() / float(n_sub)
            exempt_start = float(cfg.strike_exempt_waypoints)
            nohit = (k <= 0).view(B, 1)
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

            hand_pen_d = (hdc * w_cap + hdw * w_wr).sum(dim=(1, 2))
            hand_pen_u = (huc * w_cap + huw * w_wr).sum(dim=(1, 2))
            hand_n = (w_cap + w_wr).sum(dim=(1, 2))
            pen_d = pen_d + hand_pen_d
            pen_u = pen_u + hand_pen_u
            n_seen = n_seen + hand_n

            # --- self collision: stick capsule + wrist vs body --------------
            # The upper body rides the commanded waist; the pelvis/legs are a
            # fixed standing-pose capsule bank in the pelvis frame.
            Rt, pt = frames["torso"]
            off = lambda loc: (pt + (Rt @ loc.to(pt.dtype).view(1, 3, 1)).squeeze(-1)
                               ).reshape(B, S, 3)
            a = off(self.c_torso_a)
            b = off(self.c_torso_b)
            hc = off(self.c_head_c)
            s_t = segment_segment_sdf(
                butt_w, tip_w, a, b, self.c_stick_r, self.c_torso_r,
            )
            s_h = segment_segment_sdf(
                butt_w, tip_w, hc, hc, self.c_stick_r, self.c_head_r,
            )
            s_wt = point_capsule_sdf(wrist_w, a, b, self.c_torso_r) - self.c_wrist_r
            s_wh = point_sphere_sdf(wrist_w, hc, self.c_head_r) - self.c_wrist_r
            s_lower_st, s_lower_wr = fixed_capsule_self_sdf(
                butt_w, tip_w, wrist_w,
                self.c_lower_a, self.c_lower_b, self.c_lower_r,
                stick_radius=self.c_stick_r, wrist_radius=self.c_wrist_r,
            )
            sdf_self_st = torch.minimum(torch.minimum(s_t, s_h), s_lower_st)
            sdf_self_wr = torch.minimum(torch.minimum(s_wt, s_wh), s_lower_wr)
            hs_st = _softplus_hinge(cfg.collision_margin - sdf_self_st, cfg.beta_collision)
            hs_wr = _softplus_hinge(cfg.collision_margin - sdf_self_wr, cfg.beta_collision)
            pen_s = pen_s + (hs_st * smask + hs_wr * smask).sum(dim=1)

            # --- raw violation (penetration depth), for the multipliers -------
            vdc = torch.where(w_cap > 0.5, (-cap_b).clamp_min(0.0), torch.zeros_like(cap_b))
            vuc = torch.where(w_cap > 0.5, (-cap_u).clamp_min(0.0), torch.zeros_like(cap_u))
            vdw = torch.where(w_wr > 0.5, (-sdf_wb).clamp_min(0.0), torch.zeros_like(sdf_wb))
            vuw = torch.where(w_wr > 0.5, (-sdf_wu).clamp_min(0.0), torch.zeros_like(sdf_wu))
            vs_st = (-sdf_self_st).clamp_min(0.0) * smask
            vs_wr = (-sdf_self_wr).clamp_min(0.0) * smask
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
                peak_h += [vdc.reshape(B, -1), vuc.reshape(B, -1),
                           vdw.reshape(B, -1), vuw.reshape(B, -1),
                           vs_st, vs_wr]
                peak_m += [w_cap.reshape(B, -1), w_cap.reshape(B, -1),
                           w_wr.reshape(B, -1), w_wr.reshape(B, -1),
                           smask, smask]
            hand_max = torch.maximum(
                torch.maximum(vdc.amax(dim=(1, 2)), vuc.amax(dim=(1, 2))),
                torch.maximum(
                    torch.maximum(vdw.amax(dim=(1, 2)), vuw.amax(dim=(1, 2))),
                    torch.maximum(vs_st.amax(dim=1), vs_wr.amax(dim=1)),
                ),
            )
            v_max = hand_max if v_max is None else torch.maximum(v_max, hand_max)
            big = 1.0e3
            msd = torch.minimum(
                torch.minimum(
                    torch.where(w_cap > 0.5, cap_b, big).amin(dim=(1, 2)),
                    torch.where(w_cap > 0.5, cap_u, big).amin(dim=(1, 2)),
                ),
                torch.minimum(
                    torch.where(w_wr > 0.5, sdf_wb, big).amin(dim=(1, 2)),
                    torch.where(w_wr > 0.5, sdf_wu, big).amin(dim=(1, 2)),
                ),
            )
            msd = torch.minimum(
                msd,
                torch.minimum(
                    torch.where(smask > 0.5, sdf_self_st, big).amin(dim=1),
                    torch.where(smask > 0.5, sdf_self_wr, big).amin(dim=1),
                ),
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
            diag[f"collision_lower_min_sdf_{hand}"] = torch.minimum(
                torch.where(smask > 0.5, s_lower_st, big).amin(dim=1),
                torch.where(smask > 0.5, s_lower_wr, big).amin(dim=1),
            ).detach()

        # --- cross-hand: stick vs stick, wrist vs stick, wrist vs wrist ------
        # Everything above scores ONE hand's capsule against the drums and
        # against the torso/head; the two hands' geometry was never compared to
        # itself, so nothing in the cost forbade the two sticks from passing
        # through each other.
        #
        # The stick AXIS is recoverable from the very points the loop scored:
        # row 0 is the bead and row 1 is the first shaft sample, which BOTH
        # producers of the cloud put at the BUTT -- ``fk.stick_shaft_offsets``
        # and the per-sample randomized :func:`stick_cost_from_context` start
        # their linspace at ``butt_sigma``.  So [row 1 -> row 0] is the whole
        # 0.40 m stick.
        pl, pr = pts_by_hand["left"], pts_by_hand["right"]      # (B, S, P, 3)
        l_tip, l_butt, l_wr = pl[:, :, 0], pl[:, :, 1], pl[:, :, -1]
        r_tip, r_butt, r_wr = pr[:, :, 0], pr[:, :, 1], pr[:, :, -1]
        rs, rw = self.c_stick_r, self.c_wrist_r
        sdf_x = torch.minimum(
            torch.minimum(
                cross_hand_sdf(l_butt, l_tip, r_butt, r_tip, rs, rs),
                cross_hand_sdf(l_wr, l_wr, r_butt, r_tip, rw, rs),
            ),
            torch.minimum(
                cross_hand_sdf(r_wr, r_wr, l_butt, l_tip, rw, rs),
                cross_hand_sdf(l_wr, l_wr, r_wr, r_wr, rw, rw),
            ),
        )                                                       # (B, S)
        # A sample counts only where BOTH hands are inside their horizon: past
        # it the frames are the held tail and the two sticks are not
        # simultaneously real.
        wx = m_hand["left"] * m_hand["right"]                    # (B, S)
        hx = _softplus_hinge(cfg.collision_margin - sdf_x, cfg.beta_collision)
        n_x = wx.sum(dim=1)
        pen_x = (hx * wx).sum(dim=1)
        # Its OWN denominator, not ``n_seen``: this term has S elements per row
        # against the drum block's 2 x S x D (capsule + wrist, both hands),
        # and sharing the divisor would silently scale the weight.
        raw["collision_stick"] = (
            pen_x / n_x.clamp_min(1.0) if cfg.reduction == "mean" else pen_x
        )
        big_x = torch.full_like(sdf_x, 1e3)
        vx = (-sdf_x).clamp_min(0.0) * wx
        msd_x = torch.where(wx > 0.5, sdf_x, big_x).amin(dim=1)
        # The SHARED reductions -- the augmented-Lagrangian violation, the peak
        # group and the reported ``collision_min_sdf`` -- see the cross-hand
        # pair only once the term is PRICED.  At the default
        # ``w_stick_stick = 0`` the multiplier dynamics, the smooth max and
        # every logged number are therefore bit-identical to the pre-cross-hand
        # form; the two diagnostics below are computed either way, so a run can
        # measure its cross-hand clearance before paying for it.
        if cfg.w_stick_stick > 0.0:
            if want_peak:
                peak_h.append(vx)
                peak_m.append(wx)
            vx_max = vx.amax(dim=1)
            v_max = vx_max if v_max is None else torch.maximum(v_max, vx_max)
            min_sdf = msd_x if min_sdf is None else torch.minimum(min_sdf, msd_x)
        diag["collision_stick"] = raw["collision_stick"].detach()
        # A row with no jointly-valid sample reports the raw closest approach
        # rather than the 1e3 sentinel, which would swamp the batch mean the
        # training log records.
        diag["collision_stick_min_sdf"] = torch.where(
            n_x > 0.5, msd_x, sdf_x.amin(dim=1)
        ).detach()
        counts["collision_stick"] = n_x.detach()

        # --- tip lateral ordering (plain objective; not AL) -----------------
        # Bead tips from the same world cloud the collision loop already paid
        # for; torso frame so waist yaw cannot flip the left/right label.
        # ``d = tip_L.y - tip_R.y``; correct order is d >= margin.  Sample 0 is
        # excluded: its q is measured q0 and cannot be optimized (same reason
        # as the waist-rate knot-0 exclusion).
        Rt, pt = frames["torso"]
        Rt_bs = Rt.reshape(B, S, 3, 3)
        pt_bs = pt.reshape(B, S, 3)
        tip_l = pts_by_hand["left"][:, :, 0, :]
        tip_r = pts_by_hand["right"][:, :, 0, :]
        yl = torch.einsum(
            "bsij,bsj->bsi", Rt_bs.transpose(-1, -2), tip_l - pt_bs
        )[..., 1]
        yr = torch.einsum(
            "bsij,bsj->bsi", Rt_bs.transpose(-1, -2), tip_r - pt_bs
        )[..., 1]
        d_lat = yl - yr  # (B, S); >= 0 when left tip is more leftward
        wx_ord = m_hand["left"] * m_hand["right"]
        wx_ord = torch.cat(
            [torch.zeros_like(wx_ord[:, :1]), wx_ord[:, 1:]], dim=1
        )
        h_ord = _softplus_hinge(
            cfg.tip_crossover_margin - d_lat, cfg.beta_tip_crossover
        )
        n_ord = wx_ord.sum(dim=1)
        pen_ord = (h_ord * wx_ord).sum(dim=1)
        raw["tip_crossover"] = (
            pen_ord / n_ord.clamp_min(1.0) if cfg.reduction == "mean" else pen_ord
        )
        counts["tip_crossover"] = n_ord.detach()
        # Signed lateral separation (min over jointly-valid samples) and the
        # peak crossover depth ``relu(-d)``.
        big_lat = torch.full_like(d_lat, 1e3)
        d_masked = torch.where(wx_ord > 0.5, d_lat, big_lat)
        diag["tip_lateral_min"] = torch.where(
            n_ord > 0.5, d_masked.amin(dim=1), d_lat.amin(dim=1)
        ).detach()
        cross_depth = (-d_lat).clamp_min(0.0)
        diag["tip_crossover_max"] = torch.where(
            n_ord > 0.5,
            torch.where(wx_ord > 0.5, cross_depth, torch.zeros_like(cross_depth)
                        ).amax(dim=1),
            cross_depth.amax(dim=1),
        ).detach()

        denom = n_seen.clamp_min(1.0) if cfg.reduction == "mean" else 1.0
        raw["collision_drum"] = pen_d / denom
        raw["collision_under"] = pen_u / denom
        raw["collision_self"] = pen_s / denom
        if want_peak:
            hcat = torch.cat(peak_h, dim=1)
            mcat = torch.cat(peak_m, dim=1)
            raw["collision_peak"] = smooth_max(
                hcat, mcat, cfg.collision_peak_mode, cfg.collision_peak_p,
                cfg.collision_peak_temp, dims=(1,),
            )
        else:
            # Exactly zero, with exactly zero gradient -- the hot path never
            # builds the concatenation and the term is bit-identical to the
            # pre-peak form.
            raw["collision_peak"] = torch.zeros_like(pen_d)
        coll_live = v_max if v_max is not None else pen_d * 0
        viol["collision"] = coll_live.detach()
        # The peak group integrates the SAME per-sample max penetration as
        # ``collision`` for the dual; the priced half is a smooth max.
        viol["collision_peak"] = viol["collision"]
        live["collision"] = coll_live
        live["collision_peak"] = coll_live
        diag["collision_min_sdf"] = min_sdf.detach()
        diag["collision_pen_max"] = coll_live.detach()
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

    def _torque_off(
        self, B, ss, raw, viol, live, diag, counts, *, zero_viol: bool = True,
    ) -> None:
        """Skip coupled RNEA torque: zero hinge, peak, segment groups, effort."""
        z = ss.q.new_zeros(B)
        raw["torque"] = z
        raw["torque"].requires_grad_(False)
        raw["torque_peak"] = z
        raw["torque_peak"].requires_grad_(False)
        if zero_viol:
            raw["torque_static"] = z
            raw["torque_static"].requires_grad_(False)
            viol["torque"] = z.detach()
            viol["torque_peak"] = z.detach()
            live["torque"] = z
            live["torque_peak"] = z
            for name, _idx in self.c_seg:
                key = f"torque_{name}"
                raw[key] = z
                raw[key].requires_grad_(False)
                viol[key] = z.detach()
                live[key] = z
            diag["torque_hinge_max"] = z.detach()
            diag["torque_excess_max"] = z.detach()
            diag["torque_max"] = z.detach()
            diag["torque_frac_over"] = z.detach()
            diag["torque_static_max"] = z.detach()
        diag["torque_sq_mean"] = z.detach()
        diag["torque_abs_mean"] = z.detach()
        raw["effort"] = z
        raw["effort"].requires_grad_(False)
        counts["torque"] = z.detach()
        counts["effort"] = z.detach()
        if zero_viol:
            counts["torque_static"] = z.detach()

    def _torque_static(
        self, sp, ss, v0, B, S, D, raw, viol, live, diag, counts, lim_alpha=None,
    ) -> None:
        """Gravity-only static torque: τ_g = RNEA(q, 0, 0).

        Warning at ``w_torque_static * raw["torque_static"]``.  When
        ``torque_static_al``, also fills the ``torque_*`` AL groups from the
        hard ReLU excess (same groups as coupled RNEA).
        """
        cfg = self.cfg
        if cfg.physics_mode == "knot_fd":
            q, _qd, _qdd = sp.knot_finite_differences(v0)
            idx = torch.arange(
                sp.max_segments + 1, device=q.device, dtype=torch.long,
            ).view(1, -1, 1)
            mask = (idx <= sp.lengths.unsqueeze(1)).to(q.dtype)
        else:
            q, mask = ss.q, ss.mask
        n = q.shape[1]
        q_flat = q.reshape(B * n, D)
        zd = torch.zeros_like(q_flat)
        tau_g = self.rnea.rnea(q_flat, zd, zd).reshape(B, n, D)
        tighten = self.c_static_torque_tighten.view(1, 1, D)
        lim_hard = available_torque_torch(
            zd.reshape(B, n, D), self.tau_hard.view(1, 1, D),
            self.v_hard.view(1, 1, D),
            eta=1.0, saturation_scale=cfg.torque_saturation_scale,
            derate=cfg.torque_derate,
        ) / tighten
        lim = cfg.eta_torque * lim_hard
        excess = tau_g.abs() - lim
        h = _softplus_hinge(excess, cfg.beta_torque)
        ss_local = _Rates(q, zd.reshape(B, n, D), zd.reshape(B, n, D), mask)
        val, cnt = self._reduce(
            h if lim_alpha is None else h * lim_alpha, ss_local.mask, (1, 2),
        )
        raw["torque_static"] = val
        diag["torque_static_max"] = (h * ss_local.mask).amax(dim=(1, 2)).detach()
        counts["torque_static"] = cnt.detach()
        if not cfg.torque_static_al:
            return
        over = (tau_g.abs() - lim_hard).clamp_min(0.0) * ss_local.mask
        if cfg.w_torque_peak > 0.0:
            raw["torque_peak"] = smooth_max(
                over, ss_local.mask, cfg.torque_peak_mode, cfg.torque_peak_p,
                cfg.torque_peak_temp, dims=(1, 2),
            )
        else:
            z = tau_g.new_zeros(B)
            raw["torque_peak"] = z
            raw["torque_peak"].requires_grad_(False)
        mean_live = over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        peak_live = over.amax(dim=(1, 2))
        viol["torque"] = mean_live.detach()
        viol["torque_peak"] = peak_live.detach()
        live["torque"] = mean_live
        live["torque_peak"] = peak_live
        h_priced = h if lim_alpha is None else h * lim_alpha
        self._emit_limit_segments(
            "torque", over, h_priced, ss_local.mask, cnt, raw, viol, live,
        )
        diag["torque_hinge_max"] = (h * ss_local.mask).amax(dim=(1, 2)).detach()
        diag["torque_excess_max"] = viol["torque_peak"]
        diag["torque_max"] = (tau_g.abs() * ss_local.mask).amax(dim=(1, 2)).detach()
        diag["torque_frac_over"] = (
            (over > 0).to(tau_g.dtype).sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        ).detach()

    def _torque_level_a(
        self,
        q_flat: torch.Tensor,
        qdd_flat: torch.Tensor,
        *,
        frozen: bool,
    ) -> torch.Tensor:
        """Decoupled torque: ``g(q) + diag(M) @ qdd`` (no Coriolis / coupling)."""
        z = torch.zeros_like(q_flat)
        tau_g = self.rnea.rnea(q_flat, z, z)
        if frozen:
            m_diag = self._m_diag_ready.to(device=q_flat.device, dtype=q_flat.dtype)
            return tau_g + m_diag.unsqueeze(0) * qdd_flat
        from .pd_sim import mass_matrix

        M = mass_matrix(self.rnea, q_flat)
        m_diag = torch.diagonal(M, dim1=-2, dim2=-1)
        return tau_g + m_diag * qdd_flat

    def _torque(self, sp, ss, v0, B, S, D, ctx, raw, viol, live, diag, counts,
                lim_alpha=None, *, torque_full_blend: float = 1.0) -> torch.Tensor:
        """RNEA torque against ``eta_torque`` x the effort limits.

        Two reductions, both returned:

        * ``raw["torque"]`` -- the mean softplus over valid ``(knot, joint)``
          elements (warning, priced at ``cfg.w_torque``, never × λ).
        * ``raw["torque_peak"]`` -- a **smooth max** of ReLU excess
          ``relu(|τ| - lim_hard)`` over valid elements, priced by
          ``cfg.w_torque_peak * λ_peak``.  A feasible row is identically 0.
          ``viol`` / ``live["torque_peak"]`` stay the horizon **hard max**.

        Each half reports its own raw violation, because the two are the same
        physical quantity reduced differently and are ~``n_valid`` apart in
        magnitude:

        * ``viol["torque"]`` -- the per-element MEAN excess over the hard
          limits;
        * ``viol["torque_peak"]`` -- the per-sample MAX excess over the hard
          limits (identical to ``diag["torque_excess_max"]``), which is the
          quantity the acceptance gate thresholds.

        Both are detached, both are exactly zero on a feasible sample.
        Per-actuator-class groups (``torque_*``) carry the hierarchical dual
        residual; the family mean is a diagnostic alias, not an AL group.

        The peak exists because the mean is the wrong SHAPE for the gate it is
        supposed to move.  ``ok_torque`` in the eval harness is a zero-tolerance
        max against the hard limits, and torque violations are extremely sparse:
        on the recorded eval CSVs of the phase-2 model's ``ok_torque`` failures
        the median case is over on ONE element out of ~30 knots x 17 joints
        (0.27 % of valid elements; p90 0.51 %, worst case 1.5 %) by a median
        0.87 N*m.  One such element moves the mean by ~0.3 % of its own size --
        roughly 1e-3 N*m of loss -- while failing the case outright.  Averaging
        is exactly the operation that destroys the signal, so the peak term puts
        a max-shaped one back alongside it (the mean is kept: it is the warning
        that gives every element gradient on the safe side of the limit).

        Returns the signed RNEA ``tau`` of shape ``(B, n, D)`` so a caller that
        also wants aux torque labels can reuse it without a second RNEA.
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
        q_flat = q.reshape(B * n, D)
        qd_flat = qd.reshape(B * n, D)
        qdd_flat = qdd.reshape(B * n, D)
        blend = float(torque_full_blend)
        use_level_a = cfg.torque_physics != "full" and blend < 1.0 - 1.0e-6
        if not use_level_a:
            tau = self.rnea.rnea(q_flat, qd_flat, qdd_flat).reshape(B, n, D)
        elif blend <= 1.0e-6:
            frozen = cfg.torque_physics == "diag_grav_frozen"
            tau = self._torque_level_a(
                q_flat, qdd_flat, frozen=frozen,
            ).reshape(B, n, D)
        else:
            tau_full = self.rnea.rnea(q_flat, qd_flat, qdd_flat)
            frozen = cfg.torque_physics == "diag_grav_frozen"
            tau_simple = self._torque_level_a(q_flat, qdd_flat, frozen=frozen)
            tau = (blend * tau_full + (1.0 - blend) * tau_simple).reshape(B, n, D)
        # ONE definition of the limit, shared with the tracking hinge, the
        # plant's optional clamp and the eval gate: ``available_torque``.
        # ``eta=1.0`` is the derated HARD rating (what the raw violation is
        # measured against); the hinge sits at ``eta_torque`` x that.
        lim_hard = available_torque_torch(
            qd, self.tau_hard.view(1, 1, D), self.v_hard.view(1, 1, D),
            eta=1.0, saturation_scale=cfg.torque_saturation_scale,
            derate=cfg.torque_derate,
        )
        lim = self.c_torque_eta.view(1, 1, D) * lim_hard
        excess = tau.abs() - lim
        h = _softplus_hinge(excess, cfg.beta_torque)
        ss = _Rates(q, qd, qdd, mask)  # local alias for the reductions below
        # ``lim_alpha`` (1, 1, D) prices the hinge PER JOINT before the mean.
        # The denominator ``cnt`` is the unweighted element count, so this is a
        # re-weighted mean over the same elements, not a change of normalization.
        # Applied to the MEAN warning only: ``raw["torque_peak"]`` is the
        # unscaled smooth-max ReLU excess, and every diagnostic keeps the
        # unscaled ``h``.
        val, cnt = self._reduce(
            h if lim_alpha is None else h * lim_alpha, ss.mask, (1, 2)
        )
        raw["torque"] = val
        diag["torque_hinge_max"] = (h * ss.mask).amax(dim=(1, 2)).detach()
        counts["torque"] = cnt.detach()
        over = (tau.abs() - lim_hard).clamp_min(0.0) * ss.mask
        if cfg.w_torque_peak > 0.0:
            raw["torque_peak"] = smooth_max(
                over, ss.mask, cfg.torque_peak_mode, cfg.torque_peak_p,
                cfg.torque_peak_temp, dims=(1, 2),
            )
        else:
            raw["torque_peak"] = tau.new_zeros(B)
            raw["torque_peak"].requires_grad_(False)
        mean_live = over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        peak_live = over.amax(dim=(1, 2))
        viol["torque"] = mean_live.detach()
        viol["torque_peak"] = peak_live.detach()
        live["torque"] = mean_live
        live["torque_peak"] = peak_live
        h_priced = h if lim_alpha is None else h * lim_alpha
        self._emit_limit_segments(
            "torque", over, h_priced, ss.mask, cnt, raw, viol, live,
        )
        diag["torque_excess_max"] = viol["torque_peak"]
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
        return tau

    # ---------------------------------------------------- tracking torque --
    def _track_torque(self, sp, v0, B, D, raw, viol, live, diag, counts,
                      lim_alpha=None) -> torch.Tensor:
        """The torque the PD servo DEMANDS while tracking this plan.

        ``tau_hat[t] = tau_rnea[t] + tau_fb[t]`` per (waypoint, joint), where

        * ``tau_rnea`` is the plan's own inverse dynamics at the KNOT finite
          differences -- the scorer's convention, and the states a first-order
          hold actually commands (:meth:`TrajectorySpline.knot_finite_differences`),
          not the spline sub-samples the RNEA hinge above uses.  The estimate has
          to live on the control-tick grid because the FIR bank does.
        * ``tau_fb`` is the feedback demand from the identified per-joint FIR
          (:func:`~amortized_opt.pd_sim.feedback_torque`), DC-removed so the
          quasi-static part is counted once, by the RNEA term, and not twice.
          See the note at the top of :mod:`amortized_opt.pd_sim`.

        Both come out of :func:`~amortized_opt.pd_sim.tracking_torque_knots`,
        which is also what the ``"tracking"`` auxiliary torque label calls, so
        the quantity the model's torque head predicts and the quantity this
        term prices are the same array by construction.

        Excess is measured against the SAME ``available_torque`` the RNEA hinge
        uses, and hinged with the same ``beta_torque`` softplus, then reduced
        two ways off the ONE hinge array: a mean (``w_track_torque``, whose
        raw violation drives the ``torque`` augmented-Lagrangian group in place
        of the RNEA mean) and a smooth max (``w_track_torque_peak``).

        ``lim_alpha`` -- the per-joint curriculum ramp -- multiplies the hinge
        before BOTH reductions here (the RNEA block applies it to the mean
        only): the tracking peak carries no group multiplier of its own, so the
        ramp is the only schedule gating it, and a peak that ignored the ramp
        would arrive at full price on step 0.  The raw violations below are
        computed from the UNSCALED excess, exactly as everywhere else, so the
        multiplier dynamics cannot see the ramp.

        Returns the signed ``tau_hat`` of shape ``(B, H+1, D)`` for aux reuse.
        """
        from .pd_sim import tracking_torque_knots

        cfg = self.cfg
        if self.c_tau_kernel is None:          # e.g. a ``with_config`` clone
            self.c_tau_kernel = self.build_torque_kernels()
        # THE shared definition -- the auxiliary label calls the same function.
        tau_hat, q, qd = tracking_torque_knots(sp, v0, self.rnea, self.c_tau_kernel)
        idx = torch.arange(
            sp.max_segments + 1, device=q.device, dtype=torch.long
        ).view(1, -1, 1)
        mask = (idx <= sp.lengths.unsqueeze(1)).to(q.dtype)
        lim_hard = available_torque_torch(
            qd, self.tau_hard.view(1, 1, D), self.v_hard.view(1, 1, D),
            eta=1.0, saturation_scale=cfg.torque_saturation_scale,
            derate=cfg.torque_derate,
        )
        h = _softplus_hinge(
            tau_hat.abs() - self.c_torque_eta.view(1, 1, D) * lim_hard,
            cfg.beta_torque,
        )
        hm = h if lim_alpha is None else h * lim_alpha
        val, cnt = self._reduce(hm, mask, (1, 2))
        raw["track_torque"] = val
        raw["track_torque_peak"] = smooth_max(
            hm, mask, cfg.torque_peak_mode, cfg.torque_peak_p, cfg.torque_peak_temp,
            dims=(1, 2),
        )
        counts["track_torque"] = cnt.detach()
        over = (tau_hat.abs() - lim_hard).clamp_min(0.0) * mask
        mean_live = over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        mean_over = mean_live.detach()
        max_over = over.amax(dim=(1, 2)).detach()
        # THE constraint signal for the ``torque`` group: one multiplier, one
        # constraint, and it is now the one the hardware enforces.  The PHR
        # quadratic follows this replacement -- same residual the dual reads.
        viol["torque"] = mean_over
        live["torque"] = mean_live
        h_priced = hm
        self._emit_limit_segments(
            "torque", over, h_priced, mask, cnt, raw, viol, live,
            write_raw=False,
        )
        diag["track_torque_excess_mean"] = mean_over
        diag["track_torque_excess_max"] = max_over
        diag["track_torque_max"] = (tau_hat.abs() * mask).amax(dim=(1, 2)).detach()
        return tau_hat

    @torch.no_grad()
    def _aux_labels_from_plant(
        self,
        sp,
        ss,
        tau: torch.Tensor | None,
        v0: torch.Tensor,
        *,
        torque_mode: str,
        tau_kernel: torch.Tensor | None,
        tau_track: torch.Tensor | None,
        B: int,
        D: int,
    ) -> dict[str, torch.Tensor]:
        """Aux head labels from plant tensors the cost forward already built.

        Matches :func:`~amortized_opt.warmstart_data.aux_targets` bit-for-bit
        on the default ``rnea_segmax`` path when ``physics_mode="knot_fd"``
        (the production cost).  FK tips use the nominal stick offset -- same as
        ``aux_targets``, not the cost's possibly-randomized ``c_points``.
        """
        from .tokenizer import FK_POINT_NAMES

        if torque_mode not in ("rnea_segmax", "tracking"):
            raise ValueError(
                f"aux_mode must be 'rnea_segmax' or 'tracking', got {torque_mode!r}"
            )
        H = int(sp.max_segments)
        dt = float(self.dt)
        end = torch.full((1,), dt, device=ss.q.device, dtype=ss.q.dtype)
        _q, qd, qdd, _j = sp.eval_offsets(end)
        vel = qd[:, :, 0, :]
        acc = qdd[:, :, 0, :]

        if torque_mode == "tracking":
            from .pd_sim import nominal_torque_kernels, tracking_torque_knots

            if tau_track is not None:
                torque = tau_track[:, 1:].abs()
            else:
                kern = tau_kernel
                if kern is None:
                    if self.c_tau_kernel is None:
                        kern = nominal_torque_kernels(
                            self.rnea,
                            joint_names=tuple(self.joint_names),
                            dt=dt,
                            device=ss.q.device,
                            dtype=ss.q.dtype,
                        )
                    else:
                        kern = self.c_tau_kernel
                tau_hat, _xk, _qd = tracking_torque_knots(sp, v0, self.rnea, kern)
                torque = tau_hat[:, 1:].abs()
            n_knots = H + 1
        else:
            # Prefer the cost's knot RNEA.  knot_fd always evaluates on H+1
            # knots; spline mode with n_sub=1 does too.  Anything denser falls
            # back to an explicit knot pass so the segment-max label stays
            # defined on waypoints.
            if (
                tau is not None
                and (
                    self.cfg.physics_mode == "knot_fd"
                    or (
                        self.cfg.physics_mode == "spline"
                        and int(self.cfg.n_sub) == 1
                    )
                )
                and int(tau.shape[1]) == H + 1
            ):
                ta = tau.abs()
                torque = torch.maximum(ta[:, :-1], ta[:, 1:])
                n_knots = H + 1
            else:
                xk, qd_fd, qdd_fd = sp.knot_finite_differences(v0)
                n_knots = xk.shape[1]
                ta = self.rnea.rnea(
                    xk.reshape(B * n_knots, D),
                    qd_fd.reshape(B * n_knots, D),
                    qdd_fd.reshape(B * n_knots, D),
                ).reshape(B, n_knots, D).abs()
                torque = torch.maximum(ta[:, :-1], ta[:, 1:])

        # FK at ``sp.x_held`` -- the same knot positions ``aux_targets`` gets
        # from ``knot_finite_differences``.  One extra knot FK is cheap next to
        # the RNEA we skipped; subsample FK is only ulp-close and would drift.
        xk = sp.x_held
        pts = self._fk_aux_points_from_q(xk.reshape(B * (H + 1), D)).reshape(
            B, H + 1, len(FK_POINT_NAMES), 3
        )
        fk_pos = pts[:, 1:]
        fk_vel = (pts[:, 1:] - pts[:, :-1]) / dt
        return {
            "vel": vel,
            "acc": acc,
            "torque": torque,
            "fk_pos": fk_pos,
            "fk_vel": fk_vel,
        }

    def _fk_aux_points_from_q(self, q: torch.Tensor) -> torch.Tensor:
        """``(N, 6, 3)`` tip/wrist/elbow for flat ``q`` of shape ``(N, D)``."""
        from .tokenizer import FK_POINT_NAMES

        frames = self.fk.forward(q)
        cols: dict[str, torch.Tensor] = {}
        for hand in ("left", "right"):
            R, p = frames[f"{hand}_wrist"]
            tip_local = torch.as_tensor(
                stick_tip_offset(hand)[None, :], device=q.device, dtype=q.dtype,
            )
            cols[f"{hand}_tip"] = ChainFK.points_world(R, p, tip_local)[:, 0]
            cols[f"{hand}_wrist"] = p
            cols[f"{hand}_elbow"] = frames[f"{hand}_elbow"][1]
        return torch.stack([cols[name] for name in FK_POINT_NAMES], dim=1)

    # ----------------------------------------------------------- velocity --
    def _velocity(self, sp: TrajectorySpline, raw, viol, live, diag, counts,
                  lim_alpha=None) -> None:
        cfg = self.cfg
        B = sp.x.shape[0]
        # Off path: no segment-max |qd|, no hinge, no peak.  A run that does
        # not price velocity should not pay for the analytic locator either.
        if cfg.w_velocity <= 0.0 and cfg.w_velocity_peak <= 0.0:
            z = sp.x.new_zeros(B)
            raw["velocity"] = z
            raw["velocity"].requires_grad_(False)
            raw["velocity_peak"] = z
            raw["velocity_peak"].requires_grad_(False)
            viol["velocity"] = z.detach()
            viol["velocity_peak"] = z.detach()
            live["velocity"] = z
            live["velocity_peak"] = z
            for name, _idx in self.c_seg:
                key = f"velocity_{name}"
                raw[key] = z
                raw[key].requires_grad_(False)
                viol[key] = z.detach()
                live[key] = z
            diag["velocity_excess_max"] = z.detach()
            diag["velocity_max"] = z.detach()
            counts["velocity"] = z.detach()
            return
        vmax = (
            sp.segment_velocity_max_analytic()
            if cfg.velocity_max_mode == "analytic"
            else sp.segment_velocity_max(cfg.n_vel_probe)
        )  # (B, H, n)
        lim_hard = self.v_hard.view(1, 1, -1)
        h = _softplus_hinge(vmax - cfg.eta_velocity * lim_hard, cfg.beta_velocity)
        # Per-joint price on the hinge, before the mean; ``over`` below is the
        # unscaled excess the augmented Lagrangian reads.  See ``forward``.
        val, cnt = self._reduce(
            h if lim_alpha is None else h * lim_alpha, sp.seg_mask, (1, 2)
        )
        raw["velocity"] = val
        counts["velocity"] = cnt.detach()
        over = (vmax - lim_hard).clamp_min(0.0) * sp.seg_mask
        vel_live = over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        viol["velocity"] = vel_live.detach()
        live["velocity"] = vel_live
        h_priced = h if lim_alpha is None else h * lim_alpha
        self._emit_limit_segments(
            "velocity", over, h_priced, sp.seg_mask, cnt, raw, viol, live,
        )
        peak_live = over.amax(dim=(1, 2))
        diag["velocity_excess_max"] = peak_live.detach()
        diag["velocity_max"] = vmax.amax(dim=(1, 2)).detach()
        if cfg.w_velocity_peak > 0.0:
            raw["velocity_peak"] = smooth_max(
                over, sp.seg_mask, cfg.velocity_peak_mode, cfg.velocity_peak_p,
                cfg.velocity_peak_temp, dims=(1, 2),
            )
            viol["velocity_peak"] = peak_live.detach()
            live["velocity_peak"] = peak_live
        else:
            z = vmax.new_zeros(vmax.shape[0])
            raw["velocity_peak"] = z
            raw["velocity_peak"].requires_grad_(False)
            viol["velocity_peak"] = z.detach()
            live["velocity_peak"] = z

    # ---------------------------------------------------------- joint box --
    def _joint_box_hard_limits(self, w_waist_box: float = 1.0):
        """Soft half-width and hard URDF/posture box, optionally waist-narrowed."""
        cfg = self.cfg
        half = cfg.eta_position * self.q_half.view(1, 1, -1)
        lo_hard = self.q_lo_hard.view(1, 1, -1)
        hi_hard = self.q_hi_hard.view(1, 1, -1)
        if cfg.waist_pr_box is not None:
            m_pr = self.c_waist_pr.view(1, 1, -1)
            tight_pr = float(cfg.waist_pr_box)
            half = half + (w_waist_box * m_pr) * (cfg.eta_position * tight_pr - half)
            lo_hard = lo_hard + (w_waist_box * m_pr) * (-tight_pr - lo_hard)
            hi_hard = hi_hard + (w_waist_box * m_pr) * (tight_pr - hi_hard)
            # Keep yaw inside a conservative reach-preserving guardrail whenever
            # the waist box is enabled, using the same existing AL/cost term.
            m_yaw = self.c_waist_yaw.view(1, 1, -1)
            tight_yaw = math.radians(80.0)
            half = half + (w_waist_box * m_yaw) * (cfg.eta_position * tight_yaw - half)
            lo_hard = lo_hard + (w_waist_box * m_yaw) * (-tight_yaw - lo_hard)
            hi_hard = hi_hard + (w_waist_box * m_yaw) * (tight_yaw - hi_hard)
        return half, lo_hard, hi_hard

    def _joint_box(self, waypoints, wp_mask, raw, viol, live, diag, counts,
                   w_waist_box=1.0, lim_alpha=None) -> None:
        """Decoder waypoints against the (optionally waist-narrowed) box.

        The warning is a mean softplus over valid ``(waypoint, joint)`` pairs.
        The optional peak half is a **smooth max** of ReLU hard excess over the
        same grid (max-shaped dense grad).  ``viol`` /
        ``live["joint_box_peak"]`` stay the horizon max for the AL gate.

        ``cfg.waist_pr_box`` shrinks the ``waist_pitch`` / ``waist_roll`` rows
        to ``+-waist_pr_box`` and, when enabled, also narrows ``waist_yaw`` to
        a fixed ±80 deg guardrail.  ``w_waist_box`` linearly blends the
        half-width between the original box (0) and the shrunk one (1).
        """
        cfg = self.cfg
        half, lo_hard, hi_hard = self._joint_box_hard_limits(w_waist_box)
        dev = (waypoints - self.q_mid.view(1, 1, -1)).abs()
        h = _softplus_hinge(dev - half, cfg.beta_position)
        val, cnt = self._reduce(
            h if lim_alpha is None else h * lim_alpha, wp_mask, (1, 2)
        )
        raw["joint_box"] = val
        counts["joint_box"] = cnt.detach()
        over = (
            torch.maximum(lo_hard - waypoints, waypoints - hi_hard)
            .clamp_min(0.0) * wp_mask
        )
        box_live = over.sum(dim=(1, 2)) / cnt.clamp_min(1.0)
        viol["joint_box"] = box_live.detach()
        live["joint_box"] = box_live
        h_priced = h if lim_alpha is None else h * lim_alpha
        self._emit_limit_segments(
            "joint_box", over, h_priced, wp_mask, cnt, raw, viol, live,
        )
        peak_live = over.amax(dim=(1, 2))
        diag["joint_box_excess_max"] = peak_live.detach()
        if cfg.w_joint_box_peak > 0.0:
            raw["joint_box_peak"] = smooth_max(
                over, wp_mask, cfg.joint_box_peak_mode, cfg.joint_box_peak_p,
                cfg.joint_box_peak_temp, dims=(1, 2),
            )
            viol["joint_box_peak"] = peak_live.detach()
            live["joint_box_peak"] = peak_live
        else:
            z = waypoints.new_zeros(waypoints.shape[0])
            raw["joint_box_peak"] = z
            raw["joint_box_peak"].requires_grad_(False)
            viol["joint_box_peak"] = z.detach()
            live["joint_box_peak"] = z

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
    def _posture(self, sp, frames, ss, B, S, ctx, raw, diag, counts) -> None:
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

        # ACTIVE-ARM COMPACTNESS.  Reuse the one FK pass already paid for by
        # collision/terminal/posture.  The four proxy points and masses follow
        # the G1 URDF's rough mass distribution:
        #   shoulder group 2.095 kg, elbow link 0.600 kg,
        #   wrist+rubber hand 0.824 kg, plus a light 0.100 kg stick.
        # Exact link COM FK would require another ~14 frames in the hot pass;
        # this segment-centre proxy preserves the useful gradient at negligible
        # cost and, unlike whole-body COM, is directly controllable here.
        Rt, pt = frames["torso"]
        Rt = Rt.reshape(B, S, 3, 3)
        pt = pt.reshape(B, S, 3)
        compact = ss.q.new_zeros(B)
        compact_n = ss.q.new_zeros(B)
        compact_z_sum = ss.q.new_zeros(B)
        compact_r_sum = ss.q.new_zeros(B)
        z_peak = ss.q.new_zeros(B)
        r_peak = ss.q.new_zeros(B)
        z_ex_peak = ss.q.new_zeros(B)
        r_ex_peak = ss.q.new_zeros(B)
        masses = self.c_arm_compact_masses
        mass_sum = masses.sum()
        for hi, hand in enumerate(_HANDS):
            shoulder = frames[f"{hand}_shoulder"][1].reshape(B, S, 3)
            elbow = frames[f"{hand}_elbow"][1].reshape(B, S, 3)
            Rw, wrist_flat = frames[f"{hand}_wrist"]
            Rw = Rw.reshape(B, S, 3, 3)
            wrist = wrist_flat.reshape(B, S, 3)
            upper = 0.65 * shoulder + 0.35 * elbow
            forearm = 0.5 * (elbow + wrist)
            hand_com = wrist + torch.einsum(
                "bsij,j->bsi", Rw, self.c_mount_p[hand] * 0.25
            )
            stick_com = wrist + torch.einsum(
                "bsij,j->bsi", Rw, self.c_mount_p[hand]
            )
            arm_com = (
                masses[0] * upper + masses[1] * forearm
                + masses[2] * hand_com + masses[3] * stick_com
            ) / mass_sum
            torso_com = torch.einsum(
                "bsji,bsj->bsi", Rt, arm_com - pt
            )
            z = torso_com[..., 2]
            radius = torch.linalg.vector_norm(torso_com[..., :2], dim=-1)
            z_ex = (z - float(cfg.arm_compact_height_max)).clamp_min(0.0)
            r_ex = (radius - float(cfg.arm_compact_radius_max)).clamp_min(0.0)
            # Only scheduled arms participate.  Sample zero is measured q0 and
            # cannot be changed by the decoder, so do not add an objective floor.
            m = m_hand[hand] * ctx.active[:, hi].unsqueeze(1)
            m = torch.cat([torch.zeros_like(m[:, :1]), m[:, 1:]], dim=1)
            if cfg.arm_compact_mode == "linear":
                posture_cost = (
                    float(cfg.arm_compact_height_scale) * z
                    + float(cfg.arm_compact_radius_scale) * radius
                )
            else:
                posture_cost = (
                    float(cfg.arm_compact_height_scale) * z_ex.square()
                    + float(cfg.arm_compact_radius_scale) * r_ex.square()
                )
            compact = compact + posture_cost.mul(m).sum(dim=1)
            compact_n = compact_n + m.sum(dim=1)
            compact_z_sum = compact_z_sum + (z * m).sum(dim=1)
            compact_r_sum = compact_r_sum + (radius * m).sum(dim=1)
            z_peak = torch.maximum(z_peak, (z * m).amax(dim=1))
            r_peak = torch.maximum(r_peak, (radius * m).amax(dim=1))
            z_ex_peak = torch.maximum(z_ex_peak, (z_ex * m).amax(dim=1))
            r_ex_peak = torch.maximum(r_ex_peak, (r_ex * m).amax(dim=1))
        den_compact = compact_n.clamp_min(1.0)
        raw["posture_arm_compact"] = compact / (
            den_compact if cfg.reduction == "mean" else 1.0
        )
        counts["posture_arm_compact"] = compact_n.detach()
        diag["arm_com_height_max"] = z_peak.detach()
        diag["arm_com_radius_max"] = r_peak.detach()
        diag["arm_com_height_mean"] = (compact_z_sum / den_compact).detach()
        diag["arm_com_radius_mean"] = (compact_r_sum / den_compact).detach()
        diag["arm_com_height_excess_max"] = z_ex_peak.detach()
        diag["arm_com_radius_excess_max"] = r_ex_peak.detach()

        # UPRIGHT WELL: mean q^2 over the waist PITCH/ROLL rows only.  The mask
        # is ``ss.mask * c_waist_pr``, so (a) the mean is over 2 rows rather than
        # 17 -- the weight is then directly comparable to a target excursion in
        # rad^2 -- and (b) waist YAW is multiplied by a hard zero, so the term is
        # exactly yaw-invariant and no yaw gradient exists to leak.
        #
        # When ``ctx.waist_pr_ref`` is set the well tracks that set-point
        # instead of 0 (``docs/waist_pr_sp_head_plan.md``).  Ref 0 is
        # bit-identical to the no-ref path except sample 0 is dropped -- a
        # trajectory sitting on the announced lean must read zero, and q0 is
        # not a decision variable.  ``stopgrad(sp)`` is load-bearing: a
        # live ref made this a 20x ungated encoder follow of decoder mean
        # (w=100 vs tether 5), which is how dumps taught the head.
        m_pr = ss.mask * self.c_waist_pr.view(1, 1, -1)
        if ctx.waist_pr_ref is not None:
            pr_idx = self.c_waist_pr.nonzero(as_tuple=True)[0]  # (2,) pitch, roll
            pr_ref = ctx.waist_pr_ref  # (B, H, 2)
            H_ref = pr_ref.shape[1]
            n_sub = int(cfg.n_sub)
            S_pr = ss.q.shape[1]
            knot_idx = (
                (torch.arange(S_pr, device=ss.q.device) - 1) // n_sub
            ).clamp(0, H_ref - 1)
            ref_ss = pr_ref[:, knot_idx, :].detach()  # (B, S, 2)
            q_pr = ss.q.index_select(2, pr_idx)
            m2 = ss.mask.index_select(2, pr_idx)
            m2 = torch.cat([torch.zeros_like(m2[:, :1]), m2[:, 1:]], dim=1)
            d2 = q_pr - ref_ss
            val_up, cnt_up = self._reduce(d2 * d2, m2, (1, 2))
        else:
            val_up, cnt_up = self._reduce(ss.q * ss.q, m_pr, (1, 2))
        raw["posture_upright"] = val_up
        counts["posture_upright"] = cnt_up.detach()
        # Waist usage diagnostics, always reported: the point of the upright
        # term is to move these, and yaw is here to prove it did NOT move.
        qa = ss.q.abs() * ss.mask
        diag["waist_pr_max"] = (qa * self.c_waist_pr.view(1, 1, -1)).amax(dim=(1, 2)).detach()
        diag["waist_yaw_max"] = (qa * self.c_waist_yaw.view(1, 1, -1)).amax(dim=(1, 2)).detach()

        # V_f: P/R at the NEARER active hit, always vs 0.  Same knot the
        # strike terminal gathers (``sp.knot(hit_index)``).  Inactive hands
        # are ignored; no active hit → 0.  The later hit only lengthens
        # the waist window; it is not this term's terminal.
        k = ctx.hit_index.to(dtype=torch.long)
        act = ctx.active > 0.5
        cap = int(sp.max_segments)
        big = torch.full((), cap + 1, device=k.device, dtype=k.dtype)
        k_act = torch.where(act, k.clamp(1, cap), big.expand_as(k))
        k_first = k_act.min(dim=1).values
        has = act.any(dim=1) & (k_first <= cap)
        q_hit = sp.knot(k_first.clamp(1, cap))
        m_hit = self.c_waist_pr.to(dtype=q_hit.dtype)
        sq_hit = (q_hit * q_hit * m_hit).sum(dim=1) / m_hit.sum().clamp_min(1.0)
        raw["waist_pr_terminal"] = torch.where(has, sq_hit, torch.zeros_like(sq_hit))
        counts["waist_pr_terminal"] = (has.to(q_hit.dtype) * m_hit.sum()).detach()
        diag["waist_pr_N"] = (q_hit.abs() * m_hit).amax(dim=1).detach()

        # WAIST-YAW RATE well.  Same shape as the upright well one block up, on
        # ``qd`` instead of ``q`` and on the YAW row instead of pitch/roll, so
        # the two are exact complements: that one says where the torso may lean,
        # this one says how fast it may turn.  Reduced over the yaw row alone,
        # which is what makes the weight readable as a target rate in
        # (rad/s)^2.  The peak rate is reported unconditionally -- it is the
        # number the ablation is actually about, and a run with the weight at
        # zero still has to be able to show what its waist was doing.
        m_yaw = ss.mask * self.c_waist_yaw.view(1, 1, -1)
        # Sample 0's qd IS the measured v0 -- a constant the optimizer cannot
        # move, so pricing it (or reporting it as the peak) is an
        # un-optimizable floor; exclude it from the term and the diagnostic.
        m_yaw = torch.cat([torch.zeros_like(m_yaw[:, :1]), m_yaw[:, 1:]], dim=1)
        val_yv, cnt_yv = self._reduce(ss.qd * ss.qd, m_yaw, (1, 2))
        raw["waist_yaw_vel"] = val_yv
        counts["waist_yaw_vel"] = cnt_yv.detach()
        diag["waist_yawd_max"] = (
            ss.qd.abs() * m_yaw
        ).amax(dim=(1, 2)).detach()
        # PITCH/ROLL RATE well: the exact complement of the yaw one above, on
        # the two pitch/roll rows.  Same knot-0 exclusion, same reason -- v0 is
        # measured and un-optimizable.  ``waist_prd_max`` is its peak, reported
        # unconditionally (including at weight 0) so an ablation always has a
        # baseline to read.
        m_prv = ss.mask * self.c_waist_pr.view(1, 1, -1)
        m_prv = torch.cat([torch.zeros_like(m_prv[:, :1]), m_prv[:, 1:]], dim=1)
        val_prv, cnt_prv = self._reduce(ss.qd * ss.qd, m_prv, (1, 2))
        raw["waist_pr_vel"] = val_prv
        counts["waist_pr_vel"] = cnt_prv.detach()
        diag["waist_prd_max"] = (ss.qd.abs() * m_prv).amax(dim=(1, 2)).detach()

        # IDLE ARM: tip inside the torso-frame safety cylinder, arrive stopped.
        #
        # The mask is ``m_hand[h] * (1 - active[h])``, so (a) only a hand with no
        # scheduled strike is priced -- a striking hand's terminal objective is
        # never contested -- and (b) the cylinder hinge runs over that hand's
        # whole horizon (not just the terminal knot).  The cylinder is torso-
        # frame, so the park region rides with waist yaw by construction.
        idle_pen = ss.q.new_zeros(B)
        idle_vel = ss.q.new_zeros(B)
        idle_n = ss.q.new_zeros(B)
        idle_worst = ss.q.new_zeros(B)
        if self.c_idle_cyl is not None and (
            cfg.w_idle_cylinder > 0.0 or cfg.w_idle_vel > 0.0
        ):
            S_ = ss.q.shape[1]
            n_dof = ss.q.shape[2]
            center = self.c_idle_cyl[:3]
            axis = self.c_idle_cyl[3:6]
            radius = self.c_idle_cyl[6]
            hh = self.c_idle_cyl[7]
            beta = float(cfg.beta_idle_cyl)
            q_flat = None
            if cfg.w_idle_cylinder > 0.0:
                q_flat = ss.q.reshape(B * S_, n_dof)
            for hi, hand in enumerate(_HANDS):
                cols = self._gcols[hand]
                idle = 1.0 - ctx.active[:, hi]                          # (B,)
                m = m_hand[hand]                                        # (B, S)
                if cfg.w_idle_cylinder > 0.0:
                    tip = tip_in_torso(self.fk, q_flat, hand).reshape(B, S_, 3)
                    d = tip - center.view(1, 1, 3)
                    z = (d * axis.view(1, 1, 3)).sum(-1)
                    radial = d - z.unsqueeze(-1) * axis.view(1, 1, 3)
                    rho = torch.linalg.norm(radial, dim=-1)
                    s_r = (rho - radius).clamp_min(0.0)
                    s_z = (z.abs() - hh).clamp_min(0.0)
                    outside = torch.hypot(s_r, s_z)                     # (B, S)
                    h = _softplus_hinge(outside, beta)
                    den = m.sum(dim=1).clamp_min(1.0)
                    # Mean softplus(outside)^2 over masked knots, then * idle.
                    idle_pen = idle_pen + idle * ((h * h * m).sum(dim=1) / den)
                    idle_worst = torch.maximum(
                        idle_worst, idle * (outside * m).amax(dim=1)
                    )
                # Terminal velocity zero at THIS hand's terminal sample.
                k = (m.sum(dim=1) - 1.0).clamp(0.0, S_ - 1).long()
                g = k.view(-1, 1, 1).expand(-1, 1, cols.numel())
                qd_k = ss.qd.index_select(2, cols).gather(1, g).squeeze(1)
                idle_vel = idle_vel + idle * (qd_k * qd_k).sum(-1)
                idle_n = idle_n + idle
        den_idle = idle_n.clamp_min(1.0)
        raw["posture_idle_cylinder"] = idle_pen / den_idle
        raw["posture_idle_vel"] = idle_vel / den_idle
        counts["posture_idle"] = idle_n.detach()
        diag["idle_cyl_outside"] = idle_worst.detach()

        # Waist-aim IL: penalise yaw, pitch, and roll deviation from the
        # reference trajectory.  Yaw follows waist_aim; pitch and roll are 0.
        # The reference is at knot resolution (H); expand to sub-sample grid (S).
        if ctx.waist_yaw_ref is not None:
            yaw_ref = ctx.waist_yaw_ref  # (B, H)
            H_ref = yaw_ref.shape[1]
            n_sub = int(cfg.n_sub)
            S = ss.q.shape[1]
            # Map sub-sample index -> reference row.  ``ref[k]`` is KNOT k+1
            # (the rows-1..H convention documented on
            # ``StrikeContext.waist_yaw_ref``), and sample ``j * n_sub`` is
            # knot j, so knot j reads row j-1.  Sample 0 is the measured q0 --
            # a constant no decision variable can move -- so it is masked out
            # of the reduction below rather than priced against row 0.
            knot_idx = (
                (torch.arange(S, device=ss.q.device) - 1) // n_sub
            ).clamp(0, H_ref - 1)
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
            err = q_waist - ref_ss  # (B, S, 3), [yaw, pitch, roll]
            # Deadband applies to YAW only: inside the band the yaw error is
            # free, outside only the excess is squared.  Pitch/roll keep the
            # plain quadratic (they are a 0-reference, not a set-point), and
            # ``waist_aim_yaw_only`` drops them entirely for runs where the
            # upright well already owns that job.
            db = float(cfg.waist_aim_deadband)
            e_yaw = err[:, :, 0]
            if db > 0.0:
                e_yaw = (e_yaw.abs() - db).clamp_min(0.0)
            sq_err = e_yaw * e_yaw
            if not cfg.waist_aim_yaw_only and err.shape[-1] > 1:
                sq_err = sq_err + (err[:, :, 1:] ** 2).sum(dim=-1)
            # Reduce over time, masked by validity.  Sample 0 (the measured
            # q0) is excluded: it is constant w.r.t. the decision variables,
            # so it contributes no gradient -- only a floor that would keep
            # ``raw["waist_aim"]`` from reading zero on a perfectly tracking
            # trajectory.
            m_any = ss.mask.amax(dim=2)  # (B, S)
            m_any = torch.cat(
                [torch.zeros_like(m_any[:, :1]), m_any[:, 1:]], dim=1
            )
            val_aim = (sq_err * m_any).sum(dim=1) / m_any.sum(dim=1).clamp_min(1.0)
            raw["waist_aim"] = val_aim
        else:
            raw["waist_aim"] = ss.q.new_zeros(B)

        # TARGET-DIRECTED MONOTONIC YAW.  The desired sign comes from the
        # measured boundary pose to the analytic reference endpoint, while the
        # penalty reads every adjacent spline sample.  A positive target thus
        # makes every negative increment expensive, including the very first
        # command.  Deceleration is free as long as position keeps increasing.
        if ctx.waist_yaw_ref is not None:
            yaw_i = int(self.c_waist_yaw.nonzero(as_tuple=True)[0][0])
            q_yaw = ss.q[:, :, yaw_i]
            # One monotonic segment ends at the NEAREST active hit.  Samples
            # after that are deliberately unpriced: when the hit fires MPC
            # re-goals, recomputes the analytic aim from the new state, and the
            # next segment may choose the opposite direction immediately.
            k = ctx.hit_index.to(dtype=torch.long)
            active = ctx.active > 0.5
            H_ref = int(ctx.waist_yaw_ref.shape[1])
            big = torch.full_like(k, H_ref + 1)
            k_first = torch.where(active, k.clamp(1, H_ref), big).min(dim=1).values
            has_hit = active.any(dim=1) & (k_first <= H_ref)
            ref_i = (k_first.clamp(1, H_ref) - 1).view(-1, 1)
            target_yaw = ctx.waist_yaw_ref.gather(1, ref_i)[:, 0]
            endpoint_delta = target_yaw - q_yaw[:, 0].detach()
            db = float(cfg.waist_yaw_monotonic_target_deadband)
            direction = torch.where(
                endpoint_delta.abs() > db,
                endpoint_delta.sign(),
                torch.zeros_like(endpoint_delta),
            ).detach()
            # Causal stop-gradient on the preceding sample: a wrong-way step
            # is repaired by moving the CURRENT sample in the desired direction,
            # never by dragging the already-earlier sample backward to meet it.
            dq = q_yaw[:, 1:] - q_yaw[:, :-1].detach()
            sample_i = torch.arange(
                1, ss.q.shape[1], device=ss.q.device, dtype=torch.long,
            ).view(1, -1)
            segment_end = (k_first * int(cfg.n_sub)).view(-1, 1)
            valid = (
                ss.mask[:, 1:, yaw_i]
                * (direction != 0).to(ss.q.dtype).unsqueeze(1)
                * has_hit.to(ss.q.dtype).unsqueeze(1)
                * (sample_i <= segment_end).to(ss.q.dtype)
            )
            wrong = (-direction.unsqueeze(1) * dq).clamp_min(0.0)
            n_mono = valid.sum(dim=1).clamp_min(1.0)
            raw["waist_yaw_monotonic"] = (wrong * valid).sum(dim=1) / n_mono
            counts["waist_yaw_monotonic"] = valid.sum(dim=1).detach()

            sample_all = torch.arange(
                ss.q.shape[1], device=ss.q.device, dtype=torch.long,
            ).view(1, -1)
            valid_acc = (
                ss.mask[:, :, yaw_i]
                * (direction != 0).to(ss.q.dtype).unsqueeze(1)
                * has_hit.to(ss.q.dtype).unsqueeze(1)
                * (sample_all <= segment_end).to(ss.q.dtype)
            )
            wrong_acc = (
                -direction.unsqueeze(1) * ss.qdd[:, :, yaw_i]
            ).clamp_min(0.0)
            n_acc = valid_acc.sum(dim=1).clamp_min(1.0)
            raw["waist_yaw_accel_direction"] = (
                wrong_acc * valid_acc
            ).sum(dim=1) / n_acc
            counts["waist_yaw_accel_direction"] = valid_acc.sum(dim=1).detach()
            raw["waist_yaw_initial_accel_direction"] = (
                -direction * ss.qdd[:, 0, yaw_i]
            ).clamp_min(0.0) * has_hit.to(ss.q.dtype)
            counts["waist_yaw_initial_accel_direction"] = has_hit.to(
                ss.q.dtype
            ).detach()
        else:
            raw["waist_yaw_monotonic"] = ss.q.new_zeros(B)
            counts["waist_yaw_monotonic"] = ss.q.new_zeros(B)
            raw["waist_yaw_accel_direction"] = ss.q.new_zeros(B)
            counts["waist_yaw_accel_direction"] = ss.q.new_zeros(B)
            raw["waist_yaw_initial_accel_direction"] = ss.q.new_zeros(B)
            counts["waist_yaw_initial_accel_direction"] = ss.q.new_zeros(B)

    # --------------------------------------------------- waist P/R band ---
    def _waist_pr_band(self, waypoints, wp_mask, B, ctx, raw, viol, live) -> None:
        """Decoder P/R vs ``stopgrad(encoder sp)`` on plan waypoints.

        ``raw`` is the **sum** of ReLU excess over valid ``(waypoint, P/R)``
        so every plan row past the band gets gradient.  ``viol`` / ``live``
        stay the horizon **max**.  When ``ctx.waist_pr_ref`` is ``None`` and
        ``w_waist_pr_band > 0`` the reference defaults to zero (upright), so
        the band penalises any lean beyond ``waist_pr_band_rad``.  Truly idle
        (exact-zero) only when both ``waist_pr_ref`` is None AND the weight
        is zero.  ``sp`` is detached so this group cannot move the encoder.
        """
        cfg = self.cfg
        z = waypoints.new_zeros(B)
        if ctx.waist_pr_ref is None and cfg.w_waist_pr_band <= 0.0:
            # Band weight off AND no reference: truly idle.
            raw["waist_pr_band"] = z
            viol["waist_pr_band"] = z
            live["waist_pr_band"] = z
            return
        pr_idx = self.c_waist_pr.nonzero(as_tuple=True)[0]  # (2,) pitch, roll
        # When no encoder SP head is active, treat the reference as upright
        # (zero pitch/roll) so the band becomes "stay within band_rad of vertical".
        if ctx.waist_pr_ref is None:
            H_wp = waypoints.shape[1]
            pr_ref = waypoints.new_zeros(B, H_wp, 2)
        else:
            pr_ref = ctx.waist_pr_ref.detach()  # (B, H, 2); stopgrad(sp)
        H_ref = pr_ref.shape[1]
        H_wp = waypoints.shape[1]
        if H_wp != H_ref:
            # Waypoints are knots 1..H; ref is the same convention.
            # Truncate/pad to match if a caller mismatched horizons.
            H = min(H_wp, H_ref)
            q_pr = waypoints[:, :H].index_select(2, pr_idx)
            ref = pr_ref[:, :H]
            m2 = wp_mask[:, :H].index_select(2, pr_idx)
        else:
            q_pr = waypoints.index_select(2, pr_idx)
            ref = pr_ref
            m2 = wp_mask.index_select(2, pr_idx)
        excess = ((q_pr - ref).abs() - float(cfg.waist_pr_band_rad)).clamp_min(0.0)
        over = excess * m2
        raw["waist_pr_band"] = over.sum(dim=(1, 2))
        peak_live = over.amax(dim=(1, 2))
        live["waist_pr_band"] = peak_live
        viol["waist_pr_band"] = peak_live.detach()

    # ---------------------------------------------------------- terminal ---
    def _terminal(self, sp: TrajectorySpline, ctx: StrikeContext,
                  raw, viol, live, diag, counts) -> None:
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
            if ctx.stick_axis_local is not None:
                ax_l = ctx.stick_axis_local[:, hi].to(dtype)            # (B, 3)
                ax_l = torch.cat([ax_l, ax_l], dim=0).unsqueeze(-1)     # knot + prev
            else:
                ax_l = self.c_mount_axis[hand].to(dtype).view(1, 3, 1)
            u = (R @ ax_l).squeeze(-1)
            base_k, base_p = base[:B], base[B:]
            u_k, u_p = u[:B], u[B:]

            # Playing-shaft capsule on cymbals/hi-hat; a degenerate (bead)
            # segment on pads.  ``spec`` collapses every row to contact_sigma.
            spec_s = ctx.contact_sigma[:, hi].to(dtype)
            if ctx.stick_tip_sigma is not None:
                tip_s = ctx.stick_tip_sigma[:, hi].to(dtype)
            else:
                tip_s = spec_s.new_full((B,), float(STICK_TIP_SIGMA))
            fb = spec_s.new_full((B,), float(CYMBAL_CONTACT_SIGMA_FALLBACK))
            if cfg.contact_mode == "spec":
                sig_lo = spec_s
                sig_hi = spec_s
            else:
                shaft = ctx.shaft_allowed[:, hi]
                sig_lo = torch.where(shaft > 0.5, fb, spec_s)
                sig_hi = torch.where(shaft > 0.5, tip_s, spec_s)
                sig_lo = torch.minimum(sig_lo, sig_hi)
            a_k = base_k + sig_lo.unsqueeze(-1) * u_k
            b_k = base_k + sig_hi.unsqueeze(-1) * u_k
            a_p = base_p + sig_lo.unsqueeze(-1) * u_p
            b_p = base_p + sig_hi.unsqueeze(-1) * u_p
            _dmin, s_star, p_c, cand_s, dist, valid = segment_hit_region(
                a_k, b_k,
                ctx.p_star[:, hi], ctx.n_out[:, hi], ctx.hit_radius[:, hi],
                ctx.disk_offset[:, hi], ctx.half_plane_dir[:, hi],
                ctx.has_half_plane[:, hi],
                cfg.terminal_axial_lo, cfg.terminal_axial_hi,
                return_candidates=True,
            )
            d_ab_k = b_k - a_k
            d_ab_p = b_p - a_p
            pc_k = a_k.unsqueeze(1) + cand_s.unsqueeze(-1) * d_ab_k.unsqueeze(1)
            pc_p = a_p.unsqueeze(1) + cand_s.unsqueeze(-1) * d_ab_p.unsqueeze(1)
            v_c = (pc_k - pc_p) / dt

            logits = -(dist + 1e3 * (1.0 - valid)) / cfg.contact_softmin_temp
            wgt = torch.softmax(logits, dim=-1)
            act = ctx.active[:, hi]

            reg = reg + act * (wgt * _softplus_hinge(dist, cfg.beta_terminal)).sum(-1)
            reg_raw = reg_raw + act * (wgt * dist).sum(-1)
            # In-plane distance to the AIM (do not subtract ``disk_offset``:
            # toms/cymbals aim off-centre, and the pull is toward the star the
            # strike asked for).  Deadband then a region-style softplus so the
            # term is idle once the contact is on the interior of the head.
            e = pc_k - ctx.p_star[:, hi].unsqueeze(1)
            axial_c = (e * ctx.n_out[:, hi].unsqueeze(1)).sum(-1)
            e_in_aim = e - axial_c.unsqueeze(-1) * ctx.n_out[:, hi].unsqueeze(1)
            rho_aim = torch.sqrt((e_in_aim * e_in_aim).sum(-1) + 1e-12)
            aim_excess = (
                rho_aim - float(cfg.terminal_aim_deadband_m)
            ).clamp_min(0.0)
            aim = aim + act * (
                wgt * _softplus_hinge(aim_excess, cfg.beta_terminal)
            ).sum(-1)

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
            vel = vel + act * (wgt * vcost).sum(-1)

            # stick elevation band on |elev|
            # Interior clamp: ``asin'(±1)`` is inf, and ``clamp`` onto the pole
            # then does ``0 * inf → NaN`` in the backward.  Same floor as
            # :func:`~amortized_opt.tokenizer._tilt_angle`.
            cosang = -(u_k * ctx.n_out[:, hi]).sum(-1).clamp(
                -1.0 + 1e-6, 1.0 - 1e-6
            )
            el = torch.asin(cosang).abs()
            lo = float(np.deg2rad(cfg.elev_min_deg))
            hi_ = float(np.deg2rad(cfg.elev_max_deg))
            ep = (
                _softplus_hinge(lo - el, cfg.beta_elev)
                + _softplus_hinge(el - hi_, cfg.beta_elev)
            )
            elev_pen = elev_pen + act * ctx.elev_required[:, hi] * ep
            n_act = n_act + act

            # diagnostics at the selected capsule contact (what the harness scores)
            pk = p_c
            pp = a_p + s_star.unsqueeze(-1) * d_ab_p
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
        term_live = reg_raw / den
        viol["terminal"] = term_live.detach()
        live["terminal"] = term_live
        diag["terminal_region_dist"] = term_live.detach()
        diag["contact_pos_err"] = (sum(d_pos) / den).detach()
        diag["contact_speed_err"] = (sum(d_speed) / den).detach()
        diag["contact_dir_cos"] = (sum(d_cos) / den).detach()
        diag["stick_elev_deg"] = (sum(d_elev) / den).detach()
        counts["terminal"] = n_act.detach()
