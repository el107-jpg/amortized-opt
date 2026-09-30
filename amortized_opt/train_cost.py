"""Phase 2+3 merged: direct cost descent with AL-balanced limits and chained BPTT.

Section 7 of ``docs/amortized_optimizer_plan.md``, under resolved decisions 1,
2, 3, 5 and 6.  Phase 1 put the decoder in a trajectory-producing regime by
imitating a quintic; this module stops imitating anything and back-propagates
the differentiable cost through the decoder instead.

The objective
-------------

Per step, on one horizon-bucket-homogeneous minibatch::

    L =   w_cost * mean_b FusedCost(q0, model(init, state, ctx), v0)
        + w_aux  * Huber(aux heads, labels from the cost plant on the model's traj)
        + w_tc   * temporal-consistency disagreement on a subset
        + w_bptt * FusedCost of a SECOND solve, after a differentiable PD roll
                   through the first hit and a bounce   (ramped in at the tail)

* the auxiliary heads stay supervised throughout (the plan: "the auxiliary
  terms ... should have continued throughout the entire thing"), but their
  labels now come from the model's own output trajectory through
  ``tspline`` / ``ChainFK`` / ``knot_fd`` + RNEA -- a self-consistent world
  model rather than a frozen teacher;
* there is no imitation term.  ``init`` is a warm start, never a target.

Weight schedule
---------------

Cost weights are at **full strength from step 0** (:func:`default_weight_schedule`
= :meth:`WeightSchedule.snap_full`).  The :class:`~amortized_opt.lagrangian.AugmentedLagrangian`
adjusts λ from raw violations; explicit smoothstep ramps remain available via
CLI for A/B only.  Aux losses (BPTT, suffix Bellman) keep their own phase-in on
:class:`Phase2Config`.

Data (the plan's DAgger mix)
----------------------------

* a **fresh pool** of i.i.d. scenarios (:mod:`amortized_opt.phase2_data`), 70 %
  plan-native / 30 % eval-like, domain-randomized kit, impossible-transition
  filter on;
* a **replay** of MPC contexts produced by :class:`MPCRoller` -- a batched,
  on-GPU receding-horizon loop driven by the *current* weights: plan, execute
  one waypoint into the noised PD simulator, shift, re-plan.  Recency-biased
  sampling; the mix ramps to ``replay_frac``.

The roller is the ``mpc_gen`` loop with two deliberate differences, both for
batching: the mass matrix is frozen at the tick's own pose rather than
refreshed per knot inside a rollout, and a slot that reaches its hit is
respawned from the pool (carrying its post-bounce state with probability
``chain_frac``) instead of re-goaling in place.  Cross-hit chaining is what the
chained-BPTT term trains explicitly, so it is not lost.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import signal
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF, PELVIS_IN_ENV
from .constants import SHAFT_HIT_SAMPLES, STICK_MIN_SEP
from .strike_seed import accelerating_path_torch, strike_seed
from .cost import CostConfig, FusedCost, StrikeContext, stick_cost_from_context
from .fk import ChainFK, stick_tip_offset
from .frontier import (
    FrontierCloud,
    FrontierConfig,
    ingest_pool,
    patch_spawn_q0,
    steal_q0,
)
from .frontier import chart_id as frontier_chart_id
from .grad_balance import (
    GRAD_BALANCE_GROUPS,
    apply_al_gradnorm,
    apply_al_pcgrad,
    clip_torque_gradient_budget,
    measure_group_grad_cosines,
    output_space_gradnorm_surrogate,
    output_space_al_pcgrad,
    output_space_protected_surrogate_gradient,
    trainable_params,
    update_group_norm_ema,
)
from .idle_cylinder import derive_cylinder, resolve_idle_cylinder_library
from .idle_home import resolve_idle_home
from .ik import (
    adjust_arm_terminal,
    correct_quintic_init_arms,
    inertia_weighted_pinv,
    tip_jacobian,
)
from .lagrangian import (
    ACTUATOR_SEGMENTS,
    ACTUATOR_SEGMENT_NAMES,
    CONSTRAINT_GROUPS,
    ALConfig,
    AugmentedLagrangian,
    LIMIT_FAMILIES,
    LIMIT_GROUP_KEYS,
    limit_group_name,
)
from .lead import lead_fraction, lead_waypoints
from .limits import (
    ETA_VELOCITY,
    available_torque_torch,
    isaac_effort_limits,
    isaac_velocity_limits,
    urdf_position_limits,
)
from .model import ModelConfig
from .pd_sim import BatchedPDSim, PDSimConfig, mass_matrix
from .voxel_pool import (
    BootstrapCandidate,
    ChartTemplate,
    ProbeBuffer,
    ProbeCandidateKey,
    VoxelKey,
    VoxelPool,
    ingest_certified_candidates,
    ordinary_bootstrap_candidate,
    ordinary_slice_batches,
    post_hit_bootstrap_candidate,
    rest_bootstrap_candidate,
)
from .phase2_data import (
    KIND_FAR_FETCHED,
    KIND_POST_HIT,
    KIND_RHYTHM_IDLE,
    Phase2Noise,
    Phase2Pool,
    Phase2ScenarioConfig,
    drum_name_predicates,
    strike_context_from_arrays,
    strike_geometry_table,
)
from .rhythm import HANDS, DrumKit
from .rnea import BatchedRNEA
from .sdf import SdfPenalty, q0_collision_clear
from .strike import STICK_TIP_SIGMA
from .tensorboard_log import MetricsTensorboard
from .tokenizer import ContextInputs, FK_POINT_NAMES, build_waypoint_masks
from .train_warmstart import LossScales, _masked_huber
# The encoder's own set-point clamp, so the search cannot propose a target the
# head is structurally unable to reach (see :func:`waist_sp_bound`).
from .waist_aim import _DEFAULT_YAW_LIMIT as _WAIST_SP_LIMIT
from .warmstart import (
    GoalPoseLibrary, arm_slice, seed_waist_pr_to_sp, update_waist_pr_ema,
    waist_pr_slice, waist_slice,
)
from .warmstart_data import CONTEXT_FIELDS
from .violation_critic import (
    ViolationCriticConfig,
    ViolationCriticTrainer,
    build_critic_features,
    critic_feature_dim,
    residuals_from_gate,
    robust_transform,
)
from .distributional_critic import OnlineDistributionalCriticTrainer
from .timing_curriculum import (
    directional_pairs,
    first_exit_indices,
    sample_directions,
)
from .line_search import LineSearchConfig, check_early_triggers, latch_triggers, reset_history_for_objective_change
from .line_search_resume import pack_line_search_checkpoint_extra
from .lbfgs_ls import LbfgsLineSearchAdapter
from .line_search_step import (
    assert_replayable_extras_or_raise,
    build_objective_direction_fingerprint,
    init_line_search_runtime,
    resolve_known_search,
    needs_line_search_direction,
    run_search_or_ordinary,
    snapshot_g_preclip,
)
from .objective_replay import (
    FrozenControls,
    GlobalTorchRngState,
    ObjectiveReplayContext,
    ReplayIncompleteError,
    restore_mutable_batch_fields,
    restore_global_torch_rng,
    restore_stream,
    snapshot_global_torch_rng,
    snapshot_streams,
)

__all__ = [
    "BounceConfig",
    "LineSearchConfig",
    "MPCRoller",
    "Phase2Config",
    "ViolationCriticConfig",
    "ReplayPool",
    "WeightSchedule",
    "default_weight_schedule",
    "apply_bounce",
    "bounce_tip_velocity",
    "build_costs",
    "effective_hits",
    "encode",
    "make_cost_step",
    "make_suffix_step",
    "shift_plan",
    "suffix_offsets",
    "phase2_al_config",
    "phase2_cost_config",
    "stage_a_pass",
    "STAGE_A_MIN_ACCEPTED",
    "STAGE_A_EVAL_N",
    "cost_tuple",
    "measure_cost_term_grad_norms",
    "measure_per_row_torque_grad_norms",
    "forward_torque_viol_per_row",
    "phase2_config_from_checkpoint",
    "prime_replay_from_roller",
    "summarize_torque_grad_stratify",
    "measure_torque_directional_curvature",
    "_hard_feasibility_ok_from_cost",
    "ramp_seed",
    "retime_batch_timing",
    "seed_waist_pr_to_sp",
    "update_waist_pr_ema",
    "vel_ramp_seed",
    "resolve_idle_cylinder_library",
    "resolve_idle_home",
    "save_phase2_checkpoint",
    "load_phase2_checkpoint",
    "mpc_init_tie_loss",
    "waist_yaw_mpc_init_tie_loss",
    "temporal_consistency_loss",
    "train_phase2",
    "WAIST_SP_SELF_TERMS",
    "outcome_only_weights",
    "substitute_waist_sp",
    "substitute_waist_setpoints",
    "waist_sp_bound",
    "waist_sp_candidates",
    "waist_sp_offsets",
    "waist_sp_softmin",
    "clone_context",
    "slice_context",
    "promote_context_next_hit",
    "refresh_context_aim",
    "waist_pr_hit_teacher",
    "waist_pr_distill_gate",
    "waist_pr_distill_loss",
    "waist_pr_hold_band_excess",
    "waist_pr_hold_band_loss",
    "PR_DISTILL_GATE_KEYS",
]

_HANDS = ("left", "right")


# ============================================================== schedule =====
def _smoothstep(t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class WeightSchedule:
    """Per-step multipliers on priced cost groups.

    Defaults (via :func:`default_weight_schedule`) are already at 1.0 from step
    0.  Explicit ``ramp_start`` / ``ramp_end`` / ``*_ramp`` fields remain for
    A/B and ``--full-weights`` / CLI overrides.
    """

    ramp_start: int = 2_000
    ramp_end: int = 20_000
    #: Multiplier at step 0 for the ramped groups (only used when ramps are on).
    lo_limits: float = 0.05        # torque / velocity / joint_box
    lo_strike: float = 0.10        # impact speed + stick elevation
    #: Groups that are NOT ramped (stage-A emphasis, full from the start).
    w_collision: float = 1.0
    w_posture: float = 1.0
    w_terminal_region: float = 1.0
    w_smooth: float = 1.0

    # ---- the torque PEAK sub-term (``CostConfig.w_torque_peak``) ------------
    #: Sum of ReLU hard torque excess; AL integrates the horizon max.  Own
    #: multiplier (``lam_torque_peak``).  Ramped into ``weights["torque_peak"]``.
    torque_peak_start: int = 0
    torque_peak_ramp: int = 1_000

    # ---- the velocity PEAK sub-term (``CostConfig.w_velocity_peak``) --------
    #: Same shape as ``torque_peak``: smooth-max ReLU priced, hard max for AL.
    velocity_peak_start: int = 0
    velocity_peak_ramp: int = 1_000

    # ---- the collision PEAK sub-term (``CostConfig.w_collision_peak``) ------
    #: Sum of ReLU penetration; AL integrates the max.  Same ramp as torque.
    collision_peak_start: int = 0
    collision_peak_ramp: int = 1_000

    # ---- the joint-box WAYPOINT ReLU (``CostConfig.w_joint_box_peak``) ------
    #: Sum of ReLU on each decoder (waypoint, joint) against the hard box.
    #: Dense gradient; the priced loss scales with violation count.  The AL
    #: group still integrates the horizon max.  ``lam_joint_box_peak`` starts at
    #: ``lam_init`` on resume.
    joint_box_peak_start: int = 0
    joint_box_peak_ramp: int = 1_000

    # ---- the WAIST-UPRIGHT pair (section 7d) --------------------------------
    #: ``CostConfig.waist_pr_box`` and ``CostConfig.w_waist_upright`` share one
    #: ramp, driving ``weights["waist_box"]`` and ``weights["posture_upright"]``.
    #: The box half is NOT a price on a penalty -- it is a linear blend on the
    #: joint box's own half-width -- so ramping it is the only way to introduce
    #: the constraint without stepping ``lam_joint_box``: dropping a fine-tune's
    #: waist rows from +-0.52 to +-0.20 in one step makes most of the replay
    #: buffer instantly infeasible, and ``rho["joint_box"] = 100`` per rad of
    #: mean excess then takes the multiplier straight to ``lam_max``.  300 steps
    #: is ~3 AL updates at ``al_every = 100``, which is enough for the
    #: multiplier to track the constraint rather than chase it.
    waist_start: int = 0
    waist_ramp: int = 300

    # ---- the strike-masked EFFORT term (section 7d leg 2) -------------------
    #: ``CostConfig.w_effort`` ramps 0 -> 1 over this span into
    #: ``weights["effort"]``.  Unlike the waist box this one is a plain price on
    #: an objective term, so the ramp is only there to keep the optimizer from
    #: seeing a step change in the loss; there is no multiplier to shock.
    effort_start: int = 0
    effort_ramp: int = 300

    # ---- the IDLE-ARM pair (``w_idle_cylinder`` / ``w_idle_vel``) ------------
    #: Both idle terms share one ramp into ``weights["posture_idle"]``.  Like
    #: ``effort`` these are plain prices on objective terms with no multiplier to
    #: shock, so the ramp only exists to keep the loss continuous when a
    #: continuation switches the cylinder on.
    idle_start: int = 0
    idle_ramp: int = 300

    # ---- tip lateral ordering (``w_tip_crossover``) -------------------------
    #: Plain objective ramp into ``weights["tip_crossover"]``.
    tip_crossover_start: int = 0
    tip_crossover_ramp: int = 300

    # ---- torque physics: level A -> full RNEA (``CostConfig.torque_physics``)
    #: Step at which ``weights["torque_full_blend"]`` begins ramping 0 -> 1.
    #: ``start=0`` and ``ramp=0`` (defaults) leave blend at 1 (full RNEA).
    #: ``start < 0`` holds level A forever.
    torque_full_start: int = 0
    torque_full_ramp: int = 0

    # ---- the plan's stage-A unlock -----------------------------------------
    #: When True, ``alpha`` is HELD at 0 until :func:`stage_a_pass` has fired
    #: once, so the limit hinges cannot ramp before the model can reach a drum
    #: collision-free.  Off by default: a gate that never opens freezes the
    #: curriculum forever, which on a short run is the worse failure.  A run
    #: that resumes from a checkpoint which already passed stage A must seed
    #: the gate open (``train_phase2(stage_a_init=True)``) or the first
    #: ``eval_every`` steps run at ``lo_limits``.
    gate_stage_a: bool = False

    # ---- per-JOINT exemption from the limit ramp ----------------------------
    #: Joints whose ``torque`` / ``velocity`` / ``joint_box`` hinge is priced at
    #: FULL strength from step 0 while every other joint follows ``alpha``'s
    #: ramp.  Empty tuple = the old, uniformly ramped behaviour.
    #:
    #: WHY, measured on ``runs/phase2_19m_ctxv2``: the ramp deliberately opens a
    #: window in which the limit hinges cost ``lo_limits`` (5 %), and the model
    #: finds it.  Over steps 1 k -> 10 k the waist PITCH/ROLL excursion grew
    #: 0.24 -> 0.64 rad and the pitch/roll RATE 1.8 -> 3.0 rad/s, while the
    #: PRICED waist-yaw rate FELL 2.5 -> 1.1 rad/s.  The momentum pumping did
    #: not stop; it migrated into the degrees of freedom the curriculum had left
    #: unpriced.
    #:
    #: For these two joints that window is a lie.  RNEA, static, ready pose plus
    #: lean, against the 25 N*m waist-pitch rating: holding UPRIGHT already costs
    #: 13.1 N*m (52 % of rating), a legal box-edge lean of 0.25 rad costs
    #: 20.1 N*m (80 %), and the observed 0.64 rad blow-through costs 28.4 N*m --
    #: statically OVER rating, before any acceleration.  The 7x torque peaks
    #: those plans carry are fiction the plant clamps.
    #:
    #: The ramp's own rationale -- let the decoder learn to REACH before the
    #: limits close, which the 2026-08-13 divergence taught -- is an argument
    #: about the ARMS, whose actuators have headroom.  A waist pitch/roll
    #: actuator that is at 52 % of rating just standing there has no slack to
    #: lend, so the honest curriculum tells it the truth from step 0.
    #:
    #: AL SEMANTICS ARE UNTOUCHED.  The multipliers integrate the RAW violations
    #: (``viol["torque"]`` / ``["velocity"]`` / ``["joint_box"]``, measured
    #: against the HARD limits and detached), and the ramp never scaled those --
    #: it only ever scaled the PENALTY.  All that changes here is that the
    #: penalty's scale is per joint instead of per group: ``v_torque`` /
    #: ``v_velocity`` / ``v_joint_box`` and every multiplier update are
    #: bit-identical with the exemption on or off (``tests/test_pr_full_limits``).
    full_limit_joints: tuple[str, ...] = ("waist_pitch_joint", "waist_roll_joint")

    def alpha(self, step: int, stage_a: bool = True) -> float:
        if self.gate_stage_a and not stage_a:
            return 0.0
        span = max(int(self.ramp_end) - int(self.ramp_start), 1)
        return _smoothstep((int(step) - int(self.ramp_start)) / span)

    def torque_peak_alpha(self, step: int) -> float:
        span = max(int(self.torque_peak_ramp), 1)
        return _smoothstep((int(step) - int(self.torque_peak_start)) / span)

    def velocity_peak_alpha(self, step: int) -> float:
        span = max(int(self.velocity_peak_ramp), 1)
        return _smoothstep((int(step) - int(self.velocity_peak_start)) / span)

    def collision_peak_alpha(self, step: int) -> float:
        span = max(int(self.collision_peak_ramp), 1)
        return _smoothstep((int(step) - int(self.collision_peak_start)) / span)

    def joint_box_peak_alpha(self, step: int) -> float:
        span = max(int(self.joint_box_peak_ramp), 1)
        return _smoothstep((int(step) - int(self.joint_box_peak_start)) / span)

    def waist_alpha(self, step: int) -> float:
        span = max(int(self.waist_ramp), 1)
        return _smoothstep((int(step) - int(self.waist_start)) / span)

    def effort_alpha(self, step: int) -> float:
        span = max(int(self.effort_ramp), 1)
        return _smoothstep((int(step) - int(self.effort_start)) / span)

    def idle_alpha(self, step: int) -> float:
        span = max(int(self.idle_ramp), 1)
        return _smoothstep((int(step) - int(self.idle_start)) / span)

    def tip_crossover_alpha(self, step: int) -> float:
        span = max(int(self.tip_crossover_ramp), 1)
        return _smoothstep((int(step) - int(self.tip_crossover_start)) / span)

    def torque_full_blend(self, step: int) -> float:
        """Blend into full coupled RNEA for the torque hinge (0 = level A only)."""
        start = int(self.torque_full_start)
        if start < 0:
            return 0.0
        ramp = int(self.torque_full_ramp)
        if start == 0 and ramp <= 0:
            return 1.0
        span = max(ramp, 1)
        return _smoothstep((int(step) - start) / span)

    def limit_alpha(self, step: int, stage_a: bool = True) -> float:
        """The scalar ramp on the three limit hinges (``lo_limits`` -> 1.0)."""
        return self.lo_limits + (1.0 - self.lo_limits) * self.alpha(step, stage_a)

    def limit_alpha_vector(
        self, step: int, joint_names, stage_a: bool = True
    ) -> np.ndarray | None:
        """Per-joint limit-hinge alpha, or ``None`` when nothing is exempt.

        ``1.0`` at every joint named in :attr:`full_limit_joints`,
        :meth:`limit_alpha` everywhere else.  ``None`` (no name matched, or the
        tuple is empty) means the caller should leave ``weights["limit_alpha"]``
        unset, which is the bit-exact pre-exemption path through the cost.
        """
        names = tuple(joint_names)
        full = set(self.full_limit_joints)
        if not full.intersection(names):
            return None
        a = self.limit_alpha(step, stage_a)
        return np.array([1.0 if n in full else a for n in names], dtype=np.float64)

    def multipliers(self, step: int, stage_a: bool = True) -> dict[str, float]:
        a = self.alpha(step, stage_a)
        lim = self.limit_alpha(step, stage_a)
        strike = self.lo_strike + (1.0 - self.lo_strike) * a
        waist = self.waist_alpha(step)
        return {
            "waist_box": waist,
            "posture_upright": waist,
            "waist_aim": waist,
            "waist_yaw_vel": waist,
            "waist_pr_vel": waist,
            "waist_pr_terminal": waist,
            "effort": self.effort_alpha(step),
            "posture_idle": self.idle_alpha(step),
            "tip_crossover": self.tip_crossover_alpha(step),
            "collision": self.w_collision,
            "posture": self.w_posture,
            "smooth": self.w_smooth,
            "terminal": 1.0,
            "terminal_region": self.w_terminal_region,
            "terminal_velocity": strike,
            "terminal_elev": strike,
            # ``lim`` is the warning (softplus) ramp, consumed as
            # ``limit_alpha`` on hinge elements -- :func:`_wtensors` does
            # not fold it into λ.
            "torque": lim,
            "velocity": lim,
            "joint_box": lim,
            "torque_peak": self.torque_peak_alpha(step),
            "velocity_peak": self.velocity_peak_alpha(step),
            "collision_peak": self.collision_peak_alpha(step),
            "joint_box_peak": self.joint_box_peak_alpha(step),
            "waist_pr_band": 1.0,
        }

    def snap_full(self) -> "WeightSchedule":
        """All curriculum ramps already complete at step 0.

        Same trick as ``--waist-full`` (``start=-1, ramp=1`` so
        ``_smoothstep((0 - (-1)) / 1) = 1``), applied to every ramped group
        including the limit/strike smoothstep.  Use when a warm start already
        trained through the ramps and the new run should keep those prices at
        ceiling while optimizer / AL / roller restart.
        """
        return replace(
            self,
            ramp_start=-1,
            ramp_end=0,
            torque_peak_start=-1,
            torque_peak_ramp=1,
            velocity_peak_start=-1,
            velocity_peak_ramp=1,
            collision_peak_start=-1,
            collision_peak_ramp=1,
            joint_box_peak_start=-1,
            joint_box_peak_ramp=1,
            waist_start=-1,
            waist_ramp=1,
            effort_start=-1,
            effort_ramp=1,
            idle_start=-1,
            idle_ramp=1,
            tip_crossover_start=-1,
            tip_crossover_ramp=1,
        )


def default_weight_schedule() -> WeightSchedule:
    """Phase-2 default: all cost weights at ceiling from step 0.

    The augmented Lagrangian adjusts λ; opening a low-weight curriculum window
    (especially on waist rate vs limit hinges) was measured to let momentum pump
    through unpriced P/R DoFs.  Aux losses (BPTT, suffix) keep their own
    phase-in schedules on :class:`Phase2Config`.
    """
    return WeightSchedule().snap_full()


def phase2_cost_config(**kw) -> CostConfig:
    """``CostConfig`` rebalanced for *training*, measured at the phase-1 model.

    ``CostConfig``'s own defaults are the section-3 validation weights and are
    left alone (``tests/test_cost_validation.py`` pins them).  They are not,
    however, a usable training objective: every term is a mean over its valid
    elements, and the element COUNTS differ by three orders of magnitude between
    groups.  On the phase-1 checkpoint, at the declared weights:

    ======== ========== ==========================================
    group     value      what it actually is
    ======== ========== ==========================================
    smooth   150 - 1230  jerk integral ~1e9 / (H x 17)
    terminal 144 -  216  80 mm miss + 2.4 m/s impact-speed error
    torque     0.7 -  1  ...while ``torque_excess_max`` is 45-102 N.m
    collision  0.046     ...while ``min_sdf`` is **-44 mm**
    ======== ========== ==========================================

    The collision hinge is averaged over ``S x P x D x 2`` = ~2 000 point-drum
    pairs of which at most a handful are ever violated, so a real 5 cm
    penetration reduces to 5e-5.  Multiplying it back up is not a hack, it is
    undoing the averaging; the augmented Lagrangian then works from a sane
    starting point instead of spending a third of the run climbing to one.

    Weights here are chosen so the groups start within ~1 decade of each other
    on that same measurement.  Everything else is ``CostConfig``'s default.
    """
    base = dict(
        # collision: undo the per-pair mean (see above)
        w_collision_drum=1.0e5, w_collision_under=1.0e5, w_collision_self=1.0e5,
        # Limit families, unit-normalized so a characteristic *element-mean*
        # hinge excess costs the same at λ=1 (AL λ still differentiates).
        # Characteristic: 1 N·m torque, 0.5 rad/s velocity, 0.05 rad box.
        # Previous w_joint_box=50 made a 0.05 rad mean excess 8× cheaper
        # than 1 N·m of torque, so SGD dumped through the box to buy a
        # waist lever.  w_* / char is the dimensionless price of "one unit".
        w_torque=20.0, w_velocity=40.0, w_joint_box=400.0,
        # ...and the SOFT-MAX half of the torque term, which is what the
        # harness's zero-tolerance max gate actually checks.  ``w_torque`` alone
        # prices a MEAN over ~H x 17 = 850 elements, of which a failing case
        # exceeds only 0.27 % at the median (p90 0.51 %, worst 1.5 %) by a
        # median 0.87 N.m -- so the term that decides acceptance moves the loss
        # by ~0.3 %.  Measured over the 243 recorded expert cases at lambda = 1,
        # ``raw["torque_peak"]`` separates the gate 26x (0.30 on PASS vs 7.78 on
        # FAIL) where the mean separates it 3.4x.  At 2.0 the peak half is 0.65
        # on a clean case -- between the mean half (0.16) and terminal (1.91),
        # i.e. inside a third of a decade of the live terms -- and ~15.6 on a
        # case that fails the gate.  Ramped in by ``WeightSchedule`` and priced
        # by its OWN AL multiplier (``rho`` two decades below ``torque``'s,
        # below), not by ``lam_torque``: section 7b's two continuations show
        # that raising this weight against a multiplier calibrated on the mean
        # is what destabilizes the objective.
        w_torque_peak=2.0,
        #: Sum of per-segment ReLU velocity excess (dense).  At 2 a single
        #: 0.5 rad/s spike costs 1 at λ=1.
        w_velocity_peak=2.0,
        #: Sum of per-(waypoint,joint) ReLU on the hard box (dense grad, scales
        #: with violation count).  At 40 a single 0.05 rad violation on one plan
        #: row costs 2 at λ=1; twelve identical rows cost 24.
        w_joint_box_peak=40.0,
        # default, which otherwise IS the objective at bucket 25.  The JERK
        # weight is restored: a one-knot zig-zag is a second difference, and it
        # is exactly what produces the single-element torque spikes above, so
        # dividing it by 10 removed the cheapest lever on the binding gate.
        # This breaks trajopt's 10:1 acc:jerk ratio deliberately -- see
        # DECISIONS.
        w_smooth_acc=1.0e-4, w_smooth_jerk=1.0e-4,
        # posture: a real preference rather than a rounding error
        w_posture_mid=0.1, w_elbow_down=10.0,
        # Continuous compact active-arm mass proxy.  Unlike the original
        # squared-ReLU form, every active pose gets a gentle downward/inward
        # gradient.  Its much larger raw gradient requires a correspondingly
        # smaller outer weight.  On the step-13,500 actor, unit weight has
        # parameter-gradient norm 6.74 vs 5,527 for the rest of the objective;
        # 15 therefore allocates about 1.8% of the gradient budget.  Start
        # gentle and raise it only if the logged mean height/radius stay flat.
        w_arm_compact=15.0,
        arm_compact_mode="linear",
        arm_compact_height_max=0.28,
        arm_compact_radius_max=0.28,
        # IDLE ARM.  Both are EXACTLY zero -- no term, no gradient -- until a
        # cylinder is installed with ``FusedCost.set_idle_cylinder``, so carrying
        # them in the training config costs nothing on a run that does not use
        # one.  Prefer ``w_idle_cylinder``; fall back to ``w_idle_home`` only if
        # a mid-migration CostConfig still exposes the old field.
        **(
            {"w_idle_cylinder": 1.0}
            if "w_idle_cylinder" in CostConfig.__dataclass_fields__
            else {"w_idle_home": 1.0}
        ),
        w_idle_vel=0.02,
        # Moderate in-plane pull toward the AIM (5 mm deadband, not AL-scaled)
        # so the legal-region hinge cannot improve torque/posture by parking on
        # the rim.  This remains far below the terminal AL term on an actual
        # miss, but supplies a useful precision gradient after entering the
        # 20 mm MPC acceptance collar.  CostConfig default remains OFF;
        # this is training-only.
        w_terminal_aim=40.0,
        terminal_aim_deadband_m=0.005,
        # Decoder P/R vs stopgrad(encoder SP).  CostConfig default is 0 so
        # validation stays bit-identical; training prices the AL group (λ is
        # then the price).  CLI --w-waist-pr-band 0 for A/B.
        w_waist_pr_band=1.0,
    )
    base.update(kw)
    return CostConfig(**base)


def phase2_al_config(
    rho_scale: float = 1.0,
    rho_terminal: float | None = None,
    **kw,
) -> ALConfig:
    """Multiplier dynamics matched to :func:`phase2_cost_config`'s scales.

    ``rho`` is the dual step (Ki) per unit of *mean raw violation* in that
    group's own units -- it is NOT the PHR quadratic coefficient.  The base
    cost weights are already O(10-100), so the multipliers only have to move
    by ones.      ``lam_max`` stays 100 as a safety clip (do not lower it on
    resume: a checkpoint sitting at 100 would shock).  Slack ``target`` keeps
    the integrator from winding up on a residual the gate will never see as
    zero; PID (``kp`` / ``kd``) cheapens a feasible batch immediately.  A
    growth-frozen PHR ``pen`` (``pen_grow = 1``) adds a small quadratic on the
    live residual; ``pen_shrink = 1.5`` still makes this decay-only once a
    group is feasible.  Bertsekas growth is off because chronic groups
    (collision, joint_box_waist) never improve by ``improve_tol`` and
    compounded 1.5^n in the first segment resume.
    """
    rho_tau = float(os.environ.get("AMORTIZED_RHO_TORQUE", 0.5)) * rho_scale
    # Dual residual is max over time×joints in the class (not mean-of-max).
    rho_fam = {
        "torque": rho_tau,               # N·m, class max ReLU
        "velocity": 0.5 * rho_scale,     # rad/s, class max ReLU
        "joint_box": 15.0 * rho_scale,   # rad, class max ReLU
    }
    rho = {
        limit_group_name(fam, seg): rho_fam[fam]
        for fam in LIMIT_FAMILIES
        for seg in ACTUATOR_SEGMENT_NAMES
    }
    rho["collision"] = 5.0 * rho_scale
    if rho_terminal is None:
        rho_terminal = float(os.environ.get("AMORTIZED_RHO_TERMINAL", "10.0"))
    rho["terminal"] = float(rho_terminal) * rho_scale
    rho["torque_peak"] = float(os.environ.get("AMORTIZED_RHO_TORQUE", 0.5)) * 0.02 * rho_scale
    rho["velocity_peak"] = 0.5 * 0.02 * rho_scale
    rho["collision_peak"] = 0.5 * rho_scale
    rho["joint_box_peak"] = 15.0 * 0.02 * rho_scale
    rho["waist_pr_band"] = 1.0 * rho_scale
    lam_max = {g: 100.0 for g in rho}
    tau_max = float(os.environ.get("AMORTIZED_LAM_TORQUE_MAX", 20.0))
    _tau_waist = os.environ.get("AMORTIZED_LAM_TORQUE_WAIST_MAX", "").strip()
    tau_waist_max = float(_tau_waist) if _tau_waist else tau_max
    for seg in ACTUATOR_SEGMENT_NAMES:
        cap = tau_waist_max if seg == "waist" else tau_max
        lam_max[limit_group_name("torque", seg)] = cap
    lam_max["torque_peak"] = tau_max
    target_fam = {
        "torque": 0.05,          # N·m class max
        "velocity": 0.05,        # rad/s class max
        "joint_box": 0.02,       # rad; the 0.012 waist floor sits under this
    }
    target = {
        limit_group_name(fam, seg): target_fam[fam]
        for fam in LIMIT_FAMILIES
        for seg in ACTUATOR_SEGMENT_NAMES
    }
    target["collision"] = 1.0e-3
    target["terminal"] = 0.01
    target["torque_peak"] = 0.0
    target["velocity_peak"] = 0.0
    target["collision_peak"] = 1.0e-3
    target["joint_box_peak"] = 0.0
    target["waist_pr_band"] = 0.0
    kp = dict(rho)
    kd = {g: 0.25 * rho[g] for g in rho}
    # Growth-frozen, decay-only PHR (pen_grow=1, pen_shrink=1.5).  Bertsekas
    # growth on a chronic residual is what took the first segment resume
    # 4k → 66k in 1.7k steps: collision λ was 23 (below the cap, so the at-cap
    # freeze did not apply) and joint_box_waist V~0.16 with pen=1e4 already
    # added ~130, then 1.5^n took it to 4e6.
    # Quadratic coefficient is sized so 0.5 * pen * V_typ² ~ 1.
    # PHR on torque stacked ~5 % of decoder grad at λ≈cap with no gate benefit.
    # ``AMORTIZED_PEN_TORQUE=1`` restores the pre-2026-09 recipe for A/B.
    _pen_torque = os.environ.get("AMORTIZED_PEN_TORQUE", "0").strip().lower()
    pen_torque_on = _pen_torque not in ("", "0", "false", "off", "no")
    pen_fam = {
        "torque": 2.0 if pen_torque_on else 0.0,
        "velocity": 10.0,        # V~0.5 rad/s
        "joint_box": 800.0,      # V~0.05 rad → 0.5*pen*V² ~ 1, matching torque
    }
    pen = {
        limit_group_name(fam, seg): pen_fam[fam]
        for fam in LIMIT_FAMILIES
        for seg in ACTUATOR_SEGMENT_NAMES
    }
    pen["collision"] = 1.0e4
    pen["terminal"] = 1.0e3
    pen["torque_peak"] = 0.5 if pen_torque_on else 0.0
    pen["velocity_peak"] = 0.5
    pen["collision_peak"] = 1.0e3
    pen["joint_box_peak"] = 800.0
    pen["waist_pr_band"] = 2.0
    pen_max_fam = {
        "torque": 200.0,
        "velocity": 1.0e4,
        "joint_box": 1.0e3,
    }
    pen_max = {
        limit_group_name(fam, seg): pen_max_fam[fam]
        for fam in LIMIT_FAMILIES
        for seg in ACTUATOR_SEGMENT_NAMES
    }
    pen_max["collision"] = 1.0e5
    pen_max["terminal"] = 1.0e4
    pen_max["torque_peak"] = 50.0
    pen_max["velocity_peak"] = 50.0
    pen_max["collision_peak"] = 1.0e4
    pen_max["joint_box_peak"] = 1.0e3
    pen_max["waist_pr_band"] = 200.0
    d = dict(
        lam_init=1.0, rho=rho, lam_min=0.1, lam_max=lam_max, ema=0.5,
        decay=0.999, feasible_decay=0.95,
        target=target, kp=kp, kd=kd,
        pen=pen, pen_min=1.0e-6, pen_max=pen_max,
        pen_grow=1.0, pen_shrink=1.5, pen_patience=10, improve_tol=0.9,
    )
    d.update(kw)
    return ALConfig(**d)


@dataclass
class BounceConfig:
    """Placeholder contact model for the chained two-hit stage.

    Reflects the tip-normal velocity component with a randomized restitution and
    leaves the tangential component alone, then adds Gaussian position/velocity
    noise.  This is deliberately crude: it stands in for Isaac's contact solver
    until the RL fine-tune replaces it, and its parameters are randomized so the
    chained stage cannot overfit one particular bounce.
    """

    restitution: tuple[float, float] = (0.3, 0.6)
    #: Gaussian sd of extra tip-velocity noise, m/s, isotropic.
    tip_vel_noise: float = 0.15
    #: Gaussian sd on the post-hit joint state.
    q_noise: float = 0.01     # rad
    qd_noise: float = 0.05    # rad/s


# LineSearchConfig: single source in :mod:`amortized_opt.line_search`
# (re-exported from this module for CLI convenience).


# =============================================================== config ======
@dataclass
class Phase2Config:
    """Everything the merged phase-2/3 run needs."""

    max_steps: int = 60_000
    batch_size: int = 512
    #: Number of actor minibatches sampled together from the timing curriculum.
    #: A value of 4 with ``batch_size=1024`` draws/shuffles 4096 task rows,
    #: then consumes four non-overlapping 1024-row slices.  Timing proposals
    #: and critic labels remain live per actor step; only the underlying random
    #: state/goal/MPC task draw is cached.
    actor_superbatch_factor: int = 1
    #: Waypoint budget per step.  Memory in the cost scales with ``B x H``
    #: (the FK/SDF pass is over ``B x (H+1)`` poses), so a 100-waypoint bucket
    #: at the same batch as a 25-waypoint one is 4x the activations.  The batch
    #: is shrunk for the long buckets to hold ``B x H`` roughly constant --
    #: which also keeps the *gradient* per step comparable across buckets,
    #: since every term is a mean over valid elements.
    token_budget: int = 512 * 50
    lr: float = 1.0e-4
    #: Denominator of the cosine, in steps.  ``None`` = ``max_steps``, which is
    #: the historical behaviour.  Set it explicitly when the LOOP budget and the
    #: LR SCHEDULE must differ -- a short continuation of a long run wants the
    #: predecessor's LR trajectory (section 7b: the LR, not the objective, is
    #: what a continuation must be tuned on) but a hard step cap of its own.
    #: Only affects ``lr_at``; ``bptt_frac`` still keys off ``max_steps``.
    #: When ``line_search`` is on, cosine ownership is disabled in the train
    #: loop (plan §7) — selector owns effective LR.
    lr_horizon: int | None = None
    min_lr_frac: float = 0.05
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 300
    grad_clip: float = 1.0
    #: Per-cost-term decoder grad-norm probe (``grad_<term>`` in metrics.jsonl).
    #: 0 = off.  Each probe is one extra forward + len(TERM_KEYS) ``autograd.grad``
    #: calls, so keep this ≥ ``log_every``.
    grad_term_every: int = 250
    #: Rows used by the cost-space torque Taylor/Hessian probe attached to each
    #: grad-term diagnostic.  0 disables it.
    torque_curvature_rows: int = 8
    #: ``""`` = standard backward; ``al_gradnorm`` / ``al_pcgrad`` balance in
    #: parameter space; ``al_pcgrad_output`` balances at the trajectory and
    #: traverses the transformer once (see :mod:`amortized_opt.grad_balance`).
    #: Forced to ``""`` in ``__post_init__`` when ``line_search`` is True
    #: (plan §1–2).  ``train_phase2`` must also clear it **before**
    #: ``make_cost_step`` — compiled graphs bake balance arity (plan §14).
    grad_balance: str = ""
    grad_balance_tau_torque_cap: float = 0.30
    grad_balance_torque_gamma: float = 0.35
    grad_balance_beta: float = 0.5
    grad_balance_shape_floor: float = 1.0
    grad_balance_waist_floor: float = 2.0
    #: Exact output-space group VJPs every N visits to each horizon bucket.
    #: Intervening visits use one scalar backward with the cached norm EMA.
    grad_balance_refresh: int = 1
    grad_balance_norm_ema: float = 0.9
    #: Extra scale on the GradNorm-combined torque group contribution in
    #: output/parameter space (1 = unchanged).  ``0.1`` ≈ tenth the torque
    #: step without touching other groups' LRs.
    torque_grad_scale: float = 1.0
    #: After ``total.backward()``, cap ``||g_torque|| / ||g||`` (0=off, 1=no-op).
    #: Uses ``terms['torque']`` (hinge + AL + PHR).  Works with line search.
    torque_grad_cap: float = 0.0
    #: Re-measure the exact torque-gradient decomposition every N outer steps.
    #: Intervening steps apply an EMA-cached scalar in the loss and avoid the
    #: second parameter VJP. One preserves the exact legacy path.
    torque_grad_cap_refresh: int = 1
    torque_grad_cap_scale_ema: float = 0.8

    #: SaLSa-style line search (``docs/parameter_line_search_plan``).  When
    #: True, ``grad_balance`` is forced empty (true ``total.backward()``);
    #: selector owns LR (cosine gated — plan §7).  Implementers: force
    #: surgery off before ``make_cost_step`` and follow commit-gate /
    #: ``viol_acc`` policy in plan §14 — do not clear balance only in-loop.
    line_search: bool = False
    line_search_cfg: LineSearchConfig = field(default_factory=LineSearchConfig)
    #: Perturb the frozen terminal-head initialization seen by fresh rows.
    #: Noise uses the pool's phase-1-matched OU + white law, then a cubic
    #: envelope whose value and slope are zero at the measured initial knot.
    terminal_init_noise: bool = False
    terminal_init_noise_ramp_steps: int = 5
    #: Line-search direction backend.  ``lbfgs`` uses limited-memory BFGS on
    #: the replay-frozen batch (history updated after each executed step).
    optimizer: str = "adamw"
    lbfgs_history: int = 10

    amp_dtype: str = "bfloat16"
    compile_mode: str | None = "default"
    #: Which units go through Inductor.  The PyTorch 2.11 environment compiles
    #: the transformer cleanly.  Compiled physics forward is faster, but its
    #: backward has produced non-finite group gradients on finite-cost batches;
    #: keep ``compile_cost=False`` until that derivative bug is fixed.
    compile_model: bool = True
    compile_cost: bool = False
    compile_aux: bool = False
    tf32: bool = True

    # ---- objective ----
    w_cost: float = 1.0
    #: The self-supervised aux heads sit around 1.0 while the cost sits around
    #: 300, so this is the weight that makes them ~2 % of the objective.
    w_aux: float = 5.0
    #: WHICH torque the aux torque head is regressed onto.  ``False`` (default,
    #: and what every run before 2026-08-16 did) keeps the RNEA segment max;
    #: ``True`` switches the label to the DEMANDED PD tracking torque
    #: ``|tau_rnea + tau_fb|`` -- the same array ``w_track_torque`` prices, from
    #: the same :func:`~amortized_opt.pd_sim.tracking_torque_knots`.  Turn it on
    #: with the tracking-torque weights: otherwise the head advertises an
    #: open-loop number the objective no longer charges for.  Opt-in because
    #: flipping it mid-run silently redefines a live objective.
    aux_torque_tracking: bool = False
    #: Temporal consistency is a Huber in radians (~0.2 at the phase-1 model).
    w_tc: float = 50.0
    tc_frac: float = 0.25
    #: Cheap detached temporal-consistency target on every sampled MPC replay
    #: row.  The replay ``init`` is the previous plan shifted by one tick, so
    #: tying the current output to it on the non-padded overlap gives dense TC
    #: supervision without another solve or plant step.  Fresh/quintic rows are
    #: explicitly masked out.  Default off for backward-compatible experiments;
    #: the rhythm Bellman launcher enables it at the same scale as ``w_tc``.
    w_mpc_init_tie: float = 0.0
    #: MPC replay consistency for waist yaw only.  Unlike the all-joint tie,
    #: this is not diluted over 17 output columns: current waypoint ``j`` is
    #: compared with the stored parent at the same absolute time.  It therefore
    #: preserves the first plan's valid initial-velocity transient while
    #: directly penalizing the closed-loop yaw sign flip.
    w_waist_yaw_mpc_tie: float = 0.0
    w_bptt: float = 1.0
    bptt_start_frac: float = 0.67      # ramp in over the final third
    bptt_max_frac: float = 0.25        # fraction of steps that also do a chain
    bptt_batch: int = 128
    #: Total weight-shared decoder applications in the two-solve BPTT chain.
    #: Must be even: half refine hit 1, then half refine hit 2 after the
    #: differentiable plant/bounce transition.  The encoder outputs are reused
    #: within each solve and carry no graph when encoder parameters are frozen.
    bptt_decoder_passes: int = 2

    #: VERIFIED SUFFIX DISTILLATION (:func:`make_suffix_step`).  Parent plans
    #: passing the conservative Bellman admission gate are detached into a
    #: short FIFO ring.  That gate permits half the terminal, collision, joint,
    #: and torque violation formerly allowed by the critic MPC parent gate.
    #: A later
    #: update re-solves random tails and trains only the Bellman identity to
    #: the verified parent -- it does not run a second FusedCost on the child.
    #: ``suffix_frac = 0`` remains the backward-compatible off switch.
    #:
    #: Retained for checkpoint/CLI compatibility.  The verified replay path
    #: deliberately ignores this former child-FusedCost coefficient.
    w_suffix_cost: float = 1.0
    #: Huber (beta 0.02 rad, like TC) between the child solve and the detached
    #: verified parent tail on their overlap.
    w_suffix_tie: float = 50.0
    suffix_frac: float = 0.0
    #: Fixed next-update distillation batch.  The ring samples without
    #: replacement when full enough and reuses parents at different offsets
    #: while warming up.
    suffix_replay_batch: int = 64
    #: Global FIFO capacity across horizon buckets.  Small by design so the
    #: teacher remains close to the live policy.
    suffix_replay_capacity: int = 256
    #: Do not self-distil until this many unique verified parents exist in the
    #: selected horizon.  The effective suffix coefficient ramps linearly from
    #: this threshold to ``suffix_replay_batch``.
    suffix_replay_min_parents: int = 16
    #: Maximum rows per differentiable suffix decoder call.  Zero evaluates
    #: the full suffix subset at once.  Chunking preserves the exact weighted
    #: mean/labels while reducing peak memory for large actor minibatches.
    suffix_microbatch: int = 0
    #: Shared remaining-TTH mixture for Bellman and critic-MPC suffix draws:
    #: an asymmetric exponential peak at the freeze boundary, broad uniform
    #: support from that boundary through the parent tail, and a small
    #: below-freeze geometric component for generalization.
    terminal_suffix_peak_frac: float = 0.55
    terminal_suffix_long_frac: float = 0.42
    terminal_suffix_low_frac: float = 0.03
    terminal_suffix_peak_wp: int = 5
    terminal_suffix_tail_tau: float = 8.0
    #: Smoothstep ramp of both suffix weights over the first N steps -- early
    #: on the identity ties two garbage plans, which is coupling paid for
    #: nothing until the cost loss has shaped the output.
    suffix_warmup: int = 1_000

    #: TERMINAL DLS LOOKAHEAD.  On a deterministic subset of rows, take a
    #: detached damped-least-squares step at each active hand's strike knot,
    #: spread that arm-only correction backward through the final approach,
    #: and evaluate the ordinary full physics cost on the corrected path.  A
    #: straight-through view sends that better terminal gradient back into the
    #: actor; the Huber tie makes the uncorrected actor learn to internalize the
    #: correction.  This is deliberately not an inference-time projector.
    #:
    #: ``dls_terminal_frac=0`` is an exact off switch.  Rows are evenly spaced
    #: rather than randomly drawn so a line-search probe replays precisely the
    #: same scalar objective without another RNG payload.
    dls_terminal_frac: float = 0.0
    w_dls_terminal_cost: float = 0.0
    w_dls_terminal_tie: float = 0.0
    dls_terminal_steps: int = 1
    dls_terminal_window: int = 16
    dls_terminal_damping: float = 0.05
    dls_terminal_max_step_rad: float = 0.35
    #: Alternative to the arm-only DLS teacher.  The curvature teacher takes
    #: the gradient of the *full* FusedCost in waypoint space, then solves a
    #: positive-definite model made from the exact spline smoothness Hessian
    #: and contact position/velocity Gauss--Newton blocks.  The ordinary actor
    #: Adam/SaLSa update and the full-cost lookahead remain unchanged.
    terminal_lookahead_mode: str = "dls"  # "dls" | "curvature"
    curvature_damping: float = 1.0
    curvature_position_weight: float = 40.0
    curvature_velocity_weight: float = 20.0
    curvature_max_step_rad: float = 0.15
    #: Replace the trajectory VJP on this deterministic fraction of rows with
    #: a norm-matched analytic-curvature direction. Unlike the lookahead
    #: teacher, this changes the backward direction but not the scalar loss.
    curvature_direct_frac: float = 0.0
    curvature_direct_mix: float = 1.0

    #: Std-dev, rad, of Gaussian jitter added to the ``yaw_now`` conditioning
    #: token at batch time (the stored ``aim_yaw`` token is left alone).  The
    #: waist-sp head must then hold its set-point AT THE AIM for every jittered
    #: yaw -- the tether (``res = sp - aim`` by construction) prices any
    #: learned sensitivity to ``yaw_now`` directly.  That sensitivity is the
    #: loop that ran away in Isaac (2026-08-14: the head chased the measured
    #: yaw at ~0.45 deg/deg and the torso ratcheted to -95 deg against a
    #: +37 deg aim; ``isaac_rl.mpc.MPCConfig.yaw_cond`` is the deployment-side
    #: half of the fix).  Training never taught anchoring because ``yaw_now``
    #: only ever appeared exactly on the scenario manifold.  Drawn from its own
    #: RNG stream, so a jitter-off run sees an identical batch sequence.
    #: 0 = the pre-fix behaviour.
    yaw_now_jitter: float = 0.3

    #: THE LEARNED WAIST-YAW SET-POINT (``docs/waist_sp_head_plan.md``).  When
    #: True (requires ``ModelConfig.waist_sp_head``), the ``w_waist_aim`` term's
    #: reference is the encoder head's set-point broadcast over the horizon
    #: instead of the pool's analytic ramp -- gradient then flows through the
    #: term into the head, and the anchor/trajectory pair walks to whatever yaw
    #: the cost prefers.  The BPTT / TC sub-losses keep ``waist_yaw_ref=None``
    #: (their ``waist_aim`` term is silent); the main loss is where the anchor
    #: lives.
    waist_sp_from_head: bool = False
    #: Tether on the head's residual, ``w * mean(res^2)`` with
    #: ``res = sp - aim_yaw`` by construction (rad^-2).  Bounds how far the
    #: learned law may walk from the analytic one -- the pool's INITS still ramp
    #: to the analytic aim, so an unbounded head would part ways with the very
    #: trajectories it anchors.  Unramped: it is exactly zero at init.  The
    #: eval line reports mean |res|; if it saturates against this weight, the
    #: tether is the binding constraint and should be lowered.
    w_waist_sp_tether: float = 0.0

    #: THE LEARNED WAIST PITCH/ROLL SET-POINT (``docs/waist_pr_sp_head_plan.md``).
    #: When True (requires ``ModelConfig.waist_pr_sp_head``), the upright well
    #: tracks ``enc.waist_pr_sp`` broadcast over the horizon instead of 0, and
    #: the quintic seed eases pitch/roll to that set-point (yaw already ramps
    #: to ``aim_yaw``).
    waist_pr_sp_from_head: bool = False
    #: Tether on the P/R set-point, ``w * mean(sp²)`` -- residual IS the
    #: set-point (base is upright).  Bounds lean; the attractive consistency
    #: fixed point is otherwise the URDF rail.  Unramped (zero at init).
    w_waist_pr_sp_tether: float = 0.0
    #: Hold the encode-time waist.  ``w * mean((sp_pr - stopgrad(q0_pr))²)``
    #: so the announced lean stays near the pose at encode, and the arms do
    #: the reaching.  ``q0`` is NOT an encoder input (that is the yaw_now
    #: runaway); it is only the loss target.  Unramped.  0 = off.
    w_waist_pr_sp_hold: float = 0.0
    #: Feasible-set hinge on that hold: ``w * mean(relu(|sp - clamp(q0)|
    #: - band))``.  Same ``band`` as the decoder ``waist_pr_band``
    #: (``CostConfig.waist_pr_band_rad``).  Equal tether+hold is ``sp =
    #: q0/2``, which is already past 0.15 on a rail ``q0``; this hinge is
    #: what makes ``|sp - q0| > band`` illegal so the decoder window
    #: contains the start.  Linear (not squared): a quadratic residual
    #: equilibrates outside the window against tether.  ``q0`` is stopgrad
    #: and URDF-clamped.  Unramped.  0 = off, bit-identical.
    w_waist_pr_sp_hold_band: float = 0.0
    #: GATED strike-knot distill of decoder waist P/R onto the encoder
    #: set-point.  ``w * mean_{gated}(‖sp - stopgrad(q_pr[hit])‖²)``.  The
    #: 0.15 band is the decoder's exploration window; this term is how a
    #: *good* lean inside that window walks the announcement.  Gate is the
    #: conjunction (active strike, on-head, in-box, no collision, in-band,
    #: |q_pr| inside URDF).  Default 0 = off, bit-identical except that
    #: upright already ``stopgrad``s ``sp``.  Start near tether (10) so a
    #: gated row beats hold+tether without walking the full 0.15 in one
    #: update.
    w_waist_pr_sp_distill: float = 0.0
    #: Closed-loop leaky EMA of the *clipped* P/R command as the upright-well
    #: reference (``StrikeContext.waist_pr_ref``).  Snapshot / 81-grid rows
    #: keep ``waist_pr_ref=None`` (dense well vs 0).  Conflicts with
    #: ``waist_pr_sp_from_head``: that head was the previous owner of the
    #: slot.  Off = bit-identical.
    waist_pr_ema: bool = False
    waist_pr_ema_alpha: float = 0.1
    waist_pr_ema_leak: float = 0.02
    #: On-head slack for that gate, metres.  ``terminal_region_dist`` is 0
    #: on the priced cylinder; 0.02 matches the 81-grid / frontier hit
    #: margin (``_FRONTIER_HIT_M``).
    pr_distill_hit_m: float = 0.02
    #: Same hold on the yaw head, ``w * mean((sp_yaw - stopgrad(q0_yaw))²)``.
    #: Default 0: yaw is supposed to walk toward ``aim_yaw``.  Turn on if the
    #: torso should hold heading too and let the arms absorb the new drum.
    w_waist_sp_hold: float = 0.0
    #: Cross-segment set-point smoothness.  ``w * mean((sp_new - sp_old)²)``
    #: where ``sp_old`` is this batch's encoding and ``sp_new`` is a second
    #: encode of the same rows after promoting the soonest current goal to
    #: ``future[0]`` -- the encoder-visible context at the next per-hand
    #: segment boundary.  Intra-segment terms (``w_waist_aim``,
    #: ``w_waist_yaw_vel``) cannot see that jump: closed loop re-encodes
    #: every hit, and on fast rhythms that is every 100-200 ms.  Both
    #: encodings keep gradient, so consecutive contexts walk toward a
    #: shared announcement.  Pitches P/R too when that head is on (added,
    #: same weight).  Unramped.  0 = off, bit-identical.
    w_waist_sp_smooth: float = 0.0

    # ---- zeroth-order SEARCH DISTILLATION for the set-point head -----------
    # Why this exists.  ``w_waist_aim`` prices ||traj_yaw - sp||, a CONSISTENCY
    # term: its gradient pulls the trajectory toward the set-point AND the
    # set-point toward the trajectory, and the only outcome grounding is the
    # (deliberately weak) tether above.  On the 2026-08-14 grooves run the pair
    # co-drifted -- ``waist_sp_res`` reached ~1.0 rad by step 5 000 and pinned
    # there, with ~90 mm of systematic contact miss.  The analytic gradient is
    # also structurally biased: the true objective for ``sp`` is the ENVELOPE
    # V(psi) = cost AFTER the arms re-plan around torso yaw psi, but the
    # gradient taken at a frozen trajectory mostly says "do not break the plan
    # you already have".
    #
    # The fix is zeroth order.  On a slice of each batch, evaluate V directly at
    # K candidate yaws (no gradient), softmin-combine them into a target, and
    # regress the head onto it.  The target is a convex combination of the
    # candidates, so it is BOUNDED BY THE CANDIDATE SPAN by construction --
    # runaway is structurally impossible, not merely penalized.
    #
    #: Fraction of THE BATCH (not of steps) the search runs on, rows
    #: ``[0, frac * batch)`` -- the same leading-slice convention as
    #: ``tc_frac`` / ``suffix_frac``, so no RNG draw is needed and the feature
    #: cannot perturb the batch / noise / gain stream of an A/B partner.
    #: 0.0 (the default) disables the whole path exactly.
    waist_sp_search_frac: float = 0.0
    #: TOTAL candidates per row, including the mandatory analytic-aim column.
    #: The rest is a quadratically spaced symmetric ladder on the head's own
    #: (detached) set-point -- see :func:`waist_sp_candidates`, which rounds an
    #: odd ``k`` down.  8 gives ``{0, +-span/9, +-4span/9, +-span}`` plus aim.
    waist_sp_search_k: int = 8
    #: Half-width, rad, of that ladder.  The target can never leave
    #: ``sp +- span`` (nor the aim), which is the runaway bound.
    waist_sp_search_span: float = 0.8
    #: Softmin temperature, DIMENSIONLESS: candidate scores are min-max
    #: normalized per row before the softmax, so this knob means the same thing
    #: whatever the absolute cost scale is (and the AL multipliers move that
    #: scale by decades over a run).  -> 0 is a hard argmin, large is the plain
    #: candidate mean.  The default is deliberately NOT a hard argmin: on a
    #: clear winner it lands the target roughly two thirds of the way there, so
    #: one mis-scored candidate cannot yank the head, and the pull stays smooth
    #: across a tie.
    waist_sp_search_temp: float = 0.1
    #: Price on ``Huber(sp - detach(sp*))``, rad^-2-ish (beta 0.1 rad).  0.0
    #: (the default) means the search never runs even at a non-zero frac.
    w_waist_sp_search: float = 0.0

    schedule: WeightSchedule = field(default_factory=default_weight_schedule)
    al: ALConfig = field(default_factory=phase2_al_config)
    al_every: int = 100

    # ---- actor-conditioned task-time feasibility critic -------------------
    #: ``off`` is exactly the legacy path.  ``shadow`` learns from the exact
    #: diagnostics the main cost pass already produces but cannot influence
    #: sampling or actor training.  ``assist`` and ``control`` are reserved for
    #: the guarded frontier sampler implemented in the next rollout phase.
    timing_critic: ViolationCriticConfig = field(
        default_factory=ViolationCriticConfig,
    )
    #: State noise applied when the timing curriculum advances a certified
    #: parent plan to a random later waypoint.  The child starts from the
    #: parent's spline state at that waypoint and receives the truncated tail
    #: as its initialization; this is a cheap collocation handoff rather than
    #: a plant rollout.
    timing_mpc_q_noise: float = 0.01
    timing_mpc_qd_noise: float = 0.05
    #: Certified full parent trajectories retained per horizon. At the default
    #: batch 512, 2048 keeps roughly four actor generations available for
    #: repeatedly sampled random suffix collocations.
    timing_mpc_parent_capacity: int = 2048

    # ---- data ----
    #: Use the positive-witness voxel curriculum instead of the legacy
    #: all-fresh/replay batch switch.  Feasibility is existential and sticky:
    #: only a trajectory passing collision, terminal-region, joint-position,
    #: and available-torque gates can add a cell; a failed probe stays unknown.
    voxel_pool: bool = False
    voxel_q_cell_rad: float = 0.02
    #: Maximum static-pool rows hard-checked for bootstrap (0 = all rows).
    voxel_bootstrap_rows: int = 10_000
    voxel_bootstrap_batch: int = 256
    #: Knot stride for ordinary along-trajectory bootstrap collocations.
    voxel_bootstrap_stride: int = 4
    #: Conservative time allotted to a finishing hand's next goal after bounce.
    voxel_posthit_anchor_tth_s: float = 1.0
    #: Minimum source-clip gap used when mining idle/rest states.
    voxel_rest_gap_s: float = 1.0
    #: Optional positive-evidence cache; rejected on any hard-gate fingerprint change.
    voxel_cache: str | None = None
    voxel_probe_capacity: int = 256
    voxel_probe_refresh_every: int = 10
    #: Asymmetric timing stencil at each occupied boundary cell.
    #: Every complete pair is independently hard-gated; no intermediate layer
    #: is inferred from a farther success.
    voxel_probe_shorter_layers: int = 1
    voxel_probe_longer_layers: int = 4
    voxel_probe_timing_step: int = 1
    #: Fraction of newly enqueued probes drawn from the timing faces rather
    #: than the 578 positional face/pairwise-diagonal neighbours.  Without
    #: explicit balancing the many state directions would statistically drown
    #: out the five timing layers.
    voxel_probe_timing_frac: float = 0.5
    #: Impact speed is task variation, not voxel identity.  It is sampled on
    #: every materialization and written to both encoder and strike context.
    voxel_strike_speed_min: float = 1.0
    voxel_strike_speed_max: float = 3.0
    #: Batch strata.  Empty MPC or boundary strata are redistributed to the
    #: occupied feasible pool; these four values must sum to one.
    voxel_frac_mpc: float = 0.50
    voxel_frac_pool: float = 0.30
    voxel_frac_boundary: float = 0.15
    voxel_frac_probe: float = 0.05
    #: Fixed quadratic coefficient used on unknown probes.  Their dual
    #: multipliers are always zero and their violations never update AL state.
    voxel_probe_penalty: float = 1.0

    replay_frac_max: float = 0.9
    replay_warmup: int = 2_000
    replay_capacity: int = 40_000
    recency_bias: float = 2.0
    #: Sampling beta for replay rows with high harvested torque violations:
    #: ``w *= exp(-beta * score)`` where ``score`` is the per-row max over
    #: ``torque_*`` / ``torque_peak`` violations on the stored plan.  0 = off.
    #: This changes sampling probability only; it does not scale the loss.
    replay_viol_downweight: float = 0.0
    #: Light proficiency stratification on the MPC replay ring: per-row EMA of
    #: hit+coll+torque ``ok`` after each forward, stratified ``sample_idx``
    #: (prof / new / flex).  Reuses ``frontier_ema_*`` thresholds.
    replay_proficiency: bool = False
    replay_frac_prof: float = 0.40
    replay_frac_new: float = 0.40
    #: Max ``torque_excess_max`` (N·m) for proficiency ``ok`` on replay EMA
    #: and frontier EMA updates.  Aligns with ``frontier_auto_viol_cap``.
    proficiency_viol_cap: float = 8.0
    roller_slots: int = 256
    roller_horizon: int = 50
    roller_ticks_per_step: int = 1
    roller_chain_frac: float = 0.35
    roller_start: int = 200
    #: Per-hand IN-PLACE re-goal instead of respawn-on-hit.  ON: the roller then
    #: reproduces ``MPCGenerator.run_episode``'s reset semantics -- when one
    #: hand's hit fires, only that hand promotes its next future token, only its
    #: arm columns are re-seeded, the other hand keeps counting down on the
    #: shared waist and the PD state carries through untouched.  Respawn-only
    #: (``False``) is the pre-2026-08-10 behaviour, kept for the A/B: with it the
    #: model never sees a mid-flight goal switch, which is the diagnosed cause of
    #: the closed-loop MPC divergence (README "decisions needing review" 3, 8).
    roller_regoal: bool = True
    #: Waist-yaw grid the roller's GPU goal-pose table is precomputed on.  The
    #: yaw range is ~3.5 rad, so 129 bins is a 27 mrad quantization -- well below
    #: the library's own 22.5 deg (0.39 rad) yaw-grid spacing, i.e. the binning
    #: is never what decides which library entry is picked.
    roller_yaw_bins: int = 129
    #: Ticks a slot may live before it is respawned regardless.
    roller_max_age: int = 400
    #: THE IDLE ARM.  Path to the idle-cylinder pose library
    #: (``assets/idle_cylinder_poses.json``), ``"default"`` for that asset if it
    #: exists, or ``None`` to disable -- seeds, roller rest stretches and
    #: ``w_idle_cylinder`` / ``w_idle_vel`` key off whether a library was loaded.
    #: ``idle_home_path`` is accepted as a deprecated alias (same semantics).
    idle_cylinder_path: str | None = "default"
    idle_home_path: str | None = None
    #: Waypoints the retraction eases over before holding inside the cylinder.
    #: Stretched automatically when the arm is far enough that this would exceed
    #: its speed rating.  10 wp = 0.2 s.
    idle_settle_wp: int = 10
    #: Ticks a BOTH-HANDS-IDLE slot keeps simulating before respawn.  The roller
    #: historically deleted rest stretches outright (``~active.any()`` was an
    #: immediate reset), so without this the idle terms would train entirely off
    #: the states they are meant to shape.  25 ticks = 0.5 s of rest, which is a
    #: whole beat at 120 bpm.
    roller_rest_ticks: int = 25
    #: FINAL-APPROACH OVERSAMPLING.  Rows whose nearest ACTIVE hit is within
    #: ``short_wp`` waypoints get ``short_boost`` x the sampling weight, in the
    #: fresh pool AND in the MPC replay.  The roller harvests every slot on
    #: every tick, so its horizon distribution is roughly uniform over each
    #: goal's initial horizon: measured on ``runs/phase2b_main/best.pt`` at 256
    #: slots, only 33.7 % (bucket 25) / 18.8 % (bucket 50) of replay rows sit at
    #: ``k <= 5``.  That is the band the MPC's executed terminal comes from, and
    #: it is where the model measurably fails -- the plan's own terminal aim is
    #: 2 mm INSIDE the hit disk at a 5-waypoint horizon and 61 mm OUTSIDE it at a
    #: 1-waypoint horizon.  1.0 = uniform (the pre-7c behaviour).
    short_wp: int = 8
    short_boost: float = 1.0
    #: Fresh-pool mix over rhythm-derived families (see plan).  Sampled
    #: categorically by ``kind`` inside :meth:`Phase2Pool.sample_idx`.  While
    #: the live post-hit buffer is empty the trainer renormalizes over
    #: nonempty families (~75/25 idle/far) with a one-shot loud log; once
    #: the roller harvests on-policy post-hit inits, ``post_hit_frac`` is
    #: drawn from that buffer (not the offline IK distill).
    rhythm_idle_frac: float = 0.30
    post_hit_frac: float = 0.60
    far_fetched_frac: float = 0.10
    #: Recency-biased ring of on-policy post-hit inits harvested from the
    #: roller after a successful re-goal or chain-carry.  Smaller than the
    #: full replay: these rows are only the ~10 % fresh mix's Phase-B slice.
    #: Harvest screens q0 with the same stick+wrist SDF as Pool A (departure
    #: drum exempted); colliding aftermaths stay in replay, not this buffer.
    post_hit_capacity: int = 8_192
    #: Optional eval-like overlay (off by default under the rhythm mix).
    eval_like_mix_frac: float = 0.0
    #: REFERENCE LEAD executed by the roller / TC / BPTT plant, in seconds.
    #: MUST match ``MPCGenConfig.cmd_lead_s`` at deployment: the roller is what
    #: generates the replay states the model is trained on, so a lead present in
    #: one and absent in the other is a train/deploy mismatch, and a model that
    #: learns its own lead ON TOP of a control-side lead strikes early.
    #: See :mod:`amortized_opt.lead`.
    cmd_lead_s: float | tuple[float, ...] | None = 0.0
    #: CLOSED-LOOP PROBE.  Steps between calls to the ``probe_fn`` handed to
    #: :func:`train_phase2` (0 = never).  Open-loop grid acceptance alone
    #: selected the 7b checkpoint, which scored 175/243 on the plan and 2/55 on
    #: the plant; ``best.pt`` is now additionally gated on this.
    probe_every: int = 0
    #: SATURATE the training plant's effort, i.e. clamp the demanded generalized
    #: force to the joint's usable torque inside every PD substep of the roller,
    #: the temporal-consistency step and the BPTT chain (``PDSimConfig.
    #: saturate_effort``).  OFF by default and bit-for-bit when off.  ON, the
    #: offline plant stops flattering the planner in exactly the place Isaac
    #: does not: an unclamped plant tracks a command no actuator could execute,
    #: so a plan that only works because of a 38 N*m wrist demand still scores.
    plant_saturate_effort: bool = False

    #: Later generalization stage on a trained checkpoint; seeds the cloud
    #: from the existing Phase2Pool (starts and mid-trajectory knots).
    #: Default OFF: the trainer never constructs a cloud and does not
    #: touch batch / spawn / harvest paths.
    frontier: bool = False
    frontier_proposals: int = 64
    #: Child ring is ``chart_cap * 64``. 2048 ≈ 4× the old 512 lifetime so a
    #: child can collect 2-3 EMA looks before wrap (harvest still shortens it).
    frontier_chart_cap: int = 2048
    frontier_fresh_frac: float = 0.5
    frontier_spawn_frac: float = 0.25
    frontier_sigma_q: float = 0.05
    #: rad/s; same per-joint weights as ``frontier_sigma_q``.
    frontier_sigma_qd: float = 0.10
    #: Fraction of remaining TTH (0.05 = 5 %), not an absolute second.
    frontier_sigma_tth: float = 0.05
    #: 0.25: two hits from the 0.5 prior clear 0.70; three reach ~0.79.
    frontier_ema_alpha: float = 0.25
    #: Parent sampling holds the 0.5 prior until this many EMA updates.
    frontier_ema_nmin: int = 2
    frontier_ema_prof: float = 0.70
    frontier_landmark_k: int = 32
    frontier_landmark_tau: float = 2.0
    frontier_propose_candidates: int = 256
    frontier_hull_ema: float = 0.55
    frontier_train_inv_frac: float = 0.5
    frontier_train_inv_pool: int = 2048
    frontier_include_miss: bool = False
    #: Knot stride for pool-init suffix slices at seed time only.
    #: ``1`` = every knot; ``4`` is the plan default. Mid-traj of the *model*
    #: plan is not harvested -- the frozen pool already collocates those
    #: suffixes, and the suffix Bellman term trains them.
    frontier_traj_stride: int = 4
    #: When the cloud can fill the step's batch, the minibatch IS a uniform
    #: sample of collocations (the expanding start set). ``512`` matches the
    #: usual phase-2 batch; a smaller ``batch_for`` still trains the whole
    #: batch on the cloud. Fallback splice only if the cloud is smaller.
    frontier_sample_k: int = 512
    #: Idle/far fresh rows come from the cloud only (seed + ball children).
    #: Post-hit stays the live roller stream. Turns off whole-batch cloud
    #: takeover, q0-steal, and post-hit harvest into the cloud. Replay
    #: still gets ``frontier_splice_k`` cloud rows so ball children collect
    #: EMA looks before the child ring wraps. No-op unless ``frontier``.
    frontier_idle_far_cloud: bool = False
    #: Cloud rows spliced into every replay/fresh minibatch under
    #: ``frontier_idle_far_cloud``. 64 ≈ 3–4 looks per child at the current
    #: ingest rate (the ``chart_cap=2048`` design). 0 disables. Ignored
    #: unless idle-far-cloud; the non-idle-far path still splices
    #: ``frontier_sample_k``.
    frontier_splice_k: int = 64
    #: Auto-enable frontier when closed-loop probe ``axial_abs_p50_mm`` drops
    #: below this (0 = off).  Also honours ``<run_dir>/.frontier_activate``
    #: written by :mod:`scripts.frontier_auto_monitor`.
    frontier_auto_miss_mm: float = 0.0
    #: Snap suffix warmup like ``--full-weights`` when auto-frontier fires.
    frontier_auto_full_weights: bool = True
    frontier_auto_viol_cap: float = 8.0
    frontier_auto_viol_ratio: float = 3.0
    frontier_auto_viol_window: int = 10

    # ---- cost ----
    cost: CostConfig = field(default_factory=phase2_cost_config)
    n_sub_late: int | None = None      # bump FusedCost.n_sub after ramp_end
    bounce: BounceConfig = field(default_factory=BounceConfig)

    # ---- eval / bookkeeping ----
    eval_every: int = 2_000
    eval_cases: int = 81               # stride-subsampled acceptance grid
    log_every: int = 50
    #: Write TensorBoard scalars under ``<run_dir>/tb`` (``train/``, ``eval/``,
    #: ``probe/`` prefixes).  Requires ``tensorboard`` (``pip install -e
    #: '.[monitoring]'``).
    tensorboard: bool = True
    #: Sample one outer step with CUDA-event phase markers every N steps.
    #: Zero disables profiling; only sampled steps synchronize for reporting.
    phase_timing_every: int = 0
    ckpt_every: int = 2_000
    #: ``last.pt`` cadence between full ``ckpt_every`` bundles (frontier plot,
    #: ``frontier.pt``).  0 = only save on ``ckpt_every`` and shutdown.
    ckpt_last_every: int = 500
    #: Retained ``step_NNNNNN.pt`` snapshots (never overwritten).  0 = off.
    ckpt_keep_every: int = 500
    plateau_evals: int = 5
    target_acceptance: float = 0.60

    seed: int = 0
    device: str = "cuda:1"
    #: Uniform mixture applied after normalizing each eligible replay sampling
    #: subset: ``p = (1-gamma) * normalize(w) + gamma / n_eligible``.
    #: Appended to preserve positional compatibility of older Phase2Config
    #: construction.  0 preserves the legacy sampler.
    replay_viol_uniform_mix: float = 0.0

    def __post_init__(self) -> None:
        if float(self.w_waist_yaw_mpc_tie) < 0.0:
            raise ValueError("w_waist_yaw_mpc_tie must be nonnegative")
        if int(self.bptt_batch) < 1:
            raise ValueError("bptt_batch must be positive")
        if (
            int(self.bptt_decoder_passes) < 2
            or int(self.bptt_decoder_passes) % 2 != 0
        ):
            raise ValueError("bptt_decoder_passes must be an even integer >= 2")
        if isinstance(self.timing_critic, dict):
            self.timing_critic = ViolationCriticConfig(**self.timing_critic)
        if int(self.actor_superbatch_factor) < 1:
            raise ValueError("actor_superbatch_factor must be >= 1")
        if not math.isfinite(float(self.timing_mpc_q_noise)) or float(
            self.timing_mpc_q_noise
        ) < 0.0:
            raise ValueError("timing_mpc_q_noise must be finite and nonnegative")
        if not math.isfinite(float(self.timing_mpc_qd_noise)) or float(
            self.timing_mpc_qd_noise
        ) < 0.0:
            raise ValueError("timing_mpc_qd_noise must be finite and nonnegative")
        if int(self.timing_mpc_parent_capacity) < 1:
            raise ValueError("timing_mpc_parent_capacity must be positive")
        if int(self.suffix_microbatch) < 0:
            raise ValueError("suffix_microbatch must be >= 0")
        suffix_mix = (
            float(self.terminal_suffix_peak_frac),
            float(self.terminal_suffix_long_frac),
            float(self.terminal_suffix_low_frac),
        )
        if any(x < 0.0 or not math.isfinite(x) for x in suffix_mix) or not math.isclose(
            sum(suffix_mix), 1.0, rel_tol=0.0, abs_tol=1.0e-6,
        ):
            raise ValueError(
                "terminal suffix mixture fractions must be finite, nonnegative, and sum to 1"
            )
        if int(self.terminal_suffix_peak_wp) < 2:
            raise ValueError("terminal_suffix_peak_wp must be at least 2")
        if float(self.terminal_suffix_tail_tau) <= 0.0:
            raise ValueError("terminal_suffix_tail_tau must be positive")
        if int(self.suffix_replay_batch) < 1:
            raise ValueError("suffix_replay_batch must be >= 1")
        if int(self.suffix_replay_capacity) < 1:
            raise ValueError("suffix_replay_capacity must be >= 1")
        if not 1 <= int(self.suffix_replay_min_parents) <= int(self.suffix_replay_batch):
            raise ValueError(
                "suffix_replay_min_parents must be in [1, suffix_replay_batch]"
            )
        if int(self.suffix_replay_batch) > int(self.suffix_replay_capacity):
            raise ValueError(
                "suffix_replay_batch cannot exceed suffix_replay_capacity"
            )
        gamma = float(self.replay_viol_uniform_mix)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                f"replay_viol_uniform_mix must be in [0, 1], got {gamma}"
            )
        if int(self.torque_grad_cap_refresh) < 1:
            raise ValueError("torque_grad_cap_refresh must be positive")
        if not 0.0 <= float(self.torque_grad_cap_scale_ema) < 1.0:
            raise ValueError("torque_grad_cap_scale_ema must be in [0, 1)")
        if int(self.phase_timing_every) < 0:
            raise ValueError("phase_timing_every must be nonnegative")
        if not 0.0 <= float(self.dls_terminal_frac) <= 1.0:
            raise ValueError("dls_terminal_frac must be in [0, 1]")
        if float(self.w_dls_terminal_cost) < 0.0:
            raise ValueError("w_dls_terminal_cost must be nonnegative")
        if float(self.w_dls_terminal_tie) < 0.0:
            raise ValueError("w_dls_terminal_tie must be nonnegative")
        if int(self.dls_terminal_steps) < 1:
            raise ValueError("dls_terminal_steps must be positive")
        if int(self.dls_terminal_window) < 1:
            raise ValueError("dls_terminal_window must be positive")
        if float(self.dls_terminal_damping) <= 0.0:
            raise ValueError("dls_terminal_damping must be positive")
        if float(self.dls_terminal_max_step_rad) <= 0.0:
            raise ValueError("dls_terminal_max_step_rad must be positive")
        if self.terminal_lookahead_mode not in ("dls", "curvature"):
            raise ValueError("terminal_lookahead_mode must be 'dls' or 'curvature'")
        if float(self.curvature_damping) <= 0.0:
            raise ValueError("curvature_damping must be positive")
        if float(self.curvature_position_weight) < 0.0:
            raise ValueError("curvature_position_weight must be nonnegative")
        if float(self.curvature_velocity_weight) < 0.0:
            raise ValueError("curvature_velocity_weight must be nonnegative")
        if float(self.curvature_max_step_rad) <= 0.0:
            raise ValueError("curvature_max_step_rad must be positive")
        if not 0.0 <= float(self.curvature_direct_frac) <= 1.0:
            raise ValueError("curvature_direct_frac must be in [0, 1]")
        if not 0.0 <= float(self.curvature_direct_mix) <= 1.0:
            raise ValueError("curvature_direct_mix must be in [0, 1]")
        if self.curvature_direct_frac > 0.0 and self.dls_terminal_frac > 0.0:
            raise ValueError("direct curvature and terminal lookahead cannot both be active")
        if self.curvature_direct_frac > 0.0 and self.grad_balance:
            raise ValueError("direct curvature requires grad_balance='' ")
        if self.voxel_pool:
            # Voxel training deliberately gives the torso to the trajectory
            # decoder.  Keeping one of the legacy encoder set-point losses on
            # would quietly re-introduce SP -> strike / SP -> seed coupling,
            # even though the voxel materializer itself stores no set-point.
            sp_couplings = {
                "waist_sp_from_head": self.waist_sp_from_head,
                "waist_pr_sp_from_head": self.waist_pr_sp_from_head,
                "w_waist_sp_tether": self.w_waist_sp_tether,
                "w_waist_pr_sp_tether": self.w_waist_pr_sp_tether,
                "w_waist_pr_sp_hold": self.w_waist_pr_sp_hold,
                "w_waist_pr_sp_hold_band": self.w_waist_pr_sp_hold_band,
                "w_waist_pr_sp_distill": self.w_waist_pr_sp_distill,
                "w_waist_sp_hold": self.w_waist_sp_hold,
                "w_waist_sp_smooth": self.w_waist_sp_smooth,
                "waist_sp_search_frac": self.waist_sp_search_frac,
                "w_waist_sp_search": self.w_waist_sp_search,
                "waist_pr_ema": self.waist_pr_ema,
            }
            enabled = [name for name, value in sp_couplings.items() if bool(value)]
            if enabled:
                raise ValueError(
                    "voxel_pool uses decoder-owned torso motion and forbids "
                    "waist set-point/reference coupling: " + ", ".join(enabled)
                )
            if (
                float(self.cost.w_waist_aim) != 0.0
                or float(self.cost.w_waist_yaw_monotonic) != 0.0
                or float(self.cost.w_waist_yaw_accel_direction) != 0.0
                or float(self.cost.w_waist_yaw_initial_accel_direction) != 0.0
            ):
                raise ValueError(
                    "voxel_pool requires waist reference costs off; the voxel path "
                    "carries no waist-yaw reference"
                )
            fracs = (
                float(self.voxel_frac_mpc), float(self.voxel_frac_pool),
                float(self.voxel_frac_boundary), float(self.voxel_frac_probe),
            )
            if any(x < 0.0 for x in fracs) or not math.isclose(
                sum(fracs), 1.0, rel_tol=0.0, abs_tol=1.0e-6,
            ):
                raise ValueError(f"voxel batch fractions must be nonnegative and sum to 1, got {fracs}")
            if self.voxel_q_cell_rad <= 0.0:
                raise ValueError("voxel_q_cell_rad must be positive")
            if (
                self.voxel_bootstrap_rows < 0
                or self.voxel_bootstrap_batch < 1
                or self.voxel_bootstrap_stride < 1
            ):
                raise ValueError("invalid voxel bootstrap row/batch limit")
            if (
                not math.isfinite(float(self.voxel_posthit_anchor_tth_s))
                or float(self.voxel_posthit_anchor_tth_s) <= 0.0
                or not math.isfinite(float(self.voxel_rest_gap_s))
                or float(self.voxel_rest_gap_s) < 0.0
            ):
                raise ValueError("invalid voxel post-hit anchor/rest-gap setting")
            if self.voxel_probe_capacity < 1:
                raise ValueError("voxel_probe_capacity must be positive")
            if self.voxel_probe_refresh_every < 1:
                raise ValueError("voxel_probe_refresh_every must be positive")
            if (
                self.voxel_probe_shorter_layers < 0
                or self.voxel_probe_longer_layers < 0
                or self.voxel_probe_timing_step < 1
                or not 0.0 <= float(self.voxel_probe_timing_frac) <= 1.0
                or (
                    self.voxel_probe_shorter_layers == 0
                    and self.voxel_probe_longer_layers == 0
                )
            ):
                raise ValueError("invalid voxel timing-probe stencil")
            if (
                not math.isfinite(float(self.voxel_strike_speed_min))
                or not math.isfinite(float(self.voxel_strike_speed_max))
                or float(self.voxel_strike_speed_min) < 0.0
                or float(self.voxel_strike_speed_min) > float(self.voxel_strike_speed_max)
            ):
                raise ValueError("invalid voxel strike-speed range")
            if self.grad_balance:
                raise ValueError(
                    "voxel_pool requires grad_balance='': probe rows have zero "
                    "dual price and must not enter unmasked gradient-surgery "
                    "constraint groups"
                )
            if self.line_search:
                raise ValueError(
                    "voxel_pool currently requires line_search=False; mixed-row "
                    "replay freezing has not been implemented"
                )
        # Plan §1–2 / §6 / §14: force-empty + warn (not ValueError) so launchers
        # that still pass GRAD_BALANCE=al_pcgrad_output keep working under LS.
        # train_phase2 must reaffirm this before make_cost_step (baked arity).
        if self.line_search and self.grad_balance:
            warnings.warn(
                f"line_search=True forces grad_balance='' "
                f"(was {self.grad_balance!r}); PCGrad/GradNorm off for true "
                f"total.backward() (parameter_line_search_plan §1–2, §14)",
                UserWarning,
                stacklevel=2,
            )
            self.grad_balance = ""

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / max(self.warmup_steps, 1)
        horizon = int(self.lr_horizon or self.max_steps)
        t = (step - self.warmup_steps) / max(horizon - self.warmup_steps, 1)
        t = min(max(t, 0.0), 1.0)
        return self.lr * (
            self.min_lr_frac
            + (1.0 - self.min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * t))
        )

    def batch_for(self, bucket: int) -> int:
        if self.token_budget <= 0:
            return int(self.batch_size)
        return int(max(64, min(self.batch_size, self.token_budget // max(int(bucket), 1))))

    def resolved_idle_cylinder_path(self) -> str | None:
        """``idle_cylinder_path``, falling back to the deprecated ``idle_home_path``."""
        if self.idle_cylinder_path is not None:
            return self.idle_cylinder_path
        return self.idle_home_path

    def replay_frac(self, step: int) -> float:
        if step < self.replay_warmup:
            return self.replay_frac_max * (step / max(self.replay_warmup, 1))
        return self.replay_frac_max

    def bptt_frac(self, step: int) -> float:
        s0 = int(self.bptt_start_frac * self.max_steps)
        if step < s0 or self.bptt_max_frac <= 0.0:
            return 0.0
        span = max(self.max_steps - s0, 1)
        return self.bptt_max_frac * _smoothstep((step - s0) / span)

    def advertised_kind_fracs(self) -> dict[int, float]:
        """CLI / config mix before empty-family renormalization."""
        from .rhythm_chain_data import (
            KIND_FAR_FETCHED,
            KIND_POST_HIT,
            KIND_RHYTHM_IDLE,
        )
        from .phase2_data import KIND_EVAL_LIKE

        return {
            KIND_RHYTHM_IDLE: float(self.rhythm_idle_frac),
            KIND_POST_HIT: float(self.post_hit_frac),
            KIND_FAR_FETCHED: float(self.far_fetched_frac),
            KIND_EVAL_LIKE: float(self.eval_like_mix_frac),
        }

    def effective_kind_fracs(
        self, pool, *, log_fn=print, extra_counts: dict[int, int] | None = None,
    ) -> dict[int, float] | None:
        """Renormalized fresh mix over nonempty kinds in ``pool``.

        ``extra_counts`` overlays live families (the on-policy post-hit
        buffer) so advertised ``post_hit_frac`` starts mixing as soon as the
        roller has harvested any rows, without requiring a static cache.

        Returns ``None`` when the pool has no rhythm-derived kinds, so
        :meth:`Phase2Pool.sample_idx` falls back to the kind-blind draw used by
        legacy caches.
        """
        from .rhythm_chain_data import (
            KIND_FAR_FETCHED,
            KIND_POST_HIT,
            KIND_RHYTHM_IDLE,
            effective_kind_fracs as _eff,
        )

        counts = dict(pool.kind_counts())
        if extra_counts:
            for k, n in extra_counts.items():
                n = int(n)
                if n > 0:
                    counts[int(k)] = counts.get(int(k), 0) + n
        rhythm = (KIND_RHYTHM_IDLE, KIND_POST_HIT, KIND_FAR_FETCHED)
        if not any(int(counts.get(k, 0)) > 0 for k in rhythm):
            return None
        if not hasattr(self, "_kind_mix_log"):
            self._kind_mix_log = {}
        return _eff(
            self.advertised_kind_fracs(),
            counts,
            log_fn=log_fn,
            log_state=self._kind_mix_log,
        )


# ============================================================== helpers ======
def effective_hits(hit: torch.Tensor, active: torch.Tensor, H: int) -> torch.Tensor:
    """``(B, 2)`` horizons matching :func:`build_waypoint_masks` exactly.

    An idle hand borrows the other's horizon (and the full grid when neither is
    active), so ``FusedCost``'s per-group segment counts and the decoder's
    element masks describe the same window.  Getting this wrong silently prices
    only the first segment of an idle arm.
    """
    al, ar = active[:, 0].bool(), active[:, 1].bool()
    kl, kr = hit[:, 0].clamp_min(0), hit[:, 1].clamp_min(0)
    neg = torch.full_like(kl, -1)
    both = torch.maximum(torch.where(al, kl, neg), torch.where(ar, kr, neg))
    both = torch.where(al | ar, both, torch.full_like(both, int(H)))
    return torch.stack([torch.where(al, kl, both), torch.where(ar, kr, both)], dim=1)


def group_lengths(hit_eff: torch.Tensor) -> torch.Tensor:
    """``(B, 3)`` waist / left / right segment counts for ``FusedCost``."""
    return torch.stack(
        [torch.maximum(hit_eff[:, 0], hit_eff[:, 1]), hit_eff[:, 0], hit_eff[:, 1]], dim=1
    )


_ELEM_GROUP_CACHE: dict[tuple, torch.Tensor] = {}


def element_group(device, joint_names=G1_UPPER_BODY_JOINTS) -> torch.Tensor:
    """``(n,)`` group id per joint: 0 = waist (longer horizon), 1 = left, 2 = right."""
    key = (str(device), tuple(joint_names))
    t = _ELEM_GROUP_CACHE.get(key)
    if t is None:
        g = [1 if n.startswith("left_") else (2 if n.startswith("right_") else 0)
             for n in joint_names]
        t = torch.as_tensor(g, device=device, dtype=torch.long)
        _ELEM_GROUP_CACHE[key] = t
    return t


def per_element_lengths(hit_eff: torch.Tensor, joint_names=G1_UPPER_BODY_JOINTS) -> torch.Tensor:
    """``(B, n)`` per-joint horizon from the per-hand one."""
    return group_lengths(hit_eff).index_select(1, element_group(hit_eff.device, joint_names))


def build_costs(
    buckets,
    cfg: CostConfig,
    *,
    device,
    dtype: torch.dtype = torch.float32,
    fk: ChainFK | None = None,
    rnea: BatchedRNEA | None = None,
) -> dict[int, FusedCost]:
    """One :class:`FusedCost` per horizon bucket, sharing FK / RNEA / scene."""
    out: dict[int, FusedCost] = {}
    base = None
    for b in sorted(int(x) for x in buckets):
        if base is None:
            out[b] = FusedCost(cfg, max_horizon=b, device=device, dtype=dtype,
                               fk=fk, rnea=rnea)
            base = out[b]
        else:
            out[b] = FusedCost(cfg, max_horizon=b, device=device, dtype=dtype,
                               fk=base.fk, rnea=base.rnea, scene=base.scene)
    return out


def temporal_consistency_loss(
    traj_h: torch.Tensor,       # (B, H, n) solved at horizon H
    traj_next: torch.Tensor,    # (B, H, n) solved one tick later
    mask_h: torch.Tensor,       # (B, H, n) element mask of the first solve
    mask_next: torch.Tensor,    # (B, H, n) element mask of the second
    *,
    beta: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked disagreement on the shared time support (plan lines 102-107).

    The tail of ``traj_h`` (waypoints 1..H-1, i.e. times 2dt..H dt) and the head
    of ``traj_next`` (waypoints 0..H-2, times dt..(H-1) dt *after the tick*)
    describe the same instants.  Zero for a perfectly consistent pair.
    """
    a = traj_h[:, 1:]
    b = traj_next[:, : traj_next.shape[1] - 1]
    m = mask_h[:, 1:] * mask_next[:, : mask_next.shape[1] - 1]
    return _masked_huber(b, a, m, beta)[0], m.sum()


def mpc_init_tie_loss(
    traj: torch.Tensor,
    shifted_parent: torch.Tensor,
    elem_mask: torch.Tensor,
    mpc_rows: torch.Tensor,
    *,
    beta: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tie outputs to stored shifted-parent plans on MPC replay rows only.

    ``ReplayPool.seed`` is ``[A1, ..., A(H-1), A(H-1)]`` after executing
    ``A0``.  The repeated final waypoint is padding, so only ``[:-1]`` is
    compared.  ``shifted_parent`` is detached history: gradients update the
    current solve ``B`` but do not attempt to reconstruct the old parent graph.
    The stored ``seed`` is used rather than the model-input ``init`` because
    optional preprocessing may rewrite ``init``.
    """
    rows = mpc_rows.to(device=traj.device, dtype=traj.dtype).view(-1, 1, 1)
    mask = elem_mask[:, :-1].to(dtype=traj.dtype) * rows
    loss, _sq, n = _masked_huber(
        traj[:, :-1], shifted_parent[:, :-1].detach(), mask, beta,
    )
    return loss, n


def waist_yaw_mpc_init_tie_loss(
    traj: torch.Tensor,
    shifted_parent: torch.Tensor,
    elem_mask: torch.Tensor,
    mpc_rows: torch.Tensor,
    *,
    yaw_index: int,
    beta: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Time-aligned MPC consistency on the single waist-yaw output column.

    Returns ``(loss, valid_count, output_grad_norm)``.  The last value is the
    norm of ``d loss / d traj_yaw`` before the configured scalar weight; logging
    ``weight * output_grad_norm`` makes gradient significance directly visible.
    """
    rows = mpc_rows.to(device=traj.device, dtype=traj.dtype).view(-1, 1)
    mask = elem_mask[:, :-1, int(yaw_index)].to(dtype=traj.dtype) * rows
    pred = traj[:, :-1, int(yaw_index)]
    target = shifted_parent[:, :-1, int(yaw_index)].detach()
    loss, _sq, n = _masked_huber(pred, target, mask, beta)
    with torch.no_grad():
        d = (pred.detach() - target) * mask
        grad = torch.where(d.abs() < beta, d / beta, d.sign()) * mask / n
        grad_norm = grad.float().norm()
    return loss, n, grad_norm


# ================================================================ bounce =====
def bounce_tip_velocity(
    v_tip: torch.Tensor, normal: torch.Tensor, restitution: torch.Tensor
) -> torch.Tensor:
    """Reflect the component of ``v_tip`` along ``normal`` with restitution.

    ``normal`` is the OUTWARD head normal, so an incoming strike has
    ``v_tip . n < 0``.  The tangential component is untouched **exactly** --
    a frictionless normal impulse -- which is the property
    ``tests/test_train_cost.py`` asserts.
    """
    n = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    vn = (v_tip * n).sum(-1, keepdim=True)
    # only reflect when moving INTO the surface
    into = (vn < 0).to(v_tip.dtype)
    e = restitution.reshape(-1, 1) if restitution.dim() == 1 else restitution
    return v_tip - into * (1.0 + e) * vn * n


def apply_bounce(
    q: torch.Tensor,
    qd: torch.Tensor,
    *,
    fk: ChainFK,
    rnea: BatchedRNEA,
    hand_index: torch.Tensor,       # (B,) 0 = left, 1 = right
    normal: torch.Tensor,           # (B, 3) outward head normal
    cfg: BounceConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-contact ``(q, qd)`` for the striking hand, differentiably.

    The tip velocity is reflected in task space and the required joint-velocity
    change is mapped back with the inertia-weighted (dynamically consistent)
    pseudoinverse, so the correction is carried by the joints that can supply it
    most cheaply -- the wrists -- exactly as in :mod:`amortized_opt.starts`.
    """
    B = q.shape[0]
    dev, dt_ = q.device, q.dtype
    e = (
        cfg.restitution[0]
        + (cfg.restitution[1] - cfg.restitution[0])
        * torch.rand(B, device=dev, dtype=dt_, generator=generator)
    )
    # ``tip_jacobian`` differentiates FK, so it needs a grad-enabled input --
    # and the roller calls this inside ``no_grad``.  Take the Jacobian off a
    # detached copy there; the roller does not back-propagate through it.
    if q.requires_grad and torch.is_grad_enabled():
        J = torch.stack([tip_jacobian(fk, q, h) for h in _HANDS], dim=1)
    else:
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            J = torch.stack([tip_jacobian(fk, qg, h) for h in _HANDS], dim=1).detach()
    sel = hand_index.view(B, 1, 1, 1).expand(B, 1, 3, q.shape[1])
    Jh = J.gather(1, sel).squeeze(1)                       # (B, 3, n)
    v_tip = torch.matmul(Jh, qd.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    v_new = bounce_tip_velocity(v_tip, normal, e)
    if cfg.tip_vel_noise > 0.0:
        v_new = v_new + cfg.tip_vel_noise * torch.randn(
            B, 3, device=dev, dtype=dt_, generator=generator
        )
    M = mass_matrix(rnea, q.detach())
    P = inertia_weighted_pinv(Jh, M=M, damping=1e-4)        # (B, n, 3)
    dqd = torch.matmul(P, (v_new - v_tip).unsqueeze(-1)).squeeze(-1)
    qd_out = qd + dqd
    if cfg.qd_noise > 0.0:
        qd_out = qd_out + cfg.qd_noise * torch.randn(
            qd.shape, device=dev, dtype=dt_, generator=generator
        )
    q_out = q
    if cfg.q_noise > 0.0:
        q_out = q + cfg.q_noise * torch.randn(
            q.shape, device=dev, dtype=dt_, generator=generator
        )
    return q_out, qd_out


# ============================================================= seeding =======
def ramp_seed(
    q_cur: torch.Tensor,        # (B, n)
    terminal: torch.Tensor,     # (B, n) per-element terminal pose
    lengths: torch.Tensor,      # (B, n) per-element horizon in waypoints
    H: int,
) -> torch.Tensor:
    """Quintic-eased straight line ``q_cur -> terminal``, then hold.

    Retained for waist and idle-arm settling. Active strike arms use
    :func:`~amortized_opt.strike_seed.strike_seed`, which matches the starting
    velocity and leaves arrival velocity free.
    """
    k = torch.arange(1, H + 1, device=q_cur.device, dtype=q_cur.dtype).view(1, H, 1)
    L = lengths.clamp_min(1).to(q_cur.dtype).unsqueeze(1)     # (B, 1, n)
    s = (k / L).clamp(0.0, 1.0)
    ease = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)        # quintic smoothstep
    return q_cur.unsqueeze(1) + ease * (terminal - q_cur).unsqueeze(1)


def vel_ramp_seed(
    q_cur: torch.Tensor,        # (B, n)
    v_cur: torch.Tensor,        # (B, n) start rate, caller-clipped to the rating
    terminal: torch.Tensor,     # (B, n)
    lengths: torch.Tensor,      # (B, n) per-element window in waypoints
    H: int,
    dt: float,
) -> torch.Tensor:
    """:func:`ramp_seed` with a VELOCITY-MATCHED start.

    The boundary quintic ``p(0)=q_cur, p'(0)=v_cur, p''(0)=0`` arriving at
    ``terminal`` at rest at waypoint ``lengths`` and holding after.
    ``v_cur = 0`` reduces exactly to :func:`ramp_seed`, so a rest start is
    bit-identical to the rest-to-rest ramp.  The numpy twin is
    ``warmstart._boundary_quintic``; the two must agree element-for-element
    (pinned by ``tests/test_waist_upright_ease.py``).
    """
    k = torch.arange(1, H + 1, device=q_cur.device, dtype=q_cur.dtype).view(1, H, 1)
    L = lengths.clamp_min(1).to(q_cur.dtype)                  # (B, n)
    u = (k / L.unsqueeze(1)).clamp(0.0, 1.0)                  # (B, H, n)
    d = (terminal - q_cur).unsqueeze(1)
    V = (v_cur * L * float(dt)).unsqueeze(1)
    u3 = u * u * u
    return (
        q_cur.unsqueeze(1) + V * u
        + (10.0 * d - 6.0 * V) * u3
        + (8.0 * V - 15.0 * d) * u3 * u
        + (6.0 * d - 3.0 * V) * u3 * u * u
    )


@torch.no_grad()
def retime_batch_timing(
    batch: dict,
    hit: torch.Tensor,
    pool: Phase2Pool,
    *,
    terminal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Retime a batch to complete, atomic two-hand hit pairs in place.

    The task terminal is retained while every timing-dependent field is
    rebuilt: hit indices, decoder masks, encoder goal TTH, and the
    velocity-matched initializer.  ``terminal`` may be supplied when the
    caller changes ``q0`` before retiming (the voxel materializer does this).

    Returns the retained per-joint terminal pose for reuse by callers.
    """
    h = int(batch["bucket"])
    active = batch["active"].to(torch.bool)
    new_hit = hit.to(device=batch["q0"].device, dtype=torch.long)
    if new_hit.shape != active.shape or new_hit.ndim != 2 or new_hit.shape[1] != 2:
        raise ValueError("hit and active must have matching shape (B, 2)")
    bad_active = active & ((new_hit < 1) | (new_hit > h))
    bad_idle = ~active & (new_hit != -1)
    if bool(bad_active.any()):
        raise ValueError(f"active timing pair must be within [1, {h}]")
    if bool(bad_idle.any()):
        raise ValueError("inactive timing-pair components must be -1")

    if terminal is None:
        old_eff = effective_hits(batch["hit"], active, h)
        old_len = per_element_lengths(old_eff, pool.cfg.joint_names)
        gather = (old_len.clamp(1, h) - 1).unsqueeze(1)
        terminal = batch["seed"].gather(1, gather).squeeze(1)
    else:
        terminal = terminal.to(device=batch["q0"].device, dtype=batch["q0"].dtype)

    batch["hit"] = new_hit
    batch["masks"] = pool.masks_for(new_hit, active, h)
    goal_tth = torch.where(
        active,
        new_hit.to(batch["q0"].dtype) * float(pool.cfg.dt),
        torch.zeros_like(new_hit, dtype=batch["q0"].dtype),
    )
    batch["context"] = replace(batch["context"], goal_tth=goal_tth)
    hit_eff = effective_hits(new_hit, active, h)
    elem_len = per_element_lengths(hit_eff, pool.cfg.joint_names)
    seed = strike_seed(
        batch["q0"], batch["v0"], terminal, elem_len, h,
        float(pool.cfg.dt), active=active, joint_names=pool.cfg.joint_names,
    )
    batch["seed"] = seed
    batch["init"] = seed * batch["masks"].elem_mask
    return terminal


@torch.no_grad()
def _harvest_replay_viol_score(
    cols: dict[str, torch.Tensor],
    bucket: int,
    pool: Phase2Pool,
    cost,
    weights: dict,
) -> torch.Tensor:
    """Per-row max torque-family violation on a harvested MPC plan."""
    from .tokenizer import ContextInputs

    H = int(bucket)
    n = int(cols["q0"].shape[0])
    if n == 0:
        return torch.zeros(0, device=pool.device, dtype=pool.dtype)
    idx = torch.arange(n, device=pool.device)
    hit, active = cols["hit"], cols["active"]
    masks = pool.masks_for(hit, active, H)
    q0, v0 = cols["q0"], cols["qd0"]
    init = cols["init"] * masks.elem_mask
    ctx = ContextInputs(**{f: cols[f] for f in CONTEXT_FIELDS if f in cols})
    heads = ctx.drum_pos - pool._pelvis
    wref = cols["waist_ref"] if "waist_ref" in cols else None
    if wref is None and getattr(pool, "waist_ref_from_seed", True):
        yaw_i = tuple(pool.cfg.joint_names).index("waist_yaw_joint")
        wref = cols["init"][:, :, yaw_i]
    scols = cols
    if "start_drum" not in scols:
        scols = dict(cols)
        scols["start_drum"] = scols["drum"]
    pref = scols["waist_pr_ref"] if "waist_pr_ref" in scols else None
    strike = strike_context_from_arrays(
        scols, idx, drum_head_centers=heads, drum_radii=ctx.drum_radius,
        waist_ref=wref, n_shaft=pool.n_shaft, waist_pr_ref=pref,
    )
    hit_eff = effective_hits(hit, active, H)
    strike = replace(strike, hit_index=hit_eff)
    lengths = group_lengths(hit_eff)
    ct = cost.forward(
        q0, init, v0, strike, lengths=lengths, weights=weights,
    )
    parts = []
    for k, v in ct.violations.items():
        if k == "torque_peak" or k.startswith("torque_"):
            parts.append(v.detach().float())
    if not parts:
        return torch.zeros(n, device=pool.device, dtype=pool.dtype)
    return torch.stack(parts, dim=0).amax(dim=0)


def _replay_sampling_probs(weights: torch.Tensor, gamma: float) -> torch.Tensor:
    """Normalize replay weights and mix with uniform over this eligible subset.

    Nonfinite and negative weights are treated as zero.  If no finite positive
    mass remains (including exponential underflow), the base distribution is
    uniform.  Consequently ``gamma > 0`` gives every eligible row probability
    at least ``gamma / len(weights)``.
    """
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"replay uniform mix gamma must be in [0, 1], got {gamma}")
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("replay sampling weights must be a nonempty vector")
    clean = torch.where(
        torch.isfinite(weights) & (weights > 0.0), weights, torch.zeros_like(weights),
    )
    total = clean.sum()
    uniform = torch.full_like(clean, 1.0 / float(clean.numel()))
    base = torch.where(
        torch.isfinite(total) & (total > 0.0),
        clean / total.clamp_min(torch.finfo(clean.dtype).tiny),
        uniform,
    )
    if gamma == 0.0:
        return base
    return (1.0 - gamma) * base + gamma * uniform


def _replay_multinomial(
    weights: torch.Tensor,
    n: int,
    *,
    gamma: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Replacement draw with exact legacy behavior on valid ``gamma=0`` input."""
    if float(gamma) == 0.0:
        try:
            # Keep valid legacy runs bit-for-bit identical: multinomial sees
            # precisely the same unnormalized weights and generator state.
            return torch.multinomial(
                weights, n, replacement=True, generator=generator,
            )
        except RuntimeError:
            # Degenerate/nonfinite weights previously crashed.  Fall through
            # to the robust normalized distribution for that exceptional case.
            pass
    probs = _replay_sampling_probs(weights, gamma)
    return torch.multinomial(probs, n, replacement=True, generator=generator)


# ============================================================ replay pool ====
class ReplayPool(Phase2Pool):
    """Recency-biased ring buffer with the :class:`Phase2Pool` batch interface.

    Preallocated per bucket, so adding a wave of MPC contexts is a scatter and
    sampling is the same code path the fresh pool uses.
    """

    def __init__(self, like: Phase2Pool, capacity: int, buckets=None,
                 recency_bias: float = 2.0, viol_downweight: float = 0.0,
                 proficiency: bool = False, ema_alpha: float = 0.25,
                 waist_ref: bool = False, waist_pr_ref: bool = False,
                 viol_uniform_mix: float = 0.0):
        self.cfg = like.cfg
        self.noise = like.noise
        self.device = like.device
        self.dtype = like.dtype
        self.stats = {}
        self.q_lo, self.q_hi = like.q_lo, like.q_hi
        self._pelvis = like._pelvis
        self.recency_bias = float(recency_bias)
        self.viol_downweight = float(viol_downweight)
        self.viol_uniform_mix = float(viol_uniform_mix)
        if not 0.0 <= self.viol_uniform_mix <= 1.0:
            raise ValueError(
                f"viol_uniform_mix must be in [0, 1], got {self.viol_uniform_mix}"
            )
        self.proficiency = bool(proficiency)
        self.ema_alpha = float(ema_alpha)
        self.capacity = int(capacity)
        # ``ReplayPool`` builds its own state instead of calling
        # ``Phase2Pool.__init__``, so anything the parent sets there must be
        # copied explicitly -- ``n_shaft`` is read by
        # ``strike_context_from_arrays`` when the pool carries a randomized
        # stick, and omitting it is an AttributeError at batch time.
        self.n_shaft = getattr(like, "n_shaft", SHAFT_HIT_SAMPLES)

        src = next(iter(like.data.values()))
        keep = tuple(int(b) for b in (buckets if buckets is not None else like.buckets))
        self.data = {}
        self._ou = {}
        self._n = {b: 0 for b in keep}
        self._pos = {b: 0 for b in keep}
        self._order = {}
        for b in keep:
            d = {}
            for k, v in src.items():
                shape = (self.capacity,) + tuple(v.shape[1:])
                if k == "init":
                    shape = (self.capacity, b, v.shape[-1])
                d[k] = torch.zeros(shape, device=self.device, dtype=v.dtype)
            # The harvested waist-yaw reference, (capacity, b).  Allocated here
            # rather than mirrored from ``src`` because the fresh pool does not
            # carry the column -- it derives the reference from its seed, which
            # the replay pool cannot do (see ``waist_ref_from_seed``).  Only
            # allocated when the roller actually harvests it
            # (``MPCRoller.want_waist_ref``): an always-present column of
            # zeros would otherwise be read by ``Phase2Pool.waist_ref`` as a
            # real reference and silently anchor the replay half at yaw = 0.
            if waist_ref:
                d["waist_ref"] = torch.zeros(
                    (self.capacity, b), device=self.device, dtype=self.dtype
                )
            # Harvested leaky-EMA pitch/roll, (capacity, b, 2), broadcast
            # over the horizon.  Same "do not allocate zeros as a fake
            # reference" rule as ``waist_ref``.
            if waist_pr_ref:
                d["waist_pr_ref"] = torch.zeros(
                    (self.capacity, b, 2), device=self.device, dtype=self.dtype
                )
            if self.viol_downweight > 0.0:
                d["viol_score"] = torch.zeros(
                    (self.capacity,), device=self.device, dtype=self.dtype,
                )
            if self.proficiency:
                d["ema_m"] = torch.full(
                    (self.capacity,), 0.5, device=self.device, dtype=self.dtype,
                )
                d["ema_n"] = torch.zeros(
                    (self.capacity,), device=self.device, dtype=torch.long,
                )
            self.data[b] = d
            self._ou[b] = like._ou[b] if b in like._ou else None
            self._order[b] = torch.zeros(self.capacity, device=self.device, dtype=torch.long)

    #: The replay pool's ``init`` is the roller's shifted previous PLAN, so
    #: deriving a waist-aim reference from it would price the model against
    #: itself.  ``MPCRoller.harvest`` supplies a real ``waist_ref`` column.
    waist_ref_from_seed: bool = False

    def size(self, bucket: int) -> int:
        return int(self._n[int(bucket)])

    def distribution(self) -> dict[int, int]:
        return {b: self.size(b) for b in self.data}

    def __len__(self) -> int:
        return sum(self.size(b) for b in self.data)

    @torch.no_grad()
    def add(self, bucket: int, cols: dict[str, torch.Tensor]) -> None:
        b = int(bucket)
        d = self.data[b]
        n = int(cols["q0"].shape[0])
        if n == 0:
            return
        if "waist_ref" in d and "waist_ref" not in cols:
            # The column exists only when the trainer said the roller harvests
            # it; a wave without the reference would leave zeros behind that
            # the read path cannot tell from a real "hold yaw = 0" anchor.
            raise ValueError(
                "ReplayPool was built with waist_ref=True but this harvest "
                "carries no 'waist_ref' column -- MPCRoller(waist_ref=...) and "
                "ReplayPool(waist_ref=...) must agree"
            )
        if "waist_pr_ref" in d and "waist_pr_ref" not in cols:
            raise ValueError(
                "ReplayPool was built with waist_pr_ref=True but this harvest "
                "carries no 'waist_pr_ref' column -- MPCRoller(waist_pr_ema=...) "
                "and ReplayPool(waist_pr_ref=...) must agree"
            )
        pos = self._pos[b]
        idx = (torch.arange(n, device=self.device) + pos) % self.capacity
        for k, v in d.items():
            if k in cols:
                v.index_copy_(0, idx, cols[k].to(v.dtype))
            elif k == "viol_score" and "viol_score" not in cols:
                v.index_fill_(0, idx, 0.0)
            elif k == "ema_m":
                v.index_fill_(0, idx, 0.5)
            elif k == "ema_n":
                v.index_fill_(0, idx, 0)
        self._order[b].index_copy_(
            0, idx, torch.arange(pos, pos + n, device=self.device, dtype=torch.long)
        )
        self._pos[b] = pos + n
        self._n[b] = min(self._n[b] + n, self.capacity)

    @torch.no_grad()
    def ema_update(
        self, bucket: int, ring_idx: torch.Tensor, y: torch.Tensor,
    ) -> None:
        """In-place, permutation-invariant EMA on physical ring slots.

        Sampling is with replacement, so one slot can occur multiple times in
        a minibatch.  For ``k`` labels on a slot, this applies one batched
        observation using their mean,

        ``m_new = (1 - alpha) * m_old + alpha * mean(y_slot)``,

        and increments ``ema_n`` by ``k``.  Thus duplicate labels have explicit
        minibatch semantics independent of sample order, while the ``k = 1``
        path is identical to the original per-row EMA.
        """
        b = int(bucket)
        if "ema_m" not in self.data[b] or ring_idx.numel() == 0:
            return
        d = self.data[b]
        j = ring_idx.to(device=d["ema_m"].device, dtype=torch.long).reshape(-1)
        yi = y.to(device=d["ema_m"].device, dtype=d["ema_m"].dtype).reshape(-1)
        if j.numel() != yi.numel():
            raise ValueError(
                f"ring_idx and y must have equal size, got {j.numel()} and {yi.numel()}"
            )
        unique_j, inverse, counts = torch.unique(
            j, sorted=True, return_inverse=True, return_counts=True,
        )
        label_sum = torch.zeros(
            unique_j.numel(), device=d["ema_m"].device, dtype=d["ema_m"].dtype,
        )
        label_sum.index_add_(0, inverse, yi)
        label_mean = label_sum / counts.to(dtype=d["ema_m"].dtype)
        alpha = self.ema_alpha
        old_m = d["ema_m"].index_select(0, unique_j)
        d["ema_m"].index_copy_(
            0, unique_j, (1.0 - alpha) * old_m + alpha * label_mean,
        )
        old_n = d["ema_n"].index_select(0, unique_j)
        d["ema_n"].index_copy_(
            0, unique_j, old_n + counts.to(dtype=d["ema_n"].dtype),
        )

    def _row_weights(
        self,
        bucket: int,
        m: int,
        *,
        short_wp: int = 0,
        short_boost: float = 1.0,
        viol_downweight: float | None = None,
    ) -> torch.Tensor:
        """Raw recency × short-boost × harvested-violation sampling weight."""
        b = int(bucket)
        beta = self.viol_downweight if viol_downweight is None else float(viol_downweight)
        if self.recency_bias <= 0.0:
            w = torch.ones(m, device=self.device, dtype=self.dtype)
        else:
            order = self._order[b][:m].to(self.dtype)
            age = (float(self._pos[b] - 1) - order).clamp_min(0.0)
            w = torch.exp(-self.recency_bias * age / float(self.capacity))
        boost = short_wp > 0 and short_boost != 1.0
        if boost:
            hit = self.data[b]["hit"][:m]
            act = self.data[b]["active"][:m]
            big = torch.full_like(hit, b + 1)
            k = torch.where(act, hit.clamp_min(0), big).min(dim=1).values
            w = torch.where(k <= int(short_wp), w * float(short_boost), w)
        if beta > 0.0 and "viol_score" in self.data[b]:
            vs = self.data[b]["viol_score"][:m].to(self.dtype)
            w = w * torch.exp(-beta * vs.clamp_min(0.0))
        return w

    def sample_idx(
        self,
        bucket: int,
        n: int,
        generator: torch.Generator | None = None,
        *,
        short_wp: int = 0,
        short_boost: float = 1.0,
        viol_downweight: float | None = None,
        viol_uniform_mix: float | None = None,
        stratify: bool = False,
        frac_prof: float = 0.40,
        frac_new: float = 0.40,
        ema_nmin: int = 2,
        ema_prof: float = 0.70,
    ):
        """Recency-biased draw, optionally oversampling the final approach.

        The two weights MULTIPLY: recency decides which policy generation a row
        came from, ``short_boost`` decides how much of the batch sits in the
        1..``short_wp`` waypoint band the MPC's executed terminal is drawn from.
        When ``viol_downweight > 0``, multiply by ``exp(-beta * viol_score)``.
        Each eligible subset is then normalized and mixed with uniform using
        ``gamma = viol_uniform_mix``.  With ``stratify`` and EMA columns,
        reserve fractions for proficient / new rows and fill the rest with
        EMA-weighted flex draws; the uniform component is local to each stratum.
        """
        b = int(bucket)
        m = self.size(b)
        if m == 0:
            return None
        beta = self.viol_downweight if viol_downweight is None else float(viol_downweight)
        gamma = (
            float(getattr(self, "viol_uniform_mix", 0.0))
            if viol_uniform_mix is None else float(viol_uniform_mix)
        )
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"viol_uniform_mix must be in [0, 1], got {gamma}")
        boost = short_wp > 0 and short_boost != 1.0
        use_strat = (
            stratify
            and "ema_m" in self.data[b]
            and (frac_prof > 0.0 or frac_new > 0.0)
        )
        if (
            self.recency_bias <= 0.0
            and not boost
            and beta <= 0.0
            and gamma == 0.0
            and not use_strat
        ):
            return torch.randint(0, m, (n,), device=self.device, generator=generator)

        w_all = self._row_weights(
            b, m, short_wp=short_wp, short_boost=short_boost,
            viol_downweight=viol_downweight,
        )
        if not use_strat:
            return _replay_multinomial(
                w_all, n, gamma=gamma, generator=generator,
            )

        ema_m = self.data[b]["ema_m"][:m]
        ema_n = self.data[b]["ema_n"][:m]
        trusted = ema_n >= int(ema_nmin)
        prof_m = trusted & (ema_m >= float(ema_prof))
        new_m = ~trusted
        bad_m = trusted & (ema_m < float(ema_prof))

        k = int(n)
        k_prof = min(int(round(k * float(frac_prof))), k)
        k_new = min(int(round(k * float(frac_new))), k - k_prof)
        k_flex = k - k_prof - k_new

        def _draw(mask: torch.Tensor, count: int, *, scale_ema: bool) -> torch.Tensor | None:
            if count <= 0:
                return None
            ids = mask.nonzero(as_tuple=True)[0]
            if ids.numel() == 0:
                return None
            w = w_all.index_select(0, ids)
            if scale_ema:
                w = w * ema_m.index_select(0, ids).clamp(min=1e-6)
            return ids.index_select(
                0,
                _replay_multinomial(
                    w, count, gamma=gamma, generator=generator,
                ),
            )

        parts: list[torch.Tensor] = []
        for chunk in (
            _draw(prof_m, k_prof, scale_ema=True),
            _draw(new_m, k_new, scale_ema=False),
        ):
            if chunk is not None:
                parts.append(chunk)
        flex_m = bad_m | prof_m
        flex = _draw(flex_m, k_flex, scale_ema=True)
        if flex is not None:
            parts.append(flex)
        if not parts:
            return _replay_multinomial(
                w_all, n, gamma=gamma, generator=generator,
            )
        out = torch.cat(parts)
        if out.numel() < k:
            need = k - int(out.numel())
            top = _replay_multinomial(
                w_all, need, gamma=gamma, generator=generator,
            )
            out = torch.cat([out, top])
        return out[:k]

    def horizon_fracs(self, bucket: int, edges=(1, 3, 5, 8, 12, 25)) -> dict:
        """``{edge: fraction of live rows with nearest active hit <= edge}``."""
        b = int(bucket)
        m = self.size(b)
        if m == 0:
            return {}
        hit, act = self.data[b]["hit"][:m], self.data[b]["active"][:m]
        big = torch.full_like(hit, b + 1)
        k = torch.where(act, hit.clamp_min(0), big).min(dim=1).values
        return {int(e): float((k <= int(e)).float().mean()) for e in edges}


@torch.no_grad()
def _replay_viol_score_telemetry(
    replay: ReplayPool,
    selected: list[torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    """Cheap stored-score quantiles; no trajectory forward rescoring."""

    def _summary(parts: list[torch.Tensor]) -> dict[str, float | int]:
        if not parts:
            return {"n": 0}
        values = torch.cat([v.detach().reshape(-1).float() for v in parts])
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {"n": 0}
        q = torch.quantile(
            values,
            torch.tensor(
                [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
                device=values.device,
                dtype=values.dtype,
            ),
        )
        names = ("p0", "p25", "p50", "p75", "p90", "p99", "p100")
        return {
            "n": int(values.numel()),
            **{name: float(value) for name, value in zip(names, q)},
        }

    whole = [
        replay.data[b]["viol_score"][:replay.size(b)]
        for b in replay.data
        if "viol_score" in replay.data[b] and replay.size(b) > 0
    ]
    return {"whole_ring": _summary(whole), "selected": _summary(selected)}


# ============================================================== the roller ===
class MPCRoller:
    """Batched receding-horizon loop driven by the current model (on GPU).

    With ``regoal=True`` (the default) this reproduces
    :meth:`~amortized_opt.mpc_gen.MPCGenerator.run_episode`'s reset semantics
    rather than the old respawn-on-hit: a hand whose hit fires promotes its own
    next ``future_*`` token **in place**, re-prices its strike row against the
    slot's OWN randomized kit, and re-seeds only its own arm columns.  The other
    hand keeps counting down on the shared waist and the PD state (``q0``,
    ``qd0``, ``prev_cmd``, ``kp``, ``kd``) carries straight through.  A slot is
    respawned from the pool only when the chain actually ends -- no hand has an
    active goal -- or it ages out, or the plant diverges.
    """

    def __init__(
        self,
        pool: Phase2Pool,
        *,
        model,
        sim: BatchedPDSim,
        rnea: BatchedRNEA,
        fk: ChainFK,
        horizon: int = 50,
        slots: int = 256,
        chain_frac: float = 0.35,
        bounce: BounceConfig | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        amp_dtype: torch.dtype | None = None,
        regoal: bool = True,
        cmd_lead_s=0.0,
        library: GoalPoseLibrary | None = None,
        kit: DrumKit | None = None,
        yaw_bins: int = 129,
        max_age: int = 400,
        idle_home=None,
        idle_library=None,
        idle_cylinder=None,
        idle_settle_wp: int = 10,
        rest_ticks: int = 0,
        waist_ref: bool = False,
        n_shaft: int = SHAFT_HIT_SAMPLES,
        waist_upright_ease_wp: int | None = None,
        clips: list | None = None,
        waist_pr_ema: bool = False,
        waist_pr_ema_alpha: float = 0.1,
        waist_pr_ema_leak: float = 0.02,
        decoded_plan_transform=None,
    ):
        #: Optional diagnostic hook applied to every decoder output before the
        #: command is extracted and before the plan is shifted into replay.
        #: Training never supplies it; ``None`` preserves the historical path
        #: without an extra tensor operation.
        self.decoded_plan_transform = decoded_plan_transform
        #: Optional RCC/synth clip list indexed by slot ``clip_i`` provenance.
        #: When set, a promote refills ``future[0]`` / ``schedule[1]`` from
        #: ``clips[clip_i].next_hits``; clip-tail slots are marked for respawn.
        self.clips = list(clips) if clips is not None else None
        #: Harvest a waist-yaw reference alongside every slot, so the replay
        #: half of the batch can price ``CostConfig.w_waist_aim`` too.  Off
        #: unless that weight is non-zero: it is a python loop over the slots
        #: (``aim_yaw`` is a scalar fixed-point solve) and there is no reason to
        #: pay for it on a run that does not use the term.
        self.want_waist_ref = bool(waist_ref)
        #: Leaky EMA of clipped P/R cmd, harvested as ``waist_pr_ref``.
        self.want_waist_pr_ref = bool(waist_pr_ema)
        self.waist_pr_ema_alpha = float(waist_pr_ema_alpha)
        self.waist_pr_ema_leak = float(waist_pr_ema_leak)
        #: The seed's pitch/roll convention, and it MUST follow the pool's
        #: (``ScenarioConfig.waist_upright_ease_wp``): the fresh pool and the
        #: roller together are the training distribution, so a roller that
        #: holds the lean while the pool eases to upright would split it.
        self.waist_ease_wp = (
            None if waist_upright_ease_wp is None else int(waist_upright_ease_wp)
        )
        self.pool = pool
        #: ONE aim law for everything this roller derives from ``waist_aim``:
        #: the harvested reference and the re-goal/respawn ``aim_yaw`` refresh.
        #: It is the POOL's law (``ScenarioConfig.waist_cfg``, default
        #: ``WaistAimConfig()``) -- the fresh pool's seeds and context tokens
        #: were built with it, and two laws in one trainer means the replay
        #: half is priced/anchored against a different torso than the fresh
        #: half (the old harvest used ``deadband=0.0`` while the pool kept
        #: 0.09).
        from .waist_aim import WaistAimConfig

        self.aim_cfg = getattr(pool.cfg, "waist_cfg", None) or WaistAimConfig()
        self.model = model
        self.sim = sim
        self.rnea = rnea
        self.fk = fk
        self.H = int(horizon)
        self.R = int(slots)
        self.chain_frac = float(chain_frac)
        self.bounce = bounce or BounceConfig()
        self.device = torch.device(device or pool.device)
        self.dtype = dtype
        self.amp_dtype = amp_dtype
        self.dt = float(pool.cfg.dt)
        names = tuple(pool.cfg.joint_names)
        self.names = names
        lo, hi = urdf_position_limits(names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        self.step_max = torch.as_tensor(
            isaac_velocity_limits(names) * self.dt, device=self.device, dtype=dtype
        )
        #: Reference lead, as a per-joint fraction of a control period.  The
        #: roller is the training distribution, so this has to equal the lead
        #: the deployed loop executes -- see :mod:`amortized_opt.lead`.
        self.lead_f = lead_fraction(cmd_lead_s, self.dt, len(names),
                                    device=self.device, dtype=dtype)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(seed))
        self.src_buckets = tuple(b for b in pool.buckets if b <= self.H)
        if not self.src_buckets:
            raise ValueError(f"pool has no bucket <= roller horizon {self.H}")

        # ---- in-place re-goal machinery (mirrors MPCGenerator.run_episode) --
        self.regoal = bool(regoal)
        self.max_age = int(max_age)
        self.D = int(pool.cfg.num_drums)
        kit = kit or DrumKit.from_layout()
        self.kit = kit
        self.kit_names = tuple(kit.names) + ("",) * (self.D - kit.num_drums)
        self.pred = drum_name_predicates(
            self.kit_names, device=self.device, dtype=self.dtype
        )
        #: per-drum-index shaft eligibility -- the ``start_shaft`` table the
        #: cost's departure exemption reads when a hand leaves the drum it hit.
        self.shaft_tab = self.pred["shaft"]
        self.arm_cols = {
            h: list(arm_slice(hand, names)) for h, hand in enumerate(_HANDS)
        }
        self.waist_cols = list(waist_slice(names))
        self.pr_idx = torch.as_tensor(
            waist_pr_slice(names), device=self.device, dtype=torch.long
        )
        self.yaw_i = names.index("waist_yaw_joint")
        self.library = library if library is not None else GoalPoseLibrary(joint_names=names)
        self.n_bins = int(yaw_bins)
        self.yaw_lo = float(lo[self.yaw_i])
        self.yaw_step = (float(hi[self.yaw_i]) - self.yaw_lo) / max(self.n_bins - 1, 1)
        self.lib_q, self.lib_ok = self._goal_pose_table()
        self.geom: dict[str, torch.Tensor] = {}

        # ---- the idle arm ---------------------------------------------------
        # ``idle_library`` is an ``IdleCylinderLibrary``; when present, every hand
        # with no active goal is re-seeded toward the cylinder on EVERY tick
        # (hold if tip already inside, else nearest+nudge+quintic).  That keeps
        # the target in front of the decoder for the whole rest stretch.
        self.idle_settle_wp = int(idle_settle_wp)
        self.idle_library = idle_library
        if idle_library is None and idle_home is not None:
            # Legacy IdleHomeTable callers: no cylinder seed path.
            self.idle_library = None
        self.idle_cylinder = idle_cylinder
        if self.idle_library is not None and self.idle_cylinder is None:
            try:
                self.idle_cylinder = derive_cylinder(self.kit, self.fk)
            except Exception:
                self.idle_cylinder = getattr(self.idle_library, "cylinder", None)
        self.idle_cyl_pack = None
        if self.idle_cylinder is not None:
            self.idle_cyl_pack = torch.as_tensor(
                np.asarray(self.idle_cylinder.pack(), float),
                device=self.device, dtype=dtype,
            )
        #: Ticks a BOTH-HANDS-IDLE slot is allowed to keep simulating before it
        #: is respawned.  0 reproduces the historical behaviour, which *deletes*
        #: rest stretches rather than simulating them -- so the idle terms would
        #: otherwise train entirely off-distribution (README, design note step 3).
        self.rest_ticks = int(rest_ticks)
        #: Shaft samples in ``FusedCost.c_points`` (1 tip + n_shaft + 1 wrist),
        #: needed to rebuild a per-sample stick from the context columns.  Read
        #: off the cost rather than assumed, so the two cannot disagree about P.
        self.n_shaft = int(n_shaft)
        #: Same stick+wrist SDF Pool A uses on q0.  Harvest drops post-hit rows
        #: that collide with a drum / under-volume / torso; leftover contact
        #: with ``start_drum`` (the pad just struck) is exempted like a
        #: drum-departure start.
        self.sdf = SdfPenalty(
            fk=self.fk, device=self.device, dtype=self.dtype,
            include_self=True,
        )

        self.state: dict[str, torch.Tensor] = {}
        self._spawn(torch.arange(self.R, device=self.device), carry=None)
        #: Quintic (or pool) inits that still need a 1-step arm DLS after the
        #: encoder P/R rewrite on the next tick.  Shifted decoder plans stay off.
        self._arm_dls_pending = torch.ones(self.R, dtype=torch.bool, device=self.device)
        #: One-shot: slots that just re-goaled or chain-carried.  Harvest
        #: consumes this so the live post-hit pool gets the plant state at
        #: the start of the next strike, not the rest of the trajectory.
        self._post_hit_fresh = torch.zeros(
            self.R, dtype=torch.bool, device=self.device,
        )
        self._post_hit_reject_sdf = 0

    # ------------------------------------------------------- lookup table ---
    def _goal_pose_table(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(2, D, n_bins, 7)`` library arm poses + a ``(2, D)`` validity mask.

        :meth:`GoalPoseLibrary.select` is a host-side dict lookup, so the whole
        (hand, drum, waist-yaw) surface is precomputed once and the re-goal path
        gathers from it.  Pairs the library does not carry -- ``left/floor_tom``
        and ``right/hihat``, the no-crossover rule -- are marked invalid, and a
        promoted future landing on one counts as "no valid future" for that
        hand (it goes idle) exactly as ``generate_cost_scenarios`` rejects them.
        """
        n_arm = len(self.arm_cols[0])
        q = torch.zeros(2, self.D, self.n_bins, n_arm, device=self.device, dtype=self.dtype)
        ok = torch.zeros(2, self.D, device=self.device, dtype=torch.bool)
        yaws = [self.yaw_lo + i * self.yaw_step for i in range(self.n_bins)]
        for h, hand in enumerate(_HANDS):
            for d, name in enumerate(self.kit_names):
                if not name or not self.library.has(hand, name):
                    continue
                ok[h, d] = True
                rows = np.stack(
                    [self.library.select(hand, name, y).arm_q for y in yaws]
                )
                q[h, d] = torch.as_tensor(rows, device=self.device, dtype=self.dtype)
        return q, ok

    # ---------------------------------------------------------- geometry ---
    @torch.no_grad()
    def _refresh_geometry(self, slots: torch.Tensor) -> None:
        """Re-price the ``(D,)`` strike table for ``slots``' (randomized) kits."""
        st = self.state
        tab = strike_geometry_table(
            st["drum_pos"].index_select(0, slots) - self.pool._pelvis,
            st["drum_axis"].index_select(0, slots),
            st["drum_radius"].index_select(0, slots),
            self.kit_names,
            predicates=self.pred,
        )
        if not self.geom:
            for k, v in tab.items():
                self.geom[k] = torch.zeros(
                    (self.R,) + tuple(v.shape[1:]), device=self.device, dtype=v.dtype
                )
        for k, v in tab.items():
            self.geom[k].index_copy_(0, slots, v.contiguous())

    # ------------------------------------------------------------- spawn ---
    def _draw(self, n: int) -> dict[str, torch.Tensor]:
        """``n`` scenarios from the pool, init padded/held out to ``self.H``."""
        sizes = np.array([self.pool.size(b) for b in self.src_buckets], float)
        p = sizes / sizes.sum()
        counts = np.random.default_rng(
            int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen,
                              device=self.device).item())
        ).multinomial(n, p)
        chunks = []
        for b, c in zip(self.src_buckets, counts):
            if c == 0:
                continue
            idx = torch.randint(
                0, self.pool.size(b), (int(c),), device=self.device, generator=self.gen
            )
            d = self.pool.data[b]
            cols = {k: v.index_select(0, idx) for k, v in d.items()}
            if b < self.H:
                pad = cols["init"][:, -1:].expand(-1, self.H - b, -1)
                cols["init"] = torch.cat([cols["init"], pad], dim=1)
            chunks.append(cols)
        return {k: torch.cat([c[k] for c in chunks], 0) for k in chunks[0]}

    @torch.no_grad()
    def _spawn(self, slots: torch.Tensor, carry: dict | None) -> None:
        n = int(slots.numel())
        if n == 0:
            return
        cols = self._draw(n)
        frontier_take = None
        if carry is None:
            cloud = getattr(self, "frontier", None)
            frac = float(getattr(self, "frontier_spawn_frac", 0.0) or 0.0)
            if cloud is not None and frac > 0.0:
                patched = patch_spawn_q0(
                    cols, cloud, frac=frac, generator=self.gen,
                )
                if bool(patched.any()):
                    take = patched.nonzero(as_tuple=True)[0]
                    hit_eff = effective_hits(
                        cols["hit"].index_select(0, take),
                        cols["active"].index_select(0, take),
                        self.H,
                    )
                    per_elem = per_element_lengths(hit_eff, self.names)
                    seed = cols["init"].index_select(0, take)
                    term = seed.gather(
                        1, (per_elem.clamp(1, self.H) - 1).unsqueeze(1),
                    ).squeeze(1)
                    cols["init"].index_copy_(
                        0, take,
                        strike_seed(
                            cols["q0"].index_select(0, take),
                            cols["qd0"].index_select(0, take), term,
                            per_elem, self.H, self.dt,
                            active=cols["active"].index_select(0, take), joint_names=self.names,
                        ),
                    )
                    frontier_take = take
        st = self.state
        if not st:
            for k, v in cols.items():
                shape = (self.R,) + tuple(v.shape[1:])
                st[k] = torch.zeros(shape, device=self.device, dtype=v.dtype)
            st["prev_cmd"] = torch.zeros(self.R, N_DOF, device=self.device, dtype=self.dtype)
            st["age"] = torch.zeros(self.R, device=self.device, dtype=torch.long)
            st["rest"] = torch.zeros(self.R, device=self.device, dtype=torch.long)
            if self.want_waist_pr_ref:
                st["waist_pr_ema"] = torch.zeros(
                    self.R, 2, device=self.device, dtype=self.dtype
                )
            kp, kd, _ki = self.sim.sample_gains(self.R, generator=self.gen)
            st["kp"], st["kd"] = kp, kd
        for k, v in cols.items():
            st[k].index_copy_(0, slots, v.to(st[k].dtype))
        kp, kd, _ki = self.sim.sample_gains(n, generator=self.gen)
        st["kp"].index_copy_(0, slots, kp)
        st["kd"].index_copy_(0, slots, kd)
        st["age"].index_copy_(0, slots, torch.zeros_like(slots))
        st["rest"].index_copy_(0, slots, torch.zeros_like(slots))
        # the drawn scenario brings its own randomized kit, so the per-drum
        # strike table these slots will be re-priced against must follow it.
        self._refresh_geometry(slots)

        if carry is not None:
            # Post-hit respawn: keep the (bounced) PD state, take a NEW goal,
            # and seed the plan with a ramp from where the stick actually is --
            # the drawn scenario's quintic was built for a different start.
            take = carry["take"]
            st["q0"].index_copy_(0, take, carry["q"].to(st["q0"].dtype))
            st["qd0"].index_copy_(0, take, carry["qd"].to(st["qd0"].dtype))
            hit_eff = effective_hits(
                st["hit"].index_select(0, take), st["active"].index_select(0, take), self.H
            )
            per_elem = per_element_lengths(hit_eff, self.names)
            seed = st["init"].index_select(0, take)
            term = seed.gather(1, (per_elem.clamp(1, self.H) - 1).unsqueeze(1)).squeeze(1)
            st["init"].index_copy_(
                0, take,
                strike_seed(
                    st["q0"].index_select(0, take), st["qd0"].index_select(0, take),
                    term, per_elem, self.H, self.dt,
                    active=st["active"].index_select(0, take), joint_names=self.names,
                ),
            )
            # The drawn scenario's ``aim_yaw`` was solved at ITS OWN q0; this
            # slot keeps the carried robot state instead.  Same policy as the
            # re-goal path: re-solve the aim at the CARRIED yaw against the
            # drawn goal (the rows are sparse, so the host-side solve is
            # affordable); a row whose drawn scenario has no active goal keeps
            # ``aim_valid=False`` and the head holds.
            self._refresh_aim(take)
        elif frontier_take is not None:
            # New q0 from the cloud; re-solve aim at the patched yaw like carry.
            self._refresh_aim(slots.index_select(0, frontier_take))
        st["prev_cmd"].index_copy_(0, slots, st["q0"].index_select(0, slots))
        if self.want_waist_pr_ref:
            # New episode: start the EMA at the sampled q0.  Bounce / chain
            # carry must NOT reset it to the new measured lean (that ratifies
            # overshoot).  Save carry rows, write q0, restore.
            saved = None
            take = None if carry is None else carry.get("take")
            if take is not None and int(take.numel()) > 0:
                saved = st["waist_pr_ema"].index_select(0, take).clone()
            q_pr = st["q0"].index_select(0, slots).index_select(1, self.pr_idx)
            st["waist_pr_ema"].index_copy_(0, slots, q_pr)
            if saved is not None:
                st["waist_pr_ema"].index_copy_(0, take, saved)
        if hasattr(self, "_arm_dls_pending"):
            self._arm_dls_pending.index_copy_(
                0, slots, torch.ones_like(slots, dtype=torch.bool)
            )

    # ------------------------------------------------------------ re-goal ---
    @torch.no_grad()
    def _tick_tth(self) -> None:
        """Advance the context's FROM-NOW time tokens by one control tick.

        ``MPCGenerator`` rebuilds ``ContextInputs`` from a fresh
        :class:`~amortized_opt.rhythm.GoalSnapshot` every step, so its
        ``goal_tth`` always agrees with the mask's Fourier tau encodings (which
        are built from ``hit``).  The roller carries its context forward
        instead, so without this every harvested replay row tells the model a
        *stale* time-to-hit while the mask says something else -- a silent
        distribution bug independent of the re-goal fix.  Invalid entries are
        left at their (zero) padding rather than drifting negative.

        ``future_gap`` is deliberately NOT ticked for a hand that has a goal.
        It is HIT-RELATIVE (see
        :attr:`~amortized_opt.tokenizer.ContextInputs.future_gap`), so
        re-snapshotting at ``t + dt`` returns the SAME gaps -- goal and future
        slide together -- and ageing it would reintroduce exactly the
        context/snapshot disagreement this method exists to remove, only in the
        other direction.  The one exception is the from-now fallback: a hand
        with no active goal is anchored at NOW, so its gaps really do age, and
        that ageing is what walks an out-of-range hit into range for
        :meth:`_regoal`'s ``stale`` path.
        """
        st = self.state
        gvalid = st["goal_valid"].to(torch.bool)
        st["goal_tth"] = torch.where(
            gvalid, st["goal_tth"] - self.dt, st["goal_tth"]
        )
        from_now = st["future_valid"].to(torch.bool) & ~gvalid.unsqueeze(-1)
        st["future_gap"] = torch.where(
            from_now, st["future_gap"] - self.dt, st["future_gap"]
        )

    @staticmethod
    def _shift_future(x: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
        """Drop index 0 of the ``(R, 2, F, ...)`` future axis for rows in ``sw``."""
        pad = torch.zeros_like(x[:, :, :1])
        shifted = torch.cat([x[:, :, 1:], pad], dim=2)
        m = sw.view(sw.shape + (1,) * (x.dim() - 2))
        return torch.where(m, shifted, x)

    @torch.no_grad()
    def _separate_promoted(
        self,
        nd: torch.Tensor,      # (R, 2) promoted drum index, clamped to [0, D-1]
        npos: torch.Tensor,    # (R, 2, 3) promoted contact point, ENV frame
        ok: torch.Tensor,      # (R, 2) bool: this hand really promotes
        idle: torch.Tensor,    # (R, 2) bool: this hand drops to idle instead
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slide a promoted contact point off the other hand's, on the head.

        A promoted goal can land on the drum the OTHER hand is already playing,
        within a few millimetres of its contact point -- the pool's own
        same-drum family resamples until the two beads are
        :data:`~amortized_opt.constants.STICK_MIN_SEP` apart
        (``Phase2ScenarioConfig.pair_same_min_sep``), but nothing enforced that
        for a goal promoted mid-episode, so the roller could ask both sticks to
        occupy one point.  That is a geometry the optimizer cannot satisfy and
        the cross-hand collision term would (rightly) refuse.

        The fix keeps the goal and moves the POINT: push it away from the other
        hand's along the head disk to exactly ``STICK_MIN_SEP``, then re-seat it
        on the head -- same axial offset, in-plane radius clamped to the drum's
        own ``hit_radius`` so the pushed point is still a legal hit.  Dropping
        the goal instead would delete exactly the two-handed close-quarters
        states this data is for.

        Returns ``(npos, n_pushed)``.  Rows that are not a same-drum conflict
        get their ``npos`` back BIT FOR BIT (a single :func:`torch.where`, no
        round trip), so this cannot perturb any existing scenario.
        """
        st = self.state
        sep = float(STICK_MIN_SEP)
        flip = lambda x: torch.flip(x, dims=(1,))  # noqa: E731 -- "the other hand"

        gvalid = st["goal_valid"].to(torch.bool)
        # Each hand's state AFTER this tick: a promoting hand takes the promoted
        # goal, everyone else keeps what it had (a hand going idle is not live).
        live = (ok | (st["active"] & gvalid & (st["drum"] >= 0))) & ~idle
        drum_now = torch.where(ok, nd, st["drum"])
        pos_now = torch.where(ok.unsqueeze(-1), npos, st["goal_pos"])

        pelvis = self.pool._pelvis.to(npos.dtype)
        gi = nd.unsqueeze(-1).expand(-1, -1, 3)
        axis = self.geom["axis"].gather(1, gi).to(npos.dtype)          # unit normal
        disk_c = self.geom["disk_center"].gather(1, gi).to(npos.dtype)  # pelvis frame
        hit_r = self.geom["hit_radius"].gather(1, nd).to(npos.dtype).unsqueeze(-1)

        p_self = npos - pelvis
        p_other = flip(pos_now) - pelvis
        d = p_self - p_other
        d_ip = d - (d * axis).sum(-1, keepdim=True) * axis
        gap = d_ip.norm(dim=-1, keepdim=True)

        conflict = (
            ok & flip(live) & (drum_now >= 0) & (drum_now == flip(drum_now))
            & (gap.squeeze(-1) < sep)
        )
        # Direction to push: away from the other bead, in the head plane.  When
        # the two points coincide the direction is degenerate, so fall back to
        # the drum's half-plane direction (always in-plane, and the playable
        # side for a cut head) and, failing that, to any in-plane unit vector.
        hp = self.geom["hp_dir"].gather(1, gi).to(npos.dtype)
        fb = hp - (hp * axis).sum(-1, keepdim=True) * axis
        fbn = fb.norm(dim=-1, keepdim=True)
        alt = torch.zeros_like(axis)
        alt[..., 0] = 1.0
        alt_z = torch.zeros_like(axis)
        alt_z[..., 2] = 1.0
        alt = torch.where(axis[..., 0:1].abs() > 0.9, alt_z, alt)
        alt = alt - (alt * axis).sum(-1, keepdim=True) * axis
        alt = alt / alt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        fb = torch.where(fbn > 1e-9, fb / fbn.clamp_min(1e-30), alt)
        u = torch.where(gap > 1e-9, d_ip / gap.clamp_min(1e-30), fb)

        moved = p_other + sep * u
        rel = moved - disk_c
        rel_ip = rel - (rel * axis).sum(-1, keepdim=True) * axis
        rr = rel_ip.norm(dim=-1, keepdim=True)
        # Stay on the head: in-plane radius clamped to the drum's legal hit
        # disk.  A head too small to hold both beads keeps the rim point, which
        # is the best separation the disk allows -- the same compromise the
        # pool's sampler makes when its 12 redraws cannot separate them.
        rel_ip = rel_ip * torch.where(
            rr > hit_r, hit_r / rr.clamp_min(1e-30), torch.ones_like(rr)
        )
        # The push is a SLIDE along the head: the promoted point keeps its own
        # axial offset (its surface inset), it is not lifted off the drum.
        ax_off = ((p_self - disk_c) * axis).sum(-1, keepdim=True)
        fixed = disk_c + ax_off * axis + rel_ip + pelvis
        return torch.where(conflict.unsqueeze(-1), fixed, npos), conflict.sum()

    @torch.no_grad()
    def _regoal(self) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-hand, in-place goal switch. Returns ``(fired_mask, stats)``.

        Mirrors ``MPCGenerator.run_episode``'s post-hit block: fired hands (and
        hands whose previously out-of-range goal has ticked into the horizon --
        its ``stale`` case) promote their next future token, re-price their
        strike row against the slot's own kit, and re-seed only their own arm
        columns.  Nothing else in the slot moves.
        """
        st = self.state
        H, dt = self.H, self.dt
        active = st["active"]
        fired = active & (st["hit"] <= 0)
        # ---- future[0]'s time FROM NOW --------------------------------------
        # ``future_gap`` is measured from this hand's current goal strike, or
        # from now when it has no goal.  Both the horizon test below and the
        # promoted ``goal_tth`` / ``hit`` index need the from-now time, so undo
        # the anchoring once, here, and derive everything from ``ntth``.
        gvalid0 = st["goal_valid"].to(torch.bool)
        anchor = torch.where(
            gvalid0, st["goal_tth"], torch.zeros_like(st["goal_tth"])
        )                                                          # (R, 2)
        fut_now = torch.where(
            st["future_valid"].to(torch.bool),
            st["future_gap"] + anchor.unsqueeze(-1),
            st["future_gap"],
        )                                                          # (R, 2, F)
        # ``stale``: an idle hand whose next hit was beyond the horizon cap at
        # spawn and has now ticked inside it.  MPCGenerator gets this for free
        # from re-snapshotting; here it is the same promotion path.
        stale = (~active) & st["future_valid"][:, :, 0] & (
            fut_now[:, :, 0] <= H * dt + 1e-9
        )
        switch = fired | stale
        stats = {"regoal": 0.0, "idle": 0.0, "stale": 0.0, "sep_push": 0.0,
                 "clip_tail": 0.0}
        self._pending_clip_tail = None
        if not bool(switch.any()):
            return fired, stats

        # ---- departure exemption: the drum this hand has just struck --------
        st["start_drum"] = torch.where(fired, st["drum"], st["start_drum"])
        st["start_shaft"] = torch.where(
            fired, self.shaft_tab[st["drum"].clamp_min(0)], st["start_shaft"]
        )

        # ---- promote future[0] ----------------------------------------------
        nd = st["future_drum"][:, :, 0].clamp(0, self.D - 1)
        npos, nvel = st["future_pos"][:, :, 0], st["future_vel"][:, :, 0]
        ntth, nval = fut_now[:, :, 0], st["future_valid"][:, :, 0]
        hand_ix = torch.arange(2, device=self.device).view(1, 2).expand_as(nd)
        lib = self.lib_ok[hand_ix, nd]
        # ``goal_snapshot(horizon_cap=H*dt)`` keeps a hit that is further away
        # than the horizon can represent OUT of the goal slot and leaves it in
        # ``future``; the hand is idle until it ticks into range.  Mirror that
        # or the promoted goal_tth and the mask's tau encodings disagree again.
        in_range = ntth <= H * dt + 1e-9
        ok = switch & nval & lib & in_range
        idle = switch & ~ok
        # a token is CONSUMED when it becomes the goal, or when the library
        # cannot seed it at all (dropping it stops the hand retrying forever);
        # one that is merely still out of range stays put for the stale path.
        consume = ok | (switch & nval & ~lib)
        okv, idlev = ok.unsqueeze(-1), idle.unsqueeze(-1)

        # ---- same-drum separation guard --------------------------------------
        # BEFORE anything is written: a promoted goal landing on the drum the
        # other hand is currently playing, within ``STICK_MIN_SEP`` of its
        # contact point, has its point slid along the head instead of being
        # dropped.  Everything below (the strike re-price, ``goal_pos``, the
        # aim refresh) then reads the separated point, so the context, the goal
        # and the cost all describe the same geometry.  Rows with no conflict
        # keep ``npos`` bit for bit, so no existing scenario moves.
        npos, n_sep = self._separate_promoted(nd, npos, ok, idle)

        st["goal_drum"] = torch.where(ok, nd, torch.where(idle, torch.zeros_like(nd),
                                                          st["goal_drum"]))
        z3 = torch.zeros_like(npos)
        st["goal_pos"] = torch.where(okv, npos, torch.where(idlev, z3, st["goal_pos"]))
        st["goal_vel"] = torch.where(okv, nvel, torch.where(idlev, z3, st["goal_vel"]))
        st["goal_tth"] = torch.where(
            ok, ntth, torch.where(idle, torch.zeros_like(ntth), st["goal_tth"])
        )
        st["goal_valid"] = (st["goal_valid"] | ok) & ~idle
        st["drum"] = torch.where(
            ok, nd, torch.where(idle, torch.full_like(nd, -1), st["drum"])
        )
        # A fresh horizon is at least 2 waypoints: a 1-waypoint goal would fire
        # again on the very next tick before the plan could act on it.
        new_hit = torch.round(ntth / dt).long().clamp(2, H)
        st["hit"] = torch.where(
            ok, new_hit, torch.where(idle, torch.full_like(new_hit, -1), st["hit"])
        )
        st["active"] = (st["active"] | ok) & ~idle

        # ---- re-price the strike row against THIS slot's own kit ------------
        g = self.geom
        gi = nd.unsqueeze(-1).expand(-1, -1, 3)
        axis = g["axis"].gather(1, gi)
        disk_c = g["disk_center"].gather(1, gi)
        p_star = npos - self.pool._pelvis
        vn = nvel.norm(dim=-1, keepdim=True)
        n_out = torch.where(vn > 1e-9, -nvel / vn.clamp_min(1e-30), axis)
        off = disk_c - p_star
        off = off - (off * axis).sum(-1, keepdim=True) * axis
        e_n = torch.zeros_like(n_out)
        e_n[..., 2] = 1.0
        rows3 = {"p_star": (p_star, z3), "v_star": (nvel, z3), "n_out": (n_out, e_n),
                 "disk_off": (off, z3), "hp_dir": (g["hp_dir"].gather(1, gi), z3)}
        for k, (v_ok, v_idle) in rows3.items():
            st[k] = torch.where(okv, v_ok, torch.where(idlev, v_idle, st[k]))
        empty1 = {"hit_radius": 0.08, "has_hp": 0.0, "sigma": float(STICK_TIP_SIGMA),
                  "shaft": 0.0, "elev": 0.0}
        for k, v_idle in empty1.items():
            src = g[k].gather(1, nd)
            st[k] = torch.where(
                ok, src, torch.where(idle, torch.full_like(src, v_idle), st[k])
            )

        # ---- consume the promoted token, and RE-ANCHOR what survives --------
        # ``future_gap`` is stated relative to the hand's current goal, and the
        # switch just moved that goal, so the surviving slots have to be
        # re-expressed or the whole new segment carries the PREVIOUS strike's
        # timing.  Shifting the from-now form and re-subtracting the new anchor
        # makes every case the same line of algebra: a promoted goal
        # (anchor := its own strike, so gaps become ``old[j+1] - old[0]``), a
        # hand dropped to idle (anchor := now), and the out-of-range hand that
        # keeps its token but loses its goal (anchor := now, nothing consumed).
        # Pre-shift goal-relative gap at slot 0: absolute clip time of the
        # promoted hit is ``hand_hit_time + gap0`` when the prior goal was valid.
        gap0_saved = st["future_gap"][:, :, 0].clone()
        for f in ("future_drum", "future_pos", "future_vel", "future_valid"):
            st[f] = self._shift_future(st[f], consume)
        if "sched_drum" in st:
            for f in ("sched_drum", "sched_pos", "sched_vel", "sched_dt", "sched_valid"):
                if f in st:
                    st[f] = self._shift_future(st[f], consume)
        fut_now = self._shift_future(fut_now, consume)
        anchor_new = torch.where(
            st["goal_valid"].to(torch.bool), st["goal_tth"],
            torch.zeros_like(st["goal_tth"]),
        )
        regap = torch.where(
            st["future_valid"].to(torch.bool),
            fut_now - anchor_new.unsqueeze(-1),
            torch.zeros_like(fut_now),
        )
        # Untouched hands keep their stored gap BIT FOR BIT: their anchor never
        # moved, and a +anchor/-anchor round trip is not the identity in fp32.
        st["future_gap"] = torch.where(
            switch.unsqueeze(-1), regap, st["future_gap"]
        )

        # ---- clip provenance: refill future / schedule after promote --------
        # With ``num_future=1`` the shift above empties future[0]; Phase-B /
        # roller continuity needs the next beat from ``clips[clip_i]``.  Rows
        # without provenance (legacy pools) keep the shifted futures as-is.
        clip_tail = self._refill_from_clip_provenance(ok, gvalid0, gap0_saved)

        # ---- refresh the waist-aim context for the switched rows -------------
        # ``aim_yaw`` was computed by ``build_context_inputs`` for the goals
        # this slot was SEEDED with; after a switch it describes the wrong
        # drum, and merely marking it invalid (the old policy) degraded nearly
        # every replay anchor to "hold" in head mode.  The switched rows are
        # sparse -- a few per tick -- so the same host-side ``aim_yaw`` solve
        # ``MPCGenerator`` runs is affordable here, AFTER the strike re-price
        # above so it sees the promoted goal's own pelvis-frame ``p_star``.
        switched = (ok | idle).any(dim=1).nonzero(as_tuple=True)[0]
        self._refresh_aim(switched)

        # ---- the shared waist first, so the strike-knot waist is known -------
        # Matches Phase-B distill / warmstart: fired arm reseeds, other arm
        # leftover, waist from ``waist_reference_block`` toward the new aim --
        # not only when both hands fire.  Pitch/roll ease to the encoder SP
        # when the head is on, else the pool's ease/hold convention.
        any_ok = ok.any(dim=1).nonzero(as_tuple=True)[0]
        self._rebuild_waist_from_aim(any_ok)

        # ---- re-seed ONLY the switched hand's arm columns --------------------
        # Library at the AIM yaw (the yaw the waist ramp is going to), then
        # one-shot DLS against the post-rebuild strike-knot waist, then rebuild
        # the arm quintic.  Binning current yaw then ramping to a new aim left
        # the terminal a half-bin off after a new hit.
        yaw_src = torch.where(
            st["aim_valid"].to(torch.bool), st["aim_yaw"], st["q0"][:, self.yaw_i],
        )
        yaw_bin = torch.round(
            (yaw_src - self.yaw_lo) / self.yaw_step
        ).long().clamp(0, self.n_bins - 1)
        n = len(self.names)
        for h in (0, 1):
            rows = ok[:, h].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = self.arm_cols[h]
            hand = _HANDS[h]
            q0r = st["q0"].index_select(0, rows)
            d = st["drum"].index_select(0, rows)[:, h].clamp(0, self.D - 1)
            term_lib = self.lib_q[h][d, yaw_bin.index_select(0, rows)]
            ok_lib = self.lib_ok[h][d]
            term_lib = torch.where(ok_lib.unsqueeze(-1), term_lib, q0r[:, cols])
            k = st["hit"].index_select(0, rows)[:, h].clamp(1, self.H)
            buf = st["init"].index_select(0, rows)
            q_seed = buf.gather(
                1, (k - 1).view(-1, 1, 1).expand(-1, 1, n)
            ).squeeze(1).clone()
            q_seed[:, cols] = term_lib
            aim = st["goal_pos"].index_select(0, rows)[:, h]
            q_fix = adjust_arm_terminal(
                self.fk, q_seed, aim, hand,
                steps=1, q_bounds=(self.q_lo, self.q_hi),
                joint_names=self.names,
            )
            term = q_fix[:, cols]
            sub = accelerating_path_torch(
                q0r[:, cols], st["qd0"].index_select(0, rows)[:, cols],
                term, k, self.H, self.dt,
            ).clamp(self.q_lo[cols], self.q_hi[cols])
            buf = buf.clone()
            buf[:, :, cols] = sub
            st["init"].index_copy_(0, rows, buf)

        # Next tick's encode may rewrite P/R to a new SP; 1-shot arms then.
        if hasattr(self, "_arm_dls_pending") and bool(ok.any()):
            self._arm_dls_pending |= ok.any(dim=1)

        stats["regoal"] = float(ok.sum())
        stats["idle"] = float(idle.sum())
        stats["stale"] = float((stale & ok).sum())
        stats["sep_push"] = float(n_sep)
        stats["clip_tail"] = float(clip_tail.sum())
        self._pending_clip_tail = clip_tail
        # On-policy post-hit inits: a hand that just struck and promoted a
        # real next goal.  Harvest consumes ``_post_hit_fresh`` so only THIS
        # plant state (q0/qd0 after the hit, init re-ramped toward the new
        # drum) enters the live KIND_POST_HIT buffer.
        ph = ok.any(dim=1)
        self._post_hit_fresh |= ph
        if "kind" in st and bool(ph.any()):
            st["kind"] = torch.where(
                ph, torch.full_like(st["kind"], KIND_POST_HIT), st["kind"],
            )
        return fired, stats

    @torch.no_grad()
    def _refill_from_clip_provenance(
        self,
        ok: torch.Tensor,
        gvalid0: torch.Tensor,
        gap0_saved: torch.Tensor,
    ) -> torch.Tensor:
        """Refill future[0] / schedule[1] from clip provenance; return clip-tail mask.

        No-op (all-False mask) when the roller has no clip list or the slot
        lacks ``clip_i`` / ``hand_hit_time``.  A promote that cannot refill an
        empty future under ``num_future=1`` marks the slot for respawn.
        """
        st = self.state
        R = st["q0"].shape[0]
        clip_tail = torch.zeros(R, dtype=torch.bool, device=self.device)
        if self.clips is None or "clip_i" not in st or "hand_hit_time" not in st:
            return clip_tail
        if not bool(ok.any()):
            return clip_tail

        from .rhythm_chain_data import refill_lookahead_from_clip

        F = int(st["future_valid"].shape[2])
        has_sched = "sched_valid" in st
        # Prefer promoting leftover schedule[0] (post-shift = old schedule[1])
        # into future when the shift left future empty; then refill schedule
        # extras from the clip.
        for h, hand in enumerate(_HANDS):
            rows = ok[:, h].nonzero(as_tuple=True)[0]
            for r in rows.tolist():
                ci = int(st["clip_i"][r].item())
                if ci < 0 or ci >= len(self.clips):
                    continue
                # Absolute clip time of the NEW goal.
                if bool(gvalid0[r, h]):
                    new_t = float(st["hand_hit_time"][r, h]) + float(gap0_saved[r, h])
                else:
                    # Stale promote: no prior goal anchor; keep stored time if
                    # any, else skip provenance update (cannot recover abs time).
                    new_t = float(st["hand_hit_time"][r, h])
                    if not np.isfinite(new_t):
                        continue
                st["hand_hit_time"][r, h] = new_t

                # If post-shift future[0] is already valid (F>1 leftover), only
                # refill schedule extras when present; else require a future.
                need_future = not bool(st["future_valid"][r, h, 0])
                if has_sched and need_future and bool(st["sched_valid"][r, h, 0]):
                    # schedule[0] after shift = old schedule[1]; project to future
                    st["future_drum"][r, h, 0] = st["sched_drum"][r, h, 0]
                    st["future_pos"][r, h, 0] = st["sched_pos"][r, h, 0]
                    st["future_vel"][r, h, 0] = st["sched_vel"][r, h, 0]
                    st["future_gap"][r, h, 0] = st["sched_dt"][r, h, 0]
                    st["future_valid"][r, h, 0] = True
                    need_future = False

                n_extra = 1 if has_sched else 0
                fields, filled = refill_lookahead_from_clip(
                    self.kit,
                    self.clips[ci],
                    hand,
                    new_goal_hit_time=new_t,
                    n_future=1 if need_future else 0,
                    n_schedule_extra=n_extra,
                )
                if need_future:
                    if not filled:
                        clip_tail[r] = True
                        continue
                    st["future_drum"][r, h, 0] = int(fields["future_drum"][0])
                    st["future_pos"][r, h, 0] = torch.as_tensor(
                        fields["future_pos"][0], device=self.device, dtype=st["future_pos"].dtype
                    )
                    st["future_vel"][r, h, 0] = torch.as_tensor(
                        fields["future_vel"][0], device=self.device, dtype=st["future_vel"].dtype
                    )
                    st["future_gap"][r, h, 0] = float(fields["future_gap"][0])
                    st["future_valid"][r, h, 0] = True
                if has_sched:
                    # Mirror future[0] into schedule[0]; put the next clip hit
                    # in schedule[1] when available.
                    st["sched_drum"][r, h, 0] = st["future_drum"][r, h, 0]
                    st["sched_pos"][r, h, 0] = st["future_pos"][r, h, 0]
                    st["sched_vel"][r, h, 0] = st["future_vel"][r, h, 0]
                    st["sched_dt"][r, h, 0] = st["future_gap"][r, h, 0]
                    st["sched_valid"][r, h, 0] = st["future_valid"][r, h, 0]
                    if st["sched_valid"].shape[2] > 1 and "sched_valid" in fields:
                        if bool(fields["sched_valid"][0]):
                            st["sched_drum"][r, h, 1] = int(fields["sched_drum"][0])
                            st["sched_pos"][r, h, 1] = torch.as_tensor(
                                fields["sched_pos"][0], device=self.device,
                                dtype=st["sched_pos"].dtype,
                            )
                            st["sched_vel"][r, h, 1] = torch.as_tensor(
                                fields["sched_vel"][0], device=self.device,
                                dtype=st["sched_vel"].dtype,
                            )
                            st["sched_dt"][r, h, 1] = float(fields["sched_dt"][0])
                            st["sched_valid"][r, h, 1] = True
                        else:
                            st["sched_valid"][r, h, 1] = False
                # With F=1, a promote that left future empty and could not
                # refill is a clip tail even if we somehow skipped above.
                if F <= 1 and not bool(st["future_valid"][r, h, 0]):
                    clip_tail[r] = True
        return clip_tail

    @torch.no_grad()
    def _rebuild_waist_from_aim(self, rows: torch.Tensor) -> None:
        """Rewrite waist ``init`` columns for ``rows`` from refreshed ``aim_yaw``.

        Yaw follows :func:`~amortized_opt.waist_aim.waist_reference_block` (same
        as ``warmstart_trajectory``); pitch/roll ease to the encoder set-point
        when the P/R head is on, else hold the measured lean or ease to
        upright under ``waist_ease_wp``.  Sparse host-side loop -- only the
        switched rows.
        """
        if rows.numel() == 0:
            return
        from .limits import ETA_VELOCITY, isaac_velocity_limits
        from .waist_aim import waist_reference_block

        st = self.state
        wc = self.waist_cols
        H, dt = self.H, self.dt
        waist_names = [self.names[c] for c in wc]
        jy = waist_names.index("waist_yaw_joint")
        v_lim = isaac_velocity_limits(self.names)
        yaw_step = float(ETA_VELOCITY) * float(v_lim[self.yaw_i]) * float(dt)

        q0 = st["q0"].index_select(0, rows)
        qd0 = st["qd0"].index_select(0, rows)
        aim = st["aim_yaw"].index_select(0, rows)
        aval = (
            st["aim_valid"].index_select(0, rows).to(torch.bool)
            if "aim_valid" in st
            else torch.ones(rows.numel(), dtype=torch.bool, device=self.device)
        )
        q0_np = q0.cpu().numpy()
        qd_np = qd0.cpu().numpy()
        aim_np = aim.cpu().numpy()
        aval_np = aval.cpu().numpy()

        block = np.zeros((int(rows.numel()), H, len(wc)), dtype=np.float64)
        for i in range(int(rows.numel())):
            yaw_start = float(q0_np[i, self.yaw_i])
            yaw_tgt = float(aim_np[i]) if bool(aval_np[i]) else yaw_start
            if self.waist_ease_wp is not None:
                sp = np.zeros(len(wc), dtype=float)
            else:
                sp = np.array([float(q0_np[i, c]) for c in wc], dtype=float)
            # Keep yaw setpoint at the measured start; the block overwrites yaw.
            sp[jy] = yaw_start
            rate = float(qd_np[i, self.yaw_i])
            ref = waist_reference_block(
                waist_names,
                yaw_start=yaw_start,
                yaw_target=yaw_tgt,
                n_waypoints=H + 1,
                setpoint=sp,
                dt=dt,
                settle_wp=3,
                yaw_rate_start=rate,
                max_move_s=0.2,
            )
            # Slew-limit yaw (warmstart twin).
            yaw = ref[:, jy].copy()
            for k in range(1, yaw.shape[0]):
                d = yaw[k] - yaw[k - 1]
                if d > yaw_step:
                    yaw[k] = yaw[k - 1] + yaw_step
                elif d < -yaw_step:
                    yaw[k] = yaw[k - 1] - yaw_step
            ref[:, jy] = yaw
            block[i] = ref[1:]  # drop measured knot 0 -- init is knots 1..H

        sub = torch.as_tensor(block, device=self.device, dtype=self.dtype)
        buf = st["init"].index_select(0, rows)
        buf[:, :, wc] = sub
        # Pitch/roll: encoder SP when the head is on (yaw already ramped to
        # aim); else the pool's ease-to-0 / hold convention.
        pr_sp = self._pr_sp_for_rows(rows)
        if pr_sp is not None:
            buf = seed_waist_pr_to_sp(
                buf, q0, qd0, pr_sp,
                joint_names=self.names, dt=self.dt, ease_wp=self.waist_ease_wp,
            )
        elif self.waist_ease_wp is not None:
            q0w = q0[:, wc]
            v0w = torch.zeros_like(q0w)
            term = q0w.clone()
            pr = [j for j, c in enumerate(wc) if c != self.yaw_i]
            term[:, pr] = 0.0
            step = (ETA_VELOCITY * self.step_max[wc]).clamp_min(1e-9)
            cap = step / self.dt
            v0w[:, pr] = torch.maximum(torch.minimum(qd0[:, wc], cap), -cap)[:, pr]
            need = torch.ceil(1.875 * (term - q0w).abs() / step)
            L = need.clamp(float(self.waist_ease_wp), float(self.H))
            pr_sub = vel_ramp_seed(q0w, v0w, term, L, self.H, self.dt)
            d0 = (pr_sub[:, 0, :] - q0w).abs().unsqueeze(1)
            peak = torch.cat([d0, (pr_sub[:, 1:] - pr_sub[:, :-1]).abs()], 1).amax(1)
            ratio = peak / step
            over = ratio > 1.0
            if bool(over.any()):
                L = torch.where(over, (L * ratio).ceil().clamp(1.0, float(self.H)), L)
                pr_sub = vel_ramp_seed(q0w, v0w, term, L, self.H, self.dt)
            for j in pr:
                buf[:, :, wc[j]] = pr_sub[:, :, j]
        st["init"].index_copy_(0, rows, buf)

    def _pr_sp_for_rows(self, rows: torch.Tensor):
        """Encoder pitch/roll set-point for ``rows``, or None when the head is off."""
        core = self.model.module if hasattr(self.model, "module") else self.model
        if not getattr(getattr(core, "cfg", None), "waist_pr_sp_head", False):
            return None
        from .tokenizer import ContextInputs

        st = self.state
        ctx = ContextInputs(
            **{f: st[f].index_select(0, rows) for f in CONTEXT_FIELDS if f in st}
        )
        return core.encode_context(ctx).waist_pr_sp

    # ----------------------------------------------------------- idle arm ---
    @torch.no_grad()
    def _seed_idle(self) -> float:
        """Re-seed every IDLE hand: hold inside cylinder, else nearest+nudge+ramp.

        GPU/host twin of :func:`~amortized_opt.warmstart.warmstart_trajectory`'s
        idle-cylinder branch.  Refreshes ``idle_cyl`` / ``idle_valid`` on the
        context every tick.

        Returns the number of hand-slots seeded, for the tick's stats.
        """
        if self.idle_library is None or self.idle_cylinder is None:
            return 0.0
        from .idle_cylinder import (
            nearest_arm_pose,
            nudge_into_cylinder,
            tip_in_torso,
            tip_inside,
        )

        st = self.state
        R = st["q0"].shape[0]
        idle_v = ~st["active"]
        if "idle_valid" in st:
            m = idle_v.unsqueeze(-1).to(self.dtype)
            st["idle_valid"] = idle_v
            if self.idle_cyl_pack is not None and "idle_cyl" in st:
                st["idle_cyl"] = (
                    self.idle_cyl_pack.view(1, 1, 8).expand(R, 2, 8) * m
                )
        n = 0.0
        lo = self.q_lo.detach().cpu().numpy()
        hi = self.q_hi.detach().cpu().numpy()
        for h, hand in enumerate(_HANDS):
            rows = (~st["active"][:, h]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = self.arm_cols[h]
            q0_full = st["q0"].index_select(0, rows)
            tips = tip_in_torso(self.fk, q0_full, hand).detach().cpu().numpy()
            inside = np.asarray(tip_inside(self.idle_cylinder, tips), bool).reshape(-1)
            q_arm = q0_full[:, cols].detach().cpu().numpy()
            lib_q = self.idle_library.arm_poses(hand)
            term = q_arm.copy()
            out_i = np.where(~inside)[0]
            if out_i.size:
                q_batch = q0_full[out_i].detach().cpu().numpy().copy()
                for j, oi in enumerate(out_i):
                    _k, q_near = nearest_arm_pose(
                        lib_q, q_arm[oi], shoulder_weight=10.0
                    )
                    q_batch[j, cols] = q_near
                nudged = nudge_into_cylinder(
                    self.fk, q_batch, hand, self.idle_cylinder,
                    shoulder_weight=10.0,
                    joint_names=self.names,
                    q_bounds=(lo, hi),
                )
                for j, oi in enumerate(out_i):
                    term[oi] = nudged[j, cols]
            term_t = torch.as_tensor(term, device=self.device, dtype=self.dtype)
            q0r = q0_full[:, cols]
            # Hold rows: L = 1 (immediate hold at current).  Outside rows: settle.
            step = (ETA_VELOCITY * self.step_max[cols]).clamp_min(1e-9)
            need = torch.ceil(1.875 * (term_t - q0r).abs() / step).amax(dim=1)
            L = need.clamp(self.idle_settle_wp, self.H).long()
            hold = torch.as_tensor(inside, device=self.device)
            L = torch.where(hold, torch.ones_like(L), L)
            sub = ramp_seed(q0r, term_t, L.view(-1, 1).expand(-1, len(cols)), self.H)
            # Exact hold for already-inside tips (bit-stable vs quintic of length 1).
            # ``ramp_seed`` returns (B, H, n) waypoints 1..H — not H+1.
            if bool(hold.any()):
                sub = sub.clone()
                sub[hold] = q0r[hold].unsqueeze(1).expand(-1, self.H, -1)
            buf = st["init"].index_select(0, rows)
            buf[:, :, cols] = sub
            st["init"].index_copy_(0, rows, buf)
            n += float(rows.numel())
        return n

    # -------------------------------------------------------------- tick ---
    def masks(self):
        st = self.state
        hit_eff = effective_hits(st["hit"], st["active"], self.H)
        t = hit_eff.to(self.dtype) * self.dt
        return build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=self.H, dt=self.dt,
            active_left=st["active"][:, 0], active_right=st["active"][:, 1],
            joint_names=self.names,
        ), hit_eff

    def _aim_row_inputs(self, rows: torch.Tensor | None = None):
        """Host-side ``aim_yaw`` inputs for ``rows`` (all rows when None).

        Returns a callable ``row(i) -> (targets, goal_tth, futures)`` over the
        selected rows, in the PELVIS frame the aim law's shoulder offsets live
        in.  The futures are the context's own tokens, un-anchored back to
        FROM-NOW time the way :meth:`_regoal` does it (``future_gap`` is stated
        relative to the hand's current goal strike, or to now when it has
        none) -- the aim law's clock is from-now, and re-deriving it here is
        what keeps the roller's aim the same function of the schedule that
        ``rhythm.build_context_inputs`` computes for the fresh pool.
        """
        st = self.state
        sel = (lambda k: st[k]) if rows is None else (
            lambda k: st[k].index_select(0, rows)
        )
        p_star = sel("p_star").cpu().numpy()
        active = sel("active").cpu().numpy()
        drum = sel("drum").cpu().numpy()
        has_fut = "future_pos" in st and "future_gap" in st and "future_valid" in st
        gtth = sel("goal_tth").cpu().numpy() if "goal_tth" in st else None
        if has_fut and gtth is not None:
            gvalid = sel("goal_valid").cpu().numpy() > 0.5
            anchor = np.where(gvalid, gtth, 0.0)                    # (n, 2)
            f_pos = (sel("future_pos") - self.pool._pelvis).cpu().numpy()
            f_now = sel("future_gap").cpu().numpy() + anchor[:, :, None]
            f_val = sel("future_valid").cpu().numpy() > 0.5
        else:
            f_pos = f_now = f_val = None

        def row(i: int):
            targets: dict[str, np.ndarray] = {}
            tth: dict[str, float] = {}
            futures: dict[str, list] = {}
            for hi, hand in enumerate(_HANDS):
                if active[i, hi] > 0.5 and drum[i, hi] >= 0:
                    targets[hand] = np.asarray(p_star[i, hi], dtype=float)
                    tth[hand] = 0.0 if gtth is None else float(gtth[i, hi])
                if f_val is not None:
                    futures[hand] = [
                        (np.asarray(f_pos[i, hi, j], dtype=float), float(f_now[i, hi, j]))
                        for j in range(f_val.shape[2])
                        if f_val[i, hi, j]
                    ]
            return targets, tth, (futures or None)

        return row

    @torch.no_grad()
    def _refresh_aim(self, rows: torch.Tensor) -> None:
        """Recompute the waist-aim CONTEXT tokens for ``rows``, host-side.

        The same law (``self.aim_cfg``, the pool's) and frame handling as
        :meth:`_waist_aim_reference` and ``MPCGenerator``: ``aim_yaw`` over the
        row's ACTIVE goal positions AND its future tokens in the PELVIS frame
        (``st["p_star"]`` is already pelvis-rooted; the futures are lifted by
        :meth:`_aim_row_inputs`), evaluated at the row's current yaw.  Called
        only for the sparse re-goal / carried-respawn sets -- a few rows per
        tick -- so the scalar fixed-point solve stays off the hot path.  A row
        with nothing upcoming at all keeps ``aim_valid=False``: "hold where you
        are" is the correct anchor at a rest, and only there.
        """
        st = self.state
        if "aim_valid" not in st or rows.numel() == 0:
            return
        from .waist_aim import aim_yaw

        yaw = st["q0"].index_select(0, rows)[:, self.yaw_i]
        # ``yaw_now`` means "where the torso is NOW" -- refresh unconditionally
        # (the tick-level refresh also keeps it current between events).
        st["yaw_now"].index_copy_(0, rows, yaw.to(st["yaw_now"].dtype))
        row = self._aim_row_inputs(rows)
        yaw_np = yaw.cpu().numpy()
        n = int(rows.numel())
        aim = np.zeros(n, dtype=float)
        ok = np.zeros(n, dtype=bool)
        for i in range(n):
            targets, tth, futures = row(i)
            # A hand whose goal the horizon cap demoted still votes through its
            # futures, so ``aim_valid`` now follows "this row has SOMETHING
            # upcoming" rather than "this row has an active goal" -- a rest is
            # the only case that still holds.
            if not targets and not any((futures or {}).values()):
                continue
            with np.errstate(all="ignore"):
                aim[i] = aim_yaw(
                    targets, cfg=self.aim_cfg, yaw_current=float(yaw_np[i]),
                    goal_tth=tth, futures=futures,
                )
            ok[i] = True
        st["aim_yaw"].index_copy_(
            0, rows,
            torch.as_tensor(aim, device=self.device).to(st["aim_yaw"].dtype),
        )
        # the pool batches context bools through the float pipeline, so the
        # stored mask may be 0.0/1.0 rather than bool -- keep its dtype
        st["aim_valid"].index_copy_(
            0, rows,
            torch.as_tensor(ok, device=self.device).to(st["aim_valid"].dtype),
        )

    def _waist_aim_reference(self) -> torch.Tensor | None:
        """``(R, H)`` yaw reference per slot from ``waist_aim``, or None.

        Row ``k`` is KNOT ``k+1`` -- the rows-1..H convention documented on
        :attr:`StrikeContext.waist_yaw_ref` and shared with the fresh pool's
        seed yaw column (``WarmStart.init_traj`` drops the measured knot 0), so
        the cost reads both producers at the same knots.  The law is
        ``self.aim_cfg``, the POOL's, for the same single-source reason.
        """
        st = self.state
        H = self.H
        R = st["q0"].shape[0]
        q0 = st["q0"].cpu().numpy()
        row = self._aim_row_inputs()
        yaw_i = self.yaw_i

        from amortized_opt.waist_aim import aim_yaw, yaw_reference

        aim_cfg = self.aim_cfg
        ref = torch.zeros(R, H, dtype=self.dtype, device=self.device)
        for i in range(R):
            yaw_start = float(q0[i, yaw_i])
            targets, tth, futures = row(i)
            if not targets and not any((futures or {}).values()):
                yaw_target = yaw_start
            else:
                with np.errstate(all="ignore"):
                    yaw_target = aim_yaw(targets, cfg=aim_cfg, yaw_current=yaw_start,
                                         goal_tth=tth, futures=futures)
            # H+1 waypoints anchored at the measured q0, then DROP knot 0 --
            # exactly how the fresh pool's seed becomes ``init_traj``.
            yaw_traj = yaw_reference(
                yaw_start, yaw_target, H + 1, settle_wp=3, yawd_start=0.0,
                max_move_wp=10,
            )[1:]
            ref[i] = torch.as_tensor(yaw_traj, dtype=self.dtype, device=self.device)
        return ref

    def strike_context(self) -> StrikeContext:
        st = self.state
        _m, hit_eff = self.masks()
        heads = st["drum_pos"] - self.pool._pelvis
        s_pts = s_ax = s_sig = None
        if "stick_valid" in st:
            s_pts, s_ax, s_sig = stick_cost_from_context(
                st["stick_axis"].to(self.dtype), st["stick_tip"].to(self.dtype),
                st["stick_length"].to(self.dtype), st["stick_valid"], self.n_shaft,
            )
        return StrikeContext(
            stick_points_local=s_pts, stick_axis_local=s_ax, stick_tip_sigma=s_sig,
            hit_index=hit_eff, active=st["active"].to(self.dtype),
            target_drum=st["drum"], start_drum=st["start_drum"],
            p_star=st["p_star"], v_star=st["v_star"], n_out=st["n_out"],
            hit_radius=st["hit_radius"], disk_offset=st["disk_off"],
            half_plane_dir=st["hp_dir"], has_half_plane=st["has_hp"],
            contact_sigma=st["sigma"], shaft_allowed=st["shaft"],
            start_shaft_allowed=st["start_shaft"], elev_required=st["elev"],
            drum_head_centers=heads,
            # The slot's OWN randomized radii, from the same column
            # ``_refresh_geometry`` re-prices the strike table off, so the
            # tokens, the strike row and the collision cylinders are one kit.
            drum_radii=st["drum_radius"],
            waist_yaw_ref=self._waist_aim_reference(),
        )

    @torch.no_grad()
    def harvest(self) -> dict[int, dict[str, torch.Tensor]]:
        """The CURRENT contexts, bucketed, ready for :meth:`ReplayPool.add`."""
        st = self.state
        hit = st["hit"]
        k_max = hit.max(dim=1).values
        # Recomputed here rather than carried in ``state``: the reference is a
        # pure function of (q0, p_star, active, drum), which are exactly the
        # columns being harvested, so deriving it on the spot cannot drift out
        # of step with them the way a shifted/aged cached row could.
        ref = self._waist_aim_reference() if self.want_waist_ref else None
        out: dict[int, dict[str, torch.Tensor]] = {}
        for b in self.pool.buckets:
            if b > self.H:
                continue
            lo = 0 if b == min(self.pool.buckets) else max(
                x for x in self.pool.buckets if x < b
            )
            sel = ((k_max > lo) & (k_max <= b)).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            cols = {}
            for k, v in st.items():
                if k in ("prev_cmd", "age", "rest", "kp", "kd", "waist_pr_ema"):
                    continue
                cols[k] = v.index_select(0, sel)
            cols["init"] = cols["init"][:, :b].contiguous()
            if ref is not None:
                cols["waist_ref"] = ref.index_select(0, sel)[:, :b].contiguous()
            if self.want_waist_pr_ref:
                # Held (pitch, roll) per slot, broadcast over the horizon --
                # same shape the encoder head used to fill.
                cols["waist_pr_ref"] = (
                    st["waist_pr_ema"].index_select(0, sel)
                    .unsqueeze(1).expand(-1, b, -1).contiguous()
                )
            mark = getattr(self, "_post_hit_fresh", None)
            if mark is not None:
                ph = mark.index_select(0, sel)
                if getattr(self, "frontier", None) is not None:
                    cols["_post_hit_attempted"] = ph.clone()
                cols["_post_hit_fresh"] = self._filter_post_hit_collisions(cols, ph)
            out[b] = cols
        mark = getattr(self, "_post_hit_fresh", None)
        if mark is not None:
            mark.zero_()
        return out

    @torch.no_grad()
    def _filter_post_hit_collisions(
        self, cols: dict[str, torch.Tensor], mark: torch.Tensor,
    ) -> torch.Tensor:
        """Drop live-B rows whose q0 collides (stick/wrist vs kit + under-volume).

        ``start_drum`` is exempted so leftover contact with the pad just struck
        is not a reject; a stick through any *other* drum, the under-volume of
        another drum, a wrist in any drum, or a stick in the torso still fails.
        """
        if not bool(mark.any()):
            return mark
        idx = mark.nonzero(as_tuple=True)[0]
        q = cols["q0"].index_select(0, idx).to(self.dtype)
        heads = (
            cols["drum_pos"].index_select(0, idx).to(self.dtype)
            if "drum_pos" in cols else None
        )
        radii = (
            cols["drum_radius"].index_select(0, idx).to(self.dtype)
            if "drum_radius" in cols else None
        )
        exempt = (
            cols["start_drum"].index_select(0, idx)
            if "start_drum" in cols else None
        )
        clear = q0_collision_clear(
            self.sdf, q,
            drum_head_centers=heads, drum_radii=radii, exempt_drum=exempt,
        )
        out = mark.clone()
        out[idx] = clear.to(dtype=torch.bool)
        n_drop = int((~clear).sum())
        if n_drop:
            self._post_hit_reject_sdf += n_drop
        return out

    @torch.no_grad()
    def tick(self) -> dict[str, float]:
        st = self.state
        masks, hit_eff = self.masks()
        init = st["init"] * masks.elem_mask
        state = torch.cat([st["q0"], st["qd0"]], dim=-1)
        # Fields the slot state does not carry fall back to ``ContextInputs``'s
        # own defaults (zeros + invalid).  That is what lets a pool CACHED before
        # the idle block existed still roll: those slots simply carry no idle
        # token, which is the truth about them.
        ctx_fields = {f: st[f] for f in CONTEXT_FIELDS if f in st}
        from .tokenizer import ContextInputs

        ctx = ContextInputs(**ctx_fields)
        amp = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if (self.amp_dtype is not None and self.device.type == "cuda")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        core = self.model.module if hasattr(self.model, "module") else self.model
        was = core.training
        core.eval()
        with amp:
            enc = core.encode_context(ctx)
            if enc.waist_pr_sp is not None and core.terminal_initializer is None:
                init = seed_waist_pr_to_sp(
                    init, st["q0"], st["qd0"], enc.waist_pr_sp,
                    joint_names=self.names, dt=self.dt, ease_wp=self.waist_ease_wp,
                )
            # One-shot arm DLS on quintic/pool inits after the strike-knot
            # waist is known (P/R rewrite, kit jitter).  Not on shifted plans.
            if bool(self._arm_dls_pending.any()):
                live = (
                    self._arm_dls_pending.unsqueeze(-1)
                    & st["active"].to(torch.bool)
                    & (st["hit"] >= 1)
                )
                if core.terminal_initializer is not None:
                    seed = core.initial_trajectory(ctx, state, masks).to(self.dtype)
                    for h in (0, 1):
                        rows = live[:, h]
                        cols = self.arm_cols[h]
                        init[:, :, cols] = torch.where(
                            rows[:, None, None], seed[:, :, cols], init[:, :, cols]
                        )
                    any_live = live.any(dim=-1)
                    init[:, :, self.waist_cols] = torch.where(
                        any_live[:, None, None], seed[:, :, self.waist_cols],
                        init[:, :, self.waist_cols],
                    )
                    if enc.waist_pr_sp is not None:
                        init = seed_waist_pr_to_sp(
                            init, st["q0"], st["qd0"], enc.waist_pr_sp,
                            joint_names=self.names, dt=self.dt,
                            ease_wp=self.waist_ease_wp,
                        )
                else:
                    init = correct_quintic_init_arms(
                        self.fk, init, st["q0"], st["hit"], st["goal_pos"],
                        active=live, steps=1,
                        q_bounds=(self.q_lo, self.q_hi),
                        joint_names=self.names, qd0=st["qd0"], dt=self.dt,
                    )
                self._arm_dls_pending.zero_()
            out = core(init, state, enc, masks)
        if was:
            core.train()
        traj = out.traj.to(self.dtype)
        if self.decoded_plan_transform is not None:
            traj = self.decoded_plan_transform(
                traj=traj,
                q0=st["q0"],
                v0=st["qd0"],
                elem_mask=masks.elem_mask,
                dt=self.dt,
            )
            if not isinstance(traj, torch.Tensor) or traj.shape != out.traj.shape:
                raise ValueError(
                    "decoded_plan_transform must return a tensor shaped like "
                    f"the decoded trajectory {tuple(out.traj.shape)}"
                )

        # execute one waypoint, admissibly, at the LED reference point
        cmd = traj[:, 0] if self.lead_f is None else lead_waypoints(traj, self.lead_f)[:, 0]
        cmd = st["prev_cmd"] + torch.clamp(cmd - st["prev_cmd"], -self.step_max, self.step_max)
        cmd = torch.clamp(cmd, self.q_lo, self.q_hi)
        if self.want_waist_pr_ref:
            st["waist_pr_ema"] = update_waist_pr_ema(
                st["waist_pr_ema"], cmd.index_select(1, self.pr_idx),
                alpha=self.waist_pr_ema_alpha, leak=self.waist_pr_ema_leak,
            )
        Minv = torch.linalg.inv(mass_matrix(self.rnea, st["q0"]))
        q, qd, _int = self.sim.step(st["prev_cmd"], cmd, st["q0"], st["qd0"],
                              kp=st["kp"], kd=st["kd"], Minv=Minv)
        st["prev_cmd"] = cmd
        st["q0"], st["qd0"] = q, qd
        # ``yaw_now`` means "where the torso is NOW": the plant just moved, so
        # the token moves with it.  Writing it only on respawn/regoal (the old
        # policy) paired every harvested row's current ``q0`` with a yaw from
        # ticks ago.
        if "yaw_now" in st:
            st["yaw_now"] = st["q0"][:, self.yaw_i].to(st["yaw_now"].dtype)

        # shift the plan, count the horizon down
        st["init"] = torch.cat([traj[:, 1:], traj[:, -1:]], dim=1)
        st["hit"] = torch.where(st["hit"] >= 0, st["hit"] - 1, st["hit"])
        st["age"] += 1

        # per-hand in-place re-goal, before any respawn decision
        n_out_pre = st["n_out"]
        if self.regoal:
            n_out_pre = st["n_out"].clone()
            self._tick_tth()
            fired, rg = self._regoal()
            # REST STRETCHES.  Respawning the instant no hand has a goal deletes
            # exactly the states the idle terms are meant to shape.  With
            # ``rest_ticks > 0`` a both-idle slot keeps simulating -- as long as
            # it still has a future hit that can tick into range -- so the rest
            # is played out and harvested instead of thrown away.
            idle_both = ~st["active"].any(dim=1)
            st["rest"] = torch.where(
                idle_both, st["rest"] + 1, torch.zeros_like(st["rest"])
            )
            keep_rest = (
                idle_both
                & (st["rest"] <= self.rest_ticks)
                & st["future_valid"].any(dim=2).any(dim=1)
            )
            done = (idle_both & ~keep_rest) | (st["age"] > self.max_age)
            clip_tail = getattr(self, "_pending_clip_tail", None)
            if clip_tail is not None:
                done = done | clip_tail
                self._pending_clip_tail = None
            rg["rest"] = float(idle_both.sum())
        else:
            fired, rg = None, {"regoal": 0.0, "idle": 0.0, "stale": 0.0,
                               "sep_push": 0.0, "clip_tail": 0.0}
            done = (st["hit"].clamp_min(0).max(dim=1).values <= 1) | (
                st["age"] > self.max_age
            )

        # respawn: the chain ended, aged out, or the plant left the admissible set
        bad = (~torch.isfinite(q).all(dim=1)) | (~torch.isfinite(qd).all(dim=1))
        bad = bad | ((q < self.q_lo - 0.5).any(dim=1)) | ((q > self.q_hi + 0.5).any(dim=1))
        reset = (done | bad).nonzero(as_tuple=True)[0]
        n_reset = int(reset.numel())
        n_bad = int(bad.sum())
        # A slot that is about to be overwritten is not a post-hit init
        # unless chain-carry keeps the bounced plant (marked again below).
        if n_reset and getattr(self, "_post_hit_fresh", None) is not None:
            self._post_hit_fresh.index_fill_(0, reset, False)
        if n_reset:
            keep = (
                (torch.rand(n_reset, device=self.device, generator=self.gen)
                 < self.chain_frac) & ~bad.index_select(0, reset)
            )
            take = reset[keep]
            carry = None
            if take.numel():
                hand = st["hit"].index_select(0, take).argmax(dim=1)
                if fired is not None:
                    # the strike row has already been re-priced, so the bounce
                    # normal must come from the PRE-re-goal row of the hand that
                    # actually fired.
                    f = fired.index_select(0, take)
                    hand = torch.where(f.any(dim=1), f.long().argmax(dim=1), hand)
                normal = n_out_pre.index_select(0, take).gather(
                    1, hand.view(-1, 1, 1).expand(-1, 1, 3)
                ).squeeze(1)
                qb, vb = apply_bounce(
                    q.index_select(0, take), qd.index_select(0, take),
                    fk=self.fk, rnea=self.rnea, hand_index=hand, normal=normal,
                    cfg=self.bounce, generator=self.gen,
                )
                carry = {"q": torch.clamp(qb, self.q_lo, self.q_hi), "qd": vb, "take": take}
            self._spawn(reset, carry)
            if take.numel() and getattr(self, "_post_hit_fresh", None) is not None:
                # Bounced plant + new goal, init re-ramped from the carried q0.
                self._post_hit_fresh.index_fill_(0, take, True)
                if "kind" in st:
                    st["kind"].index_fill_(0, take, KIND_POST_HIT)
        # LAST, so it covers hands that just went idle in the re-goal AND the
        # freshly spawned slots (whose pooled seed may hold a parked arm in
        # place).  The decoder reads ``init`` as its waypoint tokens, so this is
        # also how the idle target reaches the model.
        n_idle_seed = self._seed_idle()
        dropped = float(self._post_hit_reject_sdf)
        self._post_hit_reject_sdf = 0
        return {"reset": float(n_reset), "diverged": float(n_bad),
                "idle_seed": n_idle_seed, "post_hit_sdf": dropped, **rg,
                **({"pr_ema": float(st["waist_pr_ema"].abs().mean())}
                   if self.want_waist_pr_ref else {})}


# ================================================================ losses =====
def encode(module, context):
    """``encode_context`` through a possible DDP wrapper.

    Kept OUTSIDE the compiled step on purpose: inductor mis-compiles the
    encoder's ``goal_features`` concatenation on this Triton
    (``CompilationError`` in the 3/6-boundary cat), and the encoder is 2 layers
    over 16 tokens -- eager costs almost nothing and the plan runs it once per
    context change anyway.
    """
    core = module.module if hasattr(module, "module") else module
    return core.encode_context(context)


def make_encode_step(module, *, compile_mode: str | None = None):
    """Build the context encoder as its own optional Inductor unit."""
    core = module.module if hasattr(module, "module") else module

    def fn(context):
        return core.encode_context(context)

    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode, dynamic=False)


#: The five loss-decomposition entries every metrics row starts with.
LOSS_KEYS: tuple[str, ...] = ("loss", "cost", "aux", "tc", "bptt")

#: APPEND-ONLY, like ``VIOL_KEYS``: ``train_phase2`` slices the metrics tensor
#: positionally and ``VIOL_OFFSET`` is derived from this length.  ``effort`` is
#: the strike-masked ``tau^2`` energy proxy (section 7d leg 2) -- an objective,
#: not a constraint, so it has no entry in ``VIOL_KEYS``.
TERM_KEYS: tuple[str, ...] = (
    "terminal", "collision", "torque", "velocity", "joint_box", "smooth", "posture",
    "effort", "torque_static", "waist_aim", "waist_yaw_vel", "waist_yaw_monotonic",
    "waist_yaw_accel_direction",
    "waist_yaw_initial_accel_direction",
    "waist_pr_vel", "tip_crossover",
    "terminal_aim", "waist_pr_band", "waist_pr_terminal", "waist_upright",
)
TORQUE_TERM_IDX: int = TERM_KEYS.index("torque")
#: APPEND-ONLY (``train_phase2`` reads the block positionally out of the metrics
#: tensor).  ``torque_peak`` is the MAX-shaped half of the torque hinge and its
#: own AL group; ``collision_peak`` is the same for the collision hinge -- see
#: :mod:`amortized_opt.lagrangian`.  ``collision_peak`` integrates the same
#: per-sample max penetration as ``collision``, so the two columns are equal by
#: construction; they are separate groups because the PENALTIES they scale
#: differ by ``n_seen``.
VIOL_KEYS: tuple[str, ...] = CONSTRAINT_GROUPS
DIAG_KEYS: tuple[str, ...] = (
    "terminal_region_dist", "contact_pos_err", "contact_speed_err",
    "collision_min_sdf", "torque_excess_max", "velocity_excess_max",
    "joint_box_excess_max", "posture_elbow_max", "smooth_jerk_integral",
    # the statistic the acceptance gate is actually a threshold on -- append
    # only, the earlier indices are read positionally in ``train_phase2``
    "torque_hinge_max",
    # section 7d: the two quantities the waist-upright terms exist to separate.
    # ``waist_pr_max`` is what must fall; ``waist_yaw_max`` is the control that
    # must NOT, and logging it every step is what makes "yaw was left free" a
    # measurement rather than a claim about the code.
    "waist_pr_max", "waist_yaw_max",
    # section 7d leg 2: the energy the tau^2 term exists to reduce, measured
    # over the WHOLE horizon (unfaded) so the claim covers the strike too.
    "torque_sq_mean", "torque_abs_mean",
    # collision sub-type breakdown (section 7d)
    "collision_drum", "collision_under", "collision_self",
    # time-bucketed drum collision per hand, waypoints before hit:
    # far(>10) | mid(3-10] | near(exempt..3] | exempt | no-hit
    "t_col_far_left", "t_col_mid_left", "t_col_near_left",
    "t_col_exempt_left", "t_col_nohit_left",
    "t_col_far_right", "t_col_mid_right", "t_col_near_right",
    "t_col_exempt_right", "t_col_nohit_right",
    # section 9b (waist ablation): the RATE complements of waist_pr_max /
    # waist_yaw_max.  ``waist_yawd_max`` is what ``w_waist_yaw_vel`` exists to
    # reduce; ``waist_prd_max`` is the control that shows a frozen pitch/roll
    # really is stationary rather than merely small.  Logged on every run,
    # including the ones with both weights at zero -- otherwise the ablation has
    # no baseline to compare against.
    "waist_yawd_max", "waist_prd_max",
    # CROSS-HAND (stick vs stick) clearance.  Logged on every run, including the
    # ones with ``w_stick_stick = 0`` -- the term is off by default and these
    # two columns are how a run measures whether it needs to be on.
    # ``collision_stick`` is the priced mean; ``collision_stick_min_sdf`` is the
    # closest approach of the two sticks over the horizon, in metres.
    "collision_stick", "collision_stick_min_sdf",
    # TIP LATERAL ORDERING.  Logged even at ``w_tip_crossover = 0``.
    # ``tip_lateral_min`` is min (tip_L.y - tip_R.y) in torso frame (m);
    # ``tip_crossover_max`` is max relu(-(tip_L.y - tip_R.y)).
    "tip_lateral_min", "tip_crossover_max",
    # TRACKING TORQUE (``CostConfig.w_track_torque*``).  The mean and the max of
    # the excess of the ESTIMATED PD DEMAND over the joint's usable torque, and
    # the peak demand itself -- exact zeros on a run with the term off, which is
    # the point: they are how a run measures whether it needs to turn it on.
    "track_torque_excess_mean", "track_torque_excess_max", "track_torque_max",
    # P/R at the nearer active hit (the V_f coordinate).  Appended so a
    # run with ``w_waist_pr_terminal = 0`` still has a baseline.
    "waist_pr_N",
    # Active-arm mass-proxy geometry in the torso frame.  Logged even when the
    # compactness weight is zero so thresholds/weight can be calibrated on an
    # existing policy before changing its objective.
    "arm_com_height_max", "arm_com_radius_max",
    "arm_com_height_mean", "arm_com_radius_mean",
    "arm_com_height_excess_max", "arm_com_radius_excess_max",
)
#: Short log names for :data:`DIAG_KEYS`, positionally.  The term / violation
#: names are derived from their key tuples so the two can never drift apart.
DIAG_NAMES: tuple[str, ...] = (
    "d_region", "d_pos_err", "d_speed_err", "d_min_sdf", "d_tau_excess",
    "d_vel_excess", "d_box_excess", "d_elbow", "d_jerk", "d_tau_hinge_max",
    "d_waist_pr", "d_waist_yaw", "d_tau_sq", "d_tau_abs",
    "d_c_drum", "d_c_under", "d_c_self",
    "tcf_l", "tcm_l", "tcn_l", "tce_l", "tcx_l",
    "tcf_r", "tcm_r", "tcn_r", "tce_r", "tcx_r",
    "d_waist_yawd", "d_waist_prd",
    "d_c_stick", "d_stick_sdf",
    "d_tip_lat", "d_tip_cross",
    "d_tau_track", "d_tau_track_max", "d_tau_track_peak",
    "d_waist_pr_N",
    "d_arm_com_z", "d_arm_com_r", "d_arm_com_z_mean", "d_arm_com_r_mean",
    "d_arm_com_z_ex", "d_arm_com_r_ex",
)
assert len(DIAG_NAMES) == len(DIAG_KEYS)
assert set(VIOL_KEYS) == set(CONSTRAINT_GROUPS), (VIOL_KEYS, CONSTRAINT_GROUPS)

#: Diagnostics reported per step, in order.  The first five are the loss
#: decomposition; the rest are ``TERM_KEYS`` / ``VIOL_KEYS`` / ``DIAG_KEYS``.
METRIC_NAMES: tuple[str, ...] = (
    LOSS_KEYS
    + tuple(f"c_{k}" for k in TERM_KEYS)
    + tuple(f"v_{k}" for k in VIOL_KEYS)
    + DIAG_NAMES
)
#: Index of the first ``v_*`` entry in :data:`METRIC_NAMES` / in the metrics
#: tensor :func:`cost_tuple` builds.  Derived, never literal.
VIOL_OFFSET: int = len(LOSS_KEYS) + len(TERM_KEYS)

#: Per-row plan-space features for gated P/R distill, stacked as ``(B, 4)``
#: by :func:`cost_tuple`.  Already computed inside ``FusedCost.forward``;
#: keeping them is how the gate stays a gather, not a second cost.
PR_DISTILL_GATE_KEYS: tuple[str, ...] = (
    "terminal_region_dist",
    "joint_box_excess_max",
    "collision_pen_max",
    "waist_pr_band",
)


def waist_pr_hit_teacher(traj, hit_index, active, pr_idx) -> torch.Tensor:
    """``(B, 2)`` strike-knot waist P/R, mean over active hands, detached.

    Idle hands do not vote.  A row with no active strike returns zeros
    (the gate masks it).  Not the horizon mean, not the peak.
    """
    B, H, _ = traj.shape
    idx = torch.as_tensor(pr_idx, device=traj.device, dtype=torch.long)
    k = hit_index.to(device=traj.device, dtype=torch.long).clamp(0, H - 1)
    act = active.to(device=traj.device, dtype=traj.dtype)
    b = torch.arange(B, device=traj.device)
    q_l = traj[b, k[:, 0]].index_select(-1, idx)
    q_r = traj[b, k[:, 1]].index_select(-1, idx)
    wsum = act[:, 0:1] * q_l + act[:, 1:2] * q_r
    den = act.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (wsum / den).detach()


def waist_pr_distill_gate(
    region_dist, box_excess, coll_pen, band_excess, q_hit, active, *,
    pr_limit: float, hit_m: float = 0.02,
) -> torch.Tensor:
    """Detached ``(B,)`` 0/1 mask: conjunction of the plan-space gates.

    Hit alone is not enough (dumped 2 rad plans still hit).  Band excess
    0 keeps dumps out even if the joint-box diagnostic is noisy.
    """
    has = (active.to(q_hit.device) > 0.5).any(dim=1)
    on_head = region_dist.to(q_hit.device) <= float(hit_m)
    in_box = box_excess.to(q_hit.device) <= 0.0
    no_coll = coll_pen.to(q_hit.device) <= 0.0
    in_band = band_excess.to(q_hit.device) <= 0.0
    pr_ok = q_hit.abs().amax(dim=-1) <= float(pr_limit)
    mask = has & on_head & in_box & no_coll & in_band & pr_ok
    return mask.to(dtype=q_hit.dtype).detach()


def waist_pr_distill_loss(sp, q_hit, mask) -> torch.Tensor:
    """Mean ``‖sp - q_hit‖²`` over gated rows; 0 if none. Encoder-only."""
    mse = (sp - q_hit).pow(2).mean(dim=-1)
    n = mask.sum().clamp_min(1.0)
    return (mask * mse).sum() / n


def waist_pr_hold_band_excess(sp, q0_pr, *, band_rad: float, pr_limit: float):
    """``(B, 2)`` ``relu(|sp - clamp(q0)| - band)``. Encoder-only; ``q0`` detached.

    ``pr_limit`` is the URDF half-width the head already clamps to.  A 1.55
    inherited pose is not an announcement target.
    """
    lim = float(pr_limit)
    q = q0_pr.detach().to(device=sp.device, dtype=sp.dtype).clamp(-lim, lim)
    return ((sp - q).abs() - float(band_rad)).clamp_min(0.0)


def waist_pr_hold_band_loss(sp, q0_pr, *, band_rad: float, pr_limit: float):
    """Mean linear ReLU excess of ``|sp - clamp(q0)|`` past ``band_rad``.

    0 inside the window.  Linear so a finite weight can beat
    ``w_waist_pr_sp_tether`` at the inner edge; a squared hinge
    equilibrates with a residual gap.
    """
    return waist_pr_hold_band_excess(
        sp, q0_pr, band_rad=band_rad, pr_limit=pr_limit,
    ).mean()


def _gate_from_ct(ct) -> torch.Tensor:
    """Per-row gates carried across the compiled cost boundary.

    Columns 0:4 retain the waist-P/R distillation ABI.  Columns 4:8 are the
    four model-independent voxel feasibility diagnostics: terminal-region
    distance, raw collision clearance, joint-box excess, and available-torque
    excess.  Keeping these per row avoids a second physics pass after training.
    """
    return torch.stack(
        [
            ct.diagnostics["terminal_region_dist"].float(),
            ct.diagnostics["joint_box_excess_max"].float(),
            ct.diagnostics["collision_pen_max"].float(),
            ct.violations["waist_pr_band"].float(),
            ct.diagnostics["terminal_region_dist"].float(),
            ct.diagnostics["collision_min_sdf"].float(),
            ct.diagnostics["joint_box_excess_max"].float(),
            ct.diagnostics["torque_excess_max"].float(),
        ],
        dim=1,
    )


def _hard_feasibility_ok_from_gate(
    gate: torch.Tensor,
    *,
    terminal_eps: float = 1.0e-6,
    collision_clearance: float = 0.0,
    joint_eps: float = 1.0e-6,
    torque_eps: float = 1.0e-6,
) -> torch.Tensor:
    """Hard witness predicate for the compact per-row gate tensor."""
    if gate.ndim != 2 or gate.shape[1] < 8:
        raise ValueError(f"hard feasibility gate requires shape (B, >=8), got {tuple(gate.shape)}")
    hard = gate[:, 4:8].detach().float()
    return (
        torch.isfinite(hard).all(dim=1)
        & (hard[:, 0] <= float(terminal_eps))
        & (hard[:, 1] >= float(collision_clearance))
        & (hard[:, 2] <= float(joint_eps))
        & (hard[:, 3] <= float(torque_eps))
    )


def _bellman_parent_ingress_ok(
    gate: torch.Tensor,
    cfg: ViolationCriticConfig,
) -> torch.Tensor:
    """Bellman admission at half the former critic-MPC allowances.

    The critic MPC pool now accepts every structurally advanceable actor plan.
    Bellman teachers remain selective: each physical allowance is half of the
    old MPC parent gate.  Joint and torque limits include the old robust-space
    slack before being halved.
    """
    if gate.ndim != 2 or gate.shape[1] < 8:
        raise ValueError(
            f"Bellman parent gate requires shape (B, >=8), got {tuple(gate.shape)}"
        )
    exact = gate[:, 4:8].detach().float()
    robust_slack = math.sinh(float(cfg.acceptable_normalized_violation))
    terminal_max = 0.5 * float(cfg.primary_mpc_terminal_limit)
    collision_min = 0.5 * float(cfg.primary_mpc_collision_sdf_limit)
    joint_max = 0.5 * (
        float(cfg.joint_limit) + robust_slack * float(cfg.joint_scale)
    )
    torque_max = 0.5 * (
        float(cfg.torque_limit) + robust_slack * float(cfg.torque_scale)
    )
    return (
        torch.isfinite(exact).all(dim=1)
        & (exact[:, 0] <= terminal_max)
        & (exact[:, 1] >= collision_min)
        & (exact[:, 2] <= joint_max)
        & (exact[:, 3] <= torque_max)
    )


def _timing_mpc_parent_ingress_ok(gate: torch.Tensor) -> torch.Tensor:
    """Admit every completed actor row to the critic MPC parent pool.

    ``commit_side_effects`` and the finite-gradient guard remain the outer
    transaction boundary. The ring itself separately rejects rows that have no
    later pre-hit waypoint from which a suffix can be constructed.
    """
    if gate.ndim != 2:
        raise ValueError(f"critic MPC parent gate requires a matrix, got {tuple(gate.shape)}")
    return torch.ones(int(gate.shape[0]), device=gate.device, dtype=torch.bool)


def _metrics_tensor(ct, loss, loss_cost, loss_aux, device, dtype) -> torch.Tensor:
    return _metrics_from_stacks(
        loss, loss_cost, loss_aux,
        torch.stack([ct.terms[k].mean() for k in TERM_KEYS]),
        torch.stack([ct.violations[k].float().mean() for k in VIOL_KEYS]),
        torch.stack([ct.diagnostics[k].float().mean() for k in DIAG_KEYS]),
        device, dtype,
    )


def _metrics_from_stacks(
    loss, loss_cost, loss_aux, terms, violations, diagnostics, device, dtype,
) -> torch.Tensor:
    zero = torch.zeros((), device=device, dtype=dtype)
    return torch.cat([
        torch.stack([loss.detach(), loss_cost.detach(), loss_aux.detach(),
                     zero, zero]),
        terms.detach(), violations.detach(), diagnostics.detach(),
    ])


def cost_tuple(
    cost: FusedCost, q0, traj, v0, strike, lengths, weights,
    *,
    aux_mode: str | None = None,
    aux_tau_kernel=None,
):
    """``FusedCost`` reduced to plain tensors.

    Dataclass outputs are fine for dynamo, but the whole point of this wrapper
    is that it is a *small, separately compiled* unit -- see
    :func:`make_cost_step` -- and a flat tuple is what makes the graph boundary
    unambiguous.  The fifth return is a per-row gate tensor.  Its first four
    columns are in :data:`PR_DISTILL_GATE_KEYS` order; the remaining four are
    the hard voxel witness diagnostics.  All were already paid for in the
    forward and are mean-reduced out of the log metrics.

    When ``aux_mode`` is set, five more tensors follow (vel/acc/torque/fk_pos/
    fk_vel labels), built from the cost plant so the train step does not run a
    second spline+RNEA+FK pass.
    """
    ct = cost.forward(
        q0, traj, v0, strike, lengths=lengths, weights=weights,
        aux_mode=aux_mode, aux_tau_kernel=aux_tau_kernel,
    )
    gate = _gate_from_ct(ct)
    al_ok = weights.get("_al_ok") if weights else None
    if al_ok is None:
        violation_means = torch.stack([
            ct.violations[k].float().mean() for k in VIOL_KEYS
        ])
    else:
        mask = al_ok.to(device=ct.total.device, dtype=torch.float32).reshape(-1)
        denom = mask.sum().clamp_min(1.0)
        violation_means = torch.stack([
            (ct.violations[k].float() * mask).sum() / denom for k in VIOL_KEYS
        ])
    base = (
        ct.total,
        torch.stack([ct.terms[k].mean() for k in TERM_KEYS]),
        violation_means,
        torch.stack([ct.diagnostics[k].float().mean() for k in DIAG_KEYS]),
        gate,
    )
    if aux_mode is None:
        return base
    lab = ct.aux_labels
    assert lab is not None
    return base + (lab["vel"], lab["acc"], lab["torque"], lab["fk_pos"], lab["fk_vel"])


def cost_grad_balance_tuple(
    cost: FusedCost, q0, traj, v0, strike, lengths, weights,
    *,
    aux_mode: str | None = None,
    aux_tau_kernel=None,
):
    """Cost tensors needed by AL-GradNorm/PCGrad across a compile boundary."""
    ct = cost.forward(
        q0, traj, v0, strike, lengths=lengths, weights=weights,
        aux_mode=aux_mode, aux_tau_kernel=aux_tau_kernel,
    )
    al_ok = weights.get("_al_ok") if weights else None
    if al_ok is None:
        violation_means = torch.stack([
            ct.violations[k].float().mean() for k in VIOL_KEYS
        ])
    else:
        mask = al_ok.to(device=ct.total.device, dtype=torch.float32).reshape(-1)
        denom = mask.sum().clamp_min(1.0)
        violation_means = torch.stack([
            (ct.violations[k].float() * mask).sum() / denom for k in VIOL_KEYS
        ])
    base = (
        ct.total,
        torch.stack([ct.terms[k].mean() for k in TERM_KEYS]),
        violation_means,
        torch.stack([ct.diagnostics[k].float().mean() for k in DIAG_KEYS]),
        _gate_from_ct(ct),
        torch.stack([
            sum(ct.terms[k].mean() for k in term_keys)
            for term_keys, _al_keys in GRAD_BALANCE_GROUPS.values()
        ]),
        ct.balance_terms["joint_box_waist"].mean(),
    )
    if aux_mode is None:
        return base
    lab = ct.aux_labels
    assert lab is not None
    return base + (lab["vel"], lab["acc"], lab["torque"], lab["fk_pos"], lab["fk_vel"])


@dataclass
class GradBalanceCostView:
    """Differentiable cost payload reconstructed outside compiled physics."""

    total: torch.Tensor
    group_terms: dict[str, torch.Tensor]
    protected_terms: dict[str, torch.Tensor]
    violations: dict[str, torch.Tensor]


def make_cost_step(module, cost: FusedCost, scales: LossScales, cfg: Phase2Config,
                   *, compile_mode: str | None = "default"):
    """``(batch tensors, weights) -> (loss, metrics, traj, gate)`` for one bucket.

    THREE separately compiled units rather than one, deliberately:

    * the decoder pass,
    * :func:`cost_tuple`, and
    * the auxiliary Huber (labels come from the cost plant when ``w_aux>0``).

    One fused graph over all of it is ~3x the size of the cost graph the
    section-3 bench compiles, and inductor took >13 minutes on it here without
    producing a first step -- and it mis-compiled a concatenation inside the
    context encoder on this Triton (``CompilationError`` at the 3/6 boundary),
    which is why :func:`encode` also stays outside.  Split, each unit compiles
    in the low minutes, caches to disk, and a failure degrades one unit to
    eager instead of the whole step.
    """
    def _model(init, state, enc, masks):
        return module(init, state, enc, masks)

    want_aux = float(cfg.w_aux) > 0.0
    aux_tau_mode = "tracking" if cfg.aux_torque_tracking else "rnea_segmax"
    aux_tau_kernel = None
    if want_aux and cfg.aux_torque_tracking:
        if cost.c_tau_kernel is None:
            cost.c_tau_kernel = cost.build_torque_kernels()
        aux_tau_kernel = cost.c_tau_kernel
    aux_mode = aux_tau_mode if want_aux else None

    def _cost(q0, traj, v0, strike, lengths, weights):
        if cfg.grad_balance:
            return cost_grad_balance_tuple(
                cost, q0, traj, v0, strike, lengths, weights,
                aux_mode=aux_mode, aux_tau_kernel=aux_tau_kernel,
            )
        return cost_tuple(
            cost, q0, traj, v0, strike, lengths, weights,
            aux_mode=aux_mode, aux_tau_kernel=aux_tau_kernel,
        )

    def _aux_huber(vel, acc, torque, fk_pos, fk_vel, lab_vel, lab_acc, lab_torque,
                   lab_fk_pos, lab_fk_vel, em, pm):
        pmf = pm.unsqueeze(-1).expand_as(fk_pos)
        beta = 0.1
        return (
            _masked_huber(vel / scales.vel, lab_vel / scales.vel, em, beta)[0]
            + _masked_huber(acc / scales.acc, lab_acc / scales.acc, em, beta)[0]
            + _masked_huber(torque / scales.torque, lab_torque / scales.torque,
                            em, beta)[0]
            + _masked_huber(fk_pos / scales.fk_pos, lab_fk_pos / scales.fk_pos,
                            pmf, beta)[0]
            + _masked_huber(fk_vel / scales.fk_vel, lab_fk_vel / scales.fk_vel,
                            pmf, beta)[0]
        )

    if compile_mode is not None:
        if cfg.compile_model:
            _model = torch.compile(_model, mode=compile_mode, dynamic=False)
        if cfg.compile_cost:
            _cost = torch.compile(_cost, mode=compile_mode, dynamic=False)
        if cfg.compile_aux and want_aux:
            _aux_huber = torch.compile(_aux_huber, mode=compile_mode, dynamic=False)

    def fn(
        init, state, q0, v0, enc, masks, strike, lengths, weights,
        torque_scale_delta,
    ):
        out = _model(init, state, enc, masks)
        em = masks.elem_mask.to(out.traj.dtype)
        pm = masks.point_mask.to(out.traj.dtype)
        if cfg.grad_balance:
            packed = _cost(q0, out.traj, v0, strike, lengths, weights)
            total_cost, terms, viols, diags, gate, grouped, protected_box = packed[:7]
            if want_aux:
                lab_vel, lab_acc, lab_torque, lab_fk_pos, lab_fk_vel = packed[7:]
                loss_aux = _aux_huber(
                    out.vel, out.acc, out.torque, out.fk_pos, out.fk_vel,
                    lab_vel, lab_acc, lab_torque, lab_fk_pos, lab_fk_vel, em, pm,
                )
            else:
                loss_aux = out.traj.new_zeros(())
            loss_cost = total_cost.mean()
            loss = cfg.w_cost * loss_cost + cfg.w_aux * loss_aux
            metrics = _metrics_from_stacks(
                loss, loss_cost, loss_aux, terms, viols, diags,
                loss.device, loss.dtype,
            )
            ct = GradBalanceCostView(
                total=total_cost,
                group_terms={
                    name: grouped[i]
                    for i, name in enumerate(GRAD_BALANCE_GROUPS)
                },
                protected_terms={"joint_box_waist": protected_box},
                violations={
                    name: viols[i] for i, name in enumerate(VIOL_KEYS)
                },
            )
            return loss, metrics, out.traj, gate.detach(), ct, loss_aux
        packed = _cost(q0, out.traj, v0, strike, lengths, weights)
        total, terms, viols, diags, gate = packed[:5]
        if want_aux:
            lab_vel, lab_acc, lab_torque, lab_fk_pos, lab_fk_vel = packed[5:]
            loss_aux = _aux_huber(
                out.vel, out.acc, out.torque, out.fk_pos, out.fk_vel,
                lab_vel, lab_acc, lab_torque, lab_fk_pos, lab_fk_vel, em, pm,
            )
        else:
            loss_aux = out.traj.new_zeros(())
        loss_cost = total.mean()
        loss = cfg.w_cost * loss_cost + cfg.w_aux * loss_aux
        if cfg.torque_grad_cap > 0.0:
            # On cached-cap steps this turns
            #   g_rest + g_torque -> g_rest + scale * g_torque
            # in the single ordinary backward.  A zero delta is the exact
            # legacy objective used on refresh steps.
            loss = loss + (
                torque_scale_delta.to(loss.dtype)
                * float(cfg.w_cost) * terms[TORQUE_TERM_IDX]
            )
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
        metrics = torch.cat([
            torch.stack([loss.detach(), loss_cost.detach(), loss_aux.detach(),
                         zero, zero]),
            terms.detach(), viols.detach(), diags.detach(),
        ])
        if cfg.torque_grad_cap > 0.0:
            return loss, metrics, out.traj, gate.detach(), terms[TORQUE_TERM_IDX]
        return loss, metrics, out.traj, gate.detach()

    return fn


def measure_torque_directional_curvature(
    cost: FusedCost,
    cfg: Phase2Config,
    *,
    traj: torch.Tensor,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike,
    lengths: torch.Tensor,
    weights: dict,
) -> dict[str, float]:
    """Probe torque curvature with explicit-fp32 gradient differences.

    The training graph may be inside BF16 autocast.  This diagnostic
    transactionally detaches its input and disables autocast for every cost
    evaluation, then estimates ``d.T @ H @ d`` from *gradient* differences at
    several scales.  This avoids subtracting three nearly equal objective
    values divided by ``h**2``.  Hard-limit/hinge threshold and hard-max
    crossings are reported separately so a non-smooth crossing is not called
    a bad derivative.

    No parameter ``.grad`` or caller tensor is written.  The legacy 1 mrad
    objective second difference remains for dashboards, but is explicitly
    marked unreliable and is never used to select the estimate.
    """
    n = min(int(cfg.torque_curvature_rows), int(traj.shape[0]))
    if n <= 0:
        return {}
    sl = slice(0, n)
    x = traj[sl].detach().float().requires_grad_(True)
    q = q0[sl].detach().float()
    v = v0[sl].detach().float()
    lens = lengths[sl]
    strike_n = _strike_rows(strike, sl)
    row_idx = torch.arange(n, device=traj.device, dtype=torch.long)
    weights_n = _index_batch_weights(
        weights, row_idx, batch_size=int(traj.shape[0]),
    )

    device_type = "cuda" if x.is_cuda else "cpu"

    def torque_scalar(waypoints: torch.Tensor) -> torch.Tensor:
        # Explicitly nested OFF context wins over a surrounding training
        # autocast context and restores that context on exit.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            ct = cost.forward(
                q, waypoints.float(), v, strike_n,
                lengths=lens, weights=weights_n,
            )
            return float(cfg.w_cost) * ct.terms["torque"].mean()

    def active_signature(waypoints: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Threshold/max identity for the full-RNEA torque part of the probe."""
        with torch.no_grad(), torch.amp.autocast(
            device_type=device_type, enabled=False
        ):
            sp = cost.spline.build(q, waypoints.float(), v, lens)
            if cost.cfg.physics_mode == "knot_fd":
                qr, qd, qdd = sp.knot_finite_differences(v)
                idx = torch.arange(
                    sp.max_segments + 1, device=qr.device
                ).view(1, -1, 1)
                mask = idx <= sp.lengths.unsqueeze(1)
            else:
                ss = sp.subsample(cost.cfg.n_sub, include_end=True)
                qr, qd, qdd, mask = ss.q, ss.qd, ss.qdd, ss.mask.bool()
            B, S, D = qr.shape
            tau = cost.rnea.rnea(
                qr.reshape(B * S, D),
                qd.reshape(B * S, D),
                qdd.reshape(B * S, D),
            ).reshape(B, S, D)
            lim_hard = available_torque_torch(
                qd,
                cost.tau_hard.view(1, 1, D),
                cost.v_hard.view(1, 1, D),
                eta=1.0,
                saturation_scale=cost.cfg.torque_saturation_scale,
                derate=cost.cfg.torque_derate,
            )
            live = mask.expand_as(tau)
            hard_active = (tau.abs() > lim_hard) & live
            hinge_active = (
                tau.abs() > cost.c_torque_eta.view(1, 1, D) * lim_hard
            ) & live
            signed = (tau >= 0.0) & live
            over = (tau.abs() - lim_hard).clamp_min(0.0) * live
            global_argmax = over.reshape(B, -1).argmax(dim=1)
            segment_argmax = []
            for _name, joints in ACTUATOR_SEGMENTS:
                cols = torch.tensor(
                    [cost.joint_names.index(j) for j in joints],
                    device=tau.device,
                    dtype=torch.long,
                )
                segment_argmax.append(
                    over.index_select(-1, cols).reshape(B, -1).argmax(dim=1)
                )
            return (
                hard_active,
                hinge_active,
                signed,
                global_argmax,
                torch.stack(segment_argmax, dim=1),
            )

    def signature_crossings(
        plus: tuple[torch.Tensor, ...], minus: tuple[torch.Tensor, ...]
    ) -> tuple[int, int, int, int, int]:
        return tuple(
            int((a != b).sum().item()) for a, b in zip(plus, minus)
        )

    c0 = torque_scalar(x)
    if not c0.requires_grad:
        return {
            "torque_curve_rows": float(n),
            "torque_curve_ok": 0.0,
        }
    (g,) = torch.autograd.grad(c0, x)
    g_max = g.detach().abs().amax()
    out = {
        "torque_curve_rows": float(n),
        "torque_curve_grad_max": float(g_max),
        "torque_curve_grad_rms": float(g.detach().float().square().mean().sqrt()),
    }
    if not bool(torch.isfinite(g_max).item()) or float(g_max) <= 1e-12:
        out["torque_curve_ok"] = 0.0
        return out

    d = (-g.detach() / g_max).to(x.dtype)
    slope = -(g.detach() * d).sum()
    out["torque_curve_dir_rms"] = float(d.float().square().mean().sqrt())
    out["torque_curve_pred_slope"] = float(slope)

    c0_det = c0.detach()
    alphas_mrad = (5.0, 2.0, 1.0, 0.5, 0.25)
    estimates: list[tuple[float, float, int]] = []
    legacy_values: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for alpha_mrad in alphas_mrad:
        alpha = alpha_mrad * 1e-3
        xp = (x.detach() + alpha * d).requires_grad_(True)
        xm = (x.detach() - alpha * d).requires_grad_(True)
        cp = torque_scalar(xp)
        cm = torque_scalar(xm)
        (gp,) = torch.autograd.grad(cp, xp)
        (gm,) = torch.autograd.grad(cm, xm)
        d_hd = (d * (gp - gm) / (2.0 * alpha)).sum().detach()
        crossings = signature_crossings(
            active_signature(xp.detach()), active_signature(xm.detach())
        )
        crossing_total = sum(crossings)
        key = str(alpha_mrad).replace(".", "p")
        out[f"torque_curve_gradfd_dHd_{key}mrad"] = float(d_hd)
        out[f"torque_curve_hard_cross_{key}mrad"] = float(crossings[0])
        out[f"torque_curve_hinge_cross_{key}mrad"] = float(crossings[1])
        out[f"torque_curve_sign_cross_{key}mrad"] = float(crossings[2])
        out[f"torque_curve_global_max_cross_{key}mrad"] = float(crossings[3])
        out[f"torque_curve_segment_max_cross_{key}mrad"] = float(crossings[4])
        out[f"torque_curve_active_cross_total_{key}mrad"] = float(crossing_total)
        if bool(torch.isfinite(d_hd).item()):
            estimates.append((alpha_mrad, float(d_hd), crossing_total))
        legacy_values[alpha_mrad] = (cp.detach(), cm.detach())

        # First-order trust ratio is retained; unlike the old second
        # difference it does not divide cancellation by alpha squared.
        with torch.no_grad():
            ca = cp.detach()
            pred = alpha * slope
            actual = c0_det - ca
            out[f"torque_curve_rho_{key}mrad"] = float(
                actual / pred.clamp_min(1e-12)
            )
            out[f"torque_curve_reduction_{key}mrad"] = float(actual)

    # Pick the adjacent, preferably crossing-free pair with least relative
    # disagreement.  The smaller step of that pair is the selected estimate.
    candidates = []
    for left, right in zip(estimates, estimates[1:]):
        denom = max(abs(left[1]), abs(right[1]), 1e-12)
        disagreement = abs(left[1] - right[1]) / denom
        crossings = left[2] + right[2]
        candidates.append((crossings != 0, disagreement, right, left))
    if candidates:
        crossed, disagreement, selected, _neighbor = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        selected_step, selected_dhd, selected_crossings = selected
        good = (not crossed) and disagreement <= 0.10
        out["torque_curve_hvp_ok"] = 1.0
        out["torque_curve_dHd"] = selected_dhd
        out["torque_curve_selected_step_mrad"] = selected_step
        out["torque_curve_selected_disagreement"] = disagreement
        out["torque_curve_selected_crossings"] = float(selected_crossings)
        out["torque_curve_quality_good"] = float(good)
        out["torque_curve_quality_crossing_free"] = float(not crossed)
        # 2=stable/crossing-free, 1=crossing-free but scale-disagreeing,
        # 0=active-set crossing or unavailable.
        out["torque_curve_quality_status"] = float(
            2 if good else (1 if not crossed else 0)
        )
        if abs(selected_dhd) > 1e-12:
            out["torque_curve_quad_safe_mrad"] = float(
                0.5 * float(slope) / abs(selected_dhd) * 1e3
            )
            out["torque_curve_fd_safe_mrad"] = out[
                "torque_curve_quad_safe_mrad"
            ]
    else:
        out["torque_curve_hvp_ok"] = 0.0
        out["torque_curve_quality_good"] = 0.0
        out["torque_curve_quality_crossing_free"] = 0.0
        out["torque_curve_quality_status"] = 0.0

    # Legacy dashboard field only.  Explicitly tell consumers not to use it:
    # fp32 objective-value cancellation and max crossings contaminate it.
    c_plus, c_minus = legacy_values[1.0]
    legacy_fd = (c_plus - 2.0 * c0_det + c_minus) / 1e-6
    out["torque_curve_fd_dHd_1mrad"] = float(legacy_fd)
    out["torque_curve_fd_legacy_unreliable"] = 1.0
    out["torque_curve_legacy_fields_unreliable"] = 1.0
    out["torque_curve_fd_safe_uses_gradfd"] = 1.0
    out["torque_curve_ok"] = out["torque_curve_hvp_ok"]
    return out


def measure_cost_term_grad_norms(
    module,
    cost: FusedCost,
    cfg: Phase2Config,
    *,
    init: torch.Tensor,
    state: torch.Tensor,
    enc,
    masks,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike,
    lengths: torch.Tensor,
    weights: dict,
) -> dict[str, float]:
    """``||∇_θ (w_cost · mean(term))||`` per :data:`TERM_KEYS` entry.

    Diagnostic only: one decoder forward plus one ``autograd.grad`` per term.
    """
    core = module.module if hasattr(module, "module") else module
    params = [p for p in core.parameters() if p.requires_grad]
    if not params:
        return {}
    dec = module(init, state, enc, masks)
    ct = cost.forward(
        q0, dec.traj, v0, strike, lengths=lengths, weights=weights,
    )
    norms: dict[str, float] = {}
    w_cost = float(cfg.w_cost)
    for k in TERM_KEYS:
        t = ct.terms[k].mean()
        if not t.requires_grad:
            continue
        grads = torch.autograd.grad(
            w_cost * t, params, retain_graph=True, allow_unused=True,
        )
        sq = sum(
            g.detach().float().pow(2).sum()
            for g in grads
            if g is not None
        )
        norms[f"grad_{k}"] = float(sq.sqrt().item())
    total_sq = sum(v * v for v in norms.values())
    if total_sq > 0.0 and "grad_torque" in norms:
        norms["grad_torque_frac"] = norms["grad_torque"] / math.sqrt(total_sq)
    if (
        int(cfg.torque_curvature_rows) > 0
        and not cfg.cost.torque_static_only
        and (cfg.cost.w_torque > 0.0 or cfg.cost.w_torque_peak > 0.0)
    ):
        norms.update(measure_torque_directional_curvature(
            cost, cfg, traj=dec.traj, q0=q0, v0=v0, strike=strike,
            lengths=lengths, weights=weights,
        ))
    return norms


def _dataclass_from_mapping(cls, raw: dict):
    """Rebuild a dataclass from :func:`asdict` output (nested dicts)."""
    if not isinstance(raw, dict):
        return raw
    kw: dict = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        sub = getattr(f.type, "__dataclass_fields__", None)
        if sub is not None and isinstance(val, dict):
            kw[f.name] = _dataclass_from_mapping(f.type, val)
        else:
            kw[f.name] = val
    return cls(**kw)


def phase2_config_from_checkpoint(
    ck: dict,
    *,
    device: str | None = None,
    compile_model: bool | None = None,
) -> Phase2Config:
    """Rebuild :class:`Phase2Config` saved in a phase-2 checkpoint."""
    raw = dict(ck.get("train_cfg") or {})
    if device is not None:
        raw["device"] = str(device)
    if compile_model is not None:
        raw["compile_model"] = bool(compile_model)
        if not compile_model:
            raw["compile_mode"] = None
    nested = (
        ("cost", CostConfig),
        ("al", ALConfig),
        ("line_search_cfg", LineSearchConfig),
        ("bounce", BounceConfig),
        ("schedule", WeightSchedule),
        ("timing_critic", ViolationCriticConfig),
    )
    for key, cls in nested:
        if isinstance(raw.get(key), dict):
            raw[key] = _dataclass_from_mapping(cls, raw[key])
    valid = {f.name for f in fields(Phase2Config)}
    return Phase2Config(**{k: v for k, v in raw.items() if k in valid})


def forward_torque_viol_per_row(ct) -> torch.Tensor:
    """Max torque-family violation per batch row (matches replay harvest)."""
    parts = []
    for k, v in ct.violations.items():
        if k == "torque_peak" or k.startswith("torque_"):
            parts.append(v.detach().float())
    n = int(next(iter(ct.violations.values())).shape[0]) if ct.violations else 0
    dev = next(iter(ct.violations.values())).device if ct.violations else "cpu"
    if not parts:
        return torch.zeros(n, device=dev)
    return torch.stack(parts, dim=0).amax(dim=0)


def measure_per_row_torque_grad_norms(
    module,
    cost: FusedCost,
    cfg: Phase2Config,
    *,
    init: torch.Tensor,
    state: torch.Tensor,
    enc,
    masks,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike,
    lengths: torch.Tensor,
    weights: dict,
    rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, object]:
    """``||∇_θ (w_cost · terms['torque'][i])||`` for selected rows."""
    core = module.module if hasattr(module, "module") else module
    params = [p for p in core.parameters() if p.requires_grad]
    if not params:
        return torch.zeros(0, device=init.device), None
    dec = module(init, state, enc, masks)
    ct = cost.forward(
        q0, dec.traj, v0, strike, lengths=lengths, weights=weights,
    )
    torque = ct.terms["torque"]
    b = int(torque.shape[0])
    if rows is None:
        rows = torch.arange(b, device=torque.device)
    else:
        rows = rows.to(device=torque.device, dtype=torch.long)
    if rows.numel() == 0:
        return torch.zeros(0, device=torque.device, dtype=torch.float32), ct
    w_cost = float(cfg.w_cost)
    norms = torch.zeros(rows.shape[0], device=torque.device, dtype=torch.float32)
    for j, i in enumerate(rows.tolist()):
        t = torque[int(i)]
        if not t.requires_grad:
            continue
        grads = torch.autograd.grad(
            w_cost * t,
            params,
            retain_graph=(j + 1 < len(rows)),
            allow_unused=True,
        )
        sq = sum(
            g.detach().float().pow(2).sum()
            for g in grads
            if g is not None
        )
        norms[j] = sq.sqrt()
    return norms, ct


def _stratified_row_indices(
    values: np.ndarray,
    n_rows: int,
    *,
    n_bins: int,
    seed: int,
) -> np.ndarray:
    """Pick up to ``n_rows`` indices with equal quota per value quartile."""
    n = int(values.shape[0])
    if n_rows >= n:
        return np.arange(n, dtype=np.int64)
    per = max(1, n_rows // max(n_bins, 1))
    qs = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        if b == n_bins - 1:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        k = min(per, idx.size)
        picked.extend(rng.choice(idx, size=k, replace=False).tolist())
    if len(picked) < n_rows:
        rest = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(picked, np.int64))
        if rest.size:
            k = min(n_rows - len(picked), rest.size)
            picked.extend(rng.choice(rest, size=k, replace=False).tolist())
    return np.asarray(picked[:n_rows], dtype=np.int64)


def summarize_torque_grad_stratify(
    row_norms: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    n_bins: int = 4,
    top_fracs: tuple[float, ...] = (0.10, 0.25),
) -> dict[str, object]:
    """Energy fractions of ``row_norms**2`` by label quartile and top tails."""
    norms = np.asarray(row_norms, dtype=np.float64)
    energy = norms * norms
    total = float(energy.sum())
    out: dict[str, object] = {
        "n_rows": int(norms.shape[0]),
        "total_energy": total,
        "mean_norm": float(norms.mean()) if norms.size else 0.0,
        "max_norm": float(norms.max()) if norms.size else 0.0,
    }
    if total <= 0.0:
        return out
    strat: dict[str, object] = {}
    for name, vals in labels.items():
        v = np.asarray(vals, dtype=np.float64)
        if v.shape[0] != norms.shape[0]:
            raise ValueError(
                f"label {name!r} length {v.shape[0]} != row_norms {norms.shape[0]}"
            )
        qs = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
        bins = []
        for b in range(n_bins):
            lo, hi = qs[b], qs[b + 1]
            if b == n_bins - 1:
                mask = (v >= lo) & (v <= hi)
            else:
                mask = (v >= lo) & (v < hi)
            e = float(energy[mask].sum())
            bins.append({
                "bin": b,
                "lo": float(lo),
                "hi": float(hi),
                "n": int(mask.sum()),
                "energy_frac": e / total,
                "mean_norm": float(norms[mask].mean()) if mask.any() else 0.0,
                "max_norm": float(norms[mask].max()) if mask.any() else 0.0,
            })
        order = np.argsort(v)
        tails = {}
        for frac in top_fracs:
            k = max(1, int(round(frac * v.shape[0])))
            top = order[-k:]
            bot = order[: max(1, v.shape[0] // 2)]
            tails[f"top_{int(round(frac * 100)):02d}pct"] = {
                "k": k,
                "energy_frac": float(energy[top].sum() / total),
                "mean_norm": float(norms[top].mean()),
            }
            tails[f"bot_50pct_vs_top_{int(round(frac * 100)):02d}pct_ratio"] = (
                float(norms[top].mean() / max(norms[bot].mean(), 1e-12))
            )
        strat[name] = {"quartiles": bins, "tails": tails}
    out["strata"] = strat
    return out


def prime_replay_from_roller(
    roller: MPCRoller,
    replay: ReplayPool,
    pool: Phase2Pool,
    costs: dict[int, FusedCost],
    cfg: Phase2Config,
    al: AugmentedLagrangian,
    step: int,
    n_ticks: int,
    *,
    stage_a: bool = False,
) -> dict[str, float]:
    """Fill ``replay`` the way the train loop does before drawing a batch."""
    acc: dict[str, float] = {}
    device = pool.device
    dtype = pool.dtype
    for _ in range(int(n_ticks)):
        for b, cols in roller.harvest().items():
            if b not in replay.data:
                continue
            if float(cfg.replay_viol_downweight) > 0.0:
                w_h = _wtensors(
                    cfg.schedule.multipliers(step, stage_a),
                    al.weights(), device, dtype,
                    sched=cfg.schedule, joint_names=pool.cfg.joint_names,
                    step=step, stage_a=stage_a, pen=al.pen,
                )
                vs = _harvest_replay_viol_score(
                    cols, b, pool, costs[b], w_h,
                )
                cols = dict(cols)
                cols["viol_score"] = vs
                acc["replay_viol_score"] = (
                    acc.get("replay_viol_score", 0.0) + float(vs.sum())
                )
                acc["replay_viol_n"] = acc.get("replay_viol_n", 0.0) + float(vs.numel())
            replay.add(b, cols)
        for k_, v_ in roller.tick().items():
            acc[k_] = acc.get(k_, 0.0) + float(v_)
    return acc


# =========================================================== checkpoints =====
class _ShutdownFlag:
    """Set by SIGTERM/SIGINT; checked once per training step."""

    requested: bool = False

    @classmethod
    def install(cls) -> None:
        def _handler(signum, _frame) -> None:
            cls.requested = True

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)


def _snapshot_phase2_rng_state(streams: dict) -> dict:
    """Capture the process-global and explicitly-owned phase-2 streams."""
    global_state = snapshot_global_torch_rng()
    return {
        "global_torch": {
            "cpu": global_state.cpu,
            "cuda": global_state.cuda,
        },
        "streams": snapshot_streams(streams),
    }


def _restore_phase2_rng_state(state, streams: dict) -> str | None:
    """Restore RNG state and report full, partial, or legacy coverage."""
    if not isinstance(state, dict):
        return None
    # The script passes the full checkpoint so old ``torch_rng``-only files
    # remain resumable; direct callers may pass just the nested bundle.
    nested = state.get("rng_state")
    has_bundle = isinstance(nested, dict) and bool(nested)
    if has_bundle:
        state = nested
    restored = False
    global_state = state.get("global_torch")
    if isinstance(global_state, dict):
        cpu = global_state.get("cpu")
        if cpu is not None:
            cuda = global_state.get("cuda")
            restore_global_torch_rng(GlobalTorchRngState(
                cpu=cpu, cuda=None if cuda is None else tuple(cuda),
            ))
            restored = True
    elif state.get("torch_rng") is not None:
        restore_global_torch_rng(GlobalTorchRngState(
            cpu=state["torch_rng"], cuda=None,
        ))
        restored = True
    named = state.get("streams")
    if not isinstance(named, dict):
        named = state.get("named")
    if not isinstance(named, dict):
        named = {}
    # Missing entries are expected for legacy/partial checkpoints.  Unknown
    # future streams are ignored so an older runtime can still resume.
    for name, stream in streams.items():
        if name in named:
            restore_stream(stream, named[name])
            restored = True
    if not restored:
        return None
    if not has_bundle:
        return "legacy global-only"
    expected = set(streams)
    restored_names = expected.intersection(named)
    if restored_names == expected and isinstance(global_state, dict):
        return "full"
    return "partial"


def save_phase2_checkpoint(
    path, *, model, optimizer, al: AugmentedLagrangian, step: int,
    cfg: Phase2Config, model_cfg: ModelConfig, scenario_cfg=None, extra=None,
    atomic: bool = True, rng_streams: dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    payload = {
        "step": int(step),
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "al": al.state_dict(),
        "train_cfg": asdict(cfg),
        "model_cfg": asdict(model_cfg),
        "scenario_cfg": asdict(scenario_cfg) if scenario_cfg is not None else None,
        # Selector state under ``extra["line_search"]`` when LS is on
        # (caller merges via ``_checkpoint_extra_with_ls``).
        "extra": dict(extra or {}),
        # Keep this legacy field for old readers and pre-bundle checkpoints.
        "torch_rng": torch.get_rng_state(),
    }
    if rng_streams is not None:
        rng_state = _snapshot_phase2_rng_state(rng_streams)
        payload["rng_state"] = rng_state
        cuda_state = rng_state["global_torch"]["cuda"]
        if cuda_state is not None:
            payload["torch_cuda_rng"] = cuda_state
    if atomic:
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
    else:
        torch.save(payload, path)
    return path


def _step_ckpt_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"step_{int(step):06d}.pt"


def _save_last_pt(
    run_dir: Path,
    *,
    model,
    optimizer,
    al: AugmentedLagrangian,
    step: int,
    cfg: Phase2Config,
    model_cfg: ModelConfig,
    scenario_cfg,
    ls_rt,
    extra: dict | None = None,
    rng_streams: dict | None = None,
) -> None:
    save_phase2_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        al=al,
        step=step,
        cfg=cfg,
        model_cfg=model_cfg,
        scenario_cfg=scenario_cfg,
        extra=_checkpoint_extra_with_ls(extra, ls_rt, optimizer=optimizer),
        rng_streams=rng_streams,
    )


def _save_step_pt(
    run_dir: Path,
    *,
    model,
    optimizer,
    al: AugmentedLagrangian,
    step: int,
    cfg: Phase2Config,
    model_cfg: ModelConfig,
    scenario_cfg,
    ls_rt,
    extra: dict | None = None,
    rng_streams: dict | None = None,
) -> Path | None:
    keep = int(cfg.ckpt_keep_every)
    if keep <= 0 or int(step) % keep != 0:
        return None
    path = _step_ckpt_path(run_dir, step)
    save_phase2_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        al=al,
        step=step,
        cfg=cfg,
        model_cfg=model_cfg,
        scenario_cfg=scenario_cfg,
        extra=_checkpoint_extra_with_ls(extra, ls_rt, optimizer=optimizer),
        rng_streams=rng_streams,
    )
    return path


def _checkpoint_extra_with_ls(extra: dict | None, ls_rt, *, optimizer=None) -> dict:
    out = dict(extra or {})
    if ls_rt is None:
        return out
    out.update(pack_line_search_checkpoint_extra(
        ls_rt.cfg,
        ls_rt.accept,
        ls_rt.ctrl,
        ls_rt.refs,
        fixed_group_ratios=ls_rt.fixed_group_ratios,
        eta_ref=ls_rt.eta_ref,
        seed_provenance=ls_rt.seed_provenance,
        objective_direction_fingerprint=ls_rt.objective_direction_fingerprint,
        optimizer=optimizer if optimizer is not None else getattr(
            getattr(ls_rt, "adapter", None), "opt", None,
        ),
    ))
    return out


def load_phase2_checkpoint(
    path, *, model=None, optimizer=None, al=None, map_location=None,
    rng_streams: dict | None = None,
):
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if al is not None and ck.get("al") is not None:
        al.load_state_dict(ck["al"])
    if rng_streams is not None:
        _restore_phase2_rng_state(ck, rng_streams)
    return ck


# ============================================================ extra losses ===
def make_tc_step(module, cost: FusedCost, sim: BatchedPDSim, cfg: Phase2Config,
                 *, compile_mode: str | None = "default"):
    """Compiled temporal-consistency term (plan lines 102-107).

    Solve at horizon H; advance the noised PD by exactly one control tick with
    the plan's own first waypoint; solve again from the state it reached, one
    waypoint shorter; penalize disagreement on the overlap.  Gradient flows
    through *both* solves and the differentiable plant.
    """
    names = tuple(cost.joint_names)
    lo, hi = urdf_position_limits(names)
    q_lo = torch.as_tensor(lo, device=cost.device, dtype=cost.dtype)
    q_hi = torch.as_tensor(hi, device=cost.device, dtype=cost.dtype)
    step_max = torch.as_tensor(
        isaac_velocity_limits(names) * cost.dt, device=cost.device, dtype=cost.dtype
    )
    lead_f = lead_fraction(cfg.cmd_lead_s, cost.dt, len(names),
                           device=cost.device, dtype=cost.dtype)

    def fn(init, state, q0, v0, enc, masks, masks2, kp, kd):
        core = module
        out1 = core(init, state, enc, masks)
        traj = out1.traj
        with torch.autocast(device_type="cuda", enabled=False):
            q032, v032 = q0.float(), v0.float()
            t0 = (traj[:, 0].float() if lead_f is None
                  else lead_waypoints(traj.float(), lead_f)[:, 0])
            cmd = torch.clamp(
                q032 + torch.clamp(t0 - q032, -step_max, step_max), q_lo, q_hi,
            )
            Minv = torch.linalg.inv(mass_matrix(cost.rnea, q032))
            q1, v1, _int = sim.step(q032, cmd, q032, v032, kp=kp.float(), kd=kd.float(),
                              Minv=Minv)
        init2 = torch.cat([traj[:, 1:], traj[:, -1:]], dim=1) * masks2.elem_mask
        state2 = torch.cat([q1, v1], dim=-1)
        out2 = core(init2, state2, enc, masks2)
        loss, n = temporal_consistency_loss(
            traj, out2.traj, masks.elem_mask.to(traj.dtype),
            masks2.elem_mask.to(traj.dtype),
        )
        return loss, n

    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode, dynamic=False)


def suffix_offsets(
    hit: torch.Tensor,      # (B, 2) long, -1 = idle hand
    active: torch.Tensor,   # (B, 2) bool / 0-1
    H: int,
    *,
    generator: torch.Generator | None = None,
    min_offset: int = 0,
) -> torch.Tensor:
    """``(B,)`` random suffix offsets ``k`` with every active strike kept ahead.

    ``k`` is uniform on ``[0, k_max]`` with ``k_max = min(active hits) - 1``, so
    the shifted hit index stays >= 1 -- the subproblem always still has its
    strike in front of it.  A row whose nearest strike is imminent
    (``k_max = 0``) and only that row draws ``k = 0``: the "suffix" is then the
    whole problem re-solved with the parent plan as init, i.e. idempotence.  An
    all-idle row has no strike to protect and draws from the full grid.
    """
    act = active.bool()
    span = torch.where(act, hit, torch.full_like(hit, int(H)))
    kmax = (span.min(dim=1).values - 1).clamp_min(0)
    min_offset = int(min_offset)
    if min_offset not in (0, 1):
        raise ValueError("suffix min_offset must be 0 or 1")
    u = torch.rand(kmax.shape[0], device=hit.device, generator=generator)
    if min_offset == 0:
        return torch.minimum((u * (kmax + 1).to(u.dtype)).long(), kmax)
    # A verified-replay child should start at an actual later waypoint when
    # one exists.  Imminent strikes still fall back to the k=0 idempotence row.
    positive = 1 + torch.minimum((u * kmax.to(u.dtype)).long(), (kmax - 1).clamp_min(0))
    return torch.where(kmax > 0, positive, torch.zeros_like(kmax))


def terminal_mixture_suffix_offsets(
    hit: torch.Tensor,
    active: torch.Tensor,
    H: int,
    *,
    generator: torch.Generator | None = None,
    peak_frac: float = 0.55,
    long_frac: float = 0.42,
    low_frac: float = 0.03,
    peak_wp: int = 5,
    tail_tau: float = 8.0,
    component: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample remaining TTH from the freeze-peaked three-component mixture.

    The peak component has a sharp discontinuity below ``peak_wp`` and an
    exponential long tail. The long component is uniform from the peak through
    each parent's available horizon. The low component is geometric over the
    below-freeze waypoints. Components are normalized per parent and the final
    mixture is renormalized when a short parent cannot support one of them.
    """
    mix = (float(peak_frac), float(long_frac), float(low_frac))
    peak_wp, tail_tau = int(peak_wp), float(tail_tau)
    if any(x < 0.0 or not math.isfinite(x) for x in mix) or not math.isclose(
        sum(mix), 1.0, rel_tol=0.0, abs_tol=1.0e-6,
    ):
        raise ValueError("terminal suffix mixture must sum to one")
    if peak_wp < 2 or tail_tau <= 0.0:
        raise ValueError("invalid terminal suffix peak/tail")
    rows = int(hit.shape[0])
    if rows == 0:
        return torch.zeros(0, device=hit.device, dtype=torch.long)
    act = active.bool()
    nearest = torch.where(act, hit, torch.full_like(hit, int(H))).min(1).values
    kmax = (nearest - 1).clamp_min(0)
    offsets = torch.zeros_like(kmax)
    usable = (kmax >= 1).nonzero(as_tuple=True)[0]
    if not int(usable.numel()):
        return offsets
    remaining_grid = torch.arange(
        1, int(H), device=hit.device, dtype=torch.float32,
    )
    valid = remaining_grid.unsqueeze(0) <= kmax.index_select(0, usable).unsqueeze(1)
    below = remaining_grid < float(peak_wp)
    at = remaining_grid == float(peak_wp)
    above = remaining_grid > float(peak_wp)
    peak_raw = torch.where(
        below,
        0.10 * torch.exp(-(float(peak_wp - 1) - remaining_grid)),
        torch.where(
            at, torch.ones_like(remaining_grid),
            torch.exp(-(remaining_grid - float(peak_wp)) / tail_tau),
        ),
    ).unsqueeze(0) * valid
    long_raw = (
        (remaining_grid >= float(peak_wp)).unsqueeze(0) * valid
    ).to(torch.float32)
    low_raw = torch.where(
        below, torch.pow(2.0, remaining_grid - 1.0),
        torch.zeros_like(remaining_grid),
    ).unsqueeze(0) * valid

    def _normalize(component: torch.Tensor) -> torch.Tensor:
        return component / component.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

    mixture_probability = (
        mix[0] * _normalize(peak_raw)
        + mix[1] * _normalize(long_raw)
        + mix[2] * _normalize(low_raw)
    )
    if component is None:
        probability = mixture_probability
    else:
        selected_component = component.to(hit.device).index_select(0, usable).long()
        if bool(((selected_component < 0) | (selected_component > 2)).any()):
            raise ValueError("terminal suffix component ids must be 0, 1, or 2")
        stack = torch.stack([
            _normalize(peak_raw), _normalize(long_raw), _normalize(low_raw),
        ], dim=1)
        probability = stack[
            torch.arange(int(usable.numel()), device=hit.device),
            selected_component,
        ]
        unavailable = probability.sum(dim=1) <= 0.0
        probability = torch.where(
            unavailable.unsqueeze(1), mixture_probability, probability,
        )
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    remaining = torch.multinomial(
        probability, 1, replacement=True, generator=generator,
    )[:, 0] + 1
    offsets.index_copy_(0, usable, nearest.index_select(0, usable) - remaining)
    return offsets


def shift_plan(
    traj: torch.Tensor,     # (B, H, n)
    offs: torch.Tensor,     # (B,) long, per-row left shift
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row left shift by ``offs``, repeating the last waypoint.

    Returns ``(shifted, valid)`` with ``valid`` the ``(B, H)`` mask of
    positions that genuinely overlap the source (the repeated pad rows are
    False).  The same gather serves the child's init, the tie target and the
    shifted parent element mask, so the three cannot disagree.
    """
    B, H, D = traj.shape
    ar = torch.arange(H, device=traj.device)
    idx = (ar.unsqueeze(0) + offs.unsqueeze(1)).clamp_max(H - 1)
    shifted = traj.gather(1, idx.unsqueeze(-1).expand(B, H, D))
    valid = ar.unsqueeze(0) < (H - offs).unsqueeze(1)
    return shifted, valid


def make_suffix_step(module, cost: FusedCost, cfg: Phase2Config,
                     *, compile_mode: str | None = "default"):
    """Re-solve a detached, verified parent tail and distil its Bellman identity.

    The parent was admitted by the exact task/physics gate on an earlier
    update.  The horizon shrinks to the strike, so the tail of that successful
    plan is the teacher for the state it passes through.  Both the handoff and
    target are detached: gradients update only the new child solve.  Unlike
    the former online suffix branch, this function intentionally does *not*
    evaluate ``FusedCost(child)``; ordinary primary rows remain responsible
    for physics optimization while this branch trains replanning identity.
    """
    def _model(init, state, enc, masks):
        return module(init, state, enc, masks)

    if compile_mode is not None and cfg.compile_model:
        _model = torch.compile(_model, mode=compile_mode, dynamic=False)

    def fn(traj, q0, v0, enc, em_parent, masks_s, offs):
        B, H, D = traj.shape
        parent = traj.detach()
        shifted, valid = shift_plan(parent, offs)
        # -- handoff state from the PARENT plan's own spline, detached --------
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            t32 = parent.float()
            em32 = em_parent.float()
            sp = cost.spline.from_masks(q0.float(), t32, v0.float(), em32)
            end = torch.full((1,), float(cost.dt), device=traj.device,
                             dtype=torch.float32)
            _q, qd, _qdd, _j = sp.eval_offsets(end)      # (B, H, 1, D)
            vel = qd[:, :, 0, :]
            km1 = (offs - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, D)
            qk = t32.gather(1, km1)[:, 0]
            vk = vel.gather(1, km1)[:, 0]
            fresh = (offs == 0).view(B, 1)               # k = 0: original state
            q0_s = torch.where(fresh, q0.float(), qk)
            v0_s = torch.where(fresh, v0.float(), vk)
        em_s = masks_s.elem_mask.to(traj.dtype)
        child_init = shifted.detach() * em_s
        out = _model(child_init, torch.cat([q0_s, v0_s], dim=-1), enc, masks_s)
        m = em_s * shift_plan(em_parent, offs)[0].to(traj.dtype) \
            * valid.unsqueeze(-1).to(traj.dtype)
        tie, _sq, n = _masked_huber(out.traj, shifted.detach(), m, 0.02)
        return (
            tie, n.detach(), q0_s.detach(), v0_s.detach(), child_init.detach(),
        )

    return fn


def _run_suffix_step_chunked(
    fn,
    traj: torch.Tensor,
    q0: torch.Tensor,
    v0: torch.Tensor,
    enc,
    em_parent: torch.Tensor,
    masks_s,
    offs: torch.Tensor,
    *,
    microbatch: int,
):
    """Run a suffix solve in row chunks without changing its batch mean."""
    n = int(traj.shape[0])
    cap = int(microbatch)
    if cap <= 0 or n <= cap:
        return fn(
            traj, q0, v0, enc, em_parent, masks_s, offs,
        )
    loss_tie = traj.sum() * 0.0
    tie_elems = traj.sum().detach() * 0.0
    tails: list[list[torch.Tensor]] = [[], [], []]
    device = traj.device
    for start in range(0, n, cap):
        idx = torch.arange(start, min(start + cap, n), device=device)
        masks_part = type(masks_s)(**{
            name: (
                None if getattr(masks_s, name) is None
                else getattr(masks_s, name).index_select(0, idx)
            )
            for name in masks_s.__dataclass_fields__
        })
        out = fn(
            traj.index_select(0, idx),
            q0.index_select(0, idx),
            v0.index_select(0, idx),
            _index_encoded(enc, idx),
            em_parent.index_select(0, idx),
            masks_part,
            offs.index_select(0, idx),
        )
        weight = float(idx.numel()) / float(n)
        loss_tie = loss_tie + weight * out[0]
        tie_elems = tie_elems + out[1]
        for values, value in zip(tails, out[2:]):
            values.append(value)
    return loss_tie, tie_elems, *(torch.cat(values, dim=0) for values in tails)


def _strike_rows(s: StrikeContext, sl) -> StrikeContext:
    """Row-slice every populated field of a :class:`StrikeContext`."""
    return StrikeContext(
        **{k: (None if v is None else v[sl]) for k, v in s.__dict__.items()}
    )


def _admissible_command(q0, traj, q_lo, q_hi, step_max, lead_f=None):
    """Clip a plan into the joint box and slew-limit it, differentiably.

    ``torch.clamp`` passes gradient through the interior, so a plan already
    inside the actuator's envelope is untouched and one outside it gets
    gradient only from the part the plant could actually execute -- which is
    the honest signal.
    """
    traj = lead_waypoints(traj, lead_f)
    prev = q0
    rows = []
    for i in range(traj.shape[1]):
        cur = torch.clamp(
            prev + torch.clamp(traj[:, i] - prev, -step_max, step_max), q_lo, q_hi
        )
        rows.append(cur)
        prev = cur
    return torch.stack(rows, dim=1)


def make_bptt_step(module, cost: FusedCost, sim: BatchedPDSim, cfg: Phase2Config,
                   *, compile_mode: str | None = "default"):
    """Compiled two-hit chain (the plan's phase 3, folded into the tail).

    Solve hit 1, roll the **differentiable** noised PD through the whole first
    plan, gather the state at that hand's hit knot, bounce, then solve hit 2
    from there and price it.  Two layers of BPTT: the graph is kept through the
    plant between the two solves and nothing older is retained.
    """
    names = tuple(cost.joint_names)
    lo, hi = urdf_position_limits(names)
    q_lo = torch.as_tensor(lo, device=cost.device, dtype=cost.dtype)
    q_hi = torch.as_tensor(hi, device=cost.device, dtype=cost.dtype)
    step_max = torch.as_tensor(
        isaac_velocity_limits(names) * cost.dt, device=cost.device, dtype=cost.dtype
    )
    lead_f = lead_fraction(cfg.cmd_lead_s, cost.dt, len(names),
                           device=cost.device, dtype=cost.dtype)

    passes_per_solve = int(cfg.bptt_decoder_passes) // 2

    def decoder_solve(core, init, state, enc, masks):
        out = core(init, state, enc, masks)
        for _ in range(passes_per_solve - 1):
            out = core(out.traj, state, enc, masks)
        return out

    def fn(init1, state1, q0, v0, enc1, masks1, strike1, len1, hit1,
           term2, enc2, masks2, strike2, len2, elen2, weights, kp, kd,
           hand_index, normal, e_rest, noise_q, noise_v, noise_tip):
        core = module
        out1 = decoder_solve(core, init1, state1, enc1, masks1)
        traj1 = out1.traj
        c1_total = cost_tuple(cost, q0, traj1, v0, strike1, len1, weights)[0]

        # --- differentiable PD roll through the first plan -------------------
        # The plant is driven by an ADMISSIBLE command: clipped into the joint
        # box and slew-limited to the speed rating, exactly as the roller and
        # the real controller do.  Without it a mid-training plan with a 3 rad
        # step drives the servo unstable over 25 knots and the whole chain goes
        # non-finite -- measured, this was the only NaN source in the run.
        # fp32, ALWAYS: the servo recursion is 25 knots x 5 substeps of
        # accumulation and bf16 makes it diverge outright (measured: the whole
        # chained term returns NaN under autocast, fp32 gives 219.9).
        with torch.autocast(device_type="cuda", enabled=False):
            t32 = traj1.float()
            q032, v032 = q0.float(), v0.float()
            cmd_seq = _admissible_command(q032, t32, q_lo, q_hi, step_max, lead_f)
            x_cmd = torch.cat([q032.unsqueeze(1), cmd_seq], dim=1)
            Minv = torch.linalg.inv(mass_matrix(cost.rnea, q032)).detach()
            qh, vh, _ = sim.rollout(x_cmd, qa0=q032, va0=v032, kp=kp.float(), kd=kd.float(),
                                     Minv=Minv)
            gi = hit1.clamp(1, t32.shape[1]).view(-1, 1, 1).expand(-1, 1, qh.shape[-1])
            q_hit = qh.gather(1, gi).squeeze(1)
            v_hit = vh.gather(1, gi).squeeze(1)

            # --- bounce (placeholder contact model) --------------------------
            J = torch.stack([tip_jacobian(cost.fk, q_hit, h) for h in _HANDS], dim=1)
            sel = hand_index.view(-1, 1, 1, 1).expand(-1, 1, 3, q_hit.shape[1])
            Jh = J.gather(1, sel).squeeze(1)
            v_tip = torch.matmul(Jh, v_hit.unsqueeze(-1)).squeeze(-1)
            v_new = bounce_tip_velocity(v_tip, normal.float(), e_rest.float()) + noise_tip
            M = mass_matrix(cost.rnea, q_hit.detach())
            P = inertia_weighted_pinv(Jh, M=M, damping=1e-4)
            v_post = (
                v_hit + torch.matmul(P, (v_new - v_tip).unsqueeze(-1)).squeeze(-1) + noise_v
            )
            q_post = torch.clamp(q_hit + noise_q, q_lo, q_hi)

        # --- second solve, from the post-hit state ---------------------------
        seed2 = strike_seed(
            q_post, v_post, term2, elen2, traj1.shape[1], cost.dt,
            active=strike2.active, joint_names=names,
        )
        out2 = decoder_solve(
            core, seed2 * masks2.elem_mask,
            torch.cat([q_post, v_post], -1), enc2, masks2,
        )
        c2_total = cost_tuple(cost, q_post, out2.traj, v_post, strike2, len2, weights)[0]
        return c1_total.mean(), c2_total.mean()

    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode, dynamic=False)


# ============================================================ rolling eval ===
def eval_subset(cases, n: int):
    """Evenly subsample a grid, keeping every cell represented.

    ``cases[::stride][:n]`` does NOT do that.  The grid is drum-major, and when
    ``stride`` does not divide the length the slice runs off the end and the
    tail is truncated: at 189 cases (7 drums) and ``n = 81`` the stride is 2, so
    it kept indices 0..160 and dropped FLOOR_TOM -- indices 162-188 -- entirely.
    Turning ``include_extra_drums`` on therefore changed nothing, silently.

    Spreading the picks across the whole range instead touches every drum by
    construction, whatever the ratio.
    """
    cases = list(cases)
    n = int(n)
    if n >= len(cases) or n <= 0:
        return cases
    return [cases[i * len(cases) // n] for i in range(n)]


@torch.no_grad()
def elbow_down_rate(cost: FusedCost, plans: dict[int, np.ndarray], margin: float = 0.05) -> float:
    """Fraction of trajectories whose elbows never rise above the chord.

    The stage-A gate the plan asks for.  ``CaseScore`` does not carry it, so it
    is measured here directly off the FK the cost uses.
    """
    ok = 0
    total = 0
    for Q in plans.values():
        q = torch.as_tensor(np.asarray(Q, float), device=cost.device, dtype=cost.dtype)
        fr = cost.fk.forward(q)
        worst = -1e9
        for hand in _HANDS:
            ze = fr[f"{hand}_elbow"][1][:, 2]
            zw = fr[f"{hand}_wrist"][1][:, 2]
            zs = fr[f"{hand}_shoulder"][1][:, 2]
            drop = 0.5 * (zs + zw) - ze
            worst = max(worst, float((-(drop + margin)).max()))
        ok += int(worst <= 0.0)
        total += 1
    return ok / max(total, 1)


def rolling_eval(
    model, cases, scorer, *, device, kit, library, buckets, cost=None,
    iterations: int = 1, amp_dtype=None, num_drums=8, num_future=3, init="quintic",
) -> dict:
    """Batched solve + harness scoring, reduced to the gates the plan names."""
    from .eval_harness import aggregate, run_eval
    from .planner import batched_model_plans, precomputed_planner

    plans, solve_s = batched_model_plans(
        model, cases, device=str(device), kit=kit, library=library, buckets=buckets,
        num_drums=num_drums, num_future=num_future, iterations=iterations,
        init=init, amp_dtype=amp_dtype,
    )
    rows, _ = run_eval(
        precomputed_planner(plans, solve_s / max(len(cases), 1)), cases, scorer=scorer
    )
    agg = aggregate(rows)
    n = max(agg["n"], 1)
    short = [r for r in rows if r.N <= 24]
    out = {
        "n": agg["n"],
        "accepted": agg["counts"]["accepted"]["count"],
        "acceptance": agg["counts"]["accepted"]["count"] / n,
        "ok_hit": agg["counts"]["ok_hit"]["count"] / n,
        "ok_speed": agg["counts"]["ok_speed"]["count"] / n,
        "ok_collision": agg["counts"]["ok_collision"]["count"] / n,
        "ok_torque": agg["counts"]["ok_torque"]["count"] / n,
        "ok_velocity": agg["counts"]["ok_velocity"]["count"] / n,
        "ok_joint_limits": agg["counts"]["ok_joint_limits"]["count"] / n,
        "elev_ok": agg["counts"]["elev_ok"]["count"] / n,
        "contact_pos_err_mm_p50": agg["metrics"]["contact_pos_err_mm"]["p50"],
        "speed_err_p50": agg["metrics"]["contact_speed_err_mps"]["p50"],
        "pen_max_mm_p90": agg["metrics"]["coll_pen_max_m"]["p90"],
        "tip_lateral_min_m_p10": agg["metrics"]["tip_lateral_min_m"]["p10"],
        "tau_excess_p90": agg["metrics"]["tau_excess_max_nm"]["p90"],
        "jerk_p50": agg["metrics"]["jerk_integral"]["p50"],
        "travel_p50": agg["metrics"]["travel_l1_rad"]["p50"],
        "short_n": len(short),
        "short_accepted": sum(1 for r in short if r.accepted),
        "solve_s_total": solve_s,
    }
    if cost is not None:
        out["elbow_down"] = elbow_down_rate(cost, plans)
    return out


#: Rolling-eval grid is 81 cases.  Stage A unlocks the post-hit (Phase B)
#: pool once this many are accepted -- 30/81 is enough that the long-horizon
#: plan is on the kit and MPC can clean up the rest.
STAGE_A_MIN_ACCEPTED: int = 30
STAGE_A_EVAL_N: int = 81


def stage_a_pass(ev: dict) -> bool:
    """Unlock post-hit distillation: accepted >= 30/81, coll-free >= 66 %, elbow >= 95 %."""
    n = max(int(ev.get("n", 0)), 1)
    accepted = float(ev.get("accepted", 0.0))
    return (
        accepted / n >= STAGE_A_MIN_ACCEPTED / STAGE_A_EVAL_N
        and ev.get("ok_collision", 0.0) >= 0.66
        and ev.get("elbow_down", 0.0) >= 0.95
    )


# ============================================================== the loop =====
def build_optimizer(model, cfg: Phase2Config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.dim() < 2 or "emb" in name) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=tuple(cfg.betas),
    )


def migrate_al_state(state: dict | None) -> dict | None:
    """Make a checkpointed AL state usable by the CURRENT constraint groups.

    ``torque_peak`` needs a SEEDED multiplier.  Before it was its own group the
    peak half of the torque term was multiplied by exactly ``lam_torque``, so
    seeding ``lam["torque_peak"] = lam["torque"]`` is what makes the objective
    CONTINUOUS across a resume; leaving it out would silently drop the peak
    price back to ``lam_init``.

    Actuator-class splits of torque / velocity / joint_box are seeded from the
    parent family multiplier the same way: a checkpoint that paid ``lam_torque``
    on every joint should keep paying that price on every class, not reset the
    wrists to 1 while the rest of the run sits at 100.

    ``collision_peak`` deliberately does NOT get one.  It is not a regrouping
    of a price that was already being paid -- the smooth-max half of the
    collision term did not exist in any form before it -- so there is no
    previous effective multiplier to preserve, and ``lam_init`` is the correct
    starting point.

    ``waist_pr_band`` is the same kind of new price: it did not previously
    ride ``lam_joint_box`` or ``lam_torque``, so seeding from those would
    silently apply the wrong units.  Missing keys stay at ``lam_init`` via
    :meth:`~amortized_opt.lagrangian.AugmentedLagrangian.load_state_dict`
    ``update``.

    ``avg_violation`` is deliberately NOT seeded for splits or peaks: the stored
    value is a different reduction (element mean vs mean-of-per-joint-max vs
    global max), so the EMA is better off warming up from zero.

    Returns a NEW dict (the caller's checkpoint payload is not mutated), or the
    argument unchanged when there is nothing to do.
    """
    if not state:
        return state
    lam = dict(state.get("lam", {}))
    groups = list(state.get("groups", [])) if "groups" in state else None
    changed = False
    if "torque" in lam and "torque_peak" not in lam:
        lam["torque_peak"] = float(lam["torque"])
        changed = True
    for fam in LIMIT_FAMILIES:
        src = lam.get(fam)
        if src is None:
            continue
        for seg in ACTUATOR_SEGMENT_NAMES:
            key = limit_group_name(fam, seg)
            if key not in lam:
                lam[key] = float(src)
                changed = True
    for g in ("torque_peak", "collision_peak", "joint_box_peak",
              "waist_pr_band", "velocity_peak"):
        if groups is not None and g not in groups:
            groups.append(g)
            changed = True
    if groups is not None:
        for key in LIMIT_GROUP_KEYS:
            if key not in groups:
                groups.append(key)
                changed = True
    if not changed:
        return state
    out = dict(state)
    out["lam"] = lam
    if groups is not None:
        out["groups"] = groups
    return out


def clamp_resumed_al_penalty(value: float, lower: float, upper: float) -> float:
    """Clamp an enabled resumed PHR penalty without enabling a disabled one.

    An exact zero is a semantic switch: :class:`AugmentedLagrangian` omits the
    quadratic term entirely.  Resume-time config reconciliation must therefore
    preserve zero even when the current positive-penalty floor is nonzero.
    """
    value = float(value)
    if value == 0.0:
        return 0.0
    return float(min(max(value, float(lower)), float(upper)))


#: Groups whose price :attr:`WeightSchedule.full_limit_joints` makes per joint.
_LIMIT_GROUPS: tuple[str, ...] = LIMIT_GROUP_KEYS


def _wtensors(mult: dict[str, float], lam: dict[str, float], device, dtype,
              *, sched: "WeightSchedule | None" = None, joint_names=None,
              step: int = 0, stage_a: bool = True,
              pen: dict[str, float] | None = None):
    """Curriculum multipliers and AL λ, as 0-dim tensors (compile-stable).

    Constraint-group keys (``CONSTRAINT_GROUPS``, per-segment limit keys,
    family broadcast keys, ``collision``, ``terminal``, ``waist_pr_band``)
    carry **λ only** -- the dual multiplies the ReLU residual, never the
    softplus warning.  Peak groups keep their curriculum alpha so
    ``weights["torque_peak"] = alpha * λ`` ramps the hard peak product
    without shocking a resume (same for ``collision_peak``).

    Curriculum that is not a dual (``terminal_region`` / ``terminal_velocity``
    / ``terminal_elev``, ``waist_box``, ``posture_*``, ``smooth``, ``effort``,
    ``tip_crossover``) stays as the schedule value.

    With ``sched`` and ``joint_names`` given, the schedule's per-joint limit
    ramp (:meth:`WeightSchedule.limit_alpha_vector`) is emitted as an extra
    ``(D,)`` entry ``limit_alpha`` which the cost applies to the torque /
    velocity / joint-box **softplus** hinges PER ELEMENT.  The vector never
    scales ReLU ``live``.  Family scalars are already λ-only, so the old
    de-ramp (strip ``lim`` out of the scalar once the vector carries it) is
    the default path, not a special case.  When the vector is absent, do
    **not** fold ``lim`` into λ -- warning uses full ``cfg.w_*``.

    ``pen`` (the PHR quadratic coefficients) is emitted as ``pen_<group>``
    0-dim tensors, omitted when empty / all-zero so a caller that does not
    pass it is byte-for-byte the pre-quadratic dict.

    All three consumers of this dict (the cost step, the BPTT term, the suffix
    step) forward it whole to ``FusedCost.forward(weights=...)``.
    """
    peak_keys = {"torque_peak", "velocity_peak", "collision_peak", "joint_box_peak"}
    al_keys = set(CONSTRAINT_GROUPS) | set(LIMIT_FAMILIES)
    out = {}
    for k, v in mult.items():
        if k in peak_keys:
            val = float(v) * float(lam.get(k, 1.0))
        elif k in al_keys:
            val = float(lam.get(k, 1.0))
        else:
            val = float(v)
        out[k] = torch.tensor(val, device=device, dtype=dtype)
    # Expand torque / velocity / joint_box into per-actuator-class keys so the
    # cost can price each class with its own λ.  Family keys stay in ``out``
    # (broadcast fallback); segment keys win in FusedCost when present.
    for fam in LIMIT_FAMILIES:
        if fam not in mult:
            continue
        for seg in ACTUATOR_SEGMENT_NAMES:
            key = limit_group_name(fam, seg)
            out[key] = torch.tensor(
                float(lam.get(key, lam.get(fam, 1.0))),
                device=device, dtype=dtype,
            )
    if pen:
        for g, v in pen.items():
            if float(v) == 0.0:
                continue
            out[f"pen_{g}"] = torch.tensor(float(v), device=device, dtype=dtype)
    if sched is None or joint_names is None:
        return out
    vec = sched.limit_alpha_vector(step, joint_names, stage_a)
    if vec is None:
        pass
    else:
        out["limit_alpha"] = torch.as_tensor(vec, device=device, dtype=dtype)
    out["torque_full_blend"] = torch.tensor(
        sched.torque_full_blend(step), device=device, dtype=dtype,
    )
    return out


def _voxel_mask_al_weights(
    weights: dict[str, torch.Tensor], al_ok: torch.Tensor, *, probe_penalty: float,
) -> dict[str, torch.Tensor]:
    """Zero dual prices on unknown probes and give them a fixed penalty.

    Soft objective terms are untouched.  Confirmed rows retain the current AL
    lambda/PHR coefficient; probe rows cannot influence multiplier updates and
    cannot inherit a coefficient that may have grown while solving a different
    confirmed feasible region.
    """
    out = dict(weights)
    mask = al_ok.to(
        device=next(iter(weights.values())).device, dtype=torch.float32,
    ).reshape(-1)
    dual_keys = set(CONSTRAINT_GROUPS) | set(LIMIT_FAMILIES) | set(LIMIT_GROUP_KEYS)
    for key in dual_keys:
        value = out.get(key)
        if value is not None:
            out[key] = value * mask.to(value.dtype)
    for group in CONSTRAINT_GROUPS:
        key = f"pen_{group}"
        current = out.get(key)
        if current is None:
            current = mask.new_zeros(())
        out[key] = (
            current * mask.to(current.dtype)
            + float(probe_penalty) * (1.0 - mask).to(current.dtype)
        )
    out["_al_ok"] = mask
    return out


# =========================== zeroth-order waist set-point search =============
#: Terms whose value depends on the very set-point the search is choosing.
#: ``waist_aim`` prices ``||traj_yaw - waist_yaw_ref||`` and ``waist_yaw_ref``
#: IS the candidate, so a candidate scored with it on would be graded on its
#: own consistency -- the exact self-reference the search exists to escape.
#: The residual tether is not a cost term at all (the training loop applies it
#: to ``enc.waist_yaw_res`` directly), so it never reaches this dict.
WAIST_SP_SELF_TERMS: tuple[str, ...] = ("waist_aim",)


def outcome_only_weights(weights: dict) -> dict:
    """``weights`` with :data:`WAIST_SP_SELF_TERMS` zeroed, everything else kept.

    "Everything else" is the point: terminal / hit, collision (stick-stick
    included when it is on), torque, velocity, joint box and posture keep the
    CURRENT step's effective weights -- schedule multipliers, AL multipliers and
    the ``limit_alpha`` per-joint ramp vector alike -- because a candidate yaw
    has to be judged by the same objective the optimizer is descending, minus
    only the self-referential part.
    """
    out = dict(weights)
    for k in WAIST_SP_SELF_TERMS:
        v = out.get(k)
        out[k] = torch.zeros_like(v) if torch.is_tensor(v) else 0.0
    return out


def waist_sp_bound() -> float:
    """Largest candidate magnitude, rad: URDF waist-yaw box AND the head clamp.

    The URDF box is +-2.618 rad but ``ContextEncoder`` clamps its set-point to
    ``waist_aim``'s +-1.74533 rad policy limit, so a target beyond that is
    literally unreachable -- regressing toward one asks the head for a residual
    it can never apply, and the loss would never go to zero.  The bound is the
    tighter of the two.
    """
    lo, hi = urdf_position_limits(("waist_yaw_joint",))
    return float(min(abs(float(lo[0])), float(hi[0]), _WAIST_SP_LIMIT))


def waist_sp_offsets(k: int, span: float) -> list[float]:
    """The symmetric offset ladder, ``[0, +d1, -d1, ...]``, ``|d| <= span``.

    QUADRATIC spacing, dense near the incumbent: the head is already close to a
    decent answer most of the time, so resolution near zero is what makes the
    softmin target move smoothly instead of snapping between rungs, while the
    outermost rung still reaches ``span`` so a genuinely wrong set-point can be
    escaped in one step.  ``k`` counts the TOTAL candidates including the
    mandatory aim column, so the ladder gets ``k - 1`` slots and an odd ``k``
    rounds down (``k=8`` -> ``{0, +-span/9, +-4span/9, +-span}`` + aim = 8).
    """
    n_pairs = max((int(k) - 2) // 2, 1)
    out = [0.0]
    for j in range(1, n_pairs + 1):
        d = float(span) * (j / n_pairs) ** 2
        out += [d, -d]
    return out


def waist_sp_candidates(sp, aim, *, k: int = 8, span: float = 0.8,
                        bound: float | None = None) -> torch.Tensor:
    """``(B, K)`` candidate waist yaws, clamped to :func:`waist_sp_bound`.

    Column layout is FIXED and pinned by the tests: the ladder of
    :func:`waist_sp_offsets` applied to ``sp`` first, then ``aim`` as the LAST
    column.  The analytic aim is mandatory rather than merely likely-nearby:
    it is the one candidate whose value is known to be defensible, so the
    search can always fall back to it, and "how far did the winner land from
    the analytic law" is a well-defined diagnostic on every row.

    ``sp`` should already be detached -- no gradient flows through a candidate.
    """
    if bound is None:
        bound = waist_sp_bound()
    offs = torch.as_tensor(waist_sp_offsets(k, span), device=sp.device, dtype=sp.dtype)
    cand = sp.unsqueeze(1) + offs.unsqueeze(0)
    cand = torch.cat([cand, aim.to(sp.dtype).unsqueeze(1)], dim=1)
    return cand.clamp(-bound, bound)


def substitute_waist_setpoints(
    module,
    enc,
    *,
    yaw_sp=None,
    pr_sp=None,
    yaw_now=None,
    base_yaw=None,
):
    """Rebuild yaw and/or pitch-roll set-point tokens; yaw stays last.

    Encoder extras are ``[pr?, yaw?]`` in that order.  Tokens that are not
    being replaced are copied byte-for-byte from ``enc.memory``, so a yaw-only
    call is bit-identical to :func:`substitute_waist_sp`.  Pitch/roll-only is
    safe when the yaw head is off (the old helper would write a yaw token over
    the P/R slot).
    """
    core = module.module if hasattr(module, "module") else module
    e = core.encoder
    cfg = core.cfg
    has_pr = bool(cfg.waist_pr_sp_head)
    has_yaw = bool(cfg.waist_sp_head)
    n_extra = int(has_pr) + int(has_yaw)
    if n_extra == 0:
        raise ValueError("substitute_waist_setpoints requires a waist set-point head")
    if yaw_sp is not None and not has_yaw:
        raise ValueError("yaw_sp given but waist_sp_head is off")
    if pr_sp is not None and not has_pr:
        raise ValueError("pr_sp given but waist_pr_sp_head is off")
    if yaw_sp is not None and (yaw_now is None or base_yaw is None):
        raise ValueError("yaw_sp requires yaw_now and base_yaw")

    dt = enc.memory.dtype
    mem = enc.memory
    pieces = [mem[:, :-n_extra]]
    yaw_out, yaw_res = enc.waist_yaw_sp, enc.waist_yaw_res
    pr_out, pr_res = enc.waist_pr_sp, enc.waist_pr_res

    if has_pr:
        if pr_sp is None:
            pr_tok = mem[:, -n_extra]
        else:
            pr_sp = pr_sp.to(dt)
            pr_tok = e.waist_pr_sp_proj(pr_sp) + e.waist_pr_sp_emb
            pr_out = pr_sp
            pr_res = pr_sp
        pieces.append(pr_tok.unsqueeze(1) if pr_tok.dim() == 2 else pr_tok)
    if has_yaw:
        if yaw_sp is None:
            yaw_tok = mem[:, -1]
        else:
            yaw_sp = yaw_sp.to(dt)
            yn = yaw_now.to(dt)
            yaw_tok = e.waist_sp_proj(
                torch.stack([yaw_sp, yn, yaw_sp - yn], dim=-1)
            ) + e.waist_sp_emb
            yaw_out = yaw_sp
            yaw_res = yaw_sp - base_yaw.to(dt)
        pieces.append(yaw_tok.unsqueeze(1) if yaw_tok.dim() == 2 else yaw_tok)
    mem2 = torch.cat(pieces, dim=1)
    return type(enc)(mem2, enc.mask, yaw_out, yaw_res, pr_out, pr_res)


def substitute_waist_sp(module, enc, sp_new, yaw_now, base):
    """``enc`` with its waist-set-point token rebuilt at ``sp_new``.

    The set-point reaches the DECODER through exactly one thing: the extra
    context token ``ContextEncoder.forward`` appends *after* its blocks have
    run, ``waist_sp_proj([sp, yaw_now, sp - yaw_now]) + waist_sp_emb``.  Nothing
    upstream of it depends on ``sp``, so re-planning at a candidate yaw costs
    one ``Linear`` on a ``(B, 3)`` input rather than a second pass over the
    encoder.  ``tests/test_waist_sp_search.py`` pins this against a full second
    ``encode_context`` -- if the token construction ever moves, that test fails
    rather than this silently scoring the wrong thing.

    Yaw-only wrapper around :func:`substitute_waist_setpoints`; existing
    phase-2 callers stay bit-identical (P/R token bytes are copied, not
    rebuilt).
    """
    return substitute_waist_setpoints(
        module, enc, yaw_sp=sp_new, yaw_now=yaw_now, base_yaw=base,
    )


def waist_sp_softmin(cand: torch.Tensor, scores: torch.Tensor, temp: float):
    """``(sp*, spread)``: softmin-weighted candidate mix and the V-spread.

    Scores are MIN-MAX NORMALIZED per row before the softmax, which is what
    makes ``temp`` dimensionless -- the cost scale moves by decades over a run
    (the AL multipliers see to that) and a temperature in raw cost units would
    silently anneal from "hard argmin" to "plain mean" as it did.

    ``sp*`` is a convex combination of ``cand``, hence provably inside
    ``[cand.min(), cand.max()]``: the search cannot produce a target outside
    the span it evaluated, whatever the scores are.  ``spread`` is
    ``max - min`` of the RAW scores -- the diagnostic that says whether the
    set-point matters at all on this batch.
    """
    lo = scores.min(dim=1, keepdim=True).values
    hi = scores.max(dim=1, keepdim=True).values
    spread = (hi - lo).squeeze(1)
    z = (scores - lo) / (hi - lo).clamp_min(1e-6)
    p = torch.softmax(-z / max(float(temp), 1e-3), dim=1)
    return (p * cand).sum(dim=1), spread


def clone_context(ctx: ContextInputs) -> ContextInputs:
    """Tensor-wise clone; the batch context must not be mutated by the promote."""
    return ContextInputs(**{
        f.name: getattr(ctx, f.name).clone() for f in fields(ContextInputs)
    })


def slice_context(ctx: ContextInputs, idx: torch.Tensor) -> ContextInputs:
    """``index_select`` every field along the batch axis."""
    return ContextInputs(**{
        f.name: getattr(ctx, f.name).index_select(0, idx) for f in fields(ContextInputs)
    })


def promote_context_next_hit(ctx: ContextInputs) -> tuple[ContextInputs, torch.Tensor]:
    """Clone ``ctx`` and promote each row's soonest current goal to ``future[0]``.

    This is the encoder-visible context at the next per-hand segment boundary
    (one hand fires; the other is untouched).  ``future_gap`` is re-anchored
    the same way ``MPCRoller._regoal`` does it at fire time: the stored gaps
    are already hit-relative, so the promoted hand's new ``goal_tth`` is the
    old ``future_gap[..., 0]`` and surviving gaps are ``old[j+1] - old[0]``.
    The other hand's countdown ages by the wait until that fire (aim-law
    from-now times; the encoder does not embed ``goal_tth``).  Rows with no
    active goal are cloned bit-for-bit and marked unchanged.

    Returns ``(ctx_new, changed)`` with ``changed`` a ``(B,)`` bool.
    """
    out = clone_context(ctx)
    gvalid0 = ctx.goal_valid.to(torch.bool)
    changed = gvalid0.any(dim=1)
    if not bool(changed.any()):
        return out, changed

    B = ctx.batch
    device = ctx.goal_tth.device
    inf = torch.full((), 1e9, device=device, dtype=ctx.goal_tth.dtype)
    tth = torch.where(gvalid0, ctx.goal_tth, inf)
    hand = tth.argmin(dim=1)
    switch = torch.zeros(B, 2, dtype=torch.bool, device=device)
    switch[torch.arange(B, device=device), hand] = changed

    fut_valid0 = out.future_valid.to(torch.bool)[:, :, 0]
    ok = switch & fut_valid0
    idle = switch & ~fut_valid0
    okv, idlev = ok.unsqueeze(-1), idle.unsqueeze(-1)

    nd = out.future_drum[:, :, 0]
    npos = out.future_pos[:, :, 0]
    nvel = out.future_vel[:, :, 0]
    ngap = out.future_gap[:, :, 0]
    z_drum = torch.zeros_like(out.goal_drum)
    z_tth = torch.zeros_like(out.goal_tth)
    z3 = torch.zeros_like(npos)

    out.goal_drum = torch.where(ok, nd, torch.where(idle, z_drum, out.goal_drum))
    out.goal_pos = torch.where(okv, npos, torch.where(idlev, z3, out.goal_pos))
    out.goal_vel = torch.where(okv, nvel, torch.where(idlev, z3, out.goal_vel))
    out.goal_tth = torch.where(ok, ngap, torch.where(idle, z_tth, out.goal_tth))
    out.goal_valid = ((gvalid0 | ok) & ~idle).to(out.goal_valid.dtype)

    dt = torch.where(changed, tth.gather(1, hand.unsqueeze(1)).squeeze(1), z_tth[:, 0])
    other = ~switch
    aged = (out.goal_tth - dt.unsqueeze(1)).clamp_min(0)
    out.goal_tth = torch.where(
        other & out.goal_valid.to(torch.bool), aged, out.goal_tth,
    )

    def _shift(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, :, 1:], torch.zeros_like(x[:, :, :1])], dim=2)

    consume_m = switch.view(B, 2, *([1] * (out.future_gap.dim() - 2)))
    old_gap = ctx.future_gap
    for name in ("future_drum", "future_pos", "future_vel", "future_valid",
                 "future_gap"):
        x = getattr(out, name)
        m = switch.view(B, 2, *([1] * (x.dim() - 2)))
        setattr(out, name, torch.where(m, _shift(x), x))

    new_anchor = torch.where(out.goal_valid.to(torch.bool), out.goal_tth, z_tth)
    new_gap = torch.where(
        out.future_valid.to(torch.bool),
        _shift(old_gap) - new_anchor.unsqueeze(-1),
        torch.zeros_like(out.future_gap),
    )
    out.future_gap = torch.where(consume_m, new_gap, out.future_gap)

    idle_v = out.idle_valid.to(torch.bool)
    out.idle_valid = torch.where(
        switch, ~out.goal_valid.to(torch.bool), idle_v,
    ).to(out.idle_valid.dtype)
    return out, changed


def refresh_context_aim(ctx: ContextInputs, waist_cfg=None, rows=None) -> None:
    """Recompute ``aim_yaw`` / ``aim_valid`` from the context's own tokens.

    Same law and pelvis-frame convention as ``build_context_inputs`` /
    ``MPCRoller._refresh_aim``.  In-place.  ``rows`` is a 1-D index tensor;
    omit it to refresh every row.  A row with nothing upcoming keeps
    ``aim_valid=False`` (hold).
    """
    from .waist_aim import WaistAimConfig, aim_yaw

    wcfg = waist_cfg if waist_cfg is not None else WaistAimConfig()
    if rows is not None and int(rows.numel()) == 0:
        return
    pelvis = np.asarray(PELVIS_IN_ENV, dtype=float)
    if rows is None:
        r_np = None
        n = ctx.batch
        yaw_np = ctx.yaw_now.detach().cpu().numpy()
        gpos = ctx.goal_pos.detach().cpu().numpy()
        gvalid = ctx.goal_valid.detach().cpu().numpy() > 0.5
        gtth = ctx.goal_tth.detach().cpu().numpy()
        fpos = ctx.future_pos.detach().cpu().numpy()
        fgap = ctx.future_gap.detach().cpu().numpy()
        fval = ctx.future_valid.detach().cpu().numpy() > 0.5
    else:
        r_np = rows.detach().cpu()
        n = int(r_np.numel())
        yaw_np = ctx.yaw_now.index_select(0, rows).detach().cpu().numpy()
        gpos = ctx.goal_pos.index_select(0, rows).detach().cpu().numpy()
        gvalid = ctx.goal_valid.index_select(0, rows).detach().cpu().numpy() > 0.5
        gtth = ctx.goal_tth.index_select(0, rows).detach().cpu().numpy()
        fpos = ctx.future_pos.index_select(0, rows).detach().cpu().numpy()
        fgap = ctx.future_gap.index_select(0, rows).detach().cpu().numpy()
        fval = ctx.future_valid.index_select(0, rows).detach().cpu().numpy() > 0.5

    F = int(fpos.shape[2])
    aim = np.zeros(n, dtype=float)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        targets: dict[str, np.ndarray] = {}
        tth: dict[str, float] = {}
        futures: dict[str, list] = {}
        for hi, hand in enumerate(HANDS):
            if gvalid[i, hi]:
                targets[hand] = np.asarray(gpos[i, hi] - pelvis, dtype=float)
                tth[hand] = float(gtth[i, hi])
            futures[hand] = [
                (np.asarray(fpos[i, hi, j] - pelvis, dtype=float),
                 float(fgap[i, hi, j] + (gtth[i, hi] if gvalid[i, hi] else 0.0)))
                for j in range(F)
                if fval[i, hi, j]
            ]
        if not targets and not any(futures.values()):
            continue
        with np.errstate(all="ignore"):
            aim[i] = aim_yaw(
                targets, cfg=wcfg, yaw_current=float(yaw_np[i]),
                goal_tth=tth, futures=futures,
            )
        ok[i] = True

    aim_t = torch.as_tensor(aim, device=ctx.aim_yaw.device, dtype=ctx.aim_yaw.dtype)
    ok_t = torch.as_tensor(ok, device=ctx.aim_valid.device, dtype=ctx.aim_valid.dtype)
    if rows is None:
        ctx.aim_yaw.copy_(aim_t)
        ctx.aim_valid.copy_(ok_t)
    else:
        ctx.aim_yaw.index_copy_(0, rows, aim_t)
        ctx.aim_valid.index_copy_(0, rows, ok_t)


def _waist_sp_smooth_term(*, model, ctx, enc, waist_cfg=None):
    """``mean((sp_new - sp_old)²)`` on the next-hit re-encode, plus |Δ| diagnostics.

    Returns ``(loss, yaw_abs, pr_abs, n)``.  ``n`` is the number of rows that
    actually promoted; diagnostics are 0 when nothing changed.  Yaw and P/R
    are added (each a mean of squares) when that head produced a set-point.
    """
    device = enc.memory.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    ctx_new, changed = promote_context_next_hit(ctx)
    idx = changed.nonzero(as_tuple=True)[0]
    n = int(idx.numel())
    if n == 0:
        return zero, 0.0, 0.0, 0
    refresh_context_aim(ctx_new, waist_cfg=waist_cfg, rows=idx)
    enc_new = encode(model, slice_context(ctx_new, idx))
    loss = zero
    yaw_abs = 0.0
    pr_abs = 0.0
    if enc.waist_yaw_sp is not None and enc_new.waist_yaw_sp is not None:
        d_yaw = enc_new.waist_yaw_sp.float() - enc.waist_yaw_sp.index_select(0, idx).float()
        loss = loss + d_yaw.pow(2).mean()
        yaw_abs = float(d_yaw.detach().abs().mean())
    if enc.waist_pr_sp is not None and enc_new.waist_pr_sp is not None:
        d_pr = enc_new.waist_pr_sp.float() - enc.waist_pr_sp.index_select(0, idx).float()
        loss = loss + d_pr.pow(2).mean()
        pr_abs = float(d_pr.detach().abs().mean())
    return loss, yaw_abs, pr_abs, n


def _waist_sp_search_term(*, model, cost, cfg, enc, batch, strike, lengths,
                          bucket, weights, rows):
    """Search-distillation loss on ``rows`` + its three diagnostics.

    The candidate evaluations are EAGER and under ``no_grad``: the search slice
    is a different batch size from the main step, and ``make_cost_step``
    compiles with ``dynamic=False``, so routing them through the compiled units
    would buy a recompiled graph per bucket per fused group for a forward that
    carries no gradient anyway.  They otherwise reuse exactly the step's
    machinery -- the same module, the same ``FusedCost`` for the bucket, the
    same effective weights (minus :data:`WAIST_SP_SELF_TERMS`), the same strike
    context with ``waist_yaw_ref`` broadcast over the horizon the way the main
    loss does it.
    """
    core = model.module if hasattr(model, "module") else model
    ctx = batch["context"]
    yaw_now = ctx.yaw_now[rows].float()
    base = torch.where(
        ctx.aim_valid[rows].to(torch.bool), ctx.aim_yaw[rows].float(), yaw_now
    )
    sp = enc.waist_yaw_sp[rows].float()
    cand = waist_sp_candidates(
        sp.detach(), base, k=cfg.waist_sp_search_k, span=cfg.waist_sp_search_span,
    )
    n, n_cand = cand.shape
    enc_sub = type(enc)(enc.memory[rows], enc.mask[rows],
                        enc.waist_yaw_sp[rows], enc.waist_yaw_res[rows])
    masks = type(batch["masks"])(
        **{f: getattr(batch["masks"], f)[rows]
           for f in batch["masks"].__dataclass_fields__}
    )
    st = _strike_rows(strike, rows)
    w_out = outcome_only_weights(weights)
    init, state = batch["init"][rows], batch["state"][rows]
    q0, v0 = batch["q0"][rows], batch["v0"][rows]
    lens = lengths[rows]

    # Candidates are evaluated FUSED, several at a time along the batch axis:
    # the search slice is a fraction of the batch, and K separate forwards of
    # it leave the GPU mostly idle in kernel-launch overhead -- measured 2.0x a
    # normal step at frac 0.25 / K 8, against 1.4x once fused.  The group size
    # caps the fused batch at the MAIN step's row count, so peak memory stays
    # bounded by one ordinary forward (and these carry no autograd graph at
    # all, so they are strictly cheaper than the step's own).
    per = max(int(batch["init"].shape[0]) // max(n, 1), 1)
    scores = []
    with torch.no_grad():
        for j0 in range(0, n_cand, per):
            psi = cand[:, j0:j0 + per]                       # (n, g)
            g = psi.shape[1]
            rep = lambda t: (None if t is None                # noqa: E731
                             else t.repeat_interleave(g, dim=0))
            flat = psi.reshape(-1)          # matches repeat_interleave's order
            enc_g = substitute_waist_sp(
                core, type(enc)(rep(enc_sub.memory), rep(enc_sub.mask), None, None),
                flat, rep(yaw_now), rep(base),
            )
            out = core(
                rep(init), rep(state), enc_g,
                type(masks)(**{f: rep(getattr(masks, f))
                               for f in masks.__dataclass_fields__}),
            )
            st_g = StrikeContext(**{k_: rep(v_) for k_, v_ in st.__dict__.items()})
            st_g = replace(
                st_g,
                waist_yaw_ref=flat.to(out.traj.dtype).unsqueeze(1).expand(-1, bucket),
            )
            ct = cost.forward(rep(q0), out.traj, rep(v0), st_g,
                              lengths=rep(lens), weights=w_out)
            scores.append(ct.total.float().view(n, g))
    sp_star, spread = waist_sp_softmin(cand, torch.cat(scores, dim=1),
                                       cfg.waist_sp_search_temp)
    sp_star = sp_star.detach()
    loss = F.smooth_l1_loss(sp, sp_star, beta=0.1)
    diag = (
        float((sp.detach() - sp_star).abs().mean()),
        float((sp_star - base).abs().mean()),
        float(spread.mean()),
    )
    return loss, diag


def _bptt_term(pool, replay, bptt_fn, sim, cfg, bucket, device, dtype, gen, rng, w,
               model=None):
    """Draw a two-hit chain and return its second-solve cost.

    Hit 1 comes from wherever the training mix is drawing (replay if it has
    enough), hit 2 from a *fresh* scenario, so the second context is an honest
    "next drum" rather than a continuation of the same goal.
    """
    n = min(int(cfg.bptt_batch), pool.size(bucket))
    src = replay if (bucket in getattr(replay, "data", {}) and replay.size(bucket) >= n) else pool
    i1 = (src.sample_idx(bucket, n, generator=gen) if src is replay
          else torch.randint(0, pool.size(bucket), (n,), device=device, generator=gen))
    b1 = src.batch(bucket, i1, generator=gen, noise=(src is pool))
    i2 = torch.randint(0, pool.size(bucket), (n,), device=device, generator=gen)
    b2 = pool.batch(bucket, i2, generator=gen, noise=False)

    he1 = effective_hits(b1["hit"], b1["active"], bucket)
    he2 = effective_hits(b2["hit"], b2["active"], bucket)
    # BPTT keeps ``waist_yaw_ref=None`` (see ``Phase2Config.waist_sp_from_head``):
    # the pool batch arrives with the seed's analytic yaw ramp attached, and
    # pricing the second solve against it would fight the learned anchor with
    # ``w_bptt * w_waist_aim`` of gradient.  The main loss is where the
    # ``waist_aim`` anchor lives.
    s1 = replace(b1["strike"], hit_index=he1, waist_yaw_ref=None)
    s2 = replace(b2["strike"], hit_index=he2, waist_yaw_ref=None)

    # the FIRST hit of sample 1: which hand, when, and its head normal
    big = torch.where(
        b1["active"], b1["hit"].clamp_min(0), torch.full_like(b1["hit"], bucket + 1)
    )
    k_first = big.min(dim=1).values.clamp(1, bucket)
    hand = big.argmin(dim=1)
    normal = s1.n_out.gather(1, hand.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    elen2 = per_element_lengths(he2, pool.cfg.joint_names)
    term2 = b2["seed"].gather(1, (elen2.clamp(1, bucket) - 1).unsqueeze(1)).squeeze(1)
    kp, kd, _ki = sim.sample_gains(n, generator=gen)
    bc = cfg.bounce
    e_rest = bc.restitution[0] + (bc.restitution[1] - bc.restitution[0]) * torch.rand(
        n, device=device, dtype=dtype, generator=gen
    )
    randn = lambda *s: torch.randn(*s, device=device, dtype=dtype, generator=gen)
    _c1, c2 = bptt_fn(
        b1["init"], b1["state"], b1["q0"], b1["v0"], encode(model, b1["context"]),
        b1["masks"], s1, group_lengths(he1), k_first,
        term2, encode(model, b2["context"]), b2["masks"], s2, group_lengths(he2),
        elen2, w, kp, kd,
        hand, normal, e_rest,
        bc.q_noise * randn(n, N_DOF), bc.qd_noise * randn(n, N_DOF),
        bc.tip_vel_noise * randn(n, 3),
    )
    return c2


def _train_batch_rows(batch: dict) -> int:
    """Sample count along the training minibatch axis."""
    return int(batch["q0"].shape[0])


def _require_train_batch_size(batch: dict, bs: int, bucket: int) -> int:
    """Fail fast if a sampler returned fewer/more rows than ``batch_for``."""
    actual = _train_batch_rows(batch)
    expected = int(bs)
    if actual != expected:
        raise RuntimeError(
            f"train batch size mismatch for bucket {int(bucket)}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _cat_phase2_batches(parts: list[dict]) -> dict:
    """Concatenate ``Phase2Pool.batch`` dicts along the sample axis.

    Optional dataclass fields (``StrikeContext.waist_yaw_ref``, stick points,
    ...) can be a tensor on the fresh pool and ``None`` on the live post-hit
    replay.  Missing sides are zero-filled to that part's batch size so the
    cat does not see a ``None``.
    """
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("no batches to concatenate")
    if len(parts) == 1:
        return parts[0]

    def _align_field(values):
        tensors = [v for v in values if torch.is_tensor(v)]
        if not tensors:
            return values[0]
        like = tensors[0]
        aligned = []
        for p, v in zip(parts, values):
            if v is None:
                n = int(p["q0"].shape[0])
                aligned.append(like.new_zeros((n, *like.shape[1:])))
            else:
                aligned.append(v)
        return torch.cat(aligned, 0)

    out: dict = {}
    for k, v0 in parts[0].items():
        vs = [p[k] for p in parts]
        if k == "bucket" or not (torch.is_tensor(v0) or hasattr(v0, "__dataclass_fields__")):
            out[k] = v0
            continue
        if torch.is_tensor(v0):
            out[k] = torch.cat(vs, 0)
            continue
        kwargs = {}
        for f in v0.__dataclass_fields__:
            fv = [getattr(p, f) for p in vs]
            kwargs[f] = _align_field(fv)
        out[k] = type(v0)(**kwargs)
    return out


def _add_live_post_hit(buf: ReplayPool, bucket: int, cols: dict) -> int:
    """Scatter harvest rows marked ``_post_hit_fresh`` into the live buffer."""
    mark = cols.get("_post_hit_fresh")
    if mark is None or not bool(mark.any()):
        return 0
    sel = mark.nonzero(as_tuple=True)[0]
    ph = {
        k: v.index_select(0, sel) for k, v in cols.items()
        if not str(k).startswith("_")
    }
    if "kind" in ph:
        ph["kind"] = torch.full_like(ph["kind"], KIND_POST_HIT)
    buf.add(int(bucket), ph)
    return int(sel.numel())


#: Same gates as the collocation plot / eval_harness defaults.
_FRONTIER_HIT_M = 0.02
_FRONTIER_SDF_M = -0.005


def _torque_viol_max_from_cost(ct) -> torch.Tensor:
    """Per-row max over torque-family violation tensors."""
    parts = []
    for k, v in ct.violations.items():
        if k == "torque_peak" or k.startswith("torque_"):
            parts.append(v.detach().float())
    if not parts:
        v0 = next(iter(ct.violations.values()))
        return torch.zeros(int(v0.shape[0]), device=v0.device, dtype=torch.float32)
    return torch.stack(parts, dim=0).amax(dim=0)


def _proficiency_ok_from_cost(ct, *, viol_cap: float) -> torch.Tensor:
    """Hit + collision + optional torque cap (shared replay / frontier EMA)."""
    ok = (
        (ct.diagnostics["terminal_region_dist"] <= _FRONTIER_HIT_M)
        & (ct.diagnostics["collision_min_sdf"] >= _FRONTIER_SDF_M)
    )
    cap = float(viol_cap)
    if cap > 0.0:
        tv = ct.diagnostics.get("torque_excess_max")
        if tv is None:
            tv = _torque_viol_max_from_cost(ct)
        ok = ok & (tv <= cap)
    return ok


def _hard_feasibility_ok_from_cost(
    ct,
    *,
    terminal_eps: float = 1.0e-6,
    collision_clearance: float = 0.0,
    joint_eps: float = 1.0e-6,
    torque_eps: float = 1.0e-6,
) -> torch.Tensor:
    """Model-independent positive witness gate for the voxel feasibility set.

    This intentionally ignores every soft trajectory-quality objective and all
    AL multiplier values.  A pass establishes that this particular trajectory
    reaches the legal contact region while satisfying the selected hard
    collision, joint-position, and available-torque constraints.  Failure is
    only absence of a witness; it is not evidence that the state is infeasible.
    """
    names = (
        "terminal_region_dist",
        "collision_min_sdf",
        "joint_box_excess_max",
        "torque_excess_max",
    )
    missing = [name for name in names if name not in ct.diagnostics]
    if missing:
        raise KeyError(f"hard feasibility gate missing diagnostics: {missing}")
    terminal, collision, joint, torque = (
        ct.diagnostics[name].detach().float() for name in names
    )
    shape = terminal.shape
    if any(value.shape != shape for value in (collision, joint, torque)):
        raise ValueError("hard feasibility diagnostics must have matching shapes")
    finite = (
        torch.isfinite(terminal)
        & torch.isfinite(collision)
        & torch.isfinite(joint)
        & torch.isfinite(torque)
    )
    return (
        finite
        & (terminal <= float(terminal_eps))
        & (collision >= float(collision_clearance))
        & (joint <= float(joint_eps))
        & (torque <= float(torque_eps))
    )


_VOXEL_HARD_GATE_DEFINITION = {
    "terminal_region_dist_max": 1.0e-6,
    "collision_min_sdf_min": 0.0,
    "joint_box_excess_max": 1.0e-6,
    "torque_excess_max": 1.0e-6,
    "finite": True,
}


def _voxel_runtime_gate_definition(
    cost: FusedCost, source: Phase2Pool,
) -> dict:
    """Fingerprint every runtime fact that can invalidate positive evidence."""

    def tensor_digest(items) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(items, key=lambda item: item[0]):
            if not torch.is_tensor(value):
                continue
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(name).encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    geometry_names = {
        "drum_pos", "drum_axis", "drum_radius", "drum_height", "drum_valid",
        "stick_axis", "stick_tip", "stick_length", "stick_valid",
    }
    geometry = []
    for bucket, cols in source.data.items():
        geometry.extend(
            (f"{int(bucket)}:{name}", value)
            for name, value in cols.items() if name in geometry_names
        )
    dynamics = [
        (f"cost:{name}", value)
        for name, value in vars(cost).items()
        if name in {"q_lo_hard", "q_hi_hard", "tau_hard", "v_hard"}
    ]
    dynamics.extend(
        (f"rnea:{name}", value)
        for name, value in vars(cost.rnea).items() if torch.is_tensor(value)
    )
    cfg = cost.cfg
    def json_scalar(value):
        return float(value) if isinstance(value, (int, float, np.number)) else str(value)

    return {
        **_VOXEL_HARD_GATE_DEFINITION,
        "joint_names": list(cost.joint_names),
        "control_dt": float(cost.dt),
        "dynamics_limits_sha256": tensor_digest(dynamics),
        "source_geometry_sha256": tensor_digest(geometry),
        "collision": {
            "margin": float(cfg.collision_margin),
            "shaft_samples": int(cfg.shaft_samples),
            "stick_stick_min_sep": float(STICK_MIN_SEP),
            "shaft_hit_samples": int(SHAFT_HIT_SAMPLES),
        },
        "torque": {
            "physics": str(cfg.torque_physics),
            "derate": json_scalar(cfg.torque_derate),
            "saturation_scale": json_scalar(cfg.torque_saturation_scale),
            "track_enabled": bool(cfg.w_track_torque or cfg.w_track_torque_peak),
            "track_taps": int(cfg.track_torque_taps),
            "track_tol": float(cfg.track_torque_tol),
            "static_only": bool(cfg.torque_static_only),
        },
    }


@dataclass
class _VoxelBatchMeta:
    """Python-side provenance for one mixed voxel minibatch."""

    keys: list[VoxelKey | None]
    template_rows: list[int | None]
    hit_pairs: list[tuple[int, int] | None]
    is_probe: list[bool]


def _detach_batch(batch: dict, *, device=None) -> dict:
    """Clone a Phase2 batch without retaining a model/autograd graph."""
    out: dict = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            out[key] = tensor if device is None else tensor.to(device)
        elif hasattr(value, "__dataclass_fields__"):
            kwargs = {}
            for name in value.__dataclass_fields__:
                field_value = getattr(value, name)
                if torch.is_tensor(field_value):
                    field_value = field_value.detach().clone()
                    if device is not None:
                        field_value = field_value.to(device)
                kwargs[name] = field_value
            out[key] = type(value)(**kwargs)
        else:
            out[key] = value
    return out


class _VerifiedSuffixRing:
    """Small FIFO of Bellman-admitted parent trajectories.

    Storage is batched per horizon for cheap GPU sampling, while a monotonically
    increasing sequence column enforces one global capacity across horizons.
    Sampling is uniform inside the recent FIFO window; an additional recency
    weighting would unnecessarily collapse diversity in a 256-entry ring.
    """

    _FIELDS = ("q0", "v0", "context", "hit", "active", "masks")

    def __init__(self, *, capacity: int, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.data: dict[int, dict] = {}
        self.next_sequence = 0

    def __len__(self) -> int:
        return sum(self.size(bucket) for bucket in self.data)

    def size(self, bucket: int) -> int:
        item = self.data.get(int(bucket))
        return 0 if item is None else int(item["q0"].shape[0])

    def distribution(self) -> dict[int, int]:
        return {bucket: self.size(bucket) for bucket in sorted(self.data)}

    @torch.no_grad()
    def add(
        self,
        bucket: int,
        batch: dict,
        traj: torch.Tensor,
        *,
        hard_ok: torch.Tensor,
    ) -> int:
        """Append verified parents and evict the globally oldest rows."""
        bucket = int(bucket)
        mask = hard_ok.detach().to(device=traj.device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != int(traj.shape[0]):
            raise ValueError("hard_ok must have one value per suffix parent")
        idx = mask.nonzero(as_tuple=True)[0]
        if not int(idx.numel()):
            return 0
        parent = _index_batch(
            {key: batch[key] for key in self._FIELDS}, idx,
        )
        parent["parent_traj"] = traj.detach().index_select(0, idx)
        n = int(idx.numel())
        parent["suffix_sequence"] = torch.arange(
            self.next_sequence, self.next_sequence + n,
            device=traj.device, dtype=torch.long,
        )
        self.next_sequence += n
        parent["bucket"] = bucket
        parent = _detach_batch(parent, device=self.device)
        if bucket in self.data:
            parent = _cat_phase2_batches([self.data[bucket], parent])
        self.data[bucket] = parent
        self._trim()
        return n

    @torch.no_grad()
    def _trim(self) -> None:
        total = len(self)
        if total <= self.capacity:
            return
        sequences = torch.cat([
            item["suffix_sequence"] for item in self.data.values()
        ])
        keep_from = torch.topk(
            sequences, k=self.capacity, largest=True, sorted=False,
        ).values.min()
        for bucket in list(self.data):
            item = self.data[bucket]
            idx = (item["suffix_sequence"] >= keep_from).nonzero(as_tuple=True)[0]
            if int(idx.numel()):
                self.data[bucket] = _index_batch(item, idx)
            else:
                del self.data[bucket]

    @torch.no_grad()
    def sample(
        self,
        bucket: int,
        n: int,
        *,
        min_parents: int,
        generator: torch.Generator | None = None,
    ) -> tuple[dict | None, float]:
        """Return a fixed batch and its unique-parent warmup coefficient."""
        size = self.size(bucket)
        n = int(n)
        if size < int(min_parents):
            return None, 0.0
        if size >= n:
            idx = torch.randperm(size, device=self.device, generator=generator)[:n]
        else:
            # Reused parents draw independent offsets later in the caller.
            idx = torch.randint(size, (n,), device=self.device, generator=generator)
        fill = min(float(size) / float(max(n, 1)), 1.0)
        return _index_batch(self.data[int(bucket)], idx), fill

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "capacity": self.capacity,
            "next_sequence": int(self.next_sequence),
            "data": {
                int(bucket): _detach_batch(item, device="cpu")
                for bucket, item in self.data.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported verified suffix-ring checkpoint")
        self.next_sequence = int(state.get("next_sequence", 0))
        self.data = {
            int(bucket): _detach_batch(item, device=self.device)
            for bucket, item in state.get("data", {}).items()
        }
        self._trim()


def _suffix_distill_loss(
    replay_batch: dict,
    offs: torch.Tensor,
    *,
    bucket: int,
    encode_step,
    suffix_fn,
    pool: Phase2Pool,
    microbatch: int,
):
    """Build one shifted child batch and return its teacher-only tie loss."""
    active = replay_batch["active"]
    hit_parent = replay_batch["hit"]
    hit_child = torch.where(
        active.bool(), hit_parent - offs.unsqueeze(1), hit_parent,
    )
    masks_child = pool.masks_for(hit_child.clamp_min(0), active, int(bucket))
    enc = encode_step(replay_batch["context"])
    return _run_suffix_step_chunked(
        suffix_fn,
        replay_batch["parent_traj"], replay_batch["q0"], replay_batch["v0"],
        enc, replay_batch["masks"].elem_mask, masks_child, offs,
        microbatch=int(microbatch),
    )


class _VoxelMPCParentRing:
    """The latest shifted decoder-parent generation for each horizon.

    This intentionally is not a recency replay archive.  A refresh replaces a
    bucket's entire generation, so every sampled parent is a decode from the
    preceding optimizer visit to that bucket.  State and current-goal context
    are copied unchanged; only ``init``/``seed`` become the one-knot-shifted
    detached parent plan.
    """

    def __init__(self, *, device):
        self.device = torch.device(device)
        self.data: dict[int, dict] = {}
        self.generation: dict[int, int] = {}

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.data))

    def __len__(self) -> int:
        return sum(self.size(bucket) for bucket in self.data)

    def size(self, bucket: int) -> int:
        batch = self.data.get(int(bucket))
        return 0 if batch is None else int(batch["q0"].shape[0])

    def distribution(self) -> dict[int, int]:
        return {bucket: self.size(bucket) for bucket in self.buckets}

    @torch.no_grad()
    def refresh(
        self,
        bucket: int,
        batch: dict,
        traj: torch.Tensor,
        *,
        hard_ok: torch.Tensor,
    ) -> int:
        """Replace one bucket with shifted, caller-certified parent plans.

        A voxel's sticky feasibility certificate only says that *some* witness
        exists for that cell.  It does not make an arbitrary later policy
        decode executable.  The parent ring is an initialization trajectory,
        so its ingress contract is deliberately stronger than voxel ingress:
        voxel callers require all four hard gates.  The neural timing caller
        uses its configured exact close-feasibility range.
        """
        bucket = int(bucket)
        n = int(batch["q0"].shape[0])
        if traj.ndim != 3 or int(traj.shape[0]) != n or int(traj.shape[1]) != bucket:
            raise ValueError(
                f"parent trajectory shape {tuple(traj.shape)} does not match "
                f"batch ({n}, {bucket}, D)"
            )
        mask = hard_ok.to(device=traj.device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != n:
            raise ValueError("hard_ok must have one value per batch row")
        idx = mask.nonzero(as_tuple=True)[0]
        if not int(idx.numel()):
            # A per-step parent ring must never silently retain an older
            # generation when the current policy supplied no executable
            # parent.  Leave the stratum empty so the mixer redistributes it
            # to certified pool rows on the next visit.
            self.data.pop(bucket, None)
            self.generation[bucket] = self.generation.get(bucket, 0) + 1
            return 0
        parent = _index_batch(batch, idx)
        selected = traj.detach().index_select(0, idx)
        shifted = torch.cat([selected[:, 1:], selected[:, -1:]], dim=1)
        shifted = shifted * parent["masks"].elem_mask.to(shifted.dtype)
        parent["seed"] = shifted.clone()
        parent["init"] = shifted.clone()
        rows = int(idx.numel())
        parent["mpc_init"] = torch.ones(
            rows, device=shifted.device, dtype=torch.bool,
        )
        parent["al_ok"] = torch.ones(
            rows, device=shifted.device, dtype=torch.bool,
        )
        parent["voxel_kind"] = torch.zeros(
            rows, device=shifted.device, dtype=torch.long,
        )
        # Replay-only slot provenance must never leak into this independent
        # structure; callers use mpc_init as the per-row provenance bit.
        parent.pop("replay_loc", None)
        self.data[bucket] = _detach_batch(parent, device=self.device)
        self.generation[bucket] = self.generation.get(bucket, 0) + 1
        return rows

    def sample_idx(
        self, bucket: int, n: int, *, generator: torch.Generator | None = None,
        **_unused,
    ) -> torch.Tensor:
        size = self.size(bucket)
        if size <= 0:
            raise RuntimeError(f"MPC parent ring is empty for bucket {int(bucket)}")
        return torch.randint(
            size, (int(n),), device=self.device, generator=generator,
        )

    def batch(
        self, bucket: int, idx: torch.Tensor, *, generator=None, noise=False,
    ) -> dict:
        del generator, noise
        return _index_batch(self.data[int(bucket)], idx.to(self.device))

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "generation": dict(self.generation),
            "data": {
                int(bucket): _detach_batch(batch, device="cpu")
                for bucket, batch in self.data.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported voxel MPC parent-ring checkpoint")
        self.data = {
            int(bucket): _detach_batch(batch, device=self.device)
            for bucket, batch in state.get("data", {}).items()
        }
        self.generation = {
            int(bucket): int(value)
            for bucket, value in state.get("generation", {}).items()
        }


def _shift_time_rows(value: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Left-shift a ``(B, H, ...)`` tensor independently for each row."""
    if value.ndim < 2 or int(value.shape[0]) != int(offs.shape[0]):
        raise ValueError("time tensor and offsets must share their batch dimension")
    horizon = int(value.shape[1])
    ar = torch.arange(horizon, device=value.device)
    index = (ar.unsqueeze(0) + offs.to(value.device).unsqueeze(1)).clamp_max(
        horizon - 1
    )
    expand = (int(value.shape[0]), horizon, *value.shape[2:])
    index = index.reshape(int(value.shape[0]), horizon, *([1] * (value.ndim - 2)))
    return value.gather(1, index.expand(expand))


class _TrajectorySuffixMPCParentRing:
    """Cheap timing-MPC collocations sampled from certified trajectory tails.

    Each certified full parent trajectory is stored once. Every draw samples a
    remaining-TTH component, selects a parent capable of supporting it, builds
    the detached spline state there, and returns the corresponding noisy suffix
    problem. Reusing one parent at many offsets supplies many later-in-flight
    collocations without storing every suffix or running a PD simulation.
    """

    def __init__(
        self,
        *,
        device,
        pool,
        costs: dict[int, FusedCost],
        q_noise: float = 0.01,
        qd_noise: float = 0.05,
        capacity: int = 2048,
        terminal_peak_frac: float = 0.55,
        terminal_long_frac: float = 0.42,
        terminal_low_frac: float = 0.03,
        terminal_peak_wp: int = 5,
        terminal_tail_tau: float = 8.0,
    ):
        self.device = torch.device(device)
        self.data: dict[int, dict] = {}
        self.generation: dict[int, int] = {}
        self.pool = pool
        self.costs = costs
        self.q_noise = float(q_noise)
        self.qd_noise = float(qd_noise)
        self.capacity = int(capacity)
        self.terminal_peak_frac = float(terminal_peak_frac)
        self.terminal_long_frac = float(terminal_long_frac)
        self.terminal_low_frac = float(terminal_low_frac)
        self.terminal_peak_wp = int(terminal_peak_wp)
        self.terminal_tail_tau = float(terminal_tail_tau)
        self.last_advance_offset_mean = 0.0
        self.last_advance_seconds_mean = 0.0
        self.last_unique_parent_frac = 0.0
        self.dropped_legacy_rows = 0

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.data))

    def __len__(self) -> int:
        return sum(self.size(bucket) for bucket in self.data)

    def size(self, bucket: int) -> int:
        batch = self.data.get(int(bucket))
        return 0 if batch is None else int(batch["q0"].shape[0])

    def distribution(self) -> dict[int, int]:
        return {bucket: self.size(bucket) for bucket in self.buckets}

    def sample_idx(
        self,
        bucket: int,
        n: int,
        *,
        generator: torch.Generator | None = None,
        **_unused,
    ) -> torch.Tensor:
        size = self.size(bucket)
        if size <= 0:
            raise RuntimeError(
                f"trajectory-suffix MPC ring is empty for bucket {int(bucket)}"
            )
        return torch.randint(
            size, (int(n),), device=self.device, generator=generator,
        )

    @torch.no_grad()
    def batch(
        self, bucket: int, idx: torch.Tensor, *, generator=None, noise=False,
    ) -> dict:
        del noise
        bucket = int(bucket)
        idx = idx.to(self.device).clone()
        rows = int(idx.numel())
        component = torch.multinomial(
            torch.tensor(
                [self.terminal_peak_frac, self.terminal_long_frac,
                 self.terminal_low_frac],
                device=self.device, dtype=torch.float32,
            ),
            rows, replacement=True, generator=generator,
        )
        stored = self.data[bucket]
        stored_active = stored["active"].bool()
        stored_nearest = torch.where(
            stored_active, stored["hit"],
            torch.full_like(stored["hit"], bucket),
        ).min(dim=1).values
        requirements = ((0, self.terminal_peak_wp + 1), (1, bucket))
        for component_id, nearest_required in requirements:
            positions = (component == component_id).nonzero(as_tuple=True)[0]
            candidates = (stored_nearest >= nearest_required).nonzero(as_tuple=True)[0]
            if not int(positions.numel()) or not int(candidates.numel()):
                continue
            draw = torch.randint(
                int(candidates.numel()), (int(positions.numel()),),
                device=self.device, generator=generator,
            )
            idx.index_copy_(0, positions, candidates.index_select(0, draw))
        parent = _index_batch(stored, idx)
        selected = parent.pop("_parent_traj")
        rows, _, dof = selected.shape
        offs = terminal_mixture_suffix_offsets(
            parent["hit"], parent["active"], bucket,
            generator=generator,
            peak_frac=self.terminal_peak_frac,
            long_frac=self.terminal_long_frac,
            low_frac=self.terminal_low_frac,
            peak_wp=self.terminal_peak_wp,
            tail_tau=self.terminal_tail_tau,
            component=component,
        )
        if bool((offs < 1).any()):
            raise RuntimeError("timing MPC suffix failed to advance")

        # Evaluate the parent's continuous spline at the chosen knot. Decoder
        # rows represent knots 1..H, so offset k hands off at row k-1.
        cost = self.costs[bucket]
        with torch.autocast(device_type=selected.device.type, enabled=False):
            parent32 = selected.float()
            elem32 = parent["masks"].elem_mask.float()
            spline = cost.spline.from_masks(
                parent["q0"].float(), parent32, parent["v0"].float(), elem32,
            )
            end = torch.full(
                (1,), float(cost.dt), device=selected.device, dtype=torch.float32,
            )
            _q, qd, _qdd, _jerk = spline.eval_offsets(end)
            velocity = qd[:, :, 0, :]
            km1 = (offs - 1).view(rows, 1, 1).expand(rows, 1, dof)
            q0 = parent32.gather(1, km1)[:, 0]
            v0 = velocity.gather(1, km1)[:, 0]

        if self.q_noise:
            q0 = q0 + self.q_noise * torch.randn(
                q0.shape, device=q0.device, dtype=q0.dtype, generator=generator,
            )
        if self.qd_noise:
            v0 = v0 + self.qd_noise * torch.randn(
                v0.shape, device=v0.device, dtype=v0.dtype, generator=generator,
            )
        q_lo = self.pool.q_lo.to(device=q0.device, dtype=q0.dtype)
        q_hi = self.pool.q_hi.to(device=q0.device, dtype=q0.dtype)
        q0 = q0.clamp(q_lo, q_hi).to(parent["q0"].dtype)
        v0 = v0.to(parent["v0"].dtype)
        parent["q0"] = q0
        parent["v0"] = v0
        parent["state"] = torch.cat([q0, v0], dim=-1)

        active = parent["active"].to(torch.bool)
        child_hit = torch.where(
            active,
            parent["hit"] - offs.unsqueeze(1),
            torch.full_like(parent["hit"], -1),
        )
        if bool((active & (child_hit < 1)).any()):
            raise RuntimeError("timing MPC suffix moved beyond an active strike")
        parent["hit"] = child_hit
        parent["masks"] = self.pool.masks_for(child_hit, active, bucket)
        goal_tth = torch.where(
            active,
            child_hit.to(q0.dtype) * float(self.pool.cfg.dt),
            torch.zeros_like(child_hit, dtype=q0.dtype),
        )
        context_updates = {"goal_tth": goal_tth}
        if hasattr(parent["context"], "yaw_now"):
            yaw_now = getattr(parent["context"], "yaw_now")
            if yaw_now is not None:
                yaw_i = tuple(self.pool.cfg.joint_names).index("waist_yaw_joint")
                context_updates["yaw_now"] = q0[:, yaw_i]
        parent["context"] = replace(parent["context"], **context_updates)

        strike = parent["strike"]
        strike_updates = {
            "hit_index": effective_hits(child_hit, active, bucket),
        }
        for name in ("waist_yaw_ref", "waist_pr_ref"):
            ref = getattr(strike, name, None)
            if ref is not None:
                strike_updates[name] = _shift_time_rows(ref, offs)
        parent["strike"] = replace(strike, **strike_updates)

        shifted, _valid = shift_plan(selected, offs)
        shifted = shifted * parent["masks"].elem_mask.to(shifted.dtype)
        parent["seed"] = shifted.clone()
        parent["init"] = shifted.clone()
        parent["mpc_init"] = torch.ones(rows, device=shifted.device, dtype=torch.bool)
        parent["al_ok"] = torch.ones(rows, device=shifted.device, dtype=torch.bool)
        parent["voxel_kind"] = torch.zeros(rows, device=shifted.device, dtype=torch.long)
        parent.pop("replay_loc", None)
        self.last_advance_offset_mean = float(offs.float().mean().item())
        self.last_advance_seconds_mean = (
            self.last_advance_offset_mean * float(self.pool.cfg.dt)
        )
        self.last_unique_parent_frac = float(idx.unique().numel()) / float(max(rows, 1))
        return parent

    @torch.no_grad()
    def refresh(
        self,
        bucket: int,
        batch: dict,
        traj: torch.Tensor,
        *,
        hard_ok: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> int:
        del generator
        bucket = int(bucket)
        n = int(batch["q0"].shape[0])
        if traj.ndim != 3 or int(traj.shape[0]) != n or int(traj.shape[1]) != bucket:
            raise ValueError(
                f"parent trajectory shape {tuple(traj.shape)} does not match "
                f"batch ({n}, {bucket}, D)"
            )
        mask = hard_ok.to(device=traj.device, dtype=torch.bool).reshape(-1)
        if int(mask.numel()) != n:
            raise ValueError("hard_ok must have one value per batch row")

        # A timing child must actually advance.  Rows whose nearest active hit
        # is already at knot 1 have no valid pre-hit suffix and are not stored.
        active = batch["active"].to(device=traj.device, dtype=torch.bool)
        hit = batch["hit"].to(device=traj.device, dtype=torch.long)
        span = torch.where(active, hit, torch.full_like(hit, bucket))
        advanceable = span.min(dim=1).values > 1
        idx = (mask & advanceable).nonzero(as_tuple=True)[0]
        if not int(idx.numel()):
            # Retain previously certified full parents. A quiet ingress step
            # must not erase the reuse window the pool exists to provide.
            self.generation[bucket] = self.generation.get(bucket, 0) + 1
            return 0

        parent = _index_batch(batch, idx)
        selected = traj.detach().index_select(0, idx)
        rows = int(selected.shape[0])
        # Keep the full certified parent exactly once. ``batch`` materializes a
        # fresh random suffix from it on every draw, including duplicate draws
        # of the same parent within one actor minibatch.
        parent["_parent_traj"] = selected.clone()
        # The original initializer is dead once the actor has produced this
        # certified parent. Avoid retaining two more HxD tensors; ``batch``
        # reconstructs both seed/init from the sampled parent suffix.
        parent.pop("seed", None)
        parent.pop("init", None)
        parent["mpc_init"] = torch.ones(rows, device=selected.device, dtype=torch.bool)
        parent["al_ok"] = torch.ones(rows, device=selected.device, dtype=torch.bool)
        parent["voxel_kind"] = torch.zeros(rows, device=selected.device, dtype=torch.long)
        parent.pop("replay_loc", None)
        parent = _detach_batch(parent, device=self.device)
        previous = self.data.get(bucket)
        merged = parent if previous is None else _cat_phase2_batches([previous, parent])
        total = int(merged["q0"].shape[0])
        if total > self.capacity:
            keep = torch.arange(
                total - self.capacity, total, device=self.device,
            )
            merged = _index_batch(merged, keep)
        self.data[bucket] = merged
        self.generation[bucket] = self.generation.get(bucket, 0) + 1
        return rows

    def state_dict(self) -> dict:
        return {
            "version": 3,
            "mode": "trajectory_parent_pool",
            "generation": dict(self.generation),
            "data": {
                int(bucket): _detach_batch(batch, device="cpu")
                for bucket, batch in self.data.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        version = int(state.get("version", -1))
        self.generation = {
            int(bucket): int(value)
            for bucket, value in state.get("generation", {}).items()
        }
        if version in (1, 2):
            # V1 retained stale state/TTH; V2 stored only one already-selected
            # suffix. Neither contains the certified full parent trajectory
            # needed for sample-time offset selection. Refill from live actor
            # outputs on the next visit to each bucket.
            self.dropped_legacy_rows = sum(
                int(batch["q0"].shape[0])
                for batch in state.get("data", {}).values()
            )
            self.data = {}
            return
        if version != 3 or state.get("mode") != "trajectory_parent_pool":
            raise ValueError("unsupported trajectory-parent MPC ring checkpoint")
        self.data = {
            int(bucket): _detach_batch(batch, device=self.device)
            for bucket, batch in state.get("data", {}).items()
        }


def _voxel_current_goal_only(batch: dict) -> dict:
    """Remove future and torso-reference channels from a voxel-mode batch.

    This is applied to *every* stratum, including MPC ingress copied from the
    legacy replay container.  Consequently an old pool/replay/checkpoint cannot
    smuggle future-hit tokens or learned/EMA waist references into the new
    training path.  ``num_future`` remains one for checkpoint shape
    compatibility, but its validity mask is structurally false.
    """
    ctx = batch["context"]
    active = batch.get("active", ctx.goal_valid).to(torch.bool)
    idle_valid = ~active
    batch["context"] = replace(
        ctx,
        goal_valid=active,
        goal_pos=ctx.goal_pos * active.unsqueeze(-1).to(ctx.goal_pos.dtype),
        goal_vel=ctx.goal_vel * active.unsqueeze(-1).to(ctx.goal_vel.dtype),
        goal_tth=ctx.goal_tth * active.to(ctx.goal_tth.dtype),
        future_drum=torch.zeros_like(ctx.future_drum),
        future_pos=torch.zeros_like(ctx.future_pos),
        future_vel=torch.zeros_like(ctx.future_vel),
        future_gap=torch.zeros_like(ctx.future_gap),
        future_valid=torch.zeros_like(ctx.future_valid, dtype=torch.bool),
        idle_cyl=ctx.idle_cyl * idle_valid.unsqueeze(-1).to(ctx.idle_cyl.dtype),
        idle_valid=idle_valid,
        aim_yaw=torch.zeros_like(ctx.aim_yaw),
        aim_valid=torch.zeros_like(ctx.aim_valid, dtype=torch.bool),
    )
    strike = batch["strike"]
    batch["strike"] = replace(
        strike, waist_yaw_ref=None, waist_pr_ref=None,
    )
    # Retain the Phase2Pool-shaped key for downstream ABI compatibility, but
    # erase the legacy rest/post-hit taxonomy.  Voxel mode's only task category
    # is the 64-way chart encoded by active hand and current drum.
    if "kind" in batch:
        batch["kind"] = torch.zeros_like(batch["kind"])
    return batch


def _voxel_batch_from_columns(
    source: Phase2Pool, bucket: int, cols: dict[str, torch.Tensor],
) -> dict:
    """Convert raw or sliced pool columns to the train-batch ABI.

    Unlike :meth:`Phase2Pool.batch`, this accepts the output of
    :func:`ordinary_slice_batches`; no source row number exists after a suffix
    changes buckets.  Waist references are deliberately omitted because the
    voxel contract gives torso motion to the decoder.
    """
    horizon = int(bucket)
    n = int(cols["q0"].shape[0])
    active = cols["active"].to(torch.bool)
    hit = torch.where(
        active, cols["hit"].to(torch.long),
        torch.full_like(cols["hit"].to(torch.long), -1),
    )
    masks = source.masks_for(hit, active, horizon)
    context = ContextInputs(**{
        name: cols[name] for name in CONTEXT_FIELDS if name in cols
    })
    strike_cols = cols
    if "start_drum" not in strike_cols:
        strike_cols = dict(cols)
        strike_cols["start_drum"] = strike_cols["drum"]
    strike = strike_context_from_arrays(
        strike_cols,
        torch.arange(n, device=cols["q0"].device),
        drum_head_centers=context.drum_pos - source._pelvis,
        drum_radii=context.drum_radius,
        waist_ref=None,
        waist_pr_ref=None,
        n_shaft=source.n_shaft,
    )
    seed = cols["init"]
    qd0 = cols["qd0"]
    kind = cols.get(
        "kind", torch.zeros(n, device=cols["q0"].device, dtype=torch.long),
    )
    batch = _voxel_current_goal_only({
        "bucket": horizon,
        "q0": cols["q0"],
        "qd0": qd0,
        "v0": qd0,
        "state": torch.cat([cols["q0"], qd0], dim=-1),
        "init": seed * masks.elem_mask,
        "seed": seed,
        "masks": masks,
        "context": context,
        "strike": strike,
        "hit": hit,
        "active": active,
        "kind": kind,
    })
    batch["context"] = replace(
        batch["context"],
        goal_tth=torch.where(
            active, hit.clamp_min(0).to(source.dtype) * float(source.cfg.dt),
            torch.zeros_like(hit, dtype=source.dtype),
        ),
    )
    batch["strike"] = replace(
        batch["strike"], hit_index=effective_hits(hit, active, horizon),
    )
    return batch


def _voxel_terminals(batch: dict, joint_names) -> torch.Tensor:
    """Per-element task terminal encoded by a Phase2 warm start."""
    horizon = int(batch["bucket"])
    hit_eff = effective_hits(batch["hit"], batch["active"], horizon)
    lengths = per_element_lengths(hit_eff, joint_names).clamp(1, horizon)
    return batch["seed"].gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)


class _VoxelTrainingState:
    """Trainer adapter around the compact positive-witness data structure.

    The static :class:`Phase2Pool` remains the context/template store.  The
    voxel structure only owns feasibility evidence; this adapter remembers
    which static rows supplied each certified paired timing witness.
    """

    def __init__(
        self, pool: Phase2Pool, cfg: Phase2Config, *, gate_definition=None,
    ):
        self.source = pool
        self.cfg = cfg
        self.gate_definition = dict(
            _VOXEL_HARD_GATE_DEFINITION
            if gate_definition is None else gate_definition
        )
        self.pool = VoxelPool(q_cell_rad=float(cfg.voxel_q_cell_rad))
        self.probes = ProbeBuffer(capacity=int(cfg.voxel_probe_capacity))
        self.templates: dict[VoxelKey, list[tuple[int, tuple[int, int]]]] = {}
        self.probe_templates: dict[
            ProbeCandidateKey, tuple[int, tuple[int, int]]
        ] = {}
        self.probe_finishing_hands: dict[ProbeCandidateKey, int] = {}

    def _key(self, bucket: int, q0, qd0, chart: int) -> VoxelKey:
        return VoxelKey.from_state(
            chart_id=int(chart), bucket=int(bucket),
            q0=q0, qd0=qd0, q_cell_rad=float(self.cfg.voxel_q_cell_rad),
        )

    @torch.no_grad()
    def _bootstrap_post_hit(
        self, costs: dict[int, FusedCost], weights: dict, *, model,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Mine bounce states whose finishing hand is reassigned to its next hit."""
        anchor = int(round(
            float(self.cfg.voxel_posthit_anchor_tth_s) / float(self.source.cfg.dt)
        ))
        destinations = tuple(sorted(int(b) for b in costs if int(b) >= anchor))
        if not destinations:
            return {"posthit_checked": 0, "posthit_witness_rows": 0}
        torch_gen = torch.Generator(device=self.source.device)
        torch_gen.manual_seed(int(self.cfg.seed) + 21_119)
        cap = int(self.cfg.voxel_bootstrap_rows)
        used = checked = passed = 0
        for source_bucket in self.source.buckets:
            remaining = (
                self.source.size(source_bucket) if cap == 0 else max(cap - used, 0)
            )
            if remaining <= 0:
                break
            n = min(self.source.size(source_bucket), remaining)
            idx = torch.linspace(
                0, self.source.size(source_bucket) - 1, n,
                device=self.source.device, dtype=torch.float64,
            ).round().to(torch.long).unique()
            used += int(idx.numel())
            source_batch = self.source.batch(source_bucket, idx, noise=False)
            ctx = source_batch["context"]
            if ctx.future_valid.shape[-1] < 1:
                continue
            can_finish = (
                source_batch["active"].to(torch.bool)
                & ctx.future_valid[..., 0].to(torch.bool)
            )
            score = torch.where(
                can_finish, source_batch["hit"],
                torch.full_like(source_batch["hit"], int(source_bucket) + 1),
            )
            finisher = score.argmin(dim=1)
            keep = score.min(dim=1).values <= int(source_bucket)
            rows = keep.nonzero(as_tuple=True)[0]
            if not int(rows.numel()):
                continue
            finisher = finisher.index_select(0, rows)
            hit_now = source_batch["hit"].index_select(0, rows)
            active_now = source_batch["active"].index_select(0, rows).to(torch.bool)
            contact_k = hit_now.gather(1, finisher[:, None]).squeeze(1)
            other = 1 - finisher
            other_remaining = (
                hit_now.gather(1, other[:, None]).squeeze(1) - contact_k
            )
            keep_other = (
                active_now.gather(1, other[:, None]).squeeze(1)
                & (other_remaining >= 1)
            )
            required = torch.maximum(
                torch.full_like(other_remaining, anchor),
                torch.where(keep_other, other_remaining, torch.zeros_like(other_remaining)),
            )
            q_full = torch.cat([
                source_batch["q0"].unsqueeze(1), source_batch["seed"],
            ], dim=1).index_select(0, rows)
            gather = contact_k[:, None, None].expand(-1, 1, q_full.shape[-1])
            q_contact = q_full.gather(1, gather).squeeze(1)
            prev = (contact_k - 1).clamp_min(0)[:, None, None].expand_as(gather)
            q_prev = q_full.gather(1, prev).squeeze(1)
            qd_in = (q_contact - q_prev) / float(self.source.cfg.dt)
            current_drum = ctx.goal_drum.index_select(0, rows).gather(
                1, finisher[:, None],
            ).squeeze(1)
            normal = ctx.drum_axis.index_select(0, rows).gather(
                1, current_drum[:, None, None].expand(-1, 1, 3),
            ).squeeze(1)
            q_bounce, qd_bounce = apply_bounce(
                q_contact, qd_in, fk=costs[max(costs)].fk,
                rnea=costs[max(costs)].rnea, hand_index=finisher,
                normal=normal, cfg=self.cfg.bounce, generator=torch_gen,
            )
            future_drum = ctx.future_drum.index_select(0, rows)[..., 0]
            current_goal = ctx.goal_drum.index_select(0, rows)
            modes = torch.zeros_like(current_goal)
            modes.scatter_(1, finisher[:, None], (
                future_drum.gather(1, finisher[:, None]) + 1
            ))
            modes.scatter_(1, other[:, None], torch.where(
                keep_other[:, None],
                current_goal.gather(1, other[:, None]) + 1,
                torch.zeros_like(other[:, None]),
            ))
            charts = modes[:, 0] * 8 + modes[:, 1]
            hit_pairs = torch.full_like(hit_now, -1)
            hit_pairs.scatter_(
                1, finisher[:, None], torch.full_like(finisher[:, None], anchor),
            )
            hit_pairs.scatter_(
                1, other[:, None],
                torch.where(
                    keep_other[:, None], other_remaining[:, None],
                    torch.full_like(other[:, None], -1),
                ),
            )
            for bucket in destinations:
                select = (required <= bucket)
                # Assign each proposal to the shortest valid horizon only.
                shorter = [b for b in destinations if b < bucket]
                if shorter:
                    select &= required > max(shorter)
                loc = select.nonzero(as_tuple=True)[0]
                if not int(loc.numel()):
                    continue
                loc = torch.as_tensor([
                    int(i) for i in loc.tolist()
                    if self.pool.has_template(int(charts[i]), bucket)
                ], device=self.source.device, dtype=torch.long)
                if not int(loc.numel()):
                    continue
                keys = [
                    self._key(
                        bucket, q_bounce[i].detach().cpu().numpy(),
                        qd_bounce[i].detach().cpu().numpy(), int(charts[i]),
                    )
                    for i in loc.tolist()
                ]
                pairs = [tuple(int(x) for x in hit_pairs[i].tolist()) for i in loc.tolist()]
                batch, _ = self.pool.materialize_batch(
                    keys, rng=rng, device=self.source.device, dtype=self.source.dtype,
                    joint_names=self.source.cfg.joint_names,
                    dt=float(self.source.cfg.dt),
                    hit_pairs=pairs, q0_values=q_bounce.index_select(0, loc),
                    qd0_values=qd_bounce.index_select(0, loc), probe=True,
                    strike_speed_min=float(self.cfg.voxel_strike_speed_min),
                    strike_speed_max=float(self.cfg.voxel_strike_speed_max),
                )
                batch["seed"] = correct_quintic_init_arms(
                    costs[bucket].fk, batch["seed"], batch["q0"], batch["hit"],
                    batch["context"].goal_pos, active=batch["active"], steps=3,
                    q_bounds=(self.source.q_lo, self.source.q_hi),
                    joint_names=self.source.cfg.joint_names, qd0=batch["v0"],
                    dt=float(self.source.cfg.dt),
                )
                batch["init"] = batch["seed"] * batch["masks"].elem_mask
                hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
                strike = replace(batch["strike"], hit_index=hit_eff)
                trajectories = [("seed", batch["seed"])]
                if model is not None:
                    trajectories.append(("model", model(
                        batch["init"], batch["state"],
                        encode(model, batch["context"]), batch["masks"],
                    ).traj))
                valid = []
                payload = []
                for name, traj in trajectories:
                    ct = costs[bucket].forward(
                        batch["q0"], traj, batch["v0"], strike,
                        lengths=group_lengths(hit_eff), weights=weights,
                    )
                    payload.append((name, ct))
                    valid.append(_hard_feasibility_ok_from_cost(ct))
                ok = torch.stack(valid).any(dim=0)
                for row in ok.nonzero(as_tuple=True)[0].tolist():
                    winner = next(j for j, flag in enumerate(valid) if bool(flag[row]))
                    ct = payload[winner][1]
                    hard = torch.stack([
                        ct.diagnostics["terminal_region_dist"][row],
                        -ct.diagnostics["collision_min_sdf"][row],
                        ct.diagnostics["joint_box_excess_max"][row],
                        ct.diagnostics["torque_excess_max"][row],
                    ]).float()
                    self.pool.ingest(
                        keys[row], hit_pair=pairs[row],
                        feas_margin=-float(hard.max()), source="bootstrap",
                        ingress_step=-1, finishing_hand=int(finisher[loc[row]]),
                    )
                    passed += 1
                checked += len(keys)
        return {"posthit_checked": checked, "posthit_witness_rows": passed}

    @torch.no_grad()
    def _bootstrap_rest(
        self, costs: dict[int, FusedCost], weights: dict, *, model,
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Certify chart-0 hold poses mined from source rows with long gaps."""
        idle_by_hand: list[torch.Tensor | None] = [None, None]
        for cols in self.source.data.values():
            if "idle_cyl" not in cols or "idle_valid" not in cols:
                continue
            for hand in range(2):
                rows = cols["idle_valid"][:, hand].to(torch.bool).nonzero(as_tuple=True)[0]
                if idle_by_hand[hand] is None and int(rows.numel()):
                    idle_by_hand[hand] = cols["idle_cyl"][int(rows[0]), hand].clone()
        if any(value is None for value in idle_by_hand):
            return {"rest_checked": 0, "rest_witness_rows": 0}
        idle_cyl = torch.stack([idle_by_hand[0], idle_by_hand[1]], dim=0).unsqueeze(0)
        checked = passed = 0
        yaw_i = tuple(self.source.cfg.joint_names).index("waist_yaw_joint")
        for bucket in sorted(costs):
            if any(key.chart_id == 0 for key in self.pool.keys(bucket=bucket)):
                # Never replace the task template under an existing positive
                # certificate.  Bootstrap evidence is monotone in both state
                # membership and the task it certifies.
                continue
            if bucket not in self.source.data:
                continue
            cols = self.source.data[bucket]
            if "future_gap" not in cols or "future_valid" not in cols:
                continue
            long_gap = (
                cols["future_valid"].to(torch.bool)
                & (cols["future_gap"] >= float(self.cfg.voxel_rest_gap_s))
            ).reshape(cols["q0"].shape[0], -1).any(dim=1)
            rows = long_gap.nonzero(as_tuple=True)[0]
            if not int(rows.numel()):
                continue
            # Price several stationary candidates together, then retain the
            # first genuinely executable one as chart 0's canonical template.
            # This avoids making startup depend on the first long-gap row's
            # randomized kit happening to be collision-free.
            rows = rows[:max(1, int(self.cfg.voxel_bootstrap_batch))]
            base = self.source.batch(bucket, rows, noise=False)
            q0 = base["q0"].clone()
            qd0 = torch.zeros_like(q0)
            n = int(q0.shape[0])
            hit = torch.full((n, 2), -1, device=q0.device, dtype=torch.long)
            active = torch.zeros((n, 2), device=q0.device, dtype=torch.bool)
            masks = self.source.masks_for(hit, active, bucket)
            ctx = base["context"]
            ctx = replace(
                ctx,
                goal_drum=torch.zeros_like(ctx.goal_drum),
                goal_pos=torch.zeros_like(ctx.goal_pos),
                goal_vel=torch.zeros_like(ctx.goal_vel),
                goal_tth=torch.zeros_like(ctx.goal_tth),
                goal_valid=active,
                future_drum=torch.zeros_like(ctx.future_drum),
                future_pos=torch.zeros_like(ctx.future_pos),
                future_vel=torch.zeros_like(ctx.future_vel),
                future_gap=torch.zeros_like(ctx.future_gap),
                future_valid=torch.zeros_like(ctx.future_valid, dtype=torch.bool),
                idle_cyl=idle_cyl.to(device=q0.device, dtype=q0.dtype).expand(n, -1, -1),
                idle_valid=torch.ones_like(active),
                yaw_now=q0[:, yaw_i],
                aim_yaw=q0[:, yaw_i],
                aim_valid=torch.ones(n, device=q0.device, dtype=torch.bool),
            )
            strike = replace(
                base["strike"], hit_index=torch.full_like(hit, bucket),
                active=active.to(q0.dtype), target_drum=torch.full_like(hit, -1),
                start_drum=torch.full_like(hit, -1),
                waist_yaw_ref=None, waist_pr_ref=None,
            )
            seed = q0[:, None, :].expand(-1, bucket, -1).clone()
            rest_batch = {
                "bucket": bucket, "q0": q0, "qd0": qd0, "v0": qd0,
                "state": torch.cat([q0, qd0], dim=-1), "init": seed,
                "seed": seed, "masks": masks, "context": ctx,
                "strike": strike, "hit": hit, "active": active,
                "kind": torch.zeros(n, device=q0.device, dtype=torch.long),
            }
            hit_eff = effective_hits(hit, active, bucket)
            ct = costs[bucket].forward(
                q0, seed, qd0, strike,
                lengths=group_lengths(hit_eff), weights=weights,
            )
            valid = _hard_feasibility_ok_from_cost(ct)
            checked += n
            good = valid.nonzero(as_tuple=True)[0]
            if int(good.numel()):
                winner = int(good[0])
                template = ChartTemplate.from_batch(
                    rest_batch, winner, chart_id=0, terminal=q0[winner],
                )
                self.pool.register_template(template, replace=True)
                key = self._key(
                    bucket, q0[winner].cpu().numpy(),
                    qd0[winner].cpu().numpy(), 0,
                )
                hard = torch.stack([
                    ct.diagnostics["terminal_region_dist"][winner],
                    -ct.diagnostics["collision_min_sdf"][winner],
                    ct.diagnostics["joint_box_excess_max"][winner],
                    ct.diagnostics["torque_excess_max"][winner],
                ]).float()
                self.pool.ingest(
                    key, hit_pair=(-1, -1), feas_margin=-float(hard.max()),
                    source="bootstrap", ingress_step=-1,
                )
                passed += 1
        return {"rest_checked": checked, "rest_witness_rows": passed}

    def state_dict(self) -> dict:
        return {
            "version": 2,
            "pool": self.pool.state_dict(
                gate_definition=self.gate_definition,
            ),
            "probes": self.probes.state_dict(
                gate_definition=self.gate_definition,
            ),
            # Template row numbers refer to the immutable Phase2Pool supplied
            # to this run and are consequently stable across resume.
            "templates": dict(self.templates),
            "probe_templates": dict(self.probe_templates),
            "probe_finishing_hands": dict(self.probe_finishing_hands),
        }

    def load_state_dict(self, state: dict) -> None:
        version = int(state.get("version", -1))
        if version not in (1, 2):
            raise ValueError("unsupported voxel trainer checkpoint")
        self.pool = VoxelPool.load_state_dict(
            state["pool"], expected_gate_definition=self.gate_definition,
        )
        raw_probe_templates = dict(state.get("probe_templates", {}))
        raw_finishing_hands = dict(state.get("probe_finishing_hands", {}))
        if version == 1:
            # Version 1 deduplicated probes by VoxelKey and kept the proposed
            # complete timing pair in a side table.  Rebuild the new atomic
            # (voxel, pair) identity without discarding unknown attempts.
            migrated = dict(state["probes"])
            migrated["format_version"] = 2
            migrated_records = []
            for item in state["probes"].get("records", []):
                ks = item["key"]
                old_key = VoxelKey(
                    ks["chart_id"], ks["bucket"],
                    tuple(ks["q0_cell"]), tuple(ks["qd_bin"]),
                )
                template = raw_probe_templates.get(old_key)
                if template is None:
                    raise ValueError(
                        "legacy voxel probe has no complete timing-pair metadata"
                    )
                migrated_records.append({
                    **item, "hit_pair": list(template[1]),
                })
            migrated["records"] = migrated_records
            probe_state = migrated
        else:
            probe_state = state["probes"]
        # Runtime configuration owns the retry working-set size.  Otherwise a
        # continuation from an older 8192-entry checkpoint silently defeats a
        # newly configured short rotation.  ProbeBuffer loading naturally
        # retains the newest records when this lowers capacity.
        probe_state = {
            **probe_state,
            "capacity": int(self.cfg.voxel_probe_capacity),
        }
        self.probes = ProbeBuffer.load_state_dict(
            probe_state,
            expected_gate_definition=self.gate_definition,
        )
        self.templates = dict(state.get("templates", {}))
        if version == 1:
            self.probe_templates = {}
            self.probe_finishing_hands = {}
            for record in self.probes.records():
                old_key = record.key
                self.probe_templates[record.candidate] = raw_probe_templates[old_key]
                self.probe_finishing_hands[record.candidate] = int(
                    raw_finishing_hands.get(old_key, -1)
                )
        else:
            self.probe_templates = raw_probe_templates
            self.probe_finishing_hands = {
                candidate: int(hand)
                for candidate, hand in raw_finishing_hands.items()
                if candidate in self.probes
            }
            self.probe_templates = {
                candidate: template
                for candidate, template in self.probe_templates.items()
                if candidate in self.probes
            }

    @torch.no_grad()
    def bootstrap(
        self, costs: dict[int, FusedCost], weights: dict, *, model=None,
    ) -> dict:
        """Mine collocations and certify them against canonical chart tasks.

        Source trajectories are used to propose states, including suffix
        states every ``voxel_bootstrap_stride`` knots.  They are *not* trusted
        as certificates for their row-specific randomized kit.  Each proposal
        is rematerialized with the single ``(chart, bucket)`` template that
        training will later use, then its regenerated seed and optional policy
        decode pass the four hard gates.  This keeps positive evidence
        existential, model-independent after insertion, and geometry-consistent.
        """
        checked = passed = source_rows = 0
        terminal_min = float("inf")
        gate_pass = {"terminal": 0, "collision": 0, "joint": 0, "torque": 0}
        witness_source = {"seed": 0, "model": 0}
        cap = int(self.cfg.voxel_bootstrap_rows)
        chunk = int(self.cfg.voxel_bootstrap_batch)
        bootstrap_rng = np.random.default_rng(int(self.cfg.seed) + 17_771)
        for source_bucket in self.source.buckets:
            remaining = (
                self.source.size(source_bucket)
                if cap == 0 else max(cap - source_rows, 0)
            )
            if remaining <= 0:
                break
            n = min(self.source.size(source_bucket), remaining)
            # Even spacing gives bootstrap coverage without another RNG stream.
            idx_all = torch.linspace(
                0, self.source.size(source_bucket) - 1, n,
                device=self.source.device, dtype=torch.float64,
            ).round().to(torch.long).unique()
            for start in range(0, int(idx_all.numel()), chunk):
                idx = idx_all[start:start + chunk]
                source_rows += int(idx.numel())
                cols = {
                    name: value.index_select(0, idx)
                    for name, value in self.source.data[source_bucket].items()
                    if torch.is_tensor(value)
                }
                proposals = ordinary_slice_batches(
                    cols, dt=float(self.source.cfg.dt),
                    stride=int(self.cfg.voxel_bootstrap_stride),
                    buckets=tuple(int(b) for b in costs), include_k0=True,
                )
                for bucket, proposal_cols in proposals:
                    if int(bucket) not in costs or not int(proposal_cols["q0"].shape[0]):
                        continue
                    raw = _voxel_batch_from_columns(self.source, bucket, proposal_cols)
                    charts = frontier_chart_id(
                        raw["active"], raw["context"].goal_drum,
                    )
                    terminals = _voxel_terminals(raw, self.source.cfg.joint_names)
                    # Select canonical geometry from positive evidence, not
                    # from the first random row.  Until a chart has a witness,
                    # each source row is allowed to propose its own task
                    # geometry; only a row passing all four hard gates freezes
                    # that chart's template.  Thereafter every other state is
                    # re-solved against the frozen template below.
                    raw_seed = correct_quintic_init_arms(
                        costs[bucket].fk, raw["seed"], raw["q0"], raw["hit"],
                        raw["context"].goal_pos, active=raw["active"], steps=3,
                        q_bounds=(self.source.q_lo, self.source.q_hi),
                        joint_names=self.source.cfg.joint_names,
                        qd0=raw["v0"], dt=float(self.source.cfg.dt),
                    )
                    raw_hit_eff = effective_hits(raw["hit"], raw["active"], bucket)
                    raw_ct = costs[bucket].forward(
                        raw["q0"], raw_seed, raw["v0"],
                        replace(raw["strike"], hit_index=raw_hit_eff),
                        lengths=group_lengths(raw_hit_eff), weights=weights,
                    )
                    raw_ok = _hard_feasibility_ok_from_cost(raw_ct)
                    corrected_raw = dict(raw)
                    corrected_raw["seed"] = raw_seed
                    corrected_raw["init"] = raw_seed * raw["masks"].elem_mask
                    corrected_terminals = _voxel_terminals(
                        corrected_raw, self.source.cfg.joint_names,
                    )
                    occupied_charts = {
                        key.chart_id for key in self.pool.keys(bucket=bucket)
                    }
                    for row in raw_ok.nonzero(as_tuple=True)[0].tolist():
                        chart = int(charts[row])
                        if chart in occupied_charts:
                            continue
                        template = ChartTemplate.from_batch(
                            corrected_raw, row, chart_id=chart,
                            terminal=corrected_terminals[row],
                        )
                        self.pool.register_template(template, replace=True)
                        key = self._key(
                            bucket, raw["q0"][row].detach().cpu().numpy(),
                            raw["v0"][row].detach().cpu().numpy(), chart,
                        )
                        pair = tuple(int(x) for x in raw["hit"][row].tolist())
                        hard = torch.stack([
                            raw_ct.diagnostics["terminal_region_dist"][row],
                            -raw_ct.diagnostics["collision_min_sdf"][row],
                            raw_ct.diagnostics["joint_box_excess_max"][row],
                            raw_ct.diagnostics["torque_excess_max"][row],
                        ]).float()
                        self.pool.ingest(
                            key, hit_pair=pair, feas_margin=-float(hard.max()),
                            source="bootstrap", ingress_step=-1,
                        )
                        occupied_charts.add(chart)
                        witness_source["seed"] += 1
                        passed += 1
                    checked += int(charts.numel())
                    # The first observed geometry for a chart becomes its
                    # provisional task only when no positive source row has
                    # frozen it yet.  Such provisional templates carry no
                    # evidence and may be replaced by a later valid row.
                    for chart in charts.unique().tolist():
                        chart = int(chart)
                        if self.pool.has_template(chart, bucket):
                            continue
                        row = int((charts == chart).nonzero(as_tuple=True)[0][0])
                        self.pool.register_template(ChartTemplate.from_batch(
                            raw, row, chart_id=chart, terminal=terminals[row],
                        ))

                    keys = [
                        self._key(
                            bucket,
                            raw["q0"][row].detach().cpu().numpy(),
                            raw["v0"][row].detach().cpu().numpy(),
                            int(charts[row]),
                        )
                        for row in range(int(charts.numel()))
                    ]
                    pairs = [
                        tuple(int(x) for x in raw["hit"][row].tolist())
                        for row in range(int(charts.numel()))
                    ]
                    # Exact-state overrides preserve the collocation being
                    # tested while the compact template supplies the task.
                    batch, _ = self.pool.materialize_batch(
                        keys, rng=bootstrap_rng, device=self.source.device,
                        dtype=self.source.dtype,
                        joint_names=self.source.cfg.joint_names,
                        dt=float(self.source.cfg.dt),
                        q_lo=self.source.q_lo.detach().cpu().numpy(),
                        q_hi=self.source.q_hi.detach().cpu().numpy(),
                        strike_speed_min=float(self.cfg.voxel_strike_speed_min),
                        strike_speed_max=float(self.cfg.voxel_strike_speed_max),
                        hit_pairs=pairs, q0_values=raw["q0"],
                        qd0_values=raw["v0"], probe=True,
                    )
                    batch["seed"] = correct_quintic_init_arms(
                        costs[bucket].fk, batch["seed"], batch["q0"],
                        batch["hit"], batch["context"].goal_pos,
                        active=batch["active"], steps=3,
                        q_bounds=(self.source.q_lo, self.source.q_hi),
                        joint_names=self.source.cfg.joint_names,
                        qd0=batch["v0"], dt=float(self.source.cfg.dt),
                    )
                    batch["init"] = batch["seed"] * batch["masks"].elem_mask
                    hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
                    strike = replace(batch["strike"], hit_index=hit_eff)
                    ct_seed = costs[bucket].forward(
                        batch["q0"], batch["seed"], batch["v0"], strike,
                        lengths=group_lengths(hit_eff), weights=weights,
                    )
                    candidates = [("seed", batch["seed"], ct_seed)]
                    if model is not None:
                        enc = encode(model, batch["context"])
                        traj_model = model(
                            batch["init"], batch["state"], enc, batch["masks"],
                        ).traj
                        ct_model = costs[bucket].forward(
                            batch["q0"], traj_model, batch["v0"], strike,
                            lengths=group_lengths(hit_eff), weights=weights,
                        )
                        candidates.append(("model", traj_model, ct_model))
                    candidate_ok = [
                        _hard_feasibility_ok_from_cost(item[2]) for item in candidates
                    ]
                    terminal_min = min(
                        terminal_min,
                        *(float(item[2].diagnostics["terminal_region_dist"].min())
                          for item in candidates),
                    )
                    ok = torch.stack(candidate_ok).any(dim=0)
                    checks = {
                        "terminal": ("terminal_region_dist", lambda x: x <= 1.0e-6),
                        "collision": ("collision_min_sdf", lambda x: x >= 0.0),
                        "joint": ("joint_box_excess_max", lambda x: x <= 1.0e-6),
                        "torque": ("torque_excess_max", lambda x: x <= 1.0e-6),
                    }
                    for name, (diag, predicate) in checks.items():
                        gate_pass[name] += int(torch.stack([
                            torch.isfinite(candidate[2].diagnostics[diag])
                            & predicate(candidate[2].diagnostics[diag])
                            for candidate in candidates
                        ]).any(dim=0).sum())
                    for row in ok.nonzero(as_tuple=True)[0].tolist():
                        winner = next(
                            j for j, valid in enumerate(candidate_ok) if bool(valid[row])
                        )
                        witness_source[candidates[winner][0]] += 1
                        ct = candidates[winner][2]
                        hard = torch.stack([
                            ct.diagnostics["terminal_region_dist"][row],
                            -ct.diagnostics["collision_min_sdf"][row],
                            ct.diagnostics["joint_box_excess_max"][row],
                            ct.diagnostics["torque_excess_max"][row],
                        ]).float()
                        self.pool.ingest(
                            keys[row], hit_pair=pairs[row],
                            feas_margin=-float(hard.max()),
                            source="bootstrap", ingress_step=-1,
                        )
                        passed += 1
                    checked += int(charts.numel())
        rest_stats = self._bootstrap_rest(
            costs, weights, model=model, rng=bootstrap_rng,
        )
        posthit_stats = self._bootstrap_post_hit(
            costs, weights, model=model, rng=bootstrap_rng,
        )
        return {
            "checked": checked,
            "source_rows": source_rows,
            "witness_rows": passed,
            "terminal_min": terminal_min,
            "cells": len(self.pool),
            "boundary": len(self.pool.keys(boundary=True)),
            **{f"pass_{name}": value for name, value in gate_pass.items()},
            **{f"witness_{name}": value for name, value in witness_source.items()},
            **rest_stats,
            **posthit_stats,
        }

    @staticmethod
    def _counts(n: int, fracs: tuple[float, ...]) -> list[int]:
        raw = np.asarray(fracs, dtype=np.float64) * int(n)
        out = np.floor(raw).astype(np.int64)
        order = np.argsort(-(raw - out), kind="stable")
        for i in order[:int(n) - int(out.sum())]:
            out[i] += 1
        return [int(x) for x in out]

    def _choice(self, values, n: int, rng: np.random.Generator):
        if n <= 0:
            return []
        if not values:
            return []
        pick = rng.integers(0, len(values), size=n)
        return [values[int(i)] for i in pick]

    def _probe_retry_choice(self, records, n: int, rng: np.random.Generator):
        """Cycle through buffered unknowns before giving any one extra retries."""
        if n <= 0 or not records:
            return []
        # Randomise equal-attempt ties so insertion order does not permanently
        # favour one chart, then make attempt count the primary priority.
        shuffled = [records[int(i)] for i in rng.permutation(len(records))]
        shuffled.sort(key=lambda record: int(record.attempts))
        return [record.candidate for record in shuffled[:int(n)]]

    def _new_probe(self, bucket: int, rng: np.random.Generator, step: int):
        parents = self.pool.keys(bucket=bucket, boundary=True)
        if not parents:
            return None
        source = getattr(self, "source", None)
        q_bounds = None
        if source is not None and hasattr(source, "q_lo") and hasattr(source, "q_hi"):
            q_bounds = (
                source.q_lo.detach().cpu().numpy(),
                source.q_hi.detach().cpu().numpy(),
            )
        for _ in range(max(16, len(parents))):
            parent = parents[int(rng.integers(0, len(parents)))]
            cfg = getattr(self, "cfg", None)
            proposals = self.pool.expand_at(
                parent,
                delta_hit=int(getattr(cfg, "voxel_probe_timing_step", 1)),
                shorter_layers=int(getattr(cfg, "voxel_probe_shorter_layers", 1)),
                longer_layers=int(getattr(cfg, "voxel_probe_longer_layers", 4)),
            )
            if not proposals:
                continue
            available = []
            for proposal in proposals:
                # Positional expansion can step through a face of the hard
                # joint box.  Such a cell cannot be materialized and is not a
                # useful unknown probe, so discard it before it enters the
                # retry buffer.
                if q_bounds is not None and not self.pool.q0_cell_intersects_bounds(
                    proposal.key, q_lo=q_bounds[0], q_hi=q_bounds[1],
                ):
                    continue
                candidate = ProbeCandidateKey(proposal.key, proposal.hit_pair)
                if candidate in self.probes:
                    continue
                if (
                    proposal.key in self.pool
                    and proposal.hit_pair in self.pool.cell(proposal.key).hit_pairs
                ):
                    continue
                available.append((proposal, candidate))
            if not available:
                continue
            timing = [item for item in available if item[0].key == parent]
            spatial = [item for item in available if item[0].key != parent]
            timing_frac = float(getattr(cfg, "voxel_probe_timing_frac", 0.5))
            choose_timing = bool(timing) and (
                not spatial or float(rng.random()) < timing_frac
            )
            choices = timing if choose_timing else spatial
            proposal, candidate = choices[int(rng.integers(0, len(choices)))]
            # New checkpoints carry the compact chart template in VoxelPool.
            # Retain the source-row lookup only as a migration fallback for
            # version-1 checkpoints and focused compatibility tests.
            row = -1
            has_template = getattr(self.pool, "has_template", lambda *_: False)
            if not has_template(parent.chart_id, parent.bucket):
                entries = [
                    item for item in self.templates.get(parent, [])
                    if item[1] == proposal.parent_hit_pair
                ]
                if not entries:
                    continue
                row, _parent_pair = entries[int(rng.integers(0, len(entries)))]
            # An old failed probe remains unknown, but must not masquerade as a
            # newly generated candidate and monopolize every refresh batch.
            if not self.probes.add(candidate, step=step):
                continue
            # ProbeBuffer may have evicted its oldest unknown at capacity;
            # keep the materialisation side tables bounded by the same set.
            self.probe_templates = {
                k: v for k, v in self.probe_templates.items() if k in self.probes
            }
            self.probe_finishing_hands = {
                k: v for k, v in self.probe_finishing_hands.items() if k in self.probes
            }
            self.probe_templates.setdefault(candidate, (row, proposal.hit_pair))
            self.probe_finishing_hands.setdefault(candidate, proposal.finishing_hand)
            return candidate
        return None

    def _materialize(
        self, bucket: int, keys: list[VoxelKey] | list[ProbeCandidateKey], *, probe: bool,
        generator: torch.Generator, rng: np.random.Generator,
        yaw_i: int,
    ) -> tuple[dict | None, _VoxelBatchMeta]:
        if not keys:
            return None, _VoxelBatchMeta([], [], [], [])
        candidates = keys if probe else None
        voxel_keys = (
            [candidate.key for candidate in candidates]
            if candidates is not None else keys
        )
        if all(self.pool.has_template(key.chart_id, key.bucket) for key in voxel_keys):
            explicit_pairs = None
            if probe:
                explicit_pairs = [candidate.hit_pair for candidate in candidates]
            batch, pairs = self.pool.materialize_batch(
                voxel_keys, rng=rng, device=self.source.device,
                dtype=self.source.dtype,
                joint_names=self.source.cfg.joint_names,
                dt=float(self.source.cfg.dt),
                q_lo=self.source.q_lo.detach().cpu().numpy(),
                q_hi=self.source.q_hi.detach().cpu().numpy(),
                strike_speed_min=float(self.cfg.voxel_strike_speed_min),
                strike_speed_max=float(self.cfg.voxel_strike_speed_max),
                hit_pairs=explicit_pairs, probe=probe,
            )
            _voxel_current_goal_only(batch)
            n = len(voxel_keys)
            return batch, _VoxelBatchMeta(
                keys=list(voxel_keys), template_rows=[None] * n,
                hit_pairs=list(pairs), is_probe=[probe] * n,
            )
        chosen: list[tuple[int, tuple[int, int]]] = []
        for item in keys:
            key = item.key if probe else item
            entries = ([self.probe_templates[item]] if probe
                       else self.templates.get(key, []))
            if not entries:
                raise RuntimeError(f"voxel {key} has no materialization template")
            chosen.append(entries[int(rng.integers(0, len(entries)))])
        idx = torch.as_tensor(
            [x[0] for x in chosen], device=self.source.device, dtype=torch.long,
        )
        batch = self.source.batch(bucket, idx, generator=generator, noise=False)
        # Preserve the task terminal from the source witness *before* replacing
        # its timing.  Gathering the old seed at a newly tightened knot would
        # select an in-flight pose and make the probe aim short of the drum.
        source_hit_eff = effective_hits(
            batch["hit"], batch["active"], bucket,
        )
        source_elem_len = per_element_lengths(
            source_hit_eff, self.source.cfg.joint_names,
        )
        source_gather = (source_elem_len.clamp(1, bucket) - 1).unsqueeze(1)
        terminal = batch["seed"].gather(1, source_gather).squeeze(1)
        # The pair selected above is the complete certified witness (or the
        # complete parent-derived probe proposal).  Never leave the source
        # row's original per-hand timing in the actual optimization batch.
        hit = torch.as_tensor(
            [item[1] for item in chosen], device=self.source.device,
            dtype=torch.long,
        )
        q_lo = self.source.q_lo.detach().cpu().numpy()
        q_hi = self.source.q_hi.detach().cpu().numpy()
        q0 = torch.as_tensor(
            self.pool.sample_q0(voxel_keys, rng=rng, q_lo=q_lo, q_hi=q_hi),
            device=self.source.device, dtype=self.source.dtype,
        )
        # A key includes the coarse velocity bin just as strongly as its q cell.
        # Reusing the template row's velocity would silently materialize a state
        # outside the requested voxel and invalidate the existential witness
        # semantics.  Finite tails match VoxelPool's default four-bin support.
        qd0 = torch.as_tensor(
            self.pool.sample_qd0(voxel_keys, rng=rng),
            device=self.source.device, dtype=self.source.dtype,
        )
        batch["q0"] = q0
        batch["qd0"] = qd0
        batch["v0"] = qd0
        batch["state"] = torch.cat([q0, qd0], dim=-1)
        if batch["context"].yaw_now is not None:
            batch["context"].yaw_now = q0[:, int(yaw_i)]
        retime_batch_timing(batch, hit, self.source, terminal=terminal)
        n = len(voxel_keys)
        batch["mpc_init"] = torch.zeros(n, device=q0.device, dtype=torch.bool)
        batch["al_ok"] = torch.full(
            (n,), not probe, device=q0.device, dtype=torch.bool,
        )
        batch["voxel_kind"] = torch.full(
            (n,), 3 if probe else 1, device=q0.device, dtype=torch.long,
        )
        speed = torch.empty(
            n, 2, device=q0.device, dtype=q0.dtype,
        ).uniform_(
            float(self.cfg.voxel_strike_speed_min),
            float(self.cfg.voxel_strike_speed_max),
            generator=generator,
        )
        active = batch["active"].to(torch.bool)
        goal_vel = (
            -speed.unsqueeze(-1) * batch["strike"].n_out
            * active.unsqueeze(-1).to(q0.dtype)
        )
        batch["strike"] = replace(batch["strike"], v_star=goal_vel)
        batch["context"] = replace(batch["context"], goal_vel=goal_vel)
        return batch, _VoxelBatchMeta(
            keys=list(voxel_keys), template_rows=[x[0] for x in chosen],
            hit_pairs=[x[1] for x in chosen], is_probe=[probe] * n,
        )

    def sample_batch(
        self, bucket: int, n: int, mpc_ring: _VoxelMPCParentRing, *, generator: torch.Generator,
        rng: np.random.Generator, yaw_i: int, step: int,
    ) -> tuple[dict, _VoxelBatchMeta]:
        fracs = (
            self.cfg.voxel_frac_mpc, self.cfg.voxel_frac_pool,
            self.cfg.voxel_frac_boundary, self.cfg.voxel_frac_probe,
        )
        n_mpc, n_pool, n_boundary, n_probe = self._counts(n, fracs)
        feasible = self.pool.keys(bucket=bucket)
        boundary = self.pool.keys(bucket=bucket, boundary=True)
        if not feasible:
            raise RuntimeError(f"voxel bootstrap produced no feasible cells for bucket {bucket}")
        if mpc_ring.size(bucket) == 0:
            n_pool += n_mpc
            n_mpc = 0
        if not boundary:
            n_pool += n_boundary + n_probe
            n_boundary = n_probe = 0

        parts: list[dict] = []
        meta = _VoxelBatchMeta([], [], [], [])
        if n_mpc:
            idx = mpc_ring.sample_idx(bucket, n_mpc, generator=generator)
            part = mpc_ring.batch(bucket, idx, generator=generator, noise=False)
            part["mpc_init"] = torch.ones(n_mpc, device=self.source.device, dtype=torch.bool)
            part["al_ok"] = torch.ones(n_mpc, device=self.source.device, dtype=torch.bool)
            part["voxel_kind"] = torch.zeros(n_mpc, device=self.source.device, dtype=torch.long)
            parts.append(part)
            meta.keys.extend([None] * n_mpc)
            meta.template_rows.extend([None] * n_mpc)
            meta.hit_pairs.extend([None] * n_mpc)
            meta.is_probe.extend([False] * n_mpc)

        for count, source, kind in (
            (n_pool, feasible, 1), (n_boundary, boundary, 2),
        ):
            keys = self._choice(source, count, rng)
            part, pm = self._materialize(
                bucket, keys, probe=False, generator=generator, rng=rng, yaw_i=yaw_i,
            )
            if part is not None:
                part["voxel_kind"].fill_(kind)
                parts.append(part)
                for name in ("keys", "template_rows", "hit_pairs", "is_probe"):
                    getattr(meta, name).extend(getattr(pm, name))

        probe_candidates: list[ProbeCandidateKey] = []
        refresh = int(step) % int(self.cfg.voxel_probe_refresh_every) == 0
        if refresh:
            # Rotate in fresh unknown spatial/timing candidates on schedule.
            # Failed probes stay unknown in the bounded buffer and can be
            # retried later; they are not interpreted as negatives.
            for _ in range(n_probe):
                key = self._new_probe(bucket, rng, step)
                if key is None:
                    break
                probe_candidates.append(key)
        existing = [
            record for record in self.probes.records()
            if record.key.bucket == bucket
            and record.candidate not in probe_candidates
        ]
        probe_candidates.extend(self._probe_retry_choice(
            existing, min(n_probe - len(probe_candidates), len(existing)), rng,
        ))
        while len(probe_candidates) < n_probe:
            key = self._new_probe(bucket, rng, step)
            if key is None:
                break
            probe_candidates.append(key)
        if len(probe_candidates) < n_probe:
            # Candidate exhaustion is not a negative label; use known feasible
            # rows for the unfilled optimizer slots.
            fill = self._choice(feasible, n_probe - len(probe_candidates), rng)
            part, pm = self._materialize(
                bucket, fill, probe=False, generator=generator, rng=rng, yaw_i=yaw_i,
            )
            if part is not None:
                parts.append(part)
                for name in ("keys", "template_rows", "hit_pairs", "is_probe"):
                    getattr(meta, name).extend(getattr(pm, name))
        part, pm = self._materialize(
            bucket, probe_candidates, probe=True,
            generator=generator, rng=rng, yaw_i=yaw_i,
        )
        if part is not None:
            parts.append(part)
            for name in ("keys", "template_rows", "hit_pairs", "is_probe"):
                getattr(meta, name).extend(getattr(pm, name))

        batch = _cat_phase2_batches(parts)
        _voxel_current_goal_only(batch)
        order = torch.randperm(n, device=self.source.device, generator=generator)
        batch = _index_batch(batch, order)
        order_cpu = order.detach().cpu().tolist()
        meta = _VoxelBatchMeta(**{
            name: [getattr(meta, name)[i] for i in order_cpu]
            for name in ("keys", "template_rows", "hit_pairs", "is_probe")
        })
        return batch, meta

    def record_probe_results(self, meta: _VoxelBatchMeta, hard_ok, *, step: int) -> dict:
        """Promote any successful candidate; retain failures as unknown."""
        grouped: dict[ProbeCandidateKey, list[int]] = {}
        for i, (key, pair, probe) in enumerate(zip(
            meta.keys, meta.hit_pairs, meta.is_probe,
        )):
            if probe and key is not None and pair is not None:
                grouped.setdefault(ProbeCandidateKey(key, pair), []).append(i)
        promoted = failed = 0
        ok_cpu = hard_ok.detach().cpu().tolist()
        for candidate, rows in grouped.items():
            key = candidate.key
            successes = [i for i in rows if bool(ok_cpu[i])]
            if successes:
                i = successes[0]
                pair = meta.hit_pairs[i]
                template_row = meta.template_rows[i]
                template = (
                    None if template_row is None else (int(template_row), pair)
                )
                if candidate in self.probes:
                    self.probes.promote(
                        candidate, self.pool, source="mpc",
                        ingress_step=int(step), hard_valid=True,
                        finishing_hand=int(
                            self.probe_finishing_hands.get(candidate, -1)
                        ),
                    )
                if template is not None:
                    entries = self.templates.setdefault(key, [])
                    if template not in entries:
                        entries.append(template)
                self.probe_templates.pop(candidate, None)
                self.probe_finishing_hands.pop(candidate, None)
                promoted += 1
            else:
                if candidate in self.probes:
                    self.probes.record_failure(candidate)
                failed += 1
        return {"promoted": promoted, "failed_unknown": failed}

    def record_train_ingress(
        self, meta: _VoxelBatchMeta, hard_ok: torch.Tensor, *, step: int,
    ) -> int:
        """Attach current-policy positive evidence to confirmed voxel rows."""
        ok = hard_ok.detach().to(device="cpu", dtype=torch.bool).tolist()
        ingressed = 0
        for i, (key, pair, is_probe) in enumerate(zip(
            meta.keys, meta.hit_pairs, meta.is_probe,
        )):
            if key is None or pair is None or is_probe or not bool(ok[i]):
                continue
            self.pool.ingest(
                key, hit_pair=pair, source="mpc", ingress_step=int(step),
            )
            ingressed += 1
        return ingressed

    @torch.no_grad()
    def prime_mpc_ring(
        self,
        ring: _VoxelMPCParentRing,
        model,
        costs: dict[int, FusedCost],
        weights: dict,
        *,
        generator: torch.Generator,
        rng: np.random.Generator,
        yaw_i: int,
    ) -> dict[int, int]:
        """Create the first shifted-parent generation from current decodes."""
        primed: dict[int, int] = {}
        for bucket in self.source.buckets:
            if ring.size(bucket):
                continue
            feasible = self.pool.keys(bucket=bucket)
            if not feasible:
                continue
            desired = max(
                self._counts(self.cfg.batch_for(bucket), (
                    self.cfg.voxel_frac_mpc, self.cfg.voxel_frac_pool,
                    self.cfg.voxel_frac_boundary, self.cfg.voxel_frac_probe,
                ))[0],
                1,
            )
            keys = self._choice(feasible, desired, rng)
            batch, _meta = self._materialize(
                bucket, keys, probe=False, generator=generator, rng=rng,
                yaw_i=yaw_i,
            )
            if batch is None:
                continue
            _voxel_current_goal_only(batch)
            enc = encode(model, batch["context"])
            traj = model(
                batch["init"], batch["state"], enc, batch["masks"],
            ).traj
            hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
            strike = replace(batch["strike"], hit_index=hit_eff)
            ct = costs[bucket].forward(
                batch["q0"], traj, batch["v0"], strike,
                lengths=group_lengths(hit_eff), weights=weights,
            )
            hard_ok = _hard_feasibility_ok_from_cost(ct)
            primed[int(bucket)] = ring.refresh(
                bucket, batch, traj, hard_ok=hard_ok,
            )
        return primed


def _index_batch(batch: dict, idx: torch.Tensor) -> dict:
    """Index-select tensors and dataclass-of-tensors along the sample axis."""
    out: dict = {}
    for key, val in batch.items():
        if key == "bucket" or not (
            torch.is_tensor(val) or hasattr(val, "__dataclass_fields__")
        ):
            out[key] = val
            continue
        if torch.is_tensor(val):
            out[key] = val.index_select(0, idx)
            continue
        kwargs = {}
        for f in val.__dataclass_fields__:
            fv = getattr(val, f)
            kwargs[f] = (
                None if fv is None else
                fv.index_select(0, idx) if torch.is_tensor(fv) else fv
            )
        out[key] = type(val)(**kwargs)
    return out


def _split_timing_superbatch(
    batch: dict,
    minibatch_size: int,
    factor: int,
    *,
    generator: torch.Generator,
) -> list[dict]:
    """Shuffle/split a timing batch without separating a task's timing rows.

    Boundary and interior rows share ``timing_primary_group`` and must stay in
    one actor minibatch because interiors reuse their group's selected frontier.
    Ungrouped MPC/anchor rows are independently shuffled and divided evenly.
    """
    b = _train_batch_rows(batch)
    minibatch_size = int(minibatch_size)
    factor = int(factor)
    if b != minibatch_size * factor:
        raise ValueError(
            f"superbatch rows {b} != minibatch_size*factor "
            f"({minibatch_size}*{factor})"
        )
    device = batch["q0"].device
    groups = batch.get("timing_primary_group")
    if not torch.is_tensor(groups) or not bool((groups >= 0).any()):
        order = torch.randperm(b, device=device, generator=generator)
        return [
            _index_batch(batch, order[i * minibatch_size:(i + 1) * minibatch_size])
            for i in range(factor)
        ]

    grouped = groups >= 0
    group_ids = groups[grouped].unique(sorted=True)
    if int(group_ids.numel()) % factor:
        raise ValueError("timing task groups must divide evenly across superbatch slices")
    group_ids = group_ids.index_select(
        0,
        torch.randperm(int(group_ids.numel()), device=device, generator=generator),
    )
    ungrouped_rows = (~grouped).nonzero(as_tuple=True)[0]
    ungrouped_rows = ungrouped_rows.index_select(
        0,
        torch.randperm(int(ungrouped_rows.numel()), device=device, generator=generator),
    )
    groups_per_slice = int(group_ids.numel()) // factor
    ungrouped_per_slice = int(ungrouped_rows.numel()) // factor
    if int(ungrouped_rows.numel()) != ungrouped_per_slice * factor:
        raise ValueError("ungrouped timing rows must divide evenly across superbatch slices")

    out = []
    for i in range(factor):
        chosen_groups = group_ids[
            i * groups_per_slice:(i + 1) * groups_per_slice
        ]
        chosen_mask = (groups[:, None] == chosen_groups[None, :]).any(dim=1)
        rows = torch.cat((
            chosen_mask.nonzero(as_tuple=True)[0],
            ungrouped_rows[i * ungrouped_per_slice:(i + 1) * ungrouped_per_slice],
        ))
        if int(rows.numel()) != minibatch_size:
            raise ValueError(
                f"timing superbatch slice has {int(rows.numel())} rows, "
                f"expected {minibatch_size}"
            )
        rows = rows.index_select(
            0,
            torch.randperm(minibatch_size, device=device, generator=generator),
        )
        out.append(_index_batch(batch, rows))
    return out


def _mpc_aux_rows(mpc_init: torch.Tensor, frac: float) -> torch.Tensor:
    """Pick auxiliary-loss rows exclusively from shifted-parent examples."""
    rows = mpc_init.to(dtype=torch.bool).nonzero(as_tuple=True)[0]
    if not int(rows.numel()) or float(frac) <= 0.0:
        return rows[:0]
    requested = max(int(float(frac) * int(mpc_init.numel())), 1)
    return rows[:min(requested, int(rows.numel()))]


def _suffix_rows(batch: dict, frac: float) -> torch.Tensor:
    """Proportionally select Bellman parents from every primary stratum."""
    n = int(batch["q0"].shape[0])
    requested = min(max(int(float(frac) * n), 1), n)
    kind = batch.get("timing_primary_kind")
    if not torch.is_tensor(kind):
        return torch.arange(requested, device=batch["q0"].device)
    groups = [
        (kind == value).nonzero(as_tuple=True)[0]
        for value in kind.unique(sorted=True)
    ]
    groups = [rows for rows in groups if int(rows.numel())]
    if len(groups) <= 1:
        return groups[0][:requested]

    exact = [requested * int(rows.numel()) / n for rows in groups]
    counts = [int(math.floor(value)) for value in exact]
    if requested >= len(groups):
        counts = [max(count, 1) for count in counts]
    while sum(counts) > requested:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        counts[max(candidates, key=lambda i: counts[i] - exact[i])] -= 1
    while sum(counts) < requested:
        candidates = [
            i for i, rows in enumerate(groups) if counts[i] < int(rows.numel())
        ]
        counts[max(candidates, key=lambda i: exact[i] - counts[i])] += 1
    return torch.cat([
        rows[:count] for rows, count in zip(groups, counts) if count > 0
    ])


def _index_encoded(enc, idx: torch.Tensor):
    """Index an encoder-output dataclass without depending on head layout."""
    return type(enc)(**{
        name: (
            None if getattr(enc, name) is None else
            getattr(enc, name).index_select(0, idx)
            if torch.is_tensor(getattr(enc, name)) else getattr(enc, name)
        )
        for name in enc.__dataclass_fields__
    })


def _index_batch_weights(weights: dict, idx: torch.Tensor, batch_size: int) -> dict:
    """Slice only per-row weights; preserve scalar and per-joint schedules."""
    out = {}
    for key, value in weights.items():
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and int(value.shape[0]) == int(batch_size)
            and key != "limit_alpha"
        ):
            out[key] = value.index_select(0, idx)
        else:
            out[key] = value
    return out


def _dls_rows(
    batch_size: int,
    frac: float,
    device: torch.device,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic, evenly-spaced row subset used by terminal DLS.

    Determinism is part of the objective contract: candidate optimizer steps
    in SaLSa must replay the same rows as the baseline objective.
    """
    base = (
        torch.arange(int(batch_size), device=device, dtype=torch.long)
        if eligible is None else
        torch.nonzero(eligible.to(device=device).bool(), as_tuple=False).flatten()
    )
    if base.numel() == 0:
        return base
    n = min(
        max(int(math.ceil(float(frac) * int(batch_size))), 1),
        int(base.numel()),
    )
    take = torch.div(
        torch.arange(n, device=device, dtype=torch.long) * int(base.numel()),
        n,
        rounding_mode="floor",
    )
    return base.index_select(0, take)


def _dls_contact_local(
    cost: FusedCost,
    strike: StrikeContext,
    hand_index: int,
) -> torch.Tensor:
    """Requested contact point in each row's wrist frame.

    Recover the mount from the randomized tip when necessary.  This keeps the
    DLS teacher on exactly the same stick geometry as ``FusedCost._terminal``.
    """
    hand = HANDS[int(hand_index)]
    sigma = strike.contact_sigma[:, hand_index].float().unsqueeze(-1)
    if strike.stick_axis_local is None:
        axis = cost.c_mount_axis[hand].float().view(1, 3).expand(strike.batch, -1)
    else:
        axis = strike.stick_axis_local[:, hand_index].float()
    if strike.stick_points_local is None:
        mount = cost.c_mount_p[hand].float().view(1, 3).expand(strike.batch, -1)
    else:
        tip = strike.stick_points_local[:, hand_index, 0].float()
        if strike.stick_tip_sigma is None:
            tip_sigma = sigma.new_full((strike.batch, 1), float(STICK_TIP_SIGMA))
        else:
            tip_sigma = strike.stick_tip_sigma[:, hand_index].float().unsqueeze(-1)
        mount = tip - tip_sigma * axis
    return mount + sigma * axis


def _dls_contact_position(
    fk: ChainFK,
    q: torch.Tensor,
    hand: str,
    local: torch.Tensor,
) -> torch.Tensor:
    dev_type = "cuda" if q.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=dev_type, enabled=False):
        R, p = fk.frame(q.float(), f"{hand}_wrist")
        return p + torch.einsum("bij,bj->bi", R, local.float())


def _dls_contact_position_jacobian(
    fk: ChainFK,
    q: torch.Tensor,
    hand: str,
    local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contact position and exact positional Jacobian, both detached."""
    dev_type = "cuda" if q.device.type == "cuda" else "cpu"
    with (
        torch.inference_mode(False), torch.enable_grad(),
        torch.autocast(device_type=dev_type, enabled=False),
    ):
        qg = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        qg.copy_(q.detach().float())
        qg.requires_grad_(True)
        pos = _dls_contact_position(fk, qg, hand, local.float())
        rows = []
        for axis in range(3):
            (grad,) = torch.autograd.grad(
                pos[:, axis].sum(), qg,
                retain_graph=axis < 2, create_graph=False,
            )
            rows.append(grad)
    return pos.detach(), torch.stack(rows, dim=1).detach()


def dls_terminal_teacher(
    traj: torch.Tensor,
    strike: StrikeContext,
    cost: FusedCost,
    *,
    steps: int = 1,
    window: int = 16,
    damping: float = 0.05,
    max_step_rad: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build a detached arm-only DLS terminal teacher and approach mask.

    Each hand is solved independently with the waist frozen.  Its terminal
    delta is propagated through the last ``window`` knots with a rest-to-rest
    quintic blend, avoiding a discontinuity one knot before contact.
    """
    if traj.ndim != 3:
        raise ValueError(f"traj must be (B, H, D), got {tuple(traj.shape)}")
    B, H, D = traj.shape
    teacher = traj.detach().float().clone()
    tie_mask = teacher.new_zeros(B, H, D)
    before_sum = teacher.new_zeros(())
    after_sum = teacher.new_zeros(())
    active_count = teacher.new_zeros(())
    delta_sum = teacher.new_zeros(())
    delta_count = teacher.new_zeros(())
    knot = torch.arange(1, H + 1, device=traj.device, dtype=torch.float32).view(1, H)

    for hi, hand in enumerate(HANDS):
        active_rows = torch.nonzero(
            strike.active[:, hi] > 0.5, as_tuple=False,
        ).flatten()
        if active_rows.numel() == 0:
            continue
        hit_all = strike.hit_index[:, hi].clamp(1, H)
        hit = hit_all.index_select(0, active_rows)
        gather = (hit - 1).view(-1, 1, 1).expand(-1, 1, D)
        q_hit = teacher.index_select(0, active_rows).gather(1, gather).squeeze(1)
        aim = strike.p_star[:, hi].index_select(0, active_rows).detach().float()
        local = _dls_contact_local(cost, strike, hi).index_select(0, active_rows)
        cols = torch.as_tensor(
            [i for i, name in enumerate(cost.joint_names) if name.startswith(f"{hand}_")],
            device=traj.device, dtype=torch.long,
        )
        q_start = q_hit.clone()
        pos0 = _dls_contact_position(cost.fk, q_hit, hand, local)
        before_sum = before_sum + (aim - pos0).norm(dim=-1).sum()
        active_count = active_count + float(active_rows.numel())
        eye = torch.eye(3, device=traj.device, dtype=torch.float32).expand(
            active_rows.numel(), -1, -1,
        )
        for _ in range(int(steps)):
            pos, jac = _dls_contact_position_jacobian(
                cost.fk, q_hit, hand, local,
            )
            jac = jac.index_select(2, cols)
            jt = jac.transpose(1, 2)
            rhs = (aim - pos).unsqueeze(-1)
            dq = (jt @ torch.linalg.solve(
                jac @ jt + float(damping) ** 2 * eye, rhs,
            )).squeeze(-1)
            dq = dq.clamp(-float(max_step_rad), float(max_step_rad))
            q_arm = q_hit.index_select(1, cols) + dq
            q_arm = torch.maximum(
                torch.minimum(
                    q_arm, cost.q_hi_hard.index_select(0, cols).float(),
                ),
                cost.q_lo_hard.index_select(0, cols).float(),
            )
            q_hit = q_hit.index_copy(1, cols, q_arm)
        pos1 = _dls_contact_position(cost.fk, q_hit, hand, local)
        after_sum = after_sum + (aim - pos1).norm(dim=-1).sum()
        delta = q_hit.index_select(1, cols) - q_start.index_select(1, cols)
        delta_sum = delta_sum + delta.abs().sum()
        delta_count = delta_count + float(delta.numel())

        # A quintic alpha is zero at the beginning of the approach and one at
        # the strike.  Knots after the hit are intentionally untouched.
        start = (hit - int(window)).clamp_min(0).float().unsqueeze(1)
        denom = (hit.float().unsqueeze(1) - start).clamp_min(1.0)
        u = ((knot - start) / denom).clamp(0.0, 1.0)
        alpha = u.pow(3) * (10.0 - 15.0 * u + 6.0 * u.pow(2))
        alpha = alpha * (knot <= hit.unsqueeze(1)).float()
        add = teacher.new_zeros(active_rows.numel(), H, D)
        add[:, :, cols] = alpha.unsqueeze(-1) * delta.unsqueeze(1)
        current = teacher.index_select(0, active_rows) + add
        teacher.index_copy_(0, active_rows, current)
        hand_mask = teacher.new_zeros(active_rows.numel(), H, D)
        hand_mask[:, :, cols] = (alpha > 0.0).float().unsqueeze(-1)
        tie_mask.index_copy_(
            0, active_rows,
            torch.maximum(tie_mask.index_select(0, active_rows), hand_mask),
        )

    denom = active_count.clamp_min(1.0)
    stats = {
        "dls_terminal_rows": traj.new_tensor(float(B)),
        "dls_terminal_active_hands": active_count,
        "dls_terminal_err_before_m": before_sum / denom,
        "dls_terminal_err_after_m": after_sum / denom,
        "dls_terminal_delta_abs_rad": delta_sum / delta_count.clamp_min(1.0),
    }
    return teacher.to(traj.dtype), tie_mask.to(traj.dtype), stats


def _spline_smoothness_hessians(cost: FusedCost) -> torch.Tensor:
    """Exact per-joint waypoint Hessians of this cost's spline smoothness.

    The spline gain table maps waypoints linearly to Hermite endpoint slopes.
    Thus the closed-form acceleration/jerk integrals are quadratic in the
    waypoints.  Cache their Hessians for every possible group horizon; the
    row-dependent valid-element denominator is applied in the solve below.
    The measured initial position/velocity are constants, not variables.
    """
    builder = cost.spline
    key = (
        float(cost.cfg.w_smooth_acc), float(cost.cfg.w_smooth_jerk),
        str(cost.device), str(cost.dtype),
    )
    cache = getattr(builder, "_curvature_smooth_hessians", None)
    if cache is None:
        cache = {}
        builder._curvature_smooth_hessians = cache
    if key in cache:
        return cache[key]

    H, dt = builder.max_segments, float(builder.dt)
    basis = np.zeros((H + 1, H), dtype=np.float64)
    for j in range(H):
        basis[j + 1, j] = 1.0
    hess = np.zeros((H + 1, H, H), dtype=np.float64)
    wa, wj = float(cost.cfg.w_smooth_acc), float(cost.cfg.w_smooth_jerk)
    for length in range(1, H + 1):
        x = basis.copy()
        x[length + 1:] = 0.0
        slopes = builder._Gx_np[length] @ x
        a2 = 3.0 * (x[1:length + 1] - x[:length]) / dt**2 \
            - (2.0 * slopes[:length] + slopes[1:length + 1]) / dt
        a3 = 2.0 * (x[:length] - x[1:length + 1]) / dt**3 \
            + (slopes[:length] + slopes[1:length + 1]) / dt**2
        qa = (
            4.0 * dt * (a2.T @ a2)
            + 6.0 * dt**2 * (a2.T @ a3 + a3.T @ a2)
            + 12.0 * dt**3 * (a3.T @ a3)
        )
        qj = 36.0 * dt * (a3.T @ a3)
        hess[length] = 2.0 * (wa * qa + wj * qj)
    out = torch.as_tensor(hess, device=cost.device, dtype=torch.float32)
    cache[key] = out
    return out


def curvature_terminal_teacher(
    traj: torch.Tensor,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike: StrikeContext,
    lengths: torch.Tensor,
    weights: dict,
    cost: FusedCost,
    cfg: Phase2Config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One full-cost trajectory step in an analytic, positive curvature metric.

    A0 is the exact spline acceleration/jerk Hessian plus damping, separately
    for every joint's actual horizon.  Position and finite-difference impact
    velocity Jacobians add low-rank Gauss--Newton blocks for both hands, with
    shared waist columns.  Woodbury reduces the coupled solve to banded-sized
    per-joint solves and a <=12 by 12 contact system.  The original FusedCost
    must decrease for a row's detached teacher to be admitted.
    """
    B, H, D = traj.shape
    device = traj.device
    with torch.enable_grad():
        probe = traj.detach().float().requires_grad_(True)
        before = cost.forward(q0.float(), probe, v0.float(), strike,
                              lengths=lengths, weights=weights)
        grad = torch.autograd.grad(before.total.sum(), probe)[0].detach()
    with torch.no_grad():
        group = torch.as_tensor(cost.spline._group_np, device=device)
        per_len = lengths.clamp(1, H).index_select(1, group)
        valid = (
            torch.arange(H, device=device).view(1, H, 1)
            < per_len.unsqueeze(1)
        ).to(grad.dtype)
        grad = grad * valid
        n_valid = per_len.sum(dim=1).clamp_min(1).to(grad.dtype)
        if cost.cfg.reduction != "mean":
            n_valid = torch.ones_like(n_valid)
        stiffness = _spline_smoothness_hessians(cost).index_select(
            0, per_len.reshape(-1)
        ).reshape(B, D, H, H)
        smooth_weight = weights.get("smooth", 1.0)
        stiffness = stiffness * torch.as_tensor(
            smooth_weight, device=device, dtype=grad.dtype,
        ).reshape(-1, 1, 1, 1) / n_valid.view(B, 1, 1, 1)
        eye_h = torch.eye(H, device=device, dtype=grad.dtype)
        base = stiffness + float(cfg.curvature_damping) * eye_h
        chol = torch.linalg.cholesky(base.reshape(B * D, H, H))

        # Residual rows: (left position xyz, left velocity xyz, right ...).
        jac = grad.new_zeros(B, 12, D, H)
        root_w = grad.new_zeros(B, 12)
        hit_positions = []
        for hi, hand in enumerate(HANDS):
            hit = strike.hit_index[:, hi].clamp(1, H)
            active = (strike.active[:, hi] > 0.5).to(grad.dtype)
            at_hit = probe.detach().gather(
                1, (hit - 1).view(B, 1, 1).expand(B, 1, D),
            ).squeeze(1)
            prev_all = torch.cat([q0.float().unsqueeze(1), probe.detach()[:, :-1]], dim=1)
            at_prev = prev_all.gather(
                1, (hit - 1).view(B, 1, 1).expand(B, 1, D),
            ).squeeze(1)
            local = _dls_contact_local(cost, strike, hi)
            pos_hit, j_hit = _dls_contact_position_jacobian(
                cost.fk, at_hit, hand, local,
            )
            _pos_prev, j_prev = _dls_contact_position_jacobian(
                cost.fk, at_prev, hand, local,
            )
            hit_positions.append(pos_hit)
            j_hit = j_hit * active.view(B, 1, 1)
            j_prev = j_prev * (active * (hit > 1).to(grad.dtype)).view(B, 1, 1)
            hidx = (hit - 1).view(B, 1, 1, 1).expand(B, 3, D, 1)
            pidx = (hit - 2).clamp_min(0).view(B, 1, 1, 1).expand(B, 3, D, 1)
            jpos = jac[:, hi * 6:hi * 6 + 3]
            jvel = jac[:, hi * 6 + 3:hi * 6 + 6]
            jpos.scatter_add_(3, hidx, j_hit.unsqueeze(-1))
            jvel.scatter_add_(3, hidx, (j_hit / cost.dt).unsqueeze(-1))
            jvel.scatter_add_(3, pidx, (-j_prev / cost.dt).unsqueeze(-1))
            root_w[:, hi * 6:hi * 6 + 3] = (
                float(cfg.curvature_position_weight) ** 0.5 * active.unsqueeze(1)
            )
            root_w[:, hi * 6 + 3:hi * 6 + 6] = (
                float(cfg.curvature_velocity_weight) ** 0.5 * active.unsqueeze(1)
            )

        rhs = torch.cat([
            grad.permute(0, 2, 1).unsqueeze(-1),
            jac.permute(0, 2, 3, 1),
        ], dim=-1).reshape(B * D, H, 13)
        solved = torch.cholesky_solve(rhs, chol).reshape(B, D, H, 13)
        u0, u_j = solved[..., 0], solved[..., 1:]
        ju0 = torch.einsum("brdh,bdh->br", jac, u0)
        juj = torch.einsum("brdh,bdhs->brs", jac, u_j)
        small = (
            torch.eye(12, device=device, dtype=grad.dtype).unsqueeze(0)
            + root_w.unsqueeze(2) * juj * root_w.unsqueeze(1)
        )
        small_rhs = (root_w * ju0).unsqueeze(-1)
        correction = torch.linalg.solve(small, small_rhs).squeeze(-1) * root_w
        direction = -(u0 - torch.einsum("bdhs,bs->bdh", u_j, correction))
        direction = direction.permute(0, 2, 1) * valid
        max_abs = direction.abs().amax(dim=(1, 2)).clamp_min(1e-12)
        step = (float(cfg.curvature_max_step_rad) / max_abs).clamp(max=1.0)
        accepted = torch.zeros(B, device=device, dtype=torch.bool)
        teacher = probe.detach().clone()
        improvement = grad.new_zeros(B)
        # The full cost is nonconvex and includes changing max-active sets;
        # shrink the analytic descent direction until the *actual* cost falls.
        for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
            if bool(accepted.all()):
                break
            candidate = probe.detach() + direction * (step * shrink).view(B, 1, 1)
            after = cost.forward(q0.float(), candidate, v0.float(), strike,
                                 lengths=lengths, weights=weights)
            delta_cost = before.total.detach() - after.total
            better = (~accepted) & torch.isfinite(after.total) & (delta_cost > 1e-6)
            teacher = torch.where(better.view(B, 1, 1), candidate, teacher)
            improvement = torch.where(better, delta_cost, improvement)
            accepted |= better
        tie_mask = valid * (teacher - probe.detach()).abs().gt(1e-9).to(grad.dtype)
        miss_before = grad.new_zeros(())
        miss_after = grad.new_zeros(())
        active_count = grad.new_zeros(())
        for hi, hand in enumerate(HANDS):
            hit = strike.hit_index[:, hi].clamp(1, H)
            q_after = teacher.gather(
                1, (hit - 1).view(B, 1, 1).expand(B, 1, D),
            ).squeeze(1)
            local = _dls_contact_local(cost, strike, hi)
            p_after = _dls_contact_position(cost.fk, q_after, hand, local)
            act = (strike.active[:, hi] > 0.5).to(grad.dtype)
            aim = strike.p_star[:, hi].float()
            miss_before += (hit_positions[hi] - aim).norm(dim=1).mul(act).sum()
            miss_after += (p_after - aim).norm(dim=1).mul(act).sum()
            active_count += act.sum()
        stats = {
            "dls_terminal_rows": grad.new_tensor(float(B)),
            "dls_terminal_active_hands": active_count,
            "dls_terminal_err_before_m": miss_before / active_count.clamp_min(1.0),
            "dls_terminal_err_after_m": miss_after / active_count.clamp_min(1.0),
            "dls_terminal_delta_abs_rad": (teacher - probe.detach()).abs().mean(),
            "curvature_teacher_accept_frac": accepted.float().mean(),
            "curvature_teacher_cost_improvement": improvement.mean(),
        }
    return teacher.to(traj.dtype), tie_mask.to(traj.dtype), stats


def curvature_precondition_vjps(
    grads: torch.Tensor,
    traj: torch.Tensor,
    q0: torch.Tensor,
    strike: StrikeContext,
    lengths: torch.Tensor,
    weights: dict,
    cost: FusedCost,
    cfg: Phase2Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the analytic positive trajectory metric to one or more VJPs.

    ``grads`` is [batch, waypoint, joint, rhs].  Its first RHS is the full
    actor-trajectory gradient; a second RHS can be the torque contribution
    measured by the torque-gradient budget.  Each RHS is independently
    norm-matched per row, so this changes directions rather than silently
    multiplying the actor learning rate by a spline-Hessian scale.
    """
    B, H, D, K = grads.shape
    device = grads.device
    with torch.no_grad():
        group = torch.as_tensor(cost.spline._group_np, device=device)
        per_len = lengths.clamp(1, H).index_select(1, group)
        valid = (
            torch.arange(H, device=device).view(1, H, 1)
            < per_len.unsqueeze(1)
        ).to(grads.dtype)
        n_valid = per_len.sum(dim=1).clamp_min(1).to(grads.dtype)
        if cost.cfg.reduction != "mean":
            n_valid = torch.ones_like(n_valid)
        stiffness = _spline_smoothness_hessians(cost).index_select(
            0, per_len.reshape(-1)
        ).reshape(B, D, H, H)
        smooth_weight = torch.as_tensor(
            weights.get("smooth", 1.0), device=device, dtype=grads.dtype,
        ).reshape(-1, 1, 1, 1)
        stiffness = stiffness * smooth_weight / n_valid.view(B, 1, 1, 1)
        eye_h = torch.eye(H, device=device, dtype=grads.dtype)
        chol = torch.linalg.cholesky(
            (stiffness + float(cfg.curvature_damping) * eye_h)
            .reshape(B * D, H, H)
        )

        jac = grads.new_zeros(B, 12, D, H)
        root_w = grads.new_zeros(B, 12)
        for hi, hand in enumerate(HANDS):
            hit = strike.hit_index[:, hi].clamp(1, H)
            active = (strike.active[:, hi] > 0.5).to(grads.dtype)
            at_hit = traj.gather(
                1, (hit - 1).view(B, 1, 1).expand(B, 1, D),
            ).squeeze(1)
            prev_all = torch.cat([q0.float().unsqueeze(1), traj[:, :-1]], dim=1)
            at_prev = prev_all.gather(
                1, (hit - 1).view(B, 1, 1).expand(B, 1, D),
            ).squeeze(1)
            local = _dls_contact_local(cost, strike, hi)
            _pos_hit, j_hit = _dls_contact_position_jacobian(
                cost.fk, at_hit, hand, local,
            )
            _pos_prev, j_prev = _dls_contact_position_jacobian(
                cost.fk, at_prev, hand, local,
            )
            j_hit = j_hit * active.view(B, 1, 1)
            j_prev = j_prev * (active * (hit > 1).to(grads.dtype)).view(B, 1, 1)
            hidx = (hit - 1).view(B, 1, 1, 1).expand(B, 3, D, 1)
            pidx = (hit - 2).clamp_min(0).view(B, 1, 1, 1).expand(B, 3, D, 1)
            jpos = jac[:, hi * 6:hi * 6 + 3]
            jvel = jac[:, hi * 6 + 3:hi * 6 + 6]
            jpos.scatter_add_(3, hidx, j_hit.unsqueeze(-1))
            jvel.scatter_add_(3, hidx, (j_hit / cost.dt).unsqueeze(-1))
            jvel.scatter_add_(3, pidx, (-j_prev / cost.dt).unsqueeze(-1))
            root_w[:, hi * 6:hi * 6 + 3] = (
                float(cfg.curvature_position_weight) ** 0.5 * active.unsqueeze(1)
            )
            root_w[:, hi * 6 + 3:hi * 6 + 6] = (
                float(cfg.curvature_velocity_weight) ** 0.5 * active.unsqueeze(1)
            )

        rhs = torch.cat([
            grads.permute(0, 2, 1, 3),
            jac.permute(0, 2, 3, 1),
        ], dim=-1).reshape(B * D, H, K + 12)
        solved = torch.cholesky_solve(rhs, chol).reshape(B, D, H, K + 12)
        u0, u_j = solved[..., :K], solved[..., K:]
        ju0 = torch.einsum("brdh,bdhk->brk", jac, u0)
        juj = torch.einsum("brdh,bdhs->brs", jac, u_j)
        small = (
            torch.eye(12, device=device, dtype=grads.dtype).unsqueeze(0)
            + root_w.unsqueeze(2) * juj * root_w.unsqueeze(1)
        )
        correction = torch.linalg.solve(
            small, root_w.unsqueeze(-1) * ju0,
        ) * root_w.unsqueeze(-1)
        result = (
            u0 - torch.einsum("bdhs,bsk->bdhk", u_j, correction)
        ).permute(0, 2, 1, 3) * valid.unsqueeze(-1)

        original = grads * valid.unsqueeze(-1)
        original_norm = torch.linalg.vector_norm(original.reshape(B, -1, K), dim=1)
        result_norm = torch.linalg.vector_norm(result.reshape(B, -1, K), dim=1)
        usable = (
            torch.isfinite(result).all(dim=(1, 2))
            & torch.isfinite(result_norm) & (result_norm > 1e-20)
            & torch.isfinite(original_norm)
        )
        ratio = original_norm / result_norm.clamp_min(1e-20)
        matched = result * ratio.view(B, 1, 1, K)
        mixed = (
            (1.0 - float(cfg.curvature_direct_mix)) * original
            + float(cfg.curvature_direct_mix) * matched
        )
        output = torch.where(usable.view(B, 1, 1, K), mixed, original)
        dot = (original[..., 0] * output[..., 0]).sum(dim=(1, 2))
        cosine = dot / (
            original_norm[:, 0]
            * torch.linalg.vector_norm(output[..., 0].reshape(B, -1), dim=1)
        ).clamp_min(1e-20)
        stats = {
            "curvature_direct_rows": grads.new_tensor(float(B)),
            "curvature_direct_cosine": cosine.mean(),
            "curvature_direct_fallback_frac": (~usable[:, 0]).float().mean(),
            "curvature_direct_grad_norm": original_norm[:, 0].mean(),
        }
    return output, stats


def curvature_direct_vjp_correction(
    total: torch.Tensor,
    traj: torch.Tensor,
    q0: torch.Tensor,
    strike: StrikeContext,
    lengths: torch.Tensor,
    weights: dict,
    cost: FusedCost,
    cfg: Phase2Config,
    *,
    torque_term_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Return the VJP adjustment that replaces, rather than adds to, g_x.

    The scalar forward loss is unchanged.  First take its exact VJP into the
    actor trajectory, then calculate ``P^-1 g_x - g_x``.  A later backward of
    ``traj`` with that delta replaces the ordinary trajectory VJP.  When the
    torque budget refreshes, return an equal-value proxy whose gradient is
    the separately preconditioned torque VJP, so the cap measures the actual
    direction being sent into the model.
    """
    if not 0.0 < float(cfg.curvature_direct_frac) <= 1.0:
        raise ValueError("curvature_direct_frac must be positive")
    B = int(traj.shape[0])
    rows = _dls_rows(B, cfg.curvature_direct_frac, traj.device)
    full_grad = torch.autograd.grad(total, traj, retain_graph=True)[0].detach()
    grad_list = [full_grad.index_select(0, rows).float()]
    if torque_term_mean is not None and cfg.w_cost > 0.0:
        torque_grad = torch.autograd.grad(
            float(cfg.w_cost) * torque_term_mean, traj, retain_graph=True,
        )[0].detach()
        grad_list.append(torque_grad.index_select(0, rows).float())
    preconditioned, stats = curvature_precondition_vjps(
        torch.stack(grad_list, dim=-1),
        traj.detach().index_select(0, rows).float(),
        q0.detach().index_select(0, rows).float(),
        _strike_rows(strike, rows), lengths.index_select(0, rows),
        _index_batch_weights(weights, rows, B), cost, cfg,
    )
    delta = torch.zeros_like(traj)
    delta.index_copy_(
        0, rows,
        (preconditioned[..., 0] - grad_list[0]).to(traj.dtype),
    )
    torque_proxy = torque_term_mean
    if len(grad_list) == 2:
        torque_delta = torch.zeros_like(traj)
        torque_delta.index_copy_(
            0, rows,
            (preconditioned[..., 1] - grad_list[1]).to(traj.dtype),
        )
        torque_proxy = torque_term_mean + (
            (traj - traj.detach()) * torque_delta
        ).sum() / float(cfg.w_cost)
    return delta.detach(), torque_proxy, stats


def dls_terminal_lookahead_loss(
    traj: torch.Tensor,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike: StrikeContext,
    lengths: torch.Tensor,
    weights: dict,
    cost: FusedCost,
    cfg: Phase2Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Full-cost DLS lookahead plus a detached-teacher imitation tie."""
    B = int(traj.shape[0])
    rows = _dls_rows(
        B, cfg.dls_terminal_frac, traj.device,
        eligible=(strike.active > 0.5).any(dim=1),
    )
    if rows.numel() == 0:
        zero = traj.sum() * 0.0
        stats = {
            "dls_terminal_rows": zero.detach(),
            "dls_terminal_active_hands": zero.detach(),
            "dls_terminal_err_before_m": zero.detach(),
            "dls_terminal_err_after_m": zero.detach(),
            "dls_terminal_delta_abs_rad": zero.detach(),
            "dls_terminal_cost": zero.detach(),
            "dls_terminal_tie": zero.detach(),
        }
        return zero, zero, zero, stats
    traj_rows = traj.index_select(0, rows)
    strike_rows = _strike_rows(strike, rows)
    if cfg.terminal_lookahead_mode == "curvature":
        teacher, tie_mask, stats = curvature_terminal_teacher(
            traj_rows, q0.index_select(0, rows), v0.index_select(0, rows),
            strike_rows, lengths.index_select(0, rows),
            _index_batch_weights(weights, rows, B), cost, cfg,
        )
    else:
        teacher, tie_mask, stats = dls_terminal_teacher(
            traj_rows, strike_rows, cost,
            steps=cfg.dls_terminal_steps,
            window=cfg.dls_terminal_window,
            damping=cfg.dls_terminal_damping,
            max_step_rad=cfg.dls_terminal_max_step_rad,
        )
    # Numerically this is the DLS-corrected trajectory; analytically it has an
    # identity Jacobian w.r.t. the actor trajectory (straight-through).
    corrected = traj_rows + (teacher - traj_rows.detach())
    ct = cost.forward(
        q0.index_select(0, rows), corrected,
        v0.index_select(0, rows), strike_rows,
        lengths=lengths.index_select(0, rows),
        weights=_index_batch_weights(weights, rows, B),
    )
    tie = _masked_huber(traj_rows, teacher, tie_mask, 0.02)[0]
    torque = ct.terms["torque"].mean()
    stats["dls_terminal_cost"] = ct.total.mean().detach()
    stats["dls_terminal_tie"] = tie.detach()
    return ct.total.mean(), tie, torque, stats


def _copy_verified_timing_rows(
    batch: dict,
    proposal: dict,
    dst_rows: torch.Tensor,
    src_rows: torch.Tensor,
) -> None:
    """Commit only timing-dependent fields from exactly verified proposals."""
    for key in ("hit", "seed", "init"):
        batch[key].index_copy_(0, dst_rows, proposal[key].index_select(0, src_rows))
    for name in batch["masks"].__dataclass_fields__:
        dst = getattr(batch["masks"], name)
        src = getattr(proposal["masks"], name)
        dst.index_copy_(0, dst_rows, src.index_select(0, src_rows))
    batch["context"].goal_tth.index_copy_(
        0, dst_rows, proposal["context"].goal_tth.index_select(0, src_rows),
    )


def _refresh_timing_line_search_replay(record, batch: dict) -> None:
    """Freeze critic-selected timings into an open line-search replay record.

    The record is opened before yaw jitter and model-derived initializer
    rewrites, while timing assist runs after those rewrites. Timing assist can
    replace ``hit``, masks, goal TTH, ``seed`` and ``init``. The first three
    remain on the shared finalized batch; refresh the mutable seed/init
    snapshot so every SaLSa probe starts from the same selected collocations.

    Fresh timing rows replay from their raw retimed seed because each probe
    reapplies the parameter-dependent waist/arm initializer. MPC rows were not
    retimed and retain their original shifted-plan initializer.
    """
    if record is None:
        return
    seed = batch.get("seed")
    masks = batch.get("masks")
    if not torch.is_tensor(seed) or masks is None:
        raise ReplayIncompleteError(
            "timing line-search replay requires seed and decoder masks"
        )
    record.mutable_pre["seed"] = seed.detach().clone()
    raw_init = (seed * masks.elem_mask).detach()
    old_init = record.mutable_pre.get("init")
    if old_init is None:
        record.mutable_pre["init"] = raw_init.clone()
        return
    mpc_init = batch.get("mpc_init")
    if mpc_init is None:
        record.mutable_pre["init"] = raw_init.clone()
        return
    fresh = ~mpc_init.detach().to(device=raw_init.device, dtype=torch.bool)
    record.mutable_pre["init"] = torch.where(
        fresh.view(-1, 1, 1), raw_init, old_init.to(raw_init.device),
    ).clone()


@torch.no_grad()
def _timing_assist_batch(
    *,
    batch: dict,
    encoding,
    strike,
    model,
    cost: FusedCost,
    weights: dict,
    pool: Phase2Pool,
    critic: ViolationCriticTrainer | OnlineDistributionalCriticTrainer,
    cfg: Phase2Config,
    rows: torch.Tensor | None = None,
    selection: str = "collar",
    metric_prefix: str = "timing_assist",
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, dict[str, float]]:
    """Select and directly admit critic-proposed timing pairs.

    There is deliberately no separate actor/physics admission pass.  The
    selected rows enter the ordinary actor batch, whose single normal cost
    evaluation supplies both their actor gradient and their exact critic
    labels.  Dedicated horizon-anchor rows remain in the batch independently.
    """
    tc = cfg.timing_critic
    b = int(batch["q0"].shape[0])
    if selection not in {"collar", "boundary", "interior"}:
        raise ValueError(f"unknown timing proposal selection {selection!r}")
    eligible = batch["active"].any(dim=1).nonzero(as_tuple=True)[0]
    if rows is None:
        fraction = 1.0 if tc.mode == "control" else float(tc.assist_fraction)
        n = min(int(math.ceil(fraction * b)), int(eligible.numel()))
        perm = torch.randperm(
            int(eligible.numel()), generator=critic.generator,
        )[:n]
        rows = eligible.index_select(0, perm.to(eligible.device))
    else:
        rows = rows.to(device=batch["q0"].device, dtype=torch.long).reshape(-1)
        if int(rows.numel()) and not bool(batch["active"].index_select(0, rows).any(dim=1).all()):
            raise ValueError("explicit timing proposal rows must have an active hand")
        n = int(rows.numel())
    if n <= 0:
        return None, {f"{metric_prefix}_proposed": 0.0}

    active = batch["active"].index_select(0, rows)
    directions = sample_directions(
        active, generator=critic.generator, axis_probability=0.10,
    )
    pairs, _alpha = directional_pairs(
        active, directions, horizon=int(batch["bucket"]),
        step=int(tc.coarse_step_knots),
    )
    terminal_unlocked = bool(getattr(critic, "terminal_band_unlocked", True))
    latch_floor_wp = 1 if terminal_unlocked else int(tc.short_floor_wp)
    boundary_floor_wp = int(getattr(critic, "boundary_floor_wp", 1))
    terminal_floor_wp = max(latch_floor_wp, boundary_floor_wp)
    if terminal_floor_wp > 1:
        pairs = torch.where(
            active[:, None, :].bool(),
            pairs.clamp_min(terminal_floor_wp),
            torch.full_like(pairs, -1),
        )
    k = int(pairs.shape[1])
    repeated_rows = rows.repeat_interleave(k)
    candidate = _index_batch(batch, repeated_rows)
    retime_batch_timing(candidate, pairs.reshape(-1, 2), pool)

    features = build_critic_features(
        q0=candidate["q0"], qd0=candidate["v0"],
        goal_pos=candidate["context"].goal_pos,
        goal_tth=candidate["context"].goal_tth,
    )
    if hasattr(critic, "manageable"):
        predicted_ok = critic.manageable(features).reshape(n, k)
    else:
        scores = critic.conservative_score(features).reshape(n, k)
        predicted_ok = scores <= float(tc.decision_threshold)
    _selected, last_pass, first_fail = first_exit_indices(predicted_ok)

    if selection == "collar":
        collar_prob = torch.tensor(
            [tc.collar_inside_probability, tc.collar_boundary_probability,
             tc.collar_outside_probability], dtype=torch.float32,
        )
        collar = torch.multinomial(
            collar_prob, n, replacement=True, generator=critic.generator,
        ).to(last_pass.device) - 1
        selected = (last_pass + collar).clamp(0, k - 1)
    else:
        # The final predicted pass on a random positive ray is a stochastic
        # directional sample of the two-hand Pareto frontier.
        selected = last_pass
    anchor_ok = predicted_ok[:, 0]
    # An uncertain anchor must not suppress exploration: dedicated anchor rows
    # already preserve the known long-horizon task.  Advance one ray cell when
    # no connected predicted-manageable component exists, then let the normal
    # actor pass provide the exact label for that informative point.
    explore = torch.full_like(selected, min(1, k - 1))
    selected = torch.where(anchor_ok, selected, explore)
    selected = selected.clamp_min(min(1, k - 1))
    selected_pred_ok = predicted_ok.gather(1, selected.unsqueeze(1))[:, 0]
    if selection == "interior":
        boundary_pair = pairs[
            torch.arange(n, device=selected.device), selected,
        ]
        # Component-wise slower than the conservative frontier point: this is
        # the rectangle implied by the monotonic task-time prior.
        width = (int(batch["bucket"]) - boundary_pair).clamp_min(0)
        u = torch.rand(
            boundary_pair.shape, generator=critic.generator,
            device="cpu", dtype=torch.float32,
        ).to(boundary_pair.device)
        interior_pair = boundary_pair + torch.floor(
            u * (width + 1).to(u.dtype)
        ).long()
        interior_pair = torch.where(
            active.bool(), interior_pair, torch.full_like(interior_pair, -1),
        )
        proposal_pair = interior_pair
        proposal = _index_batch(batch, rows)
        retime_batch_timing(proposal, interior_pair, pool)
    else:
        flat = torch.arange(n, device=selected.device) * k + selected
        proposal_pair = pairs[
            torch.arange(n, device=selected.device), selected,
        ]
        proposal = _index_batch(candidate, flat)

    # Geometric seed correction is deliberately deferred until every timing
    # row has been finalized, then applied once to the complete actor batch.
    src = torch.arange(n, device=rows.device)
    _copy_verified_timing_rows(batch, proposal, rows, src)

    detail_prefix = (
        "timing" if metric_prefix == "timing_assist" else metric_prefix
    )
    advanced = (
        (proposal_pair < pairs[:, 0]) & active.bool()
    ).any(dim=1)
    # Carry selection metadata through the shuffled primary batch.  The normal
    # actor pass joins it with exact feasibility below, without provenance in
    # the critic's bounded replay representation.
    if "timing_selected_pred_manageable" not in batch:
        batch["timing_selected_pred_manageable"] = torch.zeros(
            b, device=rows.device, dtype=torch.bool,
        )
        batch["timing_advanced"] = torch.zeros(
            b, device=rows.device, dtype=torch.bool,
        )
    batch["timing_selected_pred_manageable"].index_copy_(
        0, rows, selected_pred_ok,
    )
    batch["timing_advanced"].index_copy_(0, rows, advanced)
    active_tth = (
        proposal_pair[active.bool()].float() * float(pool.cfg.dt)
    )

    def _mean_or_zero(values: torch.Tensor) -> float:
        return float(values.mean()) if int(values.numel()) else 0.0

    def _median_or_zero(values: torch.Tensor) -> float:
        return float(values.median()) if int(values.numel()) else 0.0

    return None, {
        f"{metric_prefix}_proposed": float(n),
        f"{detail_prefix}_selected_pred_manageable": float(selected_pred_ok.sum()),
        f"{detail_prefix}_advanced": float(advanced.sum()),
        f"{detail_prefix}_selected_tth_mean_s": _mean_or_zero(active_tth),
        f"{detail_prefix}_selected_tth_p50_s": _median_or_zero(active_tth),
        f"{detail_prefix}_anchor_pred_fail": float((~anchor_ok).sum()),
        f"{detail_prefix}_ray_all_pred_pass": float((first_fail == k).sum()),
        f"{detail_prefix}_terminal_floor_wp": float(terminal_floor_wp),
        f"{detail_prefix}_terminal_unlocked": float(terminal_unlocked),
    }


@torch.no_grad()
def _timing_interiors_from_group_frontier(
    *,
    batch: dict,
    boundary_rows: torch.Tensor,
    interior_rows: torch.Tensor,
    pool: Phase2Pool,
    generator: torch.Generator,
    metric_prefix: str = "timing_interior",
) -> dict[str, float]:
    """Derive all interior rows from the already-selected group frontier.

    Expanded timing rows belonging to one physical task carry the same group
    id.  One critic ray search therefore determines that task's frontier; its
    four component-wise-looser samples need no additional critic evaluation.
    """
    n = int(interior_rows.numel())
    if n == 0:
        return {f"{metric_prefix}_proposed": 0.0}
    groups = batch.get("timing_primary_group")
    if not torch.is_tensor(groups):
        raise ValueError("grouped timing interiors require timing_primary_group")
    boundary_group = groups.index_select(0, boundary_rows)
    interior_group = groups.index_select(0, interior_rows)
    if bool((boundary_group < 0).any()) or bool((interior_group < 0).any()):
        raise ValueError("grouped timing rows must have nonnegative group ids")
    if int(boundary_group.unique().numel()) != int(boundary_group.numel()):
        raise ValueError("grouped timing reuse requires one boundary per task")
    lookup = torch.full(
        (int(groups.max()) + 1,), -1,
        device=groups.device, dtype=torch.long,
    )
    lookup.index_copy_(0, boundary_group, boundary_rows)
    source_rows = lookup.index_select(0, interior_group)
    if bool((source_rows < 0).any()):
        raise ValueError("an interior timing row has no group boundary")

    active = batch["active"].index_select(0, interior_rows)
    boundary_pair = batch["hit"].index_select(0, source_rows)
    horizon = int(batch["bucket"])
    width = (horizon - boundary_pair).clamp_min(0)
    u = torch.rand(
        boundary_pair.shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(boundary_pair.device)
    interior_pair = boundary_pair + torch.floor(
        u * (width + 1).to(u.dtype)
    ).long()
    interior_pair = torch.where(
        active.bool(), interior_pair, torch.full_like(interior_pair, -1),
    )
    proposal = _index_batch(batch, interior_rows)
    retime_batch_timing(proposal, interior_pair, pool)
    src = torch.arange(n, device=interior_rows.device)
    _copy_verified_timing_rows(batch, proposal, interior_rows, src)

    pred = batch["timing_selected_pred_manageable"].index_select(
        0, source_rows,
    )
    advanced = (
        (interior_pair < horizon) & active.bool()
    ).any(dim=1)
    batch["timing_selected_pred_manageable"].index_copy_(
        0, interior_rows, pred,
    )
    batch["timing_advanced"].index_copy_(0, interior_rows, advanced)
    active_tth = interior_pair[active.bool()].float() * float(pool.cfg.dt)
    mean_tth = float(active_tth.mean()) if int(active_tth.numel()) else 0.0
    median_tth = float(active_tth.median()) if int(active_tth.numel()) else 0.0
    return {
        f"{metric_prefix}_proposed": float(n),
        f"{metric_prefix}_selected_pred_manageable": float(pred.sum()),
        f"{metric_prefix}_advanced": float(advanced.sum()),
        f"{metric_prefix}_selected_tth_mean_s": mean_tth,
        f"{metric_prefix}_selected_tth_p50_s": median_tth,
        f"{metric_prefix}_reused_frontier": float(n),
    }


def _batch_from_cloud(
    pool: Phase2Pool,
    cloud: FrontierCloud,
    bucket: int,
    loc: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise: bool = False,
) -> dict:
    """Train-batch dict from cloud rows (same layout as ``Phase2Pool.batch``)."""
    from .tokenizer import ContextInputs

    H = int(bucket)
    cols = cloud.gather(H, loc)
    n = int(loc.numel())
    q0, v0, seed = cols["q0"], cols["qd0"], cols["init"]
    hit, active = cols["hit"], cols["active"]
    masks = pool.masks_for(hit, active, H)
    init = pool.noised_init(seed, H, generator=generator) if noise else seed.clone()
    ctx = ContextInputs(
        **{f: cols[f] for f in CONTEXT_FIELDS if f in cols}
    )
    heads = ctx.drum_pos - pool._pelvis
    wref = None
    if "waist_ref" in cols:
        wref = cols["waist_ref"]
    elif getattr(pool, "waist_ref_from_seed", True):
        yaw_i = tuple(pool.cfg.joint_names).index("waist_yaw_joint")
        wref = seed[:, :, yaw_i]
    if "start_drum" not in cols:
        cols["start_drum"] = cols["drum"]
    pref = cols["waist_pr_ref"] if "waist_pr_ref" in cols else None
    sctx = strike_context_from_arrays(
        cols, torch.arange(n, device=loc.device),
        drum_head_centers=heads, drum_radii=ctx.drum_radius,
        waist_ref=wref, n_shaft=pool.n_shaft,
        waist_pr_ref=pref,
    )
    kind = cols["kind"] if "kind" in cols else torch.zeros(
        n, device=loc.device, dtype=torch.long,
    )
    return {
        "bucket": H,
        "q0": q0,
        "v0": v0,
        "state": torch.cat([q0, v0], dim=-1),
        "init": init * masks.elem_mask,
        "seed": seed,
        "masks": masks,
        "context": ctx,
        "strike": sctx,
        "hit": hit,
        "active": active,
        "kind": kind,
        "cid": cols["cid"],
        "mpc_init": torch.zeros(n, device=loc.device, dtype=torch.bool),
    }


def _draw_frontier_k(
    pool: Phase2Pool,
    cloud: FrontierCloud,
    bucket: int,
    k: int,
    *,
    generator: torch.Generator,
    inv_frac: float = 0.5,
    noise: bool = True,
) -> dict | None:
    """``k`` cloud collocations, or None if the cloud is empty."""
    among = cloud.scored_loc(int(bucket))
    k = min(int(k), int(among.numel()))
    if k <= 0:
        return None
    loc = cloud.sample_idx(
        int(bucket), k, generator=generator, among=among,
        inv_frac=float(inv_frac),
    )
    return _batch_from_cloud(
        pool, cloud, int(bucket), loc, generator=generator, noise=noise,
    )


def _draw_frontier_train_batch(
    pool: Phase2Pool,
    cloud: FrontierCloud,
    bucket: int,
    bs: int,
    *,
    k: int,
    generator: torch.Generator,
    inv_frac: float = 0.5,
) -> dict | None:
    """Minibatch of cloud collocations, or None if the cloud cannot fill ``bs``."""
    take = min(int(k), int(bs))
    batch = _draw_frontier_k(
        pool, cloud, bucket, take, generator=generator,
        inv_frac=float(inv_frac), noise=True,
    )
    if batch is None or int(batch["q0"].shape[0]) < int(bs):
        return None
    return batch


def _splice_frontier_batch(
    batch: dict,
    pool: Phase2Pool,
    cloud: FrontierCloud,
    bucket: int,
    bs: int,
    *,
    k: int,
    generator: torch.Generator,
    inv_frac: float = 0.5,
    prefer_children: bool = False,
) -> dict:
    """Replace the last ``k`` rows with a cloud sample. Size stays ``bs``.

    ``prefer_children``: draw from ball children when that set can fill
    ``k`` (idle/far-cloud replay splice — seed already appears on fresh
    idle/far draws).
    """
    k = min(int(k), int(bs), cloud.size(int(bucket)))
    if k <= 0:
        if "cid" not in batch:
            batch["cid"] = torch.zeros(
                int(bs), device=batch["q0"].device, dtype=torch.long,
            )
        return batch
    among = cloud.scored_loc(int(bucket))
    if prefer_children and among.numel() > 0:
        b = int(bucket)
        fp = cloud._buf[b]["from_pool"].index_select(0, among)
        child = among[~fp]
        if int(child.numel()) >= int(k):
            among = child
    k = min(k, int(among.numel()))
    if k <= 0:
        if "cid" not in batch:
            batch["cid"] = torch.zeros(
                int(bs), device=batch["q0"].device, dtype=torch.long,
            )
        return batch
    loc = cloud.sample_idx(
        int(bucket), k, generator=generator, among=among,
        inv_frac=float(inv_frac),
    )
    cloud_part = _batch_from_cloud(
        pool, cloud, int(bucket), loc, generator=generator, noise=False,
    )
    if k >= int(bs):
        return cloud_part
    keep = torch.arange(int(bs) - k, device=batch["q0"].device)
    kept = _index_batch(batch, keep)
    if "cid" not in kept:
        kept["cid"] = torch.zeros(
            int(keep.numel()), device=batch["q0"].device, dtype=torch.long,
        )
    return _cat_phase2_batches([kept, cloud_part])


def _frontier_ema_from_traj(
    cloud: FrontierCloud,
    cost: FusedCost,
    batch: dict,
    traj: torch.Tensor,
    bucket: int,
    *,
    viol_cap: float = 8.0,
) -> None:
    """Per-sample hit+collision(+torque) EMA on rows that carry a cloud ``cid``."""
    cid = batch.get("cid")
    if not torch.is_tensor(cid) or not bool((cid > 0).any()):
        return
    idx = (cid > 0).nonzero(as_tuple=True)[0]
    sub = _index_batch(batch, idx)
    hit_eff = effective_hits(sub["hit"], sub["active"], int(bucket))
    lengths = group_lengths(hit_eff)
    strike = replace(sub["strike"], hit_index=hit_eff)
    with torch.no_grad():
        ct = cost.forward(
            sub["q0"], traj.index_select(0, idx), sub["v0"], strike,
            lengths=lengths,
        )
        ok = _proficiency_ok_from_cost(ct, viol_cap=viol_cap)
        y = ok.to(dtype=cloud.dtype)
    cloud.ema_update(int(bucket), cid.index_select(0, idx), y)


def _replay_ema_from_traj(
    replay: ReplayPool,
    cost: FusedCost,
    batch: dict,
    traj: torch.Tensor,
    bucket: int,
    *,
    viol_cap: float = 8.0,
) -> None:
    """Per-sample proficiency EMA on MPC replay ring slots."""
    loc = batch.get("replay_loc")
    if not torch.is_tensor(loc) or not bool((loc >= 0).any()):
        return
    idx = (loc >= 0).nonzero(as_tuple=True)[0]
    ring = loc.index_select(0, idx)
    sub = _index_batch(batch, idx)
    hit_eff = effective_hits(sub["hit"], sub["active"], int(bucket))
    lengths = group_lengths(hit_eff)
    strike = replace(sub["strike"], hit_index=hit_eff)
    with torch.no_grad():
        ct = cost.forward(
            sub["q0"], traj.index_select(0, idx), sub["v0"], strike,
            lengths=lengths,
        )
        ok = _proficiency_ok_from_cost(ct, viol_cap=viol_cap)
        y = ok.to(dtype=replay.dtype)
    replay.ema_update(int(bucket), ring, y)


def _ingest_frontier_harvest(
    cloud: FrontierCloud, bucket: int, cols: dict, *, include_miss: bool,
) -> int:
    """Ingest post-hit harvest rows (filtered, or attempted if include_miss)."""
    mark = cols.get("_post_hit_fresh")
    if include_miss:
        mark = cols.get("_post_hit_attempted", mark)
    if mark is None or not bool(mark.any()):
        return 0
    sel = mark.nonzero(as_tuple=True)[0]
    payload = {
        k: v.index_select(0, sel) for k, v in cols.items()
        if torch.is_tensor(v) and not str(k).startswith("_")
    }
    if "drum" not in payload or "q0" not in payload:
        return 0
    return int(cloud.ingest(int(bucket), payload))


def _frontier_steal_fresh(
    batch: dict,
    pool: Phase2Pool,
    idx: torch.Tensor,
    bucket: int,
    cloud: FrontierCloud,
    *,
    frac: float,
    generator: torch.Generator,
    noise: bool,
) -> dict:
    """Replace idle/far q0 from the cloud, then rebuild seed/init/state."""
    d = pool.data[int(bucket)]
    cols = {
        "q0": batch["q0"],
        "qd0": batch["v0"],
        "init": batch["seed"],
        "hit": batch["hit"],
        "active": batch["active"],
        "drum": d["drum"].index_select(0, idx),
    }
    yn = getattr(batch["context"], "yaw_now", None)
    if yn is not None:
        cols["yaw_now"] = yn
    cid = steal_q0(
        cols, cloud, int(bucket),
        frac=float(frac),
        kind=batch["kind"],
        idle_kind=KIND_RHYTHM_IDLE,
        far_kind=KIND_FAR_FETCHED,
        generator=generator,
    )
    batch["cid"] = cid
    stole = cid > 0
    if bool(stole.any()):
        src = stole.nonzero(as_tuple=True)[0]
        batch["state"] = torch.cat([batch["q0"], batch["v0"]], dim=-1)
        stolen_seed = batch["seed"].index_select(0, src)
        if noise:
            new_init = pool.noised_init(stolen_seed, bucket, generator=generator)
        else:
            new_init = stolen_seed.clone()
        em = batch["masks"].elem_mask.index_select(0, src)
        batch["init"].index_copy_(0, src, new_init * em)
    return batch


def _frontier_propose_ingest(
    cloud: FrontierCloud,
    bucket: int,
    k: int,
    *,
    sdf: SdfPenalty | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Propose K children on ``bucket`` and ingest graduates (SDF-gated if easy)."""
    child = cloud.propose(int(bucket), int(k), generator=generator)
    if child is None:
        return 0
    if sdf is not None and "q0" in child:
        clear = q0_collision_clear(
            sdf, child["q0"],
            drum_head_centers=child.get("drum_pos"),
            drum_radii=child.get("drum_radius"),
            exempt_drum=child.get("start_drum"),
        )
        if not bool(clear.any()):
            return 0
        if not bool(clear.all()):
            sel = clear.nonzero(as_tuple=True)[0]
            child = {
                kk: vv.index_select(0, sel) if torch.is_tensor(vv) else vv
                for kk, vv in child.items()
            }
    return int(cloud.ingest(int(bucket), child))


def _log_frontier_occupancy(cloud: FrontierCloud, buckets, *, log: bool) -> None:
    if not log:
        return
    for b in buckets:
        occ = cloud.occupancy_of(int(b))
        n_ch = int((occ > 0).sum())
        print(
            f"[frontier] bucket {int(b)}: n={cloud.size(int(b))} "
            f"occupied_charts={n_ch}/{int(occ.numel())}",
            flush=True,
        )


def _log_frontier_ingest(n_starts: int, n_slices: int, stride: int, *, log: bool) -> None:
    if not log:
        return
    print(
        f"[frontier] seed starts={int(n_starts)} traj_slices={int(n_slices)} "
        f"stride={int(stride)}",
        flush=True,
    )


def _setup_frontier_cloud(
    pool: Phase2Pool,
    cfg: Phase2Config,
    *,
    device,
    dtype,
    run_dir,
    log: bool,
) -> FrontierCloud:
    """Build the cloud and seed it from the trained run's pool (or resume)."""
    fcfg = FrontierConfig(
        enabled=True,
        proposals=int(cfg.frontier_proposals),
        chart_cap=int(cfg.frontier_chart_cap),
        fresh_frac=float(cfg.frontier_fresh_frac),
        spawn_frac=float(cfg.frontier_spawn_frac),
        sigma_q=float(cfg.frontier_sigma_q),
        sigma_qd=float(cfg.frontier_sigma_qd),
        sigma_tth=float(cfg.frontier_sigma_tth),
        ema_alpha=float(cfg.frontier_ema_alpha),
        ema_nmin=int(cfg.frontier_ema_nmin),
        ema_prof=float(cfg.frontier_ema_prof),
        landmark_k=int(cfg.frontier_landmark_k),
        landmark_tau=float(cfg.frontier_landmark_tau),
        propose_candidates=int(cfg.frontier_propose_candidates),
        hull_ema=float(cfg.frontier_hull_ema),
        train_inv_frac=float(cfg.frontier_train_inv_frac),
        train_inv_pool=int(cfg.frontier_train_inv_pool),
        include_miss=bool(cfg.frontier_include_miss),
        traj_stride=int(cfg.frontier_traj_stride),
        dt=float(pool.cfg.dt),
        buckets=tuple(pool.buckets),
    )
    cloud = FrontierCloud(
        cfg=fcfg, device=torch.device(device), dtype=dtype,
        q_lo=pool.q_lo, q_hi=pool.q_hi,
    )
    pt = Path(run_dir) / "frontier.pt"
    # Continuation starts from the trained run's pool (every static-pool row
    # plus mid-trajectory knots). A resume of a frontier stage loads
    # frontier.pt instead -- that file already contains the expanded cloud.
    if pt.exists():
        cloud.load_state_dict(
            torch.load(pt, map_location=device, weights_only=False)
        )
        cloud.q_lo, cloud.q_hi = pool.q_lo, pool.q_hi
        # A later continuation may raise chart_cap; grow the child ring
        # before the first ingest so wrap still uses the new depth.
        for b, d in cloud._buf.items():
            like = {k: v[:1] for k, v in d.items() if torch.is_tensor(v)}
            cloud._ensure_child_alloc(int(b), like)
        if log:
            print(f"[frontier] loaded {pt}", flush=True)
    else:
        n_starts = 0
        for b in pool.buckets:
            d = pool.data[b]
            if "drum" in d and "q0" in d:
                n_starts += int(d["q0"].shape[0])
        added = ingest_pool(cloud, pool)
        n_total = sum(int(v) for v in added.values())
        _log_frontier_ingest(
            n_starts, max(n_total - n_starts, 0), cfg.frontier_traj_stride,
            log=log,
        )
    _log_frontier_occupancy(cloud, pool.buckets, log=log)
    cloud.rebuild_landmarks()
    if log:
        print(
            f"[frontier] landmarks={cloud.n_landmarks()} "
            f"charts={len(cloud._landmarks)} "
            f"tau={cloud.cfg.landmark_tau} k={cloud.cfg.landmark_k}",
            flush=True,
        )
    return cloud


def _activate_frontier_runtime(
    *,
    cfg: Phase2Config,
    pool,
    roller,
    run_dir: Path,
    device,
    dtype,
    log: bool,
    rank: int,
    step: int,
    reason: str,
    emit,
) -> FrontierCloud:
    """Turn on frontier mid-run (probe threshold or monitor flag file)."""
    cfg.frontier = True
    if cfg.frontier_auto_full_weights:
        cfg.schedule = cfg.schedule.snap_full()
        cfg.suffix_warmup = 1
    cloud = _setup_frontier_cloud(
        pool, cfg, device=device, dtype=dtype, run_dir=run_dir,
        log=bool(log and rank == 0),
    )
    roller.frontier = cloud
    roller.frontier_spawn_frac = float(cfg.frontier_spawn_frac)
    flag = Path(run_dir) / ".frontier_activate"
    flag.unlink(missing_ok=True)
    done = Path(run_dir) / ".frontier_monitor.done"
    done.write_text(f"step={step} reason={reason}\n", encoding="utf-8")
    miss = None
    if reason.startswith("miss="):
        try:
            miss = float(reason.split("=", 1)[1].rstrip("mm"))
        except ValueError:
            miss = None
    emit({
        "event": "frontier_auto",
        "step": step,
        "reason": reason,
        "miss_mm": miss,
        "full_weights": bool(cfg.frontier_auto_full_weights),
    })
    if log and rank == 0:
        print(
            f"[frontier_auto] enabled @ step {step}: {reason}"
            + (f" (miss {miss:.1f} mm)" if miss is not None else "")
            + ("; full-weights snap" if cfg.frontier_auto_full_weights else ""),
            flush=True,
        )
    return cloud


def _maybe_auto_activate_frontier(
    *,
    cfg: Phase2Config,
    pool,
    roller,
    run_dir: Path,
    device,
    dtype,
    log: bool,
    rank: int,
    step: int,
    frontier_cloud: FrontierCloud | None,
    last_probe: dict | None,
    recent_train: list[dict],
    emit,
) -> FrontierCloud | None:
    if frontier_cloud is not None or cfg.frontier_auto_miss_mm <= 0:
        return frontier_cloud
    from amortized_opt.frontier_auto import (
        DEFAULT_VIOL_KEYS,
        frontier_ready,
    )

    flag = Path(run_dir) / ".frontier_activate"
    reason = ""
    if flag.is_file():
        reason = "flag_file"
    elif last_probe is not None:
        ready, why = frontier_ready(
            last_probe,
            recent_train,
            miss_mm=float(cfg.frontier_auto_miss_mm),
            viol_keys=DEFAULT_VIOL_KEYS,
            viol_cap=float(cfg.frontier_auto_viol_cap),
            viol_ratio=float(cfg.frontier_auto_viol_ratio),
            viol_window=int(cfg.frontier_auto_viol_window),
        )
        if not ready:
            return None
        reason = why
    else:
        return None
    return _activate_frontier_runtime(
        cfg=cfg,
        pool=pool,
        roller=roller,
        run_dir=run_dir,
        device=device,
        dtype=dtype,
        log=log,
        rank=rank,
        step=step,
        reason=reason,
        emit=emit,
    )


def _save_frontier_pt(run_dir, cloud: FrontierCloud | None) -> None:
    if cloud is None:
        return
    torch.save(cloud.state_dict(), Path(run_dir) / "frontier.pt")


def _maybe_frontier_cloud_plot(
    run_dir,
    cloud: FrontierCloud | None,
    model,
    costs,
    pool,
    *,
    step: int,
    start_step: int,
    cfg: Phase2Config,
) -> None:
    """3D tip-cloud dump at ~10k / ~20k extra steps (and at max_steps)."""
    if cloud is None:
        return
    extra = int(step) - int(start_step)
    every = max(int(cfg.ckpt_every), 1)
    due_10 = 10_000 <= extra < 10_000 + every
    due_20 = 20_000 <= extra < 20_000 + every
    due_end = int(step) == int(cfg.max_steps) and extra >= 8_000
    if not (due_10 or due_20 or due_end):
        return
    try:
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "plot_frontier_cloud.py"
        spec = importlib.util.spec_from_file_location("plot_frontier_cloud", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        out = Path(run_dir) / "frontier_viz"
        print(f"[frontier-plot] step={step} extra={extra} -> {out}", flush=True)
        mod.plot_from_live(
            cloud=cloud, model=model, costs=costs, pool=pool,
            out_dir=out, step=int(step),
        )
    except Exception as e:
        print(f"[frontier-plot] skipped: {type(e).__name__}: {e}", flush=True)


def _sample_fresh_batch(
    pool: Phase2Pool,
    bucket: int,
    bs: int,
    *,
    generator: torch.Generator,
    kind_fracs: dict[int, float] | None,
    live_post_hit: ReplayPool | None = None,
    short_wp: int = 0,
    short_boost: float = 1.0,
    noise: bool = True,
    frontier: FrontierCloud | None = None,
    frontier_fresh_frac: float = 0.0,
    frontier_idle_far: bool = False,
    frontier_inv_frac: float = 0.5,
) -> dict:
    """Fresh mix: static pool families plus on-policy post-hit inits.

    ``frontier_idle_far``: the non-post-hit remainder is drawn from the
    cloud (seed + ball children), not the frozen idle/far pool.
    """
    n_ph = 0
    live_n = 0 if live_post_hit is None else live_post_hit.size(bucket)
    if kind_fracs and live_n > 0 and KIND_POST_HIT in kind_fracs:
        kinds = sorted(kind_fracs)
        probs = torch.tensor(
            [float(kind_fracs[k]) for k in kinds],
            device=pool.device, dtype=torch.float32,
        )
        cat = torch.multinomial(
            probs, int(bs), replacement=True, generator=generator,
        )
        n_ph = min(int((cat == kinds.index(KIND_POST_HIT)).sum()), live_n)
    n_rest = int(bs) - n_ph
    parts: list[dict] = []
    if n_rest > 0:
        if frontier_idle_far:
            if frontier is None:
                raise RuntimeError(
                    "frontier_idle_far requires a FrontierCloud"
                )
            part = _draw_frontier_k(
                pool, frontier, bucket, n_rest,
                generator=generator, inv_frac=float(frontier_inv_frac),
                noise=noise,
            )
            if part is None or int(part["q0"].shape[0]) < n_rest:
                raise RuntimeError(
                    f"frontier idle/far mix needs >= {n_rest} scored "
                    f"collocations in bucket {int(bucket)} "
                    f"(have {0 if part is None else int(part['q0'].shape[0])})"
                )
            parts.append(part)
        else:
            # Once the live buffer is drawing the post-hit slice, drop that kind
            # from the static pool so we do not mix IK-distilled inits back in.
            # While the buffer is empty, leave POST_HIT in ``kind_fracs`` so a
            # legacy ``--post-hit-cache`` still samples.
            pool_fracs = kind_fracs
            if n_ph > 0 and kind_fracs:
                pool_fracs = {
                    k: v for k, v in kind_fracs.items() if k != KIND_POST_HIT
                } or None
            idx = pool.sample_idx(
                bucket, n_rest, generator=generator,
                short_wp=short_wp, short_boost=short_boost, kind_fracs=pool_fracs,
            )
            part = pool.batch(bucket, idx, generator=generator, noise=noise)
            part["mpc_init"] = torch.zeros(
                int(part["q0"].shape[0]), device=pool.device, dtype=torch.bool,
            )
            if frontier is not None and float(frontier_fresh_frac) > 0.0:
                part = _frontier_steal_fresh(
                    part, pool, idx, bucket, frontier,
                    frac=frontier_fresh_frac, generator=generator, noise=noise,
                )
            parts.append(part)
    if n_ph > 0:
        idx = live_post_hit.sample_idx(
            bucket, n_ph, generator=generator,
            short_wp=short_wp, short_boost=short_boost,
        )
        live_part = live_post_hit.batch(
            bucket, idx, generator=generator, noise=noise,
        )
        live_part["mpc_init"] = torch.ones(
            n_ph, device=pool.device, dtype=torch.bool,
        )
        if parts and "cid" in parts[0]:
            live_part["cid"] = torch.zeros(
                n_ph, device=pool.device, dtype=torch.long,
            )
        parts.append(live_part)
    return _cat_phase2_batches(parts)


def _timing_primary_layout(
    batch_size: int,
    cfg: ViolationCriticConfig,
) -> tuple[int, int, int, int, int]:
    """Return ``(base tasks, anchor, boundary, interior, MPC rows)``.

    The default is exactly 40 independent task states at batch 512, expanded
    to one permanent horizon anchor, one frontier and four interior timings
    apiece.  Smaller CPU batches scale the number of base tasks proportionally
    while preserving the six timing samples per task.
    """
    b = int(batch_size)
    per_task = (
        int(cfg.primary_anchor_per_task)
        + int(cfg.primary_boundary_per_task)
        + int(cfg.primary_interior_per_task)
    )
    base = max(1, int(round(
        b * int(cfg.primary_base_tasks_per_512) / 512.0
    )))
    base = min(base, b // per_task)
    anchor = base * int(cfg.primary_anchor_per_task)
    boundary = base * int(cfg.primary_boundary_per_task)
    interior = base * int(cfg.primary_interior_per_task)
    return base, anchor, boundary, interior, b - anchor - boundary - interior


def _timing_curriculum_group_count(
    *,
    base_tasks: int,
    curriculum_open: bool,
    curriculum_open_actor_step: int | None,
    actor_step: int,
    ramp_steps: int,
) -> int:
    """Return complete expanded timing-task groups for an enabled ramp.

    ``None`` opening provenance denotes an old checkpoint which was already
    open before progressive ramps existed; continuing it fully open is safer
    than silently changing its batch distribution on resume.
    """
    if not curriculum_open:
        return 0
    if int(ramp_steps) <= 0 or curriculum_open_actor_step is None:
        return int(base_tasks)
    elapsed = max(0, int(actor_step) - int(curriculum_open_actor_step))
    return min(
        int(base_tasks),
        int(base_tasks) * elapsed // int(ramp_steps),
    )


def _expand_timing_base_tasks(
    batch: dict,
    *,
    anchor_per_task: int,
    boundary_per_task: int,
    interior_per_task: int,
) -> dict:
    """Repeat each task over anchor/boundary/interior timing strata."""
    n = int(batch["q0"].shape[0])
    kinds = torch.tensor(
        [0] * int(anchor_per_task)
        + [2] * int(boundary_per_task)
        + [3] * int(interior_per_task),
        device=batch["q0"].device, dtype=torch.long,
    )
    per_task = int(kinds.numel())
    rows = torch.arange(n, device=batch["q0"].device).repeat_interleave(per_task)
    out = _index_batch(batch, rows)
    out["timing_primary_kind"] = kinds.repeat(n)
    out["timing_primary_group"] = torch.arange(
        n, device=batch["q0"].device, dtype=torch.long,
    ).repeat_interleave(per_task)
    return out


@torch.no_grad()
def _apply_terminal_initializer(
    core,
    batch: dict,
    *,
    pool: Phase2Pool | None = None,
    generator: torch.Generator | None = None,
    start_anchored_noise: bool = False,
    noise_ramp_steps: int = 5,
) -> bool:
    """Replace fresh-scenario seeds with the frozen terminal-head seed."""
    initializer = getattr(core, "terminal_initializer", None)
    if initializer is None:
        return False
    seed = core.initial_trajectory(batch["context"], batch["state"], batch["masks"])
    fresh = ~batch.get(
        "mpc_init",
        torch.zeros(seed.shape[0], device=seed.device, dtype=torch.bool),
    ).bool()
    if start_anchored_noise:
        if pool is None:
            raise ValueError("terminal init noise requires the Phase2Pool")
        H = int(seed.shape[1])
        noisy = pool.noised_init(seed, H, generator=generator)
        # The decoder trajectory starts at t=dt; measured q0/qd0 live at t=0
        # outside this tensor. Extending this cubic envelope to t=0 gives zero
        # perturbation and zero first derivative there, then reaches the usual
        # OU amplitude after ``noise_ramp_steps`` knots.
        ramp_n = max(int(noise_ramp_steps), 1)
        u = torch.arange(
            1, H + 1, device=seed.device, dtype=seed.dtype,
        ).clamp_max(ramp_n) / float(ramp_n)
        envelope = (u.square() * (3.0 - 2.0 * u)).view(1, H, 1)
        seed = seed + envelope * (noisy - seed)
        seed = seed * batch["masks"].elem_mask
    batch["init"] = torch.where(fresh[:, None, None], seed, batch["init"])
    return True


def _ls_probe_scalar_F(
    *,
    batch: dict,
    use_replay: bool,
    bucket: int,
    bs: int,
    encode_step,
    step_fns: dict,
    tc_fns: dict,
    sfx_fns: dict,
    model,
    costs: dict,
    pool: Phase2Pool,
    sim: BatchedPDSim,
    cfg: Phase2Config,
    base,
    yaw_i: int,
    pr_idx: list[int],
    pr_limit: float,
    waist_cfg,
    w: dict,
    sfx_ramp: float,
    include_bptt: bool,
    device,
    dtype,
    amp,
    rec,
    generator: torch.Generator | None = None,
    torque_grad_scale: float = 1.0,
    return_tensor: bool = False,
) -> float | torch.Tensor:
    """Re-encode + rewrite + complete ``loss+extra`` for one LS probe (no side effects).

    Restores pre-rewrite mutable fields from ``rec``, applies recorded yaw
    jitter / TC / suffix samples, and raises :class:`ReplayIncompleteError` if
    an enabled BPTT branch cannot be replayed (MVP gate).
    """
    if rec is None or not rec.mutable_pre:
        raise ReplayIncompleteError("LS probe requires rec.mutable_pre snapshot")
    restore_mutable_batch_fields(batch, rec.mutable_pre)

    samples = rec.samples
    hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
    strike = replace(batch["strike"], hit_index=hit_eff)
    lengths = group_lengths(hit_eff)

    if cfg.yaw_now_jitter > 0.0 and batch["context"].yaw_now is not None:
        if samples.yaw_now_jitter is None:
            raise ReplayIncompleteError(
                "yaw_now_jitter enabled but sample not recorded on replay record"
            )
        batch["context"].yaw_now = batch["context"].yaw_now + samples.yaw_now_jitter

    ctxm = (torch.autocast(device_type="cuda", dtype=amp) if amp is not None
            else torch.autocast(device_type="cuda", enabled=False))
    with ctxm:
        enc = encode_step(batch["context"])
        core = model.module if hasattr(model, "module") else model
        terminal_reseeded = _apply_terminal_initializer(
            core, batch, pool=pool, generator=generator,
            start_anchored_noise=bool(cfg.terminal_init_noise),
            noise_ramp_steps=int(cfg.terminal_init_noise_ramp_steps),
        )
        if cfg.waist_sp_from_head:
            sp = enc.waist_yaw_sp.float()
            strike = replace(
                strike, waist_yaw_ref=sp.unsqueeze(1).expand(-1, bucket)
            )
        if cfg.waist_pr_sp_from_head:
            pr = enc.waist_pr_sp.float()
            strike = replace(
                strike, waist_pr_ref=pr.unsqueeze(1).expand(-1, bucket, -1)
            )
        if enc.waist_pr_sp is not None:
            batch["init"] = seed_waist_pr_to_sp(
                batch["init"], batch["q0"], batch["v0"], enc.waist_pr_sp,
                joint_names=base.joint_names,
                dt=float(getattr(pool.cfg, "dt", 0.02)),
                ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
            )
        if not terminal_reseeded:
            fresh_active = batch["active"] & ~batch["mpc_init"].unsqueeze(-1)
            batch["init"] = correct_quintic_init_arms(
                costs[bucket].fk, batch["init"], batch["q0"], batch["hit"],
                batch["context"].goal_pos,
                active=fresh_active, steps=1,
                q_bounds=(pool.q_lo, pool.q_hi),
                joint_names=base.joint_names, qd0=batch["v0"],
                dt=float(getattr(pool.cfg, "dt", 0.02)),
            )
        step_out = step_fns[bucket](
            batch["init"], batch["state"], batch["q0"], batch["v0"],
            enc, batch["masks"], strike, lengths, w,
            torch.as_tensor(
                float(torque_grad_scale) - 1.0,
                device=device, dtype=torch.float32,
            ),
        )
        # LS forces grad_balance=""; step returns 4-tuple.
        loss, _metrics, _traj, gate = step_out[:4]

        loss_tc = torch.zeros((), device=device)
        if tc_fns:
            tc_rows = (
                _mpc_aux_rows(batch["mpc_init"], cfg.tc_frac)
                if cfg.voxel_pool else
                torch.arange(
                    max(int(cfg.tc_frac * bs), 1), device=batch["q0"].device,
                )
            )
            n_tc = int(tc_rows.numel())
        if tc_fns and n_tc:
            if samples.tc_kp is None or samples.tc_kd is None:
                raise ReplayIncompleteError(
                    "TC enabled but tc_kp/tc_kd missing on replay record"
                )
            masks2 = pool.masks_for(
                (batch["hit"].index_select(0, tc_rows) - 1).clamp_min(0),
                batch["active"].index_select(0, tc_rows), bucket,
            )
            kp, kd = samples.tc_kp, samples.tc_kd
            enc_sub = _index_encoded(enc, tc_rows)
            masks_sub = _index_batch({"bucket": bucket, "q0": batch["q0"],
                                      "masks": batch["masks"]}, tc_rows)["masks"]
            loss_tc, _n = tc_fns[bucket](
                batch["init"].index_select(0, tc_rows),
                batch["state"].index_select(0, tc_rows),
                batch["q0"].index_select(0, tc_rows),
                batch["v0"].index_select(0, tc_rows), enc_sub, masks_sub,
                masks2, kp, kd,
            )

        if include_bptt:
            raise ReplayIncompleteError(
                "BPTT included on baseline but probe BPTT payload replay is "
                "not wired yet (MVP: set w_bptt=0 / bptt_max_frac=0 under LS)"
            )
        loss_bptt = torch.zeros((), device=device)

        loss_sfx_tie = torch.zeros((), device=device)
        loss_mpc_tie = torch.zeros((), device=device)
        loss_waist_mpc_tie = torch.zeros((), device=device)
        waist_mpc_tie_grad = torch.zeros((), device=device)
        if cfg.w_mpc_init_tie > 0.0:
            loss_mpc_tie, _ = mpc_init_tie_loss(
                _traj, batch["seed"], batch["masks"].elem_mask,
                batch["mpc_init"],
            )
        if cfg.w_waist_yaw_mpc_tie > 0.0:
            loss_waist_mpc_tie, _, waist_mpc_tie_grad = (
                waist_yaw_mpc_init_tie_loss(
                    _traj, batch["seed"], batch["masks"].elem_mask,
                    batch["mpc_init"], yaw_index=yaw_i,
                )
            )
        if sfx_fns and sfx_ramp > 0.0:
            suffix_batch = samples.extras.get("suffix_replay_batch")
            if samples.suffix_offs is None or suffix_batch is None:
                raise ReplayIncompleteError(
                    "suffix replay active but frozen batch/offsets not recorded"
                )
            loss_sfx_tie, *_suffix_diagnostics = _suffix_distill_loss(
                suffix_batch, samples.suffix_offs,
                bucket=bucket, encode_step=encode_step,
                suffix_fn=sfx_fns[bucket], pool=pool,
                microbatch=int(cfg.suffix_microbatch),
            )
        extra = (
            cfg.w_tc * loss_tc + cfg.w_bptt * loss_bptt
            + cfg.w_mpc_init_tie * loss_mpc_tie
            + cfg.w_waist_yaw_mpc_tie * loss_waist_mpc_tie
            + sfx_ramp * cfg.w_suffix_tie * loss_sfx_tie
        )
        if (
            cfg.dls_terminal_frac > 0.0
            and (cfg.w_dls_terminal_cost > 0.0 or cfg.w_dls_terminal_tie > 0.0)
        ):
            dls_cost, dls_tie, dls_torque, _ = dls_terminal_lookahead_loss(
                _traj, batch["q0"], batch["v0"], strike, lengths, w,
                costs[bucket], cfg,
            )
            extra = extra + (
                cfg.w_dls_terminal_cost * dls_cost
                + cfg.w_dls_terminal_tie * dls_tie
            )
            # Cached torque-budget steps scale the DLS branch by the same
            # factor as the main cost.  Refresh steps use scale=1 and measure
            # the combined torque gradient in ``train_phase2``.
            if cfg.torque_grad_cap > 0.0:
                extra = extra + (
                    (float(torque_grad_scale) - 1.0)
                    * cfg.w_dls_terminal_cost * dls_torque
                )
        if cfg.w_waist_sp_tether > 0.0 and enc.waist_yaw_res is not None:
            res = enc.waist_yaw_res.float()
            extra = extra + cfg.w_waist_sp_tether * res.pow(2).mean()
        if cfg.w_waist_pr_sp_tether > 0.0 and enc.waist_pr_res is not None:
            pr = enc.waist_pr_res.float()
            extra = extra + cfg.w_waist_pr_sp_tether * pr.pow(2).mean()
        if cfg.w_waist_pr_sp_hold > 0.0 and enc.waist_pr_sp is not None:
            q_pr = batch["q0"][:, pr_idx].detach().float()
            d_pr = enc.waist_pr_sp.float() - q_pr
            extra = extra + cfg.w_waist_pr_sp_hold * d_pr.pow(2).mean()
        if cfg.w_waist_pr_sp_hold_band > 0.0 and enc.waist_pr_sp is not None:
            extra = extra + cfg.w_waist_pr_sp_hold_band * waist_pr_hold_band_loss(
                enc.waist_pr_sp.float(),
                batch["q0"][:, pr_idx],
                band_rad=float(cfg.cost.waist_pr_band_rad),
                pr_limit=pr_limit,
            )
        if cfg.w_waist_pr_sp_distill > 0.0 and enc.waist_pr_sp is not None:
            q_hit = waist_pr_hit_teacher(
                _traj, strike.hit_index, batch["active"], pr_idx,
            )
            mask = waist_pr_distill_gate(
                gate[:, 0], gate[:, 1], gate[:, 2], gate[:, 3],
                q_hit, batch["active"],
                pr_limit=pr_limit,
                hit_m=float(cfg.pr_distill_hit_m),
            )
            loss_pr_d = waist_pr_distill_loss(
                enc.waist_pr_sp.float(), q_hit.float(), mask,
            )
            extra = extra + cfg.w_waist_pr_sp_distill * loss_pr_d
        if cfg.w_waist_sp_hold > 0.0 and enc.waist_yaw_sp is not None:
            q_yaw = batch["q0"][:, yaw_i].detach().float()
            d_yaw = enc.waist_yaw_sp.float() - q_yaw
            extra = extra + cfg.w_waist_sp_hold * d_yaw.pow(2).mean()
        if (
            cfg.w_waist_sp_smooth > 0.0
            and (enc.waist_yaw_sp is not None or enc.waist_pr_sp is not None)
        ):
            loss_sm, _, _, _ = _waist_sp_smooth_term(
                model=model, ctx=batch["context"], enc=enc,
                waist_cfg=waist_cfg,
            )
            extra = extra + cfg.w_waist_sp_smooth * loss_sm
        if (
            cfg.waist_sp_search_frac > 0.0 and cfg.w_waist_sp_search > 0.0
            and enc.waist_yaw_sp is not None
        ):
            n_ss = min(max(int(cfg.waist_sp_search_frac * bs), 1), bs)
            loss_sp_search, _ = _waist_sp_search_term(
                model=model, cost=costs[bucket], cfg=cfg, enc=enc, batch=batch,
                strike=strike, lengths=lengths, bucket=bucket, weights=w,
                rows=slice(0, n_ss),
            )
            extra = extra + cfg.w_waist_sp_search * loss_sp_search
        total = loss + extra
    if return_tensor:
        return total
    return float(total.detach())


_TORQUE_GRAD_STATUS_PREFIX = "torque_grad_status_"
_TORQUE_GRAD_COUNT_KEYS = {
    "torque_grad_cap_n",
    "torque_grad_finite_n",
}


def _accumulate_torque_grad_cap_stats(
    roll_acc: dict[str, float],
    stats: dict[str, float],
) -> None:
    """Accumulate one torque-budget attempt without mixing failure and means."""
    roll_acc["torque_grad_cap_n"] = (
        roll_acc.get("torque_grad_cap_n", 0.0) + 1.0
    )
    finite = "torque_grad_frac_before" in stats
    if finite:
        roll_acc["torque_grad_finite_n"] = (
            roll_acc.get("torque_grad_finite_n", 0.0) + 1.0
        )
    for key, value in stats.items():
        if key == "torque_grad_cap":
            continue
        if key.startswith(_TORQUE_GRAD_STATUS_PREFIX) or finite:
            roll_acc[key] = roll_acc.get(key, 0.0) + float(value)


def _torque_grad_cap_log_fields(
    roll_acc: dict[str, float],
    *,
    cap: float,
) -> dict[str, float]:
    """Build train-record means and raw nonfinite counters."""
    attempts = roll_acc.get("torque_grad_cap_n", 0.0)
    if attempts <= 0.0:
        return {}
    finite_n = roll_acc.get("torque_grad_finite_n", 0.0)
    out = {
        "torque_grad_cap": float(cap),
        "torque_grad_cap_n": attempts,
        "torque_grad_finite_n": finite_n,
    }
    for key, value in roll_acc.items():
        if key in _TORQUE_GRAD_COUNT_KEYS or key == "torque_grad_cap":
            continue
        if key.startswith(_TORQUE_GRAD_STATUS_PREFIX):
            out[key] = value
        elif key.startswith("torque_grad_") and finite_n > 0.0:
            out[key] = value / finite_n
    return out


def _invocation_throughput(
    *,
    updates_attempted: int,
    updates_executed: int,
    wall_s: float,
) -> dict[str, float | int]:
    """Invocation-local rates; absolute checkpoint steps are not work counts."""
    denom = max(float(wall_s), 1e-9)
    return {
        "updates_attempted": int(updates_attempted),
        "updates_executed": int(updates_executed),
        "steps_per_s": int(updates_attempted) / denom,
        "executed_steps_per_s": int(updates_executed) / denom,
    }


class _CudaPhaseTimer:
    """Low-cadence CUDA-event timing for one outer update.

    Events are resolved with one synchronization at :meth:`finish`, so normal
    steps incur no timing synchronization and sampled steps incur exactly one.
    """

    def __init__(self, *, enabled: bool, device: torch.device):
        self.enabled = bool(enabled and device.type == "cuda")
        self.wall_start = time.perf_counter()
        self._last_name = "setup"
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._last_event = None
        if self.enabled:
            self._last_event = torch.cuda.Event(enable_timing=True)
            self._last_event.record()

    def mark(self, next_name: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        assert self._last_event is not None
        self._events.append((self._last_name, self._last_event, event))
        self._last_name = str(next_name)
        self._last_event = event

    def finish(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        self.mark("done")
        assert self._last_event is not None
        self._last_event.synchronize()
        out: dict[str, float] = {}
        cuda_total = 0.0
        for name, start, end in self._events:
            elapsed = float(start.elapsed_time(end))
            out[f"phase_{name}_cuda_ms"] = elapsed
            cuda_total += elapsed
        wall = 1_000.0 * (time.perf_counter() - self.wall_start)
        out["phase_total_cuda_ms"] = cuda_total
        out["phase_total_wall_ms"] = wall
        out["phase_host_gap_ms"] = max(wall - cuda_total, 0.0)
        return out


def train_phase2(
    *,
    model,
    pool: Phase2Pool,
    cfg: Phase2Config,
    model_cfg: ModelConfig,
    run_dir,
    scenario_cfg: Phase2ScenarioConfig | None = None,
    eval_cases=None,
    scorer=None,
    kit=None,
    library=None,
    al_state: dict | None = None,
    opt_state: dict | None = None,
    rng_state: dict | None = None,
    line_search_state: dict | None = None,
    voxel_train_state: dict | None = None,
    timing_critic_state: dict | None = None,
    suffix_replay_state: dict | None = None,
    probe_fn=None,
    stage_a_init: bool = False,
    max_hours: float | None = None,
    start_step: int = 0,
    log: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """The merged phase-2/3 loop.  Writes ``metrics.jsonl`` + checkpoints.

    ``probe_fn(model) -> dict`` is an optional CLOSED-LOOP metric run every
    ``cfg.probe_every`` steps; it must return at least ``{"in_disk": int,
    "hits": int}``.  ``best.pt`` is gated on it: a rolling-grid acceptance
    record only overwrites ``best.pt`` if the latest probe is at least as good
    as the probe recorded with the previous best.  Section 7b selected purely on
    the open-loop grid and shipped a checkpoint that scored 175/243 on the plan
    and 2/55 on the plant, so the grid alone is demonstrably not a sufficient
    selection signal.  The best-by-probe checkpoint is ALSO written, separately,
    to ``best_probe.pt``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.curvature_direct_frac > 0.0:
        if cfg.dls_terminal_frac > 0.0:
            raise ValueError("direct curvature requires dls_terminal_frac=0")
        if cfg.grad_balance:
            raise ValueError("direct curvature requires grad_balance='' ")
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
    # A single mis-compiled subgraph must degrade to eager, not abort a 2-hour
    # run.  Inductor + this Triton has at least one such kernel (see `encode`).
    torch._dynamo.config.suppress_errors = True
    amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.amp_dtype)

    core = model.module if hasattr(model, "module") else model
    if cfg.timing_critic.mode in {"assist", "control"}:
        if cfg.voxel_pool:
            raise ValueError(
                "timing critic assist/control replaces voxel probes and cannot run inside voxel_pool"
            )
        if bool(getattr(model_cfg, "enc_goal_tth", False)):
            raise ValueError(
                "timing critic assist requires enc_goal_tth=False so retiming can reuse encoding"
            )
    if (
        cfg.timing_critic.representation == "distributional"
        and cfg.timing_critic.anchor_curriculum
    ):
        # Suffix-Bellman already supplies shortened replans with an exact cost
        # and parent-tail identity.  Do not pay for a second TC child solve in
        # the neural timing curriculum.
        cfg.tc_frac = 0.0
    if cfg.voxel_pool and (
        bool(getattr(core.cfg, "waist_sp_head", False))
        or bool(getattr(core.cfg, "waist_pr_sp_head", False))
    ):
        raise ValueError(
            "voxel_pool requires a model constructed without encoder waist "
            "set-point heads; strip legacy checkpoint heads before build"
        )
    if cfg.voxel_pool and (
        cfg.waist_sp_from_head or cfg.waist_pr_sp_from_head or cfg.waist_pr_ema
        or cfg.w_waist_sp_tether or cfg.w_waist_pr_sp_tether
        or cfg.w_waist_pr_sp_hold or cfg.w_waist_pr_sp_hold_band
        or cfg.w_waist_pr_sp_distill or cfg.w_waist_sp_hold
        or cfg.w_waist_sp_smooth or cfg.waist_sp_search_frac
        or cfg.w_waist_sp_search or cfg.cost.w_waist_aim
        or cfg.cost.w_waist_yaw_monotonic
        or cfg.cost.w_waist_yaw_accel_direction
        or cfg.cost.w_waist_yaw_initial_accel_direction
    ):
        raise ValueError(
            "voxel_pool configuration was mutated to enable a forbidden "
            "waist set-point/reference coupling"
        )
    if cfg.waist_sp_from_head and not core.cfg.waist_sp_head:
        raise ValueError(
            "waist_sp_from_head requires a model built with "
            "ModelConfig.waist_sp_head=True"
        )
    if cfg.waist_pr_sp_from_head and not core.cfg.waist_pr_sp_head:
        raise ValueError(
            "waist_pr_sp_from_head requires a model built with "
            "ModelConfig.waist_pr_sp_head=True"
        )
    if int(cfg.grad_balance_refresh) < 1:
        raise ValueError("grad_balance_refresh must be >= 1")
    if not 0.0 <= float(cfg.grad_balance_norm_ema) < 1.0:
        raise ValueError("grad_balance_norm_ema must be in [0, 1)")
    # Plan §14: surgery must be off before make_cost_step — return arity /
    # cost_grad_balance_tuple vs cost_tuple are baked at build time.  Belt-
    # and-suspenders if cfg was mutated after Phase2Config.__post_init__.
    if cfg.line_search and cfg.grad_balance:
        warnings.warn(
            f"train_phase2: line_search=True clearing grad_balance="
            f"{cfg.grad_balance!r} before make_cost_step (plan §14)",
            UserWarning,
            stacklevel=2,
        )
        cfg.grad_balance = ""
    dtype = torch.float32
    costs = build_costs(pool.buckets, cfg.cost, device=device, dtype=dtype)
    base = costs[max(costs)]
    yaw_i = base.joint_names.index("waist_yaw_joint")
    pr_idx = [
        base.joint_names.index("waist_pitch_joint"),
        base.joint_names.index("waist_roll_joint"),
    ]
    _pr_lo, _pr_hi = urdf_position_limits(
        ("waist_pitch_joint", "waist_roll_joint"),
    )
    pr_limit = float(np.minimum(-_pr_lo, _pr_hi).min())
    # ONE plant for the roller, the TC step and the BPTT chain, so the effort
    # clamp cannot be on in one and off in another.
    sim = BatchedPDSim(
        PDSimConfig(dt=pool.cfg.dt, saturate_effort=bool(cfg.plant_saturate_effort)),
        device=device, dtype=dtype, rnea=base.rnea,
    )
    scales = LossScales(device, dtype)
    al = AugmentedLagrangian(cfg.al)
    if al_state:
        # A resume CONTINUES the multipliers.  Restarting them would throw away
        # the balance the run spent thousands of steps discovering, and the
        # weight ramp would have to climb it again from scratch.  Checkpoints
        # written before ``torque_peak`` became its own group are migrated so
        # the peak half keeps the multiplier it was trained under.
        al.load_state_dict(migrate_al_state(al_state))
        # Ki (``rho``), clips, PID gains, targets and Bertsekas knobs are
        # CONFIG.  ``pen`` (the adapted quadratic coefficient), ``err_prev``
        # and the held D kick are STATE -- overwriting pen from the live table
        # would throw away the wall the run grew, the same failure mode that
        # used to hit ``rho``.  Old checkpoints have no ``pen`` key; they keep
        # the live init.
        fresh = AugmentedLagrangian(cfg.al)
        al.rho = dict(fresh.rho)
        al.lam_max = dict(fresh.lam_max)
        al.lam_min = dict(fresh.lam_min)
        al.target = dict(fresh.target)
        al.kp = dict(fresh.kp)
        al.kd = dict(fresh.kd)
        al.pen_min = dict(fresh.pen_min)
        al.pen_max = dict(fresh.pen_max)
        al.cfg = fresh.cfg
        had_pen = bool((al_state or {}).get("pen"))
        if not had_pen:
            al.pen = dict(fresh.pen)
        for g in al.groups:
            al.lam[g] = float(min(max(al.lam[g], al.lam_min[g]), al.lam_max[g]))
            al.pen[g] = clamp_resumed_al_penalty(
                al.pen[g], al.pen_min[g], al.pen_max[g]
            )
        if log and rank == 0:
            tw = limit_group_name("torque", "waist")
            print(f"[resume] AL config reapplied: rho_{tw}={al.rho[tw]:.4g} "
                  f"lam_max_{tw}={al.lam_max[tw]:.4g} "
                  f"lam_{tw}={al.lam[tw]:.3f} "
                  f"pen_{tw}={al.pen[tw]:.4g} "
                  f"(pen {'from ckpt' if had_pen else 'from config'})",
                  flush=True)

    if cfg.compile_mode is not None and float(cfg.torque_grad_cap) > 0.0:
        # Torque-budget refreshes take a second VJP from the same forward graph.
        # AOTAutograd otherwise donates (and invalidates) saved buffers after
        # the first backward, which is incompatible with retain_graph=True.
        # This only changes buffer lifetime; compiled forward/backward math and
        # parameter ownership remain unchanged for SaLSa candidate writes.
        import torch._functorch.config as _functorch_config
        _functorch_config.donated_buffer = False

    encode_step = make_encode_step(
        model,
        compile_mode=cfg.compile_mode if cfg.compile_model else None,
    )
    # cfg.grad_balance must already be "" under line_search (plan §14); arity
    # of cost_tuple vs cost_grad_balance_tuple is fixed here.
    step_fns = {b: make_cost_step(model, costs[b], scales, cfg,
                                  compile_mode=cfg.compile_mode) for b in pool.buckets}
    tc_fns = (
        {b: make_tc_step(model, costs[b], sim, cfg,
                         compile_mode=cfg.compile_mode if cfg.compile_model else None)
         for b in pool.buckets}
        if cfg.w_tc > 0.0 and cfg.tc_frac > 0.0 else {}
    )
    sfx_fns = (
        {b: make_suffix_step(model, costs[b], cfg,
                             compile_mode=cfg.compile_mode if cfg.compile_model
                             else None)
         for b in pool.buckets}
        if cfg.suffix_frac > 0.0
        and cfg.w_suffix_tie > 0.0 else {}
    )
    suffix_ring = _VerifiedSuffixRing(
        capacity=int(cfg.suffix_replay_capacity), device=device,
    )
    if suffix_replay_state:
        suffix_ring.load_state_dict(suffix_replay_state)
    bptt_bucket = min(pool.buckets)
    bptt_fn = (
        make_bptt_step(model, costs[bptt_bucket], sim, cfg,
                       compile_mode=cfg.compile_mode if cfg.compile_cost else None)
        if cfg.w_bptt > 0.0 and cfg.bptt_max_frac > 0.0 else None
    )

    # ONE flag decides whether a waist-yaw reference is harvested (the roller)
    # and stored (the replay pool); computing it once is what keeps the two
    # from disagreeing -- ``ReplayPool.add`` raises if they ever do.
    want_waist_ref = (
        cfg.cost.w_waist_aim > 0.0
        or cfg.cost.w_waist_yaw_monotonic > 0.0
        or cfg.cost.w_waist_yaw_accel_direction > 0.0
        or cfg.cost.w_waist_yaw_initial_accel_direction > 0.0
    ) and not cfg.waist_sp_from_head
    want_waist_pr_ref = bool(cfg.waist_pr_ema) and not cfg.waist_pr_sp_from_head
    replay = None
    post_hit = None
    if not cfg.voxel_pool:
        replay = ReplayPool(
            pool, cfg.replay_capacity,
            buckets=[b for b in pool.buckets if b <= cfg.roller_horizon],
            recency_bias=cfg.recency_bias,
            viol_downweight=float(cfg.replay_viol_downweight),
            viol_uniform_mix=float(cfg.replay_viol_uniform_mix),
            proficiency=bool(cfg.replay_proficiency),
            ema_alpha=float(cfg.frontier_ema_alpha),
            waist_ref=want_waist_ref,
            waist_pr_ref=want_waist_pr_ref,
        )
        post_hit = ReplayPool(
            pool, int(cfg.post_hit_capacity),
            buckets=[b for b in pool.buckets if b <= cfg.roller_horizon],
            recency_bias=cfg.recency_bias,
            waist_ref=want_waist_ref,
            waist_pr_ref=want_waist_pr_ref,
        )
    # THE IDLE ARM.  One library, three consumers: the cost prices its cylinder,
    # the roller seeds against it, and (through ``scenario_cfg``) the fresh pool
    # was generated against it.  Installing the cylinder on the costs is what
    # makes ``w_idle_cylinder`` / ``w_idle_vel`` live -- without it they are
    # exact zeros.
    idle_path = cfg.resolved_idle_cylinder_path()
    idle_lib = resolve_idle_cylinder_library(
        idle_path, joint_names=pool.cfg.joint_names
    )
    idle_cyl = None
    if idle_lib is not None:
        try:
            idle_cyl = derive_cylinder(kit, base.fk)
        except Exception:
            idle_cyl = idle_lib.cylinder
        for c in costs.values():
            if hasattr(c, "set_idle_cylinder"):
                c.set_idle_cylinder(idle_cyl if idle_cyl is not None else idle_lib)
        if log and rank == 0:
            print(
                f"idle-cylinder: K={idle_lib.K} poses/hand, "
                f"R={idle_cyl.radius:.3f} m, hh={idle_cyl.half_height:.3f} m"
            )

    roller_clips = None
    src0 = next(iter(pool.data.values()))
    if not cfg.voxel_pool and "clip_i" in src0:
        from .rhythm_chain_data import roller_clip_list as _roller_clips

        roller_clips = _roller_clips(
            kit or DrumKit.from_layout(), library=library, cfg=pool.cfg,
        )
        if log and rank == 0:
            print(
                f"[data] roller clip refill: {len(roller_clips)} clips "
                f"(rcc+synth; clip_i provenance)",
                flush=True,
            )

    roller = None
    if not cfg.voxel_pool:
        roller = MPCRoller(
            pool, model=model, sim=sim, rnea=base.rnea, fk=base.fk,
            horizon=cfg.roller_horizon, slots=cfg.roller_slots,
            chain_frac=cfg.roller_chain_frac, bounce=cfg.bounce,
            device=device, dtype=dtype, seed=cfg.seed + 1301 + 17 * rank, amp_dtype=amp,
            regoal=cfg.roller_regoal, cmd_lead_s=cfg.cmd_lead_s,
            library=library, kit=kit,
            yaw_bins=cfg.roller_yaw_bins, max_age=cfg.roller_max_age,
            idle_library=idle_lib, idle_cylinder=idle_cyl,
            idle_settle_wp=cfg.idle_settle_wp,
            rest_ticks=cfg.roller_rest_ticks if idle_lib is not None else 0,
            waist_ref=want_waist_ref,
            n_shaft=costs[min(costs)].n_points - 2,
            waist_upright_ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
            clips=roller_clips,
            waist_pr_ema=want_waist_pr_ref,
            waist_pr_ema_alpha=float(cfg.waist_pr_ema_alpha),
            waist_pr_ema_leak=float(cfg.waist_pr_ema_leak),
        )
    frontier_cloud = None
    if bool(cfg.frontier):
        frontier_cloud = _setup_frontier_cloud(
            pool, cfg, device=device, dtype=dtype, run_dir=run_dir,
            log=bool(log and rank == 0),
        )
        roller.frontier = frontier_cloud
        roller.frontier_spawn_frac = float(cfg.frontier_spawn_frac)
        if bool(cfg.frontier_idle_far_cloud) and log and rank == 0:
            print(
                "[frontier] idle/far train rows from the cloud only "
                "(seed + ball children); post-hit stays the live roller; "
                f"splice {int(cfg.frontier_splice_k)} cloud rows into "
                "every replay/fresh batch; no steal / post-hit harvest "
                "into the cloud",
                flush=True,
            )
    roll_acc = {"reset": 0.0, "diverged": 0.0, "regoal": 0.0, "idle": 0.0,
                "stale": 0.0, "rest": 0.0, "idle_seed": 0.0, "sep_push": 0.0,
                "post_hit": 0.0, "ls_u_fresh_frac": 0.0, "ls_u_mom_frac": 0.0,
                "ls_u_decay_frac": 0.0, "ls_u_mom_over_fresh": 0.0, "ls_u_n": 0.0}
    replay_viol_selected: list[torch.Tensor] = []

    opt = build_optimizer(model, cfg)
    opt_restored = False
    if opt_state and str(cfg.optimizer).lower() != "lbfgs":
        # A resume continues the MOMENTS as well as the multipliers.  Adam's
        # second moment is what sets the effective per-parameter step size, and
        # restarting it makes the first few hundred steps of a continuation
        # behave like a fresh run at a much larger LR -- on a checkpoint this
        # far into cost descent that is a real risk of undoing it.
        try:
            opt.load_state_dict(opt_state)
            opt_restored = True
        except (ValueError, KeyError) as exc:  # param-group shape changed
            print(f"[resume] optimizer state NOT restored: {exc}", flush=True)
    elif opt_state and str(cfg.optimizer).lower() == "lbfgs" and log and rank == 0:
        print(
            "[resume] L-BFGS: skipping Adam optimizer state restore "
            "(fresh quasi-Newton history)",
            flush=True,
        )
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed) + 7919 + 101 * rank)
    rng = np.random.default_rng(cfg.seed + 31 * rank)
    # The suffix term draws its offsets from its OWN stream so that switching
    # it on cannot advance ``gen`` -- an A/B against a suffix-off run then sees
    # the exact same batch / noise / gain sequence, and the arms differ by the
    # loss term alone.
    sfx_gen = torch.Generator(device=device)
    sfx_gen.manual_seed(int(cfg.seed) + 4243 + 101 * rank)
    jit_gen = torch.Generator(device=device)
    jit_gen.manual_seed(int(cfg.seed) + 5501 + 101 * rank)
    phase2_rng_streams = {
        "gen": gen,
        "sfx_gen": sfx_gen,
        "jit_gen": jit_gen,
        "rng": rng,
    }

    timing_critic = None
    timing_mpc_ring = None
    timing_critic_pending: list[dict] = []
    if cfg.timing_critic.enabled:
        trainer_cls = (
            OnlineDistributionalCriticTrainer
            if cfg.timing_critic.representation == "distributional"
            else ViolationCriticTrainer
        )
        timing_critic = trainer_cls(
            critic_feature_dim(len(base.joint_names)),
            cfg.timing_critic,
            device=device,
            seed=int(cfg.seed) + 8803 + 101 * rank,
        )
        if timing_critic_state:
            timing_critic.load_state_dict(timing_critic_state)
            for item in timing_critic_state.get("pending_observations", []):
                timing_critic_pending.append({
                    "features": item["features"].detach().cpu(),
                    "gate": item["gate"].detach().cpu(),
                    "anchor": (
                        None if item.get("anchor") is None
                        else item["anchor"].detach().cpu().bool()
                    ),
                    "actor_step": int(item["actor_step"]),
                })
        if log and rank == 0:
            restored = " restored" if timing_critic_state else " fresh"
            print(
                f"[timing-critic] mode={cfg.timing_critic.mode} "
                f"representation={cfg.timing_critic.representation} "
                f"heads={cfg.timing_critic.num_heads}{restored} "
                f"recent={len(timing_critic.recent)} "
                f"audit={len(timing_critic.audit)}",
                flush=True,
            )
        if (
            cfg.timing_critic.representation == "distributional"
            and cfg.timing_critic.anchor_curriculum
        ):
            timing_mpc_ring = _TrajectorySuffixMPCParentRing(
                device=device,
                pool=pool,
                costs=costs,
                q_noise=cfg.timing_mpc_q_noise,
                qd_noise=cfg.timing_mpc_qd_noise,
                capacity=cfg.timing_mpc_parent_capacity,
                terminal_peak_frac=cfg.terminal_suffix_peak_frac,
                terminal_long_frac=cfg.terminal_suffix_long_frac,
                terminal_low_frac=cfg.terminal_suffix_low_frac,
                terminal_peak_wp=cfg.terminal_suffix_peak_wp,
                terminal_tail_tau=cfg.terminal_suffix_tail_tau,
            )
            if timing_critic_state and timing_critic_state.get("primary_mpc_ring"):
                timing_mpc_ring.load_state_dict(
                    timing_critic_state["primary_mpc_ring"]
                )
                if log and rank == 0 and timing_mpc_ring.dropped_legacy_rows:
                    print(
                        "[timing-critic] discarded "
                        f"{timing_mpc_ring.dropped_legacy_rows} legacy MPC rows "
                        "without advanced state/TTH; buckets refill on next visit",
                        flush=True,
                    )

    def _flush_timing_critic_pending() -> None:
        if timing_critic is None or not timing_critic_pending:
            return
        pending = list(timing_critic_pending)
        timing_critic_pending.clear()
        features = torch.cat([item["features"] for item in pending], dim=0)
        gates = torch.cat([item["gate"] for item in pending], dim=0)
        actor_step = max(int(item["actor_step"]) for item in pending)
        kwargs = {
            "actor_step": actor_step,
            "batch_size_multiplier": len(pending),
        }
        if cfg.timing_critic.representation == "distributional":
            anchors = [item["anchor"] for item in pending]
            kwargs["anchor_mask"] = torch.cat(anchors, dim=0)
        timing_critic.observe_gate(features, gates, **kwargs)
    if int(cfg.actor_superbatch_factor) > 1 and timing_mpc_ring is None:
        raise ValueError(
            "actor_superbatch_factor > 1 currently requires the "
            "distributional anchor timing curriculum"
        )

    voxel_state = None
    voxel_mpc_ring = None
    voxel_bootstrap_stats = None
    if cfg.voxel_pool:
        if bool(cfg.frontier):
            raise ValueError(
                "voxel_pool and legacy frontier cannot be enabled together"
            )
        voxel_state = _VoxelTrainingState(
            pool, cfg,
            gate_definition=_voxel_runtime_gate_definition(base, pool),
        )
        voxel_mpc_ring = _VoxelMPCParentRing(device=device)
        # Used both for a cold bootstrap and for hard-checking any missing
        # parent-ring bucket on resume.  Define it unconditionally: a restored
        # curriculum may legitimately have an empty bucket generation.
        bootstrap_weights = _wtensors(
            cfg.schedule.multipliers(start_step, bool(stage_a_init)),
            al.weights(), device, dtype, sched=cfg.schedule,
            joint_names=base.joint_names, step=start_step,
            stage_a=bool(stage_a_init), pen=al.pen,
        )
        restored_voxel = bool(voxel_train_state)
        if restored_voxel:
            voxel_state.load_state_dict(voxel_train_state["voxel"])
            voxel_mpc_ring.load_state_dict(voxel_train_state["mpc_ring"])
            voxel_bootstrap_stats = {"restored": True, "cells": len(voxel_state.pool)}
        else:
            voxel_cache = Path(cfg.voxel_cache) if cfg.voxel_cache else None
            if voxel_cache is not None and voxel_cache.exists():
                voxel_state.pool = VoxelPool.load_npz(
                    voxel_cache,
                    expected_gate_definition=voxel_state.gate_definition,
                )
                voxel_bootstrap_stats = {
                    "cache": True, "cells": len(voxel_state.pool),
                }
            else:
                voxel_bootstrap_stats = voxel_state.bootstrap(
                    costs, bootstrap_weights, model=model,
                )
                if voxel_cache is not None and len(voxel_state.pool):
                    voxel_cache.parent.mkdir(parents=True, exist_ok=True)
                    voxel_state.pool.save_npz(
                        voxel_cache,
                        gate_definition=voxel_state.gate_definition,
                    )
        if log and rank == 0:
            print(
                "[voxel] bootstrap "
                + " ".join(f"{k}={v}" for k, v in voxel_bootstrap_stats.items()),
                flush=True,
            )
        if not len(voxel_state.pool):
            raise RuntimeError(
                "voxel bootstrap found no hard-feasible trajectory witnesses; "
                "increase --voxel-bootstrap-rows or inspect the four hard gates"
            )
        active_cells = sum(
            key.chart_id != 0 for key in voxel_state.pool.keys()
        )
        if active_cells == 0:
            raise RuntimeError(
                "voxel bootstrap found only rest witnesses and no hard-feasible "
                "strike trajectory; use a larger bootstrap source, an initialized "
                "Phase-2 policy, or a compatible --voxel-cache"
            )
        primed = voxel_state.prime_mpc_ring(
            voxel_mpc_ring, model, costs, bootstrap_weights,
            generator=gen, rng=rng, yaw_i=yaw_i,
        )
        voxel_bootstrap_stats["mpc_primed"] = primed

    def _train_checkpoint_extra(extra: dict | None = None) -> dict:
        out = dict(extra or {})
        if sfx_fns:
            out["suffix_replay"] = suffix_ring.state_dict()
        if timing_critic is not None:
            timing_state = timing_critic.state_dict()
            timing_state["pending_observations"] = [
                {
                    "features": item["features"].clone(),
                    "gate": item["gate"].clone(),
                    "anchor": (
                        None if item["anchor"] is None
                        else item["anchor"].clone()
                    ),
                    "actor_step": int(item["actor_step"]),
                }
                for item in timing_critic_pending
            ]
            if timing_mpc_ring is not None:
                timing_state["primary_mpc_ring"] = timing_mpc_ring.state_dict()
            out["timing_critic"] = timing_state
        if voxel_state is not None:
            out["voxel_training"] = {
                "voxel": voxel_state.state_dict(),
                "mpc_ring": voxel_mpc_ring.state_dict(),
            }
        return out

    ls_rt = None
    replay_mgr = None
    if cfg.line_search:
        ls_objective_fingerprint = build_objective_direction_fingerprint(
            cfg,
            al_lam=al.lam,
            al_pen=al.pen,
            stage_a=bool(stage_a_init),
            torque_full_blend=cfg.schedule.torque_full_blend(start_step),
        )
        ls_rt = init_line_search_runtime(
            phase2_cfg=cfg,
            optimizer=opt,
            line_search_state=line_search_state,
            opt_restored=opt_restored,
            objective_direction_fingerprint=ls_objective_fingerprint,
        )
        replay_mgr = ObjectiveReplayContext(
            {"gen": gen, "sfx_gen": sfx_gen, "jit_gen": jit_gen, "rng": rng},
            capture_global=True,
        )
        if log and rank == 0:
            print(
                f"[line_search] enabled optimizer={ls_rt.optimizer_kind} "
                f"grad_balance='' "
                f"moment_on={ls_rt.cfg.moment_on} "
                f"eta_ref={ls_rt.eta_ref:.3g} "
                f"bounds=[{ls_rt.epsilon_min:.3g},{ls_rt.epsilon_max:.3g}] "
                f"seed_provenance={ls_rt.seed_provenance} "
                f"K={ls_rt.ctrl.K}",
                flush=True,
            )
            if ls_rt.fingerprint_reset_reason is not None:
                print(
                    "[line_search] selector acceptance history reset on resume: "
                    f"{ls_rt.fingerprint_reset_reason}; K=1, search pending",
                    flush=True,
                )
        ls_rt.al_lam_snapshot = dict(al.lam)
        ls_rt.al_pen_snapshot = dict(al.pen)
        ls_rt.stage_a_snapshot = bool(stage_a_init)

    # Restore only after resume-time setup (including voxel priming and line
    # search construction) so those deterministic setup draws cannot shift the
    # first resumed update.  A fresh run has ``rng_state=None`` and follows the
    # exact historical manual-seed path above.
    if rng_state is not None:
        restored_rng = _restore_phase2_rng_state(rng_state, phase2_rng_streams)
        if restored_rng and log and rank == 0:
            print(f"[resume] RNG state restored ({restored_rng})", flush=True)

    fh = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None
    tb = MetricsTensorboard(
        run_dir / "tb",
        enabled=bool(cfg.tensorboard) and rank == 0,
    )
    if rank == 0 and cfg.tensorboard and log:
        print(f"[tensorboard] logdir={run_dir / 'tb'}", flush=True)

    def emit(rec: dict) -> None:
        if fh is None:
            return
        fh.write(json.dumps(rec, default=float) + "\n")
        fh.flush()
        tb.log_record(rec)

    emit({"event": "start", "cfg": asdict(cfg), "model_cfg": asdict(model_cfg),
          "scenario_cfg": asdict(scenario_cfg) if scenario_cfg else None,
          "pool": pool.distribution(), "pool_stats": {
              k: (float(np.percentile(v, 50)) if isinstance(v, np.ndarray) else v)
              for k, v in pool.stats.items()},
          "voxel_bootstrap": voxel_bootstrap_stats,
          "world_size": world_size})

    bucket_p = np.array([
        (len(voxel_state.pool.keys(bucket=b)) if voxel_state is not None else pool.size(b))
        for b in pool.buckets
    ], float)
    bucket_p /= bucket_p.sum()
    acc = np.zeros(len(METRIC_NAMES))
    acc_n = 0
    mix_acc = [0.0, 0.0, 0.0]   # sampled-batch fraction with k_min <= 3 / 5 / 8
    mix_n = 0
    batch_b_acc = 0.0           # mean actual B over the log window (== batch_for)
    bucket_step_acc = {int(b): 0 for b in pool.buckets}
    # mean |sp - aim_yaw| of the head, the "how far has the learned law walked
    # from the analytic one" diagnostic; saturation against the tether weight
    # means the tether is the binding constraint.
    sp_res_acc, sp_res_n = 0.0, 0
    # mean |sp_pr| of the lean head; saturation against the tether at ~0.52
    # rad means the pair co-drifted to the URDF rail.
    pr_sp_acc, pr_sp_n = 0.0, 0
    # mean |sp - q0| at encode: how far the announcement walks from the pose
    # the encoder actually saw.
    pr_hold_acc, pr_hold_n = 0.0, 0
    # mean relu(|sp - clamp(q0)| - band): 0 iff the decoder window contains
    # the (URDF-clamped) start.  The quadratic hold can be large while this
    # is 0, and vice versa on an unclipped inherited dump.
    pr_hold_band_acc, pr_hold_band_n = 0.0, 0
    yaw_hold_acc, yaw_hold_n = 0.0, 0
    pr_distill_frac_acc, pr_distill_delta_acc, pr_distill_n = 0.0, 0.0, 0
    # mean |sp_new - sp_old| on the next-hit re-encode (yaw, then P/R).
    sm_acc, sm_n = [0.0, 0.0], 0
    waist_cfg = getattr(pool.cfg, "waist_cfg", None)
    # Search distillation, in ``_waist_sp_search_term``'s order:
    # mean |sp - sp*| (is the head listening), mean |sp* - analytic aim| (where
    # the ENVELOPE actually wants the torso, independent of the head), and the
    # V-spread max-min over candidates.  The spread is the load-bearing one: if
    # it is ~0 the set-point does not change the outcome and the whole head --
    # not just this loss -- is moot.
    ss_acc, ss_n = [0.0, 0.0, 0.0], 0
    sfx_acc, sfx_n = [0.0, 0.0], 0
    mpc_tie_acc, mpc_tie_frac_acc, mpc_tie_n = 0.0, 0.0, 0
    waist_mpc_tie_acc, waist_mpc_tie_grad_acc, waist_mpc_tie_n = 0.0, 0.0, 0
    viol_acc = {g: 0.0 for g in CONSTRAINT_GROUPS}
    viol_n = 0
    best = -1.0
    best_step = 0
    best_probe = -1.0          # probe score recorded with the current best.pt
    best_probe_only = -1.0     # best probe seen at all -> best_probe.pt
    last_probe: dict | None = None
    recent_train: deque = deque(maxlen=20)
    #: Highest acceptance SEEN, whether or not the probe gate let it be saved.
    #: Distinct from ``best`` on purpose: ``best`` answers "what is in best.pt",
    #: this answers "is the run still learning", and only the second one may
    #: drive the plateau stop.  Conflating them cost runs/phase2_19m_waist_sp_b
    #: 34 000 steps -- a lucky probe (0.467 at step 14 000, against a range of
    #: 0.000-0.467 on a ~23-hit single-clip metric) sealed the save gate, so
    #: every later acceptance GAIN (0.383, 0.494, 0.506) was booked as a
    #: plateau tick and the run stopped at 26 000 of 60 000 while climbing.
    best_seen = -1.0
    since_improve = 0
    n_skipped = 0
    updates_attempted = 0
    updates_executed = 0
    last_step = int(start_step)
    gb_norm_ema_by_bucket: dict[int, dict[str, torch.Tensor]] = {}
    gb_visits_by_bucket: dict[int, int] = {}
    torque_grad_cached_scale = 1.0
    torque_grad_cache_initialized = False
    torque_grad_visits = 0
    phase_timing_latest: dict[str, float] = {}
    t0 = time.perf_counter()
    t_last = t0
    stopped = "max_steps"
    eval_hist: list[dict] = []
    # A resume from a checkpoint that already passed stage A must seed the gate
    # OPEN, or ``WeightSchedule.gate_stage_a`` would hold the limit hinges at
    # ``lo_limits`` until the first rolling eval -- a 5 % torque price for
    # ``eval_every`` steps, on a model whose torque term is the binding gate.
    stage_a = bool(stage_a_init)
    # Ephemeral by design: checkpoints resume at the next freshly generated
    # super-batch instead of serializing several large collocation slices.
    actor_superbatch_queue: deque[dict] = deque()

    deadline = None if max_hours is None else t0 + 3600.0 * float(max_hours)
    start_step = int(start_step)
    if start_step < 0:
        raise ValueError(f"start_step must be >= 0, got {start_step}")
    if start_step >= int(cfg.max_steps):
        raise ValueError(
            f"start_step {start_step} >= max_steps {cfg.max_steps}: nothing to run"
        )
    if log and rank == 0 and start_step:
        print(f"[resume] continuing from step {start_step} "
              f"(schedule/eval clocks use absolute step)", flush=True)
    if rank == 0:
        _ShutdownFlag.install()
        keep_msg = (
            f"step snapshots every {int(cfg.ckpt_keep_every)}; "
            if int(cfg.ckpt_keep_every) > 0 else ""
        )
        if log and (int(cfg.ckpt_last_every) > 0 or int(cfg.ckpt_keep_every) > 0):
            print(
                f"[ckpt] last.pt every {int(cfg.ckpt_last_every)} steps; "
                f"{keep_msg}"
                f"full bundle every {int(cfg.ckpt_every)}",
                flush=True,
            )
        if log and (
            float(cfg.replay_viol_downweight) > 0.0
            or float(cfg.replay_viol_uniform_mix) > 0.0
        ):
            print(
                f"[replay] violation sampling beta="
                f"{float(cfg.replay_viol_downweight):g} "
                f"uniform_mix_gamma={float(cfg.replay_viol_uniform_mix):g} "
                f"(p=(1-gamma)*normalize(w)+gamma*uniform; loss unchanged)",
                flush=True,
            )
        if log and bool(cfg.replay_proficiency):
            print(
                f"[replay] proficiency stratify prof={float(cfg.replay_frac_prof):g} "
                f"new={float(cfg.replay_frac_new):g} "
                f"ema_nmin={int(cfg.frontier_ema_nmin)} "
                f"ema_prof={float(cfg.frontier_ema_prof):g} "
                f"viol_cap={float(cfg.proficiency_viol_cap):g}",
                flush=True,
            )
    for step in range(start_step, int(cfg.max_steps)):
        grad_term_probe = None
        waist_mpc_param_grad_norm = None
        gb_stats: dict[str, float] = {}
        if _ShutdownFlag.requested:
            stopped = "signal"
            if rank == 0:
                emit({"event": "stop", "reason": "signal", "step": step})
                if log:
                    print(f"[signal] shutdown requested @ step {step}; "
                          "saving checkpoint and exiting", flush=True)
            break
        if deadline is not None and time.perf_counter() > deadline:
            # Wall clock, not step count, is the real budget on this box.
            stopped = "max_hours"
            emit({"event": "stop", "reason": "max_hours", "step": step})
            break
        phase_timer = _CudaPhaseTimer(
            enabled=(
                int(cfg.phase_timing_every) > 0
                and (step + 1) % int(cfg.phase_timing_every) == 0
            ),
            device=device,
        )
        updates_attempted += 1
        last_step = step + 1
        update_executed = False
        # Selector owns LR under line search (plan §7): no cosine write.
        if ls_rt is not None:
            known_search, search_reasons, ls_j = resolve_known_search(ls_rt)
            ls_direction_due = needs_line_search_direction(
                ls_rt, known_search=known_search, j=ls_j,
            )
            eps_reuse = float(ls_rt.ctrl.last_selected_epsilon or ls_rt.eta_ref)
            ls_rt.adapter.set_group_lrs(eps_reuse)
            lr = eps_reuse
        else:
            known_search, search_reasons, ls_j = False, [], 0
            ls_direction_due = False
            lr = cfg.lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr

        blend_now = cfg.schedule.torque_full_blend(step)
        if ls_rt is not None and cfg.cost.torque_physics != "full":
            if step == start_step:
                ls_rt.torque_full_blend_snapshot = blend_now
            prev_blend = float(ls_rt.torque_full_blend_snapshot)
            if prev_blend < 1.0e-3 and blend_now >= 1.0e-3:
                if ls_rt.cfg.force_search_on_al_change:
                    eps_after = reset_history_for_objective_change(
                        ls_rt.accept,
                        ls_rt.ctrl,
                        cfg=ls_rt.cfg,
                        eta_ref=ls_rt.eta_ref,
                    )
                    if ls_rt.cfg.moment_on == "step":
                        ls_rt.adapter.reset_step_moments()
                    if log and rank == 0:
                        print(
                            f"[line_search] torque full-RNEA ramp @ step {step + 1}: "
                            f"acceptance history reset, ε kept at {eps_after:.3g}, "
                            f"K=1, search forced",
                            flush=True,
                        )
                elif log and rank == 0:
                    print(
                        f"[line_search] torque full-RNEA ramp @ step {step + 1}: "
                        f"blend {prev_blend:.3g} -> {blend_now:.3g}",
                        flush=True,
                    )
            ls_rt.torque_full_blend_snapshot = blend_now
            ls_rt.objective_direction_fingerprint["torque_full_blend"] = float(
                blend_now
            )

        ls_rec = None
        critic_observation = None
        critic_extra_observations = []
        include_bptt_step = False
        sfx_ramp_step = 0.0
        suffix_replay_batch = None
        suffix_replay_offs = None
        suffix_replay_fill = 0.0
        ls_log: dict = {}
        commit_side_effects = True

        # ---- keep the MPC replay fed with the CURRENT policy ---------------
        if (
            voxel_state is None
            and timing_mpc_ring is None
            and step >= cfg.roller_start
        ):
            for _ in range(int(cfg.roller_ticks_per_step)):
                for b, cols in roller.harvest().items():
                    if b in replay.data:
                        if float(cfg.replay_viol_downweight) > 0.0:
                            w_h = _wtensors(
                                cfg.schedule.multipliers(step, stage_a),
                                al.weights(), device, dtype,
                                sched=cfg.schedule, joint_names=base.joint_names,
                                step=step, stage_a=stage_a, pen=al.pen,
                            )
                            vs = _harvest_replay_viol_score(
                                cols, b, pool, costs[b], w_h,
                            )
                            cols = dict(cols)
                            cols["viol_score"] = vs
                            roll_acc["replay_viol_score"] = (
                                roll_acc.get("replay_viol_score", 0.0)
                                + float(vs.sum())
                            )
                            roll_acc["replay_viol_n"] = (
                                roll_acc.get("replay_viol_n", 0.0)
                                + float(vs.numel())
                            )
                        replay.add(b, cols)
                    if b in post_hit.data:
                        n_ph = _add_live_post_hit(post_hit, b, cols)
                        roll_acc["post_hit"] = roll_acc.get("post_hit", 0.0) + float(n_ph)
                    if (
                        frontier_cloud is not None
                        and not bool(cfg.frontier_idle_far_cloud)
                    ):
                        _ingest_frontier_harvest(
                            frontier_cloud, b, cols,
                            include_miss=bool(cfg.frontier_include_miss),
                        )
                for k_, v_ in roller.tick().items():
                    roll_acc[k_] = roll_acc.get(k_, 0.0) + float(v_)

        # ---- draw a bucket-homogeneous minibatch ---------------------------
        replay_loc = None
        voxel_meta = None
        k_splice = 0
        if (
            timing_critic is not None
            and cfg.timing_critic.representation == "distributional"
            and cfg.timing_critic.anchor_curriculum
        ):
            # Anchor-first training owns the horizon: permanent anchor rows
            # stay at the longest time while critic-selected rows are the path
            # toward shorter timings and receive labels in the normal pass.
            bi = len(pool.buckets) - 1
        else:
            bi = int(rng.choice(len(pool.buckets), p=bucket_p))
        bucket = pool.buckets[bi]
        bs = cfg.batch_for(bucket)
        if frontier_cloud is not None and int(cfg.frontier_proposals) > 0:
            _frontier_propose_ingest(
                frontier_cloud, bucket, int(cfg.frontier_proposals),
                sdf=getattr(roller, "sdf", None), generator=gen,
            )
        batch = None
        use_replay = False
        if actor_superbatch_queue:
            batch = _detach_batch(actor_superbatch_queue.popleft(), device=device)
            use_replay = bool(batch["mpc_init"].any())
            roll_acc["actor_superbatch_cached_rows"] = (
                roll_acc.get("actor_superbatch_cached_rows", 0.0) + float(bs)
            )
        idle_far_cloud = (
            frontier_cloud is not None and bool(cfg.frontier_idle_far_cloud)
        )
        if (
            frontier_cloud is not None
            and not idle_far_cloud
            and int(cfg.frontier_sample_k) > 0
        ):
            batch = _draw_frontier_train_batch(
                pool, frontier_cloud, bucket, bs,
                k=int(cfg.frontier_sample_k), generator=gen,
                inv_frac=float(cfg.frontier_train_inv_frac),
            )
        if voxel_state is not None:
            batch, voxel_meta = voxel_state.sample_batch(
                bucket, bs, voxel_mpc_ring, generator=gen, rng=rng,
                yaw_i=yaw_i, step=step,
            )
        if batch is None and timing_mpc_ring is not None:
            draw_bs = int(bs) * int(cfg.actor_superbatch_factor)
            (
                base_n, anchor_n, boundary_n, interior_n, mpc_n,
            ) = _timing_primary_layout(draw_bs, cfg.timing_critic)
            live_n = sum(post_hit.size(b) for b in post_hit.data)
            kind_fracs = cfg.effective_kind_fracs(
                pool, extra_counts={KIND_POST_HIT: live_n},
            )
            timing_n = anchor_n + boundary_n + interior_n
            ramp_steps = int(cfg.timing_critic.anchor_curriculum_ramp_steps)
            # Keep the old branch structurally intact when the new feature is
            # disabled.  In particular, it makes exactly the old random draws.
            if ramp_steps <= 0:
                if bool(timing_critic.curriculum_open):
                    base_batch = _sample_fresh_batch(
                        pool, bucket, base_n, generator=gen,
                        kind_fracs=kind_fracs, live_post_hit=None,
                        short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                        noise=True,
                    )
                    timing_part = _expand_timing_base_tasks(
                        base_batch,
                        anchor_per_task=(
                            cfg.timing_critic.primary_anchor_per_task
                        ),
                        boundary_per_task=(
                            cfg.timing_critic.primary_boundary_per_task
                        ),
                        interior_per_task=(
                            cfg.timing_critic.primary_interior_per_task
                        ),
                    )
                    expanded_n = base_n
                    independent_anchor_n = 0
                else:
                    # Before calibration opens, maximize state/task diversity:
                    # every neural timing slot is a distinct horizon anchor.
                    timing_part = _sample_fresh_batch(
                        pool, bucket, timing_n, generator=gen,
                        kind_fracs=kind_fracs, live_post_hit=None,
                        short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                        noise=True,
                    )
                    timing_part["timing_primary_kind"] = torch.zeros(
                        timing_n, device=device, dtype=torch.long,
                    )
                    timing_part["timing_primary_group"] = torch.full(
                        (timing_n,), -1, device=device, dtype=torch.long,
                    )
                    expanded_n = 0
                    independent_anchor_n = timing_n
                ramp_fraction = float(bool(timing_critic.curriculum_open))
            else:
                expanded_n = _timing_curriculum_group_count(
                    base_tasks=base_n,
                    curriculum_open=bool(timing_critic.curriculum_open),
                    curriculum_open_actor_step=getattr(
                        timing_critic, "curriculum_open_actor_step", None,
                    ),
                    actor_step=int(step),
                    ramp_steps=ramp_steps,
                )
                # Superbatch splitting keeps every timing group intact and
                # needs an equal number of groups per future minibatch.
                # Quantize only the enabled ramp; the ordinary B=512 path is
                # unaffected.
                superbatch_factor = int(cfg.actor_superbatch_factor)
                expanded_n -= expanded_n % superbatch_factor
                independent_anchor_n = timing_n - expanded_n * (
                    int(cfg.timing_critic.primary_anchor_per_task)
                    + int(cfg.timing_critic.primary_boundary_per_task)
                    + int(cfg.timing_critic.primary_interior_per_task)
                )
                ramp_fraction = float(timing_critic.curriculum_ramp_fraction(step))
                if expanded_n == 0:
                    timing_part = _sample_fresh_batch(
                        pool, bucket, timing_n, generator=gen,
                        kind_fracs=kind_fracs, live_post_hit=None,
                        short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                        noise=True,
                    )
                    timing_part["timing_primary_kind"] = torch.zeros(
                        timing_n, device=device, dtype=torch.long,
                    )
                    timing_part["timing_primary_group"] = torch.full(
                        (timing_n,), -1, device=device, dtype=torch.long,
                    )
                elif independent_anchor_n == 0:
                    base_batch = _sample_fresh_batch(
                        pool, bucket, base_n, generator=gen,
                        kind_fracs=kind_fracs, live_post_hit=None,
                        short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                        noise=True,
                    )
                    timing_part = _expand_timing_base_tasks(
                        base_batch,
                        anchor_per_task=(
                            cfg.timing_critic.primary_anchor_per_task
                        ),
                        boundary_per_task=(
                            cfg.timing_critic.primary_boundary_per_task
                        ),
                        interior_per_task=(
                            cfg.timing_critic.primary_interior_per_task
                        ),
                    )
                else:
                    base_batch = _sample_fresh_batch(
                        pool, bucket, expanded_n, generator=gen,
                        kind_fracs=kind_fracs, live_post_hit=None,
                        short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                        noise=True,
                    )
                    expanded_part = _expand_timing_base_tasks(
                        base_batch,
                        anchor_per_task=(
                            cfg.timing_critic.primary_anchor_per_task
                        ),
                        boundary_per_task=(
                            cfg.timing_critic.primary_boundary_per_task
                        ),
                        interior_per_task=(
                            cfg.timing_critic.primary_interior_per_task
                        ),
                    )
                    anchor_part = _sample_fresh_batch(
                        pool, bucket, independent_anchor_n, generator=gen,
                        kind_fracs=kind_fracs, live_post_hit=None,
                        short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                        noise=True,
                    )
                    anchor_part["timing_primary_kind"] = torch.zeros(
                        independent_anchor_n, device=device, dtype=torch.long,
                    )
                    anchor_part["timing_primary_group"] = torch.full(
                        (independent_anchor_n,), -1, device=device,
                        dtype=torch.long,
                    )
                    timing_part = _cat_phase2_batches([expanded_part, anchor_part])

            parts = [timing_part]
            if mpc_n > 0 and timing_mpc_ring.size(bucket) > 0:
                idx = timing_mpc_ring.sample_idx(
                    bucket, mpc_n, generator=gen,
                )
                mpc_part = timing_mpc_ring.batch(
                    bucket, idx, generator=gen, noise=False,
                )
                mpc_part["mpc_init"] = torch.ones(
                    mpc_n, device=device, dtype=torch.bool,
                )
                mpc_part["timing_primary_kind"] = torch.ones(
                    mpc_n, device=device, dtype=torch.long,
                )
                mpc_part["timing_primary_group"] = torch.full(
                    (mpc_n,), -1, device=device, dtype=torch.long,
                )
                parts.append(mpc_part)
                use_replay = True
                roll_acc["timing_primary_mpc"] = (
                    roll_acc.get("timing_primary_mpc", 0.0) + float(mpc_n)
                )
                roll_acc["timing_mpc_unique_parent_frac"] = (
                    roll_acc.get("timing_mpc_unique_parent_frac", 0.0)
                    + timing_mpc_ring.last_unique_parent_frac
                )
                roll_acc["timing_mpc_advance_wp"] = (
                    roll_acc.get("timing_mpc_advance_wp", 0.0)
                    + timing_mpc_ring.last_advance_offset_mean
                )
                roll_acc["timing_mpc_advance_s"] = (
                    roll_acc.get("timing_mpc_advance_s", 0.0)
                    + timing_mpc_ring.last_advance_seconds_mean
                )
            elif mpc_n > 0:
                # A cold or failed preceding generation has no safe MPC rows.
                # Fill with independent anchors, not duplicate base tasks.
                fallback = _sample_fresh_batch(
                    pool, bucket, mpc_n, generator=gen,
                    kind_fracs=kind_fracs, live_post_hit=None,
                    short_wp=cfg.short_wp, short_boost=cfg.short_boost,
                    noise=True,
                )
                fallback["timing_primary_kind"] = torch.zeros(
                    mpc_n, device=device, dtype=torch.long,
                )
                fallback["timing_primary_group"] = torch.full(
                    (mpc_n,), -1, device=device, dtype=torch.long,
                )
                parts.append(fallback)
                roll_acc["timing_primary_mpc_fallback"] = (
                    roll_acc.get("timing_primary_mpc_fallback", 0.0)
                    + float(mpc_n)
                )
            batch = _cat_phase2_batches(parts)
            if int(cfg.actor_superbatch_factor) > 1:
                slices = _split_timing_superbatch(
                    batch, bs, int(cfg.actor_superbatch_factor), generator=gen,
                )
                batch = slices[0]
                actor_superbatch_queue.extend(
                    _detach_batch(part, device="cpu") for part in slices[1:]
                )
                del slices
                roll_acc["actor_superbatch_generated_rows"] = (
                    roll_acc.get("actor_superbatch_generated_rows", 0.0)
                    + float(draw_bs)
                )
                roll_acc["actor_superbatch_refresh"] = (
                    roll_acc.get("actor_superbatch_refresh", 0.0) + 1.0
                )
            else:
                order = torch.randperm(draw_bs, device=device, generator=gen)
                batch = _index_batch(batch, order)
            roll_acc["timing_primary_base_tasks"] = (
                roll_acc.get("timing_primary_base_tasks", 0.0)
                + float(expanded_n + independent_anchor_n)
            )
            roll_acc["timing_primary_anchor"] = (
                roll_acc.get("timing_primary_anchor", 0.0)
                + float(
                    expanded_n * int(cfg.timing_critic.primary_anchor_per_task)
                    + independent_anchor_n
                )
            )
            roll_acc["timing_primary_boundary"] = (
                roll_acc.get("timing_primary_boundary", 0.0)
                + float(
                    expanded_n * int(cfg.timing_critic.primary_boundary_per_task)
                )
            )
            roll_acc["timing_primary_interior"] = (
                roll_acc.get("timing_primary_interior", 0.0)
                + float(
                    expanded_n * int(cfg.timing_critic.primary_interior_per_task)
                )
            )
            selected_n = expanded_n * (
                int(cfg.timing_critic.primary_boundary_per_task)
                + int(cfg.timing_critic.primary_interior_per_task)
            )
            roll_acc["timing_curriculum_open"] = (
                roll_acc.get("timing_curriculum_open", 0.0)
                + float(timing_critic.curriculum_open)
            )
            roll_acc["timing_curriculum_open_step"] = (
                roll_acc.get("timing_curriculum_open_step", 0.0)
                + float(
                    -1 if getattr(
                        timing_critic, "curriculum_open_actor_step", None,
                    ) is None else timing_critic.curriculum_open_actor_step
                )
            )
            roll_acc["timing_curriculum_ramp_frac"] = (
                roll_acc.get("timing_curriculum_ramp_frac", 0.0)
                + ramp_fraction
            )
            roll_acc["timing_primary_expanded_tasks"] = (
                roll_acc.get("timing_primary_expanded_tasks", 0.0)
                + float(expanded_n)
            )
            roll_acc["timing_primary_independent_anchors"] = (
                roll_acc.get("timing_primary_independent_anchors", 0.0)
                + float(independent_anchor_n)
            )
            roll_acc["timing_primary_selected_frac_neural"] = (
                roll_acc.get("timing_primary_selected_frac_neural", 0.0)
                + float(selected_n) / max(float(timing_n), 1.0)
            )
            roll_acc["timing_primary_selected_frac_batch"] = (
                roll_acc.get("timing_primary_selected_frac_batch", 0.0)
                + float(selected_n) / max(float(draw_bs), 1.0)
            )
        if batch is None:
            use_replay = (
                bucket in replay.data
                and replay.size(bucket) >= bs
                and rng.uniform() < cfg.replay_frac(step)
            )
            if use_replay:
                idx = replay.sample_idx(
                    bucket, bs, generator=gen,
                    short_wp=cfg.short_wp,
                    short_boost=cfg.short_boost,
                    viol_downweight=cfg.replay_viol_downweight,
                    viol_uniform_mix=cfg.replay_viol_uniform_mix,
                    stratify=bool(cfg.replay_proficiency),
                    frac_prof=float(cfg.replay_frac_prof),
                    frac_new=float(cfg.replay_frac_new),
                    ema_nmin=int(cfg.frontier_ema_nmin),
                    ema_prof=float(cfg.frontier_ema_prof),
                )
                replay_loc = idx
                if (
                    rank == 0
                    and "viol_score" in replay.data[bucket]
                    and idx.numel() > 0
                ):
                    replay_viol_selected.append(
                        replay.data[bucket]["viol_score"].index_select(0, idx).detach()
                    )
                batch = replay.batch(bucket, idx, generator=gen, noise=False)
                batch["mpc_init"] = torch.ones(
                    bs, device=device, dtype=torch.bool,
                )
            else:
                live_n = sum(post_hit.size(b) for b in post_hit.data)
                kind_fracs = cfg.effective_kind_fracs(
                    pool, extra_counts={KIND_POST_HIT: live_n},
                )
                fresh_kw = {}
                if frontier_cloud is not None:
                    fresh_kw = dict(
                        frontier=frontier_cloud,
                        frontier_fresh_frac=(
                            0.0 if idle_far_cloud
                            else float(cfg.frontier_fresh_frac)
                        ),
                        frontier_idle_far=bool(idle_far_cloud),
                        frontier_inv_frac=float(cfg.frontier_train_inv_frac),
                    )
                batch = _sample_fresh_batch(
                    pool, bucket, bs, generator=gen, kind_fracs=kind_fracs,
                    live_post_hit=post_hit, short_wp=cfg.short_wp,
                    short_boost=cfg.short_boost, noise=True, **fresh_kw,
                )
            splice_k = (
                int(cfg.frontier_splice_k) if idle_far_cloud
                else int(cfg.frontier_sample_k)
            )
            if frontier_cloud is not None and splice_k > 0:
                fc_sz = frontier_cloud.size(bucket)
                k_splice = min(int(splice_k), bs, int(fc_sz))
                batch = _splice_frontier_batch(
                    batch, pool, frontier_cloud, bucket, bs,
                    k=splice_k, generator=gen,
                    inv_frac=float(cfg.frontier_train_inv_frac),
                    prefer_children=bool(idle_far_cloud),
                )
        if replay_loc is not None:
            if k_splice > 0:
                keep_n = bs - k_splice
                replay_loc = replay_loc[:keep_n]
                pad = torch.full(
                    (k_splice,), -1, device=device, dtype=torch.long,
                )
                batch["replay_loc"] = torch.cat([replay_loc, pad])
            else:
                batch["replay_loc"] = replay_loc
        if "mpc_init" not in batch:
            # Full frontier batches and legacy/static callers are never shifted
            # MPC plans unless they explicitly carry the provenance marker.
            batch["mpc_init"] = torch.zeros(
                int(batch["q0"].shape[0]), device=device, dtype=torch.bool,
            )
        timing_anchor_mask = torch.zeros(
            int(batch["q0"].shape[0]), device=device, dtype=torch.bool,
        )
        if (
            timing_critic is not None
            and cfg.timing_critic.representation == "distributional"
            and cfg.timing_critic.anchor_curriculum
        ):
            timing_anchor_mask = (
                batch["active"].any(dim=1) & ~batch["mpc_init"]
            )
            anchor_rows = timing_anchor_mask.nonzero(as_tuple=True)[0]
            if int(anchor_rows.numel()):
                anchor_batch = _index_batch(batch, anchor_rows)
                anchor_hit = torch.where(
                    anchor_batch["active"],
                    torch.full_like(anchor_batch["hit"], int(bucket)),
                    torch.full_like(anchor_batch["hit"], -1),
                )
                retime_batch_timing(anchor_batch, anchor_hit, pool)
                _copy_verified_timing_rows(
                    batch, anchor_batch, anchor_rows,
                    torch.arange(anchor_rows.numel(), device=device),
                )
        # Ring refresh must preserve the actual sampled context, not the
        # conditioning-only yaw jitter or any later init rewrite.
        voxel_parent_batch = (
            _detach_batch(batch) if voxel_state is not None else None
        )
        timing_parent_batch = (
            _detach_batch(batch) if timing_mpc_ring is not None else None
        )
        terminal_band_success = None
        timing_boundary_success = None
        actual_bs = _require_train_batch_size(batch, bs, bucket)
        batch_b_acc += float(actual_bs)
        bucket_step_acc[int(bucket)] = bucket_step_acc.get(int(bucket), 0) + 1

        # the horizon mix the OPTIMIZER actually sees, not the buffer's
        _hb = batch["hit"]
        _kb = torch.where(batch["active"], _hb.clamp_min(0),
                          torch.full_like(_hb, bucket + 1)).min(dim=1).values
        mix_acc[0] += float((_kb <= 3).float().mean())
        mix_acc[1] += float((_kb <= 5).float().mean())
        mix_acc[2] += float((_kb <= 8).float().mean())
        mix_n += 1

        hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
        lengths = group_lengths(hit_eff)
        strike = replace(batch["strike"], hit_index=hit_eff)
        w = _wtensors(
            cfg.schedule.multipliers(step, stage_a), al.weights(), device, dtype,
            sched=cfg.schedule, joint_names=base.joint_names,
            step=step, stage_a=stage_a, pen=al.pen,
        )
        w_confirmed = w
        if voxel_state is not None:
            w = _voxel_mask_al_weights(
                w, batch["al_ok"], probe_penalty=float(cfg.voxel_probe_penalty),
            )

        if sfx_fns:
            suffix_replay_batch, suffix_replay_fill = suffix_ring.sample(
                bucket, int(cfg.suffix_replay_batch),
                min_parents=int(cfg.suffix_replay_min_parents),
                generator=sfx_gen,
            )
            if suffix_replay_batch is not None:
                suffix_replay_offs = terminal_mixture_suffix_offsets(
                    suffix_replay_batch["hit"], suffix_replay_batch["active"],
                    bucket, generator=sfx_gen,
                    peak_frac=cfg.terminal_suffix_peak_frac,
                    long_frac=cfg.terminal_suffix_long_frac,
                    low_frac=cfg.terminal_suffix_low_frac,
                    peak_wp=cfg.terminal_suffix_peak_wp,
                    tail_tau=cfg.terminal_suffix_tail_tau,
                )

        if (
            ls_rt is not None
            and replay_mgr is not None
            and ls_direction_due
        ):
            # Pre-rewrite snapshot of init / yaw_now / strike (plan §14).
            ls_rec = replay_mgr.begin_record(
                primary_batch=batch,
                controls=FrozenControls(
                    weights=w,
                    al_lambda=dict(al.weights()),
                    pen=dict(al.pen),
                    sfx_ramp=0.0,
                    include_bptt=False,
                    step=step,
                    stage_a=stage_a,
                    bucket=bucket,
                    use_replay=use_replay,
                ),
                snapshot_mutable=True,
            )
            if suffix_replay_batch is not None:
                ls_rec.samples.suffix_offs = suffix_replay_offs.detach()
                ls_rec.samples.extras["suffix_replay_batch"] = suffix_replay_batch

        if cfg.yaw_now_jitter > 0.0 and batch["context"].yaw_now is not None:
            # Perturb only the CONDITIONING token; q0, the cost and the stored
            # aim are untouched, so the priced problem is identical and the
            # only thing being trained is invariance of the answer to yaw_now.
            yn = batch["context"].yaw_now
            noise = cfg.yaw_now_jitter * torch.randn(
                yn.shape, device=yn.device, dtype=yn.dtype, generator=jit_gen
            )
            if ls_rec is not None:
                ls_rec.samples.yaw_now_jitter = noise.detach()
            batch["context"].yaw_now = yn + noise

        phase_timer.mark("actor_forward")
        ctxm = (torch.autocast(device_type="cuda", dtype=amp) if amp is not None
                else torch.autocast(device_type="cuda", enabled=False))
        with ctxm:
            enc = encode_step(batch["context"])
            if cfg.waist_sp_from_head:
                # The learned set-point IS the ``waist_aim`` reference: one held
                # yaw per sample, broadcast over the knots.  Gradient flows both
                # ways through the term -- trajectory toward the anchor, anchor
                # toward the trajectory's mean -- which is the design
                # (``docs/waist_sp_head_plan.md``).
                sp = enc.waist_yaw_sp.float()
                strike = replace(
                    strike, waist_yaw_ref=sp.unsqueeze(1).expand(-1, bucket)
                )
            if cfg.waist_pr_sp_from_head:
                # Learned lean IS the upright-well reference: one held
                # (pitch, roll) per sample, broadcast over the knots.
                pr = enc.waist_pr_sp.float()
                strike = replace(
                    strike, waist_pr_ref=pr.unsqueeze(1).expand(-1, bucket, -1)
                )
            if enc.waist_pr_sp is not None and core.terminal_initializer is None:
                # Quintic prior: ease P/R to the announced set-point, same job
                # ``aim_yaw`` already does for the yaw column.  Detached inside
                # ``seed_waist_pr_to_sp`` -- the seed is not a gradient path.
                batch["init"] = seed_waist_pr_to_sp(
                    batch["init"], batch["q0"], batch["v0"], enc.waist_pr_sp,
                    joint_names=base.joint_names,
                    dt=float(getattr(pool.cfg, "dt", 0.02)),
                    ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
                )
            if (
                timing_critic is not None
                and cfg.timing_critic.mode in {"assist", "control"}
                and len(timing_critic.recent) >= int(cfg.timing_critic.min_assist_samples)
                and bool(getattr(timing_critic, "curriculum_open", True))
            ):
                if (
                    cfg.timing_critic.representation == "distributional"
                    and cfg.timing_critic.anchor_curriculum
                    and "timing_primary_kind" in batch
                ):
                    boundary_rows = (
                        batch["timing_primary_kind"] == 2
                    ).nonzero(as_tuple=True)[0]
                    interior_rows = (
                        batch["timing_primary_kind"] == 3
                    ).nonzero(as_tuple=True)[0]
                    obs, stats = _timing_assist_batch(
                        batch=batch, encoding=enc, strike=strike,
                        model=model, cost=costs[bucket], weights=w,
                        pool=pool, critic=timing_critic, cfg=cfg,
                        rows=boundary_rows, selection="boundary",
                        metric_prefix="timing_boundary",
                    )
                    if obs is not None:
                        critic_extra_observations.append(obs)
                    for key, value in stats.items():
                        roll_acc[key] = roll_acc.get(key, 0.0) + float(value)
                    group = batch.get("timing_primary_group")
                    grouped = (
                        torch.is_tensor(group)
                        and int(boundary_rows.numel()) > 0
                        and int(interior_rows.numel()) > 0
                        and int(group.index_select(0, boundary_rows).unique().numel())
                        == int(boundary_rows.numel())
                    )
                    if grouped:
                        stats = _timing_interiors_from_group_frontier(
                            batch=batch,
                            boundary_rows=boundary_rows,
                            interior_rows=interior_rows,
                            pool=pool,
                            generator=timing_critic.generator,
                        )
                        obs = None
                    else:
                        obs, stats = _timing_assist_batch(
                            batch=batch, encoding=enc, strike=strike,
                            model=model, cost=costs[bucket], weights=w,
                            pool=pool, critic=timing_critic, cfg=cfg,
                            rows=interior_rows, selection="interior",
                            metric_prefix="timing_interior",
                        )
                    if obs is not None:
                        critic_extra_observations.append(obs)
                    for key, value in stats.items():
                        roll_acc[key] = roll_acc.get(key, 0.0) + float(value)
                else:
                    assist_obs, assist_stats = _timing_assist_batch(
                        batch=batch, encoding=enc, strike=strike, model=model,
                        cost=costs[bucket], weights=w, pool=pool,
                        critic=timing_critic, cfg=cfg,
                    )
                    if assist_obs is not None:
                        critic_extra_observations.append(assist_obs)
                    for key, value in assist_stats.items():
                        roll_acc[key] = roll_acc.get(key, 0.0) + float(value)
                # Critic-selected timing rows changed their decoder masks and
                # segment endpoints.  The context encoding is still valid by
                # the enc_goal_tth=False contract checked above.
                hit_eff = effective_hits(batch["hit"], batch["active"], bucket)
                lengths = group_lengths(hit_eff)
                strike = replace(strike, hit_index=hit_eff)
            # Build the frozen-head seed after timing selection has fixed every
            # row. Legacy checkpoints retain their one geometric DLS correction.
            terminal_reseeded = _apply_terminal_initializer(
                core, batch, pool=pool, generator=gen,
                start_anchored_noise=bool(cfg.terminal_init_noise),
                noise_ramp_steps=int(cfg.terminal_init_noise_ramp_steps),
            )
            if terminal_reseeded and enc.waist_pr_sp is not None:
                batch["init"] = seed_waist_pr_to_sp(
                    batch["init"], batch["q0"], batch["v0"], enc.waist_pr_sp,
                    joint_names=base.joint_names,
                    dt=float(getattr(pool.cfg, "dt", 0.02)),
                    ease_wp=getattr(pool.cfg, "waist_upright_ease_wp", None),
                )
            if not terminal_reseeded:
                fresh_active = batch["active"] & ~batch["mpc_init"].unsqueeze(-1)
                batch["init"] = correct_quintic_init_arms(
                    costs[bucket].fk, batch["init"], batch["q0"], batch["hit"],
                    batch["context"].goal_pos,
                    active=fresh_active, steps=1,
                    q_bounds=(pool.q_lo, pool.q_hi),
                    joint_names=base.joint_names, qd0=batch["v0"],
                    dt=float(getattr(pool.cfg, "dt", 0.02)),
                )
            if timing_parent_batch is not None:
                all_rows = torch.arange(
                    int(batch["q0"].shape[0]), device=device,
                )
                _copy_verified_timing_rows(
                    timing_parent_batch, batch, all_rows, all_rows,
                )
            if ls_rec is not None and timing_critic is not None:
                # The critic makes one timing selection per outer update.
                # Candidate optimizer steps replay those finalized rows and
                # never rerun the critic or resample its timing rays.
                _refresh_timing_line_search_replay(ls_rec, batch)
            grad_term_probe = None
            if (
                cfg.grad_term_every > 0
                and (step + 1) % cfg.grad_term_every == 0
                and rank == 0
                and not (ls_rt is not None and known_search)
            ):
                grad_term_probe = measure_cost_term_grad_norms(
                    model, costs[bucket], cfg,
                    init=batch["init"], state=batch["state"], enc=enc,
                    masks=batch["masks"], q0=batch["q0"], v0=batch["v0"],
                    strike=strike, lengths=lengths, weights=w,
                )
                if not cfg.grad_balance:
                    grad_term_probe.update(measure_group_grad_cosines(
                        model, costs[bucket],
                        init=batch["init"], state=batch["state"], enc=enc,
                        masks=batch["masks"], q0=batch["q0"], v0=batch["v0"],
                        strike=strike, lengths=lengths, weights=w,
                        w_cost=float(cfg.w_cost),
                    ))
            loss_aux = None
            ct = None
            torque_term_mean = None
            torque_refresh_due = bool(
                cfg.torque_grad_cap > 0.0
                and not cfg.grad_balance
                and (
                    not torque_grad_cache_initialized
                    or torque_grad_visits
                    % int(cfg.torque_grad_cap_refresh) == 0
                )
            )
            torque_step_scale = (
                1.0 if torque_refresh_due else torque_grad_cached_scale
            )
            if cfg.torque_grad_cap > 0.0:
                torque_grad_visits += 1
            step_out = step_fns[bucket](
                batch["init"], batch["state"], batch["q0"], batch["v0"],
                enc, batch["masks"], strike, lengths, w,
                torch.as_tensor(
                    torque_step_scale - 1.0,
                    device=device, dtype=torch.float32,
                ),
            )
            if cfg.grad_balance:
                loss, metrics, _traj, gate, ct, loss_aux = step_out
            elif cfg.torque_grad_cap > 0.0:
                loss, metrics, _traj, gate, torque_term_mean = step_out
            else:
                loss, metrics, _traj, gate = step_out
            phase_timer.mark("aux_forward")
            hard_ok = _hard_feasibility_ok_from_gate(gate)
            bellman_parent_ok = (
                _bellman_parent_ingress_ok(gate, cfg.timing_critic)
                if sfx_fns else None
            )
            if timing_critic is not None:
                critic_observation = (
                    build_critic_features(
                        q0=batch["q0"], qd0=batch["v0"],
                        goal_pos=batch["context"].goal_pos,
                        goal_tth=batch["context"].goal_tth,
                    ),
                    gate.detach(),
                )
                if (
                    cfg.timing_critic.representation == "distributional"
                    and "timing_primary_kind" in batch
                    and "timing_selected_pred_manageable" in batch
                ):
                    # The ordinary actor/cost pass is the sole exact evaluator
                    # for critic-selected collocations.  Join its labels with
                    # the admission metadata for telemetry; there is no
                    # pre-admission actor forward.
                    timing_residual = residuals_from_gate(
                        gate.detach(), cfg.timing_critic,
                    )
                    timing_normalized = robust_transform(
                        timing_residual, cfg.timing_critic.scales,
                    )
                    timing_exact_ok = (
                        torch.isfinite(timing_normalized).all(dim=1)
                        & (
                            timing_normalized
                            <= float(
                                cfg.timing_critic.acceptable_normalized_violation
                            )
                        ).all(dim=1)
                    )
                    timing_kind_all = batch["timing_primary_kind"]
                    timing_nearest = torch.where(
                        batch["active"].bool(), batch["hit"],
                        torch.full_like(batch["hit"], int(bucket)),
                    ).min(dim=1).values
                    mpc_rows_all = timing_kind_all == 1
                    peak_wp = int(cfg.terminal_suffix_peak_wp)
                    bands = (
                        ("below_freeze", 1, peak_wp - 1),
                        ("freeze_peak", peak_wp, peak_wp),
                        ("mid", peak_wp + 1, min(19, int(bucket))),
                        ("long", 20, int(bucket)),
                    )
                    for band_name, lo, hi in bands:
                        band = (
                            mpc_rows_all
                            & (timing_nearest >= lo)
                            & (timing_nearest <= hi)
                        )
                        roll_acc[f"timing_mpc_band_{band_name}_rows"] = (
                            roll_acc.get(f"timing_mpc_band_{band_name}_rows", 0.0)
                            + float(band.sum())
                        )
                        roll_acc[f"timing_mpc_band_{band_name}_manageable"] = (
                            roll_acc.get(
                                f"timing_mpc_band_{band_name}_manageable", 0.0,
                            ) + float((band & timing_exact_ok).sum())
                        )
                    terminal_band = (
                        mpc_rows_all
                        & (timing_nearest >= 1)
                        & (
                            timing_nearest
                            <= peak_wp
                        )
                    )
                    terminal_band_success = timing_exact_ok[terminal_band]
                    timing_boundary_success = timing_exact_ok[
                        timing_kind_all == 2
                    ]
                    timing_pred = batch[
                        "timing_selected_pred_manageable"
                    ].bool()
                    timing_advanced = batch["timing_advanced"].bool()
                    for timing_kind, timing_prefix in (
                        (2, "timing_boundary"),
                        (3, "timing_interior"),
                    ):
                        timing_rows = (
                            batch["timing_primary_kind"] == timing_kind
                        )
                        timing_true = timing_exact_ok & timing_rows
                        timing_selected = timing_pred & timing_rows
                        timing_tp = timing_selected & timing_exact_ok
                        timing_accepted_advanced = (
                            timing_true & timing_advanced
                        )
                        timing_n = int(timing_rows.sum())
                        timing_selected_n = int(timing_selected.sum())
                        timing_accepted_n = int(timing_true.sum())
                        for key, value in {
                            f"{timing_prefix}_accepted": timing_accepted_n,
                            f"{timing_prefix}_rejected": (
                                timing_n - timing_accepted_n
                            ),
                            f"{timing_prefix}_selected_true_manageable": (
                                timing_accepted_n
                            ),
                            f"{timing_prefix}_selected_false_manageable": int(
                                (timing_selected & ~timing_exact_ok).sum()
                            ),
                            f"{timing_prefix}_selected_missed_manageable": int(
                                (~timing_pred & timing_true).sum()
                            ),
                            f"{timing_prefix}_selected_true_positive": int(
                                timing_tp.sum()
                            ),
                            f"{timing_prefix}_selected_precision": (
                                int(timing_tp.sum())
                                / max(timing_selected_n, 1)
                            ),
                            f"{timing_prefix}_accepted_advanced": int(
                                timing_accepted_advanced.sum()
                            ),
                        }.items():
                            roll_acc[key] = (
                                roll_acc.get(key, 0.0) + float(value)
                            )
                        accepted_active = (
                            timing_accepted_advanced.unsqueeze(1)
                            & batch["active"].bool()
                        )
                        accepted_tth = batch[
                            "context"
                        ].goal_tth[accepted_active].float()
                        if int(accepted_tth.numel()):
                            roll_acc[
                                f"{timing_prefix}_accepted_advanced_tth_mean_s"
                            ] = roll_acc.get(
                                f"{timing_prefix}_accepted_advanced_tth_mean_s",
                                0.0,
                            ) + float(accepted_tth.mean())
                            roll_acc[
                                f"{timing_prefix}_accepted_advanced_tth_p50_s"
                            ] = roll_acc.get(
                                f"{timing_prefix}_accepted_advanced_tth_p50_s",
                                0.0,
                            ) + float(accepted_tth.median())
            if voxel_state is not None and voxel_meta is not None:
                probe_stats = voxel_state.record_probe_results(
                    voxel_meta, hard_ok, step=step,
                )
                roll_acc["voxel_promoted"] = (
                    roll_acc.get("voxel_promoted", 0.0)
                    + float(probe_stats["promoted"])
                )
                roll_acc["voxel_failed_unknown"] = (
                    roll_acc.get("voxel_failed_unknown", 0.0)
                    + float(probe_stats["failed_unknown"])
                )
            loss_tc = torch.zeros((), device=device)
            if tc_fns:
                tc_rows = (
                    _mpc_aux_rows(batch["mpc_init"], cfg.tc_frac)
                    if cfg.voxel_pool else
                    torch.arange(
                        max(int(cfg.tc_frac * bs), 1), device=batch["q0"].device,
                    )
                )
                n_tc = int(tc_rows.numel())
            if tc_fns and n_tc:
                masks2 = pool.masks_for(
                    (batch["hit"].index_select(0, tc_rows) - 1).clamp_min(0),
                    batch["active"].index_select(0, tc_rows), bucket,
                )
                kp, kd, _ki = sim.sample_gains(n_tc, generator=gen)
                if ls_rec is not None:
                    ls_rec.samples.tc_kp = kp.detach()
                    ls_rec.samples.tc_kd = kd.detach()
                enc_sub = _index_encoded(enc, tc_rows)
                masks_sub = _index_batch({
                    "bucket": bucket, "q0": batch["q0"], "masks": batch["masks"],
                }, tc_rows)["masks"]
                loss_tc, _n = tc_fns[bucket](
                    batch["init"].index_select(0, tc_rows),
                    batch["state"].index_select(0, tc_rows),
                    batch["q0"].index_select(0, tc_rows),
                    batch["v0"].index_select(0, tc_rows), enc_sub, masks_sub,
                    masks2, kp, kd,
                )
            loss_bptt = torch.zeros((), device=device)
            include_bptt_step = bool(
                bptt_fn is not None and rng.uniform() < cfg.bptt_frac(step)
            )
            if ls_rec is not None:
                ls_rec.controls.include_bptt = include_bptt_step
            if include_bptt_step:
                loss_bptt = _bptt_term(
                    pool, (
                        voxel_mpc_ring if voxel_state is not None
                        else timing_mpc_ring if timing_mpc_ring is not None
                        else replay
                    ),
                    bptt_fn, sim, cfg, bptt_bucket, device, dtype,
                    gen, rng, w_confirmed, model=model,
                )
            loss_sfx_cost = torch.zeros((), device=device)
            loss_sfx_tie = torch.zeros((), device=device)
            loss_mpc_tie = torch.zeros((), device=device)
            loss_waist_mpc_tie = torch.zeros((), device=device)
            waist_mpc_tie_grad = torch.zeros((), device=device)
            if cfg.w_mpc_init_tie > 0.0:
                loss_mpc_tie, _mpc_tie_elems = mpc_init_tie_loss(
                    _traj, batch["seed"], batch["masks"].elem_mask,
                    batch["mpc_init"],
                )
            if cfg.w_waist_yaw_mpc_tie > 0.0:
                loss_waist_mpc_tie, _, waist_mpc_tie_grad = (
                    waist_yaw_mpc_init_tie_loss(
                        _traj, batch["seed"], batch["masks"].elem_mask,
                        batch["mpc_init"], yaw_index=yaw_i,
                    )
                )
            sfx_ramp = (
                _smoothstep(min((step + 1) / max(cfg.suffix_warmup, 1), 1.0))
                * float(suffix_replay_fill)
                if suffix_replay_batch is not None else 0.0
            )
            sfx_ramp_step = float(sfx_ramp)
            if ls_rec is not None:
                ls_rec.controls.sfx_ramp = sfx_ramp_step
                ls_rec.controls.weights = w
            if suffix_replay_batch is not None and sfx_ramp > 0.0:
                loss_sfx_tie, suffix_tie_elems, *_suffix_diagnostics = (
                    _suffix_distill_loss(
                        suffix_replay_batch, suffix_replay_offs,
                        bucket=bucket, encode_step=encode_step,
                        suffix_fn=sfx_fns[bucket], pool=pool,
                        microbatch=int(cfg.suffix_microbatch),
                    )
                )
                roll_acc["suffix_replay_rows"] = (
                    roll_acc.get("suffix_replay_rows", 0.0)
                    + float(cfg.suffix_replay_batch)
                )
                roll_acc["suffix_replay_tie_elems"] = (
                    roll_acc.get("suffix_replay_tie_elems", 0.0)
                    + float(suffix_tie_elems.detach())
                )
            extra = (
                cfg.w_tc * loss_tc + cfg.w_bptt * loss_bptt
                + cfg.w_mpc_init_tie * loss_mpc_tie
                + cfg.w_waist_yaw_mpc_tie * loss_waist_mpc_tie
                + sfx_ramp * cfg.w_suffix_tie * loss_sfx_tie
            )
            dls_stats: dict[str, torch.Tensor] = {}
            if (
                cfg.dls_terminal_frac > 0.0
                and (
                    cfg.w_dls_terminal_cost > 0.0
                    or cfg.w_dls_terminal_tie > 0.0
                )
            ):
                dls_cost, dls_tie, dls_torque, dls_stats = (
                    dls_terminal_lookahead_loss(
                        _traj, batch["q0"], batch["v0"], strike, lengths,
                        w, costs[bucket], cfg,
                    )
                )
                extra = extra + (
                    cfg.w_dls_terminal_cost * dls_cost
                    + cfg.w_dls_terminal_tie * dls_tie
                )
                if cfg.torque_grad_cap > 0.0:
                    # The cached rotation must apply to every torque gradient,
                    # including the lookahead.  On refresh steps the combined
                    # scalar below is handed to the exact budget measurement.
                    extra = extra + (
                        (float(torque_step_scale) - 1.0)
                        * cfg.w_dls_terminal_cost * dls_torque
                    )
                    if torque_term_mean is not None and cfg.w_cost > 0.0:
                        torque_term_mean = torque_term_mean + (
                            cfg.w_dls_terminal_cost / cfg.w_cost
                        ) * dls_torque
                for key, value in dls_stats.items():
                    roll_acc[key] = roll_acc.get(key, 0.0) + float(value.detach())
            if cfg.w_waist_sp_tether > 0.0 and enc.waist_yaw_res is not None:
                # res = sp - aim_yaw by construction, so this IS the tether to
                # the analytic law.  Unramped: exactly zero at init.
                res = enc.waist_yaw_res.float()
                extra = extra + cfg.w_waist_sp_tether * res.pow(2).mean()
            if cfg.w_waist_pr_sp_tether > 0.0 and enc.waist_pr_res is not None:
                # res = sp_pr (base is upright 0).  Unramped: exactly zero at init.
                pr = enc.waist_pr_res.float()
                extra = extra + cfg.w_waist_pr_sp_tether * pr.pow(2).mean()
            if cfg.w_waist_pr_sp_hold > 0.0 and enc.waist_pr_sp is not None:
                # Stay near the pose AT ENCODE, not the last announcement.
                # stopgrad(q0): this is a prior on the head, not a path into
                # the plant.  Measured lean is not an encoder input.
                q_pr = batch["q0"][:, pr_idx].detach().float()
                d_pr = enc.waist_pr_sp.float() - q_pr
                extra = extra + cfg.w_waist_pr_sp_hold * d_pr.pow(2).mean()
            if cfg.w_waist_pr_sp_hold_band > 0.0 and enc.waist_pr_sp is not None:
                # Decoder band is |q - sp| ≤ rad.  Equal tether+hold leaves
                # |sp - q0| = |q0|/2, which is already past rad on a rail
                # q0.  This hinge makes that illegal; tether still pulls
                # toward 0 inside the window.
                extra = extra + cfg.w_waist_pr_sp_hold_band * waist_pr_hold_band_loss(
                    enc.waist_pr_sp.float(),
                    batch["q0"][:, pr_idx],
                    band_rad=float(cfg.cost.waist_pr_band_rad),
                    pr_limit=pr_limit,
                )
            if (
                cfg.w_waist_pr_sp_distill > 0.0
                and enc.waist_pr_sp is not None
            ):
                # Encoder copies decoder strike-knot P/R only on plans that
                # already earned it.  q_hit is stopgrad; the gate is
                # detached.  Ungated rows contribute 0; decoder sees none
                # of this term.
                q_hit = waist_pr_hit_teacher(
                    _traj, strike.hit_index, batch["active"], pr_idx,
                )
                mask = waist_pr_distill_gate(
                    gate[:, 0], gate[:, 1], gate[:, 2], gate[:, 3],
                    q_hit, batch["active"],
                    pr_limit=pr_limit,
                    hit_m=float(cfg.pr_distill_hit_m),
                )
                loss_pr_d = waist_pr_distill_loss(
                    enc.waist_pr_sp.float(), q_hit.float(), mask,
                )
                extra = extra + cfg.w_waist_pr_sp_distill * loss_pr_d
                pr_d_n_rows = float(mask.sum().detach())
                pr_d_delta = float(
                    ((q_hit - enc.waist_pr_sp.detach()).abs()
                     .mean(dim=-1) * mask).sum().detach()
                    / max(pr_d_n_rows, 1.0)
                )
                pr_distill_frac_acc += pr_d_n_rows / max(float(mask.numel()), 1.0)
                pr_distill_delta_acc += pr_d_delta
                pr_distill_n += 1
            if cfg.w_waist_sp_hold > 0.0 and enc.waist_yaw_sp is not None:
                q_yaw = batch["q0"][:, yaw_i].detach().float()
                d_yaw = enc.waist_yaw_sp.float() - q_yaw
                extra = extra + cfg.w_waist_sp_hold * d_yaw.pow(2).mean()
            if (
                cfg.w_waist_sp_smooth > 0.0
                and (enc.waist_yaw_sp is not None or enc.waist_pr_sp is not None)
            ):
                # Next-hit re-encode: price the jump the intra-segment terms
                # never see.  Guarded on the weight so a 0-default A/B is
                # bit-identical (no extra encode, no host-side aim solve).
                loss_sm, sm_yaw, sm_pr, sm_rows = _waist_sp_smooth_term(
                    model=model, ctx=batch["context"], enc=enc,
                    waist_cfg=waist_cfg,
                )
                extra = extra + cfg.w_waist_sp_smooth * loss_sm
                if sm_rows > 0:
                    sm_acc[0] += sm_yaw
                    sm_acc[1] += sm_pr
                    sm_n += 1
            # ---- zeroth-order search distillation on the set-point head ----
            # Guarded on BOTH knobs and on the head existing, and it draws no
            # random numbers at all (the slice is the leading rows), so at the
            # default ``waist_sp_search_frac = 0.0`` this block is not merely
            # cheap -- it is unreachable, and an A/B partner sees the identical
            # batch / noise / gain / jitter stream.
            if (
                cfg.waist_sp_search_frac > 0.0 and cfg.w_waist_sp_search > 0.0
                and enc.waist_yaw_sp is not None
            ):
                n_ss = min(max(int(cfg.waist_sp_search_frac * bs), 1), bs)
                loss_sp_search, ss_diag = _waist_sp_search_term(
                    model=model, cost=costs[bucket], cfg=cfg, enc=enc, batch=batch,
                    strike=strike, lengths=lengths, bucket=bucket, weights=w,
                    rows=slice(0, n_ss),
                )
                extra = extra + cfg.w_waist_sp_search * loss_sp_search
                for i_, v_ in enumerate(ss_diag):
                    ss_acc[i_] += v_
                ss_n += 1
        total = loss + extra
        phase_timer.mark("backward")
        if enc.waist_yaw_res is not None:
            sp_res_acc += float(enc.waist_yaw_res.detach().abs().mean())
            sp_res_n += 1
        if enc.waist_pr_sp is not None:
            pr_sp_acc += float(enc.waist_pr_sp.detach().abs().mean())
            pr_sp_n += 1
            q_pr = batch["q0"][:, pr_idx].detach()
            pr_hold_acc += float(
                (enc.waist_pr_sp.detach().float() - q_pr.float()).abs().mean()
            )
            pr_hold_n += 1
            pr_hold_band_acc += float(
                waist_pr_hold_band_excess(
                    enc.waist_pr_sp.detach(), q_pr,
                    band_rad=float(cfg.cost.waist_pr_band_rad),
                    pr_limit=pr_limit,
                ).mean()
            )
            pr_hold_band_n += 1
        if enc.waist_yaw_sp is not None:
            q_yaw = batch["q0"][:, yaw_i].detach()
            yaw_hold_acc += float(
                (enc.waist_yaw_sp.detach().float() - q_yaw.float()).abs().mean()
            )
            yaw_hold_n += 1
        if sfx_fns:
            sfx_acc[0] += float(loss_sfx_cost.detach())
            sfx_acc[1] += float(loss_sfx_tie.detach())
            sfx_n += 1
        if cfg.w_mpc_init_tie > 0.0:
            mpc_tie_acc += float(loss_mpc_tie.detach())
            mpc_tie_frac_acc += float(batch["mpc_init"].float().mean())
            mpc_tie_n += 1
        if cfg.w_waist_yaw_mpc_tie > 0.0:
            waist_mpc_tie_acc += float(loss_waist_mpc_tie.detach())
            waist_mpc_tie_grad_acc += float(
                cfg.w_waist_yaw_mpc_tie * waist_mpc_tie_grad.detach()
            )
            waist_mpc_tie_n += 1
            if rank == 0 and (step + 1) % cfg.log_every == 0:
                probe_grads = torch.autograd.grad(
                    cfg.w_waist_yaw_mpc_tie * loss_waist_mpc_tie,
                    trainable_params(model), retain_graph=True, allow_unused=True,
                )
                sq = sum(
                    grad.detach().double().square().sum()
                    for grad in probe_grads if grad is not None
                )
                waist_mpc_param_grad_norm = float(sq.sqrt()) if not isinstance(sq, int) else 0.0

        if not torch.isfinite(total):
            # Never let one pathological batch poison the weights or the
            # multipliers; count it instead, and report the count at the end.
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            if ls_rt is not None:
                latch_triggers(ls_rt.ctrl, ["nonfinite_baseline"])
            emit({"event": "skip", "step": step + 1, "loss": float(total.detach())})
            continue

        if ls_rt is not None and replay_mgr is not None and ls_rec is not None:
            replay_mgr.finalize_baseline(ls_rec)

        opt.zero_grad(set_to_none=True)
        g_objective = None
        if cfg.grad_balance in (
            "al_gradnorm", "al_pcgrad", "al_pcgrad_output",
        ):
            if ls_rt is not None:
                raise RuntimeError(
                    "line_search requires grad_balance=''; plain total.backward() "
                    "(plan §1–3); surgery path must not run"
                )
            if ct is None or loss_aux is None:
                raise RuntimeError(
                    f"grad_balance={cfg.grad_balance} requires cost graph from step_fn"
                )
            _gb_common = dict(
                w_cost=float(cfg.w_cost),
                torque_gamma=float(cfg.grad_balance_torque_gamma),
                beta=float(cfg.grad_balance_beta),
                torque_cap=float(cfg.grad_balance_tau_torque_cap),
                shape_floor=float(cfg.grad_balance_shape_floor),
                torque_grad_scale=float(cfg.torque_grad_scale),
                # Pairwise cosine diagnostics require flattening the five group
                # gradients and synchronizing them to host.  They are only
                # consumed on train log rows, so do not pay that cost every step.
                compute_cosines=(
                    rank == 0 and (step + 1) % cfg.log_every == 0
                ),
            )
            fast_balanced_loss = None
            if cfg.grad_balance == "al_pcgrad_output":
                visits = gb_visits_by_bucket.get(bucket, 0)
                norm_ema = gb_norm_ema_by_bucket.get(bucket, {})
                refresh = (
                    not norm_ema
                    or visits % int(cfg.grad_balance_refresh) == 0
                )
                gb_visits_by_bucket[bucket] = visits + 1
                if refresh:
                    # The group VJPs stop at the trajectory, so a compiled
                    # MODEL is safe here.  A compiled COST still cannot consume
                    # the storage-less BatchedTensor from is_grads_batched.
                    traj_grad, gb_stats = output_space_al_pcgrad(
                        _traj, ct, al,
                        waist_floor=float(cfg.grad_balance_waist_floor),
                        batched_vjp=not (
                            cfg.compile_mode is not None and cfg.compile_cost
                        ),
                        **_gb_common,
                    )
                    refresh_norms = torch.stack([
                        gb_stats[f"gb_norm_{name}"]
                        for name in GRAD_BALANCE_GROUPS
                    ])
                    refresh_finite = bool(
                        torch.isfinite(refresh_norms).all().item()
                    )
                    if refresh_finite:
                        torch.autograd.backward(
                            _traj, grad_tensors=traj_grad, retain_graph=True,
                        )
                        norm_ema = update_group_norm_ema(
                            norm_ema, gb_stats,
                            decay=float(cfg.grad_balance_norm_ema),
                        )
                        gb_norm_ema_by_bucket[bucket] = norm_ema
                        gb_stats["gb_refresh"] = 1.0
                        gb_stats["gb_refresh_failed"] = 0.0
                    elif norm_ema:
                        # Some plant derivatives can be non-finite even when
                        # their forward costs are finite.  Never poison the EMA:
                        # fall back to the last valid scalar surrogate.
                        surrogate_common = dict(_gb_common)
                        surrogate_common.pop("compute_cosines")
                        fast_balanced_loss, gb_stats = (
                            output_space_gradnorm_surrogate(
                                ct, al, norm_ema,
                                waist_floor=float(cfg.grad_balance_waist_floor),
                                **surrogate_common,
                            )
                        )
                        traj_grad, gb_stats = (
                            output_space_protected_surrogate_gradient(
                                _traj, fast_balanced_loss, ct, gb_stats,
                                w_cost=float(cfg.w_cost),
                                batched_vjp=not (
                                    cfg.compile_mode is not None
                                    and cfg.compile_cost
                                ),
                            )
                        )
                        torch.autograd.backward(
                            _traj, grad_tensors=traj_grad, retain_graph=True,
                        )
                        fast_balanced_loss = None
                        gb_stats["gb_refresh_failed"] = 1.0
                    else:
                        # First visit to this bucket: train auxiliary/extra
                        # terms and retry exact norm initialization next visit.
                        fast_balanced_loss = loss_aux.new_zeros(())
                        gb_stats["gb_refresh"] = 0.0
                        gb_stats["gb_refresh_failed"] = 1.0
                else:
                    surrogate_common = dict(_gb_common)
                    surrogate_common.pop("compute_cosines")
                    fast_balanced_loss, gb_stats = output_space_gradnorm_surrogate(
                        ct, al, norm_ema,
                        waist_floor=float(cfg.grad_balance_waist_floor),
                        **surrogate_common,
                    )
                    traj_grad, gb_stats = (
                        output_space_protected_surrogate_gradient(
                            _traj, fast_balanced_loss, ct, gb_stats,
                            w_cost=float(cfg.w_cost),
                            batched_vjp=not (
                                cfg.compile_mode is not None
                                and cfg.compile_cost
                            ),
                        )
                    )
                    torch.autograd.backward(
                        _traj, grad_tensors=traj_grad, retain_graph=True,
                    )
                    fast_balanced_loss = None
                    gb_stats["gb_refresh_failed"] = 0.0
                gb_stats["gb_refresh_interval"] = float(
                    cfg.grad_balance_refresh
                )
                for name, value in norm_ema.items():
                    gb_stats[f"gb_norm_ema_{name}"] = value.detach()
            elif cfg.grad_balance == "al_pcgrad":
                gb_stats = apply_al_pcgrad(
                    trainable_params(model), ct, al,
                    waist_floor=float(cfg.grad_balance_waist_floor),
                    # Parameter VJPs traverse a compiled model backward, whose
                    # AOT wrapper cannot consume a BatchedTensor.
                    batched_vjp=(cfg.compile_mode is None),
                    **_gb_common,
                )
            else:
                _params = trainable_params(model)
                gb_stats = apply_al_gradnorm(
                    _params, ct, al,
                    batched_vjp=(cfg.compile_mode is None),
                    **_gb_common,
                )
            if fast_balanced_loss is not None:
                (
                    fast_balanced_loss
                    + float(cfg.w_aux) * loss_aux
                    + extra
                ).backward()
            else:
                (float(cfg.w_aux) * loss_aux).backward(retain_graph=True)
                if extra.requires_grad:
                    extra.backward()
        else:
            curvature_delta = None
            if cfg.curvature_direct_frac > 0.0:
                curvature_delta, torque_term_mean, curvature_stats = (
                    curvature_direct_vjp_correction(
                        total, _traj, batch["q0"], strike, lengths, w,
                        costs[bucket], cfg,
                        torque_term_mean=(
                            torque_term_mean if torque_refresh_due else None
                        ),
                    )
                )
                for key, value in curvature_stats.items():
                    roll_acc[key] = roll_acc.get(key, 0.0) + float(value.detach())
            total.backward(
                retain_graph=torque_refresh_due or curvature_delta is not None
            )
            if ls_rt is not None and ls_direction_due:
                # Armijo needs the true scalar-objective gradient.  In direct
                # curvature mode this snapshot must precede the output-space
                # VJP correction; the live p.grad then becomes Adam's proposed
                # direction, which SaLSa checks against the true objective.
                g_objective = snapshot_g_preclip(ls_rt.adapter._params)
            if curvature_delta is not None:
                torch.autograd.backward(
                    _traj, grad_tensors=curvature_delta,
                    retain_graph=torque_refresh_due,
                )
            if torque_refresh_due and torque_term_mean is not None:
                tgc = clip_torque_gradient_budget(
                    trainable_params(model),
                    torque_term_mean,
                    w_cost=float(cfg.w_cost),
                    cap=float(cfg.torque_grad_cap),
                )
                _accumulate_torque_grad_cap_stats(roll_acc, tgc)
                roll_acc["torque_grad_cap_refresh"] = (
                    roll_acc.get("torque_grad_cap_refresh", 0.0) + 1.0
                )
                measured_scale = tgc.get("torque_grad_cap_scale")
                if measured_scale is not None and math.isfinite(
                    float(measured_scale)
                ):
                    if torque_grad_cache_initialized:
                        decay = float(cfg.torque_grad_cap_scale_ema)
                        torque_grad_cached_scale = (
                            decay * torque_grad_cached_scale
                            + (1.0 - decay) * float(measured_scale)
                        )
                    else:
                        torque_grad_cached_scale = float(measured_scale)
                    torque_grad_cache_initialized = True
            elif cfg.torque_grad_cap > 0.0:
                roll_acc["torque_grad_cap_reuse"] = (
                    roll_acc.get("torque_grad_cap_reuse", 0.0) + 1.0
                )
                roll_acc["torque_grad_cap_cached_scale"] = (
                    roll_acc.get("torque_grad_cap_cached_scale", 0.0)
                    + float(torque_grad_cached_scale)
                )

        g_preclip = None
        g_for_lbfgs = None
        if ls_rt is not None and ls_direction_due:
            g_preclip = snapshot_g_preclip(ls_rt.adapter._params)

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        phase_timer.mark("optimizer")
        if ls_rt is not None and ls_rt.optimizer_kind == "lbfgs":
            # L-BFGS curvature pairs use the same raw objective-gradient
            # convention at both ends; q clipping only selects the live step.
            g_for_lbfgs = [
                None if g is None else g.detach().clone()
                for g in g_objective
            ]
        if not torch.isfinite(gnorm):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            if ls_rt is not None:
                latch_triggers(ls_rt.ctrl, ["nonfinite_gnorm"])
            # Preserve non-LS side-effect cadence for nonfinite gnorm (§14:
            # selection-failure is the special skip; gnorm skip keeps existing
            # frontier/viol behavior below).
        elif ls_rt is not None:
            F0 = float((loss + extra).detach())
            # Early triggers before commit (refs not yet updated).
            if ls_rt.cfg.search_on_early_triggers:
                viol_now = {
                    g_: float(
                        metrics.float().detach().cpu().numpy()[VIOL_OFFSET + i]
                    )
                    for i, g_ in enumerate(VIOL_KEYS)
                }
                _g2 = 0.0
                assert g_objective is not None
                for _g in g_objective:
                    if _g is not None:
                        _g2 += float(
                            _g.detach().double().pow(2).sum().item()
                        )
                gnorm_pre = _g2 ** 0.5
                late_reasons = check_early_triggers(
                    cfg=ls_rt.cfg,
                    refs=ls_rt.refs,
                    gnorm=gnorm_pre,
                    viol_metrics=viol_now,
                )
            else:
                # ``clip_grad_norm_`` returns the pre-clip norm. Trigger EMAs
                # are unused when early searches are disabled, so avoid a
                # second full-model float64 reduction and metric host sync.
                viol_now = {}
                gnorm_pre = float(gnorm)
                late_reasons = []
            if ls_rt.cfg.mode == "fixed_epsilon":
                run_search = False
            else:
                run_search = bool(known_search) or (
                    bool(ls_rt.cfg.search_on_early_triggers)
                    and bool(late_reasons)
                )
            all_reasons = list(search_reasons) + list(late_reasons)
            if run_search and include_bptt_step:
                raise ReplayIncompleteError(
                    f"step {step}: line-search requires BPTT replay payload when "
                    "include_bptt; MVP disables this path — set w_bptt=0 or "
                    "bptt_max_frac=0 under LS, or wait for BPTT payload wiring"
                )
            if run_search and tc_fns and (
                ls_rec is None
                or ls_rec.samples.tc_kp is None
                or ls_rec.samples.tc_kd is None
            ):
                assert_replayable_extras_or_raise(
                    cfg,
                    include_bptt=include_bptt_step,
                    has_tc_samples=False,
                    has_sfx_samples=True,
                    has_bptt_payload=False,
                )
            if run_search and sfx_fns and sfx_ramp_step > 0.0 and (
                ls_rec is None
                or ls_rec.samples.suffix_offs is None
                or "suffix_replay_batch" not in ls_rec.samples.extras
            ):
                assert_replayable_extras_or_raise(
                    cfg,
                    include_bptt=False,
                    has_tc_samples=True,
                    has_sfx_samples=False,
                    has_bptt_payload=False,
                )

            def _probe_F() -> float:
                # Adam/SaLSa probes only compare scalar objective values.  A
                # graph here was pure overhead and could OOM large actor
                # batches.  The L-BFGS y-pair path below explicitly calls the
                # same helper with ``return_tensor=True`` under grad mode.
                with torch.no_grad():
                    return _ls_probe_scalar_F(
                        batch=batch,
                        use_replay=use_replay,
                        bucket=bucket,
                        bs=bs,
                        encode_step=encode_step,
                        step_fns=step_fns,
                        tc_fns=tc_fns,
                        sfx_fns=sfx_fns,
                        model=model,
                        costs=costs,
                        pool=pool,
                        sim=sim,
                        cfg=cfg,
                        base=base,
                        yaw_i=yaw_i,
                        pr_idx=pr_idx,
                        pr_limit=pr_limit,
                        waist_cfg=waist_cfg,
                        w=w,
                        sfx_ramp=sfx_ramp_step,
                        include_bptt=include_bptt_step,
                        device=device,
                        dtype=dtype,
                        amp=amp,
                        rec=ls_rec,
                        generator=gen,
                        torque_grad_scale=torque_step_scale,
                    )

            ls_result = run_search_or_ordinary(
                ls_rt,
                j=ls_j,
                run_search=run_search,
                search_reasons=all_reasons,
                F0=F0,
                    g_preclip=g_preclip,
                probe_F=_probe_F,
                g_objective=g_objective,
                replay_ctx=replay_mgr,
                replay_rec=ls_rec,
            )
            ls_log = dict(ls_result.log)
            commit_side_effects = bool(ls_result.commit_side_effects)
            update_executed = bool(ls_result.executed)
            if ls_result.epsilon is not None:
                lr = float(ls_result.epsilon)
            if ls_result.executed:
                for k_ in (
                    "ls_u_fresh_frac", "ls_u_mom_frac", "ls_u_decay_frac",
                    "ls_u_mom_over_fresh",
                ):
                    v_ = ls_log.get(k_)
                    if v_ is not None and math.isfinite(float(v_)):
                        roll_acc[k_] = roll_acc.get(k_, 0.0) + float(v_)
                if ls_log.get("ls_u_total_norm") is not None:
                    roll_acc["ls_u_n"] = roll_acc.get("ls_u_n", 0.0) + 1.0
                if (
                    ls_rt.optimizer_kind == "lbfgs"
                    and isinstance(ls_rt.adapter, LbfgsLineSearchAdapter)
                    and g_for_lbfgs is not None
                    and ls_rec is not None
                    and replay_mgr is not None
                ):
                    adapter = ls_rt.adapter
                    opt.zero_grad(set_to_none=True)
                    with replay_mgr.run_isolated(ls_rec):
                        tot = _ls_probe_scalar_F(
                            batch=batch,
                            use_replay=use_replay,
                            bucket=bucket,
                            bs=bs,
                            encode_step=encode_step,
                            step_fns=step_fns,
                            tc_fns=tc_fns,
                            sfx_fns=sfx_fns,
                            model=model,
                            costs=costs,
                            pool=pool,
                            sim=sim,
                            cfg=cfg,
                            base=base,
                            yaw_i=yaw_i,
                            pr_idx=pr_idx,
                            pr_limit=pr_limit,
                            waist_cfg=waist_cfg,
                            w=w,
                            sfx_ramp=sfx_ramp_step,
                            include_bptt=include_bptt_step,
                            device=device,
                            dtype=dtype,
                            amp=amp,
                            rec=ls_rec,
                            generator=gen,
                            torque_grad_scale=torque_step_scale,
                            return_tensor=True,
                        )
                        tot.backward()
                    g_new = snapshot_g_preclip(adapter._params)
                    y_list = [
                        (gn - go).detach()
                        if go is not None and gn is not None
                        else torch.zeros_like(p)
                        for p, go, gn in zip(adapter._params, g_for_lbfgs, g_new)
                    ]
                    s_list = adapter.displacement_from_theta0()
                    accepted = adapter.push_pair(s_list, y_list)
                    ls_log["lbfgs_hist"] = float(len(adapter._s_hist))
                    ls_log["lbfgs_pair_ok"] = 1.0 if accepted else 0.0
                    ls_rt.last_log = {**(ls_rt.last_log or {}), **ls_log}
            # Update trigger refs once per valid baseline (after decision).
            if commit_side_effects and ls_rt.cfg.search_on_early_triggers:
                ls_rt.refs.update_after_valid_baseline(gnorm_pre, viol_now)
        else:
            # Non-LS path; gnorm is finite here (nonfinite handled above).
            opt.step()
            update_executed = True

        if update_executed:
            updates_executed += 1

        phase_timer.mark("critic")

        # The label describes the pre-update actor evaluation above.  Train the
        # separate critic only after actor/line-search side effects are settled,
        # so its optimizer and RNG cannot contaminate actor selection.
        if timing_critic is not None and critic_observation is not None:
            observations = [*critic_extra_observations, critic_observation]
            observe_features = torch.cat([item[0] for item in observations], dim=0)
            observe_gate = torch.cat([item[1] for item in observations], dim=0)
            if cfg.timing_critic.representation == "distributional":
                still_anchor = torch.where(
                    batch["active"].bool(),
                    batch["hit"] == int(bucket),
                    torch.ones_like(batch["active"], dtype=torch.bool),
                ).all(dim=1)
                timing_anchor_mask = timing_anchor_mask & still_anchor
                extra_n = int(observe_features.shape[0]) - int(
                    timing_anchor_mask.numel()
                )
                observe_anchor = torch.cat((
                    torch.zeros(extra_n, device=device, dtype=torch.bool),
                    timing_anchor_mask,
                ))
                timing_critic_pending.append({
                    "features": observe_features.detach().to(
                        device="cpu", dtype=torch.float32,
                    ),
                    "gate": observe_gate.detach().to(
                        device="cpu", dtype=torch.float32,
                    ),
                    "anchor": observe_anchor.detach().to(
                        device="cpu", dtype=torch.bool,
                    ),
                    "actor_step": int(step),
                })
            else:
                timing_critic_pending.append({
                    "features": observe_features.detach().to(
                        device="cpu", dtype=torch.float32,
                    ),
                    "gate": observe_gate.detach().to(
                        device="cpu", dtype=torch.float32,
                    ),
                    "anchor": None,
                    "actor_step": int(step),
                })
            if len(timing_critic_pending) >= int(
                cfg.timing_critic.update_interval
            ):
                _flush_timing_critic_pending()

        phase_timer.mark("bookkeeping")

        if (
            sfx_fns
            and commit_side_effects
            and torch.isfinite(gnorm)
        ):
            admitted = suffix_ring.add(
                bucket, batch, _traj, hard_ok=bellman_parent_ok,
            )
            roll_acc["suffix_replay_admitted"] = (
                roll_acc.get("suffix_replay_admitted", 0.0) + float(admitted)
            )
            roll_acc["suffix_replay_size"] = (
                roll_acc.get("suffix_replay_size", 0.0)
                + float(len(suffix_ring))
            )

        if (
            voxel_state is not None
            and voxel_meta is not None
            and voxel_parent_batch is not None
            and commit_side_effects
            and torch.isfinite(gnorm)
        ):
            ingressed = voxel_state.record_train_ingress(
                voxel_meta, hard_ok, step=step,
            )
            refreshed = voxel_mpc_ring.refresh(
                bucket, voxel_parent_batch, _traj, hard_ok=hard_ok,
            )
            roll_acc["voxel_ingress"] = (
                roll_acc.get("voxel_ingress", 0.0) + float(ingressed)
            )
            roll_acc["voxel_mpc_refresh"] = (
                roll_acc.get("voxel_mpc_refresh", 0.0) + float(refreshed)
            )
        elif (
            timing_mpc_ring is not None
            and timing_parent_batch is not None
            and commit_side_effects
            and torch.isfinite(gnorm)
        ):
            if (
                terminal_band_success is not None
                and hasattr(timing_critic, "observe_terminal_band")
            ):
                timing_critic.observe_terminal_band(terminal_band_success)
            if (
                timing_boundary_success is not None
                and hasattr(timing_critic, "observe_boundary")
            ):
                timing_critic.observe_boundary(
                    timing_boundary_success,
                    actor_step=step,
                    horizon=bucket,
                )
            # The critic MPC pool is deliberately quality-agnostic. Full-cost
            # training on noisy suffix states is the recovery signal; only the
            # structural pre-hit-waypoint check inside ``refresh`` can reject a
            # row here.
            timing_parent_ok = _timing_mpc_parent_ingress_ok(gate.detach())
            refreshed = timing_mpc_ring.refresh(
                bucket, timing_parent_batch, _traj, hard_ok=timing_parent_ok,
                generator=gen,
            )
            roll_acc["timing_mpc_refresh"] = (
                roll_acc.get("timing_mpc_refresh", 0.0) + float(refreshed)
            )
            roll_acc["timing_mpc_parent_pool_size"] = (
                roll_acc.get("timing_mpc_parent_pool_size", 0.0)
                + float(timing_mpc_ring.size(bucket))
            )

        if frontier_cloud is not None and commit_side_effects:
            _frontier_ema_from_traj(
                frontier_cloud, costs[bucket], batch, _traj, bucket,
                viol_cap=float(cfg.proficiency_viol_cap),
            )
        if (
            voxel_state is None
            and bool(cfg.replay_proficiency)
            and commit_side_effects
        ):
            _replay_ema_from_traj(
                replay, costs[bucket], batch, _traj, bucket,
                viol_cap=float(cfg.proficiency_viol_cap),
            )

        # Baseline metrics always accumulate for logging; viol_acc / AL only on
        # successful executed updates (§14 commit gate).
        m = metrics.float().detach().cpu().numpy()
        m[3] = float(loss_tc.detach())
        m[4] = float(loss_bptt.detach())
        acc += m
        acc_n += 1
        if commit_side_effects:
            for i, g_ in enumerate(VIOL_KEYS):
                viol_acc[g_] += float(m[VIOL_OFFSET + i])
            viol_n += 1
        elif ls_rt is not None and rank == 0:
            emit({
                "event": "ls_selection_failed",
                "step": step + 1,
                "u": int(ls_rt.ctrl.successful_update_count),
                "abs_step": step + 1,
                **ls_log,
            })

        if (step + 1) % cfg.al_every == 0 and viol_n:
            lam_before = dict(al.lam)
            pen_before = dict(al.pen)
            viol_mean = {g_: viol_acc[g_] / viol_n for g_ in viol_acc}
            skip_torque_al = (
                cfg.cost.torque_al_off
                or (
                    cfg.cost.torque_static_only
                    and not cfg.cost.torque_static_al
                )
            )
            if skip_torque_al:
                viol_mean = {
                    g_: v for g_, v in viol_mean.items()
                    if not g_.startswith("torque")
                }
            al.update(viol_mean)
            viol_acc = {g_: 0.0 for g_ in viol_acc}
            viol_n = 0
            if ls_rt is not None:
                changed = (
                    any(abs(al.lam[g] - lam_before[g]) > 0.0 for g in al.lam)
                    or any(abs(al.pen[g] - pen_before.get(g, al.pen[g])) > 0.0
                           for g in al.pen)
                )
                if changed:
                    ls_rt.objective_direction_fingerprint["al_lam"] = {
                        str(k): float(al.lam[k]) for k in sorted(al.lam)
                    }
                    ls_rt.objective_direction_fingerprint["al_pen"] = {
                        str(k): float(al.pen[k]) for k in sorted(al.pen)
                    }
                    if ls_rt.cfg.force_search_on_al_change:
                        eps_after = reset_history_for_objective_change(
                            ls_rt.accept,
                            ls_rt.ctrl,
                            cfg=ls_rt.cfg,
                            eta_ref=ls_rt.eta_ref,
                        )
                        if ls_rt.cfg.moment_on == "step":
                            ls_rt.adapter.reset_step_moments()
                        if log and rank == 0:
                            if ls_rt.cfg.reseed_epsilon_on_objective_change:
                                print(
                                    f"[line_search] AL change @ step {step+1}: "
                                    f"acceptance history reset, ε→{eps_after:.3g} "
                                    f"({ls_rt.cfg.objective_reseed_mult:g}× prev), "
                                    "K=1, search forced",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[line_search] AL change @ step {step+1}: "
                                    f"acceptance history reset, ε kept at "
                                    f"{eps_after:.3g}, K=1, search forced",
                                    flush=True,
                                )
                    elif log and rank == 0:
                        print(
                            f"[line_search] AL change @ step {step+1}: "
                            "λ/pen updated (no search reset)",
                            flush=True,
                        )
                ls_rt.al_lam_snapshot = dict(al.lam)
                ls_rt.al_pen_snapshot = dict(al.pen)

        phase_sample = phase_timer.finish()
        if phase_sample:
            phase_timing_latest = phase_sample

        if (step + 1) % cfg.log_every == 0 and rank == 0:
            now = time.perf_counter()
            rec = {"event": "train", "step": step + 1, "lr": lr,
                   "grad_norm": float(gnorm),
                   "steps_per_s": cfg.log_every / (now - t_last),
                   "elapsed_s": now - t0,
                   "alpha": cfg.schedule.alpha(step, stage_a),
                   "alpha_tau_peak": cfg.schedule.torque_peak_alpha(step),
                   "alpha_vel_peak": cfg.schedule.velocity_peak_alpha(step),
                   "alpha_coll_peak": cfg.schedule.collision_peak_alpha(step),
                   "alpha_jbox_peak": cfg.schedule.joint_box_peak_alpha(step),
                   "alpha_waist": cfg.schedule.waist_alpha(step),
                   "alpha_torque_full": blend_now,
                   "torque_physics": cfg.cost.torque_physics,
                   "torque_static_only": cfg.cost.torque_static_only,
                   "torque_al_off": cfg.cost.torque_al_off,
                   "torque_static_al": cfg.cost.torque_static_al,
                   "w_torque_static": cfg.cost.w_torque_static,
                   "static_torque_tighten": cfg.cost.static_torque_tighten,
                   "eta_torque": cfg.cost.eta_torque,
                   "stage_a_open": stage_a,
                   "replay": (
                       voxel_mpc_ring.distribution()
                       if voxel_state is not None else replay.distribution()
                   ),
                   "post_hit": (
                       {} if voxel_state is not None else post_hit.distribution()
                   ),
                   "replay_short": (
                       {} if voxel_state is not None else
                       {b: replay.horizon_fracs(b) for b in replay.buckets}
                   ),
                   "batch_short": {"le3": mix_acc[0] / max(mix_n, 1),
                                   "le5": mix_acc[1] / max(mix_n, 1),
                                   "le8": mix_acc[2] / max(mix_n, 1)},
                   "train_B": batch_b_acc / max(mix_n, 1),
                   "train_bucket_frac": {
                       int(b): bucket_step_acc.get(int(b), 0) / max(mix_n, 1)
                       for b in pool.buckets
                   },
                   "roller": {k_: v_ / max(cfg.log_every, 1) for k_, v_ in roll_acc.items()},
                   "replay_viol_sampling_beta": float(cfg.replay_viol_downweight),
                   "replay_viol_uniform_mix": float(cfg.replay_viol_uniform_mix),
                   "lam": al.weights(),
                   "lam_i": dict(al.lam),
                   "pen": dict(al.pen)}
            if replay is not None and any(
                "viol_score" in replay.data[b] for b in replay.data
            ):
                rec["replay_viol_score"] = _replay_viol_score_telemetry(
                    replay, replay_viol_selected,
                )
            if ls_rt is not None:
                rec.update(ls_rt.last_log or ls_log)
                rec["line_search_c1"] = float(
                    ls_rt.cfg.c1_at(int(ls_rt.ctrl.successful_update_count) + 1)
                )
                rec["line_search_epsilon_floor"] = float(
                    ls_rt.cfg.epsilon_min_at(
                        int(ls_rt.ctrl.successful_update_count) + 1,
                        float(ls_rt.epsilon_min),
                    )
                )
                rec["ls_u"] = int(ls_rt.ctrl.successful_update_count)
                rec["ls_seed_provenance"] = ls_rt.seed_provenance
                ls_n = int(roll_acc.get("ls_u_n", 0.0))
                if ls_n > 0:
                    rec["ls_u_fresh_frac_mean"] = (
                        roll_acc["ls_u_fresh_frac"] / ls_n
                    )
                    rec["ls_u_mom_frac_mean"] = roll_acc["ls_u_mom_frac"] / ls_n
                    rec["ls_u_decay_frac_mean"] = (
                        roll_acc["ls_u_decay_frac"] / ls_n
                    )
                    rec["ls_u_mom_over_fresh_mean"] = (
                        roll_acc["ls_u_mom_over_fresh"] / ls_n
                    )
            if grad_term_probe is not None:
                rec.update(grad_term_probe)
            if gb_stats:
                rec.update(gb_stats)
            rec.update(_torque_grad_cap_log_fields(
                roll_acc, cap=float(cfg.torque_grad_cap),
            ))
            rec.update(phase_timing_latest)
            phase_timing_latest = {}
            if cfg.torque_grad_cap > 0.0:
                rec["torque_grad_cap_refresh_interval"] = float(
                    cfg.torque_grad_cap_refresh
                )
                rec["torque_grad_cap_cached_scale"] = float(
                    torque_grad_cached_scale
                )
            rec.update({k: float(v) for k, v in zip(METRIC_NAMES, acc / max(acc_n, 1))})
            if waist_mpc_param_grad_norm is not None:
                rec["waist_yaw_mpc_tie_param_grad_norm"] = waist_mpc_param_grad_norm
                rec["waist_yaw_mpc_tie_param_grad_fraction"] = (
                    waist_mpc_param_grad_norm / max(float(gnorm), 1.0e-12)
                )
            if sp_res_n:
                rec["waist_sp_res"] = sp_res_acc / sp_res_n
            if pr_sp_n:
                rec["waist_pr_sp_abs"] = pr_sp_acc / pr_sp_n
            if pr_hold_n:
                rec["waist_pr_sp_hold"] = pr_hold_acc / pr_hold_n
            if pr_hold_band_n:
                rec["waist_pr_sp_hold_band"] = pr_hold_band_acc / pr_hold_band_n
            if pr_distill_n:
                rec["pr_distill_frac"] = pr_distill_frac_acc / pr_distill_n
                rec["pr_distill_delta"] = pr_distill_delta_acc / pr_distill_n
            if yaw_hold_n:
                rec["waist_sp_hold"] = yaw_hold_acc / yaw_hold_n
            if sm_n:
                rec["waist_sp_smooth"] = sm_acc[0] / sm_n
                rec["waist_pr_sp_smooth"] = sm_acc[1] / sm_n
            if ss_n:
                rec["waist_sp_search_delta"] = ss_acc[0] / ss_n
                rec["waist_sp_search_aim"] = ss_acc[1] / ss_n
                rec["waist_sp_search_spread"] = ss_acc[2] / ss_n
            if sfx_n:
                rec["sfx_cost"] = sfx_acc[0] / sfx_n
                rec["sfx_tie"] = sfx_acc[1] / sfx_n
            if mpc_tie_n:
                rec["mpc_init_tie"] = mpc_tie_acc / mpc_tie_n
                rec["mpc_init_frac"] = mpc_tie_frac_acc / mpc_tie_n
            if waist_mpc_tie_n:
                rec["waist_yaw_mpc_tie"] = waist_mpc_tie_acc / waist_mpc_tie_n
                rec["waist_yaw_mpc_tie_weighted_output_grad_norm"] = (
                    waist_mpc_tie_grad_acc / waist_mpc_tie_n
                )
            if frontier_cloud is not None:
                rec["frontier"] = {
                    int(b): {
                        "n": frontier_cloud.size(int(b)),
                        "seed": int(frontier_cloud._n_seed.get(int(b), 0)),
                        "child": int(frontier_cloud._child_n.get(int(b), 0)),
                        "charts": int((frontier_cloud.occupancy_of(int(b)) > 0).sum()),
                    }
                    for b in pool.buckets
                }
                rec["frontier_landmarks"] = frontier_cloud.n_landmarks()
            if timing_critic is not None:
                rec["timing_critic"] = timing_critic.metrics(reset=True)
            emit(rec)
            recent_train.append(rec)
            if voxel_state is None:
                activated = _maybe_auto_activate_frontier(
                    cfg=cfg, pool=pool, roller=roller, run_dir=run_dir,
                    device=device, dtype=dtype, log=log, rank=rank,
                    step=step + 1, frontier_cloud=frontier_cloud,
                    last_probe=last_probe, recent_train=list(recent_train), emit=emit,
                )
                if activated is not None:
                    frontier_cloud = activated
            if log:
                print(
                    f"[{step+1:6d}] L {rec['loss']:9.3f} cost {rec['cost']:9.3f} "
                    f"tc {rec['tc']:7.4f} "
                    + (f"sfx {rec['sfx_cost']:8.3f}/{rec['sfx_tie']:6.4f} "
                       if "sfx_tie" in rec else "")
                    + f"| term {rec['c_terminal']:7.3f} "
                    f"coll {rec['c_collision']:7.3f} "
                    f"(d{rec['d_c_drum']:5.3f} u{rec['d_c_under']:5.3f} "
                    f"s{rec['d_c_self']:5.3f} x{rec['d_c_stick']:5.3f}) "
                    f"tau {rec['c_torque']:6.3f} "
                    f"| miss {rec['d_region']*1000:6.1f}mm sdf {rec['d_min_sdf']*1000:6.1f}mm "
                    + (f"| sp_res {rec['waist_sp_res']*1000:5.1f}mrad "
                       if "waist_sp_res" in rec else "")
                    + (f"pr {rec['waist_pr_sp_abs']*1000:5.1f}mrad "
                       if "waist_pr_sp_abs" in rec else "")
                    + (f"hold {rec['waist_pr_sp_hold']*1000:5.1f}mrad "
                       if "waist_pr_sp_hold" in rec else "")
                    + (f"hB {rec['waist_pr_sp_hold_band']*1000:5.1f}mrad "
                       if "waist_pr_sp_hold_band" in rec else "")
                    + (f"dG {rec['pr_distill_frac']:.2f} "
                       f"dΔ {rec['pr_distill_delta']*1000:4.1f}mrad "
                       if "pr_distill_frac" in rec else "")
                    + (f"sm {rec['waist_sp_smooth']*1000:5.1f}mrad "
                       if "waist_sp_smooth" in rec else "")
                    + (f"srch d{rec['waist_sp_search_delta']*1000:5.1f} "
                       f"a{rec['waist_sp_search_aim']*1000:5.1f}mrad "
                       f"V{rec['waist_sp_search_spread']:8.3f} "
                       if "waist_sp_search_spread" in rec else "")
                    + (f"| cloud "
                       + " ".join(
                           f"b{b} c{rec['frontier'][b]['child']}/"
                           f"s{rec['frontier'][b]['seed']}"
                           f" ch{rec['frontier'][b]['charts']}"
                           for b in rec["frontier"]
                       )
                       + (f" lm{rec['frontier_landmarks']}"
                          if "frontier_landmarks" in rec else "")
                       + " "
                       if "frontier" in rec else "")
                    + f"| {rec['steps_per_s']:.1f} it/s", flush=True,
                )
            acc[:] = 0.0
            acc_n = 0
            mix_acc = [0.0, 0.0, 0.0]
            mix_n = 0
            batch_b_acc = 0.0
            bucket_step_acc = {int(b): 0 for b in pool.buckets}
            sp_res_acc, sp_res_n = 0.0, 0
            pr_sp_acc, pr_sp_n = 0.0, 0
            pr_hold_acc, pr_hold_n = 0.0, 0
            pr_hold_band_acc, pr_hold_band_n = 0.0, 0
            yaw_hold_acc, yaw_hold_n = 0.0, 0
            pr_distill_frac_acc, pr_distill_delta_acc, pr_distill_n = 0.0, 0.0, 0
            sm_acc, sm_n = [0.0, 0.0], 0
            ss_acc, ss_n = [0.0, 0.0, 0.0], 0
            sfx_acc, sfx_n = [0.0, 0.0], 0
            mpc_tie_acc, mpc_tie_frac_acc, mpc_tie_n = 0.0, 0.0, 0
            roll_acc = {k_: 0.0 for k_ in roll_acc}
            replay_viol_selected = []
            t_last = now

        # ---- closed-loop probe (the metric the MPC is judged on) ------------
        if (
            rank == 0 and probe_fn is not None and cfg.probe_every > 0
            and ((step + 1) % cfg.probe_every == 0 or (step + 1) == cfg.max_steps)
        ):
            model.eval()
            try:
                pr = probe_fn(model)
            except Exception as exc:  # a probe must never kill a 2 h run
                pr = {"error": repr(exc), "in_disk": -1.0, "hits": 0}
            model.train()
            last_probe = pr
            pr_rec = {k: v for k, v in pr.items() if not k.startswith("_")}
            emit({"event": "probe", "step": step + 1, **pr_rec})
            if log:
                print(f"    probe@{step+1}: score {pr.get('score', float('nan')):.3f} "
                      f"in-disk {pr.get('in_disk')}/"
                      f"{pr.get('hits')} axial p50 {pr.get('axial_p50_mm', float('nan')):+.1f} "
                      f"in-plane p50 {pr.get('inplane_p50_mm', float('nan')):+.1f} mm",
                      flush=True)
            if float(pr.get("score", -1.0)) > best_probe_only + 1e-9:
                best_probe_only = float(pr.get("score", -1.0))
                save_phase2_checkpoint(
                    run_dir / "best_probe.pt", model=model, optimizer=opt, al=al,
                    step=step + 1, cfg=cfg, model_cfg=model_cfg,
                    scenario_cfg=scenario_cfg,
                    extra=_checkpoint_extra_with_ls(
                        _train_checkpoint_extra({"probe": pr_rec}),
                        ls_rt, optimizer=opt,
                    ),
                    rng_streams=phase2_rng_streams,
                )
            if voxel_state is None:
                activated = _maybe_auto_activate_frontier(
                    cfg=cfg, pool=pool, roller=roller, run_dir=run_dir,
                    device=device, dtype=dtype, log=log, rank=rank,
                    step=step + 1, frontier_cloud=frontier_cloud,
                    last_probe=last_probe, recent_train=list(recent_train), emit=emit,
                )
                if activated is not None:
                    frontier_cloud = activated

        # ---- rolling eval + gates ------------------------------------------
        if rank == 0 and eval_cases is not None and (
            (cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0)
            or (step + 1) == cfg.max_steps
        ):
            model.eval()
            ev = rolling_eval(
                model, eval_cases, scorer, device=device, kit=kit, library=library,
                buckets=pool.buckets, cost=base, amp_dtype=amp,
                num_drums=model_cfg.num_drums, num_future=model_cfg.num_future,
            )
            model.train()
            ev.update({"event": "eval", "step": step + 1,
                       "elapsed_s": time.perf_counter() - t0,
                       "stage_a": stage_a_pass(ev)})
            eval_hist.append(ev)
            stage_a_was = bool(stage_a)
            stage_a = stage_a or bool(ev["stage_a"])
            if ls_rt is not None and (not stage_a_was) and stage_a:
                eps_after = reset_history_for_objective_change(
                    ls_rt.accept,
                    ls_rt.ctrl,
                    cfg=ls_rt.cfg,
                    eta_ref=ls_rt.eta_ref,
                )
                if ls_rt.cfg.moment_on == "step":
                    ls_rt.adapter.reset_step_moments()
                ls_rt.stage_a_snapshot = True
                ls_rt.objective_direction_fingerprint["stage_a"] = True
                if log and rank == 0:
                    if ls_rt.cfg.reseed_epsilon_on_objective_change:
                        print(
                            f"[line_search] stage_a opened @ step {step+1}: "
                            f"acceptance history reset, ε→{eps_after:.3g} "
                            f"({ls_rt.cfg.objective_reseed_mult:g}× prev), "
                            "K=1, search forced",
                            flush=True,
                        )
                    else:
                        print(
                            f"[line_search] stage_a opened @ step {step+1}: "
                            f"acceptance history reset, ε kept at "
                            f"{eps_after:.3g}, K=1, search forced",
                            flush=True,
                        )
            emit(ev)
            if log:
                print(
                    f"    eval@{step+1}: acc {ev['accepted']}/{ev['n']} "
                    f"hit {ev['ok_hit']:.2f} coll {ev['ok_collision']:.2f} "
                    f"tau {ev['ok_torque']:.2f} elbow {ev.get('elbow_down', float('nan')):.2f} "
                    f"short {ev['short_accepted']}/{ev['short_n']}", flush=True,
                )
            # PLATEAU is decided here, on acceptance alone: the question "is the
            # run still learning" is about the model, not about which file the
            # save gate happened to accept.
            if ev["acceptance"] > best_seen + 1e-4:
                best_seen, since_improve = ev["acceptance"], 0
            else:
                since_improve += 1
            if ev["acceptance"] > best + 1e-4:
                # Gate on the closed loop: acceptance alone selected 7b's
                # procrastinating checkpoint.  A probe that has not run yet
                # (``last_probe is None``) does not block, so a run without
                # ``probe_fn`` behaves exactly as before.
                pscore = (float(last_probe.get("score", last_probe["in_disk"]))
                          if last_probe else None)
                if pscore is None or pscore >= best_probe - 1e-9:
                    best, best_step = ev["acceptance"], step + 1
                    if pscore is not None:
                        best_probe = pscore
                    save_phase2_checkpoint(
                        run_dir / "best.pt", model=model, optimizer=opt, al=al,
                        step=step + 1, cfg=cfg, model_cfg=model_cfg,
                        scenario_cfg=scenario_cfg,
                        extra=_checkpoint_extra_with_ls(
                            _train_checkpoint_extra({
                                "eval": ev, "probe": last_probe,
                            }), ls_rt, optimizer=opt,
                        ),
                        rng_streams=phase2_rng_streams,
                    )
                else:
                    emit({"event": "best_blocked", "step": step + 1,
                          "acceptance": ev["acceptance"], "probe": pscore,
                          "best_probe": best_probe})
            if ev["acceptance"] >= cfg.target_acceptance:
                stopped = "target"
                emit({"event": "stop", "reason": "target", "step": step + 1})
                break
            if since_improve >= cfg.plateau_evals and ev["short_n"] and (
                ev["short_accepted"] / max(ev["short_n"], 1) >= 26.0 / 54.0
            ):
                stopped = "plateau"
                emit({"event": "stop", "reason": "plateau", "step": step + 1})
                break

        if rank == 0:
            step_done = step + 1
            if step_done % cfg.ckpt_every == 0:
                _save_last_pt(
                    run_dir, model=model, optimizer=opt, al=al, step=step_done,
                    cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
                    ls_rt=ls_rt, extra=_train_checkpoint_extra(),
                    rng_streams=phase2_rng_streams,
                )
                step_path = _save_step_pt(
                    run_dir, model=model, optimizer=opt, al=al, step=step_done,
                    cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
                    ls_rt=ls_rt, extra=_train_checkpoint_extra(),
                    rng_streams=phase2_rng_streams,
                )
                if log and step_path is not None:
                    print(f"[ckpt] kept {step_path.name}", flush=True)
                _save_frontier_pt(run_dir, frontier_cloud)
                _maybe_frontier_cloud_plot(
                    run_dir, frontier_cloud, model, costs, pool,
                    step=step_done, start_step=start_step, cfg=cfg,
                )
            elif (
                int(cfg.ckpt_last_every) > 0
                and step_done % int(cfg.ckpt_last_every) == 0
            ):
                _save_last_pt(
                    run_dir, model=model, optimizer=opt, al=al, step=step_done,
                    cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
                    ls_rt=ls_rt, extra=_train_checkpoint_extra(),
                    rng_streams=phase2_rng_streams,
                )
                step_path = _save_step_pt(
                    run_dir, model=model, optimizer=opt, al=al, step=step_done,
                    cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
                    ls_rt=ls_rt, extra=_train_checkpoint_extra(),
                    rng_streams=phase2_rng_streams,
                )
                if log and step_path is not None:
                    print(f"[ckpt] kept {step_path.name}", flush=True)

    # Do not discard up to ``update_interval - 1`` exact actor labels when a
    # short run, deadline, or early-stop ends between consolidated updates.
    _flush_timing_critic_pending()
    wall = time.perf_counter() - t0
    summary = {
        "event": "done", "steps": last_step, "start_step": int(start_step),
        "wall_s": wall,
        **_invocation_throughput(
            updates_attempted=updates_attempted,
            updates_executed=updates_executed,
            wall_s=wall,
        ),
        "best_acceptance": best, "best_step": best_step, "stopped": stopped,
        # Peak acceptance the run REACHED, save gate or no.  When this exceeds
        # ``best_acceptance`` the probe gate blocked the better model and
        # ``last.pt`` may beat ``best.pt`` -- say so rather than letting the
        # "best acceptance" line under-report the run.
        "best_acceptance_seen": best_seen,
        "best_probe_with_best": best_probe, "best_probe_only": best_probe_only,
        "last_probe": ({k: v for k, v in last_probe.items()
                        if not k.startswith("_")} if last_probe else None),
        "skipped_steps": n_skipped,
        "stage_a_reached": stage_a, "lam": al.weights(),
        "replay": (
            voxel_mpc_ring.distribution()
            if voxel_state is not None else replay.distribution()
        ),
        "evals": eval_hist,
    }
    if timing_critic is not None:
        summary["timing_critic"] = timing_critic.metrics(reset=False)
    if rank == 0:
        _save_last_pt(
            run_dir, model=model, optimizer=opt, al=al, step=last_step,
            cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
            ls_rt=ls_rt, extra=_train_checkpoint_extra(),
            rng_streams=phase2_rng_streams,
        )
        _save_frontier_pt(run_dir, frontier_cloud)
        _maybe_frontier_cloud_plot(
            run_dir, frontier_cloud, model, costs, pool,
            step=last_step, start_step=start_step, cfg=cfg,
        )
        emit(summary)
        fh.close()
        tb.close()
    return summary
