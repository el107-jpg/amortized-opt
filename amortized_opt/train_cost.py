"""Phase 2+3 merged: direct cost descent with ramped weights and chained BPTT.

Section 7 of ``docs/amortized_optimizer_plan.md``, under resolved decisions 1,
2, 3, 5 and 6.  Phase 1 put the decoder in a trajectory-producing regime by
imitating a quintic; this module stops imitating anything and back-propagates
the differentiable cost through the decoder instead.

The objective
-------------

Per step, on one horizon-bucket-homogeneous minibatch::

    L =   w_cost * mean_b FusedCost(q0, model(init, state, ctx), v0)
        + w_aux  * Huber(aux heads, labels recomputed from the model's OWN traj)
        + w_tc   * temporal-consistency disagreement on a subset
        + w_bptt * FusedCost of a SECOND solve, after a differentiable PD roll
                   through the first hit and a bounce   (ramped in at the tail)

* the auxiliary heads stay supervised throughout (the plan: "the auxiliary
  terms ... should have continued throughout the entire thing"), but their
  labels now come from the model's own output trajectory through
  ``tspline`` / ``ChainFK`` / ``knot_fd`` + RNEA -- a self-consistent world
  model rather than a frozen teacher;
* there is no imitation term.  ``init`` is a warm start, never a target.

Weight schedule (resolved decision 6a)
--------------------------------------

One run, no hard stage boundary.  Obstacle / posture / terminal-position
weights are at full strength from step 0; the torque, velocity and joint-box
hinges and the strike-quality terms (impact speed, stick elevation) ramp from
``ramp_lo`` to 1.0 over ``[ramp_start, ramp_end]`` on a smoothstep.  After that
the :class:`~amortized_opt.lagrangian.AugmentedLagrangian` does the balancing:
its multipliers are updated every ``al_every`` steps from the rolling mean of
the *raw* (zero-when-feasible) violations.

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
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, N_DOF
from .cost import CostConfig, FusedCost, StrikeContext
from .fk import ChainFK, stick_tip_offset
from .idle_home import resolve_idle_home
from .ik import inertia_weighted_pinv, tip_jacobian
from .lagrangian import CONSTRAINT_GROUPS, ALConfig, AugmentedLagrangian
from .lead import lead_fraction, lead_waypoints
from .limits import (
    ETA_VELOCITY,
    isaac_effort_limits,
    isaac_velocity_limits,
    urdf_position_limits,
)
from .model import ModelConfig
from .pd_sim import BatchedPDSim, PDSimConfig, mass_matrix
from .phase2_data import (
    Phase2Noise,
    Phase2Pool,
    Phase2ScenarioConfig,
    drum_name_predicates,
    strike_geometry_table,
)
from .rhythm import DrumKit
from .rnea import BatchedRNEA
from .strike import STICK_TIP_SIGMA
from .tokenizer import FK_POINT_NAMES, build_waypoint_masks
from .train_warmstart import LossScales, _masked_huber
from .warmstart import GoalPoseLibrary, arm_slice, waist_slice
from .warmstart_data import CONTEXT_FIELDS, aux_targets

__all__ = [
    "BounceConfig",
    "MPCRoller",
    "Phase2Config",
    "ReplayPool",
    "WeightSchedule",
    "apply_bounce",
    "bounce_tip_velocity",
    "build_costs",
    "effective_hits",
    "encode",
    "make_cost_step",
    "phase2_al_config",
    "phase2_cost_config",
    "cost_tuple",
    "ramp_seed",
    "resolve_idle_home",
    "save_phase2_checkpoint",
    "load_phase2_checkpoint",
    "temporal_consistency_loss",
    "train_phase2",
]

_HANDS = ("left", "right")


# ============================================================== schedule =====
def _smoothstep(t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class WeightSchedule:
    """Ramped multipliers for the merged stage A/B curriculum.

    Stage-A emphasis (obstacle, posture, terminal *position*) is live from step
    0; the limit hinges are present but down-weighted so the model is nudged
    toward feasibility without the torque term dominating a decoder that cannot
    yet reach the drum.  Everything ramps on one smoothstep, so there is no
    discontinuity for the augmented Lagrangian to chase.
    """

    ramp_start: int = 2_000
    ramp_end: int = 20_000
    #: Multiplier at step 0 for the ramped groups.
    lo_limits: float = 0.05        # torque / velocity / joint_box
    lo_strike: float = 0.10        # impact speed + stick elevation
    #: Groups that are NOT ramped (stage-A emphasis, full from the start).
    w_collision: float = 1.0
    w_posture: float = 1.0
    w_terminal_region: float = 1.0
    w_smooth: float = 1.0

    # ---- the torque PEAK sub-term (``CostConfig.w_torque_peak``) ------------
    #: The soft-max half of the torque term gets its own, later, faster ramp.
    #: It is the one component whose scale is set by the WORST element rather
    #: than by an average.  It now also has its own augmented-Lagrangian
    #: multiplier (``lam_torque_peak``, driven by the MAX raw violation), so
    #: this ramp and that multiplier multiply into the SAME
    #: ``weights["torque_peak"]`` slot -- see :func:`_wtensors` -- and the mean
    #: half's ``lam_torque`` is no longer dragged along with it.  Ramping over
    #: ~1 k steps still matters on a resume: the multiplier is seeded from
    #: ``lam_torque`` (7.2 at the end of ``runs/phase2_main``) so the objective
    #: is continuous across the switch, and the ramp gives it time to relax to
    #: its own equilibrium as the peak falls.
    torque_peak_start: int = 0
    torque_peak_ramp: int = 1_000

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

    # ---- the IDLE-ARM pair (``w_idle_home`` / ``w_idle_vel``) ---------------
    #: Both idle terms share one ramp into ``weights["posture_idle"]``.  Like
    #: ``effort`` these are plain prices on objective terms with no multiplier to
    #: shock, so the ramp only exists to keep the loss continuous when a
    #: continuation switches the home table on.  Short: the target is a constant
    #: pose, which is the easiest thing in the objective to learn.
    idle_start: int = 0
    idle_ramp: int = 300

    # ---- the plan's stage-A unlock -----------------------------------------
    #: When True, ``alpha`` is HELD at 0 until :func:`stage_a_pass` has fired
    #: once, so the limit hinges cannot ramp before the model can reach a drum
    #: collision-free.  Off by default: a gate that never opens freezes the
    #: curriculum forever, which on a short run is the worse failure.  A run
    #: that resumes from a checkpoint which already passed stage A must seed
    #: the gate open (``train_phase2(stage_a_init=True)``) or the first
    #: ``eval_every`` steps run at ``lo_limits``.
    gate_stage_a: bool = False

    def alpha(self, step: int, stage_a: bool = True) -> float:
        if self.gate_stage_a and not stage_a:
            return 0.0
        span = max(int(self.ramp_end) - int(self.ramp_start), 1)
        return _smoothstep((int(step) - int(self.ramp_start)) / span)

    def torque_peak_alpha(self, step: int) -> float:
        span = max(int(self.torque_peak_ramp), 1)
        return _smoothstep((int(step) - int(self.torque_peak_start)) / span)

    def waist_alpha(self, step: int) -> float:
        span = max(int(self.waist_ramp), 1)
        return _smoothstep((int(step) - int(self.waist_start)) / span)

    def effort_alpha(self, step: int) -> float:
        span = max(int(self.effort_ramp), 1)
        return _smoothstep((int(step) - int(self.effort_start)) / span)

    def idle_alpha(self, step: int) -> float:
        span = max(int(self.idle_ramp), 1)
        return _smoothstep((int(step) - int(self.idle_start)) / span)

    def multipliers(self, step: int, stage_a: bool = True) -> dict[str, float]:
        a = self.alpha(step, stage_a)
        lim = self.lo_limits + (1.0 - self.lo_limits) * a
        strike = self.lo_strike + (1.0 - self.lo_strike) * a
        waist = self.waist_alpha(step)
        return {
            "waist_box": waist,
            "posture_upright": waist,
            "waist_aim": waist,
            "waist_yaw_vel": waist,
            "effort": self.effort_alpha(step),
            "posture_idle": self.idle_alpha(step),
            "collision": self.w_collision,
            "posture": self.w_posture,
            "smooth": self.w_smooth,
            "terminal": 1.0,
            "terminal_region": self.w_terminal_region,
            "terminal_velocity": strike,
            "terminal_elev": strike,
            "torque": lim,
            "velocity": lim,
            "joint_box": lim,
            "torque_peak": self.torque_peak_alpha(step),
        }


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
        # limits: the hinges are per element and small; these are their prices
        # at full ramp, after which the multipliers take over
        w_torque=20.0, w_velocity=20.0, w_joint_box=50.0,
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
        # smoothness: the ACCELERATION weight stays 10x down from the section-3
        # default, which otherwise IS the objective at bucket 25.  The JERK
        # weight is restored: a one-knot zig-zag is a second difference, and it
        # is exactly what produces the single-element torque spikes above, so
        # dividing it by 10 removed the cheapest lever on the binding gate.
        # This breaks trajopt's 10:1 acc:jerk ratio deliberately -- see
        # DECISIONS.
        w_smooth_acc=1.0e-4, w_smooth_jerk=1.0e-4,
        # posture: a real preference rather than a rounding error
        w_posture_mid=0.1, w_elbow_down=10.0,
        # IDLE ARM.  Both are EXACTLY zero -- no term, no gradient -- until a
        # home table is installed with ``FusedCost.set_idle_home``, so carrying
        # them in the training config costs nothing on a run that does not use
        # one.  Scales, against a terminal term that sits around 1.9 on a clean
        # case: a parked arm 0.5 rad from home pays 0.25 and one 0.2 rad off pays
        # 0.04, i.e. a real preference that never outbids the strike.  The
        # velocity half is two decades down because it prices rad^2/s^2 -- an arm
        # still moving at 2 rad/s where it should have stopped pays 0.08.
        w_idle_home=1.0, w_idle_vel=0.02,
    )
    base.update(kw)
    return CostConfig(**base)


def phase2_al_config(rho_scale: float = 1.0, **kw) -> ALConfig:
    """Multiplier dynamics matched to :func:`phase2_cost_config`'s scales.

    ``rho`` is per unit of *mean raw violation* in that group's own units, and
    the base weights are already O(10-100), so the multipliers only have to
    move by ones -- not by the thousands ``ALConfig``'s defaults would need at
    the declared section-3 weights.  ``lam_max`` is 100 so no group can ever
    take the objective over entirely, and ``decay`` lets a satisfied group
    relax back.
    """
    rho = {
        "torque": 0.05 * rho_scale,      # N.m, per-ELEMENT mean excess
        "velocity": 2.0 * rho_scale,     # rad/s
        "joint_box": 100.0 * rho_scale,  # rad
        "collision": 5.0 * rho_scale,    # m
        "terminal": 10.0 * rho_scale,    # m
        # N.m of per-sample MAX excess.  The SAME physical infeasibility as
        # ``torque``, reduced with a max instead of a mean over ~850
        # (knot, joint) elements, so it reads two to three decades larger: for
        # the shape these failures actually have -- ONE element over, 0.27 % of
        # them at the median -- the ratio is exactly ``n_valid``
        # (``tests/test_torque_peak_al.py`` measures 221.0x on a 221-element
        # batch), and even a grossly infeasible batch is ~1e0 mean against
        # ~1e2 max.  1/100 of ``torque``'s rho is the conservative end of that
        # range, and it lands the multiplier where the README says the peak
        # price has to live: with ``decay = 0.999`` the equilibrium is
        # ``1000 * rho * V`` = 0.5 x the batch-mean max excess, i.e. lam ~ 0.25
        # at the residual 23/243 failure level and ~1.4 on a batch that is over
        # by 2.8 N.m -- against a shared ``lam_torque`` of 7.2, which is what
        # destabilized the two continuations in section 7b.
        "torque_peak": 5.0e-4 * rho_scale,
    }
    d = dict(lam_init=1.0, rho=rho, lam_min=0.1, lam_max=100.0, ema=0.5, decay=0.999)
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


# =============================================================== config ======
@dataclass
class Phase2Config:
    """Everything the merged phase-2/3 run needs."""

    max_steps: int = 60_000
    batch_size: int = 512
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
    lr_horizon: int | None = None
    min_lr_frac: float = 0.05
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 300
    grad_clip: float = 1.0

    amp_dtype: str = "bfloat16"
    compile_mode: str | None = "default"
    #: Which units go through inductor.  The transformer compiles cleanly; the
    #: physics (``ChainFK`` / ``BatchedRNEA``) stacks 3-element spatial blocks
    #: per link, which this Triton mis-compiles (``triton_poi_fused_stack_*``,
    #: ``CompilationError`` at the 3/6 boundary) and then retries for every
    #: autotune config -- >13 minutes with no first step.  Leave the physics
    #: eager unless you have verified your toolchain compiles it.
    compile_model: bool = True
    compile_cost: bool = False
    compile_aux: bool = False
    tf32: bool = True

    # ---- objective ----
    w_cost: float = 1.0
    #: The self-supervised aux heads sit around 1.0 while the cost sits around
    #: 300, so this is the weight that makes them ~2 % of the objective.
    w_aux: float = 5.0
    #: Temporal consistency is a Huber in radians (~0.2 at the phase-1 model).
    w_tc: float = 50.0
    tc_frac: float = 0.25
    w_bptt: float = 1.0
    bptt_start_frac: float = 0.67      # ramp in over the final third
    bptt_max_frac: float = 0.25        # fraction of steps that also do a chain
    bptt_batch: int = 128

    schedule: WeightSchedule = field(default_factory=WeightSchedule)
    al: ALConfig = field(default_factory=phase2_al_config)
    al_every: int = 100

    # ---- data ----
    replay_frac_max: float = 0.9
    replay_warmup: int = 2_000
    replay_capacity: int = 40_000
    recency_bias: float = 2.0
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
    #: THE IDLE ARM.  Path to the park-posture file
    #: (``assets/idle_home_poses.json``, built by ``scripts/build_idle_home.py``),
    #: ``"default"`` for that asset if it exists, or ``None`` to disable the whole
    #: feature -- the seeds, the roller's rest stretches and the cost's
    #: ``w_idle_home`` / ``w_idle_vel`` all key off whether a table was loaded.
    idle_home_path: str | None = "default"
    #: Waypoints the retraction eases over before holding home.  Stretched
    #: automatically when the arm is far enough away that this would exceed its
    #: speed rating.  10 wp = 0.2 s.
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

    # ---- cost ----
    cost: CostConfig = field(default_factory=phase2_cost_config)
    n_sub_late: int | None = None      # bump FusedCost.n_sub after ramp_end
    bounce: BounceConfig = field(default_factory=BounceConfig)

    # ---- eval / bookkeeping ----
    eval_every: int = 2_000
    eval_cases: int = 81               # stride-subsampled acceptance grid
    log_every: int = 50
    ckpt_every: int = 2_000
    plateau_evals: int = 5
    target_acceptance: float = 0.60

    seed: int = 0
    device: str = "cuda:1"

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

    The GPU stand-in for :func:`~amortized_opt.warmstart.warmstart_trajectory`
    used when the roller respawns a slot that is carrying its post-hit state.
    Same *shape* as the numpy seed (rest-to-rest quintic timing along a joint
    straight line) without the goal-pose library lookup, which is a host-side
    dict.  It is only an initialization; the model refines it.
    """
    k = torch.arange(1, H + 1, device=q_cur.device, dtype=q_cur.dtype).view(1, H, 1)
    L = lengths.clamp_min(1).to(q_cur.dtype).unsqueeze(1)     # (B, 1, n)
    s = (k / L).clamp(0.0, 1.0)
    ease = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)        # quintic smoothstep
    return q_cur.unsqueeze(1) + ease * (terminal - q_cur).unsqueeze(1)


# ============================================================ replay pool ====
class ReplayPool(Phase2Pool):
    """Recency-biased ring buffer with the :class:`Phase2Pool` batch interface.

    Preallocated per bucket, so adding a wave of MPC contexts is a scatter and
    sampling is the same code path the fresh pool uses.
    """

    def __init__(self, like: Phase2Pool, capacity: int, buckets=None, recency_bias: float = 2.0):
        self.cfg = like.cfg
        self.noise = like.noise
        self.device = like.device
        self.dtype = like.dtype
        self.stats = {}
        self.q_lo, self.q_hi = like.q_lo, like.q_hi
        self._pelvis = like._pelvis
        self.recency_bias = float(recency_bias)
        self.capacity = int(capacity)

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
            # the replay pool cannot do (see ``waist_ref_from_seed``).
            d["waist_ref"] = torch.zeros(
                (self.capacity, b), device=self.device, dtype=self.dtype
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
        pos = self._pos[b]
        idx = (torch.arange(n, device=self.device) + pos) % self.capacity
        for k, v in d.items():
            if k in cols:
                v.index_copy_(0, idx, cols[k].to(v.dtype))
        self._order[b].index_copy_(
            0, idx, torch.arange(pos, pos + n, device=self.device, dtype=torch.long)
        )
        self._pos[b] = pos + n
        self._n[b] = min(self._n[b] + n, self.capacity)

    def sample_idx(
        self,
        bucket: int,
        n: int,
        generator: torch.Generator | None = None,
        *,
        short_wp: int = 0,
        short_boost: float = 1.0,
    ):
        """Recency-biased draw, optionally oversampling the final approach.

        The two weights MULTIPLY: recency decides which policy generation a row
        came from, ``short_boost`` decides how much of the batch sits in the
        1..``short_wp`` waypoint band the MPC's executed terminal is drawn from.
        """
        b = int(bucket)
        m = self.size(b)
        if m == 0:
            return None
        boost = short_wp > 0 and short_boost != 1.0
        if self.recency_bias <= 0.0 and not boost:
            return torch.randint(0, m, (n,), device=self.device, generator=generator)
        if self.recency_bias <= 0.0:
            w = torch.ones(m, device=self.device, dtype=self.dtype)
        else:
            order = self._order[b][:m].to(self.dtype)
            age = (float(self._pos[b] - 1) - order).clamp_min(0.0)
            w = torch.exp(-self.recency_bias * age / float(self.capacity))
        if boost:
            hit = self.data[b]["hit"][:m]
            act = self.data[b]["active"][:m]
            big = torch.full_like(hit, b + 1)
            k = torch.where(act, hit.clamp_min(0), big).min(dim=1).values
            w = torch.where(k <= int(short_wp), w * float(short_boost), w)
        return torch.multinomial(w, n, replacement=True, generator=generator)

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
        idle_settle_wp: int = 10,
        rest_ticks: int = 0,
        waist_ref: bool = False,
    ):
        #: Harvest a waist-yaw reference alongside every slot, so the replay
        #: half of the batch can price ``CostConfig.w_waist_aim`` too.  Off
        #: unless that weight is non-zero: it is a python loop over the slots
        #: (``aim_yaw`` is a scalar fixed-point solve) and there is no reason to
        #: pay for it on a run that does not use the term.
        self.want_waist_ref = bool(waist_ref)
        self.pool = pool
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
        self.yaw_i = names.index("waist_yaw_joint")
        self.library = library if library is not None else GoalPoseLibrary(joint_names=names)
        self.n_bins = int(yaw_bins)
        self.yaw_lo = float(lo[self.yaw_i])
        self.yaw_step = (float(hi[self.yaw_i]) - self.yaw_lo) / max(self.n_bins - 1, 1)
        self.lib_q, self.lib_ok = self._goal_pose_table()
        self.geom: dict[str, torch.Tensor] = {}

        # ---- the idle arm ---------------------------------------------------
        # ``idle_home`` is an ``IdleHomeTable``; when present, every hand with no
        # active goal has its arm columns re-seeded toward the park posture on
        # EVERY tick, not only on the tick it went idle.  That is what
        # keeps the target in front of the decoder (which reads ``init`` as its
        # waypoint tokens) for the whole rest stretch, and it re-derives the ramp
        # from the CURRENT measured pose, which is exactly the receding-horizon
        # convention the active arm's seed already follows.
        self.idle_settle_wp = int(idle_settle_wp)
        self.idle_q = None
        self.idle_tip = None
        if idle_home is not None:
            self.idle_q = idle_home.torch(device=self.device, dtype=dtype)  # (2, n_arm)
            self.idle_tip = idle_home.tip_tensor(device=self.device, dtype=dtype)
        #: Ticks a BOTH-HANDS-IDLE slot is allowed to keep simulating before it
        #: is respawned.  0 reproduces the historical behaviour, which *deletes*
        #: rest stretches rather than simulating them -- so the idle terms would
        #: otherwise train entirely off-distribution (README, design note step 3).
        self.rest_ticks = int(rest_ticks)

        self.state: dict[str, torch.Tensor] = {}
        self._spawn(torch.arange(self.R, device=self.device), carry=None)

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
        st = self.state
        if not st:
            for k, v in cols.items():
                shape = (self.R,) + tuple(v.shape[1:])
                st[k] = torch.zeros(shape, device=self.device, dtype=v.dtype)
            st["prev_cmd"] = torch.zeros(self.R, N_DOF, device=self.device, dtype=self.dtype)
            st["age"] = torch.zeros(self.R, device=self.device, dtype=torch.long)
            st["rest"] = torch.zeros(self.R, device=self.device, dtype=torch.long)
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
                ramp_seed(st["q0"].index_select(0, take), term, per_elem, self.H),
            )
        st["prev_cmd"].index_copy_(0, slots, st["q0"].index_select(0, slots))

    # ------------------------------------------------------------ re-goal ---
    @torch.no_grad()
    def _tick_tth(self) -> None:
        """Advance the context's time-to-hit tokens by one control tick.

        ``MPCGenerator`` rebuilds ``ContextInputs`` from a fresh
        :class:`~amortized_opt.rhythm.GoalSnapshot` every step, so its
        ``goal_tth`` / ``future_tth`` always agree with the mask's Fourier tau
        encodings (which are built from ``hit``).  The roller carries its
        context forward instead, so without this every harvested replay row
        tells the model a *stale* time-to-hit while the mask says something
        else -- a silent distribution bug independent of the re-goal fix.
        Invalid entries are left at their (zero) padding rather than drifting
        negative.
        """
        st = self.state
        st["goal_tth"] = torch.where(
            st["goal_valid"], st["goal_tth"] - self.dt, st["goal_tth"]
        )
        st["future_tth"] = torch.where(
            st["future_valid"], st["future_tth"] - self.dt, st["future_tth"]
        )

    @staticmethod
    def _shift_future(x: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
        """Drop index 0 of the ``(R, 2, F, ...)`` future axis for rows in ``sw``."""
        pad = torch.zeros_like(x[:, :, :1])
        shifted = torch.cat([x[:, :, 1:], pad], dim=2)
        m = sw.view(sw.shape + (1,) * (x.dim() - 2))
        return torch.where(m, shifted, x)

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
        # ``stale``: an idle hand whose next hit was beyond the horizon cap at
        # spawn and has now ticked inside it.  MPCGenerator gets this for free
        # from re-snapshotting; here it is the same promotion path.
        stale = (~active) & st["future_valid"][:, :, 0] & (
            st["future_tth"][:, :, 0] <= H * dt + 1e-9
        )
        switch = fired | stale
        stats = {"regoal": 0.0, "idle": 0.0, "stale": 0.0}
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
        ntth, nval = st["future_tth"][:, :, 0], st["future_valid"][:, :, 0]
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

        # ---- consume the promoted token -------------------------------------
        for f in ("future_drum", "future_pos", "future_vel", "future_tth",
                  "future_valid"):
            st[f] = self._shift_future(st[f], consume)

        # ---- re-seed ONLY the switched hand's arm columns --------------------
        yaw_bin = torch.round(
            (st["q0"][:, self.yaw_i] - self.yaw_lo) / self.yaw_step
        ).long().clamp(0, self.n_bins - 1)
        for h in (0, 1):
            rows = ok[:, h].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = self.arm_cols[h]
            q0r = st["q0"].index_select(0, rows)
            term = self.lib_q[h][
                st["drum"].index_select(0, rows)[:, h], yaw_bin.index_select(0, rows)
            ]
            L = st["hit"].index_select(0, rows)[:, h].view(-1, 1).expand(-1, len(cols))
            sub = ramp_seed(q0r[:, cols], term, L, self.H)
            buf = st["init"].index_select(0, rows)
            buf[:, :, cols] = sub
            st["init"].index_copy_(0, rows, buf)

        # ---- the shared waist -------------------------------------------------
        # MPCGenerator re-seeds the waist ONLY when both hands re-goal on the
        # same tick.  Its fresh waist comes from ``waist_reference_block``, a
        # host-side numpy aim solver that is not tensorized; here the both-case
        # holds the current measured waist pose across the horizon instead.
        # That is close to the alternative (ramping to the library entry's own
        # waist columns) precisely because the entry is selected at the CURRENT
        # yaw bin, and it never invents an aim the cost is not pricing.
        both = (ok[:, 0] & ok[:, 1]).nonzero(as_tuple=True)[0]
        if both.numel():
            wc = self.waist_cols
            hold = st["q0"].index_select(0, both)[:, wc].unsqueeze(1).expand(-1, self.H, -1)
            buf = st["init"].index_select(0, both)
            buf[:, :, wc] = hold
            st["init"].index_copy_(0, both, buf)

        stats["regoal"] = float(ok.sum())
        stats["idle"] = float(idle.sum())
        stats["stale"] = float((stale & ok).sum())
        return fired, stats

    # ----------------------------------------------------------- idle arm ---
    @torch.no_grad()
    def _seed_idle(self) -> float:
        """Re-seed every IDLE hand's arm columns: ramp to home, then hold.

        The GPU counterpart of :func:`~amortized_opt.warmstart.warmstart_trajectory`'s
        idle branch, and it must agree with it -- the roller is the training
        distribution and the numpy path is what the deployed generator runs.
        Both ease from the CURRENT pose to the park posture over a settle window
        stretched to stay inside the arm's speed rating, arrive at rest
        (``ramp_seed``'s quintic smoothstep has zero slope at both ends) and hold
        it for the rest of the horizon.  The target is a CONSTANT: arm joints are
        torso-relative, so a parked arm does not have to track the waist.

        Returns the number of hand-slots seeded, for the tick's stats.
        """
        if self.idle_q is None:
            return 0.0
        st = self.state
        R = st["q0"].shape[0]
        # ---- the idle CONTEXT token, refreshed every tick ------------------
        # Valid exactly on the hands with no active goal, which is also how
        # ``build_context_inputs`` fills it for the fresh pool.  The payload is
        # a CONSTANT per hand -- the park posture is torso-relative -- so this is
        # a broadcast, not a lookup.
        idle_v = ~st["active"]
        if "idle_valid" in st:
            m = idle_v.unsqueeze(-1).to(self.dtype)
            st["idle_valid"] = idle_v
            st["idle_home_q"] = self.idle_q.unsqueeze(0).expand(R, 2, -1) * m
            if self.idle_tip is not None and self.idle_tip.numel():
                st["idle_home_tip"] = self.idle_tip.unsqueeze(0).expand(R, 2, -1) * m
        n = 0.0
        for h in (0, 1):
            rows = (~st["active"][:, h]).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            cols = self.arm_cols[h]
            q0r = st["q0"].index_select(0, rows)[:, cols]
            term = self.idle_q[h].view(1, -1).expand(rows.numel(), -1)
            # settle length: the quintic ease peaks at 1.875x its mean rate, so
            # this is the shortest ramp that keeps the retraction inside the
            # per-joint speed rating (same rule as the numpy seed).
            step = (ETA_VELOCITY * self.step_max[cols]).clamp_min(1e-9)
            need = torch.ceil(1.875 * (term - q0r).abs() / step).amax(dim=1)
            L = need.clamp(self.idle_settle_wp, self.H).long()
            sub = ramp_seed(q0r, term, L.view(-1, 1).expand(-1, len(cols)), self.H)
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

    def _waist_aim_reference(self) -> torch.Tensor | None:
        """``(R, H)`` yaw reference per slot from ``waist_aim``, or None."""
        st = self.state
        H = self.H
        R = st["q0"].shape[0]
        q0 = st["q0"].cpu().numpy()
        p_star = st["p_star"].cpu().numpy()
        active = st["active"].cpu().numpy()
        drum = st["drum"].cpu().numpy()
        yaw_i = self.yaw_i

        from amortized_opt.waist_aim import WaistAimConfig, aim_yaw, yaw_reference

        aim_cfg = WaistAimConfig(damping=0.3, deadband=0.0)
        ref = torch.zeros(R, H, dtype=self.dtype, device=self.device)
        for i in range(R):
            yaw_start = float(q0[i, yaw_i])
            targets: dict[str, np.ndarray] = {}
            for hi, hand in enumerate(("left", "right")):
                if active[i, hi] > 0.5 and drum[i, hi] >= 0:
                    targets[hand] = np.asarray(p_star[i, hi], dtype=float)
            if not targets:
                yaw_target = yaw_start
            else:
                with np.errstate(all="ignore"):
                    yaw_target = aim_yaw(targets, cfg=aim_cfg, yaw_current=yaw_start)
            yaw_traj = yaw_reference(
                yaw_start, yaw_target, H, settle_wp=3, yawd_start=0.0, max_move_wp=10,
            )
            ref[i] = torch.as_tensor(yaw_traj, dtype=self.dtype, device=self.device)
        return ref

    def strike_context(self) -> StrikeContext:
        st = self.state
        _m, hit_eff = self.masks()
        heads = st["drum_pos"] - self.pool._pelvis
        return StrikeContext(
            hit_index=hit_eff, active=st["active"].to(self.dtype),
            target_drum=st["drum"], start_drum=st["start_drum"],
            p_star=st["p_star"], v_star=st["v_star"], n_out=st["n_out"],
            hit_radius=st["hit_radius"], disk_offset=st["disk_off"],
            half_plane_dir=st["hp_dir"], has_half_plane=st["has_hp"],
            contact_sigma=st["sigma"], shaft_allowed=st["shaft"],
            start_shaft_allowed=st["start_shaft"], elev_required=st["elev"],
            drum_head_centers=heads,
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
                if k in ("prev_cmd", "age", "rest", "kp", "kd"):
                    continue
                cols[k] = v.index_select(0, sel)
            cols["init"] = cols["init"][:, :b].contiguous()
            if ref is not None:
                cols["waist_ref"] = ref.index_select(0, sel)[:, :b].contiguous()
            out[b] = cols
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
            out = core(init, state, enc, masks)
        if was:
            core.train()
        traj = out.traj.to(self.dtype)

        # execute one waypoint, admissibly, at the LED reference point
        cmd = traj[:, 0] if self.lead_f is None else lead_waypoints(traj, self.lead_f)[:, 0]
        cmd = st["prev_cmd"] + torch.clamp(cmd - st["prev_cmd"], -self.step_max, self.step_max)
        cmd = torch.clamp(cmd, self.q_lo, self.q_hi)
        Minv = torch.linalg.inv(mass_matrix(self.rnea, st["q0"]))
        q, qd, _int = self.sim.step(st["prev_cmd"], cmd, st["q0"], st["qd0"],
                              kp=st["kp"], kd=st["kd"], Minv=Minv)
        st["prev_cmd"] = cmd
        st["q0"], st["qd0"] = q, qd

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
            rg["rest"] = float(idle_both.sum())
        else:
            fired, rg = None, {"regoal": 0.0, "idle": 0.0, "stale": 0.0}
            done = (st["hit"].clamp_min(0).max(dim=1).values <= 1) | (
                st["age"] > self.max_age
            )

        # respawn: the chain ended, aged out, or the plant left the admissible set
        bad = (~torch.isfinite(q).all(dim=1)) | (~torch.isfinite(qd).all(dim=1))
        bad = bad | ((q < self.q_lo - 0.5).any(dim=1)) | ((q > self.q_hi + 0.5).any(dim=1))
        reset = (done | bad).nonzero(as_tuple=True)[0]
        n_reset = int(reset.numel())
        n_bad = int(bad.sum())
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
        # LAST, so it covers hands that just went idle in the re-goal AND the
        # freshly spawned slots (whose pooled seed may hold a parked arm in
        # place).  The decoder reads ``init`` as its waypoint tokens, so this is
        # also how the idle target reaches the model.
        n_idle_seed = self._seed_idle()
        return {"reset": float(n_reset), "diverged": float(n_bad),
                "idle_seed": n_idle_seed, **rg}


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


#: The five loss-decomposition entries every metrics row starts with.
LOSS_KEYS: tuple[str, ...] = ("loss", "cost", "aux", "tc", "bptt")

#: APPEND-ONLY, like ``VIOL_KEYS``: ``train_phase2`` slices the metrics tensor
#: positionally and ``VIOL_OFFSET`` is derived from this length.  ``effort`` is
#: the strike-masked ``tau^2`` energy proxy (section 7d leg 2) -- an objective,
#: not a constraint, so it has no entry in ``VIOL_KEYS``.
TERM_KEYS: tuple[str, ...] = (
    "terminal", "collision", "torque", "velocity", "joint_box", "smooth", "posture",
    "effort", "waist_aim", "waist_yaw_vel",
)
#: APPEND-ONLY (``train_phase2`` reads the block positionally out of the metrics
#: tensor).  ``torque_peak`` is the MAX-shaped half of the torque hinge and its
#: own AL group -- see :mod:`amortized_opt.lagrangian`.
VIOL_KEYS: tuple[str, ...] = (
    "terminal", "collision", "torque", "velocity", "joint_box", "torque_peak",
)
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


def cost_tuple(cost: FusedCost, q0, traj, v0, strike, lengths, weights):
    """``FusedCost`` reduced to plain tensors.

    Dataclass outputs are fine for dynamo, but the whole point of this wrapper
    is that it is a *small, separately compiled* unit -- see
    :func:`make_cost_step` -- and a flat tuple is what makes the graph boundary
    unambiguous.
    """
    ct = cost.forward(q0, traj, v0, strike, lengths=lengths, weights=weights)
    return (
        ct.total,
        torch.stack([ct.terms[k].mean() for k in TERM_KEYS]),
        torch.stack([ct.violations[k].float().mean() for k in VIOL_KEYS]),
        torch.stack([ct.diagnostics[k].float().mean() for k in DIAG_KEYS]),
    )


def make_cost_step(module, cost: FusedCost, scales: LossScales, cfg: Phase2Config,
                   *, compile_mode: str | None = "default"):
    """``(batch tensors, weights) -> (loss, metrics, traj)`` for one bucket.

    THREE separately compiled units rather than one, deliberately:

    * the decoder pass,
    * :func:`cost_tuple`, and
    * the ``no_grad`` auxiliary-label pass.

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

    def _cost(q0, traj, v0, strike, lengths, weights):
        return cost_tuple(cost, q0, traj, v0, strike, lengths, weights)

    def _aux(traj, vel, acc, torque, fk_pos, fk_vel, q0, v0, em, pm):
        with torch.no_grad():
            lab = aux_targets(q0, traj.float(), v0, em.float(),
                              builder=cost.spline, fk=cost.fk, rnea=cost.rnea,
                              dt=cost.dt)
        pmf = pm.unsqueeze(-1).expand_as(fk_pos)
        beta = 0.1
        return (
            _masked_huber(vel / scales.vel, lab["vel"] / scales.vel, em, beta)[0]
            + _masked_huber(acc / scales.acc, lab["acc"] / scales.acc, em, beta)[0]
            + _masked_huber(torque / scales.torque, lab["torque"] / scales.torque,
                            em, beta)[0]
            + _masked_huber(fk_pos / scales.fk_pos, lab["fk_pos"] / scales.fk_pos,
                            pmf, beta)[0]
            + _masked_huber(fk_vel / scales.fk_vel, lab["fk_vel"] / scales.fk_vel,
                            pmf, beta)[0]
        )

    if compile_mode is not None:
        if cfg.compile_model:
            _model = torch.compile(_model, mode=compile_mode, dynamic=False)
        if cfg.compile_cost:
            _cost = torch.compile(_cost, mode=compile_mode, dynamic=False)
        if cfg.compile_aux:
            _aux = torch.compile(_aux, mode=compile_mode, dynamic=False)

    def fn(init, state, q0, v0, enc, masks, strike, lengths, weights, _unused):
        out = _model(init, state, enc, masks)
        total, terms, viols, diags = _cost(q0, out.traj, v0, strike, lengths, weights)
        loss_cost = total.mean()
        em = masks.elem_mask.to(out.traj.dtype)
        loss_aux = _aux(out.traj, out.vel, out.acc, out.torque, out.fk_pos, out.fk_vel,
                        q0, v0, em, masks.point_mask.to(out.traj.dtype))
        loss = cfg.w_cost * loss_cost + cfg.w_aux * loss_aux
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
        metrics = torch.cat([
            torch.stack([loss.detach(), loss_cost.detach(), loss_aux.detach(),
                         zero, zero]),
            terms.detach(), viols.detach(), diags.detach(),
        ])
        return loss, metrics, out.traj

    return fn


# =========================================================== checkpoints =====
def save_phase2_checkpoint(
    path, *, model, optimizer, al: AugmentedLagrangian, step: int,
    cfg: Phase2Config, model_cfg: ModelConfig, scenario_cfg=None, extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": int(step),
            "model": core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "al": al.state_dict(),
            "train_cfg": asdict(cfg),
            "model_cfg": asdict(model_cfg),
            "scenario_cfg": asdict(scenario_cfg) if scenario_cfg is not None else None,
            "extra": dict(extra or {}),
            "torch_rng": torch.get_rng_state(),
        },
        path,
    )
    return path


def load_phase2_checkpoint(path, *, model=None, optimizer=None, al=None, map_location=None):
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if al is not None and ck.get("al") is not None:
        al.load_state_dict(ck["al"])
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

    def fn(init1, state1, q0, v0, enc1, masks1, strike1, len1, hit1,
           term2, enc2, masks2, strike2, len2, elen2, weights, kp, kd,
           hand_index, normal, e_rest, noise_q, noise_v, noise_tip):
        core = module
        out1 = core(init1, state1, enc1, masks1)
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
        seed2 = ramp_seed(q_post, term2, elen2, traj1.shape[1])
        out2 = core(seed2 * masks2.elem_mask, torch.cat([q_post, v_post], -1), enc2, masks2)
        c2_total = cost_tuple(cost, q_post, out2.traj, v_post, strike2, len2, weights)[0]
        return c1_total.mean(), c2_total.mean()

    if compile_mode is None:
        return fn
    return torch.compile(fn, mode=compile_mode, dynamic=False)


# ============================================================ rolling eval ===
def eval_subset(cases, n: int):
    """Stride-subsample a grid, keeping every cell represented."""
    if n >= len(cases) or n <= 0:
        return list(cases)
    stride = max(len(cases) // int(n), 1)
    return list(cases)[::stride][: int(n)]


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


def stage_a_pass(ev: dict) -> bool:
    """The plan's stage-A unlock: hit disk >= 85 %, coll-free >= 66 %, elbow >= 95 %."""
    return (
        ev.get("ok_hit", 0.0) >= 0.85
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

    The only migration so far is ``torque_peak``.  Before it was its own group
    the peak half of the torque term was multiplied by exactly ``lam_torque``,
    so seeding ``lam["torque_peak"] = lam["torque"]`` is what makes the
    objective CONTINUOUS across a resume; leaving it out would silently drop
    the peak price back to ``lam_init`` (``AugmentedLagrangian.load_state_dict``
    is a ``dict.update``), which on a resumed run means a step change in the
    only term that moves the acceptance gate.

    ``avg_violation`` is deliberately NOT seeded: the stored value is a mean
    over elements and the new group integrates a max, two quantities ~n_valid
    apart, so the EMA is better off warming up from zero.

    Returns a NEW dict (the caller's checkpoint payload is not mutated), or the
    argument unchanged when there is nothing to do.
    """
    if not state:
        return state
    lam = dict(state.get("lam", {}))
    if "torque" in lam and "torque_peak" not in lam:
        lam["torque_peak"] = float(lam["torque"])
        out = dict(state)
        out["lam"] = lam
        if "groups" in out and "torque_peak" not in out["groups"]:
            out["groups"] = list(out["groups"]) + ["torque_peak"]
        return out
    return state


def _wtensors(mult: dict[str, float], lam: dict[str, float], device, dtype):
    """Schedule multipliers x AL multipliers, as 0-dim tensors (compile-stable)."""
    out = {}
    for k, v in mult.items():
        val = float(v) * float(lam.get(k, 1.0))
        out[k] = torch.tensor(val, device=device, dtype=dtype)
    return out


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
    s1 = replace(b1["strike"], hit_index=he1)
    s2 = replace(b2["strike"], hit_index=he2)

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
    dtype = torch.float32
    costs = build_costs(pool.buckets, cfg.cost, device=device, dtype=dtype)
    base = costs[max(costs)]
    sim = BatchedPDSim(PDSimConfig(dt=pool.cfg.dt), device=device, dtype=dtype,
                       rnea=base.rnea)
    scales = LossScales(device, dtype)
    al = AugmentedLagrangian(cfg.al)
    if al_state:
        # A resume CONTINUES the multipliers.  Restarting them would throw away
        # the balance the run spent thousands of steps discovering, and the
        # weight ramp would have to climb it again from scratch.  Checkpoints
        # written before ``torque_peak`` became its own group are migrated so
        # the peak half keeps the multiplier it was trained under.
        al.load_state_dict(migrate_al_state(al_state))

    step_fns = {b: make_cost_step(model, costs[b], scales, cfg,
                                  compile_mode=cfg.compile_mode) for b in pool.buckets}
    tc_fns = (
        {b: make_tc_step(model, costs[b], sim, cfg,
                         compile_mode=cfg.compile_mode if cfg.compile_model else None)
         for b in pool.buckets}
        if cfg.w_tc > 0.0 and cfg.tc_frac > 0.0 else {}
    )
    bptt_bucket = min(pool.buckets)
    bptt_fn = (
        make_bptt_step(model, costs[bptt_bucket], sim, cfg,
                       compile_mode=cfg.compile_mode if cfg.compile_cost else None)
        if cfg.w_bptt > 0.0 and cfg.bptt_max_frac > 0.0 else None
    )

    replay = ReplayPool(pool, cfg.replay_capacity,
                        buckets=[b for b in pool.buckets if b <= cfg.roller_horizon],
                        recency_bias=cfg.recency_bias)
    # THE IDLE ARM.  One table, three consumers: the cost prices it, the roller
    # seeds it, and (through ``scenario_cfg``) the fresh pool was generated
    # against it.  Installing it on the costs is what makes ``w_idle_home`` /
    # ``w_idle_vel`` live -- without it they are exact zeros.
    home = resolve_idle_home(cfg.idle_home_path, joint_names=pool.cfg.joint_names)
    if home is not None:
        for c in costs.values():
            c.set_idle_home(home)
        if log and rank == 0:
            print(f"idle-home: 1 park posture per hand, margin "
                  f"{home.margin * 1e3:.0f} mm, worst clearance over "
                  f"{home.min_sdf.shape[-1]} waist yaws "
                  f"{home.worst_clearance * 1e3:.1f} mm")

    roller = MPCRoller(
        pool, model=model, sim=sim, rnea=base.rnea, fk=base.fk,
        horizon=cfg.roller_horizon, slots=cfg.roller_slots,
        chain_frac=cfg.roller_chain_frac, bounce=cfg.bounce,
        device=device, dtype=dtype, seed=cfg.seed + 1301 + 17 * rank, amp_dtype=amp,
        regoal=cfg.roller_regoal, cmd_lead_s=cfg.cmd_lead_s,
        library=library, kit=kit,
        yaw_bins=cfg.roller_yaw_bins, max_age=cfg.roller_max_age,
        idle_home=home, idle_settle_wp=cfg.idle_settle_wp,
        rest_ticks=cfg.roller_rest_ticks if home is not None else 0,
        waist_ref=cfg.cost.w_waist_aim > 0.0,
    )
    roll_acc = {"reset": 0.0, "diverged": 0.0, "regoal": 0.0, "idle": 0.0,
                "stale": 0.0, "rest": 0.0, "idle_seed": 0.0}

    opt = build_optimizer(model, cfg)
    if opt_state:
        # A resume continues the MOMENTS as well as the multipliers.  Adam's
        # second moment is what sets the effective per-parameter step size, and
        # restarting it makes the first few hundred steps of a continuation
        # behave like a fresh run at a much larger LR -- on a checkpoint this
        # far into cost descent that is a real risk of undoing it.
        try:
            opt.load_state_dict(opt_state)
        except (ValueError, KeyError) as exc:  # param-group shape changed
            print(f"[resume] optimizer state NOT restored: {exc}", flush=True)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed) + 7919 + 101 * rank)
    rng = np.random.default_rng(cfg.seed + 31 * rank)

    fh = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    def emit(rec: dict) -> None:
        if fh is None:
            return
        fh.write(json.dumps(rec, default=float) + "\n")
        fh.flush()

    emit({"event": "start", "cfg": asdict(cfg), "model_cfg": asdict(model_cfg),
          "scenario_cfg": asdict(scenario_cfg) if scenario_cfg else None,
          "pool": pool.distribution(), "pool_stats": {
              k: (float(np.percentile(v, 50)) if isinstance(v, np.ndarray) else v)
              for k, v in pool.stats.items()},
          "world_size": world_size})

    bucket_p = np.array([pool.size(b) for b in pool.buckets], float)
    bucket_p /= bucket_p.sum()
    acc = np.zeros(len(METRIC_NAMES))
    acc_n = 0
    mix_acc = [0.0, 0.0, 0.0]   # sampled-batch fraction with k_min <= 3 / 5 / 8
    mix_n = 0
    viol_acc = {g: 0.0 for g in CONSTRAINT_GROUPS}
    viol_n = 0
    best = -1.0
    best_step = 0
    best_probe = -1.0          # probe score recorded with the current best.pt
    best_probe_only = -1.0     # best probe seen at all -> best_probe.pt
    last_probe: dict | None = None
    since_improve = 0
    n_skipped = 0
    t0 = time.perf_counter()
    t_last = t0
    stopped = "max_steps"
    eval_hist: list[dict] = []
    # A resume from a checkpoint that already passed stage A must seed the gate
    # OPEN, or ``WeightSchedule.gate_stage_a`` would hold the limit hinges at
    # ``lo_limits`` until the first rolling eval -- a 5 % torque price for
    # ``eval_every`` steps, on a model whose torque term is the binding gate.
    stage_a = bool(stage_a_init)

    deadline = None if max_hours is None else t0 + 3600.0 * float(max_hours)
    for step in range(int(cfg.max_steps)):
        if deadline is not None and time.perf_counter() > deadline:
            # Wall clock, not step count, is the real budget on this box.
            stopped = "max_hours"
            emit({"event": "stop", "reason": "max_hours", "step": step})
            break
        lr = cfg.lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        # ---- keep the MPC replay fed with the CURRENT policy ---------------
        if step >= cfg.roller_start:
            for _ in range(int(cfg.roller_ticks_per_step)):
                for b, cols in roller.harvest().items():
                    if b in replay.data:
                        replay.add(b, cols)
                for k_, v_ in roller.tick().items():
                    roll_acc[k_] = roll_acc.get(k_, 0.0) + float(v_)

        # ---- draw a bucket-homogeneous minibatch ---------------------------
        bi = int(rng.choice(len(pool.buckets), p=bucket_p))
        bucket = pool.buckets[bi]
        bs = cfg.batch_for(bucket)
        use_replay = (
            bucket in replay.data
            and replay.size(bucket) >= bs
            and rng.uniform() < cfg.replay_frac(step)
        )
        if use_replay:
            idx = replay.sample_idx(bucket, bs, generator=gen,
                                    short_wp=cfg.short_wp,
                                    short_boost=cfg.short_boost)
            batch = replay.batch(bucket, idx, generator=gen, noise=False)
        else:
            idx = pool.sample_idx(bucket, bs, generator=gen,
                                  short_wp=cfg.short_wp,
                                  short_boost=cfg.short_boost)
            batch = pool.batch(bucket, idx, generator=gen, noise=True)

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
            cfg.schedule.multipliers(step, stage_a), al.weights(), device, dtype
        )

        ctxm = (torch.autocast(device_type="cuda", dtype=amp) if amp is not None
                else torch.autocast(device_type="cuda", enabled=False))
        with ctxm:
            enc = encode(model, batch["context"])
            loss, metrics, _traj = step_fns[bucket](
                batch["init"], batch["state"], batch["q0"], batch["v0"],
                enc, batch["masks"], strike, lengths, w,
                torch.zeros((), device=device),
            )
            loss_tc = torch.zeros((), device=device)
            if tc_fns:
                n_tc = max(int(cfg.tc_frac * bs), 1)
                sl = slice(0, n_tc)
                masks2 = pool.masks_for(
                    (batch["hit"][sl] - 1).clamp_min(0), batch["active"][sl], bucket
                )
                kp, kd, _ki = sim.sample_gains(n_tc, generator=gen)
                enc_sub = type(enc)(enc.memory[sl], enc.mask[sl])
                loss_tc, _n = tc_fns[bucket](
                    batch["init"][sl], batch["state"][sl], batch["q0"][sl],
                    batch["v0"][sl], enc_sub,
                    type(batch["masks"])(**{f: getattr(batch["masks"], f)[sl]
                                            for f in batch["masks"].__dataclass_fields__}),
                    masks2, kp, kd,
                )
            loss_bptt = torch.zeros((), device=device)
            if bptt_fn is not None and rng.uniform() < cfg.bptt_frac(step):
                loss_bptt = _bptt_term(
                    pool, replay, bptt_fn, sim, cfg, bptt_bucket, device, dtype,
                    gen, rng, w, model=model,
                )
            total = loss + cfg.w_tc * loss_tc + cfg.w_bptt * loss_bptt

        if not torch.isfinite(total):
            # Never let one pathological batch poison the weights or the
            # multipliers; count it instead, and report the count at the end.
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            emit({"event": "skip", "step": step + 1, "loss": float(total.detach())})
            continue

        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            n_skipped += 1
            opt.zero_grad(set_to_none=True)

        m = metrics.float().detach().cpu().numpy()
        m[3] = float(loss_tc.detach())
        m[4] = float(loss_bptt.detach())
        acc += m
        acc_n += 1
        # The ``v_*`` block of the metrics tensor, in ``VIOL_KEYS`` order.  Both
        # the offset and the key list are derived, so appending a group needs no
        # edit here.
        for i, g_ in enumerate(VIOL_KEYS):
            viol_acc[g_] += float(m[VIOL_OFFSET + i])
        viol_n += 1

        if (step + 1) % cfg.al_every == 0 and viol_n:
            al.update({g_: viol_acc[g_] / viol_n for g_ in viol_acc})
            viol_acc = {g_: 0.0 for g_ in viol_acc}
            viol_n = 0

        if (step + 1) % cfg.log_every == 0 and rank == 0:
            now = time.perf_counter()
            rec = {"event": "train", "step": step + 1, "lr": lr,
                   "grad_norm": float(gnorm),
                   "steps_per_s": cfg.log_every / (now - t_last),
                   "elapsed_s": now - t0,
                   "alpha": cfg.schedule.alpha(step, stage_a),
                   "alpha_tau_peak": cfg.schedule.torque_peak_alpha(step),
                   "alpha_waist": cfg.schedule.waist_alpha(step),
                   "stage_a_open": stage_a,
                   "replay": replay.distribution(),
                   "replay_short": {b: replay.horizon_fracs(b)
                                    for b in replay.buckets},
                   "batch_short": {"le3": mix_acc[0] / max(mix_n, 1),
                                   "le5": mix_acc[1] / max(mix_n, 1),
                                   "le8": mix_acc[2] / max(mix_n, 1)},
                   "roller": {k_: v_ / max(cfg.log_every, 1) for k_, v_ in roll_acc.items()},
                   "lam": al.weights()}
            rec.update({k: float(v) for k, v in zip(METRIC_NAMES, acc / max(acc_n, 1))})
            emit(rec)
            if log:
                print(
                    f"[{step+1:6d}] L {rec['loss']:9.3f} cost {rec['cost']:9.3f} "
                    f"tc {rec['tc']:7.4f} | term {rec['c_terminal']:7.3f} "
                    f"coll {rec['c_collision']:7.3f} "
                    f"(d{rec['d_c_drum']:5.3f} u{rec['d_c_under']:5.3f} "
                    f"s{rec['d_c_self']:5.3f}) "
                    f"tau {rec['c_torque']:6.3f} "
                    f"| miss {rec['d_region']*1000:6.1f}mm sdf {rec['d_min_sdf']*1000:6.1f}mm "
                    f"| {rec['steps_per_s']:.1f} it/s", flush=True,
                )
            acc[:] = 0.0
            acc_n = 0
            mix_acc = [0.0, 0.0, 0.0]
            mix_n = 0
            roll_acc = {k_: 0.0 for k_ in roll_acc}
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
                    scenario_cfg=scenario_cfg, extra={"probe": pr_rec},
                )

        # ---- rolling eval + gates ------------------------------------------
        if rank == 0 and eval_cases is not None and (
            (step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.max_steps
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
            stage_a = stage_a or bool(ev["stage_a"])
            emit(ev)
            if log:
                print(
                    f"    eval@{step+1}: acc {ev['accepted']}/{ev['n']} "
                    f"hit {ev['ok_hit']:.2f} coll {ev['ok_collision']:.2f} "
                    f"tau {ev['ok_torque']:.2f} elbow {ev.get('elbow_down', float('nan')):.2f} "
                    f"short {ev['short_accepted']}/{ev['short_n']}", flush=True,
                )
            if ev["acceptance"] > best + 1e-4:
                # Gate on the closed loop: acceptance alone selected 7b's
                # procrastinating checkpoint.  A probe that has not run yet
                # (``last_probe is None``) does not block, so a run without
                # ``probe_fn`` behaves exactly as before.
                pscore = (float(last_probe.get("score", last_probe["in_disk"]))
                          if last_probe else None)
                if pscore is None or pscore >= best_probe - 1e-9:
                    best, best_step, since_improve = ev["acceptance"], step + 1, 0
                    if pscore is not None:
                        best_probe = pscore
                    save_phase2_checkpoint(
                        run_dir / "best.pt", model=model, optimizer=opt, al=al,
                        step=step + 1, cfg=cfg, model_cfg=model_cfg,
                        scenario_cfg=scenario_cfg,
                        extra={"eval": ev, "probe": last_probe},
                    )
                else:
                    since_improve += 1
                    emit({"event": "best_blocked", "step": step + 1,
                          "acceptance": ev["acceptance"], "probe": pscore,
                          "best_probe": best_probe})
            else:
                since_improve += 1
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

        if rank == 0 and (step + 1) % cfg.ckpt_every == 0:
            save_phase2_checkpoint(
                run_dir / "last.pt", model=model, optimizer=opt, al=al, step=step + 1,
                cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
            )

    wall = time.perf_counter() - t0
    summary = {
        "event": "done", "steps": step + 1, "wall_s": wall,
        "steps_per_s": (step + 1) / max(wall, 1e-9),
        "best_acceptance": best, "best_step": best_step, "stopped": stopped,
        "best_probe_with_best": best_probe, "best_probe_only": best_probe_only,
        "last_probe": ({k: v for k, v in last_probe.items()
                        if not k.startswith("_")} if last_probe else None),
        "skipped_steps": n_skipped,
        "stage_a_reached": stage_a, "lam": al.weights(),
        "replay": replay.distribution(), "evals": eval_hist,
    }
    if rank == 0:
        save_phase2_checkpoint(
            run_dir / "last.pt", model=model, optimizer=opt, al=al, step=step + 1,
            cfg=cfg, model_cfg=model_cfg, scenario_cfg=scenario_cfg,
        )
        emit(summary)
        fh.close()
    return summary
