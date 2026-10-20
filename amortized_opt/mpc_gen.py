"""Simulated-MPC context generator + replay store (section 5).

The plan's data engine, without Isaac:

    We simulate an actual mpc run without using isaac sim by simply starting
    with a noised quintic straight line as our init, running our transformer to
    get a result, running a simulated noised PD controller to follow the
    trajectory created and then building a new trajectory from the point the
    simulated pd controller reaches before repeating.

One tick of :meth:`MPCGenerator.run_episode`:

1. build the per-element masks for the two hands' current horizons and record a
   training sample ``(state, context, init trajectory, masks)``;
2. call the **planner** -- any callable ``(PlanRequest) -> (H, 17)``. The
   trained decoder plugs in here; :func:`quintic_planner` is the stand-in used
   by the tests and by phase-1 data generation;
3. command the plan's first waypoint and advance the noised PD simulator by one
   20 ms control tick (5 PD substeps of 4 ms, resolved decision 1);
4. shift the plan one waypoint left -- the horizon is now exactly one shorter --
   and loop.

Post-hit reset is **per hand**. When a hand's horizon reaches zero it has just
struck; that hand alone gets a new goal from the rhythm and a fresh quintic
seed built from the state the PD simulator actually reached, while the other
hand's portion of the trajectory is left byte-for-byte untouched. This is the
plan's "not resetting the entire init after a drum, decoupling the two arms".
With ``MPCGenConfig.bounce`` the striking hand also gets the BPTT placeholder
contact impulse (tip-normal restitution) before that reseed, so the next
quintic does not start from a stick still flying through the head. Off by
default: the historical generator has no drum collider.
The waist columns are rebuilt from the refreshed aim whenever ANY hand resets
(matching Phase-B distill / the training roller), so a one-hand promote cannot
leave aim tokens and the waist seed describing different drums.

Short-horizon oversampling (the plan's normalization note, "include a few more
training examples with shorter horizon lengths") is implemented by emitting
``extra_short_samples`` additional records, with independently re-noised init
trajectories, on every tick where the nearest hit is within
``short_horizon_wp`` waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from .fk import ChainFK
from .impedance import StrikeImpedance, StrikeImpedanceConfig
from .lead import is_schedule, lead_fraction, lead_waypoints_np, schedule_fraction
from .limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from .pd_sim import BatchedPDSim, PDSimConfig, frozen_minv
from .rhythm import (
    HANDS,
    DrumKit,
    GoalSnapshot,
    HandGoal,
    RhythmClip,
    aim_inputs_from_snapshot,
    build_context_inputs,
    goal_snapshot,
)
from .sdf import DrumSceneSpec, SdfPenalty
from .ik import corrected_arm_terminals
from .tokenizer import WaypointMasks, build_waypoint_masks
from .waist_aim import WaistAimConfig
from .idle_cylinder import derive_cylinder, resolve_idle_cylinder_library
from .warmstart import GoalPoseLibrary, InitNoise, arm_slice, update_waist_pr_ema, waist_pr_slice, warmstart_trajectory

__all__ = [
    "PlanRequest",
    "MPCSample",
    "EpisodeResult",
    "MPCGenConfig",
    "MPCGenerator",
    "ReplayStore",
    "quintic_planner",
    "collate",
]

PlannerFn = Callable[["PlanRequest"], np.ndarray]


# ------------------------------------------------------------------ records --
@dataclass
class PlanRequest:
    """Everything a planner sees for one MPC tick."""

    q: np.ndarray  # (17,) measured position
    qd: np.ndarray  # (17,) measured velocity
    init_traj: np.ndarray  # (H, 17) the initialization (already noised)
    snapshot: GoalSnapshot
    kit: DrumKit
    library: GoalPoseLibrary
    masks: WaypointMasks  # batch-1 tensors
    hit_index: dict[str, int]  # waypoints to each hand's hit, -1 = idle
    step: int
    t_now: float
    dt: float
    horizon: int
    #: PLANNED waist-yaw boundary condition ``(yaw, yaw_rate)`` carried forward
    #: from the previous tick's command. Seeding the yaw ramp from this instead
    #: of from the measured rate is what stops the waist winding up -- see
    #: :func:`~amortized_opt.warmstart.warmstart_trajectory`.
    yaw_state: tuple[float, float] = (0.0, 0.0)
    #: PLANNED waist row ``(n_waist,)`` from the last command. Pitch and roll
    #: are frozen at THIS rather than at the measured value; freezing at the
    #: measured value is a zero-stiffness loop and the waist random-walks out
    #: of its box (measured: 0.79 rad past a +/-0.52 rad limit in 300 ticks).
    waist_setpoint: np.ndarray | None = None
    #: The idle arm's cylinder library, so a planner that builds its own context
    #: tokens (the closed-loop probe) can fill the idle token the same way the
    #: generator seeded the idle arm.  ``None`` = no idle-cylinder in this loop.
    idle_library: object = None
    idle_cylinder: object = None
    idle_home: object = None  # legacy alias; unused when idle_library is set


@dataclass
class MPCSample:
    """One training record. Everything is numpy and detached from the loop."""

    q: np.ndarray  # (17,)
    qd: np.ndarray  # (17,)
    init_traj: np.ndarray  # (H, 17)
    elem_mask: np.ndarray  # (H, 17)
    token_mask: np.ndarray  # (H,) bool
    point_mask: np.ndarray  # (H, 6)
    tau_left: np.ndarray  # (H,)
    tau_right: np.ndarray  # (H,)
    tau0_left: float
    tau0_right: float
    hit_index: dict[str, int]
    context: dict[str, np.ndarray]  # single-row ContextInputs payload
    target_drum: np.ndarray  # (2,) long, -1 = idle -- the SDF exemption index
    min_sdf: float  # clearance of the measured state
    episode: int
    step: int
    t_now: float
    clip: str = ""
    short_horizon: bool = False


@dataclass
class EpisodeResult:
    samples: list[MPCSample]
    q_hist: np.ndarray  # (T+1, 17) measured positions
    qd_hist: np.ndarray  # (T+1, 17)
    cmd_hist: np.ndarray  # (T, 17) commanded first waypoints
    hit_steps: list[tuple[int, str, int]]  # (step, hand, drum) as they fired
    #: Pre-bounce state at each fire, aligned one-for-one with ``hit_steps``.
    #: The regular history is intentionally post-bounce for next-step rollout;
    #: contact and momentum-grace diagnostics need the state immediately before
    #: that synthetic impulse instead.
    hit_states: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    clip: str = ""
    #: True when the episode was cut short by the divergence guard (non-finite
    #: state, or a measured pose far outside the joint box).
    diverged: bool = False
    diverged_step: int = -1
    #: Ticks on which ``MPCGenConfig.freeze_wp`` suppressed the planner call and
    #: the previously committed plan was executed instead.
    frozen_ticks: int = 0
    #: Hands that received the post-hit bounce impulse (one per fire, so a
    #: two-hand tick counts twice).
    bounced_hands: int = 0


# ------------------------------------------------------------------- config --
@dataclass
class MPCGenConfig:
    horizon: int = 100
    dt: float = CONTROL_DT
    num_future: int = 3
    num_drums: int = 8
    max_steps: int | None = None
    #: Beyond this many waypoints a pending hit is not yet the "current goal".
    #: Defaults to the full horizon.
    horizon_cap_wp: int | None = None
    #: Emit this many EXTRA re-noised samples per tick once the nearest hit is
    #: within ``short_horizon_wp`` waypoints (the plan's horizon-length balance:
    #: "include a few more training examples with shorter horizon lengths").
    #: 5 waypoints = 100 ms, the point at which the plan stops refining. On a
    #: dense groove most ticks are already inside a wider window, so a larger
    #: value stops being oversampling and just multiplies the whole dataset.
    short_horizon_wp: int = 5
    extra_short_samples: int = 1
    #: Ticks between mass-matrix refreshes (1 = every tick, the honest setting).
    minv_refresh: int = 1
    #: Re-draw the PD gains at the start of every episode.
    randomize_gains: bool = True
    #: Diagnostic plant bypass.  Execute the finalized (lead/slew/box-clamped)
    #: command exactly each tick and derive measured velocity by finite
    #: difference.  This preserves the full receding-horizon loop while
    #: removing PD tracking lag, so closed-loop misses can be attributed to the
    #: plan/replanning logic rather than the servo model.
    perfect_tracking: bool = False
    hit_radius_frac: float = 0.6
    speed_min: float = 1.0
    speed_max: float = 3.0
    record_min_sdf: bool = True
    #: Ticks a just-struck drum stays exempt from this hand's collision check.
    departure_exempt_wp: int = 4
    #: THE IDLE ARM.  ``"default"`` parks a hand with no goal using the idle
    #: cylinder library in ``assets/idle_cylinder_poses.json`` if it exists,
    #: a path names another library, ``None`` restores the hold-in-place seed.
    #: This MUST match what the training roller ran with.
    idle_cylinder_path: str | None = "default"
    idle_home_path: str | None = None  # deprecated alias
    idle_settle_wp: int = 10
    #: Re-park an idle arm on EVERY tick rather than only on the tick it goes
    #: idle, which is what :class:`~amortized_opt.train_cost.MPCRoller` does.
    #: Without it the idle columns are whatever the model's own shifted plan
    #: left there, and the target quietly drifts out of the seed the model reads.
    park_idle_every_tick: bool = True
    #: Seed the waist yaw ramp from the PLANNED yaw state (the command executed
    #: last tick) rather than from the measured yaw rate. Prerequisite fix A:
    #: the measured-rate path is positive feedback and diverges. Set False only
    #: to reproduce the old behaviour in a regression test.
    planned_yaw_bc: bool = True
    #: Pitch/roll ease window on every seed (``warmstart_trajectory``'s
    #: ``waist_upright_ease_wp``).  Under ``planned_yaw_bc`` the ease recovers
    #: to the PLANNED setpoint rather than to zero -- the anchored-command
    #: analogue of ease-to-upright.  None keeps the constant-hold seed.
    waist_upright_ease_wp: int | None = None
    #: Per-joint slew bound on the quintic SEED, as a fraction of the speed
    #: rating. The seed is velocity-unlimited by construction; inside the MPC
    #: loop its first waypoint is executed, so it has to be admissible.
    seed_slew_eta: float | None = ETA_VELOCITY
    #: Per-joint slew bound applied to the executed command, whatever planner
    #: produced it -- the actuator cannot move faster than its rating, so
    #: neither may the plant's input. 1.0 = the hard rating; None disables.
    cmd_slew_eta: float | None = 1.0
    #: Clamp the executed command into the joint box before it reaches the
    #: plant.  ON, because the plant clamps: Isaac's implicit actuators saturate
    #: a position target at the joint limit and so does the hardware, and
    #: :class:`~amortized_opt.train_cost.MPCRoller` -- the loop the amortized
    #: model is TRAINED against -- has always clamped here.  Leaving it off made
    #: this loop the only place in the stack where a planner could integrate its
    #: own command outside the box: measured on ``runs/phase2_main``, all three
    #: eval clips diverged with the COMMAND leaving the box one tick before the
    #: measured pose did, walking ``waist_roll`` to -1.02 rad against a
    #: +/-0.52 rad limit at 3 % of the speed rating.  Set False only to
    #: reproduce that.
    clamp_cmd_to_box: bool = True
    #: Clamp the NEXT tick's init (shifted previous plan, idle park, post-hit
    #: quintic) into the same joint box before the planner sees it.  The command
    #: clamp above only rewrites ``traj[0]``; ``init = concat(traj[1:], tail)``
    #: otherwise hands the decoder knots that already left the box (probe
    #: prefixes: waist roll past +/-0.52 on ~91 % of ticks, peak ~2.4 rad).
    #: Freeze copies this init, so it has to be clipped *before* the freeze
    #: window, not only after the shift.  Does not clamp the decoder OUTPUT --
    #: that is ``ModelConfig.clamp_traj_to_box`` and is a training-graph choice.
    clamp_init_to_box: bool = True
    #: Abort an episode whose measured state leaves the joint box by this much
    #: (rad) or goes non-finite, instead of emitting garbage samples.
    divergence_box_margin: float = 0.5
    #: LAG COMPENSATION (plan doc, "Lag adjustment"): schedule every goal this
    #: many seconds EARLY, so the planner aims its terminal ahead of the true
    #: hit instant and a plant that lags the command still arrives on the head.
    #: Implemented by querying the rhythm at ``t_now + hit_lead_s`` rather than
    #: at ``t_now``: every downstream consumer -- the context tokens, the
    #: element masks, the terminal waypoint index, the post-hit reset -- then
    #: agrees on the shifted instant, and a hit LEAVES the snapshot once the
    #: shifted clock passes it, so it fires exactly once.  (Post-shifting the
    #: goals' ``time_to_hit`` instead makes every hit fire ``lead / dt + 1``
    #: times, because the hit stays in the clip query until its TRUE time.)
    #: A hand's recorded fire tick is therefore ``round(hit_lead_s / dt)`` ticks
    #: BEFORE the scheduled hit -- see ``EpisodeResult.hit_lead_ticks``.
    #: 0 = off (the pre-2026-08-11 behaviour); the plan's estimate is 10 ms.
    hit_lead_s: float = 0.0
    #: REFERENCE LEAD (seconds), the primary lag compensation.  The executed
    #: command is ``q_ref(t + cmd_lead_s)``, linearly interpolated between plan
    #: waypoints, so sub-tick values are meaningful and the WHOLE trajectory
    #: keeps its timing (unlike ``hit_lead_s``, which re-solves against an
    #: earlier terminal).  Scalar, or a per-joint sequence in ``joint_names``
    #: order -- ``amortized_opt.lead`` explains why per-joint is the honest
    #: version and ``scripts/fit_cmd_lead.py`` identifies it.  It must match the
    #: value the TRAINING roller executes, or the control-side lead and any
    #: lead the model has learned will stack.  0 = off.
    #: A ``{"schedule": (n_h, 17) table}`` makes the lead HORIZON-DEPENDENT --
    #: row ``h`` is used when that hand's hit is ``h`` waypoints away.  The
    #: training-side roller (``Phase2Config.cmd_lead_s``) does NOT yet accept
    #: one, so a schedule is currently a deployment-only setting.
    cmd_lead_s: float | tuple[float, ...] | dict | None = 0.0
    #: STICKY GOAL POINT.  ``rhythm.goal_snapshot`` draws the aim point with
    #: :func:`~amortized_opt.rhythm.sample_hit_point`, which JITTERS uniformly in
    #: a disk of ``hit_radius_frac`` x the legal radius about the strike aim.
    #: ``run_episode`` re-snapshots every tick, so without this the aim point for
    #: ONE scheduled hit is redrawn every 20 ms -- a target that teleports by up
    #: to ~2 x 0.6 x r_disk (0.2 m on a snare) between consecutive plans.  The
    #: training distribution has no such thing: a phase-2 scenario draws its aim
    #: once, and ``MPCRoller`` carries ``goal_pos`` forward unchanged.  With this
    #: on, the jitter is drawn ONCE per (hand, drum, absolute hit time) and held
    #: for the life of that hit, which is both the honest deployment semantics
    #: (the rhythm names one point) and the distribution the model was trained
    #: on.  Set False only to reproduce the pre-2026-08-11 behaviour.
    sticky_goal_point: bool = True
    #: Deterministic multiplier on the plant's runtime Kp / Kd tables (scalar or
    #: per-joint).  A PLANT ablation, not a randomization -- see
    #: ``PDSimConfig.kp_scale``.  Changing these invalidates the fitted
    #: ``cmd_lead_s``: the lead is a function of the servo it was fitted against.
    kp_scale: float | tuple[float, ...] = 1.0
    kd_scale: float | tuple[float, ...] = 1.0
    #: Clamp the plant's demanded torque to the joint's usable effort, as Isaac
    #: does at deployment (``PDSimConfig.saturate_effort``).  OFF = the
    #: historical, bit-for-bit plant.  A closed-loop probe measuring a policy
    #: trained against a saturating plant has to saturate too, or it is
    #: measuring a different robot.
    saturate_effort: bool = False
    #: Strike-window impedance: soften the striking hand's elbow+wrist Kp/Kd
    #: around its hit (see :mod:`amortized_opt.impedance` -- less impact torque
    #: reflected into the shoulder and waist).  ``None`` = nominal gains, the
    #: shipped behaviour.  Unlike ``kp_scale`` this is per-tick and per-hand,
    #: and it leaves the plant outside the hit windows bit-identical, so the
    #: fitted lead stays valid away from the strike.
    strike_impedance: StrikeImpedanceConfig | None = None
    #: Integral gain for PID control (N·m/rad·s).  0 = pure PD (default).
    ki: float = 0.0
    #: Anti-windup clamp on the integral error, in rad·s.  0 = no clamp.
    ki_windup: float = 0.0
    #: Reset the integral error accumulator for a hand's arm columns when that
    #: hand strikes (new goal = fresh start).  The waist integral is never reset
    #: (it is shared).
    integral_reset_on_hit: bool = True
    #: Allow the PID integral to run alongside the reference lead.  OFF by
    #: default: the lead intentionally shifts the command ahead, so the plant's
    #: error against the LED command is by design, and the integral can fight it.
    #: Set True only for sweep experiments that test combined lead + ki.
    pid_with_lead: bool = False
    #: Post-hit placeholder contact (the BPTT chain's ``BounceConfig`` /
    #: :func:`~amortized_opt.train_cost.apply_bounce`).  Reflects the striking
    #: hand's tip-normal velocity with a randomized restitution, then the
    #: per-hand quintic reseed runs from that ``(q, qd)``.  Without it the PD
    #: plant has no drum collider: the stick tunnels through the head and the
    #: next stroke is seeded from a penetrating pose.  ``None`` / ``False`` =
    #: off (historical generator).  ``True`` = default ``BounceConfig``.
    bounce: object | None = None
    #: FREEZE WINDOW (plan doc: "at ... 100ms or 5 waypoints before the hit ...
    #: we will stop refining our trajectory").  Once a HAND's hit is within this
    #: many waypoints, THAT ARM's columns of the plan are replaced by the
    #: previously committed plan shifted one waypoint, so its final approach is
    #: executed through contact instead of being re-solved from a state the
    #: model handles badly.  Measured on ``runs/phase2b_main/best.pt``: the
    #: plan's own terminal aim sits INSIDE the hit disk down to a 5-waypoint
    #: horizon (in-plane -2 mm at h = 5) and then drifts OUT of it as the
    #: horizon shrinks (+38 mm at h = 3, +61 mm at h = 1), so the last three
    #: refinements actively destroy the aim.  The waist and the other arm keep
    #: being replanned every tick -- freezing the whole row diverges.
    #: 0 = replan every element every tick (the pre-2026-08-11 behaviour).
    freeze_wp: int = 0
    #: Freeze the WAIST columns too while any hand is inside its window.  The
    #: waist carries both arms, so replanning it under a frozen arm moves that
    #: arm's tip in the world and undoes most of the freeze: measured, arm-only
    #: takes the in-plane miss 53 -> 40 mm while arm+waist takes it to -35 mm
    #: (inside the disk).  The OTHER arm is still replanned every tick either
    #: way -- freezing the whole row was measured to diverge.
    freeze_waist: bool = True
    #: Leaky EMA of the clipped P/R command (same update as ``MPCRoller``).
    #: Does not change the executed command; it is the closed-loop P/R
    #: reference the cost would read.  Off = bit-identical to the historical
    #: generator.  Skip the update on ticks where ``freeze_waist`` copied P/R.
    waist_pr_ema: bool = False
    waist_pr_ema_alpha: float = 0.1
    waist_pr_ema_leak: float = 0.02
    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)


# ------------------------------------------------------------ stand-in plan --
def quintic_planner(
    *,
    waist_cfg: WaistAimConfig | None = None,
    noise: InitNoise | None = None,
    rng: np.random.Generator | None = None,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    planned_yaw_bc: bool = True,
    slew_limit_eta: float | None = ETA_VELOCITY,
    waist_upright_ease_wp: int | None = None,
) -> PlannerFn:
    """A planner that rebuilds the quintic warm start from the current state.

    This is the phase-1 target the transformer imitates, and the stand-in the
    MPC-generator tests run against: it ignores ``init_traj`` entirely, so any
    "the loop is secretly feeding itself" bug shows up as a discontinuity
    rather than being hidden by the previous plan.

    A closed-loop property of ``waist_upright_ease_wp`` worth stating once:
    under per-tick replanning the pitch/roll ease is RE-ANCHORED at the
    measured pose on every tick.  The ease is a zero-slope quintic smoothstep,
    so one 20 ms tick advances it by only ``s(1/k)`` of the remaining lean
    (~5.8 % for ``k = 5``), and the next replan starts a fresh zero-slope
    quintic from wherever the plant actually got to -- the analytic baseline
    therefore recovers to upright roughly 3x slower in closed loop than its
    own open-loop seed suggests.  This is a known, accepted property of the
    stand-in, not a bug to compensate for; the anchored-command hold path
    (``waist_upright_ease_wp=None`` under ``planned_yaw_bc``, which pins
    pitch/roll at the PLANNED setpoint instead of easing from the measurement)
    is unaffected.
    """

    def plan(req: PlanRequest) -> np.ndarray:
        goals: dict[str, tuple[str, float] | None] = {}
        aim: dict[str, np.ndarray] = {}
        # The seed's yaw ramp aims at the same FUTURE-aware target the context
        # token carries (``build_context_inputs``); both come from this
        # snapshot through the one helper.
        _t, aim_tth, aim_futures = aim_inputs_from_snapshot(req.snapshot, frame="env")
        for hand in HANDS:
            g = req.snapshot.goal[hand]
            if not g.active or req.hit_index.get(hand, -1) < 1:
                goals[hand] = None
                aim_tth.pop(hand, None)
                continue
            goals[hand] = (g.drum_name, req.hit_index[hand] * req.dt)
            aim[hand] = np.asarray(g.position, float)
        ws = warmstart_trajectory(
            req.q,
            req.qd,
            horizon=req.horizon,
            goals=goals,
            library=req.library,
            aim_targets=aim or None,
            aim_tth=aim_tth,
            aim_futures=aim_futures,
            dt=req.dt,
            joint_names=G1_UPPER_BODY_JOINTS,
            waist_cfg=waist_cfg,
            q_bounds=q_bounds,
            yaw_state=req.yaw_state if planned_yaw_bc else None,
            waist_setpoint=req.waist_setpoint if planned_yaw_bc else None,
            waist_upright_ease_wp=waist_upright_ease_wp,
            slew_limit_eta=slew_limit_eta,
        )
        out = ws.init_traj
        if noise is not None and rng is not None:
            out = noise.apply(out, rng, hit_index=ws.hit_index)
        return out

    return plan


# --------------------------------------------------------------- the engine --
class MPCGenerator:
    """Rolls simulated MPC episodes and emits training samples."""

    def __init__(
        self,
        *,
        kit: DrumKit | None = None,
        library: GoalPoseLibrary | None = None,
        sim: BatchedPDSim | None = None,
        fk: ChainFK | None = None,
        rnea=None,
        sdf: SdfPenalty | None = None,
        noise: InitNoise | None = None,
        config: MPCGenConfig | None = None,
        waist_cfg: WaistAimConfig | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.cfg = config or MPCGenConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.names = tuple(self.cfg.joint_names)
        self.n = len(self.names)
        self.kit = kit or DrumKit.from_layout()
        self.library = library or GoalPoseLibrary(joint_names=self.names)
        self.noise = noise if noise is not None else InitNoise()
        self.waist_cfg = waist_cfg
        self.fk = fk or ChainFK(device=self.device, dtype=dtype)
        if rnea is None:
            from .rnea import BatchedRNEA

            rnea = BatchedRNEA.from_urdf(device=self.device, dtype=dtype)
        self.rnea = rnea
        self.sim = sim or BatchedPDSim(
            PDSimConfig(dt=self.cfg.dt, ki=self.cfg.ki, ki_windup=self.cfg.ki_windup,
                        kp_scale=self.cfg.kp_scale, kd_scale=self.cfg.kd_scale,
                        saturate_effort=bool(self.cfg.saturate_effort)),
            device=self.device, dtype=dtype, rnea=rnea,
        )
        self.sdf = sdf or SdfPenalty(
            DrumSceneSpec.from_layout(), fk=self.fk, device=self.device, dtype=dtype
        )
        self.q_bounds = urdf_position_limits(self.names)
        self._arm = {h: arm_slice(h, self.names) for h in HANDS}
        #: Idle-cylinder library; ``None`` keeps the historical hold-in-place seed.
        #: The DEPLOYED loop has to park exactly like the training roller does or
        #: the model is asked at run time for something it was never trained on.
        idle_path = self.cfg.idle_cylinder_path
        if idle_path is None and self.cfg.idle_home_path is not None:
            idle_path = self.cfg.idle_home_path
        self.idle_library = resolve_idle_cylinder_library(
            idle_path, joint_names=self.names
        )
        self.idle_cylinder = None
        if self.idle_library is not None:
            try:
                self.idle_cylinder = derive_cylinder(self.kit, self.fk)
            except Exception:
                self.idle_cylinder = self.idle_library.cylinder
        #: Legacy alias kept for PlanRequest / probe callers.
        self.idle_home = self.idle_library
        #: (hand, drum, absolute hit time) -> the aim point drawn for that hit.
        #: Reset per episode; see ``MPCGenConfig.sticky_goal_point``.
        self._goal_cache: dict[tuple, np.ndarray] = {}
        #: hand -> absolute time of the latest hit this episode has already
        #: STRUCK.  ``_hit_index`` pulls any goal closer than half a tick up to
        #: ``k = 1``, so without this a hit whose lead-shifted time falls inside
        #: the final half tick is handed to the planner -- and fired -- twice.
        #: Measured with ``hit_lead_s = 0.01`` (half a control period): 110
        #: recorded strikes on three clips that contain 55.
        self._consumed: dict[str, float] = {h: -1e18 for h in HANDS}

    def _clip_init(self, init: np.ndarray) -> np.ndarray:
        """Project a seed into ``q_bounds``.  No-op when ``clamp_init_to_box`` is off."""
        if not self.cfg.clamp_init_to_box:
            return init
        lo, hi = self.q_bounds
        return np.clip(init, lo, hi)

    # ------------------------------------------------------------ helpers --
    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, float), device=self.device, dtype=self.dtype)

    def _masks(self, hit_index: dict[str, int]) -> WaypointMasks:
        H = self.cfg.horizon
        dt = self.cfg.dt
        tl = self._t([max(hit_index["left"], 0) * dt])
        tr = self._t([max(hit_index["right"], 0) * dt])
        al = torch.tensor([hit_index["left"] >= 0], device=self.device)
        ar = torch.tensor([hit_index["right"] >= 0], device=self.device)
        return build_waypoint_masks(
            tl, tr, max_horizon=H, dt=dt, active_left=al, active_right=ar,
            joint_names=self.names,
        )

    def _hit_index(self, snap: GoalSnapshot) -> dict[str, int]:
        """Waypoints from now to each hand's hit; -1 when the hand is idle."""
        H = self.cfg.horizon
        cap = self.cfg.horizon_cap_wp or H
        out: dict[str, int] = {}
        for hand in HANDS:
            g = snap.goal[hand]
            if not g.active:
                out[hand] = -1
                continue
            k = int(round(g.time_to_hit / self.cfg.dt))
            # A goal is never "now": the soonest actionable terminal is the
            # first waypoint, so a sub-half-tick goal is pulled to k = 1
            # rather than collapsing to an empty horizon.
            out[hand] = max(k, 1) if 0 <= k <= cap else -1
        return out

    def _context_row(
        self,
        snap: GoalSnapshot,
        q: np.ndarray,
        *,
        yaw_now: float | None = None,
    ) -> dict[str, np.ndarray]:
        # ``yaw_now`` overrides the measured waist yaw for the context's aim
        # fields.  Under ``planned_yaw_bc`` the caller passes the PLANNED yaw
        # BC -- the same carrier (``yaw_state`` / ``PlanRequest.yaw_state``)
        # the seed's ramp starts from.  ``aim_yaw``'s dependence on
        # ``yaw_current`` has two DISCRETE switches (the +/-40 deg reach gate
        # and the 0.09 rad deadband), so evaluating the context at the
        # measured yaw while the seed uses the planned one lets >0.09 rad of
        # tracking error flip the context's base between "hold here" and full
        # commitment while the seed does the other.
        ctx = build_context_inputs(
            self.kit,
            snap,
            num_drums=self.cfg.num_drums,
            num_future=self.cfg.num_future,
            device="cpu",
            dtype=torch.float64,
            idle_library=self.idle_library,
            idle_cylinder=self.idle_cylinder,
            fk=self.fk,
            yaw_now=(
                float(yaw_now) if yaw_now is not None
                else q[self.names.index("waist_yaw_joint")]
            ),
            waist_cfg=self.waist_cfg,
        )
        return {
            k: getattr(ctx, k)[0].numpy().copy() for k in ctx.__dataclass_fields__
        }

    def _seed(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        snap: GoalSnapshot,
        hit_index: dict[str, int],
        hands: tuple[str, ...],
        rng: np.random.Generator,
        *,
        with_waist: bool,
        yaw_state: tuple[float, float] | None = None,
        waist_setpoint: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
        """Fresh noised quintic seed for ``hands`` (and optionally the waist)."""
        goals: dict[str, tuple[str, float] | None] = {}
        aim: dict[str, np.ndarray] = {}
        # Same future-aware aim inputs the context row is tokenized from, so
        # the ramp this seed lands on is the value the head parameterizes
        # against (see ``_context_row`` on why the two must not diverge).
        _t, aim_tth, aim_futures = aim_inputs_from_snapshot(snap, frame="env")
        for hand in HANDS:
            g = snap.goal[hand]
            active = g.active and hit_index[hand] >= 1
            # The aim law sees BOTH hands even when only one is being reseeded,
            # so the yaw the library pose is chosen at is the yaw the waist plan
            # is actually heading for.
            if active:
                aim[hand] = np.asarray(g.position, float)
            else:
                aim_tth.pop(hand, None)
            goals[hand] = (g.drum_name, hit_index[hand] * self.cfg.dt) if (
                active and hand in hands
            ) else None
        ws_kw = dict(
            horizon=self.cfg.horizon,
            goals=goals,
            library=self.library,
            aim_targets=aim or None,
            aim_tth=aim_tth,
            aim_futures=aim_futures,
            dt=self.cfg.dt,
            joint_names=self.names,
            waist_cfg=self.waist_cfg,
            q_bounds=self.q_bounds,
            yaw_state=yaw_state if self.cfg.planned_yaw_bc else None,
            waist_setpoint=waist_setpoint if self.cfg.planned_yaw_bc else None,
            waist_upright_ease_wp=self.cfg.waist_upright_ease_wp,
            slew_limit_eta=self.cfg.seed_slew_eta,
            idle_library=self.idle_library,
            idle_cylinder=self.idle_cylinder,
            fk=self.fk,
            idle_settle_wp=self.cfg.idle_settle_wp,
        )
        ws = warmstart_trajectory(q, qd, **ws_kw)
        if any(goals[h] is not None for h in HANDS):
            terms = corrected_arm_terminals(
                self.fk,
                [{"ws": ws, "goals": goals, "aim": aim, "hit": hit_index,
                  "sticks": None}],
                steps=1, q_bounds=self.q_bounds, joint_names=self.names,
                library=self.library,
            )[0]
            if terms:
                ws = warmstart_trajectory(q, qd, terminal_arm=terms, **ws_kw)
        noised = self.noise.apply(
            ws.init_traj, rng, hit_index=ws.hit_index, q_bounds=self.q_bounds,
            joint_names=self.names,
        )
        return noised, ws.hit_index, np.asarray(ws.yaw_ref, float)

    # ------------------------------------------------------------ episode --
    def run_episode(
        self,
        clip: RhythmClip,
        q0: np.ndarray,
        qd0: np.ndarray,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        episode: int = 0,
        planner: PlannerFn | None = None,
        max_steps: int | None = None,
        t_start: float = 0.0,
        plan_trace: list | None = None,
    ) -> EpisodeResult:
        """One simulated-MPC rollout over ``clip`` starting from ``(q0, qd0)``."""
        if rng is None:
            rng = np.random.default_rng(seed)
        cfg = self.cfg
        H, dt, n = cfg.horizon, cfg.dt, self.n
        plan = (
            planner
            if planner is not None
            else quintic_planner(
                waist_cfg=self.waist_cfg, q_bounds=self.q_bounds,
                planned_yaw_bc=cfg.planned_yaw_bc,
                slew_limit_eta=cfg.seed_slew_eta,
                waist_upright_ease_wp=cfg.waist_upright_ease_wp,
            )
        )
        self._goal_cache = {}
        self._consumed = {h: -1e18 for h in HANDS}
        yaw_i = self.names.index("waist_yaw_joint")
        q_lo, q_hi = self.q_bounds
        waist_idx = [i for i, nm in enumerate(self.names) if nm.startswith("waist_")]
        pr_idx = waist_pr_slice(self.names)
        jy_w = waist_idx.index(yaw_i)
        cmd_step_max = (
            None if cfg.cmd_slew_eta is None
            else float(cfg.cmd_slew_eta) * isaac_velocity_limits(self.names) * dt
        )
        # A horizon-dependent lead is re-read every tick from the remaining
        # horizon; a fixed one is resolved once.
        lead_sched = (
            schedule_fraction(cfg.cmd_lead_s, dt, n)
            if is_schedule(cfg.cmd_lead_s) else None
        )
        lead_f = None if lead_sched is not None else lead_fraction(cfg.cmd_lead_s, dt, n)
        lead_active = lead_sched is not None or lead_f is not None

        q = np.asarray(q0, float).reshape(n).copy()
        qd = np.asarray(qd0, float).reshape(n).copy()
        prev_cmd = q.copy()

        kp, kd, ki = (
            self.sim.sample_gains(1, generator=_torch_gen(rng, self.device))
            if cfg.randomize_gains
            else self.sim.nominal_gains(1)
        )
        imp = (
            StrikeImpedance(cfg.strike_impedance, batch=1, device=self.device)
            if cfg.strike_impedance is not None else None
        )

        steps = max_steps if max_steps is not None else cfg.max_steps
        if steps is None:
            steps = int(math.ceil((clip.duration - t_start) / dt))
        steps = max(int(steps), 0)

        snap = self._snapshot(clip, t_start, rng)
        hit_index = self._hit_index(snap)
        yaw_state = (float(q[yaw_i]), float(qd[yaw_i]))
        waist_cmd = q[waist_idx].copy()
        pr_ema = q[pr_idx].copy()
        init, _, _yr = self._seed(
            q, qd, snap, hit_index, HANDS, rng, with_waist=True, yaw_state=yaw_state,
            waist_setpoint=waist_cmd,
        )
        init = self._clip_init(init)

        samples: list[MPCSample] = []
        q_hist = [q.copy()]
        qd_hist = [qd.copy()]
        cmd_hist: list[np.ndarray] = []
        hit_steps: list[tuple[int, str, int]] = []
        hit_states: list[tuple[np.ndarray, np.ndarray]] = []
        Minv = None
        integral = np.zeros(n, dtype=float) if (cfg.ki > 0.0) else None
        # PID integral is normally disabled when the reference lead is active:
        # the lead intentionally shifts the command ahead, so the plant's
        # "error" against the led command is by design.  ``pid_with_lead``
        # overrides this for sweep experiments.
        if lead_active and not cfg.pid_with_lead:
            ki = None
            integral = None
        # hand -> (drum, ticks left on its departure exemption)
        recent: dict[str, tuple[int, int]] = {}
        recent_drum: dict[str, int] = {}
        diverged = False
        diverged_step = -1
        n_frozen = 0
        n_bounced = 0

        for step in range(steps):
            # ---- divergence guard (prerequisite fix A) ----
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))) or bool(
                np.any(q < q_lo - cfg.divergence_box_margin)
                or np.any(q > q_hi + cfg.divergence_box_margin)
            ):
                diverged, diverged_step = True, step
                break
            t_now = t_start + step * dt
            masks = self._masks(hit_index)
            ctx_row = self._context_row(
                snap, q,
                yaw_now=(yaw_state[0] if cfg.planned_yaw_bc else None),
            )
            min_sdf = (
                self._clearance(q, snap, hit_index, recent_drum)
                if cfg.record_min_sdf
                else float("nan")
            )
            target = np.array(
                [
                    snap.goal[h].drum if (snap.goal[h].active and hit_index[h] >= 0) else -1
                    for h in HANDS
                ],
                dtype=np.int64,
            )

            short = min((k for k in hit_index.values() if k >= 0), default=H) <= cfg.short_horizon_wp
            reps = 1 + (cfg.extra_short_samples if short else 0)
            for r in range(reps):
                traj = init if r == 0 else self.noise.apply(
                    init, rng, hit_index=hit_index, q_bounds=self.q_bounds,
                    joint_names=self.names,
                )
                samples.append(
                    MPCSample(
                        q=q.copy(), qd=qd.copy(), init_traj=np.asarray(traj, float).copy(),
                        elem_mask=masks.elem_mask[0].cpu().numpy().copy(),
                        token_mask=masks.token_mask[0].cpu().numpy().copy(),
                        point_mask=masks.point_mask[0].cpu().numpy().copy(),
                        tau_left=masks.tau_left[0].cpu().numpy().copy(),
                        tau_right=masks.tau_right[0].cpu().numpy().copy(),
                        tau0_left=float(masks.tau0_left[0]),
                        tau0_right=float(masks.tau0_right[0]),
                        hit_index=dict(hit_index),
                        context=ctx_row if r == 0 else {k: v.copy() for k, v in ctx_row.items()},
                        target_drum=target.copy(),
                        min_sdf=min_sdf,
                        episode=int(episode), step=int(step), t_now=float(t_now),
                        clip=clip.name, short_horizon=bool(short),
                    )
                )

            # ---- plan, then execute exactly one control tick ----
            init = self._clip_init(init)
            req = PlanRequest(
                idle_library=self.idle_library,
                idle_cylinder=self.idle_cylinder,
                idle_home=self.idle_home,
                q=q.copy(), qd=qd.copy(), init_traj=init, snapshot=snap, kit=self.kit,
                library=self.library, masks=masks, hit_index=dict(hit_index),
                step=step, t_now=t_now, dt=dt, horizon=H, yaw_state=yaw_state,
                waist_setpoint=waist_cmd.copy(),
            )
            traj = np.asarray(plan(req), float)
            if traj.shape != (H, n):
                raise ValueError(f"planner returned {traj.shape}, expected {(H, n)}")

            # ---- the freeze window (plan doc: stop refining ~100 ms out) ----
            # PER HAND, and only that hand's arm columns.  ``init`` already IS
            # the previous plan shifted one waypoint left, so copying an arm
            # block out of it is exactly "execute the last refinement through
            # contact" for that arm.  Freezing the WHOLE row instead (waist and
            # both arms) was measured to diverge: the waist stops tracking its
            # aim for 5 ticks at a time and the idle arm stops being replanned
            # at all.
            waist_frozen = False
            if cfg.freeze_wp > 0 and step > 0:
                hit_any = False
                for h in HANDS:
                    k_h = hit_index.get(h, -1)
                    if 1 <= k_h <= int(cfg.freeze_wp):
                        traj[:, self._arm[h]] = init[:, self._arm[h]]
                        hit_any = True
                if hit_any and cfg.freeze_waist:
                    traj[:, waist_idx] = init[:, waist_idx]
                    waist_frozen = True
                n_frozen += int(hit_any)

            if (
                not cfg.perfect_tracking
                and (Minv is None or (step % max(int(cfg.minv_refresh), 1) == 0))
            ):
                Minv = frozen_minv(self.rnea, self._t(q)[None, None, :])[:, 0]
            # Reference lead: execute q_ref(t + delta), interpolated along the
            # plan.  Applied to the PLAN, before the slew/box clamp, so the
            # clamp still guarantees an admissible command -- and applied on
            # EVERY tick, including the frozen ones, so the committed final
            # approach is led too.
            if lead_sched is not None:
                # Row = waypoints remaining from the EXECUTED waypoint (index 0)
                # to that hand's terminal, which sits at plan index ``kh - 1``.
                # The waist is shared, so it takes the MORE IMMINENT hand's row:
                # the near hit is the one whose tracking the torso decides.
                lead_f = np.zeros(n, dtype=float)
                last = len(lead_sched) - 1
                hs = []
                for h in HANDS:
                    kh = hit_index.get(h, -1)
                    if kh >= 1:
                        hs.append(kh - 1)
                        lead_f[self._arm[h]] = lead_sched[min(kh - 1, last)][self._arm[h]]
                if hs:
                    lead_f[waist_idx] = lead_sched[min(min(hs), last)][waist_idx]
            if lead_f is None:
                cmd = traj[0]
            else:
                # The lead may only read INSIDE the plan.  Past a hand's
                # terminal knot the decoder's output is unpriced (the element
                # mask zeroes it and no cost term touches it), so the reference
                # is HELD at the terminal there.  A consequence worth stating:
                # at the strike tick itself the hand's horizon is one waypoint
                # and the hold makes the lead a no-op -- a reference lead can
                # compensate transport delay in the middle of a trajectory, but
                # it structurally cannot advance its END, because there is no
                # trajectory beyond the end to advance into.  That is what
                # ``hit_lead_s`` is for.
                held = traj.copy()
                for h in HANDS:
                    kh = hit_index.get(h, -1)
                    if kh >= 1:
                        held[kh - 1 :, self._arm[h]] = traj[kh - 1, self._arm[h]]
                kw = max([hit_index[h] for h in HANDS if hit_index[h] >= 1],
                         default=-1)
                if kw >= 1:
                    held[kw - 1 :, waist_idx] = traj[kw - 1, waist_idx]
                cmd = lead_waypoints_np(held, lead_f)[0]
            if cmd_step_max is not None:
                cmd = prev_cmd + np.clip(cmd - prev_cmd, -cmd_step_max, cmd_step_max)
            if cfg.clamp_cmd_to_box:
                cmd = np.clip(cmd, q_lo, q_hi)
            if cfg.waist_pr_ema and not waist_frozen:
                pr_ema = np.asarray(
                    update_waist_pr_ema(
                        pr_ema, cmd[pr_idx],
                        alpha=float(cfg.waist_pr_ema_alpha),
                        leak=float(cfg.waist_pr_ema_leak),
                    ),
                    float,
                )
            if cmd is not traj[0]:
                traj = traj.copy()
                traj[0] = cmd
            if plan_trace is not None:
                # Opt-in diagnostics: the plan after freeze/lead processing,
                # before advancing the plant. Keep its true measured boundary.
                plan_trace.append({
                    "step": int(step), "time_s": float(t_now),
                    "q0": q.copy(), "v0": qd.copy(), "traj": traj.copy(),
                    "elem_mask": masks.elem_mask[0].cpu().numpy().copy(),
                    "hit": dict(hit_index),
                    "target_drum": {
                        h: int(snap.goal[h].drum) if snap.goal[h].active else -1
                        for h in HANDS
                    },
                })
            gs = None
            if imp is not None:
                # Pre-decrement ``hit_index``: an active hand has kh >= 1 and
                # strikes at the END of the tick with kh == 1, so kh - 1 is
                # this tick's ticks-to-hit (idle hands go negative = no well).
                gs = imp.scales(torch.tensor(
                    [[float(hit_index["left"] - 1), float(hit_index["right"] - 1)]],
                    device=self.device))
            if cfg.perfect_tracking:
                qd = (cmd - prev_cmd) / dt
                q = cmd.copy()
            else:
                qa, va, int_new = self.sim.step(
                    self._t(prev_cmd)[None, :], self._t(cmd)[None, :],
                    self._t(q)[None, :], self._t(qd)[None, :],
                    kp=kp, kd=kd, ki=ki, Minv=Minv,
                    integral=self._t(integral)[None, :] if integral is not None else None,
                    gain_scale=gs,
                )
                if integral is not None:
                    integral = int_new[0].detach().cpu().numpy()
                q = qa[0].detach().cpu().numpy()
                qd = va[0].detach().cpu().numpy()
            prev_cmd = cmd.copy()
            cmd_hist.append(cmd.copy())
            q_hist.append(q.copy())
            qd_hist.append(qd.copy())

            # The next tick's waist BC is the plan we just committed to, NOT the
            # measured rate: that loop is positive feedback (fix A).
            yaw_state = (
                float(traj[0, yaw_i]),
                float((traj[1, yaw_i] - traj[0, yaw_i]) / dt) if H > 1 else 0.0,
            )
            waist_cmd = traj[0, waist_idx].copy()

            # ---- shift the plan: the horizon is now one waypoint shorter ----
            init = np.concatenate([traj[1:], traj[-1][None, :]], axis=0)
            hit_index = {h: (k - 1 if k >= 0 else -1) for h, k in hit_index.items()}

            # ---- per-hand post-hit reset ----
            t_next = t_start + (step + 1) * dt
            fired = [h for h in HANDS if hit_index[h] == 0]
            q_pre_bounce = q.copy()
            qd_pre_bounce = qd.copy()
            if fired and self.cfg.bounce:
                q, qd = self._bounce_fired(q, qd, fired, snap, rng)
                q_hist[-1] = q.copy()
                qd_hist[-1] = qd.copy()
                n_bounced += len(fired)
            if imp is not None:
                imp.observe_fired(torch.tensor(
                    [["left" in fired, "right" in fired]], device=self.device))
            for h in fired:
                hit_steps.append((step + 1, h, int(snap.goal[h].drum)))
                hit_states.append((q_pre_bounce.copy(), qd_pre_bounce.copy()))
                recent[h] = (int(snap.goal[h].drum), int(cfg.departure_exempt_wp))
                # remember WHICH scheduled hit this was, so a lead that lands
                # inside the final half tick cannot hand it back next tick
                self._consumed[h] = (
                    snap.t_now + float(cfg.hit_lead_s)
                    + float(snap.goal[h].time_to_hit)
                )
            recent = {
                h: (d, k - 1) for h, (d, k) in recent.items() if k - 1 >= 0
            }
            recent_drum = {h: d for h, (d, _k) in recent.items()}
            # ---- integral reset on hit (PID anti-windup per goal) ----
            if integral is not None and cfg.integral_reset_on_hit:
                for h in fired:
                    integral[self._arm[h]] = 0.0
            snap = self._snapshot(clip, t_next, rng)
            new_index = self._hit_index(snap)
            stale = [h for h in HANDS if hit_index[h] < 0 and new_index[h] >= 0]
            reset = sorted(set(fired) | set(stale))
            # PARK THE IDLE ARMS.  Re-derived from the CURRENT pose every tick,
            # exactly as ``MPCRoller._seed_idle`` does, so the model reads the
            # same target here as it did in training.  ``hands=()`` makes every
            # hand idle inside the seed while the aim law still sees the active
            # goals, so the park pose is chosen at the yaw the waist is heading
            # for.  Hands that are re-goaling this tick are overwritten below.
            idle_hands = tuple(h for h in HANDS if new_index[h] < 0)
            if self.idle_library is not None and cfg.park_idle_every_tick and idle_hands:
                parked, _, _pyaw = self._seed(
                    q, qd, snap, new_index, (), rng, with_waist=False,
                    yaw_state=yaw_state, waist_setpoint=waist_cmd,
                )
                for h in idle_hands:
                    init[:, self._arm[h]] = parked[:, self._arm[h]]
            if reset:
                fresh, _, _fresh_yaw = self._seed(
                    q, qd, snap, new_index, tuple(reset), rng, with_waist=True,
                    yaw_state=yaw_state, waist_setpoint=waist_cmd,
                )
                for h in reset:
                    init[:, self._arm[h]] = fresh[:, self._arm[h]]
                # Waist rebuilds whenever any hand resets (not only both): the
                # aim was refreshed with the new snapshot above, and the seed's
                # waist columns track that aim -- same pin as MPCRoller._regoal.
                widx = [i for i, nm in enumerate(self.names) if nm.startswith("waist_")]
                init[:, widx] = fresh[:, widx]
                yaw_state = (
                    float(init[0, yaw_i]),
                    float((init[1, yaw_i] - init[0, yaw_i]) / dt) if H > 1 else 0.0,
                )
                waist_cmd = init[0, waist_idx].copy()
                for h in reset:
                    hit_index[h] = new_index[h]
            for h in HANDS:
                # A hand whose goal did not change keeps counting down; one whose
                # goal is gone goes idle.
                if not snap.goal[h].active:
                    hit_index[h] = -1
            init = self._clip_init(init)

        return EpisodeResult(
            samples=samples,
            q_hist=np.stack(q_hist),
            qd_hist=np.stack(qd_hist),
            cmd_hist=np.stack(cmd_hist) if cmd_hist else np.zeros((0, n)),
            hit_steps=hit_steps,
            hit_states=hit_states,
            clip=clip.name,
            diverged=diverged,
            diverged_step=diverged_step,
            frozen_ticks=int(n_frozen),
            bounced_hands=int(n_bounced),
        )

    # ------------------------------------------------------------ plumbing --
    #: A hit at exactly ``t_now`` has already been executed by the tick that
    #: reached it, so snapshots query strictly forward of the current instant.
    SNAP_EPS: float = 1e-6

    def _snapshot(self, clip: RhythmClip, t_now: float, rng: np.random.Generator) -> GoalSnapshot:
        snap = goal_snapshot(
            clip, self.kit, float(t_now) + float(self.cfg.hit_lead_s) + self.SNAP_EPS,
            num_future=self.cfg.num_future,
            rng=rng,
            hit_radius_frac=self.cfg.hit_radius_frac,
            speed_min=self.cfg.speed_min,
            speed_max=self.cfg.speed_max,
            horizon_cap=(self.cfg.horizon_cap_wp or self.cfg.horizon) * self.cfg.dt,
        )
        return self._stick(self._drop_consumed(snap))

    def _drop_consumed(self, snap: GoalSnapshot) -> GoalSnapshot:
        """Deactivate any goal this hand has already struck.

        The rhythm query is at ``t_now + hit_lead_s``, so a hit leaves the
        snapshot once the shifted clock passes it -- but ``_hit_index`` rounds
        and clamps, so a hit still 0.4 ticks in the future reads as ``k = 1``
        and is executed a tick early.  On the next tick it is STILL in the
        query window and would be executed again.  One float per hand closes it.
        """
        lead = float(self.cfg.hit_lead_s)
        eps = 0.5 * float(self.cfg.dt)

        def keep(hand: str, g):
            if not g.active:
                return g
            t_abs = snap.t_now + lead + float(g.time_to_hit)
            return g if t_abs > self._consumed[hand] + eps else HandGoal.inactive()

        return GoalSnapshot(
            t_now=snap.t_now,
            goal={h: keep(h, g) for h, g in snap.goal.items()},
            future={h: [g for g in v
                        if snap.t_now + lead + float(g.time_to_hit)
                        > self._consumed[h] + eps]
                    for h, v in snap.future.items()},
        )

    def _stick(self, snap: GoalSnapshot) -> GoalSnapshot:
        """Hold each scheduled hit's jittered aim point for the life of that hit.

        Keyed by ``(hand, drum, absolute hit time)``, which identifies the hit
        uniquely across the clip, so the goal token a plan is refining does not
        move under it between ticks.  See ``MPCGenConfig.sticky_goal_point``.
        """
        if not self.cfg.sticky_goal_point:
            return snap
        cache = self._goal_cache
        eps = 1e-6

        def fix(hand: str, g):
            if not g.active:
                return g
            key = (hand, int(g.drum),
                   round(snap.t_now + float(self.cfg.hit_lead_s)
                         + float(g.time_to_hit), 6))
            hitp = cache.get(key)
            if hitp is None:
                cache[key] = hitp = np.asarray(g.position, float).copy()
            return replace(g, position=hitp)

        return GoalSnapshot(
            t_now=snap.t_now,
            goal={h: fix(h, g) for h, g in snap.goal.items()},
            future={h: [fix(h, g) for g in v] for h, v in snap.future.items()},
        )

    def _apply_lead(self, snap: GoalSnapshot) -> GoalSnapshot:
        """Pull every scheduled hit ``cfg.hit_lead_s`` seconds earlier.

        The lag compensation the plan asks for, applied at the ONE place that
        feeds every downstream consumer: ``_hit_index`` (hence the element
        masks and the terminal waypoint), ``_context_row`` (hence ``goal_tth``)
        and ``_seed``.  Shifting the snapshot rather than the hit index keeps
        them from disagreeing.  A goal whose lead-adjusted time is already past
        is clamped to zero rather than going negative -- ``_hit_index`` then
        pulls it to the first waypoint, which is the soonest actionable
        terminal.
        """
        lead = float(self.cfg.hit_lead_s)
        if lead <= 0.0:
            return snap
        adj = lambda g: (  # noqa: E731
            g if not g.active
            else replace(g, time_to_hit=max(float(g.time_to_hit) - lead, 0.0))
        )
        return GoalSnapshot(
            t_now=snap.t_now,
            goal={h: adj(g) for h, g in snap.goal.items()},
            future={h: [adj(g) for g in v] for h, v in snap.future.items()},
        )

    def _bounce_fired(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        fired: list[str],
        snap: GoalSnapshot,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BPTT placeholder contact: reflect each firing hand's tip-normal speed.

        ``q`` is left where the PD put it (plus the config's tiny joint noise);
        only ``qd`` changes in the honest no-noise case.  Lazy-imports
        :mod:`amortized_opt.train_cost` so the historical generator path never
        pays for that module.
        """
        from .train_cost import BounceConfig, apply_bounce

        spec = self.cfg.bounce
        cfg = BounceConfig() if spec is True else spec
        q_lo, q_hi = self.q_bounds
        qt = self._t(q)[None]
        qdt = self._t(qd)[None]
        gen = torch.Generator()
        gen.manual_seed(int(rng.integers(0, 2**63 - 1)))
        for h in fired:
            g = snap.goal[h]
            if not g.active:
                continue
            normal = self._t(self.kit.axes[int(g.drum)])[None]
            hi = torch.tensor(
                [0 if h == "left" else 1], device=self.device, dtype=torch.long
            )
            qt, qdt = apply_bounce(
                qt, qdt, fk=self.fk, rnea=self.rnea, hand_index=hi,
                normal=normal, cfg=cfg, generator=gen,
            )
        qn = np.clip(qt[0].detach().cpu().numpy(), q_lo, q_hi)
        vdn = qdt[0].detach().cpu().numpy()
        return qn, vdn

    def _clearance(
        self,
        q: np.ndarray,
        snap: GoalSnapshot,
        hit_index: dict[str, int],
        recent: dict[str, int] | None = None,
    ) -> float:
        """Signed clearance of the measured pose, with the strike exemptions.

        Two drums are exempt per hand: the one it is about to strike (the stick
        is meant to reach the head) and the one it has just struck, for
        ``departure_exempt_wp`` ticks afterwards. Without the second, every
        post-hit tick reports a "collision" of about one tip radius purely
        because the stick is still resting where it landed -- the same
        departure exemption the repo's replanner needed.
        """
        recent = recent or {}
        def target(hand: str) -> int:
            if hit_index[hand] in (0, 1) and snap.goal[hand].active:
                return int(snap.goal[hand].drum)
            return int(recent.get(hand, -1))

        tl, tr = target("left"), target("right")
        with torch.no_grad():
            _, det = self.sdf.forward(
                self._t(q)[None, :],
                target_drum_left=torch.as_tensor([tl], device=self.device),
                target_drum_right=torch.as_tensor([tr], device=self.device),
                return_details=True,
            )
        return float(det.min_sdf[0])

    # ------------------------------------------------------------- driver --
    def generate(
        self,
        clips: list[RhythmClip],
        starts,
        *,
        n_episodes: int,
        seed: int = 0,
        planner: PlannerFn | None = None,
        max_steps: int | None = None,
        store: "ReplayStore | None" = None,
    ) -> "ReplayStore":
        """Roll ``n_episodes``, appending every sample to a replay store.

        ``starts`` is either a :class:`~amortized_opt.starts.StartBatch` (rows
        are consumed round-robin) or a callable ``(rng) -> (q, qd)``.
        """
        rng = np.random.default_rng(seed)
        out = store if store is not None else ReplayStore()
        for ep in range(int(n_episodes)):
            clip = clips[int(rng.integers(len(clips)))]
            if callable(starts):
                q0, qd0 = starts(rng)
            else:
                i = int(rng.integers(len(starts)))
                q0 = starts.q[i].detach().cpu().numpy()
                qd0 = starts.qd[i].detach().cpu().numpy()
            res = self.run_episode(
                clip, q0, qd0, rng=rng, episode=ep, planner=planner, max_steps=max_steps
            )
            out.extend(res.samples)
        return out


def _torch_gen(rng: np.random.Generator, device) -> torch.Generator:
    """A torch Generator seeded off the episode's numpy Generator.

    Keeps the gain draw inside the single reproducible stream instead of
    silently depending on torch's global RNG.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(rng.integers(0, 2**63 - 1)))
    return g


# ------------------------------------------------------------ replay store --
class ReplayStore:
    """Ring buffer of :class:`MPCSample` with recency-biased sampling.

    The plan: "We will continuously generate these contexts, biasing toward
    recent ones as we update the model because the trajectories created will
    change as we train". Weight of a sample ``age`` slots behind the newest is
    ``exp(-bias * age / capacity)``; ``bias = 0`` is uniform, ``bias = 3``
    makes the oldest slot ~20x less likely than the newest.
    """

    def __init__(self, capacity: int = 200_000, recency_bias: float = 1.0):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.recency_bias = float(recency_bias)
        self._buf: list[MPCSample] = []
        self._pos = 0
        self._order: list[int] = []  # insertion index per slot, for recency

    def __len__(self) -> int:
        return len(self._buf)

    def append(self, sample: MPCSample) -> None:
        if len(self._buf) < self.capacity:
            self._buf.append(sample)
            self._order.append(self._pos)
        else:
            i = self._pos % self.capacity
            self._buf[i] = sample
            self._order[i] = self._pos
        self._pos += 1

    def extend(self, samples) -> None:
        for s in samples:
            self.append(s)

    def weights(self) -> np.ndarray:
        """Recency weights over the live slots, normalized to sum 1."""
        if not self._buf:
            return np.zeros(0)
        order = np.asarray(self._order, dtype=float)
        age = float(self._pos - 1) - order  # 0 = newest
        if self.recency_bias <= 0.0:
            w = np.ones_like(age)
        else:
            w = np.exp(-self.recency_bias * age / float(self.capacity))
        return w / w.sum()

    def sample(self, n: int, rng: np.random.Generator, *, replace_: bool = True) -> list[MPCSample]:
        if not self._buf:
            return []
        idx = rng.choice(len(self._buf), size=int(n), replace=replace_, p=self.weights())
        return [self._buf[int(i)] for i in idx]

    def all(self) -> list[MPCSample]:
        return list(self._buf)


# ---------------------------------------------------------------- collation --
def collate(samples: list[MPCSample], *, device=None, dtype: torch.dtype = torch.float32):
    """Stack records into the tensors the model's forward pass wants.

    Returns ``(state, init_traj, masks, context_dict, extras)`` where ``state``
    is ``(B, 34)`` = ``[q, qd]``, ``init_traj`` is ``(B, H, 17)``, ``masks`` is
    a :class:`~amortized_opt.tokenizer.WaypointMasks`, and ``context_dict``
    holds the batched :class:`~amortized_opt.tokenizer.ContextInputs` fields.
    """
    if not samples:
        raise ValueError("no samples to collate")
    st = lambda key: torch.as_tensor(  # noqa: E731
        np.stack([getattr(s, key) for s in samples]), device=device, dtype=dtype
    )
    state = torch.cat([st("q"), st("qd")], dim=-1)
    tau0 = lambda key: torch.as_tensor(  # noqa: E731
        [getattr(s, key) for s in samples], device=device, dtype=dtype
    )
    tau0_l, tau0_r = tau0("tau0_left"), tau0("tau0_right")
    # ``hit_tth`` is not on the record and does not need to be: it is what
    # ``MPCGenerator._masks`` passed, which ``build_waypoint_masks`` defines as
    # ``tau0`` on an ACTIVE hand and 0 on an idle one -- and ``hit_index`` is
    # stored, so both cases are recoverable exactly.
    act = lambda hand: torch.as_tensor(  # noqa: E731
        [float(s.hit_index[hand] >= 0) for s in samples], device=device, dtype=dtype
    )
    masks = WaypointMasks(
        tau_left=st("tau_left"),
        tau_right=st("tau_right"),
        tau0_left=tau0_l,
        tau0_right=tau0_r,
        elem_mask=st("elem_mask"),
        token_mask=torch.as_tensor(
            np.stack([s.token_mask for s in samples]), device=device, dtype=torch.bool
        ),
        point_mask=st("point_mask"),
        hit_tth=torch.stack([tau0_l * act("left"), tau0_r * act("right")], dim=-1),
    )
    keys = samples[0].context.keys()
    ctx: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([s.context[k] for s in samples])
        if arr.dtype == bool:
            ctx[k] = torch.as_tensor(arr, device=device, dtype=torch.bool)
        elif np.issubdtype(arr.dtype, np.integer):
            ctx[k] = torch.as_tensor(arr, device=device, dtype=torch.long)
        else:
            ctx[k] = torch.as_tensor(arr, device=device, dtype=dtype)
    extras = {
        "target_drum": torch.as_tensor(
            np.stack([s.target_drum for s in samples]), device=device, dtype=torch.long
        ),
        "min_sdf": torch.as_tensor(
            np.asarray([s.min_sdf for s in samples]), device=device, dtype=dtype
        ),
        "short_horizon": torch.as_tensor(
            np.asarray([s.short_horizon for s in samples]), device=device, dtype=torch.bool
        ),
    }
    return state, st("init_traj"), masks, ctx, extras
