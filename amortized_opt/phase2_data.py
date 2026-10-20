"""Phase-2 (direct cost descent) scenario generation and on-device pool.

Section 7 of ``docs/amortized_optimizer_plan.md``.  Phase 2 has **no labels** --
the gradient comes from :class:`~amortized_opt.cost.FusedCost` -- so this module
only has to put the model in front of the right distribution of *situations*,
and to hand the cost everything it needs to price the answer.

Two scenario families (prerequisite fix C)
------------------------------------------

``plan-native`` (70 % by default)
    What :mod:`amortized_opt.warmstart_data` generates: a
    :class:`~amortized_opt.starts.StartSampler` state (90 % drum-departure,
    *both* hands placed on a drum with an outward tip velocity) plus a goal
    snapshot taken at a random instant of a random RCC clip.  Mostly dual-arm,
    mostly short horizons, always moving.

``eval-like`` (30 %)
    The deployment-relevant regime the phase-1 report found the model had never
    seen: a **rest** start (zero velocity, waist + one arm sampled in a scaled
    joint box, unused joints at the ready pose), a **single** active hand, a
    horizon drawn from 0.44-0.80 s, and no future-hit tokens.  This is the
    *shape* of the acceptance grid, drawn from a different random stream --
    post-rest first hits and single-hand passages, not the graded cases.

``cold-start`` (25 %)
    Plan-native goals on top of a start whose **waist yaw was drawn without
    looking at them**: uniform over the aim law's +/-100 deg policy range (see
    :class:`~amortized_opt.starts.StartConfig`'s ``cold_fraction``).  Every
    other family puts the torso near the action -- a drum-departure start is
    the library pose for the drum just struck, and a replayed MPC state is a
    converged one -- so ``yaw_now`` and the analytic ``aim_yaw`` arrive at the
    encoder strongly collinear, and the tether that prices ``sp - aim`` cannot
    tell the waist set-point head which of the two it is being paid to follow.
    That is what produced the 2026-08-14 closed-loop waist runaway: the head
    had learned a large ``yaw_now`` sensitivity and interpolated between the
    two instead of anchoring on the aim.  A torso 100 deg from where the beat
    needs it is the observation that separates them, and the cost then
    supervises the turn itself rather than a jitter mitigation standing in for
    it.

All three families are built against a **domain-randomized kit** (fix B) and
carry the strike geometry the terminal cost prices, derived from that same
randomized kit, so the context tokens, the collision scene and the terminal
target can never disagree.

What a scenario carries
-----------------------

============ ==============================================================
``q0/qd0``   start state
``init``     the quintic + waist-aim seed (slew-limited), the tick-0
             initialization; phase 2 noises it at batch time and never
             regresses against it
``hit``      per-hand strike waypoint index, -1 for an idle hand
strike block ``p_star``/``v_star``/``n_out``/``hit_radius``/``disk_off``/
             ``hp_dir``/``has_hp``/``sigma``/``shaft``/``start_shaft``/
             ``elev`` -- the :class:`~amortized_opt.cost.StrikeContext`
             columns, in the cost's PELVIS frame
context      the :class:`~amortized_opt.tokenizer.ContextInputs` fields, in
             the ENV frame (the tokenizer convention)
============ ==============================================================

The impossible-transition filter from phase 1 is kept: a scenario whose quintic
seed demands more than ``vel_reject_scale`` x a joint's speed rating is dropped
rather than trained on.  Cold-start rows have the furthest to travel, so their
acceptance is reported separately (``stats["cold_requested"]`` /
``stats["cold"]``) -- a filter that quietly eats the family would restore the
very collinearity it was added to break.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from .gaussian_bridge import GaussianBridgeNoise, sample_bridge
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV, ready_joint_pos
from .constants import SHAFT_HIT_SAMPLES, STICK_MIN_SEP
from .cost import StrikeContext, stick_cost_from_context
from .limits import ETA_VELOCITY, isaac_velocity_limits, urdf_position_limits
from .rhythm import (
    HAND_INDEX,
    HANDS,
    DrumKit,
    GoalSnapshot,
    HandGoal,
    KitRandomization,
    RhythmClip,
    aim_inputs_from_snapshot,
    build_context_inputs,
    default_rcc_dir,
    goal_snapshot,
    load_rcc_pool,
    sample_hit_point,
    strike_region,
    strike_velocity,
)
from .strike import (
    CYMBAL_EDGE_FRACTION,
    HIHAT_RIM_FRACTION,
    HIHAT_SHOULDER_SIGMA,
    STICK_TIP_SIGMA,
    STRIKE_SURFACE_OFFSET,
    TOM_NEAR_FRACTION,
    default_contact_sigma,
    is_cymbal_name,
    is_hihat_name,
    is_tom_name,
    shaft_hit_allowed,
    strike_elevation_required,
    strike_half_plane_dir,
)
from .tokenizer import ContextInputs, build_waypoint_masks
from .idle_cylinder import derive_cylinder, resolve_idle_cylinder_library
from .fk import ChainFK, StickRandomization, stick_tip_offset
from .ik import corrected_arm_terminals
from .warmstart import GoalPoseLibrary, warmstart_trajectory
from .warmstart_data import CONTEXT_FIELDS, bucket_for, ou_kernel
from .waist_aim import WaistAimConfig

__all__ = [
    "KIND_PLAN",
    "KIND_EVAL_LIKE",
    "KIND_COLD_START",
    "KIND_PAIR",
    "KIND_RHYTHM_IDLE",
    "KIND_POST_HIT",
    "KIND_FAR_FETCHED",
    "Phase2Noise",
    "Phase2Pool",
    "Phase2ScenarioConfig",
    "STRIKE_FIELDS",
    "drum_name_predicates",
    "generate_cost_pool",
    "generate_cost_scenarios",
    "strike_context_from_arrays",
    "strike_geometry_table",
]

KIND_PLAN = 0
KIND_EVAL_LIKE = 1
KIND_COLD_START = 2
KIND_PAIR = 3

# Rhythm-derived families (see amortized_opt.rhythm_chain_data).
from .rhythm_chain_data import (  # noqa: E402
    KIND_FAR_FETCHED,
    KIND_POST_HIT,
    KIND_RHYTHM_IDLE,
)


def pair_combinations(kit) -> tuple[list[tuple[int, int]], list[int]]:
    """``(distinct_pairs, same_drum)`` the PAIR family enumerates, as indices.

    Include crossed assignments such as left/snare + right/hi-hat. Eligibility,
    rather than lateral drum order, determines which hand/drum pairs are legal.
    """
    left_ok, right_ok = set(kit.eligible("left")), set(kit.eligible("right"))
    distinct: list[tuple[int, int]] = []
    for a in range(kit.num_drums):
        for b in range(kit.num_drums):
            if a != b and a in left_ok and b in right_ok:
                distinct.append((a, b))
    same = [d for d in range(kit.num_drums) if d in left_ok and d in right_ok]
    return distinct, same


def add_empty_schedule_provenance(
    buckets: dict[int, dict[str, np.ndarray]], *, schedule_slots: int = 2,
) -> dict[int, dict[str, np.ndarray]]:
    """Give synthetic one-goal rows the rhythm pool's schedule schema.

    Their current goals live in ``goal_*``; they have no later clip hits to
    promote, so every schedule slot is invalid and clip provenance is -1.
    This keeps all columns aligned when the rows are merged with RCC rows.
    """
    for fields in buckets.values():
        if "sched_valid" in fields:
            continue
        n = len(fields["q0"])
        if np.asarray(fields["future_valid"]).any():
            raise ValueError("empty-schedule provenance requires no future goals")
        fields.update(
            sched_drum=np.full((n, 2, schedule_slots), -1, dtype=np.int64),
            sched_pos=np.zeros((n, 2, schedule_slots, 3), dtype=np.float32),
            sched_vel=np.zeros((n, 2, schedule_slots, 3), dtype=np.float32),
            sched_dt=np.zeros((n, 2, schedule_slots), dtype=np.float32),
            sched_valid=np.zeros((n, 2, schedule_slots), dtype=np.bool_),
            clip_i=np.full(n, -1, dtype=np.int64),
            hand_hit_time=np.where(fields["goal_valid"], fields["goal_tth"],
                                   0).astype(np.float32),
        )
    return buckets

_LONG_FIELDS = frozenset({"goal_drum", "future_drum"})
_BOOL_FIELDS = frozenset(
    {"drum_valid", "goal_valid", "future_valid", "idle_valid", "stick_valid", "aim_valid"}
)

#: The per-scenario strike-geometry arrays (shapes are per sample).
STRIKE_FIELDS: tuple[str, ...] = (
    "p_star", "v_star", "n_out", "hit_radius", "disk_off", "hp_dir",
    "has_hp", "sigma", "shaft", "start_shaft", "elev",
)

#: Drum -> hand for the eval-like family. Identical to the acceptance grid's
#: map (``eval_harness.DRUM_HAND``), which is itself the repo bench's, so the
#: hand a drum is played with is one convention across the whole package.
EVAL_DRUM_HAND: dict[str, str] = {
    "snare": "right", "hihat": "left", "tom1": "right",
    "tom2": "left", "cymbal_left": "left", "cymbal_right": "right",
}


# ================================================================ config =====
@dataclass
class Phase2ScenarioConfig:
    """The phase-2 input distribution."""

    horizon: int = 100
    buckets: tuple[int, ...] = (25, 50, 100)
    dt: float = CONTROL_DT
    num_drums: int = 8
    num_future: int = 3

    #: Domain randomization of the kit -- ON, and now consistent end to end.
    randomize_kit: bool = True
    kit_randomization: KitRandomization = field(default_factory=KitRandomization)

    hit_radius_frac: float = 0.6
    speed_min: float = 1.0
    speed_max: float = 3.0

    #: Per-snapshot rhythm augmentation (see :func:`~amortized_opt.rhythm.
    #: goal_snapshot`): uniform ``+-`` jitter on each hit's time (seconds).
    #: Plan-native / cold still share the offset across a unison group.  Rhythm
    #: idle applies independent per-hit jitter.  Far is a single-strike eval
    #: grid and has no second hit to jitter.
    #: 0.0 = the pre-2026-08-14 plan-native behaviour, RNG stream included.
    #: Phase A defaults callers to ``0.05`` when constructing rhythm pools.
    hit_time_jitter_s: float = 0.0
    hit_loudness_jitter: float = 0.0

    #: Phase A / far-fetched (see ``amortized_opt.rhythm_chain_data``).
    lead_s: float = 1.0
    tip_speed_max: float = 4.0
    synth_transition_frac: float = 0.25
    #: Cover every eligible hand/drum target in far rows and single-hand clips.
    all_hand_drum_goals: bool = True
    #: Optional focused far-family coverage as (hand, drum-name) pairs.
    #: Explicit pairs override the default all-hand coverage policy.
    far_goal_pairs: tuple[tuple[str, str], ...] | None = None
    schedule_slots: int = 2
    idle_arm_noise: float = 0.05
    idle_qd_scale: float = 0.02
    #: Unused: far starts are eval-style box draws, not ready+noise.
    far_arm_noise: float = 0.18
    far_lead_s: tuple[float, float] = (0.40, 1.20)
    far_horizons: tuple[float, ...] = (0.45, 0.60, 0.80)
    far_speeds: tuple[float, ...] = (1.0, 2.0, 3.0)
    far_start_q_scale: float = 0.35
    far_exclude_eval_q_tol: float = 1e-3
    far_future_gap_s: float = 0.55
    start_sdf_tries: int = 8
    #: Torso-frame tip crossover depth cap (m) for rhythm-idle / far starts.
    #: Depth is ``max(0, tip_R.y - tip_L.y)``; ``None`` disables the gate.
    idle_max_tip_crossover_m: float | None = 0.05
    far_max_tip_crossover_m: float | None = 0.15
    #: Fresh-mix fracs for the rhythm-derived families (train wiring later).
    rhythm_idle_frac: float = 0.30
    post_hit_frac: float = 0.60
    far_fetched_frac: float = 0.10

    #: Fraction of plan-native/cold draws whose (clip, instant) is TARGETED at
    #: a rare drum instead of uniform: a drum is picked with probability
    #: proportional to 1/corpus-hit-count, one of its hits is picked uniformly,
    #: and "now" is placed 2..H waypoints before it.  This is the coverage
    #: knob for floor_tom (4.3 % of corpus hits where hihat is 35 %) and the
    #: cymbals; uniform instants inherit the corpus skew no matter how many
    #: clips are added.  Counted in ``stats["rare_boosted"]``.
    rare_drum_boost: float = 0.0

    #: Fraction of scenarios drawn from the PAIR family: both hands striking,
    #: with the (left, right) drum combination drawn UNIFORMLY instead of from
    #: the rhythm clips.
    #:
    #: Why it exists.  The clips are real grooves, so the plan-native family
    #: inherits real drumming's distribution, and that is savagely skewed:
    #: measured on the 2026-08-14 pool, two-handed rows covered only 5
    #: (left, right) pairs, with hihat+snare alone at 83.6 % and
    #: ``floor_tom`` and ``cymbal_left`` appearing in ZERO two-handed rows out
    #: of 110 k.  Quantity was never the problem -- 71 % of all strike
    #: supervision is two-handed -- but the model only ever saw one
    #: configuration of it.
    #:
    #: Enumerate every eligible ordered (left, right) drum combination,
    #: including crossed assignments.  With left/floor_tom excluded this is
    #: 36 distinct-drum pairs plus six same-drum pairs.
    pair_frac: float = 0.0
    #: Share of PAIR draws that put BOTH hands on the SAME drum.  Legal for the
    #: six drums both hands can reach (all except ``floor_tom``), so the
    #: family enumerates 36 + 6 = 42 combinations.  The two contact points are
    #: resampled until they are ``pair_same_min_sep`` apart, so the sticks are
    #: not asked to occupy one point.
    pair_same_frac: float = 6.0 / 42.0
    #: Reads :data:`~amortized_opt.constants.STICK_MIN_SEP` (0.06 m, the value
    #: this field has always had).  The MPC roller's re-goal separation guard
    #: enforces the SAME number when it promotes a goal onto the drum the other
    #: hand is already playing, so a fresh scenario and a mid-episode switch
    #: cannot disagree about how close the two beads may get.
    pair_same_min_sep: float = STICK_MIN_SEP
    #: Probability that a PAIR row's two strikes land on the SAME waypoint.
    #: Simultaneity is the hard half of two-handed play and the short bucket is
    #: where it lives (44 % of its two-handed rows are within one waypoint), so
    #: it is drawn explicitly rather than left to two independent times.
    pair_simultaneous_frac: float = 0.40
    #: Waypoint gap drawn when the two strikes are NOT simultaneous.
    pair_gap_wp: tuple[int, int] = (1, 8)
    #: Fraction of scenarios drawn from the eval-like family (fix C).
    eval_like_frac: float = 0.30
    #: Horizons the eval-like family draws from (seconds). The acceptance grid
    #: uses 0.45 / 0.60 / 0.80; this is the same band, continuous.
    eval_horizon_s: tuple[float, float] = (0.44, 0.80)
    eval_speeds: tuple[float, ...] = (1.0, 2.0, 3.0)
    eval_start_q_scale: float = 0.35
    eval_start_jitter: float = 0.05
    #: Fraction of eval-like starts given a small non-zero velocity anyway, so
    #: the family is not degenerate in ``v_start``.
    eval_moving_frac: float = 0.25
    eval_v_scale: float = 0.10

    #: Fraction of scenarios drawn COLD: plan-native goals, but a start whose
    #: waist yaw was sampled uniformly over the aim law's policy range
    #: (``waist_aim._DEFAULT_YAW_LIMIT``, +/-100 deg, clipped to the URDF box)
    #: with no reference to those goals.
    #:
    #: This is an IDENTIFIABILITY fix, not a coverage one.  On every other
    #: family the torso already sits near the action -- a drum-departure start
    #: is the library pose for the drum just struck; an MPC replay state is
    #: mid-flight and converged -- so ``yaw_now`` and the analytic ``aim_yaw``
    #: are strongly correlated inputs, and the aim law's gate makes it worse by
    #: returning ``yaw_now`` VERBATIM whenever no hand needs the torso.  A
    #: regression on two collinear inputs cannot be pinned down by the tether
    #: that prices ``sp - aim``: the 2026-08-14 waist-yaw runaway was
    #: ``waist_sp_head`` having split its weight onto ``yaw_now`` and
    #: interpolated between the two.  Cold rows decorrelate the pair BY
    #: CONSTRUCTION, so the cost supervises real large turns instead of
    #: ``Phase2Config.yaw_now_jitter`` standing in for the missing data.
    #:
    #: Drawn from the same uniform as ``eval_like_frac`` and exclusive with it,
    #: so this is a fraction of the WHOLE pool and only the plan-native share
    #: shrinks; the two must sum to <= 1.  0.0 reproduces the pre-2026-08-14
    #: distribution exactly, RNG stream included.
    cold_start_frac: float = 0.25
    #: Cold rows redraw their clip instant until the nearest hit is at least
    #: this far off (seconds), because a torso 100 deg away needs the time to
    #: get there: the seed's waist ramp is slew-limited to
    #: ``ETA_VELOCITY`` x the yaw speed rating and the ARM's quintic to its own,
    #: and ``vel_reject_scale`` drops what does not fit.  Deliberately a floor
    #: and not a window -- draws whose next hit is beyond the model's horizon
    #: (``horizon * dt``) still get through, and those rows are the valuable
    #: "turn in preparation" ones: the current goal is capped into a future
    #: token, so the only thing asking the torso to move is the beat after next.
    cold_min_lead_s: float = 0.30
    #: Redraw budget for that floor.  On a dense clip most instants already
    #: clear it; on a sparse one the floor is nearly free.  Exhausting the
    #: budget keeps the last draw rather than dropping the scenario, so a clip
    #: whose hits are all closer than the floor still contributes.
    cold_lead_tries: int = 8

    #: Plan-native "now" biased toward the run-up to a hit.
    short_horizon_frac: float = 0.25
    short_horizon_wp: int = 5

    #: Impossible-transition filter, inherited from phase 1.
    vel_reject_scale: float = 1.5
    require_active_hand: bool = True
    #: REST SCENARIOS.  Fraction of plan-native draws kept even when NEITHER hand
    #: has a goal -- i.e. the bar rest that ``require_active_hand`` otherwise
    #: throws away.  The idle-arm terms (``CostConfig.w_idle_cylinder`` /
    #: ``w_idle_vel``) are only meaningful on states where an arm is parked, so a
    #: pool with no rests trains them entirely off-distribution.  Single-hand-idle
    #: scenarios are NOT affected -- those were always kept, and are already ~45 %
    #: of the pool.  0.0 reproduces the pre-2026-08-12 distribution exactly.
    rest_frac: float = 0.0
    #: Park the idle arm via the idle-cylinder library instead of holding it
    #: where it started.  ``"default"`` uses ``assets/idle_cylinder_poses.json``
    #: when it exists, a path names another library, ``None`` keeps hold-in-place.
    idle_cylinder_path: str | None = None
    idle_home_path: str | None = None  # deprecated alias
    idle_settle_wp: int = 10
    #: RE-AIM the straight-line terminal at the drum that is ACTUALLY there.
    #: ``GoalPoseLibrary`` is keyed on (hand, drum, waist_yaw) and stores a joint
    #: pose, so it is blind to the kit randomization above; the seed's terminal
    #: therefore drifts off the head.  One damped-least-squares step on the full
    #: arm (:func:`~amortized_opt.ik.adjust_arm_terminal`), solved against the
    #: waist the yaw ramp actually REACHES at the strike index.  Default **1**
    #: (quintic-init neighborhood, ~5 cm); 0 disables.  Batched like the phase-1
    #: apex teacher's pass 2 -- do not loop ``correct_tip`` at batch 1 (200×
    #: slower).  Phase 2 seeds are always the straight-line quintic; the apex
    #: spline is a phase-1 imitation target / occasional init, not a cost-descent
    #: start.
    #:
    #: THE FITTED STICK, per scenario -- see ``ScenarioConfig.stick_rand``.
    #: ``None`` = the nominal stick, the shipped behaviour.
    stick_rand: "StickRandomization | None" = None
    correct_terminal_steps: int = 1
    correct_terminal_damping: float = 0.05
    #: The seed is EXECUTED inside the MPC roller, so bound it (fix A).
    seed_slew_eta: float | None = ETA_VELOCITY

    joint_names: tuple[str, ...] = field(default=G1_UPPER_BODY_JOINTS)
    waist_cfg: WaistAimConfig | None = None
    #: Pitch/roll ease-to-upright window for the pool's seeds
    #: (``warmstart_trajectory``'s ``waist_upright_ease_wp``).  None keeps the
    #: hold-the-measured-lean seed bit for bit.  Match the phase-1 setting: a
    #: warm start taught to recover upright, refined from seeds that hold the
    #: lean, starts every step fighting its own init.
    waist_upright_ease_wp: int | None = None

    def __post_init__(self) -> None:
        self.buckets = tuple(sorted(int(b) for b in self.buckets))
        if self.buckets[-1] != int(self.horizon):
            raise ValueError(
                f"the largest bucket {self.buckets[-1]} must equal horizon {self.horizon}"
            )
        if not 0.0 <= self.eval_like_frac <= 1.0:
            raise ValueError("eval_like_frac must be in [0, 1]")
        if not 0.0 <= self.cold_start_frac <= 1.0:
            raise ValueError("cold_start_frac must be in [0, 1]")
        if not 0.0 <= self.pair_frac <= 1.0:
            raise ValueError("pair_frac must be in [0, 1]")
        if not 0.0 <= self.pair_same_frac <= 1.0:
            raise ValueError("pair_same_frac must be in [0, 1]")
        if self.eval_like_frac + self.cold_start_frac + self.pair_frac > 1.0 + 1e-9:
            raise ValueError(
                "eval_like_frac + cold_start_frac + pair_frac must be <= 1 "
                f"(got {self.eval_like_frac} + {self.cold_start_frac} "
                f"+ {self.pair_frac})"
            )


@dataclass
class Phase2Noise:
    """OU + white perturbation of the initialization (same law as phase 1)."""

    sigma_corr_min: float = 0.02
    sigma_corr_max: float = 0.30
    sigma_white: float = 0.01
    tau_steps: float = 5.0
    clip_to_box: bool = True
    # Legacy defaults preserve reconstruction of older runs. The trainer's
    # current defaults select half-normal magnitudes and a longer OU timescale.
    magnitude_distribution: str = "log_uniform"
    sigma_scale: float = 0.30
    process: str = "ou"
    bridge: GaussianBridgeNoise | None = None

    def __post_init__(self):
        if isinstance(self.bridge, dict):
            self.bridge = GaussianBridgeNoise(**self.bridge)
        if self.process not in {"ou", "gaussian_bridge"}:
            raise ValueError("unknown initialization noise process")
        if self.process == "gaussian_bridge" and self.bridge is None:
            raise ValueError("Gaussian bridge requires its calibrated configuration")
        if self.magnitude_distribution not in {"log_uniform", "half_normal"}:
            raise ValueError("unknown initialization noise magnitude distribution")
        if not math.isfinite(self.tau_steps) or self.tau_steps <= 0:
            raise ValueError("noise tau_steps must be finite and positive")
        if not math.isfinite(self.sigma_white) or self.sigma_white < 0:
            raise ValueError("noise sigma_white must be finite and nonnegative")
        if not math.isfinite(self.sigma_scale) or self.sigma_scale < 0:
            raise ValueError("noise sigma_scale must be finite and nonnegative")
        if not (0 < self.sigma_corr_min <= self.sigma_corr_max < math.inf):
            raise ValueError("noise correlated sigma bounds must be finite and positive")


# ====================================================== strike geometry ======
def _strike_row(kit: DrumKit, drum: int, position_env: np.ndarray, velocity: np.ndarray):
    """Per-hand strike-cost columns for one goal, in the PELVIS frame."""
    name = kit.names[int(drum)]
    aim, center, disk_r, half = strike_region(kit, int(drum), frame="pelvis")
    axis = np.asarray(kit.axes[int(drum)], float)
    p = np.asarray(position_env, float) - PELVIS_IN_ENV
    # ``disk_center_offset`` is the IN-PLANE vector p_star -> disk centre, which
    # is what ``FusedCost._terminal`` subtracts before measuring the disk radius.
    off = center - p
    off = off - float(off @ axis) * axis
    v = np.asarray(velocity, float)
    nv = float(np.linalg.norm(v))
    n_out = -v / nv if nv > 1e-9 else axis
    return dict(
        p_star=p,
        v_star=v,
        n_out=n_out,
        hit_radius=float(disk_r),
        disk_off=off,
        hp_dir=np.zeros(3) if half is None else np.asarray(half, float),
        has_hp=0.0 if half is None else 1.0,
        sigma=float(default_contact_sigma(name)),
        shaft=float(shaft_hit_allowed(name)),
        start_shaft=0.0,     # filled from the DEPARTURE drum, not this one
        elev=float(strike_elevation_required(name)),
    )


def _empty_strike() -> dict:
    return dict(
        p_star=np.zeros(3), v_star=np.zeros(3), n_out=np.array([0.0, 0.0, 1.0]),
        hit_radius=0.08, disk_off=np.zeros(3), hp_dir=np.zeros(3), has_hp=0.0,
        sigma=float(STICK_TIP_SIGMA), shaft=0.0, start_shaft=0.0, elev=0.0,
    )


def drum_name_predicates(
    drum_names, *, device=None, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Per-drum-INDEX constant tables for the name predicates of :mod:`.strike`.

    The kit's drum ordering is fixed (row ``i`` is drum id ``i``) and kit
    randomization only moves positions, so every name-driven quantity --
    cymbal / hi-hat / tom class, the aim fraction, the contact sigma, shaft
    eligibility, the elevation requirement -- is a constant vector over the
    drum axis.  Names may be padded with ``""``/``None`` to the tokenizer's
    ``num_drums``; padded rows read as "plain pad", which is inert because
    nothing ever selects them.
    """
    names = list(drum_names)
    cym = torch.tensor([bool(is_cymbal_name(n)) for n in names], device=device)
    hh = torch.tensor([bool(is_hihat_name(n)) for n in names], device=device)
    tom = torch.tensor([bool(is_tom_name(n)) for n in names], device=device)
    frac = torch.where(
        hh,
        torch.full_like(cym, HIHAT_RIM_FRACTION, dtype=dtype),
        torch.where(
            cym,
            torch.full_like(cym, CYMBAL_EDGE_FRACTION, dtype=dtype),
            torch.where(
                tom,
                torch.full_like(cym, TOM_NEAR_FRACTION, dtype=dtype),
                torch.zeros(len(names), device=device, dtype=dtype),
            ),
        ),
    )
    return {
        "cymbal": cym,
        "hihat": hh,
        "tom": tom,
        #: An aim OFFSET exists for cymbals and toms; a HALF-PLANE only for
        #: cymbals (2026-08-14 -- see ``strike.strike_half_plane_dir``).  This
        #: is the batched port of that scalar rule and must track it: while it
        #: read ``cym | tom`` the two disagreed about the toms.
        "has_hp": cym,
        "aim_frac": frac,
        "sigma": torch.where(
            cym,
            torch.full_like(frac, HIHAT_SHOULDER_SIGMA),
            torch.full_like(frac, STICK_TIP_SIGMA),
        ),
        "shaft": cym.to(dtype),
        "elev": torch.tensor(
            [float(strike_elevation_required(n)) for n in names], device=device, dtype=dtype
        ),
    }


@torch.no_grad()
def strike_geometry_table(
    drum_pos_pelvis: torch.Tensor,     # (B, D, 3) head centres, PELVIS frame
    drum_axis: torch.Tensor,           # (B, D, 3) head normals (need not be unit)
    drum_radius: torch.Tensor,         # (B, D)
    drum_names,                        # length-D names, "" for padding
    *,
    surface_offset: float = STRIKE_SURFACE_OFFSET,
    rim_buffer: float = 0.01,
    predicates: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Batched port of :func:`~amortized_opt.rhythm.strike_region` and friends.

    Returns ``(B, D, ...)`` tensors ``aim`` / ``disk_center`` / ``hit_radius`` /
    ``hp_dir`` / ``has_hp`` / ``sigma`` / ``shaft`` / ``elev``, all in the frame
    the centres were handed in (the caller subtracts ``PELVIS_IN_ENV`` first, as
    :func:`_strike_row` does -- the half-plane is frame-independent because the
    robot stands at ``x = y = 0`` in both).

    ``tests/test_roller_regoal.py`` asserts this agrees with the numpy path to
    1e-6 on randomized kits, which is the contract the MPC roller relies on when
    it re-prices a hand's strike row mid-episode against the slot's OWN kit.
    """
    dev, dt_ = drum_pos_pelvis.device, drum_pos_pelvis.dtype
    p = drum_name_predicates(drum_names, device=dev, dtype=dt_) if predicates is None \
        else predicates
    c = drum_pos_pelvis
    r = drum_radius

    # ---- _unit3(axis) ----
    a_norm = drum_axis.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(drum_axis)
    up[..., 2] = 1.0
    n = torch.where(a_norm > 1e-12, drum_axis / a_norm.clamp_min(1e-30), up)

    # ---- strike_half_plane_dir ----
    toward = torch.stack([-c[..., 0], -c[..., 1], torch.zeros_like(c[..., 0])], dim=-1)
    t1 = toward - (toward * n).sum(-1, keepdim=True) * n
    t1n = t1.norm(dim=-1, keepdim=True)
    ok1 = (toward.norm(dim=-1, keepdim=True) > 1e-6) & (t1n > 1e-9)

    down = torch.zeros_like(n)
    down[..., 2] = -1.0
    t2 = down - (down * n).sum(-1, keepdim=True) * n
    t2n = t2.norm(dim=-1, keepdim=True)
    pref = torch.zeros_like(n)
    pref[..., 0] = -1.0
    t3 = pref - (pref * n).sum(-1, keepdim=True) * n
    t3n = t3.norm(dim=-1, keepdim=True)
    fallback = torch.where(
        t2n < 1e-9, t3 / t3n.clamp_min(1e-12), t2 / t2n.clamp_min(1e-12)
    )
    hp = torch.where(ok1, t1 / t1n.clamp_min(1e-30), fallback)
    has_hp = p["has_hp"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(c.shape[:-1])
    hp_dir = torch.where(has_hp.unsqueeze(-1), hp, torch.zeros_like(hp))

    # ---- _strike_aim_offset / strike_aim_on_head ----
    cym = p["cymbal"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(c.shape[:-1])
    frac = p["aim_frac"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(c.shape[:-1])
    buf = max(float(rim_buffer), 0.0)
    r_use = torch.where(cym, r, (r - buf).clamp_min(1e-3))
    off = (frac * r_use).unsqueeze(-1) * hp_dir          # zero where has_hp is False
    aim = c - float(surface_offset) * n + off
    # ``strike_disk_center_offset`` is exactly ``-off``, so the disk centre is
    # the surface-inset head centre for EVERY drum class.
    disk_center = c - float(surface_offset) * n
    hit_radius = torch.where(cym, r.clamp_min(1e-3), (r - buf).clamp_min(1e-3))

    shape = c.shape[:-1]
    return {
        "aim": aim,
        "disk_center": disk_center,
        "hit_radius": hit_radius,
        "hp_dir": hp_dir,
        "has_hp": has_hp.to(dt_),
        "sigma": p["sigma"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(shape).to(dt_),
        "shaft": p["shaft"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(shape).to(dt_),
        "elev": p["elev"].reshape((1,) * (c.dim() - 2) + (-1,)).expand(shape).to(dt_),
        "axis": n,
    }


@lru_cache(maxsize=8)
def _layout_half_plane_names(layout_path: str | None = None) -> tuple[bool, ...]:
    """Per-drum "has a half-plane", in drum-id order, under the CURRENT rule.

    Read straight off :func:`~amortized_opt.strike.strike_half_plane_dir` rather
    than re-deriving the predicate, so there is exactly one place that decides
    which heads are cut and this cannot drift from the scorer again.
    """
    from .rhythm import DrumKit

    kit = DrumKit.from_layout(layout_path)
    return tuple(
        strike_half_plane_dir(kit.positions[i], kit.axes[i], name=kit.names[i])
        is not None
        for i in range(kit.num_drums)
    )


def _layout_half_plane_mask(*, device, dtype) -> torch.Tensor:
    """``(D,)`` 0/1 gate for the terminal term's half-plane penalty."""
    return torch.as_tensor(
        _layout_half_plane_names(), device=device, dtype=dtype
    )


def strike_context_from_arrays(
    d: dict[str, torch.Tensor], idx: torch.Tensor, *, drum_head_centers: torch.Tensor,
    drum_radii: torch.Tensor | None = None,
    waist_ref: torch.Tensor | None = None, n_shaft: int | None = None,
    waist_pr_ref: torch.Tensor | None = None,
) -> StrikeContext:
    """Assemble a :class:`~amortized_opt.cost.StrikeContext` from pool columns.

    ``waist_ref`` is the ``(B, H)`` waist-yaw reference the IL term prices
    against.  It is passed in rather than picked out of ``d`` because its two
    sources differ: the fresh pool reads it off its own seed, the replay pool
    carries it as a harvested column -- see :meth:`Phase2Pool.waist_ref`.
    ``None`` leaves ``CostConfig.w_waist_aim`` with nothing to price, which is
    the pre-2026-08-12 behaviour and was silently the ONLY behaviour.
    ``waist_pr_ref`` is the same idea for pitch/roll (leaky EMA of clipped
    cmd on replay; ``None`` on the fresh pool = upright well vs 0).
    """
    pick = lambda k: d[k].index_select(0, idx)  # noqa: E731
    # THE HALF-PLANE, RECOMPUTED rather than read.  ``has_hp`` is baked into the
    # pool npz, and the cache key is the CONTEXT_FIELDS names -- it carries no
    # geometry version -- so a pool written under an older rule keeps enforcing
    # that rule in the objective forever, silently.  That is exactly what
    # happened when the toms lost their half-plane on 2026-08-14: the SCORER
    # dropped the cut (it builds its region live) while every cached pool went
    # on charging the terminal term for far-side tom contact, so the cost was
    # strictly stricter than the test it is optimising for.  Deriving it here
    # from the current rule, keyed on the target drum, makes the cached pool
    # immune to this class of drift at the price of one gather.
    hp_now = _layout_half_plane_mask(
        device=d["drum"].device, dtype=drum_head_centers.dtype
    )
    # THE FITTED STICK.  Same discipline as ``drum_head_centers``: the cost's
    # geometry is rebuilt from the very columns the model is shown, so the two
    # cannot describe different sticks.  All three stay None when the pool did
    # not randomize, and the cost then uses its baked-in nominal buffers.
    s_pts = s_ax = s_sig = None
    if n_shaft is not None and "stick_valid" in d:
        s_pts, s_ax, s_sig = stick_cost_from_context(
            pick("stick_axis").to(drum_head_centers.dtype),
            pick("stick_tip").to(drum_head_centers.dtype),
            pick("stick_length").to(drum_head_centers.dtype),
            pick("stick_valid"), n_shaft,
        )
    return StrikeContext(
        stick_points_local=s_pts, stick_axis_local=s_ax, stick_tip_sigma=s_sig,
        waist_yaw_ref=waist_ref,
        waist_pr_ref=waist_pr_ref,
        hit_index=pick("hit").clamp_min(0),
        active=pick("active").to(drum_head_centers.dtype),
        target_drum=pick("drum"),
        start_drum=pick("start_drum"),
        p_star=pick("p_star"),
        v_star=pick("v_star"),
        n_out=pick("n_out"),
        hit_radius=pick("hit_radius"),
        disk_offset=pick("disk_off"),
        half_plane_dir=pick("hp_dir"),
        # ``hp_dir`` itself is fine to reuse: it is a direction, and the cost
        # multiplies its penalty by ``has_half_plane``, so zeroing the gate is
        # enough to remove the cut.
        has_half_plane=hp_now[pick("drum").clamp_min(0).long()],
        contact_sigma=pick("sigma"),
        shaft_allowed=pick("shaft"),
        start_shaft_allowed=pick("start_shaft"),
        elev_required=pick("elev"),
        drum_head_centers=drum_head_centers,
        drum_radii=drum_radii,
        drum_axes=pick('drum_axis') if 'drum_axis' in d else None,
    )


# ============================================================ scenarios ======
def _draw_time(clip: RhythmClip, rng: np.random.Generator, cfg: Phase2ScenarioConfig) -> float:
    if clip.hits and rng.uniform() < cfg.short_horizon_frac:
        hit = clip.hits[int(rng.integers(len(clip.hits)))]
        k = int(rng.integers(1, max(int(cfg.short_horizon_wp), 1) + 1))
        return max(float(hit.time) - k * cfg.dt, 0.0)
    return float(rng.uniform(0.0, max(clip.duration, cfg.dt)))


def _draw_cold_time(
    clip: RhythmClip, rng: np.random.Generator, cfg: Phase2ScenarioConfig
) -> float:
    """A clip instant that leaves a cold torso room to turn.

    The plan-native draw is uniform (and, a quarter of the time, deliberately
    1-5 waypoints before a hit).  Handed to a start 100 deg off the beat, most
    of those are transitions no seed can build: the arm quintic and the waist
    ramp both have to cover their whole travel inside ``k_hit`` waypoints, and
    ``vel_reject_scale`` drops the result.  So cold rows redraw until the next
    hit -- EITHER hand's, since the nearest one is what sizes the seed -- is at
    least ``cold_min_lead_s`` away.

    Rejection rather than "pick a hit and step back", because stepping back by
    a lead says nothing about the hits in between: the hit that constrains the
    seed is whichever comes first, not the one that was aimed at.
    """
    t_max = max(clip.duration, cfg.dt)
    if not clip.hits:
        return float(rng.uniform(0.0, t_max))
    times = np.asarray([h.time for h in clip.hits], float)
    t = float(rng.uniform(0.0, t_max))
    for _ in range(max(int(cfg.cold_lead_tries), 1)):
        nxt = np.searchsorted(times, t - 1e-9, side="left")
        if nxt >= times.size or float(times[nxt]) - t >= float(cfg.cold_min_lead_s):
            return t
        t = float(rng.uniform(0.0, t_max))
    return t


class _BoxStartPool:
    """Chunked, collision-screened rest starts for the eval-like family.

    Mirrors ``eval_harness._sample_box_start`` with ``hold=ready``: the waist
    and the *striking* arm are drawn in a scaled posture box, unused joints
    stay at the ready pose, and the pose is rejected if it collides or if
    either tip sits in a drum's XY disk below that head.  Screening is batched
    over a chunk so it costs one SDF call per chunk rather than one per sample.
    """

    def __init__(self, sampler, cfg: Phase2ScenarioConfig, chunk: int = 256):
        self.s = sampler
        self.cfg = cfg
        self.chunk = int(chunk)
        names = tuple(cfg.joint_names)
        self.names = names
        lo, hi = urdf_position_limits(names, posture_limits=True)
        self.lo, self.hi = lo, hi
        self.mid, self.half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        self.idx = {
            h: [i for i, n in enumerate(names) if n.startswith("waist_") or n.startswith(f"{h}_")]
            for h in HANDS
        }
        self.v_limit = isaac_velocity_limits(names)
        self._buf: dict[str, list[np.ndarray]] = {h: [] for h in HANDS}

    def _refill(self, hand: str, rng: np.random.Generator, kit: DrumKit) -> None:
        cfg = self.cfg
        n = self.chunk
        idx = self.idx[hand]
        q = np.tile(ready_joint_pos(self.names), (n, 1))
        q[:, idx] = (
            self.mid[idx][None, :]
            + cfg.eval_start_q_scale * self.half[idx][None, :]
            * rng.uniform(-1.0, 1.0, size=(n, len(idx)))
        )
        q = np.clip(q, self.lo[None, :], self.hi[None, :])
        t = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a, float), device=self.s.device, dtype=self.s.dtype
        )
        heads = t(kit.positions_in("env")).unsqueeze(0).expand(n, -1, -1)
        # The screen prices the kit this scenario will be BUILT against, radii
        # included -- ``per_drum_radius`` resizes the collision cylinder, so a
        # nominal-radius screen accepts poses inside a drum that grew.
        radii = t(kit.radii).unsqueeze(0).expand(n, -1)
        qt = torch.as_tensor(q, device=self.s.device, dtype=self.s.dtype)
        with torch.no_grad():
            _p, det = self.s.sdf.forward(
                qt, drum_head_centers=heads, drum_radii=radii, return_details=True,
            )
            ok = det.min_sdf > 0.0
            # Finite under-volume SDF misses a tip that has gone *past* the
            # short under-cylinder but is still in that drum's XY shadow.
            heads_p = t(kit.positions_in("pelvis"))
            rad = t(kit.radii)
            under = torch.zeros(n, dtype=torch.bool, device=self.s.device)
            for h in HANDS:
                R, p = self.s.fk.frame(qt, f"{h}_wrist")
                loc = t(stick_tip_offset(h)).view(1, 3, 1)
                tip = p + (R @ loc).squeeze(-1)
                dxy = torch.linalg.vector_norm(
                    tip[:, None, :2] - heads_p[None, :, :2], dim=-1,
                )
                under |= (
                    (dxy <= rad) & (tip[:, None, 2] < heads_p[None, :, 2])
                ).any(dim=1)
            ok = ok & ~under
        self._buf[hand] = list(q[ok.detach().cpu().numpy()])

    def draw(self, hand: str, rng: np.random.Generator, kit: DrumKit) -> np.ndarray | None:
        if not self._buf[hand]:
            self._refill(hand, rng, kit)
        if not self._buf[hand]:
            return None
        return self._buf[hand].pop()


def generate_cost_scenarios(
    n: int,
    clips: list[RhythmClip],
    *,
    seed: int,
    cfg: Phase2ScenarioConfig | None = None,
    kit: DrumKit | None = None,
    library: GoalPoseLibrary | None = None,
    sampler=None,
    start_chunk: int = 256,
) -> dict:
    """``n`` accepted phase-2 scenarios, keyed by horizon bucket."""
    cfg = cfg or Phase2ScenarioConfig()
    rng = np.random.default_rng(seed)
    names = tuple(cfg.joint_names)
    kit = kit or DrumKit.from_layout()
    # Enumerated on the NOMINAL kit on purpose.  ``kit_randomization`` moves each
    # drum by up to 4 cm, and snare (+0.260) and tom2 (+0.180) are only 8 cm
    # apart in y -- drawing the combination set per randomized kit would let
    # those two swap sides and quietly skew the coverage this family exists to
    # make uniform.  Identity, not the jittered geometry, decides the set.
    pair_distinct, pair_same = pair_combinations(kit)
    library = library or GoalPoseLibrary(joint_names=names)
    if sampler is None:
        from .starts import StartSampler

        sampler = StartSampler(kit=kit, library=library)
    from .starts import KIND_COLD as _START_KIND_COLD

    q_bounds = urdf_position_limits(names)
    v_limit = isaac_velocity_limits(names)
    H = int(cfg.horizon)
    idle_path = cfg.idle_cylinder_path
    if idle_path is None and cfg.idle_home_path is not None:
        idle_path = cfg.idle_home_path
    idle_library = resolve_idle_cylinder_library(idle_path, joint_names=names)
    fk64 = ChainFK(device="cpu", dtype=torch.float64)
    idle_cylinder = None
    if idle_library is not None:
        try:
            idle_cylinder = derive_cylinder(kit, fk64)
        except Exception:
            idle_cylinder = idle_library.cylinder
    box = _BoxStartPool(sampler, cfg)
    eval_names = [n_ for n_ in EVAL_DRUM_HAND if n_ in kit.names]

    keys = ("q0", "qd0", "init", "hit", "active", "drum", "start_drum", "kind",
            *STRIKE_FIELDS, *CONTEXT_FIELDS)
    out: dict[int, dict[str, list]] = {b: {k: [] for k in keys} for b in cfg.buckets}
    stats = {
        "requested": int(n), "accepted": 0, "reject_idle": 0, "reject_velocity": 0,
        "reject_library": 0, "reject_box_start": 0, "dual_active": 0,
        "asymmetric": 0, "short_horizon": 0, "eval_like": 0, "rest": 0,
        "pair": 0, "pair_same": 0, "pair_same_tight": 0,
        # Cold rows are counted at DRAW time and again at accept time, so
        # ``cold / cold_requested`` is the family's acceptance rate and a
        # collapse under the impossible-transition filter is visible without
        # regenerating anything.  ``reject_cold_start`` is the other way to
        # lose one: the SDF screen exhausted its budget and the sampler handed
        # back a yaw-0 fallback, which is not a cold start.
        "cold_requested": 0, "cold": 0, "reject_cold_start": 0,
        "single_idle": 0, "terminal_corrected": 0,
        "rare_boosted": 0,
        "peak_rate_ratio": [],
    }

    # (clip, hit-time) index per drum for the targeted rare-drum draw, with
    # inverse-frequency drum weights.  Built once; ``None`` when the knob is
    # off, so the RNG stream of a boost-free run is untouched.
    rare_index = None
    if cfg.rare_drum_boost > 0.0:
        by_drum: dict[int, list[tuple[int, float]]] = {}
        for ci_r, c_ in enumerate(clips):
            for h_ in c_.hits:
                by_drum.setdefault(int(h_.drum), []).append((ci_r, float(h_.time)))
        if by_drum:
            r_drums = sorted(by_drum)
            inv = np.array([1.0 / len(by_drum[d]) for d in r_drums], float)
            rare_index = (r_drums, [by_drum[d] for d in r_drums], inv / inv.sum())

    starts = None
    si = 0
    cold_starts = None
    ci = 0
    pending: list[dict] = []
    _CHUNK = 512

    def _emit_row(
        *, q0, qd0, init, hit, active, kit_s, snap, sticks, src, kind,
        eval_like, cold, Hb, k_max,
    ):
        ctx = build_context_inputs(
            kit_s, snap, num_drums=cfg.num_drums, num_future=cfg.num_future,
            frame="env", device="cpu", dtype=torch.float64,
            idle_library=idle_library, idle_cylinder=idle_cylinder, fk=fk64,
            sticks=sticks,
            yaw_now=q0[names.index("waist_yaw_joint")], waist_cfg=cfg.waist_cfg,
        )
        rec = out[Hb]
        rec["q0"].append(q0.astype(np.float32))
        rec["qd0"].append(qd0.astype(np.float32))
        rec["init"].append(np.asarray(init, float).astype(np.float32))
        rec["hit"].append(np.array([hit["left"], hit["right"]], dtype=np.int64))
        rec["active"].append(np.array(active, dtype=bool))
        rec["drum"].append(
            np.array([snap.goal[h].drum if hit[h] >= 0 else -1 for h in HANDS], np.int64)
        )
        rec["start_drum"].append(np.asarray(src, np.int64))
        rec["kind"].append(np.int64(kind))
        rows = []
        for h in HANDS:
            if hit[h] < 0:
                rows.append(_empty_strike())
            else:
                g = snap.goal[h]
                rows.append(_strike_row(kit_s, g.drum, g.position, g.velocity))
        for hi_, s in enumerate(np.asarray(src, np.int64)):
            rows[hi_]["start_shaft"] = (
                float(shaft_hit_allowed(kit_s.names[int(s)])) if int(s) >= 0 else 0.0
            )
        for f in STRIKE_FIELDS:
            rec[f].append(
                np.stack([np.asarray(r[f], float) for r in rows]).astype(np.float32)
            )
        for f in CONTEXT_FIELDS:
            v = getattr(ctx, f)[0].numpy()
            rec[f].append(
                v.astype(np.int64) if f in _LONG_FIELDS
                else (v.astype(bool) if f in _BOOL_FIELDS else v.astype(np.float32))
            )
        stats["accepted"] += 1
        stats["eval_like"] += int(eval_like)
        stats["cold"] += int(cold)
        if all(active):
            stats["dual_active"] += 1
            if hit["left"] != hit["right"]:
                stats["asymmetric"] += 1
        if k_max <= cfg.short_horizon_wp:
            stats["short_horizon"] += 1

    def _flush_pending():
        if not pending:
            return
        if cfg.correct_terminal_steps > 0:
            terms = corrected_arm_terminals(
                fk64, pending,
                steps=int(cfg.correct_terminal_steps),
                damping=float(cfg.correct_terminal_damping),
                q_bounds=q_bounds, joint_names=names, library=library,
            )
            for p, term in zip(pending, terms):
                if term:
                    p["ws"] = warmstart_trajectory(
                        p["q0"], p["qd0"], terminal_arm=term, **p["ws_kw"]
                    )
                    stats["terminal_corrected"] += len(term)
        for p in pending:
            ws = p["ws"]
            init = np.asarray(ws.init_traj, float)
            ratio = float((np.asarray(ws.peak_rate_raw, float) / v_limit).max())
            stats["peak_rate_ratio"].append(ratio)
            if ratio > cfg.vel_reject_scale:
                stats["reject_velocity"] += 1
                continue
            _emit_row(
                q0=p["q0"], qd0=p["qd0"], init=init, hit=p["hit"],
                active=p["active"], kit_s=p["kit_s"], snap=p["snap"],
                sticks=p["sticks"], src=p["src"], kind=p["kind"],
                eval_like=p["eval_like"], cold=p["cold"], Hb=p["Hb"],
                k_max=p["k_max"],
            )
        pending.clear()

    for _ in range(int(n)):
        kit_s = kit.randomized(rng, cfg.kit_randomization) if cfg.randomize_kit else kit
        # ONE uniform decides the family, so the three are exclusive and each
        # fraction reads as a share of the whole pool: adding cold starts takes
        # from the plan-native family and leaves ``eval_like_frac`` exactly
        # where it was set.  With ``cold_start_frac = 0`` this is the old draw.
        u = float(rng.uniform())
        eval_like = u < cfg.eval_like_frac
        cold = (not eval_like) and u < cfg.eval_like_frac + cfg.cold_start_frac
        pair = (not eval_like) and (not cold) and u < (
            cfg.eval_like_frac + cfg.cold_start_frac + cfg.pair_frac
        )
        if cold:
            stats["cold_requested"] += 1

        # ---------------------------------------------------- the scenario --
        if eval_like:
            name = eval_names[int(rng.integers(len(eval_names)))]
            hand = EVAL_DRUM_HAND[name]
            di = kit_s.index(name)
            q0 = box.draw(hand, rng, kit_s)
            if q0 is None:
                stats["reject_box_start"] += 1
                continue
            sel = np.array(
                [1.0 if (nm.startswith("waist_") or nm.startswith(f"{hand}_")) else 0.0
                 for nm in names]
            )
            # The grid jitters its three repeats around one base pose; here every
            # sample is its own draw, so the jitter is just extra spread.
            q0 = np.clip(
                q0 + sel * rng.uniform(
                    -cfg.eval_start_jitter, cfg.eval_start_jitter, size=len(names)
                ),
                *q_bounds,
            )
            qd0 = np.zeros(len(names))
            if rng.uniform() < cfg.eval_moving_frac:
                qd0 = cfg.eval_v_scale * v_limit * rng.uniform(-1, 1, size=len(names)) * sel
            t_h = float(rng.uniform(*cfg.eval_horizon_s))
            k_hit = int(np.clip(round(t_h / cfg.dt), 4, H))
            speed = float(cfg.eval_speeds[int(rng.integers(len(cfg.eval_speeds)))])
            pos = sample_hit_point(kit_s, di, rng, hit_radius_frac=cfg.hit_radius_frac)
            vel = -speed * np.asarray(kit_s.axes[di], float)
            goal = {h: HandGoal.inactive() for h in HANDS}
            goal[hand] = HandGoal(
                active=True, drum=di, position=pos, velocity=vel,
                time_to_hit=k_hit * cfg.dt, drum_name=name, loudness=1.0,
            )
            snap = GoalSnapshot(t_now=0.0, goal=goal, future={h: [] for h in HANDS})
            hit = {hand: k_hit, ("right" if hand == "left" else "left"): -1}
            src = np.full(2, -1, dtype=np.int64)
            kind = KIND_EVAL_LIKE
        elif pair:
            # BOTH hands, drum combination uniform over the 42 legal ones.
            n_d, n_s = len(pair_distinct), len(pair_same)
            same = n_s > 0 and (n_d == 0 or rng.uniform() < cfg.pair_same_frac)
            if same:
                di_l = di_r = int(pair_same[int(rng.integers(n_s))])
            else:
                di_l, di_r = pair_distinct[int(rng.integers(n_d))]
            drum_of = {"left": int(di_l), "right": int(di_r)}

            # Contact points.  On the SAME drum the two beads must not be asked
            # to occupy one spot, so redraw until they are far enough apart and
            # take the best attempt if the disk is too small to separate them.
            pos = {h: sample_hit_point(kit_s, drum_of[h], rng,
                                       hit_radius_frac=cfg.hit_radius_frac)
                   for h in HANDS}
            if same:
                best = pos, -1.0
                for _ in range(12):
                    d = float(np.linalg.norm(pos["left"] - pos["right"]))
                    if d >= cfg.pair_same_min_sep:
                        break
                    if d > best[1]:
                        best = pos, d
                    pos = {h: sample_hit_point(kit_s, drum_of[h], rng,
                                               hit_radius_frac=cfg.hit_radius_frac)
                           for h in HANDS}
                else:
                    pos = best[0]
                    stats["pair_same_tight"] += 1

            # Timing: an explicit simultaneous share, else a small waypoint gap.
            t_h = float(rng.uniform(*cfg.eval_horizon_s))
            k_base = int(np.clip(round(t_h / cfg.dt), 4, H))
            if rng.uniform() < cfg.pair_simultaneous_frac:
                k = {"left": k_base, "right": k_base}
            else:
                gap = int(rng.integers(cfg.pair_gap_wp[0], cfg.pair_gap_wp[1] + 1))
                late = HANDS[int(rng.integers(2))]
                k = {h: k_base for h in HANDS}
                k[late] = int(np.clip(k_base + gap, 4, H))

            # Same start-pose machinery as the plan-native family: a sampled,
            # SDF-screened full-body pose, not a per-hand box draw -- both arms
            # are active here, so both need a sensible place to start from.
            if starts is None or si >= len(starts):
                starts = sampler.sample(int(start_chunk), rng=rng, kit=kit_s)
                si = 0
            pool_i, si = si, si + 1
            q0 = starts.q[pool_i].detach().cpu().numpy().astype(float)
            qd0 = starts.qd[pool_i].detach().cpu().numpy().astype(float)
            src = starts.drum[pool_i].detach().cpu().numpy().astype(np.int64)
            speed = float(cfg.eval_speeds[int(rng.integers(len(cfg.eval_speeds)))])
            goal = {}
            for h in HANDS:
                d_i = drum_of[h]
                goal[h] = HandGoal(
                    active=True, drum=d_i, position=pos[h],
                    velocity=-speed * np.asarray(kit_s.axes[d_i], float),
                    time_to_hit=k[h] * cfg.dt,
                    drum_name=kit_s.names[d_i], loudness=1.0,
                )
            snap = GoalSnapshot(t_now=0.0, goal=goal,
                                future={h: [] for h in HANDS})
            hit = {h: int(k[h]) for h in HANDS}
            kind = KIND_PAIR
            stats["pair"] += 1
            stats["pair_same"] += int(same)
        else:
            # Cold rows keep their own chunk rather than filtering a mixed one:
            # the family is decided per scenario here, and the goal draw below
            # depends on it, so the sampler is asked for all-cold or no-cold.
            if cold:
                if cold_starts is None or ci >= len(cold_starts):
                    cold_starts = sampler.sample(
                        int(start_chunk), rng=rng, kit=kit_s, cold_fraction=1.0
                    )
                    ci = 0
                pool_i, ci = ci, ci + 1
                if int(cold_starts.kind[pool_i]) != _START_KIND_COLD:
                    # The SDF screen fell back to the ready pose, which sits at
                    # yaw 0 -- the correlated state, not a cold one.
                    stats["reject_cold_start"] += 1
                    continue
                src_batch = cold_starts
            else:
                if starts is None or si >= len(starts):
                    starts = sampler.sample(int(start_chunk), rng=rng, kit=kit_s)
                    si = 0
                pool_i, si = si, si + 1
                src_batch = starts
            q0 = src_batch.q[pool_i].detach().cpu().numpy().astype(float)
            qd0 = src_batch.qd[pool_i].detach().cpu().numpy().astype(float)
            src = src_batch.drum[pool_i].detach().cpu().numpy().astype(np.int64)
            # Targeted rare-drum draw (``rare_drum_boost``): pick the drum
            # inverse-frequency-weighted, one of its hits uniformly, and put
            # "now" 2..H waypoints before it.  Cold rows keep their own draw --
            # their lead floor is what makes them buildable.
            if (
                rare_index is not None and not cold
                and rng.uniform() < cfg.rare_drum_boost
            ):
                r_drums, r_hits, r_p = rare_index
                di_r = int(rng.choice(len(r_drums), p=r_p))
                ci_r, t_hit = r_hits[di_r][int(rng.integers(len(r_hits[di_r])))]
                clip = clips[ci_r]
                t0 = max(t_hit - int(rng.integers(2, H + 1)) * cfg.dt, 0.0)
                stats["rare_boosted"] += 1
            else:
                clip = clips[int(rng.integers(len(clips)))]
                t0 = _draw_cold_time(clip, rng, cfg) if cold else _draw_time(clip, rng, cfg)
            snap = goal_snapshot(
                clip, kit_s, t0,
                num_future=cfg.num_future, rng=rng,
                hit_radius_frac=cfg.hit_radius_frac,
                speed_min=cfg.speed_min, speed_max=cfg.speed_max,
                horizon_cap=H * cfg.dt,
                time_jitter_s=cfg.hit_time_jitter_s,
                loudness_jitter=cfg.hit_loudness_jitter,
            )
            hit = {}
            for h in HANDS:
                g = snap.goal[h]
                hit[h] = (
                    int(np.clip(round(g.time_to_hit / cfg.dt), 1, H)) if g.active else -1
                )
            kind = KIND_COLD_START if cold else KIND_PLAN

        active = [hit[h] >= 0 for h in HANDS]
        if not any(active):
            # A both-hands-idle draw is a REST.  Keep it at ``rest_frac`` when a
            # home table is available to park the arms at (without one the row is
            # a pose held in place with nothing pricing it, which is the state
            # the idle terms exist to replace).
            if cfg.require_active_hand and (
                idle_library is None or rng.uniform() >= cfg.rest_frac
            ):
                stats["reject_idle"] += 1
                continue
            stats["rest"] += 1
        goals: dict[str, tuple[str, float] | None] = {}
        aim: dict[str, np.ndarray] = {}
        # The seed's yaw ramp and the context's ``aim_yaw`` token must land on
        # the SAME target, so both read the snapshot's futures through the one
        # helper.  An eval-like row has no future hits at all (its snapshot
        # carries an empty list per hand), so it keeps the single-goal aim.
        _t, aim_tth, aim_futures = aim_inputs_from_snapshot(snap, frame="env")
        for h in HANDS:
            g = snap.goal[h]
            if hit[h] < 0:
                goals[h] = None
                aim_tth.pop(h, None)
                continue
            goals[h] = (g.drum_name, hit[h] * cfg.dt)
            aim[h] = np.asarray(g.position, float)
        if not all(library.has(h, goals[h][0]) for h in HANDS if goals[h] is not None):
            stats["reject_library"] += 1
            continue

        k_max = max([hit[h] for h in HANDS if hit[h] >= 0], default=1)
        if any(active):
            Hb = bucket_for(k_max, cfg.buckets)
        else:
            # A rest has no hit to size the horizon from, and ``bucket_for(1)``
            # would file every one of them in the shortest bucket.  Both arms are
            # priced over the WHOLE grid when neither is active (see
            # ``effective_hits``), so spread rests across the buckets instead.
            Hb = int(cfg.buckets[int(rng.integers(len(cfg.buckets)))])
        if any(active) and not all(active):
            stats["single_idle"] += 1

        # THE FITTED STICK is drawn BEFORE the seed is built: the terminal
        # correction below must aim the stick this scenario actually carries --
        # the same stick ``build_context_inputs`` writes into the context and
        # the cost prices.  Sampling it after the correction re-aimed the seed
        # for the nominal stick while everything else saw the fitted one.
        sticks = None
        if cfg.stick_rand is not None:
            sticks = (cfg.stick_rand.sample(rng, "left"),
                      cfg.stick_rand.sample(rng, "right"))

        ws_kw = dict(
            horizon=Hb, goals=goals, library=library,
            aim_targets=aim or None, aim_tth=aim_tth, aim_futures=aim_futures,
            dt=cfg.dt, joint_names=names,
            waist_cfg=cfg.waist_cfg, q_bounds=q_bounds,
            waist_upright_ease_wp=cfg.waist_upright_ease_wp,
            slew_limit_eta=cfg.seed_slew_eta,
            idle_library=idle_library, idle_cylinder=idle_cylinder, fk=fk64,
            idle_settle_wp=cfg.idle_settle_wp,
        )
        # Straight-line quintic only.  Apex is a phase-1 imitation target (and
        # an occasional phase-1 init); cost descent always starts from the
        # same seed deployment will hand the decoder.  RE-AIM is batched in
        # ``_flush_pending``: stack every pending row of one hand, one
        # ``adjust_arm_terminal`` call.  The first build supplies the WAIST
        # at the strike index -- correcting against ``yaw_target`` instead of
        # the yaw the ramp actually reaches leaves a residual that scales
        # with whatever the ramp still had to travel (measured: 3.7 cm
        # median extra).
        pending.append({
            "ws": warmstart_trajectory(q0, qd0, **ws_kw),
            "ws_kw": ws_kw, "q0": q0, "qd0": qd0, "goals": goals,
            "aim": aim, "hit": hit, "active": active, "sticks": sticks,
            "kit_s": kit_s, "snap": snap, "src": src, "kind": kind,
            "eval_like": eval_like, "cold": cold, "Hb": Hb, "k_max": k_max,
        })
        if len(pending) >= _CHUNK:
            _flush_pending()

    _flush_pending()

    packed = {
        b: {k: np.stack(v) for k, v in rec.items()}
        for b, rec in out.items() if rec["q0"]
    }
    stats["peak_rate_ratio"] = np.asarray(stats["peak_rate_ratio"], dtype=np.float32)
    return {"buckets": packed, "stats": stats}


def _worker(args) -> dict:
    n, seed, cfg, clip_dir = args
    torch.set_num_threads(1)
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(clip_dir, kit)
    return generate_cost_scenarios(n, clips, seed=seed, cfg=cfg, kit=kit)


def _merge(parts: list[dict]) -> dict:
    buckets: dict[int, dict[str, list[np.ndarray]]] = {}
    stats: dict = {}
    for p in parts:
        for b, arrays in p["buckets"].items():
            tgt = buckets.setdefault(b, {k: [] for k in arrays})
            for k, v in arrays.items():
                tgt[k].append(v)
        for k, v in p["stats"].items():
            if isinstance(v, np.ndarray):
                stats.setdefault(k, []).append(v)
            elif isinstance(v, dict):
                target = stats.setdefault(k, {})
                for label, count in v.items():
                    target[label] = target.get(label, 0) + count
            else:
                stats[k] = stats.get(k, 0) + v
    out_b = {b: {k: np.concatenate(v) for k, v in a.items()} for b, a in buckets.items()}
    if "peak_rate_ratio" in stats:
        stats["peak_rate_ratio"] = np.concatenate(stats["peak_rate_ratio"])
    return {"buckets": out_b, "stats": stats}


# ================================================================= pool ======
class Phase2Pool:
    """Bucketed phase-2 scenario pool, resident on one device."""

    def __init__(
        self,
        buckets: dict[int, dict[str, np.ndarray]],
        *,
        cfg: Phase2ScenarioConfig,
        noise: Phase2Noise | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        stats: dict | None = None,
    ):
        self.cfg = cfg
        self.noise = noise or Phase2Noise()
        self.device = torch.device(device)
        self.dtype = dtype
        self.stats = dict(stats or {})
        #: Shaft samples in ``FusedCost.c_points``; only read when the pool
        #: carries a randomized stick.  Overridden by the trainer from the cost.
        self.n_shaft = SHAFT_HIT_SAMPLES
        names = tuple(cfg.joint_names)
        lo, hi = urdf_position_limits(names)
        self.q_lo = torch.as_tensor(lo, device=self.device, dtype=dtype)
        self.q_hi = torch.as_tensor(hi, device=self.device, dtype=dtype)
        self._pelvis = torch.as_tensor(PELVIS_IN_ENV, device=self.device, dtype=dtype)

        self.data: dict[int, dict[str, torch.Tensor]] = {}
        self._ou: dict[int, torch.Tensor] = {}
        self._ou_overrides: dict[tuple[int, float], torch.Tensor] = {}
        for H, arrays in sorted(buckets.items()):
            if not len(arrays["q0"]):
                continue
            n = len(arrays["q0"])
            wrong = {key: len(value) for key, value in arrays.items()
                     if len(value) != n}
            if wrong:
                raise ValueError(f"bucket {H} has {n} q0 rows but mismatched "
                                 f"columns: {wrong}")
            d: dict[str, torch.Tensor] = {}
            for k, v in arrays.items():
                if v.dtype == np.bool_:
                    d[k] = torch.as_tensor(v, device=self.device)
                elif np.issubdtype(v.dtype, np.integer):
                    d[k] = torch.as_tensor(v, device=self.device, dtype=torch.long)
                else:
                    d[k] = torch.as_tensor(v, device=self.device, dtype=dtype)
            self.data[H] = d
            self._ou[H] = ou_kernel(H, self.noise.tau_steps, self.device, dtype)
        # Per-bucket kind → row indices (built lazily for kind-aware sampling).
        self._kind_idx: dict[int, dict[int, torch.Tensor]] = {}

    # ---------------------------------------------------------- accessors --
    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(sorted(self.data))

    def __len__(self) -> int:
        return sum(int(d["q0"].shape[0]) for d in self.data.values())

    def size(self, bucket: int) -> int:
        return int(self.data[int(bucket)]["q0"].shape[0])

    def distribution(self) -> dict[int, int]:
        return {b: self.size(b) for b in self.buckets}

    def kind_counts(self, bucket: int | None = None) -> dict[int, int]:
        """``{kind: n}`` over one bucket, or summed across the pool."""
        if bucket is not None:
            d = self.data[int(bucket)]
            k = d["kind"]
            uniq, cnt = torch.unique(k, return_counts=True)
            return {int(u): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
        out: dict[int, int] = {}
        for b in self.buckets:
            for k, c in self.kind_counts(b).items():
                out[k] = out.get(k, 0) + c
        return out

    def indices_of_kind(self, bucket: int, kind: int) -> torch.Tensor:
        """``(M,)`` row indices of ``kind`` in ``bucket`` (cached)."""
        b, k = int(bucket), int(kind)
        cache = self._kind_idx.setdefault(b, {})
        if k not in cache:
            d = self.data[b]
            cache[k] = (d["kind"] == k).nonzero(as_tuple=True)[0]
        return cache[k]

    # ------------------------------------------------- horizon bookkeeping --
    def hit_min(self, bucket: int) -> torch.Tensor:
        """``(N,)`` nearest ACTIVE hit index per stored scenario.

        A scenario with no active hand reads ``bucket + 1``, i.e. "longer than
        anything in this bucket", so it never counts as short.  This is the
        statistic the MPC actually lives in: the closed loop spends every strike
        passing through ``k = 5, 4, 3, 2, 1`` and those are the plans whose
        terminal is executed.
        """
        d = self.data[int(bucket)]
        hit, act = d["hit"], d["active"]
        big = torch.full_like(hit, int(bucket) + 1)
        return torch.where(act, hit.clamp_min(0), big).min(dim=1).values

    def horizon_histogram(self, bucket: int, edges=(1, 2, 3, 5, 8, 12, 16, 25, 50)) -> dict:
        """``{edge: count}`` of :meth:`hit_min` at or below each edge, cumulative."""
        k = self.hit_min(bucket).float()
        n = int(k.numel())
        out = {"n": n}
        prev = 0
        for e in edges:
            c = int((k <= e).sum())
            out[f"<= {e}"] = c - prev
            prev = c
        out["> %d" % edges[-1]] = n - prev
        return out

    def short_weight(self, bucket: int, *, short_wp: int, boost: float) -> torch.Tensor:
        """Per-row sampling weight that oversamples the final-approach regime.

        The roller harvests every slot on every tick, so its horizon
        distribution is roughly uniform over each goal's initial horizon --
        which puts only a few percent of the mix inside the ``k <= 5`` window
        the MPC's terminal waypoint is drawn from.  Measured on the closed loop,
        the plan's own terminal aim is inside the hit disk down to ``k ~ 5`` and
        then drifts ~60 mm OUT of it by ``k = 1``, which is a competence gap in
        exactly the band that is under-represented.  ``boost`` multiplies those
        rows' sampling weight; 1.0 restores uniform.
        """
        k = self.hit_min(bucket)
        w = torch.ones(k.shape, device=k.device, dtype=self.dtype)
        if short_wp > 0 and boost != 1.0:
            w = torch.where(k <= int(short_wp), w * float(boost), w)
        return w

    def sample_idx(
        self,
        bucket: int,
        n: int,
        generator: torch.Generator | None = None,
        *,
        short_wp: int = 0,
        short_boost: float = 1.0,
        kind_fracs: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """Draw ``n`` row indices from ``bucket``.

        With ``kind_fracs`` (``{kind: weight}`` already renormalized over
        nonempty families), sampling is categorical by kind then uniform /
        short-boosted within the kind.  Kinds with no rows in this bucket are
        dropped and the remaining weights re-normalized.  ``kind_fracs=None``
        restores the historical kind-blind draw.
        """
        b = int(bucket)
        m = self.size(b)
        if m == 0:
            raise RuntimeError(f"bucket {b} is empty")
        if kind_fracs:
            local = {
                int(k): float(f)
                for k, f in kind_fracs.items()
                if float(f) > 0.0 and self.indices_of_kind(b, int(k)).numel() > 0
            }
            if local:
                s = sum(local.values())
                kinds = sorted(local)
                probs = torch.tensor(
                    [local[k] / s for k in kinds],
                    device=self.device, dtype=self.dtype,
                )
                cat = torch.multinomial(
                    probs, int(n), replacement=True, generator=generator
                )
                out = torch.empty(int(n), device=self.device, dtype=torch.long)
                boost = short_wp > 0 and short_boost != 1.0
                w_all = (
                    self.short_weight(b, short_wp=short_wp, boost=short_boost)
                    if boost else None
                )
                for i, k in enumerate(kinds):
                    mask = cat == i
                    ni = int(mask.sum())
                    if ni == 0:
                        continue
                    idx_k = self.indices_of_kind(b, k)
                    if boost and w_all is not None:
                        w = w_all.index_select(0, idx_k)
                        pick = torch.multinomial(
                            w, ni, replacement=True, generator=generator
                        )
                    else:
                        pick = torch.randint(
                            0, idx_k.numel(), (ni,),
                            device=self.device, generator=generator,
                        )
                    out[mask] = idx_k.index_select(0, pick)
                return out
        if short_wp <= 0 or short_boost == 1.0:
            return torch.randint(0, m, (n,), device=self.device, generator=generator)
        w = self.short_weight(bucket, short_wp=short_wp, boost=short_boost)
        return torch.multinomial(w, n, replacement=True, generator=generator)

    def masks_for(self, hit: torch.Tensor, active: torch.Tensor, H: int):
        t = hit.clamp_min(0).to(self.dtype) * float(self.cfg.dt)
        return build_waypoint_masks(
            t[:, 0], t[:, 1], max_horizon=int(H), dt=self.cfg.dt,
            active_left=active[:, 0], active_right=active[:, 1],
            joint_names=self.cfg.joint_names,
        )

    def context(self, bucket: int, idx: torch.Tensor) -> ContextInputs:
        d = self.data[int(bucket)]
        # Fields absent from the stored arrays are left to ``ContextInputs``'s
        # own defaults, which materialize zeros + invalid.  That is what lets a
        # pool cached BEFORE the idle-home block existed still load: those rows
        # simply carry no idle token, which is the truth about them.
        return ContextInputs(
            **{f: d[f].index_select(0, idx) for f in CONTEXT_FIELDS if f in d}
        )

    #: Where :meth:`waist_ref` gets the waist-yaw reference when there is no
    #: explicit ``waist_ref`` column.  True on the FRESH pool, whose ``init`` is
    #: the warm-start seed and whose waist-yaw column is therefore
    #: ``yaw_reference(yaw_start, aim_yaw(targets))`` evaluated by
    #: :func:`~amortized_opt.warmstart.warmstart_trajectory` with this pool's own
    #: ``waist_cfg`` -- i.e. exactly the reference, already computed, per sample.
    #:
    #: :class:`~amortized_opt.train_cost.ReplayPool` sets it FALSE, because its
    #: ``init`` is the roller's shifted PREVIOUS PLAN -- the model's own output.
    #: Pricing the model against that would be self-referential: any yaw
    #: trajectory it produced would be its own reference and the term would read
    #: zero no matter what the torso did.  The replay pool must carry a real
    #: harvested column or contribute no waist-aim gradient at all.
    waist_ref_from_seed: bool = True

    def waist_ref(self, bucket: int) -> torch.Tensor | None:
        """``(N, H)`` waist-yaw reference for this bucket, or ``None``.

        Rows follow the knots-1..H convention documented on
        :attr:`~amortized_opt.cost.StrikeContext.waist_yaw_ref` (the fresh
        pool's ``init`` IS ``WarmStart.init_traj``, which drops knot 0).

        A pool with ``waist_ref_from_seed=False`` (the replay pool) NEVER
        falls back to its seed: it returns a genuinely harvested ``waist_ref``
        column or ``None``.  ``ReplayPool`` only allocates that column when the
        roller harvests it, so an unharvested run cannot present a column of
        zeros here and silently anchor the replay half at yaw = 0.
        """
        d = self.data[int(bucket)]
        if "waist_ref" in d:
            return d["waist_ref"]
        if not self.waist_ref_from_seed:
            return None
        yaw_i = tuple(self.cfg.joint_names).index("waist_yaw_joint")
        return d["init"][:, :, yaw_i]

    def noised_init(
        self, init: torch.Tensor, bucket: int, *, generator: torch.Generator | None = None,
        noise: Phase2Noise | None = None,
        elem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nc = self.noise if noise is None else noise
        B, H, D = init.shape
        if nc.process == "gaussian_bridge":
            if elem_mask is None:
                raise ValueError("Gaussian bridge requires per-joint strike masks")
            out = init + sample_bridge(elem_mask, float(self.cfg.dt), nc.bridge, generator=generator)
            return out.clamp(self.q_lo,self.q_hi) if nc.clip_to_box else out
        if nc.tau_steps == self.noise.tau_steps:
            L = self._ou[int(bucket)]
        else:
            key = (int(bucket), float(nc.tau_steps))
            if key not in self._ou_overrides:
                self._ou_overrides[key] = ou_kernel(H, nc.tau_steps, self.device, self.dtype)
            L = self._ou_overrides[key]
        eps = torch.randn(B, H, D, device=init.device, dtype=init.dtype, generator=generator)
        if nc.magnitude_distribution == "half_normal":
            sigma = nc.sigma_scale * torch.randn(
                B, 1, 1, device=init.device, dtype=init.dtype, generator=generator,
            ).abs()
        else:
            u = torch.rand(B, 1, 1, device=init.device, dtype=init.dtype, generator=generator)
            lo, hi = math.log(nc.sigma_corr_min), math.log(nc.sigma_corr_max)
            sigma = torch.exp(lo + (hi - lo) * u)
        out = init + sigma * torch.einsum("ij,bjd->bid", L, eps)
        if nc.sigma_white > 0.0:
            out = out + nc.sigma_white * torch.randn(
                B, H, D, device=init.device, dtype=init.dtype, generator=generator
            )
        if nc.clip_to_box:
            out = torch.clamp(out, self.q_lo, self.q_hi)
        return out

    # -------------------------------------------------------------- batch --
    def batch(
        self,
        bucket: int,
        idx: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: bool = True,
    ) -> dict:
        H = int(bucket)
        d = self.data[H]
        pick = lambda k: d[k].index_select(0, idx)  # noqa: E731
        q0, v0, seed = pick("q0"), pick("qd0"), pick("init")
        masks = self.masks_for(pick("hit"), pick("active"), H)
        init = self.noised_init(seed, H, generator=generator) if noise else seed.clone()
        ctx = self.context(H, idx)
        heads = ctx.drum_pos - self._pelvis
        wref = self.waist_ref(H)
        pref = d["waist_pr_ref"].index_select(0, idx) if "waist_pr_ref" in d else None
        sctx = strike_context_from_arrays(
            d, idx, drum_head_centers=heads, drum_radii=ctx.drum_radius,
            waist_ref=None if wref is None else wref.index_select(0, idx),
            n_shaft=self.n_shaft,
            waist_pr_ref=pref,
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
            "hit": pick("hit"),
            "active": pick("active"),
            "kind": pick("kind"),
        }


def generate_cost_pool(
    n: int,
    *,
    seed: int,
    cfg: Phase2ScenarioConfig | None = None,
    noise: Phase2Noise | None = None,
    split: str = "train",
    clip_dir: str | Path | None = None,
    workers: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Phase2Pool:
    """Fan :func:`generate_cost_scenarios` out over processes and wrap it."""
    import multiprocessing as mp

    cfg = cfg or Phase2ScenarioConfig()
    clip_dir = str(clip_dir if clip_dir is not None else default_rcc_dir(split))
    workers = int(workers or min(os.cpu_count() or 8, 64))
    workers = max(1, min(workers, int(n)))
    per = [n // workers] * workers
    for i in range(n - sum(per)):
        per[i] += 1
    args = [(per[i], seed * 100003 + i, cfg, clip_dir) for i in range(workers)]
    if workers == 1:
        raw = _merge([_worker(args[0])])
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            raw = _merge(pool.map(_worker, args))
    return Phase2Pool(
        raw["buckets"], cfg=cfg, noise=noise, device=device, dtype=dtype,
        stats=raw["stats"],
    )
