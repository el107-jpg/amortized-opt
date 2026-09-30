"""CPU-only, actor-free trajectory optimisation for strike teaching.

The editor deliberately optimises the waypoint tensor itself.  The production
phase-2 scenario generator supplies the start state, randomized kit, analytic
quintic warm start, masks and strike context; :class:`FusedCost` supplies the
same differentiable geometry and rigid-body objective used for actor training.
No actor checkpoint (or actor module) is loaded on this path.

Waypoint numbers in the public API are one based.  This matches
``StrikeContext.hit_index``: waypoint 1 is the first sample after ``q0`` and
waypoint 50 is the last row of the ``(50, 17)`` trajectory.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from amortized_opt.constants import CONTROL_DT, PELVIS_IN_ENV, STICK_LENGTH
from amortized_opt.cost import CostTerms, FusedCost, StrikeContext
from amortized_opt.fk import (
    ChainFK,
    stick_axis_local,
    stick_contact_offset,
    stick_tip_offset,
)
from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.limits import urdf_position_limits
from amortized_opt.phase2_data import (
    Phase2Pool,
    Phase2ScenarioConfig,
    generate_cost_scenarios,
)
from amortized_opt.rhythm import DrumKit, default_rcc_dir, load_rcc_pool
from amortized_opt.sdf import segment_hit_region
from amortized_opt.strike import (
    CYMBAL_CONTACT_SIGMA_FALLBACK,
    STICK_TIP_SIGMA,
)
from amortized_opt.tokenizer import ContextInputs, WaypointMasks
from amortized_opt.train_cost import phase2_al_config, phase2_cost_config

__all__ = [
    "DirectTrajectoryOptimizer",
    "ApproachCorridorConfig",
    "FeasibilityTolerances",
    "TrajectoryScenario",
    "TaskSpaceConstraint",
    "WaypointKeyframe",
    "quintic_terminal_initialization",
    "sample_trajectory_scenario",
    "warp_terminal_approach",
]


HORIZON = 50
MIN_HIT_WAYPOINT = 20
HANDS = ("left", "right")
TASK_POSITION_WEIGHT = 2.0e6
TASK_DIRECTION_WEIGHT = 2.0e5
TASK_WAIST_WEIGHT = 1.0e5
TASK_CONTACT_TOLERANCE_M = 0.020
TASK_DIRECTION_TOLERANCE_DEG = 2.0
TASK_WAIST_TOLERANCE_RAD = 0.015
TASK_APPROACH_WINDOW_WAYPOINTS = 16
TASK_APPROACH_POSITION_WEIGHT = 2.0e5
TASK_APPROACH_DIRECTION_WEIGHT = 2.0e4
TASK_APPROACH_WAIST_WEIGHT = 1.0e4


def quintic_terminal_initialization(
    scenario: "TrajectoryScenario",
    terminal_q: torch.Tensor | np.ndarray | Iterable[float],
    *,
    hand: str,
) -> torch.Tensor:
    """Build a fresh velocity-matched quintic to an edited strike pose.

    The selected arm and waist start at the scenario's measured ``q0/qd0``,
    reach ``terminal_q`` at that hand's strike waypoint with zero terminal
    velocity, and hold the pose afterward. The opposite arm retains the
    scenario's analytic quintic warm start so its independently timed strike
    still has a useful seed. This never copies the current actor or teacher
    trajectory into the edited side.
    """
    if hand not in HANDS:
        raise ValueError("hand must be left or right")
    target = torch.as_tensor(
        terminal_q, device=scenario.q0.device, dtype=scenario.q0.dtype
    )
    if tuple(target.shape) != (17,) or not bool(torch.isfinite(target).all().item()):
        raise ValueError("terminal_q must be a finite 17-vector")
    hit = int(scenario.hit_index[0, HANDS.index(hand)].item())
    if not 1 <= hit <= scenario.horizon:
        raise ValueError(f"selected-hand hit waypoint must be in [1, {scenario.horizon}]")

    out = scenario.init.detach().clone()
    cols = torch.as_tensor(
        [
            index
            for index, name in enumerate(scenario.joint_names)
            if name.startswith("waist_") or name.startswith(f"{hand}_")
        ],
        device=out.device,
        dtype=torch.long,
    )
    q0 = scenario.q0[0].index_select(0, cols)
    qd0 = scenario.qd0[0].index_select(0, cols)
    goal = target.index_select(0, cols)
    steps = torch.arange(
        1, scenario.horizon + 1, device=out.device, dtype=out.dtype
    ).unsqueeze(1)
    duration_steps = torch.as_tensor(float(hit), device=out.device, dtype=out.dtype)
    u = (steps / duration_steps).clamp(0.0, 1.0)
    displacement = (goal - q0).unsqueeze(0)
    opening_tangent = (qd0 * duration_steps * float(scenario.dt)).unsqueeze(0)
    u3 = u.pow(3)
    curve = (
        q0.unsqueeze(0)
        + opening_tangent * u
        + (10.0 * displacement - 6.0 * opening_tangent) * u3
        + (8.0 * opening_tangent - 15.0 * displacement) * u3 * u
        + (6.0 * displacement - 3.0 * opening_tangent) * u3 * u.pow(2)
    )
    out[0, :, cols] = curve
    return out


def warp_terminal_approach(
    trajectory: torch.Tensor,
    terminal_q: torch.Tensor | np.ndarray | Iterable[float],
    *,
    hit_waypoint: int,
    joint_indices: Iterable[int],
    window_waypoints: int = TASK_APPROACH_WINDOW_WAYPOINTS,
) -> torch.Tensor:
    """Minimum-jerk warp of selected joints into a new terminal pose.

    The public waypoint is one based.  The endpoint displacement is spread
    backward over the approach window with a quintic blend whose velocity and
    acceleration vanish at both ends.  This is only an initialization: the
    differentiable physics solve remains free to refine every non-keyframed
    value afterward.
    """
    source = torch.as_tensor(trajectory)
    squeeze = tuple(source.shape) == (HORIZON, 17)
    if squeeze:
        source = source.unsqueeze(0)
    if tuple(source.shape) != (1, HORIZON, 17):
        raise ValueError("trajectory must have shape (50, 17) or (1, 50, 17)")
    hit = int(hit_waypoint)
    if not 1 <= hit <= HORIZON:
        raise ValueError(f"hit_waypoint must be in [1, {HORIZON}]")
    window = int(window_waypoints)
    if window < 1:
        raise ValueError("window_waypoints must be positive")
    cols = tuple(int(value) for value in joint_indices)
    if not cols or len(set(cols)) != len(cols) or min(cols) < 0 or max(cols) >= 17:
        raise ValueError("joint_indices must be unique indices in [0, 16]")
    target = torch.as_tensor(terminal_q, device=source.device, dtype=source.dtype)
    if tuple(target.shape) != (17,) or not bool(torch.isfinite(target).all().item()):
        raise ValueError("terminal_q must be a finite 17-vector")

    out = source.detach().clone()
    hit_row = hit - 1
    first_row = max(0, hit_row - window + 1)
    rows = torch.arange(first_row, hit_row + 1, device=out.device)
    # q0 lives immediately before row zero and has blend alpha=0.  Giving the
    # first optimized row s=1/N keeps the q0 -> waypoint-1 transition smooth
    # even when a hit occurs before a full approach window fits.
    s = torch.arange(
        1, rows.numel() + 1, device=out.device, dtype=out.dtype
    ) / float(rows.numel())
    alpha = 10.0 * s.pow(3) - 15.0 * s.pow(4) + 6.0 * s.pow(5)
    col_tensor = torch.as_tensor(cols, device=out.device, dtype=torch.long)
    endpoint = out[0, hit_row].index_select(0, col_tensor)
    delta = target.index_select(0, col_tensor) - endpoint
    warped = out[0].index_select(0, rows).index_select(1, col_tensor)
    warped = warped + alpha.unsqueeze(1) * delta.unsqueeze(0)
    out[0, rows.unsqueeze(1), col_tensor.unsqueeze(0)] = warped
    return out[0] if squeeze else out


@dataclass(frozen=True)
class ApproachCorridorConfig:
    """Signed above-head corridor over the final pre-contact waypoints."""

    window_waypoints: int = 10
    far_clearance_m: float = 0.10
    near_clearance_m: float = 0.01
    weight: float = 2.0e4

    def __post_init__(self) -> None:
        if self.window_waypoints < 1:
            raise ValueError("window_waypoints must be positive")
        if not 0.0 <= self.near_clearance_m <= self.far_clearance_m:
            raise ValueError("corridor clearances must satisfy 0 <= near <= far")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("corridor weight must be finite and positive")


@dataclass(frozen=True)
class FeasibilityTolerances:
    """Acceptance-style hard gates for declaring a teacher usable."""

    terminal_region_m: float = 0.02
    impact_speed_error_mps: float = 0.25
    impact_speed_error_fraction: float = 0.25
    collision_penetration_m: float = 1.0e-4
    torque_excess_nm: float = 1.0e-5
    velocity_excess_rad_s: float = 1.0e-5
    joint_box_excess_rad: float = 1.0e-6
    approach_corridor_m: float = 0.005


def _json_value(value: Any, *, squeeze_batch: bool = False) -> Any:
    """Return strict-JSON-compatible values (including finite checks)."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if squeeze_batch and value.ndim and value.shape[0] == 1:
            value = value[0]
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return str(value)


def _tensor_record(obj: Any) -> dict[str, Any]:
    return {
        name: _json_value(getattr(obj, name), squeeze_batch=True)
        for name in obj.__dataclass_fields__
    }


def _cost_record(out: CostTerms) -> dict[str, Any]:
    record: dict[str, Any] = {"total": _json_value(out.total, squeeze_batch=True)}
    for group in (
        "terms",
        "raw",
        "violations",
        "diagnostics",
        "counts",
        "balance_terms",
    ):
        record[group] = {
            key: _json_value(value, squeeze_batch=True)
            for key, value in getattr(out, group).items()
        }
    return record


@dataclass(frozen=True)
class WaypointKeyframe:
    """A partial joint pose at a waypoint or at one hand's own strike.

    Specify exactly one of ``waypoint`` or ``hand``.  ``hand='left'`` resolves
    to the left hand's hit index and never to the other hand's (possibly later)
    horizon.  Values may cover any subset of the 17 controlled joints.
    """

    values: Mapping[str, float]
    mode: str = "soft"
    weight: float = 1.0
    waypoint: int | None = None
    hand: str | None = None

    @classmethod
    def at_hit(
        cls,
        hand: str,
        values: Mapping[str, float],
        *,
        mode: str = "soft",
        weight: float = 1.0,
    ) -> "WaypointKeyframe":
        return cls(values=values, mode=mode, weight=weight, hand=hand)


@dataclass(frozen=True)
class TaskSpaceConstraint:
    """Persistent selected-hand terminal intent for direct optimization."""

    hand: str
    contact_position: tuple[float, float, float] | list[float]
    stick_direction: tuple[float, float, float] | list[float]
    waist: tuple[float, float, float] | list[float]
    # Joint-space IK solution for the task pose.  The direct Adam backend does
    # not need it after the endpoint warp, but the SQP backend uses the selected
    # arm columns as a redundancy/posture reference at contact.
    terminal_q: tuple[float, ...] | list[float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "hand": self.hand,
            "contact_position": [float(value) for value in self.contact_position],
            "stick_direction": [float(value) for value in self.stick_direction],
            "waist": [float(value) for value in self.waist],
            "terminal_q": (
                None
                if self.terminal_q is None
                else [float(value) for value in self.terminal_q]
            ),
            "weights": {
                "contact_position_m2": TASK_POSITION_WEIGHT,
                "stick_direction_1_minus_cos": TASK_DIRECTION_WEIGHT,
                "waist_rad2": TASK_WAIST_WEIGHT,
            },
        }


@dataclass
class TrajectoryScenario:
    """One deterministic, replayable production-format optimisation row."""

    scenario_id: str
    seed: int
    horizon: int
    dt: float
    joint_names: tuple[str, ...]
    drum_names: tuple[str, ...]
    q0: torch.Tensor
    qd0: torch.Tensor
    init: torch.Tensor
    hit_index: torch.Tensor
    active: torch.Tensor
    target_drum: torch.Tensor
    start_drum: torch.Tensor
    masks: WaypointMasks
    context: ContextInputs
    strike: StrikeContext
    sampling_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon != HORIZON or tuple(self.init.shape) != (1, HORIZON, 17):
            raise ValueError("trajectory scenarios must have init shape (1, 50, 17)")
        if tuple(self.q0.shape) != (1, 17) or tuple(self.qd0.shape) != (1, 17):
            raise ValueError("q0 and qd0 must have shape (1, 17)")
        if tuple(self.hit_index.shape) != (1, 2):
            raise ValueError("hit_index must have shape (1, 2)")

    @property
    def teacher_mask(self) -> torch.Tensor:
        return self.masks.elem_mask

    def goals_json(self) -> list[dict[str, Any]]:
        goals = []
        pelvis = torch.as_tensor(
            PELVIS_IN_ENV,
            dtype=self.context.goal_pos.dtype,
            device=self.context.goal_pos.device,
        )
        for hi, hand in enumerate(HANDS):
            active = bool(self.active[0, hi].item())
            drum_i = int(self.target_drum[0, hi].item())
            hit = int(self.hit_index[0, hi].item()) if active else -1
            goals.append(
                {
                    "hand": hand,
                    "active": active,
                    "drum": self.drum_names[drum_i] if active and drum_i >= 0 else None,
                    "drum_index": drum_i if active else -1,
                    "tth_s": float(hit * self.dt) if active else None,
                    "hit_index": hit,
                    "contact_position": (
                        _json_value(self.context.goal_pos[0, hi] - pelvis)
                        if active
                        else None
                    ),
                }
            )
        return goals

    def kit_json(self) -> dict[str, Any]:
        valid = self.context.drum_valid[0].bool()
        pos_env = self.context.drum_pos[0, valid]
        pelvis = torch.as_tensor(PELVIS_IN_ENV, dtype=pos_env.dtype)
        axes = self.context.drum_axis[0, valid]
        radii = self.context.drum_radius[0, valid]
        heights = self.context.drum_height[0, valid]
        nominal = DrumKit.from_layout()
        drums = []
        for i in range(int(valid.sum().item())):
            drums.append(
                {
                    "id": i,
                    "name": self.drum_names[i],
                    "position_env": _json_value(pos_env[i]),
                    "position_pelvis": _json_value(pos_env[i] - pelvis),
                    "axis": _json_value(axes[i]),
                    "radius": float(radii[i].item()),
                    "height": float(heights[i].item()),
                    "shaft_hit": bool(nominal.shaft_hit[i]),
                }
            )
        return {"frame": "env", "drums": drums}

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.scenario_id,
            "seed": self.seed,
            "horizon": self.horizon,
            "dt": self.dt,
            "joint_names": list(self.joint_names),
            "q0": _json_value(self.q0, squeeze_batch=True),
            "qd0": _json_value(self.qd0, squeeze_batch=True),
            "hit_index": _json_value(self.hit_index, squeeze_batch=True),
            "active": _json_value(self.active, squeeze_batch=True),
            "target_drum": _json_value(self.target_drum, squeeze_batch=True),
            "start_drum": _json_value(self.start_drum, squeeze_batch=True),
            "goals": self.goals_json(),
            "kit": self.kit_json(),
            "context": _tensor_record(self.context),
            "strike_context": _tensor_record(self.strike),
            "init": _json_value(self.init, squeeze_batch=True),
            "masks": _tensor_record(self.masks),
            "sampling_stats": _json_value(self.sampling_stats),
        }

    def training_bundle(
        self,
        teacher: torch.Tensor,
        *,
        teaching_hand: str | None = None,
    ) -> dict[str, Any]:
        """Tensor contract consumed by the IL/LoRA persistence layer.

        When ``teaching_hand`` is supplied, imitation supervision covers only
        the waist and that arm, and only through that hand's own hit.  The
        other arm remains an actor/context input and is protected by the
        trainer's base-policy distillation term, rather than being silently
        treated as a second human correction.
        """
        if tuple(teacher.shape) == (HORIZON, 17):
            teacher = teacher.unsqueeze(0)
        if tuple(teacher.shape) != (1, HORIZON, 17):
            raise ValueError("teacher must have shape (50, 17) or (1, 50, 17)")
        teacher_mask = self.teacher_mask.detach().clone()
        if teaching_hand is not None:
            if teaching_hand not in HANDS:
                raise ValueError("teaching_hand must be 'left' or 'right'")
            hi = HANDS.index(teaching_hand)
            hit = int(self.hit_index[0, hi].item())
            if not bool(self.active[0, hi].item()) or hit < 1:
                raise ValueError(f"{teaching_hand} hand has no active hit")
            selected_joints = torch.tensor(
                [
                    name.startswith("waist_") or name.startswith(f"{teaching_hand}_")
                    for name in self.joint_names
                ],
                dtype=teacher_mask.dtype,
                device=teacher_mask.device,
            ).view(1, 1, -1)
            selected_time = (
                torch.arange(HORIZON, device=teacher_mask.device) < hit
            ).to(teacher_mask.dtype).view(1, HORIZON, 1)
            teacher_mask = teacher_mask * selected_joints * selected_time
        return {
            "scenario_id": self.scenario_id,
            "init": self.init.detach().clone(),
            "state": torch.cat([self.q0, self.qd0], dim=-1).detach().clone(),
            "context": self.context,
            "strike": self.strike,
            "hit_index": self.hit_index.detach().clone(),
            "active": self.active.detach().clone(),
            "teacher_q": teacher.detach().clone(),
            "teacher_mask": teacher_mask,
            "teaching_hand": teaching_hand,
        }


def sample_trajectory_scenario(
    seed: int,
    *,
    clip_dir: str | Path | None = None,
    start_chunk: int = 16,
    max_attempts: int = 16,
    num_future: int = 1,
    timing_selector: Callable[[dict, Phase2Pool], Mapping[str, Any]] | None = None,
) -> TrajectoryScenario:
    """Sample a deterministic, dual-hand, production-format 50-step row.

    The PAIR family is used intentionally: both goal drums and both per-hand
    TTHs are always explicit, while starts, kit randomization, hit-region
    sampling, analytic warmstart and terminal correction remain the production
    implementations.  Rejections retry on a deterministic seed sequence.
    """
    seed = int(seed)
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(clip_dir or default_rcc_dir("train"), kit)
    if not clips:
        raise RuntimeError("no RCC clips are available for scenario sampling")
    cfg = Phase2ScenarioConfig(
        horizon=HORIZON,
        buckets=(HORIZON,),
        num_future=int(num_future),
        eval_like_frac=0.0,
        cold_start_frac=0.0,
        pair_frac=1.0,
        randomize_kit=True,
        # These are deliberately ordinary teaching tasks, not critic-selected
        # frontier rows.  Keep enough approach time for useful manual edits,
        # while retaining a broad range of independent two-hand timings.
        eval_horizon_s=(MIN_HIT_WAYPOINT * CONTROL_DT, 0.80),
    )
    sampled = None
    used_seed = seed
    for attempt in range(max(int(max_attempts), 1)):
        # A simple arithmetic stream is stable across Python processes and
        # avoids relying on salted hashes.
        used_seed = seed + attempt * 1_000_003
        candidate = generate_cost_scenarios(
            1,
            clips,
            seed=used_seed,
            cfg=cfg,
            kit=kit,
            start_chunk=max(int(start_chunk), 1),
        )
        if HORIZON in candidate["buckets"]:
            sampled = candidate
            break
    if sampled is None:
        raise RuntimeError(f"could not sample an accepted scenario in {max_attempts} attempts")

    pool = Phase2Pool(sampled["buckets"], cfg=cfg, device="cpu", dtype=torch.float32)
    batch = pool.batch(HORIZON, torch.tensor([0]), noise=False)
    if not bool(batch["active"].all().item()):
        raise RuntimeError("PAIR scenario unexpectedly contains an inactive hand")
    timing_pair = [int(value) for value in batch["hit"][0].tolist()]
    if min(timing_pair) < MIN_HIT_WAYPOINT:
        raise RuntimeError("PAIR scenario violates the editor minimum hit waypoint")
    timing_stats: Mapping[str, Any] = {
        "task_purpose": "il_trajectory_teaching",
        "timing_source": "random_il_pair",
        "timing_pair": timing_pair,
        "timing_tth_s": [float(value * cfg.dt) for value in timing_pair],
        "timing_min_hit_waypoint": MIN_HIT_WAYPOINT,
        "timing_actor_validated": False,
    }
    if timing_selector is not None:
        timing_stats = timing_selector(batch, pool)

    digest_data = np.concatenate(
        [
            batch["q0"].numpy().reshape(-1),
            batch["v0"].numpy().reshape(-1),
            batch["hit"].numpy().astype(np.float32).reshape(-1),
            batch["context"].goal_pos.numpy().reshape(-1),
        ]
    ).tobytes()
    scenario_id = "traj-" + hashlib.sha256(digest_data).hexdigest()[:16]
    stats = dict(sampled["stats"])
    stats["generator_seed"] = used_seed
    stats.update(dict(timing_stats))
    return TrajectoryScenario(
        scenario_id=scenario_id,
        seed=seed,
        horizon=HORIZON,
        dt=float(cfg.dt),
        joint_names=tuple(cfg.joint_names),
        drum_names=tuple(kit.names),
        q0=batch["q0"],
        qd0=batch["v0"],
        init=batch["seed"].clone(),
        hit_index=batch["hit"],
        active=batch["active"],
        target_drum=batch["strike"].target_drum,
        start_drum=batch["strike"].start_drum,
        masks=batch["masks"],
        context=batch["context"],
        strike=batch["strike"],
        sampling_stats=stats,
    )


class DirectTrajectoryOptimizer:
    """Incremental Adam optimisation over one ``(50, 17)`` waypoint tensor."""

    def __init__(
        self,
        scenario: TrajectoryScenario,
        *,
        initial_trajectory: (
            torch.Tensor | np.ndarray | Iterable[Iterable[float]] | None
        ) = None,
        keyframes: Iterable[WaypointKeyframe | Mapping[str, Any]] = (),
        task_constraints: Iterable[TaskSpaceConstraint | Mapping[str, Any]] = (),
        learning_rate: float = 1.0e-4,
        max_grad_norm: float | None = 1.0e4,
        approach: ApproachCorridorConfig | None = None,
        tolerances: FeasibilityTolerances | None = None,
        dual_update_every: int = 5,
    ) -> None:
        if scenario.horizon != HORIZON:
            raise ValueError(f"direct editor requires horizon {HORIZON}")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.scenario = scenario
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.approach = approach or ApproachCorridorConfig()
        self.tolerances = tolerances or FeasibilityTolerances()
        self.dual_update_every = max(int(dual_update_every), 1)
        self.iteration = 0
        self.progress: list[dict[str, Any]] = []
        self._q_lo, self._q_hi = (
            torch.as_tensor(x, dtype=torch.float32)
            for x in urdf_position_limits(scenario.joint_names, posture_limits=False)
        )
        self._initialization_source = (
            "analytic_warmstart_fallback"
            if initial_trajectory is None
            else "actor_proposal"
        )
        if initial_trajectory is None:
            initial_trajectory = scenario.init
        self._initial = torch.as_tensor(initial_trajectory, dtype=torch.float32)
        if tuple(self._initial.shape) == (HORIZON, 17):
            self._initial = self._initial.unsqueeze(0)
        if tuple(self._initial.shape) != (1, HORIZON, 17):
            raise ValueError(
                "initial_trajectory must be the actor proposal with shape "
                "(50, 17) or (1, 50, 17)"
            )
        if not bool(torch.isfinite(self._initial).all().item()):
            raise ValueError("initial_trajectory must contain only finite values")
        self._initial = self._initial.detach().clone()
        self.trajectory = torch.nn.Parameter(self._initial.clone())

        self.keyframes = [self._coerce_keyframe(k) for k in keyframes]
        self._resolved_keyframes = [self._resolve_keyframe(k) for k in self.keyframes]
        self.task_constraints = [
            self._coerce_task_constraint(value) for value in task_constraints
        ]
        self._resolved_task_constraints = [
            self._resolve_task_constraint(value) for value in self.task_constraints
        ]
        self._hard_mask = torch.zeros_like(self.trajectory, dtype=torch.bool)
        self._hard_values = torch.zeros_like(self.trajectory)
        self._install_hard_keyframes()

        # The actor-training default leaves cross-stick and collision-peak
        # prices at zero for checkpoint compatibility.  An offline teacher has
        # no such compatibility constraint: both are hard feasibility concerns.
        self.cost = FusedCost(
            phase2_cost_config(w_stick_stick=1.0e5, w_collision_peak=5.0e3),
            max_horizon=HORIZON,
            dt=scenario.dt,
            device="cpu",
            dtype=torch.float32,
            joint_names=scenario.joint_names,
        )
        self.al = AugmentedLagrangian(
            phase2_al_config(ema=0.0, decay=1.0, feasible_decay=0.9)
        )
        self.optimizer = torch.optim.Adam([self.trajectory], lr=self.learning_rate)
        self._geometry_fk = ChainFK(
            joint_names=scenario.joint_names,
            frames={name: name for name in scenario.joint_names},
            device="cpu",
            dtype=torch.float32,
        )
        self._install_task_approach_references()
        self._project()
        self._converged = False
        self._stop_reason = "not_started"
        self._last_best_iteration = 0
        with torch.no_grad():
            (
                initial_out, initial_soft, initial_corridor,
                initial_corridor_detail, initial_task, initial_task_detail,
            ) = self._evaluate()
        self._initial_cost = float(initial_out.total.mean().item())
        initial_feasibility = self._feasibility(
            initial_out, initial_corridor_detail, initial_task_detail
        )
        self._best_rank = self._candidate_rank(
            initial_feasibility,
            self._initial_cost + float(initial_soft.item())
            + self.approach.weight * float(initial_corridor.item())
            + float(initial_task.item()),
        )
        self._best_trajectory = self.trajectory.detach().clone()
        self._best_cost = self._initial_cost
        self._best_feasibility = initial_feasibility

    @staticmethod
    def _coerce_keyframe(value: WaypointKeyframe | Mapping[str, Any]) -> WaypointKeyframe:
        if isinstance(value, WaypointKeyframe):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("keyframes must be WaypointKeyframe objects or mappings")
        return WaypointKeyframe(**dict(value))

    @staticmethod
    def _coerce_task_constraint(
        value: TaskSpaceConstraint | Mapping[str, Any],
    ) -> TaskSpaceConstraint:
        if isinstance(value, TaskSpaceConstraint):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("task constraints must be TaskSpaceConstraint objects or mappings")
        return TaskSpaceConstraint(**dict(value))

    def _resolve_task_constraint(
        self, constraint: TaskSpaceConstraint,
    ) -> dict[str, Any]:
        if constraint.hand not in HANDS:
            raise ValueError("task constraint hand must be 'left' or 'right'")
        hi = HANDS.index(constraint.hand)
        if not bool(self.scenario.active[0, hi].item()):
            raise ValueError(f"{constraint.hand} has no active strike")

        def vec3(value: Iterable[float], name: str) -> torch.Tensor:
            tensor = self.trajectory.new_tensor(list(value))
            if tuple(tensor.shape) != (3,) or not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"task constraint {name} must be a finite 3-vector")
            return tensor

        position = vec3(constraint.contact_position, "contact_position")
        direction = vec3(constraint.stick_direction, "stick_direction")
        norm = torch.linalg.vector_norm(direction)
        if float(norm.item()) < 1.0e-8:
            raise ValueError("task constraint stick_direction must be nonzero")
        waist = vec3(constraint.waist, "waist")
        waist_cols = [
            self.scenario.joint_names.index(f"waist_{axis}_joint")
            for axis in ("yaw", "pitch", "roll")
        ]
        lo = self._q_lo[waist_cols]
        hi_limit = self._q_hi[waist_cols]
        if bool(((waist < lo) | (waist > hi_limit)).any().item()):
            raise ValueError("task constraint waist is outside the joint box")
        return {
            "hand": constraint.hand,
            "hand_index": hi,
            "waypoint": int(self.scenario.hit_index[0, hi].item()),
            "contact_position": position,
            "stick_direction": direction / norm,
            "waist": waist,
            "waist_cols": waist_cols,
        }

    def _resolve_keyframe(self, keyframe: WaypointKeyframe) -> dict[str, Any]:
        if (keyframe.waypoint is None) == (keyframe.hand is None):
            raise ValueError("a keyframe must specify exactly one of waypoint or hand")
        if keyframe.mode not in ("hard", "soft"):
            raise ValueError("keyframe mode must be 'hard' or 'soft'")
        if not math.isfinite(float(keyframe.weight)) or float(keyframe.weight) < 0.0:
            raise ValueError("keyframe weight must be finite and non-negative")
        if keyframe.hand is not None:
            if keyframe.hand not in HANDS:
                raise ValueError("keyframe hand must be 'left' or 'right'")
            hi = HANDS.index(keyframe.hand)
            if not bool(self.scenario.active[0, hi].item()):
                raise ValueError(f"{keyframe.hand} has no active strike")
            waypoint = int(self.scenario.hit_index[0, hi].item())
        else:
            waypoint = int(keyframe.waypoint)  # type: ignore[arg-type]
        if not 1 <= waypoint <= HORIZON:
            raise ValueError(f"waypoint must be in [1, {HORIZON}]")
        if not keyframe.values:
            raise ValueError("keyframe values cannot be empty")
        values: dict[int, float] = {}
        for name, raw in keyframe.values.items():
            if name not in self.scenario.joint_names:
                raise ValueError(f"unknown joint {name!r}")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"keyframe value for {name!r} must be finite")
            ji = self.scenario.joint_names.index(name)
            if not bool(self.scenario.masks.elem_mask[0, waypoint - 1, ji].item()):
                raise ValueError(
                    f"{name!r} is outside its hand horizon at waypoint {waypoint}"
                )
            if value < float(self._q_lo[ji]) or value > float(self._q_hi[ji]):
                raise ValueError(f"keyframe value for {name!r} is outside the joint box")
            values[ji] = value
        return {
            "waypoint": waypoint,
            "hand": keyframe.hand,
            "mode": keyframe.mode,
            "weight": float(keyframe.weight),
            "values": values,
        }

    def _install_hard_keyframes(self) -> None:
        for keyframe in self._resolved_keyframes:
            if keyframe["mode"] != "hard":
                continue
            row = keyframe["waypoint"] - 1
            for col, value in keyframe["values"].items():
                if self._hard_mask[0, row, col] and not torch.isclose(
                    self._hard_values[0, row, col], torch.tensor(value), atol=1e-7
                ):
                    raise ValueError("conflicting hard keyframes address the same joint")
                self._hard_mask[0, row, col] = True
                self._hard_values[0, row, col] = value

    def _install_task_approach_references(self) -> None:
        """Freeze approach anchors from the endpoint-warped initialization."""
        with torch.no_grad():
            for constraint in self._resolved_task_constraints:
                hit_row = constraint["waypoint"] - 1
                first_row = max(
                    0, hit_row - int(TASK_APPROACH_WINDOW_WAYPOINTS) + 1
                )
                q = self._initial[0, first_row].unsqueeze(0)
                hi = constraint["hand_index"]
                sigma = self.scenario.strike.contact_sigma[0, hi].to(q).view(1)
                contact = self._contact_positions(q, constraint["hand"], hi, sigma)[0]
                wrist_r, _wrist_p = self.cost.fk.frame(
                    q, f"{constraint['hand']}_wrist"
                )
                _mount, axis_local, _tip_sigma = self._stick_local_geometry(
                    constraint["hand"], hi
                )
                axis = torch.einsum("bij,j->bi", wrist_r, axis_local)[0]
                axis = axis / axis.norm().clamp_min(1.0e-12)
                waist_cols = torch.as_tensor(
                    constraint["waist_cols"], device=q.device, dtype=torch.long
                )
                constraint["approach_first_row"] = first_row
                constraint["approach_anchor_contact"] = contact.detach().clone()
                constraint["approach_anchor_direction"] = axis.detach().clone()
                constraint["approach_anchor_waist"] = (
                    q[0].index_select(0, waist_cols).detach().clone()
                )

    def _project(self) -> None:
        with torch.no_grad():
            self.trajectory.clamp_(self._q_lo, self._q_hi)
            invalid = self.scenario.masks.elem_mask <= 0.0
            self.trajectory[invalid] = self._initial[invalid]
            self.trajectory[self._hard_mask] = self._hard_values[self._hard_mask]

    def _soft_keyframe_loss(self) -> torch.Tensor:
        loss = self.trajectory.new_zeros(())
        for keyframe in self._resolved_keyframes:
            if keyframe["mode"] != "soft" or keyframe["weight"] == 0.0:
                continue
            row = keyframe["waypoint"] - 1
            cols = torch.tensor(list(keyframe["values"]), dtype=torch.long)
            target = self.trajectory.new_tensor(list(keyframe["values"].values()))
            err = self.trajectory[0, row].index_select(0, cols) - target
            loss = loss + keyframe["weight"] * err.square().mean()
        return loss

    def _task_space_constraint_loss(
        self,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Preserve terminal intent and coherently deform its approach window."""
        loss = self.trajectory.new_zeros(())
        detail: dict[str, dict[str, torch.Tensor]] = {}
        for constraint in self._resolved_task_constraints:
            hand = constraint["hand"]
            hi = constraint["hand_index"]
            row = constraint["waypoint"] - 1
            q = self.trajectory[0, row].unsqueeze(0)
            sigma = self.scenario.strike.contact_sigma[0, hi].to(q).view(1)
            contact = self._contact_positions(q, hand, hi, sigma)[0]
            wrist_r, _wrist_p = self.cost.fk.frame(q, f"{hand}_wrist")
            _mount, axis_local, _tip_sigma = self._stick_local_geometry(hand, hi)
            axis = torch.einsum("bij,j->bi", wrist_r, axis_local)[0]
            axis = axis / axis.norm().clamp_min(1.0e-12)
            position_sq = (contact - constraint["contact_position"]).square().sum()
            direction = 1.0 - (axis * constraint["stick_direction"]).sum().clamp(-1.0, 1.0)
            waist_cols = torch.as_tensor(
                constraint["waist_cols"], device=q.device, dtype=torch.long,
            )
            waist_sq = (
                q[0].index_select(0, waist_cols) - constraint["waist"]
            ).square().mean()
            loss = loss + (
                TASK_POSITION_WEIGHT * position_sq
                + TASK_DIRECTION_WEIGHT * direction
                + TASK_WAIST_WEIGHT * waist_sq
            )

            # A terminal edit must move the approach rather than creating a
            # one-knot discontinuity.  Track a minimum-jerk task-space path
            # from the endpoint-warped anchor to the requested contact pose.
            first_row = int(constraint["approach_first_row"])
            approach_rows = torch.arange(
                first_row, row, device=self.trajectory.device, dtype=torch.long
            )
            approach_position_rmse = self.trajectory.new_zeros(())
            approach_direction_rmse = self.trajectory.new_zeros(())
            approach_waist_rmse = self.trajectory.new_zeros(())
            if approach_rows.numel() > 0:
                qa = self.trajectory[0].index_select(0, approach_rows)
                n = int(approach_rows.numel()) + 1
                s = torch.arange(
                    1, n, device=qa.device, dtype=qa.dtype
                ) / float(n)
                alpha = 10.0 * s.pow(3) - 15.0 * s.pow(4) + 6.0 * s.pow(5)
                ramp_weight = alpha.square().clamp_min(0.05)

                sigma_a = self.scenario.strike.contact_sigma[0, hi].to(qa).expand(
                    approach_rows.numel()
                )
                contact_a = self._contact_positions(qa, hand, hi, sigma_a)
                start_contact = constraint["approach_anchor_contact"]
                desired_contact = (
                    (1.0 - alpha).unsqueeze(1) * start_contact.unsqueeze(0)
                    + alpha.unsqueeze(1) * constraint["contact_position"].unsqueeze(0)
                )
                # Retain the signed above-head corridor in the reference path,
                # not merely as a one-sided collision-like hinge.
                n_out = self.scenario.strike.n_out[0, hi].to(qa)
                p_star = constraint["contact_position"].to(qa)
                steps = (row - approach_rows).to(qa.dtype)
                clearance = self.approach.near_clearance_m + (
                    steps / float(max(self.approach.window_waypoints, 1))
                ).clamp(max=1.0) * (
                    self.approach.far_clearance_m
                    - self.approach.near_clearance_m
                )
                axial = ((desired_contact - p_star) * n_out).sum(dim=-1)
                desired_contact = desired_contact + (
                    clearance - axial
                ).clamp_min(0.0).unsqueeze(1) * n_out.unsqueeze(0)
                position_each = (contact_a - desired_contact).square().sum(dim=-1)
                approach_position_sq = (
                    ramp_weight * position_each
                ).sum() / ramp_weight.sum()

                wrist_ra, _wrist_pa = self.cost.fk.frame(qa, f"{hand}_wrist")
                axis_a = torch.einsum("bij,j->bi", wrist_ra, axis_local)
                axis_a = axis_a / axis_a.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
                desired_axis = (
                    (1.0 - alpha).unsqueeze(1)
                    * constraint["approach_anchor_direction"].unsqueeze(0)
                    + alpha.unsqueeze(1) * constraint["stick_direction"].unsqueeze(0)
                )
                desired_axis = desired_axis / desired_axis.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-12)
                direction_each = 1.0 - (axis_a * desired_axis).sum(dim=-1).clamp(
                    -1.0, 1.0
                )
                approach_direction = (
                    ramp_weight * direction_each
                ).sum() / ramp_weight.sum()

                waist_a = qa.index_select(1, waist_cols)
                desired_waist = (
                    (1.0 - alpha).unsqueeze(1)
                    * constraint["approach_anchor_waist"].unsqueeze(0)
                    + alpha.unsqueeze(1) * constraint["waist"].unsqueeze(0)
                )
                waist_each = (waist_a - desired_waist).square().mean(dim=-1)
                approach_waist_sq = (
                    ramp_weight * waist_each
                ).sum() / ramp_weight.sum()
                loss = loss + (
                    TASK_APPROACH_POSITION_WEIGHT * approach_position_sq
                    + TASK_APPROACH_DIRECTION_WEIGHT * approach_direction
                    + TASK_APPROACH_WAIST_WEIGHT * approach_waist_sq
                )
                approach_position_rmse = approach_position_sq.sqrt()
                approach_direction_rmse = torch.rad2deg(
                    torch.sqrt((2.0 * approach_direction).clamp_min(0.0))
                )
                approach_waist_rmse = approach_waist_sq.sqrt()
            detail[hand] = {
                "waypoint": self.trajectory.new_tensor(float(row + 1)),
                "contact_error_m": position_sq.sqrt(),
                "direction_error_deg": torch.rad2deg(
                    torch.acos((axis * constraint["stick_direction"]).sum().clamp(-1.0, 1.0))
                ),
                "waist_error_rad": waist_sq.sqrt(),
                "contact_position": contact,
                "stick_direction": axis,
                "approach_position_rmse_m": approach_position_rmse,
                "approach_direction_rmse_deg": approach_direction_rmse,
                "approach_waist_rmse_rad": approach_waist_rmse,
            }
        return loss, detail

    def _stick_local_geometry(
        self, hand: str, hi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return wrist-local ``(mount, axis, tip_sigma)`` for this sample."""
        ref = self.trajectory
        if bool(self.scenario.context.stick_valid[0, hi].item()):
            axis = self.scenario.context.stick_axis[0, hi].to(ref)
            tip = self.scenario.context.stick_tip[0, hi].to(ref)
            length = self.scenario.context.stick_length[0, hi].to(ref)
            tip_sigma = length - STICK_LENGTH / 2.0
            mount = tip - tip_sigma * axis
            return mount, axis, tip_sigma
        axis = ref.new_tensor(stick_axis_local(hand))
        tip_sigma = ref.new_tensor(float(STICK_TIP_SIGMA))
        tip = ref.new_tensor(stick_tip_offset(hand))
        return tip - tip_sigma * axis, axis, tip_sigma

    def _contact_positions(
        self, q: torch.Tensor, hand: str, hi: int, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Specified contact point for ``q`` rows, differentiable in ``q``."""
        R, p = self.cost.fk.frame(q, f"{hand}_wrist")
        mount, axis, _tip_sigma = self._stick_local_geometry(hand, hi)
        local = mount.unsqueeze(0) + sigma.reshape(-1, 1) * axis.unsqueeze(0)
        return p + torch.einsum("bij,bj->bi", R, local)

    def _approach_corridor_loss(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Signed outward-normal clearance ramp before each hand's strike.

        Positive signed distance is above/outside the drum head.  For the last
        ``window_waypoints`` before contact, required clearance ramps from
        ``far_clearance_m`` to ``near_clearance_m``.  The hit itself is left to
        the exact legal-region terminal constraint.
        """
        losses: list[torch.Tensor] = []
        detail: dict[str, Any] = {}
        W = int(self.approach.window_waypoints)
        for hi, hand in enumerate(HANDS):
            if not bool(self.scenario.active[0, hi].item()):
                continue
            hit = int(self.scenario.hit_index[0, hi].item())
            first = max(1, hit - W)
            waypoint = torch.arange(
                first, hit, device=self.trajectory.device, dtype=torch.long
            )
            if waypoint.numel() == 0:
                continue
            q = self.trajectory[0].index_select(0, waypoint - 1)
            sigma = self.scenario.strike.contact_sigma[0, hi].to(q).expand(
                waypoint.numel()
            )
            contact = self._contact_positions(q, hand, hi, sigma)
            n_out = self.scenario.strike.n_out[0, hi].to(q)
            p_star = self.scenario.strike.p_star[0, hi].to(q)
            signed = ((contact - p_star) * n_out).sum(dim=-1)
            steps = (hit - waypoint).to(q.dtype)
            alpha = steps / float(max(W, 1))
            desired = self.approach.near_clearance_m + alpha * (
                self.approach.far_clearance_m - self.approach.near_clearance_m
            )
            violation = torch.relu(desired - signed)
            losses.append(violation.square().mean() + violation.amax().square())
            detail[hand] = {
                "waypoints": waypoint,
                "signed_clearance_m": signed,
                "required_clearance_m": desired,
                "violation_m": violation,
                "violation_max_m": violation.amax(),
            }
        if not losses:
            return self.trajectory.new_zeros(()), detail
        return torch.stack(losses).mean(), detail

    def _strike_segment(
        self, q: torch.Tensor, hand: str, hi: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-space playable bead/shaft segment for one or more poses."""
        R, p = self.cost.fk.frame(q, f"{hand}_wrist")
        mount, axis, tip_sigma = self._stick_local_geometry(hand, hi)
        axis_w = torch.einsum("bij,j->bi", R, axis)
        base = p + torch.einsum("bij,j->bi", R, mount)
        spec = self.scenario.strike.contact_sigma[0, hi].to(q)
        if bool(self.scenario.strike.shaft_allowed[0, hi].item()):
            lo = spec.new_tensor(float(CYMBAL_CONTACT_SIGMA_FALLBACK))
            hi_sigma = tip_sigma.to(spec)
            lo = torch.minimum(lo, hi_sigma)
        else:
            lo = hi_sigma = spec
        return base + lo * axis_w, base + hi_sigma * axis_w

    def _terminal_metrics_per_hand(self) -> dict[str, dict[str, float]]:
        """Exact legal-region and impact-speed diagnostics for each hand."""
        result: dict[str, dict[str, float]] = {}
        cfg = self.cost.cfg
        for hi, hand in enumerate(HANDS):
            if not bool(self.scenario.active[0, hi].item()):
                continue
            hit = int(self.scenario.hit_index[0, hi].item())
            q = self.trajectory[0, hit - 1].unsqueeze(0)
            q_prev = (
                self.scenario.q0
                if hit == 1
                else self.trajectory[0, hit - 2].unsqueeze(0)
            )
            a, b = self._strike_segment(q, hand, hi)
            a_prev, b_prev = self._strike_segment(q_prev, hand, hi)
            dist, s, contact = segment_hit_region(
                a,
                b,
                self.scenario.strike.p_star[:, hi],
                self.scenario.strike.n_out[:, hi],
                self.scenario.strike.hit_radius[:, hi],
                self.scenario.strike.disk_offset[:, hi],
                self.scenario.strike.half_plane_dir[:, hi],
                self.scenario.strike.has_half_plane[:, hi],
                cfg.terminal_axial_lo,
                cfg.terminal_axial_hi,
            )
            contact_prev = a_prev + s.unsqueeze(-1) * (b_prev - a_prev)
            speed = torch.linalg.vector_norm(
                (contact - contact_prev) / self.scenario.dt, dim=-1
            )
            target_speed = torch.linalg.vector_norm(
                self.scenario.strike.v_star[:, hi], dim=-1
            )
            result[hand] = {
                "region_m": float(dist[0].detach().item()),
                "speed_mps": float(speed[0].detach().item()),
                "target_speed_mps": float(target_speed[0].detach().item()),
                "speed_error_mps": float(
                    (speed - target_speed).abs()[0].detach().item()
                ),
            }
        return result

    def _feasibility(
        self,
        out: CostTerms,
        corridor_detail: Mapping[str, Any],
        task_detail: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> dict[str, Any]:
        diag = out.diagnostics
        scalar = lambda key, default=0.0: float(  # noqa: E731
            diag[key].detach().mean().item() if key in diag else default
        )
        terminal_detail = self._terminal_metrics_per_hand()
        per_hand = {hand: row["region_m"] for hand, row in terminal_detail.items()}
        speed_per_hand = {
            hand: row["speed_error_mps"] for hand, row in terminal_detail.items()
        }
        terminal_max = max(per_hand.values(), default=0.0)
        speed_excess = max(
            (
                max(
                    0.0,
                    row["speed_error_mps"]
                    - max(
                        self.tolerances.impact_speed_error_mps,
                        self.tolerances.impact_speed_error_fraction
                        * row["target_speed_mps"],
                    ),
                )
                for row in terminal_detail.values()
            ),
            default=0.0,
        )
        collision = max(scalar("collision_pen_max"), 0.0)
        stick_pen = max(0.0, -scalar("collision_stick_min_sdf", 1.0))
        collision = max(collision, stick_pen)
        corridor = max(
            (
                float(row["violation_max_m"].detach().item())
                for row in corridor_detail.values()
            ),
            default=0.0,
        )
        metrics = {
            "terminal_region_max_m": terminal_max,
            "terminal_region_per_hand_m": per_hand,
            "impact_speed_error_per_hand_mps": speed_per_hand,
            "impact_speed_excess_max_mps": speed_excess,
            "collision_penetration_max_m": collision,
            "stick_stick_penetration_max_m": stick_pen,
            "torque_excess_max_nm": max(scalar("torque_excess_max"), 0.0),
            "velocity_excess_max_rad_s": max(scalar("velocity_excess_max"), 0.0),
            "joint_box_excess_max_rad": max(scalar("joint_box_excess_max"), 0.0),
            "approach_corridor_violation_max_m": corridor,
            "task_space": {
                hand: {
                    key: float(value.detach().item())
                    if value.numel() == 1 else _json_value(value)
                    for key, value in row.items()
                }
                for hand, row in task_detail.items()
            },
        }
        t = self.tolerances
        checks = {
            "terminal": terminal_max <= t.terminal_region_m,
            "impact_speed": speed_excess <= 0.0,
            "collision": collision <= t.collision_penetration_m,
            "torque": metrics["torque_excess_max_nm"] <= t.torque_excess_nm,
            "velocity": metrics["velocity_excess_max_rad_s"] <= t.velocity_excess_rad_s,
            "joint_box": metrics["joint_box_excess_max_rad"] <= t.joint_box_excess_rad,
            "approach_corridor": corridor <= t.approach_corridor_m,
        }
        for hand, row in task_detail.items():
            checks[f"{hand}_task_contact"] = (
                float(row["contact_error_m"].item()) <= TASK_CONTACT_TOLERANCE_M
            )
            checks[f"{hand}_task_direction"] = (
                float(row["direction_error_deg"].item())
                <= TASK_DIRECTION_TOLERANCE_DEG
            )
            checks[f"{hand}_task_waist"] = (
                float(row["waist_error_rad"].item()) <= TASK_WAIST_TOLERANCE_RAD
            )
        excess = [
            max(0.0, terminal_max - t.terminal_region_m) / 0.02,
            speed_excess / 0.25,
            max(0.0, collision - t.collision_penetration_m) / 0.005,
            max(0.0, metrics["torque_excess_max_nm"] - t.torque_excess_nm) / 5.0,
            max(0.0, metrics["velocity_excess_max_rad_s"] - t.velocity_excess_rad_s) / 0.5,
            max(0.0, metrics["joint_box_excess_max_rad"] - t.joint_box_excess_rad) / 0.02,
            max(0.0, corridor - t.approach_corridor_m) / 0.02,
        ]
        for row in task_detail.values():
            excess.extend(
                [
                    max(0.0, float(row["contact_error_m"].item()) - TASK_CONTACT_TOLERANCE_M) / 0.02,
                    max(0.0, float(row["direction_error_deg"].item()) - TASK_DIRECTION_TOLERANCE_DEG) / 5.0,
                    max(0.0, float(row["waist_error_rad"].item()) - TASK_WAIST_TOLERANCE_RAD) / 0.05,
                ]
            )
        return {
            "feasible": bool(all(checks.values())),
            # UI / persisted-teacher vocabulary.  Kept equal by construction;
            # an infeasible best-effort trajectory must never become accepted.
            "accepted": bool(all(checks.values())),
            **metrics,
            "checks": checks,
            "tolerances": _json_value(t.__dict__),
            "task_space_tolerances": {
                "contact_m": TASK_CONTACT_TOLERANCE_M,
                "direction_deg": TASK_DIRECTION_TOLERANCE_DEG,
                "waist_rad": TASK_WAIST_TOLERANCE_RAD,
            },
            "normalized_violation_score": float(max(excess) + sum(excess)),
        }

    @staticmethod
    def _candidate_rank(feasibility: Mapping[str, Any], objective: float) -> tuple[float, ...]:
        return (
            0.0 if bool(feasibility["feasible"]) else 1.0,
            0.0 if bool(feasibility["feasible"]) else float(
                feasibility["normalized_violation_score"]
            ),
            float(objective),
        )

    def _evaluate(
        self,
    ) -> tuple[
        CostTerms, torch.Tensor, torch.Tensor, dict[str, Any],
        torch.Tensor, dict[str, dict[str, torch.Tensor]],
    ]:
        out = self.cost(
            self.scenario.q0,
            self.trajectory,
            self.scenario.qd0,
            self.scenario.strike,
            elem_mask=self.scenario.masks.elem_mask,
            weights=self.al.cost_weights(),
        )
        corridor, detail = self._approach_corridor_loss()
        task, task_detail = self._task_space_constraint_loss()
        return out, self._soft_keyframe_loss(), corridor, detail, task, task_detail

    def step(self, iterations: int = 1) -> list[dict[str, Any]]:
        """Run ``iterations`` updates and return only this call's progress."""
        if int(iterations) < 0:
            raise ValueError("iterations must be non-negative")
        chunk = []
        for _ in range(int(iterations)):
            self.optimizer.zero_grad(set_to_none=True)
            out, soft, corridor, _corridor_detail, task, _task_detail = self._evaluate()
            objective = (
                out.total.mean() + soft + self.approach.weight * corridor + task
            )
            if not bool(torch.isfinite(objective).item()):
                raise RuntimeError("trajectory objective became non-finite")
            objective.backward()
            assert self.trajectory.grad is not None
            allowed = (self.scenario.masks.elem_mask > 0.0) & ~self._hard_mask
            self.trajectory.grad.mul_(allowed)
            grad_before = float(torch.linalg.vector_norm(self.trajectory.grad).item())
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_([self.trajectory], self.max_grad_norm)
            grad_after = float(torch.linalg.vector_norm(self.trajectory.grad).item())
            self.optimizer.step()
            self._project()
            self.iteration += 1
            with torch.no_grad():
                (
                    post_out, post_soft, post_corridor, post_corridor_detail,
                    post_task, post_task_detail,
                ) = self._evaluate()
            if self.iteration % self.dual_update_every == 0:
                self.al.update(post_out.violations, record=False)
            post_objective = (
                float(post_out.total.mean().item())
                + float(post_soft.item())
                + self.approach.weight * float(post_corridor.item())
                + float(post_task.item())
            )
            feasibility = self._feasibility(
                post_out, post_corridor_detail, post_task_detail
            )
            rank = self._candidate_rank(feasibility, post_objective)
            improved = rank < self._best_rank
            if improved:
                self._best_rank = rank
                self._best_trajectory = self.trajectory.detach().clone()
                self._best_cost = float(post_out.total.mean().item())
                self._best_feasibility = feasibility
                self._last_best_iteration = self.iteration
            row = {
                "iteration": self.iteration,
                "objective": post_objective,
                "physics_cost": float(post_out.total.mean().item()),
                "soft_keyframe_loss": float(post_soft.item()),
                "approach_corridor_loss": float(post_corridor.item()),
                "task_space_loss": float(post_task.item()),
                "feasible": bool(feasibility["feasible"]),
                "normalized_violation_score": float(
                    feasibility["normalized_violation_score"]
                ),
                "best_updated": improved,
                "objective_before": float(objective.detach().item()),
                "physics_cost_before": float(out.total.detach().mean().item()),
                "grad_norm": grad_before,
                "clipped_grad_norm": grad_after,
            }
            self.progress.append(row)
            chunk.append(row)
        return chunk

    def solve(
        self,
        *,
        max_iterations: int = 300,
        min_iterations: int = 40,
        patience: int = 30,
    ) -> dict[str, Any]:
        """Run a feasibility-aware solve and restore the best iterate found.

        Stagnation only terminates a *feasible* solve.  An infeasible trajectory
        consumes the complete iteration budget rather than being mislabeled as
        converged merely because its scalar objective stopped moving.
        """
        max_iterations = int(max_iterations)
        min_iterations = max(int(min_iterations), 0)
        patience = max(int(patience), 1)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self._converged = False
        self._stop_reason = "running"
        start_iteration = self.iteration
        target_iteration = self.iteration + max_iterations
        while self.iteration < target_iteration:
            self.step(1)
            if (
                self.iteration - start_iteration >= min_iterations
                and bool(self._best_feasibility["feasible"])
                and self.iteration
                - max(self._last_best_iteration, start_iteration)
                >= patience
            ):
                self._converged = True
                self._stop_reason = "feasible_stagnation"
                break
        if self._stop_reason == "running":
            if bool(self._best_feasibility["feasible"]):
                self._converged = True
                self._stop_reason = "feasible_iteration_budget"
            else:
                self._stop_reason = "max_iterations_infeasible"
        with torch.no_grad():
            self.trajectory.copy_(self._best_trajectory)
        self._project()
        # Adam's moments describe the discarded last iterate, not the restored
        # best one.  A later refinement round must start with fresh moments.
        self.optimizer.state.clear()
        return self.result_json()

    def status_json(self) -> dict[str, Any]:
        with torch.no_grad():
            out, soft, corridor, corridor_detail, task, task_detail = self._evaluate()
        feasibility = self._feasibility(out, corridor_detail, task_detail)
        return {
            "converged": bool(self._converged),
            "reason": self._stop_reason,
            "initialization_source": self._initialization_source,
            "iterations": self.iteration,
            "initial_cost": self._initial_cost,
            "final_cost": float(out.total.mean().item()),
            "best_cost": self._best_cost,
            "objective": float(
                out.total.mean().item()
                + soft.item()
                + self.approach.weight * corridor.item()
                + task.item()
            ),
            "feasibility": feasibility,
        }

    def keyframes_json(self) -> list[dict[str, Any]]:
        result = []
        for item in self._resolved_keyframes:
            result.append(
                {
                    "waypoint": item["waypoint"],
                    "hand": item["hand"],
                    "mode": item["mode"],
                    "weight": item["weight"],
                    "values": {
                        self.scenario.joint_names[col]: value
                        for col, value in item["values"].items()
                    },
                }
            )
        return result

    def geometry_json(self) -> dict[str, Any]:
        """FK paths for the 50 editable trajectory waypoints, in pelvis frame."""
        with torch.no_grad():
            q = self.trajectory.detach()[0]
            frames = self._geometry_fk(q)
            joint_pos = {name: pos for name, (_rot, pos) in frames.items()}
            paths: dict[str, list[list[float]]] = {
                name: _json_value(pos) for name, pos in joint_pos.items()
            }
            stick_data: dict[str, dict[str, torch.Tensor]] = {}
            for hi, hand in enumerate(HANDS):
                wrist_name = f"{hand}_wrist_yaw_joint"
                R, p = frames[wrist_name]
                if bool(self.scenario.context.stick_valid[0, hi].item()):
                    axis_l = self.scenario.context.stick_axis[0, hi]
                    tip_l = self.scenario.context.stick_tip[0, hi]
                    length = float(self.scenario.context.stick_length[0, hi].item())
                    tip_sigma = length - STICK_LENGTH / 2.0
                    mount_l = tip_l - tip_sigma * axis_l
                    butt_l = mount_l - (STICK_LENGTH / 2.0) * axis_l
                    sigma = float(self.scenario.strike.contact_sigma[0, hi].item())
                    contact_l = mount_l + sigma * axis_l
                else:
                    axis_l = q.new_tensor(stick_axis_local(hand))
                    tip_l = q.new_tensor(stick_tip_offset(hand))
                    butt_l = q.new_tensor(stick_contact_offset(hand, -STICK_LENGTH / 2.0))
                    sigma = float(self.scenario.strike.contact_sigma[0, hi].item())
                    contact_l = q.new_tensor(stick_contact_offset(hand, sigma))
                transform = lambda local: p + torch.einsum("bij,j->bi", R, local)  # noqa: E731
                stick_data[hand] = {
                    "wrist": p,
                    "butt": transform(butt_l),
                    "tip": transform(tip_l),
                    "contact": transform(contact_l),
                    "axis": torch.einsum("bij,j->bi", R, axis_l),
                }
                for key, value in stick_data[hand].items():
                    paths[f"{hand}_{key}"] = _json_value(value)

            frame_rows = []
            for k in range(HORIZON):
                frame_rows.append(
                    {
                        "waypoint": k + 1,
                        "time_s": float((k + 1) * self.scenario.dt),
                        "joints": {
                            name: _json_value(pos[k]) for name, pos in joint_pos.items()
                        },
                        "sticks": {
                            hand: {
                                key: _json_value(value[k])
                                for key, value in stick_data[hand].items()
                            }
                            for hand in HANDS
                        },
                    }
                )
        return {"frame": "pelvis", "frames": frame_rows, "paths": paths}

    def result_json(self, *, include_geometry: bool = True) -> dict[str, Any]:
        with torch.no_grad():
            out, soft, corridor, corridor_detail, task, task_detail = self._evaluate()
        approach_json = {
            hand: {
                key: _json_value(value)
                for key, value in row.items()
            }
            for hand, row in corridor_detail.items()
        }
        task_json = {
            hand: {key: _json_value(value) for key, value in row.items()}
            for hand, row in task_detail.items()
        }
        feasibility = self._feasibility(out, corridor_detail, task_detail)
        status = self.status_json()
        result = {
            "scenario_id": self.scenario.scenario_id,
            "iteration": self.iteration,
            "initialization_source": self._initialization_source,
            "joint_names": list(self.scenario.joint_names),
            "actor_initial_trajectory": _json_value(
                self._initial, squeeze_batch=True
            ),
            "trajectory": _json_value(self.trajectory, squeeze_batch=True),
            # Explicit aliases make the output directly saveable as a teacher.
            "teacher": _json_value(self.trajectory, squeeze_batch=True),
            "teacher_mask": _json_value(
                self.scenario.teacher_mask, squeeze_batch=True
            ),
            "keyframes": self.keyframes_json(),
            "task_constraints": [
                constraint.to_json() for constraint in self.task_constraints
            ],
            "diagnostics": {
                "objective": float(
                    out.total.mean().item()
                    + soft.item()
                    + self.approach.weight * corridor.item()
                    + task.item()
                ),
                "soft_keyframe_loss": float(soft.item()),
                "approach_corridor_loss": float(corridor.item()),
                "task_space_loss": float(task.item()),
                "task_space": task_json,
                "approach_corridor": approach_json,
                "feasibility": feasibility,
                "convergence": {
                    key: value for key, value in status.items()
                    if key != "feasibility"
                },
                "cost": _cost_record(out),
                "al_weights": _json_value(self.al.cost_weights()),
                "progress": _json_value(self.progress),
            },
        }
        if include_geometry:
            result["geometry"] = self.geometry_json()
        return result

    def dumps(self, *, include_geometry: bool = True, indent: int | None = None) -> str:
        """Serialize with strict JSON (NaN/Infinity are never emitted)."""
        return json.dumps(
            self.result_json(include_geometry=include_geometry),
            allow_nan=False,
            indent=indent,
        )
