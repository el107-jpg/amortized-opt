"""HPIPM/SQP backend for the trajectory teaching editor.

This is deliberately an optional, tools-only adapter.  The standalone
``amortized_opt`` package remains independent of the original drummer repo;
the editor discovers that checkout at runtime and reports unavailability
cleanly when it is absent.

The inner problem is a structured OCP QP, but the complete solve is SQP:
kinematics, collision distances and inverse dynamics are relinearized around
the current trajectory before each HPIPM solve.  The actor proposal is always
the initial iterate.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from amortized_opt.constants import (
    FIXED_LOWER_BODY_CAPSULES,
    PELVIS_IN_ENV,
    STICK_LENGTH,
    STICK_MOUNT_POS_LEFT,
    STICK_MOUNT_POS_RIGHT,
    STICK_MOUNT_ROT_LEFT_WXYZ,
    STICK_MOUNT_ROT_RIGHT_WXYZ,
    STICK_RADIUS,
    TIP_RADIUS,
    default_urdf_path,
)
from amortized_opt.limits import (
    isaac_effort_limits,
    isaac_velocity_limits,
    urdf_position_limits,
)

from .trajectory_opt import (
    HANDS,
    TaskSpaceConstraint,
    TrajectoryScenario,
    WaypointKeyframe,
)


class HpipmUnavailable(RuntimeError):
    """The optional upstream HPIPM backend cannot be loaded."""


@dataclass(frozen=True)
class HpipmSolveResult:
    trajectory: torch.Tensor
    iterations: int
    converged: bool
    solver_failed: bool
    message: str
    diagnostics: dict[str, Any]


def _control_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("G1_DRUMMER_CONTROL")
    if configured:
        candidates.append(Path(configured).expanduser())
    repo = os.environ.get("G1_DRUMMER_REPO")
    if repo:
        candidates.append(
            Path(repo).expanduser()
            / "source" / "g1_drummer" / "g1_drummer" / "control"
        )
    # Normal sibling-checkout layout used by this workspace.
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "g1-drummer-lab-rythmic"
        / "source" / "g1_drummer" / "g1_drummer" / "control"
    )
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in result:
            result.append(resolved)
    return result


@lru_cache(maxsize=1)
def _load_hpipm_module():
    failures: list[str] = []
    for control in _control_candidates():
        if not (control / "trajopt_hpipm").is_dir():
            failures.append(f"{control}: checkout not found")
            continue
        if str(control) not in sys.path:
            sys.path.insert(0, str(control))
        try:
            module = importlib.import_module("trajopt_hpipm")
            module.ensure_hpipm()
            return module
        except Exception as exc:  # optional dependency boundary
            failures.append(f"{control}: {type(exc).__name__}: {exc}")
    detail = "; ".join(failures) if failures else "no checkout candidates"
    raise HpipmUnavailable(f"HPIPM trajectory backend is unavailable ({detail})")


def hpipm_available() -> tuple[bool, str | None]:
    try:
        _load_hpipm_module()
    except HpipmUnavailable as exc:
        return False, str(exc)
    return True, None


def _synchronize_stick_geometry(mod) -> None:
    """Make the optional legacy solver use this repo's authoritative sticks.

    The upstream checkout has older palm offsets and a 20-degree inward flare.
    Leaving those globals untouched makes its Cartesian Jacobians optimize a
    different contact point and direction from the editor.  The adapter process
    is isolated, so synchronizing these module constants is safe and local.
    """
    values = {
        "STICK_LENGTH": STICK_LENGTH,
        "STICK_RADIUS": STICK_RADIUS,
        "TIP_RADIUS": TIP_RADIUS,
        "STICK_MOUNT_POS_LEFT": STICK_MOUNT_POS_LEFT,
        "STICK_MOUNT_POS_RIGHT": STICK_MOUNT_POS_RIGHT,
        "STICK_MOUNT_POS": STICK_MOUNT_POS_LEFT,
        "STICK_MOUNT_ROT_LEFT_WXYZ": STICK_MOUNT_ROT_LEFT_WXYZ,
        "STICK_MOUNT_ROT_RIGHT_WXYZ": STICK_MOUNT_ROT_RIGHT_WXYZ,
        "STICK_MOUNT_ROT_WXYZ": STICK_MOUNT_ROT_LEFT_WXYZ,
    }
    for module in (mod._problem, mod._robot):
        for name, value in values.items():
            if hasattr(module, name):
                setattr(module, name, value)


def _minimum_jerk_waist_reference(
    scenario: TrajectoryScenario,
    constraint: TaskSpaceConstraint,
    horizon: int,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """A discrete waist reference that agrees with the measured opening speed."""
    names = scenario.joint_names
    cols = np.asarray(
        [names.index(f"waist_{axis}_joint") for axis in ("yaw", "pitch", "roll")],
        dtype=int,
    )
    q0 = scenario.q0[0].detach().cpu().numpy().astype(float)
    v0 = scenario.qd0[0].detach().cpu().numpy().astype(float)
    target = np.asarray(constraint.waist, dtype=float)
    hit = int(scenario.hit_index[0, HANDS.index(constraint.hand)].item())
    hit = min(max(hit, 1), horizon)
    ref = np.repeat(q0[cols][None, :], horizon + 1, axis=0)
    if hit == 1:
        ref[1:] = target
    else:
        # The upstream boundary row and waist pin must agree exactly at row 1.
        ref[1] = np.clip(
            q0[cols] + float(scenario.dt) * v0[cols],
            q_lower[cols], q_upper[cols],
        )
        rows = np.arange(2, hit + 1)
        if rows.size:
            s = (rows - 1).astype(float) / float(hit - 1)
            alpha = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
            ref[rows] = (
                (1.0 - alpha)[:, None] * ref[1][None, :]
                + alpha[:, None] * target[None, :]
            )
        ref[hit:] = target
    v_start = v0.copy()
    v_start[cols] = (ref[1] - ref[0]) / float(scenario.dt)
    return ref, v_start


def _obstacles(mod, scenario: TrajectoryScenario) -> list[Any]:
    valid = scenario.context.drum_valid[0].detach().cpu().bool().numpy()
    centers = (
        scenario.context.drum_pos[0].detach().cpu().numpy().astype(float)
        - np.asarray(PELVIS_IN_ENV, dtype=float)
    )
    axes = scenario.context.drum_axis[0].detach().cpu().numpy().astype(float)
    radii = scenario.context.drum_radius[0].detach().cpu().numpy().astype(float)
    heights = scenario.context.drum_height[0].detach().cpu().numpy().astype(float)
    out: list[Any] = []
    for index, name in enumerate(scenario.drum_names):
        if index >= valid.size or not bool(valid[index]):
            continue
        axis = axes[index] / max(float(np.linalg.norm(axes[index])), 1.0e-12)
        out.append(
            mod.CylinderObstacle(
                center=centers[index], axis=axis, radius=float(radii[index]),
                height=float(heights[index]), name=name, head_frame=True,
            )
        )
        # The same below-head no-go volume used by the reference adapter.
        depth = 0.08
        out.append(
            mod.CylinderObstacle(
                center=(
                    centers[index]
                    - (float(heights[index]) + 0.01 + 0.5 * depth) * axis
                ),
                axis=axis, radius=float(radii[index]), height=depth,
                name=f"{name}__under", head_frame=False,
            )
        )
    # The upstream upper-body geometry predates the fixed-leg collision term.
    # Conservative cylinders make those pelvis-fixed capsules visible to SQP;
    # the editor's exact capsule scorer remains the final authority.
    for name, a_raw, b_raw, radius in FIXED_LOWER_BODY_CAPSULES:
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            out.append(mod.SphereObstacle(center=a, radius=float(radius), name=f"body_{name}"))
        else:
            out.append(
                mod.CylinderObstacle(
                    center=0.5 * (a + b), axis=delta / length,
                    radius=float(radius), height=length + 2.0 * float(radius),
                    name=f"body_{name}", head_frame=False,
                )
            )
    return out


def _strike_specs(
    mod,
    scenario: TrajectoryScenario,
    constraint: TaskSpaceConstraint,
    horizon: int,
) -> list[Any]:
    selected = HANDS.index(constraint.hand)
    specs: list[Any] = []
    for hi, hand in enumerate(HANDS):
        if not bool(scenario.active[0, hi].item()):
            continue
        hit = int(scenario.hit_index[0, hi].item())
        if not 1 <= hit <= horizon:
            continue
        p_original = scenario.strike.p_star[0, hi].detach().cpu().numpy().astype(float)
        position = (
            np.asarray(constraint.contact_position, dtype=float)
            if hi == selected else p_original
        )
        disk_offset = scenario.strike.disk_offset[0, hi].detach().cpu().numpy().astype(float)
        if hi == selected:
            # Keep the legal disk centred on the physical head after moving the
            # preferred contact point within that disk.
            disk_offset = p_original + disk_offset - position
        half_plane = None
        if bool(scenario.strike.has_half_plane[0, hi].item()):
            half_plane = (
                scenario.strike.half_plane_dir[0, hi]
                .detach().cpu().numpy().astype(float)
            )
        drum_index = int(scenario.target_drum[0, hi].item())
        specs.append(
            mod.StrikeSpec(
                time=float(hit * scenario.dt),
                hand=hand,
                position=position,
                velocity=scenario.strike.v_star[0, hi].detach().cpu().numpy().astype(float),
                drum=scenario.drum_names[drum_index],
                hit_radius=float(scenario.strike.hit_radius[0, hi].item()),
                half_plane_dir=half_plane,
                disk_center_offset=disk_offset,
                contact_sigma=float(scenario.strike.contact_sigma[0, hi].item()),
            )
        )
    return specs


def _coerce_keyframe_targets(
    scenario: TrajectoryScenario,
    keyframes: Iterable[WaypointKeyframe | Mapping[str, Any]],
    horizon: int,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray, float]], list[tuple[int, int, float]]]:
    soft: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    hard: list[tuple[int, int, float]] = []
    for raw in keyframes:
        item = raw if isinstance(raw, WaypointKeyframe) else WaypointKeyframe(**dict(raw))
        if item.waypoint is None:
            if item.hand not in HANDS:
                raise ValueError("hit keyframe requires a left/right hand")
            waypoint = int(scenario.hit_index[0, HANDS.index(item.hand)].item())
        else:
            waypoint = int(item.waypoint)
        if not 1 <= waypoint <= horizon:
            continue
        cols = np.asarray([scenario.joint_names.index(name) for name in item.values], dtype=int)
        values = np.asarray([float(value) for value in item.values.values()], dtype=float)
        if item.mode == "hard":
            hard.extend((waypoint, int(col), float(value)) for col, value in zip(cols, values))
        else:
            # Editor weights are intentionally human-scale.  Lift them into the
            # dt-scaled SQP smoothness objective without changing the UI knob.
            soft.append((waypoint, cols, values, max(float(item.weight), 0.0) * 1.0e6))
    return soft, hard


def solve_hpipm_trajectory(
    scenario: TrajectoryScenario,
    initial_trajectory: torch.Tensor,
    *,
    task_constraint: TaskSpaceConstraint,
    keyframes: Iterable[WaypointKeyframe | Mapping[str, Any]] = (),
    max_iterations: int = 15,
) -> HpipmSolveResult:
    """Run full-17-DoF SQP and return the actor-shaped ``(1, 50, 17)`` path."""
    mod = _load_hpipm_module()
    _synchronize_stick_geometry(mod)
    initial = torch.as_tensor(initial_trajectory, dtype=torch.float32)
    if tuple(initial.shape) == (50, 17):
        initial = initial.unsqueeze(0)
    if tuple(initial.shape) != (1, 50, 17) or not bool(torch.isfinite(initial).all()):
        raise ValueError("initial_trajectory must be a finite (1, 50, 17) tensor")
    if task_constraint.hand not in HANDS:
        raise ValueError("task constraint hand must be left or right")
    if task_constraint.terminal_q is None:
        raise ValueError("HPIPM SQP requires the task IK terminal_q")
    terminal_q = np.asarray(task_constraint.terminal_q, dtype=float)
    if terminal_q.shape != (17,) or not np.all(np.isfinite(terminal_q)):
        raise ValueError("task IK terminal_q must be a finite 17-vector")

    active_hits = [
        int(scenario.hit_index[0, hi].item())
        for hi in range(2) if bool(scenario.active[0, hi].item())
    ]
    if not active_hits:
        raise ValueError("HPIPM SQP requires at least one active strike")
    horizon = max(active_hits)
    q_lower, q_upper = urdf_position_limits(
        scenario.joint_names, posture_limits=False
    )
    waist_ref, v_start = _minimum_jerk_waist_reference(
        scenario, task_constraint, horizon, q_lower, q_upper
    )
    q0 = scenario.q0[0].detach().cpu().numpy().astype(float)
    q_seed = initial[0, :horizon].detach().cpu().numpy().astype(float)
    Q_init = np.concatenate([q0[None, :], q_seed], axis=0)
    strikes = _strike_specs(mod, scenario, task_constraint, horizon)
    soft_targets, hard_pins = _coerce_keyframe_targets(
        scenario, keyframes, horizon
    )
    selected_cols = np.asarray(
        [
            index for index, name in enumerate(scenario.joint_names)
            if name.startswith(f"{task_constraint.hand}_")
        ],
        dtype=int,
    )
    selected_hit = int(
        scenario.hit_index[0, HANDS.index(task_constraint.hand)].item()
    )
    # The task IK pose is the user's explicit selected-hand terminal command.
    # Pin only that arm at its own hit; the opposite arm remains free and the
    # waist follows its separate minimum-jerk reference.  The Cartesian costs
    # below still make the intended contact/direction explicit to SQP.
    hard_pins.extend(
        (selected_hit, int(col), float(terminal_q[col])) for col in selected_cols
    )
    direction_target = np.asarray(task_constraint.stick_direction, dtype=float)
    direction_target /= max(float(np.linalg.norm(direction_target)), 1.0e-12)

    class EditorDrumTrajOpt(mod.DrumTrajOpt):
        """Small extension adding editor-only time-local and orientation costs."""

        def __init__(self, cfg):
            self._editor_soft = soft_targets
            self._editor_hard = hard_pins
            self._editor_direction = (
                selected_hit, task_constraint.hand, direction_target, 1.0e9
            )
            super().__init__(cfg)
            qoff = self.ws.Q0
            cs = float(self.ws.cost_scale)
            for waypoint, cols, values, weight in self._editor_soft:
                if weight <= 0.0:
                    continue
                idx = qoff + np.asarray(cols, dtype=int)
                self.ws._Q_base[waypoint][idx, idx] += cs * 2.0 * weight
                self.ws._q_base[waypoint][idx] += cs * -2.0 * weight * values
            self.ws._Q = [value.copy() for value in self.ws._Q_base]
            self.ws._q = [value.copy() for value in self.ws._q_base]
            self.ws._cost_written = False

        def _write_qp(self, ev, rho, bc):
            super()._write_qp(ev, rho, bc)
            waypoint, hand, target, weight = self._editor_direction
            X = np.asarray(ev.x, dtype=float).reshape(self.cfg.N + 1, self.cfg.n)
            qbar = X[waypoint]
            self.rob.fk(qbar)
            axis = np.asarray(self.rob.stick_axis(hand), dtype=float)
            jac = np.asarray(self.rob.stick_axis_jacobian(hand), dtype=float)
            if np.all(np.isfinite(axis)) and np.all(np.isfinite(jac)):
                c = axis - jac @ qbar - target
                qoff = self.ws.Q0
                sl = slice(qoff, qoff + self.cfg.n)
                cs = float(self.ws.cost_scale)
                self.ws._Q[waypoint][sl, sl] += cs * 2.0 * weight * (jac.T @ jac)
                self.ws._q[waypoint][sl] += cs * 2.0 * weight * (jac.T @ c)
                self.ws._cost_const += cs * weight * float(c @ c)
                self.ws._cost_dirty.add(waypoint)
            for k, col, value in self._editor_hard:
                self.ws._lbx[k][col] = value
                self.ws._ubx[k][col] = value

        def _evaluate(self, x):
            ev = super()._evaluate(x)
            X = np.asarray(x, dtype=float).reshape(self.cfg.N + 1, self.cfg.n)
            extra = 0.0
            for waypoint, cols, values, weight in self._editor_soft:
                error = X[waypoint, cols] - values
                extra += weight * float(error @ error)
            waypoint, hand, target, weight = self._editor_direction
            self.rob.fk(X[waypoint])
            direction_error = np.asarray(self.rob.stick_axis(hand), dtype=float) - target
            extra += weight * float(direction_error @ direction_error)
            ev.cost += extra
            return ev

    cfg = mod.TrajOptConfig(
        urdf_path=default_urdf_path(),
        joint_names=tuple(scenario.joint_names),
        N=int(horizon),
        dt=float(scenario.dt),
        obstacles=_obstacles(mod, scenario),
        strikes=strikes,
        q_start=q0,
        q_goal=q_seed[-1].copy(),
        v_start=v_start,
        v_goal=None,
        a_start=np.zeros(17),
        a_goal=np.zeros(17),
        q_lower_override=np.asarray(q_lower, dtype=float),
        q_upper_override=np.asarray(q_upper, dtype=float),
        v_limit_override=isaac_velocity_limits(scenario.joint_names),
        tau_limit_override=isaac_effort_limits(scenario.joint_names),
        waist_reference=waist_ref,
        waist_pin_tol=1.0e-5,
        simplified_cost=False,
        seed_mode="quintic",
        seed_apex_first=False,
        strike_goal_nearest=False,
        w_posture=0.0,
        w_strike_posture=0.0,
        w_strike_goal_posture=0.0,
        w_strike_position=2.0e8,
        strike_corridor_waypoints=min(10, max(selected_hit - 1, 0)),
        max_iterations=max(1, min(int(max_iterations), 40)),
    )
    solver = EditorDrumTrajOpt(cfg)
    result = solver.solve(Q_init.reshape(-1))
    solved = np.asarray(result.Q, dtype=float)
    if solved.shape != (horizon + 1, 17) or not np.all(np.isfinite(solved)):
        raise RuntimeError("HPIPM returned a non-finite or incorrectly shaped trajectory")
    output = initial.detach().clone()
    output[0, :horizon] = torch.as_tensor(solved[1:], dtype=output.dtype)
    diagnostics = {
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "solver_failed": bool(result.solver_failed),
        "message": str(result.message),
        "max_slack": float(result.max_slack),
        "max_slack_external": float(result.max_slack_ext),
        "max_slack_self": float(result.max_slack_self),
        "max_slack_torque": float(result.max_slack_tau),
        "max_slack_velocity": float(result.max_slack_vel),
        "max_slack_corridor": float(result.max_slack_corridor),
        "horizon": int(horizon),
        "selected_hand": task_constraint.hand,
    }
    if not all(
        math.isfinite(value)
        for key, value in diagnostics.items()
        if key.startswith("max_slack")
    ):
        raise RuntimeError("HPIPM returned non-finite slack diagnostics")
    return HpipmSolveResult(
        trajectory=output,
        iterations=int(result.iterations),
        converged=bool(result.converged),
        solver_failed=bool(result.solver_failed),
        message=str(result.message),
        diagnostics=diagnostics,
    )


__all__ = [
    "HpipmSolveResult",
    "HpipmUnavailable",
    "hpipm_available",
    "solve_hpipm_trajectory",
]
