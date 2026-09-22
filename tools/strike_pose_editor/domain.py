"""Isaac-free geometry and validation for the strike-pose editor.

This module deliberately owns no persistent state.  It translates the core
robot/drum representation into small JSON-compatible documents and performs
the same FK and collision checks for both preview and save requests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from amortized_opt.constants import (
    G1_UPPER_BODY_JOINTS,
    STICK_LENGTH,
    ready_joint_pos,
)
from amortized_opt.fk import (
    ChainFK,
    stick_axis_local,
    stick_contact_offset,
    stick_tip_offset,
)
from amortized_opt.limits import urdf_position_limits
from amortized_opt.rhythm import DrumKit
from amortized_opt.sdf import SdfPenalty
from amortized_opt.strike import default_contact_sigma, shaft_hit_sigma_range
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice

__all__ = ["DomainValidationError", "EditorDomain"]


# Task-space IK is weighted least squares, not a strict lexicographic program.
# These scales make stick direction the dominant soft task when it conflicts
# with the exact drum point. Hard joint bounds, fixed waist angles, and the
# collision acceptance gate remain safety guardrails.
IK_POSITION_WEIGHT = 2.0e4
IK_DIRECTION_WEIGHT = 2.0e3
IK_ELBOW_WEIGHT = 900.0
IK_POSTURE_WEIGHT = 30.0
IK_COLLISION_WEIGHT = 2.0e4
IK_PROXIMITY_WEIGHT = 0.08
IK_CONTACT_ACCEPT_M = 0.015
IK_DIRECTION_ACCEPT_DEG = 2.0


class DomainValidationError(ValueError):
    """A malformed or physically out-of-domain editor request."""


def _list(x: torch.Tensor | np.ndarray | Iterable[float]) -> list[float]:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return [float(v) for v in np.asarray(x, dtype=float).reshape(-1)]


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise DomainValidationError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class PoseSelection:
    hand: str
    drum: str
    drum_id: int
    contact_sigma: float


class EditorDomain:
    """CPU-only facade over the optimizer's exact geometry helpers."""

    schema_version = 1

    def __init__(self) -> None:
        # This tool handles one pose at a time.  Extra BLAS workers only contend
        # with training jobs and make request latency less predictable.
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.joint_names = tuple(G1_UPPER_BODY_JOINTS)
        self.ready = ready_joint_pos(self.joint_names).astype(float)
        self.q_min, self.q_max = urdf_position_limits(
            self.joint_names, posture_limits=True
        )
        self.hard_min, self.hard_max = urdf_position_limits(
            self.joint_names, posture_limits=False
        )
        self.kit = DrumKit.from_layout()
        self.library = GoalPoseLibrary(joint_names=self.joint_names)

        # Naming every controlled joint gives the canvas a complete kinematic
        # point set without introducing a second robot description.
        frames = {name: name for name in self.joint_names}
        self.fk = ChainFK(
            joint_names=self.joint_names,
            frames=frames,
            device=self.device,
            dtype=self.dtype,
        )
        # Default scene is in env coordinates and adds PELVIS_IN_ENV to the FK;
        # this is exactly equivalent to comparing pelvis-frame FK with
        # kit.positions_in("pelvis"), which the contact diagnostics below use.
        self.sdf = SdfPenalty(device=self.device, dtype=self.dtype)
        self._yaw_i = self.joint_names.index("waist_yaw_joint")

    # --------------------------------------------------------------- config --
    def config(self) -> dict[str, Any]:
        joints = []
        for i, name in enumerate(self.joint_names):
            joints.append(
                {
                    "name": name,
                    # min/max are the actor's effective training box.
                    "min": float(self.q_min[i]),
                    "max": float(self.q_max[i]),
                    "hard_min": float(self.hard_min[i]),
                    "hard_max": float(self.hard_max[i]),
                    "ready": float(self.ready[i]),
                }
            )

        pelvis_pos = self.kit.positions_in("pelvis")
        drums = [
            {
                "id": i,
                "name": name,
                "position": _list(pelvis_pos[i]),
                "axis": _list(self.kit.axes[i]),
                "radius": float(self.kit.radii[i]),
                "height": float(self.kit.heights[i]),
                "shaft_hit": bool(self.kit.shaft_hit[i]),
                "edge_hit": bool(self.kit.edge_hit[i]),
                "contact_sigma_min": float(
                    (shaft_hit_sigma_range(name) or (default_contact_sigma(name),))[0]
                ),
                "contact_sigma_max": float(
                    (shaft_hit_sigma_range(name) or (default_contact_sigma(name),))[-1]
                ),
            }
            for i, name in enumerate(self.kit.names)
        ]
        hands: dict[str, dict[str, list[str]]] = {}
        for hand in ("left", "right"):
            hands[hand] = {
                "drums": [
                    self.kit.names[i]
                    for i in self.kit.eligible(hand)
                    if self.library.has(hand, self.kit.names[i])
                ]
            }

        seeds = []
        for hand, drum in self.library.pairs:
            cols = arm_slice(hand, self.joint_names)
            for entry in self.library.entries(hand, drum):
                q = self.ready.copy()
                q[:3] = entry.q[:3]
                q[cols] = entry.arm_q
                seeds.append(
                    {
                        "hand": hand,
                        "drum": drum,
                        "yaw_rad": float(entry.waist_yaw),
                        "q": _list(q),
                        "tip_err": float(entry.tip_err),
                        "wrist_headroom": float(entry.wrist_headroom),
                        "contact_sigma": float(default_contact_sigma(drum)),
                    }
                )
        return {
            "schema_version": self.schema_version,
            "joints": joints,
            "hands": hands,
            "yaw_bins_rad": [float(v) for v in self.library.yaw_grid_rad],
            "drums": drums,
            "seeds": seeds,
            "stick": {
                "tip_sigma": float(default_contact_sigma(None)),
                "shaft_sigma_min": float(
                    min(
                        span[0]
                        for name in self.kit.names
                        if (span := shaft_hit_sigma_range(name)) is not None
                    )
                ),
            },
        }

    # ------------------------------------------------------------ validation --
    def validate_pose(
        self, q: Iterable[float], *, allow_posture_margin: bool = False
    ) -> np.ndarray:
        try:
            out = np.asarray(list(q), dtype=float)
        except (TypeError, ValueError) as exc:
            raise DomainValidationError("q must be a numeric 17-element array") from exc
        if out.shape != (len(self.joint_names),):
            raise DomainValidationError(
                f"q must contain {len(self.joint_names)} values, got shape {out.shape}"
            )
        if not np.all(np.isfinite(out)):
            raise DomainValidationError("q must contain only finite values")
        lower = self.hard_min if allow_posture_margin else self.q_min
        upper = self.hard_max if allow_posture_margin else self.q_max
        bad = np.flatnonzero((out < lower - 1e-10) | (out > upper + 1e-10))
        if bad.size:
            details = ", ".join(
                f"{self.joint_names[i]}={out[i]:.6g} outside "
                f"[{lower[i]:.6g}, {upper[i]:.6g}]"
                for i in bad[:4]
            )
            limit_kind = "hard joint" if allow_posture_margin else "editor posture"
            raise DomainValidationError(f"pose is outside the {limit_kind} limits: {details}")
        return out

    def validate_selection(
        self, hand: str, drum: str, contact_sigma: float | None
    ) -> PoseSelection:
        if hand not in ("left", "right"):
            raise DomainValidationError("hand must be 'left' or 'right'")
        if drum not in self.kit.names:
            raise DomainValidationError(f"unknown drum {drum!r}")
        drum_id = self.kit.index(drum)
        if drum_id not in self.kit.eligible(hand) or not self.library.has(hand, drum):
            raise DomainValidationError(f"{hand} hand is not eligible for drum {drum!r}")
        sigma = (
            default_contact_sigma(drum)
            if contact_sigma is None
            else _finite_scalar(contact_sigma, "contact_sigma")
        )
        span = shaft_hit_sigma_range(drum)
        if span is None:
            expected = float(default_contact_sigma(drum))
            if abs(sigma - expected) > 1e-7:
                raise DomainValidationError(
                    f"{drum!r} is tip-only; contact_sigma must be {expected:.6g} m"
                )
        elif not (float(span[0]) - 1e-9 <= sigma <= float(span[1]) + 1e-9):
            raise DomainValidationError(
                f"contact_sigma for {drum!r} must be in "
                f"[{span[0]:.6g}, {span[1]:.6g}] m"
            )
        return PoseSelection(hand, drum, drum_id, float(sigma))

    # --------------------------------------------------------------- geometry --
    def _stick_geometry(
        self,
        q_t: torch.Tensor,
        hand: str,
        contact_sigma: float,
    ) -> dict[str, list[float]]:
        wrist_name = f"{hand}_wrist_yaw_joint"
        R, p = self.fk.frame(q_t, wrist_name)
        local = torch.as_tensor(
            np.stack(
                [
                    stick_contact_offset(hand, -STICK_LENGTH / 2.0),
                    stick_tip_offset(hand),
                    stick_contact_offset(hand, contact_sigma),
                ]
            ),
            device=self.device,
            dtype=self.dtype,
        )
        pts = ChainFK.points_world(R, p, local)[0]
        axis_local = torch.as_tensor(
            stick_axis_local(hand), device=self.device, dtype=self.dtype
        )
        axis = R[0] @ axis_local
        return {
            "wrist": _list(p[0]),
            "butt": _list(pts[0]),
            "tip": _list(pts[1]),
            "contact": _list(pts[2]),
            "axis": _list(axis),
        }

    @torch.no_grad()
    def pose_geometry(
        self,
        q: Iterable[float],
        contact_sigma: Mapping[str, float] | None = None,
        *,
        allow_posture_margin: bool = False,
    ) -> dict[str, Any]:
        """Return FK geometry for an unsaved editor pose.

        This deliberately performs no collision/target evaluation so slider
        previews stay cheap.  A scenario may supply each hand's actual
        tip/shaft contact coordinate; otherwise both hands use the bead.
        """
        q_np = self.validate_pose(q, allow_posture_margin=allow_posture_margin)
        q_t = torch.as_tensor(q_np, device=self.device, dtype=self.dtype).view(1, -1)
        frames = self.fk.forward(q_t)
        joints = {name: _list(value[1][0]) for name, value in frames.items()}
        sigma = contact_sigma or {}
        sticks = {
            hand: self._stick_geometry(
                q_t, hand, float(sigma.get(hand, default_contact_sigma(None)))
            )
            for hand in ("left", "right")
        }
        return {"joints": joints, "sticks": sticks}

    def solve_task_pose(
        self,
        q_seed: Iterable[float],
        *,
        hand: str,
        contact_position: Iterable[float],
        stick_direction: Iterable[float],
        waist: Iterable[float],
        contact_sigma: float,
        joint_guides: Mapping[str, float] | None = None,
        guide_weight: float = 25.0,
        iterations: int = 140,
        drum_head_centers: torch.Tensor | None = None,
        drum_radii: torch.Tensor | None = None,
        target_drum: int = -1,
    ) -> dict[str, Any]:
        """Solve one hand's terminal pose from contact/axis task coordinates.

        Waist angles are fixed by the editor.  Only the selected seven-DoF arm
        is optimized; manually touched arm joints become soft guide residuals.
        Multiple deterministic seeds let the objective choose an elbow-down,
        collision-clear branch without making small edits jump unpredictably.
        """
        if hand not in ("left", "right"):
            raise DomainValidationError("hand must be 'left' or 'right'")
        seed = self.validate_pose(q_seed, allow_posture_margin=True)

        def vec3(value: Iterable[float], name: str) -> np.ndarray:
            try:
                out = np.asarray(list(value), dtype=float)
            except (TypeError, ValueError) as exc:
                raise DomainValidationError(f"{name} must be a numeric 3-vector") from exc
            if out.shape != (3,) or not np.all(np.isfinite(out)):
                raise DomainValidationError(f"{name} must be a finite 3-vector")
            return out

        target_p = vec3(contact_position, "contact_position")
        target_axis = vec3(stick_direction, "stick_direction")
        axis_norm = float(np.linalg.norm(target_axis))
        if axis_norm < 1e-8:
            raise DomainValidationError("stick_direction must be nonzero")
        target_axis /= axis_norm
        waist_np = vec3(waist, "waist")
        waist_idx = np.asarray(
            [self.joint_names.index(f"waist_{axis}_joint") for axis in ("yaw", "pitch", "roll")],
            dtype=np.int64,
        )
        if np.any(waist_np < self.hard_min[waist_idx] - 1e-10) or np.any(
            waist_np > self.hard_max[waist_idx] + 1e-10
        ):
            raise DomainValidationError("waist target is outside hard joint limits")
        sigma = _finite_scalar(contact_sigma, "contact_sigma")
        if not (0 <= int(target_drum) < len(self.kit.names)):
            raise DomainValidationError("target_drum is outside the drum table")
        iterations = int(iterations)
        if not 10 <= iterations <= 400:
            raise DomainValidationError("iterations must be in [10, 400]")
        guide_weight = _finite_scalar(guide_weight, "guide_weight")
        if not 0.0 <= guide_weight <= 1.0e4:
            raise DomainValidationError("guide_weight must be in [0, 10000]")

        arm_cols = np.asarray(arm_slice(hand, self.joint_names), dtype=np.int64)
        arm_names = [self.joint_names[i] for i in arm_cols]
        guides = dict(joint_guides or {})
        unknown = sorted(set(guides) - set(arm_names))
        if unknown:
            raise DomainValidationError(
                "joint guides must belong to the selected arm: " + ", ".join(unknown[:4])
            )
        guide_cols: list[int] = []
        guide_values: list[float] = []
        for name, raw in guides.items():
            value = _finite_scalar(raw, f"joint_guides.{name}")
            index = self.joint_names.index(name)
            if not self.hard_min[index] - 1e-10 <= value <= self.hard_max[index] + 1e-10:
                raise DomainValidationError(f"joint guide {name!r} is outside hard limits")
            guide_cols.append(arm_names.index(name))
            guide_values.append(value)

        dev, dtype = self.device, self.dtype
        arm_i = torch.as_tensor(arm_cols, device=dev, dtype=torch.long)
        waist_i = torch.as_tensor(waist_idx, device=dev, dtype=torch.long)
        target_p_t = torch.as_tensor(target_p, device=dev, dtype=dtype)
        target_axis_t = torch.as_tensor(target_axis, device=dev, dtype=dtype)
        q_fixed = torch.as_tensor(seed, device=dev, dtype=dtype)
        q_fixed = q_fixed.index_copy(0, waist_i, torch.as_tensor(waist_np, device=dev, dtype=dtype))
        arm_ref = q_fixed.index_select(0, arm_i)
        arm_lo = torch.as_tensor(self.hard_min[arm_cols], device=dev, dtype=dtype)
        arm_hi = torch.as_tensor(self.hard_max[arm_cols], device=dev, dtype=dtype)
        preferred_lo = torch.as_tensor(self.q_min[arm_cols], device=dev, dtype=dtype)
        preferred_hi = torch.as_tensor(self.q_max[arm_cols], device=dev, dtype=dtype)

        # Seed zero preserves the actor branch.  The other deterministic seeds
        # explore the arm's null space without changing the requested waist.
        seed_offsets = torch.tensor(
            [
                [0.0] * 7,
                [0.0, -0.10, 0.12, 0.22, 0.0, -0.08, 0.0],
                [0.0, 0.10, -0.12, -0.18, 0.0, 0.08, 0.0],
            ],
            device=dev,
            dtype=dtype,
        )
        arm = (arm_ref.unsqueeze(0) + seed_offsets).clamp(arm_lo, arm_hi).detach()
        arm.requires_grad_(True)
        optimizer = torch.optim.Adam([arm], lr=0.035)
        wrist_name = f"{hand}_wrist_yaw_joint"
        elbow_name = f"{hand}_elbow_joint"
        shoulder_name = f"{hand}_shoulder_pitch_joint"
        contact_local = torch.as_tensor(
            stick_contact_offset(hand, sigma), device=dev, dtype=dtype,
        ).view(1, 3)
        axis_local = torch.as_tensor(stick_axis_local(hand), device=dev, dtype=dtype)
        guide_i = torch.as_tensor(guide_cols, device=dev, dtype=torch.long)
        guide_t = torch.as_tensor(guide_values, device=dev, dtype=dtype)
        heads = None if drum_head_centers is None else drum_head_centers.to(device=dev, dtype=dtype)
        radii = None if drum_radii is None else drum_radii.to(device=dev, dtype=dtype)
        target_ids = {
            "left": torch.full((arm.shape[0],), int(target_drum) if hand == "left" else -1, device=dev, dtype=torch.long),
            "right": torch.full((arm.shape[0],), int(target_drum) if hand == "right" else -1, device=dev, dtype=torch.long),
        }

        def objective(arm_q: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            full = q_fixed.unsqueeze(0).expand(arm_q.shape[0], -1).index_copy(1, arm_i, arm_q)
            wrist_r, wrist_p = self.fk.frame(full, wrist_name)
            contact = ChainFK.points_world(wrist_r, wrist_p, contact_local)[:, 0]
            axis = torch.einsum("bij,j->bi", wrist_r, axis_local)
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            position_sq = (contact - target_p_t).square().sum(dim=-1)
            direction = (1.0 - (axis * target_axis_t).sum(dim=-1).clamp(-1.0, 1.0))
            _elbow_r, elbow_p = self.fk.frame(full, elbow_name)
            _shoulder_r, shoulder_p = self.fk.frame(full, shoulder_name)
            chord_z = 0.5 * (shoulder_p[:, 2] + wrist_p[:, 2])
            elbow_violation = (elbow_p[:, 2] - chord_z + 0.02).clamp_min(0.0)
            prox = (arm_q - arm_ref).square().sum(dim=-1)
            posture = (
                (preferred_lo - arm_q).clamp_min(0.0).square()
                + (arm_q - preferred_hi).clamp_min(0.0).square()
            ).sum(dim=-1)
            guide = arm_q.new_zeros(arm_q.shape[0])
            if guide_i.numel():
                guide = (arm_q.index_select(1, guide_i) - guide_t).square().sum(dim=-1)
            collision = self.sdf(
                full,
                drum_head_centers=None if heads is None else heads.expand(arm_q.shape[0], -1, -1),
                drum_radii=None if radii is None else radii.expand(arm_q.shape[0], -1),
                target_drum_left=target_ids["left"],
                target_drum_right=target_ids["right"],
            )
            total = (
                IK_POSITION_WEIGHT * position_sq
                + IK_DIRECTION_WEIGHT * direction
                + guide_weight * guide
                + IK_ELBOW_WEIGHT * elbow_violation.square()
                + IK_POSTURE_WEIGHT * posture
                + IK_COLLISION_WEIGHT * collision
                + IK_PROXIMITY_WEIGHT * prox
            )
            return total, {
                "full": full,
                "contact": contact,
                "axis": axis,
                "position_sq": position_sq,
                "direction": direction,
                "guide": guide,
                "elbow_drop": chord_z - elbow_p[:, 2],
            }

        for step in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = objective(arm)
            loss.sum().backward()
            optimizer.step()
            with torch.no_grad():
                arm.clamp_(arm_lo, arm_hi)
            if step == int(iterations * 0.65):
                for group in optimizer.param_groups:
                    group["lr"] = 0.012

        with torch.no_grad():
            loss, parts = objective(arm)
            best = int(loss.argmin().item())
            q_best = parts["full"][best]
            geometry = self.pose_geometry(
                q_best.cpu().tolist(), {hand: sigma}, allow_posture_margin=True,
            )
            _, sdf = self.sdf(
                q_best.view(1, -1),
                drum_head_centers=heads,
                drum_radii=radii,
                target_drum_left=torch.tensor([target_drum if hand == "left" else -1], device=dev),
                target_drum_right=torch.tensor([target_drum if hand == "right" else -1], device=dev),
                return_details=True,
            )
            position_error = float(parts["position_sq"][best].sqrt().item())
            direction_error = math.degrees(
                math.acos(float((parts["axis"][best] * target_axis_t).sum().clamp(-1.0, 1.0).item()))
            )
            guide_rmse = (
                float((parts["guide"][best] / max(len(guide_cols), 1)).sqrt().item())
                if guide_cols else 0.0
            )
            min_sdf = float(sdf.extras[f"min_sdf_{hand}"][0].item())
            combined_min_sdf = float(sdf.min_sdf[0].item())
            elbow_drop = float(parts["elbow_drop"][best].item())
        return {
            "q": _list(q_best),
            "geometry": geometry,
            "metrics": {
                "contact_error_m": position_error,
                "direction_error_deg": direction_error,
                "guide_rmse_rad": guide_rmse,
                "elbow_drop_m": elbow_drop,
                "min_sdf_m": min_sdf,
                "combined_min_sdf_m": combined_min_sdf,
                "objective": float(loss[best].item()),
            },
            "accepted": bool(
                position_error <= IK_CONTACT_ACCEPT_M
                and direction_error <= IK_DIRECTION_ACCEPT_DEG
                and min_sdf >= -0.001
            ),
            "priority": {
                "hard": ["joint_limits", "fixed_waist"],
                "guardrail": ["collision_clearance"],
                "soft": [
                    "stick_azimuth_elevation",
                    "contact_position",
                    "elbow_down",
                    "manual_joint_guides",
                    "preferred_posture",
                    "seed_proximity",
                ],
                "weights": {
                    "contact_position_m2": IK_POSITION_WEIGHT,
                    "stick_direction_1_minus_cos": IK_DIRECTION_WEIGHT,
                    "collision": IK_COLLISION_WEIGHT,
                    "elbow_down_m2": IK_ELBOW_WEIGHT,
                    "manual_joint_guides_rad2": guide_weight,
                    "preferred_posture_rad2": IK_POSTURE_WEIGHT,
                    "seed_proximity_rad2": IK_PROXIMITY_WEIGHT,
                },
            },
            "guided_joints": list(guides),
        }

    @torch.no_grad()
    def evaluate(
        self,
        q: Iterable[float],
        hand: str,
        drum: str,
        contact_sigma: float | None = None,
    ) -> dict[str, Any]:
        q_np = self.validate_pose(q)
        selected = self.validate_selection(hand, drum, contact_sigma)
        q_t = torch.as_tensor(q_np, device=self.device, dtype=self.dtype).view(1, -1)
        geometry = self.pose_geometry(
            q_np,
            {selected.hand: selected.contact_sigma},
        )
        joints, sticks = geometry["joints"], geometry["sticks"]

        target_pos = self.kit.positions_in("pelvis")[selected.drum_id]
        target_axis = self.kit.axes[selected.drum_id]
        contact = np.asarray(sticks[selected.hand]["contact"], dtype=float)
        delta = contact - target_pos
        plane_error = float(delta @ target_axis)
        tangent = delta - plane_error * target_axis
        radial = float(np.linalg.norm(tangent))
        radius = float(self.kit.radii[selected.drum_id])

        target_left = torch.tensor(
            [selected.drum_id if selected.hand == "left" else -1],
            dtype=torch.long,
            device=self.device,
        )
        target_right = torch.tensor(
            [selected.drum_id if selected.hand == "right" else -1],
            dtype=torch.long,
            device=self.device,
        )
        _pen, sdf = self.sdf(
            q_t,
            target_drum_left=target_left,
            target_drum_right=target_right,
            return_details=True,
        )

        pref_margin = np.minimum(q_np - self.q_min, self.q_max - q_np)
        hard_margin = np.minimum(q_np - self.hard_min, self.hard_max - q_np)
        elbow_z = joints[f"{selected.hand}_elbow_joint"][2]
        wrist_z = sticks[selected.hand]["wrist"][2]
        min_sdf = float(sdf.min_sdf[0].item())
        warnings: list[str] = []
        if float(pref_margin.min()) < 0.05:
            warnings.append("pose is within 0.05 rad of an editor joint limit")
        if abs(plane_error) > 0.02:
            warnings.append("selected contact is more than 20 mm off the drum plane")
        if radial > radius:
            warnings.append("selected contact is outside the drum radius")
        if min_sdf < 0.0:
            warnings.append("pose intersects a non-exempt drum or the robot body")

        return {
            "valid": True,
            "selection": {
                "hand": selected.hand,
                "drum": selected.drum,
                "drum_id": selected.drum_id,
                "contact_sigma": selected.contact_sigma,
            },
            "q": _list(q_np),
            "geometry": {"joints": joints, "sticks": sticks},
            "target": {
                "position": _list(target_pos),
                "axis": _list(target_axis),
                "radius": radius,
            },
            "metrics": {
                "plane_error_m": plane_error,
                "radial_error_m": radial,
                "radial_excess_m": max(0.0, radial - radius),
                "contact_distance_m": float(np.linalg.norm(delta)),
                "min_joint_margin_rad": float(pref_margin.min()),
                "min_hard_joint_margin_rad": float(hard_margin.min()),
                "min_sdf_m": min_sdf,
                "elbow_height_m": float(elbow_z),
                "wrist_height_m": float(wrist_z),
            },
            "diagnostics": {
                "preferred_joint_margin_rad": _list(pref_margin),
                "hard_joint_margin_rad": _list(hard_margin),
                "collision": {
                    "min_sdf_m": min_sdf,
                    "left_min_sdf_m": float(sdf.extras["min_sdf_left"][0].item()),
                    "right_min_sdf_m": float(sdf.extras["min_sdf_right"][0].item()),
                    "drum_penalty": float(sdf.pen_drum[0].item()),
                    "under_penalty": float(sdf.pen_under[0].item()),
                    "self_penalty": float(sdf.pen_self[0].item()),
                },
            },
            "warnings": warnings,
        }

    def default_joint_weight(self, hand: str) -> list[float]:
        if hand not in ("left", "right"):
            raise DomainValidationError("hand must be 'left' or 'right'")
        weights = np.zeros(len(self.joint_names), dtype=float)
        weights[arm_slice(hand, self.joint_names)] = 1.0
        weights[self.joint_names.index("waist_pitch_joint")] = 0.2
        weights[self.joint_names.index("waist_roll_joint")] = 0.2
        # Waist yaw selects the style entry; it is not an imitation target.
        return _list(weights)
