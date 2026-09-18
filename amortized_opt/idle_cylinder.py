"""Torso-frame idle safety cylinder (full replace of the joint-home park).

An idle arm parks its stick tip inside a vertical cylinder **centered on the
torso / waist axis**, expressed in the torso frame.  XY at the origin means
waist yaw only spins the cylinder about itself, so the occupied world volume
is yaw-invariant.  Radius is the largest XY disk that, extruded infinitely along torso-up, does
not overlap any drum whose head is within 30 cm of the cylinder's lower face
(or above it).  Heads farther below cannot meet a parked tip.  Shared arm poses live in
:class:`IdleCylinderLibrary`; per-scenario geometry comes from
:func:`derive_cylinder`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .constants import (
    G1_UPPER_BODY_JOINTS,
    default_layout_path,
    ready_joint_pos,
)
from .fk import ChainFK
from .ik import arm_joint_indices, tip_jacobian, tip_position
from .limits import urdf_position_limits
from .rhythm import DrumKit
from .warmstart import arm_slice

__all__ = [
    "IdleCylinder",
    "IdleCylinderLibrary",
    "default_idle_cylinder_path",
    "derive_cylinder",
    "load_idle_cylinder_library",
    "nearest_arm_pose",
    "nudge_into_cylinder",
    "outside",
    "resolve_idle_cylinder_library",
    "tip_in_torso",
    "tip_inside",
    "tip_outside",
]

_HANDS = ("left", "right")
_N_SHOULDER = 3
_DEFAULT_HALF_HEIGHT = 0.12
_DEFAULT_Z_MARGIN = 0.08
_DEFAULT_N_YAW = 129
#: Air gap so the infinite-vertical XY disk is strictly inside every drum rim.
_XY_CLEARANCE = 1e-3
#: Drums this far below the cylinder's lower face do not constrain radius --
#: a tip parked in the cylinder cannot reach them.
_Z_BELOW_IGNORE = 0.30


def default_idle_cylinder_path() -> Path:
    """``assets/idle_cylinder_poses.json`` next to the drum layout."""
    return Path(default_layout_path()).with_name("idle_cylinder_poses.json")


# ---------------------------------------------------------------- geometry --
def tip_in_torso(
    fk: ChainFK,
    q: Tensor,
    hand: str,
    *,
    stick=None,
) -> Tensor:
    """``(B, 3)`` stick tip in the torso frame: ``R_torso^T (p_tip - p_torso)``."""
    tip = tip_position(fk, q, hand, stick=stick)  # pelvis
    Rt, pt = fk.frame(q, "torso")
    return torch.einsum("bij,bj->bi", Rt.transpose(1, 2), tip - pt)


@dataclass
class IdleCylinder:
    """Finite vertical cylinder in the torso frame (tip must stay inside)."""

    center: np.ndarray  # (3,)
    axis: np.ndarray  # (3,) unit
    radius: float
    half_height: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, float).reshape(3)
        ax = np.asarray(self.axis, float).reshape(3)
        n = float(np.linalg.norm(ax))
        if n < 1e-12:
            raise ValueError("IdleCylinder.axis must be non-zero")
        self.axis = ax / n
        self.radius = float(self.radius)
        self.half_height = float(self.half_height)

    def pack(self) -> np.ndarray:
        """``(8,)`` = center(3) + axis(3) + radius + half_height."""
        return np.concatenate(
            [self.center, self.axis, [self.radius, self.half_height]]
        ).astype(float)

    @classmethod
    def unpack(cls, v) -> "IdleCylinder":
        a = np.asarray(v, float).reshape(8)
        return cls(center=a[:3], axis=a[3:6], radius=float(a[6]), half_height=float(a[7]))

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "axis": self.axis.tolist(),
            "radius": float(self.radius),
            "half_height": float(self.half_height),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IdleCylinder":
        return cls(
            center=d["center"],
            axis=d["axis"],
            radius=float(d["radius"]),
            half_height=float(d["half_height"]),
        )


def outside(cyl: IdleCylinder, tip) -> np.ndarray:
    """Apex-style outside residual: ``hypot(s_r, s_z)``.

    ``s_r = max(rho - R, 0)``, ``s_z = max(|z| - hh, 0)`` along the cylinder axis.
    ``tip`` is ``(..., 3)`` in the torso frame; returns ``(...)``.
    """
    t = np.asarray(tip, float)
    d = t - cyl.center
    z = np.sum(d * cyl.axis, axis=-1)
    radial = d - z[..., None] * cyl.axis
    rho = np.linalg.norm(radial, axis=-1)
    s_r = np.maximum(rho - cyl.radius, 0.0)
    s_z = np.maximum(np.abs(z) - cyl.half_height, 0.0)
    return np.hypot(s_r, s_z)


def tip_outside(cyl: IdleCylinder, tip, *, tol: float = 1e-4) -> bool | np.ndarray:
    """True when the tip is strictly outside (``outside > tol``)."""
    return outside(cyl, tip) > float(tol)


def tip_inside(cyl: IdleCylinder, tip, *, tol: float = 1e-4) -> bool | np.ndarray:
    """True when the tip is inside or on the boundary (``outside <= tol``)."""
    return outside(cyl, tip) <= float(tol)


def derive_cylinder(
    kit: DrumKit,
    fk: ChainFK,
    q_ref: np.ndarray | None = None,
    *,
    n_yaw: int = _DEFAULT_N_YAW,
    z_margin: float = _DEFAULT_Z_MARGIN,
    radius: float | None = None,
    half_height: float = _DEFAULT_HALF_HEIGHT,
    xy_clearance: float = _XY_CLEARANCE,
    z_below_ignore: float = _Z_BELOW_IGNORE,
    joint_names=G1_UPPER_BODY_JOINTS,
    dtype: torch.dtype = torch.float64,
) -> IdleCylinder:
    """Place the idle cylinder on the torso / waist axis in the torso frame.

    Waist yaw is swept (``n_yaw`` in ~65–129) with lean at zero so every drum
    head is seen from every torso heading.  ``center_xy = (0, 0)`` -- the torso
    origin, which sits on the yaw axis -- so rotating the waist does not move
    the cylinder in the world.  ``z_lo = max(head_z_torso) + z_margin``; axis is
    torso-up ``[0, 0, 1]``.

    Radius is the largest XY disk that, extruded infinitely along torso-up,
    stays strictly inside every drum whose head is within ``z_below_ignore``
    of the cylinder's lower face (or above it).  Heads farther below cannot
    meet a tip parked in the cylinder and do not pinch the radius.  Pass
    ``radius`` to cap below that fit.
    """
    names = tuple(joint_names)
    n_yaw = int(n_yaw)
    if n_yaw < 1:
        raise ValueError("n_yaw must be >= 1")
    q0 = (
        ready_joint_pos(names)
        if q_ref is None
        else np.asarray(q_ref, float).reshape(len(names)).copy()
    )
    lo, hi = urdf_position_limits(names, posture_limits=True)
    yaw_i = names.index("waist_yaw_joint")
    yaws = (
        np.linspace(float(lo[yaw_i]), float(hi[yaw_i]), n_yaw)
        if n_yaw > 1
        else np.array([float(q0[yaw_i])])
    )

    heads = np.asarray(kit.positions_in("pelvis"), float)  # (D, 3)
    q_batch = np.repeat(q0[None, :], len(yaws), axis=0)
    q_batch[:, yaw_i] = yaws
    for nm in ("waist_pitch_joint", "waist_roll_joint"):
        q_batch[:, names.index(nm)] = 0.0

    qt = torch.as_tensor(q_batch, device=fk.device, dtype=dtype)
    with torch.no_grad():
        frames = fk.forward(qt)
        Rt, pt = frames["torso"]
        hp = torch.as_tensor(heads, device=fk.device, dtype=dtype)
        rel = hp.unsqueeze(0) - pt.unsqueeze(1)
        heads_t = torch.einsum("nij,ndj->ndi", Rt.transpose(1, 2), rel)
        z_max = float(heads_t[..., 2].amax().item())
        z_lo = z_max + float(z_margin)
        rho = torch.linalg.norm(heads_t[..., :2], dim=-1)
        rad = torch.as_tensor(kit.radii, device=fk.device, dtype=dtype).reshape(1, -1)
        # Per drum: highest torso-z over the yaw sweep.  Far-below heads
        # (snare / floor tom on the nominal kit) are unreachable from the park.
        z_hi = heads_t[..., 2].amax(dim=0)
        keep = (z_lo - z_hi) <= float(z_below_ignore)
        if bool(keep.any()):
            min_rim = float((rho - rad)[:, keep].amin().item())
        else:
            min_rim = float((rho - rad).amin().item())

    hh = float(half_height)
    # On the yaw axis.  A forward XY offset (the old ready-tip mean) would
    # orbit the kit as the waist turns; (0, 0) does not.
    center = np.array([0.0, 0.0, z_lo + hh], dtype=float)
    max_r = max(0.0, min_rim - float(xy_clearance))
    if radius is None:
        r = max_r
    else:
        r = min(float(radius), max_r)
    return IdleCylinder(
        center=center,
        axis=np.array([0.0, 0.0, 1.0]),
        radius=r,
        half_height=hh,
    )


# -------------------------------------------------------------- library API --
def _shoulder_weights(n_arm: int, shoulder_weight: float) -> np.ndarray:
    w = np.ones(n_arm, dtype=float)
    w[: min(_N_SHOULDER, n_arm)] = float(shoulder_weight)
    return w


def nearest_arm_pose(
    library_q: np.ndarray,
    q_arm: np.ndarray,
    *,
    shoulder_weight: float = 10.0,
) -> tuple[int, np.ndarray]:
    """Index + pose of the library entry nearest ``q_arm`` (shoulder-hard).

    ``library_q`` is ``(K, 7)``; ``q_arm`` is ``(7,)``.
    """
    lib = np.asarray(library_q, float)
    q = np.asarray(q_arm, float).reshape(-1)
    if lib.ndim != 2 or lib.shape[-1] != q.shape[0]:
        raise ValueError(f"library_q {lib.shape} vs q_arm {q.shape}")
    w = _shoulder_weights(q.shape[0], shoulder_weight)
    i = int(np.argmin((((lib - q) ** 2) * w).sum(-1)))
    return i, lib[i].copy()


def nudge_into_cylinder(
    fk: ChainFK,
    q: np.ndarray | Tensor,
    hand: str,
    cyl: IdleCylinder,
    *,
    steps: int = 8,
    damping: float = 0.05,
    shoulder_weight: float = 10.0,
    joint_names=G1_UPPER_BODY_JOINTS,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-4,
) -> np.ndarray:
    """Arm-only damped LS that pulls the tip into ``cyl`` (torso frame).

    Shoulder joints are inflated in the least-squares metric so distal DoFs
    absorb most of the motion.  Waist is frozen.
    """
    names = tuple(joint_names)
    cols = arm_joint_indices(hand, names)
    q_np = np.asarray(
        q.detach().cpu().numpy() if isinstance(q, Tensor) else q, float
    )
    squeeze = q_np.ndim == 1
    if squeeze:
        q_np = q_np.reshape(1, -1)
    idx = torch.as_tensor(cols, device=fk.device, dtype=torch.long)
    m = int(idx.numel())
    out = torch.as_tensor(q_np, device=fk.device, dtype=fk.dtype)
    dtype = out.dtype
    sqrt_w = torch.as_tensor(
        np.sqrt(_shoulder_weights(m, shoulder_weight)),
        device=fk.device,
        dtype=dtype,
    ).view(1, m)
    inv_sw = 1.0 / sqrt_w
    lam2 = float(damping) ** 2
    eye = torch.eye(3, device=fk.device, dtype=dtype)
    lo = hi = None
    if q_bounds is not None:
        lo = torch.as_tensor(q_bounds[0], device=fk.device, dtype=dtype).index_select(
            0, idx
        )
        hi = torch.as_tensor(q_bounds[1], device=fk.device, dtype=dtype).index_select(
            0, idx
        )
    c = torch.as_tensor(cyl.center, device=fk.device, dtype=dtype)
    ax = torch.as_tensor(cyl.axis, device=fk.device, dtype=dtype)
    R = float(cyl.radius)
    hh = float(cyl.half_height)

    for _ in range(int(steps)):
        tip = tip_in_torso(fk, out, hand)
        d = tip - c
        z = (d * ax).sum(-1)
        radial = d - z.unsqueeze(-1) * ax
        rho = torch.linalg.norm(radial, dim=-1).clamp_min(1e-12)
        s_r = (rho - R).clamp_min(0.0)
        s_z = (z.abs() - hh).clamp_min(0.0)
        if float(torch.hypot(s_r, s_z).max()) <= float(tol):
            break
        u = radial / rho.unsqueeze(-1)
        delta = -(s_r.unsqueeze(-1) * u) - (
            s_z * torch.sign(z).clamp(min=-1.0, max=1.0)
        ).unsqueeze(-1) * ax
        Rt, _pt = fk.frame(out, "torso")
        J_pelvis = tip_jacobian(fk, out, hand).index_select(2, idx)
        J = torch.einsum("bij,bjk->bik", Rt.transpose(1, 2), J_pelvis)
        # y = sqrt(W) dq; Js = J inv(sqrt(W)); minimize ||y|| s.t. Js y ≈ delta
        Js = J * inv_sw.unsqueeze(1)
        y = (
            Js.transpose(1, 2)
            @ torch.linalg.solve(Js @ Js.transpose(1, 2) + lam2 * eye, delta.unsqueeze(-1))
        ).squeeze(-1)
        dq = y * inv_sw
        upd = out.index_select(1, idx) + dq
        if lo is not None:
            upd = torch.clamp(upd, lo, hi)
        out = out.index_copy(1, idx, upd)

    result = out.detach().cpu().numpy()
    return result[0] if squeeze else result


@dataclass
class IdleCylinderLibrary:
    """Shared idle arm poses ``q_arm (2, K, 7)`` plus the nominal cylinder they clear."""

    q_arm: np.ndarray  # (2, K, 7)
    joint_names: tuple[str, ...]
    cylinder: IdleCylinder
    tip_torso: np.ndarray = field(default_factory=lambda: np.zeros((2, 0, 3)))

    def __post_init__(self) -> None:
        self.q_arm = np.asarray(self.q_arm, float)
        if self.q_arm.ndim != 3 or self.q_arm.shape[0] != 2:
            raise ValueError(f"q_arm must be (2, K, 7), got {self.q_arm.shape}")
        self.joint_names = tuple(self.joint_names)
        if self.tip_torso.size == 0:
            self.tip_torso = np.zeros((2, self.K, 3))
        else:
            self.tip_torso = np.asarray(self.tip_torso, float)

    @property
    def K(self) -> int:
        return int(self.q_arm.shape[1])

    @property
    def n_arm(self) -> int:
        return int(self.q_arm.shape[-1])

    def arm_poses(self, hand: str) -> np.ndarray:
        """``(K, 7)`` library for ``hand``."""
        return np.asarray(self.q_arm[_HANDS.index(hand)], float)

    def nearest_arm_pose(
        self,
        hand: str,
        q_arm: np.ndarray,
        *,
        shoulder_weight: float = 10.0,
    ) -> tuple[int, np.ndarray]:
        return nearest_arm_pose(
            self.arm_poses(hand), q_arm, shoulder_weight=shoulder_weight
        )

    def nudge_into_cylinder(
        self,
        fk: ChainFK,
        q: np.ndarray | Tensor,
        hand: str,
        cyl: IdleCylinder | None = None,
        **kwargs,
    ) -> np.ndarray:
        return nudge_into_cylinder(
            fk, q, hand, cyl if cyl is not None else self.cylinder, **kwargs
        )

    def full_pose(
        self,
        hand: str,
        k: int,
        *,
        yaw: float = 0.0,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(n_dof,)`` with ``hand`` set to library pose ``k`` and waist at ``yaw``."""
        names = self.joint_names
        q = ready_joint_pos(names) if base is None else np.asarray(base, float).copy()
        q[names.index("waist_yaw_joint")] = float(yaw)
        q[arm_slice(hand, names)] = self.q_arm[_HANDS.index(hand), int(k)]
        return q

    def to_dict(self) -> dict:
        return {
            "joint_names": list(self.joint_names),
            "q_arm": np.asarray(self.q_arm, float).tolist(),
            "tip_torso": np.asarray(self.tip_torso, float).tolist(),
            "cylinder": self.cylinder.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IdleCylinderLibrary":
        return cls(
            q_arm=np.asarray(d["q_arm"], float),
            joint_names=tuple(d["joint_names"]),
            cylinder=IdleCylinder.from_dict(d["cylinder"]),
            tip_torso=np.asarray(d.get("tip_torso", []), float),
        )

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path is not None else default_idle_cylinder_path()
        p.write_text(json.dumps(self.to_dict()))
        return p


def load_idle_cylinder_library(
    path: str | Path | None = None, *, joint_names=G1_UPPER_BODY_JOINTS
) -> IdleCylinderLibrary:
    """Read the cached pose library, checking it was built for ``joint_names``."""
    p = Path(path) if path is not None else default_idle_cylinder_path()
    if not p.exists():
        raise FileNotFoundError(
            f"idle-cylinder library missing at {p} -- build it with "
            "`python scripts/build_idle_cylinder_library.py`"
        )
    lib = IdleCylinderLibrary.from_dict(json.loads(p.read_text()))
    if tuple(joint_names) != tuple(lib.joint_names):
        raise ValueError(
            f"idle-cylinder library was built for {lib.joint_names}, "
            f"not {tuple(joint_names)}"
        )
    return lib


def resolve_idle_cylinder_library(
    spec, *, joint_names=G1_UPPER_BODY_JOINTS
) -> IdleCylinderLibrary | None:
    """``None`` / ``"default"`` / a path -> an :class:`IdleCylinderLibrary` or ``None``.

    ``"default"`` is permissive (missing asset -> ``None`` + warning). An explicit
    path is strict.
    """
    if spec is None:
        return None
    if str(spec) == "default":
        p = default_idle_cylinder_path()
        if not p.exists():
            print(
                f"[idle-cylinder] {p} not built; idle-cylinder terms stay off "
                "(run scripts/build_idle_cylinder_library.py)"
            )
            return None
        return load_idle_cylinder_library(p, joint_names=joint_names)
    return load_idle_cylinder_library(spec, joint_names=joint_names)
