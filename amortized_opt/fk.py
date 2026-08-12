"""Batched, differentiable, sync-free forward kinematics for the G1 upper body.

Vendored from ``control/safety/torch_fk.py`` with the sync-elimination fixes of
``scratch/amortized_perf/fast_fk.py`` folded in:

1. Chain topology (``qidx``) is hoisted to Python ints at construction. The
   original indexed ``q[:, ch["qidx"][k]]`` with a 0-dim CUDA tensor, forcing
   ``aten._local_scalar_dense`` -- a device->host sync once per joint per chain
   (~27 per forward) which serialised every launch and blocked CUDA graphs.
2. The per-joint Rodrigues skew ``K`` and ``K @ K`` are constant (the axis is
   fixed by the URDF), so they are precomputed once instead of being restacked
   from 9 slices on every call.
3. All constants are pre-cast to the working dtype at build time.

Additions over the repo original:

* an **elbow frame per arm** (``left_elbow`` / ``right_elbow``), which the
  amortized optimizer's auxiliary FK head needs (tip + wrist + elbow).

Chain parameters are parsed from the vendored URDF with stdlib XML -- no
Pinocchio import, no ``UNITREE_MODEL_DIR``. Frames are pelvis-rooted; add
``PELVIS_IN_ENV`` for env coordinates.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from .constants import (  # noqa: F401  (re-exported for convenience)
    G1_UPPER_BODY_JOINTS,
    HEAD_SPHERE,
    PELVIS_IN_ENV,
    SHAFT_TIP_CLEAR,
    STICK_LENGTH,
    STICK_MOUNT_POS_LEFT,
    STICK_MOUNT_POS_RIGHT,
    STICK_MOUNT_ROT_LEFT_WXYZ,
    STICK_MOUNT_ROT_RIGHT_WXYZ,
    STICK_RADIUS,
    TIP_RADIUS,
    TORSO_CAPSULE,
    WRIST_RADIUS,
    default_urdf_path,
)

_STICK_MOUNT = {
    "left": (STICK_MOUNT_POS_LEFT, STICK_MOUNT_ROT_LEFT_WXYZ),
    "right": (STICK_MOUNT_POS_RIGHT, STICK_MOUNT_ROT_RIGHT_WXYZ),
}


def _rpy_to_R(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF rpy -> rotation matrix (R = Rz(y) @ Ry(p) @ Rx(r))."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _quat_wxyz_to_R(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _skew_np(a) -> np.ndarray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def stick_mount_transform(hand: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, p) of the stick mount in the wrist-yaw joint frame."""
    pos, quat = _STICK_MOUNT[hand]
    return _quat_wxyz_to_R(quat), np.asarray(pos, float)


def stick_tip_offset(hand: str) -> np.ndarray:
    """Stick tip position in the wrist-yaw frame (mount +Z * L/2)."""
    R, p = stick_mount_transform(hand)
    return p + R @ np.array([0.0, 0.0, STICK_LENGTH / 2.0])


def stick_contact_offset(hand: str, sigma: float) -> np.ndarray:
    """Stick point at arc coordinate ``sigma`` in the wrist-yaw frame.

    ``sigma`` is measured along the stick axis from the mount origin (the
    stick's centre), so ``STICK_LENGTH / 2`` is the tip and smaller values walk
    back toward the hand -- the same convention as
    ``trajopt.robot.contact_position`` and ``strike.StrikeSpec.contact_sigma``.
    """
    R, p = stick_mount_transform(hand)
    return p + R @ np.array([0.0, 0.0, float(sigma)])


def stick_axis_local(hand: str) -> np.ndarray:
    """Handle->tip unit direction of the stick, in the wrist-yaw frame.

    Rotate by the wrist frame's ``R`` to get the world stick axis, which is what
    the strike-elevation band is measured against.
    """
    R, _p = stick_mount_transform(hand)
    return R @ np.array([0.0, 0.0, 1.0])


def stick_shaft_offsets(hand: str, n_samples: int) -> np.ndarray:
    """(n, 3) sample points along the handle-side shaft in the wrist frame."""
    R, p = stick_mount_transform(hand)
    z = np.linspace(
        -STICK_LENGTH / 2.0, STICK_LENGTH / 2.0 - SHAFT_TIP_CLEAR, max(int(n_samples), 2)
    )
    return np.stack([p + R @ np.array([0.0, 0.0, zi]) for zi in z], axis=0)


def _parse_float_list(s: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not s:
        return default
    return tuple(float(v) for v in s.replace(",", " ").split())


class _UrdfJoint:
    __slots__ = ("name", "jtype", "parent", "child", "R", "p", "axis")

    def __init__(self, el: ET.Element):
        self.name = el.get("name", "")
        self.jtype = el.get("type", "fixed")
        self.parent = el.find("parent").get("link")  # type: ignore[union-attr]
        self.child = el.find("child").get("link")  # type: ignore[union-attr]
        origin = el.find("origin")
        xyz = _parse_float_list(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_float_list(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        self.R = _rpy_to_R(rpy)
        self.p = np.asarray(xyz, float)
        ax = el.find("axis")
        axis = _parse_float_list(ax.get("xyz") if ax is not None else None, (1.0, 0.0, 0.0))
        a = np.asarray(axis, float)
        n = float(np.linalg.norm(a))
        self.axis = a / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _chain_to_joint(joints_by_child: dict[str, _UrdfJoint], target: _UrdfJoint) -> list[_UrdfJoint]:
    chain = [target]
    link = target.parent
    while link in joints_by_child:
        j = joints_by_child[link]
        chain.append(j)
        link = j.parent
    chain.reverse()
    return chain


class ChainFK:
    """Batched FK for pelvis -> {wrist, elbow} per arm, plus the torso frame.

    ``forward(q)`` takes ``q`` of shape ``(B, 17)`` in ``G1_UPPER_BODY_JOINTS``
    order (radians, pelvis-rooted) and returns ``{name: (R, p)}`` with shapes
    ``(B, 3, 3)`` / ``(B, 3)``. Differentiable w.r.t. ``q`` and free of
    device->host syncs.
    """

    FRAME_JOINTS: dict[str, str] = {
        "left_wrist": "left_wrist_yaw_joint",
        "right_wrist": "right_wrist_yaw_joint",
        "torso": "waist_pitch_joint",
        "left_elbow": "left_elbow_joint",
        "right_elbow": "right_elbow_joint",
    }

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        *,
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
        frames: dict[str, str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.joint_names = tuple(joint_names)
        self.device = torch.device(device)
        self.dtype = dtype
        path = Path(urdf_path) if urdf_path is not None else default_urdf_path()
        root = ET.parse(str(path)).getroot()
        joints = [_UrdfJoint(el) for el in root.findall("joint")]
        by_name = {j.name: j for j in joints}
        by_child = {j.child: j for j in joints}
        qidx_by_name = {nm: i for i, nm in enumerate(self.joint_names)}
        frame_joints = dict(self.FRAME_JOINTS if frames is None else frames)

        self._chains: dict[str, dict] = {}
        for frame, jname in frame_joints.items():
            if jname not in by_name:
                raise ValueError(f"URDF {path} lacks joint {jname!r}")
            steps_R: list[np.ndarray] = []
            steps_p: list[np.ndarray] = []
            steps_axis: list[np.ndarray] = []
            steps_qidx: list[int] = []
            acc_R, acc_p = np.eye(3), np.zeros(3)
            for j in _chain_to_joint(by_child, by_name[jname]):
                acc_p = acc_p + acc_R @ j.p
                acc_R = acc_R @ j.R
                if j.jtype in ("revolute", "continuous"):
                    if j.name not in qidx_by_name:
                        raise ValueError(
                            f"articulated joint {j.name!r} on the {frame} chain is not "
                            f"in joint_names -- cannot evaluate FK without it"
                        )
                    steps_R.append(acc_R.copy())
                    steps_p.append(acc_p.copy())
                    steps_axis.append(j.axis.copy())
                    steps_qidx.append(qidx_by_name[j.name])
                    acc_R, acc_p = np.eye(3), np.zeros(3)
                elif j.jtype != "fixed":
                    raise ValueError(f"unsupported joint type {j.jtype!r} ({j.name})")
            self._chains[frame] = {
                "qidx_np": np.asarray(steps_qidx, dtype=np.int64),
                "R_np": np.stack(steps_R),
                "p_np": np.stack(steps_p),
                "axis_np": np.stack(steps_axis),
                "R_tail_np": acc_R.copy(),
                "p_tail_np": acc_p.copy(),
            }
        self._build_tensors()

    # -------------------------------------------------------------- build --
    def _build_tensors(self) -> None:
        # Python ints for the loop bound, plus a device-resident index tensor for
        # the gather. Indexing with a python list would rebuild (and H2D-copy) an
        # index tensor on every call -- itself a synchronizing operation.
        for ch in self._chains.values():
            ch["qidx"] = [int(v) for v in ch["qidx_np"].tolist()]
            ch["qidx_t"] = torch.tensor(
                ch["qidx"], device=self.device, dtype=torch.long
            )
        self._cache: dict[torch.dtype, dict[str, dict]] = {}
        self._cast(self.dtype)

    def _cast(self, dtype: torch.dtype) -> dict[str, dict]:
        """Constants for ``dtype`` on ``self.device``, built once and cached."""
        hit = self._cache.get(dtype)
        if hit is not None:
            return hit
        dev = self.device
        out: dict[str, dict] = {}
        for name, ch in self._chains.items():
            K = np.stack([_skew_np(a) for a in ch["axis_np"]], axis=0)
            t = lambda a: torch.tensor(np.asarray(a).tolist(), device=dev, dtype=dtype)
            out[name] = {
                "R": t(ch["R_np"]),
                "p": t(ch["p_np"]),
                "axis": t(ch["axis_np"]),
                "K": t(K),
                "K2": t(K @ K),
                "R_tail": t(ch["R_tail_np"]),
                "p_tail": t(ch["p_tail_np"]),
            }
        self._cache[dtype] = out
        return out

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "ChainFK":
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self._build_tensors()
        return self

    @property
    def frame_names(self) -> tuple[str, ...]:
        return tuple(self._chains.keys())

    # --------------------------------------------------------------- eval --
    def frame(self, q: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(R, p) of one frame, shapes (B, 3, 3) / (B, 3), pelvis frame."""
        ch = self._cast(q.dtype)[name]
        raw = self._chains[name]
        qidx, qidx_t = raw["qidx"], raw["qidx_t"]
        Rk, pk, K, K2 = ch["R"], ch["p"], ch["K"], ch["K2"]
        B = q.shape[0]

        theta = q.index_select(1, qidx_t)  # (B, S) -- one gather, one sin/cos
        s = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        c = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        Rot = eye + s * K.unsqueeze(0) + (1.0 - c) * K2.unsqueeze(0)  # (B,S,3,3)
        M = Rk.unsqueeze(0) @ Rot  # fold the fixed link transform in

        R = eye.expand(B, 3, 3)
        p = q.new_zeros(B, 3)
        for k in range(len(qidx)):
            p = p + (R @ pk[k].view(1, 3, 1)).squeeze(-1)
            R = R @ M[:, k]
        p = p + (R @ ch["p_tail"].view(1, 3, 1)).squeeze(-1)
        R = R @ ch["R_tail"]
        return R, p

    def forward(self, q: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if q.dim() != 2:
            raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
        return {name: self.frame(q, name) for name in self._chains}

    __call__ = forward

    @staticmethod
    def points_world(R: torch.Tensor, p: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Transform local points (P, 3) by batched frames -> (B, P, 3)."""
        return p.unsqueeze(1) + torch.einsum("bij,pj->bpi", R, local)


#: Back-compat alias for the repo's class name.
TorchChainFK = ChainFK
