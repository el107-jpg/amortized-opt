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

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class StickSpec:
    """One hand's stick geometry, so it can vary per SAMPLE.

    ``amortized_opt/constants.py`` is still the single source of the NOMINAL
    stick -- ``g1_scene/g1.py`` authors the PhysX colliders from it and
    ``tests/test_g1_scene.py`` pins the two together at machine epsilon.  This
    is the per-sample perturbation of it, threaded explicitly so nothing can
    mutate the module-level constants underneath the collider.

    ``tilt`` / ``roll`` rotate the mount ABOUT the nominal axes rather than
    replacing them, which is what "randomize along the existing stick angles"
    means: the sampled stick is always a perturbation of the shipped one.

    LENGTH IS HANDLE-FIXED.  Nominally the mount sits at the stick's CENTRE
    (``STICK_TIP_SIGMA = STICK_LENGTH / 2``), so a naive length change would
    move the butt and the tip in opposite directions.  Fixing the handle --
    what swapping a real stick does -- keeps the butt at ``-STICK_LENGTH / 2``
    and lets the tip run out to ``length - STICK_LENGTH / 2``.  That is why
    :attr:`tip_sigma` is not ``length / 2``.
    """

    hand: str
    #: Extra rotation of the mount, radians, about the two axes perpendicular
    #: to the nominal stick direction.
    tilt: float = 0.0
    roll: float = 0.0
    #: TOTAL stick length, metres.  Nominal is ``STICK_LENGTH``.
    length: float = STICK_LENGTH

    @property
    def tip_sigma(self) -> float:
        """Arc coordinate of the tip, handle fixed at ``-STICK_LENGTH / 2``."""
        return float(self.length) - STICK_LENGTH / 2.0

    @property
    def butt_sigma(self) -> float:
        return -STICK_LENGTH / 2.0

    def rotation(self) -> np.ndarray:
        """(3, 3) mount rotation: the nominal one, then the sampled tilt/roll."""
        _pos, quat = _STICK_MOUNT[self.hand]
        R0 = _quat_wxyz_to_R(quat)
        if not self.tilt and not self.roll:
            return R0
        # Rotate about the mount's own x then y; +z is the stick direction, so
        # these two tip the stick without spinning it about its own axis.
        cx, sx = math.cos(self.tilt), math.sin(self.tilt)
        cy, sy = math.cos(self.roll), math.sin(self.roll)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
        return R0 @ Rx @ Ry


@dataclass(frozen=True)
class StickRandomization:
    """Bounds for the per-sample stick perturbation.

    Deliberately shaped like :class:`~amortized_opt.rhythm.KitRandomization`:
    every bound is a half-width, zero means "off", and the defaults are all
    zero so the shipped stick is reproduced exactly until a caller opts in.

    ``tilt`` / ``roll`` are drawn uniformly in ``[-b, b]`` and applied ABOUT the
    nominal mount rotation, so the sampled stick is always a perturbation of the
    shipped one rather than a fresh orientation.  ``length`` is a half-width in
    METRES about ``STICK_LENGTH``, handle-fixed (see :class:`StickSpec`).
    """

    tilt: float = 0.0
    roll: float = 0.0
    length: float = 0.0

    def __post_init__(self) -> None:
        for k in ("tilt", "roll", "length"):
            if float(getattr(self, k)) < 0.0:
                raise ValueError(f"{k} must be >= 0")
        # The real invariant: the shortest drawable stick must keep
        # ``tip_sigma - SHAFT_TIP_CLEAR > butt_sigma``, i.e. total length
        # ``> SHAFT_TIP_CLEAR``, or ``stick_shaft_offsets``'s linspace inverts
        # and the shaft samples walk tip-ward past the clearance band.
        if self.length >= STICK_LENGTH - SHAFT_TIP_CLEAR:
            raise ValueError(
                f"length half-width {self.length} would allow a stick shorter "
                f"than SHAFT_TIP_CLEAR ({SHAFT_TIP_CLEAR}), inverting the shaft "
                f"sample linspace"
            )

    def sample(self, rng, hand: str) -> "StickSpec":
        """One hand's stick for one scenario."""
        u = lambda b: float(rng.uniform(-b, b)) if b > 0.0 else 0.0  # noqa: E731
        return StickSpec(
            hand=hand,
            tilt=u(self.tilt),
            roll=u(self.roll),
            length=STICK_LENGTH + u(self.length),
        )


def _spec(hand: str, stick: "StickSpec | None") -> "StickSpec":
    if stick is None:
        return StickSpec(hand)
    if stick.hand != hand:
        raise ValueError(f"stick spec is for {stick.hand!r}, asked for {hand!r}")
    return stick


def stick_mount_transform(
    hand: str, stick: "StickSpec | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """(R, p) of the stick mount in the wrist-yaw joint frame."""
    sp = _spec(hand, stick)
    pos, _quat = _STICK_MOUNT[hand]
    return sp.rotation(), np.asarray(pos, float)


def stick_tip_offset(hand: str, stick: "StickSpec | None" = None) -> np.ndarray:
    """Stick tip position in the wrist-yaw frame (mount +Z * tip_sigma)."""
    sp = _spec(hand, stick)
    R, p = stick_mount_transform(hand, stick)
    return p + R @ np.array([0.0, 0.0, sp.tip_sigma])


def stick_contact_offset(
    hand: str, sigma: float, stick: "StickSpec | None" = None
) -> np.ndarray:
    """Stick point at arc coordinate ``sigma`` in the wrist-yaw frame.

    ``sigma`` is measured along the stick axis from the mount origin (the
    stick's centre), so ``STICK_LENGTH / 2`` is the tip and smaller values walk
    back toward the hand -- the same convention as
    ``trajopt.robot.contact_position`` and ``strike.StrikeSpec.contact_sigma``.
    """
    R, p = stick_mount_transform(hand, stick)
    return p + R @ np.array([0.0, 0.0, float(sigma)])


def stick_axis_local(hand: str, stick: "StickSpec | None" = None) -> np.ndarray:
    """Handle->tip unit direction of the stick, in the wrist-yaw frame.

    Rotate by the wrist frame's ``R`` to get the world stick axis, which is what
    the strike-elevation band is measured against.
    """
    R, _p = stick_mount_transform(hand, stick)
    return R @ np.array([0.0, 0.0, 1.0])


def stick_shaft_offsets(
    hand: str, n_samples: int, stick: "StickSpec | None" = None
) -> np.ndarray:
    """(n, 3) sample points along the handle-side shaft in the wrist frame."""
    sp = _spec(hand, stick)
    R, p = stick_mount_transform(hand, stick)
    z = np.linspace(
        sp.butt_sigma, sp.tip_sigma - SHAFT_TIP_CLEAR, max(int(n_samples), 2)
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


def _autograd_tensor(t: torch.Tensor) -> torch.Tensor:
    """Clone inference-mode buffers so they can be saved for backward.

    ``ChainFK`` is often first constructed inside ``play_isaac``'s
    ``inference_mode`` reset, which makes ``qidx_t`` / Rodrigues constants
    inference tensors. ``tip_jacobian`` exits that mode and graphs ``q``, then
    ``index_select`` still tries to save those constants.
    """
    return t.clone() if t.is_inference() else t


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
        Rk, pk, K, K2 = (_autograd_tensor(t) for t in (Rk, pk, K, K2))
        B = q.shape[0]

        theta = q.index_select(1, _autograd_tensor(qidx_t))  # (B, S) -- one gather, one sin/cos
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
        p = p + (R @ _autograd_tensor(ch["p_tail"]).view(1, 3, 1)).squeeze(-1)
        R = R @ _autograd_tensor(ch["R_tail"])
        return R, p

    def forward(self, q: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if q.dim() != 2:
            raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
        return {name: self.frame(q, name) for name in self._chains}

    __call__ = forward

    @staticmethod
    def points_world(R: torch.Tensor, p: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Transform local points by batched frames -> ``(B, P, 3)``.

        ``local`` is ``(P, 3)`` when every sample carries the SAME stick -- the
        shipped case, one contraction -- or ``(B, P, 3)`` when the stick is
        randomized per sample, which is the only shape change per-sample stick
        geometry needs anywhere in the hot path.
        """
        if local.dim() == 3:
            return p.unsqueeze(1) + torch.einsum("bij,bpj->bpi", R, local)
        return p.unsqueeze(1) + torch.einsum("bij,pj->bpi", R, local)


#: Back-compat alias for the repo's class name.
TorchChainFK = ChainFK
