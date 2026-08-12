"""Batched Featherstone RNEA -- sync-free, autograd-safe, armature-correct.

Vendored from ``control/trajopt/batch_rnea.py`` with the
``scratch/amortized_perf/fast_rnea.py`` refactor applied:

* **Topology as Python ints.** The original read ``int(self.body_to_q[i].item())``
  / ``body_to_v`` / ``parent`` inside the loop -- ~60-70 device->host syncs per
  call, which serialised launches and blocked CUDA-graph capture.
* **List rebind instead of in-place index assignment.** ``v[:, i] = ...`` /
  ``f[:, i] = ...`` / ``tau_pin[:, iv] = ...`` made backward raise
  ``RuntimeError: ... modified by an inplace operation``. Per-link values now
  live in Python lists whose slots are *rebound* (functional), stacked at the
  end. The parent-force accumulation ``f[par] = f[par] + ...`` is a rebind too.
* **Armature.** ``tau += diag(isaac_armature) @ qdd``. Pinocchio's ``M`` is
  link-inertia only, so rotor inertia was missing exactly where it matters --
  several N*m at strike accelerations against a 5 N*m wrist limit. Pass
  ``armature=None`` to reproduce the link-inertia-only result (that is what the
  Pinocchio conformance test compares against).

The spatial layout is Pinocchio >= 3 ``[linear; angular]`` internally;
``from_pinocchio`` detects a Pinocchio 2 ``[angular; linear]`` model and
converts gravity + inertias at build time.

Pinocchio is a **build-time** dependency only: once constructed, or once
reloaded from a saved state dict, evaluation is pure torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import G1_UPPER_BODY_JOINTS


# ------------------------------------------------------------ spatial algebra
def _skew(v: torch.Tensor) -> torch.Tensor:
    """v: [B, 3] -> [B, 3, 3]."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((o, -z, y), dim=-1),
            torch.stack((z, o, -x), dim=-1),
            torch.stack((-y, x, o), dim=-1),
        ),
        dim=-2,
    )


def _action_inverse(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``SE3(R,p).actionInverse`` for Motion [lin; ang], batched [B,6,6]."""
    Rt = R.transpose(-1, -2)
    zero = torch.zeros_like(Rt)
    top = torch.cat([Rt, -torch.bmm(Rt, _skew(p))], dim=-1)
    bot = torch.cat([zero, Rt], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _force_parent_from_child_via_Am(Am: torch.Tensor) -> torch.Tensor:
    """Child->parent Force map: ``inv(action).T`` (== ``Am.transpose`` for SE3)."""
    return Am.transpose(-1, -2)


def _motion_cross_la(v: torch.Tensor) -> torch.Tensor:
    """vx for Motion.vector = [lin; ang]."""
    lin, w = v[..., :3], v[..., 3:]
    sw = _skew(w)
    zero = torch.zeros_like(sw)
    top = torch.cat([sw, _skew(lin)], dim=-1)
    bot = torch.cat([zero, sw], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _force_cross_la(v: torch.Tensor) -> torch.Tensor:
    """Force cross for [lin; ang]."""
    lin, w = v[..., :3], v[..., 3:]
    sw = _skew(w)
    zero = torch.zeros_like(sw)
    top = torch.cat([sw, zero], dim=-1)
    bot = torch.cat([_skew(lin), sw], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _rodrigues(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """axis [3], theta [B] -> R [B, 3, 3]."""
    B = theta.shape[0]
    c, s = torch.cos(theta), torch.sin(theta)
    K = _skew(axis.view(1, 3).expand(B, 3))
    K2 = torch.bmm(K, K)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype).view(1, 3, 3).expand(B, 3, 3)
    return I + s.view(B, 1, 1) * K + (1.0 - c).view(B, 1, 1) * K2


# ------------------------------------------------------------ pinocchio glue
def _swap_spatial_halves_6x6(M: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 6), dtype=float)
    out[:3, :3] = M[3:, 3:]
    out[3:, 3:] = M[:3, :3]
    out[:3, 3:] = M[3:, :3]
    out[3:, :3] = M[:3, 3:]
    return out


def _detect_pinocchio_spatial_layout(model) -> str:
    """``"LA"`` ([lin;ang], Pinocchio>=3) or ``"AL"`` (Pinocchio 2)."""
    gv = np.asarray(model.gravity.vector, dtype=float).ravel()
    lin_n = float(np.linalg.norm(gv[:3]))
    ang_n = float(np.linalg.norm(gv[3:]))
    if lin_n > 1.0 and ang_n < 1.0:
        return "LA"
    if ang_n > 1.0 and lin_n < 1.0:
        return "AL"
    for i in range(1, model.njoints):
        mass = float(model.inertias[i].mass)
        if mass < 1e-8:
            continue
        M = np.asarray(model.inertias[i].matrix(), dtype=float)
        if abs(M[0, 0] - mass) < 1e-3 * mass:
            return "LA"
        if abs(M[3, 3] - mass) < 1e-3 * mass:
            return "AL"
        break
    return "LA"


def _joint_axis(jmodel) -> tuple[float, float, float]:
    name = jmodel.shortname()
    if "RX" in name:
        return (1.0, 0.0, 0.0)
    if "RY" in name:
        return (0.0, 1.0, 0.0)
    if "RZ" in name:
        return (0.0, 0.0, 1.0)
    a = getattr(jmodel, "axis", None)
    if a is None:
        raise ValueError(f"unsupported joint: {name}")
    ax = (float(a[0]), float(a[1]), float(a[2]))
    n = (ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) ** 0.5
    return (ax[0] / n, ax[1] / n, ax[2] / n)


def _pinocchio_parent_ids(model) -> np.ndarray:
    n = int(model.njoints)
    try:
        return np.asarray([int(model.parents[i]) for i in range(n)], dtype=np.int64)
    except (TypeError, RuntimeError, ValueError) as exc:
        if "unsigned long" not in str(exc):
            raise
    raise RuntimeError(
        "this Pinocchio build cannot expose model.parents; the repo's FK-based "
        "fallback (batch_rnea._infer_parent_ids_fk) is not vendored here"
    )


# ------------------------------------------------------------------- module --
@dataclass
class BatchedRNEA:
    """Featherstone RNEA for a fixed-base revolute tree, batched over envs.

    Assumptions (true for the G1 upper-body reduced model): every actuated joint
    is 1-DoF revolute, joint motion has zero translation, ``idx_q == idx_v``.
    """

    axis: torch.Tensor  # [n+1, 3]
    R0: torch.Tensor  # [n+1, 3, 3]
    p0: torch.Tensor  # [n+1, 3]
    I: torch.Tensor  # [n+1, 6, 6] spatial inertia, LA layout
    gravity: torch.Tensor  # [6] Motion.vector of model.gravity (LA)
    parent: list[int]
    body_to_q: list[int]
    body_to_v: list[int]
    opt2pin: list[int]
    n_bodies: int
    nq_pin: int
    nv_pin: int
    armature: torch.Tensor | None = None  # [n_opt], reflected rotor inertia

    BUILD_ID = "amortized-rnea-v1-listrebind-armature"

    def __post_init__(self) -> None:
        # Cached index tensor: building one per call from the python list would
        # add a host->device copy to every forward.
        self._opt2pin_t = torch.as_tensor(
            self.opt2pin, dtype=torch.long, device=self.gravity.device
        )

    # ---------------------------------------------------------------- build --
    @classmethod
    def from_pinocchio(
        cls,
        model,
        opt2pin: np.ndarray,
        *,
        armature: np.ndarray | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "BatchedRNEA":
        layout = _detect_pinocchio_spatial_layout(model)
        n = model.njoints - 1
        parent_ids = _pinocchio_parent_ids(model)
        for i in range(1, model.njoints):
            p = int(parent_ids[i])
            if p < 0 or p >= i:
                raise RuntimeError(
                    f"Pinocchio parent invariant broken for joint {i} "
                    f"({model.names[i]!r}): parent={p}"
                )
        axis = np.zeros((n + 1, 3))
        R0 = np.zeros((n + 1, 3, 3))
        p0 = np.zeros((n + 1, 3))
        I = np.zeros((n + 1, 6, 6))
        body_to_q = [-1] * (n + 1)
        body_to_v = [-1] * (n + 1)
        R0[0] = np.eye(3)
        for i in range(1, model.njoints):
            jmodel = model.joints[i]
            if jmodel.nq == 0 and jmodel.nv == 0:
                axis[i] = (0.0, 0.0, 1.0)
            elif jmodel.nq == 1 and jmodel.nv == 1:
                axis[i] = _joint_axis(jmodel)
                body_to_q[i] = int(jmodel.idx_q)
                body_to_v[i] = int(jmodel.idx_v)
            else:
                raise ValueError(
                    f"only 1-DoF revolute or fixed joints supported; got "
                    f"{model.names[i]!r} nq={jmodel.nq} nv={jmodel.nv}"
                )
            X = model.jointPlacements[i]
            R0[i] = np.asarray(X.rotation, dtype=float)
            p0[i] = np.asarray(X.translation, dtype=float).ravel()
            Ii = np.asarray(model.inertias[i].matrix(), dtype=float)
            I[i] = _swap_spatial_halves_6x6(Ii) if layout == "AL" else Ii

        gv = np.asarray(model.gravity.vector, dtype=float).ravel().copy()
        if layout == "AL":
            gv = np.array([gv[3], gv[4], gv[5], gv[0], gv[1], gv[2]], dtype=float)

        dev = torch.device(device)
        t = lambda a: torch.tensor(np.asarray(a, dtype=float).tolist(), device=dev, dtype=dtype)
        arm_t = None if armature is None else t(np.asarray(armature, dtype=float).ravel())
        out = cls(
            axis=t(axis),
            R0=t(R0),
            p0=t(p0),
            I=t(I),
            gravity=t(gv),
            parent=[int(v) for v in parent_ids.tolist()],
            body_to_q=body_to_q,
            body_to_v=body_to_v,
            opt2pin=[int(v) for v in np.asarray(opt2pin).ravel().tolist()],
            n_bodies=int(n),
            nq_pin=int(model.nq),
            nv_pin=int(model.nv),
            armature=arm_t,
        )
        out.spatial_layout = layout  # type: ignore[attr-defined]
        return out

    @classmethod
    def from_urdf(
        cls,
        urdf_path=None,
        *,
        joint_names: tuple[str, ...] = G1_UPPER_BODY_JOINTS,
        with_armature: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "BatchedRNEA":
        """Build from the vendored URDF (needs Pinocchio at construction only)."""
        from .limits import build_reduced_model, isaac_armature

        model, opt2pin = build_reduced_model(joint_names, urdf_path=urdf_path)
        arm = isaac_armature(joint_names) if with_armature else None
        return cls.from_pinocchio(model, opt2pin, armature=arm, device=device, dtype=dtype)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "BatchedRNEA":
        dev = torch.device(device)
        cast = lambda t: (
            None
            if t is None
            else (t.to(device=dev, dtype=dtype) if dtype is not None else t.to(dev))
        )
        out = BatchedRNEA(
            axis=cast(self.axis),
            R0=cast(self.R0),
            p0=cast(self.p0),
            I=cast(self.I),
            gravity=cast(self.gravity),
            parent=list(self.parent),
            body_to_q=list(self.body_to_q),
            body_to_v=list(self.body_to_v),
            opt2pin=list(self.opt2pin),
            n_bodies=self.n_bodies,
            nq_pin=self.nq_pin,
            nv_pin=self.nv_pin,
            armature=cast(self.armature),
        )
        if hasattr(self, "spatial_layout"):
            out.spatial_layout = self.spatial_layout  # type: ignore[attr-defined]
        return out

    def without_armature(self) -> "BatchedRNEA":
        """A view with the armature term disabled (link inertia only)."""
        out = self.to(self.gravity.device)
        out.armature = None
        return out

    # ----------------------------------------------------------------- eval --
    def _expand(self, x_opt: torch.Tensor, width: int) -> torch.Tensor:
        B = x_opt.shape[0]
        out = x_opt.new_zeros(B, width)
        # out-of-place index_copy on a fresh zero tensor: autograd-safe
        return out.index_copy(1, self._opt2pin_t, x_opt)

    def rnea(
        self, q_opt: torch.Tensor, qd_opt: torch.Tensor, qdd_opt: torch.Tensor
    ) -> torch.Tensor:
        """tau = RNEA(q, qd, qdd) + diag(armature) qdd. Tensors [B, n_opt] or [n_opt]."""
        squeeze = q_opt.ndim == 1
        if squeeze:
            q_opt, qd_opt, qdd_opt = (t.unsqueeze(0) for t in (q_opt, qd_opt, qdd_opt))

        B = q_opt.shape[0]
        q = self._expand(q_opt, self.nq_pin)
        qd = self._expand(qd_opt, self.nv_pin)
        qdd = self._expand(qdd_opt, self.nv_pin)
        n = self.n_bodies
        zero = q.new_zeros(B)

        v_list: list[torch.Tensor | None] = [None] * (n + 1)
        a_list: list[torch.Tensor | None] = [None] * (n + 1)
        f_list: list[torch.Tensor] = [q.new_zeros(B, 6) for _ in range(n + 1)]
        Af_list: list[torch.Tensor | None] = [None] * (n + 1)
        S_list: list[torch.Tensor | None] = [None] * (n + 1)

        v_list[0] = q.new_zeros(B, 6)
        a_list[0] = (-self.gravity.to(q.dtype).view(1, 6)).expand(B, 6)

        for i in range(1, n + 1):
            iq, iv = self.body_to_q[i], self.body_to_v[i]
            qi = q[:, iq] if iq >= 0 else zero
            qdi = qd[:, iv] if iv >= 0 else zero
            qddi = qdd[:, iv] if iv >= 0 else zero

            axis_i = self.axis[i].to(q.dtype)
            Rj = _rodrigues(axis_i, qi)
            R = torch.bmm(self.R0[i].to(q.dtype).expand(B, 3, 3), Rj)
            Am = _action_inverse(R, self.p0[i].to(q.dtype).expand(B, 3))
            Af_list[i] = _force_parent_from_child_via_Am(Am)

            # constant angular motion subspace, no zeros+slice-assign
            S = torch.cat([q.new_zeros(B, 3), axis_i.view(1, 3).expand(B, 3)], dim=-1)
            S_list[i] = S
            vj = S * qdi.unsqueeze(-1)
            aj = S * qddi.unsqueeze(-1)

            par = self.parent[i]
            vi = torch.bmm(Am, v_list[par].unsqueeze(-1)).squeeze(-1) + vj
            cor = torch.bmm(_motion_cross_la(vi), vj.unsqueeze(-1)).squeeze(-1)
            ai = torch.bmm(Am, a_list[par].unsqueeze(-1)).squeeze(-1) + aj + cor
            v_list[i], a_list[i] = vi, ai

        tau_cols: list[torch.Tensor | None] = [None] * self.nv_pin
        for i in range(n, 0, -1):
            Ii = self.I[i].to(q.dtype).expand(B, 6, 6)
            vi, ai = v_list[i], a_list[i]
            Iv = torch.bmm(Ii, vi.unsqueeze(-1)).squeeze(-1)
            Ia = torch.bmm(Ii, ai.unsqueeze(-1)).squeeze(-1)
            f_i = f_list[i] + Ia + torch.bmm(
                _force_cross_la(vi), Iv.unsqueeze(-1)
            ).squeeze(-1)
            f_list[i] = f_i

            iv = self.body_to_v[i]
            if iv >= 0:
                tau_cols[iv] = (S_list[i] * f_i).sum(dim=-1)

            par = self.parent[i]
            f_list[par] = f_list[par] + torch.bmm(
                Af_list[i], f_i.unsqueeze(-1)
            ).squeeze(-1)

        tau_pin = torch.stack([c if c is not None else zero for c in tau_cols], dim=-1)
        tau_opt = tau_pin.index_select(1, self._opt2pin_t)

        if self.armature is not None:
            tau_opt = tau_opt + self.armature.to(q.dtype).view(1, -1) * qdd_opt

        return tau_opt.squeeze(0) if squeeze else tau_opt

    def mass_matrix_torque(self, q_opt: torch.Tensor, qdd_opt: torch.Tensor) -> torch.Tensor:
        """``(M(q) + diag(armature)) qdd`` -- the purely inertial torque.

        RNEA is affine in ``qdd`` at fixed ``(q, qd)``, so the mass-matrix
        product falls out of two calls at ``qd = 0``; gravity is in both and
        cancels exactly, and ``qd = 0`` never generates a Coriolis term.
        Because the armature is diagonal, a ``qdd`` with some columns zeroed
        yields in THOSE rows the pure coupling reaction of the other joints'
        acceleration -- the term the Isaac plant feeds forward to the waist to
        cancel the arms' reaction torque (``waist_reaction_ff``).
        """
        z = torch.zeros_like(q_opt)
        return self.rnea(q_opt, z, qdd_opt) - self.rnea(q_opt, z, torch.zeros_like(qdd_opt))

    __call__ = rnea


def max_error_vs_pinocchio(model, data, opt2pin, batch: BatchedRNEA, n: int = 12) -> float:
    """Max |tau_torch - tau_pin| over random samples (link-inertia part only)."""
    import pinocchio as pin

    if batch.armature is not None:
        batch = batch.without_armature()
    rng = np.random.default_rng(0)
    opt2pin = np.asarray(opt2pin).ravel()
    n_opt = len(opt2pin)
    err = 0.0
    device = batch.gravity.device
    dtype = batch.gravity.dtype
    for _ in range(n):
        q_opt = rng.normal(size=n_opt)
        qd_opt = rng.normal(size=n_opt) * 0.5
        qdd_opt = rng.normal(size=n_opt) * 2.0
        q = np.zeros(model.nq)
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)
        q[opt2pin] = q_opt
        v[opt2pin] = qd_opt
        a[opt2pin] = qdd_opt
        tau_pin = np.asarray(pin.rnea(model, data, q, v, a))[opt2pin]
        with torch.no_grad():
            tau_t = batch.rnea(
                torch.tensor(q_opt.tolist(), dtype=dtype, device=device),
                torch.tensor(qd_opt.tolist(), dtype=dtype, device=device),
                torch.tensor(qdd_opt.tolist(), dtype=dtype, device=device),
            )
        err = max(err, float(np.max(np.abs(tau_t.cpu().numpy() - tau_pin))))
    return err
