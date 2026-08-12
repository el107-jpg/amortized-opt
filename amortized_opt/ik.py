"""Tip Jacobians and velocity IK -- plain and inertia-weighted (section 5).

The plan's drum-departure start sampler needs to turn a desired *tip* velocity
(pointing away from the drum head it is leaving) into *joint* velocities. Two
ways to do that:

* :func:`damped_pinv` -- vendored verbatim from ``control/velocity_ik.py``.
  Minimizes ``||qd||`` subject to ``J qd = v``: every joint is equally cheap,
  so the solution spreads the motion over the whole arm including the shoulder.
* :func:`inertia_weighted_pinv` -- the plan's actual request. Minimizes the
  kinetic energy ``qd^T M qd`` instead, giving the *dynamically consistent*
  inverse ``M^-1 J^T (J M^-1 J^T)^-1``. Because ``M^-1`` is largest at the
  distal joints, the same tip velocity is produced with proportionally more
  wrist motion and less shoulder motion --- which is both what a real drummer's
  rebound looks like and what the G1's actuator budget can afford (the wrists
  have 5 N.m and almost no inertia; the shoulder has 25 N.m and all of it).

Both return an exact solution for reachable requests: ``J @ qd == v`` to solver
precision, since they differ only in which element of the (11-dimensional, for
one 3-DoF task on a 17-DoF body) solution set they pick.

``M`` comes from :func:`~amortized_opt.pd_sim.mass_matrix` (RNEA with unit
accelerations, armature included) so the weighting is the *effective* inertia
the servo sees, not the link inertia alone --- at ``wrist_roll`` the armature is
the dominant term, and omitting it is the 27x underestimate flagged in
``CLAUDE.md``.

Everything here is batched torch and differentiable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .constants import G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV
from .fk import ChainFK, stick_contact_offset, stick_tip_offset

__all__ = [
    "damped_pinv",
    "inertia_weighted_pinv",
    "tip_position",
    "tip_jacobian",
    "arm_joint_indices",
    "solve_tip_velocity",
]


# ------------------------------------------------------------------- pinvs --
def damped_pinv(J: Tensor, damping: float = 1.0e-3) -> Tensor:
    """``J^+`` for ``J [..., m, n]`` via ``(J J^T + lambda^2 I)^-1``.

    Vendored verbatim from ``control/velocity_ik.py::damped_pinv``.
    """
    m = J.shape[-2]
    JT = J.transpose(-1, -2)
    eye = torch.eye(m, device=J.device, dtype=J.dtype)
    return JT @ torch.linalg.solve(J @ JT + (damping * damping) * eye, eye)


def inertia_weighted_pinv(
    J: Tensor,
    M: Tensor | None = None,
    *,
    Minv: Tensor | None = None,
    damping: float = 1.0e-6,
) -> Tensor:
    """Dynamically-consistent inverse ``M^-1 J^T (J M^-1 J^T + l^2 I)^-1``.

    ``J`` is ``(..., m, n)``; pass either ``M`` (``(..., n, n)``, inverted here)
    or ``Minv`` directly. With ``damping = 0`` and a full-rank task this is the
    exact minimum-kinetic-energy solution; the tiny default ridge only guards a
    singular configuration and costs ~1e-12 of tip-velocity accuracy.
    """
    if (M is None) == (Minv is None):
        raise ValueError("pass exactly one of M / Minv")
    if Minv is None:
        Minv = torch.linalg.inv(M)
    m = J.shape[-2]
    JT = J.transpose(-1, -2)
    W = Minv @ JT  # (..., n, m)
    A = J @ W  # (..., m, m)
    if damping:
        A = A + (damping * damping) * torch.eye(m, device=J.device, dtype=J.dtype)
    return W @ torch.linalg.inv(A)


# --------------------------------------------------------------- kinematics --
def _local_offset(hand: str, sigma: float | None) -> np.ndarray:
    return stick_tip_offset(hand) if sigma is None else stick_contact_offset(hand, float(sigma))


def tip_position(
    fk: ChainFK,
    q: Tensor,
    hand: str,
    *,
    sigma: float | None = None,
    frame: str = "pelvis",
) -> Tensor:
    """(B, 3) world position of the stick tip (or the ``sigma`` contact point).

    ``frame="env"`` adds ``PELVIS_IN_ENV`` so the result is comparable with the
    drum layout; ``"pelvis"`` returns the raw FK frame.
    """
    R, p = fk.frame(q, f"{hand}_wrist")
    local = torch.as_tensor(
        _local_offset(hand, sigma).tolist(), device=q.device, dtype=q.dtype
    )
    out = p + torch.einsum("bij,j->bi", R, local)
    if frame == "env":
        off = torch.as_tensor(PELVIS_IN_ENV.tolist(), device=q.device, dtype=q.dtype)
        return out + off
    if frame != "pelvis":
        raise ValueError(f"frame must be 'pelvis' or 'env', got {frame!r}")
    return out


def tip_jacobian(
    fk: ChainFK,
    q: Tensor,
    hand: str,
    *,
    sigma: float | None = None,
) -> Tensor:
    """(B, 3, n) positional Jacobian of the tip w.r.t. all 17 joints.

    Computed by reverse-mode autograd through :class:`ChainFK` -- exact (not a
    finite difference), and it inherits the FK's sync-free property. Three
    backward passes, one per Cartesian row, so it costs about 3 FK passes.
    """
    if q.dim() != 2:
        raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
    qg = q.detach().requires_grad_(True)
    pos = tip_position(fk, qg, hand, sigma=sigma)
    rows = []
    for axis in range(3):
        (g,) = torch.autograd.grad(
            pos[:, axis].sum(), qg, retain_graph=axis < 2, create_graph=False
        )
        rows.append(g)
    return torch.stack(rows, dim=1)


def arm_joint_indices(hand: str, joint_names=G1_UPPER_BODY_JOINTS) -> list[int]:
    """Indices of one arm's joints (waist excluded) in optimizer order."""
    prefix = f"{hand}_"
    return [i for i, n in enumerate(joint_names) if n.startswith(prefix)]


# ------------------------------------------------------------------- solver --
def solve_tip_velocity(
    fk: ChainFK,
    q: Tensor,
    hand: str,
    v_tip: Tensor,
    *,
    M: Tensor | None = None,
    Minv: Tensor | None = None,
    weighting: str = "inertia",
    joint_subset: list[int] | None = None,
    damping: float = 1.0e-6,
    v_limit: Tensor | None = None,
) -> Tensor:
    """(B, n) joint velocities realizing the tip velocity ``v_tip`` (B, 3).

    ``weighting`` is ``"inertia"`` (dynamically consistent, needs ``M``/``Minv``)
    or ``"plain"`` (damped least squares). ``joint_subset`` restricts which
    joints may move --- the drum-departure sampler passes one arm's 7 joints so
    the two hands cannot fight over the shared waist. ``v_limit`` (n,) clips the
    result and is applied AFTER the solve, so the returned tip velocity is only
    exact when no clip fires.
    """
    J = tip_jacobian(fk, q, hand)
    n = J.shape[-1]
    if joint_subset is not None:
        idx = torch.as_tensor(list(joint_subset), device=q.device, dtype=torch.long)
        Js = J.index_select(2, idx)
        if weighting == "inertia":
            if Minv is None:
                if M is None:
                    raise ValueError("inertia weighting needs M or Minv")
                Ms = M.index_select(1, idx).index_select(2, idx)
                Minvs = torch.linalg.inv(Ms)
            else:
                # Invert back to the sub-block: the principal minor of M, not of
                # M^-1 (they differ, and the energy metric is defined on M).
                Ms = torch.linalg.inv(Minv).index_select(1, idx).index_select(2, idx)
                Minvs = torch.linalg.inv(Ms)
            P = inertia_weighted_pinv(Js, Minv=Minvs, damping=damping)
        else:
            P = damped_pinv(Js, damping=max(damping, 1e-6))
        qd_sub = torch.matmul(P, v_tip.unsqueeze(-1)).squeeze(-1)
        qd = q.new_zeros(q.shape[0], n)
        qd = qd.index_copy(1, idx, qd_sub)
    else:
        if weighting == "inertia":
            P = inertia_weighted_pinv(J, M=M, Minv=Minv, damping=damping)
        elif weighting == "plain":
            P = damped_pinv(J, damping=max(damping, 1e-6))
        else:
            raise ValueError(f"weighting must be 'inertia' or 'plain', got {weighting!r}")
        qd = torch.matmul(P, v_tip.unsqueeze(-1)).squeeze(-1)
    if v_limit is not None:
        lim = v_limit.view(1, -1).to(qd.dtype)
        qd = torch.clamp(qd, -lim, lim)
    return qd
