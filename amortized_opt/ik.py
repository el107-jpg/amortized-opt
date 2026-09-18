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

from .constants import CONTROL_DT, G1_UPPER_BODY_JOINTS, PELVIS_IN_ENV
from .fk import ChainFK, stick_contact_offset, stick_tip_offset

__all__ = [
    "damped_pinv",
    "inertia_weighted_pinv",
    "tip_position",
    "tip_jacobian",
    "arm_joint_indices",
    "solve_tip_velocity",
    "correct_tip",
    "adjust_arm_terminal",
    "correct_quintic_init_arms",
    "corrected_arm_terminals",
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
def _local_offset(hand: str, sigma: float | None, stick=None) -> np.ndarray:
    """Stick point in the wrist-yaw frame: ``(3,)``, or ``(B, 3)`` per row.

    ``stick`` is ``None`` (the nominal stick), one
    :class:`~amortized_opt.fk.StickSpec` shared by every row, or a sequence of
    ``StickSpec | None`` -- ONE PER BATCH ROW -- for per-row randomized sticks.
    The stick only ever enters the kinematics as this constant local offset, so
    per-row support here is all :func:`tip_position` / :func:`tip_jacobian` /
    :func:`correct_tip` need to solve a whole randomized pool in one call.
    """

    def one(sp) -> np.ndarray:
        if sigma is not None:
            return stick_contact_offset(hand, float(sigma), sp)
        return stick_tip_offset(hand, sp)

    if isinstance(stick, (list, tuple)):
        return np.stack([one(sp) for sp in stick], axis=0)
    return one(stick)


def tip_position(
    fk: ChainFK,
    q: Tensor,
    hand: str,
    *,
    sigma: float | None = None,
    frame: str = "pelvis",
    stick=None,
) -> Tensor:
    """(B, 3) world position of the stick tip (or the ``sigma`` contact point).

    ``frame="env"`` adds ``PELVIS_IN_ENV`` so the result is comparable with the
    drum layout; ``"pelvis"`` returns the raw FK frame.  ``stick`` may be one
    :class:`~amortized_opt.fk.StickSpec` for the whole batch or a length-``B``
    sequence of per-row specs (see :func:`_local_offset`).
    """
    R, p = fk.frame(q, f"{hand}_wrist")
    off = _local_offset(hand, sigma, stick)
    local = torch.as_tensor(off.tolist(), device=q.device, dtype=q.dtype)
    if local.dim() == 2:
        if local.shape[0] != q.shape[0]:
            raise ValueError(
                f"per-row stick batch {local.shape[0]} != q batch {q.shape[0]}"
            )
        out = p + torch.einsum("bij,bj->bi", R, local)
    else:
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
    stick=None,
) -> Tensor:
    """(B, 3, n) positional Jacobian of the tip w.r.t. all 17 joints.

    Computed by reverse-mode autograd through :class:`ChainFK` -- exact (not a
    finite difference), and it inherits the FK's sync-free property. Three
    backward passes, one per Cartesian row, so it costs about 3 FK passes.
    """
    if q.dim() != 2:
        raise ValueError(f"q must be (B, n), got {tuple(q.shape)}")
    # Callers sit under ``@torch.no_grad()`` (roller) or ``inference_mode``
    # (``play_isaac.run_episode`` wraps reset the same way the env steps).
    # ``enable_grad`` does not exit inference mode, and an inference tensor
    # cannot grow a graph even with ``requires_grad_(True)`` -- clone it out.
    with torch.inference_mode(False), torch.enable_grad():
        # ``clone()`` of an inference tensor can stay inference; allocate fresh.
        qg = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        qg.copy_(q)
        qg.requires_grad_(True)
        pos = tip_position(fk, qg, hand, sigma=sigma, stick=stick)
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


# ------------------------------------------------------- terminal-pose fix --
def correct_tip(
    fk: ChainFK,
    q: Tensor,
    aim: Tensor,
    hand: str,
    *,
    steps: int = 3,
    damping: float = 0.05,
    sigma: float | None = None,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    stick=None,
) -> Tensor:
    """Move ONE arm's joints so its stick tip reaches ``aim`` (env frame).

    ``q`` is ``(B, n)`` and ``aim`` is ``(B, 3)``; the return is ``(B, n)`` with
    only ``hand``'s arm columns changed.  Damped least squares, ``steps``
    iterations, clamped to the joint box each step.

    Why this exists
    ---------------
    :class:`~amortized_opt.warmstart.GoalPoseLibrary` is keyed on
    ``(hand, drum, waist_yaw)`` and stores a JOINT pose.  It therefore knows
    nothing about (a) where the drum actually is once
    :class:`~amortized_opt.rhythm.KitRandomization` has moved it, or (b) what
    stick is fitted.  Both change where the tip lands for the same ``arm_q``, so
    the seed's terminal drifts off the head.  Measured over 60 randomized kits
    at the shipped jitter, the library terminal misses the head centre by a
    median 12.9 cm; three steps of this bring the median to 0.06 cm.

    The WAIST is deliberately excluded.  It is shared between the hands and is
    already set by the yaw ramp, so letting the correction move it would make
    the two arms fight over it and desynchronise the ramp the cost prices.

    Cost is dominated by the autograd Jacobian, so CALL THIS BATCHED: measured
    49 us/pose/step at batch 512 against 10.3 ms at batch 1, a 200x difference.
    Per-row randomized sticks do NOT break the batching -- pass ``stick`` as a
    length-``B`` sequence of ``StickSpec | None`` and every row is corrected
    against its own fitted stick in the same call.

    The residual is not always zero, and that is not a solver failure: the two
    cross-body reaches (``right/cymbal_right``, ``left/tom2``) put the head
    outside the arm's workspace at some kit jitters, and no joint solution
    exists.  Those rows come back at the closest reachable pose.
    """
    if q.dim() != 2 or aim.dim() != 2:
        raise ValueError(f"q must be (B, n) and aim (B, 3); got {tuple(q.shape)}, {tuple(aim.shape)}")
    if q.shape[0] != aim.shape[0]:
        raise ValueError(f"batch mismatch: q {q.shape[0]} vs aim {aim.shape[0]}")
    cols = torch.as_tensor(arm_joint_indices(hand, joint_names), device=q.device)
    out = q.clone()
    lam2 = float(damping) ** 2
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    lo = hi = None
    if q_bounds is not None:
        lo = torch.as_tensor(q_bounds[0], device=q.device, dtype=q.dtype).index_select(0, cols)
        hi = torch.as_tensor(q_bounds[1], device=q.device, dtype=q.dtype).index_select(0, cols)
    for _ in range(int(steps)):
        with torch.no_grad():
            err = aim - tip_position(fk, out, hand, sigma=sigma, frame="env", stick=stick)
        J = tip_jacobian(fk, out, hand, sigma=sigma, stick=stick).index_select(2, cols)
        JT = J.transpose(1, 2)
        dq = (JT @ torch.linalg.solve(J @ JT + lam2 * eye, err.unsqueeze(-1))).squeeze(-1)
        upd = out.index_select(1, cols) + dq
        if lo is not None:
            upd = torch.clamp(upd, lo, hi)
        out.index_copy_(1, cols, upd)
    return out


def adjust_arm_terminal(
    fk: ChainFK,
    q: Tensor,
    aim: Tensor,
    hand: str,
    *,
    steps: int = 1,
    damping: float = 0.05,
    sigma: float | None = None,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    stick=None,
) -> Tensor:
    """One-shot DLS correction of a quintic-init arm terminal.

    ``q`` is ``(B, n)`` with the waist already at the strike knot and the
    library ``arm_q`` already written.  ``aim`` is ``(B, 3)`` in the env frame
    (actual drum / ``p_star``, same as :func:`correct_tip`).  Returns ``(B, n)``
    with only ``hand``'s 7 arm columns changed.

    This is exactly :func:`correct_tip` with ``steps=1`` (the default here):
    one Newton / damped-least-squares step, full arm, waist frozen.  Damping
    stays at ``correct_tip``'s ``0.05``.

    Why one step, not three
    -----------------------
    Three steps close the residual to sub-cm (apex teacher,
    ``gen_drum_goal_poses``).  Quintic inits are a seed neighborhood, not a
    strike: one step drops the library miss from ~13 cm to ~5 cm (often
    inside a pad disk) at 1/3 the autograd cost.  Three online steps at
    batch 1 would be 10.3 ms/pose/step.

    Why full arm, not proximal-only
    -------------------------------
    Shoulders-only is a poorly-conditioned 3×3 map for a 3-D tip
    (``|Δq|`` ~0.8 rad, median miss 7 cm vs 5 cm full-arm).  Shoulder+waist
    yanks the shared torso and fights the encoder P/R set-point and the
    other hand.  :func:`correct_tip` already freezes the waist for that
    reason.

    Fold every residual into this one FK error -- do not compute a separate
    ``J_w Δq_w``.  Library yaw-bin quantization, encoder P/R vs the upright
    library, kit randomization / actual ``p_star`` / stick all show up as
    ``aim - FK(q_seed)``.

    Always batch.  Cost is dominated by the autograd Jacobian: 49 µs/pose/step
    at batch 512 vs 10.3 ms at batch 1.
    """
    # Newton step wants full precision even if the caller sits under autocast.
    dev_type = "cuda" if q.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=dev_type, enabled=False):
        return correct_tip(
            fk, q, aim, hand,
            steps=int(steps), damping=float(damping), sigma=sigma,
            q_bounds=q_bounds, joint_names=joint_names, stick=stick,
        )


def _rest_quintic(q_cur: Tensor, terminal: Tensor, lengths: Tensor, H: int) -> Tensor:
    """Rest-to-rest quintic, knots 1..H -- same formula as ``train_cost.ramp_seed``."""
    k = torch.arange(1, H + 1, device=q_cur.device, dtype=q_cur.dtype).view(1, H, 1)
    L = lengths.clamp_min(1).to(q_cur.dtype).unsqueeze(1)
    s = (k / L).clamp(0.0, 1.0)
    ease = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
    return q_cur.unsqueeze(1) + ease * (terminal - q_cur).unsqueeze(1)


def _vel_quintic(
    q_cur: Tensor, v_cur: Tensor, terminal: Tensor, lengths: Tensor, H: int, dt: float,
) -> Tensor:
    """Velocity-matched quintic -- same formula as ``train_cost.vel_ramp_seed``."""
    k = torch.arange(1, H + 1, device=q_cur.device, dtype=q_cur.dtype).view(1, H, 1)
    L = lengths.clamp_min(1).to(q_cur.dtype)
    u = (k / L.unsqueeze(1)).clamp(0.0, 1.0)
    d = (terminal - q_cur).unsqueeze(1)
    V = (v_cur * L * float(dt)).unsqueeze(1)
    u3 = u * u * u
    return (
        q_cur.unsqueeze(1) + V * u
        + (10.0 * d - 6.0 * V) * u3
        + (8.0 * V - 15.0 * d) * u3 * u
        + (6.0 * d - 3.0 * V) * u3 * u * u
    )


def correct_quintic_init_arms(
    fk: ChainFK,
    init: Tensor,
    q0: Tensor,
    hit: Tensor,
    aims: Tensor,
    *,
    active: Tensor | None = None,
    steps: int = 1,
    damping: float = 0.05,
    q_bounds: tuple | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    stick=None,
    qd0: Tensor | None = None,
    dt: float = CONTROL_DT,
) -> Tensor:
    """One-shot DLS on each active hand's strike terminal, then rebuild the arm quintic.

    ``init`` is ``(B, H, n)`` knots 1..H.  Strike knot ``k`` (``hit[b, h] >= 1``)
    is row ``k - 1``; the waist at that row is the waist the quintic will have
    at the hit (after the yaw ramp and any encoder P/R rewrite).  ``aims`` is
    ``(B, 2, 3)`` in the env frame.  Only arm columns of active hands change;
    the path still starts at ``q0``.

    Use this on **quintic inits**, not on a shifted decoder plan: after the
    correction the arm columns are a fresh ramp to the new terminal
    (``terminal_arm`` / ``ramp_seed``), not an in-place overwrite of the last
    knot.
    """
    if int(steps) <= 0:
        return init
    if init.dim() != 3:
        raise ValueError(f"init must be (B, H, n), got {tuple(init.shape)}")
    B, H, n = init.shape
    out = init.clone()
    hit = hit.to(device=init.device)
    if active is None:
        live = hit >= 1
    else:
        live = active.to(device=init.device, dtype=torch.bool) & (hit >= 1)
    names = tuple(joint_names)
    for hi, hand in enumerate(("left", "right")):
        rows = live[:, hi].nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        acols = arm_joint_indices(hand, names)
        k = hit.index_select(0, rows)[:, hi].clamp(1, H)
        q_hit = out.index_select(0, rows).gather(
            1, (k - 1).view(-1, 1, 1).expand(-1, 1, n)
        ).squeeze(1)
        aim_h = aims.to(device=init.device, dtype=init.dtype).index_select(0, rows)[:, hi]
        stick_h = None
        if stick is not None:
            # ``stick`` is ``(left, right)`` shared specs, or a length-2
            # sequence of per-row sequences.
            spec = stick[hi]
            if isinstance(spec, (list, tuple)) and spec and not hasattr(spec[0], "hand"):
                stick_h = [spec[int(r)] for r in rows.tolist()]
            else:
                stick_h = spec
        q_in = q_hit.to(device=fk.device, dtype=fk.dtype)
        aim_in = aim_h.to(device=fk.device, dtype=fk.dtype)
        q_fix = adjust_arm_terminal(
            fk, q_in, aim_in, hand,
            steps=int(steps), damping=float(damping),
            q_bounds=q_bounds, joint_names=names, stick=stick_h,
        )
        term = q_fix[:, acols].to(device=out.device, dtype=out.dtype)
        q0r = q0.to(device=out.device, dtype=out.dtype).index_select(0, rows)[:, acols]
        L = k.to(dtype=out.dtype).unsqueeze(-1).expand(-1, len(acols))
        if qd0 is not None:
            v0 = qd0.to(device=out.device, dtype=out.dtype).index_select(0, rows)[:, acols]
            sub = _vel_quintic(q0r, v0, term, L, H, float(dt))
        else:
            sub = _rest_quintic(q0r, term, L, H)
        buf = out.index_select(0, rows).clone()
        idx = torch.as_tensor(acols, device=out.device, dtype=torch.long)
        buf.index_copy_(2, idx, sub)
        out.index_copy_(0, rows, buf)
    return out


def corrected_arm_terminals(
    fk: ChainFK,
    items,
    *,
    steps: int = 1,
    damping: float = 0.05,
    q_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    joint_names=G1_UPPER_BODY_JOINTS,
    library,
) -> list[dict[str, np.ndarray]]:
    """Batched one-shot (or ``steps``) DLS for numpy quintic-pool rows.

    Each item is a dict with ``ws`` (a :class:`~amortized_opt.warmstart.WarmStart`
    whose ``q[k_hit]`` waist is the **reached** strike-knot waist), ``goals``,
    ``aim`` (env frame), ``hit``, optional ``sticks`` ``(left, right)``, and
    ``library`` is the :class:`~amortized_opt.warmstart.GoalPoseLibrary`.

    Returns one ``{hand: arm_q}`` per item (idle hands omitted).  Callers pass
    that as ``terminal_arm=`` into :func:`~amortized_opt.warmstart.warmstart_trajectory`
    so the arm quintic is rebuilt, not just the last knot overwritten.

    Apex pass 2 is the template: stack every row of one hand, one
    :func:`adjust_arm_terminal` call.  ``steps=1`` is the quintic-init default;
    the apex teacher keeps ``steps=3`` by asking for 3.
    """
    from .warmstart import arm_slice

    names = tuple(joint_names)
    n = len(names)
    out: list[dict[str, np.ndarray]] = [{} for _ in items]
    if int(steps) <= 0 or not items:
        return out
    for hand in ("left", "right"):
        acols = arm_slice(hand, names)
        rows = [i for i, it in enumerate(items) if it["goals"].get(hand) is not None]
        if not rows:
            continue
        qs, aims, sticks_rows = [], [], []
        for i in rows:
            it = items[i]
            ws = it["ws"]
            q_ws = np.asarray(ws.q, float)
            kh = int(np.clip(it["hit"][hand], 1, q_ws.shape[0] - 1))
            qz = np.zeros(n, dtype=float)
            qz[:3] = q_ws[kh, :3]
            qz[acols] = library.select(
                hand, it["goals"][hand][0], ws.yaw_target
            ).arm_q
            qs.append(qz)
            aims.append(np.asarray(it["aim"][hand], float))
            sticks = it.get("sticks")
            hi_ = 0 if hand == "left" else 1
            sticks_rows.append(None if sticks is None else sticks[hi_])
        stick_arg = None if all(sp is None for sp in sticks_rows) else sticks_rows
        tt = lambda a: torch.as_tensor(  # noqa: E731
            np.asarray(a), device=fk.device, dtype=fk.dtype
        )
        qc = adjust_arm_terminal(
            fk, tt(np.stack(qs)), tt(np.stack(aims)), hand,
            steps=int(steps), damping=float(damping),
            q_bounds=q_bounds, joint_names=names, stick=stick_arg,
        ).detach().cpu().numpy()
        for j, i in enumerate(rows):
            out[i][hand] = qc[j, acols]
    return out
