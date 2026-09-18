"""Match ``scripts/play_isaac.py`` start pose and first-hit clock.

PPO used to spawn at the Unitree USD ``init_state`` (shoulder pitch ~+0.3).
Play overwrites that pose with ``amortized_opt.ready_joint_pos`` (pitch −1.20)
before the first plan, and ``--first_hit_s 1.0`` delays the clip so the first
hand hit has the same lead idle-family training used from rest.  Episode end
still reseeds from leftover ``q`` (zero velocity), not a teleport to ready,
and waits ``clip_gap_s`` (default 1 s) after the last hit before chaining.

Isaac-free helpers; the env methods they call (``ready_q``, ``reset_to``)
are the same on the live sibling and the pinned snapshot.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.rhythm import RhythmClip, pad_first_hit_to

__all__ = [
    "amortized_ready_q",
    "describe_clip_pad",
    "ensure_clip_gap",
    "install_duration_done",
    "install_train_ready",
    "make_leftover_on_done",
    "pad_clip_pool",
    "reset_envs_to_ready",
    "upper_body_ready",
]


def amortized_ready_q(device=None, dtype=torch.float32) -> torch.Tensor:
    """The 17-DoF vector phase-2 / the idle cylinder / play_isaac start from."""
    return torch.as_tensor(
        ready_joint_pos(G1_UPPER_BODY_JOINTS), device=device, dtype=dtype,
    )


def upper_body_ready(env) -> torch.Tensor:
    """``(17,)`` from the env's ready buffer in ``G1_UPPER_BODY_JOINTS`` order."""
    ids = env.joint_ids
    return env.ready_q[0, ids].detach().to(dtype=torch.float32).clone()


def reset_envs_to_ready(env, env_ids) -> None:
    """Teleport ``env_ids`` to the env ready pose at rest."""
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1).long()
    if env_ids.numel() == 0:
        return
    q = upper_body_ready(env).to(device=env.device)
    env.reset_to(
        env_ids,
        q.unsqueeze(0).expand(env_ids.numel(), -1).contiguous(),
        None,
    )


def install_train_ready(env) -> None:
    """Overwrite USD init_state with ``ready_joint_pos`` and teleport every env.

    Same rewrite ``play_isaac._install_train_ready`` does on ``G1DrummerEnv``.
    """
    q = amortized_ready_q(device=env.device, dtype=env.ready_q.dtype)
    ids = env.joint_ids
    was = env.ready_q[0, ids].detach().clone()
    env.ready_q[:, ids] = q
    tgt = getattr(env, "joint_pos_target", None)
    if tgt is not None:
        env.joint_pos_target[:, ids] = q
    reset_envs_to_ready(env, torch.arange(env.num_envs, device=env.device))

    names = G1_UPPER_BODY_JOINTS
    ip = names.index("left_shoulder_pitch_joint")
    ie = names.index("left_elbow_joint")

    def _deg(i: int) -> tuple[float, float]:
        return float(torch.rad2deg(q[i]).item()), float(torch.rad2deg(was[i]).item())

    p_now, p_was = _deg(ip)
    e_now, e_was = _deg(ie)
    print(
        f"[ready] amortized ready_joint_pos "
        f"(pitch {p_was:+.1f}->{p_now:+.1f}deg, "
        f"elbow {e_was:+.1f}->{e_now:+.1f}deg)",
        flush=True,
    )


def pad_clip_pool(
    clips: Sequence[RhythmClip], t_first: float | None,
) -> tuple[list[RhythmClip], dict]:
    """``pad_first_hit_to`` on every clip.  ``t_first<=0`` or ``None`` is a no-op."""
    clips = list(clips)
    if t_first is None or float(t_first) <= 0.0:
        return clips, {
            "enabled": False, "t_first": 0.0, "n": len(clips),
            "n_shifted": 0, "dt_p50": 0.0,
        }
    out: list[RhythmClip] = []
    dts: list[float] = []
    for clip in clips:
        shifted, dt = pad_first_hit_to(clip, float(t_first))
        out.append(shifted)
        dts.append(float(dt))
    return out, {
        "enabled": True,
        "t_first": float(t_first),
        "n": len(out),
        "n_shifted": int(sum(1 for d in dts if d > 1e-12)),
        "dt_p50": float(np.median(dts)) if dts else 0.0,
    }


def describe_clip_pad(stats: dict) -> str:
    if not stats.get("enabled"):
        return "[clip] first-hit pad OFF (native RCC times)"
    return (
        f"[clip] first hand hit padded to {stats['t_first']:.2f}s "
        f"({stats['n_shifted']}/{stats['n']} clips shifted, "
        f"dt p50={stats['dt_p50']:.2f}s)"
    )


def ensure_clip_gap(
    clips: Sequence[RhythmClip], gap_s: float,
) -> list[RhythmClip]:
    """``duration`` at least ``last_hit + gap_s``.  ``gap_s<=0`` is a no-op.

    ``load_rcc`` already writes ``post_buffer_s=1.0`` into duration, but a
    pad or a hand-built clip can leave duration == last hit.  The gap only
    plays if :func:`install_duration_done` is also applied: the stock
    ``BatchedSchedule.done`` fires as soon as the last hit is consumed.
    """
    clips = list(clips)
    if gap_s is None or float(gap_s) <= 0.0:
        return clips
    gap = float(gap_s)
    out: list[RhythmClip] = []
    for clip in clips:
        last = max((float(h.time) for h in clip.hits), default=0.0)
        need = last + gap
        if float(clip.duration) + 1e-9 < need:
            clip = replace(clip, duration=need)
        out.append(clip)
    return out


def install_duration_done(sched, *, enabled: bool = True) -> None:
    """Chain the next clip only after ``t_now >= duration``.

    Stock ``done`` is ``(no hits left) | (t_now >= duration)``.  The first
    clause trips on the tick the last hit fires, so leftover ``q`` never
    gets the post-buffer idle between clips.
    """
    if not enabled:
        return

    def done() -> torch.Tensor:
        return sched.t_now >= sched.duration

    sched.done = done


def make_leftover_on_done(env, reset_sched: Callable, mpc) -> Callable:
    """Episode end: new clip, keep leftover ``q`` at rest, reseed MPC."""

    def on_done(ids, q, qd):
        del qd
        reset_sched(ids)
        env.reset_to(ids, q[ids].clone(), None)
        q_new, qd_new = env.read_state()
        mpc.reset(ids, q_new)
        return q_new, qd_new

    return on_done
