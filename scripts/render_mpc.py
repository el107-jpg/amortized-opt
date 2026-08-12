#!/usr/bin/env python
"""Render a closed-loop MPC episode to video.

Runs one eval clip through ``MPCGenerator`` with a trained checkpoint (same
machinery as ``diag_7c.py --probe closed``), then draws the measured motion
with the vendored FK: drums as tilted discs, both arms as torso->elbow->wrist
segments, sticks as wrist->tip segments with tip trails.  A drum flashes for a
few frames when a hit fires.  Output: mp4 (ffmpeg).

Example:
    python scripts/render_mpc.py --ckpt runs/phase2b_main/best.pt \
        --clip-index 0 --steps 250 --lead perjoint --out runs/render/ep0.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

import amortized_opt as A  # noqa: E402
from amortized_opt.fk import stick_mount_transform, stick_tip_offset  # noqa: E402

from diag_7c import PER_JOINT_LEAD, load_model, run_clip  # noqa: E402


def stick_points(fk, q, hand):
    """(tip, butt, wrist) world points for one hand over a (T,17) history."""
    R, p = fk.frame(q, f"{hand}_wrist")
    Rm, pm = stick_mount_transform(hand)
    Rm_t = torch.as_tensor(Rm, dtype=q.dtype)
    pm_t = torch.as_tensor(pm, dtype=q.dtype)
    tip_local = torch.as_tensor(stick_tip_offset(hand), dtype=q.dtype)
    # tip offset is already in the wrist frame (mount folded in by the helper)
    tip = A.ChainFK.points_world(R, p, tip_local[None, :])[:, 0]
    mount = (R @ pm_t) + p if pm_t.ndim == 1 else p
    return tip.numpy(), mount.numpy(), p.numpy()


def frame_points(fk, q, name):
    _, p = fk.frame(q, name)
    return p.numpy()


def disc(center, axis, radius, n=40):
    axis = axis / np.linalg.norm(axis)
    a = np.array([1.0, 0.0, 0.0])
    if abs(axis @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, a)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n)
    return center[None, :] + radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lead", default="perjoint", choices=["none", "perjoint"])
    ap.add_argument("--freeze-wp", type=int, default=0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--trail", type=int, default=25)
    ap.add_argument("--out", default="runs/render/mpc_episode.mp4")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")
    amp = torch.bfloat16

    kit = A.DrumKit.from_layout()
    model, cfg, _ = load_model(root / args.ckpt, dev)
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)
    clip = clips[args.clip_index]

    from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator

    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    starts = sampler.sample(args.clip_index + 1, seed=args.seed)
    q0 = starts.q[args.clip_index].cpu().numpy()
    qd0 = starts.qd[args.clip_index].cpu().numpy()

    cmd_lead = PER_JOINT_LEAD if args.lead == "perjoint" else 0.0
    t0 = time.perf_counter()
    res, record, st, wall = run_clip(
        model, cfg, kit, dev, amp, clip, q0, qd0,
        steps=args.steps, seed=1000, freeze_wp=args.freeze_wp,
        cmd_lead=cmd_lead,
    )
    T = res.q_hist.shape[0] - 1
    print(f"episode: {clip.name} ticks={T} hits={len(res.hit_steps)} "
          f"diverged={res.diverged} solve_wall={wall:.1f}s")

    frames = dict(A.ChainFK.FRAME_JOINTS)
    frames["left_shoulder"] = "left_shoulder_pitch_joint"
    frames["right_shoulder"] = "right_shoulder_pitch_joint"
    fk = A.ChainFK(device="cpu", dtype=torch.float64, frames=frames)
    q = torch.as_tensor(res.q_hist, dtype=torch.float64)
    pts = {}
    for hand in ("left", "right"):
        tip, mount, wrist = stick_points(fk, q, hand)
        pts[hand] = {
            "tip": tip, "wrist": wrist,
            "elbow": frame_points(fk, q, f"{hand}_elbow"),
            "shoulder": frame_points(fk, q, f"{hand}_shoulder"),
        }
    torso = frame_points(fk, q, "torso")
    pelvis = np.zeros_like(torso)

    hit_at = {}
    for step, hand, drum in res.hit_steps:
        hit_at.setdefault(step, []).append((hand, drum))

    fig = plt.figure(figsize=(9, 7), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 0.6))
    drums_env = kit.positions_in("pelvis")

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=3000)
    colors = {"left": "tab:blue", "right": "tab:red"}
    flash_until = {}

    with writer.saving(fig, str(out_path), dpi=110):
        for t in range(T):
            ax.cla()
            ax.set_xlim(-0.2, 1.0)
            ax.set_ylim(-0.7, 0.7)
            ax.set_zlim(-0.4, 0.6)
            ax.view_init(elev=22, azim=-50)
            for ev in hit_at.get(t, []):
                flash_until[ev[1]] = t + 6
            for i, name in enumerate(kit.names):
                ring = disc(drums_env[i], kit.axes[i], kit.radii[i])
                hot = flash_until.get(i, -1) >= t or flash_until.get(name, -1) >= t
                ax.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                        color="orange" if hot else "gray",
                        lw=3.0 if hot else 1.2)
            spine = np.stack([pelvis[t], torso[t]])
            ax.plot(spine[:, 0], spine[:, 1], spine[:, 2], color="k", lw=3.0)
            yoke = np.stack([pts["left"]["shoulder"][t], torso[t],
                             pts["right"]["shoulder"][t]])
            ax.plot(yoke[:, 0], yoke[:, 1], yoke[:, 2], color="k", lw=2.0)
            for hand in ("left", "right"):
                P = pts[hand]
                c = colors[hand]
                seg = np.stack([P["shoulder"][t], P["elbow"][t], P["wrist"][t]])
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=c, lw=2.5)
                stick = np.stack([P["wrist"][t], P["tip"][t]])
                ax.plot(stick[:, 0], stick[:, 1], stick[:, 2], color=c, lw=3.5)
                lo = max(0, t - args.trail)
                ax.plot(P["tip"][lo:t + 1, 0], P["tip"][lo:t + 1, 1],
                        P["tip"][lo:t + 1, 2], color=c, lw=1.0, alpha=0.5)
                ax.scatter(*P["tip"][t], color=c, s=25)
            recent = [f"{h}:{kit.names[d] if isinstance(d, (int, np.integer)) else d}"
                      for s, evs in hit_at.items() if 0 <= t - s <= 12
                      for h, d in [evs[0]]]
            ax.set_title(f"{clip.name}   t={t * A.CONTROL_DT:6.2f}s  tick {t}/{T}"
                         + ("   HIT " + ", ".join(recent) if recent else ""))
            writer.grab_frame()
            if t % 50 == 0:
                print(f"  rendered {t}/{T}", flush=True)

    print(f"wrote {out_path}  ({T} frames @ {args.fps} fps = "
          f"{T / args.fps:.1f}s video for {T * A.CONTROL_DT:.1f}s sim)")
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
