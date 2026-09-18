#!/usr/bin/env python
"""Where the sticks are at scheduled fire, on the Isaac plant.

Loads a PPO actor, rolls mean-policy (tiny std, no AR1) with freeze_wp matching
training, and dumps head-frame geometry at every scheduled fire.

    CUDA_VISIBLE_DEVICES=1 python isaac_rl_ppo/scripts/dump_hit_geometry.py \
        --ckpt runs/rhythm_bellman_nf1_frontier_ball/best.pt \
        --resume runs/isaac_rl_ppo_orth_frontier_ball_v7/last.pt \
        --device cuda:0 --envs 16 --steps 400 --freeze-wp 5 --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaac_rl_ppo.train_ppo import build_cfg, parse as train_parse  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--freeze-wp", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clips", type=int, default=4)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--noisy", action="store_true",
                   help="keep the trained std / AR(1) instead of mean policy")
    p.add_argument("--isaac-ready", action="store_true", default=False)
    p.add_argument("--first-hit-s", type=float, default=1.0)
    p.add_argument("--clip-gap-s", type=float, default=1.0)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    fake = [
        "--ckpt", args.ckpt, "--resume", args.resume, "--device", args.device,
        "--envs", str(args.envs), "--freeze-wp", str(args.freeze_wp),
        "--seed", str(args.seed), "--clips", str(args.clips),
        "--headless", "--name", "_hitgeom_probe", "--iterations", "1",
        "--first-hit-s", str(args.first_hit_s),
        "--clip-gap-s", str(args.clip_gap_s),
    ]
    if args.isaac_ready:
        fake.append("--isaac-ready")
    targs = train_parse(fake)
    cfg = build_cfg(targs)

    from amortized_opt.rnea import BatchedRNEA
    try:
        _rnea_cpu = BatchedRNEA.from_urdf(device="cpu")
        print("[prelaunch] BatchedRNEA OK", flush=True)
    except Exception as exc:
        _rnea_cpu = None
        print(f"[prelaunch] BatchedRNEA FAILED ({exc!r})", flush=True)

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, device=args.device).app

    import numpy as np
    import torch
    import amortized_opt as A
    from isaac_rl_ppo import bridge
    from isaac_rl_ppo.actor import AmortizedPpoActor, executed_command
    from isaac_rl_ppo.ppo_loop import RolloutDriver

    surface = bridge.load_isaac_rl()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index or 0)
    dtype = torch.float32

    envmod = surface.modules["env"]
    env_cfg = surface.DrumRLEnvCfg(
        num_envs=cfg.env.num_envs, device=args.device, seed=args.seed,
        rhythm_split="train", contact_sensors=cfg.env.contact_sensors,
        gravity_feedforward=cfg.env.gravity_feedforward, foh_ramp=cfg.env.foh_ramp,
        pin_pelvis=cfg.env.pin_pelvis, freeze_legs=cfg.env.freeze_legs,
        use_ideal_pd=cfg.env.use_ideal_pd, pd_damping_scale=cfg.env.pd_damping_scale,
        hit_mode=cfg.env.hit_mode, cmd_lead_s=cfg.env.cmd_lead_s,
        freeze_wp=cfg.env.freeze_wp, hit_lead_s=cfg.env.hit_lead_s,
    )
    isaac_cfg = envmod.build_isaac_cfg(env_cfg)
    isaac_cfg.sim.device = args.device
    env = envmod._make(isaac_cfg, env_cfg)
    env.reset()
    print(f"[env] {args.envs} envs step_dt={float(env.step_dt):.4f}", flush=True)

    from isaac_rl_ppo.start import (
        describe_clip_pad, ensure_clip_gap, install_duration_done,
        install_train_ready, make_leftover_on_done, pad_clip_pool,
    )
    if not cfg.env.isaac_ready:
        install_train_ready(env)
    else:
        print("[ready] --isaac-ready: keeping env composed pose", flush=True)

    choice_path = Path(args.ckpt)
    ck = torch.load(str(choice_path), map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    H = int((ck.get("scenario_cfg") or {}).get("horizon", 50))
    cfg.actor.horizon = H
    actor = AmortizedPpoActor(model, cfg.actor, cfg.critic,
                              num_future=mcfg.num_future, device=dev)
    blob = torch.load(args.resume, map_location=dev, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=False)
    actor.eval()
    if not args.noisy:
        with torch.no_grad():
            actor.std.fill_(actor.cfg.noise_std_min)
        actor._ar1_active = False
        print(f"[policy] MEAN (std={float(actor.std.mean()):.4f}, AR1 off)", flush=True)
    else:
        print(f"[policy] NOISY std_mean={float(actor.current_std().mean()):.3f}",
              flush=True)

    kit = A.DrumKit.from_layout()
    names = list(kit.names)
    clips = A.load_rcc_pool(A.default_rcc_dir("train"), kit)[: max(args.clips, 1)]
    clips, pad_stats = pad_clip_pool(clips, cfg.env.first_hit_s)
    clips = ensure_clip_gap(clips, cfg.env.clip_gap_s)
    print(describe_clip_pad(pad_stats), flush=True)
    print(
        f"[clip] gap after last hit {cfg.env.clip_gap_s:.2f}s "
        f"({'done waits for duration' if cfg.env.clip_gap_s > 0 else 'chain on last hit'})",
        flush=True,
    )
    ScheduleConfig, BatchedSchedule = surface.ScheduleConfig, surface.BatchedSchedule
    scfg = ScheduleConfig(horizon=H, num_drums=mcfg.num_drums,
                          num_future=mcfg.num_future, hit_lead_s=cfg.env.hit_lead_s)
    sched = BatchedSchedule(args.envs, scfg, device=dev, dtype=dtype)
    install_duration_done(sched, enabled=cfg.env.clip_gap_s > 0)
    rng = np.random.default_rng(args.seed)

    def _reset_sched(ids):
        picks = [clips[int(i) % len(clips)] for i in range(ids.numel())]
        sched.reset(ids, picks, [kit] * ids.numel(), rng=rng)

    all_ids = torch.arange(args.envs, device=dev)
    _reset_sched(all_ids)
    lead_value, lead_src = cfg.resolve_lead()
    print(f"[lead] {lead_src} freeze_wp={cfg.env.freeze_wp}", flush=True)

    IsaacMPC, MPCConfig = surface.IsaacMPC, surface.MPCConfig
    mpc = IsaacMPC(
        model, sched,
        MPCConfig(horizon=H, iterations=cfg.actor.iterations,
                  cmd_lead_s=lead_value, freeze_wp=cfg.env.freeze_wp,
                  freeze_waist=cfg.actor.freeze_waist, amp_dtype=None),
        device=dev, dtype=dtype, kit_names=tuple(kit.names))
    fk = A.ChainFK(device=dev, dtype=dtype)
    from amortized_opt.train_cost import phase2_cost_config
    cost = None
    if _rnea_cpu is not None:
        cost = A.FusedCost(phase2_cost_config(),
                           rnea=_rnea_cpu.to(dev, dtype),
                           max_horizon=H, device=dev, dtype=dtype)
    rcfg = surface.RewardConfig()
    rcfg.w_plan_cost = 0.0
    scorer = surface.ExecutedStrikeScorer(fk=fk, dt=float(env.step_dt),
                                          device=dev, dtype=dtype)
    reward = surface.ExecutedReward(rcfg, scorer=scorer, cost=cost,
                                    device=dev, dtype=dtype)
    q, qd = env.read_state()
    mpc.reset(all_ids, q)
    on_done = make_leftover_on_done(env, _reset_sched, mpc)

    driver = RolloutDriver(actor=actor, mpc=mpc, schedule=sched, reward=reward,
                           plant=env.plant, cfg=cfg, fk=fk, on_done=on_done)
    HANDS = ("left", "right")
    events = []
    t0 = time.perf_counter()
    actor.eval()
    with torch.inference_mode():
      for step in range(int(args.steps)):
        ctx = sched.strike_context()
        hit_index = sched.hit_index()
        res = mpc.plan(q, qd)
        mean = mpc.last_ref0
        dist = actor.update_distribution(mean)
        if actor._ar1_active:
            action = mean + dist.stddev * actor._sample_ar1(mean)
        else:
            action = mean
        delta = action - mean
        cmd = executed_command(res.cmd, delta, prev_cmd=mpc.prev_cmd,
                               step_max=mpc.step_max, q_lo=mpc.q_lo, q_hi=mpc.q_hi)
        q_before = q
        q, qd = driver._step_plant(mpc.prev_cmd, cmd, q, qd)
        fired = sched.advance()
        terms, cs = reward.step(
            q_before=q_before, q_after=q, qd_after=qd, ctx=ctx,
            hit_index=res.hit_index, fired=fired, ref0=mean,
            traj=res.traj, elem_mask=res.elem_mask,
            q_plan0=q_before, v_plan0=None,
            drum_heads=ctx.drum_head_centers,
            drum_axes=sched.geom.get("axis"), drum_radii=sched.drum_radius,
            drum_valid=sched.drum_valid, filter_active=None,
        )
        # plan-nominated contact at the fire knot vs executed
        _d_plan, _s_plan, p_plan = scorer.best_contact(mean, ctx)
        _d_ex, _s_ex, p_ex = scorer.best_contact(q, ctx)
        ax_plan, ip_plan = scorer.head_frame(p_plan, ctx)
        ax_ex, ip_ex = scorer.head_frame(p_ex, ctx)
        mpc.commit(cmd, q, fired)
        done = sched.done()
        if bool(done.any()):
            q, qd = on_done(done.nonzero(as_tuple=True)[0], q, qd)

        rows = fired.nonzero(as_tuple=False)
        if rows.numel() == 0:
            continue
        raw = terms.raw
        coll = float(raw.get("plan.collision", torch.zeros(1)).mean())
        viol = float(raw.get("viol.collision", torch.zeros(1)).mean())
        extra = terms.counts.get("extra_hits")
        for b, h in rows.tolist():
            drum_i = int(cs.drum[b, h].item())
            drum = names[drum_i] if 0 <= drum_i < len(names) else f"?{drum_i}"
            crossed = bool(cs.crossed[b, h])
            in_disk = bool(cs.in_disk[b, h])
            axial = float(cs.axial_m[b, h])
            inplane = float(cs.inplane_m[b, h])
            if in_disk:
                kind = "hit"
            elif crossed:
                kind = "crossed_out"
            elif axial > 0.02:
                kind = "short"
            elif axial < -0.02:
                kind = "through"
            else:
                kind = "near_plane"
            events.append({
                "step": step, "env": b, "hand": HANDS[h], "drum": drum,
                "kind": kind, "crossed": crossed, "in_disk": in_disk,
                "axial_mm": 1000.0 * axial,
                "inplane_mm": 1000.0 * inplane,
                "axial_exec_mm": 1000.0 * float(ax_ex[b, h]),
                "inplane_exec_mm": 1000.0 * float(ip_ex[b, h]),
                "axial_plan_mm": 1000.0 * float(ax_plan[b, h]),
                "inplane_plan_mm": 1000.0 * float(ip_plan[b, h]),
                "speed": float(cs.speed[b, h]),
                "tip_exec": [float(x) for x in p_ex[b, h].tolist()],
                "tip_plan": [float(x) for x in p_plan[b, h].tolist()],
                "p_star": [float(x) for x in ctx.p_star[b, h].tolist()],
                "extra": float(extra[b]) if extra is not None else 0.0,
                "plan_collision": coll,
                "viol_collision": viol,
            })

    elapsed = time.perf_counter() - t0
    n = len(events)
    kinds = Counter(e["kind"] for e in events)
    drums = Counter(e["drum"] for e in events)
    print(f"\n[done] {args.steps} steps x {args.envs} envs in {elapsed:.1f}s  "
          f"fires={n}", flush=True)
    if n == 0:
        print("no scheduled fires — try more --steps")
        return 0

    def pct(k):
        return 100.0 * kinds[k] / n

    print("outcome at FIRE (scheduled hit consumed this tick):")
    for k in ("hit", "crossed_out", "short", "through", "near_plane"):
        print(f"  {k:12s} {kinds[k]:5d}  {pct(k):5.1f}%")
    print("drums:", dict(drums))

    ax = np.array([e["axial_mm"] for e in events])
    ip = np.array([e["inplane_mm"] for e in events])
    axp = np.array([e["axial_plan_mm"] for e in events])
    ipp = np.array([e["inplane_plan_mm"] for e in events])
    extra_frac = np.mean([e["extra"] > 0 for e in events])
    print("\nhead-frame at start-of-tick (mm):  axial>0 = SHORT of head, "
          "inplane>0 = outside disk")
    print(f"  executed axial  p50={np.median(ax):+.1f}  p90={np.quantile(ax,0.9):+.1f}  "
          f"mean={ax.mean():+.1f}")
    print(f"  executed inplane p50={np.median(ip):+.1f}  p90={np.quantile(ip,0.9):+.1f}  "
          f"mean={ip.mean():+.1f}")
    print(f"  PLAN     axial  p50={np.median(axp):+.1f}  p90={np.quantile(axp,0.9):+.1f}  "
          f"mean={axp.mean():+.1f}")
    print(f"  PLAN     inplane p50={np.median(ipp):+.1f}  p90={np.quantile(ipp,0.9):+.1f}  "
          f"mean={ipp.mean():+.1f}")
    dtip = np.array([
        np.linalg.norm(np.array(e["tip_exec"]) - np.array(e["tip_plan"]))
        for e in events])
    print(f"  ||tip_exec - tip_plan|| p50={1000*np.median(dtip):.1f} mm  "
          f"p90={1000*np.quantile(dtip,0.9):.1f} mm")
    print(f"  extra-hit on that env-tick: {100*extra_frac:.1f}% of fires")
    print(f"  plan viol.collision mean {np.mean([e['viol_collision'] for e in events]):.4f}")

    by_hand = {}
    for hand in HANDS:
        sub = [e for e in events if e["hand"] == hand]
        if not sub:
            continue
        by_hand[hand] = {
            "n": len(sub),
            "hit_pct": 100.0 * sum(e["kind"] == "hit" for e in sub) / len(sub),
            "short_pct": 100.0 * sum(e["kind"] == "short" for e in sub) / len(sub),
            "axial_p50_mm": float(np.median([e["axial_mm"] for e in sub])),
            "inplane_p50_mm": float(np.median([e["inplane_mm"] for e in sub])),
        }
    print("by hand:", json.dumps(by_hand, indent=2))

    # a few concrete examples
    print("\nexamples (first 8 fires):")
    for e in events[:8]:
        te = e["tip_exec"]
        ps = e["p_star"]
        print(f"  {e['hand']:5s} {e['drum']:14s} {e['kind']:12s}  "
              f"axial={e['axial_mm']:+6.1f}mm inplane={e['inplane_mm']:+6.1f}mm  "
              f"tip=({te[0]:+.3f},{te[1]:+.3f},{te[2]:+.3f})  "
              f"aim=({ps[0]:+.3f},{ps[1]:+.3f},{ps[2]:+.3f})")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n": n, "kinds": dict(kinds), "drums": dict(drums),
            "by_hand": by_hand, "events": events[:500],
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
