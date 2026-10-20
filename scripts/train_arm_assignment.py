#!/usr/bin/env python
"""20-update graph warm start, followed by assignment PPO in Isaac.

    python scripts/train_arm_assignment.py --device cuda:0 --iterations 100

The only learned outputs are a categorical hand choice and a value estimate.
The current clean-line trajectory checkpoint and its committed-plan PID stay
frozen. All drums can be assigned to either arm; there are no action masks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="arm_assignment_mlp_20261002")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="runs/phase2_clean_lines_fromp1_20261001/last.pt")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--episodes-per-update", type=int, default=4)
    p.add_argument("--pretrain-steps", type=int, default=20)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--rl-lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lookahead-s", type=float, default=2.0)
    p.add_argument("--window-s", type=float, default=5.0)
    p.add_argument("--eval-seconds", type=float, default=13.0)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--train-dir", default="assets/rcc/train")
    p.add_argument("--eval-clip", default="assets/rcc/eval/groove_eval_drummer1_session2_62_punk_144_beat_4-4_seg04.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pretrain-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
    from amortized_opt.fk import ChainFK
    from amortized_opt.rhythm import DrumKit, load_rcc
    from isaac_rl.arm_assignment import AssignmentConfig, AssignmentPolicy, AssignmentQueue
    from isaac_rl.assignment_train import read_tips

    if args.lookahead_s <= 0 or args.window_s <= 0 or args.episodes_per_update < 1:
        raise ValueError("lookahead, window and episode count must be positive")
    run_dir = ROOT/"runs"/args.name
    if not (ROOT/args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    kit = DrumKit.from_layout()
    pcfg = AssignmentConfig(num_drums=kit.num_drums, hidden=args.hidden, lookahead_s=args.lookahead_s)
    clips, skipped = [], []
    for path in sorted((ROOT/args.train_dir).glob("*.json")):
        clip = load_rcc(path, kit, on_ineligible="keep")
        if not clip.hits:
            continue
        try:
            AssignmentQueue(clip, kit, pcfg)
        except ValueError as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        clips.append((path, clip))
    if not clips:
        raise ValueError("no usable training clips")
    cfg = dict(vars(args), run_dir=str(run_dir), policy_config=asdict(pcfg),
               training_clips=[str(p) for p, _ in clips], skipped_clips=skipped,
               critic=True, bellman=True, drum_action_masks=False,
               gamma_per_second=0.99, tail_s=0.6, hit_reward=10.0, miss_penalty=5.0,
               stray_penalty=2.0, dense_weight=0.2, joint_penalty=5.0,
               torque_penalty=0.1, collision_penalty=20.0,
               eval_clip=str((ROOT/args.eval_clip).resolve()),
               frozen_planner=str((ROOT/args.ckpt).resolve()),
               controller={"commit_plan": True, "plan_max_passes": 1,
                           "plan_acceptance": "isaac", "fresh_terminal_every_tick": True,
                           "ki": 5.0, "ki_windup": 0.25, "cmd_lead_s": "fit_kd4",
                           "contact_compliance": True, "waist_reaction_ff": True})
    if args.dry_run:
        print(json.dumps({k: v for k, v in cfg.items() if k not in ("training_clips", "skipped_clips")}, indent=2))
        print(f"training clips={len(clips)}, excluded >2-strike clips={len(skipped)}, obs_dim={pcfg.obs_dim}")
        return 0
    # Refuse to overwrite a trained run, including its initial warm start.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/"config.json").write_text(json.dumps(cfg, indent=2))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = AssignmentPolicy(pcfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.pretrain_lr)
    writer = SummaryWriter(str(run_dir/"tb"))
    if os.environ.get("TB_AUTO_REGISTER", "1") != "0":
        registry = ROOT/"runs/tensorboard_active"
        registry.mkdir(parents=True, exist_ok=True)
        (registry/args.name).symlink_to((run_dir/"tb").resolve())
    q = torch.tensor(ready_joint_pos(G1_UPPER_BODY_JOINTS), dtype=torch.float32)
    qd = torch.zeros_like(q)
    fk = ChainFK()
    tips = read_tips(fk, q)
    xs, ys = [], []
    # A brief behavior-cloning warm start, using the actual current graph code.
    # Its eligibility preference is only in the labels, never in the actor.
    while len(xs) < 4096:
        _, clip = clips[int(rng.integers(len(clips)))]
        queue = AssignmentQueue(clip, kit, pcfg)
        for group in queue.groups:
            now = max(0.0, queue.hits[group[0]].time-pcfg.lookahead_s)
            xs.append(queue.observation(now, q.numpy(), qd.numpy(), tips))
            action = queue.teacher_action()
            ys.append(action)
            queue.commit(action)
            if len(xs) >= 4096:
                break
    x = torch.as_tensor(np.stack(xs))
    y = torch.tensor(ys)
    history = []
    for step in range(1, args.pretrain_steps+1):
        ids = torch.randint(len(x), (256,))
        dist, _ = policy(x[ids])
        loss = -dist.log_prob(y[ids]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        with torch.no_grad():
            dist, _ = policy(x[ids])
            accuracy = float((dist.probs.argmax(-1) == y[ids]).float().mean())
            entropy = float(dist.entropy().mean())
        history.append({"step": step, "loss": float(loss), "accuracy": accuracy, "entropy": entropy})
        for key, value in history[-1].items():
            writer.add_scalar("pretrain/"+key, value, step)
    torch.save({"format": "arm_assignment_v1", "policy": policy.state_dict(),
                "policy_config": asdict(pcfg), "pretrain_steps": args.pretrain_steps,
                "config": cfg}, run_dir/"pretrained.pt")
    (run_dir/"pretrain.json").write_text(json.dumps(history, indent=2))
    writer.flush()
    writer.close()
    print(f"[assignment pretrain] {args.pretrain_steps} updates, "
          f"{sum(p.numel() for p in policy.parameters())} parameters, final={history[-1] if history else None}", flush=True)
    if args.pretrain_only:
        return 0
    # Use the full player environment: its contact detector and credit rules
    # are what the earlier committed-plan videos and comparisons measured.
    seed_path, _ = max(clips, key=lambda item: len(item[1].hits))
    max_seconds = max(args.eval_seconds+cfg["tail_s"], args.window_s+1+cfg["tail_s"])
    command = [sys.executable, "-u", str(ROOT/"scripts/play_isaac.py"), "--headless",
               "--device", args.device, "--model_device", args.device, "--ckpt", cfg["frozen_planner"],
               "--rhythm_path", str(seed_path), "--first_hit_s", "1",
               "--steps", str(int(np.ceil(max_seconds/0.02))+5),
               "--commit_plan", "--plan_max_passes", "1", "--plan_acceptance", "isaac",
               "--fresh-terminal-every-tick", "--cmd_lead_s", "fit_kd4", "--ki", "5",
               "--ki_windup", "0.25", "--contact_compliance", "--no_overlay",
               "--hit_diagnostics", str(run_dir/"last_episode_diagnostics.json"),
               "--assignment_train_config", str(run_dir/"config.json")]
    (run_dir/"command.json").write_text(json.dumps(command, indent=2))
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
