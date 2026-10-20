"""Event-level assignment PPO, using the player's unchanged Isaac hit detector.

Imported only after AppLauncher. The trajectory model, PID, feedforward,
compliance and actuator limits are frozen. This is not trajectory-policy PPO.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from amortized_opt.constants import PELVIS_IN_ENV
from amortized_opt.fk import stick_tip_offset
from amortized_opt.rhythm import RhythmClip, load_rcc
from isaac_rl.arm_assignment import AssignmentConfig, AssignmentPolicy, AssignmentQueue, ppo_update
from isaac_rl.schedule import clip_hit_table


def read_tips(fk, q):
    frames = fk.forward(q.unsqueeze(0))
    tips = []
    for hand in ("left", "right"):
        R, p = frames[hand+"_wrist"]
        offset = torch.as_tensor(stick_tip_offset(hand), device=q.device, dtype=q.dtype)
        tips.append((p[0]+R[0]@offset).detach().cpu().numpy())
    return np.stack(tips)


def install_beats(u, queue):
    """Replace the episode beat table; only assigned entries become valid.

    Drum geometry, timing/speed credit gates and the contact detector stay as
    the player configured them. No load_rcc eligibility reassignment is used.
    """
    n = len(queue.hits)
    if n > u.beat_valid.shape[1]:
        raise ValueError("episode exceeds scene's allocated beat slots")
    u.beat_valid.zero_()
    u.beat_time.fill_(1e6)
    u.beat_hand.zero_()
    u.beat_drum_id.zero_()
    for i, hit in enumerate(queue.hits):
        u.beat_time[0, i] = hit.time
        u.beat_drum_id[0, i] = hit.drum
        u.beat_velocity[0, i] = hit.velocity
        if getattr(u,"beat_target_speed_m_s",None) is not None:
            from amortized_opt.rhythm import strike_speed
            u.beat_target_speed_m_s[0,i] = strike_speed(hit.velocity)
        u.beat_pos[0, i] = u.drum_pos_env[0, hit.drum]
        u.beat_drum_height[0, i] = u.drum_height[hit.drum]
        u.beat_dipole_offset[0, i] = u.drum_dipole_offset[hit.drum]
    sync_credit_hands(u, queue)


def sync_credit_hands(u, queue):
    for i, hand in queue.hands.items():
        u.beat_hand[0, i] = int(hand == "right")
        u.beat_valid[0, i] = True
    u.beat_is_left.copy_((u.beat_hand == 0) & u.beat_valid)
    u.beat_is_right.copy_((u.beat_hand == 1) & u.beat_valid)
    u._field_cache_handoff = False
    u._field_cache_stale = torch.tensor([0], device=u.device)


def sync_planner(agent, queue):
    sched = agent.schedule
    tab = clip_hit_table(queue.assigned_clip(), queue.kit, None, cfg=sched.cfg)
    for name in ("hit_t", "hit_drum", "hit_pos", "hit_vel", "hit_valid"):
        dest = getattr(sched, name)
        dest[0].copy_(torch.as_tensor(tab[name], device=dest.device, dtype=dest.dtype))
    sched._sync_cursor()
    # New futures change context even when neither current goal changed.
    # Invalidate only the encoder, preserving the committed physical plan.
    agent.mpc._enc = None
    agent.mpc._ctx = None
    agent.mpc._enc_key = None
    sync_credit_hands(agent.u, queue)


class OnlineAssignment:
    def __init__(self, agent, policy, clip, *, deterministic=False, graph=False):
        self.agent, self.policy = agent, policy
        queue_cfg = policy.cfg
        if queue_cfg.lookahead_beats:
            bpm = float(getattr(agent, "assignment_bpm", 120))
            queue_cfg = replace(queue_cfg, lookahead_s=max(queue_cfg.lookahead_s,
                                queue_cfg.lookahead_beats*60/max(bpm, 1)))
        self.queue = AssignmentQueue(clip, agent.kit, queue_cfg)
        self.deterministic, self.graph = deterministic, graph
        self.records = []

    @torch.no_grad()
    def advance(self, now):
        if not self.queue.due(now):
            return
        agent = self.agent
        q = agent.u.robot.data.joint_pos[0, agent._joint_ids].to(agent.device).float()
        qd = agent.u.robot.data.joint_vel[0, agent._joint_ids].to(agent.device).float()
        tips = read_tips(agent.mpc._chain_fk(), q)
        while self.queue.due(now):
            obs = self.queue.observation(now, q.detach().cpu().numpy(),
                                         qd.detach().cpu().numpy(), tips)
            output = self.policy(torch.as_tensor(obs, device=agent.device).unsqueeze(0))
            if isinstance(output, tuple):
                dist, value = output
            else:
                # Playback of Double DQN is greedy. The categorical wrapper
                # keeps the shared trace format; no PPO update uses this path.
                dist = torch.distributions.Categorical(logits=output)
                value = output.max(-1).values
            action = (self.queue.teacher_action() if self.graph else
                      int(dist.probs.argmax(-1).item()) if self.deterministic else int(dist.sample().item()))
            self.records.append({"obs": obs.copy(), "action": action,
                                 "log_prob": float(dist.log_prob(torch.tensor([action], device=agent.device))),
                                 "value": float(value.item()), "time": now, "reward": 0.0,
                                 "discount": 1.0, "group": self.queue.next_group})
            self.queue.commit(action)
        sync_planner(agent, self.queue)


def window_clip(clip, rng, *, span_s=5.0, first_hit_s=1.0, tail_s=0.6, start_s=None):
    start = (float(rng.uniform(0, max(0.0, clip.hits[-1].time-span_s)))
             if start_s is None else float(start_s))
    hits = [h for h in clip.hits if start <= h.time < start+span_s]
    if not hits:
        if start_s is not None:
            raise ValueError("explicit training window contains no hits")
        return window_clip(clip, rng, span_s=span_s, first_hit_s=first_hit_s, tail_s=tail_s)
    offset = first_hit_s-hits[0].time
    # Preserve quantized inter-hit gaps exactly.
    hits = [replace(h, time=round(h.time+offset, 8), raw_time=h.raw_time+offset) for h in hits]
    return replace(clip, name=f"{clip.name}@{start:.3f}s", hits=hits,
                   duration=hits[-1].time+tail_s, ineligible=0)


def rollout(env, agent, policy, clip, cfg, *, deterministic=False, graph=False):
    u = env.unwrapped
    steps = int(math.ceil(clip.duration/agent.dt))
    records = []
    tick_rewards = []
    gamma = float(cfg["gamma_per_second"])
    with torch.inference_mode():
        env.reset()
        # Empty scheduler until the learned assignment has supplied the goals.
        agent._clip = replace(clip, hits=[])
        agent.reset()
        online = OnlineAssignment(agent, policy, clip, deterministic=deterministic, graph=graph)
        install_beats(u, online.queue)
        u.env_max_steps.fill_(steps+2)
        online.advance(0.0)
        # Reset seed AFTER installing the first assigned goals.
        q = u.robot.data.joint_pos[0, agent._joint_ids].to(agent.device).float()
        agent.mpc.reset(torch.tensor([0], device=agent.device), q.unsqueeze(0))
        prev_done = u.beat_done[0].clone()
        prev_expired = u.beat_expired[0].clone()
        prev_extra = float(u.episode_extra[0])
        prev_segments = 0
        total_reward = 0.0
        for tick in range(steps):
            now = tick*agent.dt
            online.advance(now)
            action = agent.act(now)
            _, _, terminated, truncated, _ = env.step(action)
            # Auto-reset would destroy terminal contact telemetry. The horizon
            # above prevents timeout resets; unexpected termination is an error.
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("Isaac reset before assignment episode end")
            agent.post_step()
            after = (tick+1)*agent.dt
            valid = u.beat_valid[0]
            new_done = u.beat_done[0] & ~prev_done & valid
            new_expired = u.beat_expired[0] & ~prev_expired & valid
            stray = max(0.0, float(u.episode_extra[0])-prev_extra)
            reward = (cfg["hit_reward"]*int(new_done.sum())
                      - cfg["miss_penalty"]*int(new_expired.sum())
                      - cfg["stray_penalty"]*stray)
            prev_done = u.beat_done[0].clone()
            prev_expired = u.beat_expired[0].clone()
            prev_extra = float(u.episode_extra[0])
            # Small, once-per-deadline continuous signal. This never changes
            # contact credit or replaces the existing Isaac hit requirements.
            for i, hit in enumerate(online.queue.hits):
                if now+1e-7 < hit.time <= after+1e-7:
                    q = u.robot.data.joint_pos[0, agent._joint_ids].float()
                    geom = agent._hit_geometry(q, online.queue.hands[i], hit.drum)
                    axial = max(0.0, -geom["axial_mm"], geom["axial_mm"]-10.0)
                    gap = math.hypot(axial, geom["lateral_beyond_mm"])/1000
                    reward += cfg["dense_weight"]/(1+(gap/0.05)**2)
            q_cpu = u.robot.data.joint_pos[0, agent._joint_ids].detach().cpu().float()
            joint_excess = torch.maximum((q_cpu-agent._q_hi).clamp_min(0),
                                         (agent._q_lo-q_cpu).clamp_min(0)).max().item()
            computed_tau = u.robot.data.computed_torque[0, agent._joint_ids].detach().cpu().float()
            torque_excess = (computed_tau.abs()/agent._tau_lim-1).clamp_min(0).max().item()
            reward -= cfg["joint_penalty"]*joint_excess*agent.dt
            reward -= cfg["torque_penalty"]*min(torque_excess, 10.0)*agent.dt
            # Analytic collision diagnostic on each new full plan, not merely
            # on the executed first waypoint. No actor/planner gradients here.
            segments = agent.mpc.plan_segments
            for seg in segments[prev_segments:]:
                collision = seg["attempts"][-1]["violations"].get("collision", 0.0)
                reward -= cfg["collision_penalty"]*min(collision, 1.0)
            prev_segments = len(segments)
            tick_rewards.append((after, float(reward)))
            total_reward += reward
        if len(online.queue.hands) != len(online.queue.hits):
            raise RuntimeError("some requested hits were never assigned")
        # Close any unresolved beats at terminal exactly once. Tail is at
        # least the detector's expiry margin; this also handles roundoff.
        unresolved = u.beat_valid[0] & ~u.beat_done[0] & ~u.beat_expired[0]
        terminal_miss = -cfg["miss_penalty"]*int(unresolved.sum())
        tick_rewards.append((steps*agent.dt, terminal_miss))
        total_reward += terminal_miss
        records = online.records
        for i, record in enumerate(records):
            end = records[i+1]["time"] if i+1 < len(records) else float("inf")
            # Rewards at the end of a tick belong to the decision that
            # preceded that tick; a new decision at exactly that time happens
            # AFTER that reward. Same-time decisions have empty intervals.
            record["reward"] = sum(gamma**max(0.0, t-record["time"])*r for t, r in tick_rewards
                                   if record["time"] < t <= end)
            record["discount"] = (gamma**(end-record["time"]) if math.isfinite(end) else 0.0)
        by_drum = {}
        for d, name in enumerate(agent.kit.names):
            mask = u.beat_valid[0] & (u.beat_drum_id[0] == d)
            by_drum[name] = {"requested": int(mask.sum()), "hits": int((mask & u.beat_done[0]).sum()),
                            "left": int((mask & (u.beat_hand[0] == 0)).sum())}
        summary = {"clip": clip.name, "return": total_reward,
                   "hits": int((u.beat_done[0] & u.beat_valid[0]).sum()),
                   "requested": len(clip.hits), "strays": float(u.episode_extra[0]),
                   "decisions": len(records), "steps": steps,
                   "by_drum": by_drum, "physics": agent.violation_summary(),
                   "assignments": [{"time": h.time, "drum": h.drum_name,
                                    "hand": online.queue.hands[i]} for i, h in enumerate(online.queue.hits)]}
    return records, summary


def run_training(env, agent, config_path):
    from torch.utils.tensorboard import SummaryWriter
    cfg = json.loads(Path(config_path).read_text())
    run_dir = Path(cfg["run_dir"])
    blob = torch.load(run_dir/"pretrained.pt", map_location=agent.device, weights_only=False)
    policy = AssignmentPolicy(AssignmentConfig(**blob["policy_config"])).to(agent.device)
    policy.load_state_dict(blob["policy"])
    agent.mpc.model.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["rl_lr"])
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    clips = [load_rcc(p, agent.kit, on_ineligible="keep") for p in cfg["training_clips"]]
    writer = SummaryWriter(str(run_dir/"tb"))
    log = (run_dir/"metrics.jsonl").open("a", buffering=1)
    print(f"[assignment] {sum(p.numel() for p in policy.parameters())} parameters; "
          "no eligibility masks; critic/Bellman ON; frozen generator/PID", flush=True)

    def save(name, iteration):
        torch.save({"format": "arm_assignment_v1", "policy": policy.state_dict(),
                    "policy_config": asdict(policy.cfg), "optimizer": optimizer.state_dict(),
                    "iteration": iteration, "config": cfg}, run_dir/name)

    # Holdout is evaluated deterministically, never used for gradient updates.
    eval_clip = load_rcc(cfg["eval_clip"], agent.kit, on_ineligible="keep")
    from amortized_opt.rhythm import pad_first_hit_to
    eval_clip, _ = pad_first_hit_to(eval_clip, 1.0)
    eval_clip = replace(eval_clip, hits=[h for h in eval_clip.hits if h.time <= cfg["eval_seconds"]],
                        duration=cfg["eval_seconds"]+cfg["tail_s"])
    for label, graph in (("graph", True), ("pretrained", False)):
        _, report = rollout(env, agent, policy, eval_clip, cfg, deterministic=True, graph=graph)
        (run_dir/f"eval_{label}.json").write_text(json.dumps(report, indent=2))
        writer.add_scalar(f"eval/{label}_hit_fraction", report["hits"]/max(report["requested"], 1), 0)
        writer.flush()
        print(f"[assignment eval] {label}: {report['hits']}/{report['requested']}", flush=True)

    best = -float("inf")
    try:
        for iteration in range(1, cfg["iterations"]+1):
            start = time.monotonic()
            episodes, reports = [], []
            for _ in range(cfg["episodes_per_update"]):
                clip = clips[int(rng.integers(len(clips)))]
                clip = window_clip(clip, rng, span_s=cfg["window_s"], tail_s=cfg["tail_s"])
                episode, report = rollout(env, agent, policy, clip, cfg)
                episodes.append(episode); reports.append(report)
            stats = ppo_update(policy, optimizer, episodes)
            (run_dir/"last_rollouts.json").write_text(json.dumps(reports, indent=2))
            stats.update(iteration=iteration, return_mean=float(np.mean([r["return"] for r in reports])),
                         hit_fraction=sum(r["hits"] for r in reports)/sum(r["requested"] for r in reports),
                         hits=sum(r["hits"] for r in reports), requested=sum(r["requested"] for r in reports),
                         seconds=time.monotonic()-start)
            log.write(json.dumps(stats)+"\n")
            for key, value in stats.items():
                writer.add_scalar("rl/"+key, value, iteration)
            for drum in agent.kit.names:
                rows = [r["by_drum"][drum] for r in reports]
                n = sum(r["requested"] for r in rows)
                if n:
                    writer.add_scalar(f"assignment/{drum}_left_fraction", sum(r["left"] for r in rows)/n, iteration)
                    writer.add_scalar(f"hits/{drum}_fraction", sum(r["hits"] for r in rows)/n, iteration)
            writer.flush()
            save("last.pt", iteration)
            if stats["return_mean"] > best:
                best = stats["return_mean"]
                save("best_training_return.pt", iteration)
            if iteration % cfg["save_interval"] == 0 or iteration == cfg["iterations"]:
                save(f"step_{iteration:05d}.pt", iteration)
                _, report = rollout(env, agent, policy, eval_clip, cfg, deterministic=True)
                (run_dir/f"eval_{iteration:05d}.json").write_text(json.dumps(report, indent=2))
                writer.add_scalar("eval/learned_hit_fraction", report["hits"]/max(report["requested"], 1), iteration)
                writer.flush()
            print(f"[assignment PPO] {stats}", flush=True)
        (run_dir/"completed.json").write_text(json.dumps({"iterations": cfg["iterations"]}))
    finally:
        writer.close()
        log.close()
