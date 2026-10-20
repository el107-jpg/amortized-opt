"""Parallel Isaac collection for the discrete assignment Double DQN critic."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from amortized_opt.fk import stick_tip_offset
from amortized_opt.rhythm import load_rcc, pad_first_hit_to
from isaac_rl.arm_assignment import AssignmentQueue
from isaac_rl.assignment_train import window_clip
from isaac_rl.assignment_dqn import AssignmentQPolicy, DoubleDQN, Replay
from isaac_rl.schedule import clip_hit_table


def load_with_tempo(path, kit, policy_cfg):
    clip = load_rcc(path, kit, on_ineligible="keep")
    bpm = float(json.loads(Path(path).read_text()).get("metadata", {}).get("original_bpm", 120))
    lookahead = max(policy_cfg.lookahead_s, policy_cfg.lookahead_beats*60/max(bpm, 1))
    return clip, replace(policy_cfg, lookahead_s=lookahead)


class ParallelAssignments:
    def __init__(self, u, mpc, policy, clips, configs, *, epsilon=0, graph=False):
        self.u, self.mpc, self.policy = u, mpc, policy
        self.queues = [AssignmentQueue(c, u.assignment_kit, pc) for c, pc in zip(clips, configs)]
        self.records = [[] for _ in clips]
        self.epsilon, self.graph = epsilon, graph
        self.fk = mpc._chain_fk()
        self.offsets = {h: torch.as_tensor(stick_tip_offset(h), device=mpc.device, dtype=torch.float32)
                        for h in ("left", "right")}

    def install(self):
        u = self.u
        u.beat_valid.zero_(); u.beat_time.fill_(1e6); u.beat_hand.zero_(); u.beat_drum_id.zero_()
        for b, queue in enumerate(self.queues):
            n = len(queue.hits)
            if n > u.beat_valid.shape[1]:
                raise ValueError("requested rhythm exceeds allocated beat slots")
            drums = torch.tensor([h.drum for h in queue.hits], device=u.device)
            u.beat_time[b, :n] = torch.tensor([h.time for h in queue.hits], device=u.device)
            u.beat_drum_id[b, :n] = drums
            u.beat_velocity[b, :n] = torch.tensor([h.velocity for h in queue.hits], device=u.device)
            u.beat_pos[b, :n] = u.drum_pos_env[b, drums]
            u.beat_drum_height[b, :n] = u.drum_height[drums]
            u.beat_dipole_offset[b, :n] = u.drum_dipole_offset[drums]
        u.beat_is_left.zero_(); u.beat_is_right.zero_()

    @torch.no_grad()
    def advance(self, now, q, qd):
        due = [b for b, queue in enumerate(self.queues) if queue.due(now)]
        if not due:
            return
        q_cpu, qd_cpu = q.detach().cpu().numpy(), qd.detach().cpu().numpy()
        tips = []
        for hand in ("left", "right"):
            R, p = self.fk.frame(q, hand+"_wrist")
            tips.append(p+torch.einsum("bij,j->bi", R, self.offsets[hand]))
        tips = torch.stack(tips, 1).detach().cpu().numpy()
        changed = set()
        while due:
            obs = np.stack([self.queues[b].observation(now, q_cpu[b], qd_cpu[b], tips[b]) for b in due])
            values = self.policy(torch.as_tensor(obs, device=q.device))
            choices = values.argmax(1)
            if self.epsilon:
                explore = torch.rand(len(due), device=q.device) < self.epsilon
                choices = torch.where(explore, torch.randint(2, (len(due),), device=q.device), choices)
            choices = choices.detach().cpu().numpy()
            for j, b in enumerate(due):
                queue = self.queues[b]
                action = queue.teacher_action() if self.graph else int(choices[j])
                self.records[b].append(dict(obs=obs[j].copy(), action=action, time=now, reward=0., discount=1.))
                for i, hit in queue.commit(action):
                    self.u.beat_hand[b, i] = int(hit.hand == "right")
                    self.u.beat_valid[b, i] = True
                changed.add(b)
            due = [b for b in due if self.queues[b].due(now)]
        sched = self.mpc.sched
        for b in changed:
            tab = clip_hit_table(self.queues[b].assigned_clip(), self.u.assignment_kit, None, cfg=sched.cfg)
            for name in ("hit_t", "hit_drum", "hit_pos", "hit_vel", "hit_valid"):
                dest = getattr(sched, name)
                dest[b].copy_(torch.as_tensor(tab[name], device=dest.device, dtype=dest.dtype))
        sched._sync_cursor()
        self.mpc._enc = None; self.mpc._ctx = None; self.mpc._enc_key = None
        self.u.beat_is_left.copy_((self.u.beat_hand == 0) & self.u.beat_valid)
        self.u.beat_is_right.copy_((self.u.beat_hand == 1) & self.u.beat_valid)
        self.u._field_cache_handoff = False
        self.u._field_cache_stale = torch.tensor(sorted(changed), device=self.u.device)


def make_start_sampler(mpc, pid, cfg):
    if not cfg.get("start_pose_pool"):
        return None
    from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
    from amortized_opt.constants import ready_joint_pos
    pool = np.load(cfg["start_pose_pool"], allow_pickle=False)
    if tuple(pool["joint_names"]) != tuple(mpc.names):
        raise ValueError("starting pose pool joint order differs from planner")
    def clearance(poses):
        parts = [pose_clearance(mpc.descent_cost, torch.as_tensor(
            poses[i:i+1024], device=mpc.device, dtype=torch.float32)).cpu().numpy()
                 for i in range(0, len(poses), 1024)]
        return np.concatenate(parts)
    return StartingPoseSampler(pool["poses"], ready_joint_pos(mpc.names),
                               pid.q_lo.cpu().numpy(), pid.q_hi.cpu().numpy(),
                               clearance=clearance, names=mpc.names)


def collect(env, mpc, pid, policy, clips, configs, cfg, *, epsilon=0, graph=False, progress=None,
            start_poses=None):
    u = env.unwrapped
    B = len(clips)
    durations = np.array([int(math.ceil(c.duration/mpc.dt)) for c in clips])
    tick_rewards = [[] for _ in clips]
    reports = [None]*B
    timings = dict(assignment=0., planner=0., controller=0., isaac=0., reward=0.)
    deadline_hits = {}
    for b, clip in enumerate(clips):
        for i, hit in enumerate(sorted(clip.hits, key=lambda h: (h.time, h.drum))):
            deadline_hits.setdefault(int(round(hit.time/mpc.dt)), []).append((b, i))
    with torch.inference_mode():
        if not hasattr(u, "assignment_nominal_ready"):
            u.assignment_nominal_ready = u.ready_joint_pos.clone()
        # Let the original task reset all servo histories and strike trackers
        # consistently from the chosen pose. Validation always restores nominal.
        u.ready_joint_pos.copy_(u.assignment_nominal_ready)
        if start_poses is not None:
            poses = torch.as_tensor(start_poses, device=u.device, dtype=torch.float32)
            if poses.shape != (B, len(pid.ids)) or not torch.isfinite(poses).all():
                raise ValueError("invalid randomized starting poses")
            u.ready_joint_pos[:, pid.ids] = poses
        env.reset()
        empty = [replace(c, hits=[]) for c in clips]
        ids = torch.arange(B, device=mpc.device)
        mpc.sched.reset(ids, empty, [u.assignment_kit]*B)
        u.env_max_steps.fill_(int(durations.max())+2)
        queues = ParallelAssignments(u, mpc, policy, clips, configs, epsilon=epsilon, graph=graph)
        queues.install()
        q = u.robot.data.joint_pos[:, pid.ids].float()
        qd = u.robot.data.joint_vel[:, pid.ids].float()
        queues.advance(0., q, qd)
        mpc.reset(ids, q)
        pid.reset(q)
        prev_done = u.beat_done.clone(); prev_expired = u.beat_expired.clone(); prev_extra = u.episode_extra.clone()
        zero_action = torch.zeros(B, u.cfg.action_space, device=u.device)
        for tick in range(int(durations.max())):
            now = tick*mpc.dt
            t = time.perf_counter()
            queues.advance(now, q, qd)
            timings["assignment"] += time.perf_counter()-t
            t = time.perf_counter()
            cmd, hit_index = mpc.plan(q, qd)
            timings["planner"] += time.perf_counter()-t
            t = time.perf_counter()
            pid.prepare(q, cmd, hit_index)
            timings["controller"] += time.perf_counter()-t
            t = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(zero_action)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("unexpected Isaac reset during parallel assignment collection")
            timings["isaac"] += time.perf_counter()-t
            t = time.perf_counter()
            q = u.robot.data.joint_pos[:, pid.ids].float()
            qd = u.robot.data.joint_vel[:, pid.ids].float()
            fired = mpc.sched.advance()
            pid.finish(fired)
            valid = u.beat_valid
            new_done = u.beat_done & ~prev_done & valid
            expired = u.beat_expired & ~prev_expired & valid
            stray = (u.episode_extra-prev_extra).clamp_min(0)
            rewards = cfg["hit_reward"]*new_done.sum(1)-cfg["miss_penalty"]*expired.sum(1)-cfg["stray_penalty"]*stray
            prev_done.copy_(u.beat_done); prev_expired.copy_(u.beat_expired); prev_extra.copy_(u.episode_extra)
            joint_excess = torch.maximum((q-pid.q_hi).clamp_min(0), (pid.q_lo-q).clamp_min(0)).amax(1)
            tau = u.robot.data.computed_torque[:, pid.ids]
            torque_excess = (tau.abs()/pid.tau_lim-1).clamp_min(0).amax(1)
            rewards -= cfg["joint_penalty"]*joint_excess*mpc.dt
            rewards -= cfg["torque_penalty"]*torque_excess.clamp_max(10)*mpc.dt
            rewards -= cfg["collision_penalty"]*mpc.collision.clamp_max(1)
            # Deadline-only FK shaping uses the same geometry helper as the
            # previous runner, without building any per-tick playback trace.
            for b, i in deadline_hits.get(tick+1, []):
                hit = queues.queues[b].hits[i]
                geom = cfg["geometry_scorer"](q[b], queues.queues[b].hands[i], hit.drum)
                axial = max(0., -geom["axial_mm"], geom["axial_mm"]-10.)
                gap = math.hypot(axial, geom["lateral_beyond_mm"])/1000
                rewards[b] += cfg["dense_weight"]/(1+(gap/0.05)**2)
            rewards = rewards.detach().cpu().numpy()
            for b in range(B):
                if tick < durations[b]:
                    tick_rewards[b].append(((tick+1)*mpc.dt, float(rewards[b])))
                if tick+1 == durations[b]:
                    unresolved = u.beat_valid[b] & ~u.beat_done[b] & ~u.beat_expired[b]
                    tick_rewards[b].append(((tick+1)*mpc.dt, -cfg["miss_penalty"]*int(unresolved.sum())))
                    drum_rows = {}
                    for d, name in enumerate(u.assignment_kit.names):
                        mask = u.beat_valid[b] & (u.beat_drum_id[b] == d)
                        drum_rows[name] = dict(requested=int(mask.sum()), hits=int((mask & u.beat_done[b]).sum()),
                                               left=int((mask & (u.beat_hand[b] == 0)).sum()))
                    reports[b] = dict(clip=clips[b].name, return_=sum(r for _, r in tick_rewards[b]),
                                      hits=int((u.beat_done[b] & u.beat_valid[b]).sum()), requested=len(clips[b].hits),
                                      strays=float(u.episode_extra[b]), steps=int(durations[b]),
                                      decoder_passes=int(mpc.decoder_passes[b]), by_drum=drum_rows,
                                      assignments=[dict(time=h.time, drum=h.drum_name, hand=queues.queues[b].hands[i])
                                                   for i, h in enumerate(queues.queues[b].hits)])
            timings["reward"] += time.perf_counter()-t
            if progress and (tick+1) % 50 == 0:
                progress(tick+1, int(durations.max()))
        gamma = cfg["gamma_per_second"]
        for b, records in enumerate(queues.records):
            for i, record in enumerate(records):
                end = records[i+1]["time"] if i+1 < len(records) else float("inf")
                record["reward"] = sum(gamma**max(0., t-record["time"])*r for t, r in tick_rewards[b]
                                       if record["time"] < t <= end)
                record["discount"] = gamma**(end-record["time"]) if math.isfinite(end) else 0.
            if len(queues.queues[b].hands) != len(clips[b].hits):
                raise RuntimeError("unassigned requested strike")
            reports[b]["decisions"] = len(records)
    return queues.records, reports, timings


def train(env, mpc, pid, cfg, policy, *, checkpoint=None, reset_replay=False):
    from torch.utils.tensorboard import SummaryWriter
    run_dir = Path(cfg["run_dir"])
    learner = DoubleDQN(policy, lr=cfg["rl_lr"])
    replay = Replay(policy.cfg.obs_dim, capacity=cfg.get("replay_capacity", 50000), seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    start_update = 1
    if checkpoint and checkpoint.get("format") == "arm_assignment_dqn_v1":
        learner.target.load_state_dict(checkpoint["target"])
        learner.optimizer.load_state_dict(checkpoint["optimizer"])
        learner.updates = checkpoint["gradient_updates"]
        if not reset_replay:
            replay.load_state_dict(checkpoint["replay"])
        rng.bit_generator.state = checkpoint["rng"]
        start_update = checkpoint["iteration"]+1
    writer = SummaryWriter(str(run_dir/"tb"))
    metrics = (run_dir/"dqn_metrics.jsonl").open("a", buffering=1)
    clips = [load_with_tempo(path, env.unwrapped.assignment_kit, policy.cfg) for path in cfg["training_clips"]]
    sampler = None
    if cfg.get("window_sampling", "uniform") == "balanced":
        from isaac_rl.assignment_sampling import BalancedWindowSampler
        sampler = BalancedWindowSampler([c for c, _ in clips], len(env.unwrapped.assignment_kit.names),
            span_s=cfg["window_s"], deduplicate=cfg.get("deduplicate_windows", False),
            max_probability=cfg.get("max_window_probability"),
            seed=7201+cfg["seed"]+9973*start_update if cfg.get("refresh_balanced_windows", False) else 7201)
        (run_dir/"window_sampling.json").write_text(json.dumps(
            dict(drums=env.unwrapped.assignment_kit.names, **sampler.summary), indent=2))
        print(f"[assignment sampling] {sampler.summary}", flush=True)
    B = env.unwrapped.num_envs
    start_sampler = make_start_sampler(mpc, pid, cfg)
    if start_sampler is not None:
        (run_dir/"start_pose_sampling.json").write_text(json.dumps(dict(
            pool=cfg["start_pose_pool"], valid_anchors=len(start_sampler.anchors),
            exact_ready_fraction=start_sampler.ready_fraction,
            perturbed_anchor_fraction=start_sampler.anchor_fraction,
            perturbed_ready_fraction=1-start_sampler.ready_fraction-start_sampler.anchor_fraction,
            joint_names=list(mpc.names), noise_std_rad=start_sampler.std.tolist(),
            noise_clip_std=2.5, min_clearance_m=start_sampler.margin,
            max_pitch_roll_rad=.08, initial_velocity="zero"), indent=2))
        print(f"[assignment starts] {len(start_sampler.anchors)} collision-clear executed anchors", flush=True)
    heldout, eval_pc = load_with_tempo(cfg["eval_clip"], env.unwrapped.assignment_kit, policy.cfg)
    heldout, _ = pad_first_hit_to(heldout, 1.0)
    heldout = replace(heldout, hits=[h for h in heldout.hits if h.time <= cfg["eval_seconds"]],
                      duration=cfg["eval_seconds"]+cfg["tail_s"])
    eval_clips = []
    if start_sampler is not None:
        excluded = []
        for path in sorted(Path(cfg["eval_clip"]).parent.glob("*.json")):
            clip, pc = load_with_tempo(path, env.unwrapped.assignment_kit, policy.cfg)
            try:
                AssignmentQueue(clip, env.unwrapped.assignment_kit, pc)
            except ValueError as exc:
                excluded.append(dict(path=str(path), reason=str(exc)))
                continue
            if clip.hits:
                eval_clips.append((path, clip, pc))
        if not eval_clips:
            raise ValueError("no eligible held-out rhythms for varied-start validation")
        (run_dir/"validation_clips.json").write_text(json.dumps(dict(
            accepted=[str(p) for p, _, _ in eval_clips], excluded=excluded,
            optimized_on=False), indent=2))

    def save(iteration):
        blob = dict(format="arm_assignment_dqn_v1", policy=policy.state_dict(), policy_config=asdict(policy.cfg),
                    target=learner.target.state_dict(), optimizer=learner.optimizer.state_dict(),
                    gradient_updates=learner.updates, iteration=iteration, replay=replay.state_dict(),
                    rng=rng.bit_generator.state, config={k: v for k, v in cfg.items() if k != "geometry_scorer"})
        tmp = run_dir/"dqn_last.tmp"
        torch.save(blob, tmp)
        tmp.replace(run_dir/"dqn_last.pt")

    print(f"[assignment Double DQN] {B} parallel Isaac environments, "
          f"{sum(p.numel() for p in policy.parameters())} parameters, "
          f"{policy.cfg.lookahead_beats:g} beats / {policy.cfg.future_hits} upcoming strikes", flush=True)
    try:
        for iteration in range(start_update, cfg["iterations"]+1):
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(policy.q_head.weight.device)
            if sampler is None:
                picks = [clips[int(rng.integers(len(clips)))] for _ in range(B)]
                windows = [window_clip(c, rng, span_s=cfg["window_s"], tail_s=cfg["tail_s"]) for c, _ in picks]
            else:
                if cfg.get("refresh_balanced_windows", False) and iteration > start_update:
                    sampler = BalancedWindowSampler([c for c, _ in clips], len(env.unwrapped.assignment_kit.names),
                        span_s=cfg["window_s"], seed=7201+cfg["seed"]+9973*iteration,
                        deduplicate=cfg.get("deduplicate_windows", False),
                        max_probability=cfg.get("max_window_probability"))
                    (run_dir/"window_sampling.json").write_text(json.dumps(
                        dict(iteration=iteration, drums=env.unwrapped.assignment_kit.names, **sampler.summary), indent=2))
                indexes, windows = sampler.sample(rng, B, tail_s=cfg["tail_s"])
                picks = [clips[i] for i in indexes]
                writer.add_scalar("sampling/unique_source_clips", len(set(indexes)), iteration)
                patterns = {tuple((round(h.time-c.hits[0].time, 8), h.drum, round(h.velocity, 6))
                                  for h in c.hits) for c in windows}
                writer.add_scalar("sampling/unique_note_patterns", len(patterns), iteration)
                writer.add_scalar("sampling/effective_candidate_windows", sampler.summary["effective_windows"], iteration)
                print(f"[assignment rhythm diversity] update={iteration} clips={len(set(indexes))} "
                      f"patterns={len(patterns)} pool={len(sampler.candidates)}", flush=True)
            start_poses = None
            if start_sampler is not None:
                start_poses, start_stats = start_sampler.sample(rng, B)
                for name, value in start_stats.items():
                    writer.add_scalar("starting_pose/"+name, value, iteration)
                print(f"[assignment starting poses] update={iteration} {start_stats}", flush=True)
            exposure = replay.size+cfg.get("exploration_decisions_offset", 0)
            epsilon = max(0.05, 0.3-0.25*exposure/cfg.get("exploration_decay_decisions", 20000))
            requested_counts = np.bincount([h.drum for c in windows for h in c.hits],
                                          minlength=len(env.unwrapped.assignment_kit.names))
            for drum, count in zip(env.unwrapped.assignment_kit.names, requested_counts):
                writer.add_scalar(f"sampling/{drum}_requested", count, iteration)
                writer.add_scalar(f"sampling/{drum}_share", count/requested_counts.sum(), iteration)
            writer.flush()
            def progress(tick, maximum):
                writer.add_scalar("collection/control_tick", tick, iteration*1000+tick)
                writer.add_scalar("collection/iteration", iteration, iteration*1000+tick)
                writer.flush()
                print(f"[assignment collect] update={iteration} tick={tick}/{maximum} envs={B}", flush=True)
            episodes, reports, parts = collect(env, mpc, pid, policy, windows, [pc for _, pc in picks], cfg,
                                               epsilon=epsilon, progress=progress, start_poses=start_poses)
            torch.cuda.synchronize(policy.q_head.weight.device)
            collect_seconds = time.perf_counter()-start
            for e in episodes:
                replay.add_episode(e)
            t = time.perf_counter()
            stats = learner.update(replay, gradient_steps=cfg.get("gradient_steps", 32))
            torch.cuda.synchronize(policy.q_head.weight.device)
            update_seconds = time.perf_counter()-t
            stats.update(iteration=iteration, epsilon=epsilon, environments=B,
                         collection_seconds=collect_seconds, update_seconds=update_seconds,
                         seconds=collect_seconds+update_seconds,
                         environment_steps=sum(r["steps"] for r in reports),
                         simulated_environment_seconds=sum(r["steps"] for r in reports)*mpc.dt,
                         return_mean=float(np.mean([r["return_"] for r in reports])),
                         hit_fraction=sum(r["hits"] for r in reports)/sum(r["requested"] for r in reports),
                         hits=sum(r["hits"] for r in reports), requested=sum(r["requested"] for r in reports),
                         **{f"requested_{drum}": int(count) for drum, count in
                            zip(env.unwrapped.assignment_kit.names, requested_counts)},
                         peak_torch_allocated_gib=torch.cuda.max_memory_allocated(policy.q_head.weight.device)/2**30,
                         peak_torch_reserved_gib=torch.cuda.max_memory_reserved(policy.q_head.weight.device)/2**30,
                         same_instant_transition_fraction=sum(r["discount"] == 1 for e in episodes for r in e)/max(1, sum(map(len, episodes))),
                         zero_reward_transition_fraction=sum(r["reward"] == 0 for e in episodes for r in e)/max(1, sum(map(len, episodes))),
                         initial_decision_fraction=sum(r["time"] == 0 for e in episodes for r in e)/max(1, sum(map(len, episodes))),
                         **{f"time_{k}": v for k, v in parts.items()})
            metrics.write(json.dumps(stats)+"\n")
            for key, value in stats.items():
                writer.add_scalar("dqn/"+key, value, iteration)
            for drum in env.unwrapped.assignment_kit.names:
                rows = [r["by_drum"][drum] for r in reports]
                n = sum(r["requested"] for r in rows)
                if n:
                    writer.add_scalar(f"assignment/{drum}_left_fraction", sum(r["left"] for r in rows)/n, iteration)
                    writer.add_scalar(f"hits/{drum}_fraction", sum(r["hits"] for r in rows)/n, iteration)
            writer.flush()
            (run_dir/"dqn_last_rollouts.json").write_text(json.dumps(reports, indent=2))
            save(iteration)
            print(f"[assignment Double DQN update] {stats}", flush=True)
            if iteration % cfg["save_interval"] == 0 or iteration == cfg["iterations"]:
                validation_windows, validation_configs = [heldout]*B, [eval_pc]*B
                validation_starts = None
                groups = {"nominal": slice(0, B)}
                if start_sampler is not None and B >= 3:
                    # Fixed independent RNG: evaluation does not change future
                    # training draws. All three cohorts share one Isaac pass.
                    eval_rng = np.random.default_rng(cfg["seed"]+100000)
                    n = B//3
                    groups = {"nominal": slice(0, n), "random_start": slice(n, 2*n),
                              "varied_rhythm_random_start": slice(2*n, B)}
                    validation_starts, _ = start_sampler.sample(eval_rng, B)
                    validation_starts[:n] = start_sampler.ready
                    for b in range(2*n, B):
                        _, clip, pc = eval_clips[int(eval_rng.integers(len(eval_clips)))]
                        validation_windows[b] = window_clip(clip, eval_rng, span_s=cfg["window_s"], tail_s=cfg["tail_s"])
                        validation_configs[b] = pc
                _, validation, _ = collect(env, mpc, pid, policy, validation_windows, validation_configs, cfg,
                                           start_poses=validation_starts)
                nominal = validation[groups["nominal"]]
                hit_fraction = sum(r["hits"] for r in nominal)/sum(r["requested"] for r in nominal)
                writer.add_scalar("dqn_eval/learned_hit_fraction", hit_fraction, iteration)
                for name, cohort in groups.items():
                    rows = validation[cohort]
                    fraction = sum(r["hits"] for r in rows)/sum(r["requested"] for r in rows)
                    writer.add_scalar(f"dqn_eval/{name}_hit_fraction", fraction, iteration)
                    for drum in env.unwrapped.assignment_kit.names:
                        requested = sum(r["by_drum"][drum]["requested"] for r in rows)
                        if requested:
                            writer.add_scalar(f"dqn_eval/{name}/{drum}_fraction",
                                sum(r["by_drum"][drum]["hits"] for r in rows)/requested, iteration)
                    for b in range(cohort.start, cohort.stop):
                        validation[b]["validation_group"] = name
                    print(f"[assignment validation] update={iteration} cohort={name} hit_fraction={fraction:.4f}", flush=True)
                writer.flush()
                (run_dir/f"dqn_eval_{iteration:04d}.json").write_text(json.dumps(validation, indent=2))
                print(f"[assignment Double DQN eval] update={iteration} hits="
                      f"{[r['hits'] for r in validation]}/{validation[0]['requested']}", flush=True)
        (run_dir/"dqn_completed.json").write_text(json.dumps({"iterations": cfg["iterations"]}))
    finally:
        writer.close(); metrics.close()
