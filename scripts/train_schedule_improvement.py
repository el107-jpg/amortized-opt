#!/usr/bin/env python3
"""Train a full-schedule critic, then distill analytic local-search edits.

Environment: frozen decoder endpoint poses, perfectly tracked Hermite motion,
analytic collision/dynamics checks and randomized contact rebound. No Isaac.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator, SingleHitGenerator, TrajectorySurrogate, hermite
from isaac_rl.assignment_sampling import BalancedWindowSampler
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.assignment_train import window_clip
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_analytic import AnalyticScheduleEnvironment
from isaac_rl.schedule_improvement import (
    MAX_HITS, EDITS, JointScheduleCritic, ScheduleEditPolicy, ScheduleNetConfig,
    cap_window, graph_seed, physical_table, observation, index_obs, suffix_obs,
    suffix_actions, critic_search, policy_search, teacher_edits,
)


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def make_pool(paths, kit, seed, *, candidates=24, allow_missing_drums=False):
    rng = np.random.default_rng(seed)
    clips = [A.load_rcc(path, kit, on_ineligible='keep') for path in paths]
    base = BalancedWindowSampler(clips, kit.num_drums, span_s=.84, seed=seed,
        candidates_per_clip=candidates, deduplicate=True, allow_missing_drums=allow_missing_drums)
    windows, excluded, truncated = [], 0, 0
    for index, start in base.candidates:
        try:
            raw = window_clip(clips[index], rng, span_s=.84, first_hit_s=.16, tail_s=0, start_s=start)
            clipped = cap_window(raw)
            truncated += int(len(clipped.hits) < len(raw.hits))
            windows.append(clipped)
        except ValueError:
            excluded += 1
    # Balance after the physical hit cap, not before it.
    sampler = BalancedWindowSampler(windows, kit.num_drums, span_s=1, seed=seed+1,
        candidates_per_clip=1, deduplicate=True, max_probability=max(.002, 1/len(windows)),
        allow_missing_drums=allow_missing_drums)
    sampler.summary.update(excluded_overlapping_or_empty=excluded, capped_windows=truncated,
        input_horizon_s=1, max_physical_hits=MAX_HITS, maximum_first_hit_s=.16,
        first_hit_offsets='Half 160 ms; half uniformly sampled 20..160 ms on the control grid')
    return sampler


def sample_windows(sampler, rng, count):
    _, clips = sampler.sample(rng, count, first_hit_s=.16, tail_s=0)
    varied = []
    for clip in clips:
        first = .16 if rng.random() < .5 else .02*int(rng.integers(1, 9))
        shift = first-clip.hits[0].time
        hits = [replace(h, time=round(h.time+shift, 8)) for h in clip.hits]
        varied.append(cap_window(replace(clip, hits=hits, duration=hits[-1].time)))
    return varied


def candidate_bank(clips, kit, rng, limit=64):
    banks, seeds, counts, fallbacks = [], [], [], 0
    for clip in clips:
        seed, fallback = graph_seed(clip, kit)
        fallbacks += fallback
        n = len(set(h.time for h in clip.hits))
        choices = [seed.copy()]
        for edit in EDITS[1:]:
            if max(edit) < n:
                row = seed.copy(); row[list(edit)] ^= 1; choices.append(row)
        mirror = seed.copy(); mirror[:n] ^= 1; choices.append(mirror)
        for _ in range(16):
            row = np.zeros(MAX_HITS, np.int64); row[:n] = rng.integers(2, size=n); choices.append(row)
        unique = list(dict.fromkeys(tuple(row) for row in choices))[:limit]
        banks.append(np.asarray(unique, np.int64)); seeds.append(seed); counts.append(len(unique))
    C = max(counts)
    actions = np.stack([np.concatenate((b, np.repeat(b[:1], C-len(b), axis=0))) for b in banks])
    valid = np.arange(C)[None] < np.array(counts)[:, None]
    return actions, valid, np.stack(seeds), fallbacks


def append_observation(storage, prefix, obs):
    for key, value in obs.items():
        storage.setdefault(prefix+key, []).append(value.detach().cpu())


def materialize(storage, device):
    return {key: torch.cat(parts).to(device) for key, parts in storage.items()}


def data_obs(data, ids, prefix=''):
    return {k: data[prefix+k][ids] for k in ('state', 'events', 'hit_valid', 'group', 'member')}


def metrics(out, *, returns_key='returns'):
    hits = out['hits'].float().sum().item()
    requested = out['requested'].float().sum().item()
    feasible = ((out['torque_excess'] <= .01) & (out['velocity_excess'] <= .01)
                & (out['collision_m'] <= .005) & (out['joint_excess_rad'] <= 1e-4))
    ret = out[returns_key][..., 0]
    return dict(hit_fraction=hits/max(requested, 1), hits=int(hits), requested=int(requested),
        return_mean=float(ret.mean()), feasible_path_fraction=float(feasible.float().mean()),
        torque_excess_mean=float(out['torque_excess'].mean()),
        velocity_excess_mean=float(out['velocity_excess'].mean()),
        collision_peak_m_mean=float(out['collision_m'].mean()))


def log(writer, prefix, values, step):
    for key, value in values.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(prefix+'/'+key, value, step)
    writer.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='schedule_improvement_analytic_20261003')
    p.add_argument('--source-config', default='runs/arm_assignment_balanced_diverse_20261002/config.json')
    p.add_argument('--generator', choices=('decoder', 'pose'), default='decoder')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--collect-batches', type=int, default=32)
    p.add_argument('--windows', type=int, default=16)
    p.add_argument('--candidates', type=int, default=64)
    p.add_argument('--bounce-samples', type=int, default=2)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--critic-steps', type=int, default=2500)
    p.add_argument('--actor-steps', type=int, default=2000)
    p.add_argument('--edit-steps', type=int, default=3)
    p.add_argument('--eval-windows', type=int, default=128)
    p.add_argument('--seed', type=int, default=3103)
    p.add_argument('--benchmark-only', action='store_true')
    args = p.parse_args()
    if args.candidates < 56:
        raise ValueError('Candidate budget must include all graph single/pair neighbors')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); device = torch.device(args.device)
    run = ROOT/'runs'/args.name
    if run.exists(): raise FileExistsError('Choose a fresh run name')
    run.mkdir(parents=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run/'tb'), flush_secs=5)
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        (registry/run.name).symlink_to((run/'tb').resolve())
    original = json.loads((ROOT/args.source_config).read_text())
    checkpoint = Path(original['frozen_planner'])
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(source), device)
        cost = A.FusedCost(A.CostConfig(**source['train_cfg']['cost']), max_horizon=100, device=device)
    model.load_state_dict(source['model'])
    kit = A.DrumKit.from_layout()
    generator_type = DecoderHitPoseGenerator if args.generator == 'decoder' else SingleHitGenerator
    generator = generator_type(model, cost, kit,
        waist_upright_ease_wp=(source.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    env = AnalyticScheduleEnvironment(generator, TrajectorySurrogate(cost, kit), chunk=args.chunk)
    net_cfg = ScheduleNetConfig()
    critic = JointScheduleCritic(net_cfg).to(device)
    actor = ScheduleEditPolicy(net_cfg).to(device)
    training_paths = original['training_clips']
    eval_paths = sorted((ROOT/'assets/rcc/eval').glob('*.json'))
    assert not set(map(str, eval_paths)) & set(training_paths)
    train_pool = make_pool(training_paths, kit, args.seed)
    eval_pool = make_pool(eval_paths, kit, args.seed+100000, allow_missing_drums=True)
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    assert tuple(anchors['joint_names']) == tuple(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q: pose_clearance(cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS)
    config = dict(vars(args), critic=True, bellman=True, bellman_type='Finite-plan suffix consistency',
        critic_objective='Monte Carlo return regression + target-network Bellman consistency',
        actor_objective='Imitate analytically evaluated single/pair local edits, including random initial schedules',
        environment='Existing perfect-tracking Hermite/contact/rebound analytic surrogate; no Isaac',
        planner_checkpoint=str(checkpoint), planner_frozen=True, commitment_features=False,
        network_config=asdict(net_cfg), critic_parameters=sum(p.numel() for p in critic.parameters()),
        actor_parameters=sum(p.numel() for p in actor.parameters()), train_pool=train_pool.summary,
        eval_pool=eval_pool.summary, reward_scale=env.reward_scale, gamma_per_second=env.gamma,
        reward='Discounted sum of per-event analytic hit/miss/constraint scores',
        drum_hand_masks=False, simultaneous_groups='One binary choice; second hit uses opposite arm',
        notes='Critic evaluates following a finite proposed plan. Event-refresh execution is evaluated separately.')
    save_json(run/'config.json', config)
    writer.add_scalar('setup/critic', 1, 0); writer.add_scalar('setup/bellman', 1, 0)
    writer.add_text('setup/config', json.dumps(config, indent=2), 0); writer.flush()
    print(json.dumps(dict(stage='ready', **{k: config[k] for k in ('critic_parameters', 'actor_parameters', 'environment')})), flush=True)
    critic_data, actor_data = {}, {}
    moving_q, moving_v = [], []
    collection_start = time.perf_counter()
    for batch in range(1, args.collect_batches+1):
        start = time.perf_counter()
        clips = sample_windows(train_pool, rng, args.windows)
        positions, pose_stats = starts.sample(rng, args.windows)
        q = torch.as_tensor(positions, device=device)
        v = torch.as_tensor(rng.normal(0, .2, size=positions.shape).astype(np.float32), device=device)
        if moving_q:
            indexes = rng.integers(len(moving_q), size=args.windows//2)
            q[:len(indexes)] = torch.stack([moving_q[i] for i in indexes])
            v[:len(indexes)] = torch.stack([moving_v[i] for i in indexes])
        table = physical_table(clips, kit, device)
        obs = observation(table, q, v, cost.fk)
        banks, bank_valid, seeds, fallbacks = candidate_bank(clips, kit, rng, args.candidates)
        candidates = torch.as_tensor(banks, device=device)
        draws = torch.as_tensor(rng.uniform(.2, .8, size=(args.windows, args.bounce_samples, MAX_HITS, 2)).astype(np.float32), device=device)
        out = env.score(table, q, v, candidates, draws)
        B, C, S, G = out['returns'].shape
        owners = torch.arange(B, device=device).repeat_interleave(C*S)
        flat_actions = candidates[:, :, None].expand(-1, -1, S, -1).reshape(-1, MAX_HITS)
        flat_obs = index_obs(obs, owners)
        roots_valid = torch.as_tensor(bank_valid, device=device)[:, :, None].expand(-1, -1, S).reshape(-1)
        for g in range(G):
            keep = roots_valid & table['valid'][owners, g]
            if g == 0:
                this = flat_obs; this_actions = flat_actions
            else:
                this = suffix_obs(flat_obs, out['poses'].reshape(-1, G, A.N_DOF)[:, g-1],
                    out['post_velocity'].reshape(-1, G, A.N_DOF)[:, g-1], cost.fk,
                    table['times'][owners, g-1], g)
                this_actions = suffix_actions(flat_actions, g)
            nxt = suffix_obs(flat_obs, out['poses'].reshape(-1, G, A.N_DOF)[:, g],
                out['post_velocity'].reshape(-1, G, A.N_DOF)[:, g], cost.fk, table['times'][owners, g], g+1)
            append_observation(critic_data, '', index_obs(this, keep))
            append_observation(critic_data, 'next_', index_obs(nxt, keep))
            for key, value in dict(actions=this_actions, next_actions=suffix_actions(flat_actions, g+1),
                target=out['returns'].reshape(-1, G)[:, g], reward=out['rewards'].reshape(-1, G)[:, g],
                discount=out['discounts'].reshape(-1, G)[:, g], done=~nxt['hit_valid'].any(1)).items():
                critic_data.setdefault(key, []).append(value[keep].detach().cpu())
        mean_returns = out['returns'][..., 0].mean(2).cpu().numpy()
        actor_owners, actor_actions, labels, gains = [], [], [], []
        for b in range(B):
            count = int(bank_valid[b].sum())
            lab, gain = teacher_edits(banks[b, :count], mean_returns[b, :count], int(table['valid'][b].sum()))
            actor_owners.extend([b]*count); actor_actions.extend(banks[b, :count]); labels.extend(lab); gains.extend(gain)
        append_observation(actor_data, '', index_obs(obs, torch.tensor(actor_owners, device=device)))
        for key, value in dict(actions=np.asarray(actor_actions), labels=np.asarray(labels), gains=np.asarray(gains)).items():
            actor_data.setdefault(key, []).append(torch.as_tensor(value))
        # Feed training genuinely moving states from collision-clear points on
        # previously scored connectors, not only ready states with synthetic qd.
        if G > 1:
            qmid, vmid, _ = hermite(out['poses'][:, 0, 0, 0], out['post_velocity'][:, 0, 0, 0],
                out['poses'][:, 0, 0, 1], out['pre_velocity'][:, 0, 0, 1],
                table['times'][:, 1]-table['times'][:, 0], q.new_full((B, 1), .5))
            good = table['valid'][:, 1] & (pose_clearance(cost, qmid[:, 0]) >= .003)
            moving_q.extend(qmid[good, 0].detach().unbind()); moving_v.extend(vmid[good, 0].detach().unbind())
            moving_q, moving_v = moving_q[-4096:], moving_v[-4096:]
        summary = dict(batch=batch, seconds=time.perf_counter()-start, candidate_worlds=B*C*S,
            unique_schedules=int(bank_valid.sum()), collocations=out['collocations'],
            graph_return_mean=float(mean_returns[:, 0].mean()), best_bank_return_mean=float(mean_returns.max(1).mean()),
            moving_start_bank=len(moving_q), graph_fallback_groups=fallbacks,
            actor_stop_fraction=float(np.mean(np.asarray(labels) == 0)))
        log(writer, 'collection', summary, batch); log(writer, 'starts', pose_stats, batch)
        counts = np.bincount([h.drum for c in clips for h in c.hits], minlength=kit.num_drums)
        for i, name in enumerate(kit.names): writer.add_scalar('drums/'+name+'/requested', int(counts[i]), batch)
        save_json(run/'progress.json', dict(stage='collection', **summary))
        print(json.dumps(dict(stage='collection', **summary)), flush=True)
        if args.benchmark_only: break
    training = materialize(critic_data, device); teaching = materialize(actor_data, device)
    torch.save(dict(critic={k: v.cpu() for k, v in training.items()}, actor={k: v.cpu() for k, v in teaching.items()}), run/'analytic_replay.pt')
    if args.benchmark_only:
        save_json(run/'benchmark.json', dict(seconds=time.perf_counter()-collection_start,
            critic_samples=len(training['target']), actor_samples=len(teaching['labels'])))
        writer.close(); return
    held = torch.randperm(len(training['target']), device=device)[:min(4096, len(training['target'])//10)]
    # This diagnostic split shares source windows. Fresh eval rhythms below are
    # the generalization test; don't present this as an independent heldout set.
    fit = torch.ones(len(training['target']), dtype=torch.bool, device=device); fit[held] = False
    fit = fit.nonzero()[:, 0]
    target = copy.deepcopy(critic).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=2e-4, weight_decay=1e-4)
    for step in range(1, args.critic_steps+1):
        ids = fit[torch.randint(len(fit), (128,), device=device)]
        prediction = critic(data_obs(training, ids), training['actions'][ids])
        with torch.no_grad():
            continuation = target(data_obs(training, ids, 'next_'), training['next_actions'][ids])
            bellman_target = training['reward'][ids]+training['discount'][ids]*continuation*(~training['done'][ids])
        mc_loss = F.smooth_l1_loss(prediction, training['target'][ids])
        td_loss = F.smooth_l1_loss(prediction, bellman_target)
        loss = mc_loss+.5*td_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for dst, src in zip(target.parameters(), critic.parameters()): dst.lerp_(src, .01)
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                val = critic(data_obs(training, held), training['actions'][held])
                mae = float((val-training['target'][held]).abs().mean())
            stats = dict(mc_loss=float(mc_loss), bellman_loss=float(td_loss), diagnostic_mae=mae)
            log(writer, 'critic', stats, step); save_json(run/'progress.json', dict(stage='critic', step=step, **stats))
            print(json.dumps(dict(stage='critic', step=step, **stats)), flush=True)
    critic.eval().requires_grad_(False)
    torch.save(dict(format='joint_analytic_schedule_critic_v1', config=config, model=critic.state_dict()), run/'critic.pt')
    actor.encoder.load_state_dict(critic.encoder.state_dict())
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=2e-4, weight_decay=1e-4)
    for step in range(1, args.actor_steps+1):
        ids = torch.randint(len(teaching['labels']), (128,), device=device)
        logits = actor(data_obs(teaching, ids), teaching['actions'][ids])
        loss = F.cross_entropy(logits, teaching['labels'][ids])
        actor_optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1); actor_optimizer.step()
        if step % 100 == 0 or step == 1:
            stats = dict(edit_loss=float(loss), batch_teacher_agreement=float((logits.argmax(1)==teaching['labels'][ids]).float().mean()))
            log(writer, 'actor', stats, step); save_json(run/'progress.json', dict(stage='actor', step=step, **stats))
            print(json.dumps(dict(stage='actor', step=step, **stats)), flush=True)
    actor.eval().requires_grad_(False)
    torch.save(dict(format='analytic_schedule_edit_policy_v1', config=config, model=actor.state_dict()), run/'actor.pt')
    eval_rng = np.random.default_rng(args.seed+200000)
    clips = sample_windows(eval_pool, eval_rng, args.eval_windows)
    positions, _ = starts.sample(eval_rng, args.eval_windows)
    q = torch.as_tensor(positions, device=device)
    v = torch.as_tensor(eval_rng.normal(0, .2, size=positions.shape).astype(np.float32), device=device)
    table = physical_table(clips, kit, device); obs = observation(table, q, v, cost.fk)
    bank_np, bank_valid, seed_np, fallbacks = candidate_bank(clips, kit, eval_rng, args.candidates)
    seed = torch.as_tensor(seed_np, device=device)
    policy_actions = policy_search(actor, obs, seed, steps=args.edit_steps)
    guarded_actions = policy_search(actor, obs, seed, steps=args.edit_steps, critic=critic)
    searched_actions = critic_search(critic, obs, seed, steps=args.edit_steps)
    bank = torch.as_tensor(bank_np, device=device)
    methods = ('graph', 'learned_edits', 'learned_edits_critic_guard', 'critic_local_search')
    chosen = torch.stack((seed, policy_actions, guarded_actions, searched_actions), 1)
    draws = torch.as_tensor(eval_rng.uniform(.2, .8, size=(args.eval_windows, args.bounce_samples, MAX_HITS, 2)).astype(np.float32), device=device)
    start = time.perf_counter()
    out = env.score(table, q, v, torch.cat((chosen, bank), 1), draws)
    comparisons = {name: metrics({k: value[:, i] for k, value in out.items() if torch.is_tensor(value)}) for i, name in enumerate(methods)}
    bank_returns = out['returns'][:, len(methods):, :, 0].mean(2)
    masked_returns = bank_returns.masked_fill(~torch.as_tensor(bank_valid, device=device), -torch.inf)
    best = masked_returns.argmax(1)
    best_out = {k: value[torch.arange(len(q), device=device), len(methods)+best] for k, value in out.items() if torch.is_tensor(value)}
    comparisons['analytic_bank_best'] = metrics(best_out)
    with torch.no_grad():
        predicted = critic_values_for_bank(critic, obs, bank)
    fresh_mae = float((predicted-bank_returns).abs()[torch.as_tensor(bank_valid, device=device)].mean())
    predicted_best = predicted.masked_fill(~torch.as_tensor(bank_valid, device=device), -torch.inf).argmax(1)
    regret = masked_returns.max(1).values-bank_returns[torch.arange(len(q), device=device), predicted_best]
    print(json.dumps(dict(stage='fresh_window_eval', comparisons=comparisons, fresh_critic_mae=fresh_mae)), flush=True)
    log(writer, 'fresh_critic', dict(mae=fresh_mae, bank_selection_regret=float(regret.mean())), args.critic_steps)
    for name, values in comparisons.items(): log(writer, 'probe/window/'+name, values, args.actor_steps)
    causal = {}
    for name in methods:
        begin = time.perf_counter()
        result_parts = []
        for bounce in range(args.bounce_samples):
            def choose(q_now, v_now, elapsed, removed):
                current = obs if removed == 0 else suffix_obs(obs, q_now, v_now, cost.fk, elapsed, removed)
                if removed == 0: current = observation(table, q_now, v_now, cost.fk)
                initial = suffix_actions(seed, removed) if removed else seed
                if name == 'graph': return initial
                if name == 'critic_local_search': return critic_search(critic, current, initial, steps=args.edit_steps)
                return policy_search(actor, current, initial, steps=args.edit_steps,
                    critic=critic if name == 'learned_edits_critic_guard' else None)
            result_parts.append(env.refresh_rollout(table, q, v, draws[:, bounce], choose))
        combined = {k: torch.cat([p[k] for p in result_parts]) for k in result_parts[0] if torch.is_tensor(result_parts[0][k])}
        causal[name] = metrics(combined)
        causal[name]['seconds'] = time.perf_counter()-begin
        log(writer, 'probe/refresh/'+name, causal[name], args.actor_steps)
        print(json.dumps(dict(stage='refresh_eval', method=name, **causal[name])), flush=True)
    report = dict(config=config, training_critic_samples=len(training['target']), training_actor_samples=len(teaching['labels']),
        fresh_windows=len(clips), fresh_critic_mae=fresh_mae, fresh_critic_bank_selection_regret=float(regret.mean()),
        fixed_window=comparisons, refresh_after_each_hit=causal, graph_fallback_groups=fallbacks,
        evaluation_seconds=time.perf_counter()-start,
        scope='Analytic geometric hits, perfect tracking, sampled rebound; not Isaac or actual hit-credit validation',
        analytic_best_scope='Best evaluated bounded candidate bank; not the exhaustive optimum',
        clips=[dict(name=c.name, times=[h.time for h in c.hits], drums=[h.drum_name for h in c.hits]) for c in clips],
        initial_positions=positions.tolist(), initial_velocities=v.cpu().tolist(),
        chosen_actions={name: chosen[:, i].cpu().tolist() for i, name in enumerate(methods)})
    save_json(run/'final_report.json', report)
    save_json(run/'progress.json', dict(stage='complete', report=str(run/'final_report.json')))
    writer.close()
    print(json.dumps(dict(stage='complete', run=str(run))), flush=True)


def critic_values_for_bank(critic, obs, bank):
    from isaac_rl.schedule_improvement import critic_values
    return critic_values(critic, obs, bank)


if __name__ == '__main__': main()
