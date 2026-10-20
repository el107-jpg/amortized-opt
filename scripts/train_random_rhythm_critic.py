#!/usr/bin/env python3
"""Continue the analytic joint critic using entirely procedural rhythm windows."""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator, TrajectorySurrogate, hermite
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.random_rhythms import RandomRhythmSampler
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_analytic import AnalyticScheduleEnvironment
from isaac_rl.schedule_improvement import (
    MAX_HITS, JointScheduleCritic, ScheduleNetConfig, graph_seed, physical_table,
    observation, index_obs, suffix_obs, suffix_actions, critic_search, critic_values,
)
from scripts.train_schedule_improvement import (
    save_json, append_observation, materialize, data_obs, log, metrics, candidate_bank,
)


def random_bank(clips, kit, rng, limit):
    """Graph/mirror seeds and uniformly random complete schedules, no hand masks."""
    banks, seeds, counts = [], [], []
    for clip in clips:
        seed, _ = graph_seed(clip, kit)
        n = len(set(h.time for h in clip.hits))
        mirror = seed.copy(); mirror[:n] ^= 1
        rows = [seed, mirror]
        # Oversample to deduplicate short schedules; enumerate all when tiny.
        if 2**n <= limit:
            for bits in range(2**n):
                row = np.zeros(MAX_HITS, np.int64)
                row[:n] = [(bits >> j) & 1 for j in range(n)]
                rows.append(row)
        else:
            while len({tuple(row) for row in rows}) < limit:
                row = np.zeros(MAX_HITS, np.int64)
                row[:n] = rng.integers(2, size=n)
                rows.append(row)
        bank = np.asarray(list(dict.fromkeys(tuple(row) for row in rows))[:limit], np.int64)
        banks.append(bank); seeds.append(seed); counts.append(len(bank))
    C = max(counts)
    bank = np.stack([np.concatenate((b, np.repeat(b[:1], C-len(b), axis=0))) for b in banks])
    valid = np.arange(C)[None] < np.asarray(counts)[:, None]
    return bank, valid, np.stack(seeds)


@torch.no_grad()
def panel_stats(critic, panel):
    predicted = critic_values(critic, panel['obs'], panel['bank'])
    actual, valid = panel['returns'], panel['valid']
    selected = predicted.masked_fill(~valid, -torch.inf).argmax(1)
    rows = torch.arange(len(actual), device=actual.device)
    best = actual.masked_fill(~valid, -torch.inf).max(1).values
    gap = best-actual[rows, selected]
    return dict(mae=float((predicted-actual).abs()[valid].mean()),
        bank_selection_regret=float(gap.mean()),
        selected_return=float(actual[rows, selected].mean()),
        best_bank_return=float(best.mean()))


@torch.no_grad()
def prepare_panel(env, cost, kit, clips, q, v, draws, rng, run, name):
    table = physical_table(clips, kit, q.device)
    obs = observation(table, q, v, cost.fk)
    bank_np, valid, seeds, _ = candidate_bank(clips, kit, rng, 64)
    bank = torch.as_tensor(bank_np, device=q.device)
    out = env.score(table, q, v, bank, draws)
    result = dict(clips=clips, q=q, v=v, draws=draws, table=table, obs=obs, bank=bank,
        valid=torch.as_tensor(valid, device=q.device), seed=torch.as_tensor(seeds, device=q.device),
        returns=out['returns'][..., 0].mean(2),
        requested=sum(len(c.hits) for c in clips),
        drum_counts=np.bincount([h.drum for c in clips for h in c.hits], minlength=kit.num_drums).tolist())
    def cpu(value):
        if torch.is_tensor(value): return value.cpu()
        if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
        return value
    torch.save(cpu(result), run/(name+'_panel.pt'))
    return result


@torch.no_grad()
def refresh_metrics(env, cost, panel, critic=None):
    parts = []
    for bounce in range(panel['draws'].shape[1]):
        def choose(q, v, elapsed, removed):
            obs = observation(panel['table'], q, v, cost.fk) if removed == 0 else suffix_obs(
                panel['obs'], q, v, cost.fk, elapsed, removed)
            seed = panel['seed'] if removed == 0 else suffix_actions(panel['seed'], removed)
            return seed if critic is None else critic_search(critic, obs, seed, steps=3)
        parts.append(env.refresh_rollout(panel['table'], panel['q'], panel['v'],
            panel['draws'][:, bounce], choose))
    return metrics({k: torch.cat([p[k] for p in parts]) for k in parts[0] if torch.is_tensor(parts[0][k])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='schedule_critic_random_rhythms_20261003')
    parser.add_argument('--source-run', default='runs/schedule_improvement_analytic_20261003_v2')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--collect-batches', type=int, default=64)
    parser.add_argument('--windows', type=int, default=32)
    parser.add_argument('--candidates', type=int, default=32)
    parser.add_argument('--bounce-samples', type=int, default=2)
    parser.add_argument('--critic-steps', type=int, default=8000)
    parser.add_argument('--eval-windows', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--seed', type=int, default=4103)
    args = parser.parse_args()
    if min(args.collect_batches, args.windows, args.candidates, args.bounce_samples,
           args.critic_steps, args.eval_windows, args.eval_every, args.chunk) < 1:
        raise ValueError('positive collection/training/evaluation sizes required')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); device = torch.device(args.device)
    run = ROOT/'runs'/args.name
    run.mkdir(parents=True, exist_ok=False)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run/'tb'), flush_secs=5)
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        (registry/run.name).symlink_to((run/'tb').resolve())
    previous = ROOT/args.source_run
    source_config = json.loads((previous/'config.json').read_text())
    source = torch.load(source_config['planner_checkpoint'], map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(source), device)
        cost = A.FusedCost(A.CostConfig(**source['train_cfg']['cost']), max_horizon=100, device=device)
    model.load_state_dict(source['model'])
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(source.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    env = AnalyticScheduleEnvironment(generator, TrajectorySurrogate(cost, kit), chunk=args.chunk)
    net_config = ScheduleNetConfig(**source_config['network_config'])
    critic = JointScheduleCritic(net_config).to(device)
    old = torch.load(previous/'critic.pt', map_location=device, weights_only=False)
    critic.load_state_dict(old['model'])
    reference = copy.deepcopy(critic).eval().requires_grad_(False)
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    assert tuple(anchors['joint_names']) == tuple(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q: pose_clearance(cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS, arm_std=.12, yaw_std=.25)
    sampler = RandomRhythmSampler(kit)
    config = dict(vars(args), critic=True, bellman=True, actor_training=False,
        planner_checkpoint=source_config['planner_checkpoint'], planner_frozen=True,
        network_config=asdict(net_config), critic_parameters=sum(p.numel() for p in critic.parameters()),
        continued_critic=str(previous/'critic.pt'), procedural_rhythms=sampler.summary,
        state_noise=dict(arm_std=.12, yaw_std=.25, lean_std=.02, velocity_std=.3,
            moving_start_fraction=.5, minimum_clearance_m=.003),
        critic_objective='Monte Carlo finite-plan return + 0.5 target-network Bellman suffix consistency',
        environment='Frozen decoder endpoints, perfect Hermite tracking, analytic constraints and rebound; no Isaac',
        reward_scale=env.reward_scale, gamma_per_second=env.gamma,
        validation_selection='Last update; validation never chooses the checkpoint')
    save_json(run/'config.json', config)
    log(writer, 'setup', dict(critic=1, bellman=1, actor_training=0), 0)
    print(json.dumps(dict(stage='ready', config=config)), flush=True)
    data, moving_q, moving_v = {}, [], []
    requested_counts = np.zeros(kit.num_drums, np.int64)
    for batch in range(1, args.collect_batches+1):
        begin = time.perf_counter()
        clips = sampler.sample(rng, args.windows)
        positions, start_stats = starts.sample(rng, args.windows)
        q = torch.as_tensor(positions, device=device)
        v = torch.as_tensor(rng.normal(0, .3, size=positions.shape).astype(np.float32), device=device)
        if moving_q:
            ids = rng.integers(len(moving_q), size=args.windows//2)
            q[:len(ids)] = torch.stack([moving_q[i] for i in ids])
            v[:len(ids)] = torch.stack([moving_v[i] for i in ids])
        table = physical_table(clips, kit, device)
        obs = observation(table, q, v, cost.fk)
        bank_np, valid, _ = random_bank(clips, kit, rng, args.candidates)
        bank = torch.as_tensor(bank_np, device=device)
        draws = torch.as_tensor(rng.uniform(.2, .8, size=(args.windows, args.bounce_samples, MAX_HITS, 2)).astype(np.float32), device=device)
        out = env.score(table, q, v, bank, draws)
        B, C, S, G = out['returns'].shape
        owners = torch.arange(B, device=device).repeat_interleave(C*S)
        actions = bank[:, :, None].expand(-1, -1, S, -1).reshape(-1, MAX_HITS)
        flat_obs = index_obs(obs, owners)
        root_valid = torch.as_tensor(valid, device=device)[:, :, None].expand(-1, -1, S).reshape(-1)
        for g in range(G):
            keep = root_valid & table['valid'][owners, g]
            this = flat_obs if g == 0 else suffix_obs(flat_obs,
                out['poses'].reshape(-1, G, A.N_DOF)[:, g-1],
                out['post_velocity'].reshape(-1, G, A.N_DOF)[:, g-1], cost.fk, table['times'][owners, g-1], g)
            nxt = suffix_obs(flat_obs, out['poses'].reshape(-1, G, A.N_DOF)[:, g],
                out['post_velocity'].reshape(-1, G, A.N_DOF)[:, g], cost.fk, table['times'][owners, g], g+1)
            append_observation(data, '', index_obs(this, keep))
            append_observation(data, 'next_', index_obs(nxt, keep))
            values = dict(actions=actions if g == 0 else suffix_actions(actions, g),
                next_actions=suffix_actions(actions, g+1),
                target=out['returns'].reshape(-1, G)[:, g], reward=out['rewards'].reshape(-1, G)[:, g],
                discount=out['discounts'].reshape(-1, G)[:, g], done=~nxt['hit_valid'].any(1),
                window_id=owners+(batch-1)*B)
            for key, value in values.items():
                data.setdefault(key, []).append(value[keep].detach().cpu())
        if G > 1:
            qm, vm, _ = hermite(out['poses'][:, 0, 0, 0], out['post_velocity'][:, 0, 0, 0],
                out['poses'][:, 0, 0, 1], out['pre_velocity'][:, 0, 0, 1],
                table['times'][:, 1]-table['times'][:, 0], q.new_full((B, 1), .5))
            good = table['valid'][:, 1] & torch.as_tensor(starts.valid(qm[:, 0].cpu().numpy()), device=device)
            moving_q.extend(qm[good, 0].detach().unbind()); moving_v.extend(vm[good, 0].detach().unbind())
            moving_q, moving_v = moving_q[-4096:], moving_v[-4096:]
        counts = np.bincount([h.drum for c in clips for h in c.hits], minlength=kit.num_drums)
        requested_counts += counts
        stats = dict(batch=batch, windows=batch*B, seconds=time.perf_counter()-begin,
            unique_schedules=int(valid.sum()), moving_start_bank=len(moving_q),
            mean_hits_per_window=float(counts.sum()/B))
        log(writer, 'collection', stats, batch); log(writer, 'starts', start_stats, batch)
        for i, name in enumerate(kit.names):
            writer.add_scalar('drums/'+name+'/requested', int(requested_counts[i]), batch)
        save_json(run/'progress.json', dict(stage='collection', **stats))
        print(json.dumps(dict(stage='collection', **stats)), flush=True)
    training = materialize(data, device)
    del data
    torch.save({k: v.cpu() for k, v in training.items()}, run/'analytic_replay.pt')
    save_json(run/'collection_summary.json', dict(windows=args.windows*args.collect_batches,
        critic_samples=len(training['target']), drum_names=kit.names, counts=requested_counts.tolist(),
        shares=(requested_counts/requested_counts.sum()).tolist()))
    panels = {}
    eval_rng = np.random.default_rng(args.seed+200000)
    clips = sampler.sample(eval_rng, args.eval_windows)
    q_np, _ = starts.sample(eval_rng, len(clips))
    q = torch.as_tensor(q_np, device=device)
    v = torch.as_tensor(eval_rng.normal(0, .3, size=q_np.shape).astype(np.float32), device=device)
    draws = torch.as_tensor(eval_rng.uniform(.2, .8, size=(len(clips), args.bounce_samples, MAX_HITS, 2)).astype(np.float32), device=device)
    print(json.dumps(dict(stage='prepare_validation', panel='generated')), flush=True)
    save_json(run/'progress.json', dict(stage='prepare_validation', panel='generated'))
    panels['generated'] = prepare_panel(env, cost, kit, clips, q, v, draws, eval_rng, run, 'generated')
    # Paired natural-rhythm comparison preserves the original evaluation panel.
    natural = torch.load(previous/'eval_panel.pt', map_location=device, weights_only=False)
    natural_count = min(args.eval_windows, len(natural['clips']))
    print(json.dumps(dict(stage='prepare_validation', panel='recorded')), flush=True)
    save_json(run/'progress.json', dict(stage='prepare_validation', panel='recorded'))
    panels['recorded'] = prepare_panel(env, cost, kit, natural['clips'][:natural_count],
        natural['q'][:natural_count], natural['v'][:natural_count], natural['draws'][:natural_count],
        np.random.default_rng(args.seed+300000), run, 'recorded')
    history = []
    def validate(step):
        critic.eval()
        stats = {name: panel_stats(critic, panel) for name, panel in panels.items()}
        for name, values in stats.items(): log(writer, 'validation/'+name, values, step)
        history.append(dict(step=step, panels=stats))
        save_json(run/'validation_history.json', history)
        print(json.dumps(dict(stage='validation', step=step, panels=stats)), flush=True)
        critic.train()
    validate(0)
    target = copy.deepcopy(critic).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    critic.train()
    for step in range(1, args.critic_steps+1):
        ids = torch.randint(len(training['target']), (128,), device=device)
        prediction = critic(data_obs(training, ids), training['actions'][ids])
        with torch.no_grad():
            continuation = target(data_obs(training, ids, 'next_'), training['next_actions'][ids])
            td_target = training['reward'][ids]+training['discount'][ids]*continuation*(~training['done'][ids])
        mc = F.smooth_l1_loss(prediction, training['target'][ids])
        td = F.smooth_l1_loss(prediction, td_target)
        optimizer.zero_grad(set_to_none=True); (mc+.5*td).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1); optimizer.step()
        with torch.no_grad():
            for dst, src in zip(target.parameters(), critic.parameters()): dst.lerp_(src, .01)
        if step % 100 == 0 or step == 1:
            stats = dict(mc_loss=float(mc), bellman_loss=float(td))
            log(writer, 'critic', stats, step)
            save_json(run/'progress.json', dict(stage='critic', step=step, **stats))
            print(json.dumps(dict(stage='critic', step=step, **stats)), flush=True)
        if step % args.eval_every == 0 or step == args.critic_steps:
            validate(step)
            torch.save(dict(format='joint_analytic_schedule_critic_v1', config=config,
                model=critic.state_dict(), optimizer=optimizer.state_dict(),
                target_model=target.state_dict(), step=step), run/'critic.pt')
    critic.eval().requires_grad_(False)
    refresh = {}
    for name, panel in panels.items():
        refresh[name] = {}
        for label, network in [('graph', None), ('previous_critic', reference), ('random_rhythm_critic', critic)]:
            values = refresh_metrics(env, cost, panel, network)
            refresh[name][label] = values
            log(writer, 'probe/refresh/'+name+'/'+label, values, args.critic_steps)
            print(json.dumps(dict(stage='refresh', panel=name, method=label, **values)), flush=True)
    report = dict(config=config, training_windows=args.windows*args.collect_batches,
        training_critic_samples=len(training['target']), previous_updates=source_config['critic_steps'],
        continuation_updates=args.critic_steps, initial_validation=history[0], final_validation=history[-1],
        refresh_after_each_hit=refresh, training_drum_counts=requested_counts.tolist(),
        evaluation_drum_counts={name: panel['drum_counts'] for name, panel in panels.items()},
        scope='Analytic geometric hits and finite-plan values; perfect tracking, no Isaac',
        comparison='Both critics evaluated with identical rhythms, starts and rebound draws',
        candidate_bank='Bounded bank, not exhaustive optimum')
    save_json(run/'final_report.json', report)
    save_json(run/'progress.json', dict(stage='complete', report=str(run/'final_report.json')))
    writer.close()
    print(json.dumps(dict(stage='complete', run=str(run))), flush=True)


if __name__ == '__main__':
    main()
