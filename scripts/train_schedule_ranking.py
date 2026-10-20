#!/usr/bin/env python3
"""Train schedule preferences and repeatedly mine the critic's analytic mistakes."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
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
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator, TrajectorySurrogate
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.random_rhythms import RandomRhythmSampler
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_analytic import AnalyticScheduleEnvironment
from isaac_rl.schedule_improvement import (
    MAX_HITS, JointScheduleCritic, ScheduleNetConfig, physical_table, observation,
    index_obs, suffix_obs, suffix_actions, critic_search, critic_values, neighbors, valid_edits,
)
from isaac_rl.schedule_ranking import select_preference_pairs, preference_loss, ranking_stats
from scripts.train_schedule_improvement import (
    save_json, append_observation, materialize, data_obs, log, candidate_bank, make_pool, sample_windows,
)
from scripts.train_random_rhythm_critic import prepare_panel, refresh_metrics


@torch.no_grad()
def proposal_bank(critic, obs, clips, kit, rng, limit):
    """Include the critic's local winner and its preferred nearby alternatives."""
    base, valid, seeds, _ = candidate_bank(clips, kit, rng, 64)
    seed = torch.as_tensor(seeds, device=obs['state'].device)
    searched = critic_search(critic, obs, seed, steps=3)
    nearby = neighbors(searched)
    values = critic_values(critic, obs, nearby).masked_fill(~valid_edits(obs), -torch.inf)
    choices = values.topk(8, dim=1).indices
    near = nearby.gather(1, choices[..., None].expand(-1, -1, MAX_HITS)).cpu().numpy()
    finite = values.gather(1, choices).isfinite().cpu().numpy()
    banks = []
    for i in range(len(clips)):
        rows = [seeds[i], searched[i].cpu().numpy(), *near[i, finite[i]], *base[i, valid[i]]]
        banks.append(np.asarray(list(dict.fromkeys(tuple(row) for row in rows))[:limit], np.int64))
    count = np.asarray([len(b) for b in banks])
    C = int(count.max())
    padded = np.stack([np.concatenate((b, np.repeat(b[:1], C-len(b), axis=0))) for b in banks])
    return (torch.as_tensor(padded, device=seed.device),
            torch.as_tensor(np.arange(C)[None] < count[:, None], device=seed.device), seed)


def append_value_rollouts(storage, obs, table, bank, valid, out, fk):
    """Pointwise value/Bellman examples; suffix pairs are not compared across states."""
    B, C, S, G = out['returns'].shape
    owners = torch.arange(B, device=bank.device).repeat_interleave(C*S)
    actions = bank[:, :, None].expand(-1, -1, S, -1).reshape(-1, MAX_HITS)
    roots = index_obs(obs, owners)
    root_valid = valid[:, :, None].expand(-1, -1, S).reshape(-1)
    for g in range(G):
        keep = root_valid & table['valid'][owners, g]
        this = roots if g == 0 else suffix_obs(roots,
            out['poses'].reshape(-1, G, A.N_DOF)[:, g-1],
            out['post_velocity'].reshape(-1, G, A.N_DOF)[:, g-1], fk, table['times'][owners, g-1], g)
        nxt = suffix_obs(roots, out['poses'].reshape(-1, G, A.N_DOF)[:, g],
            out['post_velocity'].reshape(-1, G, A.N_DOF)[:, g], fk, table['times'][owners, g], g+1)
        append_observation(storage, '', index_obs(this, keep))
        append_observation(storage, 'next_', index_obs(nxt, keep))
        values = dict(actions=actions if g == 0 else suffix_actions(actions, g),
            next_actions=suffix_actions(actions, g+1), target=out['returns'].reshape(-1, G)[:, g],
            reward=out['rewards'].reshape(-1, G)[:, g], discount=out['discounts'].reshape(-1, G)[:, g],
            done=~nxt['hit_valid'].any(1))
        for key, value in values.items():
            storage.setdefault(key, []).append(value[keep].detach().cpu())


def append_preferences(storage, obs, bank, selected):
    owners = torch.as_tensor(selected['owners'], device=bank.device)
    if not len(owners):
        return
    append_observation(storage, '', index_obs(obs, owners))
    a = torch.as_tensor(selected['candidate_a'], device=bank.device)
    b = torch.as_tensor(selected['candidate_b'], device=bank.device)
    values = dict(actions_a=bank[owners, a], actions_b=bank[owners, b],
        delta=torch.as_tensor(selected['delta']), hard=torch.as_tensor(selected['hard']))
    for key, value in values.items():
        storage.setdefault(key, []).append(value.detach().cpu())


def sample_value_batch(buffers, fractions, count, device):
    parts = []
    used = 0
    for i, (buffer, fraction) in enumerate(zip(buffers, fractions)):
        size = int(round(count*fraction)) if i+1 < len(buffers) else count-used
        used += size
        ids = torch.randint(len(buffer['target']), (size,), device=device)
        parts.append({k: v[ids] for k, v in buffer.items()})
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def cpu_dict(data):
    return {k: v.cpu() for k, v in data.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='schedule_critic_ranking_mining_20261003')
    p.add_argument('--source-run', default='runs/schedule_critic_random_rhythms_20261003')
    p.add_argument('--recorded-replay', default='runs/schedule_improvement_analytic_20261003_v2/analytic_replay.pt')
    p.add_argument('--recorded-config', default='runs/arm_assignment_balanced_diverse_20261002/config.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--batches-per-round', type=int, default=8)
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--steps-per-round', type=int, default=2000)
    p.add_argument('--candidates', type=int, default=64)
    p.add_argument('--bounce-samples', type=int, default=4)
    p.add_argument('--eval-windows', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--rank-weight', type=float, default=1.)
    p.add_argument('--minimum-gap', type=float, default=.25)
    p.add_argument('--rank-temperature', type=float, default=.5)
    p.add_argument('--hard-weight', type=float, default=3.)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--seed', type=int, default=5103)
    args = p.parse_args()
    if min(args.rounds, args.batches_per_round, args.windows, args.steps_per_round,
           args.candidates, args.bounce_samples, args.eval_windows, args.eval_every, args.chunk) < 1:
        raise ValueError('collection and training sizes must be positive')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    device = torch.device(args.device); rng = np.random.default_rng(args.seed)
    run = ROOT/'runs'/args.name; run.mkdir(parents=True, exist_ok=False)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run/'tb'), flush_secs=5)
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        (registry/run.name).symlink_to((run/'tb').resolve())
    previous = ROOT/args.source_run
    old_config = json.loads((previous/'config.json').read_text())
    source = torch.load(old_config['planner_checkpoint'], map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(source), device)
        cost = A.FusedCost(A.CostConfig(**source['train_cfg']['cost']), max_horizon=100, device=device)
    model.load_state_dict(source['model'])
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(source.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    env = AnalyticScheduleEnvironment(generator, TrajectorySurrogate(cost, kit), chunk=args.chunk)
    critic = JointScheduleCritic(ScheduleNetConfig(**old_config['network_config'])).to(device)
    critic.load_state_dict(torch.load(previous/'critic.pt', map_location=device, weights_only=False)['model'])
    reference = copy.deepcopy(critic).eval().requires_grad_(False)
    target = copy.deepcopy(reference)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    assert tuple(anchors['joint_names']) == tuple(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q: pose_clearance(cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS, arm_std=.12, yaw_std=.25)
    sampler = RandomRhythmSampler(kit)
    recorded_config = json.loads((ROOT/args.recorded_config).read_text())
    training_paths = recorded_config['training_clips']
    eval_paths = list((ROOT/'assets/rcc/eval').glob('*.json'))
    assert not set(map(str, training_paths)) & set(map(str, eval_paths))
    recorded_pool = make_pool(training_paths, kit, args.seed)
    value_keys = ('state', 'events', 'hit_valid', 'group', 'member',
        'next_state', 'next_events', 'next_hit_valid', 'next_group', 'next_member',
        'actions', 'next_actions', 'target', 'reward', 'discount', 'done')
    old_random = torch.load(previous/'analytic_replay.pt', map_location=device, weights_only=False)
    old_recorded = torch.load(ROOT/args.recorded_replay, map_location=device, weights_only=False)['critic']
    old_random = {k: old_random[k] for k in value_keys}
    old_recorded = {k: old_recorded[k] for k in value_keys}
    config = dict(vars(args), critic=True, bellman=True, actor_training=False,
        planner_checkpoint=old_config['planner_checkpoint'], planner_frozen=True,
        network_config=old_config['network_config'], continued_critic=str(previous/'critic.pt'),
        objective='MC return + 0.5 Bellman suffix consistency + weighted same-state logistic ranking',
        collection='80% generated / 20% training-recorded new roots; half starts replaced by reached suffix states when available',
        value_replay_fractions=dict(new=.4, old_generated=.4, old_training_recorded=.2),
        preference_filter=dict(minimum_gap=args.minimum_gap, minimum_draw_sign_agreement=.75),
        preference_sampling='Prioritize misrankings and predicted-winner vs graph / measured winner, plus uniform pairs',
        validation_selection='Last update; validation labels never mined or used for training',
        environment=old_config['environment'], reward_scale=env.reward_scale, gamma_per_second=env.gamma)
    save_json(run/'config.json', config)
    log(writer, 'setup', dict(critic=1, bellman=1, ranking=1, actor_training=0), 0)
    panels = {}
    for name in ('generated', 'recorded'):
        panel = torch.load(previous/(name+'_panel.pt'), map_location=device, weights_only=False)
        n = min(args.eval_windows, len(panel['clips']))
        panels[name] = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in panel.items()}
        panels[name]['clips'] = panel['clips'][:n]
        for field in ('obs', 'table'):
            panels[name][field] = {k: v[:n] for k, v in panel[field].items()}
    history = []
    def validate(step):
        critic.eval()
        stats = {name: ranking_stats(critic, panel) for name, panel in panels.items()}
        for name, values in stats.items(): log(writer, 'validation/'+name, values, step)
        history.append(dict(step=step, panels=stats)); save_json(run/'validation_history.json', history)
        print(json.dumps(dict(stage='validation', step=step, panels=stats)), flush=True)
        critic.train()
    validate(0)
    print(json.dumps(dict(stage='ready', run=str(run), rounds=args.rounds)), flush=True)
    values_storage, pairs_storage, suffix_pool, collection_history = {}, {}, [], []
    drum_counts = np.zeros(kit.num_drums, np.int64)
    step = 0
    for round_index in range(1, args.rounds+1):
        critic.eval()
        for batch in range(1, args.batches_per_round+1):
            begin = time.perf_counter()
            clips = sampler.sample(rng, args.windows)
            natural_count = int(round(args.windows*.2))
            clips[:natural_count] = sample_windows(recorded_pool, rng, natural_count)
            q_np, _ = starts.sample(rng, args.windows)
            q = torch.as_tensor(q_np, device=device)
            v = torch.as_tensor(rng.normal(0, .3, size=q_np.shape).astype(np.float32), device=device)
            reached_count = 0
            if suffix_pool:
                chosen = rng.integers(len(suffix_pool), size=args.windows//2)
                slots = rng.choice(args.windows, size=len(chosen), replace=False)
                for i, index in zip(slots, chosen):
                    clips[i], qi, vi = suffix_pool[index]
                    q[i], v[i] = qi.to(device), vi.to(device)
                reached_count = len(chosen)
            table = physical_table(clips, kit, device); obs = observation(table, q, v, cost.fk)
            with torch.no_grad():
                bank, valid, seed = proposal_bank(critic, obs, clips, kit, rng, args.candidates)
                predicted = critic_values(critic, obs, bank)
            draws = torch.as_tensor(rng.uniform(.2, .8, size=(len(clips), args.bounce_samples, MAX_HITS, 2)).astype(np.float32), device=device)
            out = env.score(table, q, v, bank, draws)
            score_samples = out['returns'][..., 0]
            selected = select_preference_pairs(predicted.cpu().numpy(), score_samples.cpu().numpy(),
                valid.cpu().numpy(), rng, minimum_gap=args.minimum_gap)
            append_preferences(pairs_storage, obs, bank, selected)
            append_value_rollouts(values_storage, obs, table, bank, valid, out, cost.fk)
            winner = predicted.masked_fill(~valid, -torch.inf).argmax(1)
            rows = torch.arange(len(clips), device=device)
            draw_delta = score_samples[rows, winner]-score_samples[:, 0]
            stable_failure = ((winner != 0) & (draw_delta.mean(-1) < -args.minimum_gap)
                              & ((draw_delta < 0).float().mean(-1) >= .75))
            # Future comparisons begin from the actual selected plan's reached
            # state, with the SAME remaining requests and shifted deadlines.
            for i, clip in enumerate(clips):
                groups = int(table['valid'][i].sum())
                if groups < 2:
                    continue
                g = int(rng.integers(groups-1)); bounce = int(rng.integers(args.bounce_samples))
                elapsed = sorted({h.time for h in clip.hits})[g]
                hits = [replace(h, time=round(h.time-elapsed, 8), raw_time=round(h.time-elapsed, 8))
                        for h in clip.hits if h.time > elapsed+1e-6]
                qi = out['poses'][i, winner[i], bounce, g].detach()
                vi = out['post_velocity'][i, winner[i], bounce, g].detach()
                joint_ok = np.all((qi.cpu().numpy() >= lo) & (qi.cpu().numpy() <= hi))
                if hits and joint_ok and torch.isfinite(vi).all():
                    suffix_pool.append((replace(clip, hits=hits, duration=hits[-1].time), qi.cpu(), vi.cpu()))
            suffix_pool = suffix_pool[-4096:]
            drum_counts += np.bincount([h.drum for c in clips for h in c.hits], minlength=kit.num_drums)
            stats = dict(round=round_index, batch=batch, seconds=time.perf_counter()-begin,
                root_states=len(clips), reached_starts=reached_count, suffix_pool=len(suffix_pool),
                recorded_roots=sum(not c.name.startswith('random_') for c in clips),
                preference_pairs=len(selected['owners']), hard_pairs=int(selected['hard'].sum()),
                changed_predicted_winners=int((winner != 0).sum()), stable_harmful_winners=int(stable_failure.sum()))
            collection_history.append(stats)
            log(writer, 'mining', stats, (round_index-1)*args.batches_per_round+batch)
            for i, name in enumerate(kit.names):
                writer.add_scalar('drums/'+name+'/requested', int(drum_counts[i]),
                    (round_index-1)*args.batches_per_round+batch)
            save_json(run/'collection_history.json', collection_history)
            save_json(run/'progress.json', dict(stage='mining', **stats))
            print(json.dumps(dict(stage='mining', **stats)), flush=True)
        fresh_values = materialize(values_storage, device)
        preferences = materialize(pairs_storage, device)
        if not len(preferences.get('delta', [])):
            raise RuntimeError('No stable preference pairs; cannot train ranking objective')
        torch.save(dict(value=cpu_dict(fresh_values), preferences=cpu_dict(preferences)), run/'mined_replay.pt')
        critic.train()
        for _ in range(args.steps_per_round):
            step += 1
            batch_values = sample_value_batch([fresh_values, old_random, old_recorded], [.4, .4, .2], 128, device)
            ids = torch.arange(len(batch_values['target']), device=device)
            prediction = critic(data_obs(batch_values, ids), batch_values['actions'])
            with torch.no_grad():
                continuation = target(data_obs(batch_values, ids, 'next_'), batch_values['next_actions'])
                td_target = batch_values['reward']+batch_values['discount']*continuation*(~batch_values['done'])
            mc = F.smooth_l1_loss(prediction, batch_values['target'])
            td = F.smooth_l1_loss(prediction, td_target)
            pair_ids = torch.randint(len(preferences['delta']), (64,), device=device)
            pair_obs = data_obs(preferences, pair_ids)
            combined_obs = {k: torch.cat((value, value)) for k, value in pair_obs.items()}
            pair_pred = critic(combined_obs, torch.cat((preferences['actions_a'][pair_ids], preferences['actions_b'][pair_ids])))
            ranking = preference_loss(pair_pred[:64], pair_pred[64:], preferences['delta'][pair_ids],
                preferences['hard'][pair_ids], temperature=args.rank_temperature, hard_weight=args.hard_weight)
            loss = mc+.5*td+args.rank_weight*ranking
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1); optimizer.step()
            with torch.no_grad():
                for dst, src in zip(target.parameters(), critic.parameters()): dst.lerp_(src, .01)
            if step % 100 == 0 or step == 1:
                stats = dict(round=round_index, step=step, mc_loss=float(mc), bellman_loss=float(td),
                    rank_loss=float(ranking), pair_batch_accuracy=float(
                        (((pair_pred[:64]-pair_pred[64:]) > 0) == (preferences['delta'][pair_ids] > 0)).float().mean()))
                log(writer, 'critic', stats, step)
                save_json(run/'progress.json', dict(stage='critic', **stats))
                print(json.dumps(dict(stage='critic', **stats)), flush=True)
            if step % args.eval_every == 0 or step % args.steps_per_round == 0:
                validate(step)
                torch.save(dict(format='joint_analytic_schedule_critic_v1', config=config,
                    model=critic.state_dict(), optimizer=optimizer.state_dict(), target_model=target.state_dict(),
                    step=step, round=round_index), run/'critic.pt')
    critic.eval().requires_grad_(False)
    # A new procedural panel is created only after all learning/mining ends.
    eval_rng = np.random.default_rng(args.seed+400000)
    clips = sampler.sample(eval_rng, args.eval_windows)
    q_np, _ = starts.sample(eval_rng, len(clips)); q = torch.as_tensor(q_np, device=device)
    v = torch.as_tensor(eval_rng.normal(0, .3, size=q_np.shape).astype(np.float32), device=device)
    draws = torch.as_tensor(eval_rng.uniform(.2, .8, size=(len(clips), args.bounce_samples, MAX_HITS, 2)).astype(np.float32), device=device)
    save_json(run/'progress.json', dict(stage='fresh_test'))
    panels['fresh_generated'] = prepare_panel(env, cost, kit, clips, q, v, draws, eval_rng, run, 'fresh_generated')
    diagnostics, refresh = {}, {}
    for name, panel in panels.items():
        diagnostics[name] = dict(previous=ranking_stats(reference, panel), ranked=ranking_stats(critic, panel))
        refresh[name] = {}
        for label, network in [('graph', None), ('previous_critic', reference), ('ranking_critic', critic)]:
            result = refresh_metrics(env, cost, panel, network)
            refresh[name][label] = result
            log(writer, 'probe/refresh/'+name+'/'+label, result, step)
            print(json.dumps(dict(stage='refresh', panel=name, method=label, **result)), flush=True)
    report = dict(config=config, continuation_updates=step,
        mined_root_states=sum(s['root_states'] for s in collection_history),
        mined_value_examples=len(fresh_values['target']), preference_examples=len(preferences['delta']),
        hard_preference_examples=int(preferences['hard'].sum()),
        training_drum_names=kit.names, training_drum_counts=drum_counts.tolist(),
        stable_harmful_winners=sum(s['stable_harmful_winners'] for s in collection_history),
        initial_validation=history[0], final_validation=history[-1], ranking_diagnostics=diagnostics,
        refresh_after_each_hit=refresh,
        scope='Analytic perfect-tracking/contact/bounce, finite-plan scores; no Isaac execution')
    save_json(run/'final_report.json', report)
    save_json(run/'progress.json', dict(stage='complete', report=str(run/'final_report.json')))
    writer.close(); print(json.dumps(dict(stage='complete', run=str(run))), flush=True)


if __name__ == '__main__':
    main()
