#!/usr/bin/env python3
"""Collect exhaustive actual-spline teachers and train the 86 -> 16 scorer."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos, PELVIS_IN_ENV
from amortized_opt.hand_assign import LIMB_LEFT, LIMB_RIGHT
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.random_rhythms import RandomRhythmConfig, RandomRhythmSampler
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
from isaac_rl.schedule_distillation import (FourHitScheduleScorer, schedule_bank,
    suffix_table, value_and_ranking_loss, relative_score_loss, selection_metrics)
from isaac_rl.schedule_improvement import physical_table, graph_seed


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def setup(checkpoint, device, chunk):
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(cp), device)
        cfg = A.CostConfig(**cp['train_cfg']['cost'])
        cfg.n_sub = 4; cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(cp['model']); kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(cp.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    # Large eager batches have nearly the same throughput and avoid retaining
    # a private CUDA pool for every new prefix duration in procedural rhythms.
    return DecoderScheduleEnvironment(generator, chunk=chunk), kit


def random_windows(sampler, rng, count):
    clips = []
    while len(clips) < count:
        for clip in sampler.sample(rng, max(32, count-len(clips))):
            if len(clip.hits) != 4:
                continue
            first = float(rng.choice([.16, .4, clip.hits[0].time], p=[.3, .3, .4]))
            shift = first-clip.hits[0].time
            hits = [replace(h, time=round(h.time+shift, 8), raw_time=round(h.raw_time+shift, 8)) for h in clip.hits]
            if hits[-1].time <= 1.+1e-7:
                clips.append(replace(clip, hits=hits, duration=hits[-1].time))
            if len(clips) == count:
                break
    return clips


@torch.no_grad()
def graph_indices(clips, kit, features):
    tip = features[:, 34:40].reshape(-1, 2, 3)
    nearest = (tip[:, :, None]-(tip.new_tensor(kit.positions)-tip.new_tensor(PELVIS_IN_ENV))[None, None]).norm(dim=-1).argmin(-1).cpu()
    actions = []
    for i, clip in enumerate(clips):
        parked = {LIMB_LEFT:kit.names[int(nearest[i, 0])], LIMB_RIGHT:kit.names[int(nearest[i, 1])]}
        action, _ = graph_seed(clip, kit, parked=parked); actions.append(action[:4])
    return (torch.as_tensor(np.asarray(actions), device=features.device)*features.new_tensor([8,4,2,1])).sum(-1).long()


@torch.no_grad()
def transitions(scorer, env, table, features, bank, out, chosen, rng):
    B = len(features); S = out['score'].shape[2]
    candidate = torch.stack((chosen, torch.as_tensor(rng.integers(16, size=B), device=features.device)), 1)
    owner = torch.arange(B, device=features.device)[:, None, None].expand(-1, 2, S).reshape(-1)
    c = candidate[:, :, None].expand(-1, -1, S).reshape(-1)
    sample = torch.arange(S, device=features.device)[None, None].expand(B, 2, -1).reshape(-1)
    raw = {k:table[k][owner, :4] for k in ('drums', 'speeds', 'times', 'valid')}
    scores = out['score'][owner, c, sample]/scorer.reward_scale
    poses = out['poses'][owner, c, sample]; velocities = out['post_velocity'][owner, c, sample]
    current = features[owner]; parts = []
    for g in range(4):
        nxt_table = suffix_table(raw, raw['times'][:, g], g+1)
        nxt = scorer.features(nxt_table, poses[:, g], velocities[:, g], env.cost.fk)
        # Finite-plan Bellman target follows the selected continuation. It does
        # not bootstrap a maximum over a different schedule or add a tail.
        parts.append(dict(obs=current, next_obs=nxt,
            action=(c % (1 << (4-g))) << g,
            next_action=(c % (1 << (3-g))) << (g+1),
            target=scores[:, g:].sum(-1), reward=scores[:, g],
            done=~nxt_table['valid'].any(-1)))
        current = nxt
    return {k:torch.cat([p[k] for p in parts]).cpu() for k in parts[0]}


@torch.no_grad()
def collect(args):
    torch.set_num_threads(2); device = torch.device(args.device)
    torch.manual_seed(args.seed+args.rank)
    env, kit = setup(args.checkpoint, device, args.chunk)
    scorer = FourHitScheduleScorer().to(device).eval()
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    assert tuple(anchors['joint_names']) == tuple(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q:pose_clearance(env.cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS, arm_std=.12, yaw_std=.25)
    sampler = RandomRhythmSampler(kit, RandomRhythmConfig(max_hits=4, pair_probability=0.))
    begin = time.monotonic(); total = 0
    for split, count, offset in (('train', args.train_windows, 0), ('validation', args.val_windows, 1000000), ('test', args.test_windows, 2000000)):
        rng = np.random.default_rng(args.seed+offset+args.rank*10000)
        local = len(range(args.rank, count, args.workers))
        folder = args.out/'data'/split; folder.mkdir(parents=True, exist_ok=True)
        for batch, start in enumerate(range(0, local, args.collection_windows)):
            n = min(args.collection_windows, local-start)
            clips = random_windows(sampler, rng, n)
            positions, stats = starts.sample(rng, n)
            std = rng.choice([.1, .3, .7], size=(n, 1), p=[.6, .3, .1])
            velocities = rng.normal(size=positions.shape)*std
            exact_ready = np.max(np.abs(positions-ready_joint_pos()), axis=1)<1e-6
            velocities[exact_ready] = 0.
            q = torch.as_tensor(positions, device=device)
            v = torch.as_tensor(velocities, device=device, dtype=torch.float32)
            table = physical_table(clips, kit, device)
            bank = schedule_bank(device)[None].expand(n, -1, -1)
            draws = torch.as_tensor(rng.uniform(.2, .8, (n, 2, 4, 2)), device=device, dtype=torch.float32)
            features = scorer.features(table, q, v, env.cost.fk)
            out = env.score(table, q, v, bank, draws)
            values = out['score'].double().sum(-1).mean(-1).float()
            chosen = values.argmax(-1)
            data = dict(features=features.cpu(), values=values.cpu(), chosen=chosen.cpu(),
                graph_index=graph_indices(clips, kit, features).cpu(), q=q.cpu(), v=v.cpu(), clips=clips,
                draws=draws.cpu(), first_hit_s=table['times'][:, 0].cpu(),
                drum_counts=torch.bincount(table['drums'][:, :4, 0].flatten(), minlength=7).cpu(),
                root_ids=[f'{split}:{args.rank}:{start+i}' for i in range(n)], start_stats=stats)
            if split == 'train':
                data['transitions'] = transitions(scorer, env, table, features, bank, out, chosen, rng)
            # Keep per-event actual physics results for all candidates, not
            # just values, so heldout quality can be audited without re-search.
            if split != 'train':
                data['physics'] = {k:t.cpu() for k,t in out.items()
                    if k not in ('poses', 'pre_velocity', 'post_velocity', 'evaluated_groups')}
            path = folder/f'w{args.rank}_{batch:04d}.pt'
            torch.save(data, path)
            total += n
            write_json(args.out/f'w{args.rank}_progress.json', dict(stage='collection', split=split,
                split_windows=start+n, completed_windows=total, elapsed_s=time.monotonic()-begin))
            if batch%16 == 0:
                print(json.dumps(dict(split=split, windows=start+n, worker=args.rank)), flush=True)
            del out, data
    write_json(args.out/f'w{args.rank}_done.json', dict(status='complete', windows=total, elapsed_s=time.monotonic()-begin))


def load_split(folder, device):
    parts = [torch.load(p, weights_only=False) for p in sorted(folder.glob('*.pt'))]
    if not parts:
        raise ValueError(f'No completed data shards in {folder}')
    data = {k:torch.cat([p[k] for p in parts]).to(device)
            for k in ('features', 'values', 'graph_index')}
    transition_parts = [p['transitions'] for p in parts if 'transitions' in p]
    if transition_parts:
        data['transitions'] = {k:torch.cat([p[k] for p in transition_parts]).to(device)
                               for k in transition_parts[0]}
    data['root_ids'] = sum((p['root_ids'] for p in parts), [])
    return data


def log(writer, prefix, values, step):
    for k,v in values.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(prefix+'/'+k, v, step)
    writer.flush()


def train(args, writer):
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    device = torch.device(args.devices[0])
    training = load_split(args.out/'data/train', device)
    validation = load_split(args.out/'data/validation', device)
    test = load_split(args.out/'data/test', device)
    if (set(training['root_ids'])&set(validation['root_ids']) or
            set(training['root_ids'])&set(test['root_ids']) or
            set(validation['root_ids'])&set(test['root_ids'])):
        raise AssertionError('Root cases leaked across train/validation/test')
    model = FourHitScheduleScorer(hidden_dims=args.hidden_dims).to(device)
    base_step = 0
    if args.continue_from is not None:
        initial = torch.load(args.continue_from/'best.pt', map_location=device, weights_only=False)
        if initial['format'] != model.format or initial['reward_scale'] != model.reward_scale:
            raise ValueError('Continuation checkpoint does not match the scorer')
        model.load_state_dict(initial['model'])
        base_step = int(initial['step'])
        torch.manual_seed(args.seed+base_step)
    else:
        torch.save(dict(format=model.format, model=model.state_dict(), reward_scale=model.reward_scale,
            hidden_dims=model.hidden_dims, step=0, critic=True, bellman=True), args.out/'initial.pt')
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = (None if args.constant_lr else
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.updates, eta_min=3e-5))
    replay = training['transitions']; best = float('inf'); best_step = 0
    sampling_rng = (torch.Generator(device=device).manual_seed(args.batch_seed+base_step)
                    if args.batch_seed is not None else None)
    if args.continue_from is not None:
        model.eval()
        with torch.no_grad():
            baseline = selection_metrics(model, validation['features'], validation['values'], validation['graph_index'])
        best = baseline['regret_mean']; best_step = base_step
        initial = dict(initial, validation=baseline, continuation_source=str(args.continue_from/'best.pt'))
        torch.save(initial, args.out/'best.pt')
        log(writer, 'validation', baseline, base_step)
        write_json(args.out/'continuation_baseline.json', dict(step=base_step, validation=baseline,
            optimizer='Fresh AdamW; source checkpoint contains model weights only', target='Initialized from restored model'))
    for update in range(1, args.updates+1):
        step = base_step+update
        model.train()
        ids = torch.randint(len(training['features']), (args.batch,), device=device, generator=sampling_rng)
        prediction = model(training['features'][ids])
        labels = training['values'][ids]/model.reward_scale
        value, ranking = value_and_ranking_loss(prediction, labels,
            top_k=args.ranking_top_k, gap_cap=args.ranking_gap_cap)
        relative = relative_score_loss(prediction, labels)
        tid = torch.randint(len(replay['obs']), (args.batch,), device=device, generator=sampling_rng)
        current = model(replay['obs'][tid]).gather(1, replay['action'][tid, None]).squeeze(-1)
        with torch.no_grad():
            next_value = target(replay['next_obs'][tid]).gather(1, replay['next_action'][tid, None]).squeeze(-1)
            bellman_target = replay['reward'][tid]+next_value*(~replay['done'][tid])
        suffix_value = F.smooth_l1_loss(current, replay['target'][tid])
        bellman = F.smooth_l1_loss(current, bellman_target)
        loss = value+args.ranking_weight*ranking+args.relative_weight*relative+.5*suffix_value+.1*bellman
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            for dst,src in zip(target.parameters(), model.parameters()):
                dst.lerp_(src, .01)
        if update%100 == 0 or update in (1, args.updates):
            model.eval()
            stats = selection_metrics(model, validation['features'], validation['values'], validation['graph_index'])
            fit = selection_metrics(model, training['features'][:2048], training['values'][:2048], training['graph_index'][:2048])
            losses = dict(value_loss=float(value), ranking_loss=float(ranking), relative_value_loss=float(relative), suffix_value_loss=float(suffix_value),
                          bellman_loss=float(bellman), total_loss=float(loss), learning_rate=optimizer.param_groups[0]['lr'])
            log(writer, 'train', losses, step); log(writer, 'train_selection', fit, step); log(writer, 'validation', stats, step)
            checkpoint = dict(format=model.format, model=model.state_dict(), reward_scale=model.reward_scale,
                hidden_dims=model.hidden_dims,
                step=step, parameters=sum(p.numel() for p in model.parameters()), critic=True, bellman=True,
                feature_spec='q17/v17/tip_position6/tip_velocity6/four*(drum_onehot7/tth/speed/valid)',
                generator_checkpoint=str(args.checkpoint), validation=stats,
                continuation_source=str(args.continue_from) if args.continue_from is not None else None,
                objective=dict(ranking_weight=args.ranking_weight, ranking_top_k=args.ranking_top_k,
                    ranking_gap_cap=args.ranking_gap_cap, relative_weight=args.relative_weight))
            torch.save(checkpoint, args.out/'last.pt')
            if stats['regret_mean'] < best:
                best = stats['regret_mean']; best_step = step; torch.save(checkpoint, args.out/'best.pt')
            write_json(args.out/'progress.json', dict(stage='training', step=step, additional_updates=update,
                best_step=best_step, **losses, validation=stats))
            print(json.dumps(dict(step=step, validation=stats, **losses)), flush=True)
    saved = torch.load(args.out/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model']); model.eval()
    stats = selection_metrics(model, test['features'], test['values'], test['graph_index'])
    log(writer, 'test', stats, best_step)
    write_json(args.out/'test_selection.json', dict(best_step=best_step, test_windows=len(test['features']), **stats))
    return model, stats


@torch.no_grad()
def physical_test(args, model, writer):
    device = torch.device(args.devices[0]); env, kit = setup(args.checkpoint, device, args.chunk)
    parts = [torch.load(p, weights_only=False) for p in sorted((args.out/'data/test').glob('*.pt'))]
    rng = np.random.default_rng(args.seed+3000000); sums = {}; total = 0; latency = []
    for part in parts:
        q = part['q'].to(device); v = part['v'].to(device); n = len(q)
        table = physical_table(part['clips'], kit, device)
        features = model.features(table, q, v, env.cost.fk)
        torch.testing.assert_close(features, part['features'].to(device), atol=2e-5, rtol=2e-5)
        torch.cuda.synchronize(device); begin = time.perf_counter()
        choice, actions = model.choose(features); torch.cuda.synchronize(device)
        latency.append((time.perf_counter()-begin)*1000)
        bank = schedule_bank(device)
        selected = torch.stack((bank[part['graph_index'].to(device)], actions,
                                bank[part['chosen'].to(device)]), 1)
        draws = torch.as_tensor(rng.uniform(.2, .8, (n, 4, 4, 2)), device=device, dtype=torch.float32)
        out = env.score(table, q, v, selected, draws)
        for c,name in enumerate(('graph', 'mlp', 'search_teacher')):
            row = sums.setdefault(name, dict(hits=0, requested=0, strict_hits=0, relaxed_hits=0, score_sum=0., collision_sum=0.))
            row['hits'] += int(out['hits'][:, c].sum()); row['requested'] += int(out['requested'][:, c].sum())
            row['strict_hits'] += int(out['strict_valid_hits'][:, c].sum()); row['relaxed_hits'] += int(out['relaxed_valid_hits'][:, c].sum())
            row['score_sum'] += float(out['score'][:, c].sum(-1).mean(-1).sum())
            row['collision_sum'] += float(out['collision_m'][:, c].amax(-1).mean(-1).sum())
        total += n
    report = dict(test_windows=total, heldout_bounce_draws=4,
        evaluation='Static four-hit schedule, actual decoder spline tracking and analytic rebound, fresh bounce draws; no Isaac',
        policy_batch_ms_median=float(np.median(latency)), policies={})
    for name,row in sums.items():
        report['policies'][name] = dict(hit_rate=row['hits']/row['requested'],
            strict_valid_hit_rate=row['strict_hits']/row['requested'],
            relaxed_valid_hit_rate=row['relaxed_hits']/row['requested'],
            score_mean=row['score_sum']/total, mean_worst_collision_mm=1000*row['collision_sum']/total,
            hits=row['hits'], requested=row['requested'])
        log(writer, 'physical_test/'+name, report['policies'][name], total)
    # End-to-end decision latency includes computing the 86-feature observation.
    timing = []
    one = {k:t[:1] for k,t in table.items()}
    for _ in range(20):
        torch.cuda.synchronize(device); begin = time.perf_counter()
        model.assign(one, q[:1], v[:1], env.cost.fk); torch.cuda.synchronize(device)
        timing.append((time.perf_counter()-begin)*1000)
    report['single_window_decision_ms_median'] = float(np.median(timing))
    write_json(args.out/'physical_test.json', report)
    return report


def parent(args):
    if args.out.exists():
        raise FileExistsError('Choose a fresh run directory')
    args.out.mkdir(parents=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'), flush_secs=5)
    registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
    (registry/args.out.name).symlink_to((args.out/'tb').resolve())
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(critic=True, bellman=True, discount=1., generator_frozen=True,
        scorer_initialization='restored best weights' if args.continue_from else 'fresh random weights',
        source='Independent procedural four-single-hit rhythms and collision-clear pose mixture',
        split='Separate root cases and RNG streams; every derivative remains in its root split',
        network='86 -> '+' -> '.join(f'{width} LN GELU' for width in args.hidden_dims)+' -> 16',
        loss=f'Root MC Huber + {args.ranking_weight} ranking + {args.relative_weight} centered root MC Huber + 0.5 suffix MC Huber + 0.1 finite-schedule Bellman Huber',
        ranking_mode='teacher top-k, gap-weighted, per-window mean' if args.ranking_top_k else 'uniform all-pairs',
        collection_cuda_graphs=False, drum_hand_masks=False, physics='Full actual decoder spline prefixes, both arms and waist')
    write_json(args.out/'config.json', config)
    writer.add_scalar('collection/completed_windows', 0, 0); writer.flush()
    workers = []; begin = time.monotonic()
    try:
        data_source = args.continue_from or args.data_from
        if data_source is not None:
            (args.out/'data').symlink_to((data_source/'data').resolve(), target_is_directory=True)
            write_json(args.out/'run.json', dict(status='running', stage='training', pid=os.getpid(),
                data_source=str(data_source),
                continuation_source=str(args.continue_from) if args.continue_from else None,
                optimization_updates=args.updates))
        else:
            for rank,device in enumerate(args.devices):
                f = (args.out/f'w{rank}.log').open('w')
                cmd = [sys.executable, str(Path(__file__).resolve()), '--collect-worker', '--out', str(args.out),
                    '--checkpoint', str(args.checkpoint), '--device', device, '--rank', str(rank),
                    '--workers', str(len(args.devices)), '--seed', str(args.seed), '--chunk', str(args.chunk),
                    '--collection-windows', str(args.collection_windows), '--train-windows', str(args.train_windows),
                    '--val-windows', str(args.val_windows), '--test-windows', str(args.test_windows)]
                workers.append((subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT), f))
            write_json(args.out/'run.json', dict(status='running', stage='collection', pid=os.getpid(), worker_pids=[p.pid for p,_ in workers]))
            while any(p.poll() is None for p,_ in workers):
                if any(p.poll() not in (None, 0) for p,_ in workers):
                    raise RuntimeError('Collection worker failed; inspect worker logs')
                progress = [json.loads(p.read_text()) for p in args.out.glob('w*_progress.json')]
                completed = sum(p['completed_windows'] for p in progress)
                writer.add_scalar('collection/completed_windows', completed, int(time.monotonic()-begin)); writer.flush()
                write_json(args.out/'progress.json', dict(stage='collection', completed_windows=completed,
                    target_windows=args.train_windows+args.val_windows+args.test_windows, elapsed_s=time.monotonic()-begin))
                time.sleep(3)
            if any(p.returncode for p,_ in workers):
                raise RuntimeError('Collection worker failed; inspect worker logs')
        model, selection = train(args, writer)
        report = physical_test(args, model, writer)
        completed = dict(status='complete', elapsed_s=time.monotonic()-begin,
            parameters=sum(p.numel() for p in model.parameters()), test_selection=selection, physical_test=report)
        write_json(args.out/'run.json', completed)
        write_json(args.out/'progress.json', dict(stage='complete', **completed))
        print(json.dumps(dict(status='complete', test_selection=selection, physical_test=report)), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc))); raise
    finally:
        for p,f in workers:
            if p.poll() is None:
                p.terminate()
            f.close()
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    p.add_argument('--device', default='cuda:0'); p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--collect-worker', action='store_true')
    p.add_argument('--train-windows', type=int, default=16384); p.add_argument('--val-windows', type=int, default=2048)
    p.add_argument('--test-windows', type=int, default=512); p.add_argument('--collection-windows', type=int, default=64)
    p.add_argument('--chunk', type=int, default=2048); p.add_argument('--updates', type=int, default=3000)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--seed', type=int, default=202610047)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--continue-from', type=Path, help='Reuse a run\'s data and start from its best scorer weights')
    source.add_argument('--data-from', type=Path, help='Reuse a run\'s data with fresh random scorer weights')
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--constant-lr', action='store_true', help='Disable cosine learning rate decay')
    p.add_argument('--ranking-weight', type=float, default=.1)
    p.add_argument('--ranking-top-k', type=int, default=0, help='Rank pairs involving teacher top-k schedules; zero uses all pairs')
    p.add_argument('--ranking-gap-cap', type=float, default=4., help='Maximum pair weight in normalized score units')
    p.add_argument('--relative-weight', type=float, default=0., help='Weight on regression of scores after subtracting the window mean')
    p.add_argument('--hidden-dims', nargs='+', type=int, help='Hidden MLP widths; default 256 256 128')
    p.add_argument('--batch-seed', type=int, help='Separate minibatch RNG for matched architecture comparisons')
    args = p.parse_args(); args.out = args.out.resolve(); args.checkpoint = args.checkpoint.resolve()
    if args.updates < 1 or args.learning_rate <= 0:
        p.error('Updates and learning rate must be positive')
    if not 0 <= args.ranking_top_k <= 16 or min(args.ranking_weight, args.relative_weight) < 0 or args.ranking_gap_cap <= 0:
        p.error('Invalid ranking or relative loss configuration')
    if args.hidden_dims is not None and min(args.hidden_dims) < 1:
        p.error('Hidden dimensions must be positive')
    data_source = args.continue_from or args.data_from
    if data_source is not None:
        data_source = data_source.resolve()
        if args.continue_from is not None:
            args.continue_from = data_source
        else:
            args.data_from = data_source
        if args.collect_worker:
            p.error('Data reuse cannot collect new shards')
        required = ['config.json', 'data/train', 'data/validation', 'data/test']
        if args.continue_from is not None:
            required.append('best.pt')
        for path in required:
            if not (data_source/path).exists():
                p.error(f'Data source is missing {path}')
        original = json.loads((data_source/'config.json').read_text())
        args.checkpoint = Path(original['checkpoint']).resolve()
        args.seed = original['seed']
        for key in ('train_windows', 'val_windows', 'test_windows'):
            setattr(args, key, original[key])
    if args.continue_from is not None:
        checkpoint_dims = torch.load(args.continue_from/'best.pt', map_location='cpu', weights_only=False).get('hidden_dims', (256,256,128))
        if args.hidden_dims is not None and tuple(args.hidden_dims) != tuple(checkpoint_dims):
            p.error('Continuation requires the checkpoint architecture; use --data-from for fresh weights')
        args.hidden_dims = list(checkpoint_dims)
    else:
        args.hidden_dims = args.hidden_dims or [256,256,128]
    worker_count = args.workers if args.collect_worker else len(args.devices)
    if min(args.train_windows, args.val_windows, args.test_windows) < worker_count:
        p.error('Every split must contain at least one root per worker')
    collect(args) if args.collect_worker else parent(args)


if __name__ == '__main__':
    main()
