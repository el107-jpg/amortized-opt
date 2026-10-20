#!/usr/bin/env python3
"""Adaptive fixed-decoder/QP transition calibration using saved one-second starts."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.ik import arm_joint_indices
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.transition_assignment import TransitionTable, minimum_statistics
from isaac_rl.transition_calibration import strike_context
from isaac_rl.transition_time_search import coarse_horizons, fine_horizons
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, metrics, refine
from scripts.probe_transition_qp import save_npz, write_json
from scripts.rebuild_transition_database import accepted_gates

POLICY = {'left': ['floor_tom'], 'right': ['hihat', 'cymbal_left']}


def decoder_rows(rows):
    """Reuse raw diagnostics, excluding every repaired witness and metric."""
    before = {k: v for k, v in rows.items() if k.startswith('before_')}
    return dict(keys=rows['keys'], **before,
        **{'after_'+k.removeprefix('before_'): v for k, v in before.items()})


def inputs(args):
    region_cohort = (args.baseline/'start_cohort.npz').exists()
    config = json.loads((args.baseline/('config.json' if region_cohort else 'qp/config.json')).read_text())
    if (not config['ignore_idle_arm'] or config['impact_speed_required']
            or config['hit_depth_mm'] != 70 or not config['allow_early_target_top_entry']):
        raise ValueError('Baseline uses incompatible contact or constraint settings')
    if region_cohort and not args.decoder_only:
        raise ValueError('Random-region cohort is decoder-only')
    if not region_cohort and config['qp']['iterations'] != 5:
        raise ValueError('Baseline must use five QP iterations')
    table = TransitionTable.load(args.baseline/('transition_table.npz' if region_cohort else
                                              'database/transition_table_qp.npz'))
    table.metadata['forbidden_drums_by_hand'] = POLICY
    with np.load(args.baseline/('start_cohort.npz' if region_cohort else 'qp/deadline_050.npz')) as data:
        rows = {k: data[k].copy() for k in data.files}
    keys = rows['keys']
    eligible = rows['departure_posture_valid'] & table.allowed_groups()[tuple(keys[:, :4].T)]
    if args.reuse_raw_from:
        prior = json.loads((args.reuse_raw_from/'config.json').read_text())
        fields = ('checkpoint', 'checkpoint_step', 'source_calibration', 'ignore_idle_arm',
                  'hit_depth_mm', 'impact_speed_required', 'allow_early_target_top_entry',
                  'collocation_substeps')
        if (any(prior[k] != config[k] for k in fields)
                or prior['forbidden_drums_by_hand'] != POLICY
                or prior['baseline_run'] != str(args.baseline)
                or len(prior['devices']) != (len(args.devices) if args.devices else args.workers)):
            raise ValueError('Raw cache has a different generator, physics, start cohort or partition')
    return config, table, rows, eligible


def worker(args):
    torch.set_num_threads(2)
    np.random.seed(20261004); torch.manual_seed(20261004)
    config, table, baseline, eligible = inputs(args)
    ids = np.flatnonzero(eligible & (np.arange(len(eligible)) % args.workers == args.rank))
    if args.limit: ids = ids[:args.limit]
    keys = baseline['keys'][ids]
    folder = args.out/f'w{args.rank}'; folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(checkpoint), device)
        cfg = A.CostConfig(**checkpoint['train_cfg']['cost'])
        cfg.n_sub = config['collocation_substeps']
        cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    cost.calibration_active_hand_only = True
    model.load_state_dict(checkpoint['model']); model.eval()
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(checkpoint.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    source = Path(config['source_calibration'])
    states = {}
    qs, vs = [], []
    for h, src, dst, y, s in keys:
        name = (h, src, y)
        if name not in states:
            with np.load(source/'samples'/f'h{h}_d{src}_y{y}.npz') as d:
                states[name] = d['q'].copy(), d['v'].copy()
        qs.append(states[name][0][s]); vs.append(states[name][1][s])
    Q = torch.as_tensor(np.asarray(qs).reshape(-1, 17), device=device)
    V = torch.as_tensor(np.asarray(vs).reshape(-1, 17), device=device)
    coarse = coarse_horizons(args.stride)
    # -1 means untested: it must never be interpreted as a failed evaluation.
    raw = np.full((len(keys), 101), -1, np.int8); repaired = raw.copy()
    raw[:, 50] = baseline['before_success'][ids]
    repaired[:, 50] = baseline['before_success' if args.decoder_only else 'after_success'][ids]
    begin = time.monotonic(); evaluations = 0
    qp_cfg = QPConfig(iterations=5, jacobian_chunk=args.jacobian_chunk)

    def evaluate(N, candidates, stage):
        nonlocal evaluations
        path = folder/f'{stage}_{N:03d}.npz'
        if path.exists():
            with np.load(path) as d: rows = {k: d[k].copy() for k in d.files}
            expected_keys = set(map(tuple, keys[candidates]))
            if set(map(tuple, rows['keys'])) != expected_keys:
                raise ValueError('Cached cohort differs from this search configuration')
            lookup = {tuple(k): i for i, k in enumerate(keys)}
            take = np.asarray([lookup[tuple(k)] for k in rows['keys']], int)
            accepted_gates(rows)
            raw[take, N] = rows['before_success']; repaired[take, N] = rows['after_success']
            evaluations += len(take)
            return
        short_cost = cost_for_horizon(cost, N)
        parts = []
        if args.decoder_only and args.reuse_raw_from:
            reused = set()
            lookup = {tuple(keys[i]): i for i in candidates}
            for cached in sorted((args.reuse_raw_from/f'w{args.rank}').glob(f'*_{N:03d}.npz')):
                with np.load(cached) as d: rows = decoder_rows({k: d[k].copy() for k in d.files})
                indexes = [i for i, key in enumerate(rows['keys'])
                    if tuple(key) in lookup and tuple(key) not in reused]
                if not indexes: continue
                rows = {k: v[indexes] for k, v in rows.items()}
                accepted_gates(rows); parts.append(rows)
                take = np.asarray([lookup[tuple(k)] for k in rows['keys']], int)
                raw[take, N] = rows['before_success']; repaired[take, N] = rows['before_success']
                reused.update(map(tuple, rows['keys'])); evaluations += len(take)
            candidates = np.asarray([i for i in candidates if tuple(keys[i]) not in reused], int)
        for hand in (0, 1):
            indexes = candidates[keys[candidates, 0] == hand]
            for start in range(0, len(indexes), args.chunk):
                take = indexes[start:start+args.chunk]; q, v = Q[take], V[take]
                B = len(take)
                targets = np.full((B, 2), -1, int); targets[:, hand] = keys[take, 2]
                sources = np.full((B, 2), -1, int); sources[:, hand] = keys[take, 1]
                with torch.no_grad():
                    plan = generator.predict(q, v, torch.as_tensor(targets[:, None], device=device),
                        q.new_full((B, 1), N*.02), q.new_full((B, 1, 2), 2.),
                        torch.ones(B, 1, device=device, dtype=torch.bool), 0, return_plan=True,
                        waist_yaw=(q[:, 0] if config.get('yaw_sampling') == 'uniform_within_region'
                                   else np.asarray(config['yaw_centers'])[keys[take, 3]]))
                    traj = plan['traj'][:, :N].clone()
                    idle = arm_joint_indices(('left', 'right')[1-hand])
                    traj[:, :, idle] = q[:, None, idle]
                    lengths = torch.full_like(plan['lengths'], N)
                    ctx = strike_context(kit, targets, sources, np.full(B, N), device,
                        speeds=np.full((B, 2), 2.))
                    ctx.target_top_entry_depth_m = .07
                if args.decoder_only:
                    started = time.monotonic()
                    before = metrics(short_cost, q, v, traj, ctx, lengths, .07)
                    after = before
                    info = {'elapsed_s': time.monotonic()-started}
                else:
                    _, before, after, info = refine(short_cost, q, v, traj, ctx, lengths,
                        hand, depth_m=.07, cfg=qp_cfg)
                rows = dict(keys=keys[take], **{'before_'+k: val for k, val in before.items()},
                    **{'after_'+k: val for k, val in after.items()})
                accepted_gates(rows); parts.append(rows)
                raw[take, N] = before['success']; repaired[take, N] = after['success']
                evaluations += B
                write_json(folder/'progress.json', dict(status='running', rank=args.rank,
                    stage=stage, deadline_s=N*.02, starts=len(keys), new_evaluations=evaluations,
                    elapsed_s=time.monotonic()-begin,
                    supported_starts=int((repaired == 1).any(-1).sum())))
                print(json.dumps(dict(stage=stage, N=N, processed=evaluations,
                    before=int(before['success'].sum()), after=int(after['success'].sum()),
                    chunk_s=info['elapsed_s'])), flush=True)
        if parts:
            save_npz(path, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})

    for N in coarse:
        if N != 50: evaluate(N, np.arange(len(keys)), 'coarse')
    needed = np.zeros_like(raw, bool)
    for i in range(len(keys)):
        # Refine raw and repaired minima independently, on identical fresh seeds.
        for outcomes in ((raw,) if args.decoder_only else (raw, repaired)):
            needed[i, fine_horizons(coarse, outcomes[i, coarse] == 1)] = True
    for N in range(1, 101):
        candidates = np.flatnonzero(needed[:, N] & (raw[:, N] == -1))
        if len(candidates): evaluate(N, candidates, 'fine')
    if args.stride == 1 and (raw[:, 1:] < 0).any():
        raise ValueError('Exhaustive sweep did not evaluate every start/deadline pair')
    save_npz(folder/'results.npz', keys=keys, raw_outcomes=raw, qp_outcomes=repaired,
        fine_required=needed)
    write_json(folder/'done.json', dict(status='complete', starts=len(keys),
        new_evaluations=evaluations, elapsed_s=time.monotonic()-begin))


def publish(args, config, table, baseline, eligible, complete):
    shape = (*table.minimum_s.shape, config['samples'])
    raw = np.full(shape, np.inf); qp = raw.copy(); admissible = np.zeros(shape, bool)
    admissible[tuple(baseline['keys'].T)] = eligible
    raw[tuple(baseline['keys'][eligible & baseline['before_success']].T)] = 1.
    qp[tuple(baseline['keys'][eligible & baseline[
        'before_success' if args.decoder_only else 'after_success']].T)] = 1.
    seen = set(); new_evaluations = 0
    for rank in range(len(args.devices)):
        folder = args.out/f'w{rank}'
        for path in sorted(folder.glob('*.npz')):
            if path.name == 'results.npz': continue
            N = int(path.stem.rsplit('_', 1)[1])
            if not 1 <= N <= 100 or N == 50:
                raise ValueError('Invalid deadline or duplicate one-second cohort')
            with np.load(path) as d: rows = {k: d[k].copy() for k in d.files}
            accepted_gates(rows)
            for i, key in enumerate(rows['keys']):
                key = tuple(key)
                if (key, N) in seen: raise ValueError('Overlapping evaluations')
                seen.add((key, N)); new_evaluations += 1
                if not admissible[key]: raise ValueError('Witness outside eligible policy')
                if rows['before_success'][i]: raw[key] = min(raw[key], N*.02)
                if rows['after_success'][i]: qp[key] = min(qp[key], N*.02)
    if complete and args.stride == 1 and len(seen) != int(eligible.sum())*99:
        raise ValueError('Completed sweep lacks full 100-deadline coverage')
    stats = [minimum_statistics(g[..., None], np.isfinite(g[..., None])) for g in (raw, qp)]
    summary = dict(status='complete' if complete else 'running',
        allowed_groups=int(table.allowed_groups().sum()), eligible_starts=int(eligible.sum()),
        new_evaluations=new_evaluations, reused_one_second_evaluations=int(eligible.sum()),
        decoder_supported_starts=int(np.isfinite(raw).sum()), qp_supported_starts=int(np.isfinite(qp).sum()),
        decoder_supported_groups=int((stats[0]['count'] > 0).sum()),
        qp_supported_groups=int((stats[1]['count'] > 0).sum()),
        forbidden_drums_by_hand=POLICY, resolution_s=.02, range_s=[.02, 2.],
        search=('Every 20 ms deadline from 20 ms through 2 seconds' if args.stride == 1 else
            f'{args.stride*.02:g} s coarse scan with complete 20 ms scans below lowest passing anchors'),
        exhaustive_range_search=args.stride == 1,
        decoder_only=args.decoder_only,
        limitation=('Lowest accepted grid time for each sampled state and fixed generator; not a physical lower bound'
            if args.stride == 1 else 'Disconnected feasible islands between failed coarse anchors can be missed'))
    if args.decoder_only:
        summary.pop('qp_supported_starts'); summary.pop('qp_supported_groups')
    out = args.out/'database'; out.mkdir(exist_ok=True)
    metadata = dict(config, forbidden_drums_by_hand=POLICY, minimum_time_search=True,
        deadline_search='full-range' if args.stride == 1 else 'coarse-to-fine',
        coarse_stride_s=args.stride*.02, resolution_s=.02,
        deadlines_s=(np.arange(1, 101)*.02).tolist() if args.stride == 1 else None,
        candidate_deadlines_s=(np.arange(1, 101)*.02).tolist(),
        duration_trials=100 if args.stride == 1 else None,
        deadline_coverage='All 100 deadlines per eligible start' if args.stride == 1 else
            'Variable per start; see worker outcome arrays (-1 means untested)',
        range_s=[.02, 2.], duration_search=summary['search'],
        timing_interpretation='Mean of per-start lowest accepted sampled deadlines; not physical lower bounds',
        database_status=summary['status'], baseline_run=str(args.baseline),
        complete_deadline_range=complete and args.stride == 1, early_stop_on_failure=False,
        pair_yaw_recalibrated=False, search_coverage=summary,
        qp={'iterations': 0} if args.decoder_only else config['qp'],
        refinement='None' if args.decoder_only else config['refinement'],
        path_generation='One frozen decoder pass' if args.decoder_only else 'Decoder plus up to five QP steps')
    for name, stat in zip(('decoder',) if args.decoder_only else ('decoder', 'qp'), stats):
        current = replace(table, minimum_s=stat['mean'], support=stat['count'],
            trials=admissible.sum(-1), metadata=metadata)
        temporary = out/f'.transition_table_{name}.npz'
        current.save(temporary); temporary.replace(out/f'transition_table_{name}.npz')
    save_npz(out/'per_start_minima.npz', decoder_minimum_s=raw,
        **({} if args.decoder_only else {'qp_minimum_s': qp}),
        eligible=admissible, allowed_groups=table.allowed_groups(),
        decoder_std_s=stats[0]['std'], qp_std_s=stats[1]['std'])
    temporary = out/'.transition_table.csv'
    with temporary.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['hand', 'source', 'destination', 'initial_yaw_rad'] + (
            ['decoder_mean_min_s', 'decoder_std_s', 'decoder_supported_starts', 'eligible_starts']
            if args.decoder_only else ['decoder_mean_min_s', 'qp_mean_min_s', 'qp_std_s',
                'decoder_supported_starts', 'qp_supported_starts', 'eligible_starts']) + (
                ['yaw_region_low_rad', 'yaw_region_high_rad'] if 'yaw_region_edges' in config else []))
        finite = lambda v: float(v) if np.isfinite(v) else ''
        for k in np.ndindex(table.minimum_s.shape):
            if not table.allowed_groups()[k]: continue
            h, src, dst, y = k
            w.writerow([('left', 'right')[h], table.drum_names[src], table.drum_names[dst],
                table.yaw_centers[y]] + ([finite(stats[0]['mean'][k]), stats[0]['std'][k],
                    stats[0]['count'][k], admissible[k].sum()] if args.decoder_only else
                    [finite(stats[0]['mean'][k]), finite(stats[1]['mean'][k]), stats[1]['std'][k],
                     stats[0]['count'][k], stats[1]['count'][k], admissible[k].sum()]) + (
                     config['yaw_region_edges'][y:y+2] if 'yaw_region_edges' in config else []))
    temporary.replace(out/'transition_table.csv')
    write_json(out/'summary.json', summary)
    return summary


def parent(args):
    from torch.utils.tensorboard import SummaryWriter
    args.out.mkdir(parents=True, exist_ok=True)
    config, table, baseline, eligible = inputs(args)
    config.update(forbidden_drums_by_hand=POLICY, baseline_run=str(args.baseline),
        coarse_horizons=coarse_horizons(args.stride), grid_s=.02, stride=args.stride,
        devices=args.devices, chunk=args.chunk, limit=args.limit, training=False,
        deadline_search='full-range' if args.stride == 1 else 'coarse-to-fine',
        deadlines_s=(np.arange(1, 101)*.02).tolist() if args.stride == 1 else None,
        duration_trials=100 if args.stride == 1 else None,
        duration_search='Every 20 ms deadline from 20 ms through 2 seconds' if args.stride == 1 else
            'Coarse scan of 20 ms to 2 s, then 20 ms refinement below passing anchors')
    config.update(decoder_only=args.decoder_only, reuse_raw_from=str(args.reuse_raw_from)
        if args.reuse_raw_from else None)
    if args.decoder_only: config.update(qp={'iterations': 0}, refinement='None')
    write_json(args.out/'config.json', config)
    writer = SummaryWriter(str(args.out/'tb'))
    registry = ROOT/'runs/tensorboard_active'/args.out.name
    if not registry.is_symlink(): registry.symlink_to((args.out/'tb').resolve())
    begin = time.monotonic(); workers = []
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('a')
            command = [sys.executable, str(Path(__file__).resolve()), '--baseline', str(args.baseline),
                '--out', str(args.out), '--device', device, '--rank', str(rank),
                '--workers', str(len(args.devices)), '--chunk', str(args.chunk),
                '--stride', str(args.stride), '--jacobian-chunk', str(args.jacobian_chunk)]
            if args.limit: command += ['--limit', str(args.limit)]
            if args.decoder_only: command.append('--decoder-only')
            if args.reuse_raw_from: command += ['--reuse-raw-from', str(args.reuse_raw_from)]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
            workers.append((subprocess.Popen(command, cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env=env), log))
        write_json(args.out/'run.json', dict(status='running', worker_pids=[p.pid for p, _ in workers]))
        while True:
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('Deadline worker failed; inspect worker logs')
            complete = all(p.poll() == 0 for p, _ in workers)
            summary = publish(args, config, table, baseline, eligible, complete)
            summary['elapsed_s'] = time.monotonic()-begin
            progress = [json.loads(p.read_text()) for p in args.out.glob('w*/progress.json')]
            summary['processed_evaluations'] = sum(p['new_evaluations'] for p in progress)
            summary['processing_rate_plans_s'] = summary['processed_evaluations']/max(summary['elapsed_s'], 1)
            write_json(args.out/'progress.json', summary)
            for key in ('new_evaluations', 'processed_evaluations', 'processing_rate_plans_s',
                        'decoder_supported_starts', 'decoder_supported_groups',
                        'qp_supported_starts', 'qp_supported_groups'):
                if key not in summary: continue
                writer.add_scalar('search/'+key, summary[key], int(summary['elapsed_s']))
            writer.flush()
            if complete:
                write_json(args.out/'summary.json', summary)
                write_json(args.out/'run.json', summary); break
            time.sleep(15)
    except BaseException as error:
        write_json(args.out/'run.json', dict(status='failed', error=str(error)))
        raise
    finally:
        writer.close()
        for process, log in workers:
            if process.poll() is None: process.terminate()
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=ROOT/'runs/transition_active_arm_1s_20261004_165905')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--devices', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--jacobian-chunk', type=int, default=256)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--decoder-only', action='store_true')
    p.add_argument('--reuse-raw-from', type=Path)
    args = p.parse_args(); args.out = args.out.resolve(); args.baseline = args.baseline.resolve()
    if args.chunk < 1 or args.jacobian_chunk < 1 or not 1 <= args.stride <= 50:
        p.error('Invalid chunk or coarse stride')
    if not 0 <= args.rank < args.workers or (args.devices and len(set(args.devices)) != len(args.devices)):
        p.error('Invalid worker rank or duplicate devices')
    parent(args) if args.devices else worker(args)


if __name__ == '__main__': main()
