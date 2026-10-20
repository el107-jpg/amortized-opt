#!/usr/bin/env python3
"""Repair every independently generated transition path with a few local QPs."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
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
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, metrics, refine
from scripts.audit_transition_deadlines import records


def write_json(path, data):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = Path(str(path)+'.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def report(parts):
    keys = np.concatenate([p['keys'] for p in parts])
    bounded = np.concatenate([p['departure_posture_valid'] for p in parts])
    before = np.concatenate([p['before_success'] for p in parts])
    after = np.concatenate([p['after_success'] for p in parts])
    out = dict(trajectories=len(keys), before_feasible=int(before.sum()),
        after_feasible=int(after.sum()), recovered=int((after & ~before).sum()),
        lost=int((before & ~after).sum()), within_bounds=dict(trajectories=int(bounded.sum()),
            before_feasible=int((before & bounded).sum()), after_feasible=int((after & bounded).sum()),
            recovered=int((after & ~before & bounded).sum())))
    for field in ('terminal_region_dist', 'torque_excess_max', 'velocity_excess_max',
                  'joint_box_excess_max', 'collision_min_sdf'):
        old = np.concatenate([p['before_'+field] for p in parts])
        new = np.concatenate([p['after_'+field] for p in parts])
        if field == 'collision_min_sdf': old, new = np.maximum(-old, 0), np.maximum(-new, 0)
        failed = (old > 1e-6) & bounded
        out[field] = dict(failures_before=int(failed.sum()), failures_after=int(((new > 1e-6)&bounded).sum()),
            median_before_on_initial_failures=float(np.median(old[failed])) if failed.any() else None,
            median_after_on_initial_failures=float(np.median(new[failed])) if failed.any() else None)
    return out


def worker(args):
    torch.set_num_threads(2)
    np.random.seed(20261004); torch.manual_seed(20261004)
    config = json.loads((args.audit/'config.json').read_text())
    table = TransitionTable.load(Path(config['source_calibration'])/'transition_table.npz')
    table.metadata.update(forbidden_drums_by_hand=config.get('forbidden_drums_by_hand', {}))
    ck = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(ck), device)
        cfg = A.CostConfig(**ck['train_cfg']['cost'])
        cfg.n_sub = config['collocation_substeps']
        cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(ck['model']); model.eval()
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    ignore_idle_arm = config.get('ignore_idle_arm', False)
    cost.calibration_active_hand_only = ignore_idle_arm
    data = records(Path(config['source_calibration']), table, device,
        cost=cost, kit=kit, ignore_idle_arm=ignore_idle_arm)
    keys = data['keys']
    selected = np.flatnonzero(data['valid'] & (np.arange(len(keys))%args.workers == args.rank))
    if args.smoke:
        wanted = {(0, 5, 6, 0, 4), (0, 5, 0, 1, 1), (0, 4, 2, 1, 1), (1, 5, 0, 3, 4)}
        selected = np.asarray([i for i in selected if tuple(keys[i]) in wanted])
    if args.limit: selected = selected[:args.limit]
    _, lo, hi = cost._joint_box_hard_limits()
    in_bounds = ((data['q'] >= lo.reshape(1, -1)-1e-6)&
                 (data['q'] <= hi.reshape(1, -1)+1e-6))
    if ignore_idle_arm:
        checked = torch.ones_like(in_bounds)
        for h in (0, 1):
            checked[keys[:, 0] == h, 3+7*(1-h):10+7*(1-h)] = False
        in_bounds = in_bounds | ~checked
    bounded = in_bounds.all(-1).cpu().numpy()
    with np.load(args.audit/'deadline_audit.npz') as audit:
        deadlines = audit['deadlines_s'].copy() if args.full_range else np.array([1.])
    qp_cfg = QPConfig(iterations=args.iterations, jacobian_chunk=args.jacobian_chunk)
    folder = args.out/f'w{args.rank}'; folder.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    completed = 0
    for deadline in deadlines:
        N = int(round(deadline/.02))
        output = folder/f'deadline_{N:03d}.npz'
        if output.exists() and (folder/f'deadline_{N:03d}.json').exists():
            with np.load(output) as cached: completed += len(cached['keys'])
            continue
        short_cost = cost_for_horizon(cost, N)
        parts, histories = [], []
        for hand in (0, 1):
            indexes = selected[keys[selected, 0] == hand]
            for start in range(0, len(indexes), args.chunk):
                ids = indexes[start:start+args.chunk]
                Q, V = data['q'][ids], data['v'][ids]
                B = len(ids)
                targets = np.full((B, 2), -1, int); targets[:, hand] = keys[ids, 2]
                sources = np.full((B, 2), -1, int); sources[:, hand] = keys[ids, 1]
                with torch.no_grad():
                    plan = generator.predict(Q, V, torch.as_tensor(targets[:, None], device=device),
                        Q.new_full((B, 1), deadline), Q.new_full((B, 1, 2), 2.),
                        torch.ones(B, 1, device=device, dtype=torch.bool), 0, return_plan=True,
                        waist_yaw=np.asarray(config['yaw_centers'])[keys[ids, 3]])
                    traj = plan['traj'][:, :N].clone()
                    idle = arm_joint_indices(('left', 'right')[1-hand])
                    traj[:, :, idle] = Q[:, None, idle]
                    lengths = torch.full_like(plan['lengths'], N)
                    ctx = strike_context(kit, targets, sources, np.full(B, N), device,
                        speeds=np.full((B, 2), 2.))
                    ctx.target_top_entry_depth_m = .07
                repaired, before, after, info = refine(short_cost, Q, V, traj, ctx,
                    lengths, hand, depth_m=.07, cfg=qp_cfg)
                part = dict(keys=keys[ids], departure_posture_valid=bounded[ids],
                    **{'before_'+k: val for k, val in before.items()},
                    **{'after_'+k: val for k, val in after.items()})
                parts.append(part)
                for key, history in zip(keys[ids], info['histories']):
                    histories.append(dict(key=key.tolist(), deadline_s=float(deadline), iterations=history))
                if args.smoke:
                    np.savez_compressed(folder/f'paths_h{hand}.npz', q=Q.cpu().numpy(), v=V.cpu().numpy(),
                        before=traj.cpu().numpy(), after=repaired.cpu().numpy(), keys=keys[ids])
                completed += B
                write_json(folder/'progress.json', dict(rank=args.rank, deadline_s=float(deadline),
                    processed=completed, expected=int(len(selected)*len(deadlines)),
                    elapsed_s=time.monotonic()-begin, current_deadline=report(parts)))
                print(json.dumps(dict(rank=args.rank, deadline_s=float(deadline), processed=completed,
                    chunk_seconds=info['elapsed_s'], before=int(before['success'].sum()),
                    after=int(after['success'].sum()))), flush=True)
        combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
        save_npz(output, **combined)
        write_json(folder/f'deadline_{N:03d}.json', dict(deadline_s=float(deadline),
            summary=report(parts), solver_histories=histories))
    write_json(folder/'done.json', dict(status='complete', processed=completed, elapsed_s=time.monotonic()-begin))


def parent(args):
    from torch.utils.tensorboard import SummaryWriter
    args.out.mkdir(parents=True, exist_ok=True)
    existing = args.out/'config.json'
    prior = json.loads(existing.read_text()) if existing.exists() else {}
    pid_file = args.out/'launcher.pid'
    start_epoch = prior.get('run_start_epoch_s', pid_file.stat().st_mtime if pid_file.exists() else time.time())
    begin = time.monotonic()-(time.time()-start_epoch)
    meta = json.loads((args.audit/'config.json').read_text())
    meta.update(qp=asdict(QPConfig(iterations=args.iterations, jacobian_chunk=args.jacobian_chunk)),
        comparison_audit=str(args.audit), duration_and_q0_v0_fixed=True, training=False,
        refinement='Full-waypoint elastic SQP, worst path point per actuator/obstacle relinearized each iteration',
        failure_interpretation='Local repair failure; no proof of physical infeasibility',
        full_range=args.full_range, devices=args.devices, chunk=args.chunk,
        run_start_epoch_s=start_epoch, qp_solver='OSQP', qp_solver_version=__import__('osqp').__version__)
    write_json(args.out/'config.json', meta)
    writer = SummaryWriter(str(args.out/'tb'))
    writer.add_scalar('qp/processed', 0, 0); writer.flush()
    registration = args.out.parent.name+'__'+args.out.name if args.out.name == 'qp' else args.out.name
    registry = ROOT/'runs/tensorboard_active'/registration
    if not registry.exists(): registry.symlink_to((args.out/'tb').resolve())
    workers = []
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('a')
            command = [sys.executable, str(Path(__file__).resolve()), '--audit', str(args.audit),
                '--out', str(args.out), '--device', device, '--rank', str(rank), '--workers', str(len(args.devices)),
                '--chunk', str(args.chunk), '--iterations', str(args.iterations),
                '--jacobian-chunk', str(args.jacobian_chunk)]
            if args.full_range: command.append('--full-range')
            if args.smoke: command.append('--smoke')
            if args.limit: command += ['--limit', str(args.limit)]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
            workers.append((subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        written = set()
        last_progress = 0
        while True:
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('QP worker failed; inspect worker logs')
            for path in (args.out/'w0').glob('deadline_*.npz'):
                if path.name in written: continue
                paths = [args.out/f'w{rank}'/path.name for rank in range(len(workers))]
                if not all(p.exists() and p.with_suffix('.json').exists() for p in paths): continue
                parts = []
                for p in paths:
                    with np.load(p) as d: parts.append({k: d[k].copy() for k in d.files})
                summary = report(parts); N = int(path.stem.rsplit('_', 1)[1])
                summary.update(deadline_s=N*.02, elapsed_s=time.monotonic()-begin)
                with np.load(args.audit/'deadline_audit.npz') as baseline:
                    lookup = {tuple(k): i for i, k in enumerate(baseline['keys'])}
                    column = int(np.flatnonzero(np.isclose(baseline['deadlines_s'], N*.02))[0])
                    merged_keys = np.concatenate([p['keys'] for p in parts])
                    if len(set(map(tuple, merged_keys))) != len(merged_keys):
                        raise ValueError('Overlapping QP worker shards')
                    ids = np.array([lookup[tuple(k)] for k in merged_keys])
                    reference = baseline['success'][ids, column]
                    actual = np.concatenate([p['before_success'] for p in parts])
                    summary['baseline_acceptance_mismatches'] = int((reference != actual).sum())
                write_json(args.out/(path.stem+'.json'), summary)
                save_npz(args.out/path.name,
                    **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
                for name in ('before_feasible', 'after_feasible', 'recovered', 'lost'):
                    writer.add_scalar('qp/'+name, summary[name], N)
                writer.flush(); written.add(path.name)
                print(json.dumps(summary), flush=True)
            progress = []
            for rank in range(len(workers)):
                p = args.out/f'w{rank}'/'progress.json'
                if p.exists(): progress.append(json.loads(p.read_text()))
            processed = sum(p['processed'] for p in progress)
            if processed > last_progress:
                elapsed = time.monotonic()-begin
                write_json(args.out/'progress.json', dict(status='running', processed=processed,
                    expected=sum(p['expected'] for p in progress), elapsed_s=elapsed, workers=progress))
                writer.add_scalar('qp/processed', processed, int(elapsed))
                writer.add_scalar('qp/processing_rate_plans_s', processed/max(elapsed, 1), int(elapsed))
                writer.flush(); last_progress = processed
            if all(p.poll() is not None for p, _ in workers): break
            time.sleep(1)
        # Join the deadline sweep into a reusable table, counting an
        # out-of-posture starting pose as unsupported rather than a witness.
        source_table = TransitionTable.load(Path(meta['source_calibration'])/'transition_table.npz')
        shape = (*source_table.minimum_s.shape, int(meta['samples']))
        minima = np.full(shape, np.inf); old_minima = minima.copy()
        admissible = np.zeros(shape, bool)
        trajectory_count = 0
        for filename in sorted(written):
            N = int(Path(filename).stem.rsplit('_', 1)[1])
            with np.load(args.out/filename) as d:
                trajectory_count += len(d['keys'])
                for i, key in enumerate(d['keys']):
                    key = tuple(key); admissible[key] = d['departure_posture_valid'][i]
                    if admissible[key] and d['before_success'][i]: old_minima[key] = min(old_minima[key], N*.02)
                    if admissible[key] and d['after_success'][i]: minima[key] = min(minima[key], N*.02)
        stats = minimum_statistics(minima[..., None], np.isfinite(minima[..., None]))
        source_table.minimum_s = stats['mean']; source_table.support = stats['count']
        source_table.trials = admissible.sum(-1)
        source_table.metadata = dict(meta, statistic='Mean of minimum accepted fixed deadlines after local QP repair',
            witness_filter='Geometry-valid and within checkpoint posture bounds at departure',
            complete_deadline_range=args.full_range and not args.limit and not args.smoke)
        source_table.save(args.out/'transition_table_qp.npz')
        np.savez_compressed(args.out/'transition_statistics.npz', per_start_minimum_s=minima,
            before_minimum_s=old_minima, departure_posture_valid=admissible, std_s=stats['std'])
        with (args.out/'per_start_minima.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['hand', 'source', 'destination', 'yaw_rad', 'start_index',
                'departure_posture_valid', 'before_min_s', 'after_min_s'])
            for h, src, dst, y, s in np.ndindex(shape):
                key = h, src, dst, y, s
                w.writerow([('left', 'right')[h], source_table.drum_names[src], source_table.drum_names[dst],
                    source_table.yaw_centers[y], s, admissible[key],
                    old_minima[key] if np.isfinite(old_minima[key]) else '',
                    minima[key] if np.isfinite(minima[key]) else ''])
        final = dict(status='complete', completed_deadlines=len(written), trajectories=trajectory_count,
            elapsed_s=time.monotonic()-begin, within_bounds=dict(starts=int(admissible.sum()),
                before_supported_starts=int(np.isfinite(old_minima).sum()),
                after_supported_starts=int(np.isfinite(minima).sum()),
                recovered_starts=int((np.isfinite(minima) & ~np.isfinite(old_minima)).sum()),
                before_supported_groups=int(np.isfinite(old_minima).any(-1).sum()),
                after_supported_groups=int(np.isfinite(minima).any(-1).sum())))
        write_json(args.out/'summary.json', final)
        write_json(args.out/'progress.json', final)
    finally:
        writer.close()
        for p, log in workers:
            if p.poll() is None: p.terminate()
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT/'runs/transition_table_full_range_top_entry_v6_full_idle_20261004/audit')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--jacobian-chunk', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--full-range', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.chunk < 1 or args.jacobian_chunk < 1 or args.iterations < 1:
        parser.error('Chunk sizes and iteration budget must be positive')
    if args.devices and len(set(args.devices)) != len(args.devices):
        parser.error('Worker devices must be distinct')
    if not 0 <= args.rank < args.workers:
        parser.error('Invalid worker rank')
    args.audit = args.audit.resolve(); args.out = args.out.resolve()
    begin = time.monotonic()
    parent(args) if args.devices else worker(args)
