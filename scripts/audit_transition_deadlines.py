#!/usr/bin/env python3
"""Reuse saved transition starts: exactly 1 second first, then descend by 20 ms."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import amortized_opt as A
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.transition_assignment import TransitionTable, minimum_statistics
from isaac_rl.transition_calibration import descending_deadlines, full_range_deadlines, evaluate_trials, static_clearance


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def register_tensorboard(out):
    registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
    name = out.parent.name+'__'+out.name if out.name == 'audit' else out.name
    alias = registry/name
    if not alias.is_symlink(): alias.symlink_to((out/'tb').resolve())
    elif alias.resolve() != (out/'tb').resolve():
        raise ValueError(f'TensorBoard registration already refers to a different run: {alias}')


def records(source, table, device, *, cost=None, kit=None, ignore_idle_arm=False):
    keys, qs, vs, valid = [], [], [], []
    shape = (*table.minimum_s.shape, int(table.metadata['samples']))
    old_minimum = np.full(shape, np.inf)
    old_geometric = old_minimum.copy()
    for h in range(2):
        for src in range(len(table.drum_names)):
            for y in range(len(table.yaw_centers)):
                with np.load(source/'samples'/f'h{h}_d{src}_y{y}.npz') as d:
                    start_valid = d['start_valid']
                    if ignore_idle_arm:
                        if cost is None or kit is None:
                            raise ValueError('Rechecking starts requires the cost and kit')
                        q = torch.as_tensor(d['q'], device=device)
                        sources = np.full((len(q), 2), -1); sources[:, h] = src
                        clearance = static_clearance(cost, kit, q, sources, ignore_idle_arm=True).cpu().numpy()
                        start_valid = ((d['ik_error_m'] <= .003) & (clearance >= -1e-6)
                            & (d['outward_speed_m_s'] > .05) & np.isfinite(d['q']).all(-1)
                            & np.isfinite(d['v']).all(-1))
                    for target in range(len(table.drum_names)):
                        key = h, src, target, y
                        for field, output in [('success', old_minimum), ('geometric', old_geometric)]:
                            output[key] = np.where(d[f'{target}_{field}'], d['durations'], np.inf).min(-1)
                        for s in range(len(d['q'])):
                            keys.append((*key, s)); qs.append(d['q'][s]); vs.append(d['v'][s])
                            valid.append(start_valid[s] and table.allowed_groups()[key])
    return dict(keys=np.asarray(keys), q=torch.as_tensor(np.asarray(qs), device=device),
                v=torch.as_tensor(np.asarray(vs), device=device), valid=np.asarray(valid, bool),
                old_minimum=old_minimum, old_geometric=old_geometric)


def describe(results, valid, keys, *, require_hit_speed=True):
    take = np.asarray(valid, bool)
    head = take & (results['terminal_region_dist'] <= 1e-6)
    hit = take & results['geometric']
    hard = take & results['success']
    def groups(mask):
        return len(set(map(tuple, keys[mask, :4])))
    return dict(transition_start_states=int(len(take)), valid_transition_start_states=int(take.sum()),
        on_head=int(head.sum()), complete_hit_conditions=int(hit.sum()), fully_feasible=int(hard.sum()),
        on_head_groups=groups(head), complete_hit_groups=groups(hit), fully_feasible_groups=groups(hard),
        failures_on_valid_starts=dict(region=int((take & ~head).sum()),
            speed_error=int((take & (results['contact_speed_err'] > .75)).sum()) if require_hit_speed else 0,
            direction=int((take & (results['contact_dir_cos'] < .5)).sum()),
            elevation=int((take & ((results['stick_elev_deg'] < 5.) | (results['stick_elev_deg'] > 45.))).sum()),
            collision=int((take & (results['collision_min_sdf'] < -1e-6)).sum()),
            torque=int((take & (results['torque_excess_max'] > 1e-6)).sum()),
            velocity=int((take & (results['velocity_excess_max'] > 1e-6)).sum()),
            joint=int((take & (results['joint_box_excess_max'] > 1e-6)).sum())),
        failure_categories_overlap=True, impact_speed_required=require_hit_speed,
        speed_error_diagnostic_count=int((take & (results['contact_speed_err'] > .75)).sum()))


def combine_worker_audits(paths, *, baseline=False):
    """Merge disjoint starting-state shards; verify identity before combining."""
    merged, keys, valid, deadlines = {}, None, None, None
    for path in paths:
        with np.load(path) as data:
            if keys is None:
                keys = data['keys'].copy()
                valid = np.zeros(len(keys), bool)
                if not baseline: deadlines = data['deadlines_s'].copy()
            elif not np.array_equal(keys, data['keys']):
                raise ValueError('parallel workers used different starting-state records')
            mask = data['valid_start']
            if (valid & mask).any(): raise ValueError('parallel starting-state shards overlap')
            valid |= mask
            for name in data.files:
                if name in ('keys', 'valid_start', 'deadlines_s'): continue
                value = data[name]
                if name not in merged:
                    merged[name] = np.full(value.shape, False if value.dtype == bool else np.nan,
                                           dtype=value.dtype)
                merged[name][mask] = value[mask]
            if not baseline and not np.array_equal(deadlines, data['deadlines_s']):
                raise ValueError('parallel workers used different deadlines')
    return keys, valid, deadlines, merged


def parallel_main(args):
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    devices = args.devices
    if len(set(devices)) != len(devices): raise ValueError('parallel devices must be distinct')
    workers = []
    baseline_written = False
    try:
        for rank, device in enumerate(devices):
            folder = out/'workers'/f'w{rank}'; folder.parent.mkdir(exist_ok=True)
            log = (folder.parent/f'w{rank}.log').open('w')
            command = [sys.executable, str(Path(__file__).resolve()), '--source', str(source),
                '--out', str(folder), '--device', device, '--chunk', str(args.chunk),
                '--deadline-search', args.deadline_search,
                '--worker-index', str(rank), '--worker-count', str(len(devices))]
            if args.hit_depth_mm is not None:
                command += ['--hit-depth-mm', str(args.hit_depth_mm)]
            if args.ignore_hit_speed:
                command += ['--ignore-hit-speed']
            if args.allow_early_target_top_entry:
                command += ['--allow-early-target-top-entry']
            if args.ignore_idle_arm:
                command += ['--ignore-idle-arm']
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            workers.append((process, folder, log))
        while any(p.poll() is None for p, _, _ in workers):
            for process, folder, _ in workers:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'Worker failed; see {folder.parent/(folder.name+".log")}')
            baselines = [folder/'one_second.npz' for _, folder, _ in workers]
            if not baseline_written and all(p.exists() for p in baselines):
                keys, valid, _, values = combine_worker_audits(baselines, baseline=True)
                report = describe(values, valid, keys, require_hit_speed=not args.ignore_hit_speed)
                report.update(deadline_s=1., devices=devices, elapsed_s=time.monotonic()-begin)
                write_json(out/'one_second.json', report)
                print(json.dumps(dict(stage='parallel_one_second', **report)), flush=True)
                baseline_written = True
            time.sleep(.5)
        if any(p.returncode != 0 for p, _, _ in workers): raise RuntimeError('Parallel worker failed')
        keys, valid, deadlines, values = combine_worker_audits(
            [folder/'deadline_audit.npz' for _, folder, _ in workers])
        table = TransitionTable.load(source/'transition_table.npz')
        metadata = json.loads((workers[0][1]/'config.json').read_text())
        metadata.update(devices=devices, worker_count=len(devices), parallel_partition='Disjoint saved starting-state records')
        metadata.pop('worker_index', None)
        write_json(out/'config.json', metadata)
        table.metadata = metadata
        baseline = describe({k: v[:, 0] for k, v in values.items()}, valid, keys,
                            require_hit_speed=not args.ignore_hit_speed)
        baseline.update(deadline_s=1., devices=devices)
        write_json(out/'one_second.json', baseline)
        # Each worker retained prior random-search witnesses. Taking the minimum
        # across their per-state arrays exactly joins the independent new searches.
        stats_files = [np.load(folder/'transition_statistics.npz') for _, folder, _ in workers]
        try:
            minima = np.minimum.reduce([d['per_start_minimum_s'] for d in stats_files])
            descending = np.minimum.reduce([d['descending_only_minimum_s'] for d in stats_files])
        finally:
            for d in stats_files: d.close()
        stats = minimum_statistics(minima[..., None], np.isfinite(minima[..., None]))
        table.minimum_s = stats['mean']; table.support = stats['count']
        table.trials = np.zeros_like(table.trials)
        np.add.at(table.trials, tuple(keys[valid, :4].T), 1)
        table.save(out/'transition_table.npz')
        np.savez_compressed(out/'transition_statistics.npz', std_s=stats['std'],
                            per_start_minimum_s=minima, descending_only_minimum_s=descending)
        np.savez_compressed(out/'deadline_audit.npz', keys=keys, valid_start=valid,
                            deadlines_s=deadlines, **values)
        with (out/'transition_table.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['hand', 'source', 'destination', 'yaw_rad', 'mean_minimum_s', 'successful_starts', 'sampled_starts'])
            for h, src, dst, y in np.ndindex(table.minimum_s.shape):
                value = float(table.minimum_s[h, src, dst, y])
                w.writerow([('left', 'right')[h], table.drum_names[src], table.drum_names[dst], table.yaw_centers[y],
                    value if np.isfinite(value) else '', table.support[h, src, dst, y], table.trials[h, src, dst, y]])
        final = dict(status='complete', checkpoint=metadata['checkpoint'], checkpoint_step=metadata['checkpoint_step'],
            one_second=baseline, devices=devices, decoder_trials=int(valid.sum()*len(deadlines)),
            previous_supported_groups=int((TransitionTable.load(source/'transition_table.npz').support > 0).sum()),
            updated_supported_groups=int((table.support > 0).sum()),
            fully_supported_groups=int(((table.support == table.trials)&(table.trials > 0)).sum()),
            failed_1s_but_succeeded_shorter=int((~values['success'][:, 0] &
                values['success'][:, deadlines < 1.].any(-1) & valid).sum()), elapsed_s=time.monotonic()-begin)
        write_json(out/'summary.json', final)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out/'tb'))
        register_tensorboard(out)
        for column, deadline in enumerate(deadlines):
            writer.add_scalar('calibration/deadline_s', deadline, column)
            writer.add_scalar('calibration/fully_feasible_at_deadline', int(values['success'][:, column].sum()), column)
        writer.close()
        print(json.dumps(final), flush=True)
    finally:
        for process, _, log in workers:
            if process.poll() is None: process.terminate()
            log.close()


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/transition_table_random_v2_20261004')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--deadline-search', choices=('descending', 'full', 'one-second'), default='descending',
                   help='full: every 20 ms from 20 ms to 2 s, independently for every valid saved start')
    p.add_argument('--hit-depth-mm', type=float, default=None,
                   help='Calibration-only inward head depth; default uses the checkpoint band')
    p.add_argument('--ignore-hit-speed', action='store_true',
                   help='Calibration-only: log impact speed but do not require the requested speed')
    p.add_argument('--allow-early-target-top-entry', action='store_true',
                   help='Allow bounded early head contact after verified top entry; reject side/underside entry')
    p.add_argument('--ignore-idle-arm', action='store_true',
                   help='Recheck saved starts and constrain only the moving arm and shared waist')
    p.add_argument('--devices', nargs='+', help='Independent GPU workers, e.g. cuda:0 cuda:1 cuda:2 cuda:3')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--worker-count', type=int, default=1)
    args = p.parse_args(argv)
    if args.chunk < 1: p.error('chunk must be positive')
    if args.hit_depth_mm is not None and (not np.isfinite(args.hit_depth_mm) or args.hit_depth_mm <= 0):
        p.error('hit depth must be finite and positive')
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        p.error('invalid worker partition')
    if args.devices:
        parallel_main(args)
        return
    torch.set_num_threads(4)
    start = time.monotonic()
    source = args.source.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    table = TransitionTable.load(source/'transition_table.npz')
    if args.ignore_idle_arm:
        table.metadata['forbidden_drums_by_hand'] = {'left': ['floor_tom'], 'right': []}
    metadata = dict(table.metadata)
    ck = torch.load(metadata['checkpoint'], map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(ck), device)
        cfg = A.CostConfig(**ck['train_cfg']['cost'])
        cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cfg.n_sub = metadata['collocation_substeps']
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(ck['model'])
    kit = A.DrumKit.from_layout()
    if tuple(kit.names) != table.drum_names: raise ValueError('kit differs from calibrated table')
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    data = records(source, table, device, cost=cost, kit=kit, ignore_idle_arm=args.ignore_idle_arm)
    keys, valid = data['keys'], data['valid']
    if args.allow_early_target_top_entry or args.ignore_idle_arm or args.deadline_search == 'one-second':
        # The old timed exemption also admitted late side entries. Its witnesses
        # are not certified under the new rule; search all durations afresh.
        data['old_minimum'].fill(np.inf)
        data['old_geometric'].fill(np.inf)
    valid = valid & (np.arange(len(keys)) % args.worker_count == args.worker_index)
    deadlines = (np.array([1.]) if args.deadline_search == 'one-second' else
        full_range_deadlines() if args.deadline_search == 'full' else descending_deadlines())
    description = ('Every supported 20 ms deadline from 20 ms to 2 seconds; 1 second reference first'
                   if args.deadline_search == 'full' else
                   'Exactly 1 second, then descend every 20 ms to 80 ms; longer fallback anchors afterward')
    if args.deadline_search == 'one-second': description = 'Exactly one second only'
    metadata.update(source_calibration=str(source), duration_search=description,
        ignore_idle_arm=args.ignore_idle_arm,
        constraint_scope='Moving arm and shared waist only' if args.ignore_idle_arm else 'Both physical arms and waist',
        starting_eligibility_recomputed=args.ignore_idle_arm,
        duration_trials=int(len(deadlines)), deadline_search=args.deadline_search,
        deadlines_s=deadlines.tolist(), identical_saved_start_states=True,
        reinitialize_each_trial=True, early_stop_on_failure=False,
        filter_by_one_second_success=False,
        merged_with_previous_random_witnesses=not (args.allow_early_target_top_entry or args.ignore_idle_arm or args.deadline_search == 'one-second'), training=False,
        worker_index=args.worker_index, worker_count=args.worker_count,
        head_membership='Exact closest tip/allowed shaft contact in legal region; 1e-6 m numerical excess tolerance',
        hit_depth_mm=(args.hit_depth_mm if args.hit_depth_mm is not None else
                      1000*(cfg.terminal_axial_hi-cfg.terminal_axial_lo)),
        hit_depth_override_scope='Head membership only; original checkpoint speed/direction diagnostics and all path constraints retained',
        impact_speed_required=not args.ignore_hit_speed,
        speed_error_tolerance_m_s=None if args.ignore_hit_speed else .75,
        allow_early_target_top_entry=args.allow_early_target_top_entry,
        target_collision_policy=('Verified physical-head top entry with bounded post-contact continuation; side/underside entry and wrists checked; terminal hit region retained'
                                if args.allow_early_target_top_entry else 'Checkpoint timed target exemption'),
        training_softmin_residual_separately_logged=True)
    write_json(out/'config.json', metadata)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out/'tb'), flush_secs=5)
    registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
    if args.worker_count == 1:
        register_tensorboard(out)
    results = {}
    total = 0
    for column, deadline in enumerate(deadlines):
        for h in range(2):
            ids = np.flatnonzero(valid & (keys[:, 0] == h))
            values = evaluate_trials(generator, data['q'][ids], data['v'][ids], h,
                keys[ids, 1], keys[ids, 2], table.yaw_centers[keys[ids, 3]],
                np.full((len(ids), 1), deadline), chunk=args.chunk,
                hit_depth_m=None if args.hit_depth_mm is None else args.hit_depth_mm/1000,
                require_hit_speed=not args.ignore_hit_speed,
                allow_early_target_top_entry=args.allow_early_target_top_entry,
                ignore_idle_arm=args.ignore_idle_arm)
            total += len(ids)
            for name, value in values.items():
                if name not in results:
                    is_bool = value.dtype == bool
                    results[name] = np.full((len(keys), len(deadlines)), False if is_bool else np.nan,
                                            dtype=bool if is_bool else np.float32)
                results[name][ids, column] = value[:, 0]
        report = describe({k: v[:, column] for k, v in results.items()}, valid, keys,
                          require_hit_speed=not args.ignore_hit_speed)
        report.update(deadline_s=float(deadline), tested_deadlines=column+1,
                      elapsed_s=time.monotonic()-start)
        if column == 0:
            write_json(out/'one_second.json', report)
            np.savez_compressed(out/'.one_second.tmp.npz', keys=keys, valid_start=valid,
                                **{k: v[:, 0] for k, v in results.items()})
            (out/'.one_second.tmp.npz').replace(out/'one_second.npz')
        write_json(out/'progress.json', report)
        writer.add_scalar('calibration/deadline_s', deadline, column)
        writer.add_scalar('calibration/fully_feasible_at_deadline', report['fully_feasible'], column)
        print(json.dumps(report), flush=True)
    np.savez_compressed(out/'deadline_audit.npz', keys=keys, valid_start=valid, deadlines_s=deadlines, **results)
    shape = data['old_minimum'].shape
    for name, previous in [('success', data['old_minimum']), ('geometric', data['old_geometric'])]:
        minima = np.full(shape, np.inf)
        minima[tuple(keys.T)] = np.where(results[name], deadlines[None], np.inf).min(-1)
        merged = np.minimum(minima, previous)
        stats = minimum_statistics(merged[..., None], np.isfinite(merged[..., None]))
        if name == 'success':
            old_support = int((table.support > 0).sum())
            table.minimum_s = stats['mean']; table.support = stats['count']
            table.trials = np.zeros_like(table.trials)
            np.add.at(table.trials, tuple(keys[valid, :4].T), 1)
            table.metadata = metadata; table.save(out/'transition_table.npz')
            np.savez_compressed(out/'transition_statistics.npz', std_s=stats['std'],
                per_start_minimum_s=merged, descending_only_minimum_s=minima)
            short_after_long_failure = int((~results['success'][:, 0] &
                results['success'][:, (deadlines < 1.)].any(-1) & valid).sum())
        else:
            np.savez_compressed(out/'geometric_statistics.npz', minimum_s=stats['mean'], support=stats['count'])
    with (out/'transition_table.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['hand', 'source', 'destination', 'yaw_rad', 'mean_minimum_s', 'successful_starts', 'sampled_starts'])
        for h, src, dst, y in np.ndindex(table.minimum_s.shape):
            value = float(table.minimum_s[h, src, dst, y])
            w.writerow([('left', 'right')[h], table.drum_names[src], table.drum_names[dst], table.yaw_centers[y],
                        value if np.isfinite(value) else '', table.support[h, src, dst, y], table.trials[h, src, dst, y]])
    final = dict(status='complete', checkpoint=metadata['checkpoint'], checkpoint_step=metadata['checkpoint_step'],
        one_second=json.loads((out/'one_second.json').read_text()), decoder_trials=total,
        previous_supported_groups=old_support, updated_supported_groups=int((table.support > 0).sum()),
        fully_supported_groups=int(((table.support == table.trials)&(table.trials > 0)).sum()),
        failed_1s_but_succeeded_shorter=short_after_long_failure,
        elapsed_s=time.monotonic()-start)
    write_json(out/'summary.json', final); writer.close()
    print(json.dumps(final), flush=True)


if __name__ == '__main__': main()
