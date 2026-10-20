#!/usr/bin/env python3
"""Search every allowed hand/drum pair using unrestricted-waist SQP.

Each source supplies random legal near-head departures. Every 20 ms deadline
is checked with the decoder. SQP concentrates on promising departures and
searches every shorter grid duration once a feasible witness exists; failed
references receive a coarse recovery sweep. Accepted paths are checked at 32
collocations per segment. There is no yaw-indexed table.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from dataclasses import asdict, replace
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
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import near_head_starts, strike_context
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, metrics, refine
from scripts.probe_transition_qp import save_npz, write_json


def allowed_pairs(names):
    allowed = np.ones((2, len(names), len(names)), bool)
    floor = names.index('floor_tom')
    allowed[0, floor, :] = False
    allowed[0, :, floor] = False
    return allowed


def retime(q, trajectory, segments):
    """Resample a seed; the rebuilt spline retains measured q0/v0 exactly."""
    old = torch.cat((q[:, None], trajectory), 1)
    positions = torch.linspace(0, trajectory.shape[1], segments+1,
                               device=q.device)[1:]
    low = positions.floor().long()
    high = positions.ceil().long().clamp_max(trajectory.shape[1])
    blend = (positions-low).reshape(1, segments, 1)
    return old[:, low]*(1-blend)+old[:, high]*blend


def aggregate(names, pieces):
    shape = (2, len(names), len(names))
    allowed = allowed_pairs(names)
    result = {name: np.full(shape, np.inf) for name in
              ('minimum_s', 'mean_minimum_s', 'median_minimum_s', 'p90_minimum_s',
               'decoder_minimum_s', 'all_start_median_s', 'all_start_p90_s')}
    result.update(allowed=allowed, completed=np.zeros(shape, bool),
                  support=np.zeros(shape, int), trials=np.zeros(shape, int))
    result['std_minimum_s'] = np.full(shape, np.nan)
    result['support_fraction'] = np.zeros(shape)
    for piece in pieces:
        h, src = int(piece['hand']), int(piece['source'])
        for dst in np.flatnonzero(allowed[h, src]):
            take = piece['destination'] == dst
            values = piece['minimum_s'][take]
            live = values[np.isfinite(values)]
            raw = piece['decoder_minimum_s'][take]
            result['completed'][h, src, dst] = bool(piece.get('completed', True))
            result['trials'][h, src, dst] = len(values)
            result['support'][h, src, dst] = len(live)
            if len(values):
                result['support_fraction'][h, src, dst] = len(live)/len(values)
                ordered = np.sort(values)
                # Empirical inverse CDF keeps failed searches as infinity.
                result['all_start_median_s'][h, src, dst] = ordered[int(np.ceil(.5*len(values)))-1]
                result['all_start_p90_s'][h, src, dst] = ordered[int(np.ceil(.9*len(values)))-1]
            if len(live):
                result['minimum_s'][h, src, dst] = live.min()
                result['mean_minimum_s'][h, src, dst] = live.mean()
                result['median_minimum_s'][h, src, dst] = np.median(live)
                result['p90_minimum_s'][h, src, dst] = np.quantile(live, .9)
                if len(live) >= 2:
                    result['std_minimum_s'][h, src, dst] = live.std(ddof=1)
            if np.isfinite(raw).any():
                result['decoder_minimum_s'][h, src, dst] = raw.min()
    return result


def worker(args):
    torch.set_num_threads(2)
    meta = json.loads((args.out/'config.json').read_text())
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = A.build_model(A.model_config_from_checkpoint(ck), device)
    model.load_state_dict(ck['model']); model.eval()
    cfg = A.CostConfig(**ck['train_cfg']['cost'])
    cfg.n_sub = 4; cfg.physics_mode = 'spline'
    cfg.w_velocity = max(cfg.w_velocity, 1.)
    cfg.w_joint_box = max(cfg.w_joint_box, 1.)
    cost = A.FusedCost(cfg, max_horizon=args.max_segments, device=device,
                       rnea=load_cached(device=device))
    cost.calibration_active_hand_only = True
    kit = A.DrumKit.from_layout()
    names = list(kit.names)
    allowed = allowed_pairs(names)
    generator = DecoderHitPoseGenerator(model, cost, kit, horizon=args.max_segments,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    qp_cfg = QPConfig(iterations=args.qp_steps, trust_rad=args.qp_trust_rad,
        jacobian_chunk=32, signed_contact=True,
        torque_time_bins=args.qp_torque_time_bins,
        line_search=tuple(2.**(-k) for k in range(12)))
    groups = [(h, src) for h in range(2) for src in range(len(names))
              if allowed[h, src].any()]
    if args.limit_sources:
        groups = groups[:args.limit_sources]
    folder = args.out/f'w{args.rank}'
    folder.mkdir(exist_ok=True)
    _, lo, hi = cost._joint_box_hard_limits()
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    costs = {}
    def get_cost(N, dense=False):
        key = N, dense
        if key not in costs:
            c = cost_for_horizon(cost, N)
            c.cfg = copy.copy(cost.cfg)
            c.cfg.n_sub = args.verify_substeps if dense else 4
            costs[key] = c
        return costs[key]
    for gi, (hand, src) in enumerate(groups):
        if gi % args.workers != args.rank:
            continue
        output = folder/f'h{hand}_source_{src}.npz'
        if output.exists():
            with np.load(output) as saved:
                if bool(saved.get('completed', True)):
                    continue
        started = time.monotonic()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, hand, src]))
        columns = list(range(3))+list(arm_joint_indices(('left', 'right')[hand]))
        qs, vs, diagnostics = [], [], []
        attempted = 0
        while sum(len(a) for a in qs) < args.samples and attempted < args.max_start_trials:
            count = min(32, args.max_start_trials-attempted)
            yaws = rng.uniform(float(lo[0]), float(hi[0]), count)
            q, v, valid, diag = near_head_starts(cost, kit, ('left', 'right')[hand],
                src, yaws, count, rng, ignore_idle_arm=True)
            bounded = ((q[:, columns] >= lo[columns]-1e-6) &
                       (q[:, columns] <= hi[columns]+1e-6)).all(-1)
            indexes = np.flatnonzero((valid & bounded).cpu().numpy())
            remaining = args.samples-sum(len(a) for a in qs)
            indexes = indexes[:remaining]
            qs.append(q[indexes]); vs.append(v[indexes])
            diagnostics.append({k: value[indexes] for k, value in diag.items()})
            attempted += count
        q0, v0 = torch.cat(qs), torch.cat(vs)
        K = len(q0)
        save_npz(folder/f'starts_h{hand}_d{src}.npz', q=q0.cpu().numpy(), v=v0.cpu().numpy(),
            **{k: np.concatenate([d[k] for d in diagnostics]) for k in diagnostics[0]})
        destinations = np.flatnonzero(allowed[hand, src])
        destination = np.repeat(destinations, K)
        sample = np.tile(np.arange(K), len(destinations))
        q, v = q0[sample], v0[sample]
        B = len(q)
        bestN = np.full(B, args.max_segments+1, int)
        decoderN = bestN.copy()
        best = [None]*B
        best_metrics = [None]*B
        raw = np.zeros((B, args.max_segments+1), bool)
        seed_merit = np.full(B, np.inf)
        attempts = np.zeros((B, args.max_segments+1), np.int16)
        fresh = {}
        qp_evaluations = 0
        def progress(stage, N=0):
            write_json(folder/'progress.json', dict(stage=stage, hand=hand, source=names[src],
                valid_departures=K, attempted_departures=attempted, duration_s=N*.02,
                source_records=B, supported_records=int((bestN <= args.max_segments).sum()),
                qp_evaluations=qp_evaluations, elapsed_s=time.monotonic()-started))
        def context(indexes, N):
            targets = np.full((len(indexes), 2), -1, int)
            targets[:, hand] = destination[indexes]
            sources = np.full_like(targets, -1); sources[:, hand] = src
            ctx = strike_context(kit, targets, sources, np.full(len(indexes), N), device,
                                 speeds=np.full((len(indexes), 2), 2.))
            ctx.target_top_entry_depth_m = .07
            L = torch.full((len(indexes), 3), N, device=device, dtype=torch.long)
            return targets, ctx, L
        def accept(indexes, N, trajectory, *, repaired=False):
            if not len(indexes):
                return
            _, ctx, L = context(indexes, N)
            dense_cost = get_cost(N, True)
            result = metrics(dense_cost, q[indexes], v[indexes], trajectory, ctx, L, .07,
                             signed_contact=True)
            if repaired and (~result['success']).any():
                failed = np.flatnonzero(~result['success'])
                _, fail_ctx, fail_L = context(indexes[failed], N)
                improved, _, after, _ = refine(dense_cost, q[indexes[failed]], v[indexes[failed]],
                    trajectory[failed], fail_ctx, fail_L, hand, depth_m=.07, cfg=replace(qp_cfg, iterations=args.dense_qp_steps, trust_rad=args.dense_qp_trust_rad))
                trajectory = trajectory.clone(); trajectory[failed] = improved
                for key in result:
                    result[key][failed] = after[key]
            for j in np.flatnonzero(result['success']):
                row = indexes[j]
                if not repaired:
                    decoderN[row] = min(decoderN[row], N)
                if N < bestN[row]:
                    bestN[row] = N
                    best[row] = trajectory[j].detach().cpu()
                    best_metrics[row] = {k: value[j] for k, value in result.items()}
        # Sweep all decoder deadlines first, including trials below failed
        # long-duration references. Keep their actual splines for fresh starts.
        for N in range(1, args.max_segments+1):
            if not B:
                break
            indexes = np.arange(B)
            targets, ctx, L = context(indexes, N)
            trajectories, results = [], []
            with torch.no_grad():
                for aim in (q[:, 0], None):
                    plan = generator.predict(q, v, torch.as_tensor(targets[:, None], device=device),
                        q.new_full((B, 1), N*.02), q.new_full((B, 1, 2), 2.),
                        torch.ones(B, 1, device=device, dtype=torch.bool), 0,
                        return_plan=True, waist_yaw=aim)
                    trajectory = plan['traj'][:, :N].clone()
                    idle = arm_joint_indices(('left', 'right')[1-hand])
                    trajectory[:, :, idle] = q[:, None, idle]
                    result = metrics(get_cost(N), q, v, trajectory, ctx, L, .07,
                                     signed_contact=True)
                    trajectories.append(trajectory); results.append(result)
            prefer_other = results[1]['merit'] < results[0]['merit']
            trajectory = trajectories[0].clone()
            trajectory[prefer_other] = trajectories[1][prefer_other]
            fresh[N] = trajectory.cpu()
            raw[:, N] = results[0]['success'] | results[1]['success']
            seed_merit = np.minimum(seed_merit, np.minimum(results[0]['merit'], results[1]['merit']))
            for trajectory, result in zip(trajectories, results):
                pending = np.flatnonzero(result['success'] & (decoderN > args.max_segments))
                for lower in range(0, len(pending), args.chunk):
                    take = pending[lower:lower+args.chunk]
                    accept(take, N, trajectory[take])
            if N % 10 == 0:
                progress('decoder_sweep', N)
        selected = np.zeros(B, bool)
        for dst in destinations:
            candidates = np.flatnonzero(destination == dst)
            order = np.lexsort((seed_merit[candidates], bestN[candidates]))
            selected[candidates[order[:args.qp_starts_per_pair]]] = True
        recovery = set([1, 5, 10, 20, 30, 40, 50, 75, args.max_segments])
        def snapshot(completed=False):
            paths = np.zeros((B, args.max_segments, 17), np.float32)
            for i, path in enumerate(best):
                if path is not None:
                    paths[i, :len(path)] = path.numpy()
            arrays = dict(hand=np.asarray(hand), source=np.asarray(src), destination=destination,
                sample=sample, q=q.cpu().numpy(), v=v.cpu().numpy(), completed=np.asarray(completed),
                minimum_s=np.where(bestN <= args.max_segments, bestN*.02, np.inf),
                selected_for_qp=selected,
                decoder_minimum_s=np.where(decoderN <= args.max_segments, decoderN*.02, np.inf),
                paths=paths, raw_outcomes=raw, qp_attempts=attempts)
            for name in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                         'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err'):
                arrays[name] = np.asarray([m[name] if m is not None else np.nan for m in best_metrics])
            save_npz(output, **arrays)
        snapshot()
        # Every shorter grid deadline receives a repair for selected starts
        # with a witness. Starts without one receive recovery anchors; missing
        # evidence remains unknown. Never stop at the first failed duration.
        for N in range(args.max_segments, 0, -1):
            indexes = np.flatnonzero(selected & (N < bestN) &
                ((bestN <= args.max_segments) | (N in recovery)))
            for lower in range(0, len(indexes), args.chunk):
                take = indexes[lower:lower+args.chunk]
                seed_kinds = ('continuation', 'fresh')
                if args.hihat_terminal_repair or args.terminal_repair:
                    seed_kinds += ('terminal_repair',)
                if args.terminal_repair:
                    seed_kinds += ('region_repair',)
                for seed_kind in seed_kinds:
                    active = take[N < bestN[take]]
                    if seed_kind == 'terminal_repair' and not args.terminal_repair:
                        active = active[destination[active] == names.index('hihat')]
                    if seed_kind == 'continuation':
                        active = np.asarray([i for i in active if best[i] is not None], int)
                    if not len(active):
                        continue
                    if seed_kind == 'continuation':
                        seeds = torch.cat([retime(q[i:i+1], best[i][None].to(device), N)
                                           for i in active])
                    else:
                        seeds = fresh[N][active].to(device)
                    _, ctx, L = context(active, N)
                    if seed_kind in ('terminal_repair', 'region_repair'):
                        from amortized_opt.train_cost import dls_terminal_teacher
                        seeds, _, _ = dls_terminal_teacher(seeds, ctx, get_cost(N), steps=4, window=32, region_target=seed_kind == 'region_repair')
                    repaired, _, after, _ = refine(get_cost(N), q[active], v[active],
                        seeds, ctx, L, hand, depth_m=.07, cfg=qp_cfg)
                    attempts[active, N] += 1
                    qp_evaluations += len(active)
                    # Small residuals can require a different collocation
                    # linearization even when the coarse gate has not passed.
                    near = ((after['terminal_region_dist'] <= .005)
                            & (after['collision_min_sdf'] >= -.005)
                            & (after['torque_excess_max'] <= 10.)
                            & (after['joint_box_excess_max'] <= .02)
                            & (after['velocity_excess_max'] <= .02))
                    passed = np.flatnonzero(after['success'] | near)
                    accept(active[passed], N, repaired[passed], repaired=True)
                progress('qp_sweep', N)
            if B and N % 5 == 0:
                snapshot()
                print(json.dumps(dict(hand=hand, source=names[src], N=N,
                    supported=int((bestN <= args.max_segments).sum()), qp_evaluations=qp_evaluations)),
                    flush=True)
        minimum = np.where(bestN <= args.max_segments, bestN*.02, np.inf)
        snapshot(completed=True)
        write_json(folder/f'h{hand}_source_{src}.json', dict(status='complete', hand=hand,
            source=names[src], sampled_departures=K, attempted_departures=attempted,
            source_records=B, supported_records=int(np.isfinite(minimum).sum()),
            qp_evaluations=qp_evaluations, elapsed_s=time.monotonic()-started))
        print(json.dumps(dict(stage='source_complete', hand=hand, source=names[src],
            supported=int(np.isfinite(minimum).sum()), records=B, seconds=time.monotonic()-started)),
            flush=True)
    write_json(folder/'done.json', {'status': 'complete'})


def publish(args, names, complete=False):
    pieces = []
    for path in args.out.glob('w*/h*_source_*.npz'):
        with np.load(path) as data:
            piece = {k: data[k].copy() for k in
                     ('hand', 'source', 'destination', 'minimum_s', 'decoder_minimum_s')}
            piece['completed'] = bool(data.get('completed', True))
            pieces.append(piece)
    table = aggregate(names, pieces)
    save_npz(args.out/'transition_table.npz', drum_names=np.asarray(names), **table)
    csv_path = args.out/'transition_table.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['hand', 'source', 'destination', 'status', 'minimum_s',
                         'mean_minimum_s', 'median_minimum_s', 'p90_minimum_s',
                         'std_minimum_s', 'decoder_minimum_s', 'all_start_median_s', 'all_start_p90_s', 'supported_starts', 'valid_starts', 'support_fraction'])
        for h, src, dst in np.ndindex(table['allowed'].shape):
            if not table['allowed'][h, src, dst]:
                status = 'excluded'
            elif not table['completed'][h, src, dst]:
                status = ('supported_searching' if table['support'][h, src, dst] else
                          'searching' if table['trials'][h, src, dst] else 'pending')
            elif not table['trials'][h, src, dst]:
                status = 'no_valid_departure_found'
            elif not table['support'][h, src, dst]:
                status = 'no_feasible_witness_found'
            else:
                status = 'supported'
            numeric = [table[key][h, src, dst] for key in
                       ('minimum_s', 'mean_minimum_s', 'median_minimum_s',
                        'p90_minimum_s', 'std_minimum_s', 'decoder_minimum_s', 'all_start_median_s', 'all_start_p90_s')]
            writer.writerow([('left', 'right')[h], names[src], names[dst], status,
                *[f'{x:.3f}' if np.isfinite(x) else '' for x in numeric],
                table['support'][h, src, dst], table['trials'][h, src, dst], table['support_fraction'][h, src, dst]])
    allowed = table['allowed']; supported = allowed & np.isfinite(table['minimum_s'])
    summary = dict(status='complete' if complete else 'running', allowed_pairs=int(allowed.sum()),
        completed_pairs=int((allowed & table['completed']).sum()), supported_pairs=int(supported.sum()),
        source_groups_completed=sum(p['completed'] for p in pieces),
        source_groups_started=len(pieces), total_source_groups=13,
        supported_start_records=int(table['support'].sum()), valid_start_records=int(table['trials'].sum()),
        shortest_found_s=float(table['minimum_s'][supported].min()) if supported.any() else None,
        longest_supported_minimum_s=float(table['minimum_s'][supported].max()) if supported.any() else None,
        no_yaw_axis=True, yaw_motion='No added yaw window or target in SQP; existing joint/physics limits retained',
        primary_statistic='Shortest dense-verified witness across sampled legal departures',
        mean_statistic='Mean of per-start minima, conditional on successful searches; support reported separately',
        std_statistic='Sample standard deviation of successful per-start minima (ddof=1); undefined with fewer than two',
        resolution_s=.02, range_s=[.02, args.max_segments*.02],
        constraint_scope='Moving arm and shared waist; idle-arm violations excluded',
        hit_depth_m=.07, impact_speed_required=False, verification_substeps=args.verify_substeps,
        limitation='Finite SQP search gives feasible witnesses, not certified physical minima; unsupported entries remain unknown')
    summary['failure_aware_statistics'] = 'Empirical inverse-CDF median/p90 over all starts; unresolved searches count as infinity (blank CSV cells)'
    summary['qp_starts_per_pair'] = args.qp_starts_per_pair
    summary['search_coverage'] = ('Full decoder grid for all starts; full shorter-duration QP grid for selected starts with witnesses; '
                                'coarse recovery anchors for selected starts without witnesses')
    write_json(args.out/'summary.json', summary)
    if pieces:
        plot(args.out, names, table, acceptance=getattr(args, 'decoder_acceptance', None))
    return summary


def plot(out, names, table, *, acceptance=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for statistic, label, filename in (
            ('minimum_s', 'Shortest feasible time found', 'minimum_times'),
            ('mean_minimum_s', 'Mean of successful per-start minimum times', 'mean_minimum_times')):
        if acceptance is not None:
            label = ('Shortest accepted decoder duration' if statistic == 'minimum_s' else
                     'Mean of accepted per-start decoder durations')
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
        for h, ax in enumerate(axes):
            values = table[statistic][h]*1000
            im = ax.imshow(np.ma.masked_invalid(np.where(np.isfinite(values), values, np.nan)),
                           vmin=0, vmax=2000, cmap='viridis')
            ax.set_xticks(range(len(names)), [n.replace('_', ' ') for n in names], rotation=40, ha='right')
            ax.set_yticks(range(len(names)), [n.replace('_', ' ') for n in names])
            ax.set_xlabel('Destination drum'); ax.set_ylabel('Source drum')
            ax.set_title(('Left hand', 'Right hand')[h])
            for src, dst in np.ndindex(values.shape):
                if not table['allowed'][h, src, dst]:
                    text, color = 'excluded', '#777777'
                elif not np.isfinite(values[src, dst]):
                    text = ('unknown' if table['completed'][h, src, dst] else
                            'searching' if table['trials'][h, src, dst] else 'pending')
                    color = '#777777'
                else:
                    text = f'{values[src, dst]:.0f} ms\n{table["support"][h, src, dst]}/{table["trials"][h, src, dst]} starts'
                    if not table['completed'][h, src, dst]:
                        text += '\nsearching'
                    color = 'white' if values[src, dst] < 1000 else '#111111'
                ax.text(dst, src, text, ha='center', va='center', fontsize=8, color=color)
        fig.colorbar(im, ax=axes, label='Time (ms)', shrink=.8)
        qualifier = (' — decoder only, '+acceptance+' constraints' if acceptance is not None else ' — unrestricted waist')
        fig.suptitle(label+qualifier+'\nUnknown means no accepted witness; times are sampled estimates', fontsize=14)
        fig.savefig(out/(filename+'.png'), dpi=160, bbox_inches='tight')
        fig.savefig(out/(filename+'.svg'), bbox_inches='tight')
        plt.close(fig)
    from scripts.plot_transition_statistics import plot_statistics, plot_standard_deviation
    plot_statistics(out, names, table, decoder_acceptance=acceptance)
    plot_standard_deviation(out, names, table, decoder_acceptance=acceptance)


def parent(args):
    pipeline_files = ['amortized_opt/cost.py', 'amortized_opt/strike.py',
        'amortized_opt/target_contact.py', 'amortized_opt/terminal_initializer.py',
        'amortized_opt/train_cost.py', 'isaac_rl/trajectory_qp.py',
        'assets/drum_goal_poses.json', 'scripts/build_unrestricted_transition_chart.py']
    fingerprint = hashlib.sha256()
    for filename in pipeline_files:
        fingerprint.update(filename.encode()); fingerprint.update((ROOT/filename).read_bytes())
    pipeline_sha256 = fingerprint.hexdigest()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.resume:
        saved = json.loads((args.out/'config.json').read_text())
        if saved.get('pipeline_sha256') != pipeline_sha256 or saved.get('checkpoint_sha256') != checkpoint_sha256:
            raise ValueError('Resume requires the identical checkpoint, geometry, and repair pipeline')
        for key, value in dict(checkpoint=str(args.checkpoint), devices=args.devices,
                samples_per_source=args.samples, max_start_trials=args.max_start_trials,
                seed=args.seed, qp_starts_per_pair=args.qp_starts_per_pair,
                verification_substeps=args.verify_substeps).items():
            if saved[key] != value:
                raise ValueError(f'Resume setting differs: {key}')
        if saved['qp']['trust_rad'] != args.qp_trust_rad:
            raise ValueError('Resume QP trust radius differs')
        if saved.get('terminal_repair_seed', False) != args.terminal_repair or saved.get('dense_qp_steps', args.qp_steps) != args.dense_qp_steps or saved.get('dense_qp_trust_rad', .03) != args.dense_qp_trust_rad:
            raise ValueError('Resume terminal/dense repair settings differ')
        if saved['qp']['iterations'] != args.qp_steps or len(saved['candidate_deadlines_s']) != args.max_segments:
            raise ValueError('Resume QP/deadline budgets differ')
        if (saved['qp'].get('torque_time_bins', 1) != args.qp_torque_time_bins
                or saved.get('hihat_terminal_repair_seed', False) != args.hihat_terminal_repair):
            raise ValueError('Resume torque bounds or terminal repair seeds differ')
        old = json.loads((args.out/'run.json').read_text())
        for pid in [old.get('launcher_pid'), *old.get('worker_pids', [])]:
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f'Old launcher/worker still exists: {pid}')
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    names = list(A.DrumKit.from_layout().names)
    qp = QPConfig(iterations=args.qp_steps, trust_rad=args.qp_trust_rad, jacobian_chunk=32,
                  torque_time_bins=args.qp_torque_time_bins,
                  signed_contact=True, line_search=tuple(2.**(-k) for k in range(12)))
    write_json(args.out/'config.json', dict(format='unrestricted_transition_chart_v2',
        pipeline_sha256=pipeline_sha256, checkpoint_sha256=checkpoint_sha256,
        checkpoint=str(args.checkpoint), devices=args.devices, samples_per_source=args.samples,
        max_start_trials=args.max_start_trials, seed=args.seed, qp=asdict(qp),
        forbidden_drums_by_hand={'left': ['floor_tom'], 'right': []},
        yaw_axis=False, initial_yaw_sampling='Uniform over the existing legal joint box, filtered by source reachability',
        yaw_constraints='No additional window; waist yaw/pitch/roll are optimization variables',
        generator_waist_aim='Both normal goal-derived and current-yaw seed hints; neither is an SQP constraint',
        physics_mode='spline', optimization_substeps=4, verification_substeps=args.verify_substeps,
        hit_depth_m=.07, requested_speed_m_s=2., impact_speed_required=False,
        target_top_entry_allowed=True, idle_arm_constraints=False, training=False,
        hihat_terminal_repair_seed=args.hihat_terminal_repair,
        terminal_repair_seed=args.terminal_repair, dense_qp_steps=args.dense_qp_steps,
        dense_qp_trust_rad=args.dense_qp_trust_rad,
        candidate_deadlines_s=(np.arange(1, args.max_segments+1)*.02).tolist(),
        early_stop_on_failure=False, limit_sources=args.limit_sources,
        qp_starts_per_pair=args.qp_starts_per_pair,
        qp_start_selection='Shortest dense-verified decoder times, then lowest constraint merit',
        failed_reference_recovery_segments=[1, 5, 10, 20, 30, 40, 50, 75, args.max_segments]))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'))
    registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
    alias = registry/args.out.name
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        if alias.exists() or alias.is_symlink():
            if not args.resume or alias.resolve() != (args.out/'tb').resolve():
                raise ValueError(f'TensorBoard registration already exists: {alias}')
        else:
            alias.symlink_to((args.out/'tb').resolve())
    workers = []; started = time.monotonic(); previous = None
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            command = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
                '--checkpoint', str(args.checkpoint), '--device', device, '--rank', str(rank),
                '--workers', str(len(args.devices)), '--samples', str(args.samples),
                '--max-start-trials', str(args.max_start_trials), '--max-segments', str(args.max_segments),
                '--qp-steps', str(args.qp_steps), '--qp-trust-rad', str(args.qp_trust_rad),
                '--dense-qp-steps', str(args.dense_qp_steps),
                '--dense-qp-trust-rad', str(args.dense_qp_trust_rad), '--chunk', str(args.chunk), '--seed', str(args.seed),
                '--qp-starts-per-pair', str(args.qp_starts_per_pair),
                '--qp-torque-time-bins', str(args.qp_torque_time_bins),
                '--verify-substeps', str(args.verify_substeps), '--limit-sources', str(args.limit_sources)]
            if args.terminal_repair:
                command.append('--terminal-repair')
            if args.hihat_terminal_repair:
                command.append('--hihat-terminal-repair')
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
            workers.append((subprocess.Popen(command, cwd=ROOT, stdout=log,
                                            stderr=subprocess.STDOUT, env=env), log))
        write_json(args.out/'run.json', dict(status='running', worker_pids=[p.pid for p, _ in workers],
                                             launcher_pid=os.getpid()))
        while any(p.poll() is None for p, _ in workers):
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('Transition worker failed; inspect worker logs')
            signature = sorted((p.name, p.stat().st_mtime_ns) for p in args.out.glob('w*/h*_source_*.npz'))
            if signature != previous:
                summary = publish(args, names)
                previous = signature
                print(json.dumps(summary), flush=True)
                step = int(time.monotonic()-started)
                writer.add_scalar('calibration/completed_pairs', summary['completed_pairs'], step)
                writer.add_scalar('calibration/supported_pairs', summary['supported_pairs'], step)
            progress = [json.loads(p.read_text()) for p in args.out.glob('w*/progress.json')]
            writer.add_scalar('calibration/current_source_qp_evaluations',
                              sum(p['qp_evaluations'] for p in progress), int(time.monotonic()-started))
            writer.flush()
            time.sleep(2)
        if any(p.returncode for p, _ in workers):
            raise RuntimeError('Transition worker failed')
        summary = publish(args, names, complete=True)
        if not args.limit_sources and summary['completed_pairs'] != 85:
            raise RuntimeError('Incomplete drum-pair coverage')
        summary['elapsed_s'] = time.monotonic()-started
        write_json(args.out/'summary.json', summary); write_json(args.out/'run.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        for process, log in workers:
            if process.poll() is None:
                process.terminate()
            log.close()
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--devices', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--max-start-trials', type=int, default=256)
    p.add_argument('--max-segments', type=int, default=100)
    p.add_argument('--qp-steps', type=int, default=30)
    p.add_argument('--qp-trust-rad', type=float, default=.03)
    p.add_argument('--dense-qp-trust-rad', type=float, default=.03)
    p.add_argument('--dense-qp-steps', type=int, default=30)
    p.add_argument('--terminal-repair', action='store_true', help='try fixed-point and nearest-region teachers for all destinations')
    p.add_argument('--qp-torque-time-bins', type=int, default=1)
    p.add_argument('--hihat-terminal-repair', action='store_true',
                   help='try a smooth terminal DLS seed for hi-hat QP calibration')
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--qp-starts-per-pair', type=int, default=2)
    p.add_argument('--verify-substeps', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026100507)
    p.add_argument('--limit-sources', type=int, default=0)
    p.add_argument('--resume', action='store_true', help='Reuse completed sources and restart partial ones')
    args = p.parse_args(); args.out = args.out.resolve(); args.checkpoint = args.checkpoint.resolve()
    if min(args.samples, args.max_start_trials, args.max_segments, args.qp_steps,
           args.chunk, args.verify_substeps, args.qp_starts_per_pair,
           args.qp_torque_time_bins, args.dense_qp_steps) < 1 or args.max_segments > 100:
        p.error('Positive budgets and at most 100 segments required')
    if min(args.qp_trust_rad, args.dense_qp_trust_rad) <= 0:
        p.error('QP trust radii must be positive')
    parent(args) if args.devices else worker(args)


if __name__ == '__main__':
    main()
