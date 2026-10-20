"""Shorten saved feasible gap witnesses with SQP and update the original table.

Only the frozen checkpoint from the table is used. Every accepted path is
verified on 32 collocations per segment. Per-start records remain the source
of truth for aggregate statistics; failed searches are never called impossible.
"""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
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
from isaac_rl.duration_search import search_duration
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, metrics, refine
from scripts.build_unrestricted_transition_chart import publish, retime
from scripts.probe_transition_qp import save_npz, write_json


def scalar_metrics(values):
    return {k: float(v[0]) for k, v in values.items()}


def near_gate(values):
    return ((values['terminal_region_dist'] <= .005)
            & (values['collision_min_sdf'] >= -.005)
            & (values['torque_excess_max'] <= 10.)
            & (values['joint_box_excess_max'] <= .02)
            & (values['velocity_excess_max'] <= .02))


def context(kit, hand, sources, destinations, N, device):
    targets = np.full((len(sources), 2), -1, int)
    starts = targets.copy()
    targets[:, hand] = destinations; starts[:, hand] = sources
    ctx = strike_context(kit, targets, starts, np.full(len(sources), N), device,
                         speeds=np.full((len(sources), 2), 2.))
    ctx.target_top_entry_depth_m = .07
    lengths = torch.full((len(sources), 3), N, device=device, dtype=torch.long)
    return targets, ctx, lengths


def setup(checkpoint, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = A.CostConfig(**ck['train_cfg']['cost'])
    cfg.physics_mode = 'spline'; cfg.n_sub = 4
    cfg.w_velocity = max(1., cfg.w_velocity); cfg.w_joint_box = max(1., cfg.w_joint_box)
    cost = A.FusedCost(cfg, max_horizon=100, device=device, rnea=load_cached(device=device))
    cost.calibration_active_hand_only = True
    return ck, cost, A.DrumKit.from_layout()


def load_piece(path):
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def witness_row(piece, case, q, v):
    if int(piece['hand']) != case['hand'] or int(piece['source']) != case['source']:
        raise ValueError('Witness hand/source does not match source record')
    row = np.flatnonzero((piece['destination'] == case['destination'])
                        & (piece['sample'] == case['sample']))
    if len(row) != 1:
        raise ValueError('Witness must identify exactly one departure row')
    i = int(row[0])
    np.testing.assert_allclose(piece['q'][i], q[0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(piece['v'][i], v[0], atol=1e-6, rtol=0)
    return i


def apply_witness(piece, case, q, v, path, values, raw_minimum_s=np.inf):
    """Commit one verified witness to its original, unchanged departure row."""
    if not values['success'][0]:
        raise ValueError('Cannot commit an unverified witness')
    i = witness_row(piece, case, q, v)
    if (path.ndim != 3 or path.shape[0] != 1 or not np.isfinite(path).all()
            or not 1 <= path.shape[1] <= piece['paths'].shape[1]
            or path.shape[2] != piece['paths'].shape[2]):
        raise ValueError('Invalid witness trajectory shape or values')
    duration = path.shape[1]*.02
    if duration >= piece['minimum_s'][i]:
        if raw_minimum_s < piece['decoder_minimum_s'][i]:
            piece['decoder_minimum_s'][i] = raw_minimum_s
            return True
        return False
    piece['minimum_s'][i] = duration
    piece['paths'][i] = 0
    piece['paths'][i, :path.shape[1]] = path[0]
    for key in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err'):
        piece[key][i] = values[key][0]
    piece['decoder_minimum_s'][i] = min(piece['decoder_minimum_s'][i], raw_minimum_s)
    piece.setdefault('gap_fill_verified', np.zeros(len(piece['minimum_s']), bool))[i] = True
    return True


def revalidate_baseline(args):
    _, cost, kit = setup(args.checkpoint, args.devices[-1])
    checked = []; invalid = []
    for filename in sorted(args.table_run.glob('w*/h*_source_*.npz')):
        piece = load_piece(filename); h = int(piece['hand']); src = int(piece['source'])
        live = np.flatnonzero(np.isfinite(piece['minimum_s']))
        for N in sorted(set(np.rint(piece['minimum_s'][live]/.02).astype(int))):
            rows = live[np.rint(piece['minimum_s'][live]/.02).astype(int) == N]
            q = torch.tensor(piece['q'][rows], device=cost.device)
            v = torch.tensor(piece['v'][rows], device=cost.device)
            traj = torch.tensor(piece['paths'][rows, :N], device=cost.device)
            _, ctx, lengths = context(kit, h, np.full(len(rows), src), piece['destination'][rows], N, cost.device)
            dense = cost_for_horizon(cost, N); dense.cfg = copy.copy(cost.cfg); dense.cfg.n_sub = 32
            result = metrics(dense, q, v, traj, ctx, lengths, .07, signed_contact=True)
            for j, row in enumerate(rows):
                record = dict(hand=h, source=src, destination=int(piece['destination'][row]),
                    sample=int(piece['sample'][row]), duration_s=N*.02,
                    metrics={k: float(val[j]) for k, val in result.items()})
                checked.append(record)
                if not result['success'][j]:
                    invalid.append(record)
                    piece['minimum_s'][row] = np.inf
                    piece['decoder_minimum_s'][row] = np.inf
                    piece['paths'][row] = 0
            # Keep surviving witnesses' diagnostics aligned with current geometry.
            for key in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                        'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err'):
                piece[key][rows] = result[key]
        if len(live):
            save_npz(filename, **piece)
    write_json(args.out/'baseline_revalidation.json', dict(checked=len(checked),
        passed=len(checked)-len(invalid), failed=len(invalid), invalid=invalid, cases=checked))
    print(json.dumps(dict(stage='baseline_revalidated', checked=len(checked), failed=len(invalid))), flush=True)


def prepare_cases(args):
    with np.load(args.table_run/'transition_table.npz') as d:
        allowed = d['allowed'].copy()
        missing = allowed & ~np.isfinite(d['minimum_s'])
        names = d['drum_names'].tolist()
    if getattr(args, 'per_start', False):
        candidates = []
        for path in sorted(args.table_run.glob('w*/h*_source_*.npz')):
            piece = load_piece(path)
            h, src = int(piece['hand']), int(piece['source'])
            for row in np.flatnonzero(~np.isfinite(piece['minimum_s'])):
                dst, sample = int(piece['destination'][row]), int(piece['sample'][row])
                if not allowed[h, src, dst]: continue
                candidates.append(dict(hand=h, source=src, destination=dst, sample=sample,
                    hand_name=('left', 'right')[h], source_name=names[src], destination_name=names[dst],
                    stride=args.stride, recovery=True, per_start=True))
        # Visit one unresolved departure from each pair before its next sample.
        candidates.sort(key=lambda c: (c['sample'], c['hand'], c['source'], c['destination']))
        for i, case in enumerate(candidates): case['case'] = i
        write_json(args.out/'cases.json', candidates)
        return candidates
    if getattr(args, 'recover', False):
        cases = [dict(case=i, hand=int(h), source=int(src), destination=int(dst),
            hand_name=('left', 'right')[h], source_name=names[src], destination_name=names[dst],
            stride=args.stride, recovery=True)
            for i, (h, src, dst) in enumerate(zip(*np.nonzero(missing)))]
        write_json(args.out/'cases.json', cases)
        return cases
    candidates = []
    for row in csv.DictReader((ROOT/'runs/hihat_fixes_probe_20261005/verified_witnesses.csv').open()):
        candidates.append(dict(hand=('left', 'right').index(row['hand']), source=names.index(row['source']),
            destination=names.index(row['destination']), sample=int(row['sample']),
            initial_s=float(row['feasible_duration_s']), artifact=row['trajectory_artifact'], field='after'))
    saved = ROOT/'runs/transition_coverage_repairs_20261005'
    for row in json.loads((saved/'witnesses.json').read_text()):
        candidates.append(dict(hand=('left', 'right').index(row['hand']), source=names.index(row['source']),
            destination=names.index(row['destination']), sample=int(row['sample']), initial_s=row['duration_s'],
            artifact=str(saved/row['artifact']), field='trajectory'))
    unique = {}
    for case in candidates:
        h, src, dst, sample = (case[k] for k in ('hand', 'source', 'destination', 'sample'))
        if not missing[h, src, dst]: continue
        key = h, src, dst, sample
        if key not in unique or case['initial_s'] < unique[key]['initial_s']: unique[key] = case
    cases = list(unique.values())
    def priority(case):
        h, src, dst = (case[k] for k in ('hand', 'source', 'destination'))
        if h == 0 and names[src] == 'tom1' and names[dst] == 'cymbal_left': return (0, case['sample'])
        if h == 0 and names[src] == 'hihat' and names[dst] == 'tom1': return (1, case['sample'])
        return (2, case['initial_s'], h, src, dst, case['sample'])
    cases.sort(key=priority)
    # Start one sample from each pair before revisiting the same pair.
    first, extra, seen = [], [], set()
    for case in cases:
        key = tuple(case[k] for k in ('hand', 'source', 'destination'))
        (extra if key in seen else first).append(case)
        seen.add(key)
    cases = first + extra
    for i, case in enumerate(cases):
        case.update(case=i, hand_name=('left', 'right')[case['hand']], source_name=names[case['source']],
                    destination_name=names[case['destination']])
        case['stride'] = 1 if case['source_name'] == 'tom1' and case['destination_name'] == 'cymbal_left' else args.stride
        piece = load_piece(next(args.table_run.glob(f'w*/h{case["hand"]}_source_{case["source"]}.npz')))
        artifact = load_piece(case['artifact'])
        witness_row(piece, case, artifact['q'], artifact['v'])
        path = artifact[case['field']]
        if path.shape[1] < round(case['initial_s']/.02) or not np.isfinite(path).all():
            raise ValueError('Invalid saved initial witness')
    write_json(args.out/'cases.json', cases)
    return cases


def import_initial_witnesses(args, cases):
    """Expose all saved witnesses immediately, before lengthy shortening."""
    _, cost, kit = setup(args.checkpoint, args.devices[-1])
    folder = args.out/'wimport'; folder.mkdir(exist_ok=True)
    checked = []
    for case in cases:
        if case.get('recovery'): continue
        data = load_piece(case['artifact'])
        N = round(case['initial_s']/.02)
        q = torch.tensor(data['q'], device=cost.device)
        v = torch.tensor(data['v'], device=cost.device)
        path = torch.tensor(data[case['field']][:, :N], device=cost.device)
        _, ctx, lengths = context(kit, case['hand'], [case['source']], [case['destination']], N, cost.device)
        dense = cost_for_horizon(cost, N); dense.cfg = copy.copy(cost.cfg); dense.cfg.n_sub = 32
        m = metrics(dense, q, v, path, ctx, lengths, .07, signed_contact=True)
        checked.append(dict(case=case, metrics=scalar_metrics(m)))
        if not m['success'][0]: continue
        artifact = folder/f'initial_{case["case"]}.npz'
        save_npz(artifact, q=data['q'], v=data['v'], trajectory=path.cpu().numpy(),
                 duration_s=np.asarray(N*.02))
        write_json(folder/f'witness_{case["case"]}_preflight.json', dict(case=case,
            status='verified_witness', minimum_s=N*.02, raw_minimum_s=None,
            final_metrics=scalar_metrics(m), artifact=str(artifact)))
    write_json(args.out/'initial_revalidation.json', dict(checked=len(checked),
        passed=sum(bool(r['metrics']['success']) for r in checked), records=checked))


def recover_witness(args, case, cost, kit, generator, c, qp, folder):
    """Search only unresolved original pairs, retaining every measured start."""
    from amortized_opt.train_cost import dls_terminal_teacher
    h, src, dst, index = (case[k] for k in ('hand', 'source', 'destination', 'case'))
    piece = load_piece(next(args.table_run.glob(f'w*/h{h}_source_{src}.npz')))
    take = piece['destination'] == dst
    if case.get('per_start'): take &= piece['sample'] == case['sample']
    rows = np.flatnonzero(take)
    if case.get('per_start') and len(rows) != 1:
        raise ValueError('Per-start recovery must identify exactly one measured departure')
    q = torch.tensor(piece['q'][rows], device=args.device)
    v = torch.tensor(piece['v'][rows], device=args.device)
    samples = piece['sample'][rows]
    anchors = (10, 20, 30, 40, 50, 75, 100)
    seeds, attempts, best = {}, [], None
    raw = np.full(len(rows), np.inf)
    accepted_minimum = np.full(len(rows), np.inf)
    departure_seeds = []
    repair_folder = ROOT/'runs/transition_coverage_repairs_20261005'
    for key, name, sample in [((1, 3, 6), 'extra_h1_tom2_to_floor_tom.npz', 1),
                              ((1, 6, 1), 'floor_departure_small_trust.npz', 6)]:
        if (h, src, dst) != key: continue
        if sample not in samples: continue
        data = load_piece(repair_folder/name)
        witness_row(piece, dict(case, sample=sample), data['q'], data['v'])
        i = int(np.flatnonzero(samples == sample)[0])
        departure_seeds.append((i, torch.tensor(data['after'], device=args.device)))
    if case.get('per_start'):
        # A verified sibling path supplies another terminal/waist seed while
        # the spline still starts at this sample's unchanged measured q/v.
        siblings = np.flatnonzero((piece['destination'] == dst) & np.isfinite(piece['minimum_s']))
        if len(siblings):
            columns = [0, 1, 2, *arm_joint_indices(('left', 'right')[h])]
            distance = np.linalg.norm(piece['q'][siblings][:, columns]-piece['q'][rows[0], columns], axis=1)
            row = int(siblings[np.argmin(distance)])
            N = round(piece['minimum_s'][row]/.02)
            path = torch.tensor(piece['paths'][row:row+1, :N], device=args.device)
            old_q = torch.tensor(piece['q'][row:row+1], device=args.device)
            t = torch.linspace(0, 1, N+1, device=args.device)[1:]
            ease = 1-(10*t**3-15*t**4+6*t**5)
            path += ease[None, :, None]*(q[:1]-old_q)[:, None]
            idle = arm_joint_indices(('left', 'right')[1-h]); path[:, :, idle] = q[:, None, idle]
            departure_seeds.append((0, path))
    def record(N, i, path, m, method):
        nonlocal best
        if N*.02 >= accepted_minimum[i]: return
        accepted_minimum[i] = N*.02
        current = dict(case, sample=int(samples[i]))
        artifact = folder/f'case_{index}_recovery_s{samples[i]}_n{N}_{method}.npz'
        save_npz(artifact, q=q[i:i+1].cpu().numpy(), v=v[i:i+1].cpu().numpy(),
                 trajectory=path.detach().cpu().numpy(), duration_s=np.asarray(N*.02))
        write_json(folder/f'witness_{index}_s{samples[i]}_n{N}_{method}.json', dict(case=current,
            status='verified_witness', minimum_s=N*.02,
            raw_minimum_s=N*.02 if method.startswith('decoder') else None,
            final_metrics=scalar_metrics(m), artifact=str(artifact)))
        if best is None or N < best[0]: best = (N, i, path.detach().clone(), current)
    for N in range(1, 101):
        targets, ctx, lengths = context(kit, h, [src]*len(rows), [dst]*len(rows), N, args.device)
        candidates = []
        with torch.no_grad():
            for yi, yaw in enumerate((q[:, 0], None)):
                plan = generator.predict(q, v, torch.tensor(targets[:, None], device=args.device),
                    q.new_full((len(q), 1), N*.02), q.new_full((len(q), 1, 2), 2.),
                    torch.ones(len(q), 1, device=args.device, dtype=torch.bool), 0,
                    return_plan=True, waist_yaw=yaw)
                traj = plan['traj'][:, :N].clone()
                idle = arm_joint_indices(('left', 'right')[1-h]); traj[:, :, idle] = q[:, None, idle]
                m = metrics(c(N), q, v, traj, ctx, lengths, .07, signed_contact=True)
                for i in np.flatnonzero(m['success']):
                    _, one, L = context(kit, h, [src], [dst], N, args.device)
                    dense = metrics(c(N, True), q[i:i+1], v[i:i+1], traj[i:i+1], one, L, .07,
                                    signed_contact=True)
                    if dense['success'][0]:
                        raw[i] = min(raw[i], N*.02)
                        record(N, i, traj[i:i+1], dense, f'decoder{yi}')
                if N in anchors: candidates.append(traj.detach())
        if N in anchors:
            for kind, traj in [('decoder0', candidates[0]), ('decoder1', candidates[1]),
                               ('region', candidates[0]), ('fixed', candidates[0])]:
                if kind in ('region', 'fixed'):
                    traj, _, _ = dls_terminal_teacher(traj, ctx, c(N), steps=4, window=32,
                                                     region_target=kind == 'region')
                m = metrics(c(N), q, v, traj, ctx, lengths, .07, signed_contact=True)
                seeds[N, kind] = (traj.cpu(), m['merit'])
            for seed_id, (i, path) in enumerate(departure_seeds):
                extra = retime(q[i:i+1], path, N)
                _, one, L = context(kit, h, [src], [dst], N, args.device)
                m = metrics(c(N), q[i:i+1], v[i:i+1], extra, one, L, .07, signed_contact=True)
                paths = candidates[0].clone(); paths[i:i+1] = extra
                merit = np.full(len(rows), np.inf); merit[i] = m['merit'][0]
                seeds[N, f'saved_path{seed_id}'] = (paths.cpu(), merit)
        if N % 10 == 0:
            write_json(folder/'progress.json', dict(stage='recovery_decoder', case=index,
                hand=case['hand_name'], source=case['source_name'], destination=case['destination_name'],
                duration_s=N*.02, passing_starts=int(np.isfinite(raw).sum())))
    if best is None:
        # Revisit independently seeded deadlines; a failed long path prunes nothing.
        for N in reversed(anchors):
            candidates = [(float(merit[i]), i, kind, paths[i:i+1].to(args.device))
                for (n, kind), (paths, merit) in seeds.items() if n == N
                for i in range(len(rows))]
            # Cover distinct departures before using their second-best seed.
            selected, seen = [], set()
            for item in sorted(candidates, key=lambda x: x[0]):
                if item[1] in seen: continue
                selected.append(item); seen.add(item[1])
                if len(selected) == args.recovery_starts: break
            for _, i, kind, seed in selected:
                _, ctx, lengths = context(kit, h, [src], [dst], N, args.device)
                path, _, _, _ = refine(c(N), q[i:i+1], v[i:i+1], seed, ctx, lengths, h,
                                      depth_m=.07, cfg=qp)
                m = metrics(c(N, True), q[i:i+1], v[i:i+1], path, ctx, lengths, .07,
                            signed_contact=True)
                if not m['success'][0] and near_gate(m)[0]:
                    path, _, m, _ = refine(c(N, True), q[i:i+1], v[i:i+1], path, ctx, lengths, h,
                        depth_m=.07, cfg=replace(qp, iterations=10, trust_rad=.001))
                attempt = dict(sample=int(samples[i]), duration_s=N*.02, seed=kind,
                               metrics=scalar_metrics(m))
                attempts.append(attempt)
                write_json(folder/'progress.json', dict(stage='recovery_qp', case=index,
                    hand=case['hand_name'], source=case['source_name'], destination=case['destination_name'],
                    **attempt))
                print(json.dumps(dict(case=index, stage='recovery_qp', **attempt)), flush=True)
                if m['success'][0]: record(N, i, path, m, 'qp')
            if best is not None: break
    write_json(folder/f'recovery_{index}.json', dict(case=case, decoder_deadlines=100,
        sample_ids=samples.tolist(), decoder_minimum_s=[float(x) if np.isfinite(x) else None for x in raw],
        qp_attempts=attempts, status='witness_found' if best else 'unresolved_budget_exhausted'))
    if best is None: return None
    N, i, path, current = best
    current['initial_s'] = N*.02
    return current, q[i:i+1], v[i:i+1], path


def worker(args):
    torch.set_num_threads(2)
    ck, cost, kit = setup(args.checkpoint, args.device)
    model = A.build_model(A.model_config_from_checkpoint(ck), args.device)
    model.load_state_dict(ck['model']); model.eval()
    generator = DecoderHitPoseGenerator(model, cost, kit, horizon=100,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    qp = QPConfig(iterations=15, trust_rad=.005, torque_time_bins=16, jacobian_chunk=32,
        signed_contact=True, line_search=tuple(2.**(-k) for k in range(12)))
    costs = {}
    def c(N, dense=False):
        if (N, dense) not in costs:
            obj = cost_for_horizon(cost, N); obj.cfg = copy.copy(cost.cfg); obj.cfg.n_sub = 32 if dense else 4
            costs[N, dense] = obj
        return costs[N, dense]
    folder = args.out/f'w{args.rank}'; folder.mkdir(exist_ok=True)
    cases = json.loads((args.out/'cases.json').read_text())
    # Spread distinct pairs over workers before their additional samples.
    for case in cases:
        if case['case'] % args.workers != args.rank: continue
        index = case['case']; h = case['hand']; src = case['source']; dst = case['destination']
        filename = folder/f'case_{index}.json'
        if filename.exists(): continue
        begin = time.monotonic()
        if case.get('recovery'):
            if case.get('per_start'):
                piece = load_piece(next(args.table_run.glob(f'w*/h{h}_source_{src}.npz')))
                row = np.flatnonzero((piece['destination'] == dst) & (piece['sample'] == case['sample']))
                if len(row) != 1: raise ValueError('Ambiguous per-start recovery row')
                if np.isfinite(piece['minimum_s'][row[0]]):
                    write_json(filename, dict(case=case, status='already_supported'))
                    continue
            recovered = recover_witness(args, case, cost, kit, generator, c, qp, folder)
            if recovered is None:
                write_json(filename, dict(case=case, status='unresolved_budget_exhausted'))
                continue
            case, q, v, initial = recovered
        else:
            with np.load(case['artifact']) as data:
                q = torch.tensor(data['q'], device=args.device); v = torch.tensor(data['v'], device=args.device)
                initial = torch.tensor(data[case['field']], device=args.device)
        N0 = round(case['initial_s']/.02); initial = initial[:, :N0]
        def ctx_for(N): return context(kit, h, [src], [dst], N, args.device)
        _, ctx, lengths = ctx_for(N0)
        initial_metrics = metrics(c(N0, True), q, v, initial, ctx, lengths, .07, signed_contact=True)
        if not initial_metrics['success'][0]:
            write_json(filename, dict(case=case, status='initial_witness_failed_revalidation',
                                     metrics=scalar_metrics(initial_metrics)))
            continue
        initial_artifact = folder/f'case_{index}_initial.npz'
        save_npz(initial_artifact, q=q.cpu().numpy(), v=v.cpu().numpy(),
                 trajectory=initial.cpu().numpy(), duration_s=np.asarray(N0*.02))
        write_json(folder/f'witness_{index}_initial.json', dict(case=case, status='verified_witness',
            minimum_s=N0*.02, raw_minimum_s=None, final_metrics=scalar_metrics(initial_metrics),
            artifact=str(initial_artifact)))
        fresh = {}; raw_minimum = np.inf; repairs = []
        with torch.no_grad():
            for N in range(1, N0):
                targets, ctx, lengths = ctx_for(N); choices = []
                for yaw in (q[:, 0], None):
                    plan = generator.predict(q, v, torch.tensor(targets[:, None], device=args.device),
                        q.new_full((1, 1), N*.02), q.new_full((1, 1, 2), 2.),
                        torch.ones(1, 1, device=args.device, dtype=torch.bool), 0, return_plan=True, waist_yaw=yaw)
                    traj = plan['traj'][:, :N].clone()
                    idle = arm_joint_indices(('left', 'right')[1-h]); traj[:, :, idle] = q[:, None, idle]
                    m = metrics(c(N, True), q, v, traj, ctx, lengths, .07, signed_contact=True)
                    if m['success'][0]: raw_minimum = min(raw_minimum, N*.02)
                    choices.append((float(m['merit'][0]), traj))
                fresh[N] = min(choices, key=lambda item: item[0])[1]
        def attempt(N, best, parent, stage):
            _, ctx, lengths = ctx_for(N)
            for kind, seed in [('continuation', retime(q, best, N)), ('fresh', fresh[N])]:
                m = metrics(c(N, True), q, v, seed, ctx, lengths, .07, signed_contact=True)
                path = seed
                if not m['success'][0]:
                    path, _, _, _ = refine(c(N), q, v, seed, ctx, lengths, h, depth_m=.07, cfg=qp)
                    m = metrics(c(N, True), q, v, path, ctx, lengths, .07, signed_contact=True)
                    if not m['success'][0] and near_gate(m)[0]:
                        path, _, m, _ = refine(c(N, True), q, v, path, ctx, lengths, h,
                            depth_m=.07, cfg=replace(qp, iterations=10, trust_rad=.001))
                row = dict(duration_s=N*.02, parent_s=parent*.02, stage=stage, seed=kind,
                           success=bool(m['success'][0]), metrics=scalar_metrics(m))
                repairs.append(row)
                write_json(folder/'progress.json', dict(case=index, hand=case['hand_name'],
                    source=case['source_name'], destination=case['destination_name'], duration_s=N*.02,
                    success=bool(m['success'][0]), elapsed_s=time.monotonic()-begin))
                # Preserve every improved passing path immediately.
                if m['success'][0]:
                    save_npz(folder/f'case_{index}_latest.npz', q=q.cpu().numpy(), v=v.cpu().numpy(),
                             trajectory=path.cpu().numpy(), duration_s=np.asarray(N*.02))
                    # Immutable artifacts allow the parent to ingest while this worker continues.
                    passing = folder/f'case_{index}_n{N}.npz'
                    save_npz(passing, q=q.cpu().numpy(), v=v.cpu().numpy(),
                             trajectory=path.cpu().numpy(), duration_s=np.asarray(N*.02))
                    write_json(folder/f'witness_{index}_n{N}.json', dict(case=case,
                        status='verified_witness', minimum_s=N*.02, raw_minimum_s=None,
                        final_metrics=scalar_metrics(m), artifact=str(passing)))
                    print(json.dumps(dict(case=index, **row)), flush=True)
                    return path, True
            return best, False
        found = search_duration(N0, initial, attempt, stride=case['stride'], refine_bands=2)
        _, ctx, lengths = ctx_for(found.horizon)
        final = metrics(c(found.horizon, True), q, v, found.path, ctx, lengths, .07, signed_contact=True)
        if not final['success'][0]: raise RuntimeError('Final duration witness failed verification')
        artifact = folder/f'case_{index}.npz'
        save_npz(artifact, q=q.cpu().numpy(), v=v.cpu().numpy(), trajectory=found.path.cpu().numpy(),
                 duration_s=np.asarray(found.horizon*.02))
        result = dict(case=case, status='complete', minimum_s=found.horizon*.02, raw_minimum_s=(
            float(raw_minimum) if np.isfinite(raw_minimum) else None), final_metrics=scalar_metrics(final),
            artifact=str(artifact), deadlines=found.deadlines, repairs=repairs,
            elapsed_s=time.monotonic()-begin, certified_global_minimum=False)
        write_json(filename, result)
        print(json.dumps(dict(case=index, minimum_s=result['minimum_s'], status='complete')), flush=True)
    write_json(folder/'done.json', dict(status='complete'))


def refresh_table(args, names, complete=False):
    table_args = argparse.Namespace(out=args.table_run, max_segments=100, verify_substeps=32, qp_starts_per_pair=2)
    summary = publish(table_args, names, complete=True)
    cases = json.loads((args.out/'cases.json').read_text())
    finished = [json.loads(p.read_text()) for p in args.out.glob('w*/case_*.json')]
    summary.update(status='complete' if complete else 'gap_fill_running', gap_fill_run=str(args.out),
        gap_fill_cases_finished=len(finished), gap_fill_cases_total=len(cases),
        gap_fill_search='60 ms coarse then 20 ms refinement; tom1-to-left-cymbal uses full 20 ms sweep',
        pipeline_checkpoint=str(args.checkpoint), baseline_backup=str(args.out/'baseline_backup'),
        verification='Existing and new witnesses revalidated on current 32-collocation physical contract',
        search_coverage='Original per-start searches plus targeted gap shortening; failed samples remain unresolved')
    summary['gap_fill_recovery'] = getattr(args, 'recover', False)
    summary['gap_fill_per_start'] = getattr(args, 'per_start', False)
    summary['gap_fill_cases_completed'] = sum(r['status'] == 'complete' for r in finished)
    summary['gap_fill_cases_unresolved'] = sum(r['status'] == 'unresolved_budget_exhausted' for r in finished)
    write_json(args.table_run/'summary.json', summary)
    write_json(args.out/'summary.json', summary)
    write_change_report(args)
    if complete:
        from scripts.plot_transition_statistics import plot_statistics, plot_standard_deviation
        table = load_piece(args.table_run/'transition_table.npz')
        plot_statistics(args.table_run, names, table)
        plot_standard_deviation(args.table_run, names, table)
    return summary


def write_change_report(args):
    baseline = load_piece(args.out/'baseline_backup/transition_table.npz')
    current = load_piece(args.table_run/'transition_table.npz')
    names = current['drum_names'].tolist()
    changes = []
    for h, src, dst in zip(*np.nonzero(current['allowed'])):
        before, after = baseline['minimum_s'][h, src, dst], current['minimum_s'][h, src, dst]
        if before == after: continue
        changes.append(dict(hand=('left', 'right')[h], source=names[src], destination=names[dst],
            before_s=float(before) if np.isfinite(before) else None,
            shortest_found_s=float(after) if np.isfinite(after) else None,
            supported_starts=int(current['support'][h, src, dst]),
            valid_starts=int(current['trials'][h, src, dst])))
    preserved = 0
    for filename in args.table_run.glob('w*/h*_source_*.npz'):
        old = load_piece(args.out/'baseline_backup'/filename.relative_to(args.table_run))
        new = load_piece(filename)
        for key in ('q', 'v', 'destination', 'sample'):
            np.testing.assert_array_equal(new[key], old[key])
        preserved += len(new['q'])
    report = dict(changes=changes, preserved_departure_records=preserved,
        new_supported_pairs=sum(r['before_s'] is None and r['shortest_found_s'] is not None for r in changes),
        still_unresolved_pairs=int((current['allowed'] & ~np.isfinite(current['minimum_s'])).sum()),
        limitation='Saved shortest-found durations; active searches can still improve them; no global certificates')
    write_json(args.out/'table_changes.json', report)
    write_json(args.table_run/'gap_fill_changes.json', report)


def ingest_results(args, applied):
    changed = False
    for filename in sorted([*args.out.glob('w*/case_*.json'), *args.out.glob('w*/witness_*.json')]):
        if str(filename) in applied: continue
        result = json.loads(filename.read_text())
        if result['status'] in ('complete', 'verified_witness'):
            case = result['case']; h, src = case['hand'], case['source']
            source_path = next(args.table_run.glob(f'w*/h{h}_source_{src}.npz'))
            piece = load_piece(source_path)
            with np.load(result['artifact']) as data:
                values = {k: np.asarray([v]) for k, v in result['final_metrics'].items()}
                committed = apply_witness(piece, case, data['q'], data['v'], data['trajectory'], values,
                    result['raw_minimum_s'] if result['raw_minimum_s'] is not None else np.inf)
            if committed:
                save_npz(source_path, **piece)
                changed = True
        applied.add(str(filename))
    return changed


def parent(args):
    args.out.mkdir(parents=True, exist_ok=False)
    # Identity checks precede all table mutations and expensive searches.
    prepare_cases(args)
    backup = args.out/'baseline_backup'; backup.mkdir()
    for pattern in ('*.json', '*.csv', '*.npz', '*.png', '*.svg', 'w*/h*_source_*.npz'):
        for filename in args.table_run.glob(pattern):
            target = backup/filename.relative_to(args.table_run); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filename, target)
    write_json(args.out/'config.json', dict(table_run=str(args.table_run), checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), devices=args.devices,
        target_scope=('All unresolved original start records' if args.per_start else 'Only gaps in original table')
            + '; no generator training', depth_m=.07, speed_required=False,
        verification_substeps=32, stride=args.stride, physics_mode='spline', idle_arm_constraints=False,
        limitation='Search budget yields verified shortest-found durations, not global certificates'))
    torch.set_num_threads(2)
    revalidate_baseline(args)
    names = list(A.DrumKit.from_layout().names)
    table_args = argparse.Namespace(out=args.table_run, max_segments=100, verify_substeps=32, qp_starts_per_pair=2)
    publish(table_args, names, complete=True)
    cases = prepare_cases(args)
    if not args.recover:
        import_initial_witnesses(args, cases)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'))
    alias = ROOT/'runs/tensorboard_active'/args.out.name
    alias.symlink_to((args.out/'tb').resolve())
    processes = []; applied = set(); started = time.monotonic()
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--table-run', str(args.table_run),
                '--out', str(args.out), '--checkpoint', str(args.checkpoint), '--device', device,
                '--rank', str(rank), '--workers', str(len(args.devices)), '--stride', str(args.stride),
                '--recovery-starts', str(args.recovery_starts)]
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
            processes.append((proc, log))
        write_json(args.out/'run.json', dict(status='running', launcher_pid=os.getpid(),
                                            worker_pids=[p.pid for p, _ in processes]))
        summary = refresh_table(args, names)
        while True:
            changed = ingest_results(args, applied)
            if changed:
                summary = refresh_table(args, names)
                step = int(time.monotonic()-started)
                writer.add_scalar('table/supported_pairs', summary['supported_pairs'], step)
                writer.add_scalar('table/finished_gap_searches', summary['gap_fill_cases_finished'], step); writer.flush()
                print(json.dumps(summary), flush=True)
            if all(p.poll() is not None for p, _ in processes):
                ingest_results(args, applied)
                break
            if any(p.poll() not in (None, 0) for p, _ in processes):
                raise RuntimeError('Gap worker failed; inspect worker logs')
            time.sleep(2)
        if any(p.returncode for p, _ in processes): raise RuntimeError('Gap worker failed')
        if len(list(args.out.glob('w*/case_*.json'))) != len(cases):
            raise RuntimeError('Missing completed worker results')
        summary = refresh_table(args, names, complete=True)
        write_json(args.out/'run.json', dict(status='complete', elapsed_s=time.monotonic()-started))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc)))
        raise
    finally:
        for proc, log in processes:
            if proc.poll() is None: proc.terminate()
            log.close()
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table-run', type=Path, default=ROOT/'runs/transition_qp_unrestricted_20261005')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--devices', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--recover', action='store_true', help='Recover remaining missing pairs from original starts')
    p.add_argument('--per-start', action='store_true', help='Recover every unresolved sample, including supported pairs')
    p.add_argument('--recovery-starts', type=int, default=3)
    args = p.parse_args()
    if args.per_start: args.recover = True
    args.table_run = args.table_run.resolve(); args.out = args.out.resolve(); args.checkpoint = args.checkpoint.resolve()
    parent(args) if args.devices else worker(args)


if __name__ == '__main__': main()
