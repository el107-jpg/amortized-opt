#!/usr/bin/env python3
"""Sample near-head departure states uniformly within explicit waist regions."""
from __future__ import annotations
import argparse
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
from isaac_rl.transition_assignment import TransitionTable
from isaac_rl.transition_calibration import near_head_starts, strike_context
from isaac_rl.trajectory_qp import cost_for_horizon, metrics
from scripts.probe_transition_qp import save_npz, write_json
from scripts.search_transition_minimum_times import POLICY


def worker(args):
    torch.set_num_threads(2)
    meta = json.loads((args.out/'config.json').read_text())
    table = TransitionTable.load(args.out/'transition_table.npz')
    ck = torch.load(meta['checkpoint'], map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    torch.manual_seed(meta['seed'])
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(ck), device)
        cfg = A.CostConfig(**ck['train_cfg']['cost'])
        cfg.n_sub = meta['collocation_substeps']
        cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    cost.calibration_active_hand_only = True
    model.load_state_dict(ck['model']); model.eval()
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    short_cost = cost_for_horizon(cost, 50)
    folder = args.out/f'w{args.rank}'; folder.mkdir(exist_ok=True)
    _, lo, hi = cost._joint_box_hard_limits()
    lo, hi = lo.reshape(-1), hi.reshape(-1)
    groups = [(h, src, y) for h in range(2) for src in range(kit.num_drums)
        if table.allowed_drums()[h, src] for y in range(len(table.yaw_centers))]
    parts = []
    for i, (h, src, y) in enumerate(groups):
        if i % args.workers != args.rank: continue
        rng = np.random.default_rng(np.random.SeedSequence([meta['seed'], h, src, y]))
        low, high = meta['yaw_region_edges'][y:y+2]
        yaws = rng.uniform(low, high, meta['samples'])
        q, v, valid, diagnostics = near_head_starts(cost, kit, ('left', 'right')[h], src,
            yaws, meta['samples'], rng, ignore_idle_arm=True)
        columns = list(range(3))+list(arm_joint_indices(('left', 'right')[h]))
        bounded = ((q[:, columns] >= lo[columns]-1e-6)&
                   (q[:, columns] <= hi[columns]+1e-6)).all(-1)
        eligible = (valid & bounded).cpu().numpy()
        save_npz(args.out/'samples'/f'h{h}_d{src}_y{y}.npz', q=q.cpu().numpy(),
            v=v.cpu().numpy(), start_valid=valid.cpu().numpy(), eligible=eligible, **diagnostics)
        destinations = np.flatnonzero(table.allowed_drums()[h, :-1])
        keys = np.asarray([(h, src, dst, y, s) for dst in destinations
                           for s in range(meta['samples'])], int)
        indexes = keys[:, -1]
        Q, V = q[indexes], v[indexes]
        targets = np.full((len(keys), 2), -1, int); targets[:, h] = keys[:, 2]
        sources = np.full_like(targets, -1); sources[:, h] = src
        with torch.no_grad():
            plan = generator.predict(Q, V, torch.as_tensor(targets[:, None], device=device),
                Q.new_full((len(keys), 1), 1.), Q.new_full((len(keys), 1, 2), 2.),
                torch.ones(len(keys), 1, device=device, dtype=torch.bool), 0,
                return_plan=True, waist_yaw=Q[:, 0])
            traj = plan['traj'][:, :50].clone()
            idle = arm_joint_indices(('left', 'right')[1-h]); traj[:, :, idle] = Q[:, None, idle]
            lengths = torch.full_like(plan['lengths'], 50)
            ctx = strike_context(kit, targets, sources, np.full(len(keys), 50), device,
                speeds=np.full((len(keys), 2), 2.))
            ctx.target_top_entry_depth_m = .07
            result = metrics(short_cost, Q, V, traj, ctx, lengths, .07)
        parts.append(dict(keys=keys, departure_posture_valid=eligible[indexes],
            **{'before_'+k: value for k, value in result.items()},
            **{'after_'+k: value for k, value in result.items()}))
        print(json.dumps(dict(hand=h, source=kit.names[src], region=y+1,
            eligible_starts=int(eligible.sum()), sampled_yaw_range=[float(yaws.min()),float(yaws.max())])),
            flush=True)
    save_npz(folder/'cohort.npz', **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    write_json(folder/'done.json', {'status': 'complete'})


def parent(args):
    args.out.mkdir(parents=True, exist_ok=False); (args.out/'samples').mkdir()
    reference = TransitionTable.load(args.reference/'database/transition_table_decoder.npz')
    meta = dict(reference.metadata)
    edges = np.round(np.linspace(args.yaw_min, args.yaw_max, args.regions+1), 12)
    centers = (edges[:-1]+edges[1:])/2
    meta.update(samples=args.samples, seed=args.seed, yaw_centers=centers.tolist(),
        yaw_region_edges=edges.tolist(), yaw_sampling='uniform_within_region',
        yaw_region_interpretation='Initial yaw interval; decoder aim uses each sampled initial yaw',
        source_calibration=str(args.out), source_reference=str(args.reference), qp={'iterations': 0},
        refinement='None', forbidden_drums_by_hand=POLICY, decoder_only=True,
        baseline_run=str(args.out), deadlines_s=[1.], duration_trials=1,
        duration_search='One-second decoder baseline for fresh random-region starting cases',
        deadline_search='one-second', yaw_range_rad=[args.yaw_min,args.yaw_max],
        failure_interpretation='Decoder failure; not a certificate of physical infeasibility')
    for key in ('search_coverage','qp_coverage','qp_solver','qp_solver_version',
                'reuse_raw_from','coarse_horizons','limit','candidate_deadlines_s'):
        meta.pop(key, None)
    write_json(args.out/'config.json', meta)
    shape = (2, len(reference.drum_names), len(reference.drum_names), args.regions)
    table = TransitionTable(reference.drum_names, centers, np.full(shape, np.inf),
        np.zeros(shape, int), np.zeros(shape, int), reference.pair_yaw, reference.pair_supported, meta)
    table.save(args.out/'transition_table.npz')
    workers = []
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            command = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
                '--device', device, '--rank', str(rank), '--workers', str(len(args.devices))]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
            workers.append((subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env),log))
        while any(p.poll() is None for p, _ in workers):
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('Region start worker failed; inspect logs')
            time.sleep(1)
        if any(p.returncode != 0 for p, _ in workers): raise RuntimeError('Region start worker failed')
        parts = []
        for rank in range(len(workers)):
            with np.load(args.out/f'w{rank}/cohort.npz') as d:
                parts.append({k: d[k].copy() for k in d.files})
        combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.lexsort(combined['keys'].T[::-1]); combined = {k:v[order] for k,v in combined.items()}
        save_npz(args.out/'start_cohort.npz', **combined)
        write_json(args.out/'summary.json', dict(status='complete', regions=args.regions,
            region_edges_rad=edges.tolist(), sampled_departure_states=int(table.allowed_drums()[:, :-1].sum())*args.regions*args.samples,
            transition_start_records=len(combined['keys']),
            eligible_transition_starts=int(combined['departure_posture_valid'].sum())))
    finally:
        for process, log in workers:
            if process.poll() is None: process.terminate()
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=ROOT/'runs/transition_min_time_decoder_only_20261004_174008')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--regions', type=int, default=3)
    p.add_argument('--yaw-min', type=float, default=-.9)
    p.add_argument('--yaw-max', type=float, default=.9)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--devices', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args(); args.out=args.out.resolve(); args.reference=args.reference.resolve()
    if args.regions < 1 or args.samples < 1 or not np.isfinite([args.yaw_min,args.yaw_max]).all() or args.yaw_min >= args.yaw_max:
        p.error('Positive regions/samples and an increasing finite yaw range required')
    if args.devices and len(set(args.devices)) != len(args.devices): p.error('Worker devices must be distinct')
    parent(args) if args.devices else worker(args)


if __name__ == '__main__': main()
