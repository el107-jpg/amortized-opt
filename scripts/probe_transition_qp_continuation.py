#!/usr/bin/env python3
"""Continue the saved failed local repairs, holding states and deadlines fixed."""
from __future__ import annotations

import argparse
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
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, refine
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance, profile
from scripts.probe_transition_qp import save_npz, write_json
from scripts.probe_small_transition_repairs import collect


def worker(args):
    torch.set_num_threads(2)
    old_config = json.loads((args.previous/'config.json').read_text())
    full = Path(old_config['full_grid'])
    config = json.loads((full/'config.json').read_text())
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    with rnea_from_cache_if_needed():
        cfg = A.CostConfig(**checkpoint['train_cfg']['cost'])
        cfg.n_sub = config['collocation_substeps']
        cfg.w_velocity = max(cfg.w_velocity, 1.)
        cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    cost.calibration_active_hand_only = True
    kit = A.DrumKit.from_layout()
    qp_cfg = QPConfig(**old_config['qp'])
    qp_cfg.iterations = args.additional_steps
    allowance = SmallViolationAllowance(**old_config['allowance'])
    with np.load(args.previous/'repair_results.npz') as z:
        failed = {tuple(k) for k in z['keys'][~z['after_strict_success']]}
    folder = args.out/'continuation'/f'w{args.rank}'
    folder.mkdir(parents=True, exist_ok=True)
    parts = []; processed = 0; begin = time.monotonic()
    paths = sorted((args.previous/'qp').glob('w*/paths_*.npz'))
    for index, path in enumerate(paths):
        if index % args.workers != args.rank:
            continue
        with np.load(path) as z:
            take = np.asarray([tuple(k) in failed for k in z['keys']])
            if not take.any():
                continue
            keys = z['keys'][take].copy()
            q = torch.as_tensor(z['q'][take], device=device)
            v = torch.as_tensor(z['v'][take], device=device)
            seed = torch.as_tensor(z['after'][take], device=device)
            original = torch.as_tensor(z['before'][take], device=device)
        B, N, _ = seed.shape
        hand = int(keys[0, 0]); assert (keys[:, 0] == hand).all()
        targets = np.full((B, 2), -1, int); targets[:, hand] = keys[:, 2]
        sources = np.full_like(targets, -1); sources[:, hand] = keys[:, 1]
        ctx = strike_context(kit, targets, sources, np.full(B, N), device,
                             speeds=np.full((B, 2), 2.))
        ctx.target_top_entry_depth_m = .07
        lengths = torch.full((B, 3), N, device=device, dtype=torch.long)
        short_cost = cost_for_horizon(cost, N)
        before = profile(short_cost, q, v, seed, ctx, lengths, allowance=allowance)
        repaired, _, repair_metrics, info = refine(short_cost, q, v, seed, ctx, lengths,
                                                  hand, depth_m=.07, cfg=qp_cfg)
        after = profile(short_cost, q, v, repaired, ctx, lengths, allowance=allowance)
        histories = info['histories']
        part = dict(keys=keys, N=np.full(B, N),
            **{'before_'+name: val for name, val in before.items()},
            **{'after_'+name: val for name, val in after.items()},
            accepted_steps=repair_metrics['accepted_steps'],
            attempted_steps=np.asarray([len(h) for h in histories]),
            last_step_accepted=np.asarray([bool(h and h[-1].get('accepted')) for h in histories]),
            first_merit=np.asarray([h[0]['merit_before'] if h else 0. for h in histories]),
            final_merit=np.asarray([h[-1]['merit_after'] if h else 0. for h in histories]),
            additional_max_correction_rad=(repaired-seed).abs().amax((1, 2)).cpu().numpy(),
            total_max_correction_rad=(repaired-original).abs().amax((1, 2)).cpu().numpy())
        parts.append(part); processed += B
        save_npz(folder/f'paths_{index:03d}.npz', keys=keys, q=q.cpu().numpy(), v=v.cpu().numpy(),
                 original=original.cpu().numpy(), before=seed.cpu().numpy(), after=repaired.cpu().numpy())
        write_json(folder/f'qp_info_{index:03d}.json', info)
        save_npz(folder/'deadline_results.npz', **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        write_json(folder/'progress.json', dict(processed=processed, elapsed_s=time.monotonic()-begin))
        print(json.dumps(dict(processed=processed, strict_repaired=int(after['strict_success'].sum()))), flush=True)
    write_json(folder/'done.json', dict(status='complete', processed=processed, elapsed_s=time.monotonic()-begin))


def parent(args):
    args.out.mkdir(parents=True, exist_ok=True)
    previous_config = json.loads((args.previous/'config.json').read_text())
    write_json(args.out/'config.json', dict(previous=str(args.previous),
        additional_steps=args.additional_steps, devices=args.devices,
        qp=dict(previous_config['qp'], iterations=args.additional_steps),
        allowance=previous_config['allowance'], training=False,
        initialization='Exact saved paths after the previous five-step repair',
        states_deadlines_constraints_fixed=True))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'))
    registry = ROOT/'runs/tensorboard_active'/args.out.name
    if not registry.is_symlink():
        registry.symlink_to((args.out/'tb').resolve())
    begin = time.monotonic(); workers = []
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--previous', str(args.previous),
                   '--out', str(args.out), '--additional-steps', str(args.additional_steps),
                   '--device', device, '--rank', str(rank), '--workers', str(len(args.devices))]
            workers.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')), log))
        write_json(args.out/'run.json', dict(status='running', worker_pids=[p.pid for p, _ in workers]))
        while any(p.poll() is None for p, _ in workers):
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('Continuation worker failed; inspect logs')
            progress = [json.loads(p.read_text()) for p in (args.out/'continuation').glob('w*/progress.json')]
            writer.add_scalar('continuation/processed', sum(p['processed'] for p in progress),
                              int(time.monotonic()-begin)); writer.flush()
            time.sleep(3)
        if any(p.returncode for p, _ in workers):
            raise RuntimeError('Continuation worker failed; inspect logs')
        results = collect(args.out, 'continuation')
        save_npz(args.out/'continuation_results.npz', **results)
        success = results['after_strict_success']
        prior = json.loads((args.previous/'summary.json').read_text())
        summary = dict(status='complete',previously_unresolved=len(success),
            additional_steps=args.additional_steps, additionally_repaired=int(success.sum()),
            total_repaired_of_original_87=prior['qp_strictly_repaired']+int(success.sum()),
            still_unresolved=int((~success).sum()),
            remaining_within_allowance=int((~success&results['after_relaxed_success']).sum()),
            remaining_stalled=int((~success&~results['last_step_accepted']).sum()),
            remaining_at_iteration_budget=int((~success&results['last_step_accepted']&
                (results['attempted_steps']==args.additional_steps)).sum()),
            max_total_correction_deg=float(np.degrees(results['total_max_correction_rad'].max())),
            elapsed_s=time.monotonic()-begin,
            failure_interpretation='Failure or stalling is not a physical infeasibility certificate')
        write_json(args.out/'summary.json', summary); write_json(args.out/'run.json', summary)
        writer.add_scalar('continuation/additionally_repaired', int(success.sum()), args.additional_steps)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc)))
        raise
    finally:
        for process, log in workers:
            if process.poll() is None:
                process.terminate()
            log.close()
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--additional-steps', type=int, default=15)
    p.add_argument('--devices', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args(); args.previous = args.previous.resolve(); args.out = args.out.resolve()
    if args.additional_steps < 1:
        p.error('--additional-steps must be positive')
    parent(args) if args.devices else worker(args)


if __name__ == '__main__':
    main()
