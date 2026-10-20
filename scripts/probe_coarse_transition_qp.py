#!/usr/bin/env python3
"""Paired 20 ms versus 60 ms + 20 ms SQP duration search on frozen departures."""
from __future__ import annotations
import argparse
import copy
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
from isaac_rl.duration_search import search_duration
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, metrics, refine
from scripts.build_unrestricted_transition_chart import retime
from scripts.probe_transition_qp import save_npz, write_json


def prepare_cases(out, original):
    specs = [(0, 3, 0, 1), (0, 0, 3, 0), (0, 2, 0, 0),
             (0, 1, 3, 4), (0, 2, 3, 6)]
    cases, qs, vs = [], [], []
    # Keep the known single-transition probe in the matched comparison.
    p = ROOT/'runs/transition_min_time_three_regions_20261004_174951/cohort/samples/h1_d2_y0.npz'
    with np.load(p) as d:
        qs.append(d['q'][2]); vs.append(d['v'][2])
    cases.append(dict(hand=1, source=2, destination=6, sample=2, initial_s=.56,
                      origin=str(p), row=2))
    for hand, src, dst, sample in specs:
        p = next(original.glob(f'w*/h{hand}_source_{src}.npz'))
        with np.load(p) as d:
            row = int(np.flatnonzero((d['destination'] == dst)&(d['sample'] == sample))[0])
            duration = float(d['decoder_minimum_s'][row])
            if not np.isfinite(duration):
                raise ValueError(f'No verified raw seed: {(hand, src, dst, sample)}')
            qs.append(d['q'][row]); vs.append(d['v'][row])
        cases.append(dict(hand=hand, source=src, destination=dst, sample=sample,
                          initial_s=duration, origin=str(p), row=row))
    save_npz(out/'departures.npz', q=np.asarray(qs), v=np.asarray(vs))
    write_json(out/'cases.json', cases)


def load_method_result(out, case_index, method):
    path = out/f'case_{case_index}_{method}.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['method'] != method or not result['final_metrics']['success']:
        raise ValueError(f'Invalid completed method result: {path}')
    # The first probe saved verified metrics before its witness-file error.
    # Those completed measurements can be reused without fabricating a path.
    result['witness_saved'] = (out/f'case_{case_index}_{method}.npz').exists()
    return result


def save_method_result(out, case_index, method, result, q, v, trajectory):
    save_npz(out/f'case_{case_index}_{method}.npz', q=q.detach().cpu().numpy(),
             v=v.detach().cpu().numpy(), trajectory=trajectory.detach().cpu().numpy(),
             duration_s=np.asarray(result['minimum_s']))
    result['witness_saved'] = True
    write_json(out/f'case_{case_index}_{method}.json', result)


def worker(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = A.build_model(A.model_config_from_checkpoint(ck), device)
    model.load_state_dict(ck['model']); model.eval()
    cfg = A.CostConfig(**ck['train_cfg']['cost'])
    cfg.n_sub = 4; cfg.physics_mode = 'spline'
    cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
    cost = A.FusedCost(cfg, max_horizon=100, device=device, rnea=load_cached(device=device))
    cost.calibration_active_hand_only = True
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    qp_cfg = QPConfig(iterations=30, trust_rad=.03, jacobian_chunk=32,
        signed_contact=True, line_search=tuple(2.**(-k) for k in range(12)))
    costs = {}
    def get_cost(N, dense=False):
        if (N, dense) not in costs:
            c = cost_for_horizon(cost, N); c.cfg = copy.copy(cost.cfg)
            c.cfg.n_sub = 32 if dense else 4
            costs[N, dense] = c
        return costs[N, dense]
    cases = json.loads((args.out/'cases.json').read_text())
    with np.load(args.out/'departures.npz') as d:
        all_q = torch.as_tensor(d['q'], device=device)
        all_v = torch.as_tensor(d['v'], device=device)
    for case_index, case in enumerate(cases):
        if case_index % args.workers != args.rank:
            continue
        if (args.out/f'case_{case_index}.json').exists():
            continue
        q, v = all_q[case_index:case_index+1], all_v[case_index:case_index+1]
        hand, src, dst = case['hand'], case['source'], case['destination']
        N0 = round(case['initial_s']/.02)
        def context(N):
            targets = np.full((1, 2), -1, int); targets[:, hand] = dst
            sources = np.full_like(targets, -1); sources[:, hand] = src
            ctx = strike_context(kit, targets, sources, [N], device, speeds=np.full((1, 2), 2.))
            ctx.target_top_entry_depth_m = .07
            return targets, ctx, torch.full((1, 3), N, device=device, dtype=torch.long)
        fresh, references = {}, []
        began = time.monotonic()
        # Shared seed preparation is excluded from paired search wall times.
        for N in range(1, N0+1):
            targets, ctx, L = context(N)
            choices, merit = [], []
            with torch.no_grad():
                for aim in (q[:, 0], None):
                    plan = generator.predict(q, v, torch.as_tensor(targets[:, None], device=device),
                        q.new_full((1, 1), N*.02), q.new_full((1, 1, 2), 2.),
                        torch.ones(1, 1, device=device, dtype=torch.bool), 0,
                        return_plan=True, waist_yaw=aim)
                    path = plan['traj'][:, :N].clone()
                    idle = arm_joint_indices(('left', 'right')[1-hand])
                    path[:, :, idle] = q[:, None, idle]
                    m = metrics(get_cost(N), q, v, path, ctx, L, .07, signed_contact=True)
                    choices.append(path); merit.append(m['merit'][0])
            fresh[N] = choices[int(np.argmin(merit))]
            if N == N0:
                for path in choices:
                    m = metrics(get_cost(N, True), q, v, path, ctx, L, .07, signed_contact=True)
                    if m['success'][0]: references.append(path)
        if not references:
            raise RuntimeError(f'Matched initial reference failed dense verification: case {case_index}')
        initial = references[0]
        seed_seconds = time.monotonic()-began
        # Alternating order reduces systematic cache/thermal timing bias.
        strides = (1, 3) if case_index % 2 == 0 else (3, 1)
        outputs = {}
        for stride in strides:
            method = 'fine20' if stride == 1 else 'coarse60_refine20'
            saved = load_method_result(args.out, case_index, method)
            if saved is not None:
                outputs[method] = saved
                print(json.dumps(dict(case=case_index, method=method, reused=True,
                                      minimum_s=saved['minimum_s'])), flush=True)
                continue
            trace, counters = [], dict(repair_calls=0, qp_solves=0, qp_iterations=0,
                                       dense_repair_calls=0, seed_repairs_succeeded=0)
            started = time.monotonic()
            def repaired(seed, N, dense=False):
                _, ctx, L = context(N)
                path, before, after, info = refine(get_cost(N, dense), q, v, seed, ctx, L,
                    hand, depth_m=.07, cfg=qp_cfg)
                counters['repair_calls'] += 1
                counters['dense_repair_calls'] += int(dense)
                counters['qp_solves'] += sum(len(h) for h in info['histories'])
                counters['qp_iterations'] += sum(h.get('iterations', 0) for hs in info['histories'] for h in hs)
                return path, before, after, info
            def attempt(N, best, parent, stage):
                _, ctx, L = context(N)
                for kind, seed in [('continuation', retime(q, best, N)), ('fresh', fresh[N])]:
                    path, before, after, info = repaired(seed, N)
                    if after['success'][0]:
                        checked = metrics(get_cost(N, True), q, v, path, ctx, L, .07,
                                          signed_contact=True)
                        if not checked['success'][0]:
                            path, _, checked, dense_info = repaired(path, N, True)
                            info['elapsed_s'] += dense_info['elapsed_s']
                        success = bool(checked['success'][0])
                    else:
                        checked = after; success = False
                    if success and not before['success'][0]:
                        counters['seed_repairs_succeeded'] += 1
                    row = dict(duration_s=N*.02, parent_s=parent*.02, stage=stage,
                        seed=kind, success=success, seconds=info['elapsed_s'],
                        initial_merit=float(before['merit'][0]), final_merit=float(checked['merit'][0]),
                        max_correction_rad=float((path-seed).abs().max()),
                        torque_excess_nm=float(checked['torque_excess_max'][0]),
                        collision_clearance_m=float(checked['collision_min_sdf'][0]),
                        head_miss_m=float(checked['terminal_region_dist'][0]))
                    trace.append(row)
                    write_json(args.out/f'progress_{args.rank}.json', dict(case=case_index,
                        method=method, duration_s=N*.02, parent_s=parent*.02,
                        seconds=time.monotonic()-started, **counters))
                    if success:
                        return path, True
                return best, False
            found = search_duration(N0, initial.clone(), attempt, stride=stride, refine_bands=2)
            _, ctx, L = context(found.horizon)
            final = metrics(get_cost(found.horizon, True), q, v, found.path, ctx, L, .07,
                            signed_contact=True)
            if not final['success'][0]: raise RuntimeError('Final witness lost feasibility')
            output = dict(method=method, minimum_s=found.horizon*.02,
                seconds=time.monotonic()-started, deadlines_checked=len(found.deadlines),
                **counters, final_metrics={k: v[0].item() for k, v in final.items()},
                deadlines=found.deadlines, repairs=trace)
            outputs[method] = output
            save_method_result(args.out, case_index, method, output, q, v, found.path)
            print(json.dumps(dict(case=case_index, **{k: output[k] for k in
                ('method', 'minimum_s', 'seconds', 'repair_calls', 'qp_solves')})), flush=True)
        fine, coarse = outputs['fine20'], outputs['coarse60_refine20']
        write_json(args.out/f'case_{case_index}.json', dict(case=case_index, **case,
            source_name=kit.names[src], destination_name=kit.names[dst], seed_preparation_s=seed_seconds,
            fine20=fine, coarse60_refine20=coarse,
            difference_ms=(coarse['minimum_s']-fine['minimum_s'])*1000,
            speedup=fine['seconds']/max(coarse['seconds'], 1e-9)))


def publish(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('case_*.json'))
            if p.stem.split('_')[-1].isdigit()]
    summary = dict(status='running', completed_cases=len(rows), total_cases=6,
        equal_minimums=sum(abs(r['difference_ms']) < 1e-6 for r in rows),
        coarse_slower_minimums=sum(r['difference_ms'] > 1e-6 for r in rows),
        coarse_better_minimums=sum(r['difference_ms'] < -1e-6 for r in rows), cases=rows)
    if rows:
        old = sum(r['fine20']['seconds'] for r in rows)
        new = sum(r['coarse60_refine20']['seconds'] for r in rows)
        summary.update(aggregate_speedup=old/max(new, 1e-9),
            fine20_repair_calls=sum(r['fine20']['repair_calls'] for r in rows),
            coarse60_repair_calls=sum(r['coarse60_refine20']['repair_calls'] for r in rows))
    write_json(out/'summary.json', summary)
    return summary


def parent(args):
    if args.resume:
        saved = json.loads((args.out/'config.json').read_text())
        if saved['checkpoint'] != str(args.checkpoint) or saved['original'] != str(args.original):
            raise ValueError('Resume checkpoint/original run differ')
        old = json.loads((args.out/'run.json').read_text())
        for pid in [old.get('launcher_pid'), *old.get('worker_pids', [])]:
            if pid:
                try: os.kill(pid, 0)
                except ProcessLookupError: pass
                else: raise ValueError(f'Probe launcher/worker still exists: {pid}')
        saved['resumed'] = True
        saved['timing_note'] = 'Shared GPUs; method timing spans process restarts, so compare solve counts as well'
        write_json(args.out/'config.json', saved)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        prepare_cases(args.out, args.original)
        write_json(args.out/'config.json', dict(checkpoint=str(args.checkpoint), original=str(args.original),
        coarse_spacing_s=.06, fine_spacing_s=.02, refinement_window_s=.12,
        failed_coarse_deadlines_prune=False, original_run_left_running=True,
        initial_references='Same dense-verified raw decoder plan for both methods',
        acceptance='70mm depth; no speed gate; active hand/waist; target top entry; dense32',
        limitation='Coarse + local refinement can miss isolated feasible fine-grid islands',
        qp=asdict(QPConfig(iterations=30, trust_rad=.03, jacobian_chunk=32,
            signed_contact=True, line_search=tuple(2.**(-k) for k in range(12))))))
    children = []
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('a' if args.resume else 'w')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
                '--checkpoint', str(args.checkpoint), '--device', device,
                '--rank', str(rank), '--workers', str(len(args.devices))]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
            children.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env=env), log))
        write_json(args.out/'run.json', dict(status='running', launcher_pid=os.getpid(),
            worker_pids=[p.pid for p, _ in children]))
        previous = -1
        while any(p.poll() is None for p, _ in children):
            if any(p.poll() not in (None, 0) for p, _ in children):
                raise RuntimeError('Probe worker failed; inspect worker logs')
            summary = publish(args.out)
            if summary['completed_cases'] != previous:
                print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)
                previous = summary['completed_cases']
            time.sleep(2)
        if any(p.returncode for p, _ in children): raise RuntimeError('Probe worker failed')
        summary = publish(args.out)
        if summary['completed_cases'] != 6: raise RuntimeError('Incomplete paired comparison')
        summary['status'] = 'complete'
        write_json(args.out/'summary.json', summary); write_json(args.out/'run.json', {'status': 'complete'})
        print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        for p, log in children:
            if p.poll() is None: p.terminate()
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--original', type=Path, default=ROOT/'runs/transition_qp_unrestricted_20261005')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--devices', nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); args.out = args.out.resolve()
    args.checkpoint = args.checkpoint.resolve(); args.original = args.original.resolve()
    parent(args) if args.devices else worker(args)


if __name__ == '__main__': main()
