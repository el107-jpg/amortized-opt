#!/usr/bin/env python3
"""Exhaustive four-hit assignments versus graph, on actual decoder splines."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos, PELVIS_IN_ENV, STICK_TIP_SIGMA
from amortized_opt.fk import stick_contact_offset
from amortized_opt.hand_assign import LIMB_LEFT, LIMB_RIGHT
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.random_rhythms import RandomRhythmConfig, RandomRhythmSampler
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
from isaac_rl.schedule_improvement import graph_seed, physical_table
from scripts.probe_transition_qp import write_json


@torch.no_grad()
def worker(args):
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    problem = torch.load(args.out/'problems.pt', weights_only=False)
    clips = problem['clips']; device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(checkpoint), device)
        cfg = A.CostConfig(**checkpoint['train_cfg']['cost'])
        cfg.n_sub = 4; cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(checkpoint['model']); model.eval()
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(checkpoint.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    assert tuple(anchors['joint_names']) == tuple(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q: pose_clearance(cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS, arm_std=.12, yaw_std=.25)
    rng = np.random.default_rng(args.seed+1)
    q_np, start_stats = starts.sample(rng, len(clips))
    v_np = rng.normal(0, .1, q_np.shape).astype(np.float32)
    ready = np.max(np.abs(q_np-ready_joint_pos()), axis=1)<1e-6
    v_np[ready] = 0.
    search_draws = rng.uniform(.2, .8, (len(clips), 2, 4, 2)).astype(np.float32)
    eval_draws = rng.uniform(.2, .8, (len(clips), 4, 4, 2)).astype(np.float32)
    owner = np.arange(len(clips))[args.rank::args.workers]
    clips = [clips[i] for i in owner]
    q = torch.as_tensor(q_np[owner], device=device); v = torch.as_tensor(v_np[owner], device=device)
    physical = physical_table(clips, kit, device)
    bit_weights = torch.tensor([8, 4, 2, 1], device=device)
    bank = ((torch.arange(16, device=device)[:, None]//bit_weights)&1)[None].expand(len(clips), -1, -1)
    graph = []; informed_graph = []; fallback = 0
    tips = []
    for hand in ('left', 'right'):
        R, p = cost.fk.frame(q, hand+'_wrist')
        tips.append(p+(R@q.new_tensor(stick_contact_offset(hand, STICK_TIP_SIGMA))[:, None]).squeeze(-1))
    nearest = (torch.stack(tips, 1)[:, :, None]-(q.new_tensor(kit.positions)-q.new_tensor(PELVIS_IN_ENV))[None, None]).norm(dim=-1).argmin(-1)
    for b, clip in enumerate(clips):
        action, n = graph_seed(clip, kit); graph.append(action[:4]); fallback += n
        parked = {LIMB_LEFT:kit.names[int(nearest[b, 0])], LIMB_RIGHT:kit.names[int(nearest[b, 1])]}
        action, n = graph_seed(clip, kit, parked=parked); informed_graph.append(action[:4]); fallback += n
    graph = torch.as_tensor(np.asarray(graph), device=device)
    informed_graph = torch.as_tensor(np.asarray(informed_graph), device=device)
    graph_index = (graph*bit_weights).sum(-1); informed_index = (informed_graph*bit_weights).sum(-1)
    env = DecoderScheduleEnvironment(generator, chunk=args.chunk)
    torch.cuda.synchronize(device); begin = time.perf_counter()
    searched = env.score(physical, q, v, bank, torch.as_tensor(search_draws[owner], device=device))
    torch.cuda.synchronize(device); search_s = time.perf_counter()-begin
    values = searched['score'].sum(-1).mean(-1)
    best = values.argmax(-1); ids = torch.arange(len(clips), device=device)
    assert (values[ids, best] >= values[ids, graph_index]-1e-5).all()
    assert (values[ids, best] >= values[ids, informed_index]-1e-5).all()
    selected = torch.stack((graph, informed_graph, bank[ids, best]), 1)
    print(json.dumps(dict(stage='search_complete', windows=len(clips), search_s=search_s)), flush=True)
    write_json(args.out/f'w{args.rank}_progress.json', dict(stage='heldout_evaluation', windows=len(clips)))
    torch.cuda.synchronize(device); begin = time.perf_counter()
    heldout = env.score(physical, q, v, selected, torch.as_tensor(eval_draws[owner], device=device))
    torch.cuda.synchronize(device); eval_s = time.perf_counter()-begin
    online_ms = []
    if args.rank == 0:
        for b in range(min(8, len(clips))):
            one_table = {k:t[b:b+1] for k,t in physical.items()}
            torch.cuda.synchronize(device); begin = time.perf_counter()
            env.score(one_table, q[b:b+1], v[b:b+1], bank[b:b+1],
                      torch.as_tensor(search_draws[owner[b:b+1]], device=device))
            torch.cuda.synchronize(device); online_ms.append((time.perf_counter()-begin)*1000)
    torch.save(dict(owner=owner, q=q.cpu(), v=v.cpu(), ready=ready[owner],
        graph_index=graph_index.cpu(), informed_index=informed_index.cpu(), best_index=best.cpu(),
        selected_actions=selected.cpu(), search={k:t.cpu() for k,t in searched.items()},
        heldout={k:t.cpu() for k,t in heldout.items()}), args.out/f'w{args.rank}_results.pt')
    write_json(args.out/f'w{args.rank}_done.json', dict(status='complete', windows=len(clips),
        search_s=search_s, heldout_s=eval_s, online_search_ms=online_ms,
        graph_fallbacks=fallback, start_stats=start_stats))


def summarize(results):
    report = {}
    for c, name in enumerate(('graph', 'graph_with_start_drums', 'exhaustive')):
        requested = int(results['requested'][:, c].sum())
        report[name] = dict(hits=int(results['hits'][:, c].sum()), requested=requested,
            hit_rate=float(results['hits'][:, c].sum()/requested),
            score_mean=float(results['score'][:, c].sum(-1).mean()),
            strict_valid_hit_rate=float(results['strict_valid_hits'][:, c].sum()/requested),
            relaxed_valid_hit_rate=float(results['relaxed_valid_hits'][:, c].sum()/requested),
            mean_peak_torque_ratio=float(results['torque_peak_ratio'][:, c].amax(-1).mean()),
            mean_peak_penetration_mm=float(results['collision_m'][:, c].amax(-1).mean()*1000),
            mean_peak_velocity_ratio=float(results['velocity_peak_ratio'][:, c].amax(-1).mean()))
    return report


def parent(args):
    args.out.mkdir(parents=True, exist_ok=True); kit = A.DrumKit.from_layout()
    rng = np.random.default_rng(args.seed)
    sampler = RandomRhythmSampler(kit, RandomRhythmConfig(max_hits=4, pair_probability=0.))
    clips = []
    while len(clips)<args.windows:
        for clip in sampler.sample(rng, max(32, args.windows-len(clips))):
            if len(clip.hits)!=4:
                continue
            first = .16 if len(clips)%2==0 else .4
            delta = first-clip.hits[0].time
            hits = [replace(h, time=round(h.time+delta, 8), raw_time=round(h.raw_time+delta, 8)) for h in clip.hits]
            if hits[-1].time>1.+1e-7:
                continue
            clips.append(replace(clip, hits=hits, duration=hits[-1].time))
            if len(clips)==args.windows:
                break
    torch.save(dict(clips=clips), args.out/'problems.pt')
    write_json(args.out/'rhythms.json', [dict(name=c.name, hits=[dict(time=h.time, drum=h.drum_name) for h in c.hits]) for c in clips])
    write_json(args.out/'config.json', dict(checkpoint=str(args.checkpoint), windows=args.windows,
        seed=args.seed, devices=args.devices, chunk=args.chunk, training=False,
        search='All 16 assignments of four single-hit events; no drum eligibility masks',
        score='Existing analytic hit/miss and peak-constraint score scale, applied to actual decoder spline prefixes',
        environment='Perfect spline tracking and the existing random analytic rebound; no Isaac',
        first_hit_s=[.16, .4], search_bounce_draws=2, heldout_bounce_draws=4,
        physics='Both arms and shared waist; 5 ms collision samples, knot_fd torque, analytic spline velocity extrema',
        hit_rules='Existing 10 mm contact region at deadline and inward normal speed greater than 0.1 m/s',
        collision_policy='Verified target top entry up to 10 mm; timed actual-source departure exemption',
        starting_states='Ready, perturbed ready, executed pose anchors; 0.1 rad/s Gaussian starting velocity except exact ready',
        graph_baselines=['Existing graph with unknown parked drums', 'Same graph initialized to nearest drum per measured hand'],
        evaluation='New procedural rhythms; heldout bounce draws shared between methods'))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'))
    registry = ROOT/'runs/tensorboard_active'/args.out.name
    if not registry.is_symlink():registry.symlink_to((args.out/'tb').resolve())
    workers = []; begin = time.monotonic()
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--checkpoint', str(args.checkpoint),
                '--out', str(args.out), '--windows', str(args.windows), '--seed', str(args.seed),
                '--chunk', str(args.chunk), '--device', device, '--rank', str(rank), '--workers', str(len(args.devices))]
            workers.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')), log))
        write_json(args.out/'run.json', dict(status='running', worker_pids=[p.pid for p,_ in workers]))
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in (None, 0) for p,_ in workers):raise RuntimeError('Probe worker failed; inspect logs')
            done = len(list(args.out.glob('w*_done.json')))
            writer.add_scalar('probe/completed_workers', done, int(time.monotonic()-begin)); writer.flush()
            time.sleep(3)
        if any(p.returncode for p,_ in workers):raise RuntimeError('Probe worker failed; inspect logs')
        parts = [torch.load(args.out/f'w{i}_results.pt', weights_only=False) for i in range(len(workers))]
        order = np.argsort(np.concatenate([p['owner'] for p in parts]))
        heldout = {k:torch.cat([p['heldout'][k] for p in parts])[order] for k in parts[0]['heldout']}
        searched = {k:torch.cat([p['search'][k] for p in parts])[order] for k in parts[0]['search']}
        report = dict(status='complete', windows=args.windows, heldout=summarize(heldout), elapsed_s=time.monotonic()-begin)
        scores = heldout['score'].sum(-1).mean(-1)
        gains = scores[:, 2]-scores[:, 1]
        hit_gains = (heldout['hits'][:, 2]-heldout['hits'][:, 1]).sum((1, 2)).float()/16
        boot = np.random.default_rng(args.seed+2).integers(args.windows, size=(5000, args.windows))
        report.update(heldout_windows_better=int((gains>1e-5).sum()), heldout_windows_worse=int((gains< -1e-5).sum()),
            heldout_windows_tied=int((gains.abs()<=1e-5).sum()),
            heldout_hit_rate_difference_95pct_ci=np.percentile(hit_gains.numpy()[boot].mean(1), [2.5, 97.5]).tolist())
        report['timing'] = dict(worker=[json.loads((args.out/f'w{i}_done.json').read_text()) for i in range(len(workers))],
            search_gpu_seconds_per_window=sum(json.loads((args.out/f'w{i}_done.json').read_text())['search_s'] for i in range(len(workers)))/args.windows,
            note='Batched throughput while four GPUs run concurrently; not single-window online latency')
        online_ms = report['timing']['worker'][0]['online_search_ms']
        report['timing'].update(online_search_ms_median=float(np.median(online_ms)),
                               online_search_ms_p95=float(np.percentile(online_ms, 95)))
        for name, m in report['heldout'].items():
            writer.add_scalar(f'probe/{name}/hit_rate', m['hit_rate'], args.windows)
            writer.add_scalar(f'probe/{name}/score', m['score_mean'], args.windows)
        torch.save(dict(heldout=heldout, searched=searched,
            q=torch.cat([p['q'] for p in parts])[order], v=torch.cat([p['v'] for p in parts])[order],
            selected_actions=torch.cat([p['selected_actions'] for p in parts])[order]), args.out/'probe_results.pt')
        write_json(args.out/'probe.json', report); write_json(args.out/'run.json', report)
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc))); raise
    finally:
        for process, log in workers:
            if process.poll() is None:process.terminate()
            log.close()
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--windows', type=int, default=256)
    p.add_argument('--seed', type=int, default=202610043); p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--devices', nargs='+'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--rank', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    args = p.parse_args(); args.out = args.out.resolve(); args.checkpoint = args.checkpoint.resolve()
    if args.windows<1 or args.chunk<1:p.error('Positive windows and chunk required')
    parent(args) if args.devices else worker(args)


if __name__=='__main__':main()
