#!/usr/bin/env python3
"""Latency and throughput of all 16 frozen-decoder schedule assignments."""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Four workers must not each create a full machine's worth of BLAS threads
# while constructing the frozen spline gain tables.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
from isaac_rl.schedule_improvement import physical_table


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--windows', type=int, default=32)
    p.add_argument('--online', type=int, default=8)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--cuda-graphs', action='store_true')
    p.add_argument('--barrier', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = ROOT/'runs/decoder_schedule_exhaustive_vs_graph_20261004_195018'
    cached = ROOT/'runs/decoder_schedule_early_pruning_20261004_210205'
    config = json.loads((source/'config.json').read_text())
    torch.manual_seed(config['seed'])
    device = torch.device(args.device)
    cp = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(cp), device)
        cfg = A.CostConfig(**cp['train_cfg']['cost'])
        cfg.n_sub = 4
        cfg.w_velocity = max(cfg.w_velocity, 1.)
        cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(cp['model']); model.eval()
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(cp.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    env = DecoderScheduleEnvironment(generator, chunk=args.chunk, cuda_graphs=args.cuda_graphs)
    original = torch.load(source/'probe_results.pt', weights_only=False)
    all_clips = torch.load(source/'problems.pt', weights_only=False)['clips']
    owners = np.arange(min(args.windows, len(all_clips)))[args.rank::args.workers]
    clips = [all_clips[i] for i in owners]
    draws_all = torch.empty(len(original['q']), 2, 4, 2)
    for path in sorted(cached.glob('w*_inputs_winners.pt')):
        part = torch.load(path, weights_only=False)
        draws_all[part['owners']] = part['draws']
    q = original['q'][owners].to(device)
    v = original['v'][owners].to(device)
    draws = draws_all[owners].to(device)
    table = physical_table(clips, kit, device)
    bank = ((torch.arange(16, device=device)[:, None]//torch.tensor([8,4,2,1], device=device))&1)
    bank = bank[None].expand(len(q), -1, -1)

    def run(n, start=0):
        tab = {k:t[start:start+n] for k,t in table.items()}
        result = env.score(tab, q[start:start+n], v[start:start+n], bank[:n], draws[start:start+n])
        values = result['score'].double().sum(-1).mean(-1)
        # Include selecting the winner and returning the decision to the CPU.
        chosen = values.argmax(-1).cpu()
        return result, chosen

    # Every distinct prefix duration has its own physics graph. Warm all
    # measured windows before reporting latency, and report setup separately.
    setup_start = time.perf_counter()
    for i in range(min(args.online, len(q))):
        run(1, i)
    torch.cuda.synchronize(device)
    online_setup_s = time.perf_counter()-setup_start
    if args.profile:
        prof = cProfile.Profile(); prof.enable(); run(1); torch.cuda.synchronize(device); prof.disable()
        with (args.out/'profile.txt').open('w') as f:
            pstats.Stats(prof, stream=f).sort_stats('cumtime').print_stats(45)
    online = []
    for i in range(min(args.online, len(q))):
        torch.cuda.synchronize(device); start = time.perf_counter()
        run(1, i); torch.cuda.synchronize(device)
        online.append((time.perf_counter()-start)*1000)
    batch_setup_start = time.perf_counter()
    warmed, _ = run(len(q))
    torch.cuda.synchronize(device)
    warm_values = warmed['score'].double().sum(-1).mean(-1).cpu()
    del warmed
    batch_setup_s = time.perf_counter()-batch_setup_start
    startup_peak_allocated_mb = torch.cuda.max_memory_allocated(device)/2**20
    startup_peak_reserved_mb = torch.cuda.max_memory_reserved(device)/2**20
    torch.cuda.reset_peak_memory_stats(device)
    if args.barrier:
        args.barrier.mkdir(parents=True, exist_ok=True)
        (args.barrier/f'ready_{args.rank}').touch()
        while not (args.barrier/'start').exists():
            time.sleep(.05)
    torch.cuda.synchronize(device); start = time.perf_counter()
    result, chosen = run(len(q)); torch.cuda.synchronize(device)
    batch_end = time.perf_counter()
    elapsed = batch_end-start
    peak_mb = torch.cuda.max_memory_allocated(device)/2**20
    peak_reserved_mb = torch.cuda.max_memory_reserved(device)/2**20
    values = result['score'].double().sum(-1).mean(-1).cpu()
    stored_reference = original['searched']['score'][owners].double().sum(-1).mean(-1)
    # GEMM kernels can vary with batch shape. Compare replay against eager
    # evaluation of the same shape, and report drift from the old chunk=64
    # teacher separately rather than silently calling a changed label identical.
    if args.cuda_graphs:
        wrapped_decoder = generator.model.decoder.forward
        wrapped_contact = env.contact.contact
        wrapped_rebound = env._rebound
        generator.model.decoder.forward = wrapped_decoder.fn
        env.contact.contact = wrapped_contact.fn
        env._rebound = wrapped_rebound.fn
        env.cuda_graphs = False
        eager_start = time.perf_counter()
        eager, eager_chosen = run(len(q)); torch.cuda.synchronize(device)
        eager_s = time.perf_counter()-eager_start
        reference = eager['score'].double().sum(-1).mean(-1).cpu()
        generator.model.decoder.forward = wrapped_decoder
        env.contact.contact = wrapped_contact
        env._rebound = wrapped_rebound
        env.cuda_graphs = True
        torch.testing.assert_close(result['poses'], eager['poses'], atol=2e-5, rtol=2e-5)
    else:
        # Validate deterministic repeated eager evaluation at the SAME batch
        # shape, since the old teacher used smaller GEMM batches.
        reference = warm_values; eager_s = elapsed
    torch.testing.assert_close(values, reference, atol=1e-3, rtol=1e-5)
    if not torch.equal(chosen, reference.argmax(-1)):
        raise AssertionError('Optimization changed the exhaustive winner')
    summary = dict(device=str(device), chunk=args.chunk, windows=len(q), candidates=16,
        allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default'),
        cuda_graphs=args.cuda_graphs, online_setup_s=online_setup_s, batch_setup_s=batch_setup_s,
        batch_start_monotonic=start, batch_end_monotonic=batch_end,
        bounce_draws=2, hits_per_window=4, online_ms=online,
        online_median_ms=float(np.median(online)) if online else None,
        online_p95_ms=float(np.percentile(online,95)) if online else None,
        batched_wall_s=elapsed, batched_ms_per_window=elapsed*1000/len(q),
        windows_per_s=len(q)/elapsed, peak_allocated_mb=peak_mb,
        peak_reserved_mb=peak_reserved_mb,
        startup_peak_allocated_mb=startup_peak_allocated_mb,
        startup_peak_reserved_mb=startup_peak_reserved_mb,
        max_score_difference=float((values-reference).abs().max()), same_winner_count=len(q),
        eager_same_batch_s=eager_s, eager_speedup=eager_s/elapsed,
        prior_chunk64_max_score_difference=float((values-stored_reference).abs().max()),
        prior_chunk64_same_winner_count=int((chosen==stored_reference.argmax(-1)).sum()),
        prior_chunk64_max_selected_regret=float((stored_reference.max(-1).values-stored_reference[torch.arange(len(q)),chosen]).max()),
        prior_chunk64_field_differences={name:float((result[name].cpu().double()-original['searched'][name][owners].double()).abs().max())
            for name in ('poses','gap','normal_speed','hits','collision_m','torque_peak_ratio','velocity_peak_ratio')},
        checkpoint=config['checkpoint'], precision='float32', physics_subsamples=4,
        timing_includes='packing, 4 sequential replans, contact/rebound, both-arm collision/torque/velocity/bounds, scoring, CPU decision',
        training=False)
    (args.out/'benchmark.json').write_text(json.dumps(summary, indent=2)+'\n')
    torch.save(dict(values=values, chosen=chosen), args.out/'results.pt')
    # Store every candidate's value, permitting supervised assignment training
    # or preference/ranking training without rerunning the teacher search.
    torch.save(dict(q=q.cpu(), v=v.cpu(), clips=clips, draws=draws.cpu(),
        candidates=bank.cpu(), values=values, chosen=chosen,
        assignments=bank.cpu()[torch.arange(len(q)), chosen], owners=owners,
        checkpoint=config['checkpoint'], source=str(source)), args.out/'teacher_labels.pt')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
