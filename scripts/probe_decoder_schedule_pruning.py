#!/usr/bin/env python3
"""Check conservative early pruning against the same exhaustive probe inputs."""
from __future__ import annotations

import argparse
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
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.schedule_decoder import DecoderScheduleEnvironment
from isaac_rl.schedule_improvement import physical_table
from scripts.probe_transition_qp import write_json, save_npz


def timed(device, fn):
    torch.cuda.synchronize(device); start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    return result, time.perf_counter()-start


@torch.no_grad()
def worker(args):
    torch.set_num_threads(2)
    config = json.loads((args.previous/'config.json').read_text())
    torch.manual_seed(config['seed']); device = torch.device(args.device)
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(checkpoint), device)
        cfg = A.CostConfig(**checkpoint['train_cfg']['cost'])
        cfg.n_sub = 4; cfg.w_velocity = max(cfg.w_velocity, 1.); cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(checkpoint['model']); model.eval(); kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(checkpoint.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    env = DecoderScheduleEnvironment(generator, chunk=args.chunk)
    original = torch.load(args.previous/'probe_results.pt', weights_only=False)
    all_clips = torch.load(args.previous/'problems.pt', weights_only=False)['clips']
    # The previous probe did not save its bounce arrays. Reproduce its RNG
    # stream, checking the reconstructed states against the saved exact states.
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q: pose_clearance(cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS, arm_std=.12, yaw_std=.25)
    rng = np.random.default_rng(config['seed']+1)
    poses, _ = starts.sample(rng, len(all_clips))
    velocity = rng.normal(0, .1, poses.shape).astype(np.float32)
    velocity[np.max(np.abs(poses-ready_joint_pos()), axis=1)<1e-6] = 0.
    np.testing.assert_array_equal(poses, original['q'].numpy())
    np.testing.assert_array_equal(velocity, original['v'].numpy())
    draws_all = rng.uniform(.2, .8, (len(all_clips), 2, 4, 2)).astype(np.float32)
    owners = np.arange(min(args.windows, len(all_clips)))[args.rank::args.workers]
    clips = [all_clips[i] for i in owners]
    q = original['q'][owners].to(device); v = original['v'][owners].to(device)
    draws = torch.as_tensor(draws_all[owners], device=device)
    table = physical_table(clips, kit, device)
    weights = torch.tensor([8,4,2,1], device=device)
    bank = ((torch.arange(16, device=device)[:,None]//weights)&1)[None].expand(len(owners),-1,-1)
    seed_indices = (original['selected_actions'][owners,1].to(device)*weights).sum(-1)
    full, full_s = timed(device, lambda:env.score(table,q,v,bank,draws))
    write_json(args.out/f'w{args.rank}_progress.json', dict(stage='pruned_search', windows=len(owners)))
    pruned, pruned_s = timed(device, lambda:env.search_pruned(table,q,v,bank,draws,seed_indices))
    values = full['score'].double().sum(-1).mean(-1)
    reference = values.max(-1).values; best = values.argmax(-1)
    selected_actual = values[torch.arange(len(owners),device=device),pruned['best_index']]
    regret = reference-selected_actual
    # Verify using exhaustive values for the CHOSEN schedule, not just the
    # partial values that the pruning implementation believes it achieved.
    if (regret>1e-3).any():raise AssertionError(f'Pruning lost exhaustive optimum: max regret {regret.max().item()}')
    discarded = ~pruned['completed']
    if (pruned['upper_bounds'][discarded]>pruned['best_value'][:,None].expand_as(discarded)[discarded]+1e-3).any():
        raise AssertionError('Discarded candidate lacks a valid incumbent bound')
    old_values = original['searched']['score'][owners].double().sum(-1).mean(-1)
    torch.testing.assert_close(values,old_values.to(device),atol=1e-3,rtol=1e-5)
    online_full = []; online_pruned = []; online_saved = []; online_index_matches = []
    if args.rank==0:
        # Include both first-hit timing cohorts in the single-window benchmark.
        for b in range(min(8, min(args.windows, len(all_clips)))):
            one_table = physical_table([all_clips[b]], kit, device)
            one_q = original['q'][b:b+1].to(device); one_v = original['v'][b:b+1].to(device)
            one_draws = torch.as_tensor(draws_all[b:b+1],device=device)
            one_bank = bank[:1]; one_seed = (original['selected_actions'][b:b+1,1].to(device)*weights).sum(-1)
            full_fn = lambda:env.score(one_table,one_q,one_v,one_bank,one_draws)
            prune_fn = lambda:env.search_pruned(one_table,one_q,one_v,one_bank,one_draws,one_seed)
            if b%2==0:
                f,tf=timed(device,full_fn); p,tp=timed(device,prune_fn)
            else:
                p,tp=timed(device,prune_fn); f,tf=timed(device,full_fn)
            fv=f['score'].double().sum(-1).mean(-1)
            if (fv.max(-1).values-fv[0,p['best_index']])>1e-3:raise AssertionError('Single-window pruning lost optimum')
            online_full.append(tf*1000); online_pruned.append(tp*1000)
            online_saved.append(int(p['saved_group_evaluations']))
            online_index_matches.append(bool(fv.argmax(-1)==p['best_index']))
    save_npz(args.out/f'w{args.rank}_cases.npz',owner=owners,
        full_index=best.cpu().numpy(),pruned_index=pruned['best_index'].cpu().numpy(),
        full_value=reference.cpu().numpy(),pruned_value=pruned['best_value'].cpu().numpy(),
        regret=regret.cpu().numpy(),completed=pruned['completed'].cpu().numpy(),
        evaluated=pruned['evaluated_groups'].cpu().numpy(),
        full_evaluated=pruned['full_evaluated_groups'].cpu().numpy())
    torch.save(dict(q=q.cpu(),v=v.cpu(),draws=draws.cpu(),owners=owners,
        winner={k:t.cpu() for k,t in pruned['winner'].items()}),args.out/f'w{args.rank}_inputs_winners.pt')
    summary = dict(status='complete',windows=len(owners),full_s=full_s,pruned_s=pruned_s,
        same_best_index=int((best==pruned['best_index']).sum()),max_regret=float(regret.max()),
        discarded_candidates=int(discarded.sum()),candidates=int(discarded.numel()),
        evaluated_groups=int(pruned['evaluated_groups'].sum()),
        exhaustive_groups=int(pruned['full_evaluated_groups'].sum()),
        saved_groups=int(pruned['saved_group_evaluations']),online_full_ms=online_full,
        online_pruned_ms=online_pruned,online_saved_groups=online_saved,
        online_same_best_index=online_index_matches)
    write_json(args.out/f'w{args.rank}_done.json',summary);print(json.dumps(summary),flush=True)


def parent(args):
    args.out.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.previous/'config.json').read_text())
    write_json(args.out/'config.json',dict(previous=str(args.previous),windows=args.windows,
        checkpoint=config['checkpoint'],devices=args.devices,chunk=args.chunk,training=False,
        incumbent='Fully evaluated graph initialized from measured hand positions',
        bound='Mean measured prefix score plus 10 per remaining requested hit; outward rounding margin',
        inputs='Exact previous starting states, rhythms and reconstructed identical bounce draws',
        reference='Fresh exhaustive search with the same current implementation',
        candidate_order='Hamming distance to graph assignment',
        timing_includes_incumbent=True))
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(args.out/'tb'))
    registry=ROOT/'runs/tensorboard_active'/args.out.name
    if not registry.is_symlink():registry.symlink_to((args.out/'tb').resolve())
    begin=time.monotonic();workers=[]
    try:
        for rank,device in enumerate(args.devices):
            log=(args.out/f'w{rank}.log').open('w')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--previous',str(args.previous),
                '--out',str(args.out),'--windows',str(args.windows),'--chunk',str(args.chunk),
                '--device',device,'--rank',str(rank),'--workers',str(len(args.devices))]
            workers.append((subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')),log))
        write_json(args.out/'run.json',dict(status='running',worker_pids=[p.pid for p,_ in workers]))
        while any(p.poll() is None for p,_ in workers):
            if any(p.poll() not in (None,0) for p,_ in workers):raise RuntimeError('Pruning probe worker failed; inspect logs')
            writer.add_scalar('probe/completed_workers',len(list(args.out.glob('w*_done.json'))),int(time.monotonic()-begin));writer.flush()
            time.sleep(3)
        if any(p.returncode for p,_ in workers):raise RuntimeError('Pruning probe worker failed; inspect logs')
        parts=[json.loads((args.out/f'w{i}_done.json').read_text()) for i in range(len(workers))]
        report=dict(status='complete',elapsed_s=time.monotonic()-begin,
            **{k:sum(p[k] for p in parts) for k in ('windows','same_best_index','discarded_candidates','candidates','evaluated_groups','exhaustive_groups','saved_groups')},
            max_regret=max(p['max_regret'] for p in parts),
            saved_group_fraction=sum(p['saved_groups'] for p in parts)/sum(p['exhaustive_groups'] for p in parts),
            batched_full_gpu_s_per_window=sum(p['full_s'] for p in parts)/sum(p['windows'] for p in parts),
            batched_pruned_gpu_s_per_window=sum(p['pruned_s'] for p in parts)/sum(p['windows'] for p in parts),
            online_full_ms_median=float(np.median(parts[0]['online_full_ms'])),
            online_pruned_ms_median=float(np.median(parts[0]['online_pruned_ms'])),
            online_full_ms_p95=float(np.percentile(parts[0]['online_full_ms'],95)),
            online_pruned_ms_p95=float(np.percentile(parts[0]['online_pruned_ms'],95)),workers=parts)
        writer.add_scalar('pruning/saved_group_fraction',report['saved_group_fraction'],args.windows)
        write_json(args.out/'probe.json',report);write_json(args.out/'run.json',report)
        print(json.dumps(report),flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json',dict(status='failed',error=str(exc)));raise
    finally:
        for process,log in workers:
            if process.poll() is None:process.terminate()
            log.close()
        writer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=ROOT/'runs/decoder_schedule_exhaustive_vs_graph_20261004_195018')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--windows',type=int,default=256)
    p.add_argument('--chunk',type=int,default=64);p.add_argument('--devices',nargs='+')
    p.add_argument('--device',default='cuda:0');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args();args.out=args.out.resolve();args.previous=args.previous.resolve()
    if args.windows<1 or args.chunk<2:p.error('Positive windows and chunk >= bounce count required')
    parent(args) if args.devices else worker(args)


if __name__=='__main__':main()
