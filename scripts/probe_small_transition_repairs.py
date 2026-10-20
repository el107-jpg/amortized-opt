#!/usr/bin/env python3
"""Audit brief/small violations across the full grid, then repair earliest candidates."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import csv

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.ik import arm_joint_indices
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.transition_assignment import TransitionTable,minimum_statistics
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import QPConfig,cost_for_horizon,refine
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance,profile
from scripts.probe_transition_qp import save_npz,write_json


def seed_rows(full,N):
    if N==50:
        config=json.loads((full/'config.json').read_text())
        paths=[Path(config['baseline_run'])/'start_cohort.npz']
    else:paths=sorted(full.glob(f'w*/coarse_{N:03d}.npz'))
    parts=[]
    for p in paths:
        with np.load(p) as d:parts.append({k:d[k].copy() for k in d.files})
    return {k:np.concatenate([a[k] for a in parts]) for k in parts[0]}


def worker(args):
    torch.set_num_threads(2)
    config=json.loads((args.full/'config.json').read_text())
    with np.load(args.full/'database/per_start_minima.npz') as d:eligible=d['eligible'].copy()
    checkpoint=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    device=torch.device(args.device)
    with rnea_from_cache_if_needed():
        model=A.build_model(A.model_config_from_checkpoint(checkpoint),device)
        cfg=A.CostConfig(**checkpoint['train_cfg']['cost']);cfg.n_sub=config['collocation_substeps'];cfg.physics_mode='spline'
        cfg.w_velocity=max(cfg.w_velocity,1.);cfg.w_joint_box=max(cfg.w_joint_box,1.)
        cost=A.FusedCost(cfg,max_horizon=100,device=device)
    cost.calibration_active_hand_only=True
    model.load_state_dict(checkpoint['model']);model.eval()
    kit=A.DrumKit.from_layout()
    generator=DecoderHitPoseGenerator(model,cost,kit,
        waist_upright_ease_wp=(checkpoint.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    allowance=SmallViolationAllowance()
    qp_cfg=QPConfig(iterations=5,trust_rad=.03,jacobian_chunk=256)
    folder=args.out/args.stage/f'w{args.rank}';folder.mkdir(parents=True,exist_ok=True)
    sample_cache={};processed=0;begin=time.monotonic()
    if args.stage=='qp':
        with np.load(args.out/'repair_candidates.npz') as d:
            repair_keys=d['keys'].copy();repair_N=d['N'].copy()
    for N in range(1,101):
        if N%args.workers!=args.rank:continue
        if args.stage=='audit':
            rows=seed_rows(args.full,N);keys=rows['keys']
            take=eligible[tuple(keys.T)]&~rows['before_success']
            take&=(rows['before_terminal_region_dist']<=1e-6)&(rows['before_contact_dir_cos']>=.5)
            # The cached angle is only a conservative pad prefilter. Cymbals
            # accept the legal shaft region; current geometry is re-evaluated below.
            from amortized_opt.strike import shaft_hit_allowed
            pad=np.asarray([not shaft_hit_allowed(kit.names[int(k[2])]) for k in keys])
            take&=(~pad)|((rows['before_stick_elev_deg']>=5)&(rows['before_stick_elev_deg']<=45))
            take&=(rows['before_velocity_excess_max']<=1e-6)&(rows['before_joint_box_excess_max']<=1e-6)
            take&=(rows['before_collision_min_sdf']>=-allowance.collision_depth_m-1e-6)
            # Loose absolute bound; the actual gate below is per-joint percent
            # and duration, measured against the checkpoint's available torque.
            take&=(rows['before_torque_excess_max']<=float(cost.tau_hard.max())*.10+1e-6)
            keys=keys[take]
        else:keys=repair_keys[repair_N==N]
        if not len(keys):continue
        parts=[];short_cost=cost_for_horizon(cost,N)
        for hand in (0,1):
            selected=keys[keys[:,0]==hand]
            for start in range(0,len(selected),args.chunk if args.stage=='audit' else 16):
                chunk=args.chunk if args.stage=='audit' else 16
                k=selected[start:start+chunk];qs=[];vs=[]
                for h,src,dst,y,s in k:
                    cache_key=(h,src,y)
                    if cache_key not in sample_cache:
                        with np.load(Path(config['source_calibration'])/'samples'/f'h{h}_d{src}_y{y}.npz') as d:
                            sample_cache[cache_key]=(d['q'].copy(),d['v'].copy())
                    qs.append(sample_cache[cache_key][0][s]);vs.append(sample_cache[cache_key][1][s])
                q=torch.as_tensor(np.asarray(qs),device=device);v=torch.as_tensor(np.asarray(vs),device=device)
                B=len(k);targets=np.full((B,2),-1,int);targets[:,hand]=k[:,2]
                sources=np.full_like(targets,-1);sources[:,hand]=k[:,1]
                with torch.no_grad():
                    plan=generator.predict(q,v,torch.as_tensor(targets[:,None],device=device),
                        q.new_full((B,1),N*.02),q.new_full((B,1,2),2.),
                        torch.ones(B,1,device=device,dtype=torch.bool),0,return_plan=True,waist_yaw=q[:,0])
                    traj=plan['traj'][:,:N].clone()
                    idle=arm_joint_indices(('left','right')[1-hand]);traj[:,:,idle]=q[:,None,idle]
                    lengths=torch.full_like(plan['lengths'],N)
                    ctx=strike_context(kit,targets,sources,np.full(B,N),device,speeds=np.full((B,2),2.))
                    ctx.target_top_entry_depth_m=.07
                before=profile(short_cost,q,v,traj,ctx,lengths,allowance=allowance)
                part=dict(keys=k,N=np.full(B,N),**{'before_'+name:val for name,val in before.items()})
                if args.stage=='qp':
                    repaired,_,repair_metrics,info=refine(short_cost,q,v,traj,ctx,lengths,hand,depth_m=.07,cfg=qp_cfg)
                    after=profile(short_cost,q,v,repaired,ctx,lengths,allowance=allowance)
                    delta=(repaired-traj).abs()
                    part.update(**{'after_'+name:val for name,val in after.items()},
                        accepted_steps=repair_metrics['accepted_steps'],
                        max_correction_rad=delta.amax((1,2)).cpu().numpy(),
                        rms_correction_rad=delta.square().mean((1,2)).sqrt().cpu().numpy())
                    save_npz(folder/f'paths_{N:03d}_h{hand}_{start:04d}.npz',keys=k,
                        q=q.cpu().numpy(),v=v.cpu().numpy(),before=traj.cpu().numpy(),after=repaired.cpu().numpy())
                    write_json(folder/f'qp_info_{N:03d}_h{hand}_{start:04d}.json',info)
                parts.append(part);processed+=B
                write_json(folder/'progress.json',dict(processed=processed,N=N,elapsed_s=time.monotonic()-begin))
                print(json.dumps(dict(stage=args.stage,N=N,processed=processed,
                    small=int(before['relaxed_success'].sum()))),flush=True)
        save_npz(folder/f'deadline_{N:03d}.npz',**{name:np.concatenate([p[name] for p in parts]) for name in parts[0]})
    write_json(folder/'done.json',dict(status='complete',processed=processed,elapsed_s=time.monotonic()-begin))


def collect(out,stage):
    parts=[]
    for p in sorted((out/stage).glob('w*/deadline_*.npz')):
        with np.load(p) as d:parts.append({k:d[k].copy() for k in d.files})
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def publish(args,audit,repairs):
    table=TransitionTable.load(args.full/'database/transition_table_decoder.npz')
    with np.load(args.full/'database/per_start_minima.npz') as d:
        strict=d['decoder_minimum_s'].copy();eligible=d['eligible'].copy()
    tolerant=strict.copy();repaired=strict.copy()
    if audit:
        for k,N,ok in zip(audit['keys'],audit['N'],audit['before_relaxed_success']):
            if ok:tolerant[tuple(k)]=min(tolerant[tuple(k)],N*.02)
    if repairs:
        for k,N,ok in zip(repairs['keys'],repairs['N'],repairs['after_strict_success']):
            if ok:repaired[tuple(k)]=min(repaired[tuple(k)],N*.02)
    stats=[minimum_statistics(a[...,None],np.isfinite(a[...,None])) for a in (strict,tolerant,repaired)]
    summary=dict(status='complete',allowed_groups=int(table.allowed_groups().sum()),
        eligible_starts=int(eligible.sum()),allowance=asdict(SmallViolationAllowance()),
        near_candidate_trials=len(audit.get('keys',[])),
        small_violation_trials=int(audit['before_relaxed_success'].sum()) if audit else 0,
        qp_trials=len(repairs.get('keys',[])),
        qp_strictly_repaired=int(repairs['after_strict_success'].sum()) if repairs else 0,
        qp_strictly_repaired_with_small_correction=int((repairs['after_strict_success']&(repairs['max_correction_rad']<=.05)).sum()) if repairs else 0,
        qp_remaining_within_allowance=int(repairs['after_relaxed_success'].sum()) if repairs else 0,
        qp_failed_strict_gate=int((~repairs['after_strict_success']).sum()) if repairs else 0,
        failure_interpretation='Failure within this local QP budget is not proof of physical infeasibility',
        duration_interpretation='Sampled duration, worst actuator measured separately; no thermal or tracking certification')
    out=args.out/'database';out.mkdir(exist_ok=True)
    for name,stat in zip(('strict','tolerant','qp'),stats):
        summary[name]=dict(supported_starts=int(stat['count'].sum()),supported_groups=int((stat['count']>0).sum()))
    for name,stat in zip(('strict','tolerant','qp'),stats):
        metadata=dict(table.metadata,small_violation_allowance=summary['allowance'],repair_summary=summary,
            qp={'iterations':5 if name=='qp' else 0,'trust_rad':.03 if name=='qp' else None},
            path_generation='Decoder with local QP repairs' if name=='qp' else 'Decoder only',
            acceptance='Small brief violations allowed' if name=='tolerant' else 'Strict calibration constraints',
            torque_duration_check='Per joint trapezoidal sampled time, 20 ms knots',
            runtime_note='Repaired minima require the stored QP correction; allowed torque excess needs tracking validation')
        current=replace(table,minimum_s=stat['mean'],support=stat['count'],trials=eligible.sum(-1),metadata=metadata)
        current.save(out/f'transition_table_{name}.npz')
    save_npz(out/'per_start_minima.npz',strict_minimum_s=strict,tolerant_minimum_s=tolerant,
        qp_minimum_s=repaired,eligible=eligible)
    with (out/'transition_table.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['hand','source','destination','yaw_region','strict_mean_min_s',
            'tolerant_mean_min_s','qp_strict_mean_min_s','strict_starts','tolerant_starts','qp_starts','eligible_starts'])
        finite=lambda v:float(v) if np.isfinite(v) else ''
        for k in np.ndindex(table.minimum_s.shape):
            if not table.allowed_groups()[k]:continue
            h,src,dst,y=k
            w.writerow([('left','right')[h],table.drum_names[src],table.drum_names[dst],y+1,
                *[finite(s['mean'][k]) for s in stats],*[s['count'][k] for s in stats],eligible[k].sum()])
    write_json(out/'summary.json',summary)
    return summary


def parent(args):
    args.out.mkdir(parents=True,exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(args.out/'tb'))
    registry=ROOT/'runs/tensorboard_active'/args.out.name
    if not registry.is_symlink():registry.symlink_to((args.out/'tb').resolve())
    write_json(args.out/'config.json',dict(full_grid=str(args.full),allowance=asdict(SmallViolationAllowance()),
        qp=asdict(QPConfig(iterations=5,trust_rad=.03)),devices=args.devices,
        selection='Earliest allowance-passing failed plan per start, only when shorter than its raw strict minimum',
        training=False,constraint_scope='Moving arm and shared waist'))
    begin=time.monotonic()
    for stage in ('audit','qp'):
        write_json(args.out/'run.json',dict(status='running',stage=stage))
        workers=[]
        try:
            for rank,device in enumerate(args.devices):
                log=(args.out/f'{stage}_w{rank}.log').open('w')
                cmd=[sys.executable,str(Path(__file__).resolve()),'--full',str(args.full),'--out',str(args.out),
                    '--device',device,'--rank',str(rank),'--workers',str(len(args.devices)),'--stage',stage]
                workers.append((subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')),log))
            write_json(args.out/'run.json',dict(status='running',stage=stage,worker_pids=[p.pid for p,_ in workers]))
            while any(p.poll() is None for p,_ in workers):
                if any(p.poll() not in (None,0) for p,_ in workers):raise RuntimeError('Repair worker failed; inspect logs')
                progress=[json.loads(p.read_text()) for p in (args.out/stage).glob('w*/progress.json')]
                writer.add_scalar(f'{stage}/processed',sum(p['processed'] for p in progress),int(time.monotonic()-begin));writer.flush()
                time.sleep(3)
            if any(p.returncode for p,_ in workers):raise RuntimeError('Repair worker failed')
        finally:
            for process,log in workers:
                if process.poll() is None:process.terminate()
                log.close()
        if stage=='audit':
            audit=collect(args.out,'audit');save_npz(args.out/'violation_audit.npz',**audit)
            with np.load(args.full/'database/per_start_minima.npz') as d:strict=d['decoder_minimum_s'].copy()
            selected={}
            if audit:
                for i,(k,N,ok) in enumerate(zip(audit['keys'],audit['N'],audit['before_relaxed_success'])):
                    k=tuple(k)
                    if ok and N*.02<strict[k] and (k not in selected or N<audit['N'][selected[k]]):selected[k]=i
            take=np.asarray(list(selected.values()),int)
            save_npz(args.out/'repair_candidates.npz',keys=audit['keys'][take] if audit else np.empty((0,5),int),
                N=audit['N'][take] if audit else np.empty(0,int))
            write_json(args.out/'audit_summary.json',dict(candidate_trials=len(audit.get('keys',[])),
                allowance_passing_trials=int(audit['before_relaxed_success'].sum()) if audit else 0,
                selected_qp_trials=len(take)))
    repairs=collect(args.out,'qp');save_npz(args.out/'repair_results.npz',**repairs)
    summary=publish(args,audit,repairs);summary['elapsed_s']=time.monotonic()-begin
    write_json(args.out/'summary.json',summary);write_json(args.out/'run.json',summary)
    writer.close();print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--devices',nargs='+');p.add_argument('--device',default='cuda:0')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--stage',choices=('audit','qp'),default='audit');p.add_argument('--chunk',type=int,default=64)
    args=p.parse_args();args.full=args.full.resolve();args.out=args.out.resolve()
    parent(args) if args.devices else worker(args)


if __name__=='__main__':main()
