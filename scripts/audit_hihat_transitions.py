#!/usr/bin/env python3
"""Reproduce hi-hat decoder failures without changing the completed table."""
from pathlib import Path
import argparse
import copy
import json
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
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import cost_for_horizon, metrics, captured_constraints


def checks(m, *, elevation_required=True):
    return dict(region=m['terminal_region_dist'] <= 1e-6,
        collision=m['collision_min_sdf'] >= -1e-6, torque=m['torque_excess_max'] <= 1e-6,
        velocity=m['velocity_excess_max'] <= 1e-6, joints=m['joint_box_excess_max'] <= 1e-6,
        direction=m['contact_dir_cos'] >= .5,
        elevation=((m['stick_elev_deg'] >= 5)&(m['stick_elev_deg'] <= 45)
                   if elevation_required else np.ones_like(m['success'], dtype=bool)))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--hands',nargs='+',type=int,choices=[0,1],default=[0,1])
    args=parser.parse_args();run=args.run.resolve();out=args.out.resolve() if args.out else run/'hihat_diagnostic';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((run/'config.json').read_text());torch.set_num_threads(2)
    checkpoint=args.checkpoint or Path(meta['checkpoint'])
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=A.build_model(A.model_config_from_checkpoint(ck),args.device);model.load_state_dict(ck['model']);model.eval()
    cfg=A.CostConfig(**ck['train_cfg']['cost']);cfg.n_sub=4;cfg.physics_mode='spline'
    cfg.w_velocity=max(cfg.w_velocity,1.);cfg.w_joint_box=max(cfg.w_joint_box,1.)
    cost=A.FusedCost(cfg,max_horizon=100,device=args.device,rnea=load_cached(device=args.device))
    cost.calibration_active_hand_only=True;kit=A.DrumKit.from_layout();hat=list(kit.names).index('hihat')
    generator=DecoderHitPoseGenerator(model,cost,kit,horizon=100,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    results=[];begin=time.monotonic()
    for hand in args.hands:
        qs=[];vs=[];sources=[];samples=[]
        for p in sorted(run.glob(f'w*/h{hand}_source_*.npz')):
            with np.load(p) as d:
                take=d['destination']==hat;qs.append(d['q'][take]);vs.append(d['v'][take])
                sources.extend([int(d['source'])]*int(take.sum()));samples.extend(d['sample'][take].tolist())
        q=torch.tensor(np.concatenate(qs),device=args.device);v=torch.tensor(np.concatenate(vs),device=args.device)
        B=len(q);targets=np.full((B,2),-1);targets[:,hand]=hat
        start=np.full_like(targets,-1);start[:,hand]=sources
        best_merit=np.full(B,np.inf);best_records=[None]*B;best_paths=[None]*B
        old_pass=np.zeros(B,bool);relaxed_pass=np.zeros(B,bool);costs={}
        dense_pass=np.zeros(B,bool);dense_min=np.full(B,np.inf);dense_paths=[None]*B
        def c(N,dense=False):
            if (N,dense) not in costs:
                obj=cost_for_horizon(cost,N);obj.cfg=copy.copy(cost.cfg);obj.cfg.n_sub=32 if dense else 4
                costs[N,dense]=obj
            return costs[N,dense]
        for N in range(1,101):
            ctx=strike_context(kit,targets,start,np.full(B,N),args.device,speeds=np.full((B,2),2.))
            ctx.target_top_entry_depth_m=.07;L=torch.full((B,3),N,device=args.device,dtype=torch.long)
            for aim in ('current','goal'):
                plan=generator.predict(q,v,torch.tensor(targets[:,None],device=args.device),q.new_full((B,1),N*.02),
                    q.new_full((B,1,2),2.),torch.ones(B,1,device=args.device,dtype=torch.bool),0,
                    return_plan=True,waist_yaw=q[:,0] if aim=='current' else None)
                traj=plan['traj'][:,:N].clone();idle=arm_joint_indices(('left','right')[1-hand]);traj[:,:,idle]=q[:,None,idle]
                m=metrics(c(N),q,v,traj,ctx,L,.07,signed_contact=True);flags=checks(m)
                all_other=np.logical_and.reduce([val for key,val in flags.items() if key!='elevation'])
                old_pass|=np.logical_and.reduce(list(flags.values()));relaxed_pass|=all_other
                verify=np.flatnonzero(all_other & ~dense_pass)
                if len(verify):
                    dctx=strike_context(kit,targets[verify],start[verify],np.full(len(verify),N),args.device,
                        speeds=np.full((len(verify),2),2.))
                    dctx.target_top_entry_depth_m=.07
                    dm=metrics(c(N,True),q[verify],v[verify],traj[verify],dctx,L[verify],.07,signed_contact=True)
                    for i in verify[dm['success']]:
                        dense_pass[i]=True;dense_min[i]=N*.02;dense_paths[i]=traj[i].cpu().clone()
                merit=m['merit']
                for i in np.flatnonzero(merit<best_merit):
                    best_merit[i]=merit[i];best_paths[i]=traj[i].cpu().clone()
                    best_records[i]=dict(hand=('left','right')[hand],source=kit.names[sources[i]],sample=int(samples[i]),
                        duration_s=N*.02,seed=aim,merit_no_elevation=float(merit[i]),
                        metrics={k:float(val[i]) for k,val in m.items()},
                        failures=[k for k,val in flags.items() if not val[i]])
            if N%20==0:print(json.dumps(dict(hand=hand,N=N,seconds=time.monotonic()-begin)),flush=True)
        # Verify each start's closest candidate on the original dense grid;
        # expose per-obstacle collision constraints as well as headline metrics.
        for i,record in enumerate(best_records):
            N=len(best_paths[i]);ctx=strike_context(kit,targets[i:i+1],start[i:i+1],[N],args.device,speeds=np.full((1,2),2.))
            ctx.target_top_entry_depth_m=.07;L=torch.full((1,3),N,device=args.device,dtype=torch.long)
            traj=best_paths[i][None].to(args.device)
            m=metrics(c(N,True),q[i:i+1],v[i:i+1],traj,ctx,L,.07,signed_contact=True)
            flags=checks(m,elevation_required=bool(ctx.elev_required[0,hand]))
            g,terms,names=captured_constraints(c(N,True),q[i:i+1],v[i:i+1],traj,ctx,L,.07,signed_contact=True)
            collisions=[]
            for k,name in enumerate(names):
                if name.startswith('collision') and float(g[0,k])>0:
                    collisions.append(dict(constraint=name,penetration_mm=float(g[0,k])*10))
            record.update(dense_metrics={k:float(val[0]) for k,val in m.items()},
                dense_failures=[k for k,val in flags.items() if not val[0]],
                positive_collision_constraints=collisions,any_old_pass=bool(old_pass[i]),
                any_no_elevation_pass=bool(relaxed_pass[i]),elevation_required=bool(ctx.elev_required[0,hand]))
            record.update(any_dense_pass=bool(dense_pass[i]),
                first_dense_verified_duration_s=float(dense_min[i]) if dense_pass[i] else None)
        np.savez_compressed(out/f'closest_h{hand}.npz',q=q.cpu().numpy(),v=v.cpu().numpy(),
            sources=sources,samples=samples,lengths=[len(p) for p in best_paths],
            paths=np.stack([np.pad(p.numpy(),((0,100-len(p)),(0,0))) for p in best_paths]))
        supported=np.flatnonzero(dense_pass)
        if len(supported):
            np.savez_compressed(out/f'dense_h{hand}.npz',q=q[supported].cpu().numpy(),v=v[supported].cpu().numpy(),
                sources=np.asarray(sources)[supported],samples=np.asarray(samples)[supported],
                lengths=np.rint(dense_min[supported]/.02).astype(int),
                paths=np.stack([np.pad(dense_paths[i].numpy(),((0,100-len(dense_paths[i])),(0,0))) for i in supported]))
        results.extend(best_records)
        (out/'cases.json').write_text(json.dumps(results,indent=2))
    summary=dict(status='complete',seconds=time.monotonic()-begin,total_starts=len(results),
        checkpoint=str(checkpoint),checkpoint_step=ck.get('step'),
        optimization_substeps=4, verification_substeps=32,
        coarse_legacy_supported_starts=sum(r['any_old_pass'] for r in results),
        coarse_no_elevation_supported_starts=sum(r['any_no_elevation_pass'] for r in results),
        dense_verified_supported_starts=sum(r['any_dense_pass'] for r in results),
        dense_verified_pairs=sorted({r['hand']+':'+r['source']+'->hihat' for r in results if r['any_dense_pass']}),
        dense_closest_failure_counts={key:sum(key in r['dense_failures'] for r in results)
            for key in ['region','collision','torque','velocity','joints','direction','elevation']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
