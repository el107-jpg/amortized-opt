"""Save reproducible repairs of previously unsupported unrestricted transitions."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.ik import arm_joint_indices
from amortized_opt.train_cost import dls_terminal_teacher
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import strike_context
from isaac_rl.trajectory_qp import QPConfig, cost_for_horizon, metrics, refine
from scripts.probe_transition_qp import save_npz, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=ROOT/'runs/transition_qp_unrestricted_20261005')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--skip-qp', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = A.build_model(A.model_config_from_checkpoint(ck), args.device)
    model.load_state_dict(ck['model']); model.eval()
    cfg = A.CostConfig(**ck['train_cfg']['cost'])
    cfg.n_sub=4; cfg.physics_mode='spline'; cfg.w_velocity=max(1.,cfg.w_velocity)
    cfg.w_joint_box=max(1.,cfg.w_joint_box)
    cost = A.FusedCost(cfg,max_horizon=100,device=args.device,rnea=load_cached(device=args.device))
    cost.calibration_active_hand_only=True
    kit=A.DrumKit.from_layout(); names=list(kit.names)
    generator=DecoderHitPoseGenerator(model,cost,kit,horizon=100,
        waist_upright_ease_wp=(ck.get('scenario_cfg')or{}).get('waist_upright_ease_wp'))
    with np.load(args.source_run/'transition_table.npz') as d:
        unresolved=d['allowed'] & ~np.isfinite(d['minimum_s'])
    prior=ROOT/'runs/hihat_fixes_probe_20261005/verified_witnesses.csv'
    for row in csv.DictReader(prior.open()):
        unresolved[('left','right').index(row['hand']),names.index(row['source']),names.index('hihat')]=False
    checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    write_json(args.out/'config.json',dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checksum,
        source_run=str(args.source_run.resolve()),n_sub=32,physics_mode='spline',hit_depth_m=.07,
        impact_speed_required=False,active_arm_and_waist_only=True,training=False,
        deadlines_s=[.2,.4,.6,.8,1.,1.5,2.],limitation='Tested witnesses, not certified minima'))
    costs={}; records={}; repairs=[]; begin=time.monotonic()
    def c(N,dense=False):
        if (N,dense) not in costs:
            obj=cost_for_horizon(cost,N);obj.cfg=copy.copy(cost.cfg);obj.cfg.n_sub=32 if dense else 4
            costs[N,dense]=obj
        return costs[N,dense]
    def context(h,src,dst,B,N):
        target=np.full((B,2),-1);target[:,h]=dst
        source=np.full_like(target,-1);source[:,h]=src
        ctx=strike_context(kit,target,source,[N]*B,args.device,speeds=np.full((B,2),2.))
        ctx.target_top_entry_depth_m=.07
        L=torch.full((B,3),N,device=args.device,dtype=torch.long)
        return target,ctx,L
    def save(h,src,dst,sample,N,q,v,traj,m,method):
        key=(h,src,dst,sample)
        if key in records and records[key]['duration_s']<=N*.02: return
        filename=f'h{h}_{names[src]}_to_{names[dst]}_s{sample}.npz'
        save_npz(args.out/filename,q=q.cpu().numpy(),v=v.cpu().numpy(),trajectory=traj.detach().cpu().numpy(),
                 duration_s=np.asarray(N*.02))
        records[key]=dict(hand=('left','right')[h],source=names[src],destination=names[dst],sample=int(sample),
            duration_s=N*.02,method=method,artifact=filename,metrics={k:float(x[0]) for k,x in m.items()})
    for h,src,dst in zip(*np.nonzero(unresolved)):
        path=next(args.source_run.glob(f'w*/h{h}_source_{src}.npz'))
        with np.load(path) as d:
            take=d['destination']==dst
            q=torch.tensor(d['q'][take],device=args.device);v=torch.tensor(d['v'][take],device=args.device)
            samples=d['sample'][take]
        for N in (10,20,30,40,50,75,100):
            target,ctx,L=context(h,src,dst,len(q),N)
            with torch.no_grad():
                for yaw in (q[:,0],None):
                    plan=generator.predict(q,v,torch.tensor(target[:,None],device=args.device),q.new_full((len(q),1),N*.02),
                        q.new_full((len(q),1,2),2.),torch.ones(len(q),1,device=args.device,dtype=torch.bool),0,
                        return_plan=True,waist_yaw=yaw)
                    traj=plan['traj'][:,:N].clone()
                    idle=arm_joint_indices(('left','right')[1-h]);traj[:,:,idle]=q[:,None,idle]
                    coarse=metrics(c(N),q,v,traj,ctx,L,.07,signed_contact=True)
                    for i in np.flatnonzero(coarse['success']):
                        _,one,oneL=context(h,src,dst,1,N)
                        dense=metrics(c(N,True),q[i:i+1],v[i:i+1],traj[i:i+1],one,oneL,.07,signed_contact=True)
                        if dense['success'][0]: save(h,src,dst,samples[i],N,q[i:i+1],v[i:i+1],traj[i:i+1],dense,'decoder')
        print(json.dumps(dict(stage='decoder_audit',hand=int(h),source=names[src],destination=names[dst],
                             witnesses=len(records),elapsed_s=time.monotonic()-begin)),flush=True)
    if not args.skip_qp:
        for hand,source,destination,sample in [('right','hihat','cymbal_right',0),('right','tom2','floor_tom',1),('left','hihat','tom1',0)]:
            h=('left','right').index(hand);src=names.index(source);dst=names.index(destination);N=50
            with np.load(next(args.source_run.glob(f'w*/h{h}_source_{src}.npz'))) as d:
                i=np.flatnonzero((d['destination']==dst)&(d['sample']==sample))[0]
                q=torch.tensor(d['q'][i:i+1],device=args.device);v=torch.tensor(d['v'][i:i+1],device=args.device)
            target,ctx,L=context(h,src,dst,1,N);candidates=[]
            with torch.no_grad():
                for yaw in (q[:,0],None):
                    plan=generator.predict(q,v,torch.tensor(target[:,None],device=args.device),q.new_full((1,1),1.),
                        q.new_full((1,1,2),2.),torch.ones(1,1,device=args.device,dtype=torch.bool),0,return_plan=True,waist_yaw=yaw)
                    traj=plan['traj'][:,:N].clone();idle=arm_joint_indices(('left','right')[1-h]);traj[:,:,idle]=q[:,None,idle]
                    m=metrics(c(N),q,v,traj,ctx,L,.07,signed_contact=True);candidates.append((float(m['merit'][0]),traj))
            traj=min(candidates,key=lambda x:x[0])[1]
            seeds=[('decoder',traj)]
            for region in (False,True):
                teacher,_,_=dls_terminal_teacher(traj,ctx,c(N),steps=4,window=32,region_target=region)
                seeds.append(('region_teacher' if region else 'fixed_teacher',teacher))
            seed_kind,seed=min(seeds,key=lambda x:metrics(c(N,True),q,v,x[1],ctx,L,.07,signed_contact=True)['merit'][0])
            qp=QPConfig(iterations=15,trust_rad=.01,torque_time_bins=8,jacobian_chunk=32,signed_contact=True,
                        line_search=tuple(2.**(-k) for k in range(12)))
            fixed,_,_,info=refine(c(N),q,v,seed,ctx,L,h,depth_m=.07,cfg=qp)
            m=metrics(c(N,True),q,v,fixed,ctx,L,.07,signed_contact=True)
            if not m['success'][0]:
                from dataclasses import replace
                fixed,_,m,dense_info=refine(c(N,True),q,v,fixed,ctx,L,h,depth_m=.07,
                                         cfg=replace(qp,iterations=5,trust_rad=.003))
                info['dense']=dense_info
            save_npz(args.out/f'repair_h{h}_{source}_to_{destination}.npz',q=q.cpu().numpy(),v=v.cpu().numpy(),
                     before=seed.detach().cpu().numpy(),after=fixed.detach().cpu().numpy())
            write_json(args.out/f'repair_h{h}_{source}_to_{destination}.json',info)
            result=dict(hand=hand,source=source,destination=destination,sample=sample,seed=seed_kind,
                        metrics={k:float(x[0]) for k,x in m.items()})
            repairs.append(result); print(json.dumps(result),flush=True)
            if m['success'][0]:save(h,src,dst,sample,N,q,v,fixed,m,'DLS+QP')
    rows=list(records.values())
    write_json(args.out/'witnesses.json',rows)
    summary=dict(status='complete',new_supported_pairs=len({k[:3] for k in records}),
        supported_starts=len(records),tested_pairs=int(unresolved.sum()),repairs=repairs,
        elapsed_s=time.monotonic()-begin,checkpoint_sha256=checksum)
    write_json(args.out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
