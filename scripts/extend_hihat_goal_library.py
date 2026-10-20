#!/usr/bin/env python3
"""Solve legal near-edge hi-hat poses for both hands, preserving other entries."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.constants import DRUM_GOAL_POSES_PATH, G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.ik import correct_tip, tip_position, arm_joint_indices
from amortized_opt.rhythm import sample_hit_point
from amortized_opt.strike import default_contact_sigma
from amortized_opt.warmstart import GoalPoseLibrary
from isaac_rl.rnea_cache import load_cached
from isaac_rl.transition_calibration import static_clearance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:1')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--install',action='store_true')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    path=Path(DRUM_GOAL_POSES_PATH);raw=json.loads(path.read_text());library=GoalPoseLibrary()
    kit=A.DrumKit.from_layout();hat=kit.names.index('hihat');sigma=default_contact_sigma('hihat')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=A.CostConfig(**ck['train_cfg']['cost'])
    cost=A.FusedCost(cfg,max_horizon=2,device=args.device,rnea=load_cached(device=args.device))
    cost.calibration_active_hand_only=True;_,lo,hi=cost._joint_box_hard_limits();lo=lo.reshape(-1);hi=hi.reshape(-1)
    bounds=(lo.cpu().numpy(),hi.cpu().numpy());rng=np.random.default_rng(2026100521)
    report=[]
    for hand in ['left','right']:
        columns=arm_joint_indices(hand);yaws=np.linspace(float(lo[0])+.01,float(hi[0])-.01,9)
        starts=[];values=[]
        for yaw in yaws:
            drums=[name for name in ['hihat','cymbal_left','snare','tom1','tom2'] if library.has(hand,name)]
            for trial in range(32):
                q=ready_joint_pos().copy();q[0]=yaw
                q[columns]=library.select(hand,drums[trial%len(drums)],yaw).arm_q
                if trial>=len(drums):q[columns]+=rng.normal(0,.35,len(columns))
                starts.append(q);values.append(yaw)
        q=torch.tensor(np.asarray(starts),device=args.device,dtype=torch.float32).clamp(lo,hi)
        target=q.new_tensor(sample_hit_point(kit,hat)).expand(len(q),-1)
        q=correct_tip(cost.fk,q,target,hand,steps=45,damping=.025,sigma=sigma,q_bounds=bounds)
        q=correct_tip(cost.fk,q,target,hand,steps=20,damping=.008,sigma=sigma,q_bounds=bounds)
        error=(tip_position(cost.fk,q,hand,sigma=sigma,frame='env')-target).norm(dim=-1)
        sources=np.full((len(q),2),-1);sources[:,0 if hand=='left' else 1]=hat
        clearance=static_clearance(cost,kit,q,sources,ignore_idle_arm=True)
        valid=(error<=.003)&(clearance>=-1e-6)&torch.isfinite(q).all(-1)
        rows=[]
        for yaw in yaws:
            ids=np.flatnonzero(np.isclose(values,yaw)&valid.cpu().numpy())
            if not len(ids):
                report.append(dict(hand=hand,yaw_rad=float(yaw),status='unresolved'));continue
            # Prefer a pose away from wrist limits and close to the ready arm.
            wrist=[j for j in columns if 'wrist' in G1_UPPER_BODY_JOINTS[j]]
            headroom=torch.minimum(q[ids][:,wrist]-lo[wrist],hi[wrist]-q[ids][:,wrist]).amin(-1)
            travel=(q[ids][:,columns]-q.new_tensor(ready_joint_pos())[columns]).square().sum(-1)
            chosen=int(ids[int(torch.argmin(travel-.5*headroom))]);pose=q[chosen].cpu().numpy()
            row=dict(waist_yaw=float(yaw),q={name:float(pose[j]) for j,name in enumerate(G1_UPPER_BODY_JOINTS)},
                tip_err=float(error[chosen]),contact_err_m=float(error[chosen]),contact_sigma=float(sigma),
                wrist_headroom=float(torch.minimum(q[chosen,wrist]-lo[wrist],hi[wrist]-q[chosen,wrist]).amin()),
                active_clearance_m=float(clearance[chosen]))
            rows.append(row);report.append(dict(hand=hand,yaw_rad=float(yaw),status='valid',
                contact_error_mm=float(error[chosen])*1000,clearance_mm=float(clearance[chosen])*1000))
        if not rows:raise RuntimeError(f'No verified hi-hat pose for {hand}; library not installed')
        raw.setdefault('yaw_grid',{}).setdefault(hand,{})['hihat']=rows
        raw.setdefault('poses',{}).setdefault(hand,{})['hihat']=min(rows,key=lambda row:abs(row['waist_yaw']))['q']
    raw['hihat_update']=dict(aim_inset_m=.02,contact_sigma=sigma,seed=2026100521,
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        validation='<=3 mm contact error, existing joint box, active-arm static clearance; idle arm ignored')
    target_path=out/'drum_goal_poses.json';target_path.write_text(json.dumps(raw,indent=2,sort_keys=True)+'\n')
    summary=dict(poses=report,counts={h:sum(r['status']=='valid' and r['hand']==h for r in report) for h in ['left','right']},
        candidate=str(target_path),installed=False)
    if args.install:
        if not (out/'drum_goal_poses_before.json').exists():
            shutil.copy2(path,out/'drum_goal_poses_before.json')
        temporary=path.with_suffix('.hihat.tmp.json');shutil.copy2(target_path,temporary);temporary.replace(path)
        summary['installed']=True
    (out/'library_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
