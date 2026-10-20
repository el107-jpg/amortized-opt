#!/usr/bin/env python3
"""Paired graph, causal Q-policy, and bounded-search comparison in the pose surrogate."""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos,STICK_TIP_SIGMA
from amortized_opt.fk import stick_contact_offset
from amortized_opt.trajectory_surrogate import SingleHitGenerator,TrajectorySurrogate,rebound_velocity
from isaac_rl.arm_assignment import AssignmentQueue,AssignmentConfig
from isaac_rl.assignment_dqn import AssignmentQPolicy
from isaac_rl.assignment_search import pack_candidates,direct_q_targets,assignment_observations
from isaac_rl.assignment_starts import StartingPoseSampler,pose_clearance
from isaac_rl.assignment_train import window_clip
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from scripts.train_arm_assignment_model import score_candidates,tempo_span


def graph_actions(clip,kit,cfg):
    queue=AssignmentQueue(clip,kit,cfg); result=[]
    while queue.next_group<len(queue.groups):
        action=queue.teacher_action(); result.append(action); queue.commit(action)
    return result


def tips(fk,q):
    points=[]
    for h in ('left','right'):
        R,p=fk.frame(q,h+'_wrist')
        points.append(p+(R @ q.new_tensor(stick_contact_offset(h,STICK_TIP_SIGMA))[:,None]).squeeze(-1))
    return torch.stack(points,1).cpu().numpy()


@torch.no_grad()
def execute(method,clips,kit,cfg,packed,generator,env,policy):
    # The first candidate contains all paired physical-hit bounce draws. Map
    # them back by physical event ordinal when each policy selects a hand.
    R=packed['rebound_samples']; indexes=[]; owners=[]
    for owner in range(len(clips)):
        ids=np.flatnonzero(packed['owner']==owner)[:R]
        indexes.extend(ids); owners.extend([owner]*R)
    idx=torch.as_tensor(indexes,device=packed['q0'].device)
    p={k:v[idx].clone() for k,v in packed.items() if torch.is_tensor(v)}
    B,G=p['valid'].shape
    p['drums'].fill_(-1); p['speeds'].zero_(); p['efficiency'].fill_(1)
    queues=[AssignmentQueue(clips[o],kit,replace(cfg,lookahead_s=max(cfg.lookahead_s,clips[o].hits[-1].time+.01)))
            for o in owners]
    q,v=p['q0'].clone(),p['v0'].clone()
    poses=[]; velocities=[]; actions=np.full((B,G),-1,int)
    for g in range(G):
        qcpu,vcpu=q.cpu().numpy(),v.cpu().numpy(); tip=tips(env.cost.fk,q)
        live=[b for b,queue in enumerate(queues) if g<len(queue.groups)]
        if method=='q':
            obs=np.stack([queues[b].observation(0 if g==0 else float(p['times'][b,g-1]),qcpu[b],vcpu[b],tip[b]) for b in live])
            choice=policy(torch.as_tensor(obs,device=q.device)).argmax(1).cpu().numpy()
        else:
            choice=[queues[b].teacher_action() for b in live]
        for j,b in enumerate(live):
            action=int(choice[j]); actions[b,g]=action
            group=queues[b].commit(action)
            reference=indexes[b]; old_action=packed['sequences'][reference][g]
            for event,(_i,hit) in enumerate(group):
                hand=(action+event)%2; old_hand=(old_action+event)%2
                p['drums'][b,g,hand]=hit.drum
                p['speeds'][b,g,hand]=1+2*hit.velocity
                p['efficiency'][b,g,hand]=packed['efficiency'][reference,g,old_hand]
        pose,pre=generator.predict(q,v,p['drums'],p['times'],p['speeds'],p['valid'],g)
        gap,J=env.contact(pose[:,None],p['drums'][:,g:g+1])
        speed=torch.einsum('bhd,bd->bh',J[:,0],pre)
        hit=(p['drums'][:,g]>=0)&p['valid'][:,g,None]&(gap[:,0]<=1e-5)&(speed<-.1)
        q=pose; v=rebound_velocity(pre,J[:,0],env.inverse_mass,p['efficiency'][:,g],hit)
        poses.append(pose); velocities.append(pre)
    poses,velocities=torch.stack(poses,1),torch.stack(velocities,1)
    out=env.evaluate(p['q0'],p['v0'],poses,torch.zeros_like(velocities),p['drums'],p['times'],
                     p['speeds'],p['valid'],p['efficiency'],pre_velocity_override=velocities)
    return {k:v.cpu() if torch.is_tensor(v) else v for k,v in out.items()},actions


def summarize(result,indexes=None):
    if indexes is None: indexes=np.arange(len(result['score']))
    r={k:v[indexes] for k,v in result.items() if torch.is_tensor(v)}
    feasible=(r['torque_excess']<=.01)&(r['velocity_excess']<=.01)&(r['collision_m']<=.005)&(r['joint_excess_rad']<=1e-4)
    return dict(hit_fraction=float(r['hits'].sum()/r['requested'].sum()),
        hits=int(r['hits'].sum()),requested=int(r['requested'].sum()),
        feasible_trial_fraction=float(feasible.float().mean()),feasible_trials=int(feasible.sum()),trials=len(indexes),
        score_per_hit=float((r['score']/r['requested']).mean()),
        mean_peak_torque_ratio=float(r['torque_excess'].mean()+1),
        mean_peak_collision_m=float(r['collision_m'].mean()))


@torch.no_grad()
def compare(clips,poses,rng,kit,cfg,generator,env,policy):
    packed=pack_candidates(clips,kit,cfg,poses,rng,rebound_samples=2,device=env.device)
    searched=score_candidates(env,generator,packed,2048)
    target,best=direct_q_targets(searched['score'],packed,len(clips),searched['requested'])
    observations=assignment_observations(clips,kit,cfg,poses,env.cost.fk,env.device)
    qfirst=policy(observations).argmax(1).cpu().numpy()
    graphfirst=np.array([graph_actions(c,kit,cfg)[0] for c in clips])
    result=dict(windows=len(clips),search=summarize(searched,best),root_comparison={})
    for method,first in [('graph',graphfirst),('q',qfirst)]:
        actual,actions=execute(method,clips,kit,cfg,packed,generator,env,policy)
        assert np.array_equal(actions[::2,0],first), 'root decisions differ between observation paths'
        result[method]=summarize(actual)
        result[method]['assignments']=actions[::2].tolist()
        result['root_comparison'][method]=dict(agreement=float((first==target.argmax(1)).mean()),
            search_regret=float((target.max(1)-target[np.arange(len(clips)),first]).mean()))
    result['root_comparison']['q_graph_agreement']=float((qfirst==graphfirst).mean())
    result['clips']=[c.name for c in clips]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/arm_assignment_pose_search4_20261002')
    parser.add_argument('--fresh-windows',type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(4); device=torch.device('cuda:0')
    run=ROOT/args.run; ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    cfg=AssignmentConfig(**ck['policy_config']); policy=AssignmentQPolicy(cfg).to(device)
    policy.load_state_dict(ck['policy']); policy.eval(); step=ck['step']; config=ck['config']; del ck
    source=torch.load(config['planner_checkpoint'],map_location='cpu',weights_only=False)
    with rnea_from_cache_if_needed():
        model=A.build_model(A.model_config_from_checkpoint(source),device)
        cost=A.FusedCost(A.CostConfig(**source['train_cfg']['cost']),max_horizon=100,device=device)
    model.load_state_dict(source['model']); kit=A.DrumKit.from_layout()
    generator=SingleHitGenerator(model,cost,kit); env=TrajectorySurrogate(cost,kit)
    pool=np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    lo,hi=A.urdf_position_limits(posture_limits=False)
    def clearance(q): return pose_clearance(cost,torch.as_tensor(q,device=device)).cpu().numpy()
    starts=StartingPoseSampler(pool['poses'],ready_joint_pos(),lo,hi,clearance=clearance,names=A.G1_UPPER_BODY_JOINTS)
    paths=sorted((ROOT/'assets/rcc/eval').glob('*.json'))
    fixed_rng=np.random.default_rng(config['seed']+100000); fixed=[]
    for path in paths:
        try:
            clip=window_clip(A.load_rcc(path,kit,on_ineligible='keep'),fixed_rng,span_s=tempo_span(path),first_hit_s=.5,tail_s=0)
            AssignmentQueue(clip,kit,cfg); fixed.append(clip)
            if len(fixed)==config['eval_states']: break
        except ValueError: continue
    q,_=starts.sample(fixed_rng,len(fixed)); q[:max(1,len(fixed)//2)]=ready_joint_pos()
    report=dict(checkpoint_step=step,assumptions='Frozen single-pose MLP, Hermite connectors, perfect tracking, no Isaac',
                q_execution='Causal decisions at each hit transition using measured surrogate q/qd and assignment history',
                fixed=compare(fixed,q,fixed_rng,kit,cfg,generator,env,policy))
    print(json.dumps(dict(checkpoint_step=step,fixed=report['fixed'])),flush=True)
    fresh_rng=np.random.default_rng(900043); fresh=[]; rejected=0
    loaded=[(A.load_rcc(p,kit,on_ineligible='keep'),tempo_span(p)) for p in paths]
    for _ in range(args.fresh_windows*20):
        clip,span=loaded[int(fresh_rng.integers(len(loaded)))]
        try:
            window=window_clip(clip,fresh_rng,span_s=span,first_hit_s=.5,tail_s=0)
            graph_actions(window,kit,cfg)
            fresh.append(window)
            if len(fresh)==args.fresh_windows: break
        except ValueError: rejected+=1
    if len(fresh)!=args.fresh_windows: raise RuntimeError('Cannot fill fresh graph-supported eval windows')
    q,_=starts.sample(fresh_rng,len(fresh)); q[:len(fresh)//2]=ready_joint_pos()
    report['fresh']=compare(fresh,q,fresh_rng,kit,cfg,generator,env,policy)
    report['fresh']['graph_unsupported_windows_excluded']=rejected
    destination=run/f'graph_comparison_step{step}.json'
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(checkpoint_step=step,fresh=report['fresh'],report=str(destination))),flush=True)


if __name__=='__main__': main()
