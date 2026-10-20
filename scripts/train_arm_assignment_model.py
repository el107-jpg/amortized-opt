#!/usr/bin/env python3
"""Distill four-beat, complete model rollout returns into a small arm Q MLP.

Single-hit generator calls, perfect tracking of Hermite pose connectors,
random frictionless bounce. No Isaac stepping and no Bellman bootstrap.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import SingleHitGenerator,DecoderHitPoseGenerator,TrajectorySurrogate
from isaac_rl.arm_assignment import AssignmentConfig,AssignmentQueue
from isaac_rl.assignment_dqn import AssignmentQPolicy
from isaac_rl.assignment_sampling import BalancedWindowSampler
from isaac_rl.assignment_search import pack_candidates,direct_q_targets,assignment_observations
from isaac_rl.assignment_starts import StartingPoseSampler,pose_clearance
from isaac_rl.assignment_train import window_clip
from isaac_rl.rnea_cache import rnea_from_cache_if_needed


def write_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def make_policy(checkpoint,device):
    old_cfg=AssignmentConfig(**checkpoint['policy_config'])
    cfg=replace(old_cfg,future_hits=128,lookahead_beats=4)
    policy=AssignmentQPolicy(cfg).to(device)
    old=checkpoint['policy']
    with torch.no_grad():
        policy.body[0].weight.zero_()
        policy.body[0].weight[:,:old_cfg.obs_dim].copy_(old['body.0.weight'])
        policy.body[0].bias.copy_(old['body.0.bias'])
        for key in ('weight','bias'):
            getattr(policy.body[2],key).copy_(old['body.2.'+key])
        # New measured returns have a different scale; keep the trained body,
        # initialize the value head near zero, and start a fresh optimizer.
        policy.q_head.weight.mul_(.1); policy.q_head.bias.zero_()
    return policy


def tempo_span(path):
    bpm=float(json.loads(Path(path).read_text()).get('metadata',{}).get('original_bpm',120))
    return max(.02,4*60/max(bpm,1)-.5)


@torch.no_grad()
def score_candidates(surrogate,generator,packed,chunk):
    worlds=len(packed['q0']); results=[]; collocations=0
    for start in range(0,worlds,chunk):
        part={k:v[start:start+chunk] for k,v in packed.items() if torch.is_tensor(v)}
        out=surrogate.rollout(generator,part)
        if not torch.isfinite(out['score']).all():
            raise RuntimeError('nonfinite model rollout scores')
        collocations+=out['collocations']
        results.append({k:v.detach().cpu() for k,v in out.items() if torch.is_tensor(v)})
    merged={k:torch.cat([r[k] for r in results]) for k in results[0]}
    merged['collocations']=collocations
    return merged


def metrics(results,packed,targets,best,observations,policy):
    with torch.no_grad():
        predicted=policy(observations).detach().cpu().numpy()
    chosen=predicted.argmax(1)
    root=np.arange(len(chosen))
    oracle=targets.max(1)
    picked=targets[root,chosen]
    graph=[]
    for groups,_,_ in packed['problems']:
        graph.append(len(groups))
    hits=results['hits'][best].sum().item()
    requested=results['requested'][best].sum().item()
    feasible = ((results['torque_excess'][best] <= .01) &
                (results['velocity_excess'][best] <= .01) &
                (results['collision_m'][best] <= .005) &
                (results['joint_excess_rad'][best] <= 1e-4))
    return dict(hit_fraction=hits/max(requested,1), score_per_hit=float(oracle.mean()),
        feasible_path_fraction=float(feasible.float().mean()),
        feasible_hit_fraction=float((results['hits'][best]*feasible).sum())/max(requested,1),
        q_root_accuracy=float((chosen==targets.argmax(1)).mean()),
        q_search_regret=float((oracle-picked).mean()),
        q_absolute_error=float(np.abs(predicted-targets).mean()),
        requested_hits=int(requested),candidate_worlds=len(packed['q0']),
        candidates_per_state=float(len(packed['q0'])/len(targets)/packed['rebound_samples']),
        collocations=results['collocations'], groups_mean=float(np.mean(graph)),
        torque_excess=float(results['torque_excess'][best].mean()),
        velocity_excess=float(results['velocity_excess'][best].mean()),
        collision_peak_m=float(results['collision_m'][best].mean()),
        joint_excess_rad=float(results['joint_excess_rad'][best].mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='arm_assignment_pose_search4_20261002')
    parser.add_argument('--generator',choices=('pose','decoder'),default='pose')
    parser.add_argument('--source-run',default='runs/arm_assignment_balanced_diverse_20261002')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--states',type=int,default=64)
    parser.add_argument('--chunk',type=int,default=2048)
    parser.add_argument('--rebound-samples',type=int,default=2)
    parser.add_argument('--gradient-steps',type=int,default=32)
    parser.add_argument('--seed',type=int,default=2901)
    parser.add_argument('--eval-interval',type=int,default=20)
    parser.add_argument('--eval-states',type=int,default=8)
    parser.add_argument('--benchmark-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Continue last.pt with optimizer, replay and RNG')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    device=torch.device(args.device)
    old_run=ROOT/args.source_run
    original=json.loads((old_run/'config.json').read_text())
    checkpoint_path=old_run/'dqn_before_model_based.pt'
    old=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    policy=make_policy(old,device); del old
    source_path=Path(original['frozen_planner'])
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    with rnea_from_cache_if_needed():
        model=A.build_model(A.model_config_from_checkpoint(source),device)
        cost=A.FusedCost(A.CostConfig(**source['train_cfg']['cost']),max_horizon=100,device=device)
    model.load_state_dict(source['model'])
    kit=A.DrumKit.from_layout()
    Generator = SingleHitGenerator if args.generator=='pose' else DecoderHitPoseGenerator
    generator=Generator(model,cost,kit,
        waist_upright_ease_wp=(source.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    surrogate=TrajectorySurrogate(cost,kit)
    del source
    clips=[A.load_rcc(p,kit,on_ineligible='keep') for p in original['training_clips']]
    spans=np.asarray([tempo_span(p) for p in original['training_clips']])
    def sampler(seed):
        return BalancedWindowSampler(clips,kit.num_drums,spans_s=spans,seed=seed,
            deduplicate=True,max_probability=.002)
    windows=sampler(args.seed)
    pool=np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    if tuple(pool['joint_names'])!=tuple(A.G1_UPPER_BODY_JOINTS):
        raise ValueError('pose anchor joint ordering mismatch')
    def clearance(q):
        return pose_clearance(cost,torch.as_tensor(q,device=device)).cpu().numpy()
    lo,hi=A.urdf_position_limits(posture_limits=False)
    starts=StartingPoseSampler(pool['poses'],ready_joint_pos(),lo,hi,
        clearance=clearance,names=A.G1_UPPER_BODY_JOINTS)
    run=ROOT/'runs'/args.name
    run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not args.resume:
        raise FileExistsError('Choose a new run name; existing checkpoint will not be overwritten')
    config=dict(vars(args),run_dir=str(run),policy_config=asdict(policy.cfg),
        critic=True,bellman=False,bellman_override='User requested complete finite-horizon search returns',
        algorithm='Direct complete-return Q regression',planner_checkpoint=str(source_path),
        resumed_assignment_body=str(checkpoint_path.resolve()),
        pose_generator=('Pretrained terminal-pose MLP; one 17-joint pose per call, analytic approach velocity'
                        if args.generator=='pose' else 'Frozen decoder; extract next-hit pose and velocity'),
        connector='Cubic Hermite; perfect tracking; independently scored physical collocations',
        bounce='Frictionless normal impulse, fixed nominal inertia, energy retention uniform[0.2,0.8]',
        search='16 exhaustive choices for four decisions, otherwise bounded complete-window candidates',
        drum_action_masks=False,crossover_margin_m=.2,
        first_hit_s=.5,window_sampling=windows.summary,start_pose_anchors=len(starts.anchors),
        source_training_clips=len(clips),source_step=32000,
        configured_candidate_worlds=args.states*16*args.rebound_samples)
    config['generator_parameters'] = sum(p.numel() for p in
        (generator.initializer.head.parameters() if args.generator=='pose' else model.parameters()))
    write_json(run/'config.json',config)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(run/'tb'),flush_secs=5)
    if os.environ.get('TB_AUTO_REGISTER','1')!='0':
        registry=ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        link=registry/run.name
        if not link.exists(): link.symlink_to((run/'tb').resolve())
    writer.add_scalar('setup/candidate_worlds',config['configured_candidate_worlds'],0)
    writer.add_scalar('setup/critic',1,0); writer.add_scalar('setup/bellman',0,0)
    writer.add_text('setup/assumptions',json.dumps(config,indent=2),0)
    writer.flush()
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    capacity=32768
    replay_obs=np.empty((capacity,policy.cfg.obs_dim),np.float32)
    replay_target=np.empty((capacity,2),np.float32)
    cursor=size=updates=0
    start_step=1
    if args.resume:
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if saved['format'] != 'arm_assignment_search_q_v1' or saved['config']['generator'] != args.generator:
            raise ValueError('Resume requires the same generator and direct-return checkpoint format')
        policy.load_state_dict(saved['policy']); optimizer.load_state_dict(saved['optimizer'])
        size=len(saved['replay_obs'])
        replay_obs[:size]=saved['replay_obs']; replay_target[:size]=saved['replay_targets']
        cursor=saved['replay_cursor']; updates=saved['gradient_updates']; start_step=saved['step']+1
        rng.bit_generator.state=saved['rng']
        del saved
    eval_rng=np.random.default_rng(args.seed+100000)
    eval_paths=sorted((ROOT/'assets/rcc/eval').glob('*.json'))
    eval_clips=[]
    for path in eval_paths:
        try:
            clip=A.load_rcc(path,kit,on_ineligible='keep')
            clip=window_clip(clip,eval_rng,span_s=tempo_span(path),first_hit_s=.5,tail_s=0)
            AssignmentQueue(clip,kit,policy.cfg)
            eval_clips.append(clip)
            if len(eval_clips)==args.eval_states: break
        except ValueError: continue
    if not eval_clips: raise RuntimeError('No eligible heldout rhythms')
    eval_poses,_=starts.sample(eval_rng,len(eval_clips))
    eval_poses[:max(1,len(eval_clips)//2)]=ready_joint_pos()
    eval_packed=pack_candidates(eval_clips,kit,policy.cfg,eval_poses,eval_rng,
        rebound_samples=args.rebound_samples,device=device)
    eval_obs=assignment_observations(eval_clips,kit,policy.cfg,eval_poses,cost.fk,device)
    evaluation=None
    t0=time.perf_counter()
    print(json.dumps(dict(status='ready',worlds=config['configured_candidate_worlds'],
        train_clips=len(clips),sampler=windows.summary,anchors=len(starts.anchors))),flush=True)
    for step in range(start_step,args.steps+1):
        start=time.perf_counter()
        if step>1 and (step-1)%10==0:
            windows=sampler(args.seed+step)
        _,batch=windows.sample(rng,args.states,first_hit_s=.5,tail_s=0)
        q,pose_stats=starts.sample(rng,args.states)
        packed=pack_candidates(batch,kit,policy.cfg,q,rng,rebound_samples=args.rebound_samples,device=device)
        obs=assignment_observations(batch,kit,policy.cfg,q,cost.fk,device)
        torch.cuda.reset_peak_memory_stats(device)
        rollout_start=time.perf_counter()
        out=score_candidates(surrogate,generator,packed,args.chunk)
        rollout_seconds=time.perf_counter()-rollout_start
        targets,best=direct_q_targets(out['score'],packed,len(batch),out['requested'])
        for observation,target in zip(obs.cpu().numpy(),targets):
            replay_obs[cursor]=observation; replay_target[cursor]=target
            cursor=(cursor+1)%capacity; size=min(size+1,capacity)
        losses=[]
        for _ in range(0 if args.benchmark_only else args.gradient_steps):
            indexes=rng.integers(size,size=min(256,max(32,size)))
            prediction=policy(torch.as_tensor(replay_obs[indexes],device=device))
            loss=F.smooth_l1_loss(prediction,torch.as_tensor(replay_target[indexes],device=device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
            optimizer.step(); updates+=1; losses.append(float(loss.detach()))
        summary=metrics(out,packed,targets,best,obs,policy)
        summary.update(rollout_seconds=rollout_seconds,step_seconds=time.perf_counter()-start,
            peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9,
            q_loss=float(np.mean(losses)) if losses else 0., replay_states=size,
            gradient_updates=updates,elapsed_seconds=time.perf_counter()-t0)
        for name,value in summary.items(): writer.add_scalar('search/'+name,value,step)
        for name,value in pose_stats.items(): writer.add_scalar('starts/'+name,value,step)
        counts=np.bincount([h.drum for clip in batch for h in clip.hits],minlength=kit.num_drums)
        for d,name in enumerate(kit.names):
            writer.add_scalar('drums/'+name+'/requested',counts[d],step)
            writer.add_scalar('drums/'+name+'/share',counts[d]/counts.sum(),step)
            writer.add_scalar('sampling/'+name+'/expected_share',windows.expected_shares[d],step)
        print(json.dumps(dict(step=step,**summary)),flush=True)
        write_json(run/'progress.json',dict(step=step,**summary))
        writer.flush()
        if not args.benchmark_only and (step==start_step or step%args.eval_interval==0):
            if evaluation is None:
                evaluation=score_candidates(surrogate,generator,eval_packed,args.chunk)
            et,eb=direct_q_targets(evaluation['score'],eval_packed,len(eval_clips),evaluation['requested'])
            em=metrics(evaluation,eval_packed,et,eb,eval_obs,policy)
            for name,value in em.items(): writer.add_scalar('heldout/'+name,value,step)
            write_json(run/'heldout.json',dict(step=step,**em,clips=[c.name for c in eval_clips]))
            print(json.dumps(dict(heldout_step=step,**em)),flush=True)
        if step==1 or step%10==0 or step==args.steps:
            blob=dict(format='arm_assignment_search_q_v1',policy=policy.state_dict(),
                policy_config=asdict(policy.cfg),optimizer=optimizer.state_dict(),step=step,
                gradient_updates=updates,config=config,rng=rng.bit_generator.state,
                replay_obs=replay_obs[:size],replay_targets=replay_target[:size],replay_cursor=cursor)
            torch.save(blob,run/'last.tmp'); (run/'last.tmp').replace(run/'last.pt')
        if args.benchmark_only: break
    writer.close()
    write_json(run/'finished.json',dict(step=step,gradient_updates=updates,benchmark=args.benchmark_only))


if __name__=='__main__': main()
