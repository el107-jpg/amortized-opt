"""Paired Isaac evaluation and behavior-constrained Double DQN adaptation."""
from __future__ import annotations
import copy
import json
import time
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from amortized_opt.constants import ready_joint_pos
from amortized_opt.rhythm import load_rcc
from isaac_rl.arm_assignment import AssignmentQueue
from isaac_rl.assignment_dqn import AnchoredDoubleDQN,Replay
from isaac_rl.assignment_parallel import ParallelPoseMPC
from isaac_rl.assignment_parallel_train import collect,make_start_sampler,load_with_tempo
from isaac_rl.assignment_sampling import BalancedWindowSampler
from isaac_rl.assignment_train import window_clip


def serial_config(cfg): return {k:v for k,v in cfg.items() if k!='geometry_scorer'}


def aggregate(reports):
    n=sum(r['requested'] for r in reports)
    by_drum={}
    for name in reports[0]['by_drum']:
        requested=sum(r['by_drum'][name]['requested'] for r in reports)
        hits=sum(r['by_drum'][name]['hits'] for r in reports)
        left=sum(r['by_drum'][name]['left'] for r in reports)
        by_drum[name]=dict(requested=requested,hits=hits,hit_fraction=hits/max(requested,1),
                           left_fraction=left/max(requested,1))
    return dict(hits=sum(r['hits'] for r in reports),requested=n,hit_fraction=sum(r['hits'] for r in reports)/max(n,1),
        return_mean=float(np.mean([r['return_'] for r in reports])),
        strays=sum(r['strays'] for r in reports),episodes=len(reports),by_drum=by_drum)


def span(path):
    bpm=float(json.loads(Path(path).read_text()).get('metadata',{}).get('original_bpm',120))
    return max(.02,4*60/max(bpm,1)-1.)


def save_json(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)


def run_close_experiment(env,decoder_mpc,pid,policy,cfg,args):
    run=Path(cfg['run_dir']); B=env.unwrapped.num_envs; kit=env.unwrapped.assignment_kit
    cfg.update(algorithm='Anchored Double DQN',rl_lr=args.close_lr,iterations=args.close_updates,
        gradient_steps=args.gradient_steps,critic=True,bellman=True,
        close_reference_checkpoint=cfg['resumed_from'],anchor_weight=args.close_anchor_weight,
        guard_max_flip_fraction=args.close_max_flips,guard_max_mean_kl=.02,
        exploration_epsilon=.02,primary_controller='direct pose MLP + committed Hermite + existing PID',
        simulated_bounce=False,replay_reset=True,optimizer_reset=True,
        eval_selection='Final accepted update; heldout rewards are reported and not used for checkpoint selection')
    save_json(run/'config.json',serial_config(cfg))
    torch.save(dict(format='arm_assignment_dqn_init_v1',policy=policy.state_dict(),
        policy_config=asdict(policy.cfg),config=serial_config(cfg)),run/'isaac_anchor.pt')
    anchor=copy.deepcopy(policy).eval().requires_grad_(False)
    pose_mpc=ParallelPoseMPC(decoder_mpc.model,decoder_mpc.sched,decoder_mpc.cfg,
        device=decoder_mpc.device,kit_names=kit.names,descent_cost=decoder_mpc.descent_cost,kit=kit)
    start_sampler=make_start_sampler(decoder_mpc,pid,cfg)
    if start_sampler is None: raise ValueError('Varied-start pose pool is required')
    eval_rng=np.random.default_rng(cfg['seed']+900001)
    eval_paths=sorted(Path(cfg['eval_clip']).parent.glob('*.json'))
    candidates=[]
    for path in eval_paths:
        try:
            c,pc=load_with_tempo(path,kit,policy.cfg)
            AssignmentQueue(c,kit,pc); candidates.append((path,c,pc))
        except ValueError: continue
    clips=[]; configs=[]; rejected=0
    for _ in range(B*20):
        path,c,pc=candidates[int(eval_rng.integers(len(candidates)))]
        try:
            window=window_clip(c,eval_rng,span_s=span(path),first_hit_s=1.,tail_s=cfg['tail_s'])
            queue=AssignmentQueue(window,kit,pc)
            while queue.next_group<len(queue.groups): queue.commit(queue.teacher_action())
            clips.append(window); configs.append(pc)
            if len(clips)==B: break
        except ValueError: rejected+=1
    if len(clips)!=B: raise ValueError('Unable to fill heldout panel')
    starts,_=start_sampler.sample(eval_rng,B); starts[:B//2]=ready_joint_pos()
    save_json(run/'isaac_eval_panel.json',dict(optimized_on=False,seed=cfg['seed']+900001,
        first_hit_s=1,graph_unsupported_windows_excluded=rejected,
        starts=starts.tolist(),clips=[dict(name=c.name,hits=[asdict(h) for h in c.hits],duration=c.duration) for c in clips]))
    writer=SummaryWriter(str(run/'tb'),flush_secs=5)
    guard=[]; comparisons={}
    def rollout(mode,which_policy,graph=False,iteration=0):
        mpc=pose_mpc if mode=='pose' else decoder_mpc
        pid.mpc=mpc
        label=f'{mode}_{"graph" if graph else "anchor" if which_policy is anchor else "learned"}'
        start=time.perf_counter()
        def progress(t,n):
            print(json.dumps(dict(stage=label,iteration=iteration,tick=t,total=n,seconds=time.perf_counter()-start)),flush=True)
        records,reports,timings=collect(env,mpc,pid,which_policy,clips,configs,cfg,graph=graph,
            progress=progress,start_poses=starts)
        summary=aggregate(reports); summary['seconds']=time.perf_counter()-start
        comparisons[label]=summary
        save_json(run/f'isaac_{label}_{iteration}.json',dict(summary=summary,reports=reports,timings=timings))
        for k in ('hit_fraction','return_mean','strays','seconds'): writer.add_scalar(f'isaac/{label}/{k}',summary[k],iteration)
        for d,row in summary['by_drum'].items():
            writer.add_scalar(f'isaac/{label}/{d}_hit_fraction',row['hit_fraction'],iteration)
        writer.flush()
        print(json.dumps(dict(stage=label,iteration=iteration,summary=summary)),flush=True)
        return records,reports
    for mode in ('pose','decoder'):
        for graph in (True,False):
            records,_=rollout(mode,anchor,graph=graph)
            guard.extend(r['obs'] for e in records for r in e)
    guard_obs=np.stack(guard)
    np.savez_compressed(run/'isaac_guard_states.npz',observations=guard_obs)
    save_json(run/'isaac_baseline_comparison.json',comparisons)
    learner=AnchoredDoubleDQN(policy,lr=args.close_lr,anchor_weight=args.close_anchor_weight,
        max_flip_fraction=args.close_max_flips)
    replay=Replay(policy.cfg.obs_dim,capacity=25000,seed=cfg['seed']+700)
    rng=np.random.default_rng(cfg['seed']+711)
    training=[load_with_tempo(p,kit,policy.cfg) for p in cfg['training_clips']]
    spans=np.array([span(p) for p in cfg['training_clips']])
    def make_windows(iteration):
        return BalancedWindowSampler([c for c,_ in training],kit.num_drums,spans_s=spans,
            deduplicate=True,max_probability=.002,seed=cfg['seed']+821+iteration)
    pid.mpc=pose_mpc
    for iteration in range(1,args.close_updates+1):
        sampler=make_windows(iteration)
        indexes,windows=sampler.sample(rng,B,first_hit_s=1.,tail_s=cfg['tail_s'])
        positions,start_stats=start_sampler.sample(rng,B)
        start=time.perf_counter()
        def progress(t,n):
            writer.add_scalar('collection/tick',t,iteration*1000+t); writer.flush()
            print(json.dumps(dict(stage='finetune_collection',iteration=iteration,tick=t,total=n,seconds=time.perf_counter()-start)),flush=True)
        records,reports,timings=collect(env,pose_mpc,pid,policy,windows,[training[i][1] for i in indexes],cfg,
            epsilon=.02,progress=progress,start_poses=positions)
        for e in records: replay.add_episode(e)
        stats=learner.update(replay,gradient_steps=args.gradient_steps,guard_obs=guard_obs)
        stats.update(iteration=iteration,collection_seconds=time.perf_counter()-start,**aggregate(reports))
        for k,v in stats.items():
            if isinstance(v,(int,float)): writer.add_scalar('close_dqn/'+k,v,iteration)
        for k,v in start_stats.items(): writer.add_scalar('starts/'+k,v,iteration)
        writer.flush()
        save_json(run/'close_progress.json',stats)
        with (run/'close_metrics.jsonl').open('a') as f: f.write(json.dumps(stats)+'\n')
        state=dict(format='arm_assignment_dqn_v1',policy=policy.state_dict(),policy_config=asdict(policy.cfg),
            target=learner.target.state_dict(),optimizer=learner.optimizer.state_dict(),gradient_updates=learner.updates,
            iteration=iteration,replay=replay.state_dict(),rng=rng.bit_generator.state,config=serial_config(cfg),
            reference_policy=learner.reference.state_dict(),anchor_weight=learner.anchor_weight,
            guard_drift=learner.drift(guard_obs))
        torch.save(state,run/'dqn_last.tmp'); (run/'dqn_last.tmp').replace(run/'dqn_last.pt')
        print(json.dumps(dict(stage='finetune_update',stats=stats)),flush=True)
        # Paired anchor and tuned execution on the same physical controller.
        if iteration==args.close_updates:
            rollout('pose',policy,iteration=iteration)
            rollout('decoder',policy,iteration=iteration)
    save_json(run/'isaac_final_comparison.json',dict(source=cfg['resumed_from'],comparisons=comparisons,
        accepted_gradient_updates=learner.updates,guard=learner.drift(guard_obs),
        guard_states=len(guard_obs),scope='Actual contact hit credit in Isaac; generator and PID frozen'))
    writer.close()
