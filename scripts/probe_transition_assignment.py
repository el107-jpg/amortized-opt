#!/usr/bin/env python3
"""Search fresh schedules with a saved decoder-only transition table."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator, TrajectorySurrogate
from isaac_rl.assignment_search import hit_groups
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.random_rhythms import RandomRhythmSampler
from isaac_rl.schedule_analytic import AnalyticScheduleEnvironment
from isaac_rl.schedule_improvement import graph_seed, physical_table, MAX_HITS
from isaac_rl.transition_assignment import TransitionTable, AssignmentScoreConfig, score_assignment, search_assignments
from scripts.calibrate_transition_assignment import live_initial_times
from scripts.probe_transition_qp import write_json


def hand_choices(clip, table):
    allowed=table.allowed_drums()
    return [[a for a in (0,1) if all(allowed[(a+m)%2,hit.drum] for m,hit in enumerate(group))]
            for group in hit_groups(clip)]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--windows',type=int,default=256)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--seed',type=int,default=202610041)
    args=p.parse_args()
    if args.windows<1 or args.chunk<1:p.error('Windows and chunk must be positive')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic();torch.set_num_threads(2);torch.manual_seed(args.seed)
    table=TransitionTable.load(args.table)
    if table.metadata['qp']['iterations']!=0:raise ValueError('Expected decoder-only table')
    config=AssignmentScoreConfig(**table.metadata['score_config'])
    device=torch.device(args.device)
    checkpoint=torch.load(table.metadata['checkpoint'],map_location='cpu',weights_only=False)
    with rnea_from_cache_if_needed():
        model=A.build_model(A.model_config_from_checkpoint(checkpoint),device)
        cfg=A.CostConfig(**checkpoint['train_cfg']['cost']);cfg.n_sub=table.metadata['collocation_substeps']
        cfg.w_velocity=max(cfg.w_velocity,1.);cfg.w_joint_box=max(cfg.w_joint_box,1.)
        cost=A.FusedCost(cfg,max_horizon=100,device=device)
    model.load_state_dict(checkpoint['model']);model.eval()
    kit=A.DrumKit.from_layout()
    if tuple(kit.names)!=table.drum_names:raise ValueError('Table and generator drum order differ')
    generator=DecoderHitPoseGenerator(model,cost,kit,
        waist_upright_ease_wp=(checkpoint.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(out/'tb'))
    registry=ROOT/'runs/tensorboard_active'/out.name
    if not registry.is_symlink():registry.symlink_to((out/'tb').resolve())
    meta=dict(status='running',table=str(args.table.resolve()),checkpoint=table.metadata['checkpoint'],
        windows=args.windows,seed=args.seed,device=args.device,chunk=args.chunk,
        score_config=asdict(config),lookahead_s=1.,max_hits=10,search='Exhaustive eligible full-window assignments',
        decoder_only=True,training=False,initial_state='Measured ready posture, zero velocity',
        environment='Frozen decoder endpoints with Hermite interpolation and analytic bounce; no Isaac',
        hit_rules='Existing analytic contact and approach-speed requirements; not the relaxed table contact gate',
        path_constraints='Both arms and shared waist evaluated by the existing analytic environment',
        forbidden_drums_by_hand=table.metadata['forbidden_drums_by_hand'])
    write_json(out/'config.json',meta)
    write_json(out/'run.json',{'status':'running','stage':'initial_state_calibration'})
    q=torch.as_tensor(ready_joint_pos(),device=device,dtype=torch.float32)[None];v=torch.zeros_like(q)
    rng=np.random.default_rng(args.seed)
    initial=live_initial_times(generator,q,v,table.yaw_centers,rng,16,args.chunk,'full-range',
        table.metadata['hit_depth_mm']/1000,False,True,True)
    initial[~table.allowed_drums()[:,:-1]]=np.inf
    np.save(out/'ready_initial_minimum_s.npy',initial)
    print(json.dumps({'stage':'initial_state_calibrated','supported':int(np.isfinite(initial).sum()),
                      'entries':int(initial.size)}),flush=True)
    sampler=RandomRhythmSampler(kit);clips=[];rejected=[]
    while len(clips)<args.windows:
        for clip in sampler.sample(rng,args.windows-len(clips)):
            if all(hand_choices(clip,table)):clips.append(clip)
            else:rejected.append({'rhythm':clip.name,'hits':[{'time':h.time,'drum':h.drum_name} for h in clip.hits]})
    write_json(out/'excluded_windows.json',rejected)
    actions=[];graph=[];details=[];runtimes=[];changed=0;groups_total=0;forced=0;fallbacks=0
    for index,clip in enumerate(clips):
        started=time.perf_counter()
        best,candidates=search_assignments(clip,table,yaw_now=float(q[0,0]),initial_minimum_s=initial,cfg=config)
        runtimes.append((time.perf_counter()-started)*1000)
        a=np.zeros(MAX_HITS,int);a[:len(best['actions'])]=best['actions']
        baseline,n_fallback=graph_seed(clip,kit);fallbacks+=n_fallback
        choices=hand_choices(clip,table)
        modifications=0
        for g,allowed in enumerate(choices):
            if baseline[g] not in allowed:
                baseline[g]=allowed[0];modifications+=1
        forced+=modifications
        graph_score=score_assignment(hit_groups(clip),baseline[:len(choices)],table,
            yaw_now=float(q[0,0]),initial_minimum_s=initial,cfg=config)
        if best['total']>graph_score['total']+1e-6:raise AssertionError('Exhaustive winner worse than eligible graph proxy')
        changed+=int((a[:len(choices)]!=baseline[:len(choices)]).sum());groups_total+=len(choices)
        actions.append(a);graph.append(baseline)
        details.append(dict(rhythm=clip.name,hits=[dict(time=h.time,drum=h.drum,name=h.drum_name) for h in clip.hits],
            candidates=len(candidates),selected=best,graph=graph_score,graph_policy_adjustments=modifications))
        if (index+1)%32==0:
            print(json.dumps({'stage':'search','windows':index+1,'elapsed_s':time.monotonic()-begin}),flush=True)
            writer.add_scalar('search/windows',index+1,index+1);writer.flush()
    write_json(out/'probe_assignments.json',details)
    write_json(out/'run.json',{'status':'running','stage':'analytic_rollout'})
    physical=physical_table(clips,kit,device)
    candidate_bank=torch.as_tensor(np.stack((np.asarray(graph),np.asarray(actions)),1),device=device)
    draws=torch.as_tensor(rng.uniform(.2,.8,(len(clips),2,MAX_HITS,2)),device=device,dtype=q.dtype)
    env=AnalyticScheduleEnvironment(generator,TrajectorySurrogate(cost,kit),chunk=args.chunk)
    result=env.score(physical,q.expand(len(clips),-1),v.expand(len(clips),-1),candidate_bank,draws)
    report=dict(status='complete',windows=len(clips),excluded_windows=len(rejected),
        bounce_draws=2,search_ms_mean=float(np.mean(runtimes)),search_ms_max=float(np.max(runtimes)),
        search_ms_p95=float(np.percentile(runtimes,95)),changed_assignment_fraction=changed/groups_total,
        graph_policy_adjustments=forced,graph_fallbacks=fallbacks,
        ready_initial_supported=int(np.isfinite(initial).sum()),ready_initial_entries=int(initial.size),
        table_missing_edges=sum(d['selected']['missing_edges'] for d in details),
        graph_missing_edges=sum(d['graph']['missing_edges'] for d in details),
        table_missing_pairs=sum(d['selected']['missing_pairs'] for d in details),
        graph_missing_pairs=sum(d['graph']['missing_pairs'] for d in details),
        mean_proxy_improvement=float(np.mean([d['graph']['total']-d['selected']['total'] for d in details])),
        elapsed_s=time.monotonic()-begin,environment=meta['environment'])
    for j,name in enumerate(('graph','transition_table')):
        hits=int(result['hits'][:,j].sum());requested=int(result['requested'][:,j].sum())
        report[name]=dict(hits=hits,requested=requested,hit_rate=hits/requested,
            return_mean=float(result['returns'][:,j,:,0].mean()),
            score_mean=float(result['score'][:,j].mean()),
            torque_excess_mean=float(result['torque_excess'][:,j].mean()),
            velocity_excess_mean=float(result['velocity_excess'][:,j].mean()),
            collision_m_mean=float(result['collision_m'][:,j].mean()))
        writer.add_scalar(f'probe/{name}_hit_rate',hits/requested,len(clips))
        writer.add_scalar(f'probe/{name}_return',report[name]['return_mean'],len(clips))
    gains=result['returns'][:,1,:,0].mean(-1)-result['returns'][:,0,:,0].mean(-1)
    report.update(windows_better=int((gains>1e-5).sum()),windows_worse=int((gains< -1e-5).sum()),
                  windows_tied=int((gains.abs()<=1e-5).sum()))
    torch.save({k:v.cpu() for k,v in result.items() if torch.is_tensor(v)},out/'probe_rollouts.pt')
    write_json(out/'probe.json',report);write_json(out/'run.json',report)
    writer.close();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
