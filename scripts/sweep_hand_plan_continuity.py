#!/usr/bin/env python3
"""Fresh Isaac local timing/pose sweeps of every legal four-group hand plan."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import GEOMETRY_VERSION
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
from isaac_rl.hand_plan_ranker import ACTION_BANK,HandPlanRanker,HandPlanRankerConfig,rank_clips
from isaac_rl.hand_plan_sweep import AXES,perturb_case,sensitivity_summary
from isaac_rl.min_time_assignment import MinimumTimeChart
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_neural_hand_plans import make_start_bank,save_torch
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def plot_results(out,points,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((out/'summary.json').read_text())
    runtime_caption='\nFresh scenes per offset; matched decoder impact speed; original frozen neural scores' if summary.get('recreate_every_point') else ''
    for case in range(cases):
        fig,axes=plt.subplots(3,4,figsize=(17,10),constrained_layout=True)
        for column,axis in enumerate(AXES):
            rows=sorted([p for p in points if p['case']==case and p['axis']==axis],key=lambda p:p['step'])
            valid=[r for r in rows if r['valid']]
            if not valid:continue
            x=[r['offset_display'] for r in rows]
            for j,action in enumerate(valid[0]['actions']):
                axes[0,column].plot(x,[r['reward'][j] if r['valid'] else np.nan for r in rows],alpha=.55,lw=1,label=f'{action:04b}')
                axes[1,column].step(x,[r['hits'][j] if r['valid'] else np.nan for r in rows],where='mid',alpha=.55,lw=1)
                axes[2,column].plot(x,[r['neural_scores'][j]-np.mean(r['neural_scores']) if r['valid'] else np.nan for r in rows],alpha=.7,lw=1)
            axes[0,column].plot(x,[r['best_reward'] if r['valid'] else np.nan for r in rows],'k',lw=2,label='Best observed')
            axes[0,column].plot(x,[r['neural_selected_reward'] if r['valid'] else np.nan for r in rows],'k--',lw=2,label='Neural pick')
            axes[0,column].set_title(axis.replace('_',' '))
            for row in range(3):axes[row,column].set_xlabel('Offset (ms)' if column<2 else 'Offset (degrees)');axes[row,column].grid(alpha=.2)
        axes[0,0].set_ylabel('Physical reward');axes[1,0].set_ylabel('Successful hits')
        axes[2,0].set_ylabel('Centered neural score (unitless)')
        axes[0,-1].legend(fontsize=6,ncol=2)
        fig.suptitle(f'Case {case}: fixed rhythm/state, all legal assignments; 250 Hz physics / 50 Hz control'+runtime_caption)
        target=out/f'case_{case:02d}.png';temp=target.with_suffix('.tmp.png');fig.savefig(temp,dpi=140);temp.replace(target);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    labels=[a.replace('_','\n') for a in AXES]
    axes[0].bar(labels,[100*summary['axes'][a]['fraction_hit_credit_changed'] for a in AXES])
    axes[0].set_ylabel('Neighbor pairs with a changed hit credit (%)')
    axes[1].bar(labels,[summary['axes'][a]['reward_difference_p95'] for a in AXES])
    axes[1].axhline(summary['repeatability']['reward_range_max'],color='tab:red',ls='--',label='Max unchanged-case spread')
    axes[1].set_ylabel('95th percentile absolute reward change');axes[1].legend(fontsize=8)
    fig.suptitle('Local sensitivity: 1 ms timing steps / 0.1 degree pose steps')
    fig.savefig(out/'sweep_summary.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=int,default=8);p.add_argument('--envs',type=int,default=4096)
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    p.add_argument('--seed',type=int,default=20261017)
    cli=p.parse_args();torch.set_num_threads(2)
    if cli.out.exists():raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True)
    cfg=json.loads((cli.run/'config.json').read_text());source=rollout_source_config(cfg)
    if cfg['simulation_geometry_version']!=GEOMETRY_VERSION:raise ValueError('Run geometry differs')
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:raise ValueError('Decoder changed')
    (cli.out/'neural_snapshot.pt').write_bytes((cli.run/'last.pt').read_bytes())
    saved=torch.load(cli.out/'neural_snapshot.pt',map_location='cpu',weights_only=False)
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network']));model.load_state_dict(saved['model']);model.eval()
    geometry=model.config.tip_geometry;bank=make_start_bank(geometry);bank.load_state_dict(saved['post_hit_bank'])
    eligibility=MinimumTimeChart.load(cfg['eligibility_source']);rng=np.random.default_rng(cli.seed)
    sampler=StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility)
    panel=sampler.sample(cli.cases,rng);states,start_metrics=bank.sample(rng,cli.cases)
    directions=rng.normal(size=(cli.cases,17));directions[:,:3]=0;directions/=np.abs(directions).max(1,keepdims=True)
    points=[];episodes=[];episode_states=[]
    for case,(clip,state) in enumerate(zip(panel,states)):
        settings=[('control',0,i) for i in range(3)]+[(a,s,0) for a in AXES for s in range(-20,21)]
        for axis,step,repeat in settings:
            offset=step*.001 if axis in AXES[:2] else np.deg2rad(step*.1)
            if axis=='control':changed,pose,reason=clip,state,None
            else:changed,pose,reason=perturb_case(clip,state,axis,offset,directions[case],geometry)
            if reason is None and axis in AXES[2:] and not bank.valid(np.array([pose['q']]),np.array([pose['qd']]),random=state['kind']=='random')[0]:
                reason='Starting pose fails joint/clearance bounds'
            point=dict(case=case,axis=axis,step=step,repeat=repeat,offset=offset,
                offset_display=step if axis in AXES[:2] else step*.1,valid=reason is None,skip_reason=reason)
            if reason is None:
                _,legal,scores=rank_clips(model,[changed],eligibility,[pose]);actions=np.flatnonzero(legal[0])
                point.update(actions=actions.tolist(),neural_scores=scores[0,actions].tolist(),
                    neural_action=int(actions[scores[0,actions].argmax()]),report_ids=list(range(len(episodes),len(episodes)+len(actions))))
                episodes.extend(assigned_clip(changed,ACTION_BANK[a]) for a in actions)
                episode_states.extend([pose]*len(actions))
            points.append(point)
    save_torch(cli.out/'plans.pt',dict(panel=panel,states=states,arm_directions=directions,points=points,
        episodes=episodes,episode_states=episode_states))
    write_json(cli.out/'config.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()},
        checkpoint=str(checkpoint),checkpoint_sha256=cfg['checkpoint_sha256'],simulation_geometry_version=GEOMETRY_VERSION,
        physics_hz=250,control_hz=50,neural_gradient_steps=saved['gradient_steps'],starting_states=start_metrics,
        axes=dict(first_delay='All hit deadlines shifted +/-20ms,1ms steps; later gaps unchanged',
            second_gap='Groups2-4 shifted +/-20ms,1ms steps; first lead unchanged',
            waist_yaw='Initial waist yaw +/-2degrees,.1degree steps',
            arm_direction='Fixed per-case arm-only direction, +/-2degrees max per joint,.1degree steps'),
        training_feedback=False,points=len(points),physical_rollouts=len(episodes),
        skipped_points=sum(not p['valid'] for p in points),baseline_repeats=3))
    args=runtime_args(['--out',str(cli.out),'--envs',str(cli.envs),'--device',cli.devices[0],
        '--rollout-devices',*cli.devices,'--physics-hz','250','--decode-chunk','512','--collision-chunk','1024',
        '--no-plan-diagnostics','--clone-kit-from-source'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cli.seed
    seed=cli.out/'scene_seed.json';write_json(seed,sampler.seed_payload());evaluator=None;started=time.monotonic()
    write_json(cli.out/'run.json',dict(status='starting',rollouts=len(episodes)))
    try:
        evaluator=DistributedIsaacEvaluator(args,checkpoint,seed)
        for folder,_ in evaluator.workers:
            if json.loads((folder/'ready.json').read_text())['tip_geometry']!=geometry:raise ValueError('Isaac/state-bank geometry differs')
        write_json(cli.out/'run.json',dict(status='running',rollouts=len(episodes)))
        reports=evaluator.evaluate(episodes,lambda done,total,tick,ticks:write_json(cli.out/'progress.json',
            dict(done=done,total=total,tick=tick,ticks=ticks,elapsed_s=time.monotonic()-started)),initial_states=episode_states)
        write_json(cli.out/'episodes.json',reports)
        summary=sensitivity_summary(points,reports)
        qerr=max(np.max(np.abs(np.array(r['initial_state']['q'])-s['q'])) for r,s in zip(reports,episode_states))
        qderr=max(np.max(np.abs(np.array(r['initial_state']['qd'])-s['qd'])) for r,s in zip(reports,episode_states))
        tiperr=max(np.linalg.norm(np.array(r['initial_state']['tips'])-s['tips'],axis=1).max() for r,s in zip(reports,episode_states))
        summary.update(status='complete',cases=cli.cases,rollouts=len(reports),elapsed_s=time.monotonic()-started,
            initial_state_parity=dict(max_q_error_rad=float(qerr),max_qd_error_rad_s=float(qderr),max_tip_error_m=float(tiperr)),
            limitation=f'Finite local sweeps of{cli.cases} fresh cases; includes1ms off-training-grid timing; not a proof of global continuity or held-out model accuracy.')
        write_json(cli.out/'summary.json',summary);write_json(cli.out/'points.json',points);plot_results(cli.out,points,cli.cases)
        write_json(cli.out/'run.json',summary);print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':main()
