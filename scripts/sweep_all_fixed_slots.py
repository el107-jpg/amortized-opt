#!/usr/bin/env python3
"""Rerun all original continuity curves with persistent per-curve simulator slots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.hand_plan_sweep import AXES,sensitivity_summary
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.sweep_hand_plan_continuity import plot_results
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sweep',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--decoder-fp32',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--fresh-scene-controls',type=int,default=0,help='Additional zero-offset controls in independently recreated scenes')
    p.add_argument('--recreate-every-point',action='store_true',help='Start fresh worker processes/scenes for every sweep offset')
    p.add_argument('--reuse-controls',type=Path,help='Reuse only the already-completed matching-runtime control rounds')
    cli=p.parse_args();torch.set_num_threads(2);cli.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((cli.sweep/'config.json').read_text());run=json.loads((Path(cfg['args']['run'])/'config.json').read_text());source=rollout_source_config(run)
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:raise ValueError('Frozen decoder changed')
    saved=torch.load(cli.sweep/'plans.pt',map_location='cpu',weights_only=False)
    point_map={(p['case'],p['axis'],p['step']):p for p in saved['points'] if p['axis'] in AXES}
    controls={p['case']:p for p in saved['points'] if p['axis']=='control' and p['repeat']==0}
    cases=sorted(controls);curves=[(case,axis,action) for case in cases for axis in AXES for action in controls[case]['actions']]
    if cli.smoke:curves=curves[:4]
    curve_slot={key:i for i,key in enumerate(curves)};devices=['cuda:0','cuda:1','cuda:2','cuda:3']
    flags=['--out',str(cli.out),'--envs',str(len(curves)),'--device',devices[0],'--rollout-devices',*devices,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed']
    if cli.decoder_fp32:flags+=['--decoder-fp32']
    args=runtime_args(flags)
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cfg['args']['seed']
    config=dict(original_sweep=str(cli.sweep.resolve()),physics_hz=250,control_hz=50,
        fixed_slots=True,curves=len(curves),cases=len(cases),baseline_repeats_per_curve=3,
        clear_reset_contacts=True,freeze_hit_scoring=True,decoder_fp32=cli.decoder_fp32,
        match_decoder_speed=True,fresh_scene_controls=cli.fresh_scene_controls,
        recreate_every_point=cli.recreate_every_point,
        training_feedback=False,checkpoint_sha256=cfg['checkpoint_sha256'],same_rhythms_states_assignments=True,
        axes=cfg['axes'],smoke=cli.smoke)
    write_json(cli.out/'config.json',config);torch.save(dict(curves=curves,curve_slot=curve_slot),cli.out/'slot_map.pt')
    (cli.out/'batches').mkdir();evaluator=None;points=[];reports=[];started=time.monotonic()
    rounds=[('control',0,r) for r in range(3)]+[('fresh_control',0,r+1) for r in range(cli.fresh_scene_controls)]+[('sweep',s,0) for s in ([-20,0,20] if cli.smoke else range(-20,21))]
    write_json(cli.out/'run.json',dict(status='starting',rounds=len(rounds),curves=len(curves)))
    try:
        evaluator=DistributedIsaacEvaluator(args,checkpoint,cli.sweep/'scene_seed.json')
        ready=[json.loads((folder/'ready.json').read_text()) for folder,_ in evaluator.workers]
        if any(r['tip_geometry']!=run['network']['tip_geometry'] for r in ready):raise ValueError('Simulator geometry differs from original states')
        config['rollout_runtime']={k:ready[0][k] for k in ('rollout_runtime_version','fixed_hit_scoring','decoder_amp','impact_speed_target')}
        write_json(cli.out/'config.json',config)
        reused_rounds=0
        if cli.reuse_controls:
            from isaac_rl.hand_plan_dataset import require_matching_rollout_runtime
            previous_config=json.loads((cli.reuse_controls/'config.json').read_text())
            require_matching_rollout_runtime(config,previous_config)
            previous_curves=torch.load(cli.reuse_controls/'slot_map.pt',weights_only=False)['curves']
            if previous_curves!=curves or previous_config['original_sweep']!=config['original_sweep']:
                raise ValueError('Reused controls differ in physical inputs or slot mapping')
            for i in range(3+cli.fresh_scene_controls):
                previous=json.loads((cli.reuse_controls/'batches'/f'round_{i:03d}.json').read_text())
                if any(p['axis']!='control' for p in previous['points']):raise ValueError('Only control rounds may be reused')
                reports.extend(previous['reports']);points.extend(previous['points'])
                write_json(cli.out/'batches'/f'round_{i:03d}.json',previous)
                reused_rounds+=1
        for round_id,(mode,step,repeat) in enumerate(rounds):
            if round_id<reused_rounds:continue
            if mode=='fresh_control' or (mode=='sweep' and cli.recreate_every_point):
                write_json(cli.out/'run.json',dict(status='recreating_scene',completed_rounds=round_id,total_rounds=len(rounds)))
                evaluator.close();evaluator=DistributedIsaacEvaluator(args,checkpoint,cli.sweep/'scene_seed.json')
                if any(json.loads((folder/'ready.json').read_text())['tip_geometry']!=run['network']['tip_geometry'] for folder,_ in evaluator.workers):
                    raise ValueError('Recreated simulator geometry differs')
            episodes=[];states=[];round_points=[];offset=len(reports)
            for case,axis,action in curves:
                original=controls[case] if mode in ('control','fresh_control') else point_map[(case,axis,step)]
                if original['valid']:
                    idx=original['report_ids'][original['actions'].index(action)]
                    episodes.append(saved['episodes'][idx]);states.append(saved['episode_states'][idx])
                else:
                    idx=controls[case]['report_ids'][controls[case]['actions'].index(action)]
                    episodes.append(None);states.append(saved['episode_states'][idx])
            for case,axis in dict.fromkeys((c,a) for c,a,_ in curves):
                original=controls[case] if mode in ('control','fresh_control') else point_map[(case,axis,step)]
                point={k:copy.deepcopy(original[k]) for k in ('case','axis','step','repeat','offset','offset_display','valid','skip_reason')}
                if mode in ('control','fresh_control'):
                    point.update(axis='control',control_axis=axis,repeat=repeat,control_kind=mode)
                actions=[a for c,ax,a in curves if c==case and ax==axis]
                if point['valid']:
                    point.update(actions=actions,neural_scores=[original['neural_scores'][original['actions'].index(a)] for a in actions],
                        neural_action=max(actions,key=lambda a:original['neural_scores'][original['actions'].index(a)]),
                        report_ids=[offset+curve_slot[(case,axis,a)] for a in actions])
                round_points.append(point)
            round_started=time.monotonic()
            trial=evaluator.evaluate(episodes,lambda done,total,tick,ticks:write_json(cli.out/'progress.json',
                dict(round=round_id+1,total_rounds=len(rounds),mode=mode,step=step,tick=tick,ticks=ticks,
                    elapsed_s=time.monotonic()-started)),initial_states=states)
            reports.extend(trial);points.extend(round_points)
            write_json(cli.out/'batches'/f'round_{round_id:03d}.json',dict(points=round_points,reports=trial,seconds=time.monotonic()-round_started))
            write_json(cli.out/'run.json',dict(status='running',completed_rounds=round_id+1,total_rounds=len(rounds),
                actual_rollouts=sum(r is not None for r in reports),seconds=time.monotonic()-started))
            print(f'[fixed-slots] round={round_id+1}/{len(rounds)} step={step} seconds={time.monotonic()-round_started:.1f}',flush=True)
        write_json(cli.out/'episodes.json',reports)
        summary=sensitivity_summary(points,reports,fresh_scene_baselines=cli.recreate_every_point)
        # Repeat controls compare identical inputs in exactly the same curve slots.
        baseline=[reports[r*len(curves):(r+1)*len(curves)] for r in range(3)]
        hits=np.array([[x['hits'] for x in rows] for rows in baseline]);rewards=np.array([[x['reward'] for x in rows] for rows in baseline])
        pair_change=[]
        for i,j in [(0,1),(0,2),(1,2)]:
            pair_change.extend(any(a['hit']!=b['hit'] for a,b in zip(x['per_hit_outcomes'],y['per_hit_outcomes'])) for x,y in zip(baseline[i],baseline[j]))
        spread=np.ptp(rewards,axis=0)
        summary.update(status='complete',elapsed_s=time.monotonic()-started,actual_rollouts=sum(r is not None for r in reports),
            fixed_slots=True,curves=len(curves),cases=len(cases),same_slot_controls=dict(hit_count_varies_fraction=float(np.mean(np.ptp(hits,axis=0)>0)),
                hit_credit_change_pair_fraction=float(np.mean(pair_change)),reward_range_median=float(np.median(spread)),
                reward_range_p95=float(np.quantile(spread,.95)),reward_range_max=float(spread.max())),
            limitation='Original eight cases/all legal plans; each curve stays in one physical slot. Local sampled sensitivity, not global continuity or a model accuracy test.')
        summary['recreate_every_point']=cli.recreate_every_point
        if cli.fresh_scene_controls:
            fresh_indices=[0]+list(range(3,3+cli.fresh_scene_controls))
            fresh=[[reports[i*len(curves)+j] for j in range(len(curves))] for i in fresh_indices]
            fresh_hits=np.array([[r['hits'] for r in rows] for rows in fresh]);fresh_reward=np.array([[r['reward'] for r in rows] for rows in fresh])
            spread=np.ptp(fresh_reward,axis=0)
            summary['fresh_scene_controls']=dict(independent_scenes=len(fresh),
                hit_count_varies_fraction=float(np.mean(np.ptp(fresh_hits,axis=0)>0)),
                varying_hit_count_curves=int((np.ptp(fresh_hits,axis=0)>0).sum()),
                reward_range_median=float(np.median(spread)),reward_range_p95=float(np.quantile(spread,.95)),reward_range_max=float(spread.max()))
        write_json(cli.out/'summary.json',summary);write_json(cli.out/'points.json',points)
        if not cli.smoke:plot_results(cli.out,points,len(cases))
        write_json(cli.out/'run.json',summary);print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:write_json(cli.out/'run.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':main()
