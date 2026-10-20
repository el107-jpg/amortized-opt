#!/usr/bin/env python3
"""Compare archived, prior, and control rankers on one alternating groove."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.constants import ready_joint_pos
from amortized_opt.rhythm import DrumKit,Hit,RhythmClip
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import DRUM_GEOMETRY_VERSION,GEOMETRY_VERSION
from isaac_rl.hand_plan_ranker import (ACTION_BANK,HandPlanRanker,
    HandPlanRankerConfig,cartesian_tips,encode_clip,legal_actions)
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def load_ranker(folder):
    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    if saved['format']!='offline_hand_plan_ranker_v1':raise ValueError('Expected archived ranking weights')
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network']))
    model.load_state_dict(saved['model']);model.eval()
    return model,saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--first-hit-s',type=float,default=.6)
    parser.add_argument('--gap-s',type=float,default=.24)
    parser.add_argument('--rollout-audit',action='store_true',help='Record v4 contact geometry and scoring diagnostics')
    args_cli=parser.parse_args()
    if args_cli.out.exists():raise FileExistsError(args_cli.out)
    args_cli.out.mkdir(parents=True);torch.set_num_threads(2)
    source_dataset=ROOT/'runs/hand_plan_dataset_v4_prefirst_200k_20261009'
    cfg=json.loads((source_dataset/'config.json').read_text())
    if cfg['simulation_geometry_version']!=GEOMETRY_VERSION:
        raise ValueError('Decoder/shared geometry changed')
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder checkpoint changed')
    source_cfg=json.loads((Path(cfg['source_dataset'])/'config.json').read_text())
    source=rollout_source_config(source_cfg)
    eligibility=MinimumTimeChart.load(cfg['eligibility_source'])
    kit=DrumKit.from_layout()
    times=[args_cli.first_hit_s+index*args_cli.gap_s for index in range(4)]
    drums=(1,0,1,0)
    clip=RhythmClip('hihat_snare_hihat_snare_125bpm',[
        Hit(float(t),float(t),d,kit.names[d],'left',.7) for t,d in zip(times,drums)],
        times[-1]+.3)
    geometry=cfg['network']['tip_geometry']
    q=np.asarray(ready_joint_pos(),np.float32);qd=np.zeros_like(q)
    state=dict(q=q.tolist(),qd=qd.tolist(),tips=cartesian_tips(q,geometry).tolist(),
        initial_drums=[-1,-1],kind='ready')
    observation=encode_clip(clip,q=q,qd=qd,tips=state['tips'],
        initial_drums=state['initial_drums'],tip_geometry=geometry)
    legal=legal_actions(clip,eligibility)
    ids=np.flatnonzero(legal)
    if len(ids)!=16:raise ValueError('This groove should have16 legal assignments')
    run_names=dict(archived='neural_hand_plans_100k_wide_wd01_dropout15_20261008',
        prior='neural_hand_plan_ranker_prior_100k_20261009',
        control='neural_hand_plan_ranker_control_100k_20261009')
    predictions={}
    for label,name in run_names.items():
        folder=ROOT/'runs'/name;model,saved=load_ranker(folder)
        with torch.inference_mode():scores=model(torch.as_tensor(observation[None])).numpy()[0]
        order=np.argsort(-np.where(legal,scores,-np.inf),kind='stable')
        predictions[label]=dict(run=str(folder),checkpoint_step=saved['gradient_steps'],
            chosen_action=int(order[0]),canonical_rank=int(np.flatnonzero(order==10)[0])+1,
            ranking=[dict(action=int(i),hands=''.join('R' if bit else 'L' for bit in ACTION_BANK[i]),
                score=float(scores[i])) for i in order])
    write_json(args_cli.out/'predictions.json',dict(rhythm=dict(times_s=times,
        drums=[kit.names[d] for d in drums],normalized_loudness=.7,
        start='ready q, zero qd, no previous hand targets'),models=predictions))
    episodes=[assigned_clip(clip,ACTION_BANK[int(i)]) for i in ids]
    seed_file=args_cli.out/'scene_seed.json'
    write_json(seed_file,StochasticHandRhythmSampler(kit,4,eligibility).seed_payload())
    runtime_tokens=['--out',str(args_cli.out),'--envs',str(len(ids)),
        '--device',args_cli.device,'--rollout-devices',args_cli.device,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024',
        '--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed']
    if args_cli.rollout_audit:runtime_tokens.append('--rollout-audit')
    runtime=runtime_args(runtime_tokens)
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(runtime,key,source['args'][key])
    runtime.groups=runtime.horizon=4;runtime.seed=20261023
    evaluator=None
    try:
        evaluator=DistributedIsaacEvaluator(runtime,checkpoint,seed_file)
        ready=[json.loads((folder/'ready.json').read_text()) for folder,_ in evaluator.workers]
        for worker in ready:
            if (worker['rollout_runtime_version']!='thick_cymbal_top_entry_side5mm_contact_v4'
                    or worker['drum_geometry_version']!=DRUM_GEOMETRY_VERSION
                    or worker['tip_geometry']!=geometry
                    or worker['drum_usd_sha256']!=cfg['drum_usd_sha256']
                    or worker['drum_layout_sha256']!=cfg['drum_layout_sha256']):
                raise ValueError('Worker does not match v4 data/runtime')
        reports=evaluator.evaluate(episodes,lambda done,total,tick,ticks:
            write_json(args_cli.out/'progress.json',dict(done=done,total=total,tick=tick,ticks=ticks)),
            initial_states=[dict(state) for _ in ids])
        write_json(args_cli.out/'episodes.json',reports)
        outcomes={int(action):dict(hands=''.join('R' if bit else 'L' for bit in ACTION_BANK[action]),
            hits=report['hits'],requested=report['requested'],reward=report['reward'],
            hit_rate=report['hit_rate'],per_hit=report['per_hit_outcomes'])
            for action,report in zip(ids,reports)}
        best_reward=max(outcomes,key=lambda action:outcomes[action]['reward'])
        best_hits=max(outcomes,key=lambda action:(outcomes[action]['hits'],outcomes[action]['reward']))
        result=dict(status='complete',rollout_runtime=ready[0]['rollout_runtime_version'],
            times_s=times,first_hit_s=args_cli.first_hit_s,gap_s=args_cli.gap_s,
            start_state=state,all_legal_assignments=outcomes,
            best_reward_action=best_reward,best_hits_action=best_hits,
            methods={name:dict(**prediction,observed=outcomes[prediction['chosen_action']])
                for name,prediction in predictions.items()})
        if args_cli.rollout_audit:
            result['contact_tracker_stats']=reports[-1]['rollout_audit']['contact_tracker_stats']
            result['contact_debug_events']=reports[-1]['rollout_audit']['contact_tracker_debug_events']
        write_json(args_cli.out/'summary.json',result)
        write_json(args_cli.out/'run.json',dict(status='complete',
            chosen={name:dict(action=value['chosen_action'],hands=value['observed']['hands'],
                hits=value['observed']['hits'],reward=value['observed']['reward'],
                canonical_rank=value['canonical_rank']) for name,value in result['methods'].items()},
            best_reward_action=best_reward))
        print(json.dumps(json.loads((args_cli.out/'run.json').read_text())),flush=True)
    except BaseException as exc:
        write_json(args_cli.out/'run.json',dict(status='failed',error=repr(exc)))
        raise
    finally:
        if evaluator is not None:evaluator.close()


if __name__=='__main__':main()
