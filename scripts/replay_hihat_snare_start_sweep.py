#!/usr/bin/env python3
"""Matched v4 Isaac outcomes for ranker choices on 256 identical groove starts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import DRUM_GEOMETRY_VERSION,GEOMETRY_VERSION
from isaac_rl.hand_plan_ranker import ACTION_BANK
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
from isaac_rl.min_time_assignment import MinimumTimeChart
from amortized_opt.rhythm import DrumKit
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    cli=parser.parse_args()
    if cli.out.exists():raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True);torch.set_num_threads(2)
    saved=torch.load(cli.sweep,map_location='cpu',weights_only=False)
    clip=saved['clip'];states=saved['states'];choices=saved['choices']
    if len(states)!=256 or set(choices)!={'archived','control','prior'}:
        raise ValueError('Expected256 matched starts and three rankers')
    dataset=ROOT/'runs/hand_plan_dataset_v4_prefirst_200k_20261009'
    cfg=json.loads((dataset/'config.json').read_text())
    if cfg['simulation_geometry_version']!=GEOMETRY_VERSION:
        raise ValueError('Decoder geometry changed')
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    source_cfg=json.loads((Path(cfg['source_dataset'])/'config.json').read_text())
    source=rollout_source_config(source_cfg)
    eligibility=MinimumTimeChart.load(cfg['eligibility_source'])
    seed_file=cli.out/'scene_seed.json'
    write_json(seed_file,StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility).seed_payload())
    args=runtime_args(['--out',str(cli.out),'--envs',str(len(states)),
        '--device',cli.device,'--rollout-devices',cli.device,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024',
        '--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=20261023
    outcome={};raw={}
    try:
        for label in ('archived','control','prior'):
            episodes=[assigned_clip(clip,ACTION_BANK[int(action)]) for action in choices[label]]
            evaluator=None
            try:
                evaluator=DistributedIsaacEvaluator(args,checkpoint,seed_file)
                ready=[json.loads((folder/'ready.json').read_text()) for folder,_ in evaluator.workers]
                for worker in ready:
                    if (worker['rollout_runtime_version']!='thick_cymbal_top_entry_side5mm_contact_v4'
                            or worker['drum_geometry_version']!=DRUM_GEOMETRY_VERSION
                            or worker['tip_geometry']!=cfg['network']['tip_geometry']
                            or worker['drum_usd_sha256']!=cfg['drum_usd_sha256']
                            or worker['drum_layout_sha256']!=cfg['drum_layout_sha256']):
                        raise ValueError('Worker does not match v4 scored geometry')
                reports=evaluator.evaluate(episodes,lambda done,total,tick,ticks:
                    write_json(cli.out/'progress.json',dict(method=label,done=done,total=total,
                        tick=tick,ticks=ticks)),initial_states=states)
            finally:
                if evaluator is not None:evaluator.close()
            write_json(cli.out/f'episodes_{label}.json',reports)
            hits=np.asarray([r['hits'] for r in reports],np.int64)
            reward=np.asarray([r['reward'] for r in reports],np.float32)
            requested=np.asarray([r['requested'] for r in reports],np.int64)
            by_hit=np.asarray([[int(row['hit']) for row in r['per_hit_outcomes']]
                for r in reports],np.int64)
            if not np.array_equal(requested,np.full(len(states),4)):
                raise ValueError('Changed requested hits')
            raw[label]=dict(hits=hits,reward=reward,by_hit=by_hit)
            outcome[label]=dict(hits=int(hits.sum()),requested=int(requested.sum()),
                hit_rate=float(hits.sum()/requested.sum()),mean_reward=float(reward.mean()),
                per_hit_credit=by_hit.mean(0).tolist(),
                by_drum={drum:dict(requested=sum(r['by_drum'][drum]['requested'] for r in reports),
                    hits=sum(r['by_drum'][drum]['hits'] for r in reports))
                    for drum in ('snare','hihat')})
            write_json(cli.out/'partial_summary.json',dict(status='running',methods=outcome))
            print('[sweep] '+json.dumps(dict(method=label,**outcome[label])),flush=True)
        change=choices['prior']!=choices['control']
        delta_hits=raw['prior']['hits']-raw['control']['hits']
        delta_reward=raw['prior']['reward']-raw['control']['reward']
        matched=~change
        result=dict(status='complete',runtime='thick_cymbal_top_entry_side5mm_contact_v4',
            physics_hz=250,states=len(states),matched_slot_and_state=True,
            methods=outcome,prior_vs_control=dict(changed_assignments=int(change.sum()),
                mean_hit_difference=float(delta_hits.mean()),
                mean_reward_difference=float(delta_reward.mean()),
                changed_only=dict(cases=int(change.sum()),
                    prior_more_hits=int((delta_hits[change]>0).sum()),
                    control_more_hits=int((delta_hits[change]<0).sum()),
                    tied_hits=int((delta_hits[change]==0).sum()),
                    mean_hit_difference=float(delta_hits[change].mean()) if change.any() else 0.,
                    mean_reward_difference=float(delta_reward[change].mean()) if change.any() else 0.),
                identical_assignment_repeatability=dict(cases=int(matched.sum()),
                    hit_counts_differ=int((delta_hits[matched]!=0).sum()),
                    rewards_differ_over_1e_3=int((np.abs(delta_reward[matched])>.001).sum()))))
        write_json(cli.out/'summary.json',result)
        write_json(cli.out/'run.json',result)
        print('[sweep-result] '+json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',error=repr(exc),methods=outcome))
        raise


if __name__=='__main__':main()
