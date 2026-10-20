#!/usr/bin/env python3
"""Replay frozen cymbal/hi-hat plans under top-entry plus 5 mm side scoring."""
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
from isaac_rl.hand_plan_dataset import load_dataset_batches
from isaac_rl.hand_plan_ranker import ACTION_BANK,HandPlanRanker,HandPlanRankerConfig
from scripts.compare_graph_neural_fresh_test import decisions
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-drum',type=int,default=16)
    p.add_argument('--device',default='cuda:0')
    cli=p.parse_args();torch.set_num_threads(2);cli.out.mkdir(parents=True,exist_ok=True)
    config=json.loads((cli.dataset/'config.json').read_text())
    source=rollout_source_config(json.loads((Path(config['source_dataset'])/'config.json').read_text()))
    decoder=Path(config['checkpoint'])
    if hashlib.sha256(decoder.read_bytes()).hexdigest()!=config['checkpoint_sha256']:
        raise ValueError('Decoder checkpoint changed')
    saved=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).eval()
    model.load_state_dict(saved['model'])
    batch=torch.load(cli.dataset/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['batch']
    data=load_dataset_batches(cli.dataset)
    choices,_,_=decisions(model,data,batch['panel'],batch['states'])
    selected=[];categories={}
    for drum in ('hihat','cymbal_left','cymbal_right'):
        candidates=[i for i,clip in enumerate(batch['panel'])
            if any(hit.drum_name==drum for hit in clip.hits)]
        taken=[i for i in candidates if i not in selected][:cli.per_drum]
        selected.extend(taken);categories[drum]=taken
    episodes=[assigned_clip(batch['panel'][i],ACTION_BANK[int(choices['neural_depth2'][i])])
        for i in selected]
    states=[batch['states'][i] for i in selected]
    args=runtime_args(['--out',str(cli.out),'--envs',str(len(episodes)),'--device',cli.device,
        '--rollout-devices',cli.device,'--physics-hz','250','--decode-chunk','512',
        '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed','--rollout-audit'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=config['args']['seed']
    evaluator=None
    try:
        evaluator=DistributedIsaacEvaluator(args,decoder,cli.dataset/'scene_seed.json')
        ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
        if ready['rollout_runtime_version']!='thick_cymbal_top_entry_side5mm_contact_v4':
            raise ValueError('Expected corrected top/side scoring runtime')
        reports=evaluator.evaluate(episodes,initial_states=states)
    finally:
        if evaluator is not None:evaluator.close()
    write_json(cli.out/'reports.json',reports)
    old_hits=np.asarray([data['hits'][i,int(choices['neural_depth2'][i])] for i in selected])
    new_hits=np.asarray([r['hits'] for r in reports])
    requested=np.asarray([r['requested'] for r in reports])
    old_by_drum={drum:dict(requested=0,hits=0) for drum in ('hihat','cymbal_left','cymbal_right')}
    new_by_drum={drum:dict(requested=0,hits=0) for drum in old_by_drum}
    source_reports=json.loads((cli.dataset/'batches/episodes_00001.json').read_text())
    lookup={(int(i),int(a)):report for (i,a),report in zip(batch['labels'],source_reports)}
    for i,report in zip(selected,reports):
        action=int(choices['neural_depth2'][i]);before=lookup[(i,action)]
        for drum in old_by_drum:
            for target,row in ((old_by_drum,before),(new_by_drum,report)):
                target[drum]['requested']+=row['by_drum'][drum]['requested']
                target[drum]['hits']+=row['by_drum'][drum]['hits']
    stats=reports[-1]['rollout_audit']['contact_tracker_stats']
    result=dict(cases=len(selected),categories=categories,case_ids=selected,
        old_label_runtime=config['rollout_runtime'],new_rollout_runtime=ready['rollout_runtime_version'],
        old_hits=int(old_hits.sum()),new_hits=int(new_hits.sum()),requested=int(requested.sum()),
        changed_case_hit_counts=int((old_hits!=new_hits).sum()),
        old_by_drum=old_by_drum,new_by_drum=new_by_drum,
        contact_tracker_stats=stats,
        contact_debug_events=reports[-1]['rollout_audit']['contact_tracker_debug_events'],
        same_initial_states_and_plans=True,diagnostic_only=True)
    write_json(cli.out/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('cases','old_hits','new_hits','requested',
        'changed_case_hit_counts','old_by_drum','new_by_drum','contact_tracker_stats')},indent=2))


if __name__=='__main__':main()
