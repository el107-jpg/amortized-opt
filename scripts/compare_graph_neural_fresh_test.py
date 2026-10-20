#!/usr/bin/env python3
"""Compare frozen neural and graph hand plans on the same cached Isaac outcomes."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.hand_assign import LIMB_LEFT,kit_graph_from_layout
from isaac_rl.assignment_search import hit_groups
from isaac_rl.hand_plan_dataset import load_dataset_batches,require_matching_rollout_runtime
from isaac_rl.hand_plan_ranker import ACTION_BANK,HandPlanRanker,HandPlanRankerConfig
from isaac_rl.min_time_assignment import graph_actions
from scripts.evaluate_fresh_hand_plan_test import old_test_batches
from scripts.train_chart_cma import write_json


def decisions(model,data,panel,states):
    if len(panel)!=len(data['observations']) or len(states)!=len(panel):raise ValueError('Case inputs differ')
    device=next(model.parameters()).device
    scores=[]
    with torch.inference_mode():
        for offset in range(0,len(panel),1024):
            scores.append(model(torch.as_tensor(data['observations'][offset:offset+1024],device=device)).cpu().numpy())
    scores=np.concatenate(scores)
    valid=data['observed'];neural=np.where(valid,scores,-np.inf).argmax(1)
    original=kit_graph_from_layout()
    modern=replace(original,banned={LIMB_LEFT:original.index['floor_tom']})
    choices={'neural_depth2':neural};repairs={};fallbacks={}
    for name,graph in (('original_graph',original),('graph_13_combinations',modern)):
        actions=[];repair_count=fallback_count=0
        for row,(clip,state) in enumerate(zip(panel,states)):
            bits,fallback=graph_actions(hit_groups(clip),
                initial_drums=tuple(state['initial_drums']),graph=graph)
            fallback_count+=fallback
            code=int(bits@np.array([8,4,2,1]))
            if not valid[row,code]:
                allowed=np.flatnonzero(valid[row])
                code=int(allowed[np.argmin((ACTION_BANK[allowed]!=bits).sum(1))])
                repair_count+=1
            actions.append(code)
        choices[name]=np.asarray(actions);repairs[name]=repair_count;fallbacks[name]=fallback_count
    return choices,repairs,fallbacks


def method_metrics(data,choices):
    rows=np.arange(len(choices));valid=data['observed']
    if not valid[rows,choices].all():raise ValueError('Method selected an unobserved plan')
    hits=data['hits'][rows,choices];requested=data['requested'][rows,choices]
    rewards=data['rewards'][rows,choices]
    best=np.where(valid,data['rewards'],-np.inf).max(1)
    return dict(cases=len(rows),hits=int(hits.sum()),requested=int(requested.sum()),
        hit_rate=float(hits.sum()/requested.sum()),mean_reward=float(rewards.mean()),
        mean_reward_regret=float((best-rewards).mean()),
        fully_hit_cases=int((hits==requested).sum())),hits,requested


def paired_interval(a_hits,a_req,b_hits,b_req,seed=20261034,repetitions=4000):
    rng=np.random.default_rng(seed);differences=np.empty(repetitions)
    for i in range(repetitions):
        ids=rng.integers(len(a_hits),size=len(a_hits))
        differences[i]=a_hits[ids].sum()/a_req[ids].sum()-b_hits[ids].sum()/b_req[ids].sum()
    return [float(x) for x in np.quantile(differences,[.025,.975])]


def compare(model,data,panel,states):
    choices,repairs,fallbacks=decisions(model,data,panel,states)
    methods={};counts={}
    for name,action in choices.items():
        methods[name],hits,requested=method_metrics(data,action)
        counts[name]=(hits,requested)
    neural_hits,neural_requested=counts['neural_depth2']
    comparisons={}
    for name in ('original_graph','graph_13_combinations'):
        graph_hits,graph_requested=counts[name]
        comparisons[name]=dict(hit_rate_difference_pp=100*(methods['neural_depth2']['hit_rate']-methods[name]['hit_rate']),
            paired_case_bootstrap_95pct_difference_pp=[100*x for x in paired_interval(
                neural_hits,neural_requested,graph_hits,graph_requested)],
            neural_better_cases=int((neural_hits>graph_hits).sum()),
            graph_better_cases=int((neural_hits<graph_hits).sum()),
            tied_hit_count_cases=int((neural_hits==graph_hits).sum()),
            different_hand_plans=int((choices['neural_depth2']!=choices[name]).sum()))
    legal=data['observed'];oracle=np.where(legal,data['hits'],-1).argmax(1)
    methods['best_observed_hit_plan'],_,_=method_metrics(data,oracle)
    return dict(methods=methods,comparisons=comparisons,
        graph_eligibility_repairs=repairs,graph_fallback_groups=fallbacks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-dataset',type=Path,required=True)
    p.add_argument('--new-test',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    cli=p.parse_args();torch.set_num_threads(2)
    old_cfg=json.loads((cli.old_dataset/'config.json').read_text())
    new_cfg=json.loads((cli.new_test/'config.json').read_text())
    if json.loads((cli.new_test/'run.json').read_text())['status']!='complete':raise ValueError('Fresh test incomplete')
    require_matching_rollout_runtime(old_cfg,new_cfg)
    saved=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    if (Path(saved['dataset']).resolve()!=cli.old_dataset.resolve()
            or saved['checkpoint_sha256']!=new_cfg['checkpoint_sha256']
            or saved['network']['tip_geometry']!=new_cfg['tip_geometry']):
        raise ValueError('Model and cached outcomes do not share decoder/geometry')
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).to(cli.device)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    old_data=old_test_batches(cli.old_dataset)
    old_inputs=torch.load(cli.old_dataset/'heldout_inputs.pt',map_location='cpu',weights_only=False)['test']
    if not np.array_equal(old_data['observations'],old_inputs['observations']):
        raise ValueError('Old held-out input order differs')
    new_data=load_dataset_batches(cli.new_test)
    new_rows=[torch.load(path,map_location='cpu',weights_only=False)['batch']
        for path in sorted((cli.new_test/'batches').glob('batch_*.pt'))]
    panel=[clip for batch in new_rows for clip in batch['panel']]
    states=[state for batch in new_rows for state in batch['states']]
    if not np.array_equal(new_data['observations'],np.concatenate([b['observations'] for b in new_rows])):
        raise ValueError('Fresh test case order differs')
    results=dict(format='matched_graph_neural_corrected_runtime_v1',
        checkpoint=str(cli.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
        checkpoint_step=saved['gradient_steps'],rollout_runtime=new_cfg['rollout_runtime'],
        old_fixed_test=compare(model,old_data,old_inputs['panel'],old_inputs['states']),
        fresh_corrected_start_test=compare(model,new_data,panel,states),
        same_cached_case_plan_outcomes=True,held_out_used_for_training=False,
        weights_selected_by_comparison=False,
        graph_history='Both graph variants receive the actual initial hand targets; illegal plans are repaired to the nearest legal assignment')
    cli.out.parent.mkdir(parents=True,exist_ok=True);write_json(cli.out,results)
    for split in ('old_fixed_test','fresh_corrected_start_test'):
        print(split)
        for name,metrics in results[split]['methods'].items():
            print(name,metrics['hits'],metrics['requested'],f'{100*metrics["hit_rate"]:.3f}%')
        print('differences',json.dumps(results[split]['comparisons']))


if __name__=='__main__':main()
