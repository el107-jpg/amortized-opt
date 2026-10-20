#!/usr/bin/env python3
"""Evaluate one frozen ranker on old and new corrected-runtime test cases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.hand_plan_dataset import load_dataset_batches,require_matching_rollout_runtime
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig
from scripts.train_chart_cma import write_json
from scripts.train_hand_plan_dataset import evaluate


def old_test_batches(folder):
    files=[]
    for path in sorted((folder/'batches').glob('metrics_*.json')):
        if json.loads(path.read_text())['split']=='test':
            files.append(folder/'batches'/path.name.replace('metrics_','batch_').replace('.json','.pt'))
    if not files:raise ValueError('Old dataset has no fixed test batches')
    rows=[torch.load(path,map_location='cpu',weights_only=False) for path in files]
    data={}
    for key in ('observations','observed'):
        data[key]=np.concatenate([row['batch'][key] for row in rows])
    for key in ('rewards','hit_rates','hits','requested'):
        data[key]=np.concatenate([row['outcomes'][key] for row in rows])
    return data


@torch.inference_mode()
def chosen_counts(model,data):
    device=next(model.parameters()).device;scores=[]
    for offset in range(0,len(data['observations']),1024):
        scores.append(model(torch.as_tensor(data['observations'][offset:offset+1024],device=device)).cpu().numpy())
    scores=np.concatenate(scores)
    choice=np.where(data['observed'],scores,-np.inf).argmax(1)
    rows=np.arange(len(choice))
    return data['hits'][rows,choice],data['requested'][rows,choice],choice


def bootstrap_hit_rate(old_hits,old_requested,new_hits,new_requested,seed=20261033,repetitions=4000):
    rng=np.random.default_rng(seed)
    old_rates=np.empty(repetitions);new_rates=np.empty(repetitions)
    for index in range(repetitions):
        old_ids=rng.integers(len(old_hits),size=len(old_hits))
        new_ids=rng.integers(len(new_hits),size=len(new_hits))
        old_rates[index]=old_hits[old_ids].sum()/old_requested[old_ids].sum()
        new_rates[index]=new_hits[new_ids].sum()/new_requested[new_ids].sum()
    return dict(seed=seed,repetitions=repetitions,
        old_95pct=np.quantile(old_rates,[.025,.975]).tolist(),
        new_95pct=np.quantile(new_rates,[.025,.975]).tolist(),
        new_minus_old_95pct=np.quantile(new_rates-old_rates,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-dataset',type=Path,required=True)
    p.add_argument('--new-test',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    cli=p.parse_args();torch.set_num_threads(2)
    old_config=json.loads((cli.old_dataset/'config.json').read_text())
    new_config=json.loads((cli.new_test/'config.json').read_text())
    new_status=json.loads((cli.new_test/'run.json').read_text())
    if new_status['status']!='complete' or new_config['purpose']!='test_only_no_training':
        raise ValueError('New untouched test collection is incomplete')
    require_matching_rollout_runtime(old_config,new_config)
    saved=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    if (Path(saved['dataset']).resolve()!=cli.old_dataset.resolve()
            or saved['checkpoint_sha256']!=new_config['checkpoint_sha256']
            or saved['simulation_geometry_version']!=new_config['simulation_geometry_version']
            or saved['network']['tip_geometry']!=new_config['tip_geometry']):
        raise ValueError('Frozen model, decoder, geometry or dataset differs')
    require_matching_rollout_runtime(saved,new_config)
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).to(cli.device)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    old_data=old_test_batches(cli.old_dataset)
    original=torch.load(cli.old_dataset/'heldout_inputs.pt',map_location='cpu',weights_only=False)['test']
    if not np.array_equal(old_data['observations'],original['observations']):
        raise ValueError('Old test inputs differ from their fixed source')
    new_data=load_dataset_batches(cli.new_test)
    if len(new_data['observations'])!=new_status['cases']:raise ValueError('Fresh test case count differs')
    old_metrics=evaluate(model,old_data,np.arange(len(old_data['observations'])))
    new_metrics=evaluate(model,new_data,np.arange(len(new_data['observations'])))
    old_hits,old_requested,_=chosen_counts(model,old_data)
    new_hits,new_requested,_=chosen_counts(model,new_data)
    bootstrap=bootstrap_hit_rate(old_hits,old_requested,new_hits,new_requested)
    def oracle(data):
        return float(np.where(data['observed'],data['hit_rates'],-np.inf).max(1).mean())
    result=dict(format='fresh_corrected_runtime_test_comparison_v1',
        checkpoint=str(cli.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(cli.checkpoint.read_bytes()).hexdigest(),
        model_gradient_steps=saved['gradient_steps'],training_dataset=str(cli.old_dataset.resolve()),
        fresh_test_dataset=str(cli.new_test.resolve()),fresh_test_bank_provenance=new_config['bank_provenance'],
        rollout_runtime=new_config['rollout_runtime'],held_out_used_for_training=False,
        weights_selected_by_new_test=False,
        old_fixed_test=dict(metrics=old_metrics,oracle_mean_hit_rate=oracle(old_data)),
        fresh_test=dict(metrics=new_metrics,oracle_mean_hit_rate=oracle(new_data)),
        differences=dict(hit_rate_pp=100*(new_metrics['hit_rate']-old_metrics['hit_rate']),
            test_loss=new_metrics['loss']-old_metrics['loss'],
            reward_regret=new_metrics['mean_reward_regret']-old_metrics['mean_reward_regret'],
            oracle_mean_hit_rate_pp=100*(oracle(new_data)-oracle(old_data))),
        bootstrap=bootstrap)
    cli.out.parent.mkdir(parents=True,exist_ok=True);write_json(cli.out,result)
    print(json.dumps(dict(checkpoint_step=saved['gradient_steps'],old_cases=old_metrics['cases'],
        fresh_cases=new_metrics['cases'],old_hit_rate=old_metrics['hit_rate'],
        fresh_hit_rate=new_metrics['hit_rate'],differences=result['differences'],bootstrap=bootstrap),indent=2))


if __name__=='__main__':main()
