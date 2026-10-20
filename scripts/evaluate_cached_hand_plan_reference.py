#!/usr/bin/env python3
"""Freeze an old policy and evaluate it on new labels for the same held-out inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.hand_plan_dataset import load_dataset_batches
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig
from scripts.train_hand_plan_dataset import evaluate
from scripts.train_chart_cma import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--model-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(2)
    config=json.loads((args.dataset/'config.json').read_text())
    if json.loads((args.dataset/'run.json').read_text())['status']!='complete':raise ValueError('Dataset collection is incomplete')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    snapshot=args.out.with_suffix('.pt');snapshot.write_bytes((args.model_run/'last.pt').read_bytes())
    saved=torch.load(snapshot,map_location='cpu',weights_only=False)
    if saved['checkpoint_sha256']!=config['checkpoint_sha256'] or saved['simulation_geometry_version']!=config['simulation_geometry_version']:
        raise ValueError('Reference decoder/geometry differ')
    if saved['network']['tip_geometry']!=config['network']['tip_geometry']:raise ValueError('Reference observation geometry differs')
    data=load_dataset_batches(args.dataset);splits=torch.load(args.dataset/'splits.pt',weights_only=False)
    old_folder=Path(saved['dataset']);old_data=load_dataset_batches(old_folder)
    old_splits=torch.load(old_folder/'splits.pt',weights_only=False)
    for name in ('validation','test'):
        if not np.array_equal(data['observations'][splits[name]],old_data['observations'][old_splits[name]]):
            raise ValueError(f'Reference {name} inputs differ')
        if not np.array_equal(data['observed'][splits[name]],old_data['observed'][old_splits[name]]):
            raise ValueError(f'Reference {name} legal plans differ')
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).to(args.device)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    result=dict(format='frozen_hand_plan_reference_v1',dataset=str(args.dataset.resolve()),
        model_run=str(args.model_run.resolve()),checkpoint_snapshot=str(snapshot.resolve()),
        reference_weights_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        model_gradient_steps=saved['gradient_steps'],model_training_dataset=str(old_folder.resolve()),
        model_training_runtime=saved.get('rollout_runtime',{'rollout_runtime_version':'legacy_contact_reset_v0'}),
        rollout_runtime=config['rollout_runtime'],heldout_inputs_match=True,
        metrics={name:evaluate(model,data,splits[name]) for name in ('validation','test')},
        weights_selected_by_evaluation=False,held_out_used_for_training=False,
        comparison='Frozen previous policy scored on freshly relabeled matched held-out inputs; both data volume and label runtime changed')
    write_json(args.out,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
