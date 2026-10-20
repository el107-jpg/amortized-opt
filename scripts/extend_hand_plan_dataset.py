#!/usr/bin/env python3
"""Combine fresh training-only cases with existing data; freeze old held-out splits."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from isaac_rl.hand_plan_dataset import extend_case_splits,load_dataset_batches,require_matching_rollout_runtime
from scripts.train_neural_hand_plans import save_torch
from scripts.train_chart_cma import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--additional',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);cli=p.parse_args()
    if cli.out.exists():raise FileExistsError(cli.out)
    a=json.loads((cli.base/'config.json').read_text());b=json.loads((cli.additional/'config.json').read_text())
    ar=json.loads((cli.base/'run.json').read_text());br=json.loads((cli.additional/'run.json').read_text())
    if ar['status']!='complete' or br['status']!='complete':raise ValueError('Both collections must finish')
    require_matching_rollout_runtime(a,b)
    for key in ('checkpoint_sha256','simulation_geometry_version','network','first_hit_policy','rhythm_sampling','start_mixture'):
        if a[key]!=b[key]:raise ValueError(f'Dataset mismatch: {key}')
    if not b.get('training_only'):raise ValueError('Additional cases must be explicitly training-only')
    old=torch.load(cli.base/'splits.pt',weights_only=False)
    base_data=load_dataset_batches(cli.base);new_data=load_dataset_batches(cli.additional)
    if len(base_data['observations'])!=ar['cases'] or len(new_data['observations'])!=br['cases']:
        raise ValueError('Stored case counts differ from completed collection metadata')
    heldout=list(old['validation'])+list(old['test'])
    signatures={base_data['observations'][i].tobytes() for i in heldout}
    if any(row.tobytes() in signatures for row in new_data['observations']):
        raise ValueError('Additional training case duplicates an original held-out observation')
    splits=extend_case_splits(old,ar['cases'],br['cases'])
    cli.out.mkdir(parents=True)
    manifest=dict(source_datasets=[str(cli.base.resolve()),str(cli.additional.resolve())],
        base_cases=ar['cases'],additional_training_cases=br['cases'],held_out_unchanged=True)
    write_json(cli.out/'manifest.json',manifest)
    cfg=dict(a,algorithm='Expanded IID dataset with frozen original held-out cases',
        cases=ar['cases']+br['cases'],split='Original validation/test frozen; all new cases training-only',
        training_only=False,source_datasets=manifest['source_datasets'])
    write_json(cli.out/'config.json',cfg);save_torch(cli.out/'splits.pt',splits)
    shutil.copy2(cli.additional/'post_hit_bank.pt',cli.out/'post_hit_bank.pt')
    write_json(cli.out/'run.json',dict(status='complete',cases=ar['cases']+br['cases'],
        rollouts=ar['rollouts']+br['rollouts'],splits={key:len(ids) for key,ids in splits.items()},
        held_out_unchanged=True))
    print(json.dumps(dict(cases=ar['cases']+br['cases'],splits={key:len(ids) for key,ids in splits.items()})))


if __name__=='__main__':main()
