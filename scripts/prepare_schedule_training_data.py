#!/usr/bin/env python3
"""Combine exhaustive-search labels and fresh all-action online outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import amortized_opt as A
from scripts.train_four_hit_schedule import graph_indices, write_json


def prepare(base, online, out):
    if out.exists():
        raise FileExistsError(out)
    original = json.loads((base/'config.json').read_text())
    additional = json.loads((online/'config.json').read_text())
    if additional['algorithm'] != 'expected':
        raise ValueError('Only complete all-16 online labels can be merged')
    if Path(original['checkpoint']).resolve() != Path(additional['checkpoint']).resolve():
        raise ValueError('Generators differ between label sources')
    if json.loads((online/'run.json').read_text())['status'] != 'complete':
        raise ValueError('Online source has not finished')
    out.mkdir(parents=True); (out/'data/train').mkdir(parents=True)
    for split in ('validation','test'):
        (out/'data'/split).symlink_to((base/'data'/split).resolve(), target_is_directory=True)
    total = 0; transition_count = 0; counts = torch.zeros(7,dtype=torch.long); ids = set()
    for path in sorted((base/'data/train').glob('*.pt')):
        row = torch.load(path,map_location='cpu',weights_only=False)
        if ids.intersection(row['root_ids']):
            raise ValueError('Duplicate base roots')
        ids.update(row['root_ids']); total += len(row['features'])
        transition_count += len(row['transitions']['obs']) if 'transitions' in row else 0
        counts += row['drum_counts']
        (out/'data/train'/('base_'+path.name)).symlink_to(path.resolve())
    heldout = set()
    for split in ('validation','test'):
        for path in sorted((base/'data'/split).glob('*.pt')):
            heldout.update(torch.load(path,map_location='cpu',weights_only=False)['root_ids'])
    kit = A.DrumKit.from_layout(); fresh = 0
    for path in sorted((online/'online').glob('batch_*.pt')):
        row = torch.load(path,map_location='cpu',weights_only=False)
        n = len(row['features'])
        if not torch.equal(row['indices'],torch.arange(16)[None].expand(n,-1)):
            raise ValueError('Expected every action exactly once, in binary schedule order')
        if ids.intersection(row['root_ids']) or heldout.intersection(row['root_ids']):
            raise ValueError('Fresh roots overlap an existing split')
        torch.testing.assert_close(row['rewards'],row['physics']['score'].double().sum(-1).mean(-1).float())
        hits = row['features'][:,46:].reshape(n,4,10)
        assert hits[...,9].bool().all()
        drum_counts = torch.bincount(hits[...,:7].argmax(-1).flatten(),minlength=7)
        data = dict(features=row['features'],values=row['rewards'],chosen=row['rewards'].argmax(-1),
            graph_index=graph_indices(row['clips'],kit,row['features']),q=row['q'],v=row['v'],
            clips=row['clips'],draws=row['draws'],first_hit_s=hits[:,0,7],drum_counts=drum_counts,
            root_ids=row['root_ids'],start_stats=row['start_stats'])
        torch.save(data,out/'data/train'/('online_'+path.name))
        ids.update(row['root_ids']); total += n; fresh += n; counts += drum_counts
    if ids.intersection(heldout):
        raise ValueError('Train roots leaked into heldout splits')
    config = dict(original,out=str(out),train_windows=total,dataset_only=True,
        source_runs=[str(base),str(online)],new_exhaustive_windows=fresh,
        suffix_transition_source=str(base/'data/train'))
    write_json(out/'config.json',config)
    manifest = dict(training_windows=total,additional_fresh_windows=fresh,transition_records=transition_count,
        drum_counts=counts.tolist(),all_root_ids_unique=len(ids)==total,
        heldout_splits_unchanged=True,root_labels='Original and fresh complete all-16 measured scores',
        bellman_transitions='Original measured suffix transitions; fresh online run did not save suffix poses')
    write_json(out/'data_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'runs/four_hit_schedule_distillation_20261004')
    p.add_argument('--online',type=Path,default=ROOT/'runs/four_hit_schedule_expected_online_20261004')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    prepare(args.base.resolve(),args.online.resolve(),args.out.resolve())


if __name__=='__main__':
    main()
