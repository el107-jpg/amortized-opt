"""Exact training-case budgets and input-only reuse of fixed held-out cases."""
import json
from pathlib import Path
import numpy as np
import torch
from isaac_rl.hand_plan_ranker import encode_clip


def collection_tasks(training_cases,contexts,heldout_counts):
    if min(training_cases,contexts)<1:raise ValueError('Positive case budgets required')
    tasks=[];offset=0
    for split,count in [('train',training_cases),*heldout_counts.items()]:
        for start in range(0,count,contexts):
            size=min(contexts,count-start)
            tasks.append(dict(split=split,count=size,split_offset=start,case_offset=offset))
            offset+=size
    return tasks


def task_splits(tasks):
    result={}
    for name in ('train','validation','test'):
        parts=[np.arange(t['case_offset'],t['case_offset']+t['count']) for t in tasks if t['split']==name]
        result[name]=np.concatenate(parts) if parts else np.array([],dtype=np.int64)
    return result


def input_batches(folder):
    folder=Path(folder)
    if (folder/'manifest.json').exists():
        for source in json.loads((folder/'manifest.json').read_text())['source_datasets']:
            yield from input_batches(source)
    else:
        for path in sorted((folder/'batches').glob('batch_*.pt')):
            # Only panel/state/observation fields are used; cached outcomes are ignored.
            yield torch.load(path,map_location='cpu',weights_only=False)['batch']


def heldout_inputs(folder,geometry):
    splits=torch.load(Path(folder)/'splits.pt',weights_only=False)
    ids={name:np.asarray(splits[name],np.int64) for name in ('validation','test')}
    if any(not len(v) for v in ids.values()):raise ValueError('Both held-out input splits required')
    if set(ids['validation'])&set(ids['test']):raise ValueError('Held-out cases overlap')
    wanted=set(np.r_[ids['validation'],ids['test']].tolist());found={};offset=0
    for batch in input_batches(folder):
        for index in sorted(wanted.intersection(range(offset,offset+len(batch['panel'])))):
            local=index-offset;clip=batch['panel'][local];state=batch['states'][local]
            encoded=encode_clip(clip,tip_geometry=geometry,
                **{key:state[key] for key in ('q','qd','initial_drums','tips')})
            if not np.array_equal(encoded,batch['observations'][local]):raise ValueError('Held-out input reconstruction differs')
            found[index]=(clip,state,encoded)
        offset+=len(batch['panel'])
        if len(found)==len(wanted):break
    if len(found)!=len(wanted):raise ValueError('Missing held-out input cases')
    return {name:dict(panel=[found[int(i)][0] for i in values],states=[found[int(i)][1] for i in values],
        observations=np.stack([found[int(i)][2] for i in values]),source_case_ids=values)
        for name,values in ids.items()}
