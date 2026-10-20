#!/usr/bin/env python3
"""Read-only matched v4 motif diagnostics for human-prior and control runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.hand_plan_ranker import ACTION_BANK,HandPlanRanker,HandPlanRankerConfig
from isaac_rl.hand_plan_dataset import dataset_metrics
from isaac_rl.hand_plan_value import DualRuntimeRewardModel,load_value_dataset
from isaac_rl.hand_plan_collocation_priors import heuristic_plan_values
from scripts.train_chart_cma import write_json


def motif_status(clip,action):
    original=hit_groups(clip)
    assigned=hit_groups(assigned_clip(clip,ACTION_BANK[action]))
    result=[]
    for index,group in enumerate(original):
        if len(group)==2 and {hit.drum for hit in group}=={0,1}:
            result.append(('simultaneous',all(hit.hand==('left' if hit.drum==0 else 'right')
                for hit in assigned[index])))
    for index in range(len(original)-1):
        left,right=original[index:index+2]
        if len(left)==len(right)==1 and {left[0].drum,right[0].drum}=={0,1}:
            check=assigned[index][0],assigned[index+1][0]
            result.append(('consecutive',all(hit.hand==('left' if hit.drum==0 else 'right')
                for hit in check)))
    return result


@torch.inference_mode()
def scores(model,observations):
    result=[];model.eval()
    for start in range(0,len(observations),512):
        result.append(model(torch.as_tensor(observations[start:start+512])).numpy())
    return np.concatenate(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='append',nargs=2,metavar=('LABEL','PATH'),required=True)
    cli=parser.parse_args();torch.set_num_threads(2)
    data=load_value_dataset(cli.dataset);cases=data['cases']
    panels=[]
    for path in sorted((cli.dataset/'batches').glob('batch_*.pt')):
        panels.extend(torch.load(path,map_location='cpu',weights_only=False)['batch']['panel'])
    if len(panels)!=len(cases['observations']):raise ValueError('Source panels and observations differ')
    result=dict(dataset=str(cli.dataset.resolve()),read_only=True,runs={})
    for label,folder in cli.run:
        folder=Path(folder);run=json.loads((folder/'run.json').read_text())
        if run['status'] not in ('complete','stopped_by_user','stopped_at_user_direction'):
            raise ValueError(f'{label} has no stable requested checkpoint')
        checkpoint=folder/'last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        network=HandPlanRankerConfig(**saved['network'])
        if saved['format']=='mixed_runtime_hand_plan_value_v1':
            model=DualRuntimeRewardModel(network)
        elif saved['format']=='offline_hand_plan_ranker_v1':
            model=HandPlanRanker(network)
        else:raise ValueError('Expected mixed value or old ranking checkpoint')
        model.load_state_dict(saved['model'])
        report=dict(step=saved.get('step',saved.get('gradient_steps')),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),splits={})
        for split in ('validation','test'):
            ids=np.asarray(data['splits'][split],np.int64)
            output=scores(model,cases['observations'][ids])
            valid=cases['observed'][ids]
            choice=np.where(valid,output,-np.inf).argmax(1)
            full={kind:dict(motifs=0,canonical=0,cases=set())
                for kind in ('simultaneous','consecutive')}
            switches=[];subset=[]
            for row,(case_id,action) in enumerate(zip(ids,choice)):
                status=motif_status(panels[int(case_id)],int(action))
                if status:subset.append(int(case_id))
                for kind,canonical in status:
                    full[kind]['motifs']+=1;full[kind]['canonical']+=int(canonical)
                    full[kind]['cases'].add(int(case_id))
                state=cases['observations'][case_id,80:]
                previous=(int(state[34:42].argmax())-1,int(state[42:50].argmax())-1)
                preference,_=heuristic_plan_values(panels[int(case_id)],previous,
                    motif_margin=0.,switch_margin=1.)
                switches.append(-float(preference[action]))
            stats=dataset_metrics(output,cases,ids)
            stats['mean_hand_drum_switches']=float(np.mean(switches))
            stats['motifs']={kind:dict(motifs=value['motifs'],cases=len(value['cases']),
                canonical_fraction=value['canonical']/value['motifs'] if value['motifs'] else None)
                for kind,value in full.items()}
            subset=np.asarray(sorted(set(subset)),np.int64)
            if len(subset):
                local=np.asarray([int(np.flatnonzero(ids==case)[0]) for case in subset])
                stats['motif_case_metrics']=dataset_metrics(output[local],cases,subset)
            report['splits'][split]=stats
        result['runs'][label]=report
    write_json(cli.out,result)
    print(json.dumps({label:{split:record['splits'][split]['motifs']
        for split in ('validation','test')} for label,record in result['runs'].items()}))


if __name__=='__main__':main()
