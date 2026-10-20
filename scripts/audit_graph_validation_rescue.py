#!/usr/bin/env python3
"""Compare graph choices against latest neural choices on cached all-plan validation labels."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.hand_assign import kit_graph_from_layout,LIMB_LEFT
from isaac_rl.assignment_search import hit_groups
from isaac_rl.hand_plan_dataset import load_dataset_batches,dataset_metrics
from isaac_rl.hand_plan_ranker import ACTION_BANK,HandPlanRanker,HandPlanRankerConfig
from isaac_rl.hand_plan_large_dataset import heldout_inputs
from isaac_rl.min_time_assignment import graph_actions
from scripts.train_chart_cma import write_json


def label_files(folder):
    folder=Path(folder)
    if (folder/'manifest.json').exists():
        for source in json.loads((folder/'manifest.json').read_text())['source_datasets']:yield from label_files(source)
    else:
        for path in sorted((folder/'batches').glob('batch_*.pt')):yield path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ablation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    cli=p.parse_args();torch.set_num_threads(2);cli.out.mkdir(exist_ok=False,parents=True)
    manifest=json.loads((cli.ablation/'config.json').read_text());dataset=Path(manifest['dataset'])
    cfg=json.loads((dataset/'config.json').read_text());data=load_dataset_batches(dataset)
    splits=torch.load(dataset/'splits.pt',weights_only=False);ids=np.asarray(splits['validation']);inputs=heldout_inputs(dataset,cfg['network']['tip_geometry'])['validation']
    obs=data['observations'][ids];legal=data['observed'][ids];assert np.array_equal(obs,inputs['observations'])
    rows=np.arange(len(ids));choices={};model_steps={}
    for run in manifest['runs']:
        source=Path(run['path'])/'last.pt';snapshot=cli.out/(Path(run['path']).name+'.pt');snapshot.write_bytes(source.read_bytes())
        saved=torch.load(snapshot,map_location='cpu',weights_only=False);model=HandPlanRanker(HandPlanRankerConfig(**saved['network']))
        model.load_state_dict(saved['model']);model.eval()
        with torch.inference_mode():scores=model(torch.as_tensor(obs)).numpy()
        choices[run['label']]=np.where(legal,scores,-np.inf).argmax(1);model_steps[run['label']]=saved['gradient_steps']
    original=kit_graph_from_layout();modern=replace(original,banned={LIMB_LEFT:original.index['floor_tom']})
    repairs={}
    for name,graph in [('original_graph',original),('graph_13_combinations',modern)]:
        actions=[];count=0
        for row,(clip,state) in enumerate(zip(inputs['panel'],inputs['states'])):
            bits,_=graph_actions(hit_groups(clip),initial_drums=tuple(state['initial_drums']),graph=graph)
            code=int(bits@np.array([8,4,2,1]))
            if not legal[row,code]:
                allowed=np.flatnonzero(legal[row]);code=int(allowed[np.argmin((ACTION_BANK[allowed]!=bits).sum(1))]);count+=1
            actions.append(code)
        choices[name]=np.array(actions);repairs[name]=count
    choices['best_hit_count_plan']=np.where(legal,data['hits'][ids],-1).argmax(1)
    # Per-hit masks distinguish recovered misses from new misses and equal-count swaps.
    masks={};wanted={int(case):row for row,case in enumerate(ids)};offset=0
    for path in label_files(dataset):
        batch=torch.load(path,map_location='cpu',weights_only=False)['batch'];n=len(batch['panel'])
        included=sorted(set(wanted).intersection(range(offset,offset+n)))
        if included:
            reports=json.loads((path.parent/('episodes_'+path.stem.split('_')[1]+'.json')).read_text())
            index={(int(case),int(action)):j for j,(case,action) in enumerate(batch['labels'])}
            for case in included:
                row=wanted[case];keys=None;local=case-offset
                for action in np.flatnonzero(legal[row]):
                    report=reports[index[(local,int(action))]]
                    entries=sorted(report['per_hit_outcomes'],key=lambda h:(h['time_s'],h['drum']))
                    signature=[(h['time_s'],h['drum']) for h in entries]
                    if keys is not None and keys!=signature:raise ValueError('Plan request identities differ')
                    keys=signature;masks[(row,int(action))]=np.array([h['hit'] for h in entries],bool)
                    if int(masks[(row,int(action))].sum())!=int(data['hits'][case,action]):raise ValueError('Hit mask/count mismatch')
        offset+=n
        if offset>max(wanted):break
    methods={}
    for name,actions in choices.items():
        fake=np.full((len(ids),16),-1.);fake[rows,actions]=1.
        stats=dataset_metrics(fake,data,ids)
        methods[name]={key:stats[key] for key in ('hits','requested','hit_rate','mean_reward','mean_reward_regret') if key in stats}
        methods[name]['fully_hit_cases']=int((data['hits'][ids,actions]==data['requested'][ids,actions]).sum())
    comparisons={};best_reward=np.where(legal,data['rewards'][ids],-np.inf).max(1)
    for name in model_steps:
        actions=choices[name];hits=data['hits'][ids,actions];requested=data['requested'][ids,actions]
        failed=hits<requested;bad_selection=data['rewards'][ids,actions]<best_reward-.02
        comparisons[name]=dict(model_update=model_steps[name],cases_with_any_miss=int(failed.sum()),
            below_near_best_reward_cases=int(bad_selection.sum()),graphs={})
        recoverable=sum(int((~masks[(i,int(a))]&np.stack([masks[(i,int(k))] for k in np.flatnonzero(legal[i])]).any(0)).sum())
            for i,a in enumerate(actions))
        comparisons[name]['missed_hits_recoverable_by_some_legal_plan']=recoverable
        comparisons[name]['missed_hits']=int((requested-hits).sum())
        for graph_name in repairs:
            graph_choice=choices[graph_name];ghits=data['hits'][ids,graph_choice];gr=data['rewards'][ids,graph_choice]
            recovered=newly_missed=0
            for i,(a,b) in enumerate(zip(actions,graph_choice)):
                neural=masks[(i,int(a))];graph=masks[(i,int(b))]
                recovered+=int((~neural&graph).sum());newly_missed+=int((neural&~graph).sum())
            comparisons[name]['graphs'][graph_name]=dict(recovered_neural_missed_hits=recovered,
                recovered_fraction_of_neural_misses=recovered/max(1,int((requested-hits).sum())),
                newly_missed_neural_successful_hits=newly_missed,net_hit_change=recovered-newly_missed,
                failed_cases_fully_hit_by_graph=int((failed&(ghits==requested)).sum()),
                failed_cases_improved_hit_count=int((failed&(ghits>hits)).sum()),
                failed_cases_worsened_hit_count=int((failed&(ghits<hits)).sum()),
                below_near_best_cases_fixed_by_graph=int((bad_selection&(gr>=best_reward-.02)).sum()))
    summary=dict(validation_cases=len(ids),runtime=cfg.get('rollout_runtime',{'rollout_runtime_version':'legacy_contact_reset_v0'}),
        model_updates=model_steps,methods=methods,rescue=comparisons,graph_eligibility_repairs=repairs,
        matched_cached_inputs_and_labels=True,held_out_used_for_training=False,
        limitation='Latest model snapshots can have different optimizer budgets; each graph comparison is paired on the same819 validation cases and stored legacy simulator outcomes. Oracle maximizes hits, not the penalized reward.')
    write_json(cli.out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
