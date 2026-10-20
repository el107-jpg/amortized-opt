#!/usr/bin/env python3
"""Explain matched neural/graph misses on recorded Groove test windows."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.hand_plan_dataset import load_dataset_batches
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig
from scripts.compare_graph_neural_fresh_test import decisions
from scripts.train_chart_cma import write_json


def window_cluster_comparison(rows,seed=20261042,repetitions=4000):
    neural=np.asarray([r['methods']['neural_depth2']['hits'] for r in rows])
    graph=np.asarray([r['methods']['original_graph']['hits'] for r in rows])
    requested=np.asarray([r['methods']['neural_depth2']['requested'] for r in rows])
    oracle=np.asarray([r['oracle_hits'] for r in rows])
    rng=np.random.default_rng(seed);difference=np.empty(repetitions)
    for index in range(repetitions):
        ids=rng.integers(len(rows),size=len(rows))
        difference[index]=100*(neural[ids].sum()-graph[ids].sum())/requested[ids].sum()
    return dict(windows=len(rows),requests=int(requested.sum()),
        neural_hits=int(neural.sum()),graph_hits=int(graph.sum()),oracle_hits=int(oracle.sum()),
        neural_hit_rate=float(neural.sum()/requested.sum()),
        graph_hit_rate=float(graph.sum()/requested.sum()),
        oracle_hit_rate=float(oracle.sum()/requested.sum()),
        neural_minus_graph_pp=float(100*(neural.sum()-graph.sum())/requested.sum()),
        window_cluster_bootstrap_95pct_difference_pp=np.quantile(difference,[.025,.975]).tolist(),
        bootstrap_seed=seed,repetitions=repetitions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    cli=p.parse_args();torch.set_num_threads(2)
    config=json.loads((cli.dataset/'config.json').read_text())
    if json.loads((cli.dataset/'run.json').read_text())['status']!='complete':raise ValueError('Groove labels incomplete')
    manifest=json.loads((Path(config['panel']).with_suffix('.json')).read_text())
    if manifest['selection']['source_split']!='test':raise ValueError('Groove panel is not test-only')
    saved=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).to(cli.device)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    data=load_dataset_batches(cli.dataset)
    panel=[];states=[];report_lookup={};offset=0
    for batch_path in sorted((cli.dataset/'batches').glob('batch_*.pt')):
        saved_batch=torch.load(batch_path,map_location='cpu',weights_only=False)['batch']
        episode_path=batch_path.with_name(batch_path.name.replace('batch_','episodes_').replace('.pt','.json'))
        reports=json.loads(episode_path.read_text())
        if len(reports)!=len(saved_batch['labels']):raise ValueError('Plan reports differ from labels')
        report_lookup.update({(offset+int(case),int(action)):report
            for (case,action),report in zip(saved_batch['labels'],reports)})
        panel.extend(saved_batch['panel']);states.extend(saved_batch['states']);offset+=len(saved_batch['panel'])
    if offset!=len(data['observations']):raise ValueError('Groove case count differs')
    choices,repairs,fallbacks=decisions(model,data,panel,states)
    if repairs['original_graph'] or fallbacks['original_graph']:
        raise ValueError('Graph comparison required a fallback or legality repair')
    methods=('neural_depth2','original_graph')
    drum_rows={name:defaultdict(lambda:dict(requested=0,hits=0,miss_distances_m=[])) for name in methods}
    window_rows=[];starts=manifest['selection']['starts_per_window']
    for window_index,window in enumerate(manifest['windows']):
        row=dict(window_index=window_index,source_midi=window['source_midi'],
            style=window['style'],bpm=window['bpm'],features=window['features'],
            starts=starts,methods={})
        case_ids=range(window_index*starts,(window_index+1)*starts)
        for method in methods:
            total_hits=total_requested=0;examples=[]
            for case in case_ids:
                action=int(choices[method][case]);report=report_lookup[(case,action)]
                outcomes=report['per_hit_outcomes']
                if len(outcomes)!=report['requested'] or sum(h['hit'] for h in outcomes)!=report['hits']:
                    raise ValueError('Per-hit outcome count differs from report')
                total_hits+=report['hits'];total_requested+=report['requested']
                misses=[]
                for outcome in outcomes:
                    drum=outcome['drum'];stats=drum_rows[method][drum]
                    stats['requested']+=1;stats['hits']+=int(outcome['hit'])
                    if not outcome['hit']:
                        distance=outcome.get('closest_distance_m')
                        if distance is not None:stats['miss_distances_m'].append(float(distance))
                        misses.append(dict(drum=drum,time_s=outcome['time_s'],
                            closest_distance_m=distance,hand=outcome['hand']))
                examples.append(dict(case=case,assignment=choices[method][case].item(),
                    hits=report['hits'],requested=report['requested'],misses=misses))
            row['methods'][method]=dict(hits=total_hits,requested=total_requested,
                hit_rate=total_hits/total_requested,starts=examples)
        oracle_hits=sum(int(np.where(data['observed'][case],data['hits'][case],-1).max())
            for case in case_ids)
        row['oracle_hits']=oracle_hits
        row['neural_minus_graph_hits']=row['methods']['neural_depth2']['hits']-row['methods']['original_graph']['hits']
        row['unreached_oracle_hits']=oracle_hits-row['methods']['neural_depth2']['hits']
        window_rows.append(row)
    drums={}
    for drum in sorted(set(drum_rows['neural_depth2'])|set(drum_rows['original_graph'])):
        values={}
        for method in methods:
            stats=drum_rows[method][drum]
            values[method]=dict(requested=stats['requested'],hits=stats['hits'],
                hit_rate=stats['hits']/stats['requested'],
                misses=stats['requested']-stats['hits'],
                median_miss_distance_m=float(np.median(stats['miss_distances_m'])) if stats['miss_distances_m'] else None)
        if values['neural_depth2']['requested']!=values['original_graph']['requested']:
            raise ValueError('Methods received different drum requests')
        drums[drum]=values
    recoverability=defaultdict(lambda:dict(recoverable_misses=0,misses_not_hit_by_any_legal_plan=0))
    for case in range(len(panel)):
        signature=None;legal_hit_masks=[]
        for action in np.flatnonzero(data['observed'][case]):
            outcomes=sorted(report_lookup[(case,int(action))]['per_hit_outcomes'],
                key=lambda hit:(hit['time_s'],hit['drum']))
            identities=[(hit['time_s'],hit['drum']) for hit in outcomes]
            if signature is None:signature=identities
            elif signature!=identities:raise ValueError('Plan hit identities differ')
            legal_hit_masks.append([bool(hit['hit']) for hit in outcomes])
        any_legal_hit=np.any(np.asarray(legal_hit_masks),axis=0)
        selected=sorted(report_lookup[(case,int(choices['neural_depth2'][case]))]['per_hit_outcomes'],
            key=lambda hit:(hit['time_s'],hit['drum']))
        for index,hit in enumerate(selected):
            if not hit['hit']:
                key='recoverable_misses' if any_legal_hit[index] else 'misses_not_hit_by_any_legal_plan'
                recoverability[hit['drum']][key]+=1
    for drum,counts in recoverability.items():drums[drum]['neural_depth2'].update(counts)
    by_physical_difficulty=sorted(window_rows,key=lambda row:(
        row['oracle_hits']/row['methods']['neural_depth2']['requested'],row['oracle_hits']))
    summary=dict(cases=len(panel),windows=len(window_rows),starts_per_window=starts,
        test_split_only=True,checkpoint_step=saved['gradient_steps'],
        all_windows=window_cluster_comparison(window_rows),
        physically_hardest_16_windows=window_cluster_comparison(by_physical_difficulty[:16]),
        physically_hardest_8_windows=window_cluster_comparison(by_physical_difficulty[:8]),
        physical_difficulty_subset_note='Selected after seeing all-plan Isaac outcomes; diagnostic subset, not an independent test estimate',
        recoverability_total={key:sum(row[key] for row in recoverability.values())
            for key in ('recoverable_misses','misses_not_hit_by_any_legal_plan')},
        drum_results=drums,
        hardest_windows=sorted(window_rows,key=lambda row:(row['methods']['neural_depth2']['hit_rate'],
            -row['unreached_oracle_hits']))[:10],
        most_graph_advantage=sorted(window_rows,key=lambda row:row['neural_minus_graph_hits'])[:10],
        most_neural_advantage=sorted(window_rows,key=lambda row:-row['neural_minus_graph_hits'])[:10],
        windows_detail=window_rows)
    cli.out.parent.mkdir(parents=True,exist_ok=True);write_json(cli.out,summary)
    with cli.out.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('source_midi','style','bpm','score','fast_gaps','paired_groups',
            'neural_hits','graph_hits','requested','oracle_hits'))
        for row in window_rows:
            writer.writerow((row['source_midi'],row['style'],row['bpm'],row['features']['score'],
                row['features']['fast_gaps'],row['features']['paired_groups'],
                row['methods']['neural_depth2']['hits'],row['methods']['original_graph']['hits'],
                row['methods']['neural_depth2']['requested'],row['oracle_hits']))
    print(json.dumps(dict(cases=summary['cases'],windows=summary['windows'],
        drum_results=drums,hardest=[dict(source=Path(r['source_midi']).name,
            neural_hits=r['methods']['neural_depth2']['hits'],graph_hits=r['methods']['original_graph']['hits'],
            requested=r['methods']['neural_depth2']['requested']) for r in summary['hardest_windows'][:5]]),indent=2))


if __name__=='__main__':main()
