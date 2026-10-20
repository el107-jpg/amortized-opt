#!/usr/bin/env python3
"""Freeze difficult recorded Groove test windows before physics evaluation."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit,Hit,RhythmClip
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.recorded_hand_rhythms import RecordedHandRhythmSampler
from scripts.train_chart_cma import write_json


def difficulty(groups):
    gaps=np.diff([t for t,_ in groups])
    paired=sum(len(hits)==2 for _,hits in groups)
    drums=[hit.drum for _,hits in groups for hit in hits]
    rare=sum(d in ('tom1','tom2','floor_tom','cymbal_left','cymbal_right') for d in drums)
    fast=int((gaps<=.12+1e-8).sum());medium=int((gaps<=.24+1e-8).sum())
    distinct=len(set(drums))
    score=3*fast+1.5*medium+2*paired+.8*distinct+1.5*rare
    return dict(score=float(score),fast_gaps=fast,gaps_s=[round(float(g),4) for g in gaps],
        paired_groups=paired,distinct_drums=distinct,rare_drum_hits=rare,
        drums=sorted(set(drums)),total_hits=len(drums))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groove-root',type=Path,default=ROOT/'assets/rcc/groove')
    p.add_argument('--eligibility',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--windows',type=int,default=64)
    p.add_argument('--starts-per-window',type=int,default=3)
    p.add_argument('--max-per-drummer',type=int,default=18)
    p.add_argument('--seed',type=int,default=20261040)
    args=p.parse_args()
    if min(args.windows,args.starts_per_window,args.max_per_drummer)<1:raise ValueError('Positive panel budgets required')
    if args.out.exists():raise FileExistsError(args.out)
    kit=DrumKit.from_layout();chart=MinimumTimeChart.load(args.eligibility)
    sampler=RecordedHandRhythmSampler(kit,4,chart,args.groove_root,split='test')
    rows={str((args.groove_root/row['midi_filename']).resolve()):row
        for row in csv.DictReader((args.groove_root/'info.csv').open()) if row['split']=='test'}
    candidates=[]
    for source in sampler.sources:
        best=None
        for index in source['windows']:
            groups=source['groups'][index:index+4]
            features=difficulty(groups)
            candidate=dict(source=source,index=index,groups=groups,features=features)
            if best is None or (features['score'],features['rare_drum_hits'],-index)>(
                    best['features']['score'],best['features']['rare_drum_hits'],-best['index']):
                best=candidate
        if best is not None:candidates.append(best)
    candidates.sort(key=lambda c:(-c['features']['score'],-c['features']['rare_drum_hits'],c['source']['path']))
    chosen=[];by_drummer=Counter()
    for candidate in candidates:
        source_path=candidate['source']['path'];drummer=rows[source_path]['drummer']
        if by_drummer[drummer]>=args.max_per_drummer:continue
        chosen.append(candidate);by_drummer[drummer]+=1
        if len(chosen)==args.windows:break
    if len(chosen)!=args.windows:raise ValueError('Insufficient independent test performances under drummer cap')
    rng=np.random.default_rng(args.seed);panel=[];manifest=[]
    for rank,candidate in enumerate(chosen):
        source_path=candidate['source']['path'];source_row=rows[source_path]
        groups=candidate['groups'];first=groups[0][0]
        leads=rng.choice(np.arange(1,46),size=args.starts_per_window,replace=False)*.02
        manifest.append(dict(rank=rank,source_midi=source_path,drummer=source_row['drummer'],
            style=source_row['style'],bpm=float(source_row['bpm']),window_start_group=candidate['index'],
            features=candidate['features'],first_delays_s=[round(float(t),4) for t in leads]))
        for repeat,lead in enumerate(leads):
            hits=[]
            for time_s,rows_at_time in groups:
                when=round(float(lead+time_s-first),8)
                for slot,row in enumerate(sorted(rows_at_time,key=lambda hit:kit.index(hit.drum))):
                    hits.append(Hit(when,when,kit.index(row.drum),row.drum,
                        ('left','right')[slot],row.velocity/127.))
            name=f'hard_groove_{source_row["drummer"]}_{source_row["id"].replace("/","_")}_w{candidate["index"]}_r{repeat}'
            panel.append(RhythmClip(name,hits,hits[-1].time+.3))
    selection=dict(source='groove_midi_test_split_difficult_four_group_windows_v1',
        no_model_or_simulator_outcomes_used_for_selection=True,
        source_split='test',performances=len(chosen),windows=len(chosen),
        starts_per_window=args.starts_per_window,cases=len(panel),seed=args.seed,
        source_performance_cap=1,drummer_cap=args.max_per_drummer,
        difficulty='3*fast_gaps_<=120ms + 1.5*gaps_<=240ms + 2*paired_groups + 0.8*distinct_drums + 1.5*tom_or_cymbal_hits',
        first_delay='three distinct uniform 20ms-grid draws from 20-900ms per window',
        drummer_counts=dict(by_drummer),drum_window_counts=dict(Counter(
            drum for item in manifest for drum in item['features']['drums'])))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(panel=panel,selection=selection,windows=manifest),args.out)
    write_json(args.out.with_suffix('.json'),dict(selection=selection,windows=manifest))
    print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
