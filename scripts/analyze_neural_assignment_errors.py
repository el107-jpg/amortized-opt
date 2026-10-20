#!/usr/bin/env python3
"""Frozen-model error cases on exhaustive held-out outcomes; no training feedback."""
import argparse
import json
from pathlib import Path
import pickle
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,ACTION_BANK,encode_clip
from isaac_rl.assignment_search import hit_groups
from scripts.train_neural_hand_plans import load_cached_panel

LABELS={'snare':'Snare','hihat':'Hi-hat','tom1':'Tom1','tom2':'Tom2',
        'cymbal_left':'Left cymbal','cymbal_right':'Right cymbal','floor_tom':'Floor tom'}


def credit_from_drum(report,drum):
    counts=report['by_drum'][drum]
    if counts['requested']==1:return bool(counts['hits'])
    if counts['hits']==0:return False
    if counts['hits']==counts['requested']:return True
    return None  # The old cache did not retain individual repeated-note credit.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    cli=p.parse_args();torch.set_num_threads(2)
    cfg=json.loads((cli.run/'config.json').read_text())
    source=torch.load(cli.run/'last.pt',map_location='cpu',weights_only=False)
    update=source['completed_updates'];model=HandPlanRanker(HandPlanRankerConfig(**source['network']))
    model.load_state_dict(source['model']);model.eval()
    cli.out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=source['model'],network=source['network'],completed_updates=update,
        audit_only=True),cli.out/'model_snapshot.pt')
    folder=Path(cfg['args']['cached_panel']);data=load_cached_panel(folder,cfg['checkpoint_sha256'])
    archived=pickle.loads((folder/'planned_trials.pkl').read_bytes());reports=json.loads((folder/'episodes.json').read_text())
    lookup={}
    for row,(context,ids) in enumerate(zip(archived['contexts'],archived['trial']['gather'])):
        for action,j in zip(context['actions'],ids):lookup[row,int(action@np.array([8,4,2,1]))]=reports[int(j)]
    observations=np.stack([encode_clip(c,tip_geometry=model.config.tip_geometry) for c in data['panel']])
    with torch.inference_mode():scores=model(torch.as_tensor(observations)).numpy()
    valid,rewards=data['observed'],data['rewards']
    selected=np.where(valid,scores,-np.inf).argmax(1)
    best=np.where(valid,rewards,-np.inf).argmax(1)
    maxhits=np.where(valid,data['hits'],-1).max(1)
    besthits=np.where(valid & (data['hits']==maxhits[:,None]),rewards,-np.inf).argmax(1)
    cases=[]
    for row,clip in enumerate(data['panel']):
        a,b=int(selected[row]),int(best[row]);chosen,better=lookup[row,a],lookup[row,b]
        groups=hit_groups(clip);beats=[]
        for group_index,group in enumerate(groups):
            for slot,hit in enumerate(group):
                beats.append(dict(group=group_index+1,time_s=hit.time,drum=hit.drum_name,
                    velocity=hit.velocity,neural_hand=('L','R')[(ACTION_BANK[a,group_index]+slot)%2],
                    better_hand=('L','R')[(ACTION_BANK[b,group_index]+slot)%2],
                    neural_hit=credit_from_drum(chosen,hit.drum_name),
                    better_hit=credit_from_drum(better,hit.drum_name)))
        bank=np.flatnonzero(valid[row]);order=bank[np.lexsort((bank,-scores[row,bank]))]
        cases.append(dict(context=row,name=clip.name,speed_tier=int(archived['tiers'][row]),
            legal_plans=int(valid[row].sum()),neural_action=a,better_action=b,
            neural_bits=ACTION_BANK[a].tolist(),better_bits=ACTION_BANK[b].tolist(),
            neural_hits=chosen['hits'],better_hits=better['hits'],requested=chosen['requested'],
            neural_reward=chosen['reward'],better_reward=better['reward'],
            lost_hits=better['hits']-chosen['hits'],reward_gap=better['reward']-chosen['reward'],
            neural_actual_reward_rank=1+int(((rewards[row]>rewards[row,a]+1e-8)&valid[row]).sum()),
            better_plan_neural_rank=int(np.flatnonzero(order==b)[0])+1,
            max_observed_hits=int(maxhits[row]),beats=beats))
    requested=sum(c['requested'] for c in cases);hits=sum(c['neural_hits'] for c in cases)
    oracle=sum(c['max_observed_hits'] for c in cases)
    summary=dict(model_update=update,contexts=len(cases),requested=requested,neural_hits=hits,
        neural_hit_rate=hits/requested,best_observed_hits=oracle,best_observed_hit_rate=oracle/requested,
        recoverable_net_hit_gap=oracle-hits,reward_suboptimal_rhythms=sum(c['reward_gap']>1e-8 for c in cases),
        more_than_005_reward_gap_rhythms=sum(c['reward_gap']>.05 for c in cases),
        hit_suboptimal_rhythms=sum(c['neural_hits']<c['max_observed_hits'] for c in cases),
        no_hit_plan_found_rhythms=sum(c['max_observed_hits']==0 for c in cases),
        exact_best_reward_rhythms=sum(c['reward_gap']<=1e-8 for c in cases),
        missed_hit_counts=Counter(c['max_observed_hits']-c['neural_hits'] for c in cases),
        scope='Fixed held-out generated four-group rhythms, nominal ready starts, cached one rollout per legal plan',
        no_training_feedback=True)
    drum_stats=[]
    for drum in LABELS:
        rows=[lookup[i,int(selected[i])]['by_drum'][drum] for i in range(len(cases))]
        bestrows=[lookup[i,int(besthits[i])]['by_drum'][drum] for i in range(len(cases))]
        req=sum(r['requested'] for r in rows);n=sum(r['hits'] for r in rows);o=sum(r['hits'] for r in bestrows)
        drum_stats.append(dict(drum=drum,requested=req,neural_hits=n,neural_misses=req-n,
            neural_hit_rate=n/req,best_plan_hits=o,net_recoverable_hits=o-n,
            independent_drum_max_hits=sum(max(lookup[i,int(a)]['by_drum'][drum]['hits']
                for a in np.flatnonzero(valid[i])) for i in range(len(cases))),
            gross_gained_hits=sum(max(0,b['hits']-a['hits']) for a,b in zip(rows,bestrows)),
            gross_lost_hits=sum(max(0,a['hits']-b['hits']) for a,b in zip(rows,bestrows))))
    tier_stats=[]
    for tier in range(3):
        rows=[c for c in cases if c['speed_tier']==tier];req=sum(c['requested'] for c in rows)
        tier_stats.append(dict(tier=('fast','medium','slow')[tier],rhythms=len(rows),requested=req,
            neural_hits=sum(c['neural_hits'] for c in rows),neural_hit_rate=sum(c['neural_hits'] for c in rows)/req,
            best_hit_rate=sum(c['max_observed_hits'] for c in rows)/req,
            hit_suboptimal_rhythms=sum(c['neural_hits']<c['max_observed_hits'] for c in rows)))
    # Examples permit exact note credit from cached counts. Never guess which
    # of two repeated drum requests was hit when only one was credited.
    examples=[]
    for tier in range(3):
        candidates=[c for c in cases if c['speed_tier']==tier and c['lost_hits']>0
            and all(b['neural_hit'] is not None and b['better_hit'] is not None for b in c['beats'])]
        if candidates:examples.append(max(candidates,key=lambda c:(c['lost_hits'],c['reward_gap'])))
    summary.update(by_drum=drum_stats,by_timing_tier=tier_stats,
                   example_contexts=[c['context'] for c in examples])
    (cli.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (cli.out/'all_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (cli.out/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    lines=[f'# Neural assignment errors: update{update}', '',
        '256 fixed held-out generated four-group rhythms, all starting from ready. Every legal plan has one cached Isaac outcome; the best observed plan is a hindsight reference, not a proven optimum. No evaluation data enters training.','',
        f'Neural{hits}/{requested} hits ({100*hits/requested:.1f}%); best observed{oracle}/{requested} ({100*oracle/requested:.1f}%). '
        f'{summary["hit_suboptimal_rhythms"]}/{len(cases)} rhythms admit a plan with more hits.','',
        '| Drum | Requests | Neural hits | Misses | Hits under best whole plans | Net recoverable gap |',
        '|---|---:|---:|---:|---:|---:|']
    for r in drum_stats:lines.append(f'| {LABELS[r["drum"]]} | {r["requested"]} | {r["neural_hits"]} | {r["neural_misses"]} | {r["best_plan_hits"]} | {r["net_recoverable_hits"]:+d} |')
    lines += ['', 'Net recovery can be negative for a drum when the best complete assignment trades that drum against other hits. The comparison is between whole plans, not independent per-drum optima.','']
    for c in examples:
        lines += [f'## Case{c["context"]}: {("fast","medium","slow")[c["speed_tier"]]}', '',
            f'Rhythm `{c["name"]}`. Neural{c["neural_hits"]}/{c["requested"]}; better observed{c["better_hits"]}/{c["requested"]}. '
            f'Reward{c["neural_reward"]:.3f}→{c["better_reward"]:.3f}. Better plan was ranked{c["better_plan_neural_rank"]} by the network.','',
            '| Hit group | Time (s) | Drum | Neural hand/result | Better hand/result |',
            '|---:|---:|---|---|---|']
        for b in c['beats']:
            lines.append(f'| {b["group"]} | {b["time_s"]:.2f} | {LABELS[b["drum"]]} | {b["neural_hand"]} / {"hit" if b["neural_hit"] else "miss"} | {b["better_hand"]} / {"hit" if b["better_hit"] else "miss"} |')
        lines += ['', 'These examples have unambiguous per-note credit from the archived per-drum counts; partial repeated-note credits are left unknown in all_cases.json.','']
    (cli.out/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    x=np.arange(len(drum_stats));width=.35
    ax.bar(x-width/2,[100*r['neural_hit_rate'] for r in drum_stats],width,label='Neural selection',color='#4c86b1')
    ax.bar(x+width/2,[100*r['best_plan_hits']/r['requested'] for r in drum_stats],width,label='Best observed complete plans',color='#65a47e')
    ax.set_xticks(x,[LABELS[r['drum']] for r in drum_stats],rotation=20)
    ax.set_ylim(0,100);ax.set_ylabel('Hit rate (%)');ax.legend()
    ax.set_title(f'Update{update}: failures by drum on fixed ready-start test panel')
    fig.savefig(cli.out/'misses_by_drum.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));print('EXAMPLES '+json.dumps(examples,indent=2))


if __name__=='__main__':main()
