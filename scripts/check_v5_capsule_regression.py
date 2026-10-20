#!/usr/bin/env python3
"""Compare v4/v5 Isaac credit on matched hi-hat/snare states and frozen decoders.

This is a diagnostic scorer comparison, not a new training dataset. It holds
the 64 physical starts, 128 assignment slots, decoder checkpoints, rhythm,
and 250-Hz controller fixed against the archived v4 reports.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import CONTACT_SCORING_VERSION,DRUM_GEOMETRY_VERSION,GEOMETRY_VERSION
from isaac_rl.hand_plan_ranker import ACTION_BANK
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def summary(reports,actions,starts):
    rows=[]
    for i,action in enumerate(actions):
        chunk=reports[i*starts:(i+1)*starts]
        rows.append(dict(action=action,hands=''.join('R' if h else 'L' for h in ACTION_BANK[action]),
            hits=sum(r['hits'] for r in chunk),requested=sum(r['requested'] for r in chunk),
            hihat_hits=sum(r['by_drum']['hihat']['hits'] for r in chunk),
            snare_hits=sum(r['by_drum']['snare']['hits'] for r in chunk),
            mean_reward=float(np.mean([r['reward'] for r in chunk]))))
    return rows


def main():
    torch.set_num_threads(2)
    out=ROOT/'runs/capsule_v5_matched_hihat_snare_20261013'
    out.mkdir(parents=True,exist_ok=True)
    case=json.loads((ROOT/'runs/decoder_v4_hihat_paired_20261011/cases.json').read_text())
    ids=case['state_ids'];actions=case['actions']
    assert len(ids)==64 and actions==[5,10]
    sweep=torch.load(ROOT/'runs/hihat_snare_alternating_ranker_prior_v4_20261009/start_sweep.pt',
        map_location='cpu',weights_only=False)
    states=[sweep['states'][int(i)] for _ in actions for i in ids]
    episodes=[assigned_clip(sweep['clip'],ACTION_BANK[a]) for a in actions for _ in ids]
    dataset=json.loads((ROOT/'runs/hand_plan_dataset_v4_prefirst_200k_20261009/config.json').read_text())
    source=rollout_source_config(json.loads((Path(dataset['source_dataset'])/'config.json').read_text()))
    seed=ROOT/'runs/hihat_snare_start_sweep_ranker_prior_v4_20261009/scene_seed.json'
    methods={
        'baseline':(ROOT/'runs/phase2_all_scratch_physics_20261009/last.pt',
                    ROOT/'runs/decoder_v4_hihat_paired_20261011/reports_baseline.json'),
        'v4_trained_frozen_head':(
            ROOT/'runs/phase2_pairs42_v4_frozen_terminal_fromp1_20261012/last.pt',
            ROOT/'runs/phase2_pairs42_v4_frozen_terminal_fromp1_20261012/matched_v4_hihat_snare_final/reports.json'),
    }
    results={}
    for label,(checkpoint,v4_path) in methods.items():
        method_out=out/label
        method_out.mkdir(exist_ok=True)
        saved=method_out/'reports_v5.json'
        if saved.exists():
            reports=json.loads(saved.read_text())
        else:
            args=runtime_args(['--out',str(method_out),'--envs',str(len(episodes)),
                '--device','cuda:1','--rollout-devices','cuda:1','--physics-hz','250',
                '--decode-chunk','256','--collision-chunk','512','--no-plan-diagnostics',
                '--clone-kit-from-source','--clear-reset-contacts','--freeze-hit-scoring',
                '--match-decoder-speed','--rollout-audit'])
            for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                        'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
                setattr(args,key,source['args'][key])
            args.groups=args.horizon=4;args.seed=20261023
            evaluator=None
            try:
                evaluator=DistributedIsaacEvaluator(args,checkpoint,seed)
                ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
                if (ready['rollout_runtime_version']!=CONTACT_SCORING_VERSION
                        or ready['simulation_geometry_version']!=GEOMETRY_VERSION
                        or ready['drum_geometry_version']!=DRUM_GEOMETRY_VERSION
                        or ready['drum_usd_sha256']!=dataset['drum_usd_sha256']
                        or ready['drum_layout_sha256']!=dataset['drum_layout_sha256']):
                    raise ValueError('Isaac capsule scorer or physical geometry differs')
                write_json(method_out/'ready_verified.json',ready)
                reports=evaluator.evaluate(episodes,initial_states=states,
                    progress=lambda done,total,tick,ticks:write_json(method_out/'progress.json',
                        dict(done=done,total=total,tick=tick,ticks=ticks)))
            finally:
                if evaluator is not None:evaluator.close()
            write_json(saved,reports)
        old=json.loads(v4_path.read_text())
        if not (len(reports)==len(old)==len(episodes)):
            raise ValueError('Matched report count changed')
        v4_hits=np.array([r['hits'] for r in old]);v5_hits=np.array([r['hits'] for r in reports])
        gained=[];lost=[]
        for old_report,new_report in zip(old,reports):
            for before,after in zip(old_report['per_hit_outcomes'],new_report['per_hit_outcomes']):
                if (before['drum'],before['hand'],before['time_s']) != (
                        after['drum'],after['hand'],after['time_s']):
                    raise ValueError('Beat labels changed between paired rollouts')
                if bool(after['hit'])!=bool(before['hit']):
                    (gained if after['hit'] else lost).append(after['drum'])
        debug_events=reports[-1].get('rollout_audit',{}).get('contact_tracker_debug_events',[])
        hihat_events=[event for event in debug_events if event['drum']==1]
        results[label]=dict(checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            v4=summary(old,actions,len(ids)),v5=summary(reports,actions,len(ids)),
            gained_hit_drums=dict(Counter(gained)),lost_hit_drums=dict(Counter(lost)),
            changed_hit_count_cases=int((v4_hits!=v5_hits).sum()),
            v5_more_hits=int((v5_hits>v4_hits).sum()),
            v4_more_hits=int((v5_hits<v4_hits).sum()),
            identical_hits=int((v5_hits==v4_hits).sum()),
            sampled_hihat_entries=dict(events=len(hihat_events),
                top=sum(event['entry']=='top' for event in hihat_events),
                upper_side=sum(event['entry']=='upper_side' for event in hihat_events),
                fraction_min=min((event['fraction'] for event in hihat_events),default=None),
                fraction_median=float(np.median([event['fraction'] for event in hihat_events]))
                    if hihat_events else None,
                force_median_N=float(np.median([event['force_N'] for event in hihat_events]))
                    if hihat_events else None,
                impact_speed_median_m_s=float(np.median(
                    [event['latched_impact_speed_m_s'] for event in hihat_events]))
                    if hihat_events else None))
        write_json(out/'partial_summary.json',dict(status='running',methods=results))
        print('[capsule-v5] '+json.dumps(dict(method=label,**results[label])),flush=True)
    write_json(out/'summary.json',dict(status='complete',runtime=CONTACT_SCORING_VERSION,
        diagnostic_only=True,no_training_labels_mixed=True,matched_start_ids=ids,
        actions=actions,same_physical_slots=True,methods=results))


if __name__=='__main__':main()
