#!/usr/bin/env python3
"""Exhaustive legal four-group hand plans, measured in matched Isaac trials."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from amortized_opt.rhythm import DrumKit
from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart, graph_actions, window_scores
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import GEOMETRY_VERSION
from scripts.train_chart_cma import parse_args as runtime_args, write_json


def enumerate_context(clip, chart, ready):
    groups = hit_groups(clip)
    bank = np.asarray(list(product(*[chart.choices(group) for group in groups])), int)
    score = window_scores(groups, bank, chart, initial_drums=(-1, -1), last_times=(0., 0.),
                          initial_minimum_s=ready)['total']
    graph, _ = graph_actions(groups)
    order = np.lexsort((np.arange(len(bank)), (bank != graph).sum(1), score))
    return dict(episodes=[assigned_clip(clip, bank[i]) for i in order],
        actions=bank[order], scores=score[order], legal_plans=len(bank),
        graph_rank=int(np.argmin((bank[order] != graph).sum(1)))+1)


def assemble_trials(contexts, capacity, repeat_controls=True):
    gather = [np.full(c['legal_plans'], -1, int) for c in contexts]
    episodes, labels = [], []
    # Candidate rank-major packing: each legal sequence is executed once.
    for rank in range(max(c['legal_plans'] for c in contexts)):
        for case, context in enumerate(contexts):
            if rank < context['legal_plans']:
                gather[case][rank] = len(episodes)
                episodes.append(context['episodes'][rank])
                labels.append(dict(context=case, predicted_rank=rank+1, repeat_control=False))
    repeats = [[] for _ in contexts]
    # If floor eligibility leaves spare slots in the small256-lane test, use
    # them to quantify repeated selected-plan variation rather than idle lanes.
    if repeat_controls and len(episodes) < capacity:
        while len(episodes) < capacity:
            case = (len(episodes)-sum(len(ids) for ids in gather)) % len(contexts)
            repeats[case].append(len(episodes))
            episodes.append(contexts[case]['episodes'][0])
            labels.append(dict(context=case, predicted_rank=1, repeat_control=True))
    return dict(episodes=episodes, gather=gather, repeat_ids=repeats, labels=labels)


def observed_rank(values, index=0, tolerance=1e-10):
    values = np.asarray(values)
    return 1+int((values > values[index]+tolerance).sum())


def summarize_contexts(contexts, trial, reports, tiers):
    result = []
    for case, (context, ids) in enumerate(zip(contexts, trial['gather'])):
        rows = [reports[int(i)] for i in ids]
        reward = np.array([r['reward'] for r in rows])
        hits = np.array([r['hits'] for r in rows])
        requested = rows[0]['requested']
        if any(r['requested'] != requested for r in rows):
            raise ValueError('Matched plans differ in requested strikes')
        best_reward = np.flatnonzero(reward >= reward.max()-1e-10)
        best_hits = np.flatnonzero(hits == hits.max())
        controls = [reports[int(i)]['reward'] for i in [ids[0], *trial['repeat_ids'][case]]]
        graph = context['graph_rank']-1
        result.append(dict(context=case, speed_tier=int(tiers[case]), legal_plans=len(ids),
            predicted_top_two_score_gap=float(context['scores'][1]-context['scores'][0]) if len(ids)>1 else None,
            selected_actual_reward_rank=observed_rank(reward),
            selected_actual_hit_rank=observed_rank(hits, tolerance=0),
            selected_reward=float(reward[0]), second_reward=float(reward[1]) if len(ids)>1 else None,
            best_observed_reward=float(reward.max()), graph_reward=float(reward[graph]),
            selected_hits=int(hits[0]), second_hits=int(hits[1]) if len(ids)>1 else None,
            best_observed_hits=int(hits.max()), graph_hits=int(hits[graph]), requested=requested,
            best_reward_predicted_rank=int(best_reward.min())+1,
            best_hits_predicted_rank=int(best_hits.min())+1,
            predicted_top_two_contains_best_reward=bool((best_reward < 2).any()),
            predicted_top_two_contains_best_hits=bool((best_hits < 2).any()),
            predicted_rewards=reward.tolist(), predicted_hits=hits.tolist(),
            selected_reward_repeats=controls,
            repeat_reward_std=float(np.std(controls, ddof=1)) if len(controls)>1 else None))
    n = len(result)
    total = sum(r['requested'] for r in result)
    summary = dict(contexts=n, unique_plan_trials=sum(r['legal_plans'] for r in result),
        physical_rollouts=len(reports), legal_plan_counts={str(k):sum(r['legal_plans']==k for r in result)
            for k in sorted({r['legal_plans'] for r in result})},
        selected_is_best_reward_fraction=sum(r['selected_actual_reward_rank']==1 for r in result)/n,
        selected_is_best_hits_fraction=sum(r['selected_actual_hit_rank']==1 for r in result)/n,
        selected_is_actual_reward_top_two_fraction=sum(r['selected_actual_reward_rank']<=2 for r in result)/n,
        selected_within_005_reward_of_best_fraction=sum(r['best_observed_reward']-r['selected_reward']<=.05 for r in result)/n,
        median_selected_actual_reward_rank=float(np.median([r['selected_actual_reward_rank'] for r in result])),
        mean_selected_actual_reward_rank=float(np.mean([r['selected_actual_reward_rank'] for r in result])),
        predicted_top_two_contains_best_reward_fraction=sum(r['predicted_top_two_contains_best_reward'] for r in result)/n,
        predicted_top_two_contains_best_hits_fraction=sum(r['predicted_top_two_contains_best_hits'] for r in result)/n,
        mean_reward_regret=float(np.mean([r['best_observed_reward']-r['selected_reward'] for r in result])),
        selected_pooled_hit_rate=sum(r['selected_hits'] for r in result)/total,
        second_pooled_hit_rate=sum(r['second_hits'] if r['second_hits'] is not None else r['selected_hits'] for r in result)/total,
        oracle_pooled_hit_rate=sum(r['best_observed_hits'] for r in result)/total,
        graph_pooled_hit_rate=sum(r['graph_hits'] for r in result)/total,
        oracle_note='Best observed plan among all legal assignments, not a learned policy or an unbiased expected-performance estimate.',
        rank_note='Ranks count strictly better outcomes; ties share the same rank.',
        no_parameter_updates=True)
    complete = [r for r in result if r['legal_plans']==16]
    summary['full_sixteen_contexts'] = dict(count=len(complete),
        selected_is_best_reward_fraction=sum(r['selected_actual_reward_rank']==1 for r in complete)/len(complete) if complete else None,
        mean_selected_actual_reward_rank=float(np.mean([r['selected_actual_reward_rank'] for r in complete])) if complete else None)
    summary['by_speed_tier'] = {}
    for tier in range(3):
        part=[r for r in result if r['speed_tier']==tier]
        if not part:continue
        total_requests=sum(r['requested'] for r in part)
        gaps=[r['predicted_top_two_score_gap'] for r in part if r['predicted_top_two_score_gap'] is not None]
        summary['by_speed_tier'][str(tier)]=dict(contexts=len(part),
            selected_hit_rate=sum(r['selected_hits'] for r in part)/total_requests,
            oracle_hit_rate=sum(r['best_observed_hits'] for r in part)/total_requests,
            selected_is_best_reward_fraction=sum(r['selected_actual_reward_rank']==1 for r in part)/len(part),
            top_two_contains_best_reward_fraction=sum(r['predicted_top_two_contains_best_reward'] for r in part)/len(part),
            mean_selected_actual_reward_rank=float(np.mean([r['selected_actual_reward_rank'] for r in part])),
            mean_reward_regret=float(np.mean([r['best_observed_reward']-r['selected_reward'] for r in part])),
            median_predicted_top_two_gap=float(np.median(gaps)) if gaps else None,
            fraction_predicted_top_two_gap_under_1e6=float(np.mean(np.array(gaps)<1e-6)) if gaps else None)
    return summary, result


def make_panel(chart, count, rng, first_hit_policy='fixed_one_second'):
    sampler = ProceduralChartRhythmSampler(DrumKit.from_layout(), 4, chart,first_hit_policy=first_hit_policy)
    # Small diagnostic panels must not inherit the first few ordered-request
    # anchors (all snare sources). Sample from a full balanced coverage pool.
    pool = sampler.sample(max(256, count), rng)
    labels = sampler.last_strata.copy()
    ids = []
    for tier in range(3):
        required = count//3+int(tier < count%3)
        ids.extend(rng.choice(np.flatnonzero(labels==tier), required, replace=False).tolist())
    rng.shuffle(ids)
    return [pool[i] for i in ids], labels[ids], sampler


def plot_report(out, summary, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ranks = [r['selected_actual_reward_rank'] for r in cases]
    axes[0].hist(ranks, bins=np.arange(.5, 17.5), color='#497aa9')
    axes[0].set_xticks(range(1,17)); axes[0].set_xlabel('Actual reward rank of chart-selected plan')
    axes[0].set_ylabel('Rhythms'); axes[0].set_title('Rank1 means best observed performance')
    percentages = [summary[k]*100 for k in ['selected_pooled_hit_rate','second_pooled_hit_rate','graph_pooled_hit_rate','oracle_pooled_hit_rate']]
    axes[1].bar(['Selected','Second','Graph','Observed oracle'], percentages,
        color=['#497aa9','#8eb0cf','#8f969c','#60a778'])
    for i,value in enumerate(percentages):axes[1].text(i,value+1,f'{value:.1f}%',ha='center')
    axes[1].set_ylim(0,100);axes[1].set_ylabel('Actual credited hits, %')
    axes[1].set_title('Same rhythms, every legal assignment tested')
    for row in cases:
        values=row['predicted_rewards'];axes[2].scatter(range(1,len(values)+1),values,s=12,alpha=.5)
    axes[2].set_xlabel('Planner-predicted rank');axes[2].set_ylabel('Measured reward')
    axes[2].set_title('Every point is an executed plan')
    fig.suptitle(f'Exhaustive hand-plan diagnostic: {summary["contexts"]} rhythms, '
        f'{summary["unique_plan_trials"]} legal plans, {summary["physical_rollouts"]} physical trials')
    fig.text(.5,.01,'Four-group ready-start windows; floor-left excluded. Observed oracle is optimistic. No training feedback.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.93));fig.savefig(out/'all_plans.png',dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/transition_time_no_history_4096_20261007')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--contexts',type=int,default=16)
    parser.add_argument('--envs',type=int,default=256)
    parser.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    parser.add_argument('--seed',type=int,default=20261008)
    parser.add_argument('--first-hit-policy',choices=('tier_random','fixed_one_second'),default='fixed_one_second')
    parser.add_argument('--repeat-controls',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    if min(args.contexts,args.envs)<1 or args.envs%len(args.devices):parser.error('Positive budgets and evenly divided environment capacity required')
    if args.summarize_only:
        import pickle
        saved=pickle.loads((args.out/'planned_trials.pkl').read_bytes())
        reports=json.loads((args.out/'episodes.json').read_text())
        previous=json.loads((args.out/'summary.json').read_text()) if (args.out/'summary.json').exists() else {}
        summary,cases=summarize_contexts(saved['contexts'],saved['trial'],reports,saved['tiers'])
        summary.update(status='complete',physics_seconds=previous.get('physics_seconds'),
            source_run=previous.get('source_run',str(args.run.resolve())))
        write_json(args.out/'summary.json',summary);write_json(args.out/'contexts.json',cases)
        plot_report(args.out,summary,cases);write_json(args.out/'run.json',summary)
        print('[all-plans-result] '+json.dumps(summary),flush=True)
        return
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    cfg=json.loads((args.run/'config.json').read_text());checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:raise ValueError('Decoder changed')
    frozen=args.out/'chart_snapshot.npz';frozen.write_bytes((args.run/'last_chart.npz').read_bytes())
    chart=ConditionalMinimumTimeChart.load(frozen,MinimumTimeChart.load(cfg['args']['table']))
    ready=np.load(cfg['args']['ready_initial']);panel,tiers,sampler=make_panel(chart,args.contexts,np.random.default_rng(args.seed),args.first_hit_policy)
    contexts=[enumerate_context(c,chart,ready) for c in panel]
    trial=assemble_trials(contexts,args.envs,args.repeat_controls)
    with (args.out/'planned_trials.pkl').open('wb') as stream:
        import pickle
        pickle.dump(dict(panel=panel,tiers=tiers,contexts=contexts,trial=trial),stream)
    write_json(args.out/'config.json',dict(source_run=str(args.run.resolve()),chart_snapshot=str(frozen.resolve()),
        chart_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),checkpoint=str(checkpoint),
        checkpoint_sha256=cfg['checkpoint_sha256'],contexts=args.contexts,physical_capacity=args.envs,
        actual_physical_trials=len(trial['episodes']),unique_legal_plans=sum(c['legal_plans'] for c in contexts),
        devices=args.devices,seed=args.seed,groups=4,physics_hz=250,control_hz=50,
        historical_calibration_penalties=False,training=False,repeat_controls=args.repeat_controls,
        simulation_geometry_version=GEOMETRY_VERSION,first_hit_policy=args.first_hit_policy,
        first_hit_ranges_s=[list(r) for r in sampler.speed_ranges]))
    runtime=runtime_args(['--out',str(args.out),'--envs',str(args.envs),'--device',args.devices[0],
        '--rollout-devices',*args.devices,'--physics-hz','250','--decode-chunk','512',
        '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(runtime,key,cfg['args'][key])
    runtime.groups=runtime.horizon=4;runtime.seed=args.seed
    seed_clip=args.out/'scene_seed.json';write_json(seed_clip,sampler.seed_payload())
    evaluator=None
    write_json(args.out/'run.json',dict(status='starting',physical_trials=len(trial['episodes'])))
    try:
        evaluator=DistributedIsaacEvaluator(runtime,checkpoint,seed_clip)
        started=time.monotonic()
        def progress(done,total,tick,ticks):
            states=json.loads((evaluator.session/'progress.json').read_text())['workers']
            wave_done=sum(s.get('episodes',0) if s['completed'] else s.get('offset',0) for s in states)
            write_json(args.out/'progress.json',dict(completed_worker_waves=wave_done,completed_all_trials=done,
                physical_trials=total,tick=tick,ticks=ticks,workers=states))
            print(f'[all-plans] completed_waves={wave_done}/{total} tick={tick}/{ticks}',flush=True)
        reports=evaluator.evaluate(trial['episodes'],progress)
        write_json(args.out/'episodes.json',reports)
        summary,cases=summarize_contexts(contexts,trial,reports,tiers)
        summary.update(status='complete',physics_seconds=time.monotonic()-started,source_run=str(args.run.resolve()))
        write_json(args.out/'summary.json',summary);write_json(args.out/'contexts.json',cases)
        plot_report(args.out,summary,cases)
        write_json(args.out/'run.json',summary)
        print('[all-plans-result] '+json.dumps(summary),flush=True)
    except BaseException as error:
        write_json(args.out/'run.json',dict(status='failed',error=repr(error)));raise
    finally:
        if evaluator is not None:evaluator.close()


if __name__=='__main__':main()
