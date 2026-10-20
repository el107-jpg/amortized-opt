#!/usr/bin/env python3
"""Compare legacy, frozen scoring and FP32 Isaac controls with command traces."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweep',type=Path,required=True);p.add_argument('--profile',choices=['legacy','frozen','fp32','cleared','cleared_fp32'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--envs',type=int,default=64)
    p.add_argument('--match-decoder-speed',action='store_true',help='Opt into corrected speed scoring; archived audit profiles default to original mapping')
    cli=p.parse_args();torch.set_num_threads(2)
    cli.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((cli.sweep/'config.json').read_text());run_cfg=json.loads((Path(cfg['args']['run'])/'config.json').read_text());source=rollout_source_config(run_cfg)
    plans=torch.load(cli.sweep/'plans.pt',map_location='cpu',weights_only=False)
    ids=[i for point in plans['points'] if point['axis']=='control' and point['repeat']==0 for i in point['report_ids']]
    episodes=[plans['episodes'][i] for i in ids];states=[plans['episode_states'][i] for i in ids]
    flags=['--out',str(cli.out),'--envs',str(cli.envs),'--device',cli.device,'--rollout-devices',cli.device,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source','--rollout-audit']
    if cli.profile in ['frozen','fp32','cleared_fp32']:flags+=['--freeze-hit-scoring']
    else:flags+=['--no-freeze-hit-scoring']
    if cli.profile in ['fp32','cleared_fp32']:flags+=['--decoder-fp32']
    if cli.profile in ['cleared','cleared_fp32']:flags+=['--clear-reset-contacts']
    else:flags+=['--no-clear-reset-contacts']
    args=runtime_args(flags)
    args.match_decoder_speed=cli.match_decoder_speed
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cfg['args']['seed']
    write_json(cli.out/'config.json',dict(profile=cli.profile,args=vars(args),cases=len(ids),training_feedback=False))
    write_json(cli.out/'run.json',dict(status='starting'));evaluator=None;all_reports=[];started=time.monotonic()
    try:
        evaluator=DistributedIsaacEvaluator(args,Path(cfg['checkpoint']),cli.sweep/'scene_seed.json')
        for repeat in range(3):
            reports=evaluator.evaluate(episodes,lambda done,total,tick,ticks:
                write_json(cli.out/'progress.json',dict(repeat=repeat+1,tick=tick,ticks=ticks)),initial_states=states)
            all_reports.append(reports);write_json(cli.out/f'repeat_{repeat}.json',reports)
            write_json(cli.out/'run.json',dict(status='running',repeats=repeat+1))
        reward=np.array([[r['reward'] for r in rows] for rows in all_reports]);hits=np.array([[r['hits'] for r in rows] for rows in all_reports]);ranges=np.ptp(reward,axis=0)
        divergence=[]
        for case in range(len(ids)):
            traces=[r[case]['rollout_audit']['trace'] for r in all_reports]
            command=np.array([t['command'] for t in traces]);q=np.array([t['q'] for t in traces]);hi=np.array([t['hit_index'] for t in traces])
            command_delta=np.ptp(command,axis=0).max(1);q_delta=np.ptp(q,axis=0).max(1)
            divergence.append(dict(case_plan=case,original_report_id=ids[case],hits=hits[:,case].tolist(),
                first_command_difference_tick=int(np.flatnonzero(command_delta>1e-6)[0]) if (command_delta>1e-6).any() else None,
                first_q_difference_tick=int(np.flatnonzero(q_delta>1e-6)[0]) if (q_delta>1e-6).any() else None,
                initial_command_max_difference_rad=float(command_delta[0]),initial_q_max_difference_rad=float(q_delta[0]),
                first_hit_index_difference_tick=int(np.flatnonzero(np.ptp(hi,axis=0).any(1))[0]) if np.ptp(hi,axis=0).any() else None))
        summary=dict(status='complete',profile=cli.profile,envs=cli.envs,elapsed_s=time.monotonic()-started,
            reward_range_median=float(np.median(ranges)),reward_range_p95=float(np.quantile(ranges,.95)),reward_range_max=float(ranges.max()),
            fraction_hit_count_varies=float(np.mean(np.ptp(hits,axis=0)>0)),
            scoring=[dict(before=r[0]['rollout_audit']['scoring_before'],after=r[0]['rollout_audit']['scoring_after']) for r in all_reports],
            divergence=divergence)
        write_json(cli.out/'summary.json',summary);write_json(cli.out/'run.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['divergence','scoring']},indent=2),flush=True)
    except BaseException as exc:write_json(cli.out/'run.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':main()
