#!/usr/bin/env python3
"""Separate same-slot repeats from shuffled-slot variation after a local sweep."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sweep',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    cli=p.parse_args();torch.set_num_threads(2)
    if cli.out.exists():raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True)
    cfg=json.loads((cli.sweep/'config.json').read_text());run_cfg=json.loads((Path(cfg['args']['run'])/'config.json').read_text())
    source=rollout_source_config(run_cfg);plans=torch.load(cli.sweep/'plans.pt',map_location='cpu',weights_only=False)
    ids=[i for point in plans['points'] if point['axis']=='control' and point['repeat']==0 for i in point['report_ids']]
    episodes=[plans['episodes'][i] for i in ids];states=[plans['episode_states'][i] for i in ids]
    devices=['cuda:0','cuda:1','cuda:2','cuda:3']
    args=runtime_args(['--out',str(cli.out),'--envs','64','--device',devices[0],'--rollout-devices',*devices,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cfg['args']['seed']
    write_json(cli.out/'config.json',dict(sweep=str(cli.sweep.resolve()),control_episodes=len(ids),env_capacity=64,
        same_slot_repeats=3,shuffled_slot_repeats=2,training_feedback=False,checkpoint_sha256=cfg['checkpoint_sha256']))
    evaluator=None;reports=[];permutations=[];started=time.monotonic()
    write_json(cli.out/'run.json',dict(status='starting'))
    try:
        evaluator=DistributedIsaacEvaluator(args,Path(cfg['checkpoint']),cli.sweep/'scene_seed.json')
        rng=np.random.default_rng(20261018)
        for repeat in range(5):
            permutation=np.arange(len(ids)) if repeat<3 else rng.permutation(len(ids))
            trial=evaluator.evaluate([episodes[i] for i in permutation],lambda done,total,tick,ticks:
                write_json(cli.out/'progress.json',dict(repeat=repeat+1,total_repeats=5,tick=tick,ticks=ticks)),
                initial_states=[states[i] for i in permutation])
            restored=[trial[i] for i in np.argsort(permutation)]
            reports.append(restored);permutations.append(permutation.tolist())
            write_json(cli.out/f'repeat_{repeat:02d}.json',restored)
            write_json(cli.out/'run.json',dict(status='running',completed_repeats=repeat+1))
        reward=np.array([[r['reward'] for r in rows] for rows in reports]);hits=np.array([[r['hits'] for r in rows] for rows in reports])
        def stats(values,count):
            delta=np.ptp(values,axis=0)
            return dict(reward_range_median=float(np.median(delta)),reward_range_p95=float(np.quantile(delta,.95)),
                reward_range_max=float(delta.max()),fraction_hit_count_varies=float(np.mean(np.ptp(count,axis=0)>0)))
        result=dict(status='complete',control_episodes=len(ids),same_slot=stats(reward[:3],hits[:3]),
            all_slots=stats(reward,hits),elapsed_s=time.monotonic()-started,
            original_report_ids=ids,permutations=permutations,
            limitation='64-env scene versus4096 in the sweep; cross-slot/GPU effects include world-coordinate precision and scene-size/batching changes.')
        write_json(cli.out/'summary.json',result);write_json(cli.out/'run.json',result);print(json.dumps(result,indent=2),flush=True)
    except BaseException as exc:write_json(cli.out/'run.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':main()
