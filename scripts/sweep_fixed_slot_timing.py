#!/usr/bin/env python3
"""Sweep one fixed plan at centered single-env slots, independently of grid order."""
import argparse
from dataclasses import replace
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
    p.add_argument('--clear-reset-contacts',action='store_true');p.add_argument('--decoder-fp32',action='store_true')
    cli=p.parse_args();torch.set_num_threads(2);cli.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((cli.sweep/'config.json').read_text());run=json.loads((Path(cfg['args']['run'])/'config.json').read_text());source=rollout_source_config(run)
    saved=torch.load(cli.sweep/'plans.pt',map_location='cpu',weights_only=False)
    base=next(r for r in saved['points'] if r['case']==3 and r['axis']=='control' and r['repeat']==0)
    rid=base['report_ids'][base['actions'].index(15)];clip=saved['episodes'][rid];state=saved['episode_states'][rid]
    offsets=[-20,-10,0,10,11,12,13,14,15,16,17,18,19,20]
    devices=['cuda:0','cuda:1','cuda:2','cuda:3']
    flags=['--out',str(cli.out),'--envs','4','--device',devices[0],'--rollout-devices',*devices,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source','--freeze-hit-scoring','--rollout-audit']
    if cli.clear_reset_contacts:flags+=['--clear-reset-contacts']
    if cli.decoder_fp32:flags+=['--decoder-fp32']
    args=runtime_args(flags)
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cfg['args']['seed'];evaluator=None;records=[];started=time.monotonic()
    write_json(cli.out/'config.json',dict(original_case=3,action=15,offsets_ms=offsets,same_slot=True,
        centered_single_env_per_gpu=True,flags=vars(args),training_feedback=False))
    write_json(cli.out/'run.json',dict(status='starting'))
    try:
        evaluator=DistributedIsaacEvaluator(args,Path(cfg['checkpoint']),cli.sweep/'scene_seed.json')
        # Four offsets run in parallel; each is then repeated in the same worker's
        # sole slot. The final short group adds explicit cross-device repeats
        # of its first offset; these are controls, not fabricated missing labels.
        for start in range(0,len(offsets),4):
            values=offsets[start:start+4]
            if len(values)<4:values+=values[:1]*(4-len(values))
            episodes=[replace(clip,hits=[replace(h,time=h.time+delta*.001,raw_time=h.raw_time+delta*.001) for h in clip.hits],
                duration=clip.duration+delta*.001) for delta in values]
            for repeat in range(2):
                reports=evaluator.evaluate(episodes,lambda done,total,tick,ticks:
                    write_json(cli.out/'progress.json',dict(batch=start//4+1,repeat=repeat+1,tick=tick,ticks=ticks)),initial_states=[state]*4)
                records.extend(dict(offset_ms=delta,worker=i,repeat=repeat,report=report) for i,(delta,report) in enumerate(zip(values,reports)))
                write_json(cli.out/'records.json',records)
                write_json(cli.out/'run.json',dict(status='running',completed_rollouts=len(records)))
        write_json(cli.out/'run.json',dict(status='complete',rollouts=len(records),elapsed_s=time.monotonic()-started))
        print('Fixed-slot timing sweep complete',flush=True)
    except BaseException as exc:write_json(cli.out/'run.json',dict(status='failed',error=repr(exc)));raise
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':main()
