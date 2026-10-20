#!/usr/bin/env python3
"""Test whether q/qd-only reset reproduces a factual pre-hit continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import shift_clip
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.hand_plan_ranker import ACTION_BANK,cartesian_tips
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def load_traces(root):
    out={}
    for path in root.glob('*/worker_*/request_*.json'):
        if '_progress' in path.name:continue
        payload=json.loads(path.read_text())
        for report in payload.get('reports',[]):
            if report is None:continue
            name=''.join('R' if hit['hand']=='right' else 'L' for hit in report['per_hit_outcomes'])
            out[name]=report
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--trace-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=int,default=168)
    p.add_argument('--time-s',type=float,default=.4)
    p.add_argument('--plans',nargs='+',default=['LRLR','RLRL'])
    p.add_argument('--device',default='cuda:0')
    cli=p.parse_args();torch.set_num_threads(2);cli.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((cli.dataset/'config.json').read_text())
    source=rollout_source_config(json.loads((Path(cfg['source_dataset'])/'config.json').read_text()))
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    batch=torch.load(cli.dataset/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['batch']
    clip=batch['panel'][cli.case];initial=batch['states'][cli.case]
    geometry=cfg['tip_geometry'];trace_map=load_traces(cli.trace_root)
    tick=int(round(cli.time_s/.02))
    if not 0<cli.time_s<min(hit.time for hit in clip.hits)-.10:
        raise ValueError('Snapshot must precede first hit by at least 100 ms')
    results={}
    for pattern in cli.plans:
        pattern=pattern.upper();action=int(np.array([hand=='R' for hand in pattern])@np.array([8,4,2,1]))
        original=trace_map[pattern];trace=original['rollout_audit']['trace']
        if trace['hits'][tick-1]!=0:raise ValueError('A hit occurred before this snapshot')
        q=np.asarray(trace['q'][tick],np.float32);qd=np.asarray(trace['qd'][tick],np.float32)
        state=dict(q=q.tolist(),qd=qd.tolist(),tips=cartesian_tips(q,geometry).tolist(),
            initial_drums=initial['initial_drums'],contact_age_ticks=trace['contact_age'][tick],
            kind='prefirst_snapshot')
        remaining=shift_clip(assigned_clip(clip,ACTION_BANK[action]),-cli.time_s)
        args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.device,
            '--rollout-devices',cli.device,'--physics-hz','250','--decode-chunk','512',
            '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
            '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed','--rollout-audit'])
        for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                    'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
            setattr(args,key,source['args'][key])
        args.groups=args.horizon=4;args.seed=cfg['args']['seed']
        evaluator=None
        try:
            evaluator=DistributedIsaacEvaluator(args,checkpoint,cli.dataset/'scene_seed.json')
            ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
            if ready['rollout_runtime_version']!='thick_cymbal_top_entry_side5mm_contact_v4':
                raise ValueError('Continuation used wrong hit-credit runtime')
            report=evaluator.evaluate([remaining],initial_states=[state])[0]
        finally:
            if evaluator is not None:evaluator.close()
        results[pattern]=dict(action=action,snapshot_time_s=cli.time_s,
            original_hits=original['hits'],replayed_hits=report['hits'],
            original_reward=original['reward'],replayed_reward=report['reward'],
            original_per_hit=[bool(row['hit']) for row in original['per_hit_outcomes']],
            replayed_per_hit=[bool(row['hit']) for row in report['per_hit_outcomes']],
            initial_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
        write_json(cli.out/f'{pattern.lower()}_replay.json',report)
        write_json(cli.out/'summary.json',dict(case=cli.case,results=results,
            source_runtime='thick_cymbal_top_entry_side5mm_contact_v4',
            physical_start_reset=True,decoder_and_contact_state_not_restored=True,
            no_training_feedback=True))
        print(pattern,results[pattern],flush=True)


if __name__=='__main__':main()
