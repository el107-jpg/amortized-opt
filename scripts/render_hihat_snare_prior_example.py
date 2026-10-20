#!/usr/bin/env python3
"""Render matched control/prior choices for a saved alternating-groove start."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.hand_plan_ranker import ACTION_BANK
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',type=int,default=0)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--camera',choices=('front','3quarter','side','top'),default='top')
    cli=parser.parse_args();torch.set_num_threads(2)
    if cli.out.exists():raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True)
    saved=torch.load(cli.sweep,map_location='cpu',weights_only=False)
    if not 0<=cli.case<len(saved['states']):raise ValueError('Invalid saved start index')
    state=saved['states'][cli.case];clip=saved['clip']
    cfg=json.loads((ROOT/'runs/hand_plan_dataset_v4_prefirst_200k_20261009/config.json').read_text())
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    source_cfg=json.loads((Path(cfg['source_dataset'])/'config.json').read_text())
    source=rollout_source_config(source_cfg)
    seed_file=ROOT/'runs/hihat_snare_start_sweep_ranker_prior_v4_20261009/scene_seed.json'
    manifest=dict(case=cli.case,state_kind=state.get('kind'),camera=cli.camera,
        state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),
        results={})
    try:
        for label in ('control','prior'):
            action=int(saved['choices'][label][cli.case])
            video=cli.out/f'{label}_{cli.camera}.mp4'
            args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.device,
                '--rollout-devices',cli.device,'--physics-hz','250',
                '--decode-chunk','512','--collision-chunk','1024','--no-plan-diagnostics',
                '--clone-kit-from-source','--clear-reset-contacts',
                '--freeze-hit-scoring','--match-decoder-speed','--rollout-audit'])
            for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                        'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
                setattr(args,key,source['args'][key])
            args.groups=args.horizon=4;args.seed=20261023
            args.capture_video=True;args.video_path=str(video.resolve())
            args.video_camera=cli.camera
            evaluator=None
            try:
                evaluator=DistributedIsaacEvaluator(args,checkpoint,seed_file)
                ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
                geometry_matches=all(np.allclose(ready['tip_geometry'][key],
                    cfg['network']['tip_geometry'][key],rtol=0.,atol=1e-6)
                    for key in cfg['network']['tip_geometry'])
                if (ready['rollout_runtime_version']!='thick_cymbal_top_entry_side5mm_contact_v4'
                        or ready['drum_usd_sha256']!=cfg['drum_usd_sha256']
                        or ready['drum_layout_sha256']!=cfg['drum_layout_sha256']
                        or not geometry_matches):
                    raise ValueError('Rendered runtime differs from v4 scored geometry')
                report=evaluator.evaluate([assigned_clip(clip,ACTION_BANK[action])],
                    initial_states=[state])[0]
            finally:
                if evaluator is not None:evaluator.close()
            write_json(cli.out/f'{label}_report.json',report)
            manifest['results'][label]=dict(action=action,
                hands=''.join('R' if bit else 'L' for bit in ACTION_BANK[action]),
                hits=report['hits'],reward=report['reward'],
                per_hit=[row['hit'] for row in report['per_hit_outcomes']],
                video=str(video.resolve()))
            write_json(cli.out/'manifest.json',manifest)
            print(label,manifest['results'][label],flush=True)
        write_json(cli.out/'run.json',dict(status='complete',**manifest))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',error=repr(exc),**manifest))
        raise


if __name__=='__main__':main()
