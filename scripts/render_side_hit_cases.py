#!/usr/bin/env python3
"""Render measured upper-side and top-entry contacts in the v4 Isaac kit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit,Hit,RhythmClip
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import DRUM_GEOMETRY_VERSION
from isaac_rl.hand_plan_ranker import ACTION_BANK
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def case_inputs(label):
    if label=='cymbal_upper_side':
        folder=ROOT/'runs/hand_plan_hard_grooves_v2_20261009'
        saved=torch.load(folder/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['batch']
        return saved['panel'][4],saved['states'][4],14,5,.4,folder/'scene_seed.json'
    folder=ROOT/'runs/hihat_snare_alternating_side_audit_speed_v4_20261009'
    original=json.loads((folder/'summary.json').read_text())
    kit=DrumKit.from_layout();times=original['times_s'];drums=(1,0,1,0)
    clip=RhythmClip('hihat_snare_hihat_snare',[
        Hit(float(t),float(t),drum,kit.names[drum],'left',.7)
        for t,drum in zip(times,drums)],float(times[-1]+.3))
    action=3 if label=='hihat_upper_side_slow' else 7
    return clip,original['start_state'],action,1,1.08,folder/'scene_seed.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('cymbal_upper_side','hihat_upper_side_slow','hihat_top_entry'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--camera',choices=('front','3quarter','side','top'),default='side')
    cli=parser.parse_args();torch.set_num_threads(2)
    if cli.out.exists():raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True)
    clip,state,action,drum,target_time,seed_file=case_inputs(cli.case)
    cfg=json.loads((ROOT/'runs/hand_plan_dataset_v4_prefirst_200k_20261009/config.json').read_text())
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    source_cfg=json.loads((Path(cfg['source_dataset'])/'config.json').read_text())
    source=rollout_source_config(source_cfg)
    video=cli.out/f'{cli.case}_{cli.camera}.mp4'
    args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.device,
        '--rollout-devices',cli.device,'--physics-hz','250','--decode-chunk','512',
        '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed',
        '--rollout-audit'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=20261024
    args.capture_video=True;args.video_path=str(video.resolve());args.video_camera=cli.camera
    evaluator=None
    try:
        evaluator=DistributedIsaacEvaluator(args,checkpoint,seed_file)
        ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
        geometry_matches=all(np.allclose(ready['tip_geometry'][key],
            cfg['network']['tip_geometry'][key],atol=1e-6,rtol=0.)
            for key in cfg['network']['tip_geometry'])
        if (ready['rollout_runtime_version']!='thick_cymbal_top_entry_side5mm_contact_v4'
                or ready['drum_geometry_version']!=DRUM_GEOMETRY_VERSION
                or ready['drum_usd_sha256']!=cfg['drum_usd_sha256']
                or ready['drum_layout_sha256']!=cfg['drum_layout_sha256']
                or not geometry_matches):
            raise ValueError('Rendered scene differs from v4 scored geometry')
        report=evaluator.evaluate([assigned_clip(clip,ACTION_BANK[action])],initial_states=[state])[0]
        write_json(cli.out/'report.json',report)
        events=[event for event in report['rollout_audit']['contact_tracker_debug_events']
            if event['drum']==drum and abs(event['time_s']-target_time)<.1]
        beats=[hit for hit in report['per_hit_outcomes']
            if hit['drum']==('hihat' if drum==1 else 'cymbal_right')
            and abs(hit['time_s']-target_time)<.001]
        result=dict(status='complete',case=cli.case,action=action,
            hands=''.join('R' if v else 'L' for v in ACTION_BANK[action]),
            target_time_s=target_time,target_drum='hihat' if drum==1 else 'cymbal_right',
            target_beats=beats,contact_entries=events,
            tracker_stats=report['rollout_audit']['contact_tracker_stats'],
            total_hits=report['hits'],requested=report['requested'],
            video=str(video.resolve()),camera=cli.camera,
            state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
        write_json(cli.out/'summary.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(cli.out/'summary.json',dict(status='failed',error=repr(exc),case=cli.case))
        raise
    finally:
        if evaluator is not None:evaluator.close()


if __name__=='__main__':main()
