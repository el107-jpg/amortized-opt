#!/usr/bin/env python3
"""Replay three frozen hand plans in fresh Isaac scenes with camera capture."""
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
from isaac_rl.hand_plan_dataset import load_dataset_batches
from isaac_rl.hand_plan_ranker import ACTION_BANK,HandPlanRanker,HandPlanRankerConfig
from scripts.compare_graph_neural_fresh_test import decisions
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=int,default=168)
    p.add_argument('--methods',nargs='+',choices=('neural','graph','best'))
    p.add_argument('--plans',nargs='*',default=[],help='Additional explicit four-hit hand patterns such as RLLR')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--camera',choices=('front','3quarter','side','top'),default='top')
    p.add_argument('--no-video',action='store_true',help='Physics-only diagnostic replay')
    p.add_argument('--new-hit-geometry',action='store_true',help='Evaluate old plans under the new top-entry/side-5mm scoring runtime')
    p.add_argument('--collect-prefirst-snapshots',action='store_true')
    cli=p.parse_args();torch.set_num_threads(2);cli.out.mkdir(parents=True,exist_ok=True)
    config=json.loads((cli.dataset/'config.json').read_text())
    if json.loads((cli.dataset/'run.json').read_text())['status']!='complete':raise ValueError('Groove outcomes incomplete')
    decoder=Path(config['checkpoint'])
    if hashlib.sha256(decoder.read_bytes()).hexdigest()!=config['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    model_state=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    if model_state['checkpoint_sha256']!=config['checkpoint_sha256']:
        raise ValueError('Ranker and simulator use different decoders')
    model=HandPlanRanker(HandPlanRankerConfig(**model_state['network'])).eval()
    model.load_state_dict(model_state['model'])
    batch=torch.load(cli.dataset/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['batch']
    data=load_dataset_batches(cli.dataset)
    if not 0<=cli.case<len(batch['panel']):raise ValueError('Case outside frozen groove panel')
    choices,repairs,fallbacks=decisions(model,data,batch['panel'],batch['states'])
    if repairs['original_graph'] or fallbacks['original_graph']:
        raise ValueError('Graph choice required a repair; choose a directly legal case')
    legal=np.flatnonzero(data['observed'][cli.case])
    best=int(legal[np.argmax(data['hits'][cli.case,legal])])
    selected=dict(neural=int(choices['neural_depth2'][cli.case]),
        graph=int(choices['original_graph'][cli.case]),best=best)
    for pattern in cli.plans:
        pattern=pattern.upper()
        if len(pattern)!=4 or any(hand not in 'LR' for hand in pattern):
            raise ValueError('Explicit plans must contain exactly four L/R hand choices')
        selected[pattern.lower()]=int(np.array([hand=='R' for hand in pattern])@np.array([8,4,2,1]))
    requested=cli.methods if cli.methods is not None else ([] if cli.plans else ['neural','graph','best'])
    requested=list(dict.fromkeys([*requested,*(pattern.lower() for pattern in cli.plans)]))
    clip=batch['panel'][cli.case];state=batch['states'][cli.case]
    source=rollout_source_config(json.loads((Path(config['source_dataset'])/'config.json').read_text()))
    expected_runtime=dict(config['rollout_runtime'])
    if cli.new_hit_geometry:expected_runtime['rollout_runtime_version']='thick_cymbal_top_entry_side5mm_contact_v4'
    manifest=dict(case=cli.case,clip=clip.name,
        source_midi=json.load(open(cli.dataset/'panel.json'))['windows'][cli.case//3]['source_midi'],
        duration_s=clip.duration,initial_state_kind=state.get('kind'),
        initial_state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),
        checkpoint_sha256=config['checkpoint_sha256'],ranker_checkpoint=str(cli.checkpoint.resolve()),
        runtime=expected_runtime,old_label_runtime=config['rollout_runtime'],results={})
    manifest_path=cli.out/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if (previous['case']!=cli.case or previous['initial_state_sha256']!=manifest['initial_state_sha256']
                or previous['checkpoint_sha256']!=manifest['checkpoint_sha256']):
            raise ValueError('Existing video manifest uses a different case or initial state')
        manifest['results'].update(previous['results'])
    seed_file=cli.dataset/'scene_seed.json'
    for name in requested:
        action=selected[name]
        if not data['observed'][cli.case,action]:raise ValueError('Selected plan was not physically observed')
        video=cli.out/f'{name}_{cli.camera}_raw.mp4'
        args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.device,
            '--rollout-devices',cli.device,'--physics-hz','250','--decode-chunk','512',
            '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
            '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed','--rollout-audit'])
        for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                    'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
            setattr(args,key,source['args'][key])
        args.groups=args.horizon=4;args.seed=config['args']['seed']
        args.capture_video=not cli.no_video
        args.video_path=None if cli.no_video else str(video.resolve())
        args.video_camera=cli.camera
        evaluator=None
        try:
            evaluator=DistributedIsaacEvaluator(args,decoder,seed_file)
            ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
            if any(not np.allclose(ready['tip_geometry'][key],config['tip_geometry'][key],
                                   rtol=0.,atol=1e-6)
                   for key in config['tip_geometry']):
                raise ValueError('Rendered stick geometry differs from stored label geometry')
            runtime={key:ready[key] for key in ('rollout_runtime_version','fixed_hit_scoring','decoder_amp','impact_speed_target')}
            runtime['scene_reset_protocol']='fresh_process_scene_single_wave_v1'
            if runtime!=expected_runtime:
                raise ValueError('Rendered rollout runtime differs from stored labels')
            report=evaluator.evaluate([assigned_clip(clip,ACTION_BANK[action])],
                initial_states=[state],collect_post_hits=False,
                collect_prefirst_snapshots=cli.collect_prefirst_snapshots)[0]
        finally:
            if evaluator is not None:evaluator.close()
        write_json(cli.out/f'{name}_replay.json',report)
        manifest['results'][name]=dict(action=action,hand_bits=ACTION_BANK[action].tolist(),
            cached_hits=int(data['hits'][cli.case,action]),cached_requested=int(data['requested'][cli.case,action]),
            replay_hits=report['hits'],replay_requested=report['requested'],
            video=None if cli.no_video else str(video.resolve()),video_camera=cli.camera,
            video_frames=len(report['rollout_audit']['trace']['q']))
        write_json(manifest_path,manifest)
        print(name,manifest['results'][name],flush=True)


if __name__=='__main__':main()
