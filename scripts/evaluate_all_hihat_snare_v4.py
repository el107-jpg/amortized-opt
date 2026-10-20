#!/usr/bin/env python3
"""Re-evaluate every legal assignment of one saved groove under v4 hit credit."""
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
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:3'])
    cli=p.parse_args();torch.set_num_threads(2)
    cli.out.mkdir(parents=True,exist_ok=True);(cli.out/'actions').mkdir(exist_ok=True)
    cfg=json.loads((cli.dataset/'config.json').read_text())
    source=rollout_source_config(json.loads((Path(cfg['source_dataset'])/'config.json').read_text()))
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    saved=torch.load(cli.checkpoint,map_location='cpu',weights_only=False)
    if saved['checkpoint_sha256']!=cfg['checkpoint_sha256']:
        raise ValueError('Ranker and Isaac decoder differ')
    model=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).eval()
    model.load_state_dict(saved['model'])
    batch=torch.load(cli.dataset/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['batch']
    outcomes=torch.load(cli.dataset/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['outcomes']
    if not 0<=cli.case<len(batch['panel']):raise ValueError('Case outside panel')
    clip=batch['panel'][cli.case];state=batch['states'][cli.case]
    legal=np.flatnonzero(batch['observed'][cli.case])
    if len(legal)!=16:raise ValueError('Expected all 16 assignments to be legal')
    with torch.inference_mode():scores=model(torch.as_tensor(batch['observations'][cli.case:cli.case+1])).numpy()[0]
    neural_rank=legal[np.lexsort((legal,-scores[legal]))]
    choices,_,_=decisions(model,load_dataset_batches(cli.dataset),batch['panel'],batch['states'])
    neural=int(choices['neural_depth2'][cli.case]);graph=int(choices['original_graph'][cli.case])
    state_hash=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
    expected_runtime='thick_cymbal_top_entry_side5mm_contact_v4'
    order=[neural,graph,10,*[int(a) for a in np.random.default_rng(20261044).permutation(16)
        if int(a) not in (neural,graph,10)]]
    order=list(dict.fromkeys(order))
    if set(order)!=set(legal.tolist()):raise ValueError('Assignment order is incomplete')
    for start in range(0,len(order),len(cli.devices)):
        group=[a for a in order[start:start+len(cli.devices)]
            if not (cli.out/'actions'/f'action_{a:02d}.json').exists()]
        if not group:continue
        devices=cli.devices[:len(group)]
        args=runtime_args(['--out',str(cli.out),'--envs',str(len(group)),
            '--device',devices[0],'--rollout-devices',*devices,
            '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024',
            '--no-plan-diagnostics','--clone-kit-from-source','--clear-reset-contacts',
            '--freeze-hit-scoring','--match-decoder-speed','--rollout-audit'])
        for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                    'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
            setattr(args,key,source['args'][key])
        args.groups=args.horizon=4;args.seed=cfg['args']['seed']
        evaluator=None
        try:
            evaluator=DistributedIsaacEvaluator(args,checkpoint,cli.dataset/'scene_seed.json')
            for worker,_ in evaluator.workers:
                ready=json.loads((worker/'ready.json').read_text())
                if (ready['rollout_runtime_version']!=expected_runtime
                        or ready['drum_geometry_version']!='thick_cymbal_top_entry_side_5mm_v1'):
                    raise ValueError('Worker has wrong hit-credit runtime')
                if any(not np.allclose(ready['tip_geometry'][key],cfg['tip_geometry'][key],
                                       rtol=0,atol=1e-6) for key in cfg['tip_geometry']):
                    raise ValueError('Worker tip geometry differs from saved case')
            episodes=[assigned_clip(clip,ACTION_BANK[a]) for a in group]
            reports=evaluator.evaluate(episodes,initial_states=[state]*len(group))
        finally:
            if evaluator is not None:evaluator.close()
        for action,device,report in zip(group,devices,reports):
            row=dict(action=action,hands=''.join('R' if hand else 'L' for hand in ACTION_BANK[action]),
                device=device,neural_score=float(scores[action]),
                old_v2_hits=int(outcomes['hits'][cli.case,action]),
                old_v2_reward=float(outcomes['rewards'][cli.case,action]),
                v4_hits=report['hits'],v4_requested=report['requested'],v4_reward=report['reward'],
                per_hit_outcomes=report['per_hit_outcomes'],
                contact_tracker_stats=report['rollout_audit']['contact_tracker_stats'],
                contact_debug_events=report['rollout_audit']['contact_tracker_debug_events'],
                initial_state_sha256=state_hash)
            write_json(cli.out/'actions'/f'action_{action:02d}.json',row)
            print(f'action {action:02d} {row["hands"]} old={row["old_v2_hits"]}/4 '
                  f'new={row["v4_hits"]}/4 device={device}',flush=True)
        write_json(cli.out/'progress.json',dict(completed_actions=len(list((cli.out/'actions').glob('action_*.json'))),
            total_actions=16,case=cli.case))
    rows=[json.loads((cli.out/'actions'/f'action_{action:02d}.json').read_text()) for action in legal]
    new_reward=np.array([row['v4_reward'] for row in rows]);new_hits=np.array([row['v4_hits'] for row in rows])
    reward_order=legal[np.lexsort((legal,-new_reward))]
    hit_order=legal[np.lexsort((legal,-new_reward,-new_hits))]
    summary=dict(status='complete',case=cli.case,clip=clip.name,
        source_midi=json.loads((cli.dataset/'panel.json').read_text())['windows'][cli.case//3]['source_midi'],
        initial_state_sha256=state_hash,checkpoint_sha256=cfg['checkpoint_sha256'],
        old_label_runtime=cfg['rollout_runtime'],new_rollout_runtime=expected_runtime,
        neural_action=neural,graph_action=graph,
        neural_predicted_order=neural_rank.tolist(),
        v4_hit_then_reward_order=hit_order.tolist(),v4_reward_order=reward_order.tolist(),
        actions=rows,all_fresh_scenes=True,one_physical_env_per_action=True,
        same_initial_state=True,held_out_used_for_training=False)
    write_json(cli.out/'summary.json',summary)
    for rank,action in enumerate(neural_rank,1):
        row=rows[int(action)]
        print(rank,row['hands'],f'score={row["neural_score"]:.3f}',
            f'v2={row["old_v2_hits"]}/4',f'v4={row["v4_hits"]}/4',flush=True)


if __name__=='__main__':main()
