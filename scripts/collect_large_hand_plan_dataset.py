#!/usr/bin/env python3
"""Fresh scenes, all legal plans, exact training budget and freshly relabeled held-outs."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import GEOMETRY_VERSION
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
from isaac_rl.hand_plan_large_dataset import collection_tasks,task_splits,heldout_inputs
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,build_ranker_batch,unpack_ranker_reports
from isaac_rl.min_time_assignment import MinimumTimeChart
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_neural_hand_plans import make_start_bank,save_torch
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--training-cases',type=int,default=100000)
    p.add_argument('--contexts',type=int,default=256);p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--start-bank-run',type=Path,default=ROOT/'runs/neural_hand_plans_iid_shared_candidate_20261007')
    p.add_argument('--heldout-source',type=Path,default=ROOT/'runs/hand_plan_dataset_iid_16384_20261007')
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:3']);p.add_argument('--resume',action='store_true')
    p.add_argument('--rhythm-source',choices=['iid','groove'],default='iid')
    p.add_argument('--groove-root',type=Path,default=ROOT/'assets/rcc/groove')
    p.add_argument('--training-only',action='store_true',help='Supplement without relabeling fixed held-out inputs')
    cli=p.parse_args();torch.set_num_threads(2)
    if min(cli.training_cases,cli.contexts)<1:raise ValueError('Positive collection budgets required')
    if cli.out.exists() and not cli.resume:raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True,exist_ok=True);(cli.out/'batches').mkdir(exist_ok=True)
    source_config=json.loads((cli.start_bank_run/'config.json').read_text())
    if source_config['simulation_geometry_version']!=GEOMETRY_VERSION:raise ValueError('Incompatible starting-state geometry')
    checkpoint=Path(source_config['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!=source_config['checkpoint_sha256']:raise ValueError('Frozen decoder changed')
    previous=torch.load(cli.start_bank_run/'last.pt',map_location='cpu',weights_only=False)
    scorer=HandPlanRanker(HandPlanRankerConfig(tip_geometry=previous['network']['tip_geometry'])).eval()
    with torch.no_grad():scorer.network[-1].weight.zero_();scorer.network[-1].bias.zero_()
    bank=make_start_bank(scorer.config.tip_geometry);bank.load_state_dict(previous['post_hit_bank'])
    eligibility=MinimumTimeChart.load(source_config['eligibility_source'])
    if cli.rhythm_source=='groove':
        from isaac_rl.recorded_hand_rhythms import RecordedHandRhythmSampler
        sampler=RecordedHandRhythmSampler(DrumKit.from_layout(),4,eligibility,cli.groove_root)
    else:sampler=StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility)
    heldout={} if cli.training_only else heldout_inputs(cli.heldout_source,scorer.config.tip_geometry)
    counts={name:len(rows['panel']) for name,rows in heldout.items()}
    tasks=collection_tasks(cli.training_cases,cli.contexts,counts)
    heldout_signatures={row.tobytes() for split in heldout.values() for row in split['observations']}
    rng=np.random.default_rng(cli.seed);completed=rollouts=training_done=0;pending=None
    config=dict(algorithm='Fresh-scene IID all-legal-plan supervised collection',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()},
        training_cases=cli.training_cases,total_cases=sum(t['count'] for t in tasks),batches=len(tasks),
        contexts_per_batch=cli.contexts,seed=cli.seed,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        simulation_geometry_version=GEOMETRY_VERSION,network=asdict(scorer.config),
        eligibility_source=source_config['eligibility_source'],source_run=str(cli.start_bank_run.resolve()),
        rhythm_sampling=sampler.summary,first_hit_policy='iid_random',physics_hz=250,control_hz=50,
        start_mixture=dict(post_hit=.8,random=.15,ready=.05),learned_weights_used=False,
        collection='Every legal assignment once; fresh processes/scenes per batch; one physical wave',
        split='Training-only supplement' if cli.training_only else 'Exactly requested training cases, followed by relabeled original validation/test inputs',
        training_only=cli.training_only,heldout_counts=counts,heldout_source=None if cli.training_only else str(cli.heldout_source.resolve()),
        heldout_inputs_unchanged=not cli.training_only,heldout_outcomes_reused=False,
        heldout_used_for_training=False,heldout_outcomes_update_state_bank=False,
        fresh_scenes_per_batch=True,dynamic_capacity='Allocate only the actual legal-plan rollouts, rounded across workers')
    if cli.resume:
        old=json.loads((cli.out/'config.json').read_text())
        for key,value in dict(rhythm_source='iid',groove_root=str(ROOT/'assets/rcc/groove'),training_only=False).items():
            old['args'].setdefault(key,value)
        for key in ('args','checkpoint_sha256','network','heldout_counts','fresh_scenes_per_batch'):
            left=dict(old[key],resume=False) if key=='args' else old[key]
            right=dict(config[key],resume=False) if key=='args' else config[key]
            if left!=right:raise ValueError(f'Resume differs in {key}')
        if old['rollout_runtime']['scene_reset_protocol']!='fresh_process_scene_single_wave_v1':raise ValueError('Resume scene protocol differs')
        config['rollout_runtime']=old['rollout_runtime']
        saved=torch.load(cli.out/'collector_state.pt',map_location='cpu',weights_only=False)
        completed,rollouts,training_done,pending=saved['completed_batches'],saved['rollouts'],saved['training_cases'],saved['pending']
        rng.bit_generator.state=saved['rng'];bank.load_state_dict(saved['post_hit_bank'])
    else:
        save_torch(cli.out/'heldout_inputs.pt',heldout)
    write_json(cli.out/'config.json',config);save_torch(cli.out/'splits.pt',task_splits(tasks))
    source=rollout_source_config(source_config)
    args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.devices[0],'--rollout-devices',*cli.devices,
        '--physics-hz','250','--decode-chunk','512','--collision-chunk','1024','--no-plan-diagnostics',
        '--clone-kit-from-source','--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride','near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cli.seed
    seed=cli.out/'scene_seed.json';write_json(seed,sampler.seed_payload())
    def save():
        save_torch(cli.out/'collector_state.pt',dict(completed_batches=completed,rollouts=rollouts,training_cases=training_done,
            pending=pending,rng=rng.bit_generator.state,post_hit_bank=bank.state_dict()))
    def journal():
        rows=[json.loads((cli.out/'batches'/f'metrics_{i+1:05d}.json').read_text()) for i in range(completed)]
        temp=cli.out/'metrics.tmp.jsonl';temp.write_text(''.join(json.dumps(r)+'\n' for r in rows));temp.replace(cli.out/'metrics.jsonl')
        return rows
    save();journal();write_json(cli.out/'run.json',dict(status='starting',completed_batches=completed,training_cases=training_done))
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(cli.out/'tb'),flush_secs=5);writer.add_text('setup/config',json.dumps(config),0);writer.flush()
    if os.environ.get('TB_AUTO_REGISTER','1')!='0':
        registry=ROOT/'runs/tensorboard_active';registry.mkdir(exist_ok=True);link=registry/cli.out.name
        if not link.exists():link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs',timeout=10)):break
            time.sleep(5)
        else:raise RuntimeError('TensorBoard did not list fresh-scene collection')
    evaluator=None
    try:
        while completed<len(tasks):
            task=tasks[completed];started=time.monotonic()
            if pending is None:
                if task['split']=='train':
                    panel=sampler.sample(task['count'],rng,minimum_plan_choices=1);states,start_metrics=bank.sample(rng,task['count'])
                else:
                    rows=heldout[task['split']];offset=task['split_offset']
                    panel=rows['panel'][offset:offset+task['count']];states=rows['states'][offset:offset+task['count']]
                    start_metrics=dict(source='Original fixed held-out physical inputs; fresh labels only')
                pending=build_ranker_batch(scorer,panel,eligibility,16,states)
                if task['split']=='train' and any(row.tobytes() in heldout_signatures for row in pending['observations']):
                    raise ValueError('Training case duplicates a held-out input')
                pending.update(start_metrics=start_metrics,case_offset=task['case_offset'],split=task['split']);save()
            # All real episodes fit in one wave; no reused-scene labels enter this batch.
            args.envs=len(pending['episodes'])
            evaluator=DistributedIsaacEvaluator(args,checkpoint,seed)
            ready=[json.loads((folder/'ready.json').read_text()) for folder,_ in evaluator.workers]
            if any(r['tip_geometry']!=scorer.config.tip_geometry for r in ready):raise ValueError('Isaac/state-bank geometry differs')
            runtime={k:ready[0][k] for k in ('rollout_runtime_version','fixed_hit_scoring','decoder_amp','impact_speed_target')}
            runtime['scene_reset_protocol']='fresh_process_scene_single_wave_v1'
            if 'rollout_runtime' in config and config['rollout_runtime']!=runtime:raise ValueError('Collection runtime changed')
            config['rollout_runtime']=runtime;write_json(cli.out/'config.json',config)
            capacity=int(np.ceil(args.envs/len(ready)))
            if max(len(pending['episodes'])*(i+1)//len(ready)-len(pending['episodes'])*i//len(ready)
                   for i in range(len(ready)))>capacity:raise ValueError('Fresh batch requires more than one physics wave')
            def progress(done,total,tick,ticks):
                write_json(cli.out/'progress.json',dict(batch=completed+1,batches=len(tasks),split=task['split'],
                    completed_training_cases=training_done,target_training_cases=cli.training_cases,
                    completed_cases=sum(t['count'] for t in tasks[:completed]),episodes=total,done=done,tick=tick,ticks=ticks))
            reports=evaluator.evaluate(pending['episodes'],progress,initial_states=pending['episode_states'],collect_post_hits=task['split']=='train')
            outcomes=unpack_ranker_reports(pending,reports)
            if not np.array_equal(pending['observed'],pending['legal']):raise ValueError('Missing legal-plan labels')
            bank_stats=(bank.add([s for r in reports for s in r.get('post_hit_states',[])],rng) if task['split']=='train'
                else dict(updated=False,reason='Held-out outcomes never update the starting-state bank'))
            save_torch(cli.out/'batches'/f'batch_{completed+1:05d}.pt',dict(batch=pending,outcomes=outcomes))
            write_json(cli.out/'batches'/f'episodes_{completed+1:05d}.json',reports)
            evaluator.close();evaluator=None
            metrics=dict(batch=completed+1,split=task['split'],cases=task['count'],rollouts=len(reports),seconds=time.monotonic()-started,
                mean_plan_hit_rate=float(np.mean([r['hit_rate'] for r in reports])),
                mean_best_observed_hit_rate=float(np.where(pending['observed'],outcomes['hit_rates'],-np.inf).max(1).mean()),
                start_states=pending['start_metrics'],bank=bank_stats,fresh_scene=True,single_physics_wave=True,
                actual_environment_capacity=int(np.ceil(len(reports)/len(ready)))*len(ready))
            write_json(cli.out/'batches'/f'metrics_{completed+1:05d}.json',metrics)
            rollouts+=len(reports);training_done+=task['count'] if task['split']=='train' else 0
            completed+=1;pending=None;save();history=journal()
            full=[r['seconds'] for r in history if r['split']=='train' and r['cases']==cli.contexts]
            eta=float(np.mean(full))*sum(t['count']/cli.contexts for t in tasks[completed:]) if full else None
            write_json(cli.out/'run.json',dict(status='running',completed_batches=completed,batches=len(tasks),
                cases=sum(t['count'] for t in tasks[:completed]),training_cases=training_done,target_training_cases=cli.training_cases,
                rollouts=rollouts,mean_full_batch_seconds=float(np.mean(full)) if full else None,estimated_remaining_hours=eta/3600 if eta is not None else None))
            for key,value in metrics.items():
                if isinstance(value,(int,float)):writer.add_scalar('dataset/'+key,value,completed)
            writer.add_scalar('dataset/training_cases',training_done,completed);writer.flush()
            print('[fresh-dataset-result] '+json.dumps(metrics),flush=True)
        save_torch(cli.out/'post_hit_bank.pt',bank.state_dict())
        write_json(cli.out/'run.json',dict(status='complete',completed_batches=completed,cases=config['total_cases'],
            training_cases=training_done,rollouts=rollouts,splits={k:len(v) for k,v in task_splits(tasks).items()}))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',completed_batches=completed,training_cases=training_done,error=repr(exc)));raise
    finally:
        if evaluator:evaluator.close()
        writer.close()


if __name__=='__main__':main()
