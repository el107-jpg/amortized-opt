#!/usr/bin/env python3
"""Collect IID256-case panels, testing every legal plan for supervised learning."""
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
from isaac_rl.decoder_scene import GEOMETRY_VERSION
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,build_ranker_batch,unpack_ranker_reports
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler,split_case_ids
from scripts.train_neural_hand_plans import make_start_bank,save_torch
from scripts.train_chart_cma import parse_args as runtime_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batches',type=int,default=32)
    p.add_argument('--contexts',type=int,default=256);p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--start-bank-run',type=Path,default=ROOT/'runs/neural_hand_plans_all16_256_20261007')
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--training-only',action='store_true',help='Fresh extension cases reserved entirely for training')
    cli=p.parse_args();torch.set_num_threads(2)
    if min(cli.batches,cli.contexts)<=0:raise ValueError('Positive collection budget required')
    if cli.out.exists() and not cli.resume:raise FileExistsError(cli.out)
    if cli.resume:
        old_cfg=json.loads((cli.out/'config.json').read_text())
        runtime=old_cfg.get('rollout_runtime',{})
        if (runtime.get('rollout_runtime_version')!='matched_decoder_speed_contact_reset_v2'
                or not runtime.get('fixed_hit_scoring') or not runtime.get('impact_speed_target')):
            raise ValueError('Collection predates matched decoder speeds/contact reset/scoring; collect fresh labels instead of mixing runtimes')
    cli.out.mkdir(parents=True,exist_ok=True);(cli.out/'batches').mkdir(exist_ok=True)
    cfg=json.loads((cli.start_bank_run/'config.json').read_text())
    if cfg.get('simulation_geometry_version')!=GEOMETRY_VERSION:raise ValueError('Starting-state bank uses old geometry')
    checkpoint=Path(cfg['checkpoint']);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest!=cfg['checkpoint_sha256']:raise ValueError('Frozen decoder changed')
    previous=torch.load(cli.start_bank_run/'last.pt',map_location='cpu',weights_only=False)
    eligibility=MinimumTimeChart.load(cfg['eligibility_source'])
    sampler=StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility)
    scorer=HandPlanRanker(HandPlanRankerConfig(tip_geometry=previous['network'].get('tip_geometry'))).eval()
    with torch.no_grad():scorer.network[-1].weight.zero_();scorer.network[-1].bias.zero_()
    bank=make_start_bank(scorer.config.tip_geometry);bank.load_state_dict(previous['post_hit_bank'])
    rng=np.random.default_rng(cli.seed);completed,rollouts,pending=0,0,None
    if cli.resume:
        saved=torch.load(cli.out/'collector_state.pt',map_location='cpu',weights_only=False)
        if (saved['batches']!=cli.batches or saved['contexts']!=cli.contexts or saved['checkpoint_sha256']!=digest
                or saved.get('training_only',False)!=cli.training_only):
            raise ValueError('Resume collection budget or decoder differs')
        completed,rollouts,pending=saved['completed_batches'],saved['rollouts'],saved['pending']
        rng.bit_generator.state=saved['rng'];bank.load_state_dict(saved['post_hit_bank'])
    config=dict(algorithm='IID all-legal-hand-plan supervised dataset',batches=cli.batches,contexts_per_batch=cli.contexts,
        seed=cli.seed,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        simulation_geometry_version=GEOMETRY_VERSION,network=asdict(scorer.config),
        eligibility_source=cfg['eligibility_source'],source_run=str(cli.start_bank_run.resolve()),
        rhythm_sampling=sampler.summary,first_hit_policy='iid_random',physics_hz=250,control_hz=50,
        start_mixture=dict(post_hit=.8,random=.15,ready=.05),
        collection='Every legal assignment once, canonical action order; no learned selection',
        learned_weights_used=False,starting_state_bank_reused=True,
        split='Training-only extension' if cli.training_only else 'Whole rhythm/start-state cases80/10/10; all their plans stay together',
        training_only=cli.training_only,split_seed=202610121)
    write_json(cli.out/'config.json',config)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(cli.out/'tb'),flush_secs=5);writer.add_text('setup/config',json.dumps(config),0);writer.flush()
    if os.environ.get('TB_AUTO_REGISTER','1')!='0':
        registry=ROOT/'runs/tensorboard_active';registry.mkdir(exist_ok=True);link=registry/cli.out.name
        if not link.exists():link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs',timeout=10)):break
            time.sleep(5)
        else:raise RuntimeError('TensorBoard did not list dataset collection')
    source=json.loads((Path(cfg['args']['source_run'])/'config.json').read_text())
    args=runtime_args(['--out',str(cli.out),'--envs',str(cli.contexts*16),'--device',cli.devices[0],
        '--rollout-devices',*cli.devices,'--physics-hz','250','--decode-chunk','512','--collision-chunk','1024',
        '--no-plan-diagnostics','--clone-kit-from-source'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cli.seed
    seed=cli.out/'scene_seed.json';write_json(seed,sampler.seed_payload());evaluator=None
    def save():
        save_torch(cli.out/'collector_state.pt',dict(completed_batches=completed,rollouts=rollouts,pending=pending,
            rng=rng.bit_generator.state,post_hit_bank=bank.state_dict(),batches=cli.batches,contexts=cli.contexts,
            checkpoint_sha256=digest,training_only=cli.training_only))
    save();write_json(cli.out/'run.json',dict(status='starting',completed_batches=completed))
    try:
        evaluator=DistributedIsaacEvaluator(args,checkpoint,seed)
        ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text());geometry=ready['tip_geometry']
        config['rollout_runtime']={key:ready[key] for key in ('rollout_runtime_version','fixed_hit_scoring','decoder_amp','impact_speed_target')}
        write_json(cli.out/'config.json',config)
        if geometry!=scorer.config.tip_geometry:raise ValueError('State-bank geometry differs from current Isaac')
        while completed<cli.batches:
            started=time.monotonic()
            if pending is None:
                panel=sampler.sample(cli.contexts,rng,minimum_plan_choices=1)
                states,start_metrics=bank.sample(rng,cli.contexts)
                pending=build_ranker_batch(scorer,panel,eligibility,16,states)
                pending['start_metrics']=start_metrics;pending['case_offset']=completed*cli.contexts;save()
            def progress(done,total,tick,ticks):
                write_json(cli.out/'progress.json',dict(batch=completed+1,batches=cli.batches,
                    completed_cases=completed*cli.contexts,episodes=total,done=done,tick=tick,ticks=ticks))
                print(f'[dataset] batch={completed+1}/{cli.batches} tick={tick}/{ticks}',flush=True)
            reports=evaluator.evaluate(pending['episodes'],progress,initial_states=pending['episode_states'],collect_post_hits=True)
            outcomes=unpack_ranker_reports(pending,reports)
            if not np.array_equal(pending['observed'],pending['legal']):raise RuntimeError('Dataset did not execute every legal action')
            bank_stats=bank.add([s for r in reports for s in r.get('post_hit_states',[])],rng)
            save_torch(cli.out/'batches'/f'batch_{completed+1:05d}.pt',dict(batch=pending,outcomes=outcomes))
            write_json(cli.out/'batches'/f'episodes_{completed+1:05d}.json',reports)
            metrics=dict(batch=completed+1,cases=cli.contexts,rollouts=len(reports),seconds=time.monotonic()-started,
                mean_plan_hit_rate=float(np.mean([r['hit_rate'] for r in reports])),
                mean_best_observed_hit_rate=float(np.where(pending['observed'],outcomes['hit_rates'],-np.inf).max(1).mean()),
                first_hit_min_s=min(c.hits[0].time for c in pending['panel']),
                first_hit_max_s=max(c.hits[0].time for c in pending['panel']),
                legal_bank_counts={str(k):int((pending['observed'].sum(1)==k).sum()) for k in np.unique(pending['observed'].sum(1))},
                start_states=pending['start_metrics'],bank=bank_stats)
            rollouts+=len(reports);completed+=1;pending=None;save()
            with (cli.out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
            for key,value in metrics.items():
                if isinstance(value,(int,float)):writer.add_scalar('dataset/'+key,value,completed)
            writer.add_scalar('dataset/total_cases',completed*cli.contexts,completed);writer.flush()
            write_json(cli.out/'run.json',dict(status='running',completed_batches=completed,
                cases=completed*cli.contexts,rollouts=rollouts))
            print('[dataset-result] '+json.dumps(metrics),flush=True)
        splits=(dict(train=np.arange(completed*cli.contexts),validation=np.array([],dtype=int),test=np.array([],dtype=int))
                if cli.training_only else split_case_ids(completed*cli.contexts,config['split_seed']))
        save_torch(cli.out/'splits.pt',splits);save_torch(cli.out/'post_hit_bank.pt',bank.state_dict())
        write_json(cli.out/'run.json',dict(status='complete',completed_batches=completed,cases=completed*cli.contexts,
            rollouts=rollouts,splits={key:len(ids) for key,ids in splits.items()}))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',completed_batches=completed,error=repr(exc)));raise
    finally:
        writer.close()
        if evaluator is not None:evaluator.close()


if __name__=='__main__':main()
