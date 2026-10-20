#!/usr/bin/env python3
"""Collect an untouched all-plan test set from corrected-runtime starts."""
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
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,build_ranker_batch,unpack_ranker_reports
from isaac_rl.min_time_assignment import MinimumTimeChart
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json
from scripts.train_neural_hand_plans import make_start_bank,save_torch


def signatures(observations):
    return {hashlib.blake2b(np.ascontiguousarray(row).tobytes(),digest_size=16).digest()
            for row in observations}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-dataset',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--panel',type=Path,help='Frozen test-only rhythm panel; defaults to IID sampling')
    p.add_argument('--cases',type=int,default=2048)
    p.add_argument('--contexts',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261032)
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:3'])
    p.add_argument('--resume',action='store_true')
    cli=p.parse_args();torch.set_num_threads(2)
    if min(cli.cases,cli.contexts)<1:raise ValueError('Positive test-case budgets required')
    cli.out.mkdir(parents=True,exist_ok=True);(cli.out/'batches').mkdir(exist_ok=True)
    if (cli.out/'config.json').exists() and not cli.resume:raise FileExistsError(cli.out/'config.json')
    source=json.loads((cli.source_dataset/'config.json').read_text())
    expected_runtime=source['rollout_runtime']
    if (expected_runtime['rollout_runtime_version']!='matched_decoder_speed_contact_reset_v2'
            or expected_runtime['scene_reset_protocol']!='fresh_process_scene_single_wave_v1'):
        raise ValueError('Source dataset does not have corrected simulator labels')
    if source['simulation_geometry_version']!=GEOMETRY_VERSION:raise ValueError('Source geometry differs')
    provenance=json.loads((cli.bank.parent/'bank_provenance.json').read_text())
    if provenance['legacy_states_used'] or provenance['source_runtime']!=expected_runtime['rollout_runtime_version']:
        raise ValueError('Starting-state bank includes legacy states')
    bank_hash=hashlib.sha256(cli.bank.read_bytes()).hexdigest()
    checkpoint=Path(source['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    geometry=source['network']['tip_geometry']
    bank=make_start_bank(geometry)
    bank.load_state_dict(torch.load(cli.bank,map_location='cpu',weights_only=False))
    if len(bank.states)!=13 or any(len(rows)!=32 for rows in bank.states.values()):
        raise ValueError('Fresh-only starting-state bank lacks full allowed hand/drum coverage')
    eligibility=MinimumTimeChart.load(source['eligibility_source'])
    sampler=StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility)
    fixed_panel=None;panel_hash=None;panel_summary=None
    if cli.panel:
        panel_hash=hashlib.sha256(cli.panel.read_bytes()).hexdigest()
        saved_panel=torch.load(cli.panel,map_location='cpu',weights_only=False)
        fixed_panel=saved_panel['panel'];panel_summary=saved_panel['selection']
        if len(fixed_panel)!=cli.cases:raise ValueError('Frozen panel case count differs')
    scorer=HandPlanRanker(HandPlanRankerConfig(tip_geometry=geometry)).eval()
    with torch.no_grad():scorer.network[-1].weight.zero_();scorer.network[-1].bias.zero_()
    rng=np.random.default_rng(cli.seed)
    old_hashes=set()
    for path in sorted((cli.source_dataset/'batches').glob('batch_*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        old_hashes.update(signatures(saved['batch']['observations']))
    completed=rollouts=0;pending=None;new_hashes=set()
    config=dict(algorithm='Untouched fresh corrected-runtime hand-plan test',
        purpose='test_only_no_training',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()},
        cases=cli.cases,contexts_per_batch=cli.contexts,seed=cli.seed,
        source_dataset=str(cli.source_dataset.resolve()),bank=str(cli.bank.resolve()),bank_sha256=bank_hash,
        bank_provenance=provenance,checkpoint=str(checkpoint),checkpoint_sha256=source['checkpoint_sha256'],
        simulation_geometry_version=GEOMETRY_VERSION,tip_geometry=geometry,
        eligibility_source=source['eligibility_source'],
        rhythm_sampling=sampler.summary if fixed_panel is None else panel_summary,
        panel=None if cli.panel is None else str(cli.panel.resolve()),panel_sha256=panel_hash,
        first_hit_policy='iid_random',physics_hz=250,control_hz=50,
        start_mixture=dict(post_hit=.8,random=.15,ready=.05),
        collection='All legal plans once, fresh scene and one physics wave per batch',
        learned_weights_used=False,held_out_used_for_training=False,rollout_runtime=expected_runtime)
    if cli.resume:
        old=json.loads((cli.out/'config.json').read_text())
        for key in ('cases','contexts_per_batch','seed','source_dataset','bank_sha256','checkpoint_sha256',
                    'tip_geometry','rollout_runtime','panel_sha256'):
            if old[key]!=config[key]:raise ValueError(f'Resume differs in {key}')
        saved=torch.load(cli.out/'collector_state.pt',map_location='cpu',weights_only=False)
        completed,rollouts,pending=saved['completed'],saved['rollouts'],saved['pending']
        rng.bit_generator.state=saved['rng']
        new_hashes=set(saved['new_hashes'])
    write_json(cli.out/'config.json',config)
    seed_file=cli.out/'scene_seed.json';write_json(seed_file,sampler.seed_payload())
    source_args=rollout_source_config(source)
    args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.devices[0],
        '--rollout-devices',*cli.devices,'--physics-hz','250','--decode-chunk','512',
        '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source_args['args'][key])
    args.groups=args.horizon=4;args.seed=cli.seed
    def save():
        save_torch(cli.out/'collector_state.pt',dict(completed=completed,rollouts=rollouts,pending=pending,
            rng=rng.bit_generator.state,new_hashes=list(new_hashes)))
    save();write_json(cli.out/'run.json',dict(status='starting',completed_batches=completed,rollouts=rollouts))
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(cli.out/'tb'),flush_secs=5)
    writer.add_text('setup/config',json.dumps(config),0);writer.flush()
    if os.environ.get('TB_AUTO_REGISTER','1')!='0':
        registry=ROOT/'runs/tensorboard_active';registry.mkdir(exist_ok=True);link=registry/cli.out.name
        if not link.exists():link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs',timeout=10)):break
            time.sleep(5)
        else:raise RuntimeError('TensorBoard did not list fresh test collection')
    evaluator=None
    try:
        total_batches=(cli.cases+cli.contexts-1)//cli.contexts
        while completed<total_batches:
            started=time.monotonic()
            if pending is None:
                count=min(cli.contexts,cli.cases-completed*cli.contexts)
                panel=(sampler.sample(count,rng,minimum_plan_choices=1) if fixed_panel is None else
                    fixed_panel[completed*cli.contexts:completed*cli.contexts+count])
                states,start_metrics=bank.sample(rng,count)
                pending=build_ranker_batch(scorer,panel,eligibility,16,states)
                hashes=signatures(pending['observations'])
                if len(hashes)!=count or hashes & old_hashes or hashes & new_hashes:
                    raise ValueError('Fresh test duplicates an existing dataset or prior test case')
                pending.update(start_metrics=start_metrics,case_offset=completed*cli.contexts)
                new_hashes.update(hashes);save()
            args.envs=len(pending['episodes'])
            evaluator=DistributedIsaacEvaluator(args,checkpoint,seed_file)
            ready=[json.loads((folder/'ready.json').read_text()) for folder,_ in evaluator.workers]
            if any(r['tip_geometry']!=geometry for r in ready):raise ValueError('Isaac tip geometry differs')
            for row in ready:
                runtime={key:row[key] for key in ('rollout_runtime_version','fixed_hit_scoring','decoder_amp','impact_speed_target')}
                runtime['scene_reset_protocol']='fresh_process_scene_single_wave_v1'
                if runtime!=expected_runtime:raise ValueError('Fresh test simulator runtime differs from training labels')
            capacity=int(np.ceil(args.envs/len(ready)))
            if max(args.envs*(i+1)//len(ready)-args.envs*i//len(ready)
                   for i in range(len(ready)))>capacity:
                raise ValueError('Fresh test requires more than one physics wave')
            def progress(done,total,tick,ticks):
                write_json(cli.out/'progress.json',dict(batch=completed+1,batches=total_batches,
                    completed_cases=completed*cli.contexts,episodes=total,done=done,tick=tick,ticks=ticks))
            reports=evaluator.evaluate(pending['episodes'],progress,
                initial_states=pending['episode_states'],collect_post_hits=False)
            outcomes=unpack_ranker_reports(pending,reports)
            if not np.array_equal(pending['observed'],pending['legal']):
                raise ValueError('Fresh test is missing legal-plan labels')
            save_torch(cli.out/'batches'/f'batch_{completed+1:05d}.pt',dict(batch=pending,outcomes=outcomes))
            write_json(cli.out/'batches'/f'episodes_{completed+1:05d}.json',reports)
            evaluator.close();evaluator=None
            metrics=dict(batch=completed+1,cases=len(pending['panel']),rollouts=len(reports),
                seconds=time.monotonic()-started,start_states=pending['start_metrics'],
                mean_plan_hit_rate=float(np.mean([r['hit_rate'] for r in reports])),
                fresh_scene=True,single_physics_wave=True)
            write_json(cli.out/'batches'/f'metrics_{completed+1:05d}.json',metrics)
            rollouts+=len(reports);completed+=1;pending=None;save()
            write_json(cli.out/'run.json',dict(status='running',completed_batches=completed,
                batches=total_batches,cases=min(cli.cases,completed*cli.contexts),rollouts=rollouts))
            writer.add_scalar('test_collection/rollouts',rollouts,completed)
            writer.add_scalar('test_collection/completed_cases',min(cli.cases,completed*cli.contexts),completed)
            writer.flush();print('[fresh-test] '+json.dumps(metrics),flush=True)
        write_json(cli.out/'run.json',dict(status='complete',completed_batches=completed,
            cases=cli.cases,rollouts=rollouts,runtime=expected_runtime))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',completed_batches=completed,error=repr(exc)));raise
    finally:
        if evaluator is not None:evaluator.close()
        writer.close()


if __name__=='__main__':main()
