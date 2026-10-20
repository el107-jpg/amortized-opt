#!/usr/bin/env python3
"""Collect all-plan labels plus factual pre-first-hit rows to a row budget.

The v4 CLI default is retained for archived reproducibility; the current
runtime is v5 and should be launched via collect_v5_prefirst_dataset.py.
"""
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
from amortized_opt.rhythm import DrumKit,shift_clip
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.decoder_scene import (GEOMETRY_VERSION,DRUM_GEOMETRY_VERSION,
    CONTACT_SCORING_VERSION)
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
from isaac_rl.hand_plan_ranker import (HandPlanRanker,HandPlanRankerConfig,OBS_DIM,
    build_ranker_batch,encode_clip,unpack_ranker_reports)
from isaac_rl.min_time_assignment import MinimumTimeChart
from scripts.evaluate_neural_hand_plans import rollout_source_config
from scripts.train_chart_cma import parse_args as runtime_args,write_json
from scripts.train_neural_hand_plans import make_start_bank,save_torch


def factual_rows(batch,reports,case_offset,split_codes):
    """One pointwise continuation target for each real pre-first-hit snapshot."""
    geometry=batch['tip_geometry'];rows=[]
    for (case,action),report in zip(batch['labels'],reports):
        case=int(case);action=int(action)
        for snapshot in report.get('prefirst_snapshots',[]):
            shifted=shift_clip(batch['panel'][case],-snapshot['time_s'])
            observation=encode_clip(shifted,q=snapshot['q'],qd=snapshot['qd'],
                tips=snapshot['tips'],initial_drums=snapshot['initial_drums'],
                tip_geometry=geometry)
            if shifted.hits[0].time<=0 or len(shifted.hits)!=len(batch['panel'][case].hits):
                raise ValueError('Snapshot lost a requested hit')
            if not np.isfinite(snapshot['return_to_go_reward']):
                raise FloatingPointError('Nonfinite factual return-to-go')
            rows.append((observation,action,float(snapshot['return_to_go_reward']),
                float(snapshot['future_hit_rate']),case_offset+case,
                float(snapshot['time_s']),int(split_codes[case])))
    if not rows:
        return dict(observations=np.zeros((0,OBS_DIM),np.float32),
            actions=np.zeros(0,np.int64),rewards=np.zeros(0,np.float32),
            hit_rates=np.zeros(0,np.float32),case_ids=np.zeros(0,np.int64),
            elapsed_s=np.zeros(0,np.float32),splits=np.zeros(0,np.int8))
    return dict(observations=np.stack([row[0] for row in rows]),
        actions=np.asarray([row[1] for row in rows],np.int64),
        rewards=np.asarray([row[2] for row in rows],np.float32),
        hit_rates=np.asarray([row[3] for row in rows],np.float32),
        case_ids=np.asarray([row[4] for row in rows],np.int64),
        elapsed_s=np.asarray([row[5] for row in rows],np.float32),
        splits=np.asarray([row[6] for row in rows],np.int8))


def main(*,default_runtime='v4'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scoring-version',choices=('v4','v5'),default=default_runtime)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-dataset',type=Path,required=True,
        help='Decoder/eligibility provenance; its old outcome labels are never reused')
    p.add_argument('--initial-bank',type=Path,required=True)
    p.add_argument('--target-datapoints',type=int,default=200000)
    p.add_argument('--contexts',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261050)
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:3'])
    p.add_argument('--resume',action='store_true')
    cli=p.parse_args();torch.set_num_threads(2)
    requested_runtime=(CONTACT_SCORING_VERSION if cli.scoring_version=='v5'
        else 'thick_cymbal_top_entry_side5mm_contact_v4')
    if min(cli.target_datapoints,cli.contexts)<1:raise ValueError('Positive collection budgets required')
    if (cli.out/'config.json').exists() and not cli.resume:raise FileExistsError(cli.out/'config.json')
    cli.out.mkdir(parents=True,exist_ok=True);(cli.out/'batches').mkdir(exist_ok=True)
    source_cfg=json.loads((cli.source_dataset/'config.json').read_text())
    if source_cfg['simulation_geometry_version']!=GEOMETRY_VERSION:
        raise ValueError('Frozen decoder/stick geometry differs')
    checkpoint=Path(source_cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source_cfg['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    bank_digest=hashlib.sha256(cli.initial_bank.read_bytes()).hexdigest()
    bank_provenance=json.loads((cli.initial_bank.parent/'bank_provenance.json').read_text())
    if bank_provenance['legacy_states_used'] or bank_provenance['source_runtime']!='matched_decoder_speed_contact_reset_v2':
        raise ValueError('Starting bank is not the verified matching-geometry corrected bank')
    geometry=source_cfg['network']['tip_geometry']
    bank=make_start_bank(geometry)
    bank.load_state_dict(torch.load(cli.initial_bank,map_location='cpu',weights_only=False))
    if len(bank.states)!=13:raise ValueError('Starting bank lacks allowed hand/drum coverage')
    eligibility=MinimumTimeChart.load(source_cfg['eligibility_source'])
    sampler=StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility)
    scorer=HandPlanRanker(HandPlanRankerConfig(tip_geometry=geometry)).eval()
    with torch.no_grad():scorer.network[-1].weight.zero_();scorer.network[-1].bias.zero_()
    rng=np.random.default_rng(cli.seed);split_rng=np.random.default_rng(cli.seed+1)
    batches=cases=plan_rollouts=snapshot_rows=0;pending=None
    splits={name:[] for name in ('train','validation','test')}
    config=dict(algorithm=f'{cli.scoring_version.upper()} all-legal-plan plus factual pre-first-hit collection',
        purpose=f'new_{cli.scoring_version}_training_dataset',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()},
        target_datapoints=cli.target_datapoints,datapoint_definition='one plan + one drum-goal window + one physical start',
        row_count='one initial row per legal plan plus each valid pre-first-hit factual snapshot',
        source_dataset=str(cli.source_dataset.resolve()),source_outcomes_reused=False,
        initial_bank=str(cli.initial_bank.resolve()),initial_bank_sha256=bank_digest,
        initial_bank_provenance=bank_provenance,
        checkpoint=str(checkpoint),checkpoint_sha256=source_cfg['checkpoint_sha256'],
        eligibility_source=source_cfg['eligibility_source'],network=asdict(scorer.config),
        simulation_geometry_version=GEOMETRY_VERSION,drum_geometry_version=DRUM_GEOMETRY_VERSION,
        first_hit_policy='iid_random',rhythm_sampling=sampler.summary,
        physics_hz=250,control_hz=50,split_probabilities=dict(train=.8,validation=.1,test=.1),
        split_unit='whole rhythm/start case; all legal plans and snapshots inherit one split',
        start_mixture=dict(post_hit=.8,random=.15,ready=.05),
        all_legal_plans_once=True,fresh_scene_per_batch=True,single_physics_wave=True,
        rollout_runtime_version=requested_runtime,
        factual_label='remaining reward after prefix torque/joint/collision costs, plus all future hit outcomes',
        snapshot_rule='25/50/75 percent of first delay on 20ms grid, at least100ms before first hit, no beat credit yet',
        decoder_frozen=True,held_out_used_for_training=False)
    if cli.resume:
        old=json.loads((cli.out/'config.json').read_text())
        for key in ('target_datapoints','source_dataset','initial_bank_sha256','checkpoint_sha256',
                    'drum_geometry_version','rollout_runtime_version'):
            if old[key]!=config[key]:raise ValueError(f'Resume differs in {key}')
        state=torch.load(cli.out/'collector_state.pt',map_location='cpu',weights_only=False)
        batches,cases,plan_rollouts,snapshot_rows,pending=(state[key]
            for key in ('batches','cases','plan_rollouts','snapshot_rows','pending'))
        splits=state['splits'];rng.bit_generator.state=state['rng']
        split_rng.bit_generator.state=state['split_rng'];bank.load_state_dict(state['bank'])
        config['rollout_runtime']=old.get('rollout_runtime')
        config['drum_layout_sha256']=old.get('drum_layout_sha256')
        config['drum_usd_sha256']=old.get('drum_usd_sha256')
    write_json(cli.out/'config.json',config)
    seed_file=cli.out/'scene_seed.json';write_json(seed_file,sampler.seed_payload())
    source=rollout_source_config(source_cfg)
    args=runtime_args(['--out',str(cli.out),'--envs','1','--device',cli.devices[0],
        '--rollout-devices',*cli.devices,'--physics-hz','250','--decode-chunk','512',
        '--collision-chunk','1024','--no-plan-diagnostics','--clone-kit-from-source',
        '--clear-reset-contacts','--freeze-hit-scoring','--match-decoder-speed'])
    for key in ('torque_weight','joint_weight','collision_weight','collision_stride',
                'near_miss_weight','near_miss_scale_m','near_miss_window_s'):
        setattr(args,key,source['args'][key])
    args.groups=args.horizon=4;args.seed=cli.seed
    def save():
        save_torch(cli.out/'collector_state.pt',dict(batches=batches,cases=cases,
            plan_rollouts=plan_rollouts,snapshot_rows=snapshot_rows,pending=pending,
            rng=rng.bit_generator.state,split_rng=split_rng.bit_generator.state,
            bank=bank.state_dict(),splits=splits))
    save();write_json(cli.out/'run.json',dict(status='starting',total_datapoints=plan_rollouts+snapshot_rows))
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(cli.out/'tb'),flush_secs=5)
    writer.add_text('setup/config',json.dumps(config),0);writer.flush()
    if os.environ.get('TB_AUTO_REGISTER','1')!='0':
        registry=ROOT/'runs/tensorboard_active';registry.mkdir(exist_ok=True);link=registry/cli.out.name
        if not link.exists():link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs',timeout=10)):break
            time.sleep(5)
        else:raise RuntimeError('TensorBoard did not list dataset collection')
    evaluator=None
    try:
        while plan_rollouts+snapshot_rows<cli.target_datapoints:
            remaining=cli.target_datapoints-plan_rollouts-snapshot_rows
            if remaining<16:break  # Never split a context's all-plan label bank.
            started=time.monotonic()
            if pending is None:
                mean_rows=(plan_rollouts+snapshot_rows)/cases if cases else 31.
                count=min(cli.contexts,max(1,int(np.ceil(remaining/mean_rows))),remaining//16)
                panel=sampler.sample(count,rng,minimum_plan_choices=1)
                states,start_metrics=bank.sample(rng,count)
                split_codes=split_rng.choice(3,size=count,p=(.8,.1,.1)).astype(np.int8)
                pending=build_ranker_batch(scorer,panel,eligibility,16,states)
                pending.update(split_codes=split_codes,start_metrics=start_metrics,
                    case_offset=cases,tip_geometry=geometry)
                save()
            args.envs=len(pending['episodes'])
            evaluator=DistributedIsaacEvaluator(args,checkpoint,seed_file)
            ready=[json.loads((folder/'ready.json').read_text()) for folder,_ in evaluator.workers]
            for row in ready:
                if (row['rollout_runtime_version']!=requested_runtime
                        or row['drum_geometry_version']!=DRUM_GEOMETRY_VERSION
                        or row['tip_geometry']!=geometry):
                    raise ValueError('Worker differs from requested geometry/scoring')
            runtime={key:ready[0][key] for key in ('rollout_runtime_version','fixed_hit_scoring',
                'decoder_amp','impact_speed_target')}
            runtime['scene_reset_protocol']='fresh_process_scene_single_wave_v1'
            if config.get('rollout_runtime') not in (None,runtime):raise ValueError('Collection runtime changed')
            for key in ('drum_layout_sha256','drum_usd_sha256'):
                if config.get(key) not in (None,ready[0][key]):raise ValueError(f'Collection changed {key}')
                config[key]=ready[0][key]
            config['rollout_runtime']=runtime;write_json(cli.out/'config.json',config)
            capacity=int(np.ceil(args.envs/len(ready)))
            if max(args.envs*(i+1)//len(ready)-args.envs*i//len(ready)
                   for i in range(len(ready)))>capacity:raise ValueError('Batch needs more than one wave')
            def progress(done,total,tick,ticks):
                write_json(cli.out/'progress.json',dict(batch=batches+1,
                    total_datapoints=plan_rollouts+snapshot_rows,target_datapoints=cli.target_datapoints,
                    completed_contexts=cases,episodes=total,done=done,tick=tick,ticks=ticks))
            reports=evaluator.evaluate(pending['episodes'],progress,
                initial_states=pending['episode_states'],collect_post_hits=True,
                collect_prefirst_snapshots=True)
            evaluator.close();evaluator=None
            if not np.array_equal(pending['observed'],pending['legal']):
                raise ValueError('Missing legal-plan labels')
            outcomes=unpack_ranker_reports(pending,reports)
            factual=factual_rows(pending,reports,cases,pending['split_codes'])
            base=int(pending['observed'].sum())
            allowed=cli.target_datapoints-(plan_rollouts+snapshot_rows)-base
            if allowed<0:raise ValueError('A full context exceeds remaining datapoint budget')
            if len(factual['actions'])>allowed:
                chosen=np.sort(rng.choice(len(factual['actions']),size=allowed,replace=False))
                factual={key:value[chosen] for key,value in factual.items()}
            rows_now=base+len(factual['actions'])
            save_torch(cli.out/'batches'/f'batch_{batches+1:05d}.pt',
                dict(batch=pending,outcomes=outcomes,factual=factual))
            write_json(cli.out/'batches'/f'episodes_{batches+1:05d}.json',reports)
            for name,code in (('train',0),('validation',1),('test',2)):
                ids=np.flatnonzero(pending['split_codes']==code)+cases
                splits[name].extend(ids.tolist())
            train_states=[state for (case,_action),report in zip(pending['labels'],reports)
                if pending['split_codes'][int(case)]==0
                for state in report.get('post_hit_states',[])]
            bank_stats=bank.add(train_states,rng)
            cases+=len(pending['panel']);plan_rollouts+=base;snapshot_rows+=len(factual['actions'])
            batches+=1;pending=None;save()
            torch.save({name:np.asarray(ids,np.int64) for name,ids in splits.items()},cli.out/'splits.pt')
            metrics=dict(batch=batches,contexts=cases,plan_rollouts=plan_rollouts,
                factual_snapshot_rows=snapshot_rows,total_datapoints=plan_rollouts+snapshot_rows,
                target_datapoints=cli.target_datapoints,split_cases={k:len(v) for k,v in splits.items()},
                current_batch_plan_rollouts=base,current_batch_snapshots=len(factual['actions']),
                mean_snapshots_per_plan=len(factual['actions'])/base,bank=bank_stats,
                seconds=time.monotonic()-started,fresh_scene=True,single_physics_wave=True)
            write_json(cli.out/'batches'/f'metrics_{batches:05d}.json',metrics)
            write_json(cli.out/'run.json',dict(status='running',**{k:metrics[k] for k in
                ('batch','contexts','plan_rollouts','factual_snapshot_rows','total_datapoints',
                 'target_datapoints','split_cases')}))
            for key in ('contexts','plan_rollouts','factual_snapshot_rows','total_datapoints'):
                writer.add_scalar('collection/'+key,metrics[key],batches)
            writer.flush();print(f'[{cli.scoring_version}-prefirst] '+json.dumps(metrics),flush=True)
        save_torch(cli.out/'post_hit_bank.pt',bank.state_dict())
        write_json(cli.out/'run.json',dict(status='complete',batches=batches,contexts=cases,
            plan_rollouts=plan_rollouts,factual_snapshot_rows=snapshot_rows,
            total_datapoints=plan_rollouts+snapshot_rows,target_datapoints=cli.target_datapoints,
            split_cases={k:len(v) for k,v in splits.items()},
            held_out_used_for_training=False,rollout_runtime=config['rollout_runtime']))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',batches=batches,contexts=cases,
            total_datapoints=plan_rollouts+snapshot_rows,error=repr(exc)))
        raise
    finally:
        if evaluator is not None:evaluator.close()
        writer.close()


if __name__=='__main__':main()
