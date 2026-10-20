#!/usr/bin/env python3
"""Fit the hand-plan ranker repeatedly on a fixed, whole-case supervised dataset."""
import argparse
from dataclasses import asdict,replace
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
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,RankerReplay,ranking_loss
from isaac_rl.hand_plan_dataset import (StochasticHandRhythmSampler,load_dataset_batches,
    dataset_metrics,require_matching_rollout_runtime)
from isaac_rl.hand_plan_collocation_priors import sample_collocations,collocation_prior_loss
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.hand_plan_diagnostics import prediction_diagnostics,weight_diagnostics
from isaac_rl.hand_plan_gradient_regularizer import InputGradientConfig,input_gradient_l1,tip_jacobians
from scripts.train_neural_hand_plans import learning_rate,save_torch
from scripts.train_chart_cma import write_json


@torch.inference_mode()
def evaluate(model,data,ids):
    model.eval();scores=[];total_loss=0.;total_winner=0.;total_pair=0.;device=next(model.parameters()).device
    for offset in range(0,len(ids),1024):
        selected=ids[offset:offset+1024]
        output=model(torch.as_tensor(data['observations'][selected],device=device))
        loss,parts=ranking_loss(output,torch.as_tensor(data['rewards'][selected],device=device),torch.as_tensor(data['observed'][selected],device=device))
        total_loss+=float(loss)*len(selected);total_winner+=float(parts['winner_loss'])*len(selected)
        total_pair+=float(parts['pair_loss'])*len(selected);scores.append(output.cpu().numpy())
    scores=np.concatenate(scores);result=dataset_metrics(scores,data,ids)
    result.update(prediction_diagnostics(scores,data['observed'][ids],data['rewards'][ids]))
    result.update(loss=total_loss/len(ids),winner_loss=total_winner/len(ids),pair_loss=total_pair/len(ids))
    return result


def validate_reference(config,reference):
    """Capacity comparisons change scorer width while preserving the objective."""
    if Path(reference['dataset']).resolve()!=Path(config['dataset']).resolve() or reference['split_counts']!=config['split_counts']:
        raise ValueError('Reference must use the exact same dataset/splits')
    for key in ('steps','batch','lr','min_lr','warmup','seed','weight_decay','dropout'):
        if reference['args'][key]!=config['args'][key]:raise ValueError(f'Comparison must match {key}')
    same_architecture=reference['network'].get('architecture','flat')==config['network']['architecture']
    different_width=reference['network']['hidden']!=config['network']['hidden']
    if same_architecture and different_width:
        defaults=dict(gradient_l1=0.,gradient_batch=64,gradient_timing_scale_s=.02,
            gradient_position_scale_deg=1.,gradient_velocity_scale_rad_s=.1,gradient_loudness_scale=.02)
        for key,default in defaults.items():
            if reference['args'].get(key,default)!=config['args'][key]:
                raise ValueError(f'Capacity comparison must match {key}')
        return 'scorer_width'
    return 'architecture' if not same_architecture else 'regularization'


def load_reference_evaluation(path,config):
    """Only evaluation-label provenance must match; old replay/optimizer never enter fitting."""
    reference=json.loads(Path(path).read_text())
    if reference['format']!='frozen_hand_plan_reference_v1' or Path(reference['dataset']).resolve()!=Path(config['dataset']).resolve():
        raise ValueError('Frozen reference uses a different evaluation dataset')
    require_matching_rollout_runtime(config,reference)
    if not reference['heldout_inputs_match'] or reference['held_out_used_for_training'] or reference['weights_selected_by_evaluation']:
        raise ValueError('Reference must be read-only and use matched held-out inputs')
    for name in ('validation','test'):
        if reference['metrics'][name]['cases']!=config['split_counts'][name]:raise ValueError('Reference split count differs')
    snapshot=Path(reference['checkpoint_snapshot'])
    if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=reference['reference_weights_sha256']:
        raise ValueError('Frozen reference weights changed')
    return reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--stop-after-updates',type=int,help='Stop at this update while preserving the --steps cosine horizon for a matched comparison')
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--min-lr',type=float,default=3e-5)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--dropout',type=float,default=0.,help='Dropout after each hidden GELU; disabled during evaluation')
    p.add_argument('--architecture',choices=['flat','shared_candidate'],default='flat')
    p.add_argument('--hidden',type=int,help='Scorer width; defaults to dataset network width')
    p.add_argument('--group-width',type=int,default=64,help='Shared hit-group encoder output width')
    p.add_argument('--state-width',type=int,default=64,help='Shared initial-state encoder output width')
    p.add_argument('--scorer-depth',type=int,choices=(2,3),default=2,help='Number of scorer hidden layers')
    p.add_argument('--device',default='cpu',help='Torch device, e.g. cpu or cuda:2')
    p.add_argument('--reference-run',type=Path,help='Read-only matching-dataset traces for an architecture comparison')
    p.add_argument('--reference-evaluation',type=Path,help='Frozen old-policy metrics re-evaluated on this dataset; never used for fitting')
    p.add_argument('--gradient-l1',type=float,default=0.,help='Coefficient of physically scaled input-gradient L1 on relative legal scores')
    p.add_argument('--gradient-batch',type=int,default=64,help='Training cases per update for the stochastic Jacobian penalty')
    p.add_argument('--gradient-timing-scale-s',type=float,default=.02)
    p.add_argument('--gradient-position-scale-deg',type=float,default=1.)
    p.add_argument('--gradient-velocity-scale-rad-s',type=float,default=.1)
    p.add_argument('--gradient-loudness-scale',type=float,default=.02)
    p.add_argument('--diagnose-input-gradients',action='store_true',help='Read-only input-gradient probes even when the penalty is zero')
    p.add_argument('--prior-weight',type=float,default=0.,help='Synthetic soft hand-assignment regularizer')
    p.add_argument('--collocation-cases',type=int,default=32)
    p.add_argument('--motif-mode',choices=('both','consecutive','simultaneous'),default='both')
    p.add_argument('--motif-probability',type=float,default=.5)
    p.add_argument('--motif-margin',type=float,default=.5,help='Canonical snare/hi-hat margin in ranking-score units')
    p.add_argument('--switch-margin',type=float,default=.05,help='Per hand drum-switch margin in ranking-score units')
    p.add_argument('--prior-scale',type=float,default=1.)
    p.add_argument('--warmup',type=int,default=100);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261010);p.add_argument('--resume',type=Path)
    cli=p.parse_args();torch.set_num_threads(2);torch.manual_seed(cli.seed)
    if min(cli.steps,cli.batch,cli.eval_every)<=0:raise ValueError('Positive fitting budgets required')
    if cli.stop_after_updates is not None and not 0<cli.stop_after_updates<=cli.steps:
        raise ValueError('Stop update must lie within the schedule horizon')
    if cli.reference_run and cli.reference_evaluation:raise ValueError('Choose one reference comparison mode')
    if not np.isfinite(cli.weight_decay) or cli.weight_decay<0:raise ValueError('Nonnegative finite weight decay required')
    if not 0.<=cli.dropout<1.:raise ValueError('Dropout must be in [0, 1)')
    if not np.isfinite(cli.gradient_l1) or cli.gradient_l1<0 or cli.gradient_batch<1:
        raise ValueError('Nonnegative finite gradient coefficient and positive gradient batch required')
    if (min(cli.prior_weight,cli.motif_margin,cli.switch_margin)<0 or
            cli.collocation_cases<1 or not 0<=cli.motif_probability<=1 or cli.prior_scale<=0):
        raise ValueError('Invalid collocation prior configuration')
    gradient_config=InputGradientConfig(cli.gradient_timing_scale_s,np.deg2rad(cli.gradient_position_scale_deg),
        cli.gradient_velocity_scale_rad_s,cli.gradient_loudness_scale)
    if cli.out.exists() and not cli.resume and any(
        item.name not in ('input_tip_jacobians.pt','train.log') for item in cli.out.iterdir()
    ):raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((cli.dataset/'config.json').read_text());status=json.loads((cli.dataset/'run.json').read_text())
    if status['status']!='complete':raise ValueError('Finish data collection before fixed-dataset training')
    data=load_dataset_batches(cli.dataset);splits=torch.load(cli.dataset/'splits.pt',weights_only=False)
    if len(data['observations'])!=status['cases']:raise ValueError('Dataset manifest/case count differs from stored panels')
    device=torch.device(cli.device)
    model=HandPlanRanker(replace(HandPlanRankerConfig(**cfg['network']),dropout=cli.dropout,
        architecture=cli.architecture,hidden=cli.hidden or cfg['network']['hidden'],
        group_width=cli.group_width,state_width=cli.state_width,scorer_depth=cli.scorer_depth)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cli.lr,weight_decay=cli.weight_decay)
    rng=np.random.default_rng(cli.seed);step=0
    prior_rng=np.random.default_rng(cli.seed+98123)
    gradient_rng=torch.Generator(device=device).manual_seed(cli.seed+153241)
    config=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()},
        algorithm='Offline supervised16-action hand-plan ranking',network=asdict(model.config),
        checkpoint=cfg['checkpoint'],checkpoint_sha256=cfg['checkpoint_sha256'],
        simulation_geometry_version=cfg['simulation_geometry_version'],first_hit_policy='iid_random',
        eligibility_source=cfg['eligibility_source'],dataset=str(cli.dataset.resolve()),
        split_counts={key:len(ids) for key,ids in splits.items()},
        source_run=cfg['source_run'],held_out_used_for_training=False,checkpoint_selection='Latest fixed-budget weights',
        loss='2*near-best-set cross entropy + weighted pairwise ranking',
        phase2_decoder_training=False,decoder_frozen=True,neural_parameters=sum(p.numel() for p in model.parameters()),
        rollout_runtime=cfg.get('rollout_runtime',{'rollout_runtime_version':'legacy_contact_reset_v0'}),
        diagnostics=dict(prediction_entropy='Temperature-one softmax over legal labeled plans; normalized by log(bank size), excluding singleton banks',
            weight_entropy='Normalized Shannon entropy of absolute matrix-weight magnitudes (scale invariant), plus the paper raw statistic -sum |w| log |w|',
            grokking='Look for delayed held-out improvement after training fit plateaus; entropy alone is not a detection rule'))
    config['input_gradient_regularizer']=dict(version='physical_relative_score_jacobian_l1_v1',
        coefficient=cli.gradient_l1,batch=min(cli.batch,cli.gradient_batch),physical_scales=asdict(gradient_config),
        estimator='Uniform legal-plan sampling; multiply sampled centered-score derivative L1 by legal count, then average cases',
        dropout_enabled=False,discrete_inputs_regularized=False,joint_tip_chain_rule=True,
        evaluation_probe_cases_per_split=256)
    diagnose_gradients=cli.gradient_l1>0 or cli.diagnose_input_gradients
    config['input_gradient_regularizer']['diagnostics_enabled']=diagnose_gradients
    if cli.gradient_l1:
        config['loss']+=' + gradient_l1*physically scaled relative-score input Jacobian L1'
    config['collocation_prior']=dict(enabled=cli.prior_weight>0,coefficient=cli.prior_weight,
        cases_per_update=cli.collocation_cases,motif_mode=cli.motif_mode,
        motif_probability=cli.motif_probability,motif_margin_score=cli.motif_margin,
        switch_margin_score=cli.switch_margin,scale=cli.prior_scale,
        source='fresh IID rhythms paired with independent train-only physical starts',
        loss='one-sided squared violation of heuristic plan-score gaps on legal pairs',
        dropout_enabled=False,independent_sampler_rng=True)
    if cli.prior_weight:config['loss']+=' + prior_weight*collocation hand-assignment margin'
    if cli.reference_run:
        reference=json.loads((cli.reference_run/'config.json').read_text())
        config['comparison_axis']=validate_reference(config,reference)
    frozen_reference=None
    if cli.reference_evaluation:
        frozen_reference=load_reference_evaluation(cli.reference_evaluation,config)
        config['comparison_axis']='more_data_and_corrected_labels_vs_frozen_previous_policy'
        config['frozen_reference_evaluation']=dict(path=str(cli.reference_evaluation.resolve()),
            model_run=frozen_reference['model_run'],model_gradient_steps=frozen_reference['model_gradient_steps'],
            reference_weights_sha256=frozen_reference['reference_weights_sha256'],
            evaluation_sha256=hashlib.sha256(cli.reference_evaluation.read_bytes()).hexdigest(),
            model_training_runtime=frozen_reference['model_training_runtime'],
            evaluation_runtime=frozen_reference['rollout_runtime'],used_for_training=False)
    if cli.resume:
        old=torch.load(cli.resume,map_location='cpu',weights_only=False)
        if old['format']!='offline_hand_plan_ranker_v1' or old['dataset']!=str(cli.dataset.resolve()):
            raise ValueError('Resume dataset/checkpoint differs')
        for key in ('steps','batch','lr','min_lr','warmup','seed'):
            if old['args'][key]!=config['args'][key]:raise ValueError(f'Resume must preserve{key}')
        if old['args'].get('weight_decay',1e-4)!=cli.weight_decay:raise ValueError('Resume must preserve weight_decay')
        if old['network'].get('dropout',0.)!=cli.dropout:raise ValueError('Resume must preserve dropout')
        if old['network'].get('architecture','flat')!=cli.architecture or old['network']['hidden']!=model.config.hidden:
            raise ValueError('Resume must preserve architecture and hidden width')
        for key,default in (('group_width',64),('state_width',64),('scorer_depth',2)):
            if old['network'].get(key,default)!=getattr(model.config,key):raise ValueError(f'Resume must preserve {key}')
        if old['args'].get('device','cpu')!=cli.device:raise ValueError('Resume must preserve device and its RNG stream')
        if old['args'].get('diagnose_input_gradients',False)!=cli.diagnose_input_gradients:
            raise ValueError('Resume must preserve diagnose_input_gradients')
        if old['args'].get('reference_evaluation')!=config['args']['reference_evaluation']:
            raise ValueError('Resume must preserve frozen-reference evaluation')
        expected=config.get('frozen_reference_evaluation',{}).get('evaluation_sha256')
        if old.get('frozen_reference_evaluation_sha256')!=expected:raise ValueError('Resume reference evaluation changed')
        gradient_defaults=dict(gradient_l1=0.,gradient_batch=64,gradient_timing_scale_s=.02,
            gradient_position_scale_deg=1.,gradient_velocity_scale_rad_s=.1,gradient_loudness_scale=.02)
        for key,default in gradient_defaults.items():
            if old['args'].get(key,default)!=getattr(cli,key):raise ValueError(f'Resume must preserve {key}')
        if old.get('collocation_prior') is not None and old['collocation_prior']!=config['collocation_prior']:
            raise ValueError('Resume must preserve collocation prior')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        step=old['gradient_steps'];rng.bit_generator.state=old['rng'];torch.set_rng_state(old['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(old['cuda_rng'])
        if cli.gradient_l1:gradient_rng.set_state(old['gradient_rng'])
        if old.get('prior_rng') is not None:prior_rng.bit_generator.state=old['prior_rng']
    write_json(cli.out/'config.json',config)
    jacobians=None
    if diagnose_gradients:
        write_json(cli.out/'run.json',dict(status='preparing_input_gradient_jacobians',gradient_steps=step))
        fingerprint=hashlib.sha256(np.ascontiguousarray(data['observations']).tobytes()).hexdigest()
        cache=cli.out/'input_tip_jacobians.pt'
        if cache.exists():
            cached=torch.load(cache,weights_only=False)
            if cached['observations_sha256']!=fingerprint or cached['tip_geometry']!=model.config.tip_geometry:
                raise ValueError('Input FK Jacobian cache differs from dataset/geometry')
            jacobians=cached['jacobians']
        else:
            print('[input-gradient] computing FK Jacobians for joint/tip chain rule',flush=True)
            jacobians=tip_jacobians(data['observations'],model.config.tip_geometry,device=device)
            save_torch(cache,dict(observations_sha256=fingerprint,tip_geometry=model.config.tip_geometry,jacobians=jacobians))
        if jacobians.shape!=(len(data['observations']),6,17):raise ValueError('Invalid input FK Jacobian cache shape')
        jacobians=jacobians.to(device)
    replay=RankerReplay(4096)
    seed_ids=np.random.default_rng(88812).choice(splits['train'],min(4096,len(splits['train'])),replace=False)
    replay.add(data['observations'][seed_ids],{'rewards':data['rewards'][seed_ids],
        'hit_rates':data['hit_rates'][seed_ids]},data['observed'][seed_ids])
    bank=torch.load(cli.dataset/'post_hit_bank.pt',weights_only=False)
    if cli.prior_weight:
        kit=DrumKit.from_layout()
        eligibility=MinimumTimeChart.load(cfg['eligibility_source'])
        collocation_sampler=StochasticHandRhythmSampler(kit,4,eligibility)
        training_starts=np.asarray(data['observations'][splits['train'],80:],np.float32)
    def state():
        return dict(format='offline_hand_plan_ranker_v1',args=config['args'],dataset=str(cli.dataset.resolve()),
            network=asdict(model.config),model=model.state_dict(),optimizer=optimizer.state_dict(),
            gradient_steps=step,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            gradient_rng=gradient_rng.get_state() if cli.gradient_l1 else None,
            prior_rng=prior_rng.bit_generator.state,collocation_prior=config['collocation_prior'],
            input_gradient_regularizer=config['input_gradient_regularizer'],
            frozen_reference_evaluation_sha256=config.get('frozen_reference_evaluation',{}).get('evaluation_sha256'),
            replay=replay,replay_case_ids=seed_ids,post_hit_bank=bank,
            rollout_runtime=config['rollout_runtime'],
            checkpoint_sha256=cfg['checkpoint_sha256'],simulation_geometry_version=cfg['simulation_geometry_version'])
    if not (cli.out/'initial.pt').exists():
        if cli.resume:(cli.out/'initial.pt').write_bytes((cli.resume.parent/'initial.pt').read_bytes())
        else:save_torch(cli.out/'initial.pt',state())
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(cli.out/'tb'),flush_secs=5);writer.add_text('setup/config',json.dumps(config),0);writer.flush()
    if os.environ.get('TB_AUTO_REGISTER','1')!='0':
        registry=ROOT/'runs/tensorboard_active';registry.mkdir(exist_ok=True);link=registry/cli.out.name
        if not link.exists():link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs',timeout=10)):break
            time.sleep(5)
        else:raise RuntimeError('TensorBoard did not list offline ranker')
    def assess():
        metrics={name:evaluate(model,data,ids) for name,ids in splits.items()}
        if diagnose_gradients:
            for split_index,(name,ids) in enumerate(splits.items()):
                probe=ids[:256]
                probe_rng=torch.Generator(device=device).manual_seed(cli.seed+88481+split_index)
                _,sensitivity=input_gradient_l1(model,torch.as_tensor(data['observations'][probe],device=device),
                    torch.as_tensor(data['observed'][probe],device=device),jacobians[torch.as_tensor(probe,device=device)],
                    generator=probe_rng,config=gradient_config,create_graph=False)
                metrics[name].update({key:float(value) for key,value in sensitivity.items()})
                metrics[name]['regularized_objective']=metrics[name]['loss']+cli.gradient_l1*metrics[name]['gradient_l1']
        weights=weight_diagnostics(model)
        record=dict(gradient_steps=step,metrics=metrics,weights=weights)
        write_json(cli.out/f'evaluation_{step:06d}.json',record)
        with (cli.out/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        for name,stats in metrics.items():
            for key,value in stats.items():writer.add_scalar(f'{name}/{key}',value,step)
        for key,value in weights.items():writer.add_scalar(f'weights/{key}',value,step)
        write_json(cli.out/'diagnostics.json',dict(**record,
            interpretation='Read-only traces: delayed held-out improvement after train fit stabilizes is the relevant evidence; entropy changes alone do not establish grokking.'))
        if frozen_reference:
            comparison=dict(gradient_steps=step,reference=frozen_reference['model_run'],
                reference_model_gradient_steps=frozen_reference['model_gradient_steps'],
                evaluation_runtime=config['rollout_runtime'],matched_heldout_inputs=True,
                weights_selected_by_comparison=False,held_out_used_for_training=False,
                reference_metrics=frozen_reference['metrics'],metrics={name:metrics[name] for name in ('validation','test')})
            comparison['differences']={name:dict(
                hit_rate_difference_pp=100*(metrics[name]['hit_rate']-frozen_reference['metrics'][name]['hit_rate']),
                reward_regret_difference=metrics[name]['mean_reward_regret']-frozen_reference['metrics'][name]['mean_reward_regret'],
                pair_accuracy_difference_pp=100*(metrics[name]['pair_accuracy']-frozen_reference['metrics'][name]['pair_accuracy']))
                for name in ('validation','test')}
            write_json(cli.out/'comparison.json',comparison)
            for name,values in comparison['differences'].items():
                for key,value in values.items():writer.add_scalar(f'comparison/{name}_{key}',value,step)
        writer.flush();print('[offline-evaluation] '+json.dumps(record),flush=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        history=[json.loads(s) for s in (cli.out/'evaluations.jsonl').read_text().splitlines()]
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for name in ('train','validation','test'):
            x=[r['gradient_steps'] for r in history]
            axes[0].plot(x,[100*r['metrics'][name]['hit_rate'] for r in history],label=name)
            axes[1].plot(x,[100*r['metrics'][name]['pair_accuracy'] for r in history],label=name)
        axes[0].set_ylabel('Selected-plan hit rate (%)');axes[1].set_ylabel('Clear-pair ordering accuracy (%)')
        if frozen_reference:
            for axis,key in zip(axes,('hit_rate','pair_accuracy')):
                axis.axhline(100*frozen_reference['metrics']['test'][key],color='tab:red',linestyle='--',label='Frozen previous model: same fresh test labels')
        if cli.reference_run:
            reference_history=[json.loads(s) for s in (cli.reference_run/'evaluations.jsonl').read_text().split('\n')[:-1]]
            reference_history=[r for r in reference_history if r['gradient_steps']<=step]
            for axis,key in zip(axes,('hit_rate','pair_accuracy')):
                axis.plot([r['gradient_steps'] for r in reference_history],
                    [100*r['metrics']['test'][key] for r in reference_history],color='tab:green',linestyle='--',label='Reference: test')
            matched=next((r for r in reference_history if r['gradient_steps']==step),None)
            if matched:
                comparison=dict(gradient_steps=step,reference=str(cli.reference_run.resolve()),
                    weights_selected_by_comparison=False,metrics=metrics,reference_metrics=matched['metrics'],
                    test_hit_rate_difference_pp=100*(metrics['test']['hit_rate']-matched['metrics']['test']['hit_rate']),
                    test_pair_accuracy_difference_pp=100*(metrics['test']['pair_accuracy']-matched['metrics']['test']['pair_accuracy']))
                write_json(cli.out/'comparison.json',comparison)
                writer.add_scalar('comparison/test_hit_rate_difference_pp',comparison['test_hit_rate_difference_pp'],step)
                writer.add_scalar('comparison/test_pair_accuracy_difference_pp',comparison['test_pair_accuracy_difference_pp'],step)
        for ax in axes:ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2);ax.legend()
        fig.suptitle(f'{cli.architecture}: fixed IID dataset, whole-case splits, latest weights')
        temp=cli.out/'learning.tmp.png';fig.savefig(temp,dpi=160);temp.replace(cli.out/'learning.png');plt.close(fig)
        diagnostic_history=[r for r in history if 'weights' in r]
        fig,axes=plt.subplots(3,2,figsize=(11,11),constrained_layout=True)
        for name in ('train','validation','test'):
            x=[r['gradient_steps'] for r in diagnostic_history]
            axes[0,0].plot(x,[r['metrics'][name]['prediction_entropy_normalized'] for r in diagnostic_history],label=name)
            axes[0,1].plot(x,[r['metrics'][name]['loss'] for r in diagnostic_history],label=name)
            axes[1,0].plot(x,[r['metrics'][name]['mean_reward_regret'] for r in diagnostic_history],label=name)
            axes[2,0].plot(x,[100*r['metrics'][name]['hit_rate'] for r in diagnostic_history],label=name)
        axes[1,1].plot(x,[r['weights']['parameter_l2'] for r in diagnostic_history],label='All-parameter L2')
        axes[2,1].plot(x,[r['weights']['absolute_weight_entropy_normalized'] for r in diagnostic_history],label='Matrix magnitude entropy')
        raw_history=[r for r in diagnostic_history if 'absolute_weight_statistic_raw' in r['weights']]
        if raw_history:
            raw_axis=axes[2,1].twinx()
            raw_axis.plot([r['gradient_steps'] for r in raw_history],
                [r['weights']['absolute_weight_statistic_raw'] for r in raw_history],color='tab:orange',label='Raw statistic (paper)')
            raw_axis.set_ylabel('Raw -sum |w| log |w|',color='tab:orange')
            raw_axis.tick_params(axis='y',labelcolor='tab:orange')
        axes[0,0].set_ylabel('Legal-plan entropy / log(plan count)');axes[0,0].set_ylim(0,1.05)
        axes[0,1].set_ylabel('Supervised ranking loss');axes[1,0].set_ylabel('Chosen-plan reward regret')
        if frozen_reference:
            axes[1,0].axhline(frozen_reference['metrics']['test']['mean_reward_regret'],color='tab:red',linestyle='--',label='Frozen previous model: test')
            axes[2,0].axhline(100*frozen_reference['metrics']['test']['hit_rate'],color='tab:red',linestyle='--',label='Frozen previous model: test')
        axes[1,1].set_ylabel('Parameter L2 norm')
        axes[2,0].set_ylabel('Selected-plan hit rate (%)');axes[2,1].set_ylabel('Normalized absolute-weight entropy')
        for ax in axes.flat:ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2);ax.legend()
        fig.suptitle(f'{cli.architecture}: weight decay {cli.weight_decay:g}, dropout {cli.dropout:g}; latest weights')
        temp=cli.out/'entropy.tmp.png';fig.savefig(temp,dpi=160);temp.replace(cli.out/'entropy.png');plt.close(fig)
        if diagnose_gradients:
            fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
            for name in splits:
                axes[0].plot([r['gradient_steps'] for r in history],[r['metrics'][name]['gradient_l1'] for r in history],label=name)
                axes[1].plot([r['gradient_steps'] for r in history],
                    [r['metrics'][name]['fraction_scaled_derivatives_below_1e_3'] for r in history],label=name)
            axes[0].set_ylabel('Relative-score input Jacobian L1 (fixed probe)')
            axes[1].set_ylabel('Fraction of scaled derivatives below 0.001')
            axes[1].set_ylim(0,1)
            for ax in axes:ax.set_xlabel('Optimizer steps');ax.grid(alpha=.2);ax.legend()
            fig.suptitle(f'Input-gradient regularization: coefficient {cli.gradient_l1:g}; physical scales fixed')
            temp=cli.out/'input_gradients.tmp.png';fig.savefig(temp,dpi=160);temp.replace(cli.out/'input_gradients.png');plt.close(fig)
        return metrics
    try:
        existing_evaluation=cli.out/f'evaluation_{step:06d}.json'
        if cli.resume and cli.resume.resolve()==(cli.out/'last.pt').resolve() and existing_evaluation.exists():
            metrics=json.loads(existing_evaluation.read_text())['metrics']
        else:
            metrics=assess()
        if not (cli.out/'initial_metrics.json').exists():
            if cli.resume:(cli.out/'initial_metrics.json').write_bytes((cli.resume.parent/'initial_metrics.json').read_bytes())
            else:write_json(cli.out/'initial_metrics.json',metrics)
        save_torch(cli.out/'last.pt',state())
        stop_step=cli.stop_after_updates or cli.steps
        while step<stop_step:
            model.train();ids=rng.choice(splits['train'],size=cli.batch,replace=True)
            step+=1;lr=learning_rate(step,cli.steps,cli.lr,cli.min_lr,cli.warmup)
            for group in optimizer.param_groups:group['lr']=lr
            inputs=torch.as_tensor(data['observations'][ids],device=device)
            observed=torch.as_tensor(data['observed'][ids],device=device)
            scores=model(inputs)
            ranking,parts=ranking_loss(scores,torch.as_tensor(data['rewards'][ids],device=device),observed)
            sensitivity={};penalty=ranking.new_zeros(())
            if cli.gradient_l1:
                n=min(len(ids),cli.gradient_batch)
                penalty,sensitivity=input_gradient_l1(model,inputs[:n],observed[:n],
                    jacobians[torch.as_tensor(ids[:n],device=device)],generator=gradient_rng,config=gradient_config)
            prior_loss=ranking.new_zeros(());prior_stats={};sample_stats={}
            if cli.prior_weight:
                synthetic,allowed,heuristic,sample_stats=sample_collocations(
                    collocation_sampler,kit,eligibility,training_starts,prior_rng,
                    cli.collocation_cases,motif_mode=cli.motif_mode,
                    motif_probability=cli.motif_probability,
                    motif_margin=cli.motif_margin,switch_margin=cli.switch_margin)
                model.eval()  # Constrain inference scores without advancing dropout RNG.
                prior_scores=model(torch.as_tensor(synthetic,device=device))
                model.train()
                prior_loss,prior_stats=collocation_prior_loss(prior_scores,
                    torch.as_tensor(heuristic,device=device),
                    torch.as_tensor(allowed,device=device),scale=cli.prior_scale)
            loss=ranking+cli.gradient_l1*penalty+cli.prior_weight*prior_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised dataset loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%100==0:
                writer.add_scalar('fit/loss',float(loss.detach()),step);writer.add_scalar('fit/learning_rate',lr,step)
                writer.add_scalar('fit/ranking_loss',float(ranking.detach()),step)
                writer.add_scalar('fit/input_gradient_weighted_loss',float(cli.gradient_l1*penalty.detach()),step)
                writer.add_scalar('fit/collocation_prior_loss',float(prior_loss.detach()),step)
                writer.add_scalar('fit/collocation_prior_weighted_loss',float((cli.prior_weight*prior_loss).detach()),step)
                writer.add_scalar('fit/collocation_violated_pairs',prior_stats.get('violated_pairs',0),step)
                writer.add_scalar('fit/collocation_motifs',sample_stats.get('motifs',0),step)
                for key,value in sensitivity.items():writer.add_scalar('fit/'+key,float(value),step)
                writer.add_scalar('fit/winner_loss',float(parts['winner_loss']),step);writer.add_scalar('fit/pair_loss',float(parts['pair_loss']),step)
                writer.add_scalar('fit/gradient_norm_before_clip',float(grad_norm),step)
                writer.add_scalar('fit/weight_decay',cli.weight_decay,step)
                write_json(cli.out/'progress.json',dict(step=step,total_steps=cli.steps,loss=float(loss.detach()),
                    ranking_loss=float(ranking.detach()),input_gradient_weighted_loss=float(cli.gradient_l1*penalty.detach()),
                    prior_loss=float(prior_loss.detach()),weighted_prior_loss=float((cli.prior_weight*prior_loss).detach()),
                    collocation_motifs=sample_stats.get('motifs',0),
                    collocation_violated_pairs=prior_stats.get('violated_pairs',0),learning_rate=lr))
            if step%cli.eval_every==0 or step==stop_step:
                metrics=assess();save_torch(cli.out/'last.pt',state())
                write_json(cli.out/'run.json',dict(status='running',gradient_steps=step,metrics=metrics))
        write_json(cli.out/'run.json',dict(status='complete' if step==cli.steps else 'stopped_at_comparison_budget',
            gradient_steps=step,schedule_horizon=cli.steps,metrics=metrics,
            training_epochs_equivalent=step*cli.batch/len(splits['train'])))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',gradient_steps=step,error=repr(exc)));raise
    finally:writer.close()


if __name__=='__main__':main()
