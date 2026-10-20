#!/usr/bin/env python3
"""Fine-tune v4 reward prediction with pair/winner ranking and v2 auxiliary data."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit
from isaac_rl.hand_plan_ranker import ACTION_BANK, HandPlanRankerConfig
from isaac_rl.hand_plan_collocation_priors import (sample_collocations,
    collocation_prior_loss)
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.hand_plan_value import (DualRuntimeRewardModel, evaluate_legacy_runtime,
    evaluate_value_model, load_legacy_plan_cases, load_value_dataset,
    reward_ranking_losses)
from scripts.train_hand_plan_value import plot_learning
from scripts.train_neural_hand_plans import learning_rate, save_torch
from scripts.train_chart_cma import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--rank-cases', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--weight-decay', type=float, default=.1)
    parser.add_argument('--pairwise-weight', type=float, default=.05)
    parser.add_argument('--winner-weight', type=float, default=.01)
    parser.add_argument('--pairwise-temperature', type=float, default=.1)
    parser.add_argument('--winner-temperature', type=float, default=.05)
    parser.add_argument('--v4-fraction', type=float, default=.5)
    parser.add_argument('--prior-weight',type=float,default=0.,
        help='Small synthetic hand-assignment margin regularizer; zero preserves the original fit')
    parser.add_argument('--collocation-cases',type=int,default=32)
    parser.add_argument('--motif-mode',choices=('both','consecutive','simultaneous'),default='both')
    parser.add_argument('--motif-probability',type=float,default=.5)
    parser.add_argument('--motif-margin',type=float,default=.06)
    parser.add_argument('--switch-margin',type=float,default=.006)
    parser.add_argument('--prior-scale',type=float,default=.05)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--resume', action='store_true')
    cli = parser.parse_args()
    if (min(cli.steps,cli.batch,cli.rank_cases,cli.eval_every,cli.warmup,
            cli.lr,cli.min_lr,cli.pairwise_temperature,cli.winner_temperature) <= 0 or
            min(cli.pairwise_weight,cli.winner_weight,cli.weight_decay) < 0 or
            not 0 < cli.v4_fraction < 1):
        raise ValueError('Invalid mixed-runtime fitting setting')
    if (min(cli.prior_weight,cli.motif_margin,cli.switch_margin)<0 or
            cli.collocation_cases<1 or not 0<=cli.motif_probability<=1 or cli.prior_scale<=0):
        raise ValueError('Invalid collocation prior setting')
    cli.manifest = cli.manifest.resolve(); cli.init_checkpoint = cli.init_checkpoint.resolve()
    cli.out = cli.out.resolve()
    if cli.out.exists() and not cli.resume and any(cli.out.iterdir()):
        raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(cli.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cli.seed)
    manifest = json.loads(cli.manifest.read_text())
    if manifest['format'] != 'mixed_runtime_hand_plan_value_v1' or manifest['inference_runtime'] != 'v4':
        raise ValueError('Wrong mixed label manifest')
    sources = {row['runtime']:row for row in manifest['sources']}
    if set(sources) != {'v4','v2'}:raise ValueError('Both runtime sources required')
    for source in sources.values():
        actual = hashlib.sha256((Path(source['path'])/'config.json').read_bytes()).hexdigest()
        if actual != source['config_sha256']:raise ValueError('Source config changed')
    initial = torch.load(cli.init_checkpoint,map_location='cpu',weights_only=False)
    if initial['format'] not in ('offline_hand_plan_value_v1','mixed_runtime_hand_plan_value_v1'):
        raise ValueError('Expected completed v4 or mixed-runtime value weights')
    v4_path = Path(sources['v4']['path'])
    v4_config = json.loads((v4_path/'config.json').read_text())
    if v4_config['rollout_runtime_version'] != 'thick_cymbal_top_entry_side5mm_contact_v4':
        raise ValueError('Manifest v4 data has wrong hit scoring')
    if initial['format']=='offline_hand_plan_value_v1':
        if initial['dataset_config_sha256'] != sources['v4']['config_sha256']:
            raise ValueError('Initial v4 model is not from the manifest dataset')
    elif initial['manifest_sha256']!=hashlib.sha256(cli.manifest.read_bytes()).hexdigest():
        raise ValueError('Initial mixed model is not from the manifest dataset')
    v4 = load_value_dataset(v4_path)
    v2 = load_legacy_plan_cases(sources['v2']['path'],
        v4_config['checkpoint_sha256'],v4_config['simulation_geometry_version'])
    points = v4['points'];old_cases = v2['cases']
    if (len(points['actions']) != sources['v4']['rows'] or
            int(old_cases['observed'].sum()) != sources['v2']['rows'] or
            sum(row['rows'] for row in sources.values()) != manifest['total_rows']):
        raise ValueError('Mixed dataset row counts differ from manifest')
    network = HandPlanRankerConfig(**initial['network'])
    model = DualRuntimeRewardModel(network)
    if initial['format']=='offline_hand_plan_value_v1':
        model.v4.load_state_dict(initial['model'])
        model.v2_scorer.load_state_dict(model.v4.candidate_scorer.scorer.state_dict())
    else:model.load_state_dict(initial['model'])
    device = torch.device(cli.device);model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cli.lr,weight_decay=cli.weight_decay)
    manifest_sha = hashlib.sha256(cli.manifest.read_bytes()).hexdigest()
    config = dict(algorithm='v4 reward MSE plus calibrated pairwise/winner ranking; v2 auxiliary runtime head',
        manifest=str(cli.manifest),manifest_sha256=manifest_sha,
        source_counts={key:dict(rows=value['rows'],train_rows=value['train_rows'])
            for key,value in sources.items()},
        source_runtime={key:json.loads((Path(value['path'])/'config.json').read_text())['rollout_runtime']
            for key,value in sources.items()},
        initialization=str(cli.init_checkpoint),initialization_sha256=hashlib.sha256(cli.init_checkpoint.read_bytes()).hexdigest(),
        initialization_step=initial['step'],network=asdict(network),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        objective='equal-weight v4/v2 source MSE + pairwise_weight*pairwise KL + winner_weight*soft-winner KL',
        source_heads='shared drum/hit/group/state encoders; separate v4 and v2 plan scorers',
        collocation_prior=dict(enabled=cli.prior_weight>0,coefficient=cli.prior_weight,
            motif_mode=cli.motif_mode,motif_probability=cli.motif_probability,
            motif_margin_reward=cli.motif_margin,switch_margin_reward=cli.switch_margin,
            scale=cli.prior_scale,cases_per_update=cli.collocation_cases,
            source='fresh IID rhythms paired with independently sampled train-only v4 physical starts',
            loss='one-sided squared violation of human reward-gap margins on legal plan pairs',
            applied_to_runtime='v4 only',dropout_enabled=False,
            independent_sampler_rng=True),
        inference_runtime='v4',decoder_frozen=True,held_out_used_for_training=False,
        checkpoint_selection='latest fixed-budget weights',
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(cli).items()})
    step = 0;rng = np.random.default_rng(cli.seed)
    prior_rng=np.random.default_rng(cli.seed+98123)
    if cli.resume:
        previous = json.loads((cli.out/'config.json').read_text())
        for key in ('manifest_sha256','initialization_sha256','network','source_counts','collocation_prior'):
            if previous[key] != config[key]:raise ValueError(f'Resume changed {key}')
        for key in ('steps','batch','rank_cases','eval_every','lr','min_lr','warmup','weight_decay',
                    'pairwise_weight','winner_weight','pairwise_temperature','winner_temperature',
                    'v4_fraction','seed','device','prior_weight','collocation_cases',
                    'motif_mode','motif_probability','motif_margin','switch_margin','prior_scale'):
            if previous['args'][key] != config['args'][key]:raise ValueError(f'Resume changed {key}')
        saved = torch.load(cli.out/'last.pt',map_location='cpu',weights_only=False)
        if saved['format'] != 'mixed_runtime_hand_plan_value_v1':raise ValueError('Wrong checkpoint format')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        step = saved['step'];rng.bit_generator.state = saved['rng']
        prior_rng.bit_generator.state=saved['prior_rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:write_json(cli.out/'config.json',config)

    bank = torch.as_tensor(ACTION_BANK,device=device)
    v4_obs = torch.as_tensor(points['observations'],device=device)
    v4_actions = torch.as_tensor(points['actions'],device=device)
    v4_rewards = torch.as_tensor(points['rewards'],device=device)
    v4_train_rows = np.flatnonzero(points['splits']==0)
    v4_cases = v4['cases'];v4_train_cases = np.asarray(v4['splits']['train'],np.int64)
    v4_case_obs = torch.as_tensor(v4_cases['observations'],device=device)
    v4_case_rewards = torch.as_tensor(v4_cases['rewards'],device=device)
    v4_case_observed = torch.as_tensor(v4_cases['observed'],device=device)
    v2_obs = torch.as_tensor(old_cases['observations'],device=device)
    v2_rewards = torch.as_tensor(old_cases['rewards'],device=device)
    v2_observed = torch.as_tensor(old_cases['observed'],device=device)
    v2_train_cases = np.asarray(v2['splits']['train'],np.int64)
    local,old_actions = np.nonzero(old_cases['observed'][v2_train_cases])
    old_case_ids = v2_train_cases[local]
    if len(old_case_ids) != sources['v2']['train_rows']:
        raise ValueError('Legacy training rows differ from manifest')
    old_case_ids = torch.as_tensor(old_case_ids,device=device)
    old_actions = torch.as_tensor(old_actions,device=device)
    if cli.prior_weight:
        kit=DrumKit.from_layout()
        eligibility=MinimumTimeChart.load(v4_config['eligibility_source'])
        collocation_sampler=StochasticHandRhythmSampler(kit,4,eligibility)
        training_starts=np.asarray(v4_cases['observations'][v4_train_cases,80:],np.float32)

    def checkpoint():
        save_torch(cli.out/'last.pt',dict(format='mixed_runtime_hand_plan_value_v1',
            step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng.bit_generator.state,prior_rng=prior_rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            manifest_sha256=manifest_sha,source_runtime=config['source_runtime'],
            network=asdict(network),inference_runtime='v4'))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(cli.out/'tb'),flush_secs=5)
    writer.add_text('setup/config',json.dumps(config),0);writer.flush()
    if os.environ.get('TB_AUTO_REGISTER','1') != '0':
        registry = ROOT/'runs/tensorboard_active';registry.mkdir(exist_ok=True)
        link = registry/cli.out.name
        if not link.exists():link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs',timeout=10)):
                break
            time.sleep(5)
        else:raise RuntimeError('TensorBoard did not list mixed value run')
    def assess():
        primary = evaluate_value_model(model,v4)
        auxiliary = evaluate_legacy_runtime(model,v2)
        record = dict(step=step,metrics=primary,legacy_metrics=auxiliary)
        write_json(cli.out/f'evaluation_{step:06d}.json',record)
        with (cli.out/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        for prefix,metrics in (('v4',primary),('v2',auxiliary)):
            for split,values in metrics.items():
                for key,value in values.items():
                    if value is not None:writer.add_scalar(f'{prefix}/{split}/{key}',value,step)
        writer.flush();plot_learning(cli.out)
        print('[mixed-value-evaluation] '+json.dumps(dict(step=step,
            v4={key:{metric:round(values[metric],5) for metric in
                ('rmse','selection_hit_rate','selection_mean_reward_regret')}
                for key,values in primary.items()},
            v2={key:{metric:round(values[metric],5) for metric in
                ('rmse','selection_hit_rate')}
                for key,values in auxiliary.items()})),flush=True)
        return primary,auxiliary
    try:
        if cli.resume and (cli.out/f'evaluation_{step:06d}.json').exists():
            record=json.loads((cli.out/f'evaluation_{step:06d}.json').read_text())
            primary,auxiliary=record['metrics'],record['legacy_metrics']
        else:primary,auxiliary=assess()
        if not cli.resume:checkpoint()
        write_json(cli.out/'run.json',dict(status='running',step=step,metrics=primary,legacy_metrics=auxiliary))
        count4=max(1,min(cli.batch-1,round(cli.batch*cli.v4_fraction)))
        count2=cli.batch-count4
        rank4=max(1,min(cli.rank_cases-1,round(cli.rank_cases*cli.v4_fraction)))
        rank2=cli.rank_cases-rank4
        while step < cli.steps:
            model.train();step+=1
            lr=learning_rate(step,cli.steps,cli.lr,cli.min_lr,cli.warmup)
            for group in optimizer.param_groups:group['lr']=lr
            ids4=torch.as_tensor(rng.choice(v4_train_rows,count4,replace=True),device=device)
            ids2=torch.as_tensor(rng.integers(len(old_case_ids),size=count2),device=device)
            legacy_case=old_case_ids[ids2];legacy_action=old_actions[ids2]
            prediction4=model(v4_obs[ids4],runtime='v4',plans=bank[v4_actions[ids4]][:,None,:])[:,0]
            prediction2=model(v2_obs[legacy_case],runtime='v2',plans=bank[legacy_action][:,None,:])[:,0]
            mse4=(prediction4-v4_rewards[ids4]).square().mean()
            mse2=(prediction2-v2_rewards[legacy_case,legacy_action]).square().mean()
            case4=torch.as_tensor(rng.choice(v4_train_cases,rank4,replace=True),device=device)
            case2=torch.as_tensor(rng.choice(v2_train_cases,rank2,replace=True),device=device)
            p4,w4=reward_ranking_losses(model(v4_case_obs[case4],runtime='v4'),
                v4_case_rewards[case4],v4_case_observed[case4],
                pair_temperature=cli.pairwise_temperature,winner_temperature=cli.winner_temperature)
            p2,w2=reward_ranking_losses(model(v2_obs[case2],runtime='v2'),
                v2_rewards[case2],v2_observed[case2],
                pair_temperature=cli.pairwise_temperature,winner_temperature=cli.winner_temperature)
            pair=(p4+p2)/2;winner=(w4+w2)/2
            prior_loss=mse4.new_zeros(());prior_stats={};sample_stats={}
            if cli.prior_weight:
                collocation_obs,collocation_legal,heuristic,sample_stats=sample_collocations(
                    collocation_sampler,kit,eligibility,training_starts,prior_rng,
                    cli.collocation_cases,motif_mode=cli.motif_mode,
                    motif_probability=cli.motif_probability,
                    motif_margin=cli.motif_margin,switch_margin=cli.switch_margin)
                model.eval()  # Penalize the deployed function, without consuming dropout RNG.
                collocation_scores=model(torch.as_tensor(collocation_obs,device=device),runtime='v4')
                model.train()
                prior_loss,prior_stats=collocation_prior_loss(collocation_scores,
                    torch.as_tensor(heuristic,device=device),
                    torch.as_tensor(collocation_legal,device=device),scale=cli.prior_scale)
            loss=(mse4+mse2)/2+cli.pairwise_weight*pair+cli.winner_weight*winner
            loss+=cli.prior_weight*prior_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite mixed reward/ranking loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step%100==0:
                values=dict(step=step,total_steps=cli.steps,loss=float(loss.detach()),
                    v4_mse=float(mse4.detach()),v2_mse=float(mse2.detach()),
                    v4_pairwise_kl=float(p4.detach()),v2_pairwise_kl=float(p2.detach()),
                    v4_winner_kl=float(w4.detach()),v2_winner_kl=float(w2.detach()),
                    prior_loss=float(prior_loss.detach()),weighted_prior_loss=float((cli.prior_weight*prior_loss).detach()),
                    prior_violated_pairs=prior_stats.get('violated_pairs',0),
                    prior_active_pairs=prior_stats.get('active_pairs',0),
                    collocation_motifs=sample_stats.get('motifs',0),
                    learning_rate=lr)
                write_json(cli.out/'progress.json',values)
                for key,value in values.items():
                    if key not in ('step','total_steps'):writer.add_scalar('fit/'+key,value,step)
                writer.add_scalar('fit/gradient_norm_before_clip',float(grad),step)
            if step%cli.eval_every==0 or step==cli.steps:
                primary,auxiliary=assess();checkpoint()
                write_json(cli.out/'run.json',dict(status='running',step=step,
                    metrics=primary,legacy_metrics=auxiliary))
        write_json(cli.out/'run.json',dict(status='complete',step=step,
            metrics=primary,legacy_metrics=auxiliary))
    except BaseException as exc:
        write_json(cli.out/'run.json',dict(status='failed',step=step,error=repr(exc)))
        raise
    finally:writer.close()


if __name__ == '__main__':main()
