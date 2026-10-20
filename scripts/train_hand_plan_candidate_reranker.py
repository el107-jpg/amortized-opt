#!/usr/bin/env python3
"""Test candidate-aware self-attention on frozen v4/v2 hand-plan values."""
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
from isaac_rl.hand_plan_ranker import HandPlanRankerConfig
from isaac_rl.hand_plan_candidate_reranker import (CandidateAwarePlanReranker,
    evaluate_candidate_reranker)
from isaac_rl.hand_plan_value import (DualRuntimeRewardModel,
    load_legacy_plan_cases, load_value_dataset,
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
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--attention-width',type=int,default=128)
    parser.add_argument('--attention-heads',type=int,default=4)
    parser.add_argument('--attention-layers',type=int,default=2)
    parser.add_argument('--attention-dropout',type=float,default=.1)
    parser.add_argument('--resume', action='store_true')
    cli = parser.parse_args()
    if (min(cli.steps,cli.batch,cli.rank_cases,cli.eval_every,cli.warmup,
            cli.lr,cli.min_lr,cli.pairwise_temperature,cli.winner_temperature) <= 0 or
            min(cli.pairwise_weight,cli.winner_weight,cli.weight_decay) < 0 or
            not 0 < cli.v4_fraction < 1):
        raise ValueError('Invalid mixed-runtime fitting setting')
    if (cli.attention_width<1 or cli.attention_heads<1 or cli.attention_layers<1 or
            cli.attention_width%cli.attention_heads or not 0<=cli.attention_dropout<1):
        raise ValueError('Invalid candidate attention setting')
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
    if initial['format'] != 'mixed_runtime_hand_plan_value_v1' or initial['step']!=20000:
        raise ValueError('Expected completed v4/v2 mixed value weights')
    v4_path = Path(sources['v4']['path'])
    v4_config = json.loads((v4_path/'config.json').read_text())
    manifest_sha=hashlib.sha256(cli.manifest.read_bytes()).hexdigest()
    if (initial['manifest_sha256']!=manifest_sha or
            v4_config['rollout_runtime_version'] != 'thick_cymbal_top_entry_side5mm_contact_v4'):
        raise ValueError('Initial model is not from the mixed dataset manifest')
    v4 = load_value_dataset(v4_path)
    v2 = load_legacy_plan_cases(sources['v2']['path'],
        v4_config['checkpoint_sha256'],v4_config['simulation_geometry_version'])
    points = v4['points'];old_cases = v2['cases']
    if (len(points['actions']) != sources['v4']['rows'] or
            int(old_cases['observed'].sum()) != sources['v2']['rows'] or
            sum(row['rows'] for row in sources.values()) != manifest['total_rows']):
        raise ValueError('Mixed dataset row counts differ from manifest')
    network = HandPlanRankerConfig(**initial['network'])
    base = DualRuntimeRewardModel(network)
    base.load_state_dict(initial['model'])
    model = CandidateAwarePlanReranker(base,width=cli.attention_width,
        heads=cli.attention_heads,layers=cli.attention_layers,
        dropout=cli.attention_dropout)
    device = torch.device(cli.device);model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=cli.lr,weight_decay=cli.weight_decay)
    config = dict(algorithm='candidate-aware attention residual over frozen v4/v2 values',
        manifest=str(cli.manifest),manifest_sha256=manifest_sha,
        source_counts={key:dict(rows=value['rows'],train_rows=value['train_rows'])
            for key,value in sources.items()},
        source_runtime={key:json.loads((Path(value['path'])/'config.json').read_text())['rollout_runtime']
            for key,value in sources.items()},
        initialization=str(cli.init_checkpoint),initialization_sha256=hashlib.sha256(cli.init_checkpoint.read_bytes()).hexdigest(),
        initialization_step=initial['step'],network=asdict(network),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        reranker=dict(width=cli.attention_width,heads=cli.attention_heads,layers=cli.attention_layers,
            dropout=cli.attention_dropout,legal_candidate_mask=True,
            candidate_position_encoding=False,correction='mean centered over legal plans',
            initialized_correction='exactly zero'),
        objective='equal-weight v4/v2 source MSE + pairwise_weight*pairwise KL + winner_weight*soft-winner KL',
        source_heads='frozen shared value encoder; shared candidate attention; separate v4/v2 corrections',
        inference_runtime='v4',decoder_frozen=True,held_out_used_for_training=False,
        checkpoint_selection='latest fixed-budget weights',
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(cli).items()})
    step = 0;rng = np.random.default_rng(cli.seed)
    if cli.resume:
        previous = json.loads((cli.out/'config.json').read_text())
        for key in ('manifest_sha256','initialization_sha256','network','source_counts','reranker'):
            if previous[key] != config[key]:raise ValueError(f'Resume changed {key}')
        for key in ('steps','batch','rank_cases','eval_every','lr','min_lr','warmup','weight_decay',
                    'pairwise_weight','winner_weight','pairwise_temperature','winner_temperature',
                    'v4_fraction','seed','device','attention_width','attention_heads',
                    'attention_layers','attention_dropout'):
            if previous['args'][key] != config['args'][key]:raise ValueError(f'Resume changed {key}')
        saved = torch.load(cli.out/'last.pt',map_location='cpu',weights_only=False)
        if saved['format'] != 'candidate_aware_hand_plan_reranker_v1':raise ValueError('Wrong checkpoint format')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        step = saved['step'];rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:write_json(cli.out/'config.json',config)

    v4_obs = torch.as_tensor(points['observations'],device=device)
    v4_actions = torch.as_tensor(points['actions'],device=device)
    v4_rewards = torch.as_tensor(points['rewards'],device=device)
    v4_point_cases = torch.as_tensor(points['case_ids'],device=device)
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

    def checkpoint():
        save_torch(cli.out/'last.pt',dict(format='candidate_aware_hand_plan_reranker_v1',
            step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            manifest_sha256=manifest_sha,source_runtime=config['source_runtime'],
            network=asdict(network),reranker=config['reranker'],
            initialization_sha256=config['initialization_sha256'],inference_runtime='v4'))
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
        else:raise RuntimeError('TensorBoard did not list candidate-aware run')
    def assess():
        primary,auxiliary = evaluate_candidate_reranker(model,v4,v2)
        record = dict(step=step,metrics=primary,legacy_metrics=auxiliary)
        write_json(cli.out/f'evaluation_{step:06d}.json',record)
        with (cli.out/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        for prefix,metrics in (('v4',primary),('v2',auxiliary)):
            for split,values in metrics.items():
                for key,value in values.items():
                    if value is not None:writer.add_scalar(f'{prefix}/{split}/{key}',value,step)
        writer.flush();plot_learning(cli.out)
        print('[candidate-aware-evaluation] '+json.dumps(dict(step=step,
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
            prediction4=model.score_observed_rows(v4_obs[ids4],v4_actions[ids4],
                v4_case_observed[v4_point_cases[ids4]])
            prediction2=model.score_observed_rows(v2_obs[legacy_case],legacy_action,
                v2_observed[legacy_case],runtime='v2')
            mse4=(prediction4-v4_rewards[ids4]).square().mean()
            mse2=(prediction2-v2_rewards[legacy_case,legacy_action]).square().mean()
            case4=torch.as_tensor(rng.choice(v4_train_cases,rank4,replace=True),device=device)
            case2=torch.as_tensor(rng.choice(v2_train_cases,rank2,replace=True),device=device)
            scores4,correction4=model(v4_case_obs[case4],runtime='v4',
                legal=v4_case_observed[case4],return_correction=True)
            scores2,correction2=model(v2_obs[case2],runtime='v2',
                legal=v2_observed[case2],return_correction=True)
            p4,w4=reward_ranking_losses(scores4,
                v4_case_rewards[case4],v4_case_observed[case4],
                pair_temperature=cli.pairwise_temperature,winner_temperature=cli.winner_temperature)
            p2,w2=reward_ranking_losses(scores2,
                v2_rewards[case2],v2_observed[case2],
                pair_temperature=cli.pairwise_temperature,winner_temperature=cli.winner_temperature)
            pair=(p4+p2)/2;winner=(w4+w2)/2
            loss=(mse4+mse2)/2+cli.pairwise_weight*pair+cli.winner_weight*winner
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite mixed reward/ranking loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step%100==0:
                values=dict(step=step,total_steps=cli.steps,loss=float(loss.detach()),
                    v4_mse=float(mse4.detach()),v2_mse=float(mse2.detach()),
                    v4_pairwise_kl=float(p4.detach()),v2_pairwise_kl=float(p2.detach()),
                    v4_winner_kl=float(w4.detach()),v2_winner_kl=float(w2.detach()),
                    v4_correction_rms=float((correction4.square()*v4_case_observed[case4]).sum().div(
                        v4_case_observed[case4].sum()).sqrt().detach()),
                    v2_correction_rms=float((correction2.square()*v2_observed[case2]).sum().div(
                        v2_observed[case2].sum()).sqrt().detach()),
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
