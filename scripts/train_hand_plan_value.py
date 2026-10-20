#!/usr/bin/env python3
"""Fit absolute v4 Isaac reward for one explicit four-group hand plan."""
import argparse
from dataclasses import asdict, replace
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
from isaac_rl.hand_plan_ranker import ACTION_BANK, HandPlanRanker, HandPlanRankerConfig
from isaac_rl.hand_plan_value import (SPLIT_NAMES, evaluate_value_model,
    load_value_dataset, reward_ranking_losses)
from scripts.train_neural_hand_plans import learning_rate, save_torch
from scripts.train_chart_cma import write_json


def plot_learning(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (out/'evaluations.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    specs = (('rmse', 'Reward RMSE'), ('selection_hit_rate', 'Selected-plan hit rate'),
             ('selection_mean_reward_regret', 'Selected-plan reward regret'))
    for axis, (key, label) in zip(axes, specs):
        for split in SPLIT_NAMES:
            axis.plot([row['step'] for row in records],
                [row['metrics'][split][key] for row in records], label=split)
        axis.set_xlabel('Optimizer updates'); axis.set_ylabel(label)
        axis.grid(alpha=.2); axis.legend()
    fig.suptitle('v4 hand-plan absolute reward prediction; latest weights')
    temporary = out/'learning.tmp.png'
    fig.savefig(temporary, dpi=150); temporary.replace(out/'learning.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--weight-decay', type=float, default=.1)
    parser.add_argument('--dropout', type=float, default=.15)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--init-checkpoint', type=Path,
        help='Initialize weights from a completed value run; use a fresh optimizer')
    parser.add_argument('--pairwise-weight', type=float, default=0.)
    parser.add_argument('--winner-weight', type=float, default=0.)
    parser.add_argument('--rank-cases', type=int, default=64)
    parser.add_argument('--pairwise-temperature', type=float, default=.1)
    parser.add_argument('--winner-temperature', type=float, default=.05)
    parser.add_argument('--resume', action='store_true')
    cli = parser.parse_args()
    if min(cli.steps, cli.batch, cli.eval_every, cli.warmup) <= 0:
        raise ValueError('Training budgets must be positive')
    if not 0 <= cli.dropout < 1 or cli.weight_decay < 0:
        raise ValueError('Invalid regularization setting')
    if (min(cli.pairwise_weight, cli.winner_weight) < 0 or cli.rank_cases < 1 or
            min(cli.pairwise_temperature, cli.winner_temperature) <= 0):
        raise ValueError('Invalid ranking setting')
    cli.dataset = cli.dataset.resolve(); cli.out = cli.out.resolve()
    if cli.init_checkpoint: cli.init_checkpoint = cli.init_checkpoint.resolve()
    if cli.out.exists() and not cli.resume and any(cli.out.iterdir()):
        raise FileExistsError(cli.out)
    cli.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(cli.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cli.seed)
    dataset_config = json.loads((cli.dataset/'config.json').read_text())
    run = json.loads((cli.dataset/'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('Expected a completed v4 dataset')
    if dataset_config['rollout_runtime_version'] != 'thick_cymbal_top_entry_side5mm_contact_v4':
        raise ValueError('Reward labels need v4 hit scoring')
    data = load_value_dataset(cli.dataset)
    points = data['points']
    if len(points['actions']) != run['total_datapoints']:
        raise ValueError('Pointwise labels differ from collection manifest')
    device = torch.device(cli.device)
    source_network = dataset_config['network']
    network = replace(HandPlanRankerConfig(**source_network), architecture='shared_candidate',
        hidden=448, group_width=128, state_width=128, scorer_depth=2, dropout=cli.dropout)
    model = HandPlanRanker(network).to(device)
    initialization = None
    if cli.init_checkpoint:
        initialization = torch.load(cli.init_checkpoint, map_location='cpu', weights_only=False)
        if (initialization['format'] != 'offline_hand_plan_value_v1' or
                initialization['dataset_config_sha256'] != hashlib.sha256((cli.dataset/'config.json').read_bytes()).hexdigest() or
                initialization['network'] != asdict(network)):
            raise ValueError('Initial value weights use another dataset or network')
        model.load_state_dict(initialization['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=cli.weight_decay)
    ranking_active = cli.pairwise_weight > 0 or cli.winner_weight > 0
    config = dict(algorithm='Pointwise absolute Isaac reward regression for explicit hand plan',
        objective='reward MSE plus calibrated pairwise and soft-winner KL' if ranking_active
            else 'mean squared error between scalar predicted reward and v4 observed reward',
        ranking_loss_weights=dict(pairwise=cli.pairwise_weight,winner=cli.winner_weight),
        ranking_temperatures=dict(pairwise=cli.pairwise_temperature,winner=cli.winner_temperature),
        data_sampling='uniform observed plan/continuation rows; separate uniform whole-case ranking batch',
        dataset=str(cli.dataset), dataset_config_sha256=hashlib.sha256((cli.dataset/'config.json').read_bytes()).hexdigest(),
        rollout_runtime=dataset_config['rollout_runtime'],
        checkpoint_sha256=dataset_config['checkpoint_sha256'],
        geometry_version=dataset_config['simulation_geometry_version'],
        network=asdict(network), parameters=sum(p.numel() for p in model.parameters()),
        split_counts=data['split_counts'], split_row_counts={name:int((points['splits']==code).sum())
            for code,name in enumerate(SPLIT_NAMES)},
        full_rollout_rows=int((points['source']==0).sum()),
        continuation_rows=int((points['source']==1).sum()),
        initialized_from=str(cli.init_checkpoint) if cli.init_checkpoint else None,
        initialization_sha256=(hashlib.sha256(cli.init_checkpoint.read_bytes()).hexdigest()
            if cli.init_checkpoint else None),
        initialization_step=initialization['step'] if initialization else None,
        args={key:str(value) if isinstance(value,Path) else value for key,value in vars(cli).items()},
        decoder_frozen=True, held_out_used_for_training=False,
        checkpoint_selection='latest fixed-budget weights')
    step = 0; rng = np.random.default_rng(cli.seed)
    if cli.resume:
        previous = json.loads((cli.out/'config.json').read_text())
        for key in ('dataset', 'dataset_config_sha256', 'network', 'split_counts', 'split_row_counts',
                    'initialization_sha256'):
            if config[key] != previous[key]: raise ValueError(f'Resume differs in {key}')
        for key in ('steps', 'batch', 'eval_every', 'lr', 'min_lr', 'warmup',
                    'weight_decay', 'dropout', 'seed', 'device', 'pairwise_weight',
                    'winner_weight', 'rank_cases', 'pairwise_temperature', 'winner_temperature'):
            if config['args'][key] != previous['args'][key]: raise ValueError(f'Resume differs in {key}')
        saved = torch.load(cli.out/'last.pt', map_location='cpu', weights_only=False)
        if saved['format'] != 'offline_hand_plan_value_v1': raise ValueError('Wrong checkpoint format')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']; rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        write_json(cli.out/'config.json', config)
    observations = torch.as_tensor(points['observations'], device=device)
    actions = torch.as_tensor(points['actions'], device=device)
    rewards = torch.as_tensor(points['rewards'], device=device)
    training_rows = np.flatnonzero(points['splits']==0)
    action_bank = torch.as_tensor(ACTION_BANK, device=device)
    if ranking_active:
        cases = data['cases']
        case_observations = torch.as_tensor(cases['observations'], device=device)
        case_rewards = torch.as_tensor(cases['rewards'], device=device)
        case_observed = torch.as_tensor(cases['observed'], device=device)
        training_cases = np.asarray(data['splits']['train'], np.int64)
    def checkpoint():
        save_torch(cli.out/'last.pt', dict(format='offline_hand_plan_value_v1',
            step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            dataset=str(cli.dataset), dataset_config_sha256=config['dataset_config_sha256'],
            rollout_runtime=config['rollout_runtime'], network=asdict(network)))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(cli.out/'tb'), flush_secs=5)
    writer.add_text('setup/config', json.dumps(config), 0); writer.flush()
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        link = registry/cli.out.name
        if not link.exists(): link.symlink_to((cli.out/'tb').resolve())
        for _ in range(6):
            if cli.out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs', timeout=10)):
                break
            time.sleep(5)
        else: raise RuntimeError('TensorBoard did not list v4 value run')
    def assess():
        metrics = evaluate_value_model(model, data)
        record = dict(step=step, metrics=metrics)
        write_json(cli.out/f'evaluation_{step:06d}.json', record)
        with (cli.out/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        for name, stats in metrics.items():
            for key, value in stats.items():
                if value is not None: writer.add_scalar(f'{name}/{key}', value, step)
        writer.flush(); plot_learning(cli.out)
        print('[value-evaluation] '+json.dumps(record), flush=True)
        return metrics
    try:
        if cli.resume and (cli.out/f'evaluation_{step:06d}.json').exists():
            metrics = json.loads((cli.out/f'evaluation_{step:06d}.json').read_text())['metrics']
        else: metrics = assess()
        if not cli.resume: checkpoint()
        write_json(cli.out/'run.json', dict(status='running', step=step, metrics=metrics))
        while step < cli.steps:
            model.train()
            ids = torch.as_tensor(rng.choice(training_rows, size=cli.batch, replace=True), device=device)
            step += 1
            lr = learning_rate(step, cli.steps, cli.lr, cli.min_lr, cli.warmup)
            for group in optimizer.param_groups: group['lr'] = lr
            prediction = model.score_candidates(observations[ids], action_bank[actions[ids]][:, None, :])[:, 0]
            error = prediction-rewards[ids]
            mse = error.square().mean()
            pair = mse.new_zeros(()); winner = mse.new_zeros(())
            if ranking_active:
                case_ids = torch.as_tensor(rng.choice(training_cases,size=cli.rank_cases,replace=True),device=device)
                case_predictions = model(case_observations[case_ids])
                pair,winner = reward_ranking_losses(case_predictions,
                    case_rewards[case_ids],case_observed[case_ids],
                    pair_temperature=cli.pairwise_temperature,
                    winner_temperature=cli.winner_temperature)
            loss = mse+cli.pairwise_weight*pair+cli.winner_weight*winner
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite reward MSE')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step % 100 == 0:
                writer.add_scalar('fit/loss', float(loss.detach()), step)
                writer.add_scalar('fit/mse', float(mse.detach()), step)
                writer.add_scalar('fit/rmse', float(mse.detach().sqrt()), step)
                if ranking_active:
                    writer.add_scalar('fit/pairwise_kl',float(pair.detach()),step)
                    writer.add_scalar('fit/winner_kl',float(winner.detach()),step)
                    writer.add_scalar('fit/weighted_pairwise',float((cli.pairwise_weight*pair).detach()),step)
                    writer.add_scalar('fit/weighted_winner',float((cli.winner_weight*winner).detach()),step)
                writer.add_scalar('fit/lr', lr, step)
                writer.add_scalar('fit/gradient_norm_before_clip', float(grad), step)
                write_json(cli.out/'progress.json', dict(step=step, total_steps=cli.steps,
                    train_batch_mse=float(mse.detach()),pairwise_kl=float(pair.detach()),
                    winner_kl=float(winner.detach()),total_loss=float(loss.detach()),learning_rate=lr))
            if step % cli.eval_every == 0 or step == cli.steps:
                metrics = assess(); checkpoint()
                write_json(cli.out/'run.json', dict(status='running', step=step, metrics=metrics))
        write_json(cli.out/'run.json', dict(status='complete', step=step, metrics=metrics,
            training_epochs_equivalent=step*cli.batch/len(training_rows)))
    except BaseException as exc:
        write_json(cli.out/'run.json', dict(status='failed', step=step, error=repr(exc)))
        raise
    finally: writer.close()


if __name__ == '__main__': main()
