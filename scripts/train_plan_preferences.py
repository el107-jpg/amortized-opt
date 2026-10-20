#!/usr/bin/env python3
"""Matched plan trials and Bayesian linear preference-cost corrections."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from amortized_opt.rhythm import DrumKit

from scripts.train_chart_cma import parse_args as runtime_args, write_json
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler, planned_transition_coverage
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.plan_preferences import PlanPreferenceLearner, select_context_plans


def build_batch(panel, chart, learner, ready, rng, candidates, exploration, selection='top_two'):
    contexts = [select_context_plans(c, chart, learner, ready, rng, candidates, exploration, selection) for c in panel]
    # Candidate-major layout splits each slot across an equal number of GPUs.
    # Corresponding context slices retain matching nominal world origins.
    episodes = [context['episodes'][k] for k in range(candidates) for context in contexts]
    return dict(episodes=episodes, contexts=contexts,
        features=np.stack([x['features'] for x in contexts]),
        base_scores=np.stack([x['base_scores'] for x in contexts]),
        graph_slots=(np.array([x['graph_slot'] for x in contexts]) if selection == 'thompson' else None),
        selection=selection,
        actions=np.stack([x['actions'] for x in contexts]))


def summarize_rewards(batch, reports):
    B, K = batch['base_scores'].shape
    if len(reports) != B*K:
        raise ValueError('One physical rollout per candidate required')
    rewards = np.array([r['reward'] for r in reports]).reshape(K, B).T
    hit_rates = np.array([r['hit_rate'] for r in reports]).reshape(K, B).T
    metrics = dict(mean_greedy_reward=float(rewards[:, 0].mean()),
        mean_best_observed_reward=float(rewards.max(1).mean()),
        mean_greedy_hit_rate=float(hit_rates[:, 0].mean()),
        mean_oracle_advantage=float((rewards.max(1)-rewards[:, 0]).mean()),
        greedy_win_fraction=float((rewards[:, 0] >= rewards.max(1)-1e-10).mean()))
    if batch.get('selection', 'thompson') == 'top_two':
        metrics.update(mean_second_reward=float(rewards[:, 1].mean()),
            mean_second_hit_rate=float(hit_rates[:, 1].mean()),
            second_beats_greedy_fraction=float((rewards[:, 1] > rewards[:, 0]).mean()))
    else:
        graph = batch['graph_slots']
        metrics.update(mean_graph_reward=float(rewards[np.arange(B), graph].mean()),
                       mean_graph_hit_rate=float(hit_rates[np.arange(B), graph].mean()))
    return rewards, metrics


def export_correction(path, chart, learner, checkpoint):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, format='plan_preference_cost_correction_v1',
            coefficients=learner.mean, covariance=learner.covariance,
            metadata_json=json.dumps(dict(iteration=learner.updates,
                base_chart=str(chart.resolve()), base_sha256=hashlib.sha256(chart.read_bytes()).hexdigest(),
                checkpoint=str(checkpoint), selection='current_preference_MAP',
                parameter_units='normalized_plan_cost_not_seconds')))
    temporary.replace(path)


def main():
    # The task optimizes transition-time anchors. Keep the erroneous cost-head
    # experiment explicitly archival so the usual entrypoint cannot relaunch it.
    if '--legacy-plan-costs' not in sys.argv[1:]:
        from scripts.train_transition_time_preferences import main as timing_main
        timing_main()
        return
    sys.argv.remove('--legacy-plan-costs')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'runs/chart_cma_4096_20261006')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--contexts', type=int, default=None)
    parser.add_argument('--candidates', type=int, default=None)
    parser.add_argument('--selection', choices=('top_two', 'thompson'), default=None)
    parser.add_argument('--groups', type=int, choices=(4, 6), default=4)
    parser.add_argument('--devices', nargs='+', default=None)
    parser.add_argument('--prior-std', type=float, default=1.)
    parser.add_argument('--temperature', type=float, default=.25)
    parser.add_argument('--exploration', type=float, default=1.)
    parser.add_argument('--replay-capacity', type=int, default=16384)
    parser.add_argument('--reward-deadband', type=float, default=.05)
    parser.add_argument('--reward-gap-scale', type=float, default=.10)
    parser.add_argument('--reward-temperature', type=float, default=.05)
    parser.add_argument('--indifference-weight', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    requested_layout = dict(contexts=cli.contexts, candidates=cli.candidates, selection=cli.selection)
    explicit_devices = cli.devices
    out = cli.out.resolve()
    if cli.resume:
        config = json.loads((cli.resume.resolve().parent/'config.json').read_text())
        learner, saved = PlanPreferenceLearner.load(cli.resume)
        for key, value in config['preference_args'].items():
            if key not in ('out', 'resume', 'iterations', 'devices', 'dry_run', *requested_layout):
                setattr(cli, key, value)
        cli.source_run = Path(cli.source_run)
        cli.devices = explicit_devices or config['preference_args']['devices']
        source_config = config['source_config']
        chart_file = cli.resume.resolve().parent/'base_chart.npz'
    else:
        cli.devices = explicit_devices or ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        source_config = json.loads((cli.source_run/'config.json').read_text())
        chart_file = cli.source_run/'last_chart.npz'
    prior_layout = config['preference_args'] if cli.resume else {}
    cli.selection = requested_layout['selection'] or prior_layout.get('selection', 'thompson' if cli.resume else 'top_two')
    selection_changed = cli.resume and cli.selection != prior_layout.get('selection', 'thompson')
    cli.candidates = (requested_layout['candidates'] if requested_layout['candidates'] is not None else
                      (2 if cli.selection == 'top_two' else 4) if selection_changed else
                      prior_layout.get('candidates', 2 if cli.selection == 'top_two' else 4))
    default_contexts = (prior_layout['contexts']*prior_layout['candidates']//cli.candidates
                        if selection_changed else prior_layout.get('contexts', 2048 if cli.selection == 'top_two' else 1024))
    cli.contexts = requested_layout['contexts'] if requested_layout['contexts'] is not None else default_contexts
    if (min(cli.iterations, cli.contexts, cli.candidates) < 1
            or len(cli.devices) % cli.candidates != 0
            or cli.contexts % (len(cli.devices)//cli.candidates) != 0):
        parser.error('GPUs must divide evenly into candidate slots, with equal context shards')
    if cli.selection == 'top_two' and cli.candidates != 2:
        parser.error('Top-two selection requires --candidates 2')
    if cli.selection == 'top_two':
        cli.exploration = 0.
    if len(set(cli.devices)) != len(cli.devices) or not np.isfinite(cli.exploration) or cli.exploration < 0:
        parser.error('Distinct devices and finite nonnegative exploration required')
    checkpoint = Path(source_config['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != source_config['checkpoint_sha256']:
        raise ValueError('Frozen decoder checkpoint changed')
    if cli.resume and saved['checkpoint_sha256'] != source_config['checkpoint_sha256']:
        raise ValueError('Resumed frozen decoder differs from learner checkpoint')
    base = MinimumTimeChart.load(source_config['args']['table'])
    chart = ConditionalMinimumTimeChart.load(chart_file, base)
    ready = np.load(source_config['args']['ready_initial'])
    if not cli.resume:
        learner = PlanPreferenceLearner(len(chart.get_parameters()), prior_std=cli.prior_std,
            temperature=cli.temperature, replay_capacity=cli.replay_capacity,
            deadband=cli.reward_deadband, gap_scale=cli.reward_gap_scale,
            reward_temperature=cli.reward_temperature, indifference_weight=cli.indifference_weight)
        saved = {}
    if cli.dry_run:
        print(json.dumps(dict(parameters=learner.dimensions, physical_environments=cli.contexts*cli.candidates,
            contexts=cli.contexts, candidates=cli.candidates, groups=cli.groups,
            correction_initialized_to_zero=not cli.resume, covariance='full Laplace posterior',
            frozen_decoder=str(checkpoint), neural_phase2_training=False), indent=2))
        return
    if out.exists() and not cli.resume:
        raise FileExistsError('New run directory required or use --resume')
    out.mkdir(parents=True, exist_ok=True)
    frozen_chart = out/'base_chart.npz'
    if chart_file.resolve() != frozen_chart:
        frozen_chart.write_bytes(chart_file.read_bytes())
    source_snapshot = out/'source_config.json'
    write_json(source_snapshot, source_config)
    # Reuse the established frozen actor, controller and nominal scene setup.
    source_args = source_config['args']
    args = runtime_args(['--out', str(out), '--envs', str(cli.contexts*cli.candidates),
        '--device', cli.devices[0], '--physics-hz', '250', '--rollout-devices', *cli.devices,
        '--decode-chunk', '512', '--collision-chunk', '1024', '--no-plan-diagnostics',
        '--clone-kit-from-source'])
    for key in ('torque_weight', 'joint_weight', 'collision_weight', 'collision_stride',
                'near_miss_weight', 'near_miss_scale_m', 'near_miss_window_s'):
        setattr(args, key, source_args[key])
    args.groups, args.horizon, args.seed = cli.groups, cli.groups, cli.seed
    args.decoder_checkpoint = str(checkpoint)
    args.overlay = str(frozen_chart)
    args.table = source_args['table']
    args.ready_initial = source_args['ready_initial']
    pref_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()}
    config = dict(args=vars(args), preference_args=pref_args, source_config=source_config,
        source_chart_sha256=hashlib.sha256(frozen_chart.read_bytes()).hexdigest(),
        algorithm='Matched Bayesian plan preferences with full covariance',
        checkpoint=str(checkpoint), checkpoint_sha256=source_config['checkpoint_sha256'],
        phase2_decoder_training=False, frozen_generator=True, context=f'nominal_ready_{cli.groups}_hit_group_windows',
        correction_units='normalized_plan_cost_not_seconds',
        chart_init='preserved_MAP_on_resume' if cli.resume else 'fixed_source_chart_zero_correction',
        selection=cli.selection, uncertainty_used_for_selection=cli.selection == 'thompson',
        plan_trial='whole window committed', physics_hz=250, control_hz=50,
        covariance='inverse weighted logistic curvature plus Gaussian prior precision',
        covariance_note='Replay rebuilt once per fit; repeated optimization steps do not count observations repeatedly')
    if cli.resume and config['source_chart_sha256'] != saved['base_sha256']:
        raise ValueError('Resume base chart differs from saved learner')
    write_json(out/'config.json', config)
    rng = np.random.default_rng(cli.seed)
    if cli.resume:
        rng.bit_generator.state = saved['rng']
    pending = saved.get('pending')
    if pending is not None and (pending.get('selection', 'thompson') != cli.selection
            or pending['base_scores'].shape != (cli.contexts, cli.candidates)):
        pending = None  # Explicit sampler/layout change retains learner/RNG and replaces unfinished trials.
    sampler = ProceduralChartRhythmSampler(DrumKit.from_layout(), cli.groups, chart)
    seed_clip = out/'scene_seed.json'
    write_json(seed_clip, sampler.seed_payload())
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out/'tb'), flush_secs=5)
    writer.add_text('setup/config', json.dumps(config), 0); writer.flush()
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        link = registry/out.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(out/'tb')
        if not link.is_symlink() or link.resolve() != out/'tb':
            raise ValueError('TensorBoard registration belongs to another run')
        for _ in range(6):
            if out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs', timeout=10)):
                break
            time.sleep(5)
        else:
            raise RuntimeError('Shared TensorBoard did not list the run')
    def save_state(batch):
        learner.save(out/'last.pkl', rng=rng.bit_generator.state, pending=batch,
            base_sha256=config['source_chart_sha256'], checkpoint_sha256=source_config['checkpoint_sha256'])
    export_correction(out/'last_correction.npz', frozen_chart, learner, checkpoint)
    save_state(pending)
    evaluator = None
    try:
        evaluator = DistributedIsaacEvaluator(args, checkpoint, seed_clip)
        while learner.updates < cli.iterations:
            started = time.monotonic()
            if pending is None:
                panel = sampler.sample(cli.contexts, rng, minimum_plan_choices=2 if cli.selection == 'top_two' else 1)
                pending = build_batch(panel, chart, learner, ready, rng, cli.candidates, cli.exploration, cli.selection)
                pending['tiers'] = sampler.last_strata.copy()
                save_state(pending)
            planning_s = time.monotonic()-started
            def progress(offset, total, tick, ticks):
                write_json(out/'progress.json', dict(iteration=learner.updates+1,
                    phase='matched_plan_trials', episodes=total, completed=offset, tick=tick, ticks=ticks))
                print(f'[preferences] iteration={learner.updates+1} tick={tick}/{ticks}', flush=True)
            stamp = time.monotonic()
            reports = evaluator.evaluate(pending['episodes'], progress)
            rollout_s = time.monotonic()-stamp
            rewards, metrics = summarize_rewards(pending, reports)
            before_mean = learner.mean.copy()
            before_covariance = learner.covariance.copy()
            metrics.update(learner.add_preferences(pending['features'], pending['base_scores'], rewards))
            stamp = time.monotonic()
            metrics.update(learner.fit())
            fit_s = time.monotonic()-stamp
            metrics.update(iteration=learner.updates, planning_seconds=planning_s,
                rollout_seconds=rollout_s, fit_seconds=fit_s,
                contexts=cli.contexts, candidates=cli.candidates, selection=cli.selection,
                logical_rollouts=len(reports), physical_rollouts=len(reports),
                mean_distinct_selected=float(np.mean([c['distinct_selected'] for c in pending['contexts']])),
                mean_predictive_uncertainty=float(np.mean([c['predictive_uncertainty'] for c in pending['contexts']])))
            metrics.update(planned_transition_coverage(pending['episodes'], len(chart.names)))
            np.savez_compressed(out/f'iteration_{learner.updates:05d}.npz',
                features=pending['features'], base_scores=pending['base_scores'], actions=pending['actions'],
                rewards=rewards, tiers=pending['tiers'],
                graph_slots=pending['graph_slots'] if pending['graph_slots'] is not None else np.empty(0, int),
                before_mean=before_mean, after_mean=learner.mean, before_covariance=before_covariance,
                after_covariance=learner.covariance)
            write_json(out/f'iteration_{learner.updates:05d}.json', dict(metrics=metrics, reports=reports))
            pending = None
            export_correction(out/'last_correction.npz', frozen_chart, learner, checkpoint)
            save_state(None)
            metrics['iteration_seconds'] = time.monotonic()-started
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics)+'\n')
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar('preferences/'+name, value, learner.updates)
            writer.flush()
            write_json(out/'run.json', dict(status='running', completed_iterations=learner.updates))
            print('[preferences-result] '+json.dumps(metrics), flush=True)
        write_json(out/'run.json', dict(status='complete', completed_iterations=learner.updates))
    except BaseException as exc:
        write_json(out/'run.json', dict(status='failed', completed_iterations=learner.updates, error=repr(exc)))
        raise
    finally:
        writer.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    main()
