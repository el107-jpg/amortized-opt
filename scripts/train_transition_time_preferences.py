#!/usr/bin/env python3
"""Matched top-ranked plan trials updating actual transition-time anchors."""
from __future__ import annotations

import argparse
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
from amortized_opt.rhythm import DrumKit
from scripts.train_chart_cma import parse_args as runtime_args, write_json
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler, planned_transition_coverage
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.transition_time_preferences import (TransitionTimePreferenceLearner,
    build_time_batch, timing_scores_and_jacobian, unpack_time_reports)


def export_chart(path, learner):
    chart = learner.current_chart()
    chart.metadata['parameter_search'] = dict(algorithm='Direct transition-time plan preferences',
        generation=learner.updates, selection='current_time_parameters', parameter_units='seconds',
        max_update_ms=learner.max_update_s*1000)
    temporary = path.with_suffix('.tmp')
    chart.save(temporary); temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'runs/chart_cma_4096_20261006')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--contexts', type=int, default=None)
    parser.add_argument('--candidates', type=int, default=None)
    parser.add_argument('--selection', choices=('top_two', 'top_k'), default=None)
    parser.add_argument('--groups', type=int, choices=(4, 6), default=4)
    parser.add_argument('--devices', nargs='+', default=None)
    parser.add_argument('--prior-std-ms', type=float, default=50.)
    parser.add_argument('--max-update-ms', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=.25)
    parser.add_argument('--score-scale', type=float, default=10.)
    parser.add_argument('--reward-temperature', type=float, default=.05)
    parser.add_argument('--replay-capacity', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args(argv)
    out = cli.out.resolve()
    requested_contexts, requested_devices, requested_cap = cli.contexts, cli.devices, cli.max_update_ms
    requested_candidates, requested_selection = cli.candidates, cli.selection
    if cli.resume:
        config = json.loads((cli.resume.resolve().parent/'config.json').read_text())
        learner, extra = TransitionTimePreferenceLearner.load(cli.resume)
        for key, value in config['time_args'].items():
            if key not in ('out', 'resume', 'iterations', 'contexts', 'devices', 'dry_run', 'candidates', 'selection'):
                setattr(cli, key, value)
        cli.contexts = requested_contexts or config['time_args']['contexts']
        cli.devices = requested_devices or config['time_args']['devices']
        cli.max_update_ms = requested_cap if requested_cap is not None else config['time_args']['max_update_ms']
        cli.candidates = requested_candidates if requested_candidates is not None else config['time_args']['candidates']
        learner.set_max_update_ms(cli.max_update_ms)
        source = config['source_config']
        initial_path = cli.resume.resolve().parent/'initial_chart.npz'
    else:
        cli.max_update_ms = requested_cap if requested_cap is not None else 50.
        cli.contexts = requested_contexts or 2048
        cli.devices = requested_devices or ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
        cli.candidates = requested_candidates if requested_candidates is not None else 2
        source = json.loads((cli.source_run/'config.json').read_text())
        initial_path = cli.source_run/'last_chart.npz'
        base = MinimumTimeChart.load(source['args']['table'])
        initial_chart = ConditionalMinimumTimeChart.load(initial_path, base)
        learner = TransitionTimePreferenceLearner(initial_chart, prior_std_ms=cli.prior_std_ms,
            max_update_ms=cli.max_update_ms, temperature=cli.temperature,
            score_scale=cli.score_scale, reward_temperature=cli.reward_temperature,
            replay_capacity=cli.replay_capacity, groups=cli.groups)
        extra = {}
    cli.selection = requested_selection or ('top_two' if cli.candidates == 2 else 'top_k')
    if (cli.candidates < 2 or min(cli.iterations, cli.contexts) <= 0 or not cli.devices
            or (cli.selection == 'top_two' and cli.candidates != 2)):
        parser.error('At least two candidates, positive budgets and consistent selection required')
    checkpoint = Path(source['checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != source['checkpoint_sha256'] or (cli.resume and digest != extra['checkpoint_sha256']):
        raise ValueError('Frozen decoder differs from timing experiment')
    if cli.dry_run:
        print(json.dumps(dict(parameters=len(learner.parameters_s), parameter_units='seconds',
            physical_envs=cli.candidates*cli.contexts, contexts=cli.contexts, candidates=cli.candidates,
            selection=f'exact_current_timing_score_top_{cli.candidates}', max_update_ms=learner.max_update_s*1000,
            prior_std_ms=learner.prior_std_s*1000, auxiliary_cost_layer=False), indent=2))
        return
    if out.exists() and not cli.resume:
        raise FileExistsError('Use a fresh directory or a timing checkpoint to resume')
    out.mkdir(parents=True, exist_ok=True)
    initial_copy = out/'initial_chart.npz'
    if initial_copy.resolve() != initial_path.resolve():
        initial_copy.write_bytes(initial_path.read_bytes())
    initial_digest = hashlib.sha256(initial_copy.read_bytes()).hexdigest()
    if cli.resume and initial_digest != extra['initial_chart_sha256']:
        raise ValueError('Initial transition chart changed on resume')
    if not (out/'before_continuation_chart.npz').exists():
        export_chart(out/'before_continuation_chart.npz', learner)
    ready = np.load(source['args']['ready_initial'])
    args = runtime_args(['--out', str(out), '--envs', str(cli.candidates*cli.contexts), '--device', cli.devices[0],
        '--physics-hz', '250', '--rollout-devices', *cli.devices, '--decode-chunk', '512',
        '--collision-chunk', '1024', '--no-plan-diagnostics', '--clone-kit-from-source'])
    for key in ('torque_weight', 'joint_weight', 'collision_weight', 'collision_stride',
                'near_miss_weight', 'near_miss_scale_m', 'near_miss_window_s'):
        setattr(args, key, source['args'][key])
    args.groups = args.horizon = cli.groups; args.seed = cli.seed
    args.table = source['args']['table']; args.ready_initial = source['args']['ready_initial']
    args.decoder_checkpoint = str(checkpoint); args.overlay = str(initial_copy)
    config = dict(args=vars(args), time_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        source_config=source, checkpoint=str(checkpoint), checkpoint_sha256=digest,
        parameter_units='seconds', auxiliary_cost_layer=False, uncertainty_used_for_selection=False,
        algorithm='Direct transition-time plan preferences', physics_hz=250, control_hz=50,
        preference_likelihood='fixed_scaled_raw_cost_difference_shared_cost_invariant',
        assignment_score='timing_slack_only', historical_calibration_penalties=False,
        missing_ready_policy='unknown_no_surcharge',
        candidate_selection=f'current_chart_top_{cli.candidates}_legal_plans',
        preference_pairs='all_legal_candidate_pairs', pair_weighting='mean_pairwise_loss_per_rhythm',
        context=f'nominal_ready_{cli.groups}_hit_group_windows',
        max_update_ms=learner.max_update_s*1000, prior_std_ms=learner.prior_std_s*1000)
    write_json(out/'config.json', config)
    rng = np.random.default_rng(cli.seed)
    if cli.resume:
        rng.bit_generator.state = extra['rng']
    pending = extra.get('pending')
    if pending is not None and pending['scores'].shape != (cli.contexts, cli.candidates):
        raise ValueError('Cannot change contexts or candidates with pending timing trials')
    sampler = ProceduralChartRhythmSampler(DrumKit.from_layout(), cli.groups, learner.template)
    seed_clip = out/'scene_seed.json'; write_json(seed_clip, sampler.seed_payload())
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out/'tb'), flush_secs=5)
    writer.add_text('setup/config', json.dumps(config), 0); writer.flush()
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        link = registry/out.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(out/'tb')
        if not link.is_symlink() or link.resolve() != out/'tb':
            raise ValueError('TensorBoard registration conflict')
        for _ in range(6):
            if out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs', timeout=10)):
                break
            time.sleep(5)
        else:
            raise RuntimeError('Shared TensorBoard did not list timing run')
    def save_state(batch):
        learner.save(out/'last.pkl', rng=rng.bit_generator.state, pending=batch,
            initial_chart_sha256=initial_digest, checkpoint_sha256=digest)
    export_chart(out/'last_chart.npz', learner); save_state(pending)
    evaluator = None
    try:
        evaluator = DistributedIsaacEvaluator(args, checkpoint, seed_clip)
        while learner.updates < cli.iterations:
            started = time.monotonic()
            if pending is None:
                panel = sampler.sample(cli.contexts, rng, minimum_plan_choices=2)
                pending = build_time_batch(panel, learner.current_chart(), ready, candidates=cli.candidates)
                pending['tiers'] = sampler.last_strata.copy()
                save_state(pending)
            planning_s = time.monotonic()-started
            # Check cached replay scorer exactly matches original chart scorer.
            scored, _ = timing_scores_and_jacobian(learner.parameters_s, pending['design'])
            np.testing.assert_allclose(scored, pending['scores'], rtol=1e-10, atol=1e-9)
            stamp = time.monotonic()
            def progress(offset, total, tick, ticks):
                write_json(out/'progress.json', dict(iteration=learner.updates+1,
                    phase=f'direct_time_top_{cli.candidates}_trials', episodes=total, tick=tick, ticks=ticks))
                print(f'[time-preferences] iteration={learner.updates+1} tick={tick}/{ticks}', flush=True)
            reports = evaluator.evaluate(pending['episodes'], progress)
            rollout_s = time.monotonic()-stamp
            rewards, hits, valid = unpack_time_reports(pending, reports)
            before = learner.parameters_s.copy()
            before_cov = learner.covariance.copy()
            metrics = learner.add_ranked_preferences(pending['design'], rewards, valid)
            stamp = time.monotonic(); metrics.update(learner.fit()); fit_s = time.monotonic()-stamp
            best_rewards, best_hits = np.nanmax(rewards, axis=1), np.nanmax(hits, axis=1)
            best_reward_ranks = np.argmax(np.where(valid, rewards, -np.inf), axis=1)+1
            metrics.update(iteration=learner.updates, contexts=cli.contexts, candidates=cli.candidates,
                physical_rollouts=len(reports), parameter_units='seconds',
                mean_first_hit_rate=float(hits[:, 0].mean()), mean_second_hit_rate=float(hits[:, 1].mean()),
                mean_first_reward=float(rewards[:, 0].mean()), mean_second_reward=float(rewards[:, 1].mean()),
                second_beats_first_fraction=float((rewards[:, 1] > rewards[:, 0]).mean()),
                mean_best_observed_hit_rate=float(best_hits.mean()),
                mean_best_observed_reward=float(best_rewards.mean()),
                selected_is_best_reward_fraction=float((rewards[:, 0] >= best_rewards-1e-12).mean()),
                selected_is_best_hits_fraction=float((hits[:, 0] >= best_hits).mean()),
                top_two_contains_best_reward_fraction=float((np.max(rewards[:, :2], axis=1) >= best_rewards-1e-12).mean()),
                mean_best_reward_predicted_rank=float(best_reward_ranks.mean()),
                mean_reward_regret=float((best_rewards-rewards[:, 0]).mean()),
                candidate_counts=valid.sum(axis=1).tolist(),
                mean_hit_rate_by_predicted_rank=[float(np.nanmean(hits[:, rank])) if valid[:, rank].any() else None
                    for rank in range(cli.candidates)],
                mean_reward_by_predicted_rank=[float(np.nanmean(rewards[:, rank])) if valid[:, rank].any() else None
                    for rank in range(cli.candidates)],
                planning_seconds=planning_s, rollout_seconds=rollout_s, fit_seconds=fit_s,
                iteration_seconds=time.monotonic()-started)
            metrics.update(planned_transition_coverage(pending['episodes'], len(learner.template.names)))
            np.savez_compressed(out/f'iteration_{learner.updates:05d}.npz',
                before_times_s=before, after_times_s=learner.parameters_s,
                before_covariance_s2=before_cov, after_covariance_s2=learner.covariance,
                actions=pending['actions'], rewards=rewards, hit_rates=hits, valid=valid,
                report_indices=pending.get('report_indices', np.arange(2*cli.contexts).reshape(2, cli.contexts).T),
                tiers=pending['tiers'],
                **pending['design'])
            write_json(out/f'iteration_{learner.updates:05d}.json', dict(metrics=metrics, reports=reports))
            pending = None
            export_chart(out/'last_chart.npz', learner); save_state(None)
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics)+'\n')
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar('time_preferences/'+key, value, learner.updates)
            writer.flush()
            write_json(out/'run.json', dict(status='running', completed_iterations=learner.updates,
                parameter_units='seconds', auxiliary_cost_layer=False))
            print('[time-preferences-result] '+json.dumps(metrics), flush=True)
        write_json(out/'run.json', dict(status='complete', completed_iterations=learner.updates,
            parameter_units='seconds', auxiliary_cost_layer=False))
    except BaseException as exc:
        write_json(out/'run.json', dict(status='failed', completed_iterations=learner.updates, error=repr(exc)))
        raise
    finally:
        writer.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    main()
