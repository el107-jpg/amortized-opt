#!/usr/bin/env python3
"""Read-only CMA variance audit and matched physical parameter-sensitivity probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.train_chart_cma import parse_args, setup_isaac, write_json
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler
from isaac_rl.min_time_assignment import receding_assignments
from amortized_opt.rhythm import DrumKit

TERMS = ('Missed-hit loss', 'Near-miss credit', 'Torque penalty', 'Joint penalty', 'Collision penalty')


def loss_terms(reports, args):
    # Constant 1 in missed-hit loss has no effect on variance or CMA rankings.
    return np.asarray([[1-r['hit_rate'], -r['near_miss_bonus'],
        args.torque_weight*r['torque_demand_excess_average_Nm']/5,
        args.joint_weight*r['joint_excess_average_rad']/np.deg2rad(5),
        args.collision_weight*r['measured_geometry_penetration_average_m']/.005]
        for r in reports])


def variance_summary(values):
    covariance = np.cov(values, rowvar=False)
    total = float(covariance.sum())
    return dict(loss_std=float(np.sqrt(total)), covariance=covariance.tolist(),
        terms=[dict(name=name, mean=float(values[:, i].mean()),
            std=float(values[:, i].std(ddof=1)),
            variance_share=float(covariance[i].sum()/total) if total else 0.)
            for i, name in enumerate(TERMS)])


def training_variance(run, optimizer, args):
    values, residuals, tiers = [], [], []
    generations = list(range(max(39, optimizer.generation-7), optimizer.generation+1))
    for generation in generations:
        record = json.loads((run/f'generation_{generation:05d}.json').read_text())
        terms = loss_terms(record['training_reports'], args)
        np.testing.assert_allclose(terms.sum(1),
            [1-r['reward'] for r in record['training_reports']], atol=1e-12)
        labels = np.asarray(record['speed_tiers'])
        centered = terms.copy()
        for tier in np.unique(labels):
            centered[labels == tier] -= terms[labels == tier].mean(0)
        values.append(terms)
        residuals.append(centered)
        tiers.append(labels)
    combined = np.concatenate(values)
    within = np.concatenate(residuals)
    overall = variance_summary(combined)
    conditional = variance_summary(within)
    return dict(generations=generations, episodes=len(combined), raw=overall,
        within_generation_and_speed_tier=conditional,
        variance_removed_by_tier_and_generation=1-conditional['loss_std']**2/overall['loss_std']**2,
        regularizer_note='Optimizer residual regularization is separate; its proposal standard deviation is reported in noise statistics.',
        variance_share_definition='Cov(term, total loss)/Var(total loss); shares sum to 1 and may be negative when terms cancel variation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rhythms', type=int, default=24)
    parser.add_argument('--directions', type=int, default=4)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--summarize-only', action='store_true')
    cli = parser.parse_args()
    snapshot = pickle.loads((cli.out/'snapshot.pkl').read_bytes())
    optimizer = snapshot['optimizer']
    args = parse_args(['--out', str(cli.out), '--resume', str(cli.out/'snapshot.pkl'), '--device', cli.device])
    settings = json.loads((cli.out/'config.json').read_text())
    rng = np.random.default_rng(cli.seed)
    stats = training_variance(cli.run, optimizer, args)
    scaling = np.broadcast_to(optimizer.es.sigma_vec.scaling, optimizer.initial.shape)
    covariance = optimizer.es.sigma**2*optimizer.es.sm.C*scaling[:, None]*scaling[None, :]
    std_ms = np.sqrt(np.diag(covariance))*optimizer.scale*1000
    pending = optimizer.pending
    noise = dict(sigma=float(optimizer.es.sigma), anchor_std_ms_percentiles=dict(zip(
        ('min', 'p25', 'median', 'p75', 'max'), map(float, np.percentile(std_ms, [0, 25, 50, 75, 100])))),
        covariance_diagonal_percentiles=np.percentile(np.diag(optimizer.es.sm.C), [0, 50, 100]).tolist(),
        median_std_ratio_to_initial_sigma=float(np.median(optimizer.es.stds/args.sigma)),
        proposal_regularization_std=float(np.std([optimizer.penalty(t) for t in pending])) if pending is not None else None)
    write_json(cli.out/'variance.json', stats)
    write_json(cli.out/'noise.json', noise)
    if cli.summarize_only:
        saved = np.load(cli.out/'probe_inputs.npz', allow_pickle=False)
        variants = [dict(alpha=float(a), theta=t) for a, t in zip(saved['alpha'], saved['theta'])]
        reports = json.loads((cli.out/'episodes.json').read_text())
        summarize_probe(cli, args, optimizer, stats, noise, saved['anchor_std_ms'], variants,
            saved['actions'], saved['gather'], saved['replica_ids'], saved['tiers'], reports)
        return
    mean_chart = optimizer.candidate_chart(optimizer.es.mean)
    sampler = ProceduralChartRhythmSampler(DrumKit.from_layout(), args.groups, mean_chart)
    panel = sampler.sample(cli.rhythms, rng)
    tiers = sampler.last_strata.copy()
    ready = np.load(args.ready_initial)
    variants = [dict(alpha=0., direction=-1, theta=optimizer.es.mean.copy())]
    factor = np.linalg.cholesky(covariance)
    attempts = 0
    for direction in range(cli.directions):
        for _ in range(2048):
            attempts += 1
            delta = factor@rng.normal(size=len(optimizer.initial))
            try:
                for sign in (-1., 1.):
                    optimizer.candidate_chart(optimizer.es.mean+sign*2*delta)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError('Could not obtain feasible matched two-sigma directions')
        for alpha in (.25, 1., 2.):
            for sign in (-1., 1.):
                variants.append(dict(alpha=alpha, direction=direction, sign=sign,
                    theta=optimizer.es.mean+sign*alpha*delta))
    bank, keys = [], {}
    gather = np.empty((len(variants), len(panel)), int)
    actions = np.empty((len(variants), len(panel), args.groups), int)
    started = time.monotonic()
    for v, variant in enumerate(variants):
        chart = optimizer.candidate_chart(variant['theta'])
        for r, clip in enumerate(panel):
            selected = receding_assignments(clip, chart, args.horizon, ready)[0]
            actions[v, r] = selected
            key = (r, tuple(selected.tolist()))
            if key not in keys:
                keys[key] = len(bank)
                bank.append(assigned_clip(clip, selected))
            gather[v, r] = keys[key]
    # Shared schedules reuse one measured mean rather than introducing fake
    # parameter response from repeat-to-repeat contact variation. Each unique
    # schedule gets physical replicas in shuffled scene slots to estimate it.
    replica_pairs = [(b, k) for b in range(len(bank)) for k in range(cli.replicates)]
    rng.shuffle(replica_pairs)
    episodes = [bank[b] for b, _ in replica_pairs]
    replica_ids = np.empty((len(bank), cli.replicates), int)
    for index, (b, k) in enumerate(replica_pairs):
        replica_ids[b, k] = index
    args.envs = len(episodes)
    args.seed = cli.seed
    seed_clip = cli.out/'scene_seed.json'
    write_json(seed_clip, sampler.seed_payload())
    write_json(cli.out/'probe_config.json', dict(snapshot_generation=optimizer.generation,
        logical_variants=len(variants), rhythms=len(panel), unique_schedules=len(bank),
        physical_episodes=len(episodes), replicates=cli.replicates, planning_seconds=time.monotonic()-started,
        physical_randomization=False, physics_hz=args.physics_hz, directions_attempted=attempts,
        chart_selection='current_CMA_mean', variant_sampling='feasible_antithetic_directions_at_0.25_1_2_times_current_covariance',
        reward_feedback_to_optimizer=False))
    np.savez_compressed(cli.out/'probe_inputs.npz', theta=np.stack([v['theta'] for v in variants]),
        alpha=np.array([v['alpha'] for v in variants]), actions=actions, gather=gather,
        replica_ids=replica_ids, tiers=tiers, anchor_std_ms=std_ms)
    print(f'[signal] {len(variants)} variants on {len(panel)} matched rhythms; {len(bank)} unique schedules, {len(episodes)} physical episodes', flush=True)
    import torch
    checkpoint = Path(settings['checkpoint'])
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    launcher = env = None
    try:
        launcher, env, evaluator = setup_isaac(args, checkpoint, source, seed_clip)
        def progress(offset, total, tick, ticks):
            write_json(cli.out/'progress.json', dict(phase='matched_physics', tick=tick, ticks=ticks))
            print(f'[signal] physics tick={tick}/{ticks}', flush=True)
        reports = evaluator.evaluate(episodes, progress)
        # Persist before Isaac app shutdown, which may terminate the process.
        write_json(cli.out/'episodes.json', reports)
        summarize_probe(cli, args, optimizer, stats, noise, std_ms, variants, actions, gather, replica_ids, tiers, reports)
    finally:
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()


def summarize_probe(cli, args, optimizer, stats, noise, std_ms, variants, actions, gather, replica_ids, tiers, reports):
    write_json(cli.out/'episodes.json', reports)
    measured = loss_terms(reports, args).sum(1)
    repeated = measured[replica_ids]
    physical_mean = repeated.mean(1)
    observed = physical_mean[gather]
    regularizer = np.array([optimizer.penalty(v['theta']) for v in variants])
    losses = observed+regularizer[:, None]
    delta = losses-losses[0]
    bootstrap_rng = np.random.default_rng(cli.seed+1)
    case_resamples = np.concatenate([bootstrap_rng.choice(np.flatnonzero(tiers == tier),
        size=(2000, int((tiers == tier).sum())), replace=True) for tier in range(3)], axis=1)
    summaries = []
    for alpha in (.25, 1., 2.):
        ids = np.flatnonzero(np.array([v['alpha'] for v in variants]) == alpha)
        changed = actions[ids] != actions[0]
        summaries.append(dict(noise_multiplier=alpha,
            changed_groups_fraction=float(changed.mean()), changed_rhythms_fraction=float(changed.any(-1).mean()),
            mean_loss_change=float(delta[ids].mean()),
            per_rhythm_loss_change_rms=float(np.sqrt(np.mean(delta[ids]**2))),
            direction_mean_loss_changes=delta[ids].mean(1).tolist(),
            direction_mean_loss_ci95=np.percentile(delta[ids][:, case_resamples].mean(-1),
                [2.5, 97.5], axis=1).T.tolist(),
            parameter_conditional_std=float(np.sqrt(np.mean(np.var(observed[ids], axis=0, ddof=1))))))
    baseline_replica_loss = repeated[gather[0]]
    simulation_std = float(np.sqrt(np.mean(np.var(baseline_replica_loss, axis=1, ddof=1))))
    result = dict(snapshot_generation=optimizer.generation, training_variance=stats, noise=noise,
        sensitivity=summaries, baseline_physical_repeat_std=simulation_std,
        baseline_physical_repeat_hits=[[reports[i]['hits'] for i in replica_ids[b]] for b in gather[0]],
        repeat_note='Replicas occupy distinct nominal scene slots; this includes scene-origin/GPU contact numerical variation, not randomized dynamics.',
        plateau_note='Chart parameters only select discrete hand assignments; identical hand sequences share identical measured outcomes in this audit. Near-miss credit is smooth in tip distance, not through the hand argmin.',
        baseline_probe_loss_std=float(losses[0].std(ddof=1)),
        baseline_probe_within_tier_loss_std=float(np.sqrt(np.mean([np.var(losses[0, tiers==t], ddof=1) for t in range(3)]))),
        unique_schedules=len(replica_ids), physical_episodes=len(reports), rhythms=len(tiers),
        replicates=cli.replicates,
        uncertainty_note='Confidence intervals resample matched rhythms within speed tiers; physical-repeat uncertainty is reported separately. Four directions are conditioned to remain feasible at both signs of two-sigma noise.',
        no_optimizer_or_training_feedback=True)
    write_json(cli.out/'summary.json', result)
    np.savez_compressed(cli.out/'probe_outputs.npz', losses=losses, repeated_loss=repeated,
        delta=delta, alpha=np.array([v['alpha'] for v in variants]))
    plot_report(cli.out, result, std_ms, delta, variants)
    write_json(cli.out/'progress.json', dict(phase='complete'))
    print('[signal-result] '+json.dumps(dict(sensitivity=summaries,physical_repeat_std=simulation_std)), flush=True)


def plot_report(out, summary, std_ms, delta, variants):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    rows = summary['training_variance']['within_generation_and_speed_tier']['terms']
    shares = [r['variance_share']*100 for r in rows]
    ax = axs[0, 0]
    ax.barh(TERMS, shares, color=['#417ab5' if x >= 0 else '#d89242' for x in shares])
    ax.axvline(0, color='gray', linewidth=.8)
    ax.invert_yaxis()
    for i, x in enumerate(shares):
        ax.text(x+1, i, f'{x:.1f}%', va='center')
    ax.set_xlim(min(shares)-9, max(shares)+15)
    ax.set_xlabel('Cov(term, total) / Var(total), %')
    ax.set_title('Loss variance after centering each speed tier / generation')
    ax = axs[0, 1]
    ax.hist(std_ms, bins=25, color='#417ab5')
    ax.axvline(np.median(std_ms), color='#c86840', label=f'Median {np.median(std_ms):.1f} ms')
    ax.set_xlabel('Current sampling standard deviation of timing anchors, ms')
    ax.set_ylabel('Parameters')
    ax.legend()
    ax.set_title(f'Exploration over {len(std_ms)} parameters')
    ax = axs[1, 0]
    s = summary['sensitivity']
    x = [r['noise_multiplier'] for r in s]
    ax.plot(x, [r['changed_groups_fraction']*100 for r in s], 'o-', label='Hit groups changed')
    ax.plot(x, [r['changed_rhythms_fraction']*100 for r in s], 'o-', label='Rhythms changed')
    ax.set_xlabel('Noise multiplier relative to current CMA distribution')
    ax.set_ylabel('Hand assignments changed, %')
    ax.legend()
    ax.set_title('Discrete response on the same rhythms')
    ax = axs[1, 1]
    alpha = np.array([v['alpha'] for v in variants])
    for a in x:
        values = delta[alpha == a].mean(1)
        ax.scatter(np.full(len(values), a), values, color='#417ab5', alpha=.75)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set_xlabel('Noise multiplier')
    ax.set_ylabel('Paired mean loss change (lower is better)')
    ax.set_title('One point per antithetic parameter perturbation')
    fig.suptitle(f'CMA signal audit — actual mean at generation {summary["snapshot_generation"]}; '
        f'{summary["rhythms"]} fixed random rhythms, {summary["replicates"]} physical replicas per unique schedule', fontsize=13)
    fig.text(.5, .01, 'Diagnostic only. No validation selection or optimizer feedback. '
        'Negative variance shares represent cancellation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(out/'loss_signal.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
