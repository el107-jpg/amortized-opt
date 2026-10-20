#!/usr/bin/env python3
"""Read-only audit of actual timing update coherence and observed preferences."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from isaac_rl.transition_time_preferences import (DESIGN_FIELDS,
    TransitionTimePreferenceLearner, timing_scores_and_jacobian)
from isaac_rl.plan_preferences import sigmoid


def cosine(a, b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(s) for s in (args.run/'metrics.jsonl').read_text().splitlines()]
    model, _ = TransitionTimePreferenceLearner.load(args.run/'last.pkl')
    iterations = [r['iteration'] for r in rows]
    snapshots = [dict(np.load(args.run/f'iteration_{i:05d}.npz')) for i in iterations]
    steps = np.stack([(s['after_times_s']-s['before_times_s'])*1000 for s in snapshots])
    total = steps.sum(0)
    active = np.abs(steps) >= 1.  # Ignore sub-ms sign chatter.
    adjacent = active[:-1] & active[1:]
    agrees = np.sign(steps[:-1]) == np.sign(steps[1:])
    coherence = np.divide(np.abs(total), np.abs(steps).sum(0),
        out=np.zeros_like(total), where=np.abs(steps).sum(0) > 0)
    changes = []
    monotonic, mixed = [], []
    for p in range(steps.shape[1]):
        signs = np.sign(steps[active[:, p], p])
        if len(signs) >= 2:
            (monotonic if np.all(signs == signs[0]) else mixed).append(p)
    for i, iteration in enumerate(iterations):
        changes.append(dict(iteration=iteration, rms_ms=float(np.sqrt(np.mean(steps[i]**2))),
            increases_ge_1ms=int((steps[i] >= 1).sum()), decreases_ge_1ms=int((steps[i] <= -1).sum()),
            previous_step_cosine=cosine(steps[i-1], steps[i]) if i else None,
            fraction_same_sign_as_previous_above_1ms=float(agrees[i-1][adjacent[i-1]].mean())
                if i and adjacent[i-1].any() else None))

    # Compare independent batch demands at a common parameter vector so
    # changing gradient evaluation points cannot manufacture disagreement.
    common = snapshots[0]['before_times_s']
    batch_gradients, tier_gradients, context_gradients, batch_signal = [], [], [], []
    for iteration, saved in zip(iterations, snapshots):
        design = {key: saved[key] for key in DESIGN_FIELDS}
        rewards, valid, tiers = saved['rewards'], saved['valid'], saved['tiers']
        scores, jacobian = timing_scores_and_jacobian(common, design)
        initial_scores, _ = timing_scores_and_jacobian(model.initial_s, design)
        gradients = np.zeros((len(rewards), len(common)))
        strong, weak_jacobian, wrong, correct, ties = 0, 0, 0, 0, 0
        for row, mask in enumerate(valid):
            ids = np.flatnonzero(mask)
            pair_count = len(ids)*(len(ids)-1)//2
            for i, a in enumerate(ids):
                for b in ids[i+1:]:
                    gap = rewards[row, a]-rewards[row, b]
                    target = sigmoid(gap/model.reward_temperature)
                    weight = max(model.indifference_weight, min(1., (abs(gap)-model.deadband)/model.gap_scale))/pair_count
                    denominator = max(model.score_scale, abs(initial_scores[row, b]-initial_scores[row, a]))*model.temperature
                    logit = (scores[row, b]-scores[row, a])/denominator
                    gradient = (jacobian[row, b]-jacobian[row, a])*model.prior_std_s/denominator
                    gradients[row] += weight*(sigmoid(logit)-target)*gradient
                    if abs(gap) > model.deadband:
                        strong += 1
                        weak_jacobian += np.linalg.norm(gradient) < 1e-6
                        alignment = gap*(scores[row, b]-scores[row, a])
                        if abs(scores[row, b]-scores[row, a]) < 1e-6:
                            ties += 1
                        elif alignment > 0:
                            correct += 1
                        else:
                            wrong += 1
        batch_gradients.append(gradients.sum(0))
        tier_gradients.append([gradients[tiers == tier].sum(0) for tier in range(3)])
        context_gradients.append(gradients)
        batch_signal.append(dict(iteration=iteration, strong_pairs=strong,
            weak_jacobian_fraction=float(weak_jacobian/strong),
            strong_pair_correct_fraction=float(correct/strong),
            strong_pair_wrong_fraction=float(wrong/strong),
            strong_pair_score_tie_fraction=float(ties/strong)))
    context_gradients = np.concatenate(context_gradients)
    positive = np.maximum(context_gradients, 0).sum(0)
    negative = np.maximum(-context_gradients, 0).sum(0)
    # Positive/negative mass measures opposition, not a statistical noise estimate.
    minority_mass_fraction = np.minimum(positive, negative).sum()/(positive+negative).sum()
    gradient_cosines = [[cosine(a, b) for b in batch_gradients] for a in batch_gradients]
    tier_sum = np.asarray(tier_gradients).sum(0)
    tier_cosines = [[cosine(a, b) for b in tier_sum] for a in tier_sum]
    mapping = model.template.parameter_mapping()
    most_reversing = np.argsort(coherence)[:12]
    examples = [dict(**mapping[p], updates_ms=steps[:, p].tolist(),
        net_ms=float(total[p]), coherence=float(coherence[p])) for p in most_reversing]
    summary = dict(completed_updates=iterations, sign_threshold_ms=1.,
        adjacent_update_sign_agreement_fraction=float(agrees[adjacent].mean()),
        adjacent_update_sign_reversal_fraction=float((~agrees[adjacent]).mean()),
        anchors_with_two_or_more_ge_1ms_steps=len(monotonic)+len(mixed),
        monotonic_anchors=len(monotonic), mixed_direction_anchors=len(mixed),
        median_anchor_net_over_total_travel=float(np.median(coherence)),
        aggregate_net_over_total_travel=float(np.abs(total).sum()/np.abs(steps).sum()),
        total_rms_ms=float(np.sqrt(np.mean(total**2))), changes=changes,
        batch_gradient_cosines_at_common_parameters=gradient_cosines,
        speed_tier_gradient_cosines_at_common_parameters=tier_cosines,
        opposed_context_gradient_minor_mass_fraction=float(minority_mass_fraction),
        batch_signal_at_common_parameters=batch_signal, examples=examples,
        interpretation='Observed reversals/conflicts do not prove chart capacity is insufficient; replay changes, nonlinear fitting, prior and simulator noise also contribute.')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(args.out/'directions.npz', iterations=iterations, steps_ms=steps,
        common_parameters_s=common, coherence=coherence, batch_gradients=batch_gradients,
        tier_gradients=tier_gradients)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    limit = max(1., float(np.abs(steps).max()))
    im = axes[0].imshow(steps, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[0].set_yticks(range(len(iterations)), iterations)
    axes[0].set_xlabel('Timing anchor index (3 per transition cell)')
    axes[0].set_ylabel('Absolute update'); axes[0].set_title('Actual changes: blue shorter, red longer')
    fig.colorbar(im, ax=axes[0], label='Change (ms)')
    im = axes[1].imshow(np.asarray(gradient_cosines, float), cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(iterations)), iterations)
    axes[1].set_yticks(range(len(iterations)), iterations)
    axes[1].set_title('Fresh-batch gradient alignment at the same chart')
    axes[1].set_xlabel('Batch/update'); axes[1].set_ylabel('Batch/update')
    fig.colorbar(im, ax=axes[1], label='Cosine: −1 opposite, +1 aligned')
    fig.suptitle(f'Timing update direction audit: {len(iterations)} completed top-eight updates')
    fig.savefig(args.out/'update_directions.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('examples',)}, indent=2))


if __name__ == '__main__':
    main()
