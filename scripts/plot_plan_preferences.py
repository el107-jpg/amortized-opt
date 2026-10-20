#!/usr/bin/env python3
"""Actual learned plan-cost corrections and posterior exploration geometry."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from isaac_rl.plan_preferences import PlanPreferenceLearner
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    learner, _ = PlanPreferenceLearner.load(args.run/'last.pkl')
    config = json.loads((args.run/'config.json').read_text())
    base = MinimumTimeChart.load(config['args']['table'])
    chart = ConditionalMinimumTimeChart.load(args.run/'base_chart.npz', base)
    mask = chart.trainable_mask
    mean = np.full((*mask.shape, 3), np.nan)
    std = mean.copy()
    mean[mask] = learner.mean.reshape(-1, 3)
    std[mask] = np.sqrt(np.diag(learner.covariance)).reshape(-1, 3)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    magnitude = max(float(np.abs(learner.mean).max()), .01)
    labels = ('Snare', 'Hi-hat', 'Tom1', 'Tom2', 'CymL', 'CymR', 'Floor')
    for h, hand in enumerate(('Left', 'Right')):
        for row, values, title in [(0, mean, 'Cost correction at proxy0'), (1, std, 'Posterior parameter standard deviation')]:
            ax = axes[row, h]
            if row == 0:
                im = ax.imshow(values[h, :, :, 1], cmap='RdBu_r',
                    norm=TwoSlopeNorm(vmin=-magnitude, vcenter=0, vmax=magnitude))
            else:
                im = ax.imshow(values[h, :, :, 1], cmap='viridis', vmin=0, vmax=learner.prior_std)
            ax.set_xticks(range(7), labels, rotation=45)
            ax.set_yticks(range(7), labels)
            ax.set_xlabel('Destination'); ax.set_ylabel('Source')
            ax.set_title(f'{hand} hand — {title}')
            fig.colorbar(im, ax=ax, shrink=.8)
    eigen = np.sqrt(np.linalg.eigvalsh(learner.covariance))/learner.prior_std
    axes[0, 2].plot(eigen)
    axes[0, 2].axhline(1, color='gray', linestyle='--', label='Initial uncertainty')
    axes[0, 2].set_xlabel('Covariance eigen-direction, ordered')
    axes[0, 2].set_ylabel('Posterior / initial uncertainty standard deviation')
    axes[0, 2].set_ylim(0, 1.05); axes[0, 2].legend()
    axes[0, 2].set_title('Correlated directions contract by observed information')
    ax = axes[1, 2]
    if learner.updates:
        record = np.load(args.run/f'iteration_{learner.updates:05d}.npz')
        differences = record['features']-record['features'][:, 0, None, :]
        before = np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', differences,
            record['before_covariance'], differences), 0))
        after = learner.uncertainty(differences)
        live = before > 1e-9
        ax.scatter(before[live], after[live], s=8, alpha=.3)
        maximum = float(before.max())
        ax.plot([0, maximum], [0, maximum], '--', color='gray')
        ax.set_xlabel('Plan-comparison uncertainty before update')
        ax.set_ylabel('Plan-comparison uncertainty after update')
        ax.set_title(f'Median retained uncertainty: {np.median(after[live]/before[live]):.1%}')
    fig.suptitle(f'Plan-preference learner — actual MAP update {learner.updates}; '
        'frozen timing chart + learned cost correction', fontsize=14)
    foot = ('Covariance is diagnostic; selection uses deterministic search ranks1/2.'
            if config.get('selection') == 'top_two' else 'Covariance guides alternative-plan exploration.')
    fig.text(.5, .01, 'Cost coefficients are not minimum times. Lower corrections favor a plan. '+foot,
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .95), h_pad=3., w_pad=2.)
    fig.savefig(args.out or args.run/'preferences.png', dpi=160)


if __name__ == '__main__':
    main()
