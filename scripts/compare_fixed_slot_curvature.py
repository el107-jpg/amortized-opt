#!/usr/bin/env python3
"""Compare completed local sweeps without treating combined fixes as causal isolation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from isaac_rl.hand_plan_sweep import AXES
from scripts.train_chart_cma import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.old / 'summary.json').read_text())
    new = json.loads((args.new / 'summary.json').read_text())
    config = json.loads((args.new / 'config.json').read_text())
    if old['status'] != 'complete' or new['status'] != 'complete':
        raise ValueError('Both sweeps must be complete')
    source = torch.load(Path(config['original_sweep']) / 'plans.pt', map_location='cpu', weights_only=False)
    originals = {(p['case'], p['axis'], p['step']): p for p in source['points'] if p['axis'] in AXES}
    controls = {p['case']: p for p in source['points'] if p['axis'] == 'control' and p['repeat'] == 0}
    points = json.loads((args.new / 'points.json').read_text())
    reports = json.loads((args.new / 'episodes.json').read_text())
    qerr = qderr = tiperr = 0.
    checked = 0
    for point in points:
        if not point['valid']:
            continue
        original = controls[point['case']] if point['axis'] == 'control' else originals[(point['case'], point['axis'], point['step'])]
        for action, report_id in zip(point['actions'], point['report_ids']):
            source_id = original['report_ids'][original['actions'].index(action)]
            expected = source['episode_states'][source_id]
            actual = reports[report_id]['initial_state']
            qerr = max(qerr, float(np.abs(np.array(actual['q']) - expected['q']).max()))
            qderr = max(qderr, float(np.abs(np.array(actual['qd']) - expected['qd']).max()))
            tiperr = max(tiperr, float(np.linalg.norm(np.array(actual['tips']) - expected['tips'], axis=1).max()))
            checked += 1
    metrics = ('fraction_hit_credit_changed', 'fraction_one_point_hit_credit_reversals', 'reward_difference_p95')
    comparison = dict(axes={axis: {metric: dict(old=old['axes'][axis][metric], new=new['axes'][axis][metric])
        for metric in metrics} for axis in AXES},
        initial_state_parity=dict(checked_rollouts=checked, max_q_error_rad=qerr,
            max_qd_error_rad_s=qderr, max_tip_error_m=tiperr),
        scene_controls=json.loads((args.new / 'scene_repeatability.json').read_text()),
        limitation='Eight original cases. Matched impact-speed grading and fresh scenes both changed; this comparison does not isolate either change alone.')
    comparison['zero_offset_replay'] = json.loads((args.new / 'zero_offset_parity.json').read_text())
    write_json(args.new / 'comparison.json', comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(AXES))
    for ax, metric, label in zip(axes, metrics[:2], ['Neighbors changing hit credit (%)', 'One-point hit-credit reversals (%)']):
        ax.bar(x - .18, [100 * old['axes'][a][metric] for a in AXES], .36, label='Earlier: reused scenes, old speed grading')
        ax.bar(x + .18, [100 * new['axes'][a][metric] for a in AXES], .36, label='Now: fresh scenes, matched speed')
        ax.set_xticks(x, [a.replace('_', '\n') for a in AXES])
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Same rhythms, starting states, assignments and fixed slots\n1 ms timing / 0.1 degree pose increments; combined changes')
    fig.savefig(args.new / 'before_after_summary.png', dpi=160)
    plt.close(fig)

    def example_axes(axes, panel_points, fresh):
        rows = sorted([p for p in panel_points if p['case'] == 3 and p['axis'] == 'first_delay' and p['valid']], key=lambda p: p['step'])
        x = [p['offset_display'] for p in rows]
        rewards = [p['reward'][p['actions'].index(15)] for p in rows]
        hits = [p['hits'][p['actions'].index(15)] for p in rows]
        scores = [p['neural_scores'][p['actions'].index(15)] - np.mean(p['neural_scores']) for p in rows]
        axes[0].plot(x, rewards, 'o-', ms=3, label='Measured reward')
        controls = [p for p in panel_points if p['case'] == 3 and p['axis'] == 'control'
            and p.get('control_axis', 'first_delay') == 'first_delay'
            and (not fresh or p.get('control_kind') == 'fresh_control' or p.get('repeat') == 0)]
        repeated = [p['reward'][p['actions'].index(15)] for p in controls]
        axes[0].scatter(np.zeros(len(repeated)), repeated, marker='x', c='tab:red', s=75,
            label='Fresh-scene repeats' if fresh else 'Unchanged-case repeats', zorder=5)
        axes[0].legend(fontsize=8)
        axes[1].step(x, hits, where='mid')
        axes[2].plot(x, scores)
        for ax, label in zip(axes, ['Physical reward', 'Successful hits', 'Centered neural score (unitless)']):
            ax.set_xlabel('First-hit delay offset (ms)')
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        return hits

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    example_axes(axes, points, True)
    fig.suptitle('One fixed assignment: local sweep and fresh-scene repeats (case 3, plan 1111)\nFresh scene at every point; matched decoder impact speed; 250 Hz physics')
    fig.savefig(args.new / 'continuity_example.png', dpi=160)
    plt.close(fig)
    legacy_points = json.loads((Path(config['original_sweep']) / 'points.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col', sharey='col', constrained_layout=True)
    example_axes(axes[0], legacy_points, False)
    example_axes(axes[1], points, True)
    axes[0, 0].set_title('Original diagnostic')
    axes[1, 0].set_title('Now: fresh scenes + corrected reset and speed grading')
    fig.suptitle('Same case 3 / plan 1111 / first-delay offsets; combined runtime changes')
    fig.savefig(args.new / 'continuity_before_after.png', dpi=160)
    plt.close(fig)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
