#!/usr/bin/env python3
"""Current conditional chart and its matched, per-drum Isaac test results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    run = json.loads((args.run/'config.json').read_text())
    test = json.loads((args.test/'config.json').read_text())
    results = json.loads((args.test/'summary.json').read_text())
    base = MinimumTimeChart.load(run['args']['table'])
    chart = ConditionalMinimumTimeChart.load(test['learned_overlay'], base)
    generation = chart.metadata['parameter_search']['generation']
    selection = chart.metadata['parameter_search']['selection']
    learned_label = 'CMA training chart' if selection in ('current_matched_training_panel', 'cma_population_mean') else 'Validation-selected chart'
    labels = [dict(snare='Snare', hihat='Hi-hat', tom1='Tom 1', tom2='Tom 2',
        cymbal_left='Cymbal L', cymbal_right='Cymbal R', floor_tom='Floor tom')[n] for n in chart.names]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    times = chart.anchors[..., 1]*1000
    vmax = np.ceil(times[chart.trainable_mask].max()/50)*50
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#d0d0d0')
    for h, hand in enumerate(('Left hand', 'Right hand')):
        ax = axes[0, h]
        im = ax.imshow(np.ma.masked_array(times[h], mask=~chart.trainable_mask[h]),
            cmap=cmap, vmin=0, vmax=vmax)
        for s in range(len(labels)):
            for d in range(len(labels)):
                if not chart.allowed[h, s, d]:
                    text, color = '—', '#555555'
                elif not chart.trainable_mask[h, s, d]:
                    text, color = '?', '#555555'
                else:
                    value = times[h, s, d]
                    text = f'{value:.0f}'+('*' if chart.initialized_unknown_mask[h, s, d] else '')
                    rgba = cmap(value/vmax)
                    color = 'white' if np.dot(rgba[:3], [.299, .587, .114]) < .48 else 'black'
                ax.text(d, s, text, ha='center', va='center', fontsize=10, color=color)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel('Destination drum'); ax.set_ylabel('Source drum')
        ax.set_title(hand+' · planning time at other-hand proxy 0')
    fig.colorbar(im, ax=axes[0], shrink=.8, label='Planning time (ms)')
    methods = results['methods']
    keys = ('initial_chart', 'learned_chart', 'graph')
    names = (test.get('initial_label', 'Starting chart'), learned_label, 'Original graph')
    colors = ('#66c2a5', '#3288bd', '#888888')
    ax = axes[1, 0]
    bars = ax.bar(names, [100*methods[k]['hit_rate'] for k in keys], color=colors)
    ax.set_ylim(0, 110); ax.set_ylabel('Credited hits / requested hits (%)')
    ax.set_title('Overall matched Isaac results')
    for bar, key in zip(bars, keys):
        metric = methods[key]
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
            f"{100*metric['hit_rate']:.1f}%\n{metric['hits']}/{metric['requested']}", ha='center')
    ax = axes[1, 1]
    positions = np.arange(len(labels))
    for offset, key, label, color in ((-.18, 'learned_chart', learned_label, colors[1]),
                                    (.18, 'graph', 'Original graph', colors[2])):
        records = methods[key]['by_drum']
        values = [100*records[n]['hits']/max(records[n]['requested'], 1) for n in chart.names]
        bars = ax.bar(positions+offset, values, width=.35, label=label, color=color)
        ax.bar_label(bars, fmt='%.0f', fontsize=8, padding=2)
    ax.set_xticks(positions, labels, rotation=35, ha='right')
    ax.set_ylim(0, 118); ax.set_ylabel('Credited hit rate (%)')
    ax.set_title('Hits by drum'); ax.legend(loc='upper right', fontsize=9)
    fig.suptitle(f"{learned_label} · 255 parameters · generation {generation} snapshot\n"
        f"Balanced recorded test: {results['scenarios']} windows · {results['unique_source_clips']} recordings", fontsize=15)
    fig.supxlabel('Heatmap values are planning times; * marks initialized unknown cells. '
        'Base measured support is preserved. Results use actual Isaac contact credit.', fontsize=9)
    out = args.out or args.test/'chart_and_results.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
