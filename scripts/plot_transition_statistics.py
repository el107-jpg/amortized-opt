#!/usr/bin/env python3
"""Render mean/median/p90 and standard deviation, optionally following a live table."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np


def plot_statistics(out, names, table, *, decoder_acceptance=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    statistics = [('mean_minimum_s', 'Mean'), ('median_minimum_s', 'Median'),
                  ('p90_minimum_s', 'P90')]
    fig, axes = plt.subplots(3, 2, figsize=(16, 20), layout='constrained')
    for row, (key, label) in enumerate(statistics):
        for hand in (0, 1):
            ax = axes[row, hand]
            values = table[key][hand]*1000
            image = ax.imshow(np.ma.masked_invalid(np.where(np.isfinite(values), values, np.nan)),
                              vmin=0, vmax=2000, cmap='viridis')
            ticks = [name.replace('_', ' ') for name in names]
            ax.set_xticks(range(len(names)), ticks, rotation=35, ha='right', fontsize=10)
            ax.set_yticks(range(len(names)), ticks, fontsize=10)
            ax.set_xlabel('Destination drum'); ax.set_ylabel('Source drum')
            ax.set_title(f'{label} — {("Left", "Right")[hand]} hand', fontsize=14)
            for src, dst in np.ndindex(values.shape):
                if not table['allowed'][hand, src, dst]:
                    text, color = 'excluded', '#777777'
                elif not np.isfinite(values[src, dst]):
                    text = ('unknown' if table['completed'][hand, src, dst] else
                            'searching' if table['trials'][hand, src, dst] else 'pending')
                    color = '#777777'
                else:
                    text = (f'{values[src, dst]:.0f} ms\n'
                            f'{table["support"][hand, src, dst]}/{table["trials"][hand, src, dst]} starts')
                    if not table['completed'][hand, src, dst]:
                        text += '\nsearching'
                    color = 'white' if values[src, dst] < 1000 else '#111111'
                ax.text(dst, src, text, ha='center', va='center', fontsize=8, color=color)
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Time (ms)', shrink=.85)
    qualifier = 'unrestricted waist' if decoder_acceptance is None else 'decoder only, '+decoder_acceptance+' constraints'
    fig.suptitle('Transition times across starting states — '+qualifier+'\n'
                 'Mean, median and P90 of accepted per-start durations; successful starts only\n'
                 'Cells show supported/valid starts; searching = provisional', fontsize=16)
    out = Path(out)
    for suffix in ('png', 'svg'):
        target = out/f'transition_statistics.{suffix}'
        temporary = out/f'transition_statistics.tmp.{suffix}'
        fig.savefig(temporary, dpi=160)
        temporary.replace(target)
    plt.close(fig)


def standard_deviation_table(out, table):
    """Support producers launched before standard deviation was added."""
    result = {key: table[key].copy() for key in ('allowed', 'completed', 'support', 'trials')}
    if 'std_minimum_s' in table:
        result['std_minimum_s'] = table['std_minimum_s'].copy()
        return result
    result['std_minimum_s'] = np.full(table['allowed'].shape, np.nan)
    for path in Path(out).glob('w*/h*_source_*.npz'):
        with np.load(path) as data:
            hand, src = int(data['hand']), int(data['source'])
            destinations = data['destination']
            minima = data['minimum_s']
            for dst in np.flatnonzero(result['allowed'][hand, src]):
                values = minima[destinations == dst]
                finite = values[np.isfinite(values)]
                result['support'][hand, src, dst] = len(finite)
                result['trials'][hand, src, dst] = len(values)
                result['completed'][hand, src, dst] = bool(data.get('completed', True))
                if len(finite) >= 2:
                    result['std_minimum_s'][hand, src, dst] = finite.std(ddof=1)
    return result


def plot_standard_deviation(out, names, table, *, decoder_acceptance=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), layout='constrained')
    for hand, ax in enumerate(axes):
        values = table['std_minimum_s'][hand]*1000
        image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1500, cmap='viridis')
        ticks = [name.replace('_', ' ') for name in names]
        ax.set_xticks(range(len(names)), ticks, rotation=35, ha='right', fontsize=10)
        ax.set_yticks(range(len(names)), ticks, fontsize=10)
        ax.set_xlabel('Destination drum'); ax.set_ylabel('Source drum')
        ax.set_title(f'Standard deviation — {("Left", "Right")[hand]} hand', fontsize=14)
        for src, dst in np.ndindex(values.shape):
            count = table['support'][hand, src, dst]
            total = table['trials'][hand, src, dst]
            done = table['completed'][hand, src, dst]
            if not table['allowed'][hand, src, dst]:
                text, color = 'excluded', '#777777'
            elif count == 1:
                text, color = f'need ≥2\n1/{total} starts', '#777777'
                if not done: text += '\nsearching'
            elif not np.isfinite(values[src, dst]):
                text = 'unknown' if done else 'searching' if total else 'pending'
                color = '#777777'
            else:
                text = f'{values[src, dst]:.0f} ms\n{count}/{total} starts'
                if not done: text += '\nsearching'
                color = 'white' if values[src, dst] < 750 else '#111111'
            ax.text(dst, src, text, ha='center', va='center', fontsize=8, color=color)
    fig.colorbar(image, ax=axes.tolist(), label='Sample standard deviation (ms)', shrink=.85)
    qualifier = 'unrestricted waist' if decoder_acceptance is None else 'decoder only, '+decoder_acceptance+' constraints'
    fig.suptitle('Variation across starting states — '+qualifier+'\n'
                 'Sample standard deviation of accepted per-start durations (ddof=1); successful starts only\n'
                 'At least two successful starts required; searching = provisional', fontsize=15)
    out = Path(out)
    for suffix in ('png', 'svg'):
        target = out/f'standard_deviation_times.{suffix}'
        temporary = out/f'standard_deviation_times.tmp.{suffix}'
        fig.savefig(temporary, dpi=160)
        temporary.replace(target)
    plt.close(fig)
    temporary = out/'transition_standard_deviations.tmp.csv'
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['hand', 'source', 'destination', 'std_minimum_s',
                         'successful_starts', 'valid_starts', 'search_complete', 'allowed'])
        for hand, src, dst in np.ndindex(table['allowed'].shape):
            value = table['std_minimum_s'][hand, src, dst]
            writer.writerow([('left', 'right')[hand], names[src], names[dst],
                f'{value:.6f}' if np.isfinite(value) else '', table['support'][hand, src, dst],
                table['trials'][hand, src, dst], bool(table['completed'][hand, src, dst]),
                bool(table['allowed'][hand, src, dst])])
    temporary.replace(out/'transition_standard_deviations.csv')


def render(out):
    with np.load(out/'transition_table.npz') as saved:
        names = saved['drum_names'].tolist()
        table = {key: saved[key].copy() for key in
                 ('allowed', 'completed', 'support', 'trials', 'mean_minimum_s',
                  'median_minimum_s', 'p90_minimum_s')}
        if 'std_minimum_s' in saved:
            table['std_minimum_s'] = saved['std_minimum_s'].copy()
    plot_statistics(out, names, table)
    plot_standard_deviation(out, names, standard_deviation_table(out, table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=10.)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.poll_seconds <= 0:
        parser.error('Polling interval must be positive')
    previous = None
    status_file = out/'statistics_chart_watch.json'
    def status(state):
        temporary = status_file.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(dict(status=state, pid=os.getpid(),
            chart=str(out/'transition_statistics.png'),
            standard_deviation_chart=str(out/'standard_deviation_times.png'),
            table_mtime_ns=previous), indent=2)+'\n')
        temporary.replace(status_file)
    try:
        while True:
            signature = (out/'transition_table.npz').stat().st_mtime_ns
            if signature != previous:
                render(out)
                previous = signature
                status('watching' if args.watch else 'rendered')
                print(f'Updated {out / "transition_statistics.png"}', flush=True)
            if not args.watch:
                return
            summary = json.loads((out/'summary.json').read_text())
            run = json.loads((out/'run.json').read_text())
            if summary['status'] == 'complete' or run['status'] == 'failed':
                # Render one final snapshot even if the producer completed
                # immediately after the table signature was read.
                render(out)
                status('complete' if summary['status'] == 'complete' else 'producer_failed')
                return
            time.sleep(args.poll_seconds)
    except BaseException:
        status('stopped')
        raise


if __name__ == '__main__':
    main()
