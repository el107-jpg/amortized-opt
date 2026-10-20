#!/usr/bin/env python3
"""Publish transition tables from a complete decoder audit and finished QP shards.

The database remains useful while the QP sweep runs: keep every certified raw
decoder witness, then add shorter/new repaired witnesses as they are written.
Only complete, atomic worker files are consumed. No trajectory generation or
training is performed here.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from isaac_rl.transition_assignment import TransitionTable, minimum_statistics


def combine_minima(table, keys, eligible, deadlines, success, repairs):
    """Take one successful minimum per admissible start, then the group mean."""
    keys = np.asarray(keys, int)
    eligible = np.asarray(eligible, bool)
    deadlines = np.asarray(deadlines, float)
    success = np.asarray(success, bool)
    shape = (*table.minimum_s.shape, int(table.metadata['samples']))
    if (keys.shape != (len(eligible), 5) or success.shape != (len(keys), len(deadlines))
            or len(set(map(tuple, keys))) != len(keys)
            or (keys < 0).any() or (keys >= np.asarray(shape)).any()
            or not np.isfinite(deadlines).all() or (deadlines <= 0).any()
            or len(np.unique(np.rint(deadlines/.02))) != len(deadlines)):
        raise ValueError('Invalid/duplicate keys or deadline dimensions')
    allowed_groups = table.allowed_groups()
    eligible = eligible & allowed_groups[tuple(keys[:, :4].T)]
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    raw = np.where(success & eligible[:, None], deadlines[None], np.inf).min(-1)
    repaired = raw.copy()
    coverage = np.zeros_like(success)
    mismatches = 0
    for deadline, rows in repairs:
        columns = np.flatnonzero(np.isclose(deadlines, deadline))
        if len(columns) != 1: raise ValueError('QP deadline outside baseline grid')
        col = int(columns[0])
        if len(set(map(tuple, rows['keys']))) != len(rows['keys']):
            raise ValueError('Duplicate keys in QP shard')
        for j, key in enumerate(rows['keys']):
            i = lookup[tuple(key)]
            if coverage[i, col]: raise ValueError('Overlapping QP shards')
            coverage[i, col] = True
            mismatches += int(bool(rows['before_success'][j]) != bool(success[i, col]))
            if eligible[i] and rows['after_success'][j]: repaired[i] = min(repaired[i], deadline)
    grids = []
    index = tuple(keys.T)
    admissible = np.zeros(shape, bool); admissible[index] = eligible
    for minima in (raw, repaired):
        grid = np.full(shape, np.inf); grid[index] = minima
        grids.append(grid)
    stats = [minimum_statistics(g[..., None], np.isfinite(g[..., None])) for g in grids]
    # Denominator includes valid, within-bound starts, including every failed
    # search. Excluded invalid starts remain visible in the CSV/summary.
    trials = admissible.sum(-1)
    return dict(raw_minima=grids[0], repaired_minima=grids[1], eligible=admissible,
        allowed_groups=allowed_groups,
        raw_stats=stats[0], repaired_stats=stats[1], trials=trials,
        coverage=coverage, baseline_acceptance_mismatches=mismatches)


def read_inputs(audit, qp):
    config = json.loads((qp/'config.json').read_text())
    if Path(config['comparison_audit']).resolve() != audit.resolve():
        raise ValueError('QP run references a different audit')
    table = TransitionTable.load(audit/'transition_table.npz')
    table.metadata['forbidden_drums_by_hand'] = {'left': ['floor_tom'], 'right': []}
    for field in ('checkpoint', 'checkpoint_step'):
        if config[field] != table.metadata[field]: raise ValueError(f'Different {field}')
    for field in ('hit_depth_mm', 'impact_speed_required', 'allow_early_target_top_entry', 'collocation_substeps'):
        if config[field] != table.metadata[field]: raise ValueError(f'Different {field}')
    if config['impact_speed_required']:
        raise ValueError('This QP experiment/database uses the no-required-impact-speed calibration')
    with np.load(audit/'deadline_audit.npz') as data:
        base = {name: data[name].copy() for name in ('keys', 'valid_start', 'deadlines_s', 'success')}
    # The completed one-second QP cohort records bounds using the exact
    # checkpoint cost (including tightened waist bounds), for all valid starts.
    with np.load(qp/'deadline_050.npz') as data:
        flags = {tuple(key): bool(flag) for key, flag in zip(data['keys'], data['departure_posture_valid'])}
    if any(tuple(k) not in flags for k in base['keys'][base['valid_start']]):
        raise ValueError('One-second cohort does not cover every geometry-valid start')
    base['eligible'] = base['valid_start'] & np.array([flags.get(tuple(k), False) for k in base['keys']])
    return table, base, config


def finished_shards(qp):
    return sorted(p for p in qp.glob('w*/deadline_*.npz')
                  if p.with_suffix('.json').exists())


def accepted_gates(rows):
    """Check each repaired witness against the recorded nonlinear diagnostics."""
    names = ('terminal_region_dist', 'collision_min_sdf', 'torque_excess_max',
             'velocity_excess_max', 'joint_box_excess_max', 'contact_dir_cos', 'stick_elev_deg')
    values = {n: rows['after_'+n] for n in names}
    ok = np.logical_and.reduce([np.isfinite(v) for v in values.values()])
    ok &= (values['terminal_region_dist'] <= 1e-6) & (values['collision_min_sdf'] >= -1e-6)
    for n in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max'): ok &= values[n] <= 1e-6
    ok &= (values['contact_dir_cos'] >= .5) & (values['stick_elev_deg'] >= 5.) & (values['stick_elev_deg'] <= 45.)
    if (rows['after_success'] & ~ok).any(): raise ValueError('QP success contradicts physical diagnostics')


def publish(audit, qp, out, table, base, config, paths):
    repairs = []
    for path in paths:
        with np.load(path) as d: rows = {k: d[k].copy() for k in d.files}
        accepted_gates(rows)
        deadline = int(path.stem.rsplit('_', 1)[1])*.02
        repairs.append((deadline, rows))
    result = combine_minima(table, base['keys'], base['eligible'], base['deadlines_s'], base['success'], repairs)
    covered = result['coverage'][base['valid_start']]
    complete = bool(covered.all())
    raw, repaired = result['raw_minima'], result['repaired_minima']
    summary = dict(status='complete' if complete else 'provisional',
        tested_deadlines_s=base['deadlines_s'].tolist(),
        minimum_time_search=len(base['deadlines_s']) > 1,
        groups=int(result['allowed_groups'].sum()),
        sampled_groups=int(table.minimum_s.size),
        excluded_groups=int((~result['allowed_groups']).sum()),
        forbidden_drums_by_hand=table.metadata.get('forbidden_drums_by_hand', {}),
        sampled_start_records=len(base['keys']),
        geometry_valid_start_records=int(base['valid_start'].sum()),
        physically_admissible_start_records=int(base['eligible'].sum()),
        admissible_start_records=int(result['eligible'].sum()),
        decoder_trials=int(base['valid_start'].sum()*len(base['deadlines_s'])),
        completed_qp_trials=int(covered.sum()), expected_qp_trials=int(covered.size),
        complete_qp_deadlines_s=base['deadlines_s'][covered.all(0)].tolist(),
        decoder_supported_starts=int(np.isfinite(raw).sum()),
        qp_supported_starts=int(np.isfinite(repaired).sum()),
        decoder_supported_groups=int((result['raw_stats']['count'] > 0).sum()),
        qp_supported_groups=int((result['repaired_stats']['count'] > 0).sum()),
        newly_supported_starts=int((np.isfinite(repaired) & ~np.isfinite(raw)).sum()),
        selected_minima_requiring_qp=int((repaired < raw).sum()),
        baseline_acceptance_mismatches=result['baseline_acceptance_mismatches'])
    manifest = [(str(p.relative_to(qp)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    fingerprint = hashlib.sha256(json.dumps(dict(shards=manifest,
        forbidden_drums_by_hand=summary['forbidden_drums_by_hand'])).encode()).hexdigest()[:16]
    final_snapshot = out/'snapshots'/fingerprint
    if final_snapshot.exists():
        return json.loads((final_snapshot/'summary.json').read_text())
    snapshot = out/'snapshots'/('.building_'+fingerprint)
    snapshot.mkdir(parents=True, exist_ok=True)
    meta = dict(config, database_kind=('Feasibility at one fixed deadline' if len(base['deadlines_s']) == 1
        else 'Mean of per-start minimum certified deadlines'),
        timing_interpretation=('Accepted fixed deadline, not an estimated minimum transition time'
            if len(base['deadlines_s']) == 1 else 'Mean of minimum accepted deadlines per supported start'),
        database_status=summary['status'], source_audit=str(audit), qp_run=str(qp),
        eligible_departures='Geometry-valid and within checkpoint posture bounds',
        forbidden_drums_by_hand=summary['forbidden_drums_by_hand'],
        support_denominator='All eligible starts, including starts with no accepted duration',
        sampled_starts_per_group=int(table.metadata['samples']),
        yaw_interpretation='Initial yaw region; QP can modify waist trajectory',
        runtime_note='A repaired minimum may require the same QP refinement at execution',
        qp_coverage=summary, pair_yaw_source=str(audit/'transition_table.npz'),
        pair_yaw_recalibrated=False)
    for name, stat in (('decoder', result['raw_stats']), ('qp', result['repaired_stats'])):
        current = replace(table, minimum_s=stat['mean'], support=stat['count'],
            trials=result['trials'], metadata=dict(meta, path_generation=(
                'One frozen decoder pass' if name == 'decoder' else 'One decoder pass, with up to five local QP steps when needed')))
        current.save(snapshot/f'transition_table_{name}.npz')
    np.savez_compressed(snapshot/'per_start_minima.npz', decoder_minimum_s=raw,
        qp_minimum_s=repaired, eligible=result['eligible'], allowed_groups=result['allowed_groups'], keys=base['keys'],
        qp_coverage=result['coverage'], deadlines_s=base['deadlines_s'],
        decoder_std_s=result['raw_stats']['std'], qp_std_s=result['repaired_stats']['std'])
    with (snapshot/'transition_table.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['hand', 'source', 'destination', 'initial_yaw_rad',
            'decoder_mean_min_s', 'qp_mean_min_s', 'qp_std_s', 'decoder_supported_starts',
            'qp_supported_starts', 'eligible_starts', 'sampled_starts'])
        finite = lambda x: float(x) if np.isfinite(x) else ''
        for h, src, dst, y in np.ndindex(table.minimum_s.shape):
            k = h, src, dst, y
            if not result['allowed_groups'][k]: continue
            w.writerow([('left', 'right')[h], table.drum_names[src], table.drum_names[dst],
                table.yaw_centers[y], finite(result['raw_stats']['mean'][k]),
                finite(result['repaired_stats']['mean'][k]), result['repaired_stats']['std'][k],
                result['raw_stats']['count'][k], result['repaired_stats']['count'][k],
                result['trials'][k], table.metadata['samples']])
    with (snapshot/'per_start_minima.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['hand', 'source', 'destination', 'initial_yaw_rad',
            'start_index', 'eligible', 'excluded_by_hand_policy', 'decoder_min_s', 'qp_min_s', 'qp_required_for_minimum'])
        for h, src, dst, y, s in map(tuple, base['keys']):
            k = h, src, dst, y, s
            w.writerow([('left', 'right')[h], table.drum_names[src], table.drum_names[dst],
                table.yaw_centers[y], s, result['eligible'][k], not result['allowed_groups'][k[:4]],
                finite(raw[k]), finite(repaired[k]), repaired[k] < raw[k]])
    (snapshot/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (snapshot/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    snapshot.rename(final_snapshot)
    snapshot = final_snapshot
    # Readers see a complete immutable snapshot when this single link changes.
    temporary = out/'.current.tmp'
    temporary.symlink_to(snapshot.relative_to(out), target_is_directory=True)
    os.replace(temporary, out/'current')
    for name in ('transition_table_qp.npz', 'transition_table_decoder.npz', 'transition_table.csv',
                 'per_start_minima.csv', 'per_start_minima.npz', 'summary.json'):
        alias = out/name
        if not alias.is_symlink(): alias.symlink_to(Path('current')/name)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=ROOT/'runs/transition_table_full_range_top_entry_v6_full_idle_20261004/audit')
    p.add_argument('--qp-run', type=Path, default=ROOT/'runs/transition_qp_five_full_range_20261004')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    audit, qp, out = args.audit.resolve(), args.qp_run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # One publisher per output directory; the expensive QP job is independent.
    import fcntl
    lock = (out/'publisher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    table, base, config = read_inputs(audit, qp)
    previous = None
    while True:
        paths = finished_shards(qp)
        signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
        if signature != previous:
            summary = publish(audit, qp, out, table, base, config, paths)
            previous = signature
            if summary['status'] == 'complete': break
        if not args.watch: break
        time.sleep(15)


if __name__ == '__main__': main()
