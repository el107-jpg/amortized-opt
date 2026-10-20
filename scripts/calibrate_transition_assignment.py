#!/usr/bin/env python3
"""Calibrate frozen decoder transitions and probe exhaustive table assignment.

No optimizer or network training. Random searches cover all 392 hand/source/
destination/yaw groups, including same-drum recovery and unrestricted hands.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator, TrajectorySurrogate
from isaac_rl.rnea_cache import rnea_from_cache_if_needed
from isaac_rl.transition_assignment import TransitionTable, AssignmentScoreConfig, minimum_statistics, search_assignments
from isaac_rl.transition_calibration import (near_head_starts, random_durations, evaluate_trials,
                                            calibrate_pair_yaws, descending_deadlines, full_range_deadlines)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


@torch.no_grad()
def live_initial_times(generator, q, v, yaws, rng, trials, chunk, duration_search='descending', hit_depth_m=None,
                       require_hit_speed=True, allow_early_target_top_entry=False, ignore_idle_arm=False):
    """First edges use actual measured q/qd, not nearest-drum substitution."""
    D, Y = generator.kit.num_drums, len(yaws)
    estimates = np.full((2, D, Y), np.inf)
    for hand in range(2):
        target = np.repeat(np.arange(D), Y)
        yaw = np.tile(yaws, D)
        grid = full_range_deadlines() if duration_search == 'full-range' else descending_deadlines()
        durations = (np.broadcast_to(grid, (len(target), len(grid)))
                     if duration_search in ('descending', 'full-range') else random_durations(rng, len(target), trials))
        out = evaluate_trials(generator, q.expand(len(target), -1), v.expand(len(target), -1),
            hand, np.full(len(target), -1), target, yaw, durations, chunk=chunk, hit_depth_m=hit_depth_m,
            require_hit_speed=require_hit_speed, allow_early_target_top_entry=allow_early_target_top_entry,
            ignore_idle_arm=ignore_idle_arm)
        minima = np.where(out['success'], durations, np.inf).min(-1)
        estimates[hand] = minima.reshape(D, Y)
    return estimates


@torch.no_grad()
def probe(generator, table, run, rng, windows, trials, chunk, duration_search='descending', hit_depth_m=None,
          require_hit_speed=True, allow_early_target_top_entry=False):
    """Fresh procedural windows, identical initial state and paired bounce draws.

    Runtime feasibility comparison uses the existing endpoint/Hermite analytic
    environment. Calibration above always uses actual decoder paths.
    """
    from isaac_rl.random_rhythms import RandomRhythmSampler
    from isaac_rl.schedule_improvement import physical_table, graph_seed, MAX_HITS
    from isaac_rl.schedule_analytic import AnalyticScheduleEnvironment
    kit = generator.kit
    q = torch.as_tensor(ready_joint_pos(), device=generator.cost.device, dtype=torch.float32)[None]
    v = torch.zeros_like(q)
    initial = live_initial_times(generator, q, v, table.yaw_centers, rng, trials, chunk, duration_search,
                                 hit_depth_m, require_hit_speed, allow_early_target_top_entry)
    np.save(run/'ready_initial_minimum_s.npy', initial)
    clips = RandomRhythmSampler(kit).sample(rng, windows)
    actions, graph, details, runtimes = [], [], [], []
    for clip in clips:
        begin = time.perf_counter()
        best, candidates = search_assignments(clip, table, yaw_now=float(q[0, 0]),
            initial_minimum_s=initial)
        runtimes.append((time.perf_counter()-begin)*1000)
        a = np.zeros(MAX_HITS, int); a[:len(best['actions'])] = best['actions']
        actions.append(a); graph.append(graph_seed(clip, kit)[0])
        details.append(dict(rhythm=clip.name, hits=[dict(time=h.time, drum=h.drum, name=h.drum_name)
                                                  for h in clip.hits],
                            candidates=len(candidates), selected=best))
    physical = physical_table(clips, kit, q.device)
    candidates = torch.as_tensor(np.stack((np.asarray(graph), np.asarray(actions)), 1), device=q.device)
    draws = torch.as_tensor(rng.uniform(.2, .8, (windows, 2, MAX_HITS, 2)), device=q.device, dtype=q.dtype)
    env = AnalyticScheduleEnvironment(generator, TrajectorySurrogate(generator.cost, kit), chunk=chunk)
    out = env.score(physical, q.expand(windows, -1), v.expand(windows, -1), candidates, draws)
    report = dict(windows=windows, bounce_draws=2,
        environment='Existing decoder endpoint/Hermite analytic rollout; no Isaac',
        search_ms_mean=float(np.mean(runtimes)), search_ms_max=float(np.max(runtimes)),
        changed_assignment_fraction=float((np.asarray(actions) != np.asarray(graph)).sum()/
            sum(len(set(h.time for h in c.hits)) for c in clips)),
        selected_missing_edge_count=sum(d['selected']['missing_edges'] for d in details),
        selected_missing_pair_count=sum(d['selected']['missing_pairs'] for d in details),
        ready_initial_supported=int(np.isfinite(initial).sum()), ready_initial_entries=int(initial.size))
    for j, name in enumerate(('graph', 'transition_table')):
        report[name] = dict(hits=int(out['hits'][:, j].sum()),
            requested=int(out['requested'][:, j].sum()),
            hit_rate=float(out['hits'][:, j].sum()/out['requested'][:, j].sum()),
            return_mean=float(out['returns'][:, j, :, 0].mean()))
    save_json(run/'probe.json', report)
    save_json(run/'probe_assignments.json', details)
    torch.save({k: value.cpu() for k, value in out.items() if torch.is_tensor(value)}, run/'probe_rollouts.pt')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--duration-trials', type=int, default=16)
    p.add_argument('--duration-search', choices=('descending', 'random'), default='descending',
                   help='Default: 1 second first, then descend every 20 ms; random reproduces the original calibration')
    p.add_argument('--pair-candidates', type=int, default=24)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--collocation-substeps', type=int, default=4)
    p.add_argument('--hit-depth-mm', type=float, default=None,
                   help='Calibration-only inward head depth; default uses the checkpoint band')
    p.add_argument('--ignore-hit-speed', action='store_true',
                   help='Calibration-only: log impact speed but do not require the requested speed')
    p.add_argument('--allow-early-target-top-entry', action='store_true',
                   help='Allow bounded early head contact after verified top entry; reject side/underside entry')
    p.add_argument('--probe-windows', type=int, default=32)
    p.add_argument('--yaw-centers', type=float, nargs='+', default=[-.9, -.3, .3, .9])
    p.add_argument('--seed', type=int, default=20261004)
    args = p.parse_args(argv)
    if min(args.samples, args.chunk, args.pair_candidates, args.collocation_substeps) < 1 or args.duration_trials < 4 or args.probe_windows < 0:
        p.error('positive sampling sizes and at least four duration trials required')
    if not np.isfinite(args.yaw_centers).all() or (np.diff(args.yaw_centers) <= 0).any():
        p.error('yaw centers must be finite and strictly increasing')
    if args.hit_depth_mm is not None and (not np.isfinite(args.hit_depth_mm) or args.hit_depth_mm <= 0):
        p.error('hit depth must be finite and positive')
    run = args.out.resolve(); run.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    duration_count = len(descending_deadlines()) if args.duration_search == 'descending' else args.duration_trials
    checkpoint = args.checkpoint.resolve()
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    device = torch.device(args.device)
    with rnea_from_cache_if_needed():
        model = A.build_model(A.model_config_from_checkpoint(source), device)
        # Ensure calibration diagnostics are active even if a training weight was off.
        cfg = A.CostConfig(**source['train_cfg']['cost'])
        cfg.w_velocity = max(cfg.w_velocity, 1.)
        cfg.w_joint_box = max(cfg.w_joint_box, 1.)
        cfg.n_sub = max(cfg.n_sub, args.collocation_substeps)
        cost = A.FusedCost(cfg, max_horizon=100, device=device)
    model.load_state_dict(source['model'])
    kit = A.DrumKit.from_layout()
    generator = DecoderHitPoseGenerator(model, cost, kit,
        waist_upright_ease_wp=(source.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run/'tb'), flush_secs=5)
    registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
    (registry/run.name).symlink_to((run/'tb').resolve())
    metadata = dict(checkpoint=str(checkpoint), checkpoint_step=int(source['step']),
        samples=args.samples, duration_trials=duration_count, duration_search=args.duration_search,
        yaw_centers=args.yaw_centers,
        seed=args.seed, generator_passes=1, generator_initialization='checkpoint initial_trajectory',
        environment='Actual decoder spline, existing collocation physics and analytic velocity extrema',
        collocation_substeps=cfg.n_sub, torque_physics_mode=cfg.physics_mode,
        head_membership='Exact closest tip/allowed shaft contact; training softmin residual logged separately',
        hit_depth_mm=(args.hit_depth_mm if args.hit_depth_mm is not None else
                      1000*(cfg.terminal_axial_hi-cfg.terminal_axial_lo)),
        hit_depth_override_scope='Head membership only; original speed/direction diagnostics and all path constraints retained',
        statistic='Mean of per-start minimum successful searched deadlines; failures reported separately',
        initial_points='Legal source region, 2..12 mm outward; 3 mm IK tolerance',
        initial_velocity='0.1..1.2 m/s outward plus Gaussian 0.1 m/s tangential; inertia-weighted arm IK',
        strike_speed_m_s=2., impact_speed_required=not args.ignore_hit_speed,
        allow_early_target_top_entry=args.allow_early_target_top_entry,
        target_collision_policy=('Verified physical-head top entry with bounded post-contact continuation; side/underside entry and wrists checked; terminal hit region retained'
                                if args.allow_early_target_top_entry else 'Checkpoint timed target exemption'),
        speed_error_tolerance_m_s=None if args.ignore_hit_speed else .75, direction_cos_min=.5,
        score_config=asdict(AssignmentScoreConfig()), training=False,
        limitations=['Finite random search gives sampled feasible times, not physical lower bounds',
            'Transition groups hold the other arm ready and request a fixed waist target',
            'Pair yaw table is a static reach/clearance approximation',
            'Means condition on successful starting states; support penalty exposes missing samples',
            'No Isaac validation or whole-schedule feasibility guarantee'])
    save_json(run/'config.json', metadata)
    print(json.dumps(dict(stage='pair_yaw_search', run=str(run), checkpoint_step=source['step'])), flush=True)
    pair_yaw, pair_supported, pair_details = calibrate_pair_yaws(cost, kit, rng,
        candidates=args.pair_candidates)
    np.savez_compressed(run/'pair_yaw_search.npz', **pair_details)
    D, Y = kit.num_drums, len(args.yaw_centers)
    shape = (2, D, D, Y)
    means = np.full(shape, np.inf)
    support = np.zeros(shape, int); trials = np.full(shape, args.samples, int)
    geometric_means = means.copy(); geometric_support = support.copy()
    std = np.zeros(shape)
    table = TransitionTable(tuple(kit.names), np.asarray(args.yaw_centers), means, support, trials,
                            pair_yaw, pair_supported, metadata)
    raw = run/'samples'; raw.mkdir()
    finished = 0
    for hand, name in enumerate(('left', 'right')):
        for source_drum in range(D):
            for yi, yaw in enumerate(args.yaw_centers):
                q, v, valid, start_diag = near_head_starts(cost, kit, name, source_drum,
                    yaw, args.samples, rng)
                # Same physical starts and searched deadlines across all destinations.
                durations = (np.broadcast_to(descending_deadlines(), (args.samples, duration_count))
                    if args.duration_search == 'descending' else random_durations(rng, args.samples, args.duration_trials))
                values = {}
                for target in range(D):
                    out = evaluate_trials(generator, q, v, hand, np.full(args.samples, source_drum),
                        np.full(args.samples, target), np.full(args.samples, yaw), durations, chunk=args.chunk,
                        hit_depth_m=None if args.hit_depth_mm is None else args.hit_depth_mm/1000,
                        require_hit_speed=not args.ignore_hit_speed,
                        allow_early_target_top_entry=args.allow_early_target_top_entry)
                    for success_name in ('success', 'geometric'):
                        out[success_name] &= valid.cpu().numpy()[:, None]
                    stats = minimum_statistics(durations[None], out['success'][None])
                    geometric = minimum_statistics(durations[None], out['geometric'][None])
                    key = hand, source_drum, target, yi
                    means[key], support[key], std[key] = stats['mean'][0], stats['count'][0], stats['std'][0]
                    geometric_means[key], geometric_support[key] = geometric['mean'][0], geometric['count'][0]
                    for field, value in out.items():
                        values[f'{target}_{field}'] = value
                np.savez_compressed(raw/f'h{hand}_d{source_drum}_y{yi}.npz',
                    q=q.cpu().numpy(), v=v.cpu().numpy(), start_valid=valid.cpu().numpy(),
                    durations=durations, **start_diag, **values)
                table.save(run/'transition_table.npz')
                finished += D
                writer.add_scalar('calibration/groups_completed', finished, finished)
                writer.add_scalar('calibration/hard_supported_fraction', float((support > 0).sum()/finished), finished)
                writer.add_scalar('calibration/start_valid_fraction', float(valid.float().mean()), finished)
                progress = dict(stage='transition_search', groups_completed=finished, total_groups=int(means.size),
                    hand=name, source=kit.names[source_drum], yaw_rad=yaw,
                    valid_starts=int(valid.sum()), hard_supported_groups=int((support > 0).sum()),
                    geometric_supported_groups=int((geometric_support > 0).sum()),
                    elapsed_s=round(time.monotonic()-begin, 2))
                save_json(run/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    np.savez_compressed(run/'transition_statistics.npz', std_s=std,
        geometric_minimum_s=geometric_means, geometric_support=geometric_support)
    report = dict(status='calibration_complete', groups=int(means.size),
        hard_supported_groups=int((support > 0).sum()), fully_supported_groups=int((support == trials).sum()),
        geometric_supported_groups=int((geometric_support > 0).sum()),
        supported_pair_goals=int(pair_supported.sum()), pair_goal_entries=int(pair_supported.size),
        decoder_trials=int(means.size*args.samples*duration_count), elapsed_s=time.monotonic()-begin)
    save_json(run/'summary.json', report)
    if args.probe_windows:
        print(json.dumps(dict(stage='fresh_rhythm_probe')), flush=True)
        report['probe'] = probe(generator, table, run, np.random.default_rng(args.seed+100000),
            args.probe_windows, args.duration_trials, args.chunk, args.duration_search,
            None if args.hit_depth_mm is None else args.hit_depth_mm/1000,
            not args.ignore_hit_speed, args.allow_early_target_top_entry)
        for name in ('graph', 'transition_table'):
            writer.add_scalar(f'probe/{name}_hit_rate', report['probe'][name]['hit_rate'], finished)
    report.update(status='complete', elapsed_s=time.monotonic()-begin)
    save_json(run/'summary.json', report)
    writer.close()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
