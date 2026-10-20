"""Measure frozen-decoder transitions on the original sampled departures.

Fresh decoder outputs only: no QP, terminal correction, saved-path transfer,
retiming, or duration continuation. Every duration receives both existing yaw
hints. Coarse checks reject candidates cheaply; all accepted witnesses pass
32 spline collocations per segment. Source q/v are copied unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.ik import arm_joint_indices
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.trajectory_qp import cost_for_horizon
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance, profile
from scripts.fill_transition_table_gaps import context, load_piece, setup
from scripts.build_unrestricted_transition_chart import allowed_pairs, publish
from scripts.probe_transition_qp import save_npz, write_json


def table_kinds(args):
    return ('strict', 'relaxed', 'average') if getattr(args, 'torque_average_limit', None) is not None else ('strict', 'relaxed')


def empty_records(source, max_segments):
    """Keep original departures, discard all old trajectory/search results."""
    piece = {k: source[k].copy() for k in ('hand', 'source', 'destination', 'sample', 'q', 'v')}
    B, D = piece['q'].shape
    piece.update(minimum_s=np.full(B, np.inf), decoder_minimum_s=np.full(B, np.inf),
        paths=np.zeros((B, max_segments, D), np.float32), completed=np.asarray(False),
        raw_outcomes=np.zeros((B, max_segments+1), bool),
        coarse_outcomes=np.zeros((B, max_segments+1, 2), bool),
        decoder_yaw_hint_index=np.full(B, -1, np.int8),
        selected_for_qp=np.zeros(B, bool), qp_attempts=np.zeros((B, max_segments+1), np.int16))
    for key in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err',
                'contact_dir_cos', 'stick_elev_deg', 'torque_peak_ratio', 'torque_over_fraction',
                'torque_over_duration_s', 'torque_longest_run_s', 'torque_excess_integral_Nms',
                'torque_excess_average_Nm', 'torque_excess_average_max_joint_Nm'):
        piece[key] = np.full(B, np.nan)
    for key in ('torque_excess_integral_per_joint_Nms', 'torque_excess_average_per_joint_Nm'):
        piece[key] = np.full((B, D), np.nan)
    return piece


def accept_decoder(piece, rows, N, paths, dense, yaw_index, gate='strict_success'):
    """Only dense-verified raw paths can populate the decoder table."""
    for j in np.flatnonzero(dense[gate]):
        row = int(rows[j])
        piece['raw_outcomes'][row, N] = True
        if N*.02 >= piece['minimum_s'][row]: continue
        piece['minimum_s'][row] = piece['decoder_minimum_s'][row] = N*.02
        piece['paths'][row] = 0
        piece['paths'][row, :N] = paths[j]
        piece['decoder_yaw_hint_index'][row] = yaw_index
        for key in ('torque_excess_max', 'velocity_excess_max', 'joint_box_excess_max',
                    'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err',
                    'contact_dir_cos', 'stick_elev_deg', 'torque_peak_ratio', 'torque_over_fraction',
                    'torque_over_duration_s', 'torque_longest_run_s', 'torque_excess_integral_Nms',
                    'torque_excess_average_Nm', 'torque_excess_average_max_joint_Nm',
                    'torque_excess_integral_per_joint_Nms', 'torque_excess_average_per_joint_Nm'):
            piece[key][row] = dense[key][j]


def worker(args):
    torch.set_num_threads(2)
    ck, cost, kit = setup(args.checkpoint, args.device)
    model = A.build_model(A.model_config_from_checkpoint(ck), args.device)
    model.load_state_dict(ck['model']); model.eval()
    generator = DecoderHitPoseGenerator(model, cost, kit, horizon=100,
        waist_upright_ease_wp=(ck.get('scenario_cfg') or {}).get('waist_upright_ease_wp'))
    folders = {kind: args.out/kind/f'w{args.rank}' for kind in table_kinds(args)}
    for folder in folders.values(): folder.mkdir(exist_ok=True)
    allowance = SmallViolationAllowance(torque_peak_ratio=args.relaxed_torque_ratio,
        torque_duration_fraction=args.relaxed_torque_fraction, torque_continuous_s=2.,
        collision_depth_m=.01, joint_excess_rad=np.deg2rad(5), terminal_distance_m=.01,
        torque_average_limit_Nm=args.torque_average_limit)
    costs = {}
    def c(N, dense=False):
        if (N, dense) not in costs:
            obj = cost_for_horizon(cost, N); obj.cfg = copy.copy(cost.cfg)
            obj.cfg.n_sub = 32 if dense else 4
            costs[N, dense] = obj
        return costs[N, dense]
    for i, source in enumerate(sorted((args.out/'inputs').glob('h*_source_*.npz'))):
        if i % args.workers != args.rank: continue
        pieces = {kind: empty_records(load_piece(source), args.max_segments) for kind in folders}
        piece = pieces['strict']
        h, src = int(piece['hand']), int(piece['source'])
        q = torch.tensor(piece['q'], device=args.device)
        v = torch.tensor(piece['v'], device=args.device)
        B = len(q)
        for kind, folder in folders.items(): save_npz(folder/source.name, **pieces[kind])
        started = time.monotonic(); dense_evaluations = 0; measurements = []
        for N in range(1, args.max_segments+1):
            targets, ctx, lengths = context(kit, h, [src]*B, piece['destination'], N, args.device)
            with torch.no_grad():
                for yi, yaw in enumerate((q[:, 0], None)):
                    plan = generator.predict(q, v, torch.tensor(targets[:, None], device=args.device),
                        q.new_full((B, 1), N*.02), q.new_full((B, 1, 2), 2.),
                        torch.ones(B, 1, device=args.device, dtype=torch.bool), 0,
                        return_plan=True, waist_yaw=yaw)
                    traj = plan['traj'][:, :N].clone()
                    idle = arm_joint_indices(('left', 'right')[1-h]); traj[:, :, idle] = q[:, None, idle]
                    coarse = profile(c(N), q, v, traj, ctx, lengths, depth_m=.07, allowance=allowance)
                    pieces['strict']['coarse_outcomes'][:, N, yi] = coarse['strict_success']
                    pieces['relaxed']['coarse_outcomes'][:, N, yi] = coarse['relaxed_magnitude_success']
                    # Duration of a torque pulse cannot be safely rejected
                    # from coarse samples: apply that gate on the dense grid.
                    candidate = coarse['strict_success'] | coarse['relaxed_magnitude_success']
                    if 'average' in pieces:
                        pieces['average']['coarse_outcomes'][:, N, yi] = coarse['relaxed_non_torque_success']
                        # The integral cannot be rejected from a sparse grid.
                        candidate |= coarse['relaxed_non_torque_success']
                    rows = np.flatnonzero(candidate)
                    if len(rows):
                        _, one, L = context(kit, h, [src]*len(rows), piece['destination'][rows], N, args.device)
                        dense = profile(c(N, True), q[rows], v[rows], traj[rows], one, L,
                                        depth_m=.07, allowance=allowance)
                        dense_evaluations += len(rows)
                        paths = traj[rows].cpu().numpy()
                        for kind in folders:
                            accept_decoder(pieces[kind], rows, N, paths, dense, yi, gate=kind+'_success')
                        if 'average' in pieces:
                            measurements.append(dict(hand=np.full(len(rows), h), source=np.full(len(rows), src),
                                destination=piece['destination'][rows], sample=piece['sample'][rows],
                                segments=np.full(len(rows), N), yaw_hint_index=np.full(len(rows), yi), **dense))
            if N % 10 == 0 or N == args.max_segments:
                for kind, folder in folders.items(): save_npz(folder/source.name, **pieces[kind])
                record = dict(hand=h, source=list(kit.names)[src], deadline_s=N*.02,
                    evaluated_deadlines=N, starts=B, supported_starts=int(np.isfinite(piece['minimum_s']).sum()),
                    relaxed_supported_starts=int(np.isfinite(pieces['relaxed']['minimum_s']).sum()),
                    average_supported_starts=int(np.isfinite(pieces['average']['minimum_s']).sum()) if 'average' in pieces else None,
                    dense_evaluations=dense_evaluations, elapsed_s=time.monotonic()-started)
                write_json(args.out/f'progress_w{args.rank}.json', record)
                print(json.dumps(record), flush=True)
        for kind, folder in folders.items():
            pieces[kind]['completed'] = np.asarray(True)
            save_npz(folder/source.name, **pieces[kind])
        if measurements:
            save_npz(args.out/f'measurements_h{h}_source_{src}.npz',
                **{key: np.concatenate([p[key] for p in measurements]) for key in measurements[0]})
    write_json(args.out/f'done_w{args.rank}.json', dict(status='complete'))


def publish_decoder(args, complete=False):
    tables = {}
    for kind in table_kinds(args):
        publish_args = argparse.Namespace(out=args.out/kind, max_segments=args.max_segments,
            verify_substeps=32, qp_starts_per_pair=0,
            decoder_acceptance=(f'average torque excess ≤ {args.torque_average_limit:g} Nm' if kind=='average' else kind))
        summary = publish(publish_args, list(A.DrumKit.from_layout().names), complete=complete)
        summary.update(path_generation='Frozen decoder only; two yaw hints; no trajectory repairs',
            acceptance=kind, decoder_only=True, qp_iterations=0, training=False,
            yaw_motion='Decoder outputs use existing yaw hints; no additional waist motion window',
            search_coverage='All original starts at every 20 ms grid duration; coarse rejection and dense acceptance',
            primary_statistic='Shortest fresh decoder duration passing the '+kind+' gate across original departures',
            limitation='Unsupported starts have no passing decoder output on this grid; this is not physical infeasibility')
        if kind == 'relaxed':
            summary.update(torque_peak_ratio_allowed=args.relaxed_torque_ratio,
                torque_over_fraction_allowed=args.relaxed_torque_fraction,
                collision_overlap_allowed_m=.01, joint_excess_allowed_deg=5., terminal_miss_allowed_m=.01)
        if kind == 'average':
            summary.update(torque_average_excess_allowed_Nm=args.torque_average_limit,
                torque_aggregation='Sum of positive excess over moving-arm and waist actuators, integrated over time and divided by trajectory duration',
                peak_torque_cap=None, overload_duration_cap=None,
                collision_overlap_allowed_m=.01, joint_excess_allowed_deg=5., terminal_miss_allowed_m=.01)
        write_json(args.out/kind/'summary.json', summary)
        tables[kind] = summary
    summary = dict(status='complete' if complete else 'running', tables=tables,
        decoder_only=True, qp_iterations=0, same_generated_paths=True)
    write_json(args.out/'summary.json', summary)
    return summary


def parent(args):
    if args.out == args.source_run or args.source_run in args.out.parents:
        raise ValueError('Decoder-only output must be separate from the QP table')
    args.out.mkdir(parents=True, exist_ok=False)
    for kind in table_kinds(args): (args.out/kind).mkdir()
    inputs = args.out/'inputs'; inputs.mkdir()
    allowed = allowed_pairs(list(A.DrumKit.from_layout().names))
    count = 0
    for file in sorted(args.source_run.glob('w*/h*_source_*.npz')):
        data = load_piece(file)
        h, src = int(data['hand']), int(data['source'])
        if not allowed[h, src, data['destination']].all(): raise ValueError('Excluded source rows')
        saved = {k: data[k] for k in ('hand', 'source', 'destination', 'sample', 'q', 'v')}
        save_npz(inputs/file.name, **saved); count += len(saved['q'])
    if count != 680 or len(list(inputs.glob('*.npz'))) != 13:
        raise ValueError('Expected the original 680 departures across 13 source groups')
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checksum != 'c6f5919625f60fcf2b3610cb3b7c3a8df91a71e507f1a3940e3788a6a855f319':
        raise ValueError('Decoder-only comparison requires the frozen original checkpoint')
    write_json(args.out/'config.json', dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checksum,
        source_run=str(args.source_run), start_records=count, samples_per_pair=8, allowed_pairs=85,
        devices=args.devices, candidate_deadlines_s=(np.arange(1, args.max_segments+1)*.02).tolist(),
        path_generation='One frozen decoder pass per yaw hint and deadline', yaw_hints=['current_yaw', 'goal_derived'],
        decoder_only=True, qp_iterations=0, terminal_corrections=False, continuation=False, retiming=False,
        training=False, physics_mode='spline', verification_substeps=32, coarse_substeps=4,
        hit_depth_m=.07, impact_speed_required=False, idle_arm_constraints=False,
        strict_constraints='Existing zero-excess physical contract',
        relaxed_constraints=dict(torque_peak_ratio=args.relaxed_torque_ratio,
            torque_over_fraction=args.relaxed_torque_fraction, torque_fraction_measurement='Maximum over actuators of trapezoidal dense-sampled overload time divided by trajectory duration',
            collision_overlap_m=.01, joint_excess_deg=5., terminal_region_miss_m=.01,
            velocity='Existing strict limit', stick_direction_and_elevation='Existing strict gates'),
        torque_average_limit_Nm=args.torque_average_limit,
        torque_integral_units='Nm s', torque_average_units='Nm',
        torque_average_definition='sum_j integral max(abs(tau_j)-available_j,0) dt / trajectory_duration',
        forbidden_drums_by_hand={'left':['floor_tom'], 'right':[]}))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'))
    (ROOT/'runs/tensorboard_active'/args.out.name).symlink_to((args.out/'tb').resolve())
    writer.add_scalar('decoder/total_start_records', count, 0); writer.flush()
    workers = []; started = time.monotonic(); previous = None
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            command = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
                '--checkpoint', str(args.checkpoint), '--device', device, '--rank', str(rank),
                '--workers', str(len(args.devices)), '--max-segments', str(args.max_segments)]
            command += ['--relaxed-torque-ratio', str(args.relaxed_torque_ratio),
                        '--relaxed-torque-fraction', str(args.relaxed_torque_fraction)]
            if args.torque_average_limit is not None:
                command += ['--torque-average-limit', str(args.torque_average_limit)]
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'))
            workers.append((proc, log))
        write_json(args.out/'run.json', dict(status='running', launcher_pid=os.getpid(),
            worker_pids=[p.pid for p, _ in workers]))
        while any(p.poll() is None for p, _ in workers):
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('Decoder worker failed; inspect worker logs')
            signature = sorted((str(p), p.stat().st_mtime_ns) for p in args.out.glob('*/w*/h*_source_*.npz'))
            if signature and signature != previous:
                summary = publish_decoder(args)
                previous = signature
                step = int(time.monotonic()-started)
                for kind, table in summary['tables'].items():
                    writer.add_scalar(kind+'/supported_pairs', table['supported_pairs'], step)
                    writer.add_scalar(kind+'/supported_starts', table['supported_start_records'], step)
                    writer.add_scalar(kind+'/completed_pairs', table['completed_pairs'], step)
                writer.flush(); print(json.dumps(summary), flush=True)
            time.sleep(2)
        if any(p.returncode for p, _ in workers): raise RuntimeError('Decoder worker failed')
        summary = publish_decoder(args, complete=True)
        if any(t['completed_pairs'] != 85 or t['valid_start_records'] != 680 for t in summary['tables'].values()):
            raise RuntimeError('Incomplete decoder-only coverage')
        summary['elapsed_s'] = time.monotonic()-started
        write_json(args.out/'summary.json', summary)
        write_json(args.out/'run.json', dict(status='complete', elapsed_s=summary['elapsed_s']))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc)))
        raise
    finally:
        for proc, log in workers:
            if proc.poll() is None: proc.terminate()
            log.close()
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'runs/transition_qp_unrestricted_20261005')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--devices', nargs='+')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-segments', type=int, default=100)
    parser.add_argument('--relaxed-torque-ratio', type=float, default=1.2)
    parser.add_argument('--relaxed-torque-fraction', type=float, default=.1)
    parser.add_argument('--torque-average-limit', type=float, help='Provisional limit on time-average summed positive excess (Nm)')
    args = parser.parse_args()
    args.source_run=args.source_run.resolve(); args.checkpoint=args.checkpoint.resolve(); args.out=args.out.resolve()
    if not 1 <= args.max_segments <= 100: raise ValueError('Duration grid must fit the 2 s decoder horizon')
    if args.torque_average_limit is not None and (not np.isfinite(args.torque_average_limit) or args.torque_average_limit < 0):
        raise ValueError('Average torque excess allowance must be finite and nonnegative')
    parent(args) if args.devices else worker(args)


if __name__ == '__main__': main()
