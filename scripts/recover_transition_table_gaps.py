"""Recover unresolved pairs alongside shortening, with one table writer.

While the saved-witness updater is active, forward immutable verified result
manifests to it. Take over table publication after it exits. This avoids two
processes reading/modifying the same per-source record concurrently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fill_transition_table_gaps import prepare_cases, ingest_results, refresh_table
from scripts.probe_transition_qp import write_json
import amortized_opt as A


def importer_active(folder):
    path = folder/'run.json'
    if not path.exists(): raise RuntimeError('Active importer has no run manifest')
    run = json.loads(path.read_text())
    if run['status'] != 'running': return False
    try:
        os.kill(run['launcher_pid'], 0)
    except ProcessLookupError:
        return False
    return True


def forward_results(out, importer, forwarded):
    folder = importer/'wforward'; folder.mkdir(exist_ok=True)
    for path in sorted([*out.glob('w*/witness_*.json'), *out.glob('w*/case_*.json')]):
        if str(path) in forwarded: continue
        result = json.loads(path.read_text())
        if result['status'] in ('complete', 'verified_witness'):
            result['status'] = 'verified_witness'
            write_json(folder/f'witness_recovery_{path.parent.name}_{path.stem}.json', result)
        forwarded.add(str(path))


def parent(args):
    args.recover = True
    args.out.mkdir(parents=True, exist_ok=False)
    cases = prepare_cases(args)
    # Preserve the authoritative pre-update baseline, whose departure rows
    # are also used by these workers. Existing importer owns all mutations.
    shutil.copytree(args.active_importer/'baseline_backup', args.out/'baseline_backup')
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checksum != 'c6f5919625f60fcf2b3610cb3b7c3a8df91a71e507f1a3940e3788a6a855f319':
        raise ValueError('Recovery requires the frozen table checkpoint')
    write_json(args.out/'config.json', dict(table_run=str(args.table_run), checkpoint=str(args.checkpoint),
        checkpoint_sha256=checksum, devices=args.devices, active_importer=str(args.active_importer),
        target_scope='Remaining unresolved pairs only; preserve original sampled q/v', training=False,
        depth_m=.07, speed_required=False, verification_substeps=32, physics_mode='spline',
        decoder_grid_s=[.02, 2.], decoder_resolution_s=.02,
        qp_recovery_anchors_s=[.2, .4, .6, .8, 1., 1.5, 2.],
        recovery_starts_per_anchor=args.recovery_starts, stride=args.stride,
        idle_arm_constraints=False, limitation='Budget failures remain unresolved, not infeasible'))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'))
    (ROOT/'runs/tensorboard_active'/args.out.name).symlink_to((args.out/'tb').resolve())
    writer.add_scalar('recovery/target_pairs', len(cases), 0); writer.flush()
    processes, forwarded, applied = [], set(), set()
    started = time.monotonic(); owns_table = False
    names = list(A.DrumKit.from_layout().names)
    try:
        for rank, device in enumerate(args.devices):
            log = (args.out/f'w{rank}.log').open('w')
            command = [sys.executable, str(ROOT/'scripts/fill_transition_table_gaps.py'),
                '--table-run', str(args.table_run), '--out', str(args.out), '--checkpoint', str(args.checkpoint),
                '--device', device, '--rank', str(rank), '--workers', str(len(args.devices)),
                '--stride', str(args.stride), '--recovery-starts', str(args.recovery_starts)]
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
            processes.append((proc, log))
        write_json(args.out/'run.json', dict(status='running', launcher_pid=os.getpid(),
            worker_pids=[p.pid for p, _ in processes], table_writer=str(args.active_importer)))
        while True:
            if not owns_table and not importer_active(args.active_importer):
                owns_table = True
                write_json(args.out/'run.json', dict(status='running', launcher_pid=os.getpid(),
                    worker_pids=[p.pid for p, _ in processes], table_writer=str(args.out)))
            if owns_table:
                if ingest_results(args, applied): refresh_table(args, names)
            else:
                forward_results(args.out, args.active_importer, forwarded)
            results = [json.loads(p.read_text()) for p in args.out.glob('w*/case_*.json')]
            local = dict(status='running', target_pairs=len(cases), finished_pairs=len(results),
                recovered_pairs=sum(r['status'] == 'complete' for r in results),
                unresolved_pairs=sum(r['status'] == 'unresolved_budget_exhausted' for r in results),
                elapsed_s=time.monotonic()-started, table_writer=str(args.out if owns_table else args.active_importer))
            write_json(args.out/'recovery_progress.json', local)
            writer.add_scalar('recovery/finished_pairs', len(results), int(local['elapsed_s']))
            writer.add_scalar('recovery/recovered_pairs', local['recovered_pairs'], int(local['elapsed_s']))
            writer.flush()
            if any(p.poll() not in (None, 0) for p, _ in processes):
                raise RuntimeError('Recovery worker failed; inspect worker logs')
            if all(p.poll() is not None for p, _ in processes) and owns_table:
                ingest_results(args, applied)
                if len(list(args.out.glob('w*/case_*.json'))) != len(cases):
                    raise RuntimeError('Missing final recovery results')
                break
            time.sleep(5)
        summary = refresh_table(args, names, complete=True)
        write_json(args.out/'run.json', dict(status='complete', elapsed_s=time.monotonic()-started))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc)))
        raise
    finally:
        for proc, log in processes:
            if proc.poll() is None: proc.terminate()
            log.close()
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-run', type=Path, default=ROOT/'runs/transition_qp_unrestricted_20261005')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--active-importer', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--recovery-starts', type=int, default=3)
    args = parser.parse_args()
    for key in ('table_run', 'checkpoint', 'out', 'active_importer'):
        setattr(args, key, getattr(args, key).resolve())
    parent(args)


if __name__ == '__main__': main()
