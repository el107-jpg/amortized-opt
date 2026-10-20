"""Run unresolved-start recovery after existing table writers have finished."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fill_transition_table_gaps import prepare_cases
from scripts.probe_transition_qp import write_json


def prerequisites_complete(folders):
    for folder in folders:
        result = json.loads((folder/'run.json').read_text())
        if result['status'] == 'failed':
            raise RuntimeError(f'Prerequisite failed: {folder}: {result.get("error", "see logs")}')
        if result['status'] != 'complete': return False
        # Completion is written before cleanup; wait for the launcher to exit.
        pid = result.get('launcher_pid')
        if pid:
            try: os.kill(pid, 0)
            except ProcessLookupError: pass
            else: return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-run', type=Path, default=ROOT/'runs/transition_qp_unrestricted_20261005')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path, nargs='+', required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    args = parser.parse_args()
    args.table_run = args.table_run.resolve(); args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve(); args.wait_for = [p.resolve() for p in args.wait_for]
    queue = args.out.with_name(args.out.name+'_queue')
    queue.mkdir(parents=True, exist_ok=False)
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checksum != 'c6f5919625f60fcf2b3610cb3b7c3a8df91a71e507f1a3940e3788a6a855f319':
        raise ValueError('Per-start recovery requires the frozen table checkpoint')
    preview = prepare_cases(argparse.Namespace(table_run=args.table_run, out=queue, per_start=True, stride=3))
    write_json(queue/'config.json', dict(table_run=str(args.table_run), checkpoint=str(args.checkpoint),
        checkpoint_sha256=checksum, wait_for=[str(p) for p in args.wait_for], out=str(args.out),
        scope='All unresolved original starting samples, including supported pairs',
        preview_unresolved_starts=len(preview), rebuild_queue_at_launch=True, training=False,
        preserved_departures=True, verification_substeps=32))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(queue/'tb'))
    (ROOT/'runs/tensorboard_active'/queue.name).symlink_to((queue/'tb').resolve())
    writer.add_scalar('queue/unresolved_starts_at_schedule', len(preview), 0)
    writer.add_scalar('queue/waiting_for_pair_searches', 1, 0); writer.flush()
    write_json(queue/'run.json', dict(status='queued', launcher_pid=os.getpid(), out=str(args.out)))
    print(json.dumps(dict(status='queued', unresolved_starts_now=len(preview), out=str(args.out))), flush=True)
    try:
        while not prerequisites_complete(args.wait_for): time.sleep(5)
        command = [sys.executable, str(ROOT/'scripts/fill_transition_table_gaps.py'),
            '--table-run', str(args.table_run), '--checkpoint', str(args.checkpoint), '--out', str(args.out),
            '--recover', '--per-start', '--recovery-starts', '1', '--devices', *args.devices]
        write_json(queue/'run.json', dict(status='launching', launcher_pid=os.getpid(), out=str(args.out)))
        writer.add_scalar('queue/waiting_for_pair_searches', 0, 1); writer.flush()
        with (queue/'search.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
            write_json(queue/'run.json', dict(status='running', launcher_pid=os.getpid(),
                search_pid=proc.pid, out=str(args.out)))
            code = proc.wait()
        if code: raise RuntimeError(f'Per-start search exited {code}; inspect {queue/"search.log"}')
        write_json(queue/'run.json', dict(status='complete', out=str(args.out)))
    except BaseException as exc:
        write_json(queue/'run.json', dict(status='failed', error=str(exc), out=str(args.out)))
        raise
    finally:
        writer.close()


if __name__ == '__main__': main()
