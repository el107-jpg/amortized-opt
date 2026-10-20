"""Run held-out dense hi-hat checks at 500 updates and training completion."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--first-probe-step', type=int, default=32500)
    args = parser.parse_args()
    run = args.run.resolve()
    state = {'status': 'waiting', 'probes': {}}
    writer = SummaryWriter(str(run/'tb'))
    seen = None
    while True:
        pid = json.loads((run/'launch.json').read_text())['pid']
        try:
            alive = 'train_phase2.py' in Path(f'/proc/{pid}/cmdline').read_text()
        except (FileNotFoundError, ProcessLookupError):
            alive = False
        finished = (run/'summary.json').exists() or not alive
        checkpoint = run/'last.pt'
        if checkpoint.exists():
            stamp = checkpoint.stat().st_mtime_ns
            if stamp != seen or finished:
                ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
                seen = stamp
                step = int(ck['step'])
                stage = 'final' if finished else 'midpoint' if step >= args.first_probe_step else None
                if stage and stage not in state['probes']:
                    snapshot = run/f'hihat_probe_checkpoint_{step}.pt'
                    torch.save(ck, snapshot)
                    out = run/f'hihat_probe_{stage}_{step}'
                    state['status'] = 'probing'
                    (run/'hihat_probe_watch.json').write_text(json.dumps(state, indent=2))
                    with (run/f'hihat_probe_{stage}.log').open('w') as log:
                        subprocess.run([sys.executable, str(ROOT/'scripts/audit_hihat_transitions.py'),
                            '--run', str(args.source_run.resolve()), '--checkpoint', str(snapshot),
                            '--out', str(out), '--device', args.device], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, check=True)
                    summary = json.loads((out/'summary.json').read_text())
                    cases = json.loads((out/'cases.json').read_text())
                    for hand in ('left', 'right'):
                        count = sum(r['hand'] == hand and r['any_dense_pass'] for r in cases)
                        writer.add_scalar(f'hihat/raw_dense_supported_starts_{hand}', count, step)
                    writer.add_scalar('hihat/raw_dense_supported_pairs', len(summary['dense_verified_pairs']), step)
                    writer.flush()
                    state['probes'][stage] = summary
                    print(json.dumps(summary), flush=True)
                    if finished:
                        # The minimum-time chart uses repaired generator plans;
                        # retain this second check separately from raw support.
                        repair_out = run/f'hihat_qp_final_{step}'
                        for hand in (0, 1):
                            with (run/f'hihat_qp_final_h{hand}.log').open('w') as log:
                                subprocess.run([sys.executable, str(ROOT/'scripts/probe_hihat_repairs.py'),
                                    '--audit', str(out), '--out', str(repair_out),
                                    '--checkpoint', str(snapshot), '--device', args.device,
                                    '--hand', str(hand)], cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, check=True)
                            repaired = json.loads((repair_out/f'summary_h{hand}.json').read_text())
                            count = sum(bool(r['after']['success']) for r in repaired)
                            writer.add_scalar(f'hihat/qp_supported_source_pairs_{("left", "right")[hand]}', count, step)
                            writer.flush()
                        state['final_qp_directory'] = str(repair_out)
                del ck
        state['status'] = 'complete' if finished else 'waiting'
        (run/'hihat_probe_watch.json').write_text(json.dumps(state, indent=2))
        if finished:
            break
        time.sleep(10)
    writer.close()


if __name__ == '__main__':
    main()
