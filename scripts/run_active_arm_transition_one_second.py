#!/usr/bin/env python3
"""Recheck every saved start, audit at 1 s, repair, then publish a scoped table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/transition_table_random_v2_20261004')
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    audit, qp, database = (out/name for name in ('audit', 'qp', 'database'))
    begin = time.monotonic()
    state = dict(status='running', deadline_s=1., constraint_scope='Moving arm and shared waist only',
        source=str(args.source.resolve()), devices=args.devices, iterations=5,
        starting_eligibility_recomputed=True, forbidden_drums_by_hand={'left': ['floor_tom'], 'right': []})

    def save(**updates):
        state.update(updates, elapsed_s=time.monotonic()-begin)
        temporary = out/'run.json.tmp'; temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(out/'run.json')

    commands = [
        ('decoder', [sys.executable, 'scripts/audit_transition_deadlines.py',
            '--source', str(args.source.resolve()), '--out', str(audit), '--devices', *args.devices,
            '--deadline-search', 'one-second', '--ignore-idle-arm', '--hit-depth-mm', '70',
            '--ignore-hit-speed', '--allow-early-target-top-entry']),
        ('qp', [sys.executable, 'scripts/probe_transition_qp.py', '--audit', str(audit),
            '--out', str(qp), '--devices', *args.devices, '--iterations', '5',
            '--chunk', '16', '--jacobian-chunk', '256']),
        ('database', [sys.executable, 'scripts/rebuild_transition_database.py',
            '--audit', str(audit), '--qp-run', str(qp), '--out', str(database)]),
    ]
    try:
        for stage, command in commands:
            folder = dict(decoder=audit, qp=qp, database=database)[stage]
            summary = folder/'summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('status') == 'complete':
                if stage == 'decoder':
                    # A dashboard registration error can occur after all audit
                    # artifacts are complete. Preserve and reuse that work.
                    config = json.loads((audit/'config.json').read_text())
                    if not config.get('ignore_idle_arm') or config['deadlines_s'] != [1.]:
                        raise ValueError('Existing audit uses a different scope or deadline')
                    registry = ROOT/'runs/tensorboard_active'/(out.name+'__audit')
                    if not registry.is_symlink(): registry.symlink_to((audit/'tb').resolve())
                save(stage=stage, resumed_completed_stage=True)
                continue
            save(stage=stage)
            with (out/f'{stage}.log').open('a') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                save(child_pid=process.pid, command=command)
                code = process.wait()
                if code: raise RuntimeError(f'{stage} exited {code}; see {out/(stage+".log")}')
        save(status='complete', stage='complete', database=str(database))
    except BaseException as error:
        save(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
