#!/usr/bin/env python3
"""Fresh random-yaw departure cohort, then decoder-only minimum-time search."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT/'runs/transition_min_time_decoder_only_20261004_174008')
    p.add_argument('--devices', nargs='+', default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    p.add_argument('--samples', type=int, default=8)
    args = p.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    begin=time.monotonic()
    state={'status':'running','regions':3,'yaw_region_edges':[-.9,-.3,.3,.9],
           'yaw_sampling':'uniform_within_region','decoder_only':True}
    def save(**updates):
        state.update(updates,elapsed_s=time.monotonic()-begin)
        temporary=out/'pipeline.json.tmp'; temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(out/'pipeline.json')
    commands=[('starts',[sys.executable,'scripts/build_transition_region_starts.py',
        '--reference',str(args.reference.resolve()),'--out',str(out/'cohort'),
        '--devices',*args.devices,'--samples',str(args.samples)]),
        ('search',[sys.executable,'scripts/search_transition_minimum_times.py',
        '--baseline',str(out/'cohort'),'--out',str(out),'--devices',*args.devices,
        '--chunk','64','--decoder-only'])]
    try:
        for stage,command in commands:
            save(stage=stage)
            with (out/f'{stage}.log').open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(child_pid=process.pid)
                code=process.wait()
                if code: raise RuntimeError(f'{stage} exited {code}; see {stage}.log')
        save(status='complete',stage='complete')
    except BaseException as error:
        save(status='failed',error=str(error)); raise


if __name__=='__main__': main()
