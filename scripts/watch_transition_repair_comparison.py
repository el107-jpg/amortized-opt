"""Evaluate the two repair continuations on identical saved departure states."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from torch.utils.tensorboard import SummaryWriter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',required=True)
    args=p.parse_args();run=args.run.resolve()
    launch=json.loads((run/'launch.json').read_text())
    source=ROOT/'runs/transition_qp_unrestricted_20261005'
    while not (run/'summary.json').exists():
        try:
            alive='train_phase2.py' in Path(f"/proc/{launch['pid']}/cmdline").read_text()
        except FileNotFoundError:alive=False
        if not alive:
            write=dict(status='training_failed',reason='Training stopped before completion; inspect launcher.log')
            (run/'repair_validation.json').write_text(json.dumps(write,indent=2));return
        time.sleep(10)
    checkpoint=run/'last.pt'
    results={}
    commands=[('hihat',ROOT/'scripts/audit_hihat_transitions.py',
               ['--run',str(source),'--checkpoint',str(checkpoint),'--out',str(run/'validation_hihat'),'--device',args.device]),
              ('departures',ROOT/'scripts/probe_transition_coverage_repairs.py',
               ['--source-run',str(source),'--checkpoint',str(checkpoint),'--out',str(run/'validation_departures'),
                '--device',args.device,'--skip-qp'])]
    for stage,script,flags in commands:
        with (run/f'validation_{stage}.log').open('w') as log:
            subprocess.run([sys.executable,str(script),*flags],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[stage]=json.loads((run/f'validation_{stage}'/'summary.json').read_text())
    with SummaryWriter(str(run/'tb')) as writer:
        cases=json.loads((run/'validation_hihat/cases.json').read_text())
        for hand in ('left','right'):
            writer.add_scalar(f'validation/hihat_supported_starts_{hand}',
                sum(row['hand']==hand and row['any_dense_pass'] for row in cases),33800)
        writer.add_scalar('validation/recovered_failed_pairs',results['departures']['new_supported_pairs'],33800)
        writer.add_scalar('validation/recovered_failed_starts',results['departures']['supported_starts'],33800)
    results.update(status='complete',checkpoint=str(checkpoint),step=33800,promoted=False,
                   scope='Saved held-out departure cohort; raw one-pass decoder, dense physical checks')
    (run/'repair_validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
