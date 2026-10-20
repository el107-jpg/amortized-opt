#!/usr/bin/env python3
"""Refresh a ranker comparison PNG until its trials finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--wider-encoders',type=Path,required=True)
    parser.add_argument('--wider-low-regularization',type=Path)
    parser.add_argument('--control-label',default='Control')
    parser.add_argument('--wider-label',default='Wider encoders')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--interval-seconds',type=int,default=30)
    args=parser.parse_args()
    if args.interval_seconds<1:raise ValueError('Positive refresh interval required')
    reporter=Path(__file__).with_name('report_hand_plan_100k_ablation.py')
    command=[sys.executable,str(reporter),'--control',str(args.control),
        '--control-label',args.control_label,'--wider-encoders',str(args.wider_encoders),
        '--wider-label',args.wider_label,
        '--out',str(args.out),'--comparison-step','100000','--live']
    if args.wider_low_regularization:
        command.extend(('--wider-low-regularization',str(args.wider_low_regularization)))
    trials=(args.control,args.wider_encoders)
    if args.wider_low_regularization:trials+=(args.wider_low_regularization,)
    while True:
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        statuses=[json.loads((path/'run.json').read_text()) for path in trials]
        print(' '.join(f'{path.name}:{status["status"]}@{status["gradient_steps"]}'
            for path,status in zip(trials,statuses)),flush=True)
        if all(status['status'] in ('complete','failed') for status in statuses):break
        time.sleep(args.interval_seconds)


if __name__=='__main__':main()
