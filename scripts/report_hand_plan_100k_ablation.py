#!/usr/bin/env python3
"""Compare fixed-step fresh-label ranker ablations without selecting checkpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--control-label',default='Control')
    parser.add_argument('--low-regularization',type=Path)
    parser.add_argument('--wider-encoders',type=Path,required=True)
    parser.add_argument('--wider-label',default='Wider encoders')
    parser.add_argument('--wider-low-regularization',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--comparison-step',type=int,default=10000)
    parser.add_argument('--live',action='store_true',help='Expand the chart to the latest available evaluation as runs progress')
    args=parser.parse_args()
    runs={args.control_label:args.control}
    if args.low_regularization:
        runs['Less regularization']=args.low_regularization
    runs[args.wider_label]=args.wider_encoders
    if args.wider_low_regularization:
        runs['Wider + less regularization']=args.wider_low_regularization
    histories={}
    configs={}
    for label,path in runs.items():
        configs[label]=json.loads((path/'config.json').read_text())
        histories[label]=[json.loads(line) for line in (path/'evaluations.jsonl').read_text().splitlines()
                          if line.strip()]
        histories[label]=[record for record in histories[label]
                          if record['gradient_steps']<=args.comparison_step]
        if not histories[label]:raise ValueError(f'{label}: no evaluations')
    datasets={config['dataset'] for config in configs.values()}
    splits={json.dumps(config['split_counts'],sort_keys=True) for config in configs.values()}
    runtimes={json.dumps(config['rollout_runtime'],sort_keys=True) for config in configs.values()}
    if len(datasets)!=1 or len(splits)!=1 or len(runtimes)!=1:
        raise ValueError('Runs must use the same dataset, splits and rollout runtime')
    summary=dict(dataset=next(iter(datasets)),comparison_step=args.comparison_step,
                 checkpoint_selection='latest weights at fixed optimizer step; no held-out selection',runs={})
    for label,path in runs.items():
        at_step=next((record for record in histories[label]
                      if record['gradient_steps']==args.comparison_step),None)
        summary['runs'][label]=dict(path=str(path.resolve()),parameters=configs[label]['neural_parameters'],
            config={key:configs[label]['args'][key] for key in
                    ('weight_decay','dropout','gradient_l1','batch','steps','seed')},
            group_width=configs[label]['network'].get('group_width',64),
            state_width=configs[label]['network'].get('state_width',64),
            latest_evaluated_update=histories[label][-1]['gradient_steps'],
            fixed_step_metrics=None if at_step is None else at_step['metrics'])
    args.out.mkdir(parents=True,exist_ok=True)
    temp_json=args.out/'comparison.tmp.json'
    temp_json.write_text(json.dumps(summary,indent=2)+'\n')
    temp_json.replace(args.out/'comparison.json')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    series=(('train','loss','Training ranking loss',1),
            ('test','loss','Test ranking loss',1),
            ('test','hit_rate','Test chosen-plan hit rate (%)',100))
    latest=max(history[-1]['gradient_steps'] for label,history in histories.items()
               if label!=args.control_label)
    x_limit=min(args.comparison_step,max(1000,latest)) if args.live else args.comparison_step
    for label,history in histories.items():
        x=[record['gradient_steps'] for record in history]
        for axis,(split,key,title,scale) in zip(axes,series):
            axis.plot(x,[scale*record['metrics'][split][key] for record in history],
                      marker='o',markersize=2,label=label)
            axis.set(title=title,xlabel='Optimizer updates',xlim=(0,x_limit))
            axis.grid(alpha=.25)
    axes[0].legend()
    title=(f'Fresh 100k-case hand-plan ranking: live through {latest:,} updates'
           if args.live else
           f'Fresh 100k-case hand-plan ranking: matched first {args.comparison_step/1000:g}k updates')
    fig.suptitle(title)
    temp_png=args.out/'comparison.tmp.png'
    fig.savefig(temp_png,dpi=170)
    temp_png.replace(args.out/'comparison.png')
    plt.close(fig)
    print(json.dumps({label:dict(step=value['latest_evaluated_update'],
        test_hit_rate=None if value['fixed_step_metrics'] is None else
        value['fixed_step_metrics']['test']['hit_rate'])
        for label,value in summary['runs'].items()},indent=2))


if __name__=='__main__':main()
