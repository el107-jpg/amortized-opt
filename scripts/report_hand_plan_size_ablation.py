#!/usr/bin/env python3
"""Read-only matched-budget plots for the scorer-size/gradient-penalty ablation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_chart_cma import write_json


def read_history(path):
    lines=path.read_text().splitlines();rows=[]
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
    return {r['gradient_steps']:r for r in rows}


def final_checks(folder,manifest,configs):
    """Latest fixed-budget weights only; paired case statistics never feed fitting."""
    if (folder/'final_pairwise.json').exists():return
    import numpy as np
    import torch
    from isaac_rl.hand_plan_dataset import load_dataset_batches
    from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig
    from isaac_rl.hand_plan_gradient_regularizer import InputGradientConfig,input_gradient_l1
    torch.set_num_threads(2)
    data=load_dataset_batches(Path(manifest['dataset']))
    splits=torch.load(Path(manifest['dataset'])/'splits.pt',weights_only=False)
    jac=torch.load(Path(manifest['runs'][2]['path'])/'input_tip_jacobians.pt',weights_only=False)['jacobians']
    cases=[];gradients=[]
    ids=np.asarray(splits['test']);rows=np.arange(len(ids))
    rewards=data['rewards'][ids];legal=data['observed'][ids]
    best=np.where(legal,rewards,-np.inf).max(1)
    for run,config in zip(manifest['runs'],configs):
        state=torch.load(Path(run['path'])/'last.pt',map_location='cpu',weights_only=False)
        if state['gradient_steps']!=config['args']['steps']:raise ValueError('Final comparison requires complete fixed-budget weights')
        model=HandPlanRanker(HandPlanRankerConfig(**state['network']));model.load_state_dict(state['model'])
        model.eval().requires_grad_(False)
        with torch.inference_mode():scores=model(torch.as_tensor(data['observations'][ids])).numpy()
        chosen=np.where(legal,scores,-np.inf).argmax(1)
        cases.append(dict(hits=data['hits'][ids,chosen],requested=data['requested'][ids,chosen],
            regret=best-rewards[rows,chosen]))
        scales=InputGradientConfig(config['args'].get('gradient_timing_scale_s',.02),
            np.deg2rad(config['args'].get('gradient_position_scale_deg',1.)),
            config['args'].get('gradient_velocity_scale_rad_s',.1),config['args'].get('gradient_loudness_scale',.02))
        stats={}
        for i,(split,probe_ids) in enumerate(splits.items()):
            probe=np.asarray(probe_ids[:256]);generator=torch.Generator().manual_seed(config['args']['seed']+88481+i)
            _,values=input_gradient_l1(model,torch.as_tensor(data['observations'][probe]),
                torch.as_tensor(data['observed'][probe]),jac[probe],generator=generator,config=scales,create_graph=False)
            stats[split]={key:float(value) for key,value in values.items()}
        gradients.append(dict(label=run['label'],metrics=stats))
    # Shared whole-case resamples preserve pairing and target count differences.
    rng=np.random.default_rng(20261008);sample=rng.integers(len(ids),size=(2000,len(ids)))
    comparisons=[]
    for name,a,b in [('size_effect_gradient_off',0,3),('size_effect_gradient_on',1,2),
                     ('gradient_effect_small',0,1),('gradient_effect_large',3,2)]:
        left,right=cases[a],cases[b]
        hit=100*(right['hits'][sample].sum(1)/right['requested'][sample].sum(1)
            -left['hits'][sample].sum(1)/left['requested'][sample].sum(1))
        regret=(right['regret'][sample]-left['regret'][sample]).mean(1)
        comparisons.append(dict(effect=name,from_condition=manifest['runs'][a]['label'],to_condition=manifest['runs'][b]['label'],
            hit_rate_difference_pp=100*(right['hits'].sum()/right['requested'].sum()-left['hits'].sum()/left['requested'].sum()),
            hit_rate_difference_95pct_case_bootstrap_ci=np.quantile(hit,[.025,.975]).tolist(),
            mean_regret_difference=float((right['regret']-left['regret']).mean()),
            regret_difference_95pct_case_bootstrap_ci=np.quantile(regret,[.025,.975]).tolist()))
    write_json(folder/'final_input_gradients.json',dict(device='cpu',gradient_steps=configs[0]['args']['steps'],
        same_probe_cases_and_sampled_legal_plans=True,conditions=gradients,used_for_training=False))
    write_json(folder/'final_pairwise.json',dict(gradient_steps=configs[0]['args']['steps'],test_cases=len(ids),
        comparisons=comparisons,latest_fixed_budget_weights=True,held_out_used_for_training=False,
        weights_selected_by_comparison=False,limitation='Case-bootstrap uncertainty conditional on one training seed and archived simulator labels; not multi-seed variability.'))


def report(folder):
    manifest=json.loads((folder/'config.json').read_text());histories=[];configs=[];latest=[];statuses=[]
    for run in manifest['runs']:
        path=Path(run['path'])
        if not (path/'evaluations.jsonl').exists():return None
        config=json.loads((path/'config.json').read_text());history=read_history(path/'evaluations.jsonl')
        if not history:return None
        if config['network']['hidden']!=run['hidden'] or config['args'].get('gradient_l1',0.)!=run['gradient_l1']:
            raise ValueError('Run differs from ablation condition')
        if Path(config['dataset']).resolve()!=Path(manifest['dataset']).resolve():raise ValueError('Ablation dataset differs')
        if configs:
            for key in ('steps','batch','seed','lr','min_lr','warmup','weight_decay','dropout'):
                if config['args'][key]!=configs[0]['args'][key]:raise ValueError(f'Ablation must preserve {key}')
            for key in ('split_counts','checkpoint_sha256','simulation_geometry_version'):
                if config[key]!=configs[0][key]:raise ValueError(f'Ablation differs in {key}')
        configs.append(config);histories.append(history)
        step=max(history);progress=json.loads((path/'progress.json').read_text()) if (path/'progress.json').exists() else {}
        status=json.loads((path/'run.json').read_text()) if (path/'run.json').exists() else {}
        statuses.append(status.get('status','starting'))
        latest.append(dict(label=run['label'],parameters=config['neural_parameters'],
            latest_evaluated_step=step,latest_training_step=progress.get('step',step),metrics=history[step]['metrics']))
    common=sorted(set.intersection(*(set(h) for h in histories)))
    if not common:return None
    step=common[-1];complete=all(s=='complete' for s in statuses)
    if complete:final_checks(folder,manifest,configs)
    result=dict(status='complete' if complete else 'failed' if 'failed' in statuses else 'partial',
        matched_step=step,latest=latest,matched=[dict(label=r['label'],parameters=c['neural_parameters'],
            metrics=h[step]['metrics']) for r,c,h in zip(manifest['runs'],configs,histories)],
        held_out_used_for_training=False,weights_selected_by_comparison=False,
        comparison='Same dataset/splits, seed, optimizer budget, dropout, decay and physical gradient scales; scorer width and gradient coefficient crossed',
        limitation='One training seed; archived legacy simulator labels. Curves use shared evaluated budgets; latest metrics are also reported separately.')
    write_json(folder/'comparison.json',result)
    metrics=[('train','mean_reward_regret','Training reward regret'),('test','mean_reward_regret','Test reward regret'),
        ('test','hit_rate','Test selected-plan hit rate (%)'),('test','pair_accuracy','Test pair ordering accuracy (%)'),
        ('test','prediction_entropy_normalized','Test legal-plan entropy / log(count)'),('test','gradient_l1','Test relative-score input Jacobian L1')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for run,history in zip(manifest['runs'],histories):
        color='tab:blue' if run['hidden']==224 else 'tab:orange';style='--' if run['gradient_l1']==0 else '-'
        for ax,(split,key,label) in zip(axes.flat,metrics):
            rows=[history[s] for s in common if key in history[s]['metrics'][split]]
            scale=100 if key in ('hit_rate','pair_accuracy') else 1
            ax.plot([r['gradient_steps'] for r in rows],[scale*r['metrics'][split][key] for r in rows],
                linestyle=style,color=color,marker='o',ms=2,label=run['label'])
    for ax,(_,_,label) in zip(axes.flat,metrics):
        ax.set_xlabel('Matched optimizer steps');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    if not any('gradient_l1' in histories[0][s]['metrics']['test'] for s in common):
        axes[1,2].text(.02,.98,'Small unregularized baseline: gradient trace unavailable',
            transform=axes[1,2].transAxes,va='top',fontsize=7)
    fig.suptitle(f'Scorer size × input-gradient L1: matched through {step:,} updates\n224/112 (106,641 parameters) vs448/224 (254,481); penalty0 vs0.1')
    temp=folder/'ablation.tmp.png';fig.savefig(temp,dpi=160);temp.replace(folder/'ablation.png');plt.close(fig)
    with (folder/'matched_metrics.tmp.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','parameters','matched_step','train_hit_rate','test_hit_rate','train_regret','test_regret','test_pair_accuracy'])
        for row in result['matched']:
            m=row['metrics'];writer.writerow([row['label'],row['parameters'],step,m['train']['hit_rate'],m['test']['hit_rate'],m['train']['mean_reward_regret'],m['test']['mean_reward_regret'],m['test']['pair_accuracy']])
    (folder/'matched_metrics.tmp.csv').replace(folder/'matched_metrics.csv')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=float,default=30.)
    args=parser.parse_args()
    if args.interval<=0:raise ValueError('Positive reporting interval required')
    last=None
    while True:
        result=report(args.run)
        if result:
            signature=(result['matched_step'],tuple(x['latest_evaluated_step'] for x in result['latest']))
            if signature!=last:
                print(json.dumps(dict(status=result['status'],matched_step=result['matched_step'],
                    latest_steps={x['label']:x['latest_evaluated_step'] for x in result['latest']})),flush=True)
                last=signature
            if result['status'] in ('complete','failed'):break
        if not args.watch:break
        time.sleep(args.interval)


if __name__=='__main__':main()
