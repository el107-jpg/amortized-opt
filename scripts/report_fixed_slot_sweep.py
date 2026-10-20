#!/usr/bin/env python3
"""Read completed fixed-slot sweep batches and publish partial plots/statistics."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from isaac_rl.hand_plan_sweep import sensitivity_summary,AXES
from scripts.sweep_hand_plan_continuity import plot_results
from scripts.train_chart_cma import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--plots',action='store_true')
    cli=p.parse_args();files=sorted((cli.run/'batches').glob('round_*.json'));reports=[];points=[]
    for f in files:
        r=json.loads(f.read_text());reports.extend(r['reports']);points.extend(r['points'])
    config=json.loads((cli.run/'config.json').read_text())
    summary=sensitivity_summary(points,reports,fresh_scene_baselines=config.get('recreate_every_point',False))
    summary.update(status='partial',completed_rounds=len(files),actual_rollouts=sum(r is not None for r in reports),
        recreate_every_point=config.get('recreate_every_point',False))
    live=cli.run/'live';live.mkdir(exist_ok=True);write_json(live/'summary.json',summary);write_json(live/'points.json',points)
    if cli.plots and all(summary['axes'][axis]['plan_neighbors']>0 for axis in AXES):
        plot_results(live,points,8)
    print(json.dumps(dict(completed_rounds=len(files),actual_rollouts=summary['actual_rollouts'],
        repeatability=summary['repeatability'],axes={a:{k:v for k,v in summary['axes'][a].items()
        if k in ['plan_neighbors','fraction_hit_credit_changed','fraction_one_point_hit_credit_reversals','reward_difference_p95']} for a in AXES})))


if __name__=='__main__':main()
