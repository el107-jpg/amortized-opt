#!/usr/bin/env python3
"""Prototype factual pre-first-hit rows from audited Isaac plan trajectories."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import shift_clip
from isaac_rl.hand_plan_ranker import ACTION_BANK,encode_clip,cartesian_tips
from scripts.train_chart_cma import write_json


def audited_reports(folder):
    reports={}
    for path in sorted(folder.glob('*/worker_*/request_*.json')):
        if '_progress' in path.name:continue
        payload=json.loads(path.read_text())
        for report in payload.get('reports',[]):
            if report is None:continue
            if 'rollout_audit' not in report:raise ValueError('Path trace was not collected')
            hands=''.join('R' if hit['hand']=='right' else 'L' for hit in report['per_hit_outcomes'])
            if len(hands)!=4 or hands in reports:raise ValueError('Prototype requires one unique four-single-hit plan')
            reports[hands]=report
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--case',type=int,default=168)
    p.add_argument('--trace-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fractions',type=float,nargs='+',default=[.25,.5,.75])
    cli=p.parse_args()
    dataset=json.loads((cli.dataset/'config.json').read_text())
    if dataset['rollout_runtime']['rollout_runtime_version']!='matched_decoder_speed_contact_reset_v2':
        raise ValueError('Source case metadata was expected to be the archived v2 dataset')
    batch=torch.load(cli.dataset/'batches/batch_00001.pt',map_location='cpu',weights_only=False)['batch']
    clip=batch['panel'][cli.case];state=batch['states'][cli.case]
    reports=audited_reports(cli.trace_root)
    if len(reports)!=16:raise ValueError('Need all 16 audited plans for this prototype')
    first=min(hit.time for hit in clip.hits)
    dt=.02;times=sorted({round(round(first*f/dt)*dt,8) for f in cli.fractions
        if 0<f<1 and round(first*f/dt)*dt<=first-.10})
    rows=[];geometry=dataset['tip_geometry']
    for pattern,report in sorted(reports.items()):
        action=int(np.array([hand=='R' for hand in pattern])@np.array([8,4,2,1]))
        assert ''.join('R' if hand else 'L' for hand in ACTION_BANK[action])==pattern
        trace=report['rollout_audit']['trace']
        for time_s in times:
            tick=int(round(time_s/dt))
            if trace['hits'][tick-1]!=0:continue
            q=np.asarray(trace['q'][tick],np.float32)
            qd=np.asarray(trace['qd'][tick],np.float32)
            tips=cartesian_tips(q,geometry)
            remaining=shift_clip(clip,-time_s)
            obs=encode_clip(remaining,q=q,qd=qd,tips=tips,
                initial_drums=state['initial_drums'],tip_geometry=geometry)
            if abs(obs[19]-(first-time_s))>1e-6:
                raise ValueError('First deadline did not shift to time-to-hit')
            rows.append(dict(observation=obs,action=action,time_s=time_s,
                remaining_first_delay_s=first-time_s,factual_future_hits=report['hits'],
                factual_future_requested=report['requested'],
                full_episode_reward_metadata=report['reward'],
                contact_age_ticks=trace['contact_age'][tick],source_case=cli.case,
                source_plan=pattern))
    if not rows:raise ValueError('No eligible pre-first-hit snapshots')
    packed=dict(format='factual_prefirst_hit_prototype_v1',
        purpose='heldout_shape_smoke_not_training',
        rollout_runtime_version='thick_cymbal_top_entry_side5mm_contact_v4',
        source_case=cli.case,source_clip=clip.name,
        observations=np.stack([row['observation'] for row in rows]),
        actions=np.asarray([row['action'] for row in rows],np.int64),
        future_hit_rate=np.asarray([row['factual_future_hits']/row['factual_future_requested'] for row in rows],np.float32),
        snapshot_times_s=np.asarray([row['time_s'] for row in rows],np.float32),
        full_episode_reward_metadata=np.asarray([row['full_episode_reward_metadata'] for row in rows],np.float32),
        rows=[{k:v for k,v in row.items() if k!='observation'} for row in rows])
    cli.out.parent.mkdir(parents=True,exist_ok=True);torch.save(packed,cli.out)
    write_json(cli.out.with_suffix('.json'),dict(format=packed['format'],purpose=packed['purpose'],
        source_case=cli.case,plans=len(reports),snapshot_times_s=times,rows=len(rows),
        shape=list(packed['observations'].shape),reward_label='Future hit fraction only; full-episode reward stored as metadata, not a return-to-go',
        training_allowed=False))
    print(cli.out,len(rows),packed['observations'].shape,flush=True)


if __name__=='__main__':main()
