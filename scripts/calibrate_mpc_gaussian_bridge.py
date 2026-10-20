#!/usr/bin/env python3
"""Calibrate Gaussian MPC perturbations from full-plan Isaac hit diagnostics."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF
from amortized_opt.gaussian_bridge import GaussianBridgeNoise, sample_bridge
from amortized_opt.tspline import TrajectorySplineBuilder


def gaussian_scale(x,axis=0):
    """Zero-mean normal scale matching empirical median absolute error."""
    return np.nanmedian(np.abs(x),axis=axis)/0.6744897501960817


def collect(payload):
    events = payload['events'];dt=float(payload['metadata']['dt'])
    states={}
    for e in events:
        for row in e['approach']:
            if 'plan_q' not in row:
                raise ValueError('trace requires --hit_diagnostics_full_plans')
            states[round(row['time_s']/dt)] = row
    H=len(next(iter(states.values()))['plan_q'])
    builder=TrajectorySplineBuilder(max_segments=H,dt=dt,dtype=torch.float64)
    records=[];terminal=[];seen=set();names=G1_UPPER_BODY_JOINTS
    # Build each plan's actual spline derivative once; duplicate event rows
    # share the same plan and measured state.
    velocities={}
    for tick,row in states.items():
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)[None]
        sp=builder.from_masks(tensor(row['q']),tensor(row['plan_q']),tensor(row['qd']),tensor(row['plan_elem_mask']))
        velocities[tick]=sp.eval_offsets(torch.tensor([dt],dtype=torch.float64))[1][0,:,0].numpy()
    for eid,e in enumerate(events):
        cols=np.asarray([j for j,n in enumerate(names) if n.startswith(e['hand']+'_')])
        # Waist samples are counted once across simultaneous hand requests.
        waist=np.asarray([j for j,n in enumerate(names) if n.startswith('waist_')])
        for row in e['approach']:
            tick=round(row['time_s']/dt);K=int(row['ticks_to_hit']);end=tick+K
            plan=np.asarray(row['plan_q']);em=np.asarray(row['plan_elem_mask']);vplan=velocities[tick]
            for age in range(1,K+1):
                future=states.get(tick+age)
                if future is None:continue
                qerror=np.asarray(future['q'])-plan[age-1]
                verror=np.asarray(future['qd'])-vplan[age-1]
                valid=np.zeros(N_DOF,dtype=bool)
                valid[cols]=em[age-1,cols]>.5
                for j in waist:
                    key=(tick,tick+age,int(j))
                    if key not in seen and em[age-1,j]>.5:
                        valid[j]=True;seen.add(key)
                # Arm overlap cannot repeat inside one hand/goal event.
                records.append((eid,age*dt,(K-age)*dt,np.where(valid,qerror,np.nan),np.where(valid,verror,np.nan)))
            future=states.get(end)
            if future is not None:
                qerror=np.asarray(future['q'])-plan[K-1]
                verror=np.asarray(future['qd'])-vplan[K-1]
                valid=np.zeros(N_DOF,dtype=bool);valid[cols]=True
                terminal.append((eid,K*dt,np.where(valid,qerror,np.nan),np.where(valid,verror,np.nan)))
    return records,terminal,H,dt


def summarize(records):
    event=np.asarray([r[0] for r in records]);age=np.asarray([r[1] for r in records]);remaining=np.asarray([r[2] for r in records])
    q=np.stack([r[3] for r in records]);v=np.stack([r[4] for r in records])
    rows=[]
    for lo,hi in [(0,.02),(.02,.10),(.10,.25),(.25,.50),(.50,1.01)]:
        take=(age>lo+1e-9)&(age<=hi+1e-9)
        if not take.any():continue
        qr=np.sqrt(np.nanmean(q[take]**2,axis=1));vr=np.sqrt(np.nanmean(v[take]**2,axis=1))
        rows.append(dict(age_lo_s=lo,age_hi_s=hi,rows=int(take.sum()),
            position_rms_median_rad=float(np.nanmedian(qr)),position_rms_p90_rad=float(np.nanpercentile(qr,90)),
            velocity_rms_median_rad_s=float(np.nanmedian(vr)),velocity_rms_p90_rad_s=float(np.nanpercentile(vr,90))))
    return event,age,remaining,q,v,rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--endpoint-fraction',type=float,default=.5)
    args=ap.parse_args();payload=json.loads(args.trace.read_text());records,terminal,H,dt=collect(payload)
    event,age,remaining,q,v,bands=summarize(records)
    split=max(1,int(len(payload['events'])*2/3));train=event<split
    tq=np.stack([r[2] for r in terminal if r[0]<split]);tv=np.stack([r[3] for r in terminal if r[0]<split])
    # Arm-owned terminal samples cover the hands. Waist final samples use all
    # valid state/plan pairs because the two hands have different strike ends.
    covered=np.isfinite(tq).any(axis=0)
    end_q=np.full(N_DOF,np.nan);end_v=np.full(N_DOF,np.nan)
    end_q[covered]=gaussian_scale(tq[:,covered]);end_v[covered]=gaussian_scale(tv[:,covered])
    missing=~np.isfinite(end_q);end_q[missing]=gaussian_scale(q[train]) [missing]
    missing=~np.isfinite(end_v);end_v[missing]=gaussian_scale(v[train]) [missing]
    diffusion=[];state_diffusion=[]
    # Fit unconditioned integrated-Brownian state variance to empirical q/v
    # scales, with each family normalized by its empirical variance units.
    for j in range(N_DOF):
        aq=[];av=[];times=[]
        for a in np.unique(age[train]):
            take=train & np.isclose(age,a) & np.isfinite(q[:,j])
            if int(take.sum())<8:continue
            times.append(a);aq.append(float(gaussian_scale(q[take,j]))**2);av.append(float(gaussian_scale(v[take,j]))**2)
        times=np.asarray(times);aq=np.asarray(aq);av=np.asarray(av)
        if not times.size:raise ValueError(f'no calibration data for joint {j}')
        qref=max(float(np.median(aq)),1e-8);vref=max(float(np.median(av)),1e-8)
        features=np.concatenate([times**3/3/np.sqrt(qref),times/np.sqrt(vref)])
        target=np.concatenate([np.maximum(aq-.01**2-times**2*.05**2,0)/np.sqrt(qref),np.maximum(av-.05**2,0)/np.sqrt(vref)])
        d2=max(float(features@target/max(features@features,1e-20)),0)
        state_diffusion.append(np.sqrt(d2))
        # Same disturbance spectrum, conditioned to pull back near the strike.
        diffusion.append(np.sqrt(d2))
    table_times=[0.];table=[[(.01**2,.05**2,0.) for _ in range(N_DOF)]]
    for lo,hi in [(0,.02),(.02,.10),(.10,.25),(.25,.50),(.50,1.01)]:
        group=train & (age>lo+1e-9) & (age<=hi+1e-9)
        if not group.any():continue
        row=[]
        for j in range(N_DOF):
            take=group & np.isfinite(q[:,j])
            qs=float(gaussian_scale(q[take,j]));vs=float(gaussian_scale(v[take,j]))
            if not np.isfinite(qs) or not np.isfinite(vs):
                raise ValueError(f'insufficient covariance coverage for joint {j}')
            if qs>1e-10 and vs>1e-10:
                r=float(np.corrcoef(np.clip(q[take,j]/qs,-3,3),np.clip(v[take,j]/vs,-3,3))[0,1])
                r=float(np.clip(r,-.95,.95)) if np.isfinite(r) else 0.
            else:r=0.
            row.append((qs**2,vs**2,r*qs*vs))
        table_times.append(float(np.median(age[group])));table.append(row)
    cfg=GaussianBridgeNoise(velocity_diffusion=tuple(diffusion),
        terminal_position_std=tuple(args.endpoint_fraction*end_q),terminal_velocity_std=tuple(args.endpoint_fraction*end_v),
        state_velocity_diffusion=tuple(state_diffusion),state_times_s=tuple(table_times),state_covariance=tuple(table))
    held=~train
    report=dict(trace=str(args.trace.resolve()),metadata=payload['metadata'],events=len(payload['events']),
        pairs=len(records),terminal_pairs=len(terminal),fit_events=split,heldout_events=len(payload['events'])-split,
        distribution='zero-mean Gaussian; robust std = median(abs(error))/0.67449; Gaussian Brownian velocity increments integrated into position',
        endpoint_fraction=args.endpoint_fraction,tracking_drift_by_elapsed_time=bands,
        empirical_terminal_position_std_rad=end_q.tolist(),empirical_terminal_velocity_std_rad_s=end_v.tolist(),
        heldout_position_gaussian_std_rad=gaussian_scale(q[held]).tolist(),heldout_velocity_gaussian_std_rad_s=gaussian_scale(v[held]).tolist(),
        limitations=['One actual Isaac clip, free waist, contact compliance, 5-waypoint freeze and fit_kd4 lead.',
          'Samples within events overlap; pair count is not independent trajectory count.',
          'Errors include tracking, contacts and subsequent replanning; they are not sensor noise.',
          'Zero-mean Gaussian is the requested augmentation model; empirical bias and heavy tails are not assumed Gaussian.',
          'State errors use a time-indexed Gaussian covariance table because measured velocity uncertainty is roughly flat rather than increasing with elapsed time.',
          'The Brownian bridge diffusion is calibrated from empirical joint position/velocity spread; it is a smooth augmentation family rather than a claim that actual tracking errors follow Brownian dynamics.',
          'Bridge conditioning preserves the original joint-space endpoint plus nonzero Gaussian endpoint jitter; drum feasibility remains the physics loss responsibility.',
          'Joint-limit projection may alter endpoint velocity and makes the applied distribution non-Gaussian.'],sampler=asdict(cfg))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.out.with_suffix('.npz'),event=event,elapsed_s=age,remaining_s=remaining,position_error=q,velocity_error=v)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    grid=np.unique(age)
    for ax,data,label in [(axs[0,0],q,'Position error (rad)'),(axs[0,1],v,'Velocity error (rad/s)')]:
        center=[];upper=[]
        for s in grid:
            take=np.isclose(age,s);rms=np.sqrt(np.nanmean(data[take]**2,axis=1))
            center.append(np.nanmedian(rms));upper.append(np.nanpercentile(rms,90))
        ax.plot(grid,center,label='Measured median RMS');ax.plot(grid,upper,label='Measured p90 RMS',alpha=.6)
        idx=0 if data is q else 1
        table_var=np.asarray(cfg.state_covariance)[:,:,idx].mean(1)
        predicted=np.sqrt(np.interp(grid,cfg.state_times_s,table_var))
        ax.plot(grid,predicted,'--',label='Calibrated Gaussian RMS');ax.set(xlabel='Time advanced from parent plan (s)',ylabel=label);ax.legend(fontsize=8)
    mask=torch.ones(512,H,N_DOF);noise,eq,ev=sample_bridge(mask,dt,cfg,generator=torch.Generator().manual_seed(17),return_endpoints=True)
    t=np.arange(1,H+1)*dt
    ax=axs[1,0];j=3
    for n in range(12):ax.plot(t,noise[n,:,j].numpy(),alpha=.55,lw=1)
    ax.set(xlabel='Time within a one-second initialization (s)',ylabel=f'{G1_UPPER_BODY_JOINTS[j]} offset (rad)',title='Gaussian bridges with nonzero endpoint errors')
    ax=axs[1,1];std=noise.std(0).numpy();ax.plot(t,np.sqrt(np.mean(std**2,axis=1)),label='Bridge position RMS std')
    ax.set(xlabel='Time within initialization (s)',ylabel='Across-joint position std (rad)',title='Endpoint jitter stays significant');ax.legend(fontsize=8)
    fig.suptitle('Actual Isaac drift and calibrated Gaussian bridge (step 32250)');fig.tight_layout();fig.savefig(args.out.with_suffix('.png'),dpi=150);plt.close(fig)
    print(json.dumps({'events':len(payload['events']),'pairs':len(records),'bands':bands,'terminal_position_std_median_rad':float(np.median(end_q)),
      'configured_endpoint_position_std_median_rad':float(np.median(cfg.terminal_position_std)),'configured_endpoint_velocity_std_median_rad_s':float(np.median(cfg.terminal_velocity_std))},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
