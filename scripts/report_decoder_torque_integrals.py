"""Report accumulated/average overload for saved paths and threshold sweeps."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from isaac_rl.trajectory_qp import cost_for_horizon
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance, profile
from scripts.fill_transition_table_gaps import context, load_piece, setup
from scripts.probe_transition_qp import write_json


def distribution(values):
    values=np.asarray(values,float)
    return dict(count=len(values),median=float(np.median(values)),p90=float(np.quantile(values,.9)),
        maximum=float(values.max()),mean=float(values.mean()))


def saved_report(args):
    torch.set_num_threads(2)
    _,cost,kit=setup(args.checkpoint,args.device)
    names=list(kit.names)
    allowance=SmallViolationAllowance(torque_peak_ratio=1.2,torque_duration_fraction=.1,
        torque_continuous_s=2.,collision_depth_m=.01,joint_excess_rad=np.deg2rad(5),
        terminal_distance_m=.01,torque_average_limit_Nm=args.threshold)
    rows=[];joint_rows=[];costs={}
    for filename in sorted(args.saved_run.glob('w*/h*_source_*.npz')):
        piece=load_piece(filename);h,src=int(piece['hand']),int(piece['source'])
        live=np.flatnonzero(np.isfinite(piece['minimum_s']))
        for N in sorted(set(np.rint(piece['minimum_s'][live]/.02).astype(int))):
            ids=live[np.rint(piece['minimum_s'][live]/.02).astype(int)==N]
            if N not in costs:
                costs[N]=cost_for_horizon(cost,N);costs[N].cfg=copy.copy(cost.cfg);costs[N].cfg.n_sub=32
            q=torch.tensor(piece['q'][ids],device=cost.device)
            v=torch.tensor(piece['v'][ids],device=cost.device)
            path=torch.tensor(piece['paths'][ids,:N],device=cost.device)
            _,ctx,L=context(kit,h,[src]*len(ids),piece['destination'][ids],N,cost.device)
            measured=profile(costs[N],q,v,path,ctx,L,depth_m=.07,allowance=allowance)
            for i,row in enumerate(ids):
                key=dict(hand=('left','right')[h],source=names[src],destination=names[int(piece['destination'][row])],
                    sample=int(piece['sample'][row]),duration_s=float(N*.02))
                rows.append(dict(**key,integral_Nms=float(measured['torque_excess_integral_Nms'][i]),
                    average_excess_Nm=float(measured['torque_excess_average_Nm'][i]),
                    worst_joint_average_excess_Nm=float(measured['torque_excess_average_max_joint_Nm'][i]),
                    peak_excess_Nm=float(measured['torque_excess_max'][i]),
                    passes_provisional_threshold=bool(measured['average_success'][i])))
                for j,name in enumerate(cost.joint_names):
                    joint_rows.append(dict(**key,joint=name,
                        integral_Nms=float(measured['torque_excess_integral_per_joint_Nms'][i,j]),
                        average_excess_Nm=float(measured['torque_excess_average_per_joint_Nm'][i,j])))
    if not rows: raise ValueError('No saved trajectories to measure')
    args.out.mkdir(parents=True,exist_ok=True)
    for name,data in [('previous_relaxed_paths.csv',rows),('previous_relaxed_per_actuator.csv',joint_rows)]:
        with (args.out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary=dict(source_run=str(args.saved_run),trajectories=len(rows),dense_substeps=32,
        formula='sum_j integral max(abs(tau_j)-available_j,0) dt / trajectory_duration',
        integral_units='Nm s',average_units='Nm',threshold_average_Nm=args.threshold,
        paths_exceeding_threshold=sum(row['average_excess_Nm']>args.threshold+1e-6 for row in rows),
        integral_Nms=distribution([r['integral_Nms'] for r in rows]),
        average_excess_Nm=distribution([r['average_excess_Nm'] for r in rows]),
        worst_joint_average_excess_Nm=distribution([r['worst_joint_average_excess_Nm'] for r in rows]),
        maximum_case=max(rows,key=lambda r:r['average_excess_Nm']))
    write_json(args.out/'previous_relaxed_summary.json',summary)
    print(json.dumps(summary),flush=True)


def threshold_support(parts,threshold):
    keys=[]
    for data in parts:
        take=data['relaxed_non_torque_success']&(data['torque_excess_average_Nm']<=threshold+1e-6)
        keys.extend(zip(data['hand'][take],data['source'][take],data['destination'][take],data['sample'][take]))
    starts=set(keys);pairs={key[:3] for key in starts}
    return dict(threshold_average_excess_Nm=float(threshold),supported_pairs=len(pairs),supported_starts=len(starts))


def sweep_report(args):
    files=sorted(args.sweep_run.glob('measurements_h*_source_*.npz'))
    if json.loads((args.sweep_run/'run.json').read_text())['status']!='complete':
        raise ValueError('Wait for the full sweep before reporting a threshold curve')
    parts=[load_piece(path) for path in files]
    curve=[threshold_support(parts,t) for t in (0.,.05,.1,.25,.5,1.,2.,5.,10.,20.)]
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'threshold_support.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
    candidates=np.concatenate([p['torque_excess_average_Nm'][p['relaxed_non_torque_success']] for p in parts])
    saved=[];saved_rows=[];names=list(A.DrumKit.from_layout().names)
    for path in (args.sweep_run/'average').glob('w*/h*_source_*.npz'):
        data=load_piece(path)
        take=np.isfinite(data['minimum_s'])
        saved.extend(data['torque_excess_average_Nm'][take].tolist())
        for i in np.flatnonzero(take):
            saved_rows.append(dict(hand=('left','right')[int(data['hand'])],source=names[int(data['source'])],
                destination=names[int(data['destination'][i])],sample=int(data['sample'][i]),
                duration_s=float(data['minimum_s'][i]),integral_Nms=float(data['torque_excess_integral_Nms'][i]),
                average_excess_Nm=float(data['torque_excess_average_Nm'][i]),
                worst_joint_average_excess_Nm=float(data['torque_excess_average_max_joint_Nm'][i]),
                peak_excess_Nm=float(data['torque_excess_max'][i]),
                collision_overlap_mm=float(max(0.,-data['collision_min_sdf'][i])*1000),
                joint_excess_deg=float(np.rad2deg(data['joint_box_excess_max'][i])),
                hit_miss_mm=float(data['terminal_region_dist'][i]*1000)))
    with (args.out/'new_minimum_paths.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(saved_rows[0]));writer.writeheader();writer.writerows(saved_rows)
    summary=dict(measured_dense_candidates=sum(len(p['sample']) for p in parts),
        candidates_passing_other_relaxed_gates=len(candidates),threshold_curve=curve,
        candidates_rejected_above_average_limit=int((candidates>args.threshold+1e-6).sum()),
        candidates_within_average_limit=int((candidates<=args.threshold+1e-6).sum()),
        all_other_gate_passing_candidates_average_excess_Nm=distribution(candidates),
        shortest_saved_paths_at_threshold_average_excess_Nm=distribution(saved),
        shortest_saved_paths_integral_Nms=distribution([r['integral_Nms'] for r in saved_rows]),
        shortest_saved_paths_peak_excess_Nm=distribution([r['peak_excess_Nm'] for r in saved_rows]),
        threshold_average_Nm=args.threshold,formula='sum of time-integrated positive active-arm/waist excess divided by trajectory duration',
        integral_units='Nm s',average_units='Nm')
    write_json(args.out/'threshold_summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key,label,maximum in [(axes[0],'supported_pairs','Supported drum pairs',85),
                                 (axes[1],'supported_starts','Supported starting samples',680)]:
        ax.plot([r['threshold_average_excess_Nm'] for r in curve],[r[key] for r in curve],'o-')
        ax.set_xscale('symlog',linthresh=.05)
        ax.set_xlabel('Allowed average total torque excess (Nm)')
        ax.set_ylabel(label);ax.set_ylim(0,maximum)
        ax.axvline(args.threshold,color='#c65431',linestyle='--',label=f'Provisional {args.threshold:g} Nm')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Decoder-only coverage versus integrated torque allowance\n'
        'Same original starts; other relaxed constraints held fixed')
    fig.savefig(args.out/'threshold_support.png',dpi=160,bbox_inches='tight')
    fig.savefig(args.out/'threshold_support.svg',bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-run',type=Path)
    parser.add_argument('--sweep-run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/phase2_clean_lines_fromp1_20261001/last.pt')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threshold',type=float,default=5.)
    args=parser.parse_args()
    if args.saved_run:saved_report(args)
    if args.sweep_run:sweep_report(args)


if __name__=='__main__':main()
