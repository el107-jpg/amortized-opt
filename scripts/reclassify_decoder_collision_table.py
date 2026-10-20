"""Tighten a completed decoder sweep's collision/terminal/average-torque gates.

Reuse the complete dense measurements to identify minimum grid durations.
Copy unchanged raw witnesses; regenerate and reverify changed witnesses using
the same original batched decoder inputs. No QP or trajectory repair.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import amortized_opt as A
from amortized_opt.ik import arm_joint_indices
from amortized_opt.trajectory_surrogate import DecoderHitPoseGenerator
from isaac_rl.trajectory_qp import cost_for_horizon
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance,profile
from scripts.build_decoder_only_transition_chart import empty_records,accept_decoder
from scripts.build_unrestricted_transition_chart import publish
from scripts.fill_transition_table_gaps import context,load_piece,setup
from scripts.probe_transition_qp import save_npz,write_json


def select_candidates(data,collision_m,terminal_m,torque_average_Nm):
    passing=(data['relaxed_non_torque_success']
        &(data['collision_min_sdf']>=-collision_m-1e-6)
        &(data['terminal_region_dist']<=terminal_m+1e-6)
        &(data['torque_excess_average_Nm']<=torque_average_Nm+1e-6))
    best={}
    for i in np.flatnonzero(passing):
        key=(int(data['destination'][i]),int(data['sample'][i]))
        order=(int(data['segments'][i]),int(data['yaw_hint_index'][i]))
        if key not in best or order<(int(data['segments'][best[key]]),int(data['yaw_hint_index'][best[key]])):
            best[key]=int(i)
    return best,passing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--collision-m',type=float,default=.005)
    parser.add_argument('--terminal-m',type=float,default=.01)
    parser.add_argument('--torque-average-nm',type=float,default=5.)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.sweep_run=args.sweep_run.resolve();args.out=args.out.resolve()
    if json.loads((args.sweep_run/'run.json').read_text())['status']!='complete':
        raise ValueError('A completed dense decoder sweep is required')
    if not 0<=args.collision_m<=.01 or not 0<=args.terminal_m<=.01:
        raise ValueError('Cached sweep covers collision/terminal allowances up to 1 cm')
    if not np.isfinite(args.torque_average_nm) or args.torque_average_nm<0:
        raise ValueError('Average torque allowance must be finite and nonnegative')
    args.out.mkdir(parents=True,exist_ok=False);folder=args.out/'w0';folder.mkdir()
    config=json.loads((args.sweep_run/'config.json').read_text());checkpoint=Path(config['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=config['checkpoint_sha256']:
        raise ValueError('Reclassification requires the identical frozen checkpoint')
    write_json(args.out/'config.json',dict(source_sweep=str(args.sweep_run),checkpoint=str(checkpoint),
        checkpoint_sha256=config['checkpoint_sha256'],collision_overlap_m=args.collision_m,
        terminal_error_m=args.terminal_m,torque_average_excess_Nm=args.torque_average_nm,
        joint_excess_deg=5.,decoder_only=True,qp_iterations=0,verification_substeps=32,
        duration_grid_s=[.02,2.],resolution_s=.02,cached_all_duration_classification=True))
    torch.set_num_threads(2);ck,cost,kit=setup(checkpoint,args.device)
    model=A.build_model(A.model_config_from_checkpoint(ck),args.device)
    model.load_state_dict(ck['model']);model.eval()
    generator=DecoderHitPoseGenerator(model,cost,kit,horizon=100,
        waist_upright_ease_wp=(ck.get('scenario_cfg')or{}).get('waist_upright_ease_wp'))
    allowance=SmallViolationAllowance(collision_depth_m=args.collision_m,
        joint_excess_rad=np.deg2rad(5),terminal_distance_m=args.terminal_m,
        torque_average_limit_Nm=args.torque_average_nm)
    costs={};reused=regenerated=0;started=time.monotonic()
    write_json(args.out/'run.json',dict(status='running'))
    try:
        for source in sorted((args.sweep_run/'inputs').glob('h*_source_*.npz')):
            original=load_piece(source);h,src=int(original['hand']),int(original['source'])
            data=load_piece(args.sweep_run/f'measurements_h{h}_source_{src}.npz')
            best,passing=select_candidates(data,args.collision_m,args.terminal_m,args.torque_average_nm)
            old_path=next((args.sweep_run/'average').glob(f'w*/{source.name}'))
            old=load_piece(old_path);piece=empty_records(original,100)
            for name in ('hand','source','q','v','destination','sample'):
                np.testing.assert_array_equal(old[name],original[name])
            assert (data['hand']==h).all() and (data['source']==src).all()
            piece.pop('coarse_outcomes')
            lookup={(int(dst),int(sample)):i for i,(dst,sample) in enumerate(zip(piece['destination'],piece['sample']))}
            for j in np.flatnonzero(passing):
                row=lookup[int(data['destination'][j]),int(data['sample'][j])]
                piece['raw_outcomes'][row,int(data['segments'][j])]=True
            q=torch.tensor(piece['q'],device=args.device);v=torch.tensor(piece['v'],device=args.device)
            regenerate={}
            for key,j in best.items():
                row=lookup[key];N=int(data['segments'][j]);yi=int(data['yaw_hint_index'][j])
                if (np.isfinite(old['minimum_s'][row]) and round(old['minimum_s'][row]/.02)==N
                        and int(old['decoder_yaw_hint_index'][row])==yi):
                    dense={name:values[j:j+1] for name,values in data.items()}
                    dense['average_success']=np.ones(1,bool)
                    accept_decoder(piece,[row],N,old['paths'][row:row+1,:N],dense,yi,'average_success')
                    reused+=1
                else:regenerate.setdefault((N,yi),[]).append(row)
            for (N,yi),rows in sorted(regenerate.items()):
                B=len(q)
                targets,ctx,L=context(kit,h,[src]*B,piece['destination'],N,args.device)
                with torch.no_grad():
                    plan=generator.predict(q,v,torch.tensor(targets[:,None],device=args.device),
                        q.new_full((B,1),N*.02),q.new_full((B,1,2),2.),
                        torch.ones(B,1,device=args.device,dtype=torch.bool),0,
                        return_plan=True,waist_yaw=q[:,0] if yi==0 else None)
                    traj=plan['traj'][:,:N].clone()
                    idle=arm_joint_indices(('left','right')[1-h]);traj[:,:,idle]=q[:,None,idle]
                    if N not in costs:
                        costs[N]=cost_for_horizon(cost,N);costs[N].cfg=copy.copy(cost.cfg);costs[N].cfg.n_sub=32
                    _,one,L=context(kit,h,[src]*len(rows),piece['destination'][rows],N,args.device)
                    dense=profile(costs[N],q[rows],v[rows],traj[rows],one,L,depth_m=.07,allowance=allowance)
                    if not dense['average_success'].all():
                        raise RuntimeError(f'Regenerated raw witness did not reproduce its tighter gate: h{h}, source {src}, N{N}, rows {rows}')
                    accept_decoder(piece,rows,N,traj[rows].cpu().numpy(),dense,yi,'average_success')
                    regenerated+=len(rows)
            for key in ('q','v','destination','sample'):
                np.testing.assert_array_equal(piece[key],original[key])
            assert int(np.isfinite(piece['minimum_s']).sum())==len(best)
            live=np.isfinite(piece['minimum_s'])
            assert (piece['collision_min_sdf'][live]>=-args.collision_m-1e-6).all()
            assert (piece['terminal_region_dist'][live]<=args.terminal_m+1e-6).all()
            assert (piece['torque_excess_average_Nm'][live]<=args.torque_average_nm+1e-6).all()
            piece['completed']=np.asarray(True);save_npz(folder/source.name,**piece)
        label=f'{args.collision_m*1000:g} mm overlap, {args.terminal_m*1000:g} mm miss, ≤{args.torque_average_nm:g} Nm average'
        summary=publish(argparse.Namespace(out=args.out,max_segments=100,verify_substeps=32,
            qp_starts_per_pair=0,decoder_acceptance=label),list(kit.names),complete=True)
        summary.update(decoder_only=True,qp_iterations=0,collision_overlap_allowed_m=args.collision_m,
            terminal_error_allowed_m=args.terminal_m,torque_average_excess_allowed_Nm=args.torque_average_nm,
            yaw_motion='Existing decoder yaw hints; no extra waist motion window',
            search_coverage='All 20 ms grid durations from the complete cached 20 ms–2 s sweep',
            primary_statistic='Shortest cached decoder duration passing the tighter constraints',
            limitation='Finite frozen-decoder grid gives shortest-found accepted outputs, not certified physical minima; unsupported starts remain unknown',
            copied_verified_raw_paths=reused,regenerated_verified_raw_paths=regenerated,
            elapsed_s=time.monotonic()-started)
        write_json(args.out/'summary.json',summary)
        write_json(args.out/'verification.json',dict(original_departures_preserved=680,
            qp_attempts=0,copied_verified_raw_paths=reused,regenerated_dense_verified_raw_paths=regenerated,
            all_saved_paths_pass_new_gate=True,all_grid_deadlines_classified=True))
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(args.out/'tb'))
        writer.add_scalar('reclassification/supported_pairs',summary['supported_pairs'],0)
        writer.add_scalar('reclassification/supported_starts',summary['supported_start_records'],0)
        writer.close()
        alias=ROOT/'runs/tensorboard_active'/(args.sweep_run.name+'_'+args.out.name)
        alias.symlink_to((args.out/'tb').resolve())
        write_json(args.out/'run.json',dict(status='complete',elapsed_s=summary['elapsed_s']))
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
