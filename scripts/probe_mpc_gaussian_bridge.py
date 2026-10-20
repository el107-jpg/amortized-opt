#!/usr/bin/env python3
"""Frozen one-pass repair of calibrated Gaussian bridges on good MPC inputs."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import amortized_opt as A
from amortized_opt.gaussian_bridge import GaussianBridgeNoise,sample_state_drift
from amortized_opt.phase2_data import Phase2Pool,Phase2Noise
from amortized_opt.train_cost import (_index_batch,_apply_mpc_init_noise,build_costs,
    phase2_config_from_checkpoint,_wtensors)
from amortized_opt.warmstart_data import ou_kernel
from isaac_rl.reward import ExecutedStrikeScorer
from probe_replay_boundary import _scenario_from_ckpt,_load_al
from probe_mpc_plan_drift import make_ring,evaluate,_stats,BANDS
from probe_mpc_expert_noise import select_feasible_references


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True);p.add_argument('--bridge-config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:2')
    p.add_argument('--rows',type=int,default=256)
    args=p.parse_args();dev=torch.device(args.device);torch.cuda.set_device(dev)
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    cfg=phase2_config_from_checkpoint(ck,device=str(dev),compile_model=False);scenario=_scenario_from_ckpt(ck)
    H=50;pool=Phase2Pool({},cfg=scenario,device=dev)
    pool._ou[H]=ou_kernel(H,5,dev,pool.dtype)
    costs=build_costs([H],cfg.cost,device=dev);cost=costs[H]
    al=_load_al(cfg,ck);step=int(ck['step'])
    weights=_wtensors(cfg.schedule.multipliers(step,True),al.weights(),dev,pool.dtype,sched=cfg.schedule,joint_names=scenario.joint_names,step=step,stage_a=True,pen=al.pen)
    model=A.build_model(A.model_config_from_checkpoint(ck),dev);model.load_state_dict(ck['model']);model.eval()
    ring=make_ring(ck['extra']['timing_critic']['primary_mpc_ring'],cfg,pool,costs,dev)
    stored=ring.data[H];stored['probe_parent_id']=torch.arange(ring.size(H),device=dev)
    stored['probe_parent_nearest']=torch.where(stored['active'],stored['hit'],H).min(1).values
    class AmpModel:
        def encode_context(self,ctx):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model.encode_context(ctx)
        def __call__(self,*a):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model(*a)
    matched_model=AmpModel()
    gen=torch.Generator(device=dev).manual_seed(20261001)
    batch,selection=select_feasible_references(ring,H,matched_model,cost,weights,gen,args.rows,128,20000)
    bridge=GaussianBridgeNoise(**json.loads(args.bridge_config.read_text()))
    broad=Phase2Noise(tau_steps=20,sigma_white=.003,magnitude_distribution='half_normal',sigma_scale=.3)
    softer=replace(bridge,terminal_position_std=tuple(.5*x for x in bridge.terminal_position_std),
                   terminal_velocity_std=tuple(.5*x for x in bridge.terminal_velocity_std))
    policies=[('clean',None,False),('broad_ou',broad,False),('bridge',Phase2Noise(process='gaussian_bridge',bridge=bridge),False),
              ('bridge_half_endpoint',Phase2Noise(process='gaussian_bridge',bridge=softer),False),
              ('bridge_with_measured_state_drift',Phase2Noise(process='gaussian_bridge',bridge=bridge),True)]
    scorer=ExecutedStrikeScorer(fk=cost.fk,device=dev)
    reports=[];arrays={};baseline_ok=None
    def shifts(bt,clean,noisy):
        result=[];rows=torch.arange(len(clean),device=dev)
        for hand in [0,1]:
            k=(bt['hit'][:,hand]-1).clamp_min(0)
            qr=clean[rows,k];qn=noisy[rows,k]
            _,fraction,pr=scorer.best_contact(qr,bt['strike']);lo,hi=scorer._sigma_span(bt['strike'])
            pn,_=scorer.contact_points(qn,lo+fraction*(hi-lo))
            result.append((pn[:,hand]-pr[:,hand]).norm(dim=-1)*1000)
        return torch.stack(result,1)
    for name,noise,state_noise in policies:
        metrics=[];movement=[];clipped=[];perturb=[];nearest=[]
        generator=torch.Generator(device=dev).manual_seed(77)
        for start in range(0,len(batch['q0']),128):
            bt=_index_batch(batch,torch.arange(start,min(start+128,len(batch['q0'])),device=dev));clean=bt['init'].clone()
            if state_noise:
                nearest_t=torch.where(bt['active'],bt['hit'],H).min(1).values
                elapsed=(bt['probe_parent_nearest']-nearest_t).float()*.02
                dq,dv=sample_state_drift(elapsed,bt['q0'],bridge,generator=generator)
                bt['q0']=(bt['q0']+dq).clamp(pool.q_lo,pool.q_hi);bt['v0']=bt['v0']+dv
                bt['state']=torch.cat([bt['q0'],bt['v0']],-1)
                yaw=tuple(scenario.joint_names).index('waist_yaw_joint')
                bt['context']=replace(bt['context'],yaw_now=bt['q0'][:,yaw])
            if noise is not None:_apply_mpc_init_noise(bt,pool=pool,generator=generator,enabled=True,noise=noise)
            movement.append(shifts(bt,clean,bt['init']).cpu().numpy())
            mask=bt['masks'].elem_mask;delta=bt['init']-clean
            perturb.append(((delta.square()*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)).sqrt().cpu().numpy())
            clipped.append(((((bt['init']<=pool.q_lo+1e-6)|(bt['init']>=pool.q_hi-1e-6))&mask.bool()).sum((1,2))/mask.sum((1,2)).clamp_min(1)).cpu().numpy())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(bt['init'],bt['state'],model.encode_context(bt['context']),bt['masks']).traj.float()
            metrics.append(evaluate(cost,bt,out,weights));nearest.append(torch.where(bt['active'],bt['hit'],H).min(1).values.cpu().numpy())
        values={k:np.concatenate([x[k] for x in metrics]) for k in metrics[0]};ks=np.concatenate(nearest)
        if name=='clean':baseline_ok=values['hard_ok'].copy()
        values={k:v[baseline_ok] for k,v in values.items()};ks=ks[baseline_ok]
        rates={'all':float(values['hard_ok'].mean())}
        rates.update({band:float(values['hard_ok'][(ks>=lo)&(ks<=hi)].mean()) if ((ks>=lo)&(ks<=hi)).any() else None for band,(lo,hi) in BANDS.items()})
        row=dict(policy=name,rows=int(baseline_ok.sum()),candidates=len(batch['q0']),hard_feasible_rate=rates,endpoint_shift_mm=_stats(np.concatenate(movement)[baseline_ok[:,None]&batch['active'].cpu().numpy()]),
            input_position_rms_rad=_stats(np.concatenate(perturb)[baseline_ok]),frozen_output_region_excess_mm=_stats(1000*values['terminal_region_dist']),
            fraction_valid_elements_at_joint_limit=float(np.concatenate(clipped)[baseline_ok].mean()))
        reports.append(row);arrays.update({name+'/'+k:v for k,v in values.items()});print(json.dumps(row),flush=True)
    payload=dict(checkpoint=str(args.ckpt.resolve()),bridge_config=str(args.bridge_config.resolve()),selection=selection,bridge=asdict(bridge),reports=reports,
      definitions={'reference':'Distinct MPC suffix cases with hard-feasible clean input and BF16 one-pass clean output. Final baseline qualification uses identical batch shapes to each perturbed condition; a few numerical boundary cases may be excluded.',
        'endpoint_shift':'Same contact point along each active playing stick, before/after initialization noise',
        'state_drift':'Separate condition: empirically calibrated q/qd errors with unchanged scheduled strike, often harder near contact',
        'limits':'Frozen-model probe, no training; empirical calibration uses one actual Isaac clip.'})
    args.out.write_text(json.dumps(payload,indent=2)+'\n');np.savez_compressed(args.out.with_suffix('.npz'),**arrays)


if __name__=='__main__':main()
