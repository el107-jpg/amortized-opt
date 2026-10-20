#!/usr/bin/env python3
"""Model-free waypoint descent from a snare-side terminal trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS,STICK_RADIUS,TIP_RADIUS
from amortized_opt.cost import FusedCost,StrikeContext
from amortized_opt.fk import ChainFK,stick_tip_offset
from amortized_opt.limits import urdf_position_limits
from amortized_opt.sdf import DrumSceneSpec,point_cylinder_sdf
from amortized_opt.train_cost import phase2_cost_config


OUT=ROOT/'runs/snare_side_trajectory_descent_v5_20261013'
SOURCE=ROOT/'runs/capsule_v5_matched_hihat_snare_20261013/baseline/reports_v5.json'
DEVICE='cuda:1'
START_TICK=20  # 0.40 s: measured trajectory is still above the snare
HIT_TICK=42    # 0.84 s: first right-hand snare beat
H=HIT_TICK-START_TICK
STEPS=70


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def context(head,normal,hit_radius):
    dev=head.device;dtype=head.dtype
    z=lambda *shape:torch.zeros(*shape,device=dev,dtype=dtype)
    return StrikeContext(
        hit_index=torch.tensor([[0,H]],device=dev),
        active=torch.tensor([[0.,1.]],device=dev,dtype=dtype),
        target_drum=torch.zeros(1,2,device=dev,dtype=torch.long),
        start_drum=torch.full((1,2),-1,device=dev,dtype=torch.long),
        p_star=(head-.01*normal)[None,None].expand(1,2,3),
        v_star=(-2.4*normal)[None,None].expand(1,2,3),
        n_out=normal[None,None].expand(1,2,3),
        hit_radius=torch.full((1,2),hit_radius,device=dev,dtype=dtype),
        disk_offset=z(1,2,3),half_plane_dir=z(1,2,3),has_half_plane=z(1,2),
        contact_sigma=torch.full((1,2),.2,device=dev,dtype=dtype),
        shaft_allowed=z(1,2),start_shaft_allowed=z(1,2),
        elev_required=torch.ones(1,2,device=dev,dtype=dtype))


def main():
    global OUT,STEPS,DEVICE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side-offset-mm',type=float,default=15.)
    parser.add_argument('--terminal-depth-mm',type=float,default=5.,
        help='Initial terminal bead depth below the drum top along its normal')
    parser.add_argument('--steps',type=int,default=70)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--device',default=DEVICE)
    parser.add_argument('--resume',action='store_true',
        help='Continue saved final waypoints to --steps total iterations')
    cli=parser.parse_args()
    if cli.side_offset_mm<=0 or cli.terminal_depth_mm<=0 or cli.steps<1:
        raise ValueError('Positive offset, depth, and step budget required')
    STEPS=cli.steps
    DEVICE=cli.device
    if cli.out is not None:OUT=cli.out.resolve()
    elif cli.terminal_depth_mm!=5.:
        OUT=OUT/f'offset{cli.side_offset_mm:g}mm_depth{cli.terminal_depth_mm:g}mm'
    elif cli.side_offset_mm!=15.:OUT=OUT/f'offset{cli.side_offset_mm:g}mm'
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads(SOURCE.read_text())[0]
    assert source['per_hit_outcomes'][1]['drum']=='snare'
    assert source['per_hit_outcomes'][1]['hand']=='right'
    assert source['per_hit_outcomes'][1]['hit']
    measured=source['rollout_audit']['trace']
    q_trace=torch.tensor(measured['q'][START_TICK:HIT_TICK+1],device=DEVICE,dtype=torch.float32)
    v0=torch.tensor(measured['qd'][START_TICK:START_TICK+1],device=DEVICE,dtype=torch.float32)
    q0=q_trace[:1]
    scene=DrumSceneSpec.from_layout(frame='pelvis')
    drum=scene.names.index('snare')
    head=torch.tensor(scene.head_centers[drum],device=DEVICE,dtype=q0.dtype)
    normal=torch.tensor(scene.axes[drum],device=DEVICE,dtype=q0.dtype)
    radius=float(scene.radii[drum]);hit_radius=radius-.01
    ctx=context(head,normal,hit_radius)
    fk=ChainFK(joint_names=G1_UPPER_BODY_JOINTS,frames=dict(FusedCost.FRAMES),
               device=DEVICE,dtype=torch.float32)
    local_tip=torch.tensor(stick_tip_offset('right'),device=DEVICE,dtype=q0.dtype)
    lo_np,hi_np=urdf_position_limits()
    lower=torch.tensor(lo_np,device=DEVICE,dtype=q0.dtype)
    upper=torch.tensor(hi_np,device=DEVICE,dtype=q0.dtype)

    def tip(q):
        R,p=fk.frame(q,'right_wrist')
        return p+(R@local_tip.view(1,3,1)).squeeze(-1)

    # A modest local IK perturbation of a measured successful trajectory.
    # The final tip is outside the physical rim at the requested depth.
    reference=q_trace[-1].detach()
    original_tip=tip(reference[None])[0].detach()
    displacement=original_tip-head
    axial=(displacement*normal).sum()
    radial=displacement-axial*normal
    radial_dir=radial/radial.norm()
    side_target=(head+(radius+cli.side_offset_mm/1000)*radial_dir
                 -(cli.terminal_depth_mm/1000)*normal)
    q_side=reference.clone()
    ik_errors=[]
    for _ in range(30):
        arm=q_side[10:].clone().requires_grad_(True)
        def arm_tip(values):
            return tip(torch.cat((reference[:10],values))[None])[0]
        position=arm_tip(arm)
        error=side_target-position
        ik_errors.append(float(error.norm()))
        if float(error.norm())<1e-5:break
        J=torch.autograd.functional.jacobian(arm_tip,arm)
        update=J.T@torch.linalg.solve(J@J.T+.01**2*torch.eye(3,device=DEVICE),error)
        q_side[10:]=(q_side[10:]+update.clamp(-.08,.08)).clamp(lower[10:],upper[10:])
    q_side=q_side.detach()
    side_tip=tip(q_side[None])[0]
    if float((side_tip-side_target).norm())>.002:
        raise RuntimeError('Side-pose IK failed to place tip within 2 mm')
    delta=(q_side-reference).detach()
    blend=torch.linspace(0,1,H+1,device=DEVICE,dtype=q0.dtype)[1:]
    blend=blend.square()*(3-2*blend)
    initial=q_trace[1:].clone()+blend[:,None]*delta[None]
    initial=initial.clamp(lower[None],upper[None]).detach()
    initial[:,:10]=q_trace[1:,:10]  # right-arm trajectory alone is optimized
    q0=q0.detach();v0=v0.detach()
    cfg=phase2_cost_config()
    cost=FusedCost(cfg,max_horizon=H,device=DEVICE,dtype=torch.float32)
    arm_mask=torch.zeros_like(initial);arm_mask[:,10:]=1.
    physical_half_height=float(scene.heights[drum])/2
    body=head-physical_half_height*normal

    def geometry(waypoints):
        with torch.no_grad():
            sp=cost.spline.build(q0,waypoints[None],v0,
                                 torch.tensor([[H,0,H]],device=DEVICE))
            sample=cost._geometry_samples(sp,v0)
            frames=cost.fk.forward(sample.q.reshape(-1,17))
            probe=cost._points_world(frames,'right',ctx).reshape(1,-1,cost.n_points,3)
            tips=probe[0,:,0]
            d=tips-head
            z=(d*normal).sum(-1)
            rho=torch.linalg.vector_norm(d-z[:,None]*normal,dim=-1)
            sdf=point_cylinder_sdf(tips,body,normal,radius,physical_half_height)-TIP_RADIUS
            contact=(sdf<0).nonzero(as_tuple=True)[0]
            first=int(contact[0]) if len(contact) else None
            return dict(tip=tips.cpu().numpy(),rho=rho.cpu().numpy(),z=z.cpu().numpy(),
                sdf=sdf.cpu().numpy(),first_contact_index=first)

    path=initial.detach().clone()
    history=[];snapshots={};waypoint_snapshots={}
    start_step=0
    if cli.resume:
        saved=np.load(OUT/'trajectories.npz')
        previous=json.loads((OUT/'summary.json').read_text())
        if (previous['target_side_offset_m']!=cli.side_offset_mm/1000
                or previous['target_axial_m']!=-cli.terminal_depth_mm/1000
                or previous['contact_scoring_mode']!=cfg.contact_scoring_mode
                or previous['physics_loss_version']!=cfg.physics_loss_version
                or not np.allclose(saved['q0'],q0.cpu().numpy(),atol=1e-6)
                or not np.allclose(saved['initial'],initial.cpu().numpy(),atol=1e-6)):
            raise ValueError('Saved trajectory/objective differs from continuation')
        path=torch.tensor(saved['final'],device=DEVICE,dtype=q0.dtype)
        history=previous['history'][:-1]  # re-evaluate the saved last step, then advance
        start_step=int(previous['history'][-1]['step'])
        if STEPS<=start_step:raise ValueError('New budget must exceed saved step')
        for key in saved.files:
            if key.startswith('tip_step_'):
                index=int(key.rsplit('_',1)[1]);tips=saved[key]
                points=torch.tensor(tips,device=DEVICE,dtype=q0.dtype)
                sdf=(point_cylinder_sdf(points,body,normal,radius,physical_half_height)
                     -TIP_RADIUS).cpu().numpy()
                snapshots[index]=dict(tip=tips,rho=saved[f'rho_step_{index}'],
                                      z=saved[f'axial_step_{index}'],sdf=sdf)
            elif key.startswith('q_step_'):
                waypoint_snapshots[int(key.rsplit('_',1)[1])]=saved[key]
        print(f'[descent] resuming from step {start_step} to {STEPS}',flush=True)
    for step in range(start_step,STEPS+1):
        variable=path.detach().clone().requires_grad_(True)
        result=cost.forward(q0,variable[None],v0,ctx,lengths=torch.tensor([[H,0,H]],device=DEVICE))
        objective=result.total.sum()
        if not torch.isfinite(objective):raise FloatingPointError(f'Nonfinite cost at step {step}')
        gradient=torch.autograd.grad(objective,variable)[0]*arm_mask
        with torch.no_grad():
            tip_end=tip(variable[-1:])[0]
            d=tip_end-head
            axial=float((d*normal).sum())
            radial=float(torch.linalg.vector_norm(d-(d*normal).sum()*normal))
            geom=geometry(variable)
            record=dict(step=step,cost=float(objective),terminal_rho_m=radial,
                terminal_axial_m=axial,
                terminal_region_m=float(result.diagnostics['terminal_region_dist'][0]),
                terminal_aim_cost=float(result.terms['terminal_aim'][0]),
                terminal_cost=float(result.terms['terminal'][0]),
                collision_cost=float(result.terms['collision'][0]),
                collision_min_sdf_m=float(result.diagnostics['collision_min_sdf'][0]),
                target_tip_min_sdf_m=float(np.min(geom['sdf'])),
                first_target_contact_index=geom['first_contact_index'],
                first_target_contact_axial_m=(float(geom['z'][geom['first_contact_index']])
                    if geom['first_contact_index'] is not None else None),
                first_target_contact_radial_m=(float(geom['rho'][geom['first_contact_index']])
                    if geom['first_contact_index'] is not None else None),
                grad_norm=float(torch.linalg.vector_norm(gradient)))
            history.append(record)
            if step in (0,1,2,5,10,20,40,60,70,80,100,150,200) or step==STEPS:
                snapshots[step]=geom
                waypoint_snapshots[step]=variable.detach().cpu().numpy()
            if step==STEPS:break
            direction=-gradient/torch.linalg.vector_norm(gradient).clamp_min(1e-12)
            accepted=False
            # Monotone projected steepest descent, with an ordinary backtrack
            # in joint-waypoint space. No neural model or learned optimizer.
            for trial in range(10):
                scale=.08*(.5**trial)
                candidate=(path+scale*direction).clamp(lower[None],upper[None])
                candidate[:,:10]=initial[:,:10]
                candidate_cost=cost.forward(q0,candidate[None],v0,ctx,
                    lengths=torch.tensor([[H,0,H]],device=DEVICE)).total[0]
                if float(candidate_cost)<float(objective)-1e-7:
                    path=candidate.detach();accepted=True;record['accepted_step_l2_rad']=scale
                    break
            if not accepted:
                record['accepted_step_l2_rad']=0.
                print(f'[descent] stopped at {step}: no decreasing projected step',flush=True)
                break
        if step%10==0:
            atomic_json(OUT/'progress.json',record)
            torch.save(dict(step=step,waypoints=path.detach().cpu(),history=history),
                       OUT/'descent_checkpoint.pt')
            print('[descent] '+json.dumps({k:record[k] for k in ('step','cost','terminal_rho_m',
                'terminal_axial_m','terminal_region_m','target_tip_min_sdf_m')}),flush=True)

    # Include the final accepted path even when descent stopped before budget.
    final_step=history[-1]['step']
    if final_step not in snapshots:
        snapshots[final_step]=geometry(path)
        waypoint_snapshots[final_step]=path.detach().cpu().numpy()
    np.savez_compressed(OUT/'trajectories.npz',q0=q0.detach().cpu().numpy(),
        initial=initial.detach().cpu().numpy(),final=path.detach().cpu().numpy(),
        head=head.detach().cpu().numpy(),normal=normal.detach().cpu().numpy(),
        **{f'tip_step_{step}':value['tip'] for step,value in snapshots.items()},
        **{f'rho_step_{step}':value['rho'] for step,value in snapshots.items()},
        **{f'axial_step_{step}':value['z'] for step,value in snapshots.items()},
        **{f'q_step_{step}':value for step,value in waypoint_snapshots.items()})
    atomic_json(OUT/'summary.json',dict(status='complete',source_report=str(SOURCE),source_case=0,
        source_beat_index=1,model_used_for_optimization=False,
        seed='measured Isaac trajectory with deterministic right-arm IK perturbation',
        contact_scoring_mode=cfg.contact_scoring_mode,
        physics_loss_version=cfg.physics_loss_version,
        n_sub=cfg.n_sub,geometry_motion_mode=cfg.geometry_motion_mode,
        optimized_waypoints=H,optimized_joint_columns=list(range(10,17)),
        target_side_offset_m=cli.side_offset_mm/1000,
        target_axial_m=-cli.terminal_depth_mm/1000,
        ik_final_error_m=float((side_tip-side_target).norm()),
        ik_max_joint_delta_rad=float(delta.abs().max()),
        initial=history[0],final=history[-1],history=history))

    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax=axes[0,0]
    ax.fill_between([0,radius],[-.10,-.10],[0,0],color='#888888',alpha=.45,
                    label='solid drum body')
    ax.fill_between([0,hit_radius],[-.01,-.01],[0,0],color='#3cba71',alpha=.45,
                    label='terminal hit region')
    for step in sorted(snapshots):
        if step not in (0,1,5,10,20,40,60,80,final_step):continue
        data=snapshots[step]
        ax.plot(data['rho'],data['z'],lw=1.4,alpha=.8,label=f'step {step}')
        ax.scatter(data['rho'][-1],data['z'][-1],s=15)
    ax.axvline(radius,color='black',lw=1,ls=':')
    ax.axhline(0,color='black',lw=1,ls=':')
    ax.set(xlabel='tip radial distance from snare center (m)',ylabel='tip height along head normal (m)',
           title='Full tip path on a 4 ms geometry grid',xlim=(0,.36),ylim=(-.08,.35))
    ax.legend(fontsize=7,ncol=2,loc='upper left')

    ax=axes[0,1]
    steps=np.array([row['step'] for row in history]);rho=np.array([row['terminal_rho_m'] for row in history]);
    height=np.array([row['terminal_axial_m'] for row in history])
    ax.fill_between([0,hit_radius],[-.01,-.01],[0,0],color='#3cba71',alpha=.35)
    ax.fill_between([0,radius],[-.10,-.10],[0,0],color='#888888',alpha=.35)
    ax.plot(rho,height,'-o',lw=1,ms=2,c='#2360a6')
    for j in (0,1,2,5,10,20,40,60,80,final_step):
        if j<len(history):ax.annotate(str(j),(rho[j],height[j]),fontsize=8)
    ax.axvline(radius,color='black',lw=1,ls=':');ax.axhline(0,color='black',lw=1,ls=':')
    ax.set(xlabel='terminal radial distance (m)',ylabel='terminal height (m)',
           title='Terminal tip under gradient descent',xlim=(.0,.25),ylim=(-.055,.08))

    ax=axes[1,0]
    ax.plot(steps,[row['cost'] for row in history],label='total')
    ax.plot(steps,[row['terminal_cost'] for row in history],label='terminal')
    ax.plot(steps,[row['collision_cost'] for row in history],label='collision')
    ax.set(xlabel='descent step',ylabel='weighted cost',title='Current full FusedCost',yscale='symlog')
    ax.legend()

    ax=axes[1,1]
    ax.plot(steps,np.array([row['terminal_region_m'] for row in history])*1000,
            label='terminal region miss (mm)')
    ax.plot(steps,-np.array([row['target_tip_min_sdf_m'] for row in history])*1000,
            label='target tip penetration (mm; positive is overlap)')
    ax.axhline(0,color='black',lw=1,ls=':')
    ax.set(xlabel='descent step',ylabel='mm',title='Hit-position repair vs physical target overlap')
    ax.legend()
    fig.suptitle('Model-free waypoint descent from a side-of-snare terminal pose',fontsize=14)
    fig.savefig(OUT/'trajectory_descent.png',dpi=165)
    plt.close(fig)
    print('[result] '+json.dumps(dict(initial=history[0],final=history[-1],
        figure=str((OUT/'trajectory_descent.png').resolve()))),flush=True)


if __name__=='__main__':main()
