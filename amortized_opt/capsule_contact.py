"""Continuous distal-shaft contact witnesses shared by Isaac and the decoder.

The shaft is parameterized from the distal playing-shaft start (s=0) to the
bead (s=1). Nominally that start is the wrist mount.
Candidate fractions contain every boundary of the top, upper-side, and
incidence inequalities, plus one point in every interval between boundaries.
Thus a nonzero-width valid interval cannot fall between fixed shaft probes.
The bead is evaluated separately with its larger physical radius.
"""
from __future__ import annotations

import torch

from .constants import STICK_RADIUS, TIP_RADIUS
from .target_contact import (SIDE_CREDIT_ABOVE_M, SIDE_CREDIT_DEPTH_M,
                             SIDE_CREDIT_RADIAL_EXTRA_M,
                             SIDE_CREDIT_RADIAL_INSET_M,
                             v4_contact_probe_distance,
                             v4_static_probe_distance)


def _linear_root(a, b, threshold):
    slope = b-a
    return (threshold-a)/torch.where(slope.abs()>1e-10, slope,
                                     torch.ones_like(slope))


def _quadratic_roots(a, b, c):
    discriminant = b*b-4*a*c
    root = torch.sqrt(discriminant.clamp_min(0))
    safe_a = torch.where(a.abs()>1e-12,a,torch.ones_like(a))
    safe_b = torch.where(b.abs()>1e-12,b,torch.ones_like(b))
    r0 = (-b-root)/(2*safe_a)
    r1 = (-b+root)/(2*safe_a)
    linear = -c/safe_b
    r0 = torch.where(a.abs()>1e-12,r0,linear)
    r1 = torch.where(a.abs()>1e-12,r1,linear)
    valid = ((discriminant>=0)&(a.abs()>1e-12)) | ((a.abs()<=1e-12)&(b.abs()>1e-12))
    return torch.where(valid,r0,torch.full_like(r0,-1.)),torch.where(
        valid,r1,torch.full_like(r1,-1.))


def capsule_candidate_fractions(previous_mount, previous_tip, current_mount,
                                current_tip, head, normal, body_radius,
                                body_height, previous_mount_velocity=None,
                                previous_tip_velocity=None,
                                incidence_ratio=1.5):
    """Return segment fractions covering every open geometric validity interval.

    All leading dimensions broadcast and must already match. Velocity roots
    make the pre-impact incidence condition exact along the segment too.
    Invalid or out-of-range roots collapse to an endpoint; duplicates are fine.
    """
    n=torch.nn.functional.normalize(normal,dim=-1)
    roots=[torch.zeros_like(body_radius),torch.ones_like(body_radius)]
    def axial(points):return ((points-head)*n).sum(-1)
    z0a,z0b=axial(previous_mount),axial(previous_tip)
    z1a,z1b=axial(current_mount),axial(current_tip)
    rp=STICK_RADIUS
    for a,b,t in ((z0a,z0b,rp-.002),
                  (z1a,z1b,rp+.006),(z1a,z1b,-body_height-rp-SIDE_CREDIT_RADIAL_EXTRA_M),
                  (z1a,z1b,-SIDE_CREDIT_DEPTH_M),(z1a,z1b,SIDE_CREDIT_ABOVE_M),
                  (z0a-z1a,z0b-z1b,.0005)):
        roots.append(_linear_root(a,b,t))
    def radial_roots(start,end,threshold):
        d0=start-head;d1=end-start
        d0=d0-(d0*n).sum(-1)[...,None]*n
        d1=d1-(d1*n).sum(-1)[...,None]*n
        qa=(d1*d1).sum(-1)
        qb=2*(d0*d1).sum(-1)
        qc=(d0*d0).sum(-1)-threshold.square()
        roots.extend(_quadratic_roots(qa,qb,qc))
        roots.append(-qb/(2*qa.clamp_min(1e-12)))
    radial_roots(previous_mount,previous_tip,body_radius+.002)
    for threshold in (body_radius+.002,
                      body_radius-SIDE_CREDIT_RADIAL_INSET_M,
                      body_radius+rp+SIDE_CREDIT_RADIAL_EXTRA_M):
        radial_roots(current_mount,current_tip,threshold)
    if previous_mount_velocity is not None:
        va=previous_mount_velocity;vb=previous_tip_velocity-previous_mount_velocity
        na=-(va*n).sum(-1);nb=-(vb*n).sum(-1)
        ta=va+na[...,None]*n;tb=vb+nb[...,None]*n
        k=float(incidence_ratio)**2
        qa=(tb*tb).sum(-1)-k*nb.square()
        qb=2*((ta*tb).sum(-1)-k*na*nb)
        qc=(ta*ta).sum(-1)-k*na.square()
        roots.extend(_quadratic_roots(qa,qb,qc))
        roots.append(_linear_root(na,na+nb,1e-6))
    ordered=torch.stack([r.clamp(0,1) for r in roots],-1).sort(-1).values
    mid=(ordered[...,1:]+ordered[...,:-1])*.5
    # Fractions are witnesses chosen by a hard active-set search. Differentiating
    # through a grazing quadratic root has an infinite derivative at zero
    # discriminant and contaminates otherwise finite trajectory gradients.
    # The contact residual still differentiates through the material points.
    return torch.cat((ordered,mid),-1).detach()


def capsule_points(mount,tip,fractions):
    return mount[...,None,:]+fractions[...,None]*(tip-mount)[...,None,:]


def capsule_contact_geometry(previous_mount,previous_tip,current_mount,current_tip,
                             head,normal,body_radius,body_height,
                             previous_mount_velocity,previous_tip_velocity,
                             *,incidence_ratio=1.5):
    """Hard top/side/inside and speed witnesses for the playing shaft + bead.

    Output has a last candidate axis. The bead has its own 10-mm radius and
    is appended after the continuous 8-mm shaft candidates.
    """
    s=capsule_candidate_fractions(previous_mount,previous_tip,current_mount,
        current_tip,head,normal,body_radius,body_height,
        previous_mount_velocity,previous_tip_velocity,incidence_ratio)
    s=torch.cat((s,torch.ones_like(s[...,:1])),-1)
    prior=capsule_points(previous_mount,previous_tip,s)
    current=capsule_points(current_mount,current_tip,s)
    vel=capsule_points(previous_mount_velocity,previous_tip_velocity,s)
    rp=torch.full_like(s,STICK_RADIUS)
    rp=torch.cat((rp[...,:-1],torch.full_like(rp[...,-1:],TIP_RADIUS)),-1)
    n=torch.nn.functional.normalize(normal,dim=-1)[...,None,:]
    h=head[...,None,:]
    def coords(p):
        d=p-h;z=(d*n).sum(-1)
        return z,torch.linalg.vector_norm(d-z[...,None]*n,dim=-1)
    z0,r0=coords(prior);z1,r1=coords(current)
    R=body_radius[...,None];H=body_height[...,None]
    speed=-(vel*n).sum(-1)
    tangent=torch.linalg.vector_norm(vel+speed[...,None]*n,dim=-1)
    incidence=(speed>1e-6)&(tangent<=incidence_ratio*speed)
    top=((z0>=rp-.002)&(z1<=rp+.006)&((z0-z1)>.0005)
         &(r0<=R+.002)&(r1<=R+.002)&incidence)
    side=((z1>=-SIDE_CREDIT_DEPTH_M-1e-6)&(z1<=SIDE_CREDIT_ABOVE_M)
          &(r1>=R-SIDE_CREDIT_RADIAL_INSET_M)
          &(r1<=R+rp+SIDE_CREDIT_RADIAL_EXTRA_M)&incidence)
    inside=(r1<=R+rp+SIDE_CREDIT_RADIAL_EXTRA_M)&(z1>=-H-rp-SIDE_CREDIT_RADIAL_EXTRA_M)
    return s,top,side,inside,speed,current


def capsule_sweep_distance(previous_mount,previous_tip,current_mount,current_tip,
                           head,normal,body_radius,body_height,*,dt,
                           incidence_ratio=1.5):
    """Differentiable distance to valid top/side contact along the capsule."""
    velocity_mount=(current_mount-previous_mount)/dt
    velocity_tip=(current_tip-previous_tip)/dt
    s=capsule_candidate_fractions(previous_mount,previous_tip,current_mount,
        current_tip,head,normal,body_radius,body_height,
        velocity_mount,velocity_tip,incidence_ratio)
    s=torch.cat((s,torch.ones_like(s[...,:1])),-1)
    previous=capsule_points(previous_mount,previous_tip,s)
    current=capsule_points(current_mount,current_tip,s)
    radii=torch.cat((torch.full((s.shape[-1]-1,),STICK_RADIUS,device=s.device,dtype=s.dtype),
                     torch.full((1,),TIP_RADIUS,device=s.device,dtype=s.dtype)))
    distance,top,side=v4_contact_probe_distance(previous,current,head,normal,
        body_radius,body_height,radii,dt=dt,incidence_ratio=incidence_ratio)
    return distance,top,side,s,current,previous


def capsule_static_distance(mount,tip,head,normal,body_radius,body_height):
    """Pose-only distance to the physical playing surface along the capsule."""
    s=capsule_candidate_fractions(mount,tip,mount,tip,head,normal,
        body_radius,body_height)
    s=torch.cat((s,torch.ones_like(s[...,:1])),-1)
    points=capsule_points(mount,tip,s)
    radii=torch.cat((torch.full((s.shape[-1]-1,),STICK_RADIUS,device=s.device,dtype=s.dtype),
                     torch.full((1,),TIP_RADIUS,device=s.device,dtype=s.dtype)))
    return (*v4_static_probe_distance(points,head,normal,body_radius,radii),s,points)


def target_capsule_entry_mask(mount,tip,head,normal,body_radius,body_height,
                              body_sdf,*,dt,already_contacting_source,
                              incidence_ratio=1.5,contact_margin=.003):
    """Trajectory proxy for the Isaac capsule's force-qualified hit region.

    The hard latch tracks one material fraction through its contact episode;
    the body SDF stands in for PhysX force. The repair term remains
    differentiable and searches all continuous capsule witnesses.
    """
    B,S,_=mount.shape
    previous_mount=torch.cat((mount[:,:1],mount[:,:-1]),1)
    previous_tip=torch.cat((tip[:,:1],tip[:,:-1]),1)
    step_vm=(mount-previous_mount)/dt
    step_vt=(tip-previous_tip)/dt
    # At the first transition no earlier sampled velocity exists; use its
    # own incoming finite difference. Later steps use true pre-step velocity.
    vm=torch.cat((step_vm[:,:2],step_vm[:,1:-1]),1)
    vt=torch.cat((step_vt[:,:2],step_vt[:,1:-1]),1)
    head_s=head[:,None].expand(-1,S,-1)
    normal_s=normal[:,None].expand(-1,S,-1)
    radius_s=body_radius[:,None].expand(-1,S)
    height_s=body_height[:,None].expand(-1,S)
    fractions,top,side,_inside,speed,_=capsule_contact_geometry(
        previous_mount,previous_tip,mount,tip,head_s,normal_s,
        radius_s,height_s,vm,vt,incidence_ratio=incidence_ratio)
    touching=body_sdf<=contact_margin
    top=top&touching[...,None]
    side=side&touching[...,None]
    best=torch.where(top,speed,torch.full_like(speed,-1.)).argmax(-1)
    selected=fractions.gather(-1,best[...,None]).squeeze(-1)
    selected_radius=torch.where(best==fractions.shape[-1]-1,
        torch.full_like(selected,TIP_RADIUS),torch.full_like(selected,STICK_RADIUS))
    event=top.any(-1)
    latch=already_contacting_source&touching[:,0]
    init_distance,_,_,init_fractions,_=capsule_static_distance(
        mount[:,0],tip[:,0],head,normal,body_radius,body_height)
    init_best=init_distance.argmin(-1)
    initial_s=init_fractions.gather(-1,init_best[:,None]).squeeze(-1)
    held_s=torch.where(already_contacting_source,initial_s,
                       torch.ones_like(initial_s))
    initial_radius=torch.where(init_best==init_fractions.shape[-1]-1,
        torch.full_like(held_s,TIP_RADIUS),torch.full_like(held_s,STICK_RADIUS))
    held_radius=torch.where(already_contacting_source,initial_radius,
                            torch.full_like(held_s,TIP_RADIUS))
    allowed=[]
    n=torch.nn.functional.normalize(normal,dim=-1)
    for i in range(S):
        held_s=torch.where(event[:,i],selected[:,i],held_s)
        held_radius=torch.where(event[:,i],selected_radius[:,i],held_radius)
        latch=(latch|event[:,i])&touching[:,i]
        point=mount[:,i]+held_s[:,None]*(tip[:,i]-mount[:,i])
        delta=point-head
        axial=(delta*n).sum(-1)
        radial=torch.linalg.vector_norm(delta-axial[:,None]*n,dim=-1)
        inside=(radial<=body_radius+held_radius+SIDE_CREDIT_RADIAL_EXTRA_M)
        inside &= axial>=-body_height-held_radius-SIDE_CREDIT_RADIAL_EXTRA_M
        allowed.append((latch&inside)|side[:,i].any(-1))
    allowed=torch.stack(allowed,1)
    distance,top_distance,side_distance,_,_,_=capsule_sweep_distance(
        previous_mount.reshape(B*S,3),previous_tip.reshape(B*S,3),
        mount.reshape(B*S,3),tip.reshape(B*S,3),
        head_s.reshape(B*S,3),normal_s.reshape(B*S,3),
        radius_s.reshape(B*S),height_s.reshape(B*S),
        dt=dt,incidence_ratio=incidence_ratio)
    repair=distance.amin(-1).reshape(B,S)
    # The caller prices the first *penetrating* substep. Proxy touching may
    # begin up to 3 mm earlier, so preserve the repair gap throughout an
    # unapproved episode instead of zeroing it after the first proxy sample.
    repair=torch.where(touching&~allowed,repair,torch.zeros_like(repair))
    return allowed,event,repair,side.any(-1)
