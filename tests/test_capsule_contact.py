"""The new playing-shaft hit rule must cover gaps between old probes."""
from types import SimpleNamespace

import numpy as np
import torch

from amortized_opt.constants import SHAFT_HIT_SPAN
from amortized_opt.capsule_contact import (capsule_contact_geometry,
    capsule_static_distance,capsule_sweep_distance,target_capsule_entry_mask)
from amortized_opt.fk import stick_axis_local,stick_contact_offset,stick_tip_offset
from amortized_opt.target_contact import v4_contact_probe_distance
from g1_scene.g1 import stick_shaft_offsets_in_wrist
from isaac_rl.cymbal_contact_scoring import make_capsule_tracker


def _geometry(z0=.02,z1=.009):
    # The head is centered midway between the tip and first old shaft probe.
    p0=torch.tensor([[-.2,0.,z0]],requires_grad=True)
    p1=torch.tensor([[0.,0.,z0]],requires_grad=True)
    q0=torch.tensor([[-.2,0.,z1]],requires_grad=True)
    q1=torch.tensor([[0.,0.,z1]],requires_grad=True)
    head=torch.tensor([[-.025,0.,0.]])
    normal=torch.tensor([[0.,0.,1.]])
    radius=torch.tensor([.009])
    height=torch.tensor([.1])
    return p0,p1,q0,q1,head,normal,radius,height


def test_capsule_finds_top_contact_between_all_old_probes():
    p0,p1,q0,q1,h,n,r,H=_geometry()
    s,top,side,_,speed,_=capsule_contact_geometry(
        p0,p1,q0,q1,h,n,r,H,
        torch.tensor([[0.,0.,-2.]]),torch.tensor([[0.,0.,-2.]]))
    assert top.any() and (speed[top]>1.9).all()
    assert ((s[top]>.75)&(s[top]<1.)).any()
    # Old probes sit at s=0,.25,.5,.75,1 and all miss this tiny head.
    old=torch.tensor([0.,.25,.5,.75,1.])
    x=-.2+.2*old
    assert ((x+.025).abs()>r.item()+.002).all()


def test_decoder_capsule_endpoints_match_isaac_distal_playing_shaft():
    for hand in ('left','right'):
        tip=stick_tip_offset(hand)
        playing_start=tip-SHAFT_HIT_SPAN*stick_axis_local(hand)
        np.testing.assert_allclose(playing_start,stick_contact_offset(hand,0.),atol=1e-12)
        isaac=np.asarray(stick_shaft_offsets_in_wrist(hand),dtype=float)
        for i in range(4):
            expected=tip-(i+1)*SHAFT_HIT_SPAN/4*stick_axis_local(hand)
            np.testing.assert_allclose(isaac[i],expected,atol=1e-12)


def test_capsule_terminal_loss_repairs_an_interior_hit_old_probes_miss():
    p0,p1,q0,q1,h,n,r,H=_geometry()
    fractions=torch.tensor([1.,.75,.5,.25,0.])
    previous=p0[:,None]+fractions[None,:,None]*(p1-p0)[:,None]
    current=q0[:,None]+fractions[None,:,None]*(q1-q0)[:,None]
    old,_,_=v4_contact_probe_distance(previous,current,h,n,r,H,
        torch.tensor([.01,.008,.008,.008,.008]),dt=.005)
    new,*_=capsule_sweep_distance(p0,p1,q0,q1,h,n,r,H,dt=.005)
    assert old.min()>.007
    assert new.min()<1e-6


def test_capsule_rejects_deep_side_and_supplies_finite_repair_gradient():
    p0,p1,q0,q1,h,n,r,H=_geometry(z0=-.003,z1=-.006)
    _,top,side,_,_,_=capsule_contact_geometry(
        p0,p1,q0,q1,h,n,r,H,
        torch.tensor([[0.,0.,-2.]]),torch.tensor([[0.,0.,-2.]]))
    assert not top.any() and not side.any()
    d,*_=capsule_sweep_distance(p0,p1,q0,q1,h,n,r,H,dt=.005)
    d.min().backward()
    assert d.min()>0
    assert all(torch.isfinite(x.grad).all() for x in (p0,p1,q0,q1))


def test_capsule_repair_gradient_points_back_toward_playing_surface():
    p0,p1,q0,q1,h,n,r,H=_geometry()
    with torch.no_grad():
        for point in (p0,p1,q0,q1):point[:,1]=.03
    d,*_=capsule_sweep_distance(p0,p1,q0,q1,h,n,r,H,dt=.005)
    d.min().backward()
    assert d.min()>.005
    assert q0.grad[0,1]+q1.grad[0,1]>0


def test_capsule_proxy_latches_top_at_same_material_fraction():
    p0,p1,q0,q1,h,n,r,H=_geometry()
    mount=torch.stack((p0.detach(),q0.detach()),1)
    bead=torch.stack((p1.detach(),q1.detach()),1)
    allowed,entry,_,side=target_capsule_entry_mask(mount,bead,h,n,r,H,
        torch.tensor([[.01,-.001]]),dt=.005,
        already_contacting_source=torch.tensor([False]))
    assert allowed.tolist()==[[False,True]]
    assert entry.tolist()==[[False,True]]
    assert not side.any()


def test_capsule_proxy_keeps_repair_gap_for_unapproved_deep_side():
    p0,p1,q0,q1,h,n,r,H=_geometry(z0=-.003,z1=-.006)
    mount=torch.stack((p0.detach(),q0.detach()),1)
    bead=torch.stack((p1.detach(),q1.detach()),1)
    allowed,entry,repair,side=target_capsule_entry_mask(mount,bead,h,n,r,H,
        torch.tensor([[.01,-.001]]),dt=.005,
        already_contacting_source=torch.tensor([False]))
    assert not allowed.any() and not entry.any() and not side.any()
    assert repair[0,1]>.0005


def test_pose_capsule_has_continuous_hit_distance():
    _,_,q0,q1,h,n,r,H=_geometry()
    d,_,_,s,_=capsule_static_distance(q0,q1,h,n,r,H)
    assert d.min()<1e-6
    assert ((s>.75)&(s<1.)).any()


def test_isaac_capsule_tracker_credits_interior_top_entry_with_real_force():
    class Base:
        def __init__(self,detector):
            self.detector=detector;self.num_envs=self.num_drums=1;self.device='cpu'
            self.force_threshold=1.;self.max_tangential_ratio=1.5
            self.require_rearm=True;self.clear_latch_on_exit=False
            shape=(1,2,1)
            self.entry_speed=torch.zeros(shape)
            self.entry_time=torch.full(shape,-1e6)
            self.events=torch.zeros(shape,dtype=torch.bool)
            self.contact_step=torch.zeros(shape,dtype=torch.bool)
            self.peak_force=torch.zeros(shape)
            for name in ('n_contacts','n_entries','n_rejected_band','n_rejected_rearm',
                         'n_rejected_grazing'):
                setattr(self,name,torch.zeros((),dtype=torch.long))
        def reset(self,ids):
            self.entry_speed[ids]=0;self.entry_time[ids]=-1e6
        def sample(self,*_args):pass
        def stats(self):return {}
    detector=SimpleNamespace(head_normal=torch.tensor([[0.,0.,1.]]),
        stray_min_speed=.1,strike_rearm_axial=.04)
    kit=SimpleNamespace(radii=[.009],heights=[.1],edge_hit=[True])
    tracker=make_capsule_tracker(Base,kit)(detector)
    tracker.debug_contact_geometry=True
    tracker.reset(torch.tensor([0]))
    def sample(z,force):
        x=torch.tensor([0.,-.05,-.1,-.15,-.2])
        points=torch.stack((x,torch.zeros_like(x),torch.full_like(x,z)),-1)
        points=points[None,None].expand(1,2,-1,-1).clone()
        velocity=torch.zeros_like(points);velocity[...,2]=-2.
        f=torch.zeros(1,2,1,3);f[...,2]=force
        tracker.sample(points,velocity,torch.tensor([[-.025,0.,0.]]),
                       torch.tensor([0.02]),f)
    sample(.045,0.)
    sample(.009,2.)
    assert tracker.events.all()
    assert torch.allclose(tracker.entry_speed,torch.full_like(tracker.entry_speed,2.))
    assert all(.75<event['fraction']<1. for event in tracker.debug_events)
