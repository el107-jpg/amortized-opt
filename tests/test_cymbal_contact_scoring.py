from types import SimpleNamespace
import torch

from isaac_rl.cymbal_contact_scoring import cymbal_entry_geometry,make_side_sliver_tracker


def classify(previous,current):
    p=torch.tensor(previous,dtype=torch.float32).reshape(1,1,1,3)
    q=torch.tensor(current,dtype=torch.float32).reshape(1,1,1,3)
    top,side,inside,through=cymbal_entry_geometry(q,p,torch.zeros(1,3),
        torch.tensor([[0.,0.,1.]]),torch.tensor([.15]),torch.tensor([.10]),
        torch.tensor([True]),torch.tensor([True]),torch.tensor([[True]]))
    return tuple(bool(mask[0,0,0,0]) for mask in (top,side,inside,through))


def test_top_entry_counts_even_after_deep_penetration():
    assert classify([.05,0.,.04],[.05,0.,-.06])==(True,False,True,False)


def test_only_upper_five_millimetres_of_side_is_eligible():
    assert classify([.19,0.,-.003],[.16,0.,-.003])==(False,True,True,False)
    assert classify([.19,0.,-.006],[.16,0.,-.006])==(False,False,True,False)
    assert classify([.19,0.,-.06],[.16,0.,-.06])==(False,False,True,False)


def test_top_entry_anywhere_on_face_but_not_from_underneath():
    assert classify([0.,0.,.04],[0.,0.,.014])[0]
    assert not classify([0.,0.,-.04],[0.,0.,-.06])[0]


def test_tilted_head_uses_its_own_normal_for_top_and_side():
    previous=torch.tensor([[[[.05,.04,0.]]]])
    current=torch.tensor([[[[.05,-.06,0.]]]])
    top,side,inside,through=cymbal_entry_geometry(current,previous,torch.zeros(1,3),
        torch.tensor([[0.,1.,0.]]),torch.tensor([.15]),torch.tensor([.10]),
        torch.tensor([True]),torch.tensor([True]),torch.tensor([[True]]))
    assert top.item() and inside.item() and not side.item() and not through.item()


def test_unsupported_complete_phase_through_is_not_accepted():
    top,side,inside,through=classify([0.,0.,.05],[0.,0.,-.20])
    assert top and not side and not inside and through


def test_contact_episode_latches_top_entry_but_rejects_deep_side():
    class Detector:
        num_probes=1
        head_normal=torch.tensor([[0.,0.,1.]])
        probe_allowed=torch.tensor([[True]])
        band_z_tol_above=.024
        def band(self,points,_centres,_tol):
            return torch.ones((*points.shape[:3],1),dtype=torch.bool)
    class Tracker:
        def __init__(self,detector):
            self.detector=detector;self.num_envs=self.num_drums=1;self.device='cpu'
            self.force_threshold=1.;self.qualified=[]
        def reset(self,_ids):pass
        def sample(self,points,_velocity,centres,_time,force):
            self.qualified.append(bool((self.detector.band(points,centres,.024).any(2)
                & (torch.linalg.vector_norm(force,dim=-1)>1.))[0,0,0]))
        def stats(self):return {}
    kit=SimpleNamespace(radii=[.15],heights=[.10],edge_hit=[True])
    tracker=make_side_sliver_tracker(Tracker,kit)(Detector())
    centres=torch.zeros(1,3);velocity=torch.zeros(1,2,1,3);time=torch.zeros(1)
    def sample(x,z,force):
        points=torch.tensor([[[[x,0.,z]],[[x,0.,z]]]])
        contact=torch.zeros(1,2,1,3);contact[...,2]=force
        tracker.sample(points,velocity,centres,time,contact)
    sample(.05,.04,0.)
    sample(.05,-.06,2.)
    sample(.05,-.08,2.)
    assert tracker.qualified==[False,True,True]
    tracker.reset(torch.tensor([0]))
    sample(.19,-.03,0.)
    sample(.16,-.03,2.)
    assert tracker.qualified[-1] is False
    tracker.reset(torch.tensor([0]))
    sample(.19,-.003,0.)
    sample(.16,-.003,2.)
    assert tracker.qualified[-1] is True
