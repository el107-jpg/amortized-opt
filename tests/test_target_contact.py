import numpy as np
import pytest
import torch

from amortized_opt.sdf import segment_cylinder_sdf
from amortized_opt.target_contact import (target_top_entry_mask,
    target_upper_side_entry,target_v4_top_entry_mask,
    v4_contact_probe_distance,v4_static_probe_distance,v4_impact_quality)
from isaac_rl.cymbal_contact_scoring import cymbal_entry_geometry


def entry_mask(tips, *, butts=None, initial_source=False, radius=.15):
    tip = torch.tensor(tips, dtype=torch.float32)[None]
    butt = tip+torch.tensor([0., 0., .4]) if butts is None else torch.tensor(butts, dtype=torch.float32)[None]
    center = torch.tensor([[0., 0., -.1]])
    normal = torch.tensor([[0., 0., 1.]])
    sdf = segment_cylinder_sdf(butt, tip, center[:, None], normal[:, None], radius, .1, ra=.008)
    return target_top_entry_mask(butt, tip, tip, tip, torch.zeros(1, 3), normal,
        torch.tensor([radius]), sdf,
        stick_radius=.008, depth=.07,
        already_contacting_source=torch.tensor([initial_source]))[0][0]


def test_early_top_entry_allows_continued_shallow_contact():
    mask = entry_mask([[0, 0, .1], [0, 0, .02], [0, 0, -.02], [0, 0, -.055]])
    assert mask.tolist() == [False, False, True, True]


def test_side_entry_does_not_get_a_target_exemption():
    mask = entry_mask([[.25, 0, -.03], [.18, 0, -.03], [.12, 0, -.03], [.08, 0, -.03]])
    assert not mask.any()


def test_side_reentry_cannot_inherit_an_earlier_top_entry():
    mask = entry_mask([[0, 0, .1], [0, 0, -.02], [.25, 0, -.02], [.08, 0, -.02]])
    assert mask[1]
    assert not mask[2:].any()


def test_shaft_side_entry_before_tip_reaches_top_is_not_exempt():
    mask = entry_mask([[.25, 0, .1], [.25, 0, .02], [.08, 0, -.02]],
                      butts=[[.25, 0, .5], [0, 0, -.1], [.08, 0, .38]])
    assert not mask.any()


def test_post_top_contact_continuation_remains_allowed_within_depth_proxy():
    mask = entry_mask([[0, 0, .1], [0, 0, -.02], [0, 0, -.05]],
                      butts=[[0, 0, .5], [0, 0, .38], [.2, 0, -.05]])
    assert mask.tolist() == [False, True, True]


def test_entry_outside_physical_head_is_not_exempt():
    mask = entry_mask([[.16, 0, .1], [.16, 0, -.02]])
    assert not mask.any()


def test_physical_top_entry_may_be_outside_the_smaller_endpoint_goal_disk():
    assert entry_mask([[.13, 0, .1], [.13, 0, -.02]])[1]


def test_deeper_than_contact_band_is_not_exempt():
    mask = entry_mask([[0, 0, .1], [0, 0, -.02], [0, 0, -.08]])
    assert mask[1] and not mask[2]


def test_known_source_contact_may_start_in_head_but_other_sources_may_not():
    tips = [[0, 0, -.005], [0, 0, -.02]]
    assert entry_mask(tips, initial_source=True).all()
    assert not entry_mask(tips).any()


def test_rotated_head_uses_normal_rather_than_world_height():
    tip = torch.tensor([[[0., 0., .1], [0., 0., -.02], [0., 0., -.04]]])
    butt = tip+torch.tensor([0., 0., .4])
    angle = .7
    R = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]], dtype=torch.float32)
    normal = (R@torch.tensor([0., 0., 1.]))[None]
    tip = tip@R.T; butt = butt@R.T
    sdf = segment_cylinder_sdf(butt, tip, (-.1*normal)[:, None], normal[:, None], .15, .1, ra=.008)
    mask, _ = target_top_entry_mask(butt, tip, tip, tip, torch.zeros(1, 3), normal,
        torch.tensor([.15]), sdf, stick_radius=.008, depth=.07,
        already_contacting_source=torch.tensor([False]))
    assert mask[0].tolist() == [False, True, True]


def test_playing_shaft_top_contact_when_both_endpoints_are_outside_disk():
    # A cymbal is struck with its shaft: neither endpoint is inside the disk,
    # while the segment's center descends through its playable top.
    a = torch.tensor([[[-.18, 0., .1], [-.18, 0., -.02], [-.18, 0., -.05]]])
    b = a+torch.tensor([.36, 0., 0.])
    n = torch.tensor([[0., 0., 1.]])
    sdf = segment_cylinder_sdf(a, b, torch.tensor([[[0., 0., -.1]]]), n[:, None], .15, .1, ra=.008)
    mask, _ = target_top_entry_mask(a, b, a, b, torch.zeros(1, 3), n,
        torch.tensor([.15]), sdf, stick_radius=.008, depth=.07,
        already_contacting_source=torch.tensor([False]))
    assert mask[0].tolist() == [False, True, True]


def test_failed_entry_has_a_repair_gradient_without_granting_an_exemption():
    tip = torch.tensor([[[.155, 0., .03], [.155, 0., .005], [.155, 0., -.02]]], requires_grad=True)
    butt = tip + torch.tensor([0., 0., .4])
    n = torch.tensor([[0., 0., 1.]])
    sdf = segment_cylinder_sdf(butt, tip, torch.tensor([[[0., 0., -.1]]]), n[:, None], .15, .1, ra=.01)
    mask, _, distance = target_top_entry_mask(butt, tip, tip, tip, torch.zeros(1, 3), n,
        torch.tensor([.15]), sdf, stick_radius=.01, depth=.07,
        already_contacting_source=torch.tensor([False]), return_repair_distance=True)
    assert not mask.any()
    assert distance.max() > .001
    gradient = torch.autograd.grad(distance.sum(), tip)[0]
    assert gradient[..., 0].max() > .5


def test_downward_top_contact_near_rim_may_overhang_the_head():
    # The centerline is on the physical top face. The outer part of the
    # capsule overhangs the rim, which does not make this a side entry.
    mask = entry_mask([[.145, 0, .03], [.145, 0, .005], [.145, 0, -.02]])
    assert mask.tolist() == [False, True, True]


def test_same_near_rim_position_reached_from_side_is_rejected():
    assert not entry_mask([[.19, 0, -.02], [.145, 0, -.02], [.13, 0, -.02]]).any()


def test_upper_side_surrogate_matches_v4_geometry_and_rejects_horizontal_graze():
    head=torch.zeros(1,3);normal=torch.tensor([[0.,0.,1.]])
    radius=torch.tensor([.15]);probe_radii=torch.tensor([.01])
    body_sdf=torch.full((1,3),-.001)
    def classify(before,after):
        points=torch.tensor([[[[.159,0.,.08]],[[*before,]],[[*after,]]]],dtype=torch.float32)
        geom,inc,allowed,repair=target_upper_side_entry(points,head,normal,radius,
            probe_radii,body_sdf,dt=.004)
        top,side,_,_=cymbal_entry_geometry(points[:,2:3],points[:,1:2],head,
            normal,radius,torch.tensor([.1]),torch.tensor([True]),
            torch.tensor([True]),torch.tensor([[True]]))
        assert not top.item()
        assert geom[0,2,0].item()==side.item()
        return inc[0,2,0].item(),allowed[0,2].item(),repair[0,2].item()
    assert classify((.159,0.,.001),(.155,0.,-.003))[1]  # downward diagonal
    inc,allowed,repair=classify((.159,0.,-.003),(.155,0.,-.003))
    assert not inc and not allowed and repair>.003  # horizontal graze
    inc,allowed,repair=classify((.159,0.,-.002),(.155,0.,-.006))
    assert inc and not allowed and repair>.0009  # 6 mm below the top


def test_upper_side_repair_has_gradients_toward_height_and_downward_incidence():
    common=dict(head=torch.zeros(1,3),normal=torch.tensor([[0.,0.,1.]]),
        body_radius=torch.tensor([.15]),probe_radii=torch.tensor([.01]),
        body_sdf=torch.full((1,3),-.001),dt=.004)
    deep=torch.tensor([[[[.159,0.,.08]],[[.159,0.,-.002]],[[.155,0.,-.006]]]],requires_grad=True)
    _,_,allowed,repair=target_upper_side_entry(deep,**common)
    assert not allowed[0,2]
    gradient=torch.autograd.grad(repair[0,2],deep)[0]
    assert gradient[0,2,0,2]<0
    flat=torch.tensor([[[[.159,0.,.08]],[[.159,0.,-.003]],[[.155,0.,-.003]]]],requires_grad=True)
    _,_,allowed,repair=target_upper_side_entry(flat,**common)
    assert not allowed[0,2]
    gradient=torch.autograd.grad(repair[0,2],flat)[0]
    assert gradient[0,2,0,2]>0  # descent moves the contact down into the head


def test_v4_speed_quality_matches_the_slow_isaac_side_entry():
    velocity=torch.tensor([[0.,0.,-.16412315]],requires_grad=True)
    normal=torch.tensor([[0.,0.,1.]])
    a,t,quality,incidence,deficit=v4_impact_quality(velocity,normal,torch.tensor([2.4]))
    assert a.item()==pytest.approx(.16412315)
    assert t.item()==pytest.approx(0.,abs=1e-7)
    assert quality.item()==pytest.approx(.173,abs=.003)
    assert deficit.item()>0 and incidence.item()==0
    gradient=torch.autograd.grad(deficit.sum(),velocity)[0]
    assert gradient[0,2]>0  # descent asks for more downward impact speed
    _,_,_,incidence,_=v4_impact_quality(torch.tensor([[-1.,0.,0.]]),normal,
        torch.tensor([2.4]))
    assert incidence.item()==pytest.approx(1.)


def test_v4_terminal_probe_union_has_zero_at_top_or_downward_upper_side():
    head=torch.zeros(1,3);normal=torch.tensor([[0.,0.,1.]])
    common=dict(head=head,normal=normal,body_radius=torch.tensor([.15]),
        body_height=torch.tensor([.10]),probe_radii=torch.tensor([.01]),dt=.004)
    def distance(a,b):
        return v4_contact_probe_distance(torch.tensor([[a]]),torch.tensor([[b]]),**common)
    union,top,side=distance([.05,0.,.03],[.05,0.,.01])
    assert top.item()<1e-6 and union.item()<1e-6
    union,top,side=distance([.159,0.,.001],[.155,0.,-.003])
    assert side.item()<1e-6 and union.item()<1e-6 and top.item()>.001
    union,_,side=distance([.159,0.,-.003],[.155,0.,-.003])
    assert union.item()>.001 and side.item()>.003  # side graze
    union,_,side=distance([.159,0.,-.002],[.155,0.,-.006])
    assert union.item()>.0009 and side.item()>.0009  # too deep
    union,_,_=distance([.05,0.,-.04],[.05,0.,-.06])
    assert union.item()>.001  # underneath cannot inherit a top entry


def test_v4_terminal_probe_distance_has_a_downward_gradient_at_horizontal_side():
    before=torch.tensor([[[.159,0.,-.003]]])
    after=torch.tensor([[[.155,0.,-.003]]],requires_grad=True)
    union,_,_=v4_contact_probe_distance(before,after,torch.zeros(1,3),
        torch.tensor([[0.,0.,1.]]),torch.tensor([.15]),torch.tensor([.1]),
        torch.tensor([.01]),dt=.004)
    gradient=torch.autograd.grad(union.sum(),after)[0]
    assert gradient[0,0,2]>0


def test_v4_top_entry_latches_only_within_the_same_contact_episode():
    points=torch.tensor([[[[0.,0.,.1]],[[0.,0.,.02]],
                          [[0.,0.,-.02]],[[0.,0.,-.05]]]])
    common=dict(head=torch.zeros(1,3),normal=torch.tensor([[0.,0.,1.]]),
        body_radius=torch.tensor([.15]),body_height=torch.tensor([.1]),
        probe_radii=torch.tensor([.01]),dt=.02,
        already_contacting_source=torch.tensor([False]))
    sdf=torch.tensor([[.08,.01,-.001,-.001]])
    allowed,entry,_=target_v4_top_entry_mask(points,body_sdf=sdf,**common)
    assert allowed[0].tolist()==[False,False,True,True]
    assert entry[0].tolist()==[False,False,True,False]
    # Leaving contact clears the top certificate; coming back from the side
    # 2 cm down cannot inherit the earlier top-face event.
    side=points.clone();side[0,1,0]=torch.tensor([0.,0.,-.02])
    side[0,2,0]=torch.tensor([.25,0.,-.02]);side[0,3,0]=torch.tensor([.155,0.,-.02])
    allowed,_,_=target_v4_top_entry_mask(side,
        body_sdf=torch.tensor([[.08,-.001,.10,-.001]]),**common)
    assert allowed[0,1] and not allowed[0,3]


def test_v4_top_entry_can_be_certified_by_a_shaft_probe():
    points=torch.tensor([[[[.25,0.,.1],[0.,0.,.1]],
                          [[.25,0.,.02],[0.,0.,.02]],
                          [[.25,0.,-.02],[0.,0.,-.02]]]])
    allowed,entry,_=target_v4_top_entry_mask(points,torch.zeros(1,3),
        torch.tensor([[0.,0.,1.]]),torch.tensor([.15]),torch.tensor([.1]),
        torch.tensor([.01,.008]),torch.tensor([[.1,.01,-.001]]),dt=.02,
        already_contacting_source=torch.tensor([False]))
    assert entry[0,2] and allowed[0,2]


def test_v4_static_pose_accepts_far_top_and_upper_side_but_not_deep_side():
    common=dict(head=torch.zeros(1,3),normal=torch.tensor([[0.,0.,1.]]),
        body_radius=torch.tensor([.15]),probe_radii=torch.tensor([.01]))
    def distance(x,z):
        return v4_static_probe_distance(torch.tensor([[[x,0.,z]]]),**common)[0].item()
    assert distance(-.14,.01)<1e-6  # Full top disk, including the far side.
    assert distance(.155,-.003)<1e-6
    assert distance(.155,-.006)>.0009
    assert distance(.155,-.03)>.02
