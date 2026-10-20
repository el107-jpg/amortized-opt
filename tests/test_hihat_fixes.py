from types import SimpleNamespace

import numpy as np
import torch

from amortized_opt.rhythm import DrumKit
from amortized_opt.strike import (HIHAT_AIM_INSET_M, STRIKE_SURFACE_OFFSET,
                                 strike_aim_on_head, strike_disk_center_offset)
from isaac_rl.trajectory_qp import fully_feasible


def test_inset_aim_keeps_the_legal_disk_fixed_across_head_sizes_and_tilts():
    normal=np.array([.2,.1,1.]);normal/=np.linalg.norm(normal)
    center=np.array([.5,.5,1.])
    for radius in (.08,.15,.25):
        aim,legal_radius=strike_aim_on_head(center,normal,radius,name='hihat')
        disk_offset=strike_disk_center_offset(center,normal,radius,name='hihat')
        np.testing.assert_allclose(aim+disk_offset,center-STRIKE_SURFACE_OFFSET*normal,atol=1e-12)
        assert np.isclose(np.linalg.norm(disk_offset),radius-HIHAT_AIM_INSET_M)
        assert legal_radius==radius


def test_elevation_gate_exempts_cymbals_but_preserves_each_active_pad():
    z=torch.zeros(4)
    diagnostics={k:z.clone() for k in ['collision_min_sdf','collision_stick_min_sdf',
        'torque_excess_max','velocity_excess_max','joint_box_excess_max']}
    angles=torch.tensor([[60.,0.],[60.,0.],[2.,70.],[20.,70.]])
    diagnostics.update(contact_dir_cos=z+1,stick_elev_deg=angles.mean(-1),
                       stick_elev_deg_per_hand=angles)
    terms=SimpleNamespace(total=z,diagnostics=diagnostics)
    ctx=SimpleNamespace(active=torch.tensor([[1.,0.],[1.,0.],[1.,1.],[1.,1.]]),
        elev_required=torch.tensor([[0.,0.],[1.,0.],[1.,0.],[1.,0.]]))
    assert fully_feasible(terms,z,ctx).tolist()==[True,False,False,True]
    diagnostics['torque_excess_max'][0]=.01
    assert not fully_feasible(terms,z,ctx)[0]


def test_training_eligibility_includes_right_hihat():
    kit=DrumKit.from_layout();hat=kit.names.index('hihat');floor=kit.names.index('floor_tom')
    assert hat in kit.eligible('left') and hat in kit.eligible('right')
    assert floor not in kit.eligible('left') and floor in kit.eligible('right')


def test_right_hihat_initializer_uses_valid_contact_without_changing_checkpoint_keys():
    from amortized_opt.constants import N_DOF, ready_joint_pos
    from amortized_opt.fk import ChainFK
    from amortized_opt.ik import tip_position
    from amortized_opt.rhythm import sample_hit_point
    from amortized_opt.strike import HIHAT_SHOULDER_SIGMA
    from amortized_opt.terminal_initializer import TerminalInitializer
    from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
    kit=DrumKit.from_layout();hat=kit.names.index('hihat')
    initializer=TerminalInitializer(7)
    ctx=ContextInputs.zeros(1,num_drums=7,num_future=1,device='cpu')
    ctx.drum_pos[:]=torch.tensor(kit.positions);ctx.drum_axis[:]=torch.tensor(kit.axes)
    ctx.drum_radius[:]=torch.tensor(kit.radii);ctx.drum_valid[:]=True
    ctx.goal_drum[0,1]=hat;ctx.goal_valid[0,1]=True;ctx.goal_tth[0,1]=1
    target=torch.tensor(sample_hit_point(kit,hat),dtype=torch.float32)
    ctx.goal_pos[0,1]=target;ctx.aim_valid[:]=True;ctx.aim_yaw[:]=0
    state=torch.zeros(1,2*N_DOF);state[:,:N_DOF]=torch.tensor(ready_joint_pos())
    masks=build_waypoint_masks(ctx.goal_tth[:,0],ctx.goal_tth[:,1],max_horizon=50,
        active_left=ctx.goal_valid[:,0],active_right=ctx.goal_valid[:,1])
    keys=set(initializer.state_dict());pose=initializer.terminal_pose(ctx,state,masks)
    contact=tip_position(ChainFK(),pose,'right',sigma=HIHAT_SHOULDER_SIGMA,frame='env')
    assert (contact[0]-target).norm()<.001
    assert set(initializer.state_dict())==keys
    assert not any('hihat' in key for key in keys)
    assert not pose.requires_grad


def test_actor_retention_keeps_gradients_off_new_goals_and_padding():
    from amortized_opt.train_cost import actor_retention_loss
    class Teacher(torch.nn.Module):
        terminal_initializer = SimpleNamespace(hihat_index=1)
        def encode_context(self, ctx):
            return None
        def refine(self, init, state, encoding, masks, iterations):
            return SimpleNamespace(traj=torch.zeros_like(init))
    model=torch.nn.Linear(1,1);keys=set(model.state_dict())
    object.__setattr__(model,'_actor_retention_teacher',Teacher().eval().requires_grad_(False))
    traj=torch.full((3,2,17),.1,requires_grad=True)
    mask=torch.ones_like(traj);mask[2,1]=0
    ctx=SimpleNamespace(goal_valid=torch.tensor([[True,False],[False,True],[True,False]]),
                        goal_drum=torch.tensor([[1,0],[0,1],[1,1]]))
    batch=dict(context=ctx,init=torch.zeros_like(traj),state=torch.zeros(3,34),
               masks=SimpleNamespace(elem_mask=mask))
    loss=actor_retention_loss(model,traj,batch)
    gradient=torch.autograd.grad(loss,traj)[0]
    assert gradient[0].abs().sum()>0 and gradient[2,0].abs().sum()>0
    assert not gradient[1].any() and not gradient[2,1].any()
    assert set(model.state_dict())==keys


def test_retention_can_release_hihat_departures_without_releasing_unrelated_rows():
    from amortized_opt.train_cost import actor_retention_loss
    class Teacher:
        terminal_initializer = SimpleNamespace(hihat_index=1)
        def encode_context(self, ctx): return None
        def refine(self, init, state, encoding, masks, iterations):
            return SimpleNamespace(traj=torch.zeros_like(init))
    model=SimpleNamespace(_actor_retention_teacher=Teacher(),
                          _actor_retention_hihat_departures=True)
    traj=torch.full((4,2,17),.1,requires_grad=True)
    ctx=SimpleNamespace(goal_valid=torch.ones(4,2,dtype=torch.bool),
                        goal_drum=torch.tensor([[2,3],[2,3],[2,3],[2,1]]))
    strike=SimpleNamespace(active=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]),
        start_drum=torch.tensor([[1,-1],[-1,1],[2,1],[-1,3]]),target_drum=ctx.goal_drum)
    batch=dict(context=ctx,strike=strike,init=torch.zeros_like(traj),state=torch.zeros(4,34),
               masks=SimpleNamespace(elem_mask=torch.ones_like(traj)))
    grad=torch.autograd.grad(actor_retention_loss(model,traj,batch),traj)[0]
    assert not grad[0].any() and not grad[1].any() and not grad[3].any()
    assert grad[2].abs().sum()>0  # an inactive hi-hat hand must not release this row
