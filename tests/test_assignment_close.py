"""Real-return fine-tuning respects the frozen decision reference."""
import copy
import numpy as np
import torch
from isaac_rl.arm_assignment import AssignmentConfig
from isaac_rl.assignment_dqn import AssignmentQPolicy,AnchoredDoubleDQN,Replay


def fixture(lr):
    policy=AssignmentQPolicy(AssignmentConfig(hidden=4,future_hits=2,pending_per_arm=1))
    for p in policy.parameters(): torch.nn.init.zeros_(p)
    with torch.no_grad(): policy.q_head.bias.copy_(torch.tensor([1.,0.]))
    learner=AnchoredDoubleDQN(policy,lr=lr,anchor_weight=0,max_flip_fraction=0)
    replay=Replay(policy.cfg.obs_dim,capacity=8)
    zero=np.zeros(policy.cfg.obs_dim,np.float32)
    replay.add_episode([dict(obs=zero,action=1,reward=10.,discount=0.) for _ in range(8)])
    return learner,replay,zero[None]


def test_excessive_decision_drift_rolls_back_parameters_target_and_optimizer():
    learner,replay,guard=fixture(.5)
    before=copy.deepcopy(learner.policy.state_dict()); target=copy.deepcopy(learner.target.state_dict())
    stats=learner.update(replay,gradient_steps=8,batch=8,guard_obs=guard)
    assert not stats['update_accepted'] and stats['proposed_guard_flip_fraction']==1
    assert stats['guard_flip_fraction']==0 and learner.updates==0
    assert learner.optimizer.param_groups[0]['lr']==.25
    assert not learner.optimizer.state
    for k,v in before.items(): torch.testing.assert_close(learner.policy.state_dict()[k],v)
    for k,v in target.items(): torch.testing.assert_close(learner.target.state_dict()[k],v)


def test_small_accepted_bellman_update_retains_reference():
    learner,replay,guard=fixture(.0001)
    ref=copy.deepcopy(learner.reference.state_dict())
    stats=learner.update(replay,gradient_steps=1,batch=8,guard_obs=guard)
    assert stats['update_accepted'] and stats['guard_flip_fraction']==0
    assert stats['gradient_updates']==1 and stats['value_bellman_loss']>0
    assert learner.policy.q_head.bias[1]>0
    for k,v in ref.items(): torch.testing.assert_close(learner.reference.state_dict()[k],v)


def test_direct_pose_controller_does_not_interrupt_current_hit_for_a_future_assignment():
    from types import SimpleNamespace
    from torch import nn
    from dataclasses import replace
    import amortized_opt as A
    from amortized_opt.cost import CostConfig,FusedCost
    from amortized_opt.constants import ready_joint_pos
    from isaac_rl.assignment_parallel import ParallelPoseMPC
    from isaac_rl.mpc import MPCConfig
    from isaac_rl.schedule import BatchedSchedule,ScheduleConfig
    from isaac_rl.rnea_cache import build_or_load
    kit=A.DrumKit.from_layout()
    class Head(nn.Module):
        def terminal_pose(self,ctx,state,masks): return state[:,:17]+.001
    model=SimpleNamespace(cfg=SimpleNamespace(num_drums=kit.num_drums,num_future=3),terminal_initializer=Head())
    hit=A.Hit(.5,.5,0,kit.names[0],'left',.5)
    clip=A.RhythmClip('test',[hit],.9)
    schedule=BatchedSchedule(1,ScheduleConfig(horizon=50,num_drums=kit.num_drums),device='cpu')
    schedule.reset(torch.tensor([0]),[clip],[kit],rng=None)
    cost=FusedCost(CostConfig(),max_horizon=50,device='cpu',rnea=build_or_load(device='cpu'))
    ctrl=ParallelPoseMPC(model,schedule,MPCConfig(horizon=50,cmd_lead_s=None,
         cmd_slew_eta=None,fresh_terminal_every_tick=True),device='cpu',descent_cost=cost,kit=kit)
    q=torch.tensor(ready_joint_pos()[None],dtype=torch.float32)
    ctrl.reset(torch.tensor([0]),q)
    ctrl.plan(q,torch.zeros_like(q)); before=ctrl.reference.clone()
    # A later right-hand request joins the scheduler without changing the
    # earliest deadline. A committed one-pose reference must stay identical.
    future=replace(hit,time=.8,raw_time=.8,hand='right')
    schedule.reset(torch.tensor([0]),[replace(clip,hits=[hit,future])],[kit],rng=None)
    ctrl.plan(q+.01,torch.zeros_like(q))
    torch.testing.assert_close(ctrl.reference,before)
    assert ctrl.decoder_passes.tolist()==[1]
