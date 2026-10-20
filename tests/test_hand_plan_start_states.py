from types import SimpleNamespace
import numpy as np
import torch

from amortized_opt.constants import ready_joint_pos, N_DOF
from amortized_opt.limits import urdf_position_limits
from isaac_rl.hand_plan_ranker import (cartesian_tips, encode_clip, build_ranker_batch,
                                      HandPlanRanker, HandPlanRankerConfig)
from isaac_rl.hand_plan_start_states import PostHitStateBank
from isaac_rl.chart_cma_isaac import IsaacChartEvaluator
from tests.test_neural_hand_plans import clip, eligibility


def state(hand=0,drum=0):
    q=ready_joint_pos().copy();q[0]+=.05*drum
    qd=np.full(N_DOF,.25,np.float32)
    return dict(q=q.tolist(),qd=qd.tolist(),tips=cartesian_tips(q).tolist(),
                initial_drums=[drum if hand==0 else -1,drum if hand==1 else -1],
                source_hand=hand,source_drum=drum,kind='post_hit',contact_age_ticks=[2,2])


def test_cartesian_tip_inputs_are_consistent_and_change_with_arm_waist_pose():
    c=clip();q=ready_joint_pos();tips=cartesian_tips(q)
    np.testing.assert_allclose(encode_clip(c)[-6:],tips.reshape(-1))
    np.testing.assert_array_equal(encode_clip(c,q=q),encode_clip(c,q=q,tips=tips))
    moved=q.copy();moved[0]+=.4;moved[3]+=.2
    assert not np.allclose(cartesian_tips(moved),tips)


def test_candidate_trials_share_whole_position_velocity_and_tip_state():
    states=[state(0,0),state(1,1)]
    model=HandPlanRanker(HandPlanRankerConfig(hidden=16))
    batch=build_ranker_batch(model,[clip(),clip()],eligibility(),8,states)
    for (row,action),physical in zip(batch['labels'],batch['episode_states']):
        assert physical==states[row]
    assert not np.array_equal(batch['observations'][0],batch['observations'][1])


def test_bank_reservoir_mixture_has_real_velocities_and_filters_invalid_states():
    lower,upper=urdf_position_limits(posture_limits=False)
    bank=PostHitStateBank(lower,upper,lambda q:np.full(len(q),.1),per_combination=3)
    rng=np.random.default_rng(2)
    result=bank.add([state(0,0)]*12+[state(1,1)]*12+[state(0,6)],rng)
    assert result['combinations']==2 and result['bank_size']==6
    assert bank.seen[(0,0)]==12
    rows,metrics=bank.sample(rng,256)
    assert metrics['post_hit_count']==205 and metrics['random_count']==38 and metrics['ready_count']==13
    assert len(rows)==256 and metrics['waist_yaw_std_rad']>0
    assert metrics['arm_position_std_mean_rad']>0
    assert metrics['velocity_rms_rad_s']>0
    assert all(np.asarray(s['qd']).any() for s in rows if s['kind']=='post_hit')
    invalid=state();invalid['q'][0]=100.
    assert bank.add([invalid],rng)['accepted']==0
    fresh=PostHitStateBank(lower,upper,lambda q:np.full(len(q),.1))
    fresh.load_state_dict(bank.state_dict())
    assert fresh.states==bank.states and fresh.seen==bank.seen


def test_physical_reset_preserves_qd_and_restores_nominal_on_ready_trial():
    class Robot:
        def __init__(self):
            self.data=SimpleNamespace(joint_pos=torch.zeros(2,N_DOF),joint_vel=torch.zeros(2,N_DOF))
        def write_joint_state_to_sim(self,q,qd,joint_ids,ids):
            self.data.joint_pos[ids]=q;self.data.joint_vel[ids]=qd
    robot=Robot();u=SimpleNamespace(robot=robot,num_envs=2,device='cpu',
        ready_joint_pos=torch.as_tensor(np.stack([ready_joint_pos()]*2),dtype=torch.float32))
    forces=SimpleNamespace(force_matrix_w=torch.full((2,1,7,3),4.),net_forces_w=torch.ones(2,1,3))
    u.contact_sensors={'left':SimpleNamespace(data=forces)}
    def ordinary_reset(ids):
        robot.data.joint_pos[ids]=u.ready_joint_pos[ids];robot.data.joint_vel[ids]=0
    u._reset_idx=ordinary_reset
    env=SimpleNamespace(reset=lambda:u._reset_idx(torch.arange(2)))
    evaluator=object.__new__(IsaacChartEvaluator);evaluator.u=u;evaluator.env=env
    evaluator.clear_reset_contacts=True
    evaluator.pid=SimpleNamespace(ids=list(range(N_DOF)))
    states=[state(0,0),state(1,1)]
    evaluator._reset_from_states(states)
    torch.testing.assert_close(robot.data.joint_pos,torch.as_tensor([s['q'] for s in states]))
    torch.testing.assert_close(robot.data.joint_vel,torch.as_tensor([s['qd'] for s in states]))
    assert u._reset_idx is ordinary_reset
    assert forces.force_matrix_w.count_nonzero()==0
    forces.force_matrix_w.fill_(4.)
    evaluator._reset_from_states(None)
    torch.testing.assert_close(robot.data.joint_pos,torch.as_tensor(np.stack([ready_joint_pos()]*2),dtype=torch.float32))
    assert robot.data.joint_vel.count_nonzero()==0
    assert forces.force_matrix_w.count_nonzero()==0
