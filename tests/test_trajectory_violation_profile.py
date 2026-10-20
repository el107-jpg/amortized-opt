import numpy as np
import torch
from isaac_rl.trajectory_violation_profile import sampled_time_stats, sampled_excess_integral, profile


def test_duration_gate_measures_each_joint_instead_of_diluting_a_persistent_overload():
    over=np.zeros((2,51,17),bool)
    over[0,20,3]=True
    over[1,:,3]=True
    total,longest=sampled_time_stats(over,1.)
    np.testing.assert_allclose(total[:,3],[.02,1.])
    np.testing.assert_allclose(longest[:,3],[.02,1.])
    np.testing.assert_allclose(total[1].max(),1.)


def test_separated_pulses_have_distinct_total_and_continuous_duration():
    over=np.zeros((1,51,1),bool)
    over[0,[5,10,15,20,25],0]=True
    total,longest=sampled_time_stats(over,1.)
    np.testing.assert_allclose(total,.10)
    np.testing.assert_allclose(longest,.02)


def test_integral_retains_magnitude_endpoints_and_sums_actuators_without_dilution():
    excess = np.zeros((2, 5, 2))
    excess[0, 1, 0] = 4.  # Triangle: height 4 Nm, width .5 s -> 1 Nm s.
    excess[0, 0, 1] = 4.  # Half-width endpoint triangle -> .5 Nm s.
    excess[1, :, :] = 5.  # Two actuators at 5 Nm for a full second.
    integral = sampled_excess_integral(excess, 1.)
    np.testing.assert_allclose(integral, [[1., .5], [5., 5.]])
    np.testing.assert_allclose(integral.sum(-1), [1.5, 10.])
    np.testing.assert_allclose(sampled_excess_integral(excess, 2.), integral*2)


def test_profile_capture_preserves_cost_and_scoped_constraint_peaks():
    from amortized_opt.cost import FusedCost,CostConfig
    from amortized_opt.constants import ready_joint_pos
    from amortized_opt.rhythm import DrumKit
    from isaac_rl.rnea_cache import build_or_load
    from isaac_rl.transition_calibration import strike_context
    torch.set_num_threads(1)
    cost=FusedCost(CostConfig(physics_mode='knot_fd',w_velocity=1.,w_joint_box=1.),
        max_horizon=4,device='cpu',rnea=build_or_load(device='cpu'))
    cost.calibration_active_hand_only=True
    q=torch.as_tensor(ready_joint_pos()[None],dtype=torch.float32);q[:,0]=.3
    v=torch.zeros_like(q);traj=q[:,None].expand(-1,4,-1).clone()
    lengths=torch.full((1,3),4)
    ctx=strike_context(DrumKit.from_layout(),[[-1,2]],[[-1,2]],[4],'cpu')
    ctx.target_top_entry_depth_m=.07
    original=cost(q,traj,v,ctx,lengths=lengths)
    capture={};observed=cost(q,traj,v,ctx,lengths=lengths,constraint_profile_capture=capture)
    torch.testing.assert_close(original.total,observed.total,atol=0,rtol=0)
    assert capture['torque_tau'].shape==(1,5,17)
    assert capture['collision_right_body'].shape[1]==4*cost.cfg.n_sub+1
    assert not capture['torque_mask'][:,:,3:10].any()
    result=profile(cost,q,v,traj,ctx,lengths)
    assert result['torque_over_fraction'].shape==(1,)
    assert np.isfinite(result['collision_over_duration_s']).all()
    np.testing.assert_array_equal(result['torque_excess_integral_per_joint_Nms'][:,3:10],0.)
    np.testing.assert_allclose(result['torque_excess_average_Nm'],result['torque_excess_integral_Nms']/.08)
