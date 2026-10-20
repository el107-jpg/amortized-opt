import numpy as np
from tests.test_neural_hand_plans import clip
from amortized_opt.constants import ready_joint_pos
from isaac_rl.hand_plan_ranker import cartesian_tips
from isaac_rl.hand_plan_sweep import perturb_case,sensitivity_summary


def state():
    q=np.array(ready_joint_pos());return dict(q=q.tolist(),qd=np.full(17,.2).tolist(),
        tips=cartesian_tips(q).tolist(),initial_drums=[0,1],kind='ready')


def test_timing_changes_one_interval_and_preserves_all_discrete_requests():
    base=clip();s=state();direction=np.r_[np.zeros(3),np.ones(14)]
    lead,pose,reason=perturb_case(base,s,'first_delay',.007,direction,None)
    assert reason is None;np.testing.assert_allclose([h.time for h in lead.hits],np.array([h.time for h in base.hits])+.007)
    assert pose==s and pose is not s
    later,_,_=perturb_case(base,s,'second_gap',-.013,direction,None)
    np.testing.assert_allclose([h.time for h in later.hits],np.array([h.time for h in base.hits])+[0,-.013,-.013,-.013])
    assert [(h.drum,h.velocity,h.hand) for h in later.hits]==[(h.drum,h.velocity,h.hand) for h in base.hits]
    assert perturb_case(base,s,'first_delay',-1.,direction,None)[2] is not None


def test_pose_sweep_keeps_velocities_and_history_and_refreshes_cartesian_tips():
    base=clip();s=state();direction=np.r_[np.zeros(3),np.ones(14)]
    _,pose,reason=perturb_case(base,s,'arm_direction',.01,direction,None)
    assert reason is None and pose['qd']==s['qd'] and pose['initial_drums']==s['initial_drums']
    np.testing.assert_allclose(np.array(pose['q'])-s['q'],direction*.01,atol=1e-6)
    np.testing.assert_allclose(pose['tips'],cartesian_tips(pose['q']))
    assert not np.array_equal(pose['tips'],s['tips'])
    _,yaw,_=perturb_case(base,s,'waist_yaw',.02,direction,None)
    np.testing.assert_allclose(np.array(yaw['q'])[1:],np.array(s['q'])[1:])


def test_sensitivity_separates_repeatability_from_hit_boundary_and_near_tie_switch():
    points=[];reports=[]
    def point(axis,step,rewards,hits):
        ids=[]
        for r,h in zip(rewards,hits):
            ids.append(len(reports));reports.append(dict(reward=r,hits=h,near_miss_bonus=0.,per_hit_outcomes=[dict(hit=bool(h))]))
        points.append(dict(case=0,axis=axis,step=step,valid=True,actions=[0,1],neural_action=1,report_ids=ids))
    point('control',0,[.1,.21],[0,1]);point('control',0,[.1,.21],[0,1])
    point('first_delay',-1,[.09,.22],[0,1]);point('first_delay',0,[.1,.21],[0,1]);point('first_delay',1,[.9,.2],[1,1])
    point('first_delay',3,[10.,10.],[1,1])  # Missing step2 must not create an apparent adjacent jump.
    result=sensitivity_summary(points,reports)
    assert result['repeatability']['reward_range_max']==0
    axis=result['axes']['first_delay']
    assert axis['plan_neighbors']==4 and axis['case_point_neighbors']==2
    assert axis['fraction_hit_credit_changed']==.25
    assert axis['reward_difference_max']==.8
    assert axis['strict_best_action_switches']==1 and axis['disjoint_near_best_set_switches']==1


def test_fresh_scene_noise_bound_excludes_reused_scene_controls():
    reports=[dict(reward=r,hits=0,near_miss_bonus=0.,per_hit_outcomes=[dict(hit=False)]) for r in [.2,.8,.2,.2]]
    points=[dict(case=0,axis='control',control_axis='first_delay',control_kind=kind,
        repeat=repeat,step=0,valid=True,actions=[0],neural_action=0,report_ids=[i])
        for i,(kind,repeat) in enumerate([('control',0),('control',1),('fresh_control',1),('fresh_control',2)])]
    assert sensitivity_summary(points,reports)['repeatability']['reward_range_max']>.5
    assert sensitivity_summary(points,reports,fresh_scene_baselines=True)['repeatability']['reward_range_max']==0
