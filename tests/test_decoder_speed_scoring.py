"""Conformance of the installed Isaac reward path with decoder impact commands."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from amortized_opt.rhythm import DrumKit,Hit,RhythmClip,strike_speed,strike_velocity
from isaac_rl.chart_cma_isaac import IsaacChartEvaluator
from isaac_rl.hand_plan_dataset import require_matching_rollout_runtime


def installed_reward_functions():
    # Extract the actual lightweight speed-target method and reward expressions,
    # without importing Isaac's GUI/native bindings into the unit-test process.
    root=Path(importlib.util.find_spec('g1_drummer').origin).parent
    path=root/'tasks/direct/g1_drum/g1_drummer_env.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='G1DrummerEnv')
    helper=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_beat_target_speeds')
    rewards=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
    expressions=[copy.deepcopy(n) for n in rewards.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id in ('v_target','speed_ratio','speed_quality') for t in n.targets)]
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(helper)],type_ignores=[])),str(path),'exec'),ns)
    code=compile(ast.fix_missing_locations(ast.Module(body=expressions,type_ignores=[])),str(path),'exec')
    return ns['_beat_target_speeds'],code


def test_installed_reward_grades_exact_decoder_speed_and_preserves_normalized_loudness():
    kit=DrumKit.from_layout();velocities=[0.,.2,.5,.8,1.]
    targets=[strike_speed(v) for v in velocities]
    for drum in range(7):
        np.testing.assert_allclose([np.linalg.norm(strike_velocity(kit,drum,v)) for v in velocities],targets)
    helper,code=installed_reward_functions()
    loudness=torch.tensor([velocities]);env=SimpleNamespace(beat_velocity=loudness,cfg=SimpleNamespace(strike_speed_ref=2.),
        beat_target_speed_m_s=torch.tensor([targets]))
    env._beat_target_speeds=lambda:helper(env)
    ns={'self':env,'torch':torch,'beat_entry_speed':torch.tensor([targets]),'m_shape':1.}
    exec(code,ns)
    torch.testing.assert_close(ns['speed_quality'],torch.ones_like(loudness))
    torch.testing.assert_close(env.beat_velocity,loudness)
    env.beat_target_speed_m_s=None
    torch.testing.assert_close(helper(env),2*loudness)


def test_mismatch_can_reject_requested_ghost_strike_but_correct_mapping_accepts_it():
    helper,code=installed_reward_functions()
    env=SimpleNamespace(beat_velocity=torch.tensor([[.2]]),cfg=SimpleNamespace(strike_speed_ref=2.),beat_target_speed_m_s=None)
    env._beat_target_speeds=lambda:helper(env)
    ns={'self':env,'torch':torch,'beat_entry_speed':torch.tensor([[strike_speed(.2)]]),'m_shape':1.}
    exec(code,ns);timing_quality=np.exp(-(.2/.4)**2)
    assert float(ns['speed_quality'])*timing_quality<.25
    env.beat_target_speed_m_s=torch.tensor([[strike_speed(.2)]])
    exec(code,ns)
    assert float(ns['speed_quality'])*timing_quality>.25


def test_episode_install_uses_scheduler_mapping_without_rewriting_musical_velocity():
    kit=DrumKit.from_layout();clip=RhythmClip('speed-test',[Hit(.2+i*.2,.2+i*.2,i,kit.names[i],'left' if i%2==0 else 'right',v)
        for i,v in enumerate([.2,.5,.8,1.])],1.3)
    u=SimpleNamespace(assignment_kit=kit,device='cpu',num_envs=1,beat_valid=torch.zeros(1,8,dtype=torch.bool),
        beat_time=torch.zeros(1,8),beat_velocity=torch.ones(1,8),beat_hand=torch.zeros(1,8,dtype=torch.long),
        beat_drum_id=torch.zeros(1,8,dtype=torch.long),beat_pos=torch.zeros(1,8,3),beat_drum_height=torch.zeros(1,8),
        beat_dipole_offset=torch.zeros(1,8,3),drum_pos_env=torch.zeros(1,7,3),drum_height=torch.zeros(7),drum_dipole_offset=torch.zeros(7,3),
        beat_is_left=torch.zeros(1,8,dtype=torch.bool),beat_is_right=torch.zeros(1,8,dtype=torch.bool))
    e=object.__new__(IsaacChartEvaluator);e.u=u;e.match_decoder_speed=True
    e.mpc=SimpleNamespace(sched=SimpleNamespace(cfg=SimpleNamespace(speed_min=.7,speed_max=2.8)))
    e._install([clip])
    torch.testing.assert_close(u.beat_velocity[0,:4],torch.tensor([h.velocity for h in clip.hits]))
    torch.testing.assert_close(u.beat_target_speed_m_s[0,:4],torch.tensor([strike_speed(h.velocity,speed_min=.7,speed_max=2.8) for h in clip.hits]))
    assert u.beat_target_speed_m_s[~u.beat_valid].count_nonzero()==0


def test_speed_scoring_versions_cannot_mix_even_with_same_geometry_and_decoder():
    old={'rollout_runtime':{'rollout_runtime_version':'cleared_contact_reset_v1'}}
    new={'rollout_runtime':{'rollout_runtime_version':'matched_decoder_speed_contact_reset_v2',
        'impact_speed_target':{'version':'decoder_linear_impact_speed_v1','speed_min_m_s':1.,'speed_max_m_s':3.}}}
    with pytest.raises(ValueError,match='regenerate labels'):require_matching_rollout_runtime(old,new)
