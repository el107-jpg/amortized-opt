from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.build_decoder_only_transition_chart import empty_records, accept_decoder
from isaac_rl.trajectory_violation_profile import SmallViolationAllowance, profile


def source_piece():
    return dict(hand=np.asarray(0), source=np.asarray(2), destination=np.array([1, 1, 4]),
        sample=np.array([0, 1, 0]), q=np.arange(9, dtype=np.float32).reshape(3, 3),
        v=np.full((3, 3), .1, np.float32), minimum_s=np.full(3, .1),
        decoder_minimum_s=np.full(3, .2), paths=np.ones((3, 100, 3)))


def test_new_decoder_records_preserve_departures_and_discard_all_old_witnesses():
    source = source_piece(); piece = empty_records(source, 100)
    for key in ('q', 'v', 'sample', 'destination', 'hand', 'source'):
        np.testing.assert_array_equal(piece[key], source[key])
    assert np.isinf(piece['minimum_s']).all()
    assert np.isinf(piece['decoder_minimum_s']).all()
    assert not piece['paths'].any() and not piece['qp_attempts'].any()
    assert not piece['selected_for_qp'].any()


def test_same_raw_output_populates_only_its_passing_dense_gate_and_preserves_best():
    source = source_piece(); strict = empty_records(source, 100); relaxed = empty_records(source, 100)
    dense = {key: np.zeros(2) for key in ('torque_excess_max', 'velocity_excess_max',
        'joint_box_excess_max', 'terminal_region_dist', 'collision_min_sdf', 'contact_speed_err',
        'contact_dir_cos', 'stick_elev_deg', 'torque_peak_ratio', 'torque_over_fraction',
        'torque_over_duration_s', 'torque_longest_run_s', 'torque_excess_integral_Nms',
        'torque_excess_average_Nm', 'torque_excess_average_max_joint_Nm')}
    for key in ('torque_excess_integral_per_joint_Nms','torque_excess_average_per_joint_Nm'):
        dense[key]=np.zeros((2,3))
    dense.update(strict_success=np.array([False, True]), relaxed_success=np.array([True, True]))
    rows = np.array([0, 2]); paths = np.full((2, 10, 3), 2.)
    accept_decoder(strict, rows, 10, paths, dense, 0, 'strict_success')
    accept_decoder(relaxed, rows, 10, paths, dense, 0, 'relaxed_success')
    assert np.isinf(strict['minimum_s'][0]) and strict['minimum_s'][2] == .2
    assert relaxed['minimum_s'][0] == relaxed['minimum_s'][2] == .2
    assert np.isinf(relaxed['minimum_s'][1])
    assert not relaxed['paths'][:, 10:].any()
    accept_decoder(relaxed, rows, 20, np.full((2, 20, 3), 3.), dense, 1, 'relaxed_success')
    assert relaxed['minimum_s'][0] == .2
    assert (relaxed['paths'][0, :10] == 2.).all()
    for piece in (strict, relaxed):
        np.testing.assert_array_equal(piece['q'], source['q'])
        np.testing.assert_array_equal(piece['v'], source['v'])


@pytest.mark.parametrize('bad', [None, 'torque_peak', 'torque_duration', 'collision', 'joint', 'hit', 'velocity'])
def test_relaxed_gate_enforces_each_explicit_allowance_without_diluting_torque_time(monkeypatch, bad):
    import isaac_rl.trajectory_violation_profile as profiling
    ratio = 1.21 if bad == 'torque_peak' else 1.15
    count = 11 if bad == 'torque_duration' else 6
    depth = .0101 if bad == 'collision' else .005
    joint = np.deg2rad(5.01 if bad == 'joint' else 4.)
    region = .0101 if bad == 'hit' else .005
    tau = torch.full((1, 101, 17), .5)
    tau[:, 30:30+count, 3] = ratio
    diagnostics = dict(torque_excess_max=torch.tensor([ratio-1]),
        velocity_excess_max=torch.tensor([.001 if bad == 'velocity' else 0.]),
        joint_box_excess_max=torch.tensor([joint]), contact_dir_cos=torch.tensor([.9]),
        stick_elev_deg=torch.tensor([20.]), contact_speed_err=torch.tensor([0.]),
        collision_min_sdf=torch.tensor([-depth]), collision_stick_min_sdf=torch.tensor([-depth]))
    class Cost:
        spline = SimpleNamespace(dt=.02)
        def __call__(self, q, traj, v, ctx, *, lengths, constraint_profile_capture):
            constraint_profile_capture.update(torque_tau=tau, torque_available=torch.ones_like(tau),
                torque_mask=torch.ones_like(tau), collision_right_body=torch.full((1, 101, 1), depth))
            return SimpleNamespace(total=torch.ones(1), diagnostics=diagnostics)
    monkeypatch.setattr(profiling, 'exact_contact_region', lambda *a, **kw: torch.tensor([region]))
    allowance = SmallViolationAllowance(torque_peak_ratio=1.2, torque_duration_fraction=.1,
        torque_continuous_s=2., collision_depth_m=.01, joint_excess_rad=np.deg2rad(5), terminal_distance_m=.01)
    out = profile(Cost(), torch.zeros((1, 17)), torch.zeros((1, 17)),
        torch.zeros((1, 50, 17)), None, torch.ones((1, 3)), allowance=allowance)
    assert not out['strict_success'][0]
    assert bool(out['relaxed_success'][0]) == (bad is None)
    assert out['torque_over_fraction'][0] == pytest.approx(count/100)


@pytest.mark.parametrize('overloads,mask_last,passes', [([4.9],True,True),([5.1],True,False),
    ([2.6,2.6],True,False),([2.6,2.6],False,True)])
def test_average_gate_replaces_peak_duration_rules_and_aggregates_only_active_actuators(
        monkeypatch,overloads,mask_last,passes):
    import isaac_rl.trajectory_violation_profile as profiling
    tau=torch.full((1,101,17),.5);mask=torch.ones_like(tau)
    for i,excess in enumerate(overloads):tau[:,:,i]=1+excess
    if not mask_last:mask[:,:,len(overloads)-1]=0
    peak=max(overloads[:len(overloads) if mask_last else -1])
    diagnostics=dict(torque_excess_max=torch.tensor([peak]),velocity_excess_max=torch.zeros(1),
        joint_box_excess_max=torch.zeros(1),contact_dir_cos=torch.ones(1),stick_elev_deg=torch.full((1,),20.),
        contact_speed_err=torch.zeros(1),collision_min_sdf=torch.full((1,),.001),
        collision_stick_min_sdf=torch.full((1,),.001))
    class Cost:
        spline=SimpleNamespace(dt=.02)
        def __call__(self,q,traj,v,ctx,*,lengths,constraint_profile_capture):
            constraint_profile_capture.update(torque_tau=tau,torque_available=torch.ones_like(tau),
                torque_mask=mask,collision_right_body=torch.full((1,101,1),-.001))
            return SimpleNamespace(total=torch.ones(1),diagnostics=diagnostics)
    monkeypatch.setattr(profiling,'exact_contact_region',lambda *a,**kw:torch.zeros(1))
    out=profile(Cost(),torch.zeros((1,17)),torch.zeros((1,17)),torch.zeros((1,50,17)),None,
        torch.ones((1,3)),allowance=SmallViolationAllowance(torque_average_limit_Nm=5.))
    assert not out['relaxed_success'][0]  # Persistent, large peak: old rule rejects.
    assert bool(out['average_success'][0])==passes
    expected=sum(overloads[:len(overloads) if mask_last else -1])
    assert out['torque_excess_average_Nm'][0]==pytest.approx(expected)
    assert out['torque_excess_integral_Nms'][0]==pytest.approx(expected)
