"""Magnitude and sampled duration of constraint violations, without averaging joints."""
from dataclasses import dataclass
import numpy as np
import torch
from isaac_rl.trajectory_qp import fully_feasible, elevation_feasible
from isaac_rl.transition_calibration import exact_contact_region


@dataclass(frozen=True)
class SmallViolationAllowance:
    torque_peak_ratio: float = 1.10
    torque_duration_fraction: float = .10
    torque_continuous_s: float = .040
    collision_depth_m: float = .005
    joint_excess_rad: float = 0.
    terminal_distance_m: float = 0.
    torque_average_limit_Nm: float | None = None


def sampled_time_stats(over, duration):
    """Trapezoidal time weights on the control/collocation grid, per actuator.

    Duration and longest run are estimates from the sampled demand. They do
    not claim to resolve pulses between samples or model thermal limits.
    """
    over=np.asarray(over,bool)
    if over.ndim!=3 or over.shape[1]<2 or not np.isfinite(duration) or duration<=0:
        raise ValueError('Expected batch/time/channel flags and positive duration')
    weights=np.full(over.shape[1],duration/(over.shape[1]-1))
    weights[[0,-1]]*=.5
    total=(over*weights[None,:,None]).sum(1)
    run=np.zeros_like(total);longest=run.copy()
    for i,w in enumerate(weights):
        run=np.where(over[:,i],run+w,0.)
        longest=np.maximum(longest,run)
    return total,longest


def sampled_excess_integral(excess, duration):
    """Per-actuator positive excess area (Nm s), with trapezoidal weights."""
    excess = np.asarray(excess, dtype=np.float64)
    if (excess.ndim != 3 or excess.shape[1] < 2 or not np.isfinite(duration)
            or duration <= 0 or not np.isfinite(excess).all() or (excess < 0).any()):
        raise ValueError('Expected finite nonnegative batch/time/channel excess and positive duration')
    return (excess[:, 1:]+excess[:, :-1]).sum(1)*(duration/(excess.shape[1]-1))*.5


@torch.no_grad()
def profile(cost,q,v,traj,ctx,lengths,*,depth_m=.07,allowance=None):
    allowance=allowance or SmallViolationAllowance()
    capture={}
    terms=cost(q,traj,v,ctx,lengths=lengths,constraint_profile_capture=capture)
    region=exact_contact_region(cost,traj,ctx,depth_band_m=depth_m)
    diagnostics=terms.diagnostics
    duration=traj.shape[1]*cost.spline.dt
    tau=capture['torque_tau'];limit=capture['torque_available'];mask=capture['torque_mask']>.5
    excess=torch.where(mask,(tau.abs()-limit).clamp_min(0),0.)
    ratio=torch.where(mask,tau.abs()/limit.clamp_min(1e-9),0.)
    over=(excess>1e-6).cpu().numpy()
    total,longest=sampled_time_stats(over,duration)
    integral = sampled_excess_integral(excess.cpu().numpy(), duration)
    signed=torch.cat([val for name,val in capture.items() if name.startswith('collision_')],-1)
    collision=signed.amax(-1)
    coll_total,coll_longest=sampled_time_stats((collision>1e-6).cpu().numpy()[...,None],duration)
    out={name:diagnostics[name].cpu().numpy() for name in ('torque_excess_max',
        'velocity_excess_max','joint_box_excess_max','contact_dir_cos','stick_elev_deg','contact_speed_err')}
    out.update(terminal_region_dist=region.cpu().numpy(),
        collision_min_sdf=torch.minimum(diagnostics['collision_min_sdf'],diagnostics['collision_stick_min_sdf']).cpu().numpy(),
        strict_success=fully_feasible(terms,region,ctx).cpu().numpy(),
        torque_peak_ratio=ratio.amax((1,2)).cpu().numpy(),
        torque_over_duration_s=total.max(-1),torque_over_fraction=total.max(-1)/duration,
        torque_longest_run_s=longest.max(-1),
        torque_excess_integral_per_joint_Nms=integral,
        torque_excess_average_per_joint_Nm=integral/duration,
        torque_excess_integral_Nms=integral.sum(-1),
        torque_excess_average_Nm=integral.sum(-1)/duration,
        torque_excess_average_max_joint_Nm=integral.max(-1)/duration,
        collision_over_duration_s=coll_total[:,0],collision_longest_run_s=coll_longest[:,0],
        collision_over_fraction=coll_total[:,0]/duration)
    np.testing.assert_allclose(excess.amax((1,2)).cpu().numpy(),out['torque_excess_max'],atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(-signed.amax((1,2)).cpu().numpy(),out['collision_min_sdf'],atol=1e-5,rtol=1e-5)
    non_torque=(torch.isfinite(terms.total).cpu().numpy()
        &(out['terminal_region_dist']<=allowance.terminal_distance_m+1e-6)
        &(out['velocity_excess_max']<=1e-6)&(out['joint_box_excess_max']<=allowance.joint_excess_rad+1e-6)
        &(out['contact_dir_cos']>=.5)&elevation_feasible(diagnostics,ctx).cpu().numpy())
    out['relaxed_non_torque_success']=(non_torque
        &(out['collision_min_sdf']>=-allowance.collision_depth_m-1e-6))
    out['relaxed_magnitude_success']=(out['relaxed_non_torque_success']
        &(out['torque_peak_ratio']<=allowance.torque_peak_ratio+1e-6))
    out['relaxed_success']=(out['relaxed_magnitude_success']
        &(out['torque_over_fraction']<=allowance.torque_duration_fraction+1e-6)
        &(out['torque_longest_run_s']<=allowance.torque_continuous_s+1e-6))
    if allowance.torque_average_limit_Nm is not None:
        out['average_success']=(out['relaxed_non_torque_success']
            &(out['torque_excess_average_Nm']<=allowance.torque_average_limit_Nm+1e-6))
    return out
