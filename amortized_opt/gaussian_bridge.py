"""Gaussian motion noise with soft strike endpoints and coherent state drift.

Velocity follows Brownian motion; position is its exact time integral. Plan
noise is conditioned on its final position and on the terminal velocity of
this project's actual clamped/natural spline, then given Gaussian endpoint
errors. Gaussianity holds before any joint-limit projection by the caller.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

from .constants import N_DOF
from .tspline import spline_gain_matrices


@dataclass
class GaussianBridgeNoise:
    velocity_diffusion: tuple[float, ...] = (1.0,)
    terminal_position_std: tuple[float, ...] = (0.03,)
    terminal_velocity_std: tuple[float, ...] = (0.10,)
    state_velocity_diffusion: tuple[float, ...] = (0.0,)
    state_position_floor: float = 0.01
    state_velocity_floor: float = 0.05
    scale: float = 1.0
    state_times_s: tuple[float, ...] = ()
    # At each time: per-joint (position variance, velocity variance, covariance).
    state_covariance: tuple[tuple[tuple[float, float, float], ...], ...] = ()

    def __post_init__(self):
        for name in ('velocity_diffusion', 'terminal_position_std',
                     'terminal_velocity_std', 'state_velocity_diffusion'):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) not in (1, N_DOF) or any(not math.isfinite(x) or x < 0 for x in value):
                raise ValueError(f'{name} requires 1 or {N_DOF} finite nonnegative values')
            setattr(self, name, value)
        self.state_times_s = tuple(float(x) for x in self.state_times_s)
        self.state_covariance = tuple(tuple(tuple(float(x) for x in joint) for joint in row)
                                      for row in self.state_covariance)
        if self.state_times_s or self.state_covariance:
            if (len(self.state_times_s) != len(self.state_covariance) or len(self.state_times_s) < 2
                or self.state_times_s[0] != 0 or any(not math.isfinite(x) or x < 0 for x in self.state_times_s)
                or any(b <= a for a,b in zip(self.state_times_s,self.state_times_s[1:]))):
                raise ValueError('state covariance times must start at zero and increase')
            for row in self.state_covariance:
                if len(row) != N_DOF:
                    raise ValueError('state covariance table requires every joint')
                for joint in row:
                    if (len(joint) != 3 or any(not math.isfinite(x) for x in joint) or min(joint[:2]) < 0
                        or joint[2]**2 > joint[0]*joint[1]+1e-12):
                        raise ValueError('state covariance must be finite positive semidefinite')
        for name in ('state_position_floor', 'state_velocity_floor', 'scale'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def _joint_scale(values, like):
    return torch.as_tensor(values,device=like.device,dtype=like.dtype).expand(like.shape[-1])


@lru_cache(maxsize=16)
def _conditioning_tables(H, dt, device, dtype):
    gain = np.zeros((H+1,H,2),dtype=np.float64)
    velocity = np.zeros((H+1,H),dtype=np.float64)
    conditional_variance = np.zeros((H+1,H),dtype=np.float64)
    for K in range(1,H+1):
        t = np.arange(1,K+1,dtype=np.float64)*dt
        lo,hi = np.minimum(t[:,None],t[None,:]),np.maximum(t[:,None],t[None,:])
        cov = lo**2*(3*hi-lo)/6
        gx,_ = spline_gain_matrices(K,K,dt)
        velocity[K,:K] = gx[K,1:]
        if K == 1:
            # One waypoint leaves only one independent endpoint degree of
            # freedom. Its spline velocity is determined by its position.
            gain[K,0,0] = 1
            continue
        A = np.stack([np.eye(K)[-1],gx[K,1:]])
        G = np.linalg.solve(A@cov@A.T,A@cov).T
        gain[K,:K] = G
        conditional_variance[K,:K] = np.maximum(np.diag(cov-G@A@cov),0)
    to = lambda x: torch.as_tensor(x,device=device,dtype=dtype)
    return to(gain),to(velocity),to(conditional_variance)


@torch.no_grad()
def sample_bridge(elem_mask, dt, cfg: GaussianBridgeNoise, *, generator=None,
                  return_endpoints=False):
    """Return joint position noise; start position/velocity remain unchanged.

    Prefix masks supply a separate strike horizon for every joint. The final
    position and actual spline velocity have the configured Gaussian errors.
    For one-waypoint horizons, velocity follows the sole position waypoint.
    """
    B,H,D = elem_mask.shape
    if D != N_DOF or not dt > 0:
        raise ValueError('bridge requires the robot joint count and positive dt')
    ref = elem_mask.float()
    randn = lambda *shape: torch.randn(*shape,device=ref.device,dtype=ref.dtype,generator=generator)
    sigma = _joint_scale(cfg.velocity_diffusion,ref)*cfg.scale
    z1,z2 = randn(B,H,D),randn(B,H,D)
    dv = sigma*math.sqrt(dt)*z1
    vprev = torch.cumsum(dv,1)-dv
    dq = torch.cumsum(dt*vprev+sigma*dt**1.5*(.5*z1+z2/math.sqrt(12)),1)
    lengths = ref.sum(1).round().long()
    G,L,_ = _conditioning_tables(H,float(dt),str(ref.device),ref.dtype)
    q_last = dq.gather(1,(lengths-1).clamp_min(0)[:,None,:])[:,0]
    v_last = (L[lengths]*dq.transpose(1,2)).sum(-1)
    target_q = randn(B,D)*_joint_scale(cfg.terminal_position_std,ref)*cfg.scale
    target_v = randn(B,D)*_joint_scale(cfg.terminal_velocity_std,ref)*cfg.scale
    error = torch.stack([q_last-target_q,v_last-target_v],-1)
    corrected = dq.transpose(1,2)-torch.einsum('bdhi,bdi->bdh',G[lengths],error)
    corrected = corrected.transpose(1,2)*ref
    target_q = torch.where(lengths>0,target_q,torch.zeros_like(target_q))
    target_v = torch.where(lengths==1,L[lengths][...,0]*target_q,target_v)
    target_v = torch.where(lengths>0,target_v,torch.zeros_like(target_v))
    if return_endpoints:
        return corrected,target_q,target_v
    return corrected


@torch.no_grad()
def sample_state_drift(elapsed_s, like, cfg: GaussianBridgeNoise, *, generator=None):
    """Jointly Gaussian (position, velocity) drift after advancing a plan.

    A calibrated covariance table interpolates the measured error law, including
    position/velocity correlation. Otherwise the covariance follows integrated
    Brownian velocity plus initial position/velocity uncertainty.
    """
    s = torch.as_tensor(elapsed_s,device=like.device,dtype=like.dtype).reshape(-1,1)
    if bool((s < 0).any()):
        raise ValueError('elapsed time must be nonnegative')
    randn = lambda: torch.randn(like.shape,device=like.device,dtype=like.dtype,generator=generator)
    if cfg.state_times_s:
        times = torch.as_tensor(cfg.state_times_s,device=like.device,dtype=like.dtype)
        cov = torch.as_tensor(cfg.state_covariance,device=like.device,dtype=like.dtype)
        capped = s[:,0].clamp_max(times[-1])
        hi = torch.searchsorted(times,capped).clamp(1,len(times)-1)
        lo = hi-1
        u = ((capped-times[lo])/(times[hi]-times[lo]))[:,None,None]
        values = (1-u)*cov[lo]+u*cov[hi]
        qvar,vvar,qvcov = values.unbind(-1)
        zq,zv = randn(),randn()
        qstd = qvar.sqrt()
        shared = qvcov/qstd.clamp_min(1e-12)
        q = qstd*zq
        v = shared*zq+(vvar-shared.square()).clamp_min(0).sqrt()*zv
        return cfg.scale*q,cfg.scale*v
    qbase,vbase = cfg.state_position_floor*randn(),cfg.state_velocity_floor*randn()
    z1,z2 = randn(),randn()
    sigma = _joint_scale(cfg.state_velocity_diffusion,like)
    q = qbase+s*vbase+sigma*s**1.5*(.5*z1+z2/math.sqrt(12))
    v = vbase+sigma*s.sqrt()*z1
    return cfg.scale*q,cfg.scale*v
