"""Gaussian path and state noise: covariance, physical boundaries and replay."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.gaussian_bridge import GaussianBridgeNoise, sample_bridge, sample_state_drift
from amortized_opt.phase2_data import Phase2Noise
from amortized_opt.train_cost import Phase2Config, _apply_mpc_init_noise, phase2_config_from_checkpoint
from amortized_opt.tspline import TrajectorySplineBuilder
from test_mpc_init_noise import make_pool


def test_bridge_matches_actual_spline_endpoints_on_different_horizons():
    B,H=6,20
    mask=torch.zeros(B,H,N_DOF)
    lengths=[0,1,2,5,13,20]
    for i,k in enumerate(lengths):mask[i,:k]=1
    cfg=GaussianBridgeNoise(velocity_diffusion=(2.,),terminal_position_std=(.03,),terminal_velocity_std=(.4,))
    q,target_q,target_v=sample_bridge(mask,.02,cfg,generator=torch.Generator().manual_seed(4),return_endpoints=True)
    builder=TrajectorySplineBuilder(max_segments=H,dt=.02)
    sp=builder.from_masks(torch.zeros(B,N_DOF),q,torch.zeros(B,N_DOF),mask)
    assert torch.equal(sp.m[:,0],torch.zeros(B,N_DOF))
    for i,k in enumerate(lengths):
        if k==0:assert not q[i].any();continue
        torch.testing.assert_close(q[i,k-1],target_q[i],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(sp.m[i,k],target_v[i],atol=1e-4,rtol=1e-4)
        assert not q[i,k:].any()
    assert target_q[3:].abs().max()>.02


def test_conditioned_noise_has_gaussian_marginals_and_middle_variance():
    cfg=GaussianBridgeNoise(velocity_diffusion=(2.,),terminal_position_std=(0.,),terminal_velocity_std=(0.,))
    q=sample_bridge(torch.ones(4096,50,N_DOF),.02,cfg,generator=torch.Generator().manual_seed(7))
    x=q[:,24].flatten();x=(x-x.mean())/x.std()
    assert abs(float(x.pow(3).mean()))<.08
    assert abs(float(x.pow(4).mean())-3)<.12
    assert q[:,-1].abs().max()<1e-5
    assert q[:,24].std()>q[:,0].std()*10


def test_integrated_brownian_state_covariance_and_time_growth():
    cfg=GaussianBridgeNoise(state_velocity_diffusion=(.7,),state_position_floor=0,state_velocity_floor=0)
    like=torch.zeros(20000,N_DOF)
    for s in [.10,.50]:
        q,v=sample_state_drift(torch.full((len(like),),s),like,cfg,generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(q.square().mean(),torch.tensor(.7**2*s**3/3),rtol=.02,atol=1e-7)
        torch.testing.assert_close(v.square().mean(),torch.tensor(.7**2*s),rtol=.02,atol=1e-7)
        torch.testing.assert_close((q*v).mean(),torch.tensor(.7**2*s**2/2),rtol=.02,atol=1e-7)


def test_mpc_bridge_is_applied_once_without_changing_references():
    bridge=GaussianBridgeNoise(velocity_diffusion=(1.,),terminal_position_std=(.03,),terminal_velocity_std=(.10,))
    noise=Phase2Noise(process='gaussian_bridge',bridge=bridge,clip_to_box=False)
    pool=make_pool(Phase2Noise(),H=8)
    init=torch.zeros(2,8,N_DOF)
    from types import SimpleNamespace
    mask=torch.ones_like(init);mask[1,5:]=0
    batch=dict(init=init.clone(),seed=init.clone(),mpc_init=torch.tensor([False,True]),masks=SimpleNamespace(elem_mask=mask))
    gen=torch.Generator().manual_seed(12);before=gen.get_state().clone()
    expected=sample_bridge(mask,.02,bridge,generator=torch.Generator().set_state(before))
    _apply_mpc_init_noise(batch,pool=pool,generator=gen,enabled=True,noise=noise)
    assert torch.equal(batch['init'][0],init[0])
    torch.testing.assert_close(batch['init'][1],expected[1])
    assert torch.equal(batch['seed'],init)
    cfg=Phase2Config(mpc_noise=noise,mpc_init_noise=True,timing_mpc_state_noise=bridge)
    restored=phase2_config_from_checkpoint({'train_cfg':asdict(cfg)})
    assert restored.mpc_noise.bridge==bridge and restored.timing_mpc_state_noise==bridge


@pytest.mark.parametrize('kwargs',[{'velocity_diffusion':(-1,)},{'terminal_position_std':(float('nan'),)}, {'scale':-1}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):GaussianBridgeNoise(**kwargs)


def test_calibrated_state_covariance_interpolates_without_inventing_velocity_growth():
    table=(tuple((.01,.25,.025) for _ in range(N_DOF)),tuple((.09,.25,.075) for _ in range(N_DOF)))
    cfg=GaussianBridgeNoise(state_times_s=(0.,1.),state_covariance=table)
    like=torch.zeros(20000,N_DOF)
    q,v=sample_state_drift(torch.full((len(like),),.5),like,cfg,generator=torch.Generator().manual_seed(22))
    torch.testing.assert_close(q.square().mean(),torch.tensor(.05),rtol=.02,atol=1e-5)
    torch.testing.assert_close(v.square().mean(),torch.tensor(.25),rtol=.02,atol=1e-5)
    torch.testing.assert_close((q*v).mean(),torch.tensor(.05),rtol=.03,atol=1e-5)
