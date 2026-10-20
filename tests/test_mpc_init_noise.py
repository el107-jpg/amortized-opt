"""Training input corruption: statistical law, clean references, and replay."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from amortized_opt.constants import N_DOF, ready_joint_pos
from amortized_opt.objective_replay import ObjectiveReplayContext, restore_mutable_batch_fields
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool, Phase2ScenarioConfig
from amortized_opt.train_cost import (
    Phase2Config, _apply_mpc_init_noise, _refresh_timing_line_search_replay,
    phase2_config_from_checkpoint,
)
from scripts.train_phase2 import parse_args


def make_pool(noise, H=50):
    return Phase2Pool(
        {H: {"q0": np.zeros((1, N_DOF), dtype=np.float32)}},
        cfg=Phase2ScenarioConfig(horizon=H, buckets=(H,)), noise=noise,
    )


def test_long_ou_has_requested_correlation_and_half_normal_magnitude():
    pool = make_pool(Phase2Noise(tau_steps=20, sigma_white=0,
        magnitude_distribution="half_normal", sigma_scale=.30, clip_to_box=False))
    noise = pool.noised_init(torch.zeros(4096, 50, N_DOF), 50,
                             generator=torch.Generator().manual_seed(17))
    # Statistical checks catch an ignored tau, log-uniform amplitude, or
    # accidentally drawing one magnitude for the entire minibatch.
    a, b = noise[:, :-1].flatten(), noise[:, 1:].flatten()
    correlation = ((a-a.mean())*(b-b.mean())).mean() / (a.std()*b.std())
    assert abs(float(correlation) - np.exp(-1/20)) < .01
    assert abs(float(noise.std()) - .30) < .02
    rms = noise.square().mean((1,2)).sqrt()
    assert .16 < float(rms.median()) < .25
    assert float((rms > .4).float().mean()) > .12
    assert torch.isfinite(noise).all()


def test_mpc_corruption_preserves_fresh_rows_targets_and_measured_state():
    init = torch.zeros(2, 8, N_DOF)
    mask = torch.ones_like(init)
    mask[1, 6:] = 0
    batch = dict(init=init.clone(), seed=init.clone(), state=torch.randn(2, 2*N_DOF),
                 q0=torch.randn(2, N_DOF), v0=torch.randn(2, N_DOF),
                 mpc_init=torch.tensor([False, True]), masks=SimpleNamespace(elem_mask=mask))
    unchanged = {key:batch[key].clone() for key in ['seed','state','q0','v0']}
    pool = SimpleNamespace(noised_init=lambda init, H, **kw: init+1,
                           noise=Phase2Noise(clip_to_box=False))
    _apply_mpc_init_noise(batch, pool=pool, enabled=True)
    assert torch.equal(batch['init'][0],init[0])
    torch.testing.assert_close(batch['init'][1,0],torch.full((N_DOF,),.104))
    assert torch.equal(batch['init'][1,4],torch.ones(N_DOF))
    assert not batch['init'][1,6:].any()
    for key,value in unchanged.items():
        assert torch.equal(batch[key],value)


def test_mpc_noise_replays_identically_without_accumulating_or_advancing_rng():
    pool = make_pool(Phase2Noise(tau_steps=20, sigma_white=.003,
        magnitude_distribution='half_normal',sigma_scale=.30))
    init = torch.tensor(ready_joint_pos(),dtype=torch.float32)[None,None].expand(3,50,-1).clone()
    batch = dict(init=init.clone(),seed=init.clone(),state=torch.zeros(3,2*N_DOF),
        masks=SimpleNamespace(elem_mask=torch.ones_like(init)),mpc_init=torch.tensor([False,True,True]))
    gen = torch.Generator().manual_seed(32)
    mgr = ObjectiveReplayContext({'gen':gen},capture_global=False)
    rec = mgr.begin_record(primary_batch=batch)
    _refresh_timing_line_search_replay(rec,batch)
    _apply_mpc_init_noise(batch,pool=pool,generator=gen,enabled=True)
    expected = batch['init'].clone()
    mgr.finalize_baseline(rec)
    post = gen.get_state().clone()
    for _ in range(3):
        with mgr.run_isolated(rec,restore_pre=True):
            restore_mutable_batch_fields(batch,rec.mutable_pre)
            _apply_mpc_init_noise(batch,pool=pool,generator=gen,enabled=True)
            assert torch.equal(batch['init'],expected)
            assert torch.equal(batch['seed'],init)
        assert torch.equal(gen.get_state(),post)
    assert (expected >= pool.q_lo).all() and (expected <= pool.q_hi).all()


def test_noise_config_checkpoint_roundtrip_and_old_checkpoint_defaults():
    noise = Phase2Noise(tau_steps=20,magnitude_distribution='half_normal',sigma_white=.003)
    cfg = Phase2Config(init_noise=noise,mpc_init_noise=True,terminal_init_noise=True)
    restored = phase2_config_from_checkpoint({'train_cfg':asdict(cfg)})
    assert restored.init_noise == noise
    assert restored.mpc_init_noise and restored.terminal_init_noise
    legacy = phase2_config_from_checkpoint({'train_cfg':{}})
    assert legacy.init_noise.tau_steps == 5
    assert legacy.init_noise.magnitude_distribution == 'log_uniform'
    assert not legacy.mpc_init_noise


def test_training_cli_defaults_and_explicit_legacy_noise():
    args = parse_args([])
    assert args.terminal_init_noise and args.mpc_init_noise
    assert args.init_noise_tau_steps == 20
    assert args.init_noise_magnitude_distribution == 'half_normal'
    assert args.init_noise_sigma_scale == .30 and args.init_noise_sigma_white == .003
    off = parse_args(['--no-mpc-init-noise','--no-terminal-init-noise',
        '--init-noise-tau-steps','5','--init-noise-magnitude-distribution','log_uniform',
        '--init-noise-sigma-white','.01'])
    assert not off.mpc_init_noise and not off.terminal_init_noise
    assert off.init_noise_tau_steps == 5 and off.init_noise_sigma_white == .01


@pytest.mark.parametrize('kwargs',[
    {'tau_steps':0},{'tau_steps':float('nan')},{'sigma_scale':-1},
    {'sigma_white':-1},{'magnitude_distribution':'unknown'},
])
def test_invalid_noise_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        Phase2Noise(**kwargs)
