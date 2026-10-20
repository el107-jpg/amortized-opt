"""Initialization ablations retain states/teachers and match their probe inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from amortized_opt.constants import N_DOF, ready_joint_pos
from amortized_opt.phase2_data import Phase2Noise
from amortized_opt.train_cost import Phase2Config, _apply_terminal_initializer
from scripts.train_phase2 import parse_args
from test_mpc_init_noise import make_pool


def test_rebuild_all_rows_keeps_states_provenance_and_teacher():
    line = torch.full((2, 8, N_DOF), .2)
    core = SimpleNamespace(terminal_initializer=object(), initial_trajectory=lambda *a: line)
    batch = dict(init=torch.ones_like(line), seed=torch.ones_like(line),
        state=torch.randn(2, 2*N_DOF), context=object(), masks=object(),
        mpc_init=torch.tensor([False, True]))
    before = {k:batch[k].clone() for k in ['seed','state','mpc_init']}
    assert _apply_terminal_initializer(core, batch, all_rows=True)
    assert torch.equal(batch['init'],line)
    for k,v in before.items():
        assert torch.equal(batch[k],v)


def test_mpc_noise_override_leaves_fresh_sampler_unchanged():
    pool = make_pool(Phase2Noise())
    broad = Phase2Noise(tau_steps=20,magnitude_distribution='half_normal',sigma_scale=.3,sigma_white=.003)
    init = torch.tensor(ready_joint_pos(),dtype=torch.float32)[None,None].expand(16,50,-1).clone()
    gen = torch.Generator().manual_seed(5)
    first = pool.noised_init(init,50,generator=gen)
    pool.noised_init(init,50,generator=gen,noise=broad)
    second = pool.noised_init(init,50,generator=torch.Generator().manual_seed(5))
    assert torch.equal(first,second)
    assert pool.noise == Phase2Noise()


def test_saved_recipe_overrides_and_noise_mode_validation(tmp_path):
    recipe = tmp_path/'args.json'
    recipe.write_text(json.dumps(dict(steps=60000,lr=.0001,timing_critic_mode='control',
        terminal_init_all_rows=True,terminal_init_noise=False,mpc_init_noise=False)))
    args = parse_args(['--args-json',str(recipe),'--device','cuda:2'])
    assert args.steps == 60000 and args.device == 'cuda:2'
    assert args.terminal_init_all_rows and not args.mpc_init_noise
    with pytest.raises(ValueError,match='requires both input noise flags off'):
        Phase2Config(terminal_init_all_rows=True,mpc_init_noise=True)


def test_probe_rebuilds_from_each_measured_state(monkeypatch):
    import amortized_opt.probe as probe
    from amortized_opt.tokenizer import build_waypoint_masks
    init_seen, state_seen = [], []
    class Core:
        terminal_initializer = object()
        def encode_context(self,ctx):
            return SimpleNamespace(waist_pr_sp=None)
        def initial_trajectory(self,ctx,state,masks):
            state_seen.append(state.clone())
            return state[:,:N_DOF,None].transpose(1,2).expand_as(masks.elem_mask).clone()
        def refine(self,init,state,enc,masks,iterations):
            init_seen.append(init.clone())
            return SimpleNamespace(traj=init)
    monkeypatch.setattr(probe,'build_context_inputs',lambda *a,**kw:object())
    monkeypatch.setattr(probe,'correct_quintic_init_arms',lambda *a,**kw:pytest.fail('DLS changed clean initialization'))
    masks = build_waypoint_masks(torch.tensor([.16]),torch.tensor([.16]),max_horizon=8,dt=.02)
    plan,_ = probe.mpc_model_planner(Core(),num_drums=8,num_future=1,device='cpu',fresh_line_init=True)
    for i in range(2):
        q = np.asarray(ready_joint_pos()) + .01*i
        req = SimpleNamespace(kit=object(),snapshot=SimpleNamespace(goal={h:SimpleNamespace(active=False) for h in ['left','right']}),
            masks=masks,init_traj=np.full((8,N_DOF),9.),q=q,qd=np.zeros(N_DOF),
            hit_index={'left':8-i,'right':8-i},dt=.02)
        plan(req)
        torch.testing.assert_close(init_seen[-1],state_seen[-1][:,:N_DOF,None].transpose(1,2).expand_as(init_seen[-1]))
    assert len(state_seen)==2


def test_bridge_state_augmentation_is_separate_and_validated():
    args = parse_args(['--mpc-noise-bridge-config','bridge.json'])
    assert args.mpc_state_noise_scale == 0
    args = parse_args(['--mpc-noise-bridge-config','bridge.json','--mpc-state-noise-scale','.25'])
    assert args.mpc_state_noise_scale == .25
    for options in [ ['--mpc-state-noise-scale','.25'],
                     ['--mpc-state-noise-scale','-1'],
                     ['--mpc-state-noise-scale','nan'] ]:
        with pytest.raises(SystemExit):
            parse_args(options)
