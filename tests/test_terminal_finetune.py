from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from amortized_opt.model import AmortizedOptimizer, ModelConfig, model_config_from_checkpoint
from amortized_opt.terminal_finetune import TerminalRoleLoss, cap_terminal_gradient
from amortized_opt.terminal_pose import TerminalPoseConfig, TerminalPoseAL
from amortized_opt.train_cost import _apply_terminal_initializer
from tests.test_terminal_initializer import _batch


def small_model(trainable):
    return AmortizedOptimizer(ModelConfig(
        max_horizon=40, num_drums=7, terminal_head=True,
        terminal_head_trainable=trainable, terminal_hihat_library_fallback=False,
        d_ctx=32, ctx_heads=4, d_model=32, dec_heads=4, dec_layers=1))


def test_decoder_loss_reaches_terminal_head_through_actual_seed_builder():
    torch.set_num_threads(2)
    model = small_model(True)
    ctx, state, masks = _batch()
    batch = dict(context=ctx, state=state, masks=masks,
                 init=torch.zeros(3, 40, 17), mpc_init=torch.ones(3, dtype=torch.bool))
    assert _apply_terminal_initializer(model, batch, all_rows=True)
    assert batch["init"].requires_grad
    out = model(batch["init"], state, model.encode_context(ctx), masks)
    out.traj.square().mean().backward()
    norm = sum(float(p.grad.square().sum()) for p in model.terminal_initializer.head.parameters()
               if p.grad is not None)
    assert np.isfinite(norm) and norm > 0
    assert any(p.grad is not None for p in model.decoder.parameters())


def test_legacy_freeze_and_checkpoint_trainability_are_preserved():
    model = small_model(False)
    ctx, state, masks = _batch()
    state.requires_grad_(True)
    assert not model.initial_trajectory(ctx, state, masks).requires_grad
    assert not any(p.requires_grad for p in model.terminal_initializer.head.parameters())
    cfg = model_config_from_checkpoint(dict(model_cfg=asdict(small_model(True).cfg)))
    restored = AmortizedOptimizer(cfg)
    assert all(p.requires_grad for p in restored.terminal_initializer.head.parameters())
    with torch.no_grad():
        assert not restored.initial_trajectory(ctx, state, masks).requires_grad


def test_original_role_objective_has_independent_nonzero_head_gradients(tmp_path):
    model = small_model(True)
    path = tmp_path / "original.pt"
    al = TerminalPoseAL()
    original = dict(config=asdict(TerminalPoseConfig()), al=al.state_dict(),
                    train_config=dict(start_bank_size=0, seed=0))
    torch.save(original, path)
    role = TerminalRoleLoss(path, "cpu")
    batch = role.draw(32, np.random.default_rng(33))
    result = role.evaluate(model, batch)
    assert set(result.terms) == {"joint", "start", "motion", "com", "elev_drum",
                                  "elev_floor", "elev_cymbal", "shaft", "waist", "hit", "collision"}
    result.total.mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.terminal_initializer.head.parameters()
               if p.grad is not None) > 0
    assert all(p.grad is None for p in model.decoder.parameters())
    with torch.no_grad():
        first = role.evaluate(model, batch).total
        second = role.evaluate(model, batch).total
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert role.weights == al.cost_weights()


@pytest.mark.parametrize("rest", [4., 0.])
def test_gradient_cap_bounds_head_fraction_without_changing_rest(rest):
    model = nn.Module()
    model.terminal_initializer = nn.Module()
    model.terminal_initializer.head = nn.Linear(1, 1, bias=False)
    model.decoder = nn.Linear(1, 1, bias=False)
    model.terminal_initializer.head.weight.grad = torch.tensor([[3.]])
    model.decoder.weight.grad = torch.tensor([[rest]])
    original = model.decoder.weight.grad.clone()
    stats = cap_terminal_gradient(model, .3)
    torch.testing.assert_close(model.decoder.weight.grad, original, rtol=0, atol=0)
    assert stats["terminal_head_grad_fraction_after"] <= .300001
    assert stats["terminal_head_grad_norm_after"] < stats["terminal_head_grad_norm_before"]


def test_phase2_joint_training_arguments():
    from scripts.train_phase2 import parse_args
    args = parse_args(["--train-terminal-head", "--terminal-role-checkpoint", "original.pt",
                       "--terminal-head-grad-fraction-cap", ".3"])
    assert args.train_terminal_head and args.w_terminal_role == 1.


def test_posture_override_changes_only_original_terminal_com_term(tmp_path):
    model = small_model(True)
    path = tmp_path / "original.pt"
    original = dict(config=asdict(TerminalPoseConfig()), al=TerminalPoseAL().state_dict(),
                    train_config=dict(start_bank_size=0, seed=0))
    torch.save(original, path)
    before = TerminalRoleLoss(path, "cpu")
    after = TerminalRoleLoss(path, "cpu", w_com=2.)
    batch = before.draw(16, np.random.default_rng(91))
    a, b = before.evaluate(model, batch), after.evaluate(model, batch)
    torch.testing.assert_close(b.terms["com"], 2*a.terms["com"], rtol=0, atol=0)
    for key in a.terms:
        if key != "com":
            torch.testing.assert_close(a.terms[key], b.terms[key], rtol=0, atol=0)
    assert before.weights == after.weights
    assert after.metadata["original_config"]["w_com"] == 1.
    assert after.metadata["overrides"] == {"w_com": 2.}
    from scripts.train_phase2 import parse_args
    args = parse_args(["--w-elbow-down", "20", "--w-arm-compact", "30", "--terminal-role-w-com", "2"])
    assert (args.w_elbow_down, args.w_arm_compact, args.terminal_role_w_com) == (20., 30., 2.)


def test_v4_role_upgrade_resets_incompatible_legacy_contact_duals(tmp_path):
    path=tmp_path/'legacy_role.pt'
    old=asdict(TerminalPoseConfig())
    al=TerminalPoseAL().state_dict();al['lam']['hit']=10.;al['lam']['collision']=20.
    torch.save(dict(config=old,al=al,train_config=dict(start_bank_size=0,seed=0)),path)
    legacy=TerminalRoleLoss(path,'cpu')
    assert legacy.cost.cfg.contact_scoring_mode=='legacy'
    assert legacy.weights['hit']==10.
    upgraded=TerminalRoleLoss(path,'cpu',contact_scoring_mode='isaac_v4')
    assert upgraded.cost.cfg.contact_scoring_mode=='isaac_v4'
    assert upgraded.metadata['contact_al_reset_on_upgrade']
    assert upgraded.weights['hit']==1. and upgraded.weights['collision']==1.
    upgraded.load_state_dict(dict(config=old,al=al))
    assert upgraded.weights['hit']==1.  # A legacy resume cannot restore stale duals.
    model=small_model(True)
    batch=upgraded.draw(8,np.random.default_rng(31))
    result=upgraded.evaluate(model,batch)
    assert torch.isfinite(result.total).all()
    result.total.mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.terminal_initializer.head.parameters())


@pytest.mark.parametrize("head_size,rest_size", [(2, 200), (20, 2000)])
def test_parameter_rms_cap_is_invariant_to_network_size(head_size, rest_size):
    model = nn.Module()
    model.terminal_initializer = nn.Module()
    model.terminal_initializer.head = nn.Linear(head_size, 1, bias=False)
    model.decoder = nn.Linear(rest_size, 1, bias=False)
    model.terminal_initializer.head.weight.grad = torch.full((1, head_size), 3.)
    model.decoder.weight.grad = torch.full((1, rest_size), 4.)
    original_rest = model.decoder.weight.grad.clone()
    stats = cap_terminal_gradient(model, .3, mode="parameter_rms")
    torch.testing.assert_close(model.terminal_initializer.head.weight.grad,
                               torch.full((1, head_size), 1.2), atol=1e-6, rtol=0)
    torch.testing.assert_close(model.decoder.weight.grad, original_rest, atol=0, rtol=0)
    assert stats["terminal_head_grad_rms_ratio_after"] == pytest.approx(.3)
    assert stats["terminal_head_parameter_count"] == head_size
    assert stats["terminal_rest_parameter_count"] == rest_size


def test_zero_gradients_count_toward_rms_parameter_denominator():
    model = nn.Module()
    model.terminal_initializer = nn.Module()
    model.terminal_initializer.head = nn.Linear(2, 1, bias=False)
    model.decoder = nn.Linear(200, 1, bias=False)
    model.unused = nn.Parameter(torch.zeros(200))
    model.frozen = nn.Parameter(torch.zeros(10000), requires_grad=False)
    model.terminal_initializer.head.weight.grad = torch.full((1, 2), 3.)
    model.decoder.weight.grad = torch.full((1, 200), 4.)
    stats = cap_terminal_gradient(model, .3, mode="parameter_rms")
    assert stats["terminal_rest_parameter_count"] == 400
    assert stats["terminal_head_grad_rms_after"] == pytest.approx(.3*4/np.sqrt(2))


def test_dynamic_terminal_al_updates_only_on_commit_and_replays_frozen_prices(tmp_path):
    model = small_model(True)
    path = tmp_path / "original.pt"
    saved_al = TerminalPoseAL(lam_init=100.)
    torch.save(dict(config=asdict(TerminalPoseConfig()), al=saved_al.state_dict(),
                    train_config=dict(start_bank_size=0, seed=0)), path)
    role = TerminalRoleLoss(path, "cpu", dynamic_al=True, lam_max=1000.)
    batch = role.draw(32, np.random.default_rng(171))
    prices = dict(role.weights)
    initial_state = role.al.state_dict()
    baseline = role.evaluate(model, batch, weights=prices).total.detach()
    # Forward/probe evaluation never advances the dual or its EMA.
    role.evaluate(model, batch, weights=prices)
    assert role.al.state_dict() == initial_state
    violations = dict(hit=torch.full((32,), .01), collision=torch.full((32,), .02))
    assert not role.commit(violations, executed=False)
    assert role.al.state_dict() == initial_state
    assert role.commit(violations, executed=True)
    assert role.committed_al_updates == 1
    assert role.al.steps == saved_al.steps+1
    assert role.al.lam["hit"] > 100 and role.al.lam["collision"] > 100
    # Live dual prices changed, but replay of an old objective remains exact.
    replay = role.evaluate(model, batch, weights=prices).total.detach()
    torch.testing.assert_close(baseline, replay, rtol=0, atol=0)
    restored = TerminalRoleLoss(path, "cpu", dynamic_al=True, lam_max=1000.)
    restored.load_state_dict(role.state_dict())
    assert restored.al.state_dict() == role.al.state_dict()
    assert restored.committed_al_updates == 1
    assert restored.al.lam_max == {"hit": 1000., "collision": 1000.}


def test_fixed_terminal_al_does_not_update_even_after_a_committed_step(tmp_path):
    path = tmp_path / "original.pt"
    torch.save(dict(config=asdict(TerminalPoseConfig()), al=TerminalPoseAL().state_dict(),
                    train_config=dict(start_bank_size=0, seed=0)), path)
    role = TerminalRoleLoss(path, "cpu")
    before = role.al.state_dict()
    assert not role.commit(dict(hit=.1, collision=.1), executed=True)
    assert role.al.state_dict() == before
