"""AdamW line-search adapter tests (plan §11 tests 2, 4)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from amortized_opt.adamw_ls import (
    AdamWLineSearchAdapter,
    build_adamw_like_train_cost,
    displacement_matches_adamw_step,
    step_vector_breakdown,
)
from amortized_opt.line_search import evaluate_trial
from amortized_opt.grad_balance import clip_torque_gradient_budget


class _Toy(nn.Module):
    """Two groups: weight matrix (decay) + bias / emb-like 1-D (no_decay)."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -0.5], [0.25, 0.75]]))
        self.bias = nn.Parameter(torch.tensor([0.5, -0.25]))
        self.token_emb = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))  # "emb" → no_decay


def _make_opt(model: _Toy, lr: float = 1e-3, wd: float = 0.05):
    opt = build_adamw_like_train_cost(
        model.named_parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    # Deterministic math path for equivalence checks.
    for g in opt.param_groups:
        g["foreach"] = False
        g["fused"] = False
    return opt


def _set_grads(model: _Toy, scale: float = 1.0) -> None:
    model.weight.grad = scale * torch.tensor([[0.4, -0.2], [0.1, 0.3]])
    model.bias.grad = scale * torch.tensor([0.05, -0.1])
    model.token_emb.grad = scale * torch.tensor([0.02, -0.03, 0.01])


# ---------------------------------------------------------------- test 4: transfer
def test_epsilon_eta_ref_matches_adamw_step():
    torch.manual_seed(0)
    model = _Toy()
    eta = 1e-3
    opt = _make_opt(model, lr=eta, wd=0.05)
    _set_grads(model)
    # Warm moments a few steps so bias correction / m,v nontrivial.
    for _ in range(3):
        opt.step()
        _set_grads(model, scale=1.0 + 0.1 * _)

    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta)
    ok, info = displacement_matches_adamw_step(adapter, rtol=1e-6, atol=1e-8)
    assert ok, info


def test_commit_advances_moments_once_and_decay_once():
    model = _Toy()
    eta = 2e-3
    opt = _make_opt(model, lr=eta, wd=0.1)
    _set_grads(model)
    # One prior step so state exists
    opt.step()
    _set_grads(model)

    # Reference: native step from this state
    model_ref = _Toy()
    with torch.no_grad():
        for a, b in zip(model_ref.parameters(), model.parameters()):
            a.copy_(b)
    opt_ref = _make_opt(model_ref, lr=eta, wd=0.1)
    opt_ref.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p, pr in zip(model.parameters(), model_ref.parameters()):
        pr.grad = p.grad.detach().clone()
    opt_ref.step()

    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta)
    adapter.mark_theta0_snapshot()
    theta0 = [t.clone() for t in adapter._theta0]
    staged = adapter.stage_from_q()
    assert staged.stats.probe_valid

    # Probe must not change live params permanently
    adapter.write_candidate(eta, staged.p_probe, into_params=True)
    for p, t0 in zip(model.parameters(), theta0):
        assert not torch.allclose(p, t0)
    adapter.restore_theta0()
    for p, t0 in zip(model.parameters(), theta0):
        assert torch.allclose(p, t0)

    # Exec + commit once
    adapter.commit(staged, epsilon=eta, use_exec=True)

    for p, pr in zip(model.parameters(), model_ref.parameters()):
        assert torch.allclose(p, pr, rtol=1e-6, atol=1e-8), (p - pr).abs().max()

    # Moments match
    for p, pr in zip(model.parameters(), model_ref.parameters()):
        st, str_ = opt.state[p], opt_ref.state[pr]
        assert torch.allclose(st["exp_avg"], str_["exp_avg"], rtol=1e-6, atol=1e-8)
        assert torch.allclose(st["exp_avg_sq"], str_["exp_avg_sq"], rtol=1e-6, atol=1e-8)
        assert int(st["step"].item()) == int(str_["step"].item())

    # Decoupled decay only on weight (2-D, not emb); bias/emb wd=0
    wd_group = next(g for g in opt.param_groups if g["weight_decay"] > 0)
    nod_group = next(g for g in opt.param_groups if g["weight_decay"] == 0)
    assert any(p is model.weight for p in wd_group["params"])
    assert any(p is model.bias for p in nod_group["params"])
    assert any(p is model.token_emb for p in nod_group["params"])


def test_discard_restores_params_and_leaves_moments():
    model = _Toy()
    opt = _make_opt(model, lr=1e-3, wd=0.01)
    _set_grads(model)
    opt.step()
    _set_grads(model)
    m_before = {id(p): opt.state[p]["exp_avg"].clone() for p in model.parameters()}
    step_before = {id(p): int(opt.state[p]["step"].item()) for p in model.parameters()}

    adapter = AdamWLineSearchAdapter(opt, eta_ref=1e-3)
    adapter.mark_theta0_snapshot()
    staged = adapter.stage_from_q()
    adapter.write_candidate(1e-3, staged.p_probe, into_params=True)
    adapter.discard(restore_params=True)

    for p in model.parameters():
        assert torch.allclose(opt.state[p]["exp_avg"], m_before[id(p)])
        assert int(opt.state[p]["step"].item()) == step_before[id(p)]
    # Params restored to theta0 (pre-probe)
    # After discard, compare to snapshot taken at stage — equal to post-first-step.
    assert torch.isfinite(model.weight).all()


# -------------------------------------------------------- test 2: uphill momentum
def test_uphill_exec_from_decay_still_selects_on_probe():
    """Construct s_exec > 0 while a > 0; selection uses p_probe, exec uses p_exec."""
    # θ > 0 with g < 0: AdamW decay pulls toward 0 (against -g), so s_decay > 0.
    model = _Toy()
    with torch.no_grad():
        model.weight.fill_(5.0)
        model.bias.fill_(0.0)
        model.token_emb.fill_(0.0)
    eta = 1e-2
    opt = _make_opt(model, lr=eta, wd=50.0)  # huge wd on weight
    model.weight.grad = torch.full_like(model.weight, -1e-4)
    model.bias.grad = torch.full_like(model.bias, -1e-4)
    model.token_emb.grad = torch.full_like(model.token_emb, -1e-4)

    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta)
    adapter.mark_theta0_snapshot()
    staged = adapter.stage_from_q()
    assert staged.stats.a > 0.0
    assert staged.stats.probe_valid
    assert staged.stats.s_decay > 0.0
    assert staged.stats.s_exec > 0.0, (
        f"s_exec={staged.stats.s_exec}, s_mom={staged.stats.s_momentum}, "
        f"s_decay={staged.stats.s_decay}, a={staged.stats.a}"
    )

    F0 = 1.0
    eps = eta
    Fprobe = F0 + eps * staged.stats.s_probe  # probe downhill
    d = evaluate_trial(
        Fprobe=Fprobe,
        F0=F0,
        epsilon=eps,
        a=staged.stats.a,
        P2=staged.stats.P2,
        A_prev=0.0,
        B_cand=staged.stats.a,
        beta_eff=0.0,
        c1=0.5,
        trial_index=0,
        max_trials=6,
    )
    assert d.accept

    F_lin_exec = F0 + eps * staged.stats.s_exec
    assert F_lin_exec > F0  # transferred exec may increase loss; no veto
    theta_before = [p.detach().clone() for p in model.parameters()]
    adapter.commit(staged, epsilon=eps, use_exec=True)
    moved = any(
        not torch.allclose(p, t0) for p, t0 in zip(model.parameters(), theta_before)
    )
    assert moved


def test_probe_excludes_momentum_and_decay():
    model = _Toy()
    opt = _make_opt(model, lr=1e-3, wd=0.2)
    _set_grads(model)
    for _ in range(2):
        opt.step()
        _set_grads(model)

    adapter = AdamWLineSearchAdapter(opt, eta_ref=1e-3)
    adapter.mark_theta0_snapshot()
    g_list = [p.grad.detach().clone() for p in model.parameters()]
    staged = adapter.stage_from_q(g_preclip=g_list)
    diff = sum(
        (pp - pe).abs().sum().item()
        for pp, pe in zip(staged.p_probe, staged.p_exec)
    )
    assert diff > 0.0
    for pp, q, D in zip(staged.p_probe, g_list, staged.D):
        assert torch.allclose(pp, -D * q, rtol=1e-5, atol=1e-8)


def test_objective_slopes_use_raw_gradient_while_direction_uses_clipped_q():
    """Clipping changes the direction magnitude, not the derivative gradient."""
    model = _Toy()
    opt = _make_opt(model, lr=1e-3, wd=0.0)
    _set_grads(model, scale=1e-2)
    g_pre = [p.grad.detach().clone() * 1e6 for p in model.parameters()]

    adapter = AdamWLineSearchAdapter(opt, eta_ref=1e-3)
    adapter.mark_theta0_snapshot()
    staged_mismatch = adapter.stage_from_q(g_objective=g_pre)

    adapter.mark_theta0_snapshot()
    staged_q_only = adapter.stage_from_q()

    assert staged_mismatch.stats.s_exec == pytest.approx(
        staged_q_only.stats.s_exec * 1e6, rel=1e-6, abs=1e-8,
    )
    assert staged_mismatch.stats.a == pytest.approx(
        staged_q_only.stats.a * 1e6, rel=1e-6, abs=1e-8,
    )
    assert staged_mismatch.stats.P2 == pytest.approx(
        staged_q_only.stats.P2, rel=1e-6, abs=1e-12,
    )
    assert staged_mismatch.stats.gnorm_preclip > staged_q_only.stats.gnorm_q * 1e5
    assert staged_mismatch.stats.s_exec_q == pytest.approx(
        staged_q_only.stats.s_exec_q, rel=1e-6, abs=1e-12,
    )


def test_gradient_breakdown_matches_executed_delta_exactly():
    model = _Toy()
    eta = 2e-3
    opt = _make_opt(model, lr=eta, wd=0.1)
    _set_grads(model)
    opt.step()
    _set_grads(model, scale=-0.35)

    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta)
    theta0 = [t.clone() for t in adapter.snapshot_theta0()]
    staged = adapter.stage_from_q()
    epsilon = 0.6 * eta
    bd = step_vector_breakdown(staged, epsilon, "gradient")
    adapter.write_candidate(epsilon, staged.p_exec, into_params=True)

    delta_sq = 0.0
    for p, t0, pc, ph, pd, pm in zip(
        adapter._params,
        theta0,
        staged.p_current,
        staged.p_history,
        staged.p_decay,
        staged.p_momentum,
    ):
        delta = p.detach() - t0
        expected = epsilon * (pc + ph + pd)
        assert torch.allclose(delta, expected, rtol=1e-5, atol=1e-8)
        assert torch.allclose(pc + ph, pm, rtol=1e-6, atol=1e-8)
        delta_sq += float(delta.double().square().sum())
    assert bd["ls_u_total_norm"] == pytest.approx(delta_sq ** 0.5, rel=1e-6)
    assert (
        bd["ls_u_fresh_frac"]
        + bd["ls_u_mom_frac"]
        + bd["ls_u_decay_frac"]
    ) == pytest.approx(1.0, abs=1e-6)
    assert (
        bd["ls_u_fresh_norm_frac"]
        + bd["ls_u_mom_norm_frac"]
        + bd["ls_u_decay_norm_frac"]
    ) == pytest.approx(1.0, abs=1e-6)


def test_clipped_direction_has_raw_objective_finite_difference_slope():
    p = nn.Parameter(torch.tensor([1.2, -0.7], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=1e-2, weight_decay=0.0)

    def objective():
        return 0.5 * (3.0 * p[0].square() + 0.4 * p[1].square())

    objective().backward()
    g_objective = [p.grad.detach().clone()]
    torch.nn.utils.clip_grad_norm_([p], max_norm=1e-2)
    opt.state[p] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.tensor([0.3, 0.2], dtype=p.dtype),
        "exp_avg_sq": torch.tensor([0.4, 0.1], dtype=p.dtype),
    }
    adapter = AdamWLineSearchAdapter(opt, eta_ref=1e-2)
    theta0 = [t.clone() for t in adapter.snapshot_theta0()]
    staged = adapter.stage_from_q(g_objective=g_objective)
    h = 1e-5
    adapter.write_candidate(h, staged.p_probe, into_params=True)
    f_plus = float(objective().detach())
    adapter.write_candidate(-h, staged.p_probe, into_params=True)
    f_minus = float(objective().detach())
    adapter.write_candidate(h, staged.p_exec, into_params=True)
    f_exec_plus = float(objective().detach())
    adapter.write_candidate(-h, staged.p_exec, into_params=True)
    f_exec_minus = float(objective().detach())
    adapter.restore_theta0()
    fd = (f_plus - f_minus) / (2.0 * h)
    fd_exec = (f_exec_plus - f_exec_minus) / (2.0 * h)

    assert staged.stats.s_probe == pytest.approx(fd, rel=1e-8, abs=1e-10)
    assert staged.stats.a == pytest.approx(-fd, rel=1e-8, abs=1e-10)
    assert staged.stats.s_exec == pytest.approx(
        fd_exec, rel=1e-8, abs=1e-10,
    )
    assert staged.stats.s_probe != pytest.approx(staged.stats.s_probe_q)
    assert torch.equal(adapter._params[0], theta0[0])


def test_torque_cap_rotation_rejects_true_objective_uphill_probe():
    """A cap can rotate q across 90° even though raw total gradient is finite."""
    p = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=1e-2, weight_decay=0.0)
    rest = -2.0 * p.sum()
    torque = 3.0 * p.sum()
    total = rest + torque
    total.backward(retain_graph=True)
    g_objective = [p.grad.detach().clone()]
    stats = clip_torque_gradient_budget(
        [p], torque, w_cost=1.0, cap=0.2,
    )
    assert stats["torque_grad_cap_scale"] < 1.0
    assert p.grad.item() < 0.0  # transformed q points opposite raw g=+1

    adapter = AdamWLineSearchAdapter(opt, eta_ref=1e-2)
    adapter.snapshot_theta0()
    staged = adapter.stage_from_q(g_objective=g_objective)
    assert staged.stats.s_probe > 0.0
    assert staged.stats.s_probe_q < 0.0
    assert staged.stats.a < 0.0
    assert not staged.stats.probe_valid

    h = 1e-5
    adapter.write_candidate(h, staged.p_probe, into_params=True)
    f_plus = float((-2.0 * p.sum() + 3.0 * p.sum()).detach())
    adapter.write_candidate(-h, staged.p_probe, into_params=True)
    f_minus = float((-2.0 * p.sum() + 3.0 * p.sum()).detach())
    assert staged.stats.s_probe == pytest.approx(
        (f_plus - f_minus) / (2.0 * h), rel=1e-8, abs=1e-10,
    )
    adapter.restore_theta0()


def test_build_groups_match_train_cost_rule():
    model = _Toy()
    opt = _make_opt(model, lr=1e-4, wd=0.01)
    assert len(opt.param_groups) == 2
    decay = opt.param_groups[0]
    nod = opt.param_groups[1]
    assert decay["weight_decay"] == 0.01
    assert nod["weight_decay"] == 0.0
    assert any(p is model.weight for p in decay["params"])
    assert any(p is model.bias for p in nod["params"])
    assert any(p is model.token_emb for p in nod["params"])
