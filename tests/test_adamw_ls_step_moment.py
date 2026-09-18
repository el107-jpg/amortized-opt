"""Step-mode Adam: EMA on executed u = ε p_probe + p_mom + p_decay."""

from __future__ import annotations

import torch

from amortized_opt.adamw_ls import (
    MOMENT_ON_STEP,
    AdamWLineSearchAdapter,
    build_adamw_like_train_cost,
    step_vector_breakdown,
)
from tests.test_adamw_ls import _Toy, _make_opt, _set_grads


def test_step_mode_moment_is_ema_of_executed_u():
    model = _Toy()
    eta = 1e-3
    opt = _make_opt(model, lr=eta, wd=0.0)
    _set_grads(model)
    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta, moment_on=MOMENT_ON_STEP)
    adapter.snapshot_theta0()
    st = adapter.stage_from_q()
    eps = 0.25
    adapter.write_executed_step(eps, st, into_params=True)
    adapter.commit(st, params_already_written=True, epsilon=eps)

    w = model.weight
    u0 = (st.p_probe[0].mul(eps) + st.p_momentum[0] + st.p_decay[0]).detach()
    beta1 = opt.param_groups[0]["betas"][0]
    m = opt.state[w]["exp_avg"]
    assert torch.allclose(m, (1.0 - beta1) * u0, rtol=1e-4, atol=1e-6)


def test_step_vector_breakdown_fractions():
    model = _Toy()
    eta = 1e-2
    opt = _make_opt(model, lr=eta, wd=0.01)
    _set_grads(model, scale=1.0)
    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta, moment_on=MOMENT_ON_STEP)
    adapter.snapshot_theta0()
    st = adapter.stage_from_q()
    bd = step_vector_breakdown(st, eta, MOMENT_ON_STEP)
    assert bd["ls_u_total_norm"] > 0.0
    assert abs(bd["ls_u_fresh_frac"] + bd["ls_u_mom_frac"] + bd["ls_u_decay_frac"] - 1.0) < 1e-4
    assert 0.0 <= bd["ls_u_mom_frac"] <= 1.0


def test_step_mode_small_epsilon_still_moves_carry():
    model = _Toy()
    eta = 1e-2
    opt = _make_opt(model, lr=eta, wd=0.0)
    _set_grads(model, scale=1.0)
    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta, moment_on=MOMENT_ON_STEP)

    # First step builds m(u).
    adapter.snapshot_theta0()
    st1 = adapter.stage_from_q()
    adapter.write_executed_step(eta, st1, into_params=True)
    adapter.commit(st1, params_already_written=True, epsilon=eta)
    theta_after_first = model.weight.detach().clone()

    _set_grads(model, scale=0.5)
    adapter.snapshot_theta0()
    st2 = adapter.stage_from_q()
    tiny = 1e-8
    u_carry = st2.p_momentum[0] + st2.p_decay[0]
    assert u_carry.abs().max().item() > 0.0
    adapter.write_executed_step(tiny, st2, into_params=True)
    delta = model.weight.detach() - theta_after_first
    # Move should include unscaled carry, not only tiny * probe.
    assert delta.abs().max().item() > tiny * 10.0
