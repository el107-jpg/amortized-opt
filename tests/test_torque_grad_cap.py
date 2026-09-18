"""Torque gradient budget cap (LS-compatible, no PCGrad)."""

import pytest
import torch

from amortized_opt.grad_balance import clip_torque_gradient_budget
from amortized_opt.train_cost import (
    _accumulate_torque_grad_cap_stats,
    _torque_grad_cap_log_fields,
    phase2_al_config,
)


def _run_budget(torque, rest, *, cap, eps=1e-8):
    torque = torch.tensor(torque, dtype=torch.float64)
    rest = torch.tensor(rest, dtype=torch.float64)
    param = torch.nn.Parameter(torch.zeros_like(torque))
    torque_mean = torch.dot(param, torque)
    total = torque_mean + torch.dot(param, rest)
    total.backward(retain_graph=True)
    before = param.grad.detach().clone()
    stats = clip_torque_gradient_budget(
        [param], torque_mean, w_cost=1.0, cap=cap, eps=eps,
    )
    alpha = stats["torque_grad_cap_scale"]
    actual_ratio = float(
        (alpha * torque).norm()
        / max(float(param.grad.norm()), eps)
    )
    assert stats["torque_grad_frac_after"] == pytest.approx(
        actual_ratio, rel=1e-12, abs=1e-12,
    )
    assert actual_ratio <= cap + 1e-12
    return param, before, stats


def test_torque_budget_aligned_gradients():
    _param, _before, stats = _run_budget(
        [2.0, 0.0], [1.0, 0.0], cap=0.4,
    )
    assert stats["torque_grad_cap_scale"] == pytest.approx(1.0 / 3.0)
    assert stats["torque_grad_rest_torque_cosine"] == pytest.approx(1.0)
    assert stats["torque_grad_frac_before"] == pytest.approx(2.0 / 3.0)
    assert stats["torque_grad_full_norm_before"] == pytest.approx(3.0)
    assert stats["torque_grad_torque_norm_before"] == pytest.approx(2.0)
    assert stats["torque_grad_rest_norm"] == pytest.approx(1.0)
    assert stats["torque_grad_full_norm_after"] == pytest.approx(5.0 / 3.0)
    assert stats["torque_grad_torque_norm_after"] == pytest.approx(2.0 / 3.0)


def test_torque_budget_orthogonal_gradients():
    _param, _before, stats = _run_budget(
        [2.0, 0.0], [0.0, 1.0], cap=0.4,
    )
    expected = (0.16 / 3.36) ** 0.5
    assert stats["torque_grad_cap_scale"] == pytest.approx(expected)
    assert stats["torque_grad_rest_torque_cosine"] == pytest.approx(0.0)


def test_torque_budget_exact_cancellation():
    _param, _before, stats = _run_budget(
        [2.0, 0.0], [-2.0, 0.0], cap=0.4,
    )
    assert stats["torque_grad_cap_scale"] == pytest.approx(2.0 / 7.0)
    assert stats["torque_grad_rest_torque_cosine"] == pytest.approx(-1.0)
    assert stats["torque_grad_frac_before"] == pytest.approx(2.0e8)


def test_torque_budget_zero_remainder_uses_eps_floor():
    param, _before, stats = _run_budget(
        [3.0, 0.0], [0.0, 0.0], cap=0.4, eps=1e-8,
    )
    assert stats["torque_grad_cap_scale"] == pytest.approx(0.4e-8 / 3.0)
    assert float(param.grad.norm()) == pytest.approx(0.4e-8)


def test_torque_budget_inactive_preserves_gradients():
    param, before, stats = _run_budget(
        [0.1, 0.0], [0.0, 1.0], cap=0.7,
    )
    assert stats["torque_grad_cap_scale"] == 1.0
    assert stats["torque_grad_frac_after"] == pytest.approx(
        stats["torque_grad_frac_before"],
    )
    assert torch.equal(param.grad, before)


def test_torque_budget_nonfinite_full_gradient_is_untouched():
    param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    torque_mean = param.sum()
    (torque_mean + (param * float("nan")).sum()).backward(retain_graph=True)
    before = param.grad.detach().clone()

    stats = clip_torque_gradient_budget(
        [param], torque_mean, w_cost=1.0, cap=0.4,
    )

    assert stats["torque_grad_status_nonfinite_full"] == 1.0
    assert torch.allclose(param.grad, before, equal_nan=True)


def test_torque_budget_nonfinite_torque_gradient_is_untouched():
    param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    torque_mean = (param * float("nan")).sum()
    (2.0 * param).sum().backward(retain_graph=True)
    before = param.grad.detach().clone()

    stats = clip_torque_gradient_budget(
        [param], torque_mean, w_cost=1.0, cap=0.4,
    )

    assert stats["torque_grad_status_nonfinite_torque"] == 1.0
    assert torch.equal(param.grad, before)


def test_torque_budget_train_record_aggregation():
    roll_acc = {}
    finite_a = {
        "torque_grad_cap": 0.4,
        "torque_grad_frac_before": 0.8,
        "torque_grad_frac_after": 0.4,
        "torque_grad_cap_scale": 0.25,
        "torque_grad_rest_torque_cosine": -0.5,
        "torque_grad_rest_norm": 3.0,
        "torque_grad_torque_norm_before": 4.0,
    }
    finite_b = {
        "torque_grad_cap": 0.4,
        "torque_grad_frac_before": 0.6,
        "torque_grad_frac_after": 0.3,
        "torque_grad_cap_scale": 0.75,
        "torque_grad_rest_torque_cosine": 0.5,
        "torque_grad_rest_norm": 5.0,
        "torque_grad_torque_norm_before": 8.0,
    }
    for stats in (
        finite_a,
        {"torque_grad_cap": 0.4,
         "torque_grad_status_nonfinite_full": 1.0},
        finite_b,
        {"torque_grad_cap": 0.4,
         "torque_grad_status_nonfinite_torque": 1.0},
    ):
        _accumulate_torque_grad_cap_stats(roll_acc, stats)

    record = _torque_grad_cap_log_fields(roll_acc, cap=0.4)

    assert record["torque_grad_cap"] == 0.4
    assert record["torque_grad_cap_n"] == 4.0
    assert record["torque_grad_finite_n"] == 2.0
    assert record["torque_grad_frac_before"] == pytest.approx(0.7)
    assert record["torque_grad_frac_after"] == pytest.approx(0.35)
    assert record["torque_grad_cap_scale"] == pytest.approx(0.5)
    assert record["torque_grad_rest_torque_cosine"] == pytest.approx(0.0)
    assert record["torque_grad_rest_norm"] == pytest.approx(4.0)
    assert record["torque_grad_torque_norm_before"] == pytest.approx(6.0)
    assert record["torque_grad_status_nonfinite_full"] == 1.0
    assert record["torque_grad_status_nonfinite_torque"] == 1.0


def test_rho_terminal_override():
    al = phase2_al_config(rho_terminal=42.0)
    assert al.rho["terminal"] == 42.0
