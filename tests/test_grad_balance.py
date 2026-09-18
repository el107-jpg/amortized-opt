"""AL-informed GradNorm priorities and combine."""

from __future__ import annotations

import pytest
import torch

from amortized_opt.grad_balance import (
    GRAD_BALANCE_GROUPS,
    al_gradnorm_priorities,
    al_gradnorm_tau,
    apply_al_gradnorm,
    apply_al_pcgrad,
    compute_group_gradients,
    group_grad_cosines,
    output_space_gradnorm_surrogate,
    output_space_al_pcgrad,
    output_space_protected_surrogate_gradient,
    pcgrad_protect_reference,
    pcgrad_two_groups,
    update_group_norm_ema,
)
from amortized_opt.lagrangian import AugmentedLagrangian
from amortized_opt.train_cost import TERM_KEYS, phase2_al_config


def test_tau_torque_cap_renorms_to_one():
    p = {k: 1.0 for k in GRAD_BALANCE_GROUPS}
    p["torque"] = 10.0
    tau = al_gradnorm_tau(p, torque_cap=0.30)
    assert tau["torque"] == pytest.approx(0.30)
    assert sum(tau.values()) == pytest.approx(1.0)


def test_torque_priority_damped_vs_feas_other():
    al = AugmentedLagrangian(phase2_al_config())
    viol = {k: 0.0 for k in al.groups}
    viol["torque_waist"] = 5.0
    viol["joint_box_waist"] = 5.0
    al.lam["torque_waist"] = 10.0
    al.lam["joint_box_waist"] = 10.0
    pri = al_gradnorm_priorities(al, viol, torque_gamma=0.35, beta=1.0)
    assert pri["torque"] < pri["feas_other"]


def test_group_grad_cosines_unit():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([1.0, 0.0])
    c = torch.tensor([-1.0, 0.0])
    params = [a, b]
    cos = group_grad_cosines(params, {
        "torque": (a, torch.zeros(1)),
        "strike": (b, torch.zeros(1)),
        "shape": (c, torch.zeros(1)),
        "feas_other": (a, torch.zeros(1)),
        "waist": (b, torch.zeros(1)),
    })
    assert cos["cos_torque_strike"] == pytest.approx(1.0)
    assert cos["cos_torque_shape"] == pytest.approx(-1.0)
    assert cos["cos_min"] == pytest.approx(-1.0)


def test_apply_al_gradnorm_writes_finite_grad():
    x = torch.nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
    y = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

    class _Terms:
        terms = {k: (x * 0.1).pow(2).mean() for k in TERM_KEYS}
        terms["torque"] = (x * 3.0).pow(2).mean()
        terms["velocity"] = (y * 1.0).pow(2).mean()
        violations = {k: x.new_zeros(2) for k in (
            "torque_peak", "torque_waist", "joint_box_waist", "velocity_waist",
            "collision", "terminal", "waist_pr_band", "velocity_peak",
            "collision_peak", "joint_box_peak",
        )}

    al = AugmentedLagrangian(phase2_al_config())
    stats = apply_al_gradnorm(
        [x, y], _Terms(), al, w_cost=1.0, torque_cap=0.30, torque_gamma=0.35,
    )
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is not None and float(y.grad.abs().sum()) > 0.0
    assert stats["gb_tau_torque"] <= 0.30 + 1e-6
    assert "cos_torque_strike" in stats


def test_batched_group_vjp_matches_sequential():
    x = torch.nn.Parameter(torch.tensor([0.4, -0.7], requires_grad=True))
    y = torch.nn.Parameter(torch.tensor([1.2], requires_grad=True))

    class _Terms:
        terms = {
            k: ((i + 1) * x.square().mean() + (i % 3) * y.square().mean())
            for i, k in enumerate(TERM_KEYS)
        }

    seq = compute_group_gradients([x, y], _Terms(), batched_vjp=False)
    batched = compute_group_gradients([x, y], _Terms(), batched_vjp=True)
    assert seq.keys() == batched.keys()
    for name in seq:
        for gs, gb in zip(seq[name], batched[name]):
            assert gs is not None and gb is not None
            assert torch.allclose(gs, gb, rtol=1e-6, atol=1e-7)


def test_pcgrad_removes_conflicting_component():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-1.0, 0.0])
    ga, gb = pcgrad_two_groups((a,), (b,))
    assert float(ga[0][0]) == pytest.approx(0.0, abs=1e-5)
    assert float(gb[0][0]) == pytest.approx(0.0, abs=1e-5)


def test_waist_group_includes_upright_term():
    """``waist_upright`` is priced in the waist grad-balance bucket (PCGrad)."""
    term_keys, _ = GRAD_BALANCE_GROUPS["waist"]
    assert "waist_upright" in term_keys
    assert "waist_yaw_vel" in term_keys
    shape_keys, _ = GRAD_BALANCE_GROUPS["shape"]
    assert "posture" in shape_keys


def test_waist_priority_floor():
    al = AugmentedLagrangian(phase2_al_config())
    viol = {k: 0.0 for k in al.groups}
    pri = al_gradnorm_priorities(al, viol, waist_floor=2.0)
    assert pri["waist"] >= 2.0


def test_apply_al_pcgrad_writes_finite_grad():
    x = torch.nn.Parameter(torch.tensor([1.0, 2.0], requires_grad=True))
    y = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

    class _Terms:
        terms = {k: (x * 0.1).pow(2).mean() for k in TERM_KEYS}
        terms["torque"] = (x * 3.0).pow(2).mean()
        terms["waist_yaw_vel"] = (y * 2.0).pow(2).mean()
        violations = {k: x.new_zeros(2) for k in (
            "torque_peak", "torque_waist", "joint_box_waist", "velocity_waist",
            "collision", "terminal", "waist_pr_band", "velocity_peak",
            "collision_peak", "joint_box_peak",
        )}

    al = AugmentedLagrangian(phase2_al_config())
    stats = apply_al_pcgrad(
        [x, y], _Terms(), al, w_cost=1.0, torque_cap=0.30, waist_floor=2.0,
    )
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert stats.get("gb_pcgrad_tw") == 1.0
    assert stats["gb_pri_waist"] >= 2.0


def test_output_space_matches_parameter_space_for_identity_model():
    traj = torch.nn.Parameter(torch.tensor([0.4, -0.7], requires_grad=True))

    class _Terms:
        terms = {
            k: ((i + 1) * traj.square()).mean()
            for i, k in enumerate(TERM_KEYS)
        }
        violations = {k: traj.new_zeros(2) for k in (
            "torque_peak", "torque_waist", "joint_box_waist", "velocity_waist",
            "collision", "terminal", "waist_pr_band", "velocity_peak",
            "collision_peak", "joint_box_peak",
        )}
        balance_terms = {"joint_box_waist": traj.square().mean()}

    al = AugmentedLagrangian(phase2_al_config())
    apply_al_pcgrad([traj], _Terms(), al)
    parameter_grad = traj.grad.detach().clone()
    output_grad, stats = output_space_al_pcgrad(traj, _Terms(), al)
    assert torch.allclose(output_grad, parameter_grad, rtol=1e-6, atol=1e-7)
    assert stats["gb_output_space"] == 1.0


def test_ema_surrogate_matches_refresh_without_pcgrad_conflict():
    traj = torch.nn.Parameter(torch.tensor([0.4, -0.7], requires_grad=True))

    class _Terms:
        terms = {
            k: ((i + 1) * traj.square()).mean()
            for i, k in enumerate(TERM_KEYS)
        }
        violations = {k: traj.new_zeros(2) for k in (
            "torque_peak", "torque_waist", "joint_box_waist", "velocity_waist",
            "collision", "terminal", "waist_pr_band", "velocity_peak",
            "collision_peak", "joint_box_peak",
        )}
        balance_terms = {"joint_box_waist": traj.square().mean()}

    ct = _Terms()
    al = AugmentedLagrangian(phase2_al_config())
    exact, stats = output_space_al_pcgrad(traj, ct, al)
    ema = update_group_norm_ema({}, stats, decay=0.9)
    surrogate, fast_stats = output_space_gradnorm_surrogate(ct, al, ema)
    (fast,) = torch.autograd.grad(surrogate, traj)
    assert torch.allclose(fast, exact, rtol=1e-6, atol=1e-7)
    assert fast_stats["gb_refresh"] == 0.0


def test_one_sided_pcgrad_changes_only_torque_subject():
    torque = (torch.tensor([1.0, -1.0]),)
    waist_box = (torch.tensor([-1.0, 0.0]),)
    projected = pcgrad_protect_reference(torque, waist_box)
    assert torch.allclose(projected[0], torch.tensor([0.0, -1.0]))
    assert torch.equal(waist_box[0], torch.tensor([-1.0, 0.0]))


def test_fast_surrogate_projects_torque_off_waist_box():
    traj = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

    class _View:
        group_terms = {
            "torque": traj[0],
            "feas_other": 0.0 * traj.sum(),
            "strike": 0.0 * traj.sum(),
            "shape": 0.0 * traj.sum(),
            "waist": 0.0 * traj.sum(),
        }
        protected_terms = {"joint_box_waist": -traj[0] + traj[1]}
        violations = {}

    stats = {"gb_scale_torque": traj.new_tensor(1.0)}
    grad, out = output_space_protected_surrogate_gradient(
        traj, _View.group_terms["torque"], _View(), stats,
    )
    box_grad = torch.tensor([-1.0, 1.0])
    assert torch.dot(grad, box_grad).item() == pytest.approx(0.0, abs=1e-7)
    assert out["gb_pcgrad_torque_box_conflict"] == 1.0
