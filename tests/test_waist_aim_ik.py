"""Vendored waist-aim conformance + the inertia-weighted velocity IK."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt import constants as C
from amortized_opt import waist_aim as W
from amortized_opt.fk import ChainFK
from amortized_opt.ik import (
    arm_joint_indices,
    damped_pinv,
    inertia_weighted_pinv,
    solve_tip_velocity,
    tip_jacobian,
    tip_position,
)
from amortized_opt.pd_sim import mass_matrix
from amortized_opt.rnea import BatchedRNEA
from conftest import import_repo

N = C.N_DOF
_WAIST = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")


# --------------------------------------------------------------- waist aim --
def test_waist_aim_conformance_with_repo(capsys):
    repo = import_repo("waist_aim")
    rng = np.random.default_rng(0)
    worst = {"aim_yaw": 0.0, "yaw_reference": 0.0, "block": 0.0, "v_start": 0.0}

    for _ in range(200):
        tl = rng.uniform([0.0, -0.6, 0.7], [0.8, 0.6, 1.2])
        tr = rng.uniform([0.0, -0.6, 0.7], [0.8, 0.6, 1.2])
        yaw_now = float(rng.uniform(-1.5, 1.5))
        for targets in ({"left": tl, "right": tr}, {"left": tl}, {"right": tr}):
            a = W.aim_yaw(targets, yaw_current=yaw_now)
            b = repo.aim_yaw(targets, yaw_current=yaw_now)
            worst["aim_yaw"] = max(worst["aim_yaw"], abs(a - b))

    for _ in range(200):
        y0, y1 = rng.uniform(-1.7, 1.7, size=2)
        n_wp = int(rng.integers(4, 60))
        rate = float(rng.normal(0.0, 0.05))
        a = W.yaw_reference(y0, y1, n_wp, settle_wp=3, yawd_start=rate, max_move_wp=10)
        b = repo.yaw_reference(y0, y1, n_wp, settle_wp=3, yawd_start=rate, max_move_wp=10)
        worst["yaw_reference"] = max(worst["yaw_reference"], float(np.abs(a - b).max()))

        sp = rng.normal(0.0, 0.05, size=3)
        A = W.waist_reference_block(_WAIST, y0, y1, n_wp, setpoint=sp, dt=0.02,
                                    yaw_rate_start=rate, max_move_s=0.2)
        B = repo.waist_reference_block(_WAIST, y0, y1, n_wp, setpoint=sp, dt=0.02,
                                       yaw_rate_start=rate, max_move_s=0.2)
        worst["block"] = max(worst["block"], float(np.abs(A - B).max()))
        worst["v_start"] = max(
            worst["v_start"],
            float(np.abs(W.waist_start_velocity(A, 0.02) - repo.waist_start_velocity(B, 0.02)).max()),
        )

    with capsys.disabled():
        print("\n[waist_aim vs repo] " + ", ".join(f"{k}={v:.1e}" for k, v in worst.items()))
    assert all(v <= 1e-8 for v in worst.values())


def test_waist_controller_conformance_with_repo(capsys):
    repo = import_repo("waist_aim")
    rng = np.random.default_rng(1)
    mine = W.WaistYawController(dt=0.02)
    theirs = repo.WaistYawController(dt=0.02)
    worst = 0.0
    for step in range(120):
        targets = {
            "left": rng.uniform([0.2, 0.0, 0.8], [0.7, 0.6, 1.2]),
            "right": rng.uniform([0.2, -0.6, 0.8], [0.7, 0.0, 1.2]),
        }
        ttb = float(rng.uniform(0.1, 1.4))
        assert mine.update(step, targets, ttb) == theirs.update(step, targets, ttb)
        worst = max(worst, float(np.abs(mine.command_at(step) - theirs.command_at(step)).max()))
    with capsys.disabled():
        print(f"\n[WaistYawController vs repo] max|dcmd|={worst:.1e}")
    assert worst <= 1e-8


def test_aim_targets_from_layout_is_yaw_only_and_uses_the_vendored_kit():
    t = W.aim_targets_from_layout({"left": "hihat", "right": "snare"})
    assert set(t) == {"left", "right"}
    env = W.aim_targets_from_layout({"left": "hihat"}, frame="env")
    assert np.allclose(env["left"][:2], t["left"][:2])  # azimuth is frame-free
    assert np.isclose(env["left"][2] - t["left"][2], C.PELVIS_IN_ENV[2])
    with pytest.raises(KeyError):
        W.aim_targets_from_layout({"left": "no_such_drum"})


def test_waist_reference_freezes_pitch_and_roll():
    sp = np.array([0.0, 0.11, -0.07])
    ref = W.waist_reference_block(_WAIST, 0.0, 0.9, 30, setpoint=sp, dt=0.02, max_move_s=0.2)
    assert np.allclose(ref[:, 1], sp[1]) and np.allclose(ref[:, 2], sp[2])
    assert abs(ref[-1, 0] - 0.9) < 1e-12


# ---------------------------------------------------------------- velocity --
@pytest.fixture(scope="module")
def kin():
    fk = ChainFK(dtype=torch.float64)
    rnea = BatchedRNEA.from_urdf(dtype=torch.float64)
    rng = np.random.default_rng(0)
    q = torch.as_tensor(
        C.ready_joint_pos()[None, :] + rng.normal(0.0, 0.15, size=(32, N)), dtype=torch.float64
    )
    return fk, rnea, q, mass_matrix(rnea, q)


def test_tip_jacobian_matches_finite_differences(kin, capsys):
    fk, _rnea, q, _M = kin
    J = tip_jacobian(fk, q, "right")
    eps = 1e-6
    worst = 0.0
    for j in range(N):
        dq = torch.zeros_like(q)
        dq[:, j] = eps
        num = (tip_position(fk, q + dq, "right") - tip_position(fk, q - dq, "right")) / (2 * eps)
        worst = max(worst, float((num - J[:, :, j]).abs().max()))
    with capsys.disabled():
        print(f"\n[tip jacobian vs central difference] max err={worst:.2e}")
    assert worst < 1e-7


def test_pinvs_hit_the_requested_tip_velocity(kin, capsys):
    fk, _rnea, q, M = kin
    rng = np.random.default_rng(3)
    v = torch.as_tensor(rng.normal(0.0, 1.5, size=(q.shape[0], 3)), dtype=torch.float64)
    J = tip_jacobian(fk, q, "right")
    errs = {}
    for mode, kw in (("plain", dict(damping=1e-9)), ("inertia", dict(damping=0.0))):
        qd = solve_tip_velocity(fk, q, "right", v, M=M, weighting=mode, **kw)
        errs[mode] = float((torch.einsum("bij,bj->bi", J, qd) - v).abs().max())
    with capsys.disabled():
        print("\n[velocity IK tip accuracy] " + ", ".join(f"{k}={e:.2e}" for k, e in errs.items()))
    assert all(e <= 1e-6 for e in errs.values())


def test_inertia_weighting_shifts_motion_into_the_wrists(kin, capsys):
    """M^-1 weighting must prefer the low-inertia distal joints."""
    fk, _rnea, q, M = kin
    rng = np.random.default_rng(5)
    v = torch.as_tensor(rng.normal(0.0, 1.5, size=(q.shape[0], 3)), dtype=torch.float64)
    wrist = [i for i, n in enumerate(C.G1_UPPER_BODY_JOINTS) if n.startswith("right_wrist")]
    frac = {}
    for mode, kw in (("plain", dict(damping=1e-9)), ("inertia", dict(damping=0.0))):
        qd = solve_tip_velocity(fk, q, "right", v, M=M, weighting=mode, **kw)
        frac[mode] = float((qd[:, wrist].abs().sum(1) / qd.abs().sum(1)).mean())
    # ...and the arm-only subset, which is what the start sampler restricts to.
    idx = arm_joint_indices("right")
    sub = {}
    for mode, kw in (("plain", dict(damping=1e-9)), ("inertia", dict(damping=0.0))):
        qd = solve_tip_velocity(fk, q, "right", v, M=M, weighting=mode, joint_subset=idx, **kw)
        sub[mode] = float((qd[:, wrist].abs().sum(1) / qd.abs().sum(1)).mean())
    with capsys.disabled():
        print(f"\n[wrist |qd| fraction] all 17 DoF: plain={frac['plain']:.3f} -> "
              f"inertia={frac['inertia']:.3f}; arm-only subset: plain={sub['plain']:.3f} -> "
              f"inertia={sub['inertia']:.3f}")
    assert frac["inertia"] > frac["plain"]


def test_inertia_pinv_minimises_kinetic_energy(kin):
    """The dynamically-consistent solution is the min-energy one."""
    fk, _rnea, q, M = kin
    rng = np.random.default_rng(7)
    v = torch.as_tensor(rng.normal(0.0, 1.0, size=(q.shape[0], 3)), dtype=torch.float64)
    J = tip_jacobian(fk, q, "right")
    qd_i = solve_tip_velocity(fk, q, "right", v, M=M, weighting="inertia", damping=0.0)
    qd_p = solve_tip_velocity(fk, q, "right", v, M=M, weighting="plain", damping=1e-9)
    ke = lambda x: torch.einsum("bi,bij,bj->b", x, M, x)  # noqa: E731
    assert bool((ke(qd_i) <= ke(qd_p) + 1e-9).all())
    # ...and plain pinv is the min-norm one.
    assert bool((qd_p.norm(dim=-1) <= qd_i.norm(dim=-1) + 1e-9).all())


def test_damped_pinv_matches_the_repo_helper():
    repo = import_repo("velocity_ik")
    rng = np.random.default_rng(0)
    J = torch.as_tensor(rng.normal(size=(4, 3, N)), dtype=torch.float64)
    assert float((damped_pinv(J, 1e-3) - repo.damped_pinv(J, 1e-3)).abs().max()) == 0.0


def test_tip_jacobian_under_inference_mode(kin):
    """``play_isaac.run_episode`` wraps reset in inference_mode; J still needed."""
    fk, _rnea, q, _M = kin
    q = q[:4]
    with torch.no_grad():
        J_ref = tip_jacobian(fk, q, "right")
    with torch.inference_mode():
        J = tip_jacobian(fk, q, "right")
    assert torch.allclose(J, J_ref, atol=1e-12)


def test_tip_jacobian_fk_built_under_inference_mode():
    """IsaacMPC first-builds ChainFK during inference_mode reset."""
    from amortized_opt.fk import ChainFK

    q = torch.zeros(2, N, dtype=torch.float32)
    with torch.inference_mode():
        fk = ChainFK(device="cpu", dtype=torch.float32)
        J = tip_jacobian(fk, q, "right")
    assert torch.isfinite(J).all()
    assert J.shape == (2, 3, N)


def test_inertia_pinv_rejects_ambiguous_arguments(kin):
    _fk, _rnea, _q, M = kin
    J = torch.zeros(2, 3, N, dtype=torch.float64)
    with pytest.raises(ValueError):
        inertia_weighted_pinv(J)
    with pytest.raises(ValueError):
        inertia_weighted_pinv(J, M=M[:2], Minv=M[:2])
