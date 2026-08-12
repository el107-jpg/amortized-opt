"""Batched noised-PD simulator: repo conformance, substep behaviour, speed."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from amortized_opt import constants as C
from amortized_opt.pd_sim import (
    BatchedPDSim,
    PDSimConfig,
    frozen_minv,
    mass_matrix,
    simulate_pd,
)
from amortized_opt.rnea import BatchedRNEA
from conftest import import_repo

N = C.N_DOF


def _random_problem(B=8, K=20, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(B, K + 1, N)) * 0.1
    A = rng.normal(size=(B, K, N, N)) * 0.05
    Minv = np.tile(np.eye(N), (B, K, 1, 1)) * 3.0 + A @ A.transpose(0, 1, 3, 2)
    kp = np.full(N, 40.0)
    kd = np.full(N, 2.0)
    q0 = rng.normal(size=(B, N)) * 0.05
    v0 = rng.normal(size=(B, N)) * 0.05
    return x, Minv, kp, kd, q0, v0


def test_conformance_with_repo_closed_loop(capsys):
    """substeps=1 must reproduce ``closed_loop.simulate_actual_state`` exactly.

    The FOH position ramp collapses to the new target at one substep, so this
    is not an approximation -- it is the same recursion.
    """
    cl = import_repo("trajopt.closed_loop")
    x, Minv, kp, kd, q0, v0 = _random_problem()
    dt = C.CONTROL_DT
    ref = [cl.simulate_actual_state(x[b], Minv[b], kp, kd, dt, q0[b], v0[b]) for b in range(len(x))]
    qn = np.stack([r[0] for r in ref])
    vn = np.stack([r[1] for r in ref])
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    qa, va, _int = simulate_pd(t(x), t(Minv), t(kp), t(kd), dt, t(q0), t(v0), substeps=1)
    eq = float(np.abs(qa.numpy() - qn).max())
    ev = float(np.abs(va.numpy() - vn).max())
    with capsys.disabled():
        print(f"\n[pd conformance fp64] max|dq|={eq:.3e}  max|dv|={ev:.3e}")
    assert eq <= 1e-10 and ev <= 1e-10


def test_ff_scale_conformance():
    """The feedforward knob must match the repo's for a non-default value."""
    cl = import_repo("trajopt.closed_loop")
    x, Minv, kp, kd, q0, v0 = _random_problem(B=3, K=12, seed=3)
    dt = C.CONTROL_DT
    for ff in (0.0, 0.5):
        ref = [
            cl.simulate_actual_state(x[b], Minv[b], kp, kd, dt, q0[b], v0[b], ff)
            for b in range(len(x))
        ]
        t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
        qa, va, _int = simulate_pd(
            t(x), t(Minv), t(kp), t(kd), dt, t(q0), t(v0), substeps=1, ff_scale=ff
        )
        assert float(np.abs(va.numpy() - np.stack([r[1] for r in ref])).max()) <= 1e-10


def test_substeps_converge_to_the_control_dt_solution(capsys):
    """Refining the substep count is a convergent sequence, not a new plant.

    There is no reference for 250 Hz stepping, so the check is structural: the
    difference from the control-dt solution must shrink monotonically as the
    substep count rises, i.e. all the modes are resolved rather than aliased.
    """
    x, Minv, kp, kd, q0, v0 = _random_problem(B=4, K=25, seed=7)
    dt = C.CONTROL_DT
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    args = (t(x), t(Minv), t(kp), t(kd), dt, t(q0), t(v0))
    base = simulate_pd(*args, substeps=1)[1][:, -1]
    fine = simulate_pd(*args, substeps=200)[1][:, -1]
    errs = []
    for s in (1, 2, 5, 10, 25):
        va = simulate_pd(*args, substeps=s)[1][:, -1]
        errs.append(float((va - fine).abs().max()))
    with capsys.disabled():
        print("\n[pd substep convergence] |v_S - v_200| for S=1,2,5,10,25: "
              + ", ".join(f"{e:.3e}" for e in errs))
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    assert errs[0] > 0.0  # 5 substeps is genuinely a different (better) answer
    assert float((base - fine).abs().max()) > 1e-6


def test_constant_target_settles_with_zero_velocity(capsys):
    """A held target must be reached exactly, at rest -- the plant is a servo."""
    rnea = BatchedRNEA.from_urdf(dtype=torch.float64)
    sim = BatchedPDSim(PDSimConfig(), dtype=torch.float64, rnea=rnea)
    B, K = 3, 300
    tgt = torch.zeros(B, N, dtype=torch.float64)
    tgt[:, 3] = 0.2
    tgt[:, 10] = -0.15
    x = tgt.unsqueeze(1).expand(B, K + 1, N).contiguous()
    Minv = frozen_minv(rnea, torch.zeros(B, K, N, dtype=torch.float64))
    kp, kd, _ki = sim.nominal_gains(B)
    z = torch.zeros(B, N, dtype=torch.float64)
    qa, va, _int = sim.rollout(x, qa0=z, va0=z, kp=kp, kd=kd, Minv=Minv)
    dq = float((qa[:, -1] - tgt).abs().max())
    dv = float(va[:, -1].abs().max())
    with capsys.disabled():
        print(f"\n[pd settling, {K} ticks] |q-target|={dq:.2e}  |v|={dv:.2e}")
    assert dq < 1e-6 and dv < 1e-6
    assert torch.isfinite(qa).all() and torch.isfinite(va).all()


def test_mass_matrix_matches_pinocchio_crba(capsys):
    """The frozen inertia must be CRBA + armature, not link inertia alone."""
    pin = pytest.importorskip("pinocchio")
    from amortized_opt.limits import build_reduced_model, isaac_armature

    rnea = BatchedRNEA.from_urdf(dtype=torch.float64)
    rng = np.random.default_rng(0)
    q = torch.as_tensor(rng.normal(size=(5, N)) * 0.3, dtype=torch.float64)
    M = mass_matrix(rnea, q)
    model, opt2pin = build_reduced_model()
    data = model.createData()
    worst = 0.0
    for b in range(q.shape[0]):
        qp = np.zeros(model.nq)
        qp[opt2pin] = q[b].numpy()
        Mp = np.asarray(pin.crba(model, data, qp))
        Mp = np.triu(Mp) + np.triu(Mp, 1).T
        Mp = Mp[np.ix_(opt2pin, opt2pin)] + np.diag(isaac_armature())
        worst = max(worst, float(np.abs(M[b].numpy() - Mp).max()))
    sym = float((M - M.transpose(1, 2)).abs().max())
    with capsys.disabled():
        print(f"\n[mass matrix] max|M - (CRBA+armature)|={worst:.2e}  asymmetry={sym:.2e}")
    assert worst < 1e-9 and sym < 1e-9


def test_gain_randomization_is_positive_and_centred(capsys):
    sim = BatchedPDSim(PDSimConfig(kp_sigma=0.2, kd_sigma=0.2), dtype=torch.float64)
    g = torch.Generator()
    g.manual_seed(0)
    kp, kd, ki = sim.sample_gains(20_000, generator=g)
    assert bool((kp > 0).all()) and bool((kd > 0).all())
    assert ki is None  # ki=0 by default
    rp = (kp / sim.kp_nominal).flatten()
    rd = (kd / sim.kd_nominal).flatten()
    with capsys.disabled():
        print(f"\n[gain randomization] Kp ratio median={float(rp.median()):.4f} "
              f"sd(log)={float(rp.log().std()):.4f}; Kd median={float(rd.median()):.4f} "
              f"sd(log)={float(rd.log().std()):.4f}")
    assert abs(float(rp.median()) - 1.0) < 0.02
    assert abs(float(rp.log().std()) - 0.2) < 0.02
    assert abs(float(rd.log().std()) - 0.2) < 0.02
    # sigma = 0 must be exactly the nominal, not "approximately".
    s0 = BatchedPDSim(PDSimConfig(kp_sigma=0.0, kd_sigma=0.0), dtype=torch.float64)
    kp0, _, _ki0 = s0.sample_gains(4)
    assert torch.allclose(kp0, s0.kp_nominal.expand_as(kp0))


def test_gradient_flows_to_the_commanded_trajectory(capsys):
    x, Minv, kp, kd, q0, v0 = _random_problem(B=4, K=30, seed=11)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    xg = t(x).requires_grad_(True)
    kpg = t(kp).requires_grad_(True)
    qa, va, _int = simulate_pd(xg, t(Minv), kpg, t(kd), C.CONTROL_DT, t(q0), t(v0), substeps=5)
    va[:, -1].square().sum().backward()
    assert bool(xg.grad.isfinite().all()) and float(xg.grad.abs().max()) > 0.0
    assert bool(kpg.grad.isfinite().all())
    with capsys.disabled():
        print(f"\n[pd gradient] |d(v_K^2)/dx|max={float(xg.grad.abs().max()):.3f} "
              f"|d/dKp|max={float(kpg.grad.abs().max()):.3f}")


@pytest.mark.parametrize("substeps", [5])
def test_throughput_1024x50(cuda_device, capsys, substeps):
    """The plan's data generator needs >= 1024 rollouts x 50 knots per batch."""
    dev = cuda_device
    B, K = 1024, 50
    x = torch.randn(B, K + 1, N, device=dev) * 0.1
    Minv = (torch.eye(N, device=dev) * 3.0).expand(B, K, N, N).contiguous()
    kp = torch.full((N,), 40.0, device=dev)
    kd = torch.full((N,), 2.0, device=dev)
    q0 = torch.randn(B, N, device=dev) * 0.05
    v0 = torch.randn(B, N, device=dev) * 0.05

    def bench(fn, reps=10):
        for _ in range(3):
            with torch.no_grad():
                fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            with torch.no_grad():
                fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / reps * 1e3

    run = lambda f: f(x, Minv, kp, kd, 0.02, q0, v0, substeps=substeps)  # noqa: E731
    eager = bench(lambda: run(simulate_pd))
    torch._dynamo.reset()
    compiled_fn = torch.compile(simulate_pd, mode="reduce-overhead", dynamic=False)
    run(compiled_fn)
    compiled = bench(lambda: run(compiled_fn))
    with capsys.disabled():
        print(f"\n[pd throughput {B}x{K}, {substeps} substeps, {dev}] "
              f"eager {eager:.2f} ms | compiled {compiled:.2f} ms")
    assert compiled < 15.0
