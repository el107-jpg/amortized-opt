"""RNEA conformance: vs Pinocchio, vs the repo original, and the armature term."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.limits import build_reduced_model, isaac_armature
from amortized_opt.rnea import BatchedRNEA, max_error_vs_pinocchio
from conftest import import_repo


@pytest.fixture(scope="module")
def model_pair():
    pytest.importorskip("pinocchio")
    return build_reduced_model()


def _sample(n: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 17, generator=g, dtype=torch.float64)
    qd = torch.randn(n, 17, generator=g, dtype=torch.float64) * 0.5
    qdd = torch.randn(n, 17, generator=g, dtype=torch.float64) * 2.0
    return q, qd, qdd


def test_rnea_matches_pinocchio(model_pair, capsys):
    """Link-inertia part vs Pinocchio on this exact URDF."""
    import pinocchio as pin

    model, opt2pin = model_pair
    data = model.createData()
    r64 = BatchedRNEA.from_pinocchio(model, opt2pin, armature=None, dtype=torch.float64)
    r32 = BatchedRNEA.from_pinocchio(model, opt2pin, armature=None, dtype=torch.float32)
    e64 = max_error_vs_pinocchio(model, data, opt2pin, r64, n=24)
    e32 = max_error_vs_pinocchio(model, data, opt2pin, r32, n=24)
    with capsys.disabled():
        print(f"\n[rnea vs pinocchio] fp64 {e64:.3e} N*m, fp32 {e32:.3e} N*m")
    assert e64 <= 1e-5
    assert e32 <= 1e-3  # fp32 roundoff on ~100 N*m torques
    _ = pin  # imported for the skip guard


def test_armature_is_exactly_diag_times_qdd(model_pair, capsys):
    model, opt2pin = model_pair
    arm = isaac_armature()
    with_a = BatchedRNEA.from_pinocchio(model, opt2pin, armature=arm, dtype=torch.float64)
    no_a = with_a.without_armature()
    q, qd, qdd = _sample(32)
    d = with_a.rnea(q, qd, qdd) - no_a.rnea(q, qd, qdd)
    want = torch.tensor(arm.tolist(), dtype=torch.float64).view(1, -1) * qdd
    err = float((d - want).abs().max())
    with capsys.disabled():
        print(f"\n[armature] |tau_with - tau_without - diag(a)qdd| = {err:.3e}")
    # Exact up to the roundoff of (tau + a*qdd) - tau in fp64.
    assert err <= 1e-14


def test_armature_matters_at_strike_accelerations(model_pair, capsys):
    """Sanity: the term is non-negligible against the 5 N*m wrist limit."""
    model, opt2pin = model_pair
    r = BatchedRNEA.from_pinocchio(model, opt2pin, armature=isaac_armature(), dtype=torch.float64)
    q = torch.zeros(1, 17, dtype=torch.float64)
    qdd = torch.zeros(1, 17, dtype=torch.float64)
    qdd[0, :] = 300.0  # strike-scale wrist acceleration (rad/s^2)
    extra = float((r.rnea(q, q, qdd) - r.without_armature().rnea(q, q, qdd)).abs().max())
    with capsys.disabled():
        print(f"\n[armature magnitude @300 rad/s^2] {extra:.3f} N*m")
    assert extra > 1.0


def test_forward_matches_repo_original(model_pair, capsys):
    """Same forward values as the repo's (forward-only) BatchedRNEA."""
    repo = import_repo("trajopt.batch_rnea")
    model, opt2pin = model_pair
    ref = repo.BatchedRNEA.from_pinocchio(model, np.asarray(opt2pin), dtype=torch.float64)
    ours = BatchedRNEA.from_pinocchio(model, opt2pin, armature=None, dtype=torch.float64)
    q, qd, qdd = _sample(64, seed=5)
    with torch.no_grad():
        a = ref.rnea(q, qd, qdd)
        b = ours.rnea(q, qd, qdd)
    err = float((a - b).abs().max())
    with capsys.disabled():
        print(f"\n[rnea vs repo original, fp64] max abs err = {err:.3e}")
    assert err <= 1e-12


def test_repo_original_backward_still_raises(model_pair):
    """Documents the defect this vendored copy fixes (skips if it ever gets fixed)."""
    repo = import_repo("trajopt.batch_rnea")
    model, opt2pin = model_pair
    ref = repo.BatchedRNEA.from_pinocchio(model, np.asarray(opt2pin), dtype=torch.float64)
    q, qd, qdd = _sample(4)
    q.requires_grad_(True)
    try:
        ref.rnea(q, qd, qdd).sum().backward()
    except RuntimeError as exc:
        assert "inplace" in str(exc)
        return
    pytest.skip("repo BatchedRNEA backward now works; vendored fix no longer unique")


def test_rnea_backward_is_finite(model_pair):
    model, opt2pin = model_pair
    r = BatchedRNEA.from_pinocchio(model, opt2pin, armature=isaac_armature(), dtype=torch.float64)
    q, qd, qdd = _sample(16)
    for t in (q, qd, qdd):
        t.requires_grad_(True)
    r.rnea(q, qd, qdd).square().sum().backward()
    for t in (q, qd, qdd):
        assert torch.isfinite(t.grad).all()
        assert float(t.grad.abs().max()) > 0.0
