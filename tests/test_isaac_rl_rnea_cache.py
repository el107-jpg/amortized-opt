"""The frozen RNEA tree must be the live one, not merely close to it.

``isaac_rl.rnea_cache`` exists because Pinocchio cannot expose ``model.parents``
inside an Isaac process, which silently disabled the gravity/Coriolis
feedforward and the whole analytic half of the reward in the RL-1 smoke run.
The cache restores both -- but a cache that is *approximately* right would be a
wrong-physics bug that raises nowhere, so these tests pin exactness and pin the
staleness guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amortized_opt as A  # noqa: E402
from isaac_rl.rnea_cache import (  # noqa: E402
    CACHE_PATH,
    build_cache,
    build_or_load,
    cache_fingerprint,
    load_cached,
)


@pytest.fixture(scope="module")
def live():
    return A.BatchedRNEA.from_urdf(device="cpu", dtype=torch.float64)


@pytest.fixture(scope="module")
def cached():
    if not CACHE_PATH.is_file():
        build_cache()
    return load_cached(device="cpu", dtype=torch.float64)


def _states(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = len(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits()
    lo = torch.as_tensor(lo, dtype=torch.float64)
    hi = torch.as_tensor(hi, dtype=torch.float64)
    q = lo + (hi - lo) * torch.rand(n, d, generator=g, dtype=torch.float64)
    qd = 2.0 * torch.randn(n, d, generator=g, dtype=torch.float64)
    qdd = 20.0 * torch.randn(n, d, generator=g, dtype=torch.float64)
    return q, qd, qdd


def test_cache_exists_in_tree():
    """It is a committed artifact -- Isaac cannot regenerate it itself."""
    assert CACHE_PATH.is_file(), (
        f"{CACHE_PATH} missing; run isaac_rl/scripts/build_rnea_cache.py")


def test_topology_is_identical(live, cached):
    assert list(cached.parent) == list(live.parent)
    assert list(cached.body_to_q) == list(live.body_to_q)
    assert list(cached.body_to_v) == list(live.body_to_v)
    assert list(cached.opt2pin) == list(live.opt2pin)
    assert (cached.n_bodies, cached.nq_pin, cached.nv_pin) == (
        live.n_bodies, live.nq_pin, live.nv_pin)


def test_spatial_layout_is_carried(live, cached):
    """``spatial_layout`` is set as an ATTRIBUTE by ``from_pinocchio``, not a
    dataclass field, so a naive round-trip drops it and the inertia halves
    silently stop being swapped."""
    assert getattr(cached, "spatial_layout", None) == getattr(live, "spatial_layout", None)


def test_torques_match_to_1e6(live, cached):
    q, qd, qdd = _states()
    err = (live.rnea(q, qd, qdd) - cached.rnea(q, qd, qdd)).abs().max().item()
    assert err <= 1e-6, err


def test_gravity_only_torques_match(live, cached):
    """What the env's feedforward actually evaluates (qdd = 0)."""
    q, _qd, _qdd = _states(seed=3)
    z = torch.zeros_like(q)
    err = (live.rnea(q, z, z) - cached.rnea(q, z, z)).abs().max().item()
    assert err <= 1e-6, err


def test_armature_survives_the_round_trip(live, cached):
    assert (cached.armature is None) == (live.armature is None)
    if live.armature is not None:
        assert torch.allclose(cached.armature, live.armature, atol=0, rtol=0)
        # the armature is the whole point of the vendored RNEA (27x
        # underestimate at wrist_roll without it) -- it must be nonzero
        assert float(cached.armature.abs().sum()) > 0.0


def test_stale_cache_is_refused(tmp_path):
    """A stale tree must raise, not be used: wrong physics that raises nowhere
    is the same failure class as the sim.device default that cost RL-1 hours."""
    p = tmp_path / "rnea.npz"
    build_cache(p)
    d = dict(np.load(p, allow_pickle=False))
    d["fingerprint"] = np.asarray("not-the-right-urdf")
    np.savez_compressed(p, **d)
    with pytest.raises(RuntimeError, match="stale RNEA cache"):
        load_cached(p, device="cpu", dtype=torch.float64)
    # ...unless the caller explicitly opts out
    assert load_cached(p, device="cpu", dtype=torch.float64, check=False) is not None


def test_fingerprint_tracks_the_joint_set():
    a = cache_fingerprint()
    b = cache_fingerprint(joint_names=A.G1_UPPER_BODY_JOINTS[:-1])
    assert a != b


def test_build_or_load_prefers_the_live_build(live):
    """Outside Isaac the live path must win, so a stale cache cannot poison a
    normal process."""
    r = build_or_load(device="cpu", dtype=torch.float64, verbose=False)
    assert r is not None
    q, qd, qdd = _states(n=64, seed=7)
    assert torch.allclose(r.rnea(q, qd, qdd), live.rnea(q, qd, qdd), atol=1e-9)


def test_cached_rnea_drives_fused_cost():
    """The point of the fix: FusedCost must be constructible from the cache."""
    rnea = load_cached(device="cpu", dtype=torch.float32)
    fk = A.ChainFK(device="cpu", dtype=torch.float32)
    # FusedCost builds its own SDF from the scene; it takes fk + rnea only.
    cost = A.FusedCost(A.CostConfig(), fk=fk, rnea=rnea,
                       max_horizon=12, device="cpu", dtype=torch.float32)
    assert cost is not None


def test_fused_cost_evaluates_end_to_end_from_the_cache():
    """Constructing FusedCost is not enough -- the posture term reads
    ``frames["left_shoulder"]``, which the DEFAULT ChainFK frame set does not
    carry, so handing it a shared ``fk`` raises ``KeyError: 'left_shoulder'``
    only once a cost is actually evaluated.  This pins the whole path.
    """
    import torch as T

    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    H, B = 12, 2
    rnea = load_cached(device="cpu", dtype=T.float32)
    cost = A.FusedCost(A.CostConfig(), rnea=rnea, max_horizon=H,
                       device="cpu", dtype=T.float32)
    kit = A.DrumKit.from_layout()
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    sched = BatchedSchedule(B, ScheduleConfig(horizon=H), device="cpu")
    sched.reset(T.arange(B), clips * B, [kit] * B, rng=None)
    ctx = sched.strike_context()

    from amortized_opt.train_cost import group_lengths
    q0 = T.zeros(B, 17)
    traj = q0.unsqueeze(1).expand(B, H, 17).contiguous()
    out = cost(q0, traj, T.zeros_like(q0), ctx,
               lengths=group_lengths(ctx.hit_index).float())
    assert T.isfinite(out.total).all()
    assert out.total.shape == (B,)
    for k, v in out.terms.items():
        assert T.isfinite(v).all(), k
