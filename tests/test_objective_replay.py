"""Bit-stable RNG snapshot/restore + isolated-scope leak tests for objective replay."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from amortized_opt.objective_replay import (
    FrozenControls,
    ObjectiveReplayContext,
    RealizedSamples,
    ReplayIncompleteError,
    restore_global_torch_rng,
    restore_mutable_batch_fields,
    restore_numpy_generator,
    restore_python_random,
    restore_torch_generator,
    snapshot_global_torch_rng,
    snapshot_mutable_batch_fields,
    snapshot_numpy_generator,
    snapshot_python_random,
    snapshot_torch_generator,
)


def _cpu_gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


# ------------------------------------------------ torch.Generator bit-stable --


def test_torch_generator_snapshot_restore_bit_stable():
    g = _cpu_gen(123)
    snap = snapshot_torch_generator(g)
    a = torch.rand(8, generator=g)
    # Advance further so a naive "same seed" would not match.
    _ = torch.rand(8, generator=g)
    restore_torch_generator(g, snap)
    b = torch.rand(8, generator=g)
    assert torch.equal(a, b)


def test_torch_generator_snapshot_is_detached_from_live_state():
    """Cloned get_state() must not track later set_state / draws on ``g``."""
    g = _cpu_gen(7)
    snap = snapshot_torch_generator(g)
    _ = torch.rand(4, generator=g)
    # Mutate live generator to a different seed; snap must still restore original.
    g.manual_seed(999)
    restore_torch_generator(g, snap)
    # Re-seeded path would differ; snap path matches the first-draw sequence.
    g2 = _cpu_gen(7)
    assert torch.equal(torch.rand(4, generator=g), torch.rand(4, generator=g2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_generator_on_device_bit_stable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    g = torch.Generator(device=device)
    g.manual_seed(42)
    snap = snapshot_torch_generator(g)
    a = torch.rand(5, device=device, generator=g)
    restore_torch_generator(g, snap)
    b = torch.rand(5, device=device, generator=g)
    assert torch.equal(a, b)


# --------------------------------------------- numpy / python Random -----------


def test_numpy_generator_snapshot_restore_bit_stable():
    rng = np.random.default_rng(31)
    snap = snapshot_numpy_generator(rng)
    a = rng.random(5)
    _ = rng.integers(0, 100, size=10)
    restore_numpy_generator(rng, snap)
    b = rng.random(5)
    assert np.array_equal(a, b)


def test_python_random_snapshot_restore_bit_stable():
    rng = random.Random(9)
    snap = snapshot_python_random(rng)
    a = [rng.random() for _ in range(4)]
    _ = [rng.random() for _ in range(4)]
    restore_python_random(rng, snap)
    b = [rng.random() for _ in range(4)]
    assert a == b


# -------------------------------------------------------- global torch RNG ----


def test_global_cpu_rng_snapshot_restore_bit_stable():
    torch.manual_seed(0)
    snap = snapshot_global_torch_rng(capture_cuda=False)
    a = torch.rand(6)
    _ = torch.rand(6)
    restore_global_torch_rng(snap)
    b = torch.rand(6)
    assert torch.equal(a, b)


def test_global_cuda_rng_snapshot_restore_bit_stable():
    if not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    torch.cuda.manual_seed_all(11)
    snap = snapshot_global_torch_rng(capture_cuda=True)
    a = torch.rand(4, device="cuda")
    _ = torch.rand(4, device="cuda")
    restore_global_torch_rng(snap)
    b = torch.rand(4, device="cuda")
    assert torch.equal(a, b)


# --------------------------------------------- isolated scope does not leak --


def test_run_isolated_does_not_leak_named_or_global_draws():
    gen = _cpu_gen(1)
    sfx = _cpu_gen(2)
    jit = _cpu_gen(3)
    rng = np.random.default_rng(4)
    torch.manual_seed(5)

    mgr = ObjectiveReplayContext(
        {"gen": gen, "sfx_gen": sfx, "jit_gen": jit, "rng": rng},
        capture_global=True,
    )
    rec = mgr.begin_record(controls=FrozenControls(step=0))
    # Simulate baseline draws that should stick.
    _ = torch.rand(3, generator=gen)
    _ = torch.rand(3, generator=sfx)
    _ = float(rng.random())
    _ = torch.rand(3)
    mgr.finalize_baseline(rec)

    # Ground-truth post-baseline positions (probes must restore here).
    post_gen = snapshot_torch_generator(gen)
    post_sfx = snapshot_torch_generator(sfx)
    post_jit = snapshot_torch_generator(jit)
    post_np = snapshot_numpy_generator(rng)
    post_global = snapshot_global_torch_rng(capture_cuda=False)

    with mgr.run_isolated(rec):
        for _ in range(20):
            _ = torch.rand(8, generator=gen)
            _ = torch.rand(8, generator=sfx)
            _ = torch.rand(8, generator=jit)
            _ = rng.random()
            _ = torch.rand(8)

    assert torch.equal(snapshot_torch_generator(gen), post_gen)
    assert torch.equal(snapshot_torch_generator(sfx), post_sfx)
    assert torch.equal(snapshot_torch_generator(jit), post_jit)
    assert snapshot_numpy_generator(rng) == post_np

    # Global continuity: next draw equals a twin restored to post_global.
    twin_state = post_global
    got = torch.rand(3)
    restore_global_torch_rng(twin_state)
    expect = torch.rand(3)
    assert torch.equal(got, expect)


def test_run_isolated_restore_pre_then_returns_to_post():
    gen = _cpu_gen(10)
    mgr = ObjectiveReplayContext({"gen": gen}, capture_global=False)
    rec = mgr.begin_record()
    pre = snapshot_torch_generator(gen)
    assert torch.equal(rec.rng_pre.named["gen"], pre)

    drawn_at_baseline = torch.rand(4, generator=gen)
    mgr.finalize_baseline(rec)
    post = snapshot_torch_generator(gen)

    with mgr.run_isolated(rec, restore_pre=True):
        again = torch.rand(4, generator=gen)
        assert torch.equal(again, drawn_at_baseline)
        # Extra probe draws...
        _ = torch.rand(100, generator=gen)

    # Back to post-baseline, not pre and not probe-advanced.
    assert torch.equal(snapshot_torch_generator(gen), post)
    # Continuing from post matches a twin restored to post.
    twin = _cpu_gen(0)
    restore_torch_generator(twin, post)
    assert torch.equal(torch.rand(4, generator=gen), torch.rand(4, generator=twin))


def test_run_isolated_requires_finalize_by_default():
    gen = _cpu_gen(1)
    mgr = ObjectiveReplayContext({"gen": gen}, capture_global=False)
    rec = mgr.begin_record()
    with pytest.raises(ReplayIncompleteError):
        with mgr.run_isolated(rec):
            pass
    with mgr.run_isolated(rec, require_finalized=False):
        _ = torch.rand(2, generator=gen)


def test_run_isolated_fn_returns_value_without_leak():
    gen = _cpu_gen(3)
    mgr = ObjectiveReplayContext({"gen": gen}, capture_global=False)
    rec = mgr.begin_record()
    mgr.finalize_baseline(rec)
    before = snapshot_torch_generator(gen)

    def probe():
        return float(torch.rand((), generator=gen))

    val = mgr.run_isolated_fn(probe, rec)
    assert isinstance(val, float)
    assert torch.equal(snapshot_torch_generator(gen), before)


# --------------------------------------------- multi-stream joint restore -----


def test_multi_stream_begin_finalize_roundtrip():
    streams = {
        "gen": _cpu_gen(1),
        "sfx_gen": _cpu_gen(2),
        "jit_gen": _cpu_gen(3),
        "rng": np.random.default_rng(4),
        "py": random.Random(5),
    }
    mgr = ObjectiveReplayContext(streams, capture_global=True)
    rec = mgr.begin_record(controls=FrozenControls(include_bptt=True, sfx_ramp=0.5))
    # Consume each stream once during "baseline".
    torch.rand(2, generator=streams["gen"])
    torch.rand(2, generator=streams["sfx_gen"])
    torch.rand(2, generator=streams["jit_gen"])
    streams["rng"].random()
    streams["py"].random()
    torch.rand(2)
    mgr.finalize_baseline(rec)

    post = {k: snapshot_torch_generator(streams[k])
            for k in ("gen", "sfx_gen", "jit_gen")}
    post_np = snapshot_numpy_generator(streams["rng"])
    post_py = snapshot_python_random(streams["py"])
    post_glob = snapshot_global_torch_rng(capture_cuda=False)

    with mgr.run_isolated(rec):
        for _ in range(5):
            torch.rand(16, generator=streams["gen"])
            torch.rand(16, generator=streams["sfx_gen"])
            torch.rand(16, generator=streams["jit_gen"])
            streams["rng"].bytes(32)
            streams["py"].gauss(0.0, 1.0)
            torch.rand(16)

    for k in ("gen", "sfx_gen", "jit_gen"):
        assert torch.equal(snapshot_torch_generator(streams[k]), post[k])
    assert snapshot_numpy_generator(streams["rng"]) == post_np
    assert snapshot_python_random(streams["py"]) == post_py
    # Global: compare by draw parity from restored twin snap.
    restore_global_torch_rng(post_glob)
    a = torch.rand(3)
    restore_global_torch_rng(post_glob)
    b = torch.rand(3)
    assert torch.equal(a, b)


# --------------------------------------------- mutable batch helpers ----------


def test_mutable_batch_snapshot_restore():
    class _Ctx:
        def __init__(self):
            self.yaw_now = torch.tensor([0.1, 0.2])

    batch = {
        "init": torch.randn(2, 4, 3),
        "state": torch.randn(2, 6),
        "context": _Ctx(),
    }
    snap = snapshot_mutable_batch_fields(batch)
    batch["init"].add_(1.0)
    batch["context"].yaw_now.add_(0.5)
    restore_mutable_batch_fields(batch, snap)
    assert torch.equal(batch["init"], snap["init"])
    assert torch.equal(batch["context"].yaw_now, snap["yaw_now"])


def test_realized_samples_slot_contract():
    """Smoke: record can hold detached TC / suffix / jitter tensors."""
    rec_samples = RealizedSamples(
        tc_kp=torch.ones(2, 3),
        tc_kd=torch.zeros(2, 3),
        suffix_offs=torch.tensor([0, 1]),
        yaw_now_jitter=torch.randn(2),
    )
    assert rec_samples.tc_kp.shape == (2, 3)
    assert int(rec_samples.suffix_offs.sum()) == 1


def test_double_finalize_raises():
    mgr = ObjectiveReplayContext({"gen": _cpu_gen(0)}, capture_global=False)
    rec = mgr.begin_record()
    mgr.finalize_baseline(rec)
    with pytest.raises(ReplayIncompleteError):
        mgr.finalize_baseline(rec)
