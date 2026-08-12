"""The RL-2 update math: segment credit, rank shaping, advantage weights, NES.

Every one of these is a place where a plausible-looking implementation trains
something *slightly* wrong and nothing raises, so each test pins a property the
design argument depends on rather than a value the implementation happens to
produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_rl.perturb import AdaptiveSigma, AdaptiveSigmaConfig  # noqa: E402
from isaac_rl.psn import (PSNConfig, SegmentBuffer, TickBuffer,  # noqa: E402
                          advantage_weights, centered_ranks, nes_direction)


# ------------------------------------------------------------ rank shaping --
def test_centered_ranks_span_and_order():
    x = torch.tensor([3.0, -1.0, 7.0, 0.0])
    u = centered_ranks(x)
    assert float(u.min()) == pytest.approx(-0.5)
    assert float(u.max()) == pytest.approx(+0.5)
    # order preserved, spacing uniform
    assert torch.equal(torch.argsort(u), torch.argsort(x))
    d = torch.sort(u).values.diff()
    assert torch.allclose(d, torch.full_like(d, 1.0 / 3.0))


def test_centered_ranks_is_scale_invariant():
    """The whole point: a +10 hit bonus must not outvote its own rank."""
    x = torch.tensor([0.1, 0.2, 0.3, 1000.0])
    y = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert torch.allclose(centered_ranks(x), centered_ranks(y))


def test_centered_ranks_degenerate():
    assert centered_ranks(torch.zeros(0)).numel() == 0
    assert float(centered_ranks(torch.tensor([5.0]))[0]) == 0.0


# ------------------------------------------------------ advantage weighting --
def test_advantage_weights_zero_below_quantile():
    r = torch.arange(20.0)
    w = advantage_weights(r, quantile=0.25, beta=1.0)
    assert int((w > 0).sum()) == 5
    # and they are the TOP five
    assert torch.equal((w > 0).nonzero().flatten(), torch.arange(15, 20))


def test_advantage_weights_kept_mean_is_one_whatever_the_quantile():
    """``w_bc`` must mean the same thing at q=0.1 and q=0.5.

    The BC term normalises by the weight sum, so what has to be stable is the
    mean over the KEPT set (which is also what makes ``weight_clip`` a
    meaningful multiple).
    """
    r = torch.randn(64)
    for q in (0.1, 0.25, 0.5, 1.0):
        w = advantage_weights(r, quantile=q, beta=2.0, weight_clip=0.0)
        kept = w[w > 0]
        assert float(kept.mean()) == pytest.approx(1.0, rel=1e-5)


def test_advantage_weights_clip_bounds_a_single_outlier():
    r = torch.cat([torch.zeros(31), torch.tensor([1e6])])
    w = advantage_weights(r, quantile=0.25, beta=10.0, weight_clip=5.0)
    # nothing may own more than ``clip`` times the kept-set mean weight
    assert float(w.max()) <= 5.0 + 1e-4


def test_advantage_weights_beta_zero_is_uniform_over_kept():
    r = torch.randn(40)
    w = advantage_weights(r, quantile=0.25, beta=0.0)
    kept = w[w > 0]
    assert torch.allclose(kept, kept[0].expand_as(kept))


# ------------------------------------------------------------------- NES ----
def test_nes_direction_recovers_the_good_epsilon():
    """Antithetic pair, one good one bad: the direction must be the good one."""
    torch.manual_seed(0)
    e = torch.randn(1, 6)
    eps = torch.cat([e, -e], dim=0)
    d = nes_direction(eps, torch.tensor([1.0, 0.0]), sigma=0.02, step=1.0)
    assert torch.nn.functional.cosine_similarity(d, e[0], dim=0) > 0.999
    d2 = nes_direction(eps, torch.tensor([0.0, 1.0]), sigma=0.02, step=1.0)
    assert torch.nn.functional.cosine_similarity(d2, e[0], dim=0) < -0.999


def test_nes_direction_rms_is_step_times_sigma():
    torch.manual_seed(1)
    eps = torch.randn(16, 32)
    r = torch.randn(16)
    for sigma, step in ((0.02, 0.5), (0.1, 1.0)):
        d = nes_direction(eps, r, sigma=sigma, step=step)
        assert float(d.square().mean().sqrt()) == pytest.approx(step * sigma, rel=1e-4)


def test_nes_direction_is_invariant_to_return_scale():
    torch.manual_seed(2)
    eps = torch.randn(8, 12)
    r = torch.randn(8)
    a = nes_direction(eps, r, sigma=0.02)
    b = nes_direction(eps, 1000 * r + 7, sigma=0.02)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------- buffers ---
def test_tick_buffer_roundtrip_and_overflow():
    tb = TickBuffer(4, device="cpu")
    lo, hi = tb.append({"x": torch.arange(2.0).view(2, 1)})
    assert (lo, hi) == (0, 2)
    tb.append({"x": torch.arange(2.0, 4.0).view(2, 1)})
    g = tb.gather(torch.tensor([0, 3]))
    assert torch.allclose(g["x"].float().flatten(), torch.tensor([0.0, 3.0]))
    with pytest.raises(RuntimeError, match="overflow"):
        tb.append({"x": torch.zeros(1, 1)})


def test_tick_buffer_preserves_integer_dtypes():
    tb = TickBuffer(2, device="cpu")
    tb.append({"i": torch.tensor([[3], [4]], dtype=torch.long),
               "f": torch.zeros(2, 1)})
    assert tb.data["i"].dtype == torch.long
    assert tb.data["f"].dtype == torch.float16


def _segbuf(n=2):
    return SegmentBuffer(n, device="cpu"), TickBuffer(64, device="cpu")


def test_segment_return_is_the_sum_of_its_own_ticks():
    sb, tb = _segbuf(2)
    eps = torch.eye(2, 5)
    for r in ([1.0, 10.0], [2.0, 20.0], [3.0, 30.0]):
        lo, _ = tb.append({"x": torch.zeros(2, 1)})
        sb.add(torch.tensor(r), lo)
    sb.close(torch.tensor([0, 1]), eps)
    rets = sorted(s.ret for s in sb.closed)
    assert rets == pytest.approx([6.0, 60.0])
    assert all(s.ticks == 3 for s in sb.closed)


def test_open_segments_are_dropped_not_truncated():
    sb, tb = _segbuf(2)
    eps = torch.eye(2, 5)
    for _ in range(3):
        lo, _ = tb.append({"x": torch.zeros(2, 1)})
        sb.add(torch.ones(2), lo)
    sb.close(torch.tensor([0]), eps)        # only env 0 fired
    assert len(sb.closed) == 1 and sb.closed[0].env == 0
    rows, w, ret, e = sb.harvest(PSNConfig(quantile=1.0))
    assert ret.numel() == 1
    assert rows.numel() == 3                # env 0's three ticks, env 1's none


def test_segment_rows_index_only_that_env():
    sb, tb = _segbuf(2)
    for t in range(3):
        lo, _ = tb.append({"x": torch.arange(2.0).view(2, 1) + 10 * t})
        sb.add(torch.ones(2), lo)
    sb.close(torch.tensor([1]), torch.eye(2, 5))
    rows = sb.closed[0].rows
    vals = tb.gather(rows)["x"].float().flatten()
    assert torch.allclose(vals, torch.tensor([1.0, 11.0, 21.0]))


def test_segment_close_records_the_epsilon_in_force():
    """The loop redraws AT the fire event -- the buffer must hold the OLD one."""
    sb, tb = _segbuf(2)
    lo, _ = tb.append({"x": torch.zeros(2, 1)})
    sb.add(torch.ones(2), lo)
    old = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sb.close(torch.tensor([0]), old)
    old[0] = 99.0                            # the redraw
    assert torch.allclose(sb.closed[0].eps, torch.tensor([1.0, 2.0]))


def test_close_skips_envs_with_no_accumulated_ticks():
    sb, _tb = _segbuf(2)
    assert sb.close(torch.tensor([0, 1]), torch.eye(2, 3)) == 0
    assert sb.closed == []


def test_harvest_weights_broadcast_to_every_tick_of_a_segment():
    sb, tb = _segbuf(2)
    for _ in range(4):
        lo, _ = tb.append({"x": torch.zeros(2, 1)})
        sb.add(torch.tensor([0.0, 5.0]), lo)
    sb.close(torch.tensor([0, 1]), torch.eye(2, 3))
    rows, w, ret, _e = sb.harvest(PSNConfig(quantile=0.5, beta=0.0))
    assert rows.numel() == 4                 # one winning segment, four ticks
    assert torch.allclose(w, w[0].expand_as(w))


def test_normalize_by_ticks_changes_the_ranking():
    """A long mediocre segment beats a short great one on raw return only."""
    cfg_raw = PSNConfig(quantile=0.5, normalize_by_ticks=False)
    cfg_pt = PSNConfig(quantile=0.5, normalize_by_ticks=True)
    for cfg, winner in ((cfg_raw, 0), (cfg_pt, 1)):
        sb, tb = _segbuf(2)
        for t in range(10):
            lo, _ = tb.append({"x": torch.zeros(2, 1)})
            # env 0: 10 ticks x 1.0 = 10 ; env 1: 2 ticks x 4.0 = 8
            sb.add(torch.tensor([1.0, 4.0 if t < 2 else 0.0]), lo)
            if t == 1:
                sb.close(torch.tensor([1]), torch.eye(2, 3))
        sb.close(torch.tensor([0]), torch.eye(2, 3))
        segs = {s.env: s for s in sb.closed}
        assert segs[0].ret == pytest.approx(10.0) and segs[0].ticks == 10
        assert segs[1].ret == pytest.approx(8.0) and segs[1].ticks == 2
        rets = torch.tensor([s.ret for s in sb.closed])
        tks = torch.tensor([float(s.ticks) for s in sb.closed])
        key = rets / tks if cfg.normalize_by_ticks else rets
        assert sb.closed[int(key.argmax())].env == winner


# ------------------------------------------------------- adaptive sigma -----
def test_adaptive_sigma_moves_toward_the_target():
    a = AdaptiveSigma(AdaptiveSigmaConfig(target_mm=3.0, adapt_rate=1.05, ema=0.0))
    for _ in range(50):
        a.observe(0.5)                       # far below target -> grow
    assert a.scale > 5.0
    b = AdaptiveSigma(AdaptiveSigmaConfig(target_mm=3.0, adapt_rate=1.05, ema=0.0))
    for _ in range(50):
        b.observe(30.0)                      # far above -> shrink
    assert b.scale < 0.2


def test_adaptive_sigma_is_clamped_and_ignores_nan():
    a = AdaptiveSigma(AdaptiveSigmaConfig(scale_min=0.5, scale_max=2.0,
                                          adapt_rate=2.0, ema=0.0, target_mm=3.0))
    for _ in range(20):
        a.observe(0.0)
    assert a.scale == pytest.approx(2.0)
    before = a.scale
    assert a.observe(float("nan")) == before
