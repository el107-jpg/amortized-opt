"""``ContextInputs.future_gap``: hit-relative future timing, everywhere.

The context encoder's future tokens used to carry ``future_tth`` -- time to each
upcoming hit measured FROM NOW.  Measured from now, it ticks down every 20 ms,
which is what forced the deployment MPC to re-encode the whole context every
control tick (``isaac_rl/README.md`` DECISION 2) -- and that per-tick re-encode
is the path by which the robot's own state echoed back into the encoder and ran
the waist away in closed loop.

``future_gap`` states the same schedule relative to the hand's CURRENT goal
strike instead.  Three things follow, and this file pins all three:

1. **the anchor rule** -- gap = future hit time - current goal strike time, with
   a fall back to "from now" for a hand whose current goal is inactive (no next
   hit, or one the horizon cap has demoted).  The fallback is what makes
   ``future_gap[..., 0]`` degrade gracefully to the old semantics, which is
   exactly how the IDLE token still reads it;
2. **time invariance** -- two contexts built at different instants of the SAME
   goal segment carry IDENTICAL future tokens.  This is the property the whole
   change exists for: it is what makes a per-segment encoder cache correct;
3. **producer parity** -- ``isaac_rl.schedule.BatchedSchedule.context`` (the
   deployment/RL path) and ``rhythm.build_context_inputs`` (the training path)
   anchor identically.  A rewrite that is *nearly* right is the worst outcome
   here, same argument as ``tests/test_isaac_rl_schedule.py``.

The fourth site, ``MPCRoller``'s re-anchoring across a per-hand re-goal, is
pinned in ``tests/test_roller_regoal.py``
(``test_regoal_reanchors_the_surviving_future_gaps``) -- it lives there because
it needs that module's scenario-pool/roller fixture, which costs two minutes to
build.
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
from amortized_opt.rhythm import (  # noqa: E402
    Hit,
    RhythmClip,
    build_context_inputs,
    goal_snapshot,
)

DT = A.CONTROL_DT
LEFT, RIGHT = 0, 1


@pytest.fixture(scope="module")
def kit():
    return A.DrumKit.from_layout()


def _clip(kit, left: list[float], right: list[float], *, duration: float = 8.0):
    """A synthetic clip with hand-chosen hit times, already on the 20 ms grid."""
    name = kit.names[0]
    hits = [
        Hit(time=float(t), raw_time=float(t), drum=0, drum_name=name, hand=hand,
            velocity=1.0)
        for hand, times in (("left", left), ("right", right))
        for t in times
    ]
    hits.sort(key=lambda h: h.time)
    return RhythmClip(name="synthetic", hits=hits, duration=duration)


def _ctx(kit, clip, t_now: float, *, horizon_cap=None, num_future: int = 3):
    snap = goal_snapshot(clip, kit, t_now, num_future=num_future, rng=None,
                         horizon_cap=horizon_cap)
    return build_context_inputs(kit, snap, num_future=num_future), snap


# ------------------------------------------------------------- 1. anchoring --
def test_gaps_are_measured_from_the_current_goal_strike(kit):
    """gap[j] = t_future[j] - t_goal, hand by hand, against hand arithmetic."""
    left = [0.30, 0.90, 1.50, 2.40]
    right = [0.50, 1.00]
    clip = _clip(kit, left, right)
    ctx, _snap = _ctx(kit, clip, 0.0)

    assert bool(ctx.goal_valid[0, LEFT]) and bool(ctx.goal_valid[0, RIGHT])
    assert float(ctx.goal_tth[0, LEFT]) == pytest.approx(0.30, abs=1e-6)

    # left: three futures, each stated as its distance from the 0.30 s strike
    for j, t in enumerate(left[1:]):
        assert bool(ctx.future_valid[0, LEFT, j])
        assert float(ctx.future_gap[0, LEFT, j]) == pytest.approx(t - 0.30, abs=1e-6)
    # right: one future, then padding that stays at exactly zero + invalid
    assert float(ctx.future_gap[0, RIGHT, 0]) == pytest.approx(1.00 - 0.50, abs=1e-6)
    assert not bool(ctx.future_valid[0, RIGHT, 1])
    assert float(ctx.future_gap[0, RIGHT, 1]) == 0.0

    # gaps are non-decreasing over the valid slots -- they are a schedule
    g = ctx.future_gap[0, LEFT]
    assert bool(torch.all(g[1:] >= g[:-1]))


def test_gaps_are_positive_after_the_strike(kit):
    """Every gap is a *wait after* the current hit, so none of them is negative.

    A future token that landed before the goal would mean the goal is not the
    next hit -- the ordering ``goal_snapshot`` guarantees, restated in the
    tokenizer's units so a regression in either shows up here.
    """
    clip = _clip(kit, [0.20, 0.40, 0.60, 1.20], [0.30, 0.34, 2.00])
    for step in range(12):
        ctx, _ = _ctx(kit, clip, step * DT)
        valid = ctx.future_valid[0]
        assert bool(torch.all(ctx.future_gap[0][valid] > 0.0))


def test_inactive_goal_falls_back_to_from_now(kit):
    """The documented fallback: no current goal, so the anchor is NOW.

    ``goal_snapshot(horizon_cap=...)`` leaves a hit that is further out than the
    horizon can represent in ``future[0]`` and marks the hand idle.  That hand
    has no strike to measure from, so ``future_gap[..., 0]`` is its plain
    time-to-hit -- which is what the IDLE token reads it as
    (``ContextEncoder.forward``), and what lets the roller's ``stale`` path
    walk the hit into range by ageing it.
    """
    clip = _clip(kit, [0.30, 0.90, 1.50], [4.00])
    ctx, _ = _ctx(kit, clip, 0.10, horizon_cap=0.15)

    assert not bool(ctx.goal_valid[0, LEFT]), "0.20 s out with a 0.15 s cap"
    assert float(ctx.goal_tth[0, LEFT]) == 0.0
    # the demoted hit is future[0], stated from now
    assert bool(ctx.future_valid[0, LEFT, 0])
    assert float(ctx.future_gap[0, LEFT, 0]) == pytest.approx(0.20, abs=1e-6)
    assert float(ctx.future_gap[0, LEFT, 1]) == pytest.approx(0.80, abs=1e-6)


# -------------------------------------------------------- 2. time invariance --
def test_future_gap_is_identical_across_a_goal_segment(kit):
    """THE invariant the rewrite exists for -- asserted exactly, not approximately.

    Both hands keep the same current goal across these ticks, so every future
    token must be bit-identical: the goal strike and the future hits slide by
    the same dt and the difference does not move.  Anything weaker (allclose)
    would let a per-tick drift back in, and a drift is precisely what forces
    the deployment MPC to re-encode every tick.
    """
    clip = _clip(kit, [0.50, 1.10, 1.70, 2.30], [0.60, 1.20, 1.80])
    base, _ = _ctx(kit, clip, 0.10)
    later, _ = _ctx(kit, clip, 0.10 + 3 * DT)

    # same segment: the goal did not change hands or drums
    assert torch.equal(base.goal_drum, later.goal_drum)
    assert torch.equal(base.goal_valid, later.goal_valid)
    # the from-now token DID move (this is the one the decoder will take over)
    assert float(later.goal_tth[0, LEFT]) < float(base.goal_tth[0, LEFT])
    # ...and the hit-relative ones did not, exactly
    assert torch.equal(base.future_gap, later.future_gap)
    assert torch.equal(base.future_valid, later.future_valid)


def test_gaps_change_only_when_the_goal_does(kit):
    """Walk a clip tick by tick: ``future_gap`` moves iff the goal segment does.

    The strong form of the invariant over a whole clip rather than one window.
    """
    clip = _clip(kit, [0.20, 0.60, 1.00, 1.60], [0.40, 0.80, 1.40])
    prev = None
    for step in range(70):
        ctx, _ = _ctx(kit, clip, step * DT)
        if prev is not None:
            for hi in (LEFT, RIGHT):
                same_goal = (
                    bool(prev.goal_valid[0, hi]) == bool(ctx.goal_valid[0, hi])
                    and float(prev.goal_tth[0, hi]) - float(ctx.goal_tth[0, hi])
                    == pytest.approx(DT, abs=1e-6)
                )
                if same_goal:
                    assert torch.equal(
                        prev.future_gap[0, hi], ctx.future_gap[0, hi]
                    ), f"step {step}, hand {hi}"
        prev = ctx


# ------------------------------------------------------------ 3. schedule parity --
def _sched(kit, clip, *, cap=None, num_future=3):
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

    cfg = ScheduleConfig(horizon=50, num_drums=8, num_future=num_future,
                         horizon_cap_wp=cap, hit_radius_frac=0.0)
    s = BatchedSchedule(1, cfg, device="cpu")
    s.reset(torch.arange(1), [clip], [kit], rng=None)
    return s


@pytest.mark.parametrize("cap", [None, 8])
def test_batched_schedule_anchors_like_build_context_inputs(kit, cap):
    """Field-by-field parity on the same clip, kit and instant.

    ``BatchedSchedule`` re-derives the tokens on the GPU instead of going
    through ``goal_snapshot``, so it carries its own copy of the anchor rule --
    including the demoted-goal case, where the out-of-range hit becomes
    ``future[0]`` and the hand's anchor has to fall back to now.
    """
    clip = _clip(kit, [0.30, 0.90, 1.50, 2.40], [0.50, 1.00, 2.20])
    s = _sched(kit, clip, cap=cap)
    # ``ScheduleConfig.cap_wp`` defaults to the full horizon, so "no cap" is
    # still a cap of ``horizon`` waypoints on the snapshot side.
    horizon_cap = (cap if cap is not None else 50) * DT
    # ``MPCGenerator.SNAP_EPS``: the schedule reads the clip strictly forward of
    # the instant, so a hit at exactly ``t_now`` is already spent.  Query the
    # snapshot the same way or the two disagree on every downbeat.
    eps = 1e-6
    for step in range(60):
        want, _ = _ctx(kit, clip, step * DT + eps, horizon_cap=horizon_cap)
        got = s.context()
        assert torch.equal(got.goal_valid, want.goal_valid), step
        assert torch.equal(got.future_valid, want.future_valid), step
        assert torch.allclose(
            got.future_gap.to(torch.float64), want.future_gap.to(torch.float64),
            atol=1e-6,
        ), (step, got.future_gap, want.future_gap)
        s.advance()


def test_schedule_gap_is_frozen_between_strikes(kit):
    """The same segment-constancy check, on the deployment producer."""
    clip = _clip(kit, [0.50, 1.10, 1.70], [0.60, 1.20])
    s = _sched(kit, clip)
    seen = [s.context().future_gap.clone()]
    for _ in range(3):
        assert not bool(s.advance().any()), "no hit fires inside this window"
        seen.append(s.context().future_gap.clone())
    for g in seen[1:]:
        assert torch.equal(g, seen[0])


def test_probe_and_mpc_gen_go_through_build_context_inputs():
    """No producer may hand-roll the anchoring (the reason it lives in one place)."""
    import inspect

    from amortized_opt import mpc_gen, probe

    for mod in (mpc_gen, probe):
        src = inspect.getsource(mod)
        assert "future_gap" not in src, (
            f"{mod.__name__} touches future_gap directly; the anchor rule has "
            "exactly one implementation, in rhythm.build_context_inputs"
        )


def test_context_fields_carry_the_rename():
    """The pool cache's schema marker is the CONTEXT_FIELDS names.

    The rename is what invalidates every pool and checkpoint written under the
    from-now semantics: same-name/new-meaning would have reloaded silently.
    """
    from amortized_opt.warmstart_data import CONTEXT_FIELDS

    assert "future_gap" in CONTEXT_FIELDS
    assert "future_tth" not in CONTEXT_FIELDS
