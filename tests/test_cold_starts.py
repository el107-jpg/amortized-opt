"""Cold-start scenarios: a waist yaw drawn without looking at the goals.

``Phase2ScenarioConfig.cold_start_frac`` exists to break the collinearity
between the encoder's ``yaw_now`` token and its analytic ``aim_yaw`` token --
the identifiability hole behind the 2026-08-14 waist set-point runaway.  These
tests pin the mechanism (the yaw really is independent and really does span the
policy range), the self-consistency of the row it produces (q0, the goal
context and the aim token all describe ONE robot), and the acceptance
bookkeeping that would make a collapse of the family visible.

**What the measurement actually says, on the fresh pool (2026-08-14, n ~ 700
per cell, three seeds).**  The plan-native family was ALREADY decorrelated:
:class:`~amortized_opt.starts.StartSampler` draws the waist from
``GoalPoseLibrary.yaw_grid_rad``, which is +/-100 deg in nine steps, and the
goal comes from an unrelated instant of an unrelated clip -- so |r| between
``yaw_now`` and ``aim_yaw`` is 0.13-0.20 with cold starts OFF and 0.14-0.21
with them at 100 %.  There is no contrast to assert there, and
:func:`test_cold_yaw_is_decorrelated_from_the_aim` says so with a bound rather
than a comparison.

The collinear family is the EVAL-LIKE one, whose rest start draws the waist in
the posture box (sd 0.54 rad, |yaw| <= 0.95): from that close in, the aim law's
gate never opens, so ``aim_yaw`` is returned as ``yaw_now`` VERBATIM on 61 % of
its rows and |r| is 0.39.  That is the real contrast and
:func:`test_cold_rows_are_less_collinear_than_eval_like` pins it -- 15 % held
against 61 %, which is what a cold row is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from amortized_opt.constants import PELVIS_IN_ENV
from amortized_opt.phase2_data import (
    KIND_COLD_START,
    KIND_EVAL_LIKE,
    Phase2ScenarioConfig,
    generate_cost_pool,
)
from amortized_opt.rhythm import HANDS
from amortized_opt.waist_aim import _DEFAULT_YAW_LIMIT, WaistAimConfig, aim_yaw

SCENARIOS = 900
WORKERS = 2


def _pool(*, eval_like_frac: float, cold_start_frac: float, seed: int = 0):
    cfg = Phase2ScenarioConfig(
        horizon=100, buckets=(25, 50, 100),
        eval_like_frac=eval_like_frac, cold_start_frac=cold_start_frac,
    )
    return generate_cost_pool(SCENARIOS, seed=seed, cfg=cfg, workers=WORKERS)


def _cols(pool, *fields) -> dict[str, np.ndarray]:
    """The named columns, concatenated across horizon buckets."""
    return {
        f: np.concatenate([pool.data[b][f].detach().cpu().numpy() for b in pool.buckets])
        for f in fields
    }


@pytest.fixture(scope="module")
def pool_warm():
    """Cold starts OFF: the plan-native family on its own."""
    return _pool(eval_like_frac=0.0, cold_start_frac=0.0)


@pytest.fixture(scope="module")
def pool_cold():
    """Cold starts ONLY."""
    return _pool(eval_like_frac=0.0, cold_start_frac=1.0)


@pytest.fixture(scope="module")
def pool_mixed():
    """Half eval-like, half cold -- the two families side by side."""
    return _pool(eval_like_frac=0.5, cold_start_frac=0.5)


def _pearson(pool) -> float:
    c = _cols(pool, "yaw_now", "aim_yaw")
    return float(np.corrcoef(c["yaw_now"], c["aim_yaw"])[0, 1])


def test_cold_yaw_is_decorrelated_from_the_aim(pool_warm, pool_cold):
    """|r|(yaw_now, aim_yaw) stays low when every row is cold.

    Stated as a BOUND, not as a contrast against the warm pool, because the
    warm pool is already decorrelated (see the module docstring): the assertion
    that earns its keep is that a cold row cannot become correlated, which is
    what would happen if the family were ever rebased on the drum-departure
    starts -- there the pose IS the library's IK solution at that yaw.
    """
    r_warm, r_cold = _pearson(pool_warm), _pearson(pool_cold)
    print(f"\n|r|(yaw_now, aim_yaw): cold_start_frac=0.0 -> {r_warm:+.3f}, "
          f"cold_start_frac=1.0 -> {r_cold:+.3f}")
    assert abs(r_cold) < 0.40
    # Seed-to-seed spread of either cell is ~0.07, so this is ~3 sd of headroom
    # over "cold is no worse than the family it replaces".
    assert abs(r_cold) < abs(r_warm) + 0.25


def test_cold_yaw_spans_the_policy_range(pool_cold):
    """Wide, two-sided, and inside the aim law's own limit."""
    yaw = _cols(pool_cold, "yaw_now")["yaw_now"]
    print(f"\ncold yaw: sd={yaw.std():.3f} min={yaw.min():+.3f} max={yaw.max():+.3f}")
    assert yaw.std() > 0.5
    # Both extremes are reached, so the family is not a one-sided lean.
    assert yaw.min() < -1.4 and yaw.max() > 1.4
    # ...and never past the range the aimer is allowed to command.
    assert np.abs(yaw).max() <= _DEFAULT_YAW_LIMIT + 1e-6


def test_cold_context_is_self_consistent(pool_cold):
    """q0's waist yaw, the goal tokens and the aim token describe one robot.

    The point of the cold family is a yaw that was overridden late; if any of
    the three were computed from the pre-override pose the row would be a
    fiction.  So recompute the aim law from the STORED goal positions at the
    STORED yaw and require the stored aim token back -- including the row's
    FUTURE tokens, which the aim law reads too since 2026-08-14.
    """
    c = _cols(pool_cold, "q0", "yaw_now", "aim_yaw", "goal_pos", "goal_valid",
              "goal_tth", "future_pos", "future_gap", "future_valid", "kind")
    assert (c["kind"] == KIND_COLD_START).all()

    yaw_i = tuple(Phase2ScenarioConfig().joint_names).index("waist_yaw_joint")
    # The token is the STATE's yaw, not a free parameter beside it.
    assert np.array_equal(c["q0"][:, yaw_i], c["yaw_now"])

    cfg = WaistAimConfig()
    err = np.empty(len(c["yaw_now"]))
    # The law is FUTURE-aware (2026-08-14), so the recomputation has to feed it
    # this row's future tokens too, un-anchored back to from-now time the way
    # every producer does: ``future_gap`` is stated relative to the hand's
    # current goal strike, or to now when it has none.
    anchor = np.where(c["goal_valid"], c["goal_tth"], 0.0)            # (n, 2)
    f_now = c["future_gap"] + anchor[:, :, None]                      # (n, 2, F)
    for i in range(len(err)):
        targets = {
            h: c["goal_pos"][i, hi].astype(float) - PELVIS_IN_ENV
            for hi, h in enumerate(HANDS)
            if c["goal_valid"][i, hi]
        }
        tth = {
            h: float(c["goal_tth"][i, hi])
            for hi, h in enumerate(HANDS)
            if c["goal_valid"][i, hi]
        }
        futures = {
            h: [
                (c["future_pos"][i, hi, j].astype(float) - PELVIS_IN_ENV,
                 float(f_now[i, hi, j]))
                for j in range(c["future_valid"].shape[2])
                if c["future_valid"][i, hi, j]
            ]
            for hi, h in enumerate(HANDS)
        }
        got = aim_yaw(targets, cfg=cfg, yaw_current=float(c["yaw_now"][i]),
                      goal_tth=tth, futures=futures)
        err[i] = abs(got - float(c["aim_yaw"][i]))
    print(f"\naim recomputation error: p50={np.median(err):.2e} max={err.max():.2e}")
    # The columns are stored float32 and the law has two hard thresholds (the
    # reach gate and the deadband), so a row sitting within ~1e-6 of one could
    # legitimately flip.  Nothing has yet: the bound is on the 99.9th
    # percentile, with one row of slack for that case.
    assert np.quantile(err, 0.999) < 1e-4
    assert (err > 1e-4).sum() <= 1


def test_stats_expose_the_cold_acceptance_rate(pool_cold, pool_warm):
    """``pool.stats`` carries the family's draw and accept counts.

    A cold row has the furthest to travel, so it is the row the
    impossible-transition filter would eat first; without these two counters a
    collapse would read as a normal pool with a quiet correlation back.
    """
    s = pool_cold.stats
    assert s["cold_requested"] == SCENARIOS
    rate = s["cold"] / s["cold_requested"]
    print(f"\ncold acceptance: {s['cold']}/{s['cold_requested']} = {rate:.1%} "
          f"(|qd| reject {s['reject_velocity']}, idle {s['reject_idle']}, "
          f"start fallback {s['reject_cold_start']})")
    assert s["cold"] == s["accepted"] == len(pool_cold)
    assert rate > 0.5
    # ...and the counters stay at zero when the family is off, so a pool that
    # never asked for a cold row cannot look like one that got them.
    assert pool_warm.stats["cold_requested"] == 0
    assert pool_warm.stats["cold"] == 0


def test_cold_rows_are_less_collinear_than_eval_like(pool_mixed):
    """The contrast that is real: against the family that HAS the problem.

    The aim law returns ``yaw_current`` verbatim whenever its reach gate stays
    shut, and a row where ``aim_yaw == yaw_now`` exactly teaches the set-point
    head nothing except "follow yaw_now".  The eval-like rest start sits close
    enough to centre that the gate rarely opens; a cold start is 100 deg out,
    where it always does.
    """
    c = _cols(pool_mixed, "yaw_now", "aim_yaw", "kind")
    out = {}
    for name, k in (("eval-like", KIND_EVAL_LIKE), ("cold", KIND_COLD_START)):
        m = c["kind"] == k
        assert m.sum() > 50, f"{name}: only {m.sum()} rows"
        y, a = c["yaw_now"][m], c["aim_yaw"][m]
        out[name] = (np.mean(np.abs(a - y) < 1e-9), float(y.std()),
                     float(np.corrcoef(y, a)[0, 1]))
        print(f"\n{name:9s} n={m.sum():4d} held(aim==yaw_now)={out[name][0]:.1%} "
              f"yaw sd={out[name][1]:.3f} r={out[name][2]:+.3f}")
    held_eval, sd_eval, _ = out["eval-like"]
    held_cold, sd_cold, r_cold = out["cold"]
    assert held_cold < 0.5 * held_eval
    assert sd_cold > 1.5 * sd_eval
    assert abs(r_cold) < 0.40
