"""The three section-7c closed-loop knobs on ``MPCGenerator``.

* ``sticky_goal_point`` -- one scheduled hit keeps ONE aim point;
* ``hit_lead_s``        -- the lag compensation, and the double-fire guard;
* ``freeze_wp`` / ``freeze_waist`` -- the plan doc's "stop refining ~100 ms out".

All three default to the values section 7c measured, so these tests are also the
regression fence on those defaults: each one asserts the OFF behaviour too, so a
future change that silently flips a default fails here rather than in a two-hour
closed-loop eval.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator
from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from amortized_opt.warmstart import arm_slice

N = len(G1_UPPER_BODY_JOINTS)


@pytest.fixture(scope="module")
def kit():
    return DrumKit.from_layout()


def _clip(kit: DrumKit, times_left=(0.30, 0.80), times_right=(0.44, 0.96)):
    """A tiny two-hand clip on drums both hands can reach."""
    li, ri = kit.index("hihat"), kit.index("snare")
    hits = [Hit(time=float(t), raw_time=float(t), drum=li, drum_name="hihat",
                hand="left", velocity=0.7) for t in times_left]
    hits += [Hit(time=float(t), raw_time=float(t), drum=ri, drum_name="snare",
                 hand="right", velocity=0.7) for t in times_right]
    hits.sort(key=lambda h: h.time)
    return RhythmClip(name="unit", hits=hits, duration=1.6)


def _gen(kit, **cfg):
    base = dict(horizon=25, randomize_gains=False, record_min_sdf=False)
    base.update(cfg)
    return MPCGenerator(kit=kit, config=MPCGenConfig(**base), dtype=torch.float64)


def _start(gen):
    q = np.zeros(N)
    return q, np.zeros(N)


# ----------------------------------------------------------- sticky aim point --
def test_sticky_goal_point_holds_one_aim_per_hit(kit):
    """The same scheduled hit must not be re-jittered every tick."""
    clip = _clip(kit)
    seen = {True: {}, False: {}}
    for sticky in (True, False):
        gen = _gen(kit, sticky_goal_point=sticky)
        rng = np.random.default_rng(0)
        rows = []
        for step in range(20):
            snap = gen._snapshot(clip, step * CONTROL_DT, rng)
            g = snap.goal["left"]
            if g.active:
                rows.append((round(snap.t_now + g.time_to_hit, 6),
                             np.asarray(g.position, float).copy()))
        # group by the absolute hit time
        by_t: dict[float, list] = {}
        for t, p in rows:
            by_t.setdefault(t, []).append(p)
        seen[sticky] = by_t

    # sticky: every draw of one hit is bit-identical
    for t, ps in seen[True].items():
        assert len(ps) >= 2, "the test needs a hit that is queried on several ticks"
        for p in ps[1:]:
            assert np.array_equal(p, ps[0]), f"aim moved for hit at t={t}"
    # not sticky: it moves, by a lot (that is the bug this closes)
    spread = max(
        float(np.linalg.norm(np.asarray(ps) - np.asarray(ps[0]), axis=1).max())
        for ps in seen[False].values() if len(ps) >= 2
    )
    assert spread > 0.01, f"expected the un-sticky path to jitter, got {spread:.4f} m"


# ------------------------------------------------------------------- the lead --
def test_hit_lead_shifts_the_terminal_by_whole_ticks(kit):
    """A lead of k ticks makes the planner's horizon k waypoints shorter."""
    clip = _clip(kit)
    rng = np.random.default_rng(0)
    k0 = _gen(kit, hit_lead_s=0.0)._hit_index(
        _gen(kit, hit_lead_s=0.0)._snapshot(clip, 0.0, np.random.default_rng(0))
    )
    g = _gen(kit, hit_lead_s=2 * CONTROL_DT)
    k2 = g._hit_index(g._snapshot(clip, 0.0, rng))
    for hand in ("left", "right"):
        assert k0[hand] - k2[hand] == 2, (hand, k0, k2)


def test_hit_lead_does_not_double_fire(kit):
    """Every scheduled hit must be executed exactly once, at any lead.

    Without the ``_consumed`` guard a lead that lands inside the final half tick
    hands the same hit back on the next tick: ``_hit_index`` pulls anything under
    half a period up to ``k = 1``, so it fires again.  Measured before the fix at
    ``hit_lead_s = 0.01``: 110 recorded strikes on three clips holding 55.
    """
    clip = _clip(kit)
    n_sched = len(clip.hits)
    for lead in (0.0, 0.005, 0.01, 0.02, 0.03, 0.04):
        gen = _gen(kit, hit_lead_s=lead)
        q0, qd0 = _start(gen)
        res = gen.run_episode(clip, q0, qd0, seed=0, max_steps=100)
        fired = [(h, d) for (_s, h, d) in res.hit_steps]
        assert len(fired) == n_sched, (
            f"lead={lead}: {len(fired)} strikes for {n_sched} scheduled hits "
            f"({fired})"
        )
        assert len(set(res.hit_steps)) == len(res.hit_steps)


def test_hit_lead_fires_earlier_in_wall_clock(kit):
    clip = _clip(kit)
    gen0 = _gen(kit, hit_lead_s=0.0)
    gen2 = _gen(kit, hit_lead_s=2 * CONTROL_DT)
    a = gen0.run_episode(clip, *_start(gen0), seed=0, max_steps=100).hit_steps
    b = gen2.run_episode(clip, *_start(gen2), seed=0, max_steps=100).hit_steps
    assert len(a) == len(b)
    for (ka, ha, da), (kb, hb, db) in zip(a, b):
        assert (ha, da) == (hb, db)
        assert kb == ka - 2, (a, b)


# ------------------------------------------------------------------ the freeze --
def test_freeze_holds_the_striking_arm_and_replans_the_rest(kit):
    """Inside the window, that arm's command is the previous plan, shifted.

    The other arm and (with ``freeze_waist=False``) the waist must keep taking
    the planner's fresh output, so the freeze cannot be implemented by simply
    ignoring the planner.
    """
    clip = _clip(kit)
    calls: list[dict] = []

    def planner(req):
        # a planner that returns something DIFFERENT every tick, so "the frozen
        # arm followed the old plan" is unambiguous
        out = np.full((req.horizon, N), float(req.step) * 0.002)
        calls.append({"step": req.step, "hit": dict(req.hit_index),
                      "init": np.asarray(req.init_traj, float).copy(),
                      "traj": out.copy()})
        return out

    gen = _gen(kit, freeze_wp=5, freeze_waist=False)
    res = gen.run_episode(clip, *_start(gen), seed=0, max_steps=60, planner=planner)
    assert res.frozen_ticks > 0, "the window never opened -- the test clip is wrong"

    waist = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("waist_")]
    n_checked = 0
    for c, cmd in zip(calls, res.cmd_hist):
        frozen_hands = [h for h in ("left", "right")
                        if 1 <= c["hit"].get(h, -1) <= 5]
        if not frozen_hands or c["step"] == 0:
            continue
        n_checked += 1
        for h in frozen_hands:
            cols = list(arm_slice(h, tuple(G1_UPPER_BODY_JOINTS)))
            # the executed command is the SHIFTED PREVIOUS plan on that arm, not
            # the planner's fresh row (up to the slew/box clamp, which is inert
            # here because the values are tiny)
            assert np.allclose(cmd[cols], c["init"][0, cols], atol=1e-9), c["step"]
            assert not np.allclose(cmd[cols], c["traj"][0, cols]), c["step"]
        # the waist still takes the fresh plan
        assert np.allclose(cmd[waist], c["traj"][0, waist], atol=1e-9), c["step"]
    assert n_checked >= 3


def test_freeze_waist_also_holds_the_waist(kit):
    clip = _clip(kit)
    calls: list[dict] = []

    def planner(req):
        out = np.full((req.horizon, N), float(req.step) * 0.002)
        calls.append({"step": req.step, "hit": dict(req.hit_index),
                      "init": np.asarray(req.init_traj, float).copy(),
                      "traj": out.copy()})
        return out

    gen = _gen(kit, freeze_wp=5, freeze_waist=True)
    res = gen.run_episode(clip, *_start(gen), seed=0, max_steps=60, planner=planner)
    waist = [i for i, n in enumerate(G1_UPPER_BODY_JOINTS) if n.startswith("waist_")]
    n_checked = 0
    for c, cmd in zip(calls, res.cmd_hist):
        if c["step"] == 0 or not any(
            1 <= c["hit"].get(h, -1) <= 5 for h in ("left", "right")
        ):
            continue
        n_checked += 1
        assert np.allclose(cmd[waist], c["init"][0, waist], atol=1e-9), c["step"]
    assert n_checked >= 3


def test_freeze_off_is_bit_identical_to_the_old_loop(kit):
    """``freeze_wp = 0`` must not perturb a single command."""
    clip = _clip(kit)

    def make():
        seq = []

        def planner(req):
            out = np.full((req.horizon, N), 0.0005 * float(req.step))
            seq.append(out.copy())
            return out

        return planner, seq

    p1, _ = make()
    g1 = _gen(kit, freeze_wp=0)
    a = g1.run_episode(clip, *_start(g1), seed=3, max_steps=50, planner=p1)
    p2, _ = make()
    g2 = _gen(kit, freeze_wp=0, freeze_waist=True)
    b = g2.run_episode(clip, *_start(g2), seed=3, max_steps=50, planner=p2)
    assert np.array_equal(a.cmd_hist, b.cmd_hist)
    assert a.frozen_ticks == 0 and b.frozen_ticks == 0


def test_planner_is_still_called_every_tick_under_freeze(kit):
    """The freeze replaces columns of the answer, it does not skip the solve.

    The plan doc uses the window to "reset our context and generate the
    initialization for an immediate trajectory generation after the hit", which
    only works if the loop keeps running.
    """
    clip = _clip(kit)
    n_calls = [0]

    def planner(req):
        n_calls[0] += 1
        return np.zeros((req.horizon, N))

    gen = _gen(kit, freeze_wp=5, freeze_waist=True)
    res = gen.run_episode(clip, *_start(gen), seed=0, max_steps=40, planner=planner)
    assert n_calls[0] == res.cmd_hist.shape[0] == 40
    assert res.frozen_ticks > 0


def test_bounce_off_by_default(kit):
    clip = _clip(kit)
    gen = _gen(kit)
    res = gen.run_episode(clip, *_start(gen), seed=3, max_steps=80)
    assert len(res.hit_steps) > 0
    assert res.bounced_hands == 0


def test_bounce_counts_each_fire(kit):
    clip = _clip(kit)
    gen = _gen(kit, bounce=True)
    res = gen.run_episode(clip, *_start(gen), seed=3, max_steps=80)
    assert res.bounced_hands == len(res.hit_steps) >= 1


def test_bounce_fired_reflects_incoming_tip(kit):
    """The wiring is apply_bounce: incoming tip-normal speed reverses, q stays."""
    from amortized_opt.ik import tip_jacobian
    from amortized_opt.train_cost import BounceConfig

    bcfg = BounceConfig(
        restitution=(0.5, 0.5), tip_vel_noise=0.0, q_noise=0.0, qd_noise=0.0,
    )
    gen = _gen(kit, bounce=bcfg)
    rng = np.random.default_rng(0)
    snap = gen._snapshot(_clip(kit), 0.20, rng)
    assert snap.goal["left"].active
    drum = int(snap.goal["left"].drum)
    q = np.zeros(N)
    n = np.asarray(kit.axes[drum], float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    J = tip_jacobian(gen.fk, gen._t(q)[None], "left")[0].detach().cpu().numpy()
    qd, *_ = np.linalg.lstsq(J, -2.0 * n, rcond=None)
    vn0 = float((J @ qd) @ n)
    assert vn0 < -1.0
    q2, qd2 = gen._bounce_fired(q, qd, ["left"], snap, rng)
    J2 = tip_jacobian(gen.fk, gen._t(q2)[None], "left")[0].detach().cpu().numpy()
    vn1 = float((J2 @ qd2) @ n)
    assert vn1 > 0.0
    np.testing.assert_allclose(q2, q, atol=1e-9)
