"""The batched scheduler must agree with ``MPCGenerator``, tick for tick.

``isaac_rl.schedule`` is a GPU rewrite of the host-side snapshot path that
``amortized_opt.mpc_gen.MPCGenerator`` runs once per env per tick.  A rewrite
that is *nearly* right is the worst outcome: the model would be handed contexts
one tick or one drum off from the ones it was trained on and nothing would
raise.  So the tests here are conformance tests against the original, not
property tests of the rewrite.

Four semantics are pinned, each of which the original documents as having cost
something to get right:

1. the snapshot queries the clip STRICTLY FORWARD of the instant (``SNAP_EPS``);
2. ``hit_index`` is ``max(round(tth/dt), 1)``, ``-1`` past the horizon cap;
3. ``goal_snapshot(horizon_cap=...)`` leaves an out-of-range hit as ``future[0]``
   rather than dropping it;
4. ``hit_lead_s`` shifts the QUERY, so a hit fires exactly once (the first
   implementation post-shifted ``time_to_hit`` and every hit fired
   ``lead/dt + 1`` times -- 110 recorded strikes on three clips containing 55).
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
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator  # noqa: E402
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig, clip_hit_table  # noqa: E402

DT = A.CONTROL_DT
H = 50


@pytest.fixture(scope="module")
def kit():
    return A.DrumKit.from_layout()


@pytest.fixture(scope="module")
def clips(kit):
    return A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:2]


def _sched(kit, clips, n, *, hit_lead=0.0, cap=None):
    cfg = ScheduleConfig(horizon=H, num_drums=8, num_future=3,
                         hit_lead_s=hit_lead, horizon_cap_wp=cap)
    s = BatchedSchedule(n, cfg, device="cpu")
    s.reset(torch.arange(n), [clips[i % len(clips)] for i in range(n)],
            [kit] * n, rng=None)
    return s


# ------------------------------------------------ conformance vs MPCGenerator --
def test_hit_index_matches_mpc_generator(kit, clips):
    """The batched ``hit_index`` must equal ``MPCGenerator._hit_index`` at every tick."""
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=H, hit_radius_frac=0.0))
    s = _sched(kit, clips, 2)
    for step in range(120):
        t_now = step * DT
        for b, clip in enumerate(clips[:2]):
            snap = gen._snapshot(clip, t_now, np.random.default_rng(0))
            want = gen._hit_index(snap)
            got = s.hit_index()[b]
            assert int(got[0]) == want["left"], (step, b, "left", int(got[0]), want)
            assert int(got[1]) == want["right"], (step, b, "right", int(got[1]), want)
        s.advance()


def test_goal_tth_matches_mpc_generator(kit, clips):
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=H, hit_radius_frac=0.0))
    s = _sched(kit, clips, 2)
    for step in range(60):
        ctx = s.context()
        for b, clip in enumerate(clips[:2]):
            snap = gen._snapshot(clip, step * DT, np.random.default_rng(0))
            for hi, hand in enumerate(("left", "right")):
                g = snap.goal[hand]
                assert bool(ctx.goal_valid[b, hi]) == bool(g.active)
                if g.active:
                    assert float(ctx.goal_tth[b, hi]) == pytest.approx(
                        g.time_to_hit, abs=1e-5)
        s.advance()


def test_future_tokens_match(kit, clips):
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=H, hit_radius_frac=0.0))
    s = _sched(kit, clips, 1)
    for step in range(40):
        ctx = s.context()
        snap = gen._snapshot(clips[0], step * DT, np.random.default_rng(0))
        for hi, hand in enumerate(("left", "right")):
            for j, fut in enumerate(snap.future[hand][:3]):
                assert bool(ctx.future_valid[0, hi, j]) == bool(fut.active)
                if fut.active:
                    assert float(ctx.future_tth[0, hi, j]) == pytest.approx(
                        fut.time_to_hit, abs=1e-5)
                    assert int(ctx.future_drum[0, hi, j]) == int(fut.drum)
        s.advance()


# ------------------------------------------------------------- exactly once --
@pytest.mark.parametrize("lead", [0.0, 0.005, 0.01, 0.015, 0.02, 0.025])
def test_every_hit_fires_exactly_once(kit, clips, lead):
    """The regression the original paid for: post-shifting fires each hit
    ``lead/dt + 1`` times.  Shifting the QUERY fires it once, at any lead."""
    clip = clips[0]
    s = _sched(kit, [clip], 1, hit_lead=lead)
    steps = int(np.ceil(clip.duration / DT)) + 10
    fired = torch.zeros(2)
    for _ in range(steps):
        fired += s.advance().float()[0]
    # A hit at exactly t_start is already consumed by the strictly-forward
    # query (``MPCGenerator.SNAP_EPS``), so it never fires -- and these clips
    # do open on a downbeat.  Everything after it must fire exactly once.
    n_left = sum(1 for h in clip.hits if h.hand == "left" and h.time > 0.0)
    n_right = sum(1 for h in clip.hits if h.hand == "right" and h.time > 0.0)
    assert int(fired[0]) == n_left
    assert int(fired[1]) == n_right


def test_fire_tick_leads_the_schedule_by_the_lead(kit, clips):
    """With ``hit_lead_s``, a hand's fire tick is ``round(lead/dt)`` ticks early."""
    clip = clips[0]
    lead = 0.02
    first = {h: min((x.time for x in clip.hits if x.hand == h), default=None)
             for h in ("left", "right")}
    for hand, hi in (("left", 0), ("right", 1)):
        if first[hand] is None:
            continue
        s0 = _sched(kit, [clip], 1, hit_lead=0.0)
        sl = _sched(kit, [clip], 1, hit_lead=lead)
        t0 = tl = -1
        for k in range(200):
            if t0 < 0 and bool(s0.advance()[0, hi]):
                t0 = k
            if tl < 0 and bool(sl.advance()[0, hi]):
                tl = k
            if t0 >= 0 and tl >= 0:
                break
        assert t0 - tl == pytest.approx(round(lead / DT), abs=1)


# --------------------------------------------------------------- horizon cap --
def test_out_of_range_goal_becomes_future_zero(kit, clips):
    """``goal_snapshot(horizon_cap=...)`` semantics: not dropped, demoted."""
    s = _sched(kit, clips, 1, cap=3)      # 3 waypoints = 60 ms: almost never in range
    ctx = s.context()
    hidx = s.hit_index()
    for hi in (0, 1):
        if int(hidx[0, hi]) == -1:
            # the hit still exists, it is just not the goal yet
            assert bool(ctx.future_valid[0, hi, 0]) or not bool(s.hit_valid[0, hi].any())
            assert not bool(ctx.goal_valid[0, hi])


def test_strictly_forward_query_drops_a_hit_at_t_zero(kit):
    """A hit at exactly ``t_now`` has already been executed by the tick that
    reached it -- ``MPCGenerator.SNAP_EPS``."""
    from amortized_opt.rhythm import Hit, RhythmClip
    clip = RhythmClip(name="t0", duration=2.0, hits=[
        Hit(time=0.0, raw_time=0.0, drum=0, drum_name=kit.names[0], hand="left",
            velocity=1.0),
        Hit(time=0.5, raw_time=0.5, drum=0, drum_name=kit.names[0], hand="left",
            velocity=1.0),
    ])
    s = _sched(kit, [clip], 1)
    assert int(s.cursor[0, 0]) == 1, "the t=0 hit must already be consumed"
    assert float(s.context().goal_tth[0, 0]) == pytest.approx(0.5, abs=1e-4)


# ------------------------------------------------------------ strike context --
def test_strike_context_matches_the_numpy_row(kit, clips):
    """``strike_context`` must equal ``phase2_data._strike_row``, field by field."""
    from amortized_opt.phase2_data import _strike_row
    s = _sched(kit, clips, 1)
    sc = s.strike_context()
    ctx = s.context()
    for hi in (0, 1):
        if not bool(ctx.goal_valid[0, hi]):
            continue
        row = _strike_row(kit, int(ctx.goal_drum[0, hi]),
                          ctx.goal_pos[0, hi].numpy(), ctx.goal_vel[0, hi].numpy())
        assert sc.p_star[0, hi].numpy() == pytest.approx(row["p_star"], abs=1e-6)
        assert sc.n_out[0, hi].numpy() == pytest.approx(row["n_out"], abs=1e-6)
        assert float(sc.hit_radius[0, hi]) == pytest.approx(row["hit_radius"], abs=1e-6)
        assert sc.disk_offset[0, hi].numpy() == pytest.approx(row["disk_off"], abs=1e-6)
        assert sc.half_plane_dir[0, hi].numpy() == pytest.approx(row["hp_dir"], abs=1e-6)
        assert float(sc.has_half_plane[0, hi]) == pytest.approx(row["has_hp"])
        assert float(sc.contact_sigma[0, hi]) == pytest.approx(row["sigma"])
        assert float(sc.shaft_allowed[0, hi]) == pytest.approx(row["shaft"])
        assert float(sc.elev_required[0, hi]) == pytest.approx(row["elev"])


def test_departure_exemption_runs_after_a_strike(kit, clips):
    """The hand that just struck must be exempt from colliding with that head."""
    s = _sched(kit, clips, 1)
    for _ in range(200):
        fired = s.advance()
        if bool(fired.any()):
            hi = int(fired[0].long().argmax())
            sc = s.strike_context()
            assert int(sc.start_drum[0, hi]) >= 0
            return
    pytest.skip("no strike inside the probe window")


# ----------------------------------------------------------- sticky aim point --
def test_aim_point_is_sticky(kit, clips):
    """The aim is drawn ONCE per scheduled hit (7c: the eval/deployment mismatch
    where one hit's target teleported by up to 0.2 m every 20 ms)."""
    rng = np.random.default_rng(0)
    tab = clip_hit_table(clips[0], kit, rng, cfg=ScheduleConfig(horizon=H))
    s = BatchedSchedule(1, ScheduleConfig(horizon=H), device="cpu")
    s.reset(torch.tensor([0]), [clips[0]], [kit], rng=np.random.default_rng(0))
    seen = []
    for _ in range(30):
        seen.append(s.context().goal_pos[0, 0].clone())
        if bool(s.advance()[0, 0]):
            break
    stacked = torch.stack(seen)
    assert torch.allclose(stacked, stacked[0].expand_as(stacked), atol=1e-9)
    assert tab["hit_valid"].any()
