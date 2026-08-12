"""Simulated-MPC generator: horizon bookkeeping, per-hand reset, replay store."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from amortized_opt import constants as C
from amortized_opt.mpc_gen import (
    MPCGenConfig,
    MPCGenerator,
    MPCSample,
    ReplayStore,
    collate,
    quintic_planner,
)
from amortized_opt.rhythm import DrumKit, default_rcc_dir, load_rcc_pool
from amortized_opt.starts import StartSampler
from amortized_opt.warmstart import arm_slice

H = 60
STEPS = 120


@pytest.fixture(scope="module")
def rig():
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(default_rcc_dir("train"), kit)
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=H))
    starts = StartSampler(kit=kit, library=gen.library, fk=gen.fk, rnea=gen.rnea, sdf=gen.sdf)
    return kit, clips, gen, starts


@pytest.fixture(scope="module")
def episode(rig):
    _kit, clips, gen, starts = rig
    b = starts.sample(1, seed=0)
    t0 = time.time()
    res = gen.run_episode(
        clips[0], b.q[0].numpy(), b.qd[0].numpy(), seed=7, max_steps=STEPS
    )
    res_wall = time.time() - t0
    return res, res_wall


def test_end_to_end_episode_is_finite_and_populated(episode, capsys):
    res, wall = episode
    sd = np.asarray([s.min_sdf for s in res.samples])
    short = np.asarray([s.short_horizon for s in res.samples])
    with capsys.disabled():
        print(f"\n[mpc episode] {STEPS} ticks -> {len(res.samples)} samples "
              f"({float(short.mean()) * 100:.0f}% short-horizon extras), "
              f"{len(res.hit_steps)} strikes fired, {wall:.2f} s "
              f"({wall / STEPS * 1e3:.1f} ms/tick)")
        print(f"[mpc clearance] min={sd.min():.4f} m, median={np.median(sd):.4f} m, "
              f"fraction < 0 = {float((sd < 0).mean()):.3f}")
    assert len(res.samples) >= STEPS
    assert all(np.isfinite(s.init_traj).all() for s in res.samples)
    assert np.isfinite(res.q_hist).all() and np.isfinite(res.qd_hist).all()
    assert np.isfinite(sd).all()
    # Bounded, not exploding: no penetration deeper than a stick length.
    assert sd.min() > -0.2
    assert len(res.hit_steps) > 0

    # The servo must not diverge. It DOES exceed the joint speed ratings on
    # short horizons, because the quintic stand-in is unconstrained -- the
    # commanded rate is over the rating before the servo ever sees it. That is
    # the seed's known weakness and what the phase-2 cost curriculum fixes;
    # what is checked here is that the simulator reports it rather than
    # blowing up.
    from amortized_opt.limits import isaac_velocity_limits

    vl = isaac_velocity_limits()
    qd = np.abs(res.qd_hist)
    cmd_rate = np.abs(np.diff(np.vstack([res.q_hist[:1], res.cmd_hist]), axis=0)) / C.CONTROL_DT
    over = float((qd > vl[None, :]).any(axis=1).mean())
    with capsys.disabled():
        print(f"[mpc velocities] measured |qd| median={np.median(qd):.2f} "
              f"p99={np.percentile(qd, 99):.1f} max={qd.max():.1f} rad/s; "
              f"commanded rate max={cmd_rate.max():.1f}; "
              f"ticks over a joint rating={over:.3f}")
    assert qd.max() < 5.0 * float(vl.max())
    assert qd.max() < 1.5 * cmd_rate.max() + 1.0  # tracking, not resonating
    assert np.median(qd) < 5.0


def test_horizons_shrink_by_exactly_one_per_tick(episode, capsys):
    res, _ = episode
    first = {}
    for s in res.samples:
        first.setdefault(s.step, s)
    steps = sorted(first)
    shrink = 0
    resets = 0
    for a, b in zip(steps, steps[1:]):
        if b != a + 1:
            continue
        for hand in ("left", "right"):
            ka, kb = first[a].hit_index[hand], first[b].hit_index[hand]
            if ka >= 2:
                assert kb == ka - 1, f"step {a}->{b} {hand}: {ka} -> {kb}"
                shrink += 1
            elif ka == 1:
                # This tick executes the strike; the hand is re-goaled.
                assert kb == -1 or kb >= 1
                resets += 1
    with capsys.disabled():
        print(f"\n[mpc horizons] {shrink} verified single-tick decrements, "
              f"{resets} post-hit resets")
    assert shrink > 50 and resets > 0


def test_masks_are_consistent_with_the_horizons(episode):
    res, _ = episode
    for s in res.samples:
        for hand, cols in (("left", arm_slice("left")), ("right", arm_slice("right"))):
            k = s.hit_index[hand]
            other = s.hit_index["right" if hand == "left" else "left"]
            # An idle hand is predicted over the other hand's horizon, or the
            # whole grid when neither hand has a goal (tokenizer semantics).
            eff = k if k >= 0 else (other if other >= 0 else s.elem_mask.shape[0])
            m = s.elem_mask[:, cols]
            assert np.all(m[:eff] == 1.0), (hand, k, eff)
            assert np.all(m[eff:] == 0.0), (hand, k, eff)
        # Waist follows the LONGER horizon; token_mask is that same row.
        waist = [i for i, n in enumerate(C.G1_UPPER_BODY_JOINTS) if n.startswith("waist_")]
        longer = max(s.hit_index["left"], s.hit_index["right"])
        longer = longer if longer >= 0 else s.elem_mask.shape[0]
        assert np.all(s.elem_mask[:longer, waist] == 1.0)
        assert np.all(s.elem_mask[longer:, waist] == 0.0)
        assert np.array_equal(s.token_mask, s.elem_mask[:, waist[0]] > 0.5)
        # tau is time-to-hit on the fixed grid.
        t = (np.arange(1, s.elem_mask.shape[0] + 1)) * C.CONTROL_DT
        for hand, tau in (("left", s.tau_left), ("right", s.tau_right)):
            k = s.hit_index[hand]
            if k >= 0:
                assert np.allclose(tau, k * C.CONTROL_DT - t)


def test_one_hand_reset_leaves_the_other_hand_untouched(rig, capsys):
    """The plan's decoupling requirement, checked on the trajectory itself.

    The next tick's init is ``shift(planner_output)`` with the RESET hand's
    columns overwritten by a fresh seed. So the property is measured against
    the planner's own output, captured by a spy -- comparing against the
    previous tick's *init* would prove nothing here, since the quintic stand-in
    ignores its init and rebuilds every column from scratch.
    """
    _kit, clips, gen, starts = rig
    b = starts.sample(1, seed=1)
    base = quintic_planner(q_bounds=gen.q_bounds)
    outs: dict[int, np.ndarray] = {}

    def spy(req):
        traj = base(req)
        outs[req.step] = np.asarray(traj, float).copy()
        return traj

    res = gen.run_episode(
        clips[1], b.q[0].numpy(), b.qd[0].numpy(), seed=3, max_steps=STEPS, planner=spy
    )
    first: dict[int, object] = {}
    for s in res.samples:
        first.setdefault(s.step, s)

    waist = [i for i, n in enumerate(C.G1_UPPER_BODY_JOINTS) if n.startswith("waist_")]
    checked = 0
    for a in sorted(first):
        if a + 1 not in first or a not in outs:
            continue
        sa, sb = first[a], first[a + 1]
        fired = [h for h in ("left", "right") if sa.hit_index[h] == 1]
        if len(fired) != 1 or sa.hit_index["right" if fired[0] == "left" else "left"] < 2:
            continue
        held = "right" if fired[0] == "left" else "left"
        shifted = np.concatenate([outs[a][1:], outs[a][-1][None, :]], axis=0)
        cols_h, cols_r = arm_slice(held), arm_slice(fired[0])
        assert np.array_equal(sb.init_traj[:, cols_h], shifted[:, cols_h]), (
            f"step {a}: {held} arm was disturbed by {fired[0]}'s reset"
        )
        assert np.array_equal(sb.init_traj[:, waist], shifted[:, waist]), (
            f"step {a}: the waist was disturbed by a single-hand reset"
        )
        assert not np.array_equal(sb.init_traj[:, cols_r], shifted[:, cols_r]), (
            f"step {a}: {fired[0]} was NOT reseeded after its strike"
        )
        checked += 1
    with capsys.disabled():
        print(f"\n[mpc decoupling] {checked} single-hand resets verified: the other "
              f"arm's and the waist's columns are byte-identical to the shifted plan")
    assert checked > 0


def test_planner_stand_in_is_actually_called(rig):
    _kit, clips, gen, starts = rig
    calls = {"n": 0, "shapes": set()}
    base = quintic_planner(q_bounds=gen.q_bounds)

    def spy(req):
        calls["n"] += 1
        calls["shapes"].add(req.init_traj.shape)
        assert req.masks.elem_mask.shape == (1, H, C.N_DOF)
        return base(req)

    b = starts.sample(1, seed=2)
    res = gen.run_episode(
        clips[0], b.q[0].numpy(), b.qd[0].numpy(), seed=5, max_steps=20, planner=spy
    )
    assert calls["n"] == 20
    assert calls["shapes"] == {(H, C.N_DOF)}
    assert res.cmd_hist.shape == (20, C.N_DOF)


def test_episode_is_reproducible(rig):
    _kit, clips, gen, starts = rig
    b = starts.sample(1, seed=0)
    a1 = gen.run_episode(clips[0], b.q[0].numpy(), b.qd[0].numpy(), seed=11, max_steps=30)
    a2 = gen.run_episode(clips[0], b.q[0].numpy(), b.qd[0].numpy(), seed=11, max_steps=30)
    a3 = gen.run_episode(clips[0], b.q[0].numpy(), b.qd[0].numpy(), seed=12, max_steps=30)
    assert np.array_equal(a1.q_hist, a2.q_hist)
    assert all(np.array_equal(x.init_traj, y.init_traj) for x, y in zip(a1.samples, a2.samples))
    assert not np.array_equal(a1.q_hist, a3.q_hist)


def test_pd_step_actually_tracks_but_lags(episode, capsys):
    """The commanded and measured paths must differ (that is the whole point)
    while staying bounded -- the generator would be useless if either the servo
    were perfect or it diverged."""
    res, _ = episode
    err = np.abs(res.q_hist[1:] - res.cmd_hist).max(axis=1)
    step = np.abs(np.diff(np.vstack([res.q_hist[:1], res.cmd_hist]), axis=0)).max(axis=1)
    with capsys.disabled():
        print(f"\n[mpc tracking error] mean={err.mean() * 1e3:.1f} mrad, "
              f"p95={np.percentile(err, 95) * 1e3:.1f} mrad, max={err.max() * 1e3:.1f} mrad; "
              f"commanded step max={step.max() * 1e3:.0f} mrad/tick")
    assert err.mean() > 1e-4  # a perfect servo would make the generator pointless
    # The error must stay proportional to what is being asked of the servo,
    # i.e. bounded by a small multiple of the per-tick command step, and it
    # must not grow monotonically (which would be divergence, not lag).
    assert err.max() < 3.0 * step.max()
    assert np.median(err) < 0.5
    assert err[-1] < err.max()


def test_short_horizon_oversampling(rig, episode, capsys):
    _kit, _clips, gen, _starts = rig
    res, _ = episode
    per_step: dict[int, int] = {}
    for s in res.samples:
        per_step[s.step] = per_step.get(s.step, 0) + 1
    short_steps = {s.step for s in res.samples if s.short_horizon}
    extra = gen.cfg.extra_short_samples
    with capsys.disabled():
        print(f"\n[mpc short-horizon oversampling] {len(short_steps)}/{len(per_step)} ticks "
              f"inside {gen.cfg.short_horizon_wp} waypoints -> +{extra} sample(s) each")
    assert short_steps, "no tick ever got within the short-horizon window"
    assert len(short_steps) < len(per_step), "every tick is short -- the window is too wide"
    for st, n in per_step.items():
        assert n == (1 + extra if st in short_steps else 1)
    # ...and the extra records really are re-noised, not duplicates.
    dupes = [s for s in res.samples if s.short_horizon and s.step == min(short_steps)]
    assert len(dupes) == 1 + extra
    assert not np.array_equal(dupes[0].init_traj, dupes[1].init_traj)
    assert np.array_equal(dupes[0].q, dupes[1].q)


def test_replay_store_recency_bias(capsys):
    def mk(i):
        return MPCSample(
            q=np.zeros(1), qd=np.zeros(1), init_traj=np.zeros((1, 1)),
            elem_mask=np.zeros((1, 1)), token_mask=np.zeros(1, bool),
            point_mask=np.zeros((1, 1)), tau_left=np.zeros(1), tau_right=np.zeros(1),
            tau0_left=0.0, tau0_right=0.0, hit_index={}, context={},
            target_drum=np.zeros(2, np.int64), min_sdf=0.0, episode=0, step=i, t_now=0.0,
        )

    store = ReplayStore(capacity=100, recency_bias=3.0)
    for i in range(400):
        store.append(mk(i))
    assert len(store) == 100
    assert {s.step for s in store.all()} == set(range(300, 400))
    w = store.weights()
    newest = max(store.all(), key=lambda s: s.step).step
    rng = np.random.default_rng(0)
    drawn = np.asarray([s.step for s in store.sample(20000, rng)])
    old_half = float((drawn < 350).mean())
    with capsys.disabled():
        print(f"\n[replay store] bias=3.0 -> newest/oldest weight ratio "
              f"{w.max() / w.min():.2f}, older-half draw share {old_half:.3f}")
    assert newest == 399
    assert old_half < 0.45
    uni = ReplayStore(capacity=100, recency_bias=0.0)
    for i in range(400):
        uni.append(mk(i))
    d2 = np.asarray([s.step for s in uni.sample(20000, np.random.default_rng(0))])
    assert abs(float((d2 < 350).mean()) - 0.5) < 0.03


def test_collate_produces_model_ready_tensors(episode):
    from amortized_opt.model import AmortizedOptimizer, ModelConfig
    from amortized_opt.tokenizer import ContextInputs

    res, _ = episode
    batch = res.samples[:8]
    state, init, masks, ctx, extras = collate(batch)
    assert state.shape == (8, 2 * C.N_DOF)
    assert init.shape == (8, H, C.N_DOF)
    assert masks.elem_mask.shape == (8, H, C.N_DOF)
    inputs = ContextInputs(**ctx)
    model = AmortizedOptimizer(
        ModelConfig(max_horizon=H, num_drums=inputs.num_drums, num_future=inputs.num_future)
    )
    out = model(init, state, model.encode_context(inputs), masks)
    assert out.traj.shape == (8, H, C.N_DOF)
    assert torch.isfinite(out.traj).all()
    assert extras["target_drum"].shape == (8, 2)


def test_generate_driver_fills_a_store(rig):
    _kit, clips, gen, starts = rig
    b = starts.sample(8, seed=0)
    store = gen.generate(clips[:2], b, n_episodes=2, seed=0, max_steps=15)
    assert len(store) >= 30
    assert all(isinstance(s, MPCSample) for s in store.all())
    eps = {s.episode for s in store.all()}
    assert eps == {0, 1}
