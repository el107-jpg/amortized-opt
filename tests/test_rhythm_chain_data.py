"""Tests for rhythm-derived schedule schema and synthetic dual transitions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.fk import ChainFK
from amortized_opt.idle_cylinder import resolve_idle_cylinder_library
from amortized_opt.phase2_data import (
    KIND_FAR_FETCHED,
    KIND_RHYTHM_IDLE,
    Phase2Pool,
    Phase2ScenarioConfig,
)
from amortized_opt.rhythm import DrumKit, Hit, default_rcc_dir, load_rcc_pool
from amortized_opt.rhythm_chain_data import (
    DualTransitionConfig,
    SCHEDULE_FIELDS,
    SCHEDULE_SCHEMA,
    build_dual_transition_clips,
    build_idle_entry_index,
    dual_transition_kept_count,
    generate_far_fetched_scenarios,
    generate_rhythm_idle_scenarios,
    pack_schedule_from_hits,
    project_schedule_slot0_to_future,
    tip_crossover_depth_m,
    tip_straight_line_speed,
)
from amortized_opt.constants import ready_joint_pos, G1_UPPER_BODY_JOINTS
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice


@pytest.fixture(scope="module")
def kit():
    return DrumKit.from_layout()


@pytest.fixture(scope="module")
def library():
    return GoalPoseLibrary()


@pytest.fixture(scope="module")
def rcc_clips(kit):
    return load_rcc_pool(default_rcc_dir("train"), kit)[:12]


def test_dual_transition_kept_count_in_ballpark(kit, library):
    n = dual_transition_kept_count(kit, library, cfg=DualTransitionConfig(sep_max=0.45))
    # Nominal aims: tens, not hundreds; not empty.
    assert 20 <= n <= 80, f"unexpected kept count {n}"


def test_dual_transition_rejects_arm_split(kit, library):
    """L snare→hihat × R tom1→cymbal_right should fail end/start proximity."""
    cfg = DualTransitionConfig(sep_max=0.45, allow_same_drum_layer=False)
    clips = build_dual_transition_clips(kit, library, cfg=cfg)
    names = {c.name for c in clips}
    bad = "synth_dual_snare_hihat__tom1_cymbal_right"
    assert bad not in names


def test_dual_transition_clips_two_layers(kit, library):
    clips = build_dual_transition_clips(
        kit, library, cfg=DualTransitionConfig(sep_max=0.45, gap_s=0.60)
    )
    assert clips
    c = clips[0]
    assert len(c.hits) == 4
    t0 = sorted({h.time for h in c.hits})
    assert len(t0) == 2
    assert t0[0] == pytest.approx(0.0)
    left0 = {h.drum for h in c.hits if h.hand == "left" and h.time == t0[0]}
    right0 = {h.drum for h in c.hits if h.hand == "right" and h.time == t0[0]}
    assert len(left0) == 1 and len(right0) == 1


def test_idle_entry_index_synth_first_layer_only(kit, library):
    synth = build_dual_transition_clips(
        kit, library, cfg=DualTransitionConfig(sep_max=0.45)
    )[:3]
    entries, clips = build_idle_entry_index([], synth)
    assert len(clips) == 3
    # Two hands × 3 clips = 6 first-layer anchors (no second-layer entries).
    assert len(entries) == 6
    assert all(e.synth and e.hit_time == 0.0 for e in entries)


def test_schedule_pack_and_project(kit):
    h0 = Hit(0.5, 0.5, 0, kit.names[0], "left", 0.5)
    h1 = Hit(1.0, 1.0, 1, kit.names[1], "left", 0.5)
    sched = pack_schedule_from_hits(
        kit,
        {"left": [h0, h1], "right": []},
        goal_hit_time={"left": 0.0, "right": 0.0},
        n_slots=2,
    )
    assert sched["sched_valid"][0, 0] and sched["sched_valid"][0, 1]
    assert not sched["sched_valid"][1, 0]
    assert sched["sched_dt"][0, 0] == pytest.approx(0.5)
    fut = project_schedule_slot0_to_future(sched)
    assert fut["future_valid"].shape == (2, 1)
    assert fut["future_valid"][0, 0]
    assert not fut["future_valid"][1, 0]
    np.testing.assert_array_equal(fut["future_drum"][:, 0], sched["sched_drum"][:, 0])


def test_schedule_schema_constant():
    assert SCHEDULE_SCHEMA.startswith("rhythm_chain_")


def test_refill_lookahead_from_clip_num_future_one(kit, library):
    from amortized_opt.rhythm import Hit, RhythmClip
    from amortized_opt.rhythm_chain_data import (
        hits_after_clip_time,
        refill_lookahead_from_clip,
    )

    hits = [
        Hit(0.0, 0.0, 0, kit.names[0], "left", 0.5),
        Hit(0.5, 0.5, 1, kit.names[1], "left", 0.5),
        Hit(1.0, 1.0, 2, kit.names[2], "left", 0.5),
        Hit(1.5, 1.5, 3, kit.names[3], "left", 0.5),
    ]
    clip = RhythmClip(name="toy", hits=hits, duration=3.0)
    # After promoting the hit at t=0.5, refill future + schedule[1].
    fields, ok = refill_lookahead_from_clip(
        kit, clip, "left", new_goal_hit_time=0.5, n_future=1, n_schedule_extra=1,
    )
    assert ok
    assert fields["future_valid"][0]
    assert int(fields["future_drum"][0]) == 2
    assert fields["future_gap"][0] == pytest.approx(0.5)
    assert fields["sched_valid"][0]
    assert int(fields["sched_drum"][0]) == 3
    # Clip tail: no hits after the last goal.
    _fields2, ok2 = refill_lookahead_from_clip(
        kit, clip, "left", new_goal_hit_time=1.5, n_future=1, n_schedule_extra=1,
    )
    assert not ok2
    assert hits_after_clip_time(clip, "left", 1.5, 2) == []


def _phase_a_cfg(**kw):
    base = dict(
        horizon=100,
        buckets=(25, 50, 100),
        eval_like_frac=0.0,
        cold_start_frac=0.0,
        pair_frac=0.0,
        randomize_kit=False,
        hit_time_jitter_s=0.05,
        lead_s=1.0,
        tip_speed_max=4.0,
        synth_transition_frac=0.25,
        num_future=1,
        idle_cylinder_path="default",
    )
    base.update(kw)
    return Phase2ScenarioConfig(**base)


def test_phase_a_lead_near_one_second_and_future_valid(rcc_clips, kit, library):
    cfg = _phase_a_cfg()
    raw = generate_rhythm_idle_scenarios(
        80, rcc_clips, seed=7, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["accepted"] >= 8, raw["stats"]
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device="cpu")
    assert len(pool) == raw["stats"]["accepted"]

    hits, act, kinds = [], [], []
    fut_ok = []
    for b in pool.buckets:
        d = pool.data[b]
        hits.append(d["hit"].numpy())
        act.append(d["active"].numpy())
        kinds.append(d["kind"].numpy())
        fv = d["future_valid"].numpy()
        for i in range(fv.shape[0]):
            for hi in range(2):
                if act[-1][i, hi]:
                    fut_ok.append(bool(fv[i, hi, 0]))
        for f in SCHEDULE_FIELDS:
            assert f in d
        assert "clip_i" in d and "hand_hit_time" in d

    hit = np.concatenate(hits, axis=0)
    active = np.concatenate(act, axis=0)
    kind = np.concatenate(kinds, axis=0)
    assert np.all(kind == KIND_RHYTHM_IDLE)
    assert all(fut_ok)

    dt = float(cfg.dt)
    tths = []
    for i in range(hit.shape[0]):
        for hi in range(2):
            if active[i, hi]:
                tths.append(hit[i, hi] * dt)
    assert tths
    med = float(np.median(tths))
    # lead_s=1.0 ± jitter ± relative dual offset; median should sit near 1 s.
    assert 0.70 <= med <= 1.35, f"median tth {med}"


def test_phase_a_tip_speed_reject(rcc_clips, kit, library):
    cfg = _phase_a_cfg(tip_speed_max=0.05)  # essentially impossible from idle
    raw = generate_rhythm_idle_scenarios(
        40, rcc_clips, seed=11, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["reject_tip_speed"] > 0
    assert raw["stats"]["accepted"] == 0


def test_tip_straight_line_speed_helper(kit):
    fk = ChainFK(device="cpu", dtype=torch.float64)
    idle = resolve_idle_cylinder_library("default")
    assert idle is not None
    q0 = idle.full_pose("left", 0, yaw=0.0)
    q0 = idle.full_pose("right", 0, yaw=0.0, base=q0)
    p = np.asarray(kit.positions[0], float)
    spd = tip_straight_line_speed(fk, q0, "left", p, 1.0)
    assert spd > 0.0
    assert tip_straight_line_speed(fk, q0, "left", p, 0.01) > spd


def test_far_fetched_is_eval_style_grid(kit, library):
    """Far is rest + one strike + empty future on the eval horizon/speed set."""
    cfg = _phase_a_cfg()
    raw = generate_far_fetched_scenarios(
        40, None, seed=5, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["accepted"] >= 8, raw["stats"]
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device="cpu")
    allowed_n = {max(int(round(h / cfg.dt)), 4) for h in (0.45, 0.60, 0.80)}
    for b in pool.buckets:
        d = pool.data[b]
        assert np.all(d["kind"].numpy() == KIND_FAR_FETCHED)
        act = d["active"].numpy()
        fv = d["future_valid"].numpy()
        hit = d["hit"].numpy()
        qd0 = d["qd0"].numpy()
        for i in range(act.shape[0]):
            assert int(act[i].sum()) == 1
            assert np.allclose(qd0[i], 0.0, atol=1e-6)
            for hi in range(2):
                if act[i, hi]:
                    assert not fv[i, hi, 0]
                    assert int(hit[i, hi]) in allowed_n


def test_far_fetched_excludes_eval_grid_starts(kit, library):
    from amortized_opt.eval_harness import TrajectoryScorer, generate_grid

    sc = TrajectoryScorer(device="cpu")
    ev = generate_grid(
        seed=0, grid="standard", scorer=sc, include_extra_drums=True,
        start_above_drums=("snare", "floor_tom"),
    )
    eq = np.stack([np.asarray(c.q_start, float) for c in ev])
    cfg = _phase_a_cfg()
    raw = generate_far_fetched_scenarios(
        60, None, seed=0, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["accepted"] >= 5, raw["stats"]
    for arrays in raw["buckets"].values():
        for q0 in arrays["q0"]:
            dist = np.linalg.norm(eq - np.asarray(q0, float)[None, :], axis=1).min()
            assert dist > 1e-3, dist


def test_phase_a_per_hit_jitter_independence(kit, library):
    """Different seeds with synth-only entries should not share hit columns."""
    cfg = _phase_a_cfg(
        synth_transition_frac=1.0, hit_time_jitter_s=0.05, randomize_kit=False
    )
    a = generate_rhythm_idle_scenarios(30, [], seed=0, cfg=cfg, kit=kit, library=library)
    b = generate_rhythm_idle_scenarios(30, [], seed=1, cfg=cfg, kit=kit, library=library)
    assert a["stats"]["accepted"] >= 1 and b["stats"]["accepted"] >= 1
    ha = np.concatenate([v["hit"] for v in a["buckets"].values()], axis=0)
    hb = np.concatenate([v["hit"] for v in b["buckets"].values()], axis=0)
    n = min(len(ha), len(hb))
    assert not np.array_equal(ha[:n], hb[:n])


def test_effective_kind_fracs_empty_b_renorm():
    from amortized_opt.rhythm_chain_data import (
        KIND_FAR_FETCHED,
        KIND_POST_HIT,
        KIND_RHYTHM_IDLE,
        effective_kind_fracs,
    )

    logs = []
    state = {}
    out = effective_kind_fracs(
        {KIND_RHYTHM_IDLE: 0.30, KIND_POST_HIT: 0.60, KIND_FAR_FETCHED: 0.10},
        {KIND_RHYTHM_IDLE: 100, KIND_FAR_FETCHED: 20},
        log_fn=logs.append,
        log_state=state,
    )
    assert out[KIND_RHYTHM_IDLE] == pytest.approx(0.75)
    assert out[KIND_FAR_FETCHED] == pytest.approx(0.25)
    assert KIND_POST_HIT not in out
    assert logs and "N_B=0" in logs[0]
    # One-shot: second call does not re-log.
    effective_kind_fracs(
        {KIND_RHYTHM_IDLE: 0.30, KIND_POST_HIT: 0.60, KIND_FAR_FETCHED: 0.10},
        {KIND_RHYTHM_IDLE: 100, KIND_FAR_FETCHED: 20},
        log_fn=logs.append,
        log_state=state,
    )
    assert len(logs) == 1


def test_effective_kind_fracs_live_b_overlay():
    from amortized_opt.rhythm_chain_data import (
        KIND_FAR_FETCHED,
        KIND_POST_HIT,
        KIND_RHYTHM_IDLE,
        effective_kind_fracs,
    )

    logs = []
    state = {}
    fracs = {KIND_RHYTHM_IDLE: 0.30, KIND_POST_HIT: 0.60, KIND_FAR_FETCHED: 0.10}
    empty = effective_kind_fracs(
        fracs, {KIND_RHYTHM_IDLE: 100, KIND_FAR_FETCHED: 20},
        log_fn=logs.append, log_state=state,
    )
    assert KIND_POST_HIT not in empty
    out = effective_kind_fracs(
        fracs,
        {KIND_RHYTHM_IDLE: 100, KIND_FAR_FETCHED: 20, KIND_POST_HIT: 50},
        log_fn=logs.append, log_state=state,
    )
    assert out[KIND_POST_HIT] == pytest.approx(0.60)
    assert out[KIND_RHYTHM_IDLE] == pytest.approx(0.30)
    assert any("KIND_POST_HIT online" in x for x in logs)


def test_roller_clip_list_matches_idle_index(kit, library, rcc_clips):
    from amortized_opt.rhythm_chain_data import roller_clip_list

    synth = build_dual_transition_clips(kit, library)
    entries, clips = build_idle_entry_index(list(rcc_clips), synth)
    rolled = roller_clip_list(kit, library=library, rcc_clips=list(rcc_clips))
    assert [c.name for c in rolled] == [c.name for c in clips]
    assert max(e.clip_i for e in entries) < len(rolled)


def test_phase2pool_sample_idx_by_kind(kit, library):
    cfg = _phase_a_cfg(synth_transition_frac=1.0)
    a = generate_rhythm_idle_scenarios(40, [], seed=3, cfg=cfg, kit=kit, library=library)
    f = generate_far_fetched_scenarios(20, None, seed=4, cfg=cfg, kit=kit, library=library)
    from amortized_opt.phase2_data import _merge
    from amortized_opt.rhythm_chain_data import KIND_POST_HIT, effective_kind_fracs

    raw = _merge([a, f])
    pool = Phase2Pool(raw["buckets"], cfg=cfg, device="cpu")
    assert pool.kind_counts().get(KIND_RHYTHM_IDLE, 0) >= 1
    assert pool.kind_counts().get(KIND_FAR_FETCHED, 0) >= 1

    fracs = effective_kind_fracs(
        {KIND_RHYTHM_IDLE: 0.30, KIND_POST_HIT: 0.60, KIND_FAR_FETCHED: 0.10},
        pool.kind_counts(),
    )
    g = torch.Generator().manual_seed(0)
    bucket = None
    for b in pool.buckets:
        kc = pool.kind_counts(b)
        if kc.get(KIND_RHYTHM_IDLE, 0) and kc.get(KIND_FAR_FETCHED, 0):
            bucket = b
            break
    if bucket is None:
        bucket = pool.buckets[0]
    idx = pool.sample_idx(bucket, 64, generator=g, kind_fracs=fracs)
    kinds = pool.data[bucket]["kind"].index_select(0, idx)
    assert set(kinds.tolist()) <= {KIND_RHYTHM_IDLE, KIND_FAR_FETCHED}


def test_sample_post_hit_tip_modes(kit):
    from amortized_opt.rhythm_chain_data import sample_post_hit_tip

    rng = np.random.default_rng(0)
    aim = np.asarray(kit.positions[0], float)
    n = np.asarray(kit.axes[0], float)
    p_hit = sample_post_hit_tip(kit, 0, aim, n, mode="hit", rng=rng)
    assert p_hit.shape == (3,)
    assert float(np.linalg.norm(p_hit - aim)) < 0.25


def test_rhythm_idle_q0_near_ready_and_crossover_capped(kit, library):
    cfg = _phase_a_cfg(synth_transition_frac=1.0, idle_arm_noise=0.05,
                       idle_max_tip_crossover_m=0.05)
    raw = generate_rhythm_idle_scenarios(
        60, [], seed=9, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["accepted"] >= 5, raw["stats"]
    fk = ChainFK(device="cpu", dtype=torch.float64)
    ready = ready_joint_pos(G1_UPPER_BODY_JOINTS)
    for b, arrays in raw["buckets"].items():
        if len(arrays["q0"]) == 0:
            continue
        for q0 in arrays["q0"]:
            depth = tip_crossover_depth_m(fk, q0)
            assert depth <= 0.05 + 1e-6, depth
            # Arm DoFs stay near ready (noise 0.05 rad; allow generous L2).
            err = 0.0
            for hand in ("left", "right"):
                cols = arm_slice(hand, G1_UPPER_BODY_JOINTS)
                err += float(np.linalg.norm(q0[cols] - ready[cols]))
            assert err < 1.5, err


def test_far_starts_not_under_drums(kit, library):
    from amortized_opt.rhythm_chain_data import _tip_under_kit

    cfg = _phase_a_cfg()
    raw = generate_far_fetched_scenarios(
        40, None, seed=12, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["accepted"] >= 5, raw["stats"]
    fk = ChainFK(device="cpu", dtype=torch.float64)
    for arrays in raw["buckets"].values():
        for q0 in arrays.get("q0", []):
            assert not _tip_under_kit(fk, q0, kit)


def test_reject_crossover_counted_when_cap_tiny(kit, library):
    cfg = _phase_a_cfg(
        synth_transition_frac=1.0, idle_arm_noise=0.35,
        idle_max_tip_crossover_m=1e-4, start_sdf_tries=2,
    )
    raw = generate_rhythm_idle_scenarios(
        40, [], seed=21, cfg=cfg, kit=kit, library=library
    )
    assert raw["stats"]["reject_crossover"] > 0, raw["stats"]
