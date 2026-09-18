"""Evaluation-harness behaviour: does it score good and bad trajectories right?

Four things are checked, none of which need the repo:

1. **Reproducibility** -- same seed, byte-identical grid; different seed, a
   different grid; the standard grid is the repo's 162-case shape.
2. **Good trajectory** -- stored trajopt-HPIPM solutions (fixture NPZ, carved
   out of the recorded baseline) land in the hit disk at the commanded speed,
   with no collision, torque, velocity or joint-box violation.  Currently
   skipped: those plans targeted a ±20 mm slab, not the 1 cm interior cylinder.
3. **Bad trajectories** -- a straight joint-space line into the drum, an
   overspeed replay, and an out-of-box pose each light up the corresponding
   metric and nothing else.
4. **Round-tripping** -- cases and trajectories survive save/load, and
   re-scoring a stored run reproduces the same numbers exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import amortized_opt as A
from amortized_opt.eval_harness import (
    DRUM_HAND,
    EvalCase,
    AcceptanceThresholds,
    TrajectoryScorer,
    _stick_under_any_drum,
    _tip_under_any_drum,
    aggregate,
    generate_grid,
    load_run,
    run_eval,
    save_run,
)

_HERE = Path(__file__).resolve().parent
FIXTURE = _HERE / "data" / "trajopt_hpipm_fixture.npz"
FIXTURE_CASES = _HERE / "data" / "trajopt_hpipm_fixture_cases.json"


@pytest.fixture(scope="module")
def scorer() -> TrajectoryScorer:
    return TrajectoryScorer(device="cpu")


@pytest.fixture(scope="module")
def std_cases(scorer):
    return generate_grid(seed=0, grid="standard", scorer=scorer)


# ============================================================ reproducibility


def test_standard_grid_shape(std_cases):
    """6 drums x 3 horizons x 3 speeds x 3 repeats -- the repo's 162-case grid."""
    assert len(std_cases) == 162
    from collections import Counter

    per_drum = Counter(c.drum for c in std_cases)
    assert set(per_drum) == set(DRUM_HAND)
    assert set(per_drum.values()) == {27}
    # both hands equally covered
    per_hand = Counter(c.hand for c in std_cases)
    assert per_hand["left"] == per_hand["right"] == 81
    assert sorted({round(c.horizon_s, 3) for c in std_cases}) == [0.44, 0.6, 0.8]
    assert sorted({c.strike_speed for c in std_cases}) == [1.0, 2.0, 3.0]


def test_grid_is_reproducible(scorer, std_cases):
    again = generate_grid(seed=0, grid="standard", scorer=scorer)
    assert len(again) == len(std_cases)
    for a, b in zip(std_cases, again):
        assert a.drum == b.drum and a.hand == b.hand and a.N == b.N
        assert a.strike_speed == b.strike_speed
        np.testing.assert_array_equal(a.q_start, b.q_start)
        np.testing.assert_array_equal(a.qdot_start, b.qdot_start)
        np.testing.assert_array_equal(a.strike.position, b.strike.position)
        np.testing.assert_array_equal(a.strike.velocity, b.strike.velocity)


def test_grid_depends_on_seed(scorer, std_cases):
    other = generate_grid(seed=1, grid="standard", scorer=scorer)
    assert len(other) == len(std_cases)
    diff = sum(
        1 for a, b in zip(std_cases, other) if not np.array_equal(a.q_start, b.q_start)
    )
    assert diff > 0.8 * len(std_cases)


def test_start_above_drums_keeps_tips_over_kit(scorer):
    """Box starts can hang under snare/floor_tom (pelvis z=0); the floor rejects them."""
    cases = generate_grid(
        seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
        start_above_drums=("snare", "floor_tom"),
    )
    assert cases
    names = list(scorer.scene.names)
    z_floor = max(
        float(scorer.scene.head_centers[names.index("snare")][2]),
        float(scorer.scene.head_centers[names.index("floor_tom")][2]),
    )
    n_under = 0
    for c in cases:
        Qt = scorer._t(np.asarray(c.q_start, float).reshape(1, -1))
        for h in ("left", "right"):
            z = float(scorer._points_world(Qt, h)[0, 0, 2])
            assert z >= z_floor - 1e-6, (c.case_id, h, z, z_floor)
        n_under += int(_tip_under_any_drum(scorer, c.q_start))
    assert n_under == 0
    # Default (no floor, idle at URDF zero) still produces hanging starts,
    # so the flags are doing work.
    raw = generate_grid(
        seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
        start_above_drums=(), start_not_under_drums=False,
    )
    n_low = 0
    n_raw_under = 0
    for c in raw:
        Qt = scorer._t(np.asarray(c.q_start, float).reshape(1, -1))
        z = min(
            float(scorer._points_world(Qt, h)[0, 0, 2]) for h in ("left", "right")
        )
        n_low += int(z < z_floor)
        n_raw_under += int(_tip_under_any_drum(scorer, c.q_start))
    assert n_low > 10
    assert n_raw_under > 10


def test_default_grid_rejects_stick_under_kit(scorer):
    """Default eval starts must not park a stick under a cymbal / pad."""
    cases = generate_grid(
        seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
    )
    assert cases
    for c in cases:
        assert not _stick_under_any_drum(scorer, c.q_start), (c.case_id, c.drum, c.hand)
    near = generate_grid(seed=0, grid="near_drum", scorer=scorer)
    assert near
    for c in near:
        src = scorer.drum_index.get(str(c.start_drum), -1)
        assert not _stick_under_any_drum(scorer, c.q_start, exempt_drum=src), (
            c.case_id, c.drum, c.start_drum
        )


def _q0_min_sdf(scorer: TrajectoryScorer, q: np.ndarray, *, exempt_drum: int | None = None) -> float:
    Qt = scorer._t(np.asarray(q, float).reshape(1, -1))
    worst = float("inf")
    skip = int(exempt_drum) if exempt_drum is not None and int(exempt_drum) >= 0 else -1
    for h in ("left", "right"):
        pts = scorer._points_world(Qt, h)
        cap_b, cap_u = scorer._stick_drum_sdf(pts[:, 1], pts[:, 0])
        wr_b, wr_u = scorer._wrist_drum_sdf(pts[:, -1])
        if skip >= 0:
            cap_b, cap_u = cap_b.clone(), cap_u.clone()
            cap_b[:, skip] = float("inf")
            cap_u[:, skip] = float("inf")
        s = scorer._self_sdf(Qt, pts)
        worst = min(
            worst,
            float(cap_b.min()), float(cap_u.min()),
            float(wr_b.min()), float(wr_u.min()),
            float(s.min()),
        )
    return worst


def test_start_clearance_rejects_kissing_q0(scorer):
    """Rolling-eval default is a 2 cm SDF floor at q_start, not min_sdf > 0."""
    cases = generate_grid(
        seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
        start_above_drums=("snare", "floor_tom"),
        start_clearance_m=0.02,
    )
    assert cases
    for c in cases:
        sdf = _q0_min_sdf(scorer, c.q_start)
        assert sdf > 0.02 - 1e-9, (c.case_id, c.drum, sdf)
    # Zero floor still allows sub-cm starts (the old kissing bug).
    kissing = generate_grid(
        seed=0, grid="standard", scorer=scorer, include_extra_drums=True,
        start_above_drums=("snare", "floor_tom"),
        start_clearance_m=0.0,
    )
    n_close = sum(1 for c in kissing if _q0_min_sdf(scorer, c.q_start) < 0.01)
    assert n_close > 0


def test_near_drum_grid(scorer):
    cases = generate_grid(seed=0, grid="near_drum", scorer=scorer)
    assert cases, "near_drum grid is empty"
    for c in cases:
        assert c.start_mode == "near_drum"
        assert c.start_drum is not None and c.start_drum != c.drum
        # a departure start carries real joint velocity
        assert np.linalg.norm(c.qdot_start) > 1e-3
    # no-crossover: the right hand never departs from the leftmost drum
    assert not any(c.hand == "right" and c.start_drum == "hihat" for c in cases)


def test_case_roundtrip(std_cases):
    for c in std_cases[:20]:
        d = json.loads(json.dumps(c.to_dict()))
        b = EvalCase.from_dict(d)
        np.testing.assert_array_equal(c.q_start, b.q_start)
        np.testing.assert_array_equal(c.qdot_start, b.qdot_start)
        np.testing.assert_array_equal(c.strike.position, b.strike.position)
        assert b.strike.resolved_contact_sigma() == c.strike.resolved_contact_sigma()
        assert b.strike.is_shaft_contact() == c.strike.is_shaft_contact()


def test_cymbals_use_a_shaft_contact(std_cases):
    """Ported strike rules survive the trip through the grid generator."""
    for c in std_cases:
        s = c.strike
        if c.drum.startswith("cymbal") or c.drum == "hihat":
            assert s.is_shaft_contact()
            assert s.resolved_contact_sigma() == pytest.approx(0.10)
            assert s.half_plane_dir is not None
            assert s.disk_center_offset is not None
        else:
            assert not s.is_shaft_contact()
            assert s.resolved_contact_sigma() == pytest.approx(0.20)
        if c.drum == "snare":  # a flat pad: whole head, no offset
            assert s.half_plane_dir is None and s.disk_center_offset is None


# =========================================================== good trajectory


def _load_fixture():
    if not (FIXTURE.exists() and FIXTURE_CASES.exists()):
        pytest.skip(
            "trajopt fixture missing; regenerate with "
            "scripts/make_test_fixture.py after a baseline run"
        )
    cases = [EvalCase.from_dict(d) for d in json.loads(FIXTURE_CASES.read_text())["cases"]]
    z = np.load(FIXTURE)
    trajs = {int(k.split("_")[1]): np.asarray(z[k], float) for k in z.files
             if k.startswith("case_")}
    return cases, trajs


def test_good_trajectory_scores_well(scorer):
    """A stored trajopt-HPIPM solution must score as a clean, legal strike."""
    pytest.skip(
        "HPIPM fixtures targeted a ±20 mm slab around a 5 mm-inset aim; "
        "the legal volume is now a 1 cm cylinder on the inner face of the head."
    )
    cases, trajs = _load_fixture()
    assert len(cases) >= 6
    for c in cases:
        r = scorer.score(c, trajs[c.case_id])
        assert r.on_head, f"{c.case_id} {c.drum}: contact left the hit disk"
        assert r.ok_hit
        assert r.coll_pen_count == 0, (
            f"{c.case_id} {c.drum}: striking arm penetrates by "
            f"{r.coll_pen_max_m * 1000:.1f} mm"
        )
        assert r.coll_min_sdf_m > 0.0
        assert r.contact_speed_err_mps < 0.5 * max(r.contact_speed_target_mps, 1.0)
        assert r.contact_dir_cos > 0.8, "strike does not arrive into the head"
        assert r.tau_excess_max_nm == 0.0
        assert r.vel_excess_max_rads == 0.0
        assert r.qpos_posture_excess_max_rad <= 1e-9
        assert r.start_pos_err_rad < 1e-6, "planner moved the pinned start pose"
        assert np.isfinite(r.jerk_integral) and r.jerk_integral > 0.0


def test_good_trajectory_accepted_rate(scorer):
    pytest.skip(
        "HPIPM fixtures targeted a ±20 mm slab around a 5 mm-inset aim; "
        "the legal volume is now a 1 cm cylinder on the inner face of the head."
    )
    cases, trajs = _load_fixture()
    rows = [scorer.score(c, trajs[c.case_id]) for c in cases]
    agg = aggregate(rows)
    assert agg["counts"]["accepted"]["count"] == len(rows)


# =========================================================== bad trajectories


def _straight_line(case: EvalCase, q_end: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, case.n_waypoints)[:, None]
    return np.asarray(case.q_start)[None, :] * (1 - s) + np.asarray(q_end)[None, :] * s


def _ik_to_aim(scorer: TrajectoryScorer, case: EvalCase, iters: int = 150) -> np.ndarray:
    q = np.asarray(case.q_start, float).copy()
    sigma = case.strike.resolved_contact_sigma()
    target = np.asarray(case.strike.position, float)
    lo, hi = scorer.q_lo_post, scorer.q_hi_post
    for _ in range(iters):
        p, _t, _d = scorer.contact_world(q, case.hand, sigma)
        e = target - p
        if np.linalg.norm(e) < 1e-4:
            break
        J = scorer.tip_jacobian(q, case.hand, sigma)
        q = np.clip(q + 0.5 * (J.T @ np.linalg.solve(J @ J.T + 2.5e-3 * np.eye(3), e)), lo, hi)
    return q


def test_line_through_the_kit_is_flagged(scorer, std_cases):
    """An obstacle-blind straight line into the head must report penetration.

    It also lands the contact point on the head (the endpoint is an IK solution),
    so this isolates the collision metric from the hit metric: the harness must
    say "good hit, illegal path", not fail everything at once.
    """
    flagged = 0
    checked = 0
    # Scorer test, not a sampler test: a 2 cm q0 floor makes the joint-space
    # line miss the kit more often, so pin the old kissing grid.
    cases = generate_grid(seed=0, grid="standard", scorer=scorer, start_clearance_m=0.0)
    for c in cases[::17]:
        Q = _straight_line(c, _ik_to_aim(scorer, c))
        r = scorer.score(c, Q)
        checked += 1
        if r.coll_pen_count > 0:
            flagged += 1
            assert r.coll_pen_max_m > 0.0
            assert r.coll_min_sdf_m < 0.0
            assert not r.ok_collision
            assert not r.accepted
    assert checked >= 5
    assert flagged >= checked - 1, (
        f"only {flagged}/{checked} obstacle-blind lines were flagged"
    )


def test_overspeed_trajectory_is_flagged(scorer, std_cases):
    """Replaying the same path faster must trip velocity, then torque.

    Three speeds of one path: as planned (a quintic blend, so it starts and ends
    at rest and carries no endpoint acceleration impulse), 8x compressed (torque
    only -- the joint speed ratings are 22-37 rad/s and an 8x replay of a ~2 rad
    move still fits under them), and the whole move in a single control period
    (velocity too).  Checking the ordering, not just a single threshold, is what
    catches a scorer that reports a constant.
    """
    c = std_cases[0]
    q_end = _ik_to_aim(scorer, c)
    T = c.n_waypoints
    s = np.linspace(0.0, 1.0, T)[:, None]
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    Q = np.asarray(c.q_start)[None, :] * (1 - blend) + q_end[None, :] * blend
    fast = Q[np.clip(np.arange(T) * 8, 0, T - 1)]
    teleport = Q[np.clip(np.arange(T) * (T - 1), 0, T - 1)]

    r_slow, r_fast, r_tel = (scorer.score(c, X) for X in (Q, fast, teleport))

    assert r_slow.vel_excess_max_rads == 0.0
    assert r_slow.tau_excess_max_nm == 0.0, (
        "the smooth reference path should not exceed a torque limit; got "
        f"{r_slow.tau_excess_max_nm:.2f} N.m"
    )

    # 8x: torque blows well before the speed ratings do
    assert r_fast.tau_excess_max_nm > 0.0, "8x replay did not exceed a torque limit"
    assert not r_fast.ok_torque and not r_fast.accepted
    assert r_fast.tau_max_nm > r_slow.tau_max_nm

    # whole move in one control period: velocity too
    assert r_tel.vel_excess_max_rads > 0.0, "teleport replay not flagged"
    assert not r_tel.ok_velocity
    assert r_tel.vel_frac_over > 0.0
    assert r_tel.tau_excess_max_nm > r_fast.tau_excess_max_nm
    assert not r_tel.accepted


def test_out_of_box_pose_is_flagged(scorer, std_cases):
    c = std_cases[0]
    Q = np.repeat(np.asarray(c.q_start)[None, :], c.n_waypoints, axis=0)
    Q[c.n_waypoints // 2] = scorer.q_hi_post + 0.5
    r = scorer.score(c, Q)
    assert r.qpos_posture_excess_max_rad > 0.4
    assert not r.ok_joint_limits and not r.accepted


def test_holding_still_misses(scorer, std_cases):
    """The degenerate planner: no travel, no violations, no hit."""
    c = std_cases[0]
    Q = np.repeat(np.asarray(c.q_start)[None, :], c.n_waypoints, axis=0)
    r = scorer.score(c, Q)
    assert not r.on_head and not r.ok_hit and not r.accepted
    assert r.travel_l1_rad == pytest.approx(0.0, abs=1e-9)
    assert r.contact_speed_mps == pytest.approx(0.0, abs=1e-9)
    assert r.contact_speed_err_mps == pytest.approx(c.strike_speed, abs=1e-9)
    assert r.coll_pen_count == 0  # the grid only emits collision-free starts


def test_target_drum_exemption_is_local(scorer, std_cases):
    """The exemption must cover the strike, not the whole path.

    Parking the stick inside the target drum for the entire trajectory has to be
    flagged; only the last ``strike_exempt_waypoints`` are forgiven.
    """
    c = std_cases[0]
    q_end = _ik_to_aim(scorer, c)
    # sink 4 cm past the head along the strike direction, then hold there
    Q = np.repeat(q_end[None, :], c.n_waypoints, axis=0)
    r_all = scorer.score(c, Q)
    tight = TrajectoryScorer(device="cpu", strike_exempt_waypoints=0)
    r_none = tight.score(c, Q)
    assert r_none.coll_pen_count >= r_all.coll_pen_count
    assert r_none.coll_min_sdf_m <= r_all.coll_min_sdf_m


def test_shape_metrics_are_sane(scorer, std_cases):
    c = std_cases[0]
    q_end = _ik_to_aim(scorer, c)
    Q = _straight_line(c, q_end)
    r = scorer.score(c, Q)
    # a monotone straight line wastes nothing and never reverses
    assert r.travel_waste == pytest.approx(1.0, abs=1e-6)
    assert r.reversals == 0
    assert r.travel_l1_rad == pytest.approx(
        float(np.abs(q_end - c.q_start).sum()), rel=1e-9
    )
    assert r.travel_wrist_rad <= r.travel_l1_rad + 1e-12


def test_wrong_shape_is_rejected(scorer, std_cases):
    c = std_cases[0]
    with pytest.raises(ValueError):
        scorer.score(c, np.zeros((c.n_waypoints, 5)))
    with pytest.raises(ValueError):
        scorer.score(c, np.zeros((c.n_waypoints + 3, 17)))
    bad = np.repeat(np.asarray(c.q_start)[None, :], c.n_waypoints, axis=0)
    bad[2, 0] = np.nan
    with pytest.raises(ValueError):
        scorer.score(c, bad)


# ================================================================ round-trip


def test_run_eval_and_persistence(scorer, std_cases, tmp_path):
    cases = std_cases[:8]

    def planner(case):
        return A.PlanResult(
            Q=_straight_line(case, _ik_to_aim(scorer, case)),
            solve_s=0.001,
            meta={"converged": True, "iterations": 3, "message": "ok"},
        )

    rows, trajs = run_eval(planner, cases, scorer=scorer)
    assert len(rows) == len(cases) == len(trajs)
    paths = save_run(tmp_path, cases, rows, trajs, prefix="t", meta={"x": 1})
    assert all(p.exists() for p in paths.values())

    cases2, trajs2, meta = load_run(tmp_path, prefix="t")
    assert meta["x"] == 1
    assert len(cases2) == len(cases)
    for c in cases:
        np.testing.assert_allclose(trajs2[c.case_id], trajs[c.case_id])
    rows2 = [scorer.score(c, trajs2[c.case_id]) for c in cases2]
    for a, b in zip(rows, rows2):
        assert a.contact_pos_err_mm == pytest.approx(b.contact_pos_err_mm, rel=1e-12)
        assert a.coll_min_sdf_m == pytest.approx(b.coll_min_sdf_m, rel=1e-12)
        assert a.tau_excess_max_nm == pytest.approx(b.tau_excess_max_nm, abs=1e-12)


def test_planner_exception_is_contained(scorer, std_cases):
    def boom(case):
        raise RuntimeError("planner exploded")

    rows, _ = run_eval(boom, std_cases[:3], scorer=scorer)
    assert len(rows) == 3
    for r in rows:
        assert r.planner_failed and not r.accepted
        assert "planner exploded" in r.message


_STANDALONE = """
import sys
sys.path.insert(0, ".")
import numpy as np, amortized_opt as A

bad = [m for m in sys.modules if m.startswith(("g1_drummer", "trajopt"))]
assert not bad, f"repo leaked into the standalone harness run: {bad}"

sc = A.TrajectoryScorer(device="cpu")
cases = A.generate_grid(seed=0, grid="standard+near_drum", scorer=sc)
assert len(cases) > 200

def hold(case):
    return np.repeat(np.asarray(case.q_start)[None, :], case.n_waypoints, axis=0)

rows, trajs = A.run_eval(hold, cases[:4], scorer=sc)
assert len(rows) == 4 and all(not r.accepted for r in rows)
assert A.format_summary(rows).count("\\n") > 10
print("HARNESS_STANDALONE_OK")
"""


def test_harness_runs_without_the_repo(capsys):
    """Grid generation + scoring must work with nothing from the repo on the path."""
    import os
    import subprocess
    import sys as _sys

    env = {k: v for k, v in os.environ.items() if k != "UNITREE_MODEL_DIR"}
    env["PYTHONPATH"] = ""
    out = subprocess.run(
        [_sys.executable, "-c", _STANDALONE],
        cwd=str(_HERE.parent), env=env, capture_output=True, text=True, timeout=900,
    )
    with capsys.disabled():
        print(f"\n[harness standalone] rc={out.returncode} {out.stdout.strip()!r}")
        if out.returncode:
            print(out.stderr[-3000:])
    assert out.returncode == 0 and "HARNESS_STANDALONE_OK" in out.stdout


def test_scoring_is_deterministic(scorer, std_cases):
    c = std_cases[3]
    Q = _straight_line(c, _ik_to_aim(scorer, c))
    a, b = scorer.score(c, Q), scorer.score(c, Q)
    assert a == b


def test_hit_margin_covers_2cm_axial_overshoot(scorer, std_cases):
    """Eval's 2 cm slack accepts a strike 15 mm past the 1 cm slab; tight does not."""
    from amortized_opt.strike import strike_outward_normal

    tight = TrajectoryScorer(
        device="cpu", thresholds=AcceptanceThresholds(hit_margin_m=0.0),
    )
    found = False
    for c in std_cases[::9]:
        n_out = strike_outward_normal(c.strike)
        target = np.asarray(c.strike.position, float) + 0.015 * n_out
        q = np.asarray(c.q_start, float).copy()
        sigma = c.strike.resolved_contact_sigma()
        lo, hi = scorer.q_lo_post, scorer.q_hi_post
        for _ in range(200):
            p, _t, _d = scorer.contact_world(q, c.hand, sigma)
            e = target - p
            if np.linalg.norm(e) < 2e-3:
                break
            J = scorer.tip_jacobian(q, c.hand, sigma)
            q = np.clip(
                q + 0.5 * (J.T @ np.linalg.solve(J @ J.T + 2.5e-3 * np.eye(3), e)),
                lo, hi,
            )
        Q = np.repeat(q[None, :], c.n_waypoints, axis=0)
        r_fat = scorer.score(c, Q)
        ax = r_fat.axial_dist_m
        if not (0.011 < ax < 0.029):
            continue
        r_tight = tight.score(c, Q)
        assert not r_tight.on_head, (c.case_id, ax)
        assert r_fat.on_head, (c.case_id, ax)
        found = True
        break
    assert found, "no case landed 11-29 mm past the aim along n_out"
