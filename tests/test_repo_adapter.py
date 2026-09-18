"""The repo-only trajopt adapter: does it hand the solver the same problem?

Skips without ``source/g1_drummer``.  These are the joins where a silent
mismatch would corrupt the baseline: the reduced joint ordering, the joint box,
the drum frame, and the 10 -> 17 DoF embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_CONTROL, repo_available

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.skipif(
    not repo_available(), reason=f"repo originals not present at {REPO_CONTROL}"
)

import amortized_opt as A  # noqa: E402
from amortized_opt.constants import G1_UPPER_BODY_JOINTS, N_DOF  # noqa: E402


@pytest.fixture(scope="module")
def adapter():
    try:
        from repo_adapters import trajopt_hpipm_planner as P
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import the adapter: {exc}")
    return P


@pytest.fixture(scope="module")
def planner(adapter):
    try:
        return adapter.make_planner(backend="hpipm")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot build the HPIPM planner: {exc}")


@pytest.fixture(scope="module")
def scorer():
    return A.TrajectoryScorer(device="cpu")


def test_arm_index_maps_names(adapter):
    for hand, names in adapter.ARM_JOINTS.items():
        assert len(names) == 10
        idx = adapter.ARM_INDEX[hand]
        assert [G1_UPPER_BODY_JOINTS[i] for i in idx] == list(names)
        # the waist is shared, the arm is not
        assert sum(1 for n in names if n.startswith("waist_")) == 3
        assert all(n.startswith(("waist_", f"{hand}_")) for n in names)


def test_reduced_joint_box_is_the_vendored_box_minus_limit_margin(planner, adapter, scorer):
    """The solver plans inside a box shrunk by ``TrajOptConfig.limit_margin``.

    Not equality: the repo shrinks the URDF box by ``limit_margin`` (0.02 rad by
    default) *before* applying the posture narrowing, while the vendored
    ``urdf_position_limits`` defaults to a 0 margin.  The direction matters and
    is asserted here -- the solver's box is a strict SUBSET of the box the
    harness checks, so trajopt can never be scored as a joint-limit violator for
    a margin it was never given.  (Recorded in the baseline report.)
    """
    margin = float(planner.mod.TrajOptConfig().limit_margin)
    assert margin > 0.0
    for hand, rob in planner.robots.items():
        idx = adapter.ARM_INDEX[hand]
        lo_repo = np.asarray(rob.q_lower, float)
        hi_repo = np.asarray(rob.q_upper, float)
        lo_mine, hi_mine = scorer.q_lo_post[idx], scorer.q_hi_post[idx]
        assert np.all(lo_repo >= lo_mine - 1e-12), hand
        assert np.all(hi_repo <= hi_mine + 1e-12), hand
        # ... and never tighter than the margin: a joint either ties (the posture
        # clamp binds first) or is inset by exactly limit_margin.
        assert float(np.max(np.abs(lo_repo - lo_mine))) <= margin + 1e-12
        assert float(np.max(np.abs(hi_repo - hi_mine))) <= margin + 1e-12


def test_reduced_velocity_limits_match(planner, adapter, scorer):
    for hand, rob in planner.robots.items():
        idx = adapter.ARM_INDEX[hand]
        np.testing.assert_allclose(
            np.asarray(rob.v_limit, float), scorer.v_limit[idx], rtol=0, atol=1e-12
        )


def test_reduced_torque_limit_is_scaled(planner, adapter, scorer):
    """The solver is held to ``torque_scale`` (0.85) of the Isaac effort limit.

    The harness scores against the FULL effort limit, which is the physically
    meaningful bound.  Asserting the ratio here documents that the baseline's
    ``tau_excess_max_nm = 0`` means "inside 100 % of the limit" while the solver
    was actually asked for 85 % -- i.e. the expert has 15 % of headroom the
    metric does not credit it for.
    """
    scale = float(planner.mod.TrajOptConfig().torque_scale)
    assert 0.0 < scale <= 1.0
    for hand, rob in planner.robots.items():
        idx = adapter.ARM_INDEX[hand]
        np.testing.assert_allclose(
            np.asarray(rob.tau_limit, float), scale * scorer.tau_limit[idx],
            rtol=1e-12, atol=1e-12,
        )


def test_drum_scene_frames_agree(planner, scorer):
    """``drums_from_layout`` (pelvis frame) vs ``DrumSceneSpec(frame='pelvis')``."""
    for name, d in planner.drums.items():
        i = scorer.drum_index[name]
        np.testing.assert_allclose(
            np.asarray(d.center, float),
            np.asarray(scorer.scene.head_centers[i], float),
            atol=1e-12,
        )
        ax = np.asarray(d.axis, float)
        ax = ax / np.linalg.norm(ax)
        np.testing.assert_allclose(ax, np.asarray(scorer.scene.axes[i], float), atol=1e-12)
        assert float(d.radius) == pytest.approx(float(scorer.scene.radii[i]))


def test_contact_point_matches_repo_robot(planner, adapter, scorer):
    """Vendored FK contact point == ``robot.contact_position`` at the same q."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for hand in ("left", "right"):
        rob = planner.robots[hand]
        idx = adapter.ARM_INDEX[hand]
        for _ in range(12):
            q17 = np.zeros(N_DOF)
            q10 = rng.uniform(rob.q_lower, rob.q_upper)
            q17[idx] = q10
            for sigma in (0.20, 0.10, 0.05):
                rob.fk(q10)
                p_repo = rob.contact_position(hand, sigma)
                p_mine, _tip, _d = scorer.contact_world(q17, hand, sigma)
                worst = max(worst, float(np.max(np.abs(p_repo - p_mine))))
    print(f"max |contact point delta| = {worst:.3e} m")
    assert worst <= 1e-9


def test_stick_axis_matches_repo_robot(planner, adapter, scorer):
    rng = np.random.default_rng(6)
    worst = 0.0
    for hand in ("left", "right"):
        rob = planner.robots[hand]
        idx = adapter.ARM_INDEX[hand]
        for _ in range(12):
            q17 = np.zeros(N_DOF)
            q10 = rng.uniform(rob.q_lower, rob.q_upper)
            q17[idx] = q10
            rob.fk(q10)
            d_repo = rob.stick_axis(hand)
            _p, _tip, d_mine = scorer.contact_world(q17, hand, 0.2)
            worst = max(worst, float(np.max(np.abs(d_repo - d_mine))))
    print(f"max |stick axis delta| = {worst:.3e}")
    assert worst <= 1e-9


def test_embedding_preserves_solution_and_freezes_idle_arm(planner, adapter, scorer):
    """The 10 -> 17 embedding: solved joints kept, idle arm held at its start."""
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)
    c = next(x for x in cases if x.hand == "right")
    res = planner(c)
    Q = res.Q
    assert Q.shape == (c.n_waypoints, N_DOF)

    idx = adapter.ARM_INDEX[c.hand]
    idle = [i for i in range(N_DOF) if i not in idx]
    assert len(idle) == 7
    # idle arm: constant, and equal to the case's start value (ready pose;
    # generate_grid no longer parks unused joints at URDF zero)
    assert np.allclose(Q[:, idle], c.q_start[None, idle], atol=0)
    # start pose is pinned
    np.testing.assert_allclose(Q[0], c.q_start, atol=1e-9)
    # the solved columns actually move
    assert float(np.abs(np.diff(Q[:, idx], axis=0)).sum()) > 0.1


def test_planner_result_metadata(planner, scorer):
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)
    res = planner(cases[0])
    for k in ("failed", "converged", "iterations", "message", "backend"):
        assert k in res.meta
    assert res.meta["backend"] == "hpipm"
    assert res.solve_s > 0.0


def test_describe_records_the_defaults_in_force(planner, adapter):
    d = adapter.describe(planner)
    t = d["trajopt_effective_after_validate"]
    # the seed the plan calls out explicitly: apex-in-cylinder, apex first.
    # validate() must not touch any seed_* field.
    assert t["seed_mode"] == "apex"
    assert t["seed_apex_first"] is True
    assert t["seed_apex_height"] == pytest.approx(0.30)
    assert t["start_velocity_mode"] == "auto"  # == "dynamic" on the HPIPM backend
    assert t["simplified_cost"] is True
    assert t["enable_torque_limits"] is True
    assert d["waist_policy"] == "free (no reference)"


def test_validate_silently_zeroes_weights(planner, adapter):
    """Gap 8: the declared dataclass defaults are NOT what the solver sees."""
    d = adapter.describe(planner)
    dec = d["trajopt_declared_defaults"]
    eff = d["trajopt_effective_after_validate"]
    changed = {k for k in dec if dec[k] != eff[k]}
    # the ones that matter for reading the baseline
    assert "w_strike_position" in changed and eff["w_strike_position"] == 0.0
    assert "w_stick_elev" in changed and eff["w_stick_elev"] == 0.0
    assert "w_posture" in changed and eff["w_posture"] == 0.0
    assert eff["strike_min_velocity_fraction"] == 0.0
    # ... and the ones that must survive
    for k in ("seed_mode", "seed_apex_first", "seed_apex_height", "torque_scale",
              "limit_margin", "w_strike_velocity", "max_iterations"):
        assert dec[k] == eff[k], k


def test_waist_aim_planner_pins_the_reference(adapter, scorer):
    """v2 policy: yaw ramps to the aim target, pitch/roll hold, ref[0] == q0."""
    try:
        pl = adapter.make_planner(backend="hpipm", waist_aim=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot build the waist-aim planner: {exc}")
    d = adapter.describe(pl)
    assert d["waist_policy"] == "aim"
    assert d["waist_aim"]["damping"] == pytest.approx(0.3)
    assert d["waist_aim"]["yaw_limit_rad"] == pytest.approx(1.74533)

    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)
    moved = 0
    for c in cases[::13]:
        idx = adapter.ARM_INDEX[c.hand]
        q0 = pl.robots[c.hand].clamp(np.asarray(c.q_start, float)[idx])
        w = pl.waist_idx[c.hand]
        yaw_col = pl.yaw_col[c.hand]
        psi = pl._yaw_target(c, q0)
        ref = pl._waist_reference(c, q0, psi, int(c.N))

        assert ref.shape == (c.N + 1, 3)
        # anchored at the start pose -- Block 1 / Block 2 consistency
        np.testing.assert_allclose(ref[0], q0[w], atol=1e-12)
        # pitch and roll held at the start value for the whole horizon
        for col in range(3):
            if col != yaw_col:
                np.testing.assert_allclose(ref[:, col], q0[w][col], atol=1e-12)
        # yaw arrives at the target and is stationary well before contact
        assert ref[-1, yaw_col] == pytest.approx(psi, abs=1e-9)
        settle = max(int(round(pl.waist_settle_s / c.dt)), 0)
        tail = ref[max(len(ref) - settle - 1, 0):, yaw_col]
        np.testing.assert_allclose(tail, psi, atol=1e-9)
        # v_start's waist rows agree with the reference's own first difference
        v = pl._with_waist_bc(c, np.zeros(len(idx)), ref)
        np.testing.assert_allclose(v[w], (ref[1] - ref[0]) / c.dt, atol=1e-12)
        moved += abs(psi - float(q0[w][yaw_col])) > 1e-9
    assert moved > 0, "the aim gate never opened anywhere on the grid"


def test_waist_aim_solution_follows_the_reference(adapter, scorer):
    """The solved trajectory's waist must BE the reference (Block 2 pins it)."""
    try:
        pl = adapter.make_planner(backend="hpipm", waist_aim=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot build the waist-aim planner: {exc}")
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)
    c = cases[0]
    res = pl(c)
    idx = adapter.ARM_INDEX[c.hand]
    q0 = pl.robots[c.hand].clamp(np.asarray(c.q_start, float)[idx])
    ref = pl._waist_reference(c, q0, res.meta["yaw_target_rad"], int(c.N))
    waist17 = [G1_UPPER_BODY_JOINTS.index(n) for n in adapter.ARM_JOINTS[c.hand][:3]]
    np.testing.assert_allclose(res.Q[:, waist17], ref, atol=1e-6)
