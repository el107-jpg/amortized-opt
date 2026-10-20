"""``amortized_opt.strike`` vs the repo original, on randomized inputs.

Repo-only: skips cleanly when ``source/g1_drummer`` is absent.  The strike rules
are tuned, not derived, so a port that drifts silently would put the harness and
the section-3 cost on a different target from the RL hit detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import import_repo  # noqa: E402

import amortized_opt.strike as S  # noqa: E402

TOL = 1e-12  # the brief asks for <= 1e-6; these are literal ports, so demand more

DRUMS = (
    "snare", "hihat", "tom1", "tom2", "cymbal_left", "cymbal_right", "floor_tom",
    "hi_hat", "hi-hat", "ride_cymbal", "kick", None,
)


@pytest.fixture(scope="module")
def repo():
    return import_repo("trajopt.problem")


def _cases(n: int = 240):
    rng = np.random.default_rng(20260810)
    out = []
    for i in range(n):
        name = DRUMS[i % len(DRUMS)]
        center = np.array(
            [rng.uniform(0.1, 0.8), rng.uniform(-0.6, 0.6), rng.uniform(0.6, 1.3)]
        )
        axis = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 1.0])
        axis = axis / np.linalg.norm(axis)
        out.append((name, center, axis, float(rng.uniform(0.08, 0.25))))
    return out


def test_predicates_match(repo):
    for name in DRUMS:
        assert S.is_cymbal_name(name) == repo.is_cymbal_name(name), name
        assert S.is_hihat_name(name) == repo.is_hihat_name(name), name
        assert S.is_tom_name(name) == repo.is_tom_name(name), name
        assert S.shaft_hit_allowed(name) == repo.shaft_hit_allowed(name), name
        assert S.strike_elevation_required(name) == repo.strike_elevation_required(name)


def test_constants_match(repo):
    assert S.STICK_TIP_SIGMA == repo.STICK_TIP_SIGMA
    assert S.HIHAT_SHOULDER_SIGMA == repo.HIHAT_SHOULDER_SIGMA
    assert S.CYMBAL_CONTACT_SIGMA_MIN == repo.CYMBAL_CONTACT_SIGMA_MIN
    assert S.CYMBAL_CONTACT_SIGMA_FALLBACK == repo.CYMBAL_CONTACT_SIGMA_FALLBACK
    assert S.CYMBAL_EDGE_FRACTION == repo.CYMBAL_EDGE_FRACTION
    assert S.HIHAT_RIM_FRACTION == repo.HIHAT_RIM_FRACTION
    assert S.TOM_NEAR_FRACTION == repo.TOM_NEAR_FRACTION
    # the elevation band lives on TrajOptConfig in the repo
    cfg = repo.TrajOptConfig()
    assert S.STICK_ELEV_MIN_DEG == float(cfg.stick_elev_min_deg)
    assert S.STICK_ELEV_MAX_DEG == float(cfg.stick_elev_max_deg)


def test_contact_sigma_match(repo):
    for name in DRUMS:
        assert S.default_contact_sigma(name) == repo.default_contact_sigma(name), name
        assert S.contact_sigma_candidates(name) == repo.contact_sigma_candidates(name), name


def test_aim_and_disk_match(repo):
    worst_aim = worst_r = worst_off = worst_hp = 0.0
    for name, c, a, r in _cases():
        # Ours insets 10 mm by default; the repo still uses 5 mm.  Pin the
        # comparison to the repo inset so this test stays a port check, not a
        # default-value check.
        p_m, R_m = S.strike_aim_on_head(c, a, r, name=name, surface_offset=0.005)
        p_r, R_r = repo.strike_aim_on_head(c, a, r, name=name)
        if S.is_hihat_name(name):
            # Local aim is 20 mm inward; the world-space legal disk is unchanged.
            t = S.strike_half_plane_dir(c, a, name=name)
            np.testing.assert_allclose(p_m, p_r-S.HIHAT_AIM_INSET_M*t, atol=1e-12)
            p_m = p_m+S.HIHAT_AIM_INSET_M*t
        worst_aim = max(worst_aim, float(np.max(np.abs(p_m - p_r))))
        worst_r = max(worst_r, abs(float(R_m) - float(R_r)))

        o_m = S.strike_disk_center_offset(c, a, r, name=name)
        o_r = repo.strike_disk_center_offset(c, a, r, name=name)
        if S.is_hihat_name(name):
            o_m = o_m-S.HIHAT_AIM_INSET_M*t
        assert (o_m is None) == (o_r is None), name
        if o_m is not None:
            worst_off = max(worst_off, float(np.max(np.abs(o_m - o_r))))

        t_m = S.strike_half_plane_dir(c, a, name=name)
        t_r = repo.strike_half_plane_dir(c, a, name=name)
        assert (t_m is None) == (t_r is None), name
        if t_m is not None:
            worst_hp = max(worst_hp, float(np.max(np.abs(t_m - t_r))))
    print(f"aim {worst_aim:.3e}  radius {worst_r:.3e}  "
          f"offset {worst_off:.3e}  half-plane {worst_hp:.3e}")
    assert max(worst_aim, worst_r, worst_off, worst_hp) <= TOL


def test_velocity_weight_match(repo):
    rng = np.random.default_rng(7)
    cfg = repo.TrajOptConfig()
    worst = 0.0
    for i in range(120):
        v = rng.normal(size=3) * (0.0 if i % 20 == 0 else rng.uniform(0.1, 4.0))
        W_r = repo.strike_velocity_weight(cfg, v, drum=DRUMS[i % len(DRUMS)])
        W_m = S.strike_velocity_weight(
            v,
            w_strike_velocity=float(cfg.w_strike_velocity),
            a_normal=float(cfg.w_strike_velocity_normal),
            a_tangential=float(cfg.w_strike_velocity_tangential),
            drum=DRUMS[i % len(DRUMS)],
        )
        worst = max(worst, float(np.max(np.abs(W_m - W_r))))
    print(f"velocity weight {worst:.3e}")
    assert worst <= TOL * max(1.0, float(cfg.w_strike_velocity))


def test_strikespec_resolution_matches(repo):
    rng = np.random.default_rng(11)
    for name in DRUMS:
        for sigma in (None, 0.2, 0.15, 0.10, 0.05, -0.5, 9.0, float("nan")):
            kw = dict(
                time=0.5, hand="right",
                position=rng.normal(size=3), velocity=rng.normal(size=3),
                drum=name, contact_sigma=sigma,
            )
            a = S.StrikeSpec(**kw)
            b = repo.StrikeSpec(**kw)
            assert a.resolved_contact_sigma() == b.resolved_contact_sigma(), (name, sigma)
            assert a.is_shaft_contact() == b.is_shaft_contact(), (name, sigma)
            assert a.waypoint(0.02) == b.waypoint(0.02)
            assert a.resolved_hit_radius(0.08) == b.resolved_hit_radius(0.08)


def test_elevation_matches_repo_robot(repo):
    """The elevation band is a robot method upstream; check the formula agrees."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        n = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 1.0])
        n /= np.linalg.norm(n)
        ours = S.stick_elevation_from_surface(d, n)
        theirs = float(np.arcsin(np.clip(-float(d @ n), -1.0, 1.0)))
        assert abs(ours - theirs) <= TOL
        cfg = repo.TrajOptConfig()
        band = (
            np.deg2rad(float(cfg.stick_elev_min_deg)) - 1e-3
            <= abs(theirs)
            <= np.deg2rad(float(cfg.stick_elev_max_deg)) + 1e-3
        )
        assert S.stick_elevation_ok(d, n) == band
