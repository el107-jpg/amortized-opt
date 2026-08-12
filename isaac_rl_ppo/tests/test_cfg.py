"""Config invariants: checkpoint resolution and the lead convention.

Both encode decisions that cost a training run if they silently go wrong:
selecting the known-regressed phase2c checkpoint, or training against a lead
convention that differs from the deployed one.
"""

from __future__ import annotations

import json

import pytest

from isaac_rl_ppo.cfg import BLOCKLIST, PpoRunConfig, resolve_checkpoint


def _make_run(tmp_path, name: str, summary: dict | None = None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "best.pt").write_bytes(b"stub")
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    return d


# ================================================= checkpoint resolution ======
def test_resolves_to_phase2b_when_phase2d_absent(tmp_path):
    _make_run(tmp_path, "phase2b_main")
    got = resolve_checkpoint(runs_dir=tmp_path)
    assert got.run == "phase2b_main"
    assert "champion" in got.reason


def test_phase2d_needs_an_explicit_acceptance_mark(tmp_path):
    """File existence is NOT enough -- two legs write into that directory."""
    _make_run(tmp_path, "phase2b_main")
    _make_run(tmp_path, "phase2d_waist")            # present, unmarked
    got = resolve_checkpoint(runs_dir=tmp_path)
    assert got.run == "phase2b_main", "unmarked phase2d must not be selected"
    why = dict(got.considered)["phase2d_waist"]
    assert "NOT marked accepted" in why


def test_phase2d_selected_when_marked_accepted(tmp_path):
    _make_run(tmp_path, "phase2b_main")
    _make_run(tmp_path, "phase2d_waist", summary={"accepted": True})
    got = resolve_checkpoint(runs_dir=tmp_path)
    assert got.run == "phase2d_waist"
    assert "accepted" in got.reason


def test_phase2d_explicitly_rejected_falls_back(tmp_path):
    _make_run(tmp_path, "phase2b_main")
    _make_run(tmp_path, "phase2d_waist", summary={"accepted": False})
    got = resolve_checkpoint(runs_dir=tmp_path)
    assert got.run == "phase2b_main"


def test_phase2d_verdict_string_is_honoured(tmp_path):
    _make_run(tmp_path, "phase2b_main")
    _make_run(tmp_path, "phase2d_waist", summary={"verdict": "trade ACCEPTED"})
    assert resolve_checkpoint(runs_dir=tmp_path).run == "phase2d_waist"


def test_phase2c_is_never_resolved(tmp_path):
    """phase2c is known-regressed; it must not be reachable by resolution."""
    _make_run(tmp_path, "phase2c_main", summary={"accepted": True})
    got = resolve_checkpoint(runs_dir=tmp_path)
    assert got.run is None, "phase2c must never be auto-selected"
    assert "phase2c_main" in BLOCKLIST


def test_explicit_phase2c_path_is_refused(tmp_path):
    d = _make_run(tmp_path, "phase2c_main")
    with pytest.raises(ValueError, match="known-regressed"):
        resolve_checkpoint(d / "best.pt")


def test_explicit_path_wins(tmp_path):
    _make_run(tmp_path, "phase2b_main")
    d = _make_run(tmp_path, "some_other_run")
    got = resolve_checkpoint(d / "best.pt", runs_dir=tmp_path)
    assert got.run == "some_other_run"
    assert got.reason == "explicit --ckpt"


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "nope" / "best.pt")


def test_real_repo_resolution_is_phase2b():
    """Against the ACTUAL runs/ directory, today."""
    got = resolve_checkpoint()
    assert got.run in ("phase2b_main", "phase2d_waist")
    assert got.run != "phase2c_main"


# ====================================================== the lead convention ===
def test_isaac_regime_loads_the_published_fit():
    """The Isaac default resolves to RL-1b's measured vector, from the file."""
    cfg = PpoRunConfig()
    assert cfg.env.plant_regime == "isaac"
    assert cfg.env.cmd_lead_s is None                # -> resolved, not hardcoded
    cfg.validate()

    vec, source = cfg.resolve_lead()
    assert len(vec) == 17
    assert "lead_isaac.json" in source and "best_strike_s" in source
    # the wrists are the signature of the ISAAC fit: the analytic vector is
    # flat-zero there, this one is not (RL-1b: 0.0 -> 38.8 ms median)
    wrists = [vec[i] for i in (7, 8, 9, 14, 15, 16)]
    assert max(wrists) > 0.01, f"wrist leads {wrists} look like the analytic fit"


def test_analytic_preset_is_refused_in_isaac():
    """'fit' is the pd_sim vector; on PhysX the wrist entries invert."""
    cfg = PpoRunConfig()
    cfg.env.cmd_lead_s = "fit"
    with pytest.raises(ValueError, match="ANALYTIC preset"):
        cfg.validate()


def test_lead_can_be_disabled_explicitly():
    cfg = PpoRunConfig()
    cfg.env.cmd_lead_s = 0.0
    cfg.validate()
    assert cfg.resolve_lead()[0] == 0.0


def test_missing_lead_file_names_the_publisher(tmp_path):
    cfg = PpoRunConfig()
    cfg.env.lead_json = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="fit_lead_isaac"):
        cfg.resolve_lead()


def test_bad_lead_key_is_reported(tmp_path):
    p = tmp_path / "lead.json"
    p.write_text(json.dumps({"best_strike_s": [0.0] * 17}))
    cfg = PpoRunConfig()
    cfg.env.lead_json = str(p)
    cfg.env.lead_key = "typo"
    with pytest.raises(KeyError, match="typo"):
        cfg.resolve_lead()


def test_analytic_regime_resolves_the_fit_preset():
    cfg = PpoRunConfig()
    cfg.env.plant_regime = "analytic"
    cfg.validate()
    assert cfg.resolve_lead()[0] == "fit"


def test_unknown_plant_regime_is_refused():
    cfg = PpoRunConfig()
    cfg.env.plant_regime = "mujoco"
    with pytest.raises(ValueError, match="plant_regime"):
        cfg.validate()


def test_aux_gate_defaults_to_auto():
    """The gate opens without a code change when RL-1b's RNEA fix lands."""
    cfg = PpoRunConfig()
    assert cfg.aux.in_isaac == "auto"
    assert cfg.aux.enabled


def test_extra_lead_mechanisms_are_refused():
    """README 7c decision 2: the lead ALONE; the 3-mechanism stack is worse."""
    for field, value in (("hit_lead_s", 0.02), ("freeze_wp", 3)):
        cfg = PpoRunConfig()
        setattr(cfg.env, field, value)
        with pytest.raises(ValueError, match="hit_lead_s / freeze_wp"):
            cfg.validate()


def test_validate_rejects_a_bad_sigma_init():
    cfg = PpoRunConfig()
    cfg.actor.noise_std_init = 2.0                   # above noise_std_max
    with pytest.raises(ValueError, match="noise_std_init"):
        cfg.validate()


def test_track_reference_choice_is_checked():
    cfg = PpoRunConfig()
    cfg.track_reference = "led"
    with pytest.raises(ValueError, match="track_reference"):
        cfg.validate()


def test_plan_cost_reward_is_off_by_default():
    """The analytic cost is priced ONCE, as a differentiable aux loss."""
    cfg = PpoRunConfig()
    assert cfg.reward_plan_cost_weight == 0.0
    assert cfg.aux.enabled and cfg.aux.weight > 0.0


# ================================================ the shared-FK trap ==========
def test_a_default_chainfk_cannot_be_shared_with_fusedcost():
    """A general-purpose ``ChainFK`` lacks the frames ``FusedCost`` needs.

    Regression test for a real smoke-run failure: passing the scorer's
    ``ChainFK`` into ``FusedCost(fk=...)`` to 'share the FK pass' raises
    ``KeyError('left_shoulder')`` on the first reward call, because the cost's
    posture term reads shoulder frames that a default ChainFK does not build.
    The entry point therefore lets ``FusedCost`` build its own.

    This is asserted rather than commented because the sharing looks like an
    obvious optimisation and costs a full Isaac startup to rediscover.
    """
    import amortized_opt as A
    from amortized_opt.cost import FusedCost

    default_frames = set(A.ChainFK(device="cpu")._chains)
    needed = set(FusedCost.FRAMES)
    missing = needed - default_frames
    assert missing, (
        "a default ChainFK now carries every frame FusedCost needs; if that is "
        "intentional, train_ppo.py may share one instance again")
    assert any("shoulder" in m for m in missing)
