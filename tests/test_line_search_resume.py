"""Unit tests for SaLSa resume LR migration + checkpoint pack/unpack (§7 / §11)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from amortized_opt.line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    TriggerRefs,
)
from amortized_opt.line_search_resume import (
    has_line_search_selector_extra,
    logging_dict_for_seed_resolution,
    optimizer_has_adam_state,
    pack_line_search_checkpoint_extra,
    read_last_effective_lrs,
    reference_lr_from_lrs,
    resolve_epsilon_seed,
    unpack_line_search_checkpoint_extra,
)
from amortized_opt.line_search_step import (
    build_objective_direction_fingerprint,
    init_line_search_runtime,
    runtime_state_dict,
)
from scripts.train_phase2 import _line_search_resume_policy, _resume_training_states


def _tiny_adam(lr: float = 1e-4) -> torch.optim.AdamW:
    m = nn.Linear(2, 2, bias=True)
    return torch.optim.AdamW(m.parameters(), lr=lr)


def _phase2_stub(
    *,
    torque_physics: str = "full",
    ls_cfg: LineSearchConfig | None = None,
):
    return SimpleNamespace(
        lr=1e-4,
        optimizer="adamw",
        torque_grad_cap=0.2,
        torque_grad_cap_refresh=1,
        torque_grad_cap_scale_ema=0.8,
        grad_clip=1.0,
        cost=SimpleNamespace(torque_physics=torque_physics),
        line_search_cfg=ls_cfg or LineSearchConfig(),
    )


def _prime_adam_state(opt: torch.optim.AdamW) -> None:
    """One dummy step so ``opt.state`` is non-empty (legacy-resume signal)."""
    for g in opt.param_groups:
        for p in g["params"]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_fresh_selector_resume_keeps_absolute_activation_and_defaults_safety():
    delay, strict, reject, skip = _line_search_resume_policy(
        resume_step=32_000,
        line_search_fresh=True,
        activate_after_updates=10_000,
        strict_actual_update=None,
        reject_current_probe_increase=None,
        skip_on_probe_exhaustion=None,
        selector_update_count=31_950,
    )
    assert delay == 0
    assert strict and reject and skip


def test_fresh_selector_resume_subtracts_prior_selector_clock_from_delay():
    delay, strict, reject, skip = _line_search_resume_policy(
        resume_step=32_000,
        line_search_fresh=True,
        activate_after_updates=40_000,
        strict_actual_update=None,
        reject_current_probe_increase=None,
        skip_on_probe_exhaustion=None,
        selector_update_count=31_500,
    )
    assert delay == 8_500
    assert strict and reject and skip


def test_new_training_retains_delay_and_legacy_safety_defaults():
    delay, strict, reject, skip = _line_search_resume_policy(
        resume_step=0,
        line_search_fresh=False,
        activate_after_updates=10_000,
        strict_actual_update=None,
        reject_current_probe_increase=None,
        skip_on_probe_exhaustion=None,
    )
    assert delay == 10_000
    assert not strict and not reject and not skip


def test_fresh_resume_policy_honors_explicit_safety_overrides():
    delay, strict, reject, skip = _line_search_resume_policy(
        resume_step=32_000,
        line_search_fresh=True,
        activate_after_updates=10_000,
        strict_actual_update=False,
        reject_current_probe_increase=False,
        skip_on_probe_exhaustion=False,
    )
    assert delay == 0
    assert not strict and not reject and not skip


def test_reset_optimizer_plus_fresh_selector_keeps_resume_payload_but_not_state():
    source = nn.Linear(2, 2)
    old_opt = torch.optim.AdamW(source.parameters(), lr=8e-7)
    _prime_adam_state(old_opt)
    old_selector = {
        "cfg": {
            "epsilon_seed": 8e-7,
            "strict_actual_update": False,
            "reject_current_probe_increase": False,
        },
        "eta_ref": 8e-7,
    }
    checkpoint = {
        "model": source.state_dict(),
        "al": {"lam": {"torque": 7.0}},
        "optimizer": old_opt.state_dict(),
        "step": 12_345,
        "extra": {"line_search": old_selector},
    }

    al_state, opt_state, start_step, selector_state = _resume_training_states(
        checkpoint,
        reset_optimizer=True,
        restore_line_search=False,
    )
    resumed = nn.Linear(2, 2)
    resumed.load_state_dict(checkpoint["model"])
    assert all(
        torch.equal(a, b)
        for a, b in zip(source.parameters(), resumed.parameters())
    )
    assert al_state == checkpoint["al"]
    assert start_step == 12_345
    assert opt_state is None
    assert selector_state is None
    assert checkpoint["optimizer"] is not None  # resolver is non-mutating
    _, kept_opt, kept_step, kept_selector = _resume_training_states(
        checkpoint,
        reset_optimizer=False,
        restore_line_search=True,
    )
    assert kept_opt is checkpoint["optimizer"]
    assert kept_step == 12_345
    assert kept_selector is old_selector

    fresh_opt = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    assert fresh_opt.state == {}
    current_ls = LineSearchConfig(
        epsilon_seed=3e-6,
        epsilon_min=1e-7,
        epsilon_max=9e-6,
        strict_actual_update=True,
        reject_current_probe_increase=True,
    )
    runtime = init_line_search_runtime(
        phase2_cfg=_phase2_stub(ls_cfg=current_ls),
        optimizer=fresh_opt,
        line_search_state=selector_state,
        opt_restored=False,
    )
    assert runtime.eta_ref == pytest.approx(3e-6)
    assert runtime.epsilon_min == pytest.approx(1e-7)
    assert runtime.epsilon_max == pytest.approx(9e-6)
    assert runtime.seed_provenance == "fresh_epsilon_seed"
    assert runtime.cfg.strict_actual_update is True
    assert runtime.cfg.reject_current_probe_increase is True
    assert fresh_opt.state == {}


def test_fresh_uses_cfg_lr():
    cfg = LineSearchConfig(epsilon_seed=9e-4)
    opt = _tiny_adam(lr=3e-4)
    assert not optimizer_has_adam_state(opt)
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg,
        cfg_lr=3e-4,
        optimizer=opt,
        has_selector_ckpt=False,
        explicit_override=None,
    )
    assert seed == pytest.approx(3e-4)
    assert prov == "fresh_cfg_lr"


def test_fresh_falls_back_to_epsilon_seed_when_cfg_lr_invalid():
    cfg = LineSearchConfig(epsilon_seed=2e-4)
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg,
        cfg_lr=0.0,
        optimizer=None,
        has_selector_ckpt=False,
        explicit_override=None,
    )
    assert seed == pytest.approx(2e-4)
    assert prov == "fresh_epsilon_seed"


def test_explicit_override_wins():
    cfg = LineSearchConfig(epsilon_seed=1e-4, legacy_resume_seed=5e-5)
    opt = _tiny_adam(lr=1e-6)
    _prime_adam_state(opt)
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg,
        cfg_lr=1e-4,
        optimizer=opt,
        has_selector_ckpt=True,
        explicit_override=7e-5,
    )
    assert seed == pytest.approx(7e-5)
    assert prov == "explicit_override"


def test_selector_checkpoint_uses_cfg_epsilon_seed():
    cfg = LineSearchConfig(epsilon_seed=4e-5)
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg,
        cfg_lr=1e-4,
        optimizer=None,
        has_selector_ckpt=True,
        explicit_override=None,
    )
    assert seed == pytest.approx(4e-5)
    assert prov == "selector_checkpoint"


def test_legacy_resume_reads_optimizer_lr_not_cfg_lr():
    cfg = LineSearchConfig(epsilon_seed=1e-4)
    opt = _tiny_adam(lr=1e-4)
    _prime_adam_state(opt)
    for g in opt.param_groups:
        g["lr"] = 5e-6  # late-run cosine floor
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg,
        cfg_lr=1e-4,
        optimizer=opt,
        has_selector_ckpt=False,
        explicit_override=None,
    )
    assert seed == pytest.approx(5e-6)
    assert prov == "legacy_optimizer_lr"
    assert seed != pytest.approx(cfg.epsilon_seed)


def test_legacy_resume_seed_config_field():
    cfg = LineSearchConfig(epsilon_seed=1e-4, legacy_resume_seed=8e-6)
    opt = _tiny_adam(lr=1e-4)
    _prime_adam_state(opt)
    for g in opt.param_groups:
        g["lr"] = 5e-6
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg,
        cfg_lr=1e-4,
        optimizer=opt,
        has_selector_ckpt=False,
        explicit_override=None,
    )
    assert seed == pytest.approx(8e-6)
    assert prov == "legacy_resume_seed_config"


def test_legacy_missing_positive_lr_fails_clearly():
    cfg = LineSearchConfig()
    opt = _tiny_adam(lr=1e-4)
    _prime_adam_state(opt)
    for g in opt.param_groups:
        g["lr"] = 0.0
    with pytest.raises(ValueError, match="explicit_override|non-positive|missing"):
        resolve_epsilon_seed(
            line_search_cfg=cfg,
            cfg_lr=1e-4,
            optimizer=opt,
            has_selector_ckpt=False,
            explicit_override=None,
        )


def test_reference_lr_ignores_frozen_zero_groups():
    assert reference_lr_from_lrs([0.0, 2e-5, 1e-5]) == pytest.approx(2e-5)
    with pytest.raises(ValueError, match="no positive"):
        reference_lr_from_lrs([0.0, float("nan")])


def test_pack_unpack_roundtrip_stores_lrs_and_grad_balance():
    cfg = LineSearchConfig(grad_balance="al_pcgrad_output", epsilon_seed=2e-4)
    accept = AcceptanceEMAState(A_prev=1.1, B_prev=0.5, n_history=2)
    ctrl = IntervalControllerState(
        successful_update_count=7,
        last_search_update=7,
        K=2,
        last_selected_epsilon=2e-4,
    )
    refs = TriggerRefs(gnorm_ema=0.3, initialized=True)
    opt = _tiny_adam(lr=2e-4)
    for g in opt.param_groups:
        g["lr"] = 1.5e-4

    frag = pack_line_search_checkpoint_extra(
        cfg,
        accept,
        ctrl,
        refs,
        fixed_group_ratios=[1.0],
        eta_ref=1.5e-4,
        seed_provenance="legacy_optimizer_lr",
        optimizer=opt,
        objective_direction_fingerprint={"contract": "test-v2"},
    )
    assert "line_search" in frag
    assert frag["line_search"]["grad_balance"] == ""
    assert frag["line_search"]["last_effective_lrs"] == pytest.approx(
        read_last_effective_lrs(opt)
    )
    assert frag["line_search"]["seed_provenance"] == "legacy_optimizer_lr"
    assert has_line_search_selector_extra(frag)

    cfg2, acc2, ctrl2, refs2, meta = unpack_line_search_checkpoint_extra(frag)
    assert cfg2.grad_balance == ""
    assert acc2.A_prev == pytest.approx(1.1)
    assert ctrl2.K == 2
    assert refs2.gnorm_ema == pytest.approx(0.3)
    assert meta["grad_balance"] == ""
    assert meta["seed_provenance"] == "legacy_optimizer_lr"
    assert meta["eta_ref"] == pytest.approx(1.5e-4)
    assert meta["last_effective_lrs"] == pytest.approx([1.5e-4])
    assert meta["fixed_group_ratios"] == pytest.approx([1.0])
    assert meta["objective_direction_fingerprint"] == {"contract": "test-v2"}


def test_resume_same_fingerprint_preserves_acceptance_history():
    phase = _phase2_stub()
    fp = build_objective_direction_fingerprint(
        phase, al_lam={"torque": 2.0}, al_pen={"torque": 4.0}, stage_a=True,
    )
    rt = init_line_search_runtime(
        phase2_cfg=phase,
        optimizer=_tiny_adam(),
        objective_direction_fingerprint=fp,
    )
    rt.accept.A_prev = 3.0
    rt.accept.B_prev = 2.0
    rt.accept.n_history = 4
    rt.ctrl.K = 5
    rt.ctrl.pending_search = False
    blob = runtime_state_dict(rt)

    resumed = init_line_search_runtime(
        phase2_cfg=phase,
        optimizer=_tiny_adam(),
        line_search_state=blob,
        objective_direction_fingerprint=fp,
    )
    assert resumed.fingerprint_reset_reason is None
    assert resumed.accept.A_prev == pytest.approx(3.0)
    assert resumed.accept.n_history == 4
    assert resumed.ctrl.K == 5


def test_curvature_lookahead_changes_direction_fingerprint_only_when_enabled():
    phase = _phase2_stub()
    legacy = build_objective_direction_fingerprint(phase)
    assert "terminal_lookahead_mode" not in legacy
    phase.terminal_lookahead_mode = "curvature"
    phase.curvature_damping = 1.0
    phase.curvature_position_weight = 40.0
    phase.curvature_velocity_weight = 20.0
    phase.curvature_max_step_rad = 0.15
    enabled = build_objective_direction_fingerprint(phase)
    assert enabled != legacy
    phase.curvature_damping = 2.0
    assert build_objective_direction_fingerprint(phase) != enabled


def test_direct_curvature_changes_direction_fingerprint_without_changing_legacy():
    phase = _phase2_stub()
    legacy = build_objective_direction_fingerprint(phase)
    phase.curvature_direct_frac = 0.0
    assert build_objective_direction_fingerprint(phase) == legacy
    phase.curvature_direct_frac = 0.25
    phase.curvature_direct_mix = 1.0
    phase.curvature_damping = 1.0
    phase.curvature_position_weight = 40.0
    phase.curvature_velocity_weight = 20.0
    enabled = build_objective_direction_fingerprint(phase)
    assert enabled != legacy
    phase.curvature_direct_mix = 0.5
    assert build_objective_direction_fingerprint(phase) != enabled


@pytest.mark.parametrize("change", ["torque_physics", "al", "stage_a"])
def test_resume_objective_fingerprint_change_resets_selector(change):
    old_phase = _phase2_stub(torque_physics="static")
    old_fp = build_objective_direction_fingerprint(
        old_phase,
        al_lam={"torque": 2.0},
        al_pen={"torque": 4.0},
        stage_a=False,
    )
    rt = init_line_search_runtime(
        phase2_cfg=old_phase,
        optimizer=_tiny_adam(),
        objective_direction_fingerprint=old_fp,
    )
    rt.accept.A_prev = 3.0
    rt.accept.B_prev = 2.0
    rt.accept.n_history = 4
    rt.ctrl.K = 7
    rt.ctrl.pending_search = False
    blob = runtime_state_dict(rt)

    new_phase = _phase2_stub(
        torque_physics="full" if change == "torque_physics" else "static"
    )
    new_fp = build_objective_direction_fingerprint(
        new_phase,
        al_lam={"torque": 3.0 if change == "al" else 2.0},
        al_pen={"torque": 4.0},
        stage_a=(change == "stage_a"),
    )
    resumed = init_line_search_runtime(
        phase2_cfg=new_phase,
        optimizer=_tiny_adam(),
        line_search_state=blob,
        objective_direction_fingerprint=new_fp,
    )
    assert resumed.fingerprint_reset_reason == "objective_fingerprint_changed"
    assert resumed.accept.A_prev == 0.0
    assert resumed.accept.B_prev == 0.0
    assert resumed.accept.n_history == 0
    assert resumed.ctrl.K == 1
    assert resumed.ctrl.pending_search
    assert "objective_fingerprint_changed" in resumed.ctrl.pending_trigger_reasons


def test_resume_current_safety_config_overrides_checkpoint_and_resets():
    old_phase = _phase2_stub(ls_cfg=LineSearchConfig(
        strict_actual_update=False,
        reject_current_probe_increase=False,
        skip_on_probe_exhaustion=False,
        actual_update_diagnostic_every=0,
    ))
    old_rt = init_line_search_runtime(
        phase2_cfg=old_phase, optimizer=_tiny_adam(),
    )
    old_rt.accept.A_prev = 1.0
    blob = runtime_state_dict(old_rt)

    current_phase = _phase2_stub(ls_cfg=LineSearchConfig(
        strict_actual_update=True,
        reject_current_probe_increase=True,
        skip_on_probe_exhaustion=True,
        actual_update_diagnostic_every=7,
        force_search_on_al_change=True,
        reseed_epsilon_on_objective_change=True,
    ))
    resumed = init_line_search_runtime(
        phase2_cfg=current_phase,
        optimizer=_tiny_adam(),
        line_search_state=blob,
    )
    assert resumed.cfg.strict_actual_update
    assert resumed.cfg.reject_current_probe_increase
    assert resumed.cfg.skip_on_probe_exhaustion
    assert resumed.cfg.actual_update_diagnostic_every == 7
    assert resumed.cfg.force_search_on_al_change
    assert resumed.cfg.reseed_epsilon_on_objective_change
    assert resumed.accept.A_prev == 0.0
    assert resumed.ctrl.K == 1 and resumed.ctrl.pending_search


def test_resume_current_fixed_three_step_cadence_overrides_checkpoint():
    old_phase = _phase2_stub(ls_cfg=LineSearchConfig(
        search_interval_mode="adaptive",
        initial_interval=1,
        min_interval=1,
        max_interval=10,
        warmup_updates=10,
        search_on_early_triggers=True,
    ))
    old_rt = init_line_search_runtime(
        phase2_cfg=old_phase, optimizer=_tiny_adam(),
    )
    blob = runtime_state_dict(old_rt)

    current_phase = _phase2_stub(ls_cfg=LineSearchConfig(
        search_interval_mode="fixed",
        initial_interval=3,
        min_interval=3,
        max_interval=3,
        warmup_updates=0,
        search_on_early_triggers=False,
    ))
    resumed = init_line_search_runtime(
        phase2_cfg=current_phase,
        optimizer=_tiny_adam(),
        line_search_state=blob,
    )
    assert resumed.cfg.search_interval_mode == "fixed"
    assert resumed.cfg.initial_interval == 3
    assert resumed.cfg.min_interval == 3
    assert resumed.cfg.max_interval == 3
    assert resumed.cfg.warmup_updates == 0
    assert not resumed.cfg.search_on_early_triggers


def test_resume_legacy_selector_without_fingerprint_resets_safely():
    phase = _phase2_stub()
    rt = init_line_search_runtime(phase2_cfg=phase, optimizer=_tiny_adam())
    rt.accept.A_prev = 1.0
    rt.accept.n_history = 2
    blob = runtime_state_dict(rt)
    blob.pop("objective_direction_fingerprint")

    resumed = init_line_search_runtime(
        phase2_cfg=phase,
        optimizer=_tiny_adam(),
        line_search_state=blob,
    )
    assert resumed.fingerprint_reset_reason == "objective_fingerprint_missing"
    assert resumed.accept.n_history == 0
    assert resumed.ctrl.K == 1 and resumed.ctrl.pending_search


def test_unpack_missing_selector_raises():
    with pytest.raises(KeyError, match="line_search"):
        unpack_line_search_checkpoint_extra({})
    assert not has_line_search_selector_extra(None)
    assert not has_line_search_selector_extra({"other": 1})


def test_logging_dict_for_seed_resolution():
    d = logging_dict_for_seed_resolution(
        5e-6,
        "legacy_optimizer_lr",
        last_effective_lrs=[5e-6, 0.0],
        eta_ref=5e-6,
        K=1,
    )
    assert d["grad_balance"] == ""
    assert d["line_search"] == 1
    assert d["ls_epsilon_seed"] == pytest.approx(5e-6)
    assert d["ls_seed_provenance"] == "legacy_optimizer_lr"
    assert d["ls_eta_ref"] == pytest.approx(5e-6)
    assert d["ls_last_effective_lrs"] == pytest.approx([5e-6, 0.0])
    assert d["ls_K"] == 1
