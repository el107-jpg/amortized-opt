"""Unit tests for SaLSa-style line-search library (plan §11 tests 1,3,5)."""

from __future__ import annotations

import pytest

from amortized_opt.line_search_step import (
    LineSearchRuntime,
    init_line_search_runtime,
    resolve_known_search,
    run_search_or_ordinary,
)
from amortized_opt.line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    TriggerRefs,
    A_candidate,
    B_candidate,
    accepts_smoothed,
    advance_successful_update,
    apply_pending_history_reset,
    check_early_triggers,
    choose_search_seed,
    clamp_epsilon,
    commit_acceptance,
    curvature_residual,
    effective_beta,
    evaluate_trial,
    force_grad_balance_off,
    line_search_state_dict,
    load_line_search_state_dict,
    logging_dict_from_search,
    on_selection_failure,
    propose_next_epsilon,
    retry_seed_after_failure,
    reset_history_for_objective_change,
    resolve_runtime_bounds,
    objective_reseed_epsilon,
    should_search,
    update_interval_after_search,
)


# ------------------------------------------------------------------- test 1: quadratic
def test_quadratic_probe_slope_and_curvature():
    """F(θ)=0.5 L ‖θ‖² along p: slope and fitted κ match; unsmoothed Armijo."""
    # 1-D: F(ε) = F0 + ε s + 0.5 κ ε² with s=-a, a>0 downhill in probe coords.
    # Equivalently F(θ0+εp)= 0.5 L (θ0+εp)² with p=-1, θ0=1 → F0=0.5L,
    # Fprobe(ε)=0.5 L (1-ε)², decrease = F0-Fprobe = 0.5L(1-(1-ε)²)=0.5L(2ε-ε²)
    # a = -F'(0)*sign for p=-1: F'(θ)=L θ, at 1 is L; p_probe=-1; a=-g·p = L.
    L = 4.0
    F0 = 0.5 * L * 1.0**2
    a = L  # = g · (D g) with D=1, g=L*θ0 if scaled; here a=L
    P2 = 1.0
    eps = 0.25
    Fprobe = 0.5 * L * (1.0 - eps) ** 2
    R, kappa, decrease = curvature_residual(Fprobe, F0, eps, a)
    assert decrease == pytest.approx(0.5 * L * (2 * eps - eps**2))
    assert R == pytest.approx(0.5 * L * eps**2)  # true quadratic residual
    assert kappa == pytest.approx(L)  # κ = F'' = L for unit ‖p‖

    # beta=0 → unsmoothed: accept iff decrease >= c1 * ε * a
    beta_eff = 0.0
    Bc = B_candidate(0.0, a, beta_eff)
    Ac = A_candidate(0.0, decrease, beta_eff)
    assert Bc == pytest.approx(a)
    assert Ac == pytest.approx(decrease)
    c1 = 0.5
    assert accepts_smoothed(Ac, Bc, eps, c1)
    # Too large a step fails Armijo on this quadratic for c1=0.5 when ε big?
    # decrease = 0.5 L (2ε-ε²), need >= c1 ε a = 0.5 ε L → 0.5L(2ε-ε²) >= 0.5 ε L
    # → 2ε-ε² >= ε → ε <= 1. Always true for ε in (0,1]. Check ε=1.5:
    eps_big = 1.5
    Fprobe_big = 0.5 * L * (1.0 - eps_big) ** 2
    _, _, dec_big = curvature_residual(Fprobe_big, F0, eps_big, a)
    Ac_big = A_candidate(0.0, dec_big, 0.0)
    # 0.5*4*(2*1.5-2.25)=2*(3-2.25)=1.5; c1*ε*a=0.5*1.5*4=3 → reject
    assert not accepts_smoothed(Ac_big, a, eps_big, c1)

    # Quadratic proposal after reject clamps into [0.1ε, 0.5ε]
    nxt, eps_quad, _, reason = propose_next_epsilon(
        epsilon=eps_big,
        a=a,
        P2=P2,
        R=eps_big * a - dec_big,
        proposal_rule="quadratic_with_safeguards",
        backtrack_min_factor=0.1,
        backtrack_max_factor=0.5,
        fallback_shrink=0.5,
        epsilon_min=1e-12,
    )
    assert nxt is not None
    assert 0.1 * eps_big - 1e-12 <= nxt <= 0.5 * eps_big + 1e-12
    assert reason == "quadratic_clamp"
    assert eps_quad is not None


def test_propose_next_epsilon_tiny_eps_clamps_to_floor():
    """Denormal proposals are clamped to ``epsilon_min``, not rejected."""
    emin = 1e-10
    nxt, _, _, reason = propose_next_epsilon(
        epsilon=1e-160,
        a=1.0,
        P2=1.0,
        R=1.0,
        proposal_rule="quadratic_with_safeguards",
        backtrack_min_factor=0.1,
        backtrack_max_factor=0.5,
        fallback_shrink=0.5,
        epsilon_min=emin,
    )
    assert nxt == pytest.approx(emin)
    assert reason.endswith("_clamped_to_floor")


def test_evaluate_trial_first_step_unsmoothed():
    cfg = LineSearchConfig(beta_accept_per_update=0.0, c1=0.5, max_trials=6)
    F0, a, P2, eps = 10.0, 2.0, 4.0, 0.5
    # Sufficient decrease: decrease = 2.0 >= c1*eps*a = 0.5
    Fprobe = F0 - 2.0
    d = evaluate_trial(
        Fprobe=Fprobe,
        F0=F0,
        epsilon=eps,
        a=a,
        P2=P2,
        A_prev=0.0,
        B_cand=a,
        beta_eff=0.0,
        c1=cfg.c1,
        max_trials=cfg.max_trials,
        trial_index=0,
    )
    assert d.accept and d.reason == "accepted"


# --------------------------------------------------------------- test 3: smoothing
def test_smoothing_permits_negative_current_decrease():
    """Positive A history can accept a probe that increases current loss."""
    accept = AcceptanceEMAState(A_prev=5.0, B_prev=1.0, n_history=3, last_history_update=10)
    j = 11
    beta_eff, gap = effective_beta(accept.last_history_update, j, 0.99)
    assert gap == 1
    a = 1.0
    Bc = B_candidate(accept.B_prev, a, beta_eff)
    # Current decrease negative (loss rose), but smoothed A still large.
    decrease = -0.5
    Ac = A_candidate(accept.A_prev, decrease, beta_eff)
    eps = 0.1
    c1 = 0.5
    assert Ac > 0
    assert accepts_smoothed(Ac, Bc, eps, c1)

    # Rejected trials must not advance A or B.
    A_before, B_before, n_before = accept.A_prev, accept.B_prev, accept.n_history
    d = evaluate_trial(
        Fprobe=10.0,  # F0=9 → decrease=-1, reject if history weak
        F0=9.0,
        epsilon=1.0,
        a=a,
        P2=1.0,
        A_prev=0.0,  # no history → reject
        B_cand=a,
        beta_eff=0.0,
        c1=c1,
        trial_index=0,
        max_trials=6,
    )
    assert not d.accept
    assert accept.A_prev == A_before
    assert accept.B_prev == B_before
    assert accept.n_history == n_before

    # Commit only once after success.
    commit_acceptance(accept, Ac, Bc, j, beta_eff)
    assert accept.n_history == n_before + 1
    assert accept.last_history_update == j
    assert accept.A_prev == pytest.approx(Ac)
    assert accept.B_prev == pytest.approx(Bc)


def test_current_probe_increase_guard_overrides_smoothed_acceptance():
    """The opt-in raw guard vetoes an increase that history would admit."""
    common = dict(
        Fprobe=10.5,
        F0=10.0,
        epsilon=0.1,
        a=1.0,
        P2=1.0,
        A_prev=5.0,
        B_cand=1.0,
        beta_eff=0.99,
        c1=0.5,
        trial_index=0,
        max_trials=2,
    )
    legacy = evaluate_trial(**common)
    guarded = evaluate_trial(
        **common, reject_current_probe_increase=True,
    )
    assert legacy.accept
    assert not guarded.accept
    assert guarded.reason.startswith("rejected_current_probe_increase_")
    assert guarded.next_epsilon is not None


def test_beta_effective_uses_update_gap():
    beta, gap = effective_beta(None, j=1, beta_accept_per_update=0.99)
    assert gap == 1 and beta == pytest.approx(0.99)
    beta10, gap10 = effective_beta(1, j=11, beta_accept_per_update=0.99)
    assert gap10 == 10
    assert beta10 == pytest.approx(0.99**10)


# ------------------------------------------- test 5: expansion / reset / failure
def test_budget_failure_schedules_history_reset_and_retry_seed():
    cfg = LineSearchConfig(max_trials=2, epsilon_seed=1e-3, fallback_shrink=0.5)
    accept = AcceptanceEMAState(A_prev=1.0, B_prev=2.0, n_history=5, last_history_update=3)
    ctrl = IntervalControllerState(successful_update_count=3, last_selected_epsilon=1e-3, K=4)
    emin, emax = cfg.resolve_bounds()

    # Exhaust trials without accept.
    eps = 1e-3
    last_eps = eps
    for t in range(cfg.max_trials):
        # Always reject: Fprobe >> F0 with beta=0
        d = evaluate_trial(
            Fprobe=100.0,
            F0=1.0,
            epsilon=eps,
            a=1.0,
            P2=1.0,
            A_prev=0.0,
            B_cand=1.0,
            beta_eff=0.0,
            c1=0.5,
            trial_index=t,
            max_trials=cfg.max_trials,
            epsilon_min=emin,
        )
        last_eps = eps
        if d.accept:
            break
        if d.stop:
            break
        assert d.next_epsilon is not None
        eps = d.next_epsilon
    assert not d.accept and d.stop

    # History NOT erased mid-search.
    assert accept.A_prev == 1.0
    on_selection_failure(
        ctrl, accept, last_tried_epsilon=last_eps, fallback_shrink=0.5, epsilon_min=emin,
    )
    assert accept.pending_history_reset
    assert accept.A_prev == 1.0  # still intact
    assert ctrl.K == 1
    assert ctrl.pending_search
    assert ctrl.retry_seed == pytest.approx(max(emin, last_eps))

    # Next iteration applies reset.
    assert apply_pending_history_reset(accept)
    assert accept.A_prev == 0.0 and accept.n_history == 0
    assert accept.last_history_update is None
    assert accept.retention_product == 1.0

    j = ctrl.successful_update_count + 1
    run, reasons = should_search(ctrl, cfg, j)
    assert run
    seed = choose_search_seed(
        ctrl, cfg, epsilon_max=emax, epsilon_min=emin, reasons=reasons,
    )
    assert seed.reason == "retry_seed"
    assert seed.epsilon == pytest.approx(ctrl.retry_seed)
    assert not seed.expand


def test_force_search_on_al_change_defaults_off():
    assert LineSearchConfig().force_search_on_al_change is False
    assert LineSearchConfig().strict_actual_update is False
    assert LineSearchConfig().reject_current_probe_increase is False
    assert LineSearchConfig().skip_on_probe_exhaustion is False
    assert LineSearchConfig().actual_update_diagnostic_every == 0


def test_objective_change_reset_reseeds_epsilon_when_enabled():
    cfg = LineSearchConfig(
        epsilon_seed=1e-4,
        objective_reseed_mult=100.0,
        reseed_epsilon_on_objective_change=True,
    )
    accept = AcceptanceEMAState(A_prev=3.0, B_prev=1.0, n_history=2, last_history_update=5)
    ctrl = IntervalControllerState(
        successful_update_count=5,
        last_search_update=4,
        last_selected_epsilon=1e-10,
        retry_seed=5e-11,
        K=7,
        stable_search_streak=2,
        stability_eligible=True,
    )
    reseed = reset_history_for_objective_change(
        accept, ctrl, cfg=cfg, eta_ref=1e-4,
    )
    assert reseed == pytest.approx(1e-8)
    assert accept.n_history == 0 and accept.A_prev == 0.0
    assert ctrl.K == 1 and ctrl.pending_search
    assert not ctrl.stability_eligible
    assert ctrl.retry_seed is None
    assert ctrl.last_selected_epsilon == pytest.approx(1e-8)
    emin, emax = cfg.resolve_bounds(seed=1e-4)
    seed = choose_search_seed(
        ctrl, cfg, epsilon_max=emax, epsilon_min=emin,
        reasons=["objective_changed", "pending_search"],
    )
    assert seed.reason == "objective_reseed"
    assert seed.epsilon == pytest.approx(1e-8)
    assert not seed.expand


def test_objective_change_reset_keeps_epsilon_by_default():
    cfg = LineSearchConfig(epsilon_seed=1e-4, objective_reseed_mult=100.0)
    accept = AcceptanceEMAState(A_prev=3.0, B_prev=1.0, n_history=2, last_history_update=5)
    ctrl = IntervalControllerState(
        successful_update_count=5,
        last_search_update=4,
        last_selected_epsilon=2.5e-7,
        retry_seed=5e-11,
        K=7,
        stable_search_streak=2,
        stability_eligible=True,
    )
    kept = reset_history_for_objective_change(
        accept, ctrl, cfg=cfg, eta_ref=1e-4,
    )
    assert kept == pytest.approx(2.5e-7)
    assert ctrl.last_selected_epsilon == pytest.approx(2.5e-7)
    emin, emax = cfg.resolve_bounds(seed=1e-4)
    seed = choose_search_seed(
        ctrl, cfg, epsilon_max=emax, epsilon_min=emin,
        reasons=["objective_changed", "pending_search"],
    )
    assert seed.reason == "objective_changed"
    assert seed.epsilon == pytest.approx(2.5e-7)
    assert not seed.expand


def test_objective_reseed_clamps_and_fallback():
    cfg = LineSearchConfig(epsilon_seed=1e-4, objective_reseed_mult=100.0)
    emin, emax = cfg.resolve_bounds(seed=1e-4)
    assert objective_reseed_epsilon(
        previous=1e-5, cfg=cfg, eta_ref=1e-4,
    ) == pytest.approx(emax)
    assert objective_reseed_epsilon(
        previous=None, cfg=cfg, eta_ref=1e-4,
    ) == pytest.approx(1e-4)


def test_expansion_on_normal_interval_search():
    cfg = LineSearchConfig(expansion_factor=1.1, epsilon_seed=1e-4, warmup_updates=0)
    ctrl = IntervalControllerState(
        successful_update_count=20,
        last_search_update=15,
        last_selected_epsilon=1e-3,
        K=5,
        stability_eligible=True,
        pending_search=False,
    )
    j = 21
    run, reasons = should_search(ctrl, cfg, j)
    assert run and "interval" in reasons
    _, emax = cfg.resolve_bounds(1e-3)
    emin, _ = cfg.resolve_bounds(1e-3)
    seed = choose_search_seed(
        ctrl, cfg, epsilon_max=emax, epsilon_min=emin, reasons=reasons,
    )
    assert seed.expand
    assert seed.epsilon == pytest.approx(1.1e-3)


def test_first_search_no_expand_when_last_selected_preset():
    """Init sets last_selected_epsilon before the first search (plan §5)."""
    cfg = LineSearchConfig(expansion_factor=1.1, epsilon_seed=1e-4)
    ctrl = IntervalControllerState(
        last_selected_epsilon=1e-4,
        last_search_update=None,
        pending_search=True,
    )
    emin, emax = cfg.resolve_bounds(1e-4)
    seed = choose_search_seed(
        ctrl, cfg, epsilon_max=emax, epsilon_min=emin,
        reasons=["never_searched", "pending_search", "warmup"],
    )
    assert not seed.expand
    assert seed.epsilon == pytest.approx(1e-4)
    assert seed.reason == "first_search"


def test_resolve_runtime_bounds_and_clamp_epsilon():
    cfg = LineSearchConfig(epsilon_seed=1e-6)
    emin, emax = resolve_runtime_bounds(cfg, 1e-6)
    assert emin == pytest.approx(1e-10)
    assert emax == pytest.approx(1e-5)
    assert clamp_epsilon(1e-19, emin) == pytest.approx(1e-10)
    assert clamp_epsilon(2e-5, emin) == pytest.approx(2e-5)


def test_retry_seed_floored_at_epsilon_min():
    """Failure retry_seed clamps to floor; no post-failure halving below ``emin``."""
    cfg = LineSearchConfig(fallback_shrink=0.5, epsilon_seed=1e-4)
    emin, emax = cfg.resolve_bounds()
    accept = AcceptanceEMAState()
    ctrl = IntervalControllerState(
        successful_update_count=10,
        last_selected_epsilon=2e-6,
        pending_search=True,
    )
    ctrl.pending_trigger_reasons.append("selection_failed")
    on_selection_failure(
        ctrl, accept, last_tried_epsilon=1.5e-10, fallback_shrink=0.5, epsilon_min=emin,
    )
    assert ctrl.retry_seed == pytest.approx(max(emin, 1.5e-10))
    on_selection_failure(
        ctrl, accept, last_tried_epsilon=5e-11, fallback_shrink=0.5, epsilon_min=emin,
    )
    assert ctrl.retry_seed == pytest.approx(emin)
    on_selection_failure(
        ctrl, accept, last_tried_epsilon=emin, fallback_shrink=0.5, epsilon_min=emin,
    )
    assert ctrl.retry_seed == pytest.approx(emin)
    assert retry_seed_after_failure(emin, fallback_shrink=0.5, epsilon_min=emin) == emin
    run, reasons = should_search(ctrl, cfg, ctrl.successful_update_count + 1)
    seed = choose_search_seed(
        ctrl, cfg, epsilon_max=emax, epsilon_min=emin, reasons=reasons,
    )
    assert seed.epsilon == pytest.approx(emin)


def test_interval_promote_and_demote():
    cfg = LineSearchConfig(
        warmup_updates=0,
        stable_searches_to_increase=3,
        stable_rate_ratio_bound=1.15,
        stable_max_trials=2,
        substantial_rate_drop_ratio=0.75,
        costly_search_trials=3,
        min_interval=1,
        max_interval=10,
    )
    ctrl = IntervalControllerState(
        successful_update_count=20,
        last_selected_epsilon=1e-3,
        K=2,
        stability_eligible=True,
    )
    # Stable search → streak
    info = update_interval_after_search(
        ctrl, cfg, epsilon_new=1.05e-3, trial_count=1, j=21
    )
    assert info["stable"] and ctrl.stable_search_streak == 1
    advance_successful_update(ctrl, 21)

    info = update_interval_after_search(
        ctrl, cfg, epsilon_new=1.0e-3, trial_count=2, j=22
    )
    assert ctrl.stable_search_streak == 2
    info = update_interval_after_search(
        ctrl, cfg, epsilon_new=1.02e-3, trial_count=1, j=23
    )
    assert info["K_action"] == "promote"
    assert ctrl.K == 3
    assert ctrl.stable_search_streak == 0

    # Substantial rate drop → K=1
    update_interval_after_search(ctrl, cfg, epsilon_new=0.5e-3, trial_count=1, j=24)
    assert ctrl.K == 1

    # Costly trials demote
    ctrl.K = 8
    update_interval_after_search(ctrl, cfg, epsilon_new=1.0e-3, trial_count=3, j=25)
    assert ctrl.K == 1

    # Nonstable otherwise halves
    ctrl.K = 8
    ctrl.stable_search_streak = 0
    ctrl.stability_eligible = True
    ctrl.last_selected_epsilon = 1e-3
    update_interval_after_search(ctrl, cfg, epsilon_new=1.5e-3, trial_count=1, j=26)
    assert ctrl.K == 4


def test_no_forced_epsilon_min_accept():
    d = evaluate_trial(
        Fprobe=1e9,
        F0=0.0,
        epsilon=1e-9,
        a=1.0,
        P2=1.0,
        A_prev=0.0,
        B_cand=1.0,
        beta_eff=0.0,
        c1=0.5,
        trial_index=5,
        max_trials=6,
        epsilon_min=1e-6,
    )
    assert not d.accept
    assert d.stop
    assert d.next_epsilon is None


def test_checkpoint_roundtrip_and_grad_balance_forced():
    assert force_grad_balance_off("al_pcgrad_output", True) == ""
    assert force_grad_balance_off("al_pcgrad_output", False) == "al_pcgrad_output"

    cfg = LineSearchConfig(grad_balance="al_pcgrad_output", epsilon_seed=2e-4)
    accept = AcceptanceEMAState(A_prev=1.5, B_prev=0.7, n_history=4, last_history_update=9)
    accept.retention_product = 0.9
    ctrl = IntervalControllerState(
        successful_update_count=9,
        last_search_update=9,
        K=3,
        last_selected_epsilon=2e-4,
        stability_eligible=True,
    )
    refs = TriggerRefs(gnorm_ema=1.2, initialized=True)
    blob = line_search_state_dict(
        cfg, accept, ctrl, refs, fixed_group_ratios=[1.0, 1.0], eta_ref=2e-4, seed_provenance="fresh"
    )
    assert blob["grad_balance"] == ""
    cfg2, acc2, ctrl2, refs2 = load_line_search_state_dict(blob)
    assert cfg2.grad_balance == ""
    assert acc2.A_prev == pytest.approx(1.5)
    assert ctrl2.K == 3
    assert refs2.gnorm_ema == pytest.approx(1.2)

    log = logging_dict_from_search(
        F0=1.0, epsilon=2e-4, searched=True, accepted=True, K=3, grad_balance=""
    )
    assert log["grad_balance"] == "" and log["line_search"] == 1


def test_early_triggers_pure():
    cfg = LineSearchConfig()
    refs = TriggerRefs(gnorm_ema=1.0, initialized=True, viol_ema={"torque": 0.1})
    refs.viol_abs_thresholds["torque"] = 0.01
    reasons = check_early_triggers(
        cfg=cfg, refs=refs, gnorm=4.0, viol_metrics={"torque": 0.5}
    )
    assert "grad_norm_spike" in reasons
    assert any(r.startswith("violation_spike:") for r in reasons)

    reasons2 = check_early_triggers(
        cfg=cfg, refs=refs, gnorm=1.1, objective_changed=True
    )
    assert reasons2 == ["objective_changed"]


def test_armijo_a_cap_is_deprecated_and_true_derivative_remains_authoritative():
    cfg = LineSearchConfig(
        c1=0.5, beta_accept_per_update=0.0, armijo_a_cap_gnorm_sq=1.0,
    )
    a_true = 1.0e12
    eps = 1.0e-4
    F0 = 13000.0
    Fprobe = 11000.0  # decrease 2000
    beta_eff, _ = effective_beta(None, 1, cfg.beta_accept_per_update)
    dec_true = evaluate_trial(
        Fprobe=Fprobe,
        F0=F0,
        epsilon=eps,
        a=a_true,
        P2=1.0,
        A_prev=0.0,
        B_cand=B_candidate(0.0, a_true, beta_eff),
        beta_eff=beta_eff,
        c1=cfg.c1,
    )
    assert not dec_true.accept
    assert cfg.armijo_a_cap_gnorm_sq == 1.0  # serialized compatibility only


def test_warmup_forces_search_each_update():
    cfg = LineSearchConfig(warmup_updates=10, initial_interval=10, search_interval_mode="adaptive")
    ctrl = IntervalControllerState(
        successful_update_count=3,
        last_search_update=3,
        last_selected_epsilon=1e-4,
        K=1,
        pending_search=False,
        stability_eligible=True,
    )
    run, reasons = should_search(ctrl, cfg, j=4)
    assert run and "warmup" in reasons


def test_fixed_epsilon_skips_search_scheduler():
    from unittest.mock import MagicMock

    cfg = LineSearchConfig(mode="fixed_epsilon", epsilon_seed=1e-4)
    rt = LineSearchRuntime(
        cfg=cfg,
        accept=AcceptanceEMAState(),
        ctrl=IntervalControllerState(pending_search=True, successful_update_count=0),
        refs=TriggerRefs(),
        adapter=MagicMock(),
        optimizer_kind="lbfgs",
        eta_ref=1e-4,
        epsilon_min=1e-8,
        epsilon_max=1e-3,
        seed_provenance="test",
        fixed_group_ratios=[1.0],
    )
    run, reasons, _j = resolve_known_search(rt)
    assert not run
    assert reasons == []
