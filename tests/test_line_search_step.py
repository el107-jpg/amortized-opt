"""Integration smoke tests for SaLSa train-loop wiring helpers."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from amortized_opt.adamw_ls import build_adamw_like_train_cost
from amortized_opt.line_search import LineSearchConfig
from amortized_opt.line_search_step import (
    init_line_search_runtime,
    needs_line_search_direction,
    resolve_known_search,
    run_search_or_ordinary,
    snapshot_g_preclip,
)
from amortized_opt.objective_replay import (
    FrozenControls,
    ObjectiveReplayContext,
    snapshot_mutable_batch_fields,
    snapshot_torch_generator,
    restore_mutable_batch_fields,
)
from amortized_opt.train_cost import Phase2Config, _line_search_scheduled_rate


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, -0.5]))
        self.b = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return (self.w * x + self.b).pow(2).mean()


class _ScalarObjective(nn.Module):
    def __init__(self, value: float = 1.0, target: float = 0.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([value]))
        self.target = float(target)

    def objective(self):
        return 0.5 * (self.w - self.target).pow(2).sum()


class _DecayUphillObjective(nn.Module):
    def __init__(self):
        super().__init__()
        # Matrix shape keeps this parameter in the optimizer's decay group.
        self.w = nn.Parameter(torch.tensor([[5.0]]))

    def objective(self):
        return 0.5 * (self.w - 10.0).pow(2).sum()


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for av, ev in zip(actual, expected):
            _assert_nested_equal(av, ev)
    else:
        assert actual == expected


def test_init_runtime_fresh_bypasses_cosine_seed():
    m = _Tiny()
    cfg = Phase2Config(line_search=True, lr=3e-4, grad_balance="")
    # Factory epsilon_seed is 1e-4; should inherit cfg.lr.
    opt = build_adamw_like_train_cost(m.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt, opt_restored=False)
    assert abs(rt.eta_ref - 3e-4) < 1e-12
    assert rt.seed_provenance == "fresh_cfg_lr"
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12


def test_init_runtime_forces_grad_balance_empty():
    m = _Tiny()
    cfg = Phase2Config(line_search=True, lr=1e-4, grad_balance="")
    cfg.line_search_cfg.grad_balance = "al_pcgrad_output"
    opt = build_adamw_like_train_cost(m.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    assert rt.cfg.grad_balance == ""


def test_fixed_interval_ordinary_update_skips_transactional_direction():
    model = _Tiny()
    cfg = Phase2Config(line_search=True, lr=1e-3, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=1e-3,
        warmup_updates=0,
        search_interval_mode="fixed",
        initial_interval=3,
        search_on_early_triggers=False,
        strict_actual_update=False,
        actual_update_diagnostic_every=0,
    )
    opt = build_adamw_like_train_cost(model.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    # Model a successful first searched update. The immediately following
    # update is a pure AdamW step and needs neither parameter nor gradient
    # snapshots; the third update is the next scheduled search.
    rt.ctrl.pending_search = False
    rt.ctrl.successful_update_count = 1
    rt.ctrl.last_search_update = 1
    rt.ctrl.K = 3
    assert not needs_line_search_direction(rt, known_search=False, j=2)
    assert needs_line_search_direction(rt, known_search=True, j=3)

    x = torch.randn(2)
    opt.zero_grad(set_to_none=True)
    loss = model(x)
    loss.backward()
    result = run_search_or_ordinary(
        rt,
        j=2,
        run_search=False,
        search_reasons=[],
        F0=float(loss.detach()),
        g_preclip=None,
        g_objective=None,
        probe_F=lambda: float(model(x).detach()),
    )
    assert result.executed and not result.searched
    assert result.log["ls_fast_ordinary"] == 1.0


def test_delayed_activation_follows_cosine_floor_then_searches():
    model = _Tiny()
    cfg = Phase2Config(line_search=True, lr=1e-3, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=1e-3,
        epsilon_min=1e-5,
        epsilon_min_start=1e-3,
        epsilon_min_ramp_updates=10,
        activate_after_updates=10,
        warmup_updates=0,
        search_interval_mode="fixed",
        initial_interval=3,
        search_on_early_triggers=False,
    )
    opt = build_adamw_like_train_cost(model.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)

    known, reasons, j = resolve_known_search(rt)
    assert (known, reasons, j) == (False, [], 1)
    assert not needs_line_search_direction(rt, known_search=known, j=j)

    x = torch.randn(2)
    opt.zero_grad(set_to_none=True)
    loss = model(x)
    loss.backward()
    result = run_search_or_ordinary(
        rt,
        j=j,
        run_search=False,
        search_reasons=[],
        F0=float(loss.detach()),
        g_preclip=None,
        g_objective=None,
        probe_F=lambda: (_ for _ in ()).throw(AssertionError("probe ran")),
    )
    expected = cfg.line_search_cfg.epsilon_min_at(1, 1e-5)
    assert result.executed and not result.searched
    assert result.epsilon == expected
    assert result.log["ls_delayed_activation"] == 1

    rt.ctrl.successful_update_count = 10
    known, reasons, j = resolve_known_search(rt)
    assert known and j == 11
    assert "never_searched" in reasons


def test_delayed_cosine_matches_native_adamw_and_hands_rate_to_search():
    model = _ScalarObjective()
    native = copy.deepcopy(model)
    cfg = Phase2Config(line_search=True, line_search_cosine_before_activation=True,
                       lr=1e-4, lr_horizon=10000, warmup_steps=300)
    cfg.line_search_cfg = LineSearchConfig(
        activate_after_updates=10000, epsilon_min=1e-10,
        strict_actual_update=True, warmup_updates=0,
        search_on_early_triggers=False,
    )
    opt = build_adamw_like_train_cost(model.named_parameters(), lr=cfg.lr)
    native_opt = build_adamw_like_train_cost(native.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    rates = []
    for step in (0, 299, 5000, 9999):
        rt.ctrl.successful_update_count = step
        known, reasons, update = resolve_known_search(rt)
        rate = _line_search_scheduled_rate(cfg, step, rt, update)
        assert not known
        rates.append(rate)
        opt.zero_grad(set_to_none=True)
        loss = model.objective()
        loss.backward()
        result = run_search_or_ordinary(
            rt, j=update, run_search=known, search_reasons=reasons,
            F0=float(loss.detach()), g_preclip=None,
            probe_F=lambda: (_ for _ in ()).throw(AssertionError("probe before activation")),
            pre_activation_epsilon=rate,
        )
        native_opt.zero_grad(set_to_none=True)
        native.objective().backward()
        for group in native_opt.param_groups:
            group["lr"] = rate
        native_opt.step()
        assert result.executed and result.epsilon == rate
        torch.testing.assert_close(model.w, native.w, rtol=0, atol=0)
    assert rates[0] == pytest.approx(1e-4 / 300)
    assert rates[1] == 1e-4
    assert rates[3] == pytest.approx(5e-6, abs=1e-10)
    known, reasons, update = resolve_known_search(rt)
    assert known and update == 10001
    # Changing the absolute cosine clock cannot overwrite the inherited rate
    # once the selector becomes active.
    cfg.lr_at = lambda step: (_ for _ in ()).throw(AssertionError("cosine after handoff"))
    assert _line_search_scheduled_rate(cfg, 999999, rt, update) == rates[-1]
    opt.zero_grad(set_to_none=True)
    loss = model.objective()
    loss.backward()
    result = run_search_or_ordinary(
        rt, j=update, run_search=known, search_reasons=reasons,
        F0=float(loss.detach()), g_preclip=snapshot_g_preclip(list(model.parameters())),
        probe_F=lambda: float(model.objective().detach()),
        pre_activation_epsilon=0.5,  # ignored after activation
    )
    assert result.executed and result.searched
    assert result.log["ls_epsilon_seed"] == rates[-1]
    assert result.epsilon < 0.5


def test_resume_uses_absolute_cosine_step_until_remaining_activation_delay():
    model = _ScalarObjective()
    cfg = Phase2Config(line_search=True, line_search_cosine_before_activation=True,
                       lr=1e-4, lr_horizon=10000, warmup_steps=300)
    cfg.line_search_cfg = LineSearchConfig(activate_after_updates=8800)
    opt = build_adamw_like_train_cost(model.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    _, _, update = resolve_known_search(rt)
    assert _line_search_scheduled_rate(cfg, 1200, rt, update) == cfg.lr_at(1200)
    assert _line_search_scheduled_rate(cfg, 1200, rt, update) != cfg.lr_at(0)
    rt.ctrl.successful_update_count = 8800
    rt.ctrl.last_selected_epsilon = 5e-6
    _, _, update = resolve_known_search(rt)
    assert _line_search_scheduled_rate(cfg, 10000, rt, update) == 5e-6


def test_fast_ordinary_update_is_bit_exact_native_adamw():
    torch.manual_seed(91)
    fast = _Tiny()
    native = copy.deepcopy(fast)
    cfg = Phase2Config(line_search=True, lr=2e-3, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=2e-3,
        warmup_updates=0,
        search_interval_mode="fixed",
        initial_interval=3,
        search_on_early_triggers=False,
        actual_update_diagnostic_every=0,
    )
    opt_fast = build_adamw_like_train_cost(
        fast.named_parameters(), lr=cfg.lr, weight_decay=0.01,
    )
    opt_native = build_adamw_like_train_cost(
        native.named_parameters(), lr=cfg.lr, weight_decay=0.01,
    )
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt_fast)
    rt.ctrl.pending_search = False
    rt.ctrl.successful_update_count = 1
    rt.ctrl.last_search_update = 1
    rt.ctrl.K = 3

    x = torch.randn(2)
    for model, optimizer in ((fast, opt_fast), (native, opt_native)):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    result = run_search_or_ordinary(
        rt,
        j=2,
        run_search=False,
        search_reasons=[],
        F0=float(fast(x).detach()),
        g_preclip=None,
        g_objective=None,
        probe_F=lambda: float(fast(x).detach()),
    )
    opt_native.step()
    assert result.log["ls_fast_ordinary"] == 1.0
    for fast_parameter, native_parameter in zip(
        fast.parameters(), native.parameters(), strict=True,
    ):
        torch.testing.assert_close(
            fast_parameter, native_parameter, rtol=0.0, atol=0.0,
        )
        fast_state = opt_fast.state[fast_parameter]
        native_state = opt_native.state[native_parameter]
        assert fast_state.keys() == native_state.keys()
        for key in fast_state:
            if torch.is_tensor(fast_state[key]):
                torch.testing.assert_close(
                    fast_state[key], native_state[key], rtol=0.0, atol=0.0,
                )
            else:
                assert fast_state[key] == native_state[key]


def test_phase2_cli_exposes_line_search_safety_policies():
    from scripts.train_phase2 import parse_args

    args = parse_args([
        "--line-search-actual-diagnostic-every", "7",
        "--line-search-strict-actual-update",
        "--line-search-reject-current-probe-increase",
        "--line-search-skip-on-probe-exhaustion",
        "--reset-optimizer",
        "--log-every", "3",
    ])
    assert args.line_search_actual_diagnostic_every == 7
    assert args.line_search_strict_actual_update
    assert args.line_search_reject_current_probe_increase
    assert args.line_search_skip_on_probe_exhaustion
    assert args.reset_optimizer
    assert args.log_every == 3


def test_phase2_cli_can_explicitly_disable_fresh_resume_safety_defaults():
    from scripts.train_phase2 import parse_args

    args = parse_args([
        "--no-line-search-strict-actual-update",
        "--no-line-search-reject-current-probe-increase",
        "--no-line-search-skip-on-probe-exhaustion",
    ])
    assert args.line_search_strict_actual_update is False
    assert args.line_search_reject_current_probe_increase is False
    assert args.line_search_skip_on_probe_exhaustion is False


def test_ordinary_then_search_commit_once():
    torch.manual_seed(0)
    m = _Tiny()
    cfg = Phase2Config(line_search=True, lr=1e-2, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=1e-2, max_trials=4, warmup_updates=1, c1=0.0,
        beta_accept_per_update=0.0, expansion_factor=1.0,
    )
    opt = build_adamw_like_train_cost(m.named_parameters(), lr=cfg.lr, weight_decay=0.0)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    # First update: known search (pending / never searched).
    known, reasons, j = resolve_known_search(rt)
    assert known and j == 1

    x = torch.randn(2)
    opt.zero_grad(set_to_none=True)
    F0_t = m(x)
    F0_t.backward()
    gpc = snapshot_g_preclip(rt.adapter._params)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)

    def probe_F():
        with torch.no_grad():
            return float(m(x).detach())

    F0 = float(F0_t.detach())
    step0 = [float(opt.state[p]["step"]) if p in opt.state and "step" in opt.state[p] else 0.0
             for p in rt.adapter._params]
    res = run_search_or_ordinary(
        rt, j=j, run_search=True, search_reasons=reasons,
        F0=F0, g_preclip=gpc, probe_F=probe_F,
    )
    assert res.executed and res.searched and res.commit_side_effects
    # Moments advanced once (step counters bumped).
    for p in rt.adapter._params:
        assert p in opt.state and int(opt.state[p]["step"].item()) == 1

    # Ordinary reuse: no second staging search.
    known2, reasons2, j2 = resolve_known_search(rt)
    # After one success with K possibly 1 during warmup, may still search.
    opt.zero_grad(set_to_none=True)
    F1 = m(x)
    F1.backward()
    gpc2 = snapshot_g_preclip(rt.adapter._params)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    res2 = run_search_or_ordinary(
        rt, j=j2, run_search=False, search_reasons=[],
        F0=float(F1.detach()), g_preclip=gpc2, probe_F=probe_F,
    )
    assert res2.executed and not res2.searched
    for p in rt.adapter._params:
        assert int(opt.state[p]["step"].item()) == 2


def test_selection_failure_skips_side_effects_flag():
    torch.manual_seed(1)
    m = _Tiny()
    # Pathological: zero grad → invalid probe.
    cfg = Phase2Config(line_search=True, lr=1e-3, grad_balance="")
    opt = build_adamw_like_train_cost(m.named_parameters(), lr=cfg.lr)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    opt.zero_grad(set_to_none=True)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    gpc = snapshot_g_preclip(rt.adapter._params)
    res = run_search_or_ordinary(
        rt, j=1, run_search=True, search_reasons=["warmup"],
        F0=1.0, g_preclip=gpc, probe_F=lambda: 1.0,
    )
    assert not res.executed
    assert not res.commit_side_effects
    assert rt.ctrl.pending_search


def test_stochastic_failure_can_retry_retained_rate():
    model = _ScalarObjective()
    cfg = Phase2Config(line_search=True, lr=0.01, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=0.01, epsilon_min=1e-10,
        retry_retained_rate=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    opt.zero_grad(set_to_none=True)
    model.w.grad = torch.zeros_like(model.w)
    result = run_search_or_ordinary(
        rt, j=1, run_search=True, search_reasons=["never_searched"],
        F0=0.5, g_preclip=snapshot_g_preclip(rt.adapter._params),
        probe_F=lambda: 0.5,
    )
    assert not result.executed
    assert rt.ctrl.retry_seed == pytest.approx(0.01)


def test_actual_diagnostic_replays_exec_before_commit_without_rng_leak():
    model = _ScalarObjective()
    cfg = Phase2Config(line_search=True, lr=0.1, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=0.1,
        epsilon_min=0.01,
        epsilon_max=0.2,
        max_trials=2,
        c1=0.0,
        beta_accept_per_update=0.0,
        actual_update_diagnostic_every=2,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.1, betas=(0.9, 0.95), weight_decay=0.0,
    )
    # Nontrivial prior momentum makes probe and executed candidates differ.
    opt.state[model.w] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.ones_like(model.w),
        "exp_avg_sq": torch.ones_like(model.w),
    }
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()
    g_pre = snapshot_g_preclip(rt.adapter._params)

    gen = torch.Generator().manual_seed(123)
    replay = ObjectiveReplayContext({"gen": gen}, capture_global=False)
    rec = replay.begin_record(controls=FrozenControls(step=1))
    replay.finalize_baseline(rec)
    rng_post = snapshot_torch_generator(gen)

    def objective_with_accidental_draw():
        torch.rand((), generator=gen)
        return float(model.objective().detach())

    result = run_search_or_ordinary(
        rt,
        j=2,
        run_search=True,
        search_reasons=["interval"],
        F0=F0,
        g_preclip=g_pre,
        probe_F=objective_with_accidental_draw,
        replay_ctx=replay,
        replay_rec=rec,
    )
    assert result.executed
    assert torch.equal(snapshot_torch_generator(gen), rng_post)
    assert result.log["ls_Factual"] == pytest.approx(
        float(model.objective().detach())
    )
    assert result.log["ls_actual_decrease"] == pytest.approx(
        F0 - result.log["ls_Factual"]
    )
    assert result.log["ls_probe_exec_discrepancy"] == pytest.approx(
        result.log["ls_Factual"] - result.log["ls_Fprobe"]
    )
    assert abs(result.log["ls_probe_exec_discrepancy"]) > 1e-6


def test_strict_actual_update_backtracks_until_actual_decreases():
    model = _ScalarObjective()
    cfg = Phase2Config(line_search=True, lr=1.5, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=1.5,
        epsilon_min=0.1,
        epsilon_max=2.0,
        max_trials=3,
        fallback_shrink=0.5,
        c1=0.0,
        beta_accept_per_update=0.0,
        strict_actual_update=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=1.5, betas=(0.9, 0.95), weight_decay=0.0,
    )
    opt.state[model.w] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.ones_like(model.w),
        "exp_avg_sq": torch.ones_like(model.w),
    }
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()

    result = run_search_or_ordinary(
        rt,
        j=1,
        run_search=True,
        search_reasons=["never_searched"],
        F0=F0,
        g_preclip=snapshot_g_preclip(rt.adapter._params),
        probe_F=lambda: float(model.objective().detach()),
    )
    assert result.executed and result.commit_side_effects
    assert result.epsilon == pytest.approx(0.75)
    assert result.log["ls_actual_trial_count"] == 2
    assert result.log["ls_Factual"] <= F0
    assert int(opt.state[model.w]["step"].item()) == 2


@pytest.mark.parametrize("run_search", [True, False])
def test_strict_actual_veto_restores_params_optimizer_and_commit_controller(
    run_search,
):
    model = _DecayUphillObjective()
    cfg = Phase2Config(line_search=True, lr=1e-2, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(
        epsilon_seed=1e-2,
        epsilon_min=2.5e-3,
        epsilon_max=1e-2,
        max_trials=3,
        fallback_shrink=0.5,
        c1=0.0,
        beta_accept_per_update=0.0,
        strict_actual_update=True,
    )
    opt = build_adamw_like_train_cost(
        model.named_parameters(), lr=1e-2, weight_decay=1.0,
    )
    rt = init_line_search_runtime(phase2_cfg=cfg, optimizer=opt)
    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()
    params_before = [p.detach().clone() for p in model.parameters()]
    optimizer_before = copy.deepcopy(opt.state_dict())
    committed_controller_before = {
        "successful_update_count": rt.ctrl.successful_update_count,
        "last_search_update": rt.ctrl.last_search_update,
        "searches_completed": rt.ctrl.searches_completed,
        "last_selected_epsilon": rt.ctrl.last_selected_epsilon,
    }

    result = run_search_or_ordinary(
        rt,
        j=1,
        run_search=run_search,
        search_reasons=["never_searched"] if run_search else [],
        F0=F0,
        g_preclip=snapshot_g_preclip(rt.adapter._params),
        probe_F=lambda: float(model.objective().detach()),
    )
    assert not result.executed
    assert not result.commit_side_effects
    assert result.failure_reason == "actual_update_increase"
    assert result.log["ls_actual_trial_count"] == 3
    assert result.log["ls_actual_veto"] == 1
    assert result.log["ls_Factual"] > F0
    for param, before in zip(model.parameters(), params_before):
        assert torch.equal(param, before)
    _assert_nested_equal(opt.state_dict(), optimizer_before)
    assert {
        "successful_update_count": rt.ctrl.successful_update_count,
        "last_search_update": rt.ctrl.last_search_update,
        "searches_completed": rt.ctrl.searches_completed,
        "last_selected_epsilon": rt.ctrl.last_selected_epsilon,
    } == committed_controller_before
    assert rt.ctrl.pending_search
    assert "selection_failed" in rt.ctrl.pending_trigger_reasons


def test_strike_snapshot_roundtrip():
    from dataclasses import dataclass

    @dataclass
    class FakeStrike:
        a: torch.Tensor
        b: torch.Tensor | None = None

    batch = {
        "init": torch.randn(2, 3),
        "strike": FakeStrike(a=torch.randn(2), b=torch.ones(2)),
        "context": type("C", (), {"yaw_now": torch.tensor([0.1, 0.2])})(),
    }
    snap = snapshot_mutable_batch_fields(batch)
    assert "strike" in snap and "yaw_now" in snap
    batch["init"].zero_()
    batch["strike"].a.zero_()
    batch["context"].yaw_now.zero_()
    restore_mutable_batch_fields(batch, snap)
    assert not torch.equal(batch["init"], torch.zeros_like(batch["init"]))
    assert torch.allclose(batch["strike"].a, snap["strike"].a)


def test_explicit_minimum_probe_rate_is_honored_on_selector_resume():
    from amortized_opt.line_search_step import runtime_state_dict

    m = _Tiny()
    cfg = Phase2Config(line_search=True, lr=1e-4, grad_balance="")
    cfg.line_search_cfg = LineSearchConfig(epsilon_seed=1e-4)
    opt = build_adamw_like_train_cost(m.named_parameters(), lr=cfg.lr)
    saved = runtime_state_dict(init_line_search_runtime(phase2_cfg=cfg, optimizer=opt))
    cfg.line_search_cfg = LineSearchConfig(epsilon_seed=1e-4, epsilon_min=1e-7)
    restored = init_line_search_runtime(
        phase2_cfg=cfg, optimizer=opt, line_search_state=saved, opt_restored=True,
    )
    assert restored.epsilon_min == pytest.approx(1e-7)
    assert restored.cfg.epsilon_min == pytest.approx(1e-7)
