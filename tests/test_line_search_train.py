"""Integration tests for SaLSa line-search train wiring (plan §11 tests 13–16 / §14).

Covers Phase2Config surgery fork, commit-gate policy, ``line_search_step``
orchestration on a tiny AdamW-grouped quadratic, and checkpoint pack/unpack.
Full ``train_phase2`` cosine-ownership smokes stay skipped until that lands.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amortized_opt.adamw_ls import AdamWLineSearchAdapter, build_adamw_like_train_cost
from amortized_opt.line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    TriggerRefs,
    commit_acceptance,
    line_search_state_dict,
    load_line_search_state_dict,
    on_selection_failure,
)
from amortized_opt.line_search_step import (
    init_line_search_runtime,
    params_finite,
    resolve_known_search,
    run_search_or_ordinary,
    runtime_state_dict,
    snapshot_g_preclip,
)
from amortized_opt.line_search_train_util import (
    apply_selection_failure,
    group_lrs_from_epsilon,
    pack_line_search_extra,
    search_epsilon_along_probe,
    should_advance_controllers,
    unpack_line_search_extra,
)
from amortized_opt.train_cost import Phase2Config


# ----------------------------------------------------------- §14 / cfg smoke --


def test_phase2config_line_search_forces_grad_balance_empty():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = Phase2Config(line_search=True, grad_balance="al_pcgrad_output")
    assert cfg.line_search is True
    assert cfg.grad_balance == ""
    assert any("forces grad_balance" in str(w.message) for w in caught)

    cfg2 = Phase2Config(line_search=True, grad_balance="")
    assert cfg2.grad_balance == ""

    cfg3 = Phase2Config(line_search=False, grad_balance="al_pcgrad_output")
    assert cfg3.grad_balance == "al_pcgrad_output"


def test_should_advance_controllers_commit_gate():
    # Plan §14 table: success advances; selection failure and skipped step do not.
    assert should_advance_controllers(executed=True, selection_failed=False) is True
    assert should_advance_controllers(executed=False, selection_failed=False) is False
    assert should_advance_controllers(executed=True, selection_failed=True) is False
    assert should_advance_controllers(executed=False, selection_failed=True) is False


def test_group_lrs_from_epsilon():
    assert group_lrs_from_epsilon(2e-4, [1.0, 0.5]) == pytest.approx([2e-4, 1e-4])
    with pytest.raises(ValueError):
        group_lrs_from_epsilon(0.0, [1.0])


# ---------------------------------- quadratic + AdamW groups (ε transfer) --


class _QuadTrainLike(nn.Module):
    """Mimic ``build_optimizer`` groups: 2-D decay + 1-D / emb no_decay."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.5, -0.8], [0.4, 1.1]]))
        self.bias = nn.Parameter(torch.tensor([0.7, -0.3]))
        self.token_emb = nn.Parameter(torch.tensor([0.2, -0.1, 0.5]))

    def objective(self) -> torch.Tensor:
        return 0.5 * (
            (self.weight ** 2).sum()
            + (self.bias ** 2).sum()
            + (self.token_emb ** 2).sum()
        )


def _make_quad_opt(model: _QuadTrainLike, lr: float = 1e-3, wd: float = 0.01):
    opt = build_adamw_like_train_cost(
        model.named_parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    for g in opt.param_groups:
        g["foreach"] = False
        g["fused"] = False
    return opt


def _phase2_ls_stub(*, lr: float = 1e-3, ls_cfg: LineSearchConfig | None = None):
    return SimpleNamespace(
        lr=lr,
        line_search_cfg=ls_cfg
        or LineSearchConfig(epsilon_seed=lr, beta_accept_per_update=0.0, c1=0.1),
    )


def test_total_backward_matches_autograd_grad_on_quadratic():
    """§11.13 lite: plain ``total.backward()`` matches ``autograd.grad(F)``."""
    model = _QuadTrainLike()
    F = model.objective()
    g_list = torch.autograd.grad(F, list(model.parameters()), retain_graph=True)
    model.zero_grad(set_to_none=True)
    F2 = model.objective()
    F2.backward()
    for p, g in zip(model.parameters(), g_list):
        assert torch.allclose(p.grad, g, rtol=1e-6, atol=1e-8)


def test_snapshot_g_preclip_and_params_finite():
    model = _QuadTrainLike()
    model.objective().backward()
    params = list(model.parameters())
    g0 = snapshot_g_preclip(params)
    assert all(g is not None for g in g0)
    params[0].grad.fill_(999.0)
    assert not torch.allclose(g0[0], params[0].grad)
    assert params_finite(params)
    with torch.no_grad():
        params[0].fill_(float("nan"))
    assert not params_finite(params)


def test_line_search_step_epsilon_transfer_on_quadratic():
    """``run_search_or_ordinary`` selects ε on p_probe and commits p_exec."""
    torch.manual_seed(0)
    model = _QuadTrainLike()
    eta = 1e-3
    opt = _make_quad_opt(model, lr=eta, wd=0.02)
    phase2 = _phase2_ls_stub(lr=eta)
    rt = init_line_search_runtime(phase2_cfg=phase2, optimizer=opt)
    assert rt.seed_provenance in ("fresh_cfg_lr", "fresh_epsilon_seed")
    assert rt.cfg.grad_balance == ""

    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()
    g_pre = snapshot_g_preclip(list(model.parameters()))
    run, reasons, j = resolve_known_search(rt)
    assert run and j == 1

    n0 = rt.accept.n_history
    result = run_search_or_ordinary(
        rt,
        j=j,
        run_search=True,
        search_reasons=reasons,
        F0=F0,
        g_preclip=g_pre,
        probe_F=lambda: float(model.objective().detach()),
    )
    assert result.executed and result.searched
    assert result.epsilon is not None and result.epsilon > 0.0
    assert result.commit_side_effects is True
    assert should_advance_controllers(
        executed=result.executed,
        selection_failed=bool(result.searched and not result.executed),
    )
    assert rt.accept.n_history == n0 + 1
    assert rt.ctrl.successful_update_count == j
    assert rt.ctrl.last_selected_epsilon == pytest.approx(result.epsilon)
    assert result.log.get("grad_balance") == ""


def test_line_search_step_selection_exhausted_commits_at_floor():
    torch.manual_seed(1)
    model = _QuadTrainLike()
    eta = 1e-3
    opt = _make_quad_opt(model, lr=eta, wd=0.0)
    ls_cfg = LineSearchConfig(
        epsilon_seed=10.0,
        # Large enough to produce a representable float32 displacement, so
        # this test distinguishes a legacy floor commit from a skipped update.
            epsilon_min=0.1,
        epsilon_max=100.0,
        max_trials=1,
        c1=10.0,
        beta_accept_per_update=0.0,
        proposal_rule="geometric",
        fallback_shrink=0.5,
        warmup_updates=0,
    )
    phase2 = _phase2_ls_stub(lr=eta, ls_cfg=ls_cfg)
    rt = init_line_search_runtime(phase2_cfg=phase2, optimizer=opt)
    rt.accept.A_prev = 1.25
    rt.accept.B_prev = 0.8
    rt.accept.n_history = 3
    rt.accept.last_history_update = 5
    rt.accept.retention_product = 0.7
    rt.ctrl.successful_update_count = 5
    rt.ctrl.K = 4
    snap = (
        rt.accept.A_prev,
        rt.accept.B_prev,
        rt.accept.n_history,
        rt.accept.retention_product,
    )

    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()
    g_pre = snapshot_g_preclip(list(model.parameters()))
    theta0 = [p.detach().clone() for p in model.parameters()]

    result = run_search_or_ordinary(
        rt,
        j=6,
        run_search=True,
        search_reasons=["pending_search"],
        F0=F0,
        g_preclip=g_pre,
        probe_F=lambda: float(model.objective().detach()),
    )
    assert result.executed
    assert result.searched
    assert result.commit_side_effects is True
    assert result.failure_reason == "selection_exhausted"
    assert result.epsilon == pytest.approx(float(ls_cfg.epsilon_min))
    assert result.log.get("ls_floor_commit") == 1
    assert result.log.get("ls_accepted") == 0
    assert (
        rt.accept.A_prev,
        rt.accept.B_prev,
        rt.accept.n_history,
        rt.accept.retention_product,
    ) == snap
    assert rt.accept.pending_history_reset is True
    assert rt.ctrl.K == 1
    assert rt.ctrl.successful_update_count == 6
    assert rt.ctrl.last_selected_epsilon == pytest.approx(float(ls_cfg.epsilon_min))
    assert should_advance_controllers(
        executed=result.executed,
        selection_failed=bool(result.searched and not result.executed),
    ) is True
    assert any(
        not torch.allclose(p, t0)
        for p, t0 in zip(model.parameters(), theta0)
    )


def test_line_search_step_can_skip_probe_selection_exhaustion():
    model = _QuadTrainLike()
    eta = 1e-3
    opt = _make_quad_opt(model, lr=eta, wd=0.0)
    ls_cfg = LineSearchConfig(
        epsilon_seed=10.0,
        epsilon_min=1e-12,
        epsilon_max=100.0,
        max_trials=1,
        c1=10.0,
        beta_accept_per_update=0.0,
        proposal_rule="geometric",
        skip_on_probe_exhaustion=True,
    )
    rt = init_line_search_runtime(
        phase2_cfg=_phase2_ls_stub(lr=eta, ls_cfg=ls_cfg),
        optimizer=opt,
    )
    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()
    theta0 = [p.detach().clone() for p in model.parameters()]

    result = run_search_or_ordinary(
        rt,
        j=1,
        run_search=True,
        search_reasons=["never_searched"],
        F0=F0,
        g_preclip=snapshot_g_preclip(list(model.parameters())),
        probe_F=lambda: float(model.objective().detach()),
    )
    assert not result.executed
    assert not result.commit_side_effects
    assert result.failure_reason == "selection_exhausted"
    assert result.epsilon is None
    assert result.log["ls_floor_commit"] == 0
    assert result.log["ls_accepted"] == 0
    assert rt.ctrl.successful_update_count == 0
    assert all(
        torch.equal(param, before)
        for param, before in zip(model.parameters(), theta0)
    )


def test_adapter_selector_helper_still_transfers_epsilon():
    """Util ``search_epsilon_along_probe`` remains a thin host-loop alternative."""
    torch.manual_seed(0)
    model = _QuadTrainLike()
    eta = 1e-3
    opt = _make_quad_opt(model, lr=eta, wd=0.02)
    F0_t = model.objective()
    F0 = float(F0_t.detach())
    F0_t.backward()
    adapter = AdamWLineSearchAdapter(opt, eta_ref=eta)
    adapter.mark_theta0_snapshot()
    staged = adapter.stage_from_q()
    cfg = LineSearchConfig(beta_accept_per_update=0.0, c1=0.1, epsilon_seed=eta)
    accept = AcceptanceEMAState()
    outcome = search_epsilon_along_probe(
        evaluate_F=lambda: float(model.objective().detach()),
        adapter=adapter,
        staged=staged,
        F0=F0,
        cfg=cfg,
        accept=accept,
        j=1,
    )
    assert outcome.selected
    adapter.commit(staged, epsilon=float(outcome.epsilon), use_exec=True)
    commit_acceptance(
        accept, float(outcome.A_candidate), float(outcome.B_candidate), 1, outcome.beta_eff
    )
    assert accept.n_history == 1


def test_util_selection_failure_does_not_advance_acceptance_ema():
    model = _QuadTrainLike()
    opt = _make_quad_opt(model, lr=1e-3, wd=0.0)
    F0 = float(model.objective().detach())
    model.objective().backward()
    adapter = AdamWLineSearchAdapter(opt, eta_ref=1e-3)
    adapter.mark_theta0_snapshot()
    staged = adapter.stage_from_q()
    cfg = LineSearchConfig(
        beta_accept_per_update=0.0,
        c1=10.0,
        epsilon_seed=10.0,
        epsilon_min=1e-12,
        epsilon_max=100.0,
        max_trials=1,
        proposal_rule="geometric",
    )
    accept = AcceptanceEMAState(A_prev=1.25, B_prev=0.8, n_history=3, last_history_update=5)
    ctrl = IntervalControllerState(successful_update_count=5, K=4, pending_search=True)
    outcome = search_epsilon_along_probe(
        evaluate_F=lambda: float(model.objective().detach()),
        adapter=adapter,
        staged=staged,
        F0=F0,
        cfg=cfg,
        accept=accept,
        j=6,
    )
    assert not outcome.selected
    apply_selection_failure(ctrl, accept, outcome)
    assert accept.A_prev == 1.25 and accept.n_history == 3
    assert accept.pending_history_reset


# ------------------------------------------------------ checkpoint / resume --


def test_line_search_checkpoint_extra_roundtrip_extends_coverage():
    cfg = LineSearchConfig(epsilon_seed=3e-4, initial_interval=2, grad_balance="junk")
    accept = AcceptanceEMAState(
        A_prev=2.0, B_prev=1.1, n_history=7, last_history_update=40
    )
    accept.retention_product = 0.85
    ctrl = IntervalControllerState(
        successful_update_count=40,
        last_search_update=38,
        K=5,
        last_selected_epsilon=3e-4,
        retry_seed=1.5e-4,
        pending_search=False,
        pending_trigger_reasons=["grad_norm_spike"],
        searches_completed=12,
        stability_eligible=True,
        stable_search_streak=2,
    )
    refs = TriggerRefs(gnorm_ema=3.3, initialized=True, viol_ema={"torque": 0.02})
    blob = pack_line_search_extra(
        cfg,
        accept,
        ctrl,
        refs,
        fixed_group_ratios=[1.0, 1.0],
        eta_ref=3e-4,
        seed_provenance="legacy_param_group_lr",
    )
    assert blob["version"] == 2
    assert blob["grad_balance"] == ""
    assert blob["fixed_group_ratios"] == [1.0, 1.0]
    assert blob["eta_ref"] == pytest.approx(3e-4)
    assert blob["seed_provenance"] == "legacy_param_group_lr"

    cfg2, acc2, ctrl2, refs2 = load_line_search_state_dict(blob)
    assert cfg2.grad_balance == ""
    assert cfg2.epsilon_seed == pytest.approx(3e-4)
    assert acc2.retention_product == pytest.approx(0.85)
    assert ctrl2.retry_seed == pytest.approx(1.5e-4)
    assert ctrl2.pending_trigger_reasons == ["grad_norm_spike"]
    assert refs2.viol_ema["torque"] == pytest.approx(0.02)

    bundle = unpack_line_search_extra({"line_search": blob})
    assert bundle is not None
    assert bundle.fixed_group_ratios == [1.0, 1.0]
    assert bundle.eta_ref == pytest.approx(3e-4)
    assert bundle.seed_provenance == "legacy_param_group_lr"
    assert unpack_line_search_extra({}) is None

    blob2 = line_search_state_dict(
        bundle.cfg,
        bundle.accept,
        bundle.ctrl,
        bundle.refs,
        fixed_group_ratios=bundle.fixed_group_ratios,
        eta_ref=bundle.eta_ref,
        seed_provenance=bundle.seed_provenance,
    )
    assert blob2["controller"]["stable_search_streak"] == 2


def test_legacy_resume_seed_from_optimizer_group_lr():
    """§11.15: legacy checkpoint with small last LR seeds first search there."""
    model = _QuadTrainLike()
    small_lr = 5e-6
    opt = _make_quad_opt(model, lr=small_lr, wd=0.01)
    for g in opt.param_groups:
        g["lr"] = small_lr
    phase2 = _phase2_ls_stub(lr=1e-3)  # cfg.lr must NOT override legacy groups
    rt = init_line_search_runtime(
        phase2_cfg=phase2, optimizer=opt, opt_restored=True,
    )
    assert rt.seed_provenance == "legacy_optimizer_group_lr"
    assert rt.eta_ref == pytest.approx(small_lr)
    assert rt.ctrl.last_selected_epsilon == pytest.approx(small_lr)
    assert rt.cfg.epsilon_seed == pytest.approx(small_lr)
    assert abs(rt.eta_ref - float(phase2.lr)) > 1e-9


def test_runtime_state_dict_roundtrip_via_init():
    model = _QuadTrainLike()
    opt = _make_quad_opt(model, lr=2e-4)
    phase2 = _phase2_ls_stub(lr=2e-4)
    rt = init_line_search_runtime(phase2_cfg=phase2, optimizer=opt)
    rt.accept.A_prev = 0.5
    rt.ctrl.successful_update_count = 11
    rt.ctrl.K = 3
    blob = runtime_state_dict(rt)
    opt2 = _make_quad_opt(_QuadTrainLike(), lr=2e-4)
    rt2 = init_line_search_runtime(
        phase2_cfg=phase2, optimizer=opt2, line_search_state=blob,
    )
    assert rt2.accept.A_prev == pytest.approx(0.5)
    assert rt2.ctrl.K == 3
    assert rt2.ctrl.successful_update_count == 11


def test_on_selection_failure_then_should_not_advance():
    accept = AcceptanceEMAState(A_prev=9.0, B_prev=4.0, n_history=2)
    ctrl = IntervalControllerState(K=7, successful_update_count=10)
    on_selection_failure(ctrl, accept, last_tried_epsilon=1e-3, fallback_shrink=0.5)
    assert accept.A_prev == 9.0
    assert accept.pending_history_reset
    assert not should_advance_controllers(executed=False, selection_failed=True)


# -------------------------------------- train_phase2 wiring probes --


def test_train_phase2_gates_cosine_under_line_search():
    """§11.10: selector owns LR; cosine ``lr_at`` only on the non-LS branch."""
    src = (
        Path(__file__).resolve().parents[1]
        / "amortized_opt"
        / "train_cost.py"
    ).read_text(encoding="utf-8")
    assert "Selector owns LR under line search" in src
    assert "ls_rt.adapter.set_group_lrs(eps_reuse)" in src
    # Cosine write must remain behind the non-LS else branch.
    gate_idx = src.index("Selector owns LR under line search")
    snippet = src[gate_idx : gate_idx + 800]
    assert "cfg.lr_at(step)" in snippet
    assert "if ls_rt is not None:" in snippet
    assert "else:" in snippet


def test_train_phase2_uses_commit_side_effects_gate():
    src = (
        Path(__file__).resolve().parents[1]
        / "amortized_opt"
        / "train_cost.py"
    ).read_text(encoding="utf-8")
    assert "commit_side_effects" in src
    # Pure policy helper exists for integrator / tests to share.
    assert should_advance_controllers(executed=True, selection_failed=False)
