"""SaLSa line search for the terminal-pose network.

Uses the same selector as the trajectory optimizer: a preconditioned
gradient probe, a smoothed Armijo test, and AdamW steps at the chosen
rate.  Between searches the last rate is reused.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adamw_ls import AdamWLineSearchAdapter, build_adamw_like_train_cost
from .line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    advance_successful_update,
    commit_acceptance,
    effective_beta,
    evaluate_trial,
    B_candidate,
    choose_search_seed,
    on_selection_failure,
    should_search,
    update_interval_after_search,
)
from .terminal_pose import TerminalPoseCost


@dataclass
class TerminalPoseLineSearch:
    cfg: LineSearchConfig
    adapter: AdamWLineSearchAdapter
    accept: AcceptanceEMAState
    ctrl: IntervalControllerState
    epsilon_min: float
    epsilon_max: float
    last_epsilon: float


def make_terminal_pose_line_search(
    net: nn.Module,
    *,
    cfg: LineSearchConfig | None = None,
    weight_decay: float = 0.0,
) -> TerminalPoseLineSearch:
    """AdamW adapter whose reference rate is the SaLSa seed (10⁻⁴)."""
    cfg = cfg or LineSearchConfig()
    opt = build_adamw_like_train_cost(
        net.named_parameters(),
        lr=float(cfg.epsilon_seed),
        weight_decay=float(weight_decay),
    )
    emin, emax = cfg.resolve_bounds()
    adapter = AdamWLineSearchAdapter(opt, eta_ref=float(cfg.epsilon_seed))
    return TerminalPoseLineSearch(
        cfg=cfg,
        adapter=adapter,
        accept=AcceptanceEMAState(),
        ctrl=IntervalControllerState(),
        epsilon_min=emin,
        epsilon_max=emax,
        last_epsilon=float(cfg.epsilon_seed),
    )


def _scalar_loss(cost: TerminalPoseCost, net: nn.Module, batch, weights) -> torch.Tensor:
    return cost(net(batch.features), batch, weights).total.mean()


def line_search_update(
    ls: TerminalPoseLineSearch,
    net: nn.Module,
    cost: TerminalPoseCost,
    batch,
    weights: dict,
    *,
    step: int,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One terminal-pose update.  Returns the chosen rate and whether it searched."""
    cfg = ls.cfg
    params = list(ls.adapter._params)
    net.zero_grad(set_to_none=True)
    first = cost(net(batch.features), batch, weights)
    loss = first.total.mean()
    violations = {k: v.detach() for k, v in first.violations.items()}
    loss.backward()
    # Same order as the adapter.  Optimizer groups are not parameter() order.
    preclip = [
        (None if p.grad is None else p.grad.detach().clone()) for p in params
    ]
    torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
    f0 = float(loss.detach())

    search, reasons = should_search(ls.ctrl, cfg, int(step))
    ls.adapter.snapshot_theta0()
    staged = ls.adapter.stage_from_q(g_objective=preclip, use_param_grad_as_q=True)
    epsilon = ls.last_epsilon
    searched = False
    accepted = False
    trials = 0

    if search and staged.stats.probe_valid:
        searched = True
        seed = choose_search_seed(
            ls.ctrl, cfg,
            epsilon_max=ls.epsilon_max,
            epsilon_min=ls.epsilon_min,
            reasons=reasons,
        )
        epsilon = float(seed.epsilon)
        beta_eff, _gap = effective_beta(
            ls.accept.last_history_update, int(step), cfg.beta_accept_per_update,
        )
        b_cand = B_candidate(ls.accept.B_prev, staged.stats.a, beta_eff)
        decision = None
        for trial in range(int(cfg.max_trials)):
            trials = trial + 1
            ls.adapter.apply_epsilon_to_params(epsilon, staged.p_probe)
            with torch.no_grad():
                f_probe = float(_scalar_loss(cost, net, batch, weights))
            ls.adapter.restore_theta0()
            decision = evaluate_trial(
                Fprobe=f_probe,
                F0=f0,
                epsilon=epsilon,
                a=staged.stats.a,
                P2=staged.stats.P2,
                A_prev=ls.accept.A_prev,
                B_cand=b_cand,
                beta_eff=beta_eff,
                c1=cfg.c1,
                proposal_rule=cfg.proposal_rule,
                backtrack_min_factor=cfg.backtrack_min_factor,
                backtrack_max_factor=cfg.backtrack_max_factor,
                fallback_shrink=cfg.fallback_shrink,
                epsilon_min=ls.epsilon_min,
                trial_index=trial,
                max_trials=int(cfg.max_trials),
                reject_current_probe_increase=cfg.reject_current_probe_increase,
            )
            if decision.accept or decision.stop or decision.next_epsilon is None:
                break
            epsilon = float(decision.next_epsilon)
        if decision is not None and decision.accept:
            accepted = True
            commit_acceptance(ls.accept, decision.A_candidate, decision.B_candidate, int(step), beta_eff)
            update_interval_after_search(
                ls.ctrl, cfg,
                epsilon_new=epsilon,
                trial_count=trials,
                j=int(step),
                trigger_reasons=reasons,
                epsilon_min=ls.epsilon_min,
            )
        else:
            on_selection_failure(
                ls.ctrl, ls.accept,
                last_tried_epsilon=epsilon,
                fallback_shrink=cfg.fallback_shrink,
                epsilon_min=ls.epsilon_min,
            )
            epsilon = float(ls.epsilon_min)
        ls.adapter.commit(staged, epsilon=epsilon, params_already_written=False)
    else:
        ls.adapter.execute_ordinary_step(epsilon)

    ls.last_epsilon = float(epsilon)
    advance_successful_update(ls.ctrl, int(step))
    net.zero_grad(set_to_none=True)
    return {
        "loss": f0,
        "epsilon": float(epsilon),
        "searched": float(searched),
        "accepted": float(accepted),
        "trials": float(trials),
        "violations": violations,
    }
