"""Thin SaLSa line-search orchestrator for :func:`train_phase2`.

Keeps selector/adapter bookkeeping out of the main train loop. Probe ``F``
evaluation still lives next to the live objective assembly in ``train_cost``
(callables are injected) so term coverage stays in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW

from .adamw_ls import AdamWLineSearchAdapter, group_ratios_from_optimizer, step_vector_breakdown
from .lbfgs_ls import LbfgsLineSearchAdapter, step_vector_breakdown_lbfgs
from .line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    TriggerRefs,
    advance_successful_update,
    apply_pending_history_reset,
    choose_search_seed,
    clamp_epsilon,
    commit_acceptance,
    dataclasses_replace,
    effective_beta,
    B_candidate,
    evaluate_trial,
    latch_triggers,
    line_search_state_dict,
    load_line_search_state_dict,
    logging_dict_from_search,
    on_selection_failure,
    resolve_runtime_bounds,
    should_search,
    update_interval_after_search,
)
from .objective_replay import ObjectiveReplayContext, ReplayIncompleteError, ReplayRecord

__all__ = [
    "LineSearchRuntime",
    "SearchStepResult",
    "init_line_search_runtime",
    "snapshot_g_preclip",
    "params_finite",
    "build_objective_direction_fingerprint",
    "resolve_known_search",
    "needs_line_search_direction",
    "run_search_or_ordinary",
    "assert_replayable_extras_or_raise",
    "runtime_state_dict",
]


@dataclass
class LineSearchRuntime:
    cfg: LineSearchConfig
    accept: AcceptanceEMAState
    ctrl: IntervalControllerState
    refs: TriggerRefs
    adapter: AdamWLineSearchAdapter | LbfgsLineSearchAdapter
    optimizer_kind: str
    eta_ref: float
    epsilon_min: float
    epsilon_max: float
    seed_provenance: str
    fixed_group_ratios: list[float]
    objective_direction_fingerprint: dict[str, Any] = field(default_factory=dict)
    fingerprint_reset_reason: str | None = None
    last_log: dict[str, Any] = field(default_factory=dict)
    #: Absolute training step of last AL λ/pen snapshot (for change detection).
    al_lam_snapshot: dict[str, float] = field(default_factory=dict)
    al_pen_snapshot: dict[str, float] = field(default_factory=dict)
    stage_a_snapshot: bool | None = None
    #: Last ``WeightSchedule.torque_full_blend`` (level A -> full RNEA).
    torque_full_blend_snapshot: float = 1.0


@dataclass
class SearchStepResult:
    """Outcome of the post-backward LS branch (ordinary or searched)."""

    executed: bool
    searched: bool
    epsilon: float | None
    log: dict[str, Any] = field(default_factory=dict)
    #: When False, caller must skip frontier EMA + viol_acc (§14 commit gate).
    commit_side_effects: bool = True
    failure_reason: str | None = None


def snapshot_g_preclip(params: Sequence[Tensor]) -> list[Tensor | None]:
    """Clone ``.grad`` before ``clip_grad_norm_`` (probe slope uses pre-clip ``g``)."""
    out: list[Tensor | None] = []
    for p in params:
        if p.grad is None:
            out.append(None)
        else:
            out.append(p.grad.detach().clone())
    return out


def params_finite(params: Sequence[Tensor]) -> bool:
    for p in params:
        if not torch.isfinite(p).all():
            return False
    return True


def assert_replayable_extras_or_raise(
    cfg_phase2: Any,
    *,
    include_bptt: bool,
    has_tc_samples: bool,
    has_sfx_samples: bool,
    has_bptt_payload: bool,
) -> None:
    """Enablement gate: never silently omit an active extra from probe ``F``."""
    if (
        float(getattr(cfg_phase2, "w_tc", 0.0) or 0.0) > 0.0
        and float(getattr(cfg_phase2, "tc_frac", 0.0) or 0.0) > 0.0
        and not has_tc_samples
    ):
        raise ReplayIncompleteError(
            "line search probe: TC is enabled but tc_kp/tc_kd were not recorded "
            "on the replay record (objective_replay checklist §3)"
        )
    if (
        float(getattr(cfg_phase2, "suffix_frac", 0.0) or 0.0) > 0.0
        and (
            float(getattr(cfg_phase2, "w_suffix_cost", 0.0) or 0.0) > 0.0
            or float(getattr(cfg_phase2, "w_suffix_tie", 0.0) or 0.0) > 0.0
        )
        and not has_sfx_samples
    ):
        # Suffix may be inactive when sfx_ramp==0; caller should only require
        # samples when the baseline actually ran the suffix term.
        raise ReplayIncompleteError(
            "line search probe: suffix term ran on baseline but suffix_offs "
            "were not recorded"
        )
    if include_bptt and not has_bptt_payload:
        raise ReplayIncompleteError(
            "line search probe: BPTT was included on this step but no "
            "BpttReplayPayload was recorded — BPTT replay is not yet wired "
            "(MVP: disable w_bptt / bptt_max_frac for LS, or freeze include_bptt=False)"
        )


def _positive_group_lrs(opt: AdamW) -> list[float]:
    return [float(g["lr"]) for g in opt.param_groups if float(g["lr"]) > 0.0]


def _step_vector_breakdown(
    adapter: AdamWLineSearchAdapter | LbfgsLineSearchAdapter,
    staged: Any,
    epsilon: float,
    moment_on: str,
) -> dict[str, float]:
    if isinstance(adapter, LbfgsLineSearchAdapter):
        return step_vector_breakdown_lbfgs(staged, epsilon)
    return step_vector_breakdown(staged, epsilon, moment_on)


def _evaluate_replayed_F(
    probe_F: Callable[[], float],
    *,
    replay_ctx: ObjectiveReplayContext | None,
    replay_rec: ReplayRecord | None,
) -> float:
    """Evaluate the bound objective without advancing registered RNG streams."""
    if replay_ctx is not None and replay_rec is not None:
        with replay_ctx.run_isolated(replay_rec):
            return float(probe_F())
    return float(probe_F())


def _write_actual_candidate(
    adapter: AdamWLineSearchAdapter | LbfgsLineSearchAdapter,
    staged: Any,
    epsilon: float,
) -> None:
    """Write the exact candidate that :meth:`commit` will execute."""
    if adapter.moment_on == "step":
        adapter.write_executed_step(epsilon, staged, into_params=True)
    else:
        adapter.write_candidate(epsilon, staged.p_exec, into_params=True)


def _next_actual_epsilon(
    epsilon: float,
    *,
    cfg: LineSearchConfig,
    epsilon_min: float,
    epsilon_max: float,
) -> float | None:
    """Geometrically backtrack an actual candidate within runtime bounds."""
    factor = float(cfg.fallback_shrink)
    if not (math.isfinite(factor) and 0.0 < factor < 1.0):
        factor = 0.5
    nxt = min(float(epsilon_max), max(float(epsilon_min), float(epsilon) * factor))
    if not math.isfinite(nxt) or nxt >= float(epsilon):
        return None
    return float(nxt)


def build_objective_direction_fingerprint(
    phase2_cfg: Any,
    *,
    al_lam: Mapping[str, float] | None = None,
    al_pen: Mapping[str, float] | None = None,
    stage_a: bool | None = None,
    torque_full_blend: float | None = None,
) -> dict[str, Any]:
    """JSON-stable selector semantics that invalidate acceptance history."""
    ls_cfg = phase2_cfg.line_search_cfg.with_forced_grad_balance()
    cost_cfg = getattr(phase2_cfg, "cost", None)

    def _float_map(values: Mapping[str, float] | None) -> dict[str, float] | None:
        if values is None:
            return None
        return {str(k): float(values[k]) for k in sorted(values)}

    return {
        "contract": (
            "raw_objective_gradient_curvature_direct_v1"
            if float(getattr(phase2_cfg, "curvature_direct_frac", 0.0) or 0.0) > 0.0
            else "raw_objective_gradient_v2"
        ),
        "optimizer": str(
            getattr(phase2_cfg, "optimizer", "adamw") or "adamw"
        ).lower(),
        "moment_on": str(ls_cfg.moment_on),
        "torque_physics": str(
            getattr(cost_cfg, "torque_physics", "unknown")
        ),
        "torque_full_blend": (
            None if torque_full_blend is None else float(torque_full_blend)
        ),
        "torque_grad_cap": float(
            getattr(phase2_cfg, "torque_grad_cap", 0.0) or 0.0
        ),
        "torque_grad_cap_refresh": int(
            getattr(phase2_cfg, "torque_grad_cap_refresh", 1) or 1
        ),
        "torque_grad_cap_scale_ema": float(
            getattr(phase2_cfg, "torque_grad_cap_scale_ema", 0.8)
        ),
        # Training-time DLS changes both the scalar objective and its direction;
        # never carry SaLSa acceptance history across an enable/weight change.
        "dls_terminal_frac": float(
            getattr(phase2_cfg, "dls_terminal_frac", 0.0) or 0.0
        ),
        "w_dls_terminal_cost": float(
            getattr(phase2_cfg, "w_dls_terminal_cost", 0.0) or 0.0
        ),
        "w_dls_terminal_tie": float(
            getattr(phase2_cfg, "w_dls_terminal_tie", 0.0) or 0.0
        ),
        "dls_terminal_steps": int(
            getattr(phase2_cfg, "dls_terminal_steps", 1) or 1
        ),
        "dls_terminal_window": int(
            getattr(phase2_cfg, "dls_terminal_window", 16) or 16
        ),
        "dls_terminal_damping": float(
            getattr(phase2_cfg, "dls_terminal_damping", 0.05)
        ),
        "dls_terminal_max_step_rad": float(
            getattr(phase2_cfg, "dls_terminal_max_step_rad", 0.35)
        ),
        # Do not invalidate older DLS selector history when the new mode is
        # disabled.  Enabling curvature (or changing its metric) *does* change
        # the scalar lookahead objective and must reset acceptance history.
        **({
            "terminal_lookahead_mode": "curvature",
            "curvature_damping": float(phase2_cfg.curvature_damping),
            "curvature_position_weight": float(
                phase2_cfg.curvature_position_weight
            ),
            "curvature_velocity_weight": float(
                phase2_cfg.curvature_velocity_weight
            ),
            "curvature_max_step_rad": float(phase2_cfg.curvature_max_step_rad),
        } if getattr(phase2_cfg, "terminal_lookahead_mode", "dls") == "curvature"
          else {}),
        **({
            "curvature_direct_frac": float(phase2_cfg.curvature_direct_frac),
            "curvature_direct_mix": float(phase2_cfg.curvature_direct_mix),
            "curvature_damping": float(phase2_cfg.curvature_damping),
            "curvature_position_weight": float(
                phase2_cfg.curvature_position_weight
            ),
            "curvature_velocity_weight": float(
                phase2_cfg.curvature_velocity_weight
            ),
        } if float(getattr(phase2_cfg, "curvature_direct_frac", 0.0) or 0.0) > 0.0
          else {}),
        "grad_clip": float(getattr(phase2_cfg, "grad_clip", 0.0) or 0.0),
        "strict_actual_update": bool(ls_cfg.strict_actual_update),
        "reject_current_probe_increase": bool(
            ls_cfg.reject_current_probe_increase
        ),
        "skip_on_probe_exhaustion": bool(ls_cfg.skip_on_probe_exhaustion),
        "force_search_on_al_change": bool(
            ls_cfg.force_search_on_al_change
        ),
        "search_on_early_triggers": bool(ls_cfg.search_on_early_triggers),
        "reseed_epsilon_on_objective_change": bool(
            ls_cfg.reseed_epsilon_on_objective_change
        ),
        "stage_a": None if stage_a is None else bool(stage_a),
        "al_lam": _float_map(al_lam),
        "al_pen": _float_map(al_pen),
    }


def init_line_search_runtime(
    *,
    phase2_cfg: Any,
    optimizer: AdamW,
    line_search_state: Mapping[str, Any] | None = None,
    opt_restored: bool = False,
    objective_direction_fingerprint: Mapping[str, Any] | None = None,
) -> LineSearchRuntime:
    """Resolve ε seed (fresh / selector / legacy) and build adapter + controllers."""
    requested_cfg = phase2_cfg.line_search_cfg.with_forced_grad_balance()
    ls_cfg = requested_cfg
    current_fingerprint = dict(
        objective_direction_fingerprint
        if objective_direction_fingerprint is not None
        else build_objective_direction_fingerprint(phase2_cfg)
    )
    fingerprint_reset_reason = None
    accept = AcceptanceEMAState()
    ctrl = IntervalControllerState()
    refs = TriggerRefs()
    seed_provenance = "fresh_cfg_lr"
    eta_ref: float

    if line_search_state is not None:
        ls_cfg, accept, ctrl, refs = load_line_search_state_dict(
            line_search_state, cfg=ls_cfg, accept=accept, ctrl=ctrl, refs=refs,
        )
        # Safety policy belongs to the current invocation.  A checkpoint may
        # restore selector tuning/rates, but must not disable or enable current
        # veto behavior behind the CLI's back.
        ls_cfg = dataclasses_replace(
            ls_cfg,
            actual_update_diagnostic_every=int(
                requested_cfg.actual_update_diagnostic_every
            ),
            moment_on=str(requested_cfg.moment_on),
            strict_actual_update=bool(requested_cfg.strict_actual_update),
            reject_current_probe_increase=bool(
                requested_cfg.reject_current_probe_increase
            ),
            skip_on_probe_exhaustion=bool(
                requested_cfg.skip_on_probe_exhaustion
            ),
            force_search_on_al_change=bool(
                requested_cfg.force_search_on_al_change
            ),
            search_interval_mode=str(requested_cfg.search_interval_mode),
            initial_interval=int(requested_cfg.initial_interval),
            min_interval=int(requested_cfg.min_interval),
            max_interval=int(requested_cfg.max_interval),
            warmup_updates=int(requested_cfg.warmup_updates),
            search_on_early_triggers=bool(
                requested_cfg.search_on_early_triggers
            ),
            reseed_epsilon_on_objective_change=bool(
                requested_cfg.reseed_epsilon_on_objective_change
            ),
        ).with_forced_grad_balance()
        saved_fingerprint = line_search_state.get(
            "objective_direction_fingerprint"
        )
        fingerprint_reset_reason = (
            "objective_fingerprint_missing"
            if not isinstance(saved_fingerprint, Mapping)
            else (
                "objective_fingerprint_changed"
                if dict(saved_fingerprint) != current_fingerprint
                else None
            )
        )
        eta = line_search_state.get("eta_ref")
        if eta is None or not (float(eta) > 0 and math.isfinite(float(eta))):
            # Fall back to last_selected or config seed.
            eta = ctrl.last_selected_epsilon or ls_cfg.epsilon_seed
        eta_ref = float(eta)
        ratios = line_search_state.get("fixed_group_ratios")
        if ratios is None:
            ratios = group_ratios_from_optimizer(optimizer, eta_ref)
        else:
            ratios = [float(r) for r in ratios]
        seed_provenance = str(
            line_search_state.get("seed_provenance") or "selector_checkpoint"
        )
        if ctrl.last_selected_epsilon is None:
            ctrl.last_selected_epsilon = float(eta_ref)
        emin, emax = resolve_runtime_bounds(ls_cfg, eta_ref)
        ctrl.last_selected_epsilon = clamp_epsilon(
            float(ctrl.last_selected_epsilon), emin,
        )
    elif opt_restored:
        override = ls_cfg.legacy_resume_seed
        if override is not None:
            eta_ref = float(override)
            seed_provenance = "legacy_resume_seed_override"
        else:
            lrs = _positive_group_lrs(optimizer)
            if not lrs:
                raise RuntimeError(
                    "line_search legacy resume: optimizer has no positive group LR; "
                    "set line_search_cfg.legacy_resume_seed or epsilon_seed explicitly "
                    "(plan §7)"
                )
            eta_ref = float(max(lrs))
            seed_provenance = "legacy_optimizer_group_lr"
        ls_cfg = dataclasses_replace(ls_cfg, epsilon_seed=float(eta_ref)).with_forced_grad_balance()
        emin, emax = resolve_runtime_bounds(ls_cfg, eta_ref)
        ctrl.last_selected_epsilon = clamp_epsilon(float(eta_ref), emin)
        ctrl.pending_search = True
        ctrl.K = 1
        ratios = group_ratios_from_optimizer(optimizer, eta_ref)
    else:
        # Fresh: prefer explicit ls epsilon_seed; if it is still the factory
        # default (1e-4) while Phase2Config.lr differs, inherit cfg.lr so
        # ``--lr`` alone seeds the selector (plan §7).
        seed = float(ls_cfg.epsilon_seed)
        if abs(seed - 1.0e-4) < 1e-15 and abs(float(phase2_cfg.lr) - seed) > 1e-15:
            seed = float(phase2_cfg.lr)
            seed_provenance = "fresh_cfg_lr"
        else:
            seed_provenance = "fresh_epsilon_seed"
        eta_ref = float(seed)
        ls_cfg = dataclasses_replace(ls_cfg, epsilon_seed=eta_ref).with_forced_grad_balance()
        emin, emax = resolve_runtime_bounds(ls_cfg, eta_ref)
        ctrl.last_selected_epsilon = clamp_epsilon(float(eta_ref), emin)
        ctrl.pending_search = True
        ratios = group_ratios_from_optimizer(optimizer, eta_ref)

    # Align live group LRs to clamped selector rate before the first step.
    if ls_cfg.mode == "fixed_epsilon":
        from .line_search import _bound_search_epsilon

        eps0 = _bound_search_epsilon(
            float(ls_cfg.epsilon_seed),
            epsilon_min=float(emin),
            epsilon_max=float(emax),
        )
        ctrl.pending_search = False
        ctrl.pending_trigger_reasons = []
        ctrl.last_selected_epsilon = float(eps0)
        eta_ref = float(eps0)
    else:
        eps0 = clamp_epsilon(
            float(ctrl.last_selected_epsilon or eta_ref), float(emin),
        )
    if fingerprint_reset_reason is not None:
        accept.reset()
        ctrl.K = 1
        ctrl.stable_search_streak = 0
        ctrl.stability_eligible = False
        ctrl.retry_seed = None
        ctrl.pending_search = True
        if fingerprint_reset_reason not in ctrl.pending_trigger_reasons:
            ctrl.pending_trigger_reasons.append(fingerprint_reset_reason)
    for gi, group in enumerate(optimizer.param_groups):
        group["lr"] = eps0 * float(ratios[gi])

    optimizer_kind = str(getattr(phase2_cfg, "optimizer", "adamw") or "adamw").lower()
    if optimizer_kind == "lbfgs":
        if float(getattr(phase2_cfg, "torque_grad_cap", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "line-search L-BFGS does not support torque_grad_cap: "
                "the cap must be replayed when forming every history gradient"
            )
        history_size = int(getattr(phase2_cfg, "lbfgs_history", 10) or 10)
        adapter = LbfgsLineSearchAdapter(
            optimizer,
            eta_ref=eta_ref,
            group_ratios=ratios,
            history_size=history_size,
        )
    elif optimizer_kind == "adamw":
        adapter = AdamWLineSearchAdapter(
            optimizer,
            eta_ref=eta_ref,
            group_ratios=ratios,
            moment_on=ls_cfg.moment_on,
        )
    else:
        raise ValueError(
            f"line_search optimizer must be 'adamw' or 'lbfgs', got {optimizer_kind!r}"
        )
    return LineSearchRuntime(
        cfg=ls_cfg,
        accept=accept,
        ctrl=ctrl,
        refs=refs,
        adapter=adapter,
        optimizer_kind=optimizer_kind,
        eta_ref=eta_ref,
        epsilon_min=float(emin),
        epsilon_max=float(emax),
        seed_provenance=seed_provenance,
        fixed_group_ratios=list(ratios),
        objective_direction_fingerprint=current_fingerprint,
        fingerprint_reset_reason=fingerprint_reset_reason,
    )


def fixed_epsilon_rate(rt: LineSearchRuntime) -> float:
    """Pinned scalar step for ``mode == \"fixed_epsilon\"``."""
    from .line_search import _bound_search_epsilon

    return _bound_search_epsilon(
        float(rt.cfg.epsilon_seed),
        epsilon_min=float(rt.epsilon_min),
        epsilon_max=float(rt.epsilon_max),
    )


def resolve_known_search(rt: LineSearchRuntime) -> tuple[bool, list[str], int]:
    """Return ``(known_search, reasons, j)`` for proposed update ``j = u+1``."""
    j = int(rt.ctrl.successful_update_count) + 1
    if rt.cfg.mode == "fixed_epsilon":
        return False, [], j
    run, reasons = should_search(rt.ctrl, rt.cfg, j)
    return run, reasons, j


def needs_line_search_direction(
    rt: LineSearchRuntime,
    *,
    known_search: bool,
    j: int,
) -> bool:
    """Whether this update needs the expensive transactional Adam direction.

    Gradient-mode AdamW on a scheduled non-search update is exactly the native
    optimizer step.  Parameter/gradient snapshots are only necessary when a
    search can still happen, an actual-step diagnostic is due, or a backend
    whose ordinary update is implemented by the adapter is selected.
    """
    diagnostic_every = int(rt.cfg.actual_update_diagnostic_every)
    diagnostic_due = bool(
        diagnostic_every > 0 and int(j) % diagnostic_every == 0
    )
    return bool(
        known_search
        or rt.cfg.search_on_early_triggers
        or rt.cfg.strict_actual_update
        or diagnostic_due
        or rt.optimizer_kind == "lbfgs"
        or getattr(rt.adapter, "moment_on", "gradient") == "step"
    )


def run_search_or_ordinary(
    rt: LineSearchRuntime,
    *,
    j: int,
    run_search: bool,
    search_reasons: Sequence[str],
    F0: float,
    g_preclip: Sequence[Tensor | None] | None,
    probe_F: Callable[[], float],
    g_objective: Sequence[Tensor | None] | None = None,
    replay_ctx: ObjectiveReplayContext | None = None,
    replay_rec: ReplayRecord | None = None,
) -> SearchStepResult:
    """Stage / trial / commit, or ordinary ``opt.step`` with retained ε.

    ``probe_F`` must evaluate complete ``F`` at the *current* live parameters
    (caller writes probe candidates via the adapter before calling).  It must
    not advance live RNG or write MPC/frontier state.
    """
    objective_grad = g_preclip if g_objective is None else g_objective
    floor = float(rt.epsilon_min)
    if rt.cfg.mode == "fixed_epsilon":
        eps_cur = fixed_epsilon_rate(rt)
        rt.ctrl.last_selected_epsilon = float(eps_cur)
    else:
        eps_cur = clamp_epsilon(
            float(rt.ctrl.last_selected_epsilon or rt.eta_ref), floor,
        )
    log: dict[str, Any] = {}
    diagnostic_every = int(rt.cfg.actual_update_diagnostic_every)
    actual_diagnostic_due = bool(
        diagnostic_every > 0 and int(j) % diagnostic_every == 0
    )

    if not run_search:
        u_breakdown: dict[str, float] = {}
        if rt.cfg.strict_actual_update or actual_diagnostic_due:
            rt.adapter.snapshot_theta0()
            st = rt.adapter.stage_from_q(
                g_objective=objective_grad,
                use_param_grad_as_q=True,
            )
            actual_trial_count = 0
            Factual = float("nan")
            epsilon = float(eps_cur)
            actual_accepted = False
            max_actual_trials = (
                int(rt.cfg.max_trials) if rt.cfg.strict_actual_update else 1
            )
            try:
                for actual_index in range(max_actual_trials):
                    _write_actual_candidate(rt.adapter, st, epsilon)
                    actual_trial_count = actual_index + 1
                    if params_finite(rt.adapter._params):
                        Factual = _evaluate_replayed_F(
                            probe_F,
                            replay_ctx=replay_ctx,
                            replay_rec=replay_rec,
                        )
                    else:
                        Factual = float("nan")
                    if not rt.cfg.strict_actual_update:
                        actual_accepted = True
                        break
                    if math.isfinite(Factual) and Factual <= float(F0):
                        actual_accepted = True
                        break
                    rt.adapter.restore_theta0()
                    if (actual_index + 1) >= max_actual_trials:
                        break
                    next_actual = _next_actual_epsilon(
                        epsilon,
                        cfg=rt.cfg,
                        epsilon_min=float(rt.epsilon_min),
                        epsilon_max=float(rt.epsilon_max),
                    )
                    if next_actual is None:
                        break
                    epsilon = float(next_actual)

                if rt.cfg.strict_actual_update and not actual_accepted:
                    failure_reason = (
                        "nonfinite_actual_objective"
                        if not math.isfinite(Factual)
                        else "actual_update_increase"
                    )
                    rt.adapter.discard(restore_params=True)
                    on_selection_failure(
                        rt.ctrl,
                        rt.accept,
                        last_tried_epsilon=float(epsilon),
                        fallback_shrink=float(rt.cfg.fallback_shrink),
                        epsilon_min=float(rt.epsilon_min),
                    )
                    log = logging_dict_from_search(
                        F0=F0,
                        Factual=Factual,
                        actual_decrease=float(F0) - float(Factual),
                        epsilon=float(epsilon),
                        trial_count=0,
                        a=st.stats.a,
                        P2=st.stats.P2,
                        s_exec=st.stats.s_exec,
                        s_momentum=st.stats.s_momentum,
                        s_decay=st.stats.s_decay,
                        ls_a_q=-st.stats.s_probe_q,
                        ls_s_exec_q=st.stats.s_exec_q,
                        K=int(rt.ctrl.K),
                        searched=False,
                        accepted=None,
                        failure_reason=failure_reason,
                        grad_balance="",
                        ls_actual_trial_count=actual_trial_count,
                        ls_actual_veto=1,
                    )
                    rt.last_log = log
                    return SearchStepResult(
                        executed=False,
                        searched=False,
                        epsilon=None,
                        log=log,
                        commit_side_effects=False,
                        failure_reason=failure_reason,
                    )

                u_breakdown = _step_vector_breakdown(
                    rt.adapter, st, epsilon, rt.cfg.moment_on,
                )
                rt.adapter.commit(
                    st, params_already_written=True, epsilon=float(epsilon),
                )
                rt.adapter.set_group_lrs(float(epsilon))
            except Exception:
                rt.adapter.discard(restore_params=True)
                raise

            if rt.cfg.strict_actual_update:
                rt.ctrl.last_selected_epsilon = float(epsilon)
            advance_successful_update(rt.ctrl, j)
            log = logging_dict_from_search(
                F0=F0,
                Factual=Factual,
                actual_decrease=float(F0) - float(Factual),
                epsilon=float(epsilon),
                K=int(rt.ctrl.K),
                searched=False,
                accepted=None,
                failure_reason=None,
                grad_balance="",
                ls_a_q=-st.stats.s_probe_q,
                ls_s_exec_q=st.stats.s_exec_q,
                ls_actual_trial_count=actual_trial_count,
                ls_actual_veto=0,
                **u_breakdown,
            )
            rt.last_log = log
            return SearchStepResult(
                executed=True,
                searched=False,
                epsilon=float(epsilon),
                log=log,
                commit_side_effects=True,
            )

        if isinstance(rt.adapter, LbfgsLineSearchAdapter):
            st = rt.adapter.execute_ordinary_step(
                eps_cur, g_preclip=objective_grad,
            )
            u_breakdown = _step_vector_breakdown(
                rt.adapter, st, eps_cur, rt.cfg.moment_on,
            )
        elif rt.adapter.moment_on == "step":
            st = rt.adapter.execute_ordinary_step(
                eps_cur, g_preclip=objective_grad,
            )
            if st is not None:
                u_breakdown = _step_vector_breakdown(
                    rt.adapter, st, eps_cur, rt.adapter.moment_on,
                )
        else:
            # Exact fast path for gradient-mode AdamW.  Staging computes the
            # same moment update as ``opt.step`` but also snapshots all model
            # parameters and constructs every SaLSa direction. None of those
            # tensors are consumed on an ordinary scheduled update.
            rt.adapter.set_group_lrs(eps_cur)
            rt.adapter.opt.step()
            rt.adapter.clear_transaction()
            u_breakdown["ls_fast_ordinary"] = 1.0
        advance_successful_update(rt.ctrl, j)
        log = logging_dict_from_search(
            epsilon=eps_cur,
            K=int(rt.ctrl.K),
            searched=False,
            accepted=None,
            grad_balance="",
            fixed_epsilon=(rt.cfg.mode == "fixed_epsilon"),
            **u_breakdown,
        )
        rt.last_log = log
        return SearchStepResult(
            executed=True, searched=False, epsilon=eps_cur, log=log,
            commit_side_effects=True,
        )

    if rt.cfg.mode == "fixed_epsilon":
        raise RuntimeError("fixed_epsilon mode must not enter searched update path")

    # ---- searched update ---------------------------------------------------
    apply_pending_history_reset(rt.accept)
    rt.adapter.snapshot_theta0()
    staged = rt.adapter.stage_from_q(
        g_objective=objective_grad,
        use_param_grad_as_q=True,
    )
    stats = staged.stats
    if not stats.probe_valid:
        rt.adapter.discard(restore_params=True)
        on_selection_failure(
            rt.ctrl, rt.accept,
            last_tried_epsilon=eps_cur,
            fallback_shrink=float(rt.cfg.fallback_shrink),
            epsilon_min=float(rt.epsilon_min),
        )
        log = logging_dict_from_search(
            F0=F0, a=stats.a, P2=stats.P2, s_exec=stats.s_exec,
            K=int(rt.ctrl.K), searched=True, accepted=False,
            failure_reason="invalid_probe_direction",
            trigger_reasons=list(search_reasons),
            grad_balance="",
            ls_a_q=-stats.s_probe_q,
            ls_s_exec_q=stats.s_exec_q,
        )
        rt.last_log = log
        return SearchStepResult(
            executed=False, searched=True, epsilon=None, log=log,
            commit_side_effects=False,
            failure_reason="invalid_probe_direction",
        )

    beta_eff, gap = effective_beta(
        rt.accept.last_history_update, j, float(rt.cfg.beta_accept_per_update),
    )
    B0 = float(rt.accept.B_prev)
    A0 = float(rt.accept.A_prev)
    # Curvature and Armijo use the exact scalar-objective derivative.  The
    # legacy armijo_a_cap_gnorm_sq field is intentionally diagnostic-only.
    a_armijo = float(stats.a)
    Bc = B_candidate(B0, a_armijo, beta_eff)

    seed_choice = choose_search_seed(
        rt.ctrl,
        rt.cfg,
        epsilon_max=rt.epsilon_max,
        epsilon_min=float(rt.epsilon_min),
        reasons=search_reasons,
    )
    epsilon = float(seed_choice.epsilon)
    selected = False
    Ac = float("nan")
    last_decision = None
    Fprobe_sel = None
    Fprobe_actual = None
    Factual = None
    actual_failure_reason = None
    actual_trial_count = 0
    trial_count = 0

    try:
        for trial_index in range(int(rt.cfg.max_trials)):
            trial_count = trial_index + 1
            rt.adapter.write_candidate(
                epsilon, staged.p_probe, into_params=True,
            )
            Fp = _evaluate_replayed_F(
                probe_F, replay_ctx=replay_ctx, replay_rec=replay_rec,
            )
            decision = evaluate_trial(
                Fprobe=Fp,
                F0=F0,
                epsilon=epsilon,
                a=a_armijo,
                P2=stats.P2,
                A_prev=A0,
                B_cand=Bc,
                beta_eff=beta_eff,
                c1=float(rt.cfg.c1),
                proposal_rule=str(rt.cfg.proposal_rule),
                backtrack_min_factor=float(rt.cfg.backtrack_min_factor),
                backtrack_max_factor=float(rt.cfg.backtrack_max_factor),
                fallback_shrink=float(rt.cfg.fallback_shrink),
                epsilon_min=float(rt.epsilon_min),
                trial_index=trial_index,
                max_trials=int(rt.cfg.max_trials),
                reject_current_probe_increase=bool(
                    rt.cfg.reject_current_probe_increase
                ),
            )
            last_decision = decision
            if decision.accept:
                if rt.cfg.strict_actual_update:
                    _write_actual_candidate(rt.adapter, staged, epsilon)
                    actual_trial_count += 1
                    Fprobe_actual = float(Fp)
                    if params_finite(rt.adapter._params):
                        Factual = _evaluate_replayed_F(
                            probe_F,
                            replay_ctx=replay_ctx,
                            replay_rec=replay_rec,
                        )
                    else:
                        Factual = float("nan")
                    rt.adapter.restore_theta0()
                    if math.isfinite(float(Factual)) and float(Factual) <= float(F0):
                        selected = True
                        Ac = float(decision.A_candidate)
                        Fprobe_sel = float(Fp)
                        actual_failure_reason = None
                        break
                    actual_failure_reason = (
                        "nonfinite_actual_objective"
                        if not math.isfinite(float(Factual))
                        else "actual_update_increase"
                    )
                    if (trial_index + 1) >= int(rt.cfg.max_trials):
                        break
                    next_actual = _next_actual_epsilon(
                        epsilon,
                        cfg=rt.cfg,
                        epsilon_min=float(rt.epsilon_min),
                        epsilon_max=float(rt.epsilon_max),
                    )
                    if next_actual is None:
                        break
                    epsilon = float(next_actual)
                    continue
                selected = True
                Ac = float(decision.A_candidate)
                Fprobe_sel = float(Fp)
                break
            if decision.stop or decision.next_epsilon is None:
                break
            epsilon = float(decision.next_epsilon)

        # Discard probe params; commit uses p_exec from theta0.
        rt.adapter.restore_theta0()

        last_tried_epsilon = float(epsilon)
        floor_commit = not selected
        if floor_commit:
            on_selection_failure(
                rt.ctrl, rt.accept,
                last_tried_epsilon=last_tried_epsilon,
                fallback_shrink=float(rt.cfg.fallback_shrink),
                epsilon_min=float(rt.epsilon_min),
            )
            skip_exhausted = bool(
                rt.cfg.strict_actual_update or rt.cfg.skip_on_probe_exhaustion
            )
            if skip_exhausted:
                rt.adapter.discard(restore_params=True)
                failure_reason = (
                    actual_failure_reason
                    if actual_failure_reason is not None
                    else "selection_exhausted"
                )
                discrepancy = (
                    None
                    if Factual is None or Fprobe_actual is None
                    else float(Factual) - float(Fprobe_actual)
                )
                log = logging_dict_from_search(
                    F0=F0,
                    Fprobe=Fprobe_actual,
                    Factual=Factual,
                    actual_decrease=(
                        None if Factual is None else float(F0) - float(Factual)
                    ),
                    probe_exec_discrepancy=discrepancy,
                    epsilon=last_tried_epsilon,
                    epsilon_seed_used=float(seed_choice.epsilon),
                    trial_count=trial_count,
                    a=stats.a,
                    P2=stats.P2,
                    s_exec=stats.s_exec,
                    s_momentum=stats.s_momentum,
                    s_decay=stats.s_decay,
                    A_prev=A0,
                    B_prev=B0,
                    B_cand=Bc,
                    beta_eff=beta_eff,
                    gap=gap,
                    R=None if last_decision is None else last_decision.R,
                    kappa=None if last_decision is None else last_decision.kappa,
                    K=int(rt.ctrl.K),
                    searched=True,
                    accepted=False,
                    trigger_reasons=list(search_reasons),
                    stable_streak=int(rt.ctrl.stable_search_streak),
                    failure_reason=failure_reason,
                    grad_balance="",
                    ls_a_q=-stats.s_probe_q,
                    ls_s_exec_q=stats.s_exec_q,
                    ls_armijo_a_cap_ignored=float(
                        rt.cfg.armijo_a_cap_gnorm_sq
                    ),
                    ls_floor_commit=0,
                    ls_actual_trial_count=actual_trial_count,
                    ls_actual_veto=1 if actual_failure_reason is not None else 0,
                )
                rt.last_log = log
                return SearchStepResult(
                    executed=False,
                    searched=True,
                    epsilon=None,
                    log=log,
                    commit_side_effects=False,
                    failure_reason=failure_reason,
                )
            epsilon = floor

        # Write executed step; commit moments (step mode) or ε·p_exec (gradient).
        u_breakdown = _step_vector_breakdown(
            rt.adapter, staged, float(epsilon), rt.cfg.moment_on,
        )
        _write_actual_candidate(rt.adapter, staged, epsilon)
        if not params_finite(rt.adapter._params):
            rt.adapter.discard(restore_params=True)
            on_selection_failure(
                rt.ctrl, rt.accept,
                last_tried_epsilon=last_tried_epsilon,
                fallback_shrink=float(rt.cfg.fallback_shrink),
                epsilon_min=float(rt.epsilon_min),
            )
            latch_triggers(rt.ctrl, ["nonfinite_exec_params"])
            log = logging_dict_from_search(
                F0=F0, Fprobe=Fprobe_sel, epsilon=epsilon,
                trial_count=trial_count, a=stats.a, P2=stats.P2,
                searched=True, accepted=False,
                failure_reason="nonfinite_exec_params",
                grad_balance="",
            )
            rt.last_log = log
            return SearchStepResult(
                executed=False, searched=True, epsilon=None, log=log,
                commit_side_effects=False,
                failure_reason="nonfinite_exec_params",
            )

        run_actual_diagnostic = bool(
            rt.cfg.strict_actual_update
            or actual_diagnostic_due
        )
        if run_actual_diagnostic and not rt.cfg.strict_actual_update:
            actual_trial_count += 1
            Factual = _evaluate_replayed_F(
                probe_F, replay_ctx=replay_ctx, replay_rec=replay_rec,
            )

        rt.adapter.commit(
            staged, params_already_written=True, epsilon=float(epsilon),
        )
        # Keep group LRs aligned with selected ε for logging / next ordinary step.
        rt.adapter.set_group_lrs(float(epsilon))
        if selected:
            commit_acceptance(rt.accept, Ac, Bc, j, beta_eff)
            update_interval_after_search(
                rt.ctrl, rt.cfg,
                epsilon_new=float(epsilon),
                trial_count=trial_count,
                j=j,
                trigger_reasons=search_reasons,
                epsilon_min=floor,
            )
        else:
            rt.ctrl.last_selected_epsilon = clamp_epsilon(float(epsilon), floor)
        advance_successful_update(rt.ctrl, j)
        decrease = None if Fprobe_sel is None else (F0 - float(Fprobe_sel))
        actual_decrease = (
            None if Factual is None else float(F0) - float(Factual)
        )
        discrepancy = (
            None
            if Factual is None or Fprobe_sel is None
            else float(Factual) - float(Fprobe_sel)
        )
        log = logging_dict_from_search(
            F0=F0,
            Fprobe=Fprobe_sel,
            Factual=Factual,
            actual_decrease=actual_decrease,
            probe_exec_discrepancy=discrepancy,
            epsilon=float(epsilon),
            epsilon_seed_used=float(seed_choice.epsilon),
            trial_count=trial_count,
            a=stats.a, P2=stats.P2,
            s_exec=stats.s_exec,
            s_momentum=stats.s_momentum,
            s_decay=stats.s_decay,
            decrease=decrease,
            A_prev=A0, A_cand=Ac if selected else None,
            B_prev=B0,
            B_cand=Bc,
            beta_eff=beta_eff, gap=gap,
            R=None if last_decision is None else last_decision.R,
            kappa=None if last_decision is None else last_decision.kappa,
            K=int(rt.ctrl.K),
            searched=True, accepted=selected,
            trigger_reasons=list(search_reasons),
            stable_streak=int(rt.ctrl.stable_search_streak),
            failure_reason=None if selected else "selection_exhausted",
            grad_balance="",
            ls_a_q=-stats.s_probe_q,
            ls_s_exec_q=stats.s_exec_q,
            ls_armijo_a_cap_ignored=float(rt.cfg.armijo_a_cap_gnorm_sq),
            ls_floor_commit=0 if selected else 1,
            ls_actual_trial_count=actual_trial_count,
            ls_actual_veto=0,
            **u_breakdown,
        )
        rt.last_log = log
        return SearchStepResult(
            executed=True, searched=True, epsilon=float(epsilon), log=log,
            commit_side_effects=True,
            failure_reason=None if selected else "selection_exhausted",
        )
    except Exception:
        rt.adapter.discard(restore_params=True)
        raise


def runtime_state_dict(rt: LineSearchRuntime) -> dict[str, Any]:
    return line_search_state_dict(
        rt.cfg, rt.accept, rt.ctrl, rt.refs,
        fixed_group_ratios=rt.fixed_group_ratios,
        eta_ref=rt.eta_ref,
        seed_provenance=rt.seed_provenance,
        objective_direction_fingerprint=rt.objective_direction_fingerprint,
    )
