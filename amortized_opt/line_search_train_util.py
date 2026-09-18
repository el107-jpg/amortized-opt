"""Pure helpers for SaLSa line-search train wiring (plan §14 commit gate).

Integrator contract: call these from ``train_phase2`` rather than inlining
policy.  Keeps tests green before the full loop lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .adamw_ls import AdamWLineSearchAdapter, StagedAdamW
from .line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    TriggerRefs,
    B_candidate,
    effective_beta,
    evaluate_trial,
    on_selection_failure,
)
from .line_search_resume import (
    pack_line_search_checkpoint_extra,
    unpack_line_search_checkpoint_extra,
)

__all__ = [
    "should_advance_controllers",
    "ProbeSearchOutcome",
    "search_epsilon_along_probe",
    "LineSearchCheckpointBundle",
    "pack_line_search_extra",
    "unpack_line_search_extra",
    "group_lrs_from_epsilon",
    "apply_selection_failure",
]


def should_advance_controllers(*, executed: bool, selection_failed: bool) -> bool:
    """Commit-gate for frontier EMA / ``viol_acc`` / ``al.update`` (plan §14).

    Advance only when a parameter update actually executed.  Selection failure
    and nonfinite / skipped ``opt.step`` paths must not advance those
    controllers (dual-clock vs ``al_every``).
    """
    if selection_failed:
        return False
    return bool(executed)


@dataclass
class ProbeSearchOutcome:
    selected: bool
    epsilon: float | None
    trial_count: int
    A_candidate: float | None
    B_candidate: float | None
    last_tried_epsilon: float | None
    failure_reason: str | None
    beta_eff: float
    gap: int


def search_epsilon_along_probe(
    *,
    evaluate_F: Callable[[], float],
    adapter: AdamWLineSearchAdapter,
    staged: StagedAdamW,
    F0: float,
    cfg: LineSearchConfig,
    accept: AcceptanceEMAState,
    j: int,
    epsilon_seed: float | None = None,
) -> ProbeSearchOutcome:
    """Host trial loop along ``p_probe``; restores ``theta0``; does not touch A/B.

    Caller commits acceptance EMA only after a successful executed update.
    On failure the caller should invoke :func:`on_selection_failure`.
    """
    if not staged.stats.probe_valid:
        return ProbeSearchOutcome(
            selected=False,
            epsilon=None,
            trial_count=0,
            A_candidate=None,
            B_candidate=None,
            last_tried_epsilon=None,
            failure_reason="invalid_probe",
            beta_eff=1.0,
            gap=1,
        )

    beta_eff, gap = effective_beta(
        accept.last_history_update, int(j), cfg.beta_accept_per_update
    )
    Bc = B_candidate(accept.B_prev, staged.stats.a, beta_eff)
    seed = float(cfg.epsilon_seed if epsilon_seed is None else epsilon_seed)
    emin, _emax = cfg.resolve_bounds(seed)
    epsilon = seed
    last_tried: float | None = None
    last_reason = "budget_exhausted"

    for trial_index in range(int(cfg.max_trials)):
        last_tried = float(epsilon)
        adapter.write_candidate(epsilon, staged.p_probe, into_params=True)
        try:
            Fprobe = float(evaluate_F())
        finally:
            adapter.restore_theta0()

        decision = evaluate_trial(
            Fprobe=Fprobe,
            F0=float(F0),
            epsilon=float(epsilon),
            a=staged.stats.a,
            P2=staged.stats.P2,
            A_prev=accept.A_prev,
            B_cand=Bc,
            beta_eff=beta_eff,
            c1=cfg.c1,
            proposal_rule=cfg.proposal_rule,
            backtrack_min_factor=cfg.backtrack_min_factor,
            backtrack_max_factor=cfg.backtrack_max_factor,
            fallback_shrink=cfg.fallback_shrink,
            epsilon_min=emin,
            trial_index=trial_index,
            max_trials=int(cfg.max_trials),
            reject_current_probe_increase=bool(
                cfg.reject_current_probe_increase
            ),
        )
        if decision.accept:
            return ProbeSearchOutcome(
                selected=True,
                epsilon=float(epsilon),
                trial_count=trial_index + 1,
                A_candidate=float(decision.A_candidate),
                B_candidate=float(decision.B_candidate),
                last_tried_epsilon=float(epsilon),
                failure_reason=None,
                beta_eff=float(beta_eff),
                gap=int(gap),
            )
        last_reason = decision.reason
        if decision.stop or decision.next_epsilon is None:
            break
        epsilon = float(decision.next_epsilon)

    return ProbeSearchOutcome(
        selected=False,
        epsilon=None,
        trial_count=int(cfg.max_trials) if last_tried is not None else 0,
        A_candidate=None,
        B_candidate=None,
        last_tried_epsilon=last_tried,
        failure_reason=last_reason,
        beta_eff=float(beta_eff),
        gap=int(gap),
    )


def group_lrs_from_epsilon(
    epsilon: float, fixed_group_ratios: Sequence[float]
) -> list[float]:
    """``lr_g = ε * r_g`` for ordinary reuse / post-commit group LR writes."""
    eps = float(epsilon)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"epsilon must be positive finite, got {eps}")
    return [eps * float(r) for r in fixed_group_ratios]


@dataclass
class LineSearchCheckpointBundle:
    """Full selector blob for ``extra['line_search']`` in phase-2 checkpoints."""

    cfg: LineSearchConfig
    accept: AcceptanceEMAState
    ctrl: IntervalControllerState
    refs: TriggerRefs
    fixed_group_ratios: list[float] | None = None
    eta_ref: float | None = None
    seed_provenance: str | None = None
    objective_direction_fingerprint: dict[str, Any] | None = None
    last_effective_lrs: list[float] | None = None
    raw: dict[str, Any] | None = None


def pack_line_search_extra(
    cfg: LineSearchConfig,
    accept: AcceptanceEMAState,
    ctrl: IntervalControllerState,
    refs: TriggerRefs | None = None,
    *,
    fixed_group_ratios: Sequence[float] | None = None,
    eta_ref: float | None = None,
    seed_provenance: str | None = None,
    last_effective_lrs: Sequence[float] | None = None,
    optimizer: Any | None = None,
    objective_direction_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the dict stored under checkpoint ``extra['line_search']``.

    Delegates to :func:`pack_line_search_checkpoint_extra` so §7/§11 fields
    (``last_effective_lrs``, ``grad_balance=""``, provenance) stay in sync.
    """
    frag = pack_line_search_checkpoint_extra(
        cfg,
        accept,
        ctrl,
        refs,
        fixed_group_ratios=fixed_group_ratios,
        eta_ref=eta_ref,
        seed_provenance=seed_provenance,
        last_effective_lrs=last_effective_lrs,
        optimizer=optimizer,
        objective_direction_fingerprint=objective_direction_fingerprint,
    )
    return frag["line_search"]


def unpack_line_search_extra(
    extra: Mapping[str, Any] | None,
    *,
    key: str = "line_search",
) -> LineSearchCheckpointBundle | None:
    """Restore selector state from checkpoint ``extra``; ``None`` if absent."""
    if not extra or key not in extra:
        return None
    blob = extra[key]
    if not isinstance(blob, Mapping):
        raise TypeError(f"extra[{key!r}] must be a mapping, got {type(blob)}")
    # Normalize alternate keys into the resume helper's expected shape.
    wrapped: Mapping[str, Any] = (
        extra if key == "line_search" else {"line_search": blob}
    )
    cfg, accept, ctrl, refs, meta = unpack_line_search_checkpoint_extra(wrapped)
    return LineSearchCheckpointBundle(
        cfg=cfg,
        accept=accept,
        ctrl=ctrl,
        refs=refs,
        fixed_group_ratios=meta.get("fixed_group_ratios"),
        eta_ref=meta.get("eta_ref"),
        seed_provenance=meta.get("seed_provenance"),
        objective_direction_fingerprint=meta.get(
            "objective_direction_fingerprint"
        ),
        last_effective_lrs=meta.get("last_effective_lrs"),
        raw=dict(blob),
    )


def apply_selection_failure(
    ctrl: IntervalControllerState,
    accept: AcceptanceEMAState,
    outcome: ProbeSearchOutcome,
    *,
    fallback_shrink: float = 0.5,
    epsilon_min: float = 1e-10,
) -> None:
    """Thin wrapper so train can share the failure path with tests."""
    on_selection_failure(
        ctrl,
        accept,
        last_tried_epsilon=outcome.last_tried_epsilon,
        fallback_shrink=fallback_shrink,
        epsilon_min=epsilon_min,
    )
