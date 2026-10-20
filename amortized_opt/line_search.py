"""SaLSa-style learning-rate selection (library): config, EMAs, trials, interval.

Train-loop wiring lives elsewhere. This module assumes plain ``∇F`` (no
``grad_balance`` surgery): probe and exec both use clipped ``q`` in ``.grad``;
``p_probe = -D * q``, ``p_exec`` = AdamW momentum from ``q`` plus decoupled decay.

See ``docs/parameter_line_search_plan (2).md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Sequence

__all__ = [
    "LineSearchConfig",
    "AcceptanceEMAState",
    "IntervalControllerState",
    "TriggerRefs",
    "TrialDecision",
    "SearchSeedChoice",
    "force_grad_balance_off",
    "effective_beta",
    "B_candidate",
    "A_candidate",
    "accepts_smoothed",
    "commit_acceptance",
    "curvature_residual",
    "epsilon_floor",
    "clamp_epsilon",
    "resolve_runtime_bounds",
    "propose_next_epsilon",
    "evaluate_trial",
    "choose_search_seed",
    "should_search",
    "check_early_triggers",
    "update_interval_after_search",
    "on_selection_failure",
    "retry_seed_after_failure",
    "apply_pending_history_reset",
    "objective_reseed_epsilon",
    "reset_history_for_objective_change",
    "advance_successful_update",
    "latch_triggers",
    "logging_dict_from_search",
    "line_search_state_dict",
    "load_line_search_state_dict",
]

_STATE_VERSION = 2


# --------------------------------------------------------------------------- config
@dataclass
class LineSearchConfig:
    """Initial engineering defaults from the plan (Section 6)."""

    mode: str = "salsa_transfer"
    acceptance: str = "smoothed_probe"
    # Forced empty whenever line search is enabled (recipe fork).
    grad_balance: str = ""
    beta_accept_per_update: float = 0.99
    c1: float = 0.5
    #: Optional easy-to-strict Armijo curriculum. ``None`` or a non-positive
    #: ramp preserves the fixed-``c1`` rule exactly. The schedule is keyed to
    #: successful optimizer-update slots, not wall time or probe count.
    c1_start: float | None = None
    c1_ramp_updates: int = 0
    proposal_rule: str = "quadratic_with_safeguards"  # or "geometric"
    expansion_factor: float = 1.1
    backtrack_min_factor: float = 0.1
    backtrack_max_factor: float = 0.5
    fallback_shrink: float = 0.5
    max_trials: int = 6
    epsilon_seed: float = 1.0e-4
    epsilon_max_mult: float = 10.0
    epsilon_min_mult: float = 1.0e-6
    # Absolute bounds; if None, derived from seed * mult at resolve time.
    epsilon_max: float | None = None
    epsilon_min: float | None = None
    #: Optional high-to-low cosine curriculum for the live epsilon floor.
    #: ``epsilon_min`` remains the final floor and checkpoint/runtime bound.
    epsilon_min_start: float | None = None
    epsilon_min_ramp_updates: int = 0
    search_interval_mode: str = "adaptive"  # or "fixed"
    initial_interval: int = 1
    min_interval: int = 1
    max_interval: int = 10
    warmup_updates: int = 10
    #: Keep Armijo completely inactive through this many successful updates.
    #: The ordinary AdamW path follows ``epsilon_min_at`` during the delay;
    #: the first search therefore starts from the completed cosine floor.
    activate_after_updates: int = 0
    stable_searches_to_increase: int = 3
    stable_rate_ratio_bound: float = 1.15
    stable_max_trials: int = 2
    substantial_rate_drop_ratio: float = 0.75
    costly_search_trials: int = 3
    # Early-trigger thresholds (pure checks; refs owned by TriggerRefs).
    violation_ratio_trigger: float = 2.0
    grad_norm_ratio_trigger: float = 3.0
    grad_norm_ref_floor: float = 1.0e-12
    #: Permit gradient/violation spikes to force a search before the scheduled
    #: interval. Throughput-oriented runs can disable this while retaining
    #: scheduled and pending/nonfinite recovery searches.
    search_on_early_triggers: bool = True
    true_gradient_mode: str = "plain_total_backward"
    curriculum_history_policy: str = "approximate_gradual"
    legacy_resume_seed: float | None = None
    grouped_scaling: bool = False
    #: Optional same-batch ``F(theta_exec)`` diagnostic cadence; 0 = off.
    actual_update_diagnostic_every: int = 0
    #: Ablation: re-check / veto the executed momentum step (off by default).
    strict_actual_update: bool = False
    #: Reject a current-batch probe loss increase even when the acceptance EMA
    #: would admit it.  Off preserves the original smoothed SaLSa rule.
    reject_current_probe_increase: bool = False
    #: On probe budget exhaustion, skip the update instead of committing the
    #: configured epsilon floor.  Off preserves the legacy floor-commit path.
    skip_on_probe_exhaustion: bool = False
    #: A failed stochastic minibatch should not permanently reduce the rate
    #: that worked on prior batches. Retry the next batch from that rate.
    retry_retained_rate: bool = False
    enabled: bool = True
    #: Deprecated compatibility field. It is checkpointed and logged but never
    #: changes Armijo/curvature, which require the exact scalar-objective
    #: directional derivative.
    armijo_a_cap_gnorm_sq: float = 1.0
    #: On discrete objective / AL jumps, reseed ε to ``mult * last_ε`` (clamped).
    objective_reseed_mult: float = 100.0
    #: When False (default), stage-A / forced objective jumps only reset A/B
    #: history and force search; ``last_selected_epsilon`` is kept for SaLSa
    #: to adapt.  AL ``al_every`` updates do not force search unless
    #: ``force_search_on_al_change`` is True.
    reseed_epsilon_on_objective_change: bool = False
    #: When True, an actual ``al.update`` λ/pen change resets SaLSa A/B, sets
    #: ``K=1``, and forces the next search (plan §8 default).  When False
    #: (default), AL moves freely: only refresh ``al_*_snapshot`` and let the
    #: interval controller / triggers decide searches.
    force_search_on_al_change: bool = False
    #: ``gradient``: EMA ``m,v`` on clipped ``q`` (SaLSa transfer). ``step``: EMA on
    #: executed step ``u = ε p_probe + p_mom + p_decay`` (carry not scaled by ε).
    moment_on: str = "gradient"

    def c1_at(self, update: int) -> float:
        """Cosine-ramp Armijo ``c1`` from ``c1_start`` to the final ``c1``."""
        if self.c1_start is None or int(self.c1_ramp_updates) <= 0:
            return float(self.c1)
        u = min(max(float(update) / float(self.c1_ramp_updates), 0.0), 1.0)
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        return float(self.c1_start) + (float(self.c1) - float(self.c1_start)) * blend

    def epsilon_min_at(self, update: int, final_floor: float) -> float:
        """Cosine-ramp the live epsilon floor to ``final_floor``."""
        if self.epsilon_min_start is None or int(self.epsilon_min_ramp_updates) <= 0:
            return float(final_floor)
        u = min(
            max(float(update) / float(self.epsilon_min_ramp_updates), 0.0),
            1.0,
        )
        blend = 0.5 * (1.0 - math.cos(math.pi * u))
        return float(self.epsilon_min_start) + (
            float(final_floor) - float(self.epsilon_min_start)
        ) * blend

    def active_at(self, update: int) -> bool:
        """Whether Armijo probing is active for proposed update ``update``."""
        return int(update) > max(int(self.activate_after_updates), 0)

    def resolve_bounds(self, seed: float | None = None) -> tuple[float, float]:
        """Return ``(epsilon_min, epsilon_max)`` for a resolved seed."""
        s = float(self.epsilon_seed if seed is None else seed)
        emin = self.epsilon_min
        emax = self.epsilon_max
        if emin is None:
            emin = self.epsilon_min_mult * s
        if emax is None:
            emax = self.epsilon_max_mult * s
        if not (math.isfinite(emin) and math.isfinite(emax) and emin > 0 and emax >= emin):
            raise ValueError(f"invalid epsilon bounds: min={emin}, max={emax}, seed={s}")
        if not (emin <= s <= emax):
            raise ValueError(
                f"epsilon seed {s} outside bounds [{emin}, {emax}]; "
                "set explicit bounds that contain the seed"
            )
        return float(emin), float(emax)

    def with_forced_grad_balance(self) -> LineSearchConfig:
        if self.grad_balance == "":
            return self
        return dataclasses_replace(self, grad_balance="")


def dataclasses_replace(obj: LineSearchConfig, **kwargs: Any) -> LineSearchConfig:
    d = asdict(obj)
    d.update(kwargs)
    return LineSearchConfig(**d)


def force_grad_balance_off(grad_balance: str | None, line_search_enabled: bool) -> str:
    """When LS is on, always return ``\"\"`` regardless of launcher value."""
    if line_search_enabled:
        return ""
    return "" if grad_balance is None else str(grad_balance)


def epsilon_floor(cfg: LineSearchConfig, eta_ref: float) -> float:
    """Absolute step floor: ``epsilon_min_mult * eta_ref``, at least ``1e-10``."""
    if cfg.epsilon_min is not None:
        return float(cfg.epsilon_min)
    return max(1e-10, float(cfg.epsilon_min_mult) * float(eta_ref))


def clamp_epsilon(eps: float, floor: float) -> float:
    """Lower-clamp selector step / group LR (``max(eps, floor)``)."""
    f = float(floor)
    if not math.isfinite(float(eps)):
        return f
    return max(f, float(eps))


def resolve_runtime_bounds(
    cfg: LineSearchConfig, eta_ref: float
) -> tuple[float, float]:
    """``(epsilon_min, epsilon_max)`` for live search — always anchored to ``eta_ref``."""
    ref = float(eta_ref)
    emin = epsilon_floor(cfg, ref)
    emax = cfg.epsilon_max
    if emax is None:
        emax = float(cfg.epsilon_max_mult) * ref
    else:
        emax = float(emax)
    return emin, max(emax, emin)


def _bound_search_epsilon(
    eps: float, *, epsilon_min: float, epsilon_max: float
) -> float:
    return min(clamp_epsilon(eps, epsilon_min), float(epsilon_max))


# ---------------------------------------------------------------- acceptance EMA
@dataclass
class AcceptanceEMAState:
    """Smoothed probe decrease (A) and positive slope magnitude (B) history."""

    A_prev: float = 0.0
    B_prev: float = 0.0
    n_history: int = 0
    last_history_update: int | None = None
    retention_product: float = 1.0
    pending_history_reset: bool = False

    def reset(self) -> None:
        self.A_prev = 0.0
        self.B_prev = 0.0
        self.n_history = 0
        self.last_history_update = None
        self.retention_product = 1.0
        self.pending_history_reset = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "A_prev": float(self.A_prev),
            "B_prev": float(self.B_prev),
            "n_history": int(self.n_history),
            "last_history_update": self.last_history_update,
            "retention_product": float(self.retention_product),
            "pending_history_reset": bool(self.pending_history_reset),
        }

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.A_prev = float(d.get("A_prev", 0.0))
        self.B_prev = float(d.get("B_prev", 0.0))
        self.n_history = int(d.get("n_history", 0))
        lhu = d.get("last_history_update", None)
        self.last_history_update = None if lhu is None else int(lhu)
        self.retention_product = float(d.get("retention_product", 1.0))
        self.pending_history_reset = bool(d.get("pending_history_reset", False))

    def bias_corrected(self) -> tuple[float, float]:
        """Optional reporting: divide by ``1 - retention_product`` (variable gaps)."""
        denom = 1.0 - self.retention_product
        if abs(denom) < 1e-15:
            return self.A_prev, self.B_prev
        return self.A_prev / denom, self.B_prev / denom


def effective_beta(
    last_history_update: int | None,
    j: int,
    beta_accept_per_update: float,
) -> tuple[float, int]:
    """Return ``(beta_effective, gap)`` for proposed update index ``j``."""
    gap = 1 if last_history_update is None else int(j) - int(last_history_update)
    if gap < 1:
        gap = 1
    return float(beta_accept_per_update) ** gap, gap


def B_candidate(B_prev: float, a: float, beta_eff: float) -> float:
    return float(beta_eff) * float(B_prev) + (1.0 - float(beta_eff)) * float(a)


def A_candidate(A_prev: float, decrease: float, beta_eff: float) -> float:
    return float(beta_eff) * float(A_prev) + (1.0 - float(beta_eff)) * float(decrease)


def accepts_smoothed(
    A_cand: float,
    B_cand: float,
    epsilon: float,
    c1: float,
) -> bool:
    return float(A_cand) >= float(c1) * float(epsilon) * float(B_cand)


def commit_acceptance(
    state: AcceptanceEMAState,
    A_cand: float,
    B_cand: float,
    j: int,
    beta_eff: float,
) -> None:
    """Commit A/B once after a searched update is successfully executed."""
    state.A_prev = float(A_cand)
    state.B_prev = float(B_cand)
    state.n_history = int(state.n_history) + 1
    state.last_history_update = int(j)
    state.retention_product = float(state.retention_product) * float(beta_eff)


# --------------------------------------------------------------- trial / curvature
@dataclass
class TrialDecision:
    accept: bool
    stop: bool
    next_epsilon: float | None
    decrease: float
    A_candidate: float
    B_candidate: float
    R: float | None
    kappa: float | None
    L_measured: float | None
    epsilon_quad: float | None
    reason: str
    Fprobe_finite: bool


def curvature_residual(
    Fprobe: float,
    F0: float,
    epsilon: float,
    a: float,
) -> tuple[float, float, float]:
    """Return ``(R, kappa, decrease)`` with ``s_probe = -a``.

    ``R = Fprobe - F0 - epsilon * s_probe = epsilon * a - decrease``.
    """
    decrease = float(F0) - float(Fprobe)
    R = float(epsilon) * float(a) - decrease
    if abs(float(epsilon)) < 1e-30:
        kappa = float("nan")
    else:
        kappa = 2.0 * R / (float(epsilon) ** 2)
    return R, kappa, decrease


def _is_resolved_positive(R: float, epsilon: float, a: float) -> bool:
    """Detect cancellation / unresolved residual → fall back to geometric shrink."""
    if not math.isfinite(R) or R <= 0.0:
        return False
    # Relative cancellation: R tiny vs scale of F terms in the residual.
    scale = max(abs(float(epsilon) * float(a)), abs(R), 1e-30)
    return abs(R) > 1e-12 * scale


def propose_next_epsilon(
    *,
    epsilon: float,
    a: float,
    P2: float,
    R: float,
    proposal_rule: str,
    backtrack_min_factor: float,
    backtrack_max_factor: float,
    fallback_shrink: float,
    epsilon_min: float,
) -> tuple[float | None, float | None, float | None, str]:
    """Propose next trial ε after rejection. Returns ``(eps, eps_quad, L, reason)``.

    Clamps a valid quadratic proposal to ``[min_factor, max_factor] * epsilon``.
    Invalid / nonfinite residual → geometric shrink. Proposals below ``epsilon_min``
    are clamped to the floor (not rejected).
    """
    L_measured = None
    eps = float(epsilon)
    eps2 = eps * eps
    if math.isfinite(P2) and P2 > 0.0 and math.isfinite(R) and math.isfinite(eps2) and eps2 > 0.0:
        kappa = 2.0 * R / eps2
        if math.isfinite(kappa):
            L_measured = max(0.0, kappa / float(P2))

    eps_quad = None
    if proposal_rule == "quadratic_with_safeguards" and _is_resolved_positive(R, epsilon, a):
        if math.isfinite(eps2) and eps2 > 0.0 and math.isfinite(R) and abs(R) > 0.0:
            eps_quad = float(a) * eps2 / (2.0 * float(R))
        lo = float(backtrack_min_factor) * eps
        hi = float(backtrack_max_factor) * eps
        if math.isfinite(eps_quad) and eps_quad > 0.0:
            nxt = min(max(eps_quad, lo), hi)
            reason = "quadratic_clamp"
        else:
            nxt = float(fallback_shrink) * float(epsilon)
            reason = "quadratic_invalid_halve"
            eps_quad = None
    else:
        nxt = float(fallback_shrink) * float(epsilon)
        reason = "geometric_shrink" if proposal_rule == "geometric" else "residual_halve"

    if not math.isfinite(nxt):
        return None, eps_quad, L_measured, "nonfinite_proposal"
    floor = float(epsilon_min)
    if nxt < floor:
        reason = f"{reason}_clamped_to_floor"
        nxt = floor
    return float(nxt), eps_quad, L_measured, reason


def retry_seed_after_failure(
    last_tried_epsilon: float,
    *,
    fallback_shrink: float,
    epsilon_min: float,
) -> float:
    """``retry_seed`` after budget exhaustion: clamp to ``epsilon_min``, no halving.

    Do not apply ``fallback_shrink`` here — trial backtracking already reduced ε.
    On failure, keep ``max(last_tried, epsilon_min)`` so repeated failures sit on
    the floor instead of denormal death-spiral halving.
    """
    del fallback_shrink  # kept for call-site stability; shrink is trial-only
    emin = float(epsilon_min)
    lt = float(last_tried_epsilon)
    if not math.isfinite(lt):
        return emin
    return max(emin, lt)


def evaluate_trial(
    *,
    Fprobe: float,
    F0: float,
    epsilon: float,
    a: float,
    P2: float,
    A_prev: float,
    B_cand: float,
    beta_eff: float,
    c1: float,
    proposal_rule: str = "quadratic_with_safeguards",
    backtrack_min_factor: float = 0.1,
    backtrack_max_factor: float = 0.5,
    fallback_shrink: float = 0.5,
    epsilon_min: float = 1e-10,
    trial_index: int = 0,
    max_trials: int = 6,
    reject_current_probe_increase: bool = False,
) -> TrialDecision:
    """One probe trial: smoothed Armijo test + safeguarded next-ε proposal."""
    Fprobe_finite = math.isfinite(float(Fprobe))
    if not Fprobe_finite:
        nxt, eps_quad, L_m, why = propose_next_epsilon(
            epsilon=epsilon,
            a=a,
            P2=P2,
            R=float("nan"),
            proposal_rule="geometric",
            backtrack_min_factor=backtrack_min_factor,
            backtrack_max_factor=backtrack_max_factor,
            fallback_shrink=fallback_shrink,
            epsilon_min=epsilon_min,
        )
        stop = nxt is None or (trial_index + 1) >= max_trials
        return TrialDecision(
            accept=False,
            stop=stop,
            next_epsilon=None if stop else nxt,
            decrease=float("nan"),
            A_candidate=float("nan"),
            B_candidate=float(B_cand),
            R=None,
            kappa=None,
            L_measured=L_m,
            epsilon_quad=eps_quad,
            reason="nonfinite_Fprobe",
            Fprobe_finite=False,
        )

    R, kappa, decrease = curvature_residual(Fprobe, F0, epsilon, a)
    A_cand = A_candidate(A_prev, decrease, beta_eff)
    smoothed_accept = accepts_smoothed(A_cand, B_cand, epsilon, c1)
    current_increase_veto = (
        bool(reject_current_probe_increase) and decrease < 0.0
    )
    if smoothed_accept and not current_increase_veto:
        return TrialDecision(
            accept=True,
            stop=True,
            next_epsilon=None,
            decrease=decrease,
            A_candidate=A_cand,
            B_candidate=float(B_cand),
            R=R,
            kappa=kappa if math.isfinite(kappa) else None,
            L_measured=(max(0.0, kappa / P2) if (math.isfinite(kappa) and P2 > 0) else None),
            epsilon_quad=None,
            reason="accepted",
            Fprobe_finite=True,
        )

    nxt, eps_quad, L_m, why = propose_next_epsilon(
        epsilon=epsilon,
        a=a,
        P2=P2,
        R=R,
        proposal_rule=proposal_rule,
        backtrack_min_factor=backtrack_min_factor,
        backtrack_max_factor=backtrack_max_factor,
        fallback_shrink=fallback_shrink,
        epsilon_min=epsilon_min,
    )
    trials_left = (trial_index + 1) < max_trials
    stop = (nxt is None) or (not trials_left)
    reason_prefix = (
        "rejected_current_probe_increase_"
        if current_increase_veto
        else "rejected_"
    )
    return TrialDecision(
        accept=False,
        stop=stop,
        next_epsilon=None if stop else nxt,
        decrease=decrease,
        A_candidate=A_cand,
        B_candidate=float(B_cand),
        R=R,
        kappa=kappa if math.isfinite(kappa) else None,
        L_measured=L_m,
        epsilon_quad=eps_quad,
        reason=reason_prefix + why,
        Fprobe_finite=True,
    )


# ----------------------------------------------------------- interval / triggers
@dataclass
class IntervalControllerState:
    """Adaptive search cadence ``K ∈ [1, 10]`` and rate seeds."""

    successful_update_count: int = 0  # u
    last_search_update: int | None = None
    K: int = 1
    stable_search_streak: int = 0
    last_selected_epsilon: float | None = None
    retry_seed: float | None = None
    pending_search: bool = True
    pending_trigger_reasons: list[str] = field(default_factory=list)
    searches_completed: int = 0
    # First search after init / objective reset does not establish stability.
    stability_eligible: bool = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "successful_update_count": int(self.successful_update_count),
            "last_search_update": self.last_search_update,
            "K": int(self.K),
            "stable_search_streak": int(self.stable_search_streak),
            "last_selected_epsilon": self.last_selected_epsilon,
            "retry_seed": self.retry_seed,
            "pending_search": bool(self.pending_search),
            "pending_trigger_reasons": list(self.pending_trigger_reasons),
            "searches_completed": int(self.searches_completed),
            "stability_eligible": bool(self.stability_eligible),
        }

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.successful_update_count = int(d.get("successful_update_count", 0))
        lsu = d.get("last_search_update", None)
        self.last_search_update = None if lsu is None else int(lsu)
        self.K = int(d.get("K", 1))
        self.stable_search_streak = int(d.get("stable_search_streak", 0))
        lse = d.get("last_selected_epsilon", None)
        self.last_selected_epsilon = None if lse is None else float(lse)
        rs = d.get("retry_seed", None)
        self.retry_seed = None if rs is None else float(rs)
        self.pending_search = bool(d.get("pending_search", True))
        self.pending_trigger_reasons = list(d.get("pending_trigger_reasons", []))
        self.searches_completed = int(d.get("searches_completed", 0))
        self.stability_eligible = bool(d.get("stability_eligible", False))


@dataclass
class TriggerRefs:
    """Moving references for cheap early-search triggers (updated per valid baseline)."""

    gnorm_ema: float | None = None
    gnorm_ema_decay: float = 0.99
    viol_ema: dict[str, float] = field(default_factory=dict)
    viol_ema_decay: float = 0.99
    viol_abs_thresholds: dict[str, float] = field(default_factory=dict)
    initialized: bool = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "gnorm_ema": self.gnorm_ema,
            "gnorm_ema_decay": float(self.gnorm_ema_decay),
            "viol_ema": dict(self.viol_ema),
            "viol_ema_decay": float(self.viol_ema_decay),
            "viol_abs_thresholds": dict(self.viol_abs_thresholds),
            "initialized": bool(self.initialized),
        }

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        g = d.get("gnorm_ema", None)
        self.gnorm_ema = None if g is None else float(g)
        self.gnorm_ema_decay = float(d.get("gnorm_ema_decay", 0.99))
        self.viol_ema = {str(k): float(v) for k, v in dict(d.get("viol_ema", {})).items()}
        self.viol_ema_decay = float(d.get("viol_ema_decay", 0.99))
        self.viol_abs_thresholds = {
            str(k): float(v) for k, v in dict(d.get("viol_abs_thresholds", {})).items()
        }
        self.initialized = bool(d.get("initialized", False))

    def update_after_valid_baseline(
        self,
        gnorm: float,
        viol_metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Update refs once per valid logical baseline (not per trial)."""
        g = float(gnorm)
        if not math.isfinite(g):
            return
        if self.gnorm_ema is None or not self.initialized:
            self.gnorm_ema = g
        else:
            d = float(self.gnorm_ema_decay)
            self.gnorm_ema = d * float(self.gnorm_ema) + (1.0 - d) * g
        if viol_metrics:
            vd = float(self.viol_ema_decay)
            for k, v in viol_metrics.items():
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                if k not in self.viol_ema or not self.initialized:
                    self.viol_ema[k] = fv
                else:
                    self.viol_ema[k] = vd * self.viol_ema[k] + (1.0 - vd) * fv
        self.initialized = True


@dataclass
class SearchSeedChoice:
    epsilon: float
    expand: bool
    reason: str


def should_search(
    ctrl: IntervalControllerState,
    cfg: LineSearchConfig,
    j: int,
    *,
    extra_reasons: Sequence[str] = (),
) -> tuple[bool, list[str]]:
    """Pure scheduling: whether proposed update ``j = u+1`` should search."""
    reasons: list[str] = []
    if ctrl.last_search_update is None:
        reasons.append("never_searched")
    if ctrl.pending_search:
        reasons.append("pending_search")
    if ctrl.pending_trigger_reasons:
        reasons.extend(list(ctrl.pending_trigger_reasons))
    reasons.extend(list(extra_reasons))

    # Warmup: search each of the first ``warmup_updates`` successful updates.
    if int(ctrl.successful_update_count) < int(cfg.warmup_updates):
        reasons.append("warmup")

    if cfg.search_interval_mode == "fixed":
        K = int(cfg.initial_interval)
    else:
        K = int(ctrl.K)
    K = max(int(cfg.min_interval), min(int(cfg.max_interval), K))

    if ctrl.last_search_update is not None:
        if int(j) - int(ctrl.last_search_update) >= K:
            reasons.append("interval")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return bool(uniq), uniq


def check_early_triggers(
    *,
    cfg: LineSearchConfig,
    refs: TriggerRefs,
    gnorm: float,
    viol_metrics: Mapping[str, float] | None = None,
    objective_changed: bool = False,
    group_ratios_changed: bool = False,
    nonfinite_baseline: bool = False,
) -> list[str]:
    """Pure early-search trigger checks (before applying the current update).

    Compares CURRENT baseline signals to references captured before incorporating
    this batch (caller must not have updated ``refs`` yet for this baseline).
    """
    reasons: list[str] = []
    if objective_changed:
        reasons.append("objective_changed")
    if group_ratios_changed:
        reasons.append("group_ratios_changed")
    if nonfinite_baseline:
        reasons.append("nonfinite_baseline")
        return reasons

    if refs.initialized and refs.gnorm_ema is not None and math.isfinite(float(gnorm)):
        ref = max(float(refs.gnorm_ema), float(cfg.grad_norm_ref_floor))
        if float(gnorm) > float(cfg.grad_norm_ratio_trigger) * ref:
            reasons.append("grad_norm_spike")

    if viol_metrics and refs.initialized:
        for k, v in viol_metrics.items():
            fv = float(v)
            if not math.isfinite(fv):
                continue
            ref_v = refs.viol_ema.get(k)
            if ref_v is None:
                continue
            abs_thr = float(refs.viol_abs_thresholds.get(k, 0.0))
            if fv > float(cfg.violation_ratio_trigger) * float(ref_v) and (
                fv - float(ref_v)
            ) > abs_thr:
                reasons.append(f"violation_spike:{k}")
    return reasons


def choose_search_seed(
    ctrl: IntervalControllerState,
    cfg: LineSearchConfig,
    *,
    epsilon_max: float,
    epsilon_min: float,
    reasons: Sequence[str],
) -> SearchSeedChoice:
    """First trial ε: expansion, retained rate, or pending retry (no forced min)."""
    # Budget failure: next search starts from pending smaller retry_seed, no expansion.
    if ctrl.retry_seed is not None and (
        "selection_failed" in reasons or "pending_search" in reasons
    ):
        eps = _bound_search_epsilon(
            float(ctrl.retry_seed),
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
        )
        return SearchSeedChoice(epsilon=eps, expand=False, reason="retry_seed")

    # Discrete objective reset: optional ε reseed; always retain last selected
    # when reseed is off so the selector shrinks/expands from the current rate.
    if "objective_changed" in reasons:
        eps = _bound_search_epsilon(
            float(ctrl.last_selected_epsilon or cfg.epsilon_seed),
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
        )
        reason = (
            "objective_reseed"
            if cfg.reseed_epsilon_on_objective_change
            else "objective_changed"
        )
        return SearchSeedChoice(epsilon=eps, expand=False, reason=reason)

    # First executed search at this selector clock (plan §5: ``epsilon_seed`` /
    # retained ``last_selected_epsilon``, no expansion).  ``last_selected`` is
    # preset during init, so do not key off ``last_selected_epsilon is None``.
    if ctrl.last_search_update is None:
        eps = _bound_search_epsilon(
            float(ctrl.last_selected_epsilon or cfg.epsilon_seed),
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
        )
        return SearchSeedChoice(epsilon=eps, expand=False, reason="first_search")

    # Legacy edge: no retained rate on record.
    if ctrl.last_selected_epsilon is None:
        eps = _bound_search_epsilon(
            float(cfg.epsilon_seed),
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
        )
        return SearchSeedChoice(
            epsilon=eps, expand=False, reason="seed_initial"
        )

    # Early trigger / recovery: retained rate, no expansion.
    no_expand_tags = {
        "group_ratios_changed",
        "grad_norm_spike",
        "nonfinite_baseline",
        "selection_failed",
    }
    if any(r in no_expand_tags or r.startswith("violation_spike:") for r in reasons):
        eps = _bound_search_epsilon(
            float(ctrl.last_selected_epsilon),
            epsilon_min=epsilon_min,
            epsilon_max=epsilon_max,
        )
        return SearchSeedChoice(
            epsilon=eps, expand=False, reason="retained_no_expand",
        )

    # Normal later search (interval / warmup after the first): expand last selected.
    eps = _bound_search_epsilon(
        float(cfg.expansion_factor) * float(ctrl.last_selected_epsilon),
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
    )
    return SearchSeedChoice(epsilon=eps, expand=True, reason="expanded")

def update_interval_after_search(
    ctrl: IntervalControllerState,
    cfg: LineSearchConfig,
    *,
    epsilon_new: float,
    trial_count: int,
    j: int,
    trigger_reasons: Sequence[str] = (),
    epsilon_min: float | None = None,
) -> dict[str, Any]:
    """Update ``K`` / streak after a successful searched+executed update."""
    info: dict[str, Any] = {}
    prev = ctrl.last_selected_epsilon
    eps_new = float(epsilon_new)
    if epsilon_min is not None:
        eps_new = clamp_epsilon(eps_new, float(epsilon_min))
    ctrl.last_search_update = int(j)
    ctrl.searches_completed = int(ctrl.searches_completed) + 1
    ctrl.retry_seed = None
    ctrl.pending_search = False
    ctrl.pending_trigger_reasons = []

    ratio = None
    if prev is not None and prev > 0 and math.isfinite(eps_new):
        ratio = float(eps_new) / float(prev)
    info["rate_ratio"] = ratio

    demote_hard = False
    if ratio is not None and ratio <= float(cfg.substantial_rate_drop_ratio):
        demote_hard = True
        info["demote"] = "rate_drop"
    if int(trial_count) >= int(cfg.costly_search_trials):
        demote_hard = True
        info["demote"] = "costly_trials"
    # Early triggers latch K=1 before the search; do not re-demote a successful
    # post-trigger search solely because of those reasons.

    # During warmup keep K=1 and do not accumulate a promotion streak.
    in_warmup = int(j) <= int(cfg.warmup_updates)

    stable = False
    if (
        ratio is not None
        and max(ratio, 1.0 / ratio) <= float(cfg.stable_rate_ratio_bound)
        and int(trial_count) <= int(cfg.stable_max_trials)
    ):
        stable = True
    info["stable"] = stable

    if demote_hard or in_warmup:
        ctrl.K = int(cfg.min_interval)
        ctrl.stable_search_streak = 0
        info["K_action"] = "hold_or_reset_to_min"
    elif not ctrl.stability_eligible:
        # First search after init / objective reset: do not establish stability.
        ctrl.stable_search_streak = 0
        info["K_action"] = "first_after_reset"
        ctrl.stability_eligible = True
    elif stable:
        ctrl.stable_search_streak = int(ctrl.stable_search_streak) + 1
        if ctrl.stable_search_streak >= int(cfg.stable_searches_to_increase):
            ctrl.K = min(int(ctrl.K) + 1, int(cfg.max_interval))
            ctrl.stable_search_streak = 0
            info["K_action"] = "promote"
        else:
            info["K_action"] = "stable_streak"
    else:
        ctrl.K = max(int(cfg.min_interval), int(math.floor(ctrl.K / 2)))
        ctrl.stable_search_streak = 0
        info["K_action"] = "halve"

    if cfg.search_interval_mode == "fixed":
        ctrl.K = int(cfg.initial_interval)
        info["K_action"] = "fixed"

    ctrl.last_selected_epsilon = eps_new
    info["K"] = int(ctrl.K)
    info["trigger_reasons"] = list(trigger_reasons)
    return info


def on_selection_failure(
    ctrl: IntervalControllerState,
    accept: AcceptanceEMAState,
    *,
    last_tried_epsilon: float | None,
    fallback_shrink: float = 0.5,
    epsilon_min: float = 1e-10,
) -> None:
    """Armijo exhaustion bookkeeping: K=1, retry seed, schedule A/B reset.

    The train loop may still commit a parameter update at ``epsilon_min`` after
    this returns; invalid-probe / nonfinite paths skip that commit.
    """
    ctrl.K = 1
    ctrl.stable_search_streak = 0
    ctrl.pending_search = True
    if "selection_failed" not in ctrl.pending_trigger_reasons:
        ctrl.pending_trigger_reasons.append("selection_failed")
    if last_tried_epsilon is not None and math.isfinite(last_tried_epsilon):
        ctrl.retry_seed = retry_seed_after_failure(
            float(last_tried_epsilon),
            fallback_shrink=float(fallback_shrink),
            epsilon_min=float(epsilon_min),
        )
    # Do not erase history mid-search; schedule reset for the NEXT iteration.
    accept.pending_history_reset = True


def apply_pending_history_reset(accept: AcceptanceEMAState) -> bool:
    """Apply a scheduled history reset at the start of the next search setup."""
    if not accept.pending_history_reset:
        return False
    accept.reset()
    return True


def objective_reseed_epsilon(
    *,
    previous: float | None,
    cfg: LineSearchConfig,
    eta_ref: float,
    factor: float | None = None,
) -> float:
    """Reseed ε after a discrete objective change: ``mult * last_ε``, clamped.

    When there is no usable previous rate, fall back to ``eta_ref`` (not
    ``mult * eta_ref``).
    """
    emin, emax = resolve_runtime_bounds(cfg, float(eta_ref))
    ref = float(eta_ref)
    mult = float(cfg.objective_reseed_mult if factor is None else factor)
    if previous is None or not (float(previous) > 0 and math.isfinite(float(previous))):
        return min(emax, clamp_epsilon(ref, emin))
    reseed = mult * float(previous)
    return min(emax, clamp_epsilon(reseed, emin))


def reset_history_for_objective_change(
    accept: AcceptanceEMAState,
    ctrl: IntervalControllerState,
    *,
    cfg: LineSearchConfig,
    eta_ref: float,
) -> float:
    """Discrete objective / AL jump: clear A/B, K=1, force search; optional ε reseed."""
    accept.reset()
    ctrl.K = 1
    ctrl.stable_search_streak = 0
    ctrl.pending_search = True
    ctrl.stability_eligible = False
    ctrl.retry_seed = None
    floor = epsilon_floor(cfg, float(eta_ref))
    if cfg.reseed_epsilon_on_objective_change:
        reseed = objective_reseed_epsilon(
            previous=ctrl.last_selected_epsilon,
            cfg=cfg,
            eta_ref=float(eta_ref),
        )
        ctrl.last_selected_epsilon = reseed
    else:
        reseed = clamp_epsilon(
            float(ctrl.last_selected_epsilon or cfg.epsilon_seed or eta_ref),
            floor,
        )
        ctrl.last_selected_epsilon = reseed
    if "objective_changed" not in ctrl.pending_trigger_reasons:
        ctrl.pending_trigger_reasons.append("objective_changed")
    return reseed


def advance_successful_update(ctrl: IntervalControllerState, j: int) -> None:
    ctrl.successful_update_count = int(j)


def latch_triggers(ctrl: IntervalControllerState, reasons: Sequence[str]) -> None:
    for r in reasons:
        if r not in ctrl.pending_trigger_reasons:
            ctrl.pending_trigger_reasons.append(str(r))
    if reasons:
        ctrl.pending_search = True
        ctrl.K = 1
        ctrl.stable_search_streak = 0


# ------------------------------------------------------------------- logging / ckpt
def logging_dict_from_search(
    *,
    F0: float | None = None,
    Fprobe: float | None = None,
    Fexec: float | None = None,
    Factual: float | None = None,
    actual_decrease: float | None = None,
    probe_exec_discrepancy: float | None = None,
    epsilon: float | None = None,
    epsilon_seed_used: float | None = None,
    trial_count: int | None = None,
    a: float | None = None,
    P2: float | None = None,
    s_exec: float | None = None,
    s_momentum: float | None = None,
    s_decay: float | None = None,
    decrease: float | None = None,
    A_prev: float | None = None,
    A_cand: float | None = None,
    B_prev: float | None = None,
    B_cand: float | None = None,
    beta_eff: float | None = None,
    gap: int | None = None,
    R: float | None = None,
    kappa: float | None = None,
    K: int | None = None,
    searched: bool | None = None,
    accepted: bool | None = None,
    trigger_reasons: Sequence[str] | None = None,
    stable_streak: int | None = None,
    failure_reason: str | None = None,
    grad_balance: str = "",
    ls_u_fresh_norm: float | None = None,
    ls_u_mom_norm: float | None = None,
    ls_u_decay_norm: float | None = None,
    ls_u_total_norm: float | None = None,
    ls_u_fresh_frac: float | None = None,
    ls_u_mom_frac: float | None = None,
    ls_u_decay_frac: float | None = None,
    ls_u_mom_over_fresh: float | None = None,
    ls_u_fresh_norm_frac: float | None = None,
    ls_u_mom_norm_frac: float | None = None,
    ls_u_decay_norm_frac: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Flat logging dict; omit Nones for sparse ordinary-update logs."""
    d: dict[str, Any] = {"grad_balance": grad_balance, "line_search": 1}
    mapping = {
        "ls_F0": F0,
        "ls_Fprobe": Fprobe,
        "ls_Fexec": Fexec,
        "ls_Factual": Factual,
        "ls_actual_decrease": actual_decrease,
        "ls_probe_exec_discrepancy": probe_exec_discrepancy,
        "ls_epsilon": epsilon,
        "ls_epsilon_seed": epsilon_seed_used,
        "ls_trial_count": trial_count,
        "ls_a": a,
        "ls_P2": P2,
        "ls_s_exec": s_exec,
        "ls_s_momentum": s_momentum,
        "ls_s_decay": s_decay,
        "ls_decrease": decrease,
        "ls_A_prev": A_prev,
        "ls_A_cand": A_cand,
        "ls_B_prev": B_prev,
        "ls_B_cand": B_cand,
        "ls_beta_eff": beta_eff,
        "ls_gap": gap,
        "ls_R": R,
        "ls_kappa": kappa,
        "ls_K": K,
        "ls_searched": None if searched is None else int(bool(searched)),
        "ls_accepted": None if accepted is None else int(bool(accepted)),
        "ls_stable_streak": stable_streak,
        "ls_failure": failure_reason,
        "ls_u_fresh_norm": ls_u_fresh_norm,
        "ls_u_mom_norm": ls_u_mom_norm,
        "ls_u_decay_norm": ls_u_decay_norm,
        "ls_u_total_norm": ls_u_total_norm,
        "ls_u_fresh_frac": ls_u_fresh_frac,
        "ls_u_mom_frac": ls_u_mom_frac,
        "ls_u_decay_frac": ls_u_decay_frac,
        "ls_u_mom_over_fresh": ls_u_mom_over_fresh,
        "ls_u_fresh_norm_frac": ls_u_fresh_norm_frac,
        "ls_u_mom_norm_frac": ls_u_mom_norm_frac,
        "ls_u_decay_norm_frac": ls_u_decay_norm_frac,
    }
    for k, v in mapping.items():
        if v is not None:
            d[k] = v
    for k, v in extra.items():
        if v is not None:
            d[k] = v
    if trigger_reasons is not None:
        d["ls_triggers"] = ",".join(trigger_reasons)
    return d


def line_search_state_dict(
    cfg: LineSearchConfig,
    accept: AcceptanceEMAState,
    ctrl: IntervalControllerState,
    refs: TriggerRefs | None = None,
    *,
    fixed_group_ratios: Sequence[float] | None = None,
    eta_ref: float | None = None,
    seed_provenance: str | None = None,
    objective_direction_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": _STATE_VERSION,
        "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
        "acceptance": accept.state_dict(),
        "controller": ctrl.state_dict(),
        "trigger_refs": None if refs is None else refs.state_dict(),
        "fixed_group_ratios": None if fixed_group_ratios is None else list(map(float, fixed_group_ratios)),
        "eta_ref": eta_ref,
        "seed_provenance": seed_provenance,
        "objective_direction_fingerprint": (
            None
            if objective_direction_fingerprint is None
            else dict(objective_direction_fingerprint)
        ),
        "grad_balance": "",
        "direction_convention": (
            "p_probe=-D*q;a=-dot(g_objective,p_probe);"
            "s_exec=dot(g_objective,p_exec)"
        ),
        "controller_identity": "adaptive_K_rate_stability",
    }


def load_line_search_state_dict(
    d: Mapping[str, Any],
    *,
    cfg: LineSearchConfig | None = None,
    accept: AcceptanceEMAState | None = None,
    ctrl: IntervalControllerState | None = None,
    refs: TriggerRefs | None = None,
) -> tuple[LineSearchConfig, AcceptanceEMAState, IntervalControllerState, TriggerRefs]:
    """Restore selector state; forces ``grad_balance=\"\"``."""
    ver = int(d.get("version", 0))
    if ver > _STATE_VERSION:
        raise ValueError(f"unsupported line-search state version {ver}")

    cfg_out = cfg if cfg is not None else LineSearchConfig()
    raw_cfg = d.get("config")
    if isinstance(raw_cfg, Mapping):
        known = {f.name for f in fields(LineSearchConfig)}
        kwargs = {k: v for k, v in raw_cfg.items() if k in known}
        cfg_out = LineSearchConfig(**{**asdict(cfg_out), **kwargs})
    cfg_out = cfg_out.with_forced_grad_balance()

    accept_out = accept if accept is not None else AcceptanceEMAState()
    if isinstance(d.get("acceptance"), Mapping):
        accept_out.load_state_dict(d["acceptance"])  # type: ignore[arg-type]

    ctrl_out = ctrl if ctrl is not None else IntervalControllerState()
    if isinstance(d.get("controller"), Mapping):
        ctrl_out.load_state_dict(d["controller"])  # type: ignore[arg-type]

    refs_out = refs if refs is not None else TriggerRefs()
    if isinstance(d.get("trigger_refs"), Mapping):
        refs_out.load_state_dict(d["trigger_refs"])  # type: ignore[arg-type]

    return cfg_out, accept_out, ctrl_out, refs_out
