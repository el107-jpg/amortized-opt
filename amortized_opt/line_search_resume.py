"""Checkpoint resume LR migration + logging helpers for SaLSa line search.

Train-loop ownership stays in ``train_phase2``; this module is the reusable
substrate for plan §7 (legacy resume seed), §11 (checkpoint / log fields),
and §14 (last applied ``param_groups[].lr`` provenance until LS owns rates).

Integrator call (merge into ``save_phase2_checkpoint(..., extra=...)``)::

    from amortized_opt.line_search_resume import (
        pack_line_search_checkpoint_extra,
        unpack_line_search_checkpoint_extra,
        resolve_epsilon_seed,
        logging_dict_for_seed_resolution,
    )

    # After optimizer.load_state_dict (and optional selector unpack):
    seed, prov = resolve_epsilon_seed(
        line_search_cfg=cfg.line_search_cfg,
        cfg_lr=cfg.lr,
        optimizer=opt,
        has_selector_ckpt="line_search" in (ck.get("extra") or {}),
        explicit_override=cli_epsilon_seed,  # or None
    )
    # Sync seed into cfg.line_search_cfg.epsilon_seed / eta_ref / bounds.

    extra = {
        **(other_extra or {}),
        **pack_line_search_checkpoint_extra(
            ls_cfg, accept, ctrl, refs,
            fixed_group_ratios=ratios,
            eta_ref=seed,
            seed_provenance=prov,
            optimizer=opt,
        ),
    }
    save_phase2_checkpoint(..., extra=extra)
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .line_search import (
    AcceptanceEMAState,
    IntervalControllerState,
    LineSearchConfig,
    TriggerRefs,
    line_search_state_dict,
    load_line_search_state_dict,
    logging_dict_from_search,
)

__all__ = [
    "resolve_epsilon_seed",
    "read_last_effective_lrs",
    "reference_lr_from_lrs",
    "optimizer_has_adam_state",
    "pack_line_search_checkpoint_extra",
    "unpack_line_search_checkpoint_extra",
    "logging_dict_for_seed_resolution",
    "has_line_search_selector_extra",
]


def _positive_finite(x: float, label: str) -> float:
    fx = float(x)
    if not (math.isfinite(fx) and fx > 0.0):
        raise ValueError(
            f"line search: {label} must be positive and finite, got {x!r}"
        )
    return fx


def optimizer_has_adam_state(optimizer: Any) -> bool:
    """True after ``load_state_dict`` / at least one Adam step (moments present)."""
    state = getattr(optimizer, "state", None)
    if state is None:
        return False
    try:
        return len(state) > 0
    except TypeError:
        return False


def read_last_effective_lrs(optimizer: Any) -> list[float]:
    """Per-group ``lr`` values as currently stored on the optimizer."""
    if optimizer is None:
        raise ValueError("line search: optimizer is required to read last effective LRs")
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        raise ValueError(
            "line search legacy resume: optimizer has no param_groups; "
            "cannot recover last applied LR (plan §7). Pass explicit_override."
        )
    out: list[float] = []
    for i, g in enumerate(groups):
        if "lr" not in g:
            raise ValueError(
                f"line search legacy resume: param_groups[{i}] missing 'lr'; "
                "pass explicit_override (plan §7)."
            )
        out.append(float(g["lr"]))
    return out


def reference_lr_from_lrs(lrs: Sequence[float]) -> float:
    """Reference η from saved group rates: max of positive finite entries.

    Zero / frozen groups are ignored for the reference (ratios stay ``lr/η``).
    Distinct positive rates are allowed; ``max`` is the η_ref convention so
    ratios remain ≤ 1 for the hottest group.
    """
    positive = [float(x) for x in lrs if math.isfinite(float(x)) and float(x) > 0.0]
    if not positive:
        raise ValueError(
            "line search legacy resume: no positive finite optimizer "
            "param_groups[].lr after restore; refuse to guess cfg.lr "
            "(plan §7). Pass explicit_override / --line-search-epsilon-seed."
        )
    return float(max(positive))


def resolve_epsilon_seed(
    *,
    line_search_cfg: LineSearchConfig,
    cfg_lr: float,
    optimizer: Any | None,
    has_selector_ckpt: bool,
    explicit_override: float | None,
) -> tuple[float, str]:
    """Resolve the initial / migration ε seed and a stable provenance string.

    Precedence (plan §7):

    1. ``explicit_override`` (CLI ``--line-search-epsilon-seed``, etc.)
    2. Selector checkpoint present → use ``line_search_cfg.epsilon_seed``
       (integrator should sync from :func:`unpack_line_search_checkpoint_extra`)
    3. ``line_search_cfg.legacy_resume_seed`` when set
    4. Legacy resume: positive ``optimizer.param_groups[].lr`` when Adam state
       is present (moments loaded from a non-selector checkpoint)
    5. Fresh run: ``cfg_lr``, falling back to ``line_search_cfg.epsilon_seed``

    Raises ``ValueError`` with a clear message when a legacy resume cannot
    identify a positive finite last effective rate.
    """
    if explicit_override is not None:
        seed = _positive_finite(explicit_override, "explicit_override")
        return seed, "explicit_override"

    if has_selector_ckpt:
        seed = _positive_finite(
            line_search_cfg.epsilon_seed, "selector checkpoint epsilon_seed"
        )
        return seed, "selector_checkpoint"

    legacy_cfg = getattr(line_search_cfg, "legacy_resume_seed", None)
    if legacy_cfg is not None:
        seed = _positive_finite(legacy_cfg, "line_search_cfg.legacy_resume_seed")
        return seed, "legacy_resume_seed_config"

    if optimizer is not None and optimizer_has_adam_state(optimizer):
        try:
            lrs = read_last_effective_lrs(optimizer)
            seed = reference_lr_from_lrs(lrs)
        except ValueError as exc:
            raise ValueError(
                "line search enabled on a legacy checkpoint but last effective "
                "LR is missing or non-positive; pass explicit_override / "
                "--line-search-epsilon-seed (plan §7)."
            ) from exc
        return seed, "legacy_optimizer_lr"

    # Fresh run (no Adam moments yet): configured Phase2 reference LR.
    try:
        seed = _positive_finite(cfg_lr, "cfg.lr")
        return seed, "fresh_cfg_lr"
    except ValueError:
        seed = _positive_finite(
            line_search_cfg.epsilon_seed, "line_search_cfg.epsilon_seed"
        )
        return seed, "fresh_epsilon_seed"


def pack_line_search_checkpoint_extra(
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
    """Build an ``extra`` fragment with versioned selector state + LR fields.

    Merge into ``save_phase2_checkpoint(..., extra={**frag, ...})``. Always
    records ``grad_balance=""`` (plan §1 / §11).
    """
    if last_effective_lrs is None and optimizer is not None:
        last_effective_lrs = read_last_effective_lrs(optimizer)

    state = line_search_state_dict(
        cfg,
        accept,
        ctrl,
        refs,
        fixed_group_ratios=fixed_group_ratios,
        eta_ref=eta_ref,
        seed_provenance=seed_provenance,
        objective_direction_fingerprint=objective_direction_fingerprint,
    )
    state["grad_balance"] = ""
    state["last_effective_lrs"] = (
        None if last_effective_lrs is None else [float(x) for x in last_effective_lrs]
    )
    if seed_provenance is not None:
        state["seed_provenance"] = str(seed_provenance)
    return {"line_search": state}


def has_line_search_selector_extra(extra: Mapping[str, Any] | None) -> bool:
    if not isinstance(extra, Mapping):
        return False
    blob = extra.get("line_search")
    return isinstance(blob, Mapping) and bool(blob)


def unpack_line_search_checkpoint_extra(
    extra: Mapping[str, Any] | None,
    *,
    cfg: LineSearchConfig | None = None,
    accept: AcceptanceEMAState | None = None,
    ctrl: IntervalControllerState | None = None,
    refs: TriggerRefs | None = None,
) -> tuple[
    LineSearchConfig,
    AcceptanceEMAState,
    IntervalControllerState,
    TriggerRefs,
    dict[str, Any],
]:
    """Restore selector objects from checkpoint ``extra``.

    Returns ``(cfg, accept, ctrl, refs, meta)`` where ``meta`` includes
    ``last_effective_lrs``, ``seed_provenance``, ``eta_ref``,
    ``fixed_group_ratios``, and ``grad_balance`` (forced ``""``).

    Raises ``KeyError`` if ``extra["line_search"]`` is missing.
    """
    if not isinstance(extra, Mapping) or "line_search" not in extra:
        raise KeyError(
            "checkpoint extra has no 'line_search' selector state; "
            "treat as legacy resume via resolve_epsilon_seed(..., "
            "has_selector_ckpt=False)"
        )
    blob = extra["line_search"]
    if not isinstance(blob, Mapping):
        raise TypeError(
            f"extra['line_search'] must be a mapping, got {type(blob).__name__}"
        )

    cfg_out, accept_out, ctrl_out, refs_out = load_line_search_state_dict(
        blob, cfg=cfg, accept=accept, ctrl=ctrl, refs=refs
    )

    raw_lrs = blob.get("last_effective_lrs")
    last_lrs: list[float] | None
    if raw_lrs is None:
        last_lrs = None
    else:
        last_lrs = [float(x) for x in raw_lrs]

    ratios = blob.get("fixed_group_ratios")
    meta: dict[str, Any] = {
        "last_effective_lrs": last_lrs,
        "seed_provenance": blob.get("seed_provenance"),
        "eta_ref": None if blob.get("eta_ref") is None else float(blob["eta_ref"]),
        "fixed_group_ratios": None if ratios is None else [float(x) for x in ratios],
        "grad_balance": "",
        "version": int(blob.get("version", 0)),
        "objective_direction_fingerprint": blob.get(
            "objective_direction_fingerprint"
        ),
    }
    return cfg_out, accept_out, ctrl_out, refs_out, meta


def logging_dict_for_seed_resolution(
    seed: float,
    provenance: str,
    *,
    last_effective_lrs: Sequence[float] | None = None,
    eta_ref: float | None = None,
    searched: bool | None = None,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Metrics fragment for seed migration / resume (§11).

    Always includes ``grad_balance=""`` and ``line_search=1``. Extra kwargs
    are forwarded to :func:`logging_dict_from_search`.
    """
    d = logging_dict_from_search(
        epsilon_seed_used=float(seed),
        grad_balance="",
        searched=searched,
        **search_kwargs,
    )
    d["ls_seed_provenance"] = str(provenance)
    if eta_ref is not None:
        d["ls_eta_ref"] = float(eta_ref)
    if last_effective_lrs is not None:
        d["ls_last_effective_lrs"] = [float(x) for x in last_effective_lrs]
    return d
