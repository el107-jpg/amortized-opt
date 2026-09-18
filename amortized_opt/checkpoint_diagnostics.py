"""Pure helpers for comparing model and Adam checkpoint state.

The phase-2 loss-spike CLI lives in ``scripts/diagnose_loss_spike.py``.  This
module deliberately contains only tensor bookkeeping so its numerics can be
unit-tested without constructing the G1 model, FK, RNEA, or a scenario pool.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


def parameter_module(name: str, depth: int = 2) -> str:
    """Return the leading module path used for aggregate diagnostics."""
    parts = str(name).split(".")
    return ".".join(parts[: max(1, int(depth))])


class _NormAccumulator:
    def __init__(self) -> None:
        self.tensors = 0
        self.numel = 0
        self.nonfinite = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.max_abs = 0.0

    def add(self, value: torch.Tensor) -> None:
        x = value.detach().double().reshape(-1)
        self.tensors += 1
        self.numel += int(x.numel())
        finite = torch.isfinite(x)
        self.nonfinite += int((~finite).sum().item())
        if not bool(finite.any()):
            return
        x = x[finite]
        self.sum_sq += float(torch.dot(x, x).item())
        self.sum_abs += float(x.abs().sum().item())
        self.max_abs = max(self.max_abs, float(x.abs().max().item()))

    def report(self) -> dict[str, float | int]:
        n = max(self.numel - self.nonfinite, 1)
        return {
            "tensors": self.tensors,
            "numel": self.numel,
            "nonfinite": self.nonfinite,
            "l2": math.sqrt(max(self.sum_sq, 0.0)),
            "rms": math.sqrt(max(self.sum_sq, 0.0) / n),
            "mean_abs": self.sum_abs / n,
            "max_abs": self.max_abs,
        }


class _DeltaAccumulator:
    def __init__(self) -> None:
        self.tensors = 0
        self.numel = 0
        self.nonfinite = 0
        self.before_sq = 0.0
        self.after_sq = 0.0
        self.delta_sq = 0.0
        self.delta_abs = 0.0
        self.delta_max = 0.0
        self.before_after_dot = 0.0

    def add(self, before: torch.Tensor, after: torch.Tensor) -> None:
        a = before.detach().double().reshape(-1)
        b = after.detach().double().reshape(-1)
        self.tensors += 1
        self.numel += int(a.numel())
        finite = torch.isfinite(a) & torch.isfinite(b)
        self.nonfinite += int((~finite).sum().item())
        if not bool(finite.any()):
            return
        a, b = a[finite], b[finite]
        d = b - a
        self.before_sq += float(torch.dot(a, a).item())
        self.after_sq += float(torch.dot(b, b).item())
        self.delta_sq += float(torch.dot(d, d).item())
        self.delta_abs += float(d.abs().sum().item())
        self.delta_max = max(self.delta_max, float(d.abs().max().item()))
        self.before_after_dot += float(torch.dot(a, b).item())

    def report(self) -> dict[str, float | int | None]:
        finite_n = max(self.numel - self.nonfinite, 1)
        before_l2 = math.sqrt(max(self.before_sq, 0.0))
        after_l2 = math.sqrt(max(self.after_sq, 0.0))
        delta_l2 = math.sqrt(max(self.delta_sq, 0.0))
        denom = before_l2 * after_l2
        return {
            "tensors": self.tensors,
            "numel": self.numel,
            "nonfinite": self.nonfinite,
            "before_l2": before_l2,
            "after_l2": after_l2,
            "delta_l2": delta_l2,
            "delta_rms": delta_l2 / math.sqrt(finite_n),
            "delta_mean_abs": self.delta_abs / finite_n,
            "delta_max_abs": self.delta_max,
            "relative_delta_l2": delta_l2 / max(before_l2, 1e-30),
            "before_after_cosine": (
                self.before_after_dot / denom if denom > 0.0 else None
            ),
        }


def tensor_mapping_report(
    tensors: Mapping[str, torch.Tensor], *, module_depth: int = 2
) -> dict:
    """Norm summaries overall and by leading parameter module."""
    overall = _NormAccumulator()
    modules: dict[str, _NormAccumulator] = {}
    for name, value in tensors.items():
        if not torch.is_tensor(value):
            continue
        overall.add(value)
        modules.setdefault(
            parameter_module(name, module_depth), _NormAccumulator()
        ).add(value)
    return {
        "overall": overall.report(),
        "by_module": {k: v.report() for k, v in sorted(modules.items())},
    }


def tensor_mapping_delta_report(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    *,
    names: Sequence[str] | None = None,
    module_depth: int = 2,
) -> dict:
    """Delta summaries over common, equal-shaped floating tensors."""
    selected = set(names) if names is not None else set(before) | set(after)
    only_before: list[str] = []
    only_after: list[str] = []
    shape_mismatch: list[dict] = []
    skipped_non_float: list[str] = []
    overall = _DeltaAccumulator()
    modules: dict[str, _DeltaAccumulator] = {}
    for name in sorted(selected):
        if name not in before:
            only_after.append(name)
            continue
        if name not in after:
            only_before.append(name)
            continue
        a, b = before[name], after[name]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatch.append(
                {"name": name, "before": list(a.shape), "after": list(b.shape)}
            )
            continue
        if not (torch.is_floating_point(a) and torch.is_floating_point(b)):
            skipped_non_float.append(name)
            continue
        overall.add(a, b)
        modules.setdefault(
            parameter_module(name, module_depth), _DeltaAccumulator()
        ).add(a, b)
    return {
        "overall": overall.report(),
        "by_module": {k: v.report() for k, v in sorted(modules.items())},
        "only_before": only_before,
        "only_after": only_after,
        "shape_mismatch": shape_mismatch,
        "skipped_non_float": skipped_non_float,
    }


class _AlignmentAccumulator:
    def __init__(self) -> None:
        self.tensors = 0
        self.numel = 0
        self.nonfinite = 0
        self.observed_sq = 0.0
        self.proposed_sq = 0.0
        self.dot = 0.0

    def add(self, observed: torch.Tensor, proposed: torch.Tensor) -> None:
        a = observed.detach().double().reshape(-1)
        b = proposed.detach().double().reshape(-1)
        self.tensors += 1
        self.numel += int(a.numel())
        finite = torch.isfinite(a) & torch.isfinite(b)
        self.nonfinite += int((~finite).sum().item())
        if not bool(finite.any()):
            return
        a, b = a[finite], b[finite]
        self.observed_sq += float(torch.dot(a, a).item())
        self.proposed_sq += float(torch.dot(b, b).item())
        self.dot += float(torch.dot(a, b).item())

    def report(self) -> dict[str, float | int | None]:
        obs = math.sqrt(max(self.observed_sq, 0.0))
        prop = math.sqrt(max(self.proposed_sq, 0.0))
        denom = obs * prop
        projection = self.dot / self.proposed_sq if self.proposed_sq > 0.0 else None
        residual_sq = (
            self.observed_sq
            - self.dot * self.dot / self.proposed_sq
            if self.proposed_sq > 0.0
            else self.observed_sq
        )
        return {
            "tensors": self.tensors,
            "numel": self.numel,
            "nonfinite": self.nonfinite,
            "observed_delta_l2": obs,
            "proposed_l2": prop,
            "cosine": self.dot / denom if denom > 0.0 else None,
            "observed_to_proposed_l2": obs / prop if prop > 0.0 else None,
            "best_fit_scale": projection,
            "orthogonal_residual_l2": math.sqrt(max(residual_sq, 0.0)),
        }


def direction_alignment_report(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    proposed_update: Mapping[str, torch.Tensor],
    *,
    module_depth: int = 2,
) -> dict:
    """Compare an observed parameter delta to a proposed optimizer update."""
    overall = _AlignmentAccumulator()
    modules: dict[str, _AlignmentAccumulator] = {}
    missing: list[str] = []
    shape_mismatch: list[str] = []
    for name, proposed in proposed_update.items():
        if name not in before or name not in after:
            missing.append(name)
            continue
        a, b = before[name], after[name]
        if tuple(a.shape) != tuple(b.shape) or tuple(a.shape) != tuple(proposed.shape):
            shape_mismatch.append(name)
            continue
        observed = b - a
        overall.add(observed, proposed)
        modules.setdefault(
            parameter_module(name, module_depth), _AlignmentAccumulator()
        ).add(observed, proposed)
    return {
        "overall": overall.report(),
        "by_module": {k: v.report() for k, v in sorted(modules.items())},
        "missing_parameters": missing,
        "shape_mismatch": shape_mismatch,
    }


def _state_entry(state: Mapping, param_id):
    if param_id in state:
        return state[param_id]
    key = str(param_id)
    return state.get(key)


def map_optimizer_state_to_names(
    optimizer_state: Mapping | None,
    parameter_order: Sequence[str],
    parameters: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Mapping], dict[str, Mapping], list[str]]:
    """Map serialized optimizer integer ids to phase-2 parameter names.

    ``build_optimizer`` uses two groups: matrix-like decay parameters followed
    by bias/norm/embedding no-decay parameters.  A one-group named-order
    fallback supports small synthetic checkpoints and older tools.
    """
    if not optimizer_state:
        return {}, {}, ["optimizer state is absent"]
    groups = list(optimizer_state.get("param_groups") or [])
    states = optimizer_state.get("state") or {}
    names = [n for n in parameter_order if n in parameters]
    decay = [n for n in names if parameters[n].ndim >= 2 and "emb" not in n]
    decay_names = set(decay)
    no_decay = [n for n in names if n not in decay_names]
    expected: list[list[str]] | None = None
    if len(groups) == 2 and [len(g.get("params", ())) for g in groups] == [
        len(decay),
        len(no_decay),
    ]:
        expected = [decay, no_decay]
    elif len(groups) == 1 and len(groups[0].get("params", ())) == len(names):
        expected = [names]
    elif sum(len(g.get("params", ())) for g in groups) == len(names):
        # Generic fallback.  It cannot prove names across unusual groupings,
        # but preserves serialized group order and reports that assumption.
        expected = []
        pos = 0
        for group in groups:
            count = len(group.get("params", ()))
            expected.append(names[pos : pos + count])
            pos += count
    else:
        return {}, {}, [
            "cannot map optimizer ids: group sizes "
            f"{[len(g.get('params', ())) for g in groups]} do not match "
            f"{len(names)} model parameters"
        ]

    mapped: dict[str, Mapping] = {}
    hyper: dict[str, Mapping] = {}
    warnings: list[str] = []
    if len(groups) not in (1, 2):
        warnings.append("optimizer names mapped by flattened group order")
    for group, group_names in zip(groups, expected):
        ids = list(group.get("params", ()))
        if len(ids) != len(group_names):
            warnings.append("optimizer group/name length mismatch")
            continue
        for param_id, name in zip(ids, group_names):
            entry = _state_entry(states, param_id)
            if entry is not None:
                mapped[name] = entry
            hyper[name] = group
    return mapped, hyper, warnings


@dataclass
class AdamAnalysis:
    """JSON report plus tensors used for adjacent-checkpoint comparisons."""

    report: dict
    exp_avg: dict[str, torch.Tensor]
    exp_avg_sq: dict[str, torch.Tensor]
    direction: dict[str, torch.Tensor]
    proposed_update: dict[str, torch.Tensor]


def analyze_adam_state(
    optimizer_state: Mapping | None,
    parameter_order: Sequence[str],
    parameters: Mapping[str, torch.Tensor],
    *,
    module_depth: int = 2,
) -> AdamAnalysis:
    """Summarize bias-corrected Adam moments and one saved-state update."""
    mapped, hypers, warnings = map_optimizer_state_to_names(
        optimizer_state, parameter_order, parameters
    )
    exp_avg: dict[str, torch.Tensor] = {}
    exp_avg_sq: dict[str, torch.Tensor] = {}
    directions: dict[str, torch.Tensor] = {}
    updates: dict[str, torch.Tensor] = {}
    steps: list[float] = []
    negative_v = zero_v = v_numel = 0
    malformed: list[str] = []

    for name, state in mapped.items():
        m, v = state.get("exp_avg"), state.get("exp_avg_sq")
        if not (torch.is_tensor(m) and torch.is_tensor(v)):
            malformed.append(name)
            continue
        if tuple(m.shape) != tuple(parameters[name].shape) or tuple(v.shape) != tuple(
            parameters[name].shape
        ):
            malformed.append(name)
            continue
        group = hypers[name]
        betas = group.get("betas", (0.9, 0.999))
        beta1, beta2 = float(betas[0]), float(betas[1])
        step_raw = state.get("step", 0)
        step = float(step_raw.item()) if torch.is_tensor(step_raw) else float(step_raw)
        steps.append(step)
        step_for_bias = max(step, 1.0)
        bc1 = max(1.0 - beta1**step_for_bias, 1e-30)
        bc2 = max(1.0 - beta2**step_for_bias, 1e-30)
        m_f = m.detach().float()
        v_f = v.detach().float()
        v_used = state.get("max_exp_avg_sq") if group.get("amsgrad", False) else v_f
        if not torch.is_tensor(v_used):
            v_used = v_f
        v_used = v_used.detach().float()
        eps = float(group.get("eps", 1e-8))
        m_hat = m_f / bc1
        v_hat = v_used / bc2
        direction = m_hat / (v_hat.clamp_min(0.0).sqrt() + eps)
        lr = float(group.get("lr", 0.0))
        weight_decay = float(group.get("weight_decay", 0.0))
        update = -lr * (
            direction + weight_decay * parameters[name].detach().float()
        )
        exp_avg[name] = m_f
        exp_avg_sq[name] = v_f
        directions[name] = direction
        updates[name] = update
        negative_v += int((v_f < 0).sum().item())
        zero_v += int((v_f == 0).sum().item())
        v_numel += int(v_f.numel())

    report = {
        "available": bool(directions),
        "model_parameter_tensors": len(parameter_order),
        "mapped_state_tensors": len(mapped),
        "adam_state_tensors": len(directions),
        "coverage": len(directions) / max(len(parameter_order), 1),
        "warnings": warnings,
        "malformed_or_missing_moments": malformed,
        "step": {
            "min": min(steps) if steps else None,
            "mean": sum(steps) / len(steps) if steps else None,
            "max": max(steps) if steps else None,
        },
        "second_moment": {
            "negative_count": negative_v,
            "zero_count": zero_v,
            "numel": v_numel,
            "zero_fraction": zero_v / max(v_numel, 1),
        },
        "exp_avg": tensor_mapping_report(exp_avg, module_depth=module_depth),
        "exp_avg_sq": tensor_mapping_report(exp_avg_sq, module_depth=module_depth),
        "effective_direction": tensor_mapping_report(
            directions, module_depth=module_depth
        ),
        "proposed_update": tensor_mapping_report(updates, module_depth=module_depth),
    }
    return AdamAnalysis(report, exp_avg, exp_avg_sq, directions, updates)


def adjacent_adam_report(
    before_parameters: Mapping[str, torch.Tensor],
    after_parameters: Mapping[str, torch.Tensor],
    before: AdamAnalysis,
    after: AdamAnalysis,
    *,
    module_depth: int = 2,
) -> dict:
    """Moment jumps and alignment with the observed checkpoint delta."""
    return {
        "exp_avg_delta": tensor_mapping_delta_report(
            before.exp_avg, after.exp_avg, module_depth=module_depth
        ),
        "exp_avg_sq_delta": tensor_mapping_delta_report(
            before.exp_avg_sq, after.exp_avg_sq, module_depth=module_depth
        ),
        "before_effective_update_alignment": direction_alignment_report(
            before_parameters,
            after_parameters,
            before.proposed_update,
            module_depth=module_depth,
        ),
        "after_effective_update_alignment": direction_alignment_report(
            before_parameters,
            after_parameters,
            after.proposed_update,
            module_depth=module_depth,
        ),
    }
