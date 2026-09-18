"""Diagnostic attribution of per-row torque gradients as vectors.

This module is intentionally disconnected from the training loop.  It measures
the gradient of the batch-mean torque term exactly as training scales it, then
uses fresh VJPs to expose cancellation between row gradients without storing a
gradient vector for every row.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch


GradTuple = tuple[torch.Tensor | None, ...]


def _detached_grads(
    scalar: torch.Tensor,
    params: list[torch.Tensor],
) -> GradTuple:
    if not scalar.requires_grad:
        return tuple(None for _ in params)
    grads = torch.autograd.grad(
        scalar,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    return tuple(
        None if grad is None else grad.detach().to(dtype=torch.float32)
        for grad in grads
    )


def _dot(left: GradTuple, right: GradTuple) -> float:
    value = None
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        term = (lhs * rhs).sum()
        value = term if value is None else value + term
    return 0.0 if value is None else float(value.item())


def _norm(grads: GradTuple) -> float:
    return math.sqrt(max(_dot(grads, grads), 0.0))


def _norm_tensor(grads: GradTuple, device: torch.device) -> torch.Tensor:
    value = None
    for grad in grads:
        if grad is None:
            continue
        term = grad.square().sum()
        value = term if value is None else value + term
    if value is None:
        return torch.zeros((), device=device, dtype=torch.float32)
    return value.clamp_min(0.0).sqrt()


def _linear_combination(
    left: GradTuple,
    left_scale: float,
    right: GradTuple,
    right_scale: float,
) -> GradTuple:
    out: list[torch.Tensor | None] = []
    for lhs, rhs in zip(left, right):
        if lhs is None and rhs is None:
            out.append(None)
        elif lhs is None:
            out.append(rhs * right_scale)
        elif rhs is None:
            out.append(lhs * left_scale)
        else:
            out.append(lhs * left_scale + rhs * right_scale)
    return tuple(out)


def _vector_metrics(group: GradTuple, total: GradTuple) -> dict[str, float]:
    group_norm = _norm(group)
    total_norm = _norm(total)
    dot = _dot(group, total)
    denom = group_norm * total_norm
    total_sq = total_norm * total_norm
    return {
        "gradient_norm": group_norm,
        "vector_sum_gradient_norm": group_norm,
        "cosine_with_total": dot / denom if denom > 0.0 else 0.0,
        "projection_onto_total": dot / total_norm if total_norm > 0.0 else 0.0,
        "signed_contribution_fraction": (
            dot / total_sq if total_sq > 0.0 else 0.0
        ),
    }


def _to_cpu_vector(
    name: str,
    values: Any,
    batch_size: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    out = torch.as_tensor(values).detach().to(device="cpu", dtype=dtype).reshape(-1)
    if out.numel() != batch_size:
        raise ValueError(
            f"{name!r} has {out.numel()} rows, expected batch size {batch_size}"
        )
    if not torch.isfinite(out).all():
        raise ValueError(f"{name!r} contains non-finite values")
    return out


def _rank_quantile_indices(values: torch.Tensor, n_bins: int) -> list[torch.Tensor]:
    """Equal-count rank quantiles; stable ordering makes tied labels reproducible."""
    n = int(values.numel())
    order = torch.argsort(values, stable=True)
    return [
        chunk
        for chunk in torch.tensor_split(order, min(max(int(n_bins), 1), n))
        if chunk.numel() > 0
    ]


def _summary(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"min": 0.0, "p50": 0.0, "mean": 0.0, "max": 0.0}
    vals = values.to(dtype=torch.float64)
    return {
        "min": float(vals.min()),
        "p50": float(vals.quantile(0.5)),
        "mean": float(vals.mean()),
        "max": float(vals.max()),
    }


def _sampling_group_summary(
    sampling: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, dict[str, float]]:
    return {
        name: _summary(values.index_select(0, indices))
        for name, values in sampling.items()
    }


def measure_batch_torque_gradient_attribution(
    forward_rows: Callable[
        [], tuple[torch.Tensor, Mapping[str, torch.Tensor]]
    ],
    parameters: Iterable[torch.Tensor],
    *,
    labels: Mapping[str, Any] | None = None,
    sampling: Mapping[str, Any] | None = None,
    n_bins: int = 4,
    top_fracs: tuple[float, ...] = (0.10, 0.25),
) -> dict[str, object]:
    """Attribute the actual batch-mean torque gradient to row groups.

    ``forward_rows`` is called exactly once and must perform a *fresh* model and
    cost forward.  It returns already weighted per-row torque losses (normally
    ``cfg.w_cost * ct.terms["torque"]``) and any labels produced by that same
    forward, notably the current torque violation.  The measured total is
    therefore ``grad(forward_rows()[0].mean())``.

    One row VJP at a time supplies ``sum_i ||g_i||`` for cancellation ratios.
    Group VJPs are immediately reduced to scalar metrics and discarded, so the
    routine retains the autograd graph plus the total and one group gradient,
    never all row gradient vectors.
    """
    params = [param for param in parameters if param.requires_grad]
    if not params:
        raise ValueError("no trainable parameters supplied")

    row_losses, forward_labels = forward_rows()
    if row_losses.ndim != 1:
        raise ValueError(
            f"forward_rows loss must be rank 1, got shape {tuple(row_losses.shape)}"
        )
    batch_size = int(row_losses.numel())
    if batch_size == 0:
        raise ValueError("cannot attribute an empty batch")

    all_labels: dict[str, Any] = dict(labels or {})
    overlap = set(all_labels).intersection(forward_labels)
    if overlap:
        raise ValueError(f"duplicate forward/external labels: {sorted(overlap)}")
    all_labels.update(forward_labels)
    label_cpu = {
        name: _to_cpu_vector(name, values, batch_size)
        for name, values in all_labels.items()
    }
    sampling_cpu = {
        name: _to_cpu_vector(name, values, batch_size)
        for name, values in (sampling or {}).items()
    }

    # This is the actual training scaling for the torque component: every group
    # contribution below retains the same 1/B denominator.
    total_grad = _detached_grads(row_losses.mean(), params)
    total_norm = _norm(total_grad)

    row_grad_norms = torch.zeros(
        batch_size, device=row_losses.device, dtype=torch.float32
    )
    for row in range(batch_size):
        row_grad = _detached_grads(row_losses[row], params)
        row_grad_norms[row] = _norm_tensor(row_grad, row_losses.device)

    mean_row_norm = float(row_grad_norms.mean().item())
    total_row_energy = float(row_grad_norms.square().sum().item())
    out: dict[str, object] = {
        "batch_size": batch_size,
        "batch_mean_denominator": batch_size,
        "group_gradient_definition": "grad(sum(group weighted torque rows) / batch_size)",
        "weighted_torque_loss_mean": float(
            row_losses.detach().float().mean().item()
        ),
        "actual_batch_gradient": {
            "gradient_norm": total_norm,
            "mean_row_gradient_norm": mean_row_norm,
            "max_row_gradient_norm": float(row_grad_norms.max().item()),
            "sum_row_gradient_energy": total_row_energy,
            "cancellation_ratio": (
                total_norm / mean_row_norm if mean_row_norm > 0.0 else 0.0
            ),
        },
        "labels": {},
    }

    for label_name, values in label_cpu.items():
        label_report: dict[str, object] = {
            "value_summary": _summary(values),
            "quantiles": [],
            "tails": {},
        }
        fraction_sum = 0.0
        rank_start = 0
        for quantile, indices_cpu in enumerate(
            _rank_quantile_indices(values, n_bins)
        ):
            indices = indices_cpu.to(device=row_losses.device)
            group_grad = _detached_grads(
                row_losses.index_select(0, indices).sum() / batch_size,
                params,
            )
            metrics = _vector_metrics(group_grad, total_grad)
            denominator = float(
                row_grad_norms.index_select(0, indices).sum().item()
            ) / batch_size
            metrics["sum_row_gradient_norm_over_batch"] = denominator
            metrics["cancellation_ratio"] = (
                metrics["gradient_norm"] / denominator
                if denominator > 0.0
                else 0.0
            )
            group_energy = float(
                row_grad_norms.index_select(0, indices).square().sum().item()
            )
            metrics["row_gradient_energy_fraction"] = (
                group_energy / total_row_energy if total_row_energy > 0.0 else 0.0
            )
            metrics.update({
                "quantile": quantile,
                "rank_fraction_lo": float(rank_start) / batch_size,
                "rank_fraction_hi": float(
                    rank_start + indices_cpu.numel()
                ) / batch_size,
                "rows": int(indices_cpu.numel()),
                "row_fraction": float(indices_cpu.numel()) / batch_size,
                "label_min": float(values.index_select(0, indices_cpu).min()),
                "label_max": float(values.index_select(0, indices_cpu).max()),
            })
            if sampling_cpu:
                metrics["sampling"] = _sampling_group_summary(
                    sampling_cpu, indices_cpu
                )
            fraction_sum += float(metrics["signed_contribution_fraction"])
            label_report["quantiles"].append(metrics)
            rank_start += int(indices_cpu.numel())

        label_report["quantile_signed_contribution_fraction_sum"] = fraction_sum
        order = torch.argsort(values, stable=True)
        for fraction in top_fracs:
            if not 0.0 < float(fraction) < 1.0:
                raise ValueError(f"top fraction must be in (0, 1), got {fraction}")
            count = max(1, min(batch_size - 1, int(round(fraction * batch_size))))
            indices_cpu = order[-count:]
            indices = indices_cpu.to(device=row_losses.device)
            removed_grad = _detached_grads(
                row_losses.index_select(0, indices).sum() / batch_size,
                params,
            )
            removed = _vector_metrics(removed_grad, total_grad)
            denominator = float(
                row_grad_norms.index_select(0, indices).sum().item()
            ) / batch_size
            removed["sum_row_gradient_norm_over_batch"] = denominator
            removed["cancellation_ratio"] = (
                removed["gradient_norm"] / denominator
                if denominator > 0.0
                else 0.0
            )
            removed_energy = float(
                row_grad_norms.index_select(0, indices).square().sum().item()
            )
            removed["row_gradient_energy_fraction"] = (
                removed_energy / total_row_energy
                if total_row_energy > 0.0
                else 0.0
            )
            removed.update({
                "rows": count,
                "row_fraction": float(count) / batch_size,
                "label_min": float(values.index_select(0, indices_cpu).min()),
                "label_max": float(values.index_select(0, indices_cpu).max()),
            })
            if sampling_cpu:
                removed["sampling"] = _sampling_group_summary(
                    sampling_cpu, indices_cpu
                )

            fixed_denominator_grad = _linear_combination(
                total_grad, 1.0, removed_grad, -1.0
            )
            remaining = batch_size - count
            rescale = float(batch_size) / remaining
            fixed_metrics = _vector_metrics(
                fixed_denominator_grad, total_grad
            )
            fixed_norm = float(fixed_metrics["gradient_norm"])
            fixed_dot_total = (
                float(fixed_metrics["projection_onto_total"]) * total_norm
            )
            delta_sq = (
                rescale * rescale * fixed_norm * fixed_norm
                + total_norm * total_norm
                - 2.0 * rescale * fixed_dot_total
            )
            counterfactual = {
                "gradient_norm": rescale * fixed_norm,
                "vector_sum_gradient_norm": rescale * fixed_norm,
                "cosine_with_total": fixed_metrics["cosine_with_total"],
                "projection_onto_total": (
                    rescale * fixed_metrics["projection_onto_total"]
                ),
                "signed_contribution_fraction": (
                    rescale * fixed_metrics["signed_contribution_fraction"]
                ),
            }
            counterfactual.update({
                "rows": remaining,
                "mean_denominator": remaining,
                "gradient_norm_ratio_to_full": (
                    counterfactual["gradient_norm"] / total_norm
                    if total_norm > 0.0
                    else 0.0
                ),
                "fixed_batch_denominator_gradient_norm": fixed_norm,
                "delta_from_full_gradient_norm": math.sqrt(
                    max(delta_sq, 0.0)
                ),
            })
            key = f"top_{int(round(100.0 * fraction)):02d}pct"
            label_report["tails"][key] = {
                "removed_group": removed,
                "counterfactual_excluding": counterfactual,
            }
        out["labels"][label_name] = label_report

    if sampling_cpu:
        all_indices = torch.arange(batch_size)
        out["sampling_on_batch_rows"] = _sampling_group_summary(
            sampling_cpu, all_indices
        )
    return out


def replay_sampling_probabilities(
    replay,
    bucket: int,
    draws: int,
    *,
    short_wp: int = 0,
    short_boost: float = 1.0,
    viol_downweight: float | None = None,
    viol_uniform_mix: float | None = None,
    stratify: bool = False,
    frac_prof: float = 0.40,
    frac_new: float = 0.40,
    ema_nmin: int = 2,
    ema_prof: float = 0.70,
) -> dict[str, object]:
    """Reconstruct the with-replacement probabilities used by ReplayPool.

    For proficiency stratification there is no single IID draw probability.
    ``expected_draw_fraction`` and ``inclusion_probability`` combine the fixed
    proficient/new/flex quotas and any base-distribution top-up exactly.
    """
    bucket = int(bucket)
    draws = int(draws)
    size = int(replay.size(bucket))
    if size <= 0 or draws <= 0:
        raise ValueError(f"need positive pool size/draws, got {size}/{draws}")
    raw_weight = replay._row_weights(
        bucket,
        size,
        short_wp=short_wp,
        short_boost=short_boost,
        viol_downweight=viol_downweight,
    ).detach().float()
    expected_count = torch.zeros_like(raw_weight)
    survival = torch.ones_like(raw_weight)
    stages: list[dict[str, object]] = []
    gamma = (
        float(getattr(replay, "viol_uniform_mix", 0.0))
        if viol_uniform_mix is None else float(viol_uniform_mix)
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"viol_uniform_mix must be in [0, 1], got {gamma}")

    def probability(mask: torch.Tensor, *, scale_ema: bool) -> torch.Tensor:
        ids = mask.nonzero(as_tuple=True)[0]
        probs = torch.zeros_like(raw_weight)
        if ids.numel() == 0:
            return probs
        weight = raw_weight.index_select(0, ids)
        if scale_ema:
            weight = weight * replay.data[bucket]["ema_m"][:size].float().index_select(
                0, ids
            ).clamp_min(1e-6)
        clean = torch.where(
            torch.isfinite(weight) & (weight > 0.0),
            weight,
            torch.zeros_like(weight),
        )
        total = clean.sum()
        uniform = torch.full_like(clean, 1.0 / int(ids.numel()))
        base = torch.where(
            torch.isfinite(total) & (total > 0.0),
            clean / total.clamp_min(torch.finfo(clean.dtype).tiny),
            uniform,
        )
        local = (1.0 - gamma) * base + gamma * uniform
        probs.index_copy_(0, ids, local)
        return probs

    def add_stage(
        name: str,
        mask: torch.Tensor,
        count: int,
        *,
        scale_ema: bool,
    ) -> bool:
        nonlocal expected_count, survival
        if count <= 0 or not bool(mask.any()):
            return False
        probs = probability(mask, scale_ema=scale_ema)
        expected_count = expected_count + count * probs
        survival = survival * (1.0 - probs).pow(count)
        stages.append({
            "name": name,
            "draws": int(count),
            "eligible_rows": int(mask.sum()),
            "ema_scaled": bool(scale_ema),
            "uniform_mix_gamma": gamma,
        })
        return True

    all_rows = torch.ones(size, device=raw_weight.device, dtype=torch.bool)
    use_stratified = (
        bool(stratify)
        and "ema_m" in replay.data[bucket]
        and (float(frac_prof) > 0.0 or float(frac_new) > 0.0)
    )
    if not use_stratified:
        add_stage("base", all_rows, draws, scale_ema=False)
        scheme = "weighted_with_replacement"
    else:
        ema_m = replay.data[bucket]["ema_m"][:size]
        ema_n = replay.data[bucket]["ema_n"][:size]
        trusted = ema_n >= int(ema_nmin)
        proficient = trusted & (ema_m >= float(ema_prof))
        new = ~trusted
        flex = trusted
        k_prof = min(int(round(draws * float(frac_prof))), draws)
        k_new = min(int(round(draws * float(frac_new))), draws - k_prof)
        k_flex = draws - k_prof - k_new
        used = 0
        used += k_prof if add_stage(
            "proficient", proficient, k_prof, scale_ema=True
        ) else 0
        used += k_new if add_stage(
            "new", new, k_new, scale_ema=False
        ) else 0
        used += k_flex if add_stage(
            "flex", flex, k_flex, scale_ema=True
        ) else 0
        if used == 0:
            add_stage("base_fallback", all_rows, draws, scale_ema=False)
        elif used < draws:
            add_stage("base_topup", all_rows, draws - used, scale_ema=False)
        scheme = "stratified_weighted_with_replacement"

    return {
        "scheme": scheme,
        "replacement": True,
        "pool_size": size,
        "draws": draws,
        "viol_sampling_beta": (
            float(getattr(replay, "viol_downweight", 0.0))
            if viol_downweight is None else float(viol_downweight)
        ),
        "viol_uniform_mix": gamma,
        "raw_weight": raw_weight,
        "expected_draw_count": expected_count,
        "expected_draw_fraction": expected_count / draws,
        "inclusion_probability": 1.0 - survival,
        "stages": stages,
    }


def summarize_replay_sampling(
    diagnostics: Mapping[str, object],
    selected_indices: torch.Tensor,
) -> dict[str, object]:
    """JSON-friendly population and selected-row sampling summary."""
    selected = selected_indices.detach().to(
        device="cpu", dtype=torch.long
    ).reshape(-1)
    fields = (
        "raw_weight",
        "expected_draw_count",
        "expected_draw_fraction",
        "inclusion_probability",
    )
    tensors = {
        field: torch.as_tensor(diagnostics[field]).detach().cpu().float()
        for field in fields
    }
    return {
        "scheme": diagnostics["scheme"],
        "replacement": diagnostics["replacement"],
        "pool_size": diagnostics["pool_size"],
        "draws": diagnostics["draws"],
        "viol_sampling_beta": diagnostics.get("viol_sampling_beta", 0.0),
        "viol_uniform_mix": diagnostics.get("viol_uniform_mix", 0.0),
        "unique_selected_rows": int(selected.unique().numel()),
        "expected_unique_rows": float(tensors["inclusion_probability"].sum()),
        "stages": diagnostics["stages"],
        "population": {
            field: _summary(values) for field, values in tensors.items()
        },
        "selected_batch_rows": {
            field: _summary(values.index_select(0, selected))
            for field, values in tensors.items()
        },
        "note": "diagnostic probabilities only; training applies no importance correction",
    }
