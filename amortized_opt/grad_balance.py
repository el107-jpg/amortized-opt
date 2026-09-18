"""AL-informed GradNorm + optional PCGrad on torque/waist.

Priority ``p_i`` follows the augmented Lagrangian (``λ`` × excess violation).
Allocation ``τ_i`` splits a unit budget; each group contributes ``τ_i · ĝ_i``
with ``ĝ_i = g_i / ‖g_i‖``.  Torque is damped (RNEA inflates raw ‖g‖) and
capped so it cannot monopolize the step.

Parameter-space ``al_pcgrad`` retains the historical symmetric torque/waist
projection.  Output-space PCGrad protects the dedicated yaw/pitch/roll
``joint_box_waist`` direction one-sided: only a conflicting torque gradient is
projected, while the box and all other gradients remain unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .lagrangian import ACTUATOR_SEGMENT_NAMES, AugmentedLagrangian

__all__ = [
    "GRAD_BALANCE_GROUPS",
    "KEY_COS_PAIRS",
    "al_gradnorm_priorities",
    "al_gradnorm_tau",
    "apply_al_gradnorm",
    "apply_al_pcgrad",
    "clip_torque_gradient_budget",
    "compute_group_gradients",
    "output_space_al_pcgrad",
    "output_space_gradnorm_surrogate",
    "output_space_protected_surrogate_gradient",
    "pcgrad_protect_reference",
    "pcgrad_two_groups",
    "update_group_norm_ema",
    "grad_list_norm",
    "group_grad_cosines",
    "measure_group_grad_cosines",
]

#: ``name -> (cost term keys summed for ∇, AL viol keys for priority)``
GRAD_BALANCE_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "torque": (
        ("torque",),
        ("torque_peak",)
        + tuple(f"torque_{s}" for s in ACTUATOR_SEGMENT_NAMES),
    ),
    "feas_other": (
        ("velocity", "joint_box", "collision"),
        ("velocity_peak", "collision_peak", "joint_box_peak")
        + tuple(f"velocity_{s}" for s in ACTUATOR_SEGMENT_NAMES)
        + tuple(f"joint_box_{s}" for s in ACTUATOR_SEGMENT_NAMES)
        + ("collision",),
    ),
    "strike": (
        ("terminal", "terminal_aim"),
        ("terminal",),
    ),
    "shape": (
        ("smooth", "posture", "effort", "tip_crossover"),
        (),
    ),
    "waist": (
        ("waist_upright", "waist_aim", "waist_yaw_vel", "waist_pr_vel",
         "waist_pr_band", "waist_pr_terminal"),
        ("waist_pr_band",),
    ),
}


#: Logged every ``log_every`` train row when group gradients are available.
KEY_COS_PAIRS: tuple[tuple[str, str], ...] = (
    ("torque", "strike"),
    ("torque", "shape"),
    ("torque", "feas_other"),
    ("torque", "waist"),
    ("strike", "shape"),
    ("strike", "feas_other"),
)


def _flat_group_grad(
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor | None],
) -> torch.Tensor | None:
    if not params:
        return None
    parts: list[torch.Tensor] = []
    for p, g in zip(params, grads):
        if g is None:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            parts.append(g.reshape(-1))
    return torch.cat(parts)


def group_grad_cosines(
    params: Sequence[torch.Tensor],
    group_grads: dict[str, tuple[torch.Tensor | None, ...]],
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Pairwise cosine similarity of flattened group gradients."""
    flats: dict[str, torch.Tensor] = {}
    for name, grads in group_grads.items():
        flat = _flat_group_grad(params, grads)
        if flat is not None:
            flats[name] = flat
    stats: dict[str, float] = {}
    cosines: list[float] = []
    for a, b in KEY_COS_PAIRS:
        va, vb = flats.get(a), flats.get(b)
        if va is None or vb is None:
            continue
        na, nb = float(va.norm().item()), float(vb.norm().item())
        if na < eps or nb < eps:
            cos = 0.0
        else:
            cos = float((va @ vb) / (na * nb))
        stats[f"cos_{a}_{b}"] = cos
        cosines.append(cos)
    if cosines:
        stats["cos_min"] = min(cosines)
        stats["cos_mean"] = sum(cosines) / len(cosines)
    return stats


def compute_group_gradients(
    params: list[torch.Tensor],
    ct,
    *,
    w_cost: float = 1.0,
    batched_vjp: bool = True,
) -> dict[str, tuple[torch.Tensor | None, ...]]:
    """Gradients per group, using one batched VJP by default.

    ``is_grads_batched=True`` asks autograd/vmap to evaluate the five scalar
    group VJPs together.  The sequential path remains available for regression
    checks and toolchains with a problematic batching rule.
    """
    names = list(GRAD_BALANCE_GROUPS)
    if getattr(ct, "group_terms", None) is not None:
        scalars = [w_cost * ct.group_terms[gname] for gname in names]
    else:
        scalars = []
        for gname in names:
            term_keys, _ = GRAD_BALANCE_GROUPS[gname]
            scalars.append(w_cost * sum(ct.terms[k].mean() for k in term_keys))

    if batched_vjp:
        stacked = torch.stack(scalars)
        eye = torch.eye(
            len(names), device=stacked.device, dtype=stacked.dtype,
        )
        grads_by_param = torch.autograd.grad(
            stacked, params, grad_outputs=eye, is_grads_batched=True,
            retain_graph=True, allow_unused=True,
        )
        return {
            gname: tuple(
                None if g is None else g[gi] for g in grads_by_param
            )
            for gi, gname in enumerate(names)
        }

    group_grads: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for gname, scalar in zip(names, scalars):
        if not scalar.requires_grad:
            group_grads[gname] = tuple(None for _ in params)
            continue
        group_grads[gname] = torch.autograd.grad(
            scalar, params, retain_graph=True, allow_unused=True,
        )
    return group_grads


def _compute_named_scalar_gradients(
    params: list[torch.Tensor],
    scalars: dict[str, torch.Tensor],
    *,
    batched_vjp: bool,
) -> dict[str, tuple[torch.Tensor | None, ...]]:
    """VJPs for arbitrary named scalars, optionally in one batched call."""
    names = list(scalars)
    values = [scalars[name] for name in names]
    if batched_vjp:
        stacked = torch.stack(values)
        eye = torch.eye(len(names), device=stacked.device, dtype=stacked.dtype)
        grads_by_param = torch.autograd.grad(
            stacked, params, grad_outputs=eye, is_grads_batched=True,
            retain_graph=True, allow_unused=True,
        )
        return {
            name: tuple(None if g is None else g[i] for g in grads_by_param)
            for i, name in enumerate(names)
        }
    out: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for name, scalar in scalars.items():
        out[name] = torch.autograd.grad(
            scalar, params, retain_graph=True, allow_unused=True,
        )
    return out


def measure_group_grad_cosines(
    module,
    cost,
    *,
    init: torch.Tensor,
    state: torch.Tensor,
    enc,
    masks,
    q0: torch.Tensor,
    v0: torch.Tensor,
    strike,
    lengths: torch.Tensor,
    weights: dict,
    w_cost: float = 1.0,
) -> dict[str, float]:
    """Diagnostic cosines without writing ``param.grad``."""
    params = trainable_params(module)
    if not params:
        return {}
    dec = module(init, state, enc, masks)
    ct = cost.forward(
        q0, dec.traj, v0, strike, lengths=lengths, weights=weights,
    )
    return group_grad_cosines(
        params,
        compute_group_gradients(params, ct, w_cost=w_cost),
    )


def grad_list_norm(grads: Sequence[torch.Tensor | None]) -> float:
    """Host scalar norm for diagnostics/tests; training uses the tensor form."""
    norm = _grad_list_norm_tensor(grads)
    return 0.0 if norm is None else float(norm.item())


def _grad_list_norm_tensor(
    grads: Sequence[torch.Tensor | None],
) -> torch.Tensor | None:
    """GPU-resident FP32 norm without a device synchronization."""
    terms = [
        g.detach().float().pow(2).sum()
        for g in grads
        if g is not None
    ]
    if not terms:
        return None
    return torch.stack(terms).sum().sqrt()


def _grad_list_dot(
    a: Sequence[torch.Tensor | None],
    b: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    out = None
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        term = (x * y).sum()
        out = term if out is None else out + term
    if out is None:
        ref = next((t for t in a if t is not None), None)
        if ref is None:
            ref = next((t for t in b if t is not None), None)
        return ref.new_zeros(()) if ref is not None else torch.zeros(())
    return out


def _grad_list_project_off(
    a: Sequence[torch.Tensor | None],
    b: Sequence[torch.Tensor | None],
    *,
    eps: float = 1e-8,
) -> list[torch.Tensor | None]:
    """Return *a* projected off *b* on conflict, entirely on device."""
    dot = _grad_list_dot(a, b)
    b_sq = [g.detach().float().pow(2).sum() for g in b if g is not None]
    if not b_sq:
        return [g.clone() if g is not None else None for g in a]
    norm_b_sq = torch.stack(b_sq).sum()
    scale = torch.where(
        (dot < 0.0) & (norm_b_sq > eps),
        dot / norm_b_sq.clamp_min(eps),
        dot.new_zeros(()),
    )
    return [
        g - scale.to(dtype=g.dtype) * h
        if g is not None and h is not None else g
        for g, h in zip(a, b)
    ]


def pcgrad_two_groups(
    g_a: Sequence[torch.Tensor | None],
    g_b: Sequence[torch.Tensor | None],
    *,
    eps: float = 1e-8,
) -> tuple[tuple[torch.Tensor | None, ...], tuple[torch.Tensor | None, ...]]:
    """Two-task PCGrad: project each group off the other when ``cos < 0``."""
    ga = _grad_list_project_off(g_a, g_b, eps=eps)
    gb = _grad_list_project_off(g_b, g_a, eps=eps)
    return tuple(ga), tuple(gb)


def pcgrad_protect_reference(
    subject: Sequence[torch.Tensor | None],
    reference: Sequence[torch.Tensor | None],
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor | None, ...]:
    """Project only ``subject`` off a conflicting protected ``reference``."""
    return tuple(_grad_list_project_off(subject, reference, eps=eps))


def _protected_term(ct, name: str) -> torch.Tensor:
    terms = getattr(ct, "protected_terms", None)
    if terms is None:
        terms = getattr(ct, "balance_terms", None)
    if terms is None or name not in terms:
        raise KeyError(f"cost payload has no protected term {name!r}")
    return terms[name]


def al_gradnorm_priorities(
    al: AugmentedLagrangian,
    viol_means: dict[str, float],
    *,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    shape_floor: float = 1.0,
    waist_floor: float = 0.0,
) -> dict[str, float]:
    """Per-group priority before normalization (higher → larger τ slice)."""
    lam = al.weights()
    out: dict[str, float] = {}
    for name, (_terms, al_keys) in GRAD_BALANCE_GROUPS.items():
        if not al_keys:
            out[name] = float(shape_floor)
            continue
        lam_i = max(float(lam.get(k, 1.0)) for k in al_keys)
        excess = 0.0
        for k in al_keys:
            v = float(viol_means.get(k, 0.0))
            t = float(al.target.get(k, 0.0))
            excess = max(excess, max(0.0, v - t))
        p = lam_i * (excess ** float(beta) if excess > 0.0 else 0.0)
        if p <= 0.0:
            p = float(shape_floor) * 0.25
        if name == "torque":
            p *= float(torque_gamma)
        if name == "waist" and float(waist_floor) > 0.0:
            p = max(p, float(waist_floor))
        out[name] = p
    return out


def al_gradnorm_tau(
    priorities: dict[str, float],
    *,
    torque_cap: float = 0.30,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Normalize priorities to ``τ`` on the simplex, then cap torque."""
    names = list(GRAD_BALANCE_GROUPS)
    raw = [max(float(priorities.get(n, 0.0)), 0.0) for n in names]
    s = sum(raw)
    if s <= eps:
        tau = {n: 1.0 / len(names) for n in names}
    else:
        tau = {n: v / s for n, v in zip(names, raw)}
    cap = float(torque_cap)
    if cap > 0.0 and "torque" in tau and tau["torque"] > cap:
        excess = tau["torque"] - cap
        tau["torque"] = cap
        others = [n for n in names if n != "torque"]
        osum = sum(tau[n] for n in others)
        if osum > eps:
            for n in others:
                tau[n] += excess * (tau[n] / osum)
    return tau


def _al_gradnorm_priorities_tensor(
    al: AugmentedLagrangian,
    violations: dict[str, torch.Tensor],
    ref: torch.Tensor,
    *,
    torque_gamma: float,
    beta: float,
    shape_floor: float,
    waist_floor: float,
) -> dict[str, torch.Tensor]:
    """GPU-resident equivalent of :func:`al_gradnorm_priorities`."""
    lam = al.weights()
    out: dict[str, torch.Tensor] = {}
    fallback = ref.new_tensor(float(shape_floor) * 0.25)
    for name, (_terms, al_keys) in GRAD_BALANCE_GROUPS.items():
        if not al_keys:
            out[name] = ref.new_tensor(float(shape_floor))
            continue
        excesses = [
            violations[k].detach().float().mean()
            - float(al.target.get(k, 0.0))
            for k in al_keys
            if k in violations
        ]
        excess = (
            torch.stack(excesses).clamp_min(0.0).amax()
            if excesses else ref.new_zeros(())
        )
        lam_i = max(float(lam.get(k, 1.0)) for k in al_keys)
        p = float(lam_i) * excess.pow(float(beta))
        p = torch.where(p > 0.0, p, fallback)
        if name == "torque":
            p = p * float(torque_gamma)
        if name == "waist" and float(waist_floor) > 0.0:
            p = torch.maximum(p, ref.new_tensor(float(waist_floor)))
        out[name] = p
    return out


def _al_gradnorm_tau_tensor(
    priorities: dict[str, torch.Tensor],
    ref: torch.Tensor,
    *,
    torque_cap: float,
    eps: float,
) -> dict[str, torch.Tensor]:
    """GPU-resident simplex normalization and torque cap."""
    names = list(GRAD_BALANCE_GROUPS)
    raw = torch.stack([
        priorities.get(n, ref.new_zeros(())).clamp_min(0.0) for n in names
    ])
    total = raw.sum()
    tau_vec = torch.where(
        total > eps,
        raw / total.clamp_min(eps),
        torch.full_like(raw, 1.0 / len(names)),
    )
    if float(torque_cap) > 0.0 and "torque" in names:
        ti = names.index("torque")
        old_torque = tau_vec[ti]
        capped = old_torque.clamp_max(float(torque_cap))
        excess = (old_torque - float(torque_cap)).clamp_min(0.0)
        other_sum = (tau_vec.sum() - old_torque).clamp_min(eps)
        redistributed = tau_vec + excess * tau_vec / other_sum
        tau_vec = torch.stack([
            capped if i == ti else redistributed[i]
            for i in range(len(names))
        ])
    return {n: tau_vec[i] for i, n in enumerate(names)}


def _combine_group_gradients(
    targets: Sequence[torch.Tensor],
    ct,
    al: AugmentedLagrangian,
    group_grads: dict[str, tuple[torch.Tensor | None, ...]],
    *,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    torque_cap: float = 0.30,
    shape_floor: float = 1.0,
    waist_floor: float = 0.0,
    torque_grad_scale: float = 1.0,
    pcgrad_torque_waist: bool = False,
    torque_protect_ref: Sequence[torch.Tensor | None] | None = None,
    compute_cosines: bool = True,
    eps: float = 1e-8,
) -> tuple[list[torch.Tensor], dict[str, float | torch.Tensor]]:
    """Normalize and combine already-computed gradients on ``targets``."""
    ref = targets[0].detach().float().new_zeros(())
    priorities = _al_gradnorm_priorities_tensor(
        al, ct.violations, ref,
        torque_gamma=torque_gamma, beta=beta, shape_floor=shape_floor,
        waist_floor=waist_floor,
    )
    tau = _al_gradnorm_tau_tensor(
        priorities, ref, torque_cap=torque_cap, eps=eps,
    )
    torque_grad_scale = float(torque_grad_scale)

    stats: dict[str, float | torch.Tensor] = (
        group_grad_cosines(targets, group_grads) if compute_cosines else {}
    )

    if pcgrad_torque_waist:
        g_t, g_w = pcgrad_two_groups(
            group_grads.get("torque", ()), group_grads.get("waist", ()),
            eps=eps,
        )
        group_grads = dict(group_grads)
        group_grads["torque"] = g_t
        group_grads["waist"] = g_w
        stats["gb_pcgrad_tw"] = 1.0
    if torque_protect_ref is not None:
        torque = group_grads.get("torque", ())
        dot = _grad_list_dot(torque, torque_protect_ref)
        norm_t = _grad_list_norm_tensor(torque)
        norm_r = _grad_list_norm_tensor(torque_protect_ref)
        group_grads = dict(group_grads)
        group_grads["torque"] = pcgrad_protect_reference(
            torque, torque_protect_ref, eps=eps,
        )
        stats["gb_pcgrad_tw"] = 0.0
        stats["gb_pcgrad_torque_box"] = 1.0
        stats["gb_pcgrad_torque_box_conflict"] = (dot < 0.0).float().detach()
        if compute_cosines and norm_t is not None and norm_r is not None:
            stats["cos_torque_joint_box_waist"] = (
                dot / (norm_t * norm_r).clamp_min(eps)
            ).detach()

    names = list(GRAD_BALANCE_GROUPS)
    combined = [torch.zeros_like(t) for t in targets]
    for gname in names:
        grads = group_grads.get(gname, ())
        norm = _grad_list_norm_tensor(grads)
        if norm is None:
            stats[f"gb_norm_{gname}"] = ref
            stats[f"gb_tau_{gname}"] = tau[gname].detach()
            stats[f"gb_pri_{gname}"] = priorities[gname].detach()
            continue
        stats[f"gb_norm_{gname}"] = norm.detach()
        stats[f"gb_tau_{gname}"] = tau[gname].detach()
        stats[f"gb_pri_{gname}"] = priorities[gname].detach()
        scale = tau[gname] / norm.clamp_min(eps)
        if gname == "torque" and torque_grad_scale != 1.0:
            scale = scale * scale.new_tensor(torque_grad_scale)
        stats[f"gb_scale_{gname}"] = scale.detach()
        for i, g in enumerate(grads):
            if g is not None:
                combined[i] = combined[i] + scale.to(g.dtype) * g

    stats["gb_tau_torque"] = tau.get("torque", ref).detach()
    stats["gb_torque_grad_scale"] = ref.new_tensor(torque_grad_scale)
    return combined, stats


def _apply_group_grad_balance(
    params: list[torch.Tensor],
    ct,
    al: AugmentedLagrangian,
    *,
    w_cost: float = 1.0,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    torque_cap: float = 0.30,
    shape_floor: float = 1.0,
    waist_floor: float = 0.0,
    torque_grad_scale: float = 1.0,
    pcgrad_torque_waist: bool = False,
    compute_cosines: bool = True,
    batched_vjp: bool = True,
    eps: float = 1e-8,
) -> dict[str, float | torch.Tensor]:
    """Compute parameter-space group gradients, combine, and write ``.grad``."""
    if not params:
        return {}
    group_grads = compute_group_gradients(
        params, ct, w_cost=w_cost, batched_vjp=batched_vjp,
    )
    combined, stats = _combine_group_gradients(
        params, ct, al, group_grads,
        torque_gamma=torque_gamma, beta=beta, torque_cap=torque_cap,
        shape_floor=shape_floor, waist_floor=waist_floor,
        torque_grad_scale=torque_grad_scale,
        pcgrad_torque_waist=pcgrad_torque_waist,
        compute_cosines=compute_cosines, eps=eps,
    )
    stats["gb_batched_vjp"] = 1.0 if batched_vjp else 0.0
    stats["gb_output_space"] = 0.0
    for p, g in zip(params, combined):
        p.grad = g
    return stats


def output_space_al_pcgrad(
    traj: torch.Tensor,
    ct,
    al: AugmentedLagrangian,
    *,
    w_cost: float = 1.0,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    torque_cap: float = 0.30,
    shape_floor: float = 1.0,
    waist_floor: float = 2.0,
    torque_grad_scale: float = 1.0,
    compute_cosines: bool = True,
    batched_vjp: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """PCGrad in trajectory space for a single downstream model backward."""
    targets = [traj]
    if getattr(ct, "group_terms", None) is not None:
        scalars = {
            name: float(w_cost) * ct.group_terms[name]
            for name in GRAD_BALANCE_GROUPS
        }
    else:
        scalars = {
            name: float(w_cost) * sum(
                ct.terms[k].mean() for k in term_keys
            )
            for name, (term_keys, _al_keys) in GRAD_BALANCE_GROUPS.items()
        }
    protect_name = "_joint_box_waist_protect"
    scalars[protect_name] = float(w_cost) * _protected_term(
        ct, "joint_box_waist",
    )
    all_grads = _compute_named_scalar_gradients(
        targets, scalars, batched_vjp=batched_vjp,
    )
    protect_grad = all_grads.pop(protect_name)
    group_grads = all_grads
    combined, stats = _combine_group_gradients(
        targets, ct, al, group_grads,
        torque_gamma=torque_gamma, beta=beta, torque_cap=torque_cap,
        shape_floor=shape_floor, waist_floor=waist_floor,
        torque_grad_scale=torque_grad_scale,
        torque_protect_ref=protect_grad,
        compute_cosines=compute_cosines, eps=eps,
    )
    stats["gb_batched_vjp"] = 1.0 if batched_vjp else 0.0
    stats["gb_output_space"] = 1.0
    return combined[0], stats


def update_group_norm_ema(
    previous: dict[str, torch.Tensor],
    stats: dict[str, float | torch.Tensor],
    *,
    decay: float,
) -> dict[str, torch.Tensor]:
    """Update detached per-group norm EMAs from an exact refresh."""
    out: dict[str, torch.Tensor] = {}
    d = float(decay)
    for name in GRAD_BALANCE_GROUPS:
        value = stats.get(f"gb_norm_{name}")
        if not isinstance(value, torch.Tensor):
            continue
        current = value.detach()
        old = previous.get(name)
        out[name] = current if old is None else d * old + (1.0 - d) * current
    return out


def output_space_gradnorm_surrogate(
    ct,
    al: AugmentedLagrangian,
    norm_ema: dict[str, torch.Tensor],
    *,
    w_cost: float = 1.0,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    torque_cap: float = 0.30,
    shape_floor: float = 1.0,
    waist_floor: float = 2.0,
    torque_grad_scale: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """One-backward approximation using norms from the latest exact refresh."""
    missing = [name for name in GRAD_BALANCE_GROUPS if name not in norm_ema]
    if missing:
        raise ValueError(f"missing gradient-norm EMA for groups: {missing}")

    if getattr(ct, "group_terms", None) is not None:
        group_terms = ct.group_terms
    else:
        group_terms = {
            name: sum(ct.terms[k].mean() for k in term_keys)
            for name, (term_keys, _al_keys) in GRAD_BALANCE_GROUPS.items()
        }
    ref = next(iter(group_terms.values())).detach().float().new_zeros(())
    priorities = _al_gradnorm_priorities_tensor(
        al, ct.violations, ref,
        torque_gamma=torque_gamma, beta=beta, shape_floor=shape_floor,
        waist_floor=waist_floor,
    )
    tau = _al_gradnorm_tau_tensor(
        priorities, ref, torque_cap=torque_cap, eps=eps,
    )
    torque_grad_scale = float(torque_grad_scale)

    loss = ref
    stats: dict[str, float | torch.Tensor] = {}
    for name in GRAD_BALANCE_GROUPS:
        norm = norm_ema[name].detach().float()
        scale = tau[name] / norm.clamp_min(eps)
        if name == "torque" and torque_grad_scale != 1.0:
            scale = scale * scale.new_tensor(torque_grad_scale)
        loss = loss + scale.to(group_terms[name].dtype) * (
            float(w_cost) * group_terms[name]
        )
        stats[f"gb_norm_{name}"] = norm
        stats[f"gb_tau_{name}"] = tau[name].detach()
        stats[f"gb_pri_{name}"] = priorities[name].detach()
        stats[f"gb_scale_{name}"] = scale.detach()
    stats["gb_tau_torque"] = tau.get("torque", ref).detach()
    stats["gb_torque_grad_scale"] = ref.new_tensor(torque_grad_scale)
    stats["gb_batched_vjp"] = 0.0
    stats["gb_output_space"] = 1.0
    stats["gb_pcgrad_tw"] = 0.0
    stats["gb_refresh"] = 0.0
    return loss, stats


def output_space_protected_surrogate_gradient(
    traj: torch.Tensor,
    surrogate: torch.Tensor,
    ct,
    stats: dict[str, float | torch.Tensor],
    *,
    w_cost: float = 1.0,
    batched_vjp: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Differentiate an EMA surrogate while protecting the waist box.

    The surrogate already contains the unprojected torque contribution.  Three
    VJPs recover its total gradient, raw torque gradient and protected
    yaw/pitch/roll box direction; the returned correction replaces only the
    torque component with its one-sided PCGrad projection.
    """
    torque_term = (
        ct.group_terms["torque"]
        if getattr(ct, "group_terms", None) is not None
        else sum(
            ct.terms[k].mean()
            for k in GRAD_BALANCE_GROUPS["torque"][0]
        )
    )
    named = {
        "surrogate": surrogate,
        "torque": float(w_cost) * torque_term,
        "joint_box_waist": float(w_cost) * _protected_term(
            ct, "joint_box_waist",
        ),
    }
    grads = _compute_named_scalar_gradients(
        [traj], named, batched_vjp=batched_vjp,
    )
    base = grads["surrogate"][0]
    torque = grads["torque"]
    reference = grads["joint_box_waist"]
    if base is None or torque[0] is None:
        raise RuntimeError("trajectory is disconnected from balance surrogate")
    projected = pcgrad_protect_reference(torque, reference, eps=eps)
    correction = projected[0] - torque[0]
    scale = stats["gb_scale_torque"]
    if not isinstance(scale, torch.Tensor):
        scale = base.new_tensor(float(scale))
    combined = base + scale.to(base.dtype) * correction

    dot = _grad_list_dot(torque, reference)
    norm_t = _grad_list_norm_tensor(torque)
    norm_r = _grad_list_norm_tensor(reference)
    out = dict(stats)
    out["gb_pcgrad_torque_box"] = 1.0
    out["gb_pcgrad_torque_box_conflict"] = (dot < 0.0).float().detach()
    if norm_t is not None and norm_r is not None:
        out["cos_torque_joint_box_waist"] = (
            dot / (norm_t * norm_r).clamp_min(eps)
        ).detach()
    return combined, out


def apply_al_gradnorm(
    params: list[torch.Tensor],
    ct,
    al: AugmentedLagrangian,
    *,
    w_cost: float = 1.0,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    torque_cap: float = 0.30,
    shape_floor: float = 1.0,
    torque_grad_scale: float = 1.0,
    compute_cosines: bool = True,
    batched_vjp: bool = True,
    eps: float = 1e-8,
) -> dict[str, float | torch.Tensor]:
    return _apply_group_grad_balance(
        params, ct, al,
        w_cost=w_cost, torque_gamma=torque_gamma, beta=beta,
        torque_cap=torque_cap, shape_floor=shape_floor,
        torque_grad_scale=torque_grad_scale,
        compute_cosines=compute_cosines, batched_vjp=batched_vjp, eps=eps,
    )


def apply_al_pcgrad(
    params: list[torch.Tensor],
    ct,
    al: AugmentedLagrangian,
    *,
    w_cost: float = 1.0,
    torque_gamma: float = 0.35,
    beta: float = 0.5,
    torque_cap: float = 0.30,
    shape_floor: float = 1.0,
    waist_floor: float = 2.0,
    torque_grad_scale: float = 1.0,
    compute_cosines: bool = True,
    batched_vjp: bool = True,
    eps: float = 1e-8,
) -> dict[str, float | torch.Tensor]:
    """AL-GradNorm with PCGrad on torque/waist and a higher waist priority floor."""
    return _apply_group_grad_balance(
        params, ct, al,
        w_cost=w_cost, torque_gamma=torque_gamma, beta=beta,
        torque_cap=torque_cap, shape_floor=shape_floor,
        waist_floor=waist_floor, torque_grad_scale=torque_grad_scale,
        pcgrad_torque_waist=True,
        compute_cosines=compute_cosines, batched_vjp=batched_vjp, eps=eps,
    )


def trainable_params(module) -> list[torch.Tensor]:
    core = module.module if hasattr(module, "module") else module
    return [p for p in core.parameters() if p.requires_grad]


def clip_torque_gradient_budget(
    params: Sequence[torch.Tensor],
    torque_term_mean: torch.Tensor,
    *,
    w_cost: float,
    cap: float,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Cap the measured torque share after ``total.backward(retain_graph=True)``.

    Writing ``g_total = g_rest + g_torque``, choose the largest
    ``alpha in [0, 1]`` such that

    ``||alpha * g_torque|| / max(||g_rest + alpha * g_torque||, eps) <= cap``.

    The quadratic is solved from float64 dot products.  ``p.grad`` is left
    untouched when the cap is inactive or either input gradient is nonfinite;
    otherwise the scaled total is reconstructed exactly once.
    """
    cap = float(cap)
    if not math.isfinite(cap):
        raise ValueError(f"cap must be finite, got {cap!r}")
    if cap <= 0.0 or cap >= 1.0:
        return {}
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps!r}")
    active = [p for p in params if p.requires_grad]
    if not active:
        return {}

    g_full: list[torch.Tensor | None] = []
    nf_sq = torque_term_mean.new_zeros((), dtype=torch.float64)
    full_finite = True
    for p in active:
        if p.grad is None:
            g_full.append(None)
            continue
        g = p.grad.detach()
        g_full.append(g)
        full_finite = full_finite and bool(torch.isfinite(g).all().item())
        nf_sq = nf_sq + g.double().square().sum()
    if not full_finite or not bool(torch.isfinite(nf_sq).item()):
        return {
            "torque_grad_cap": cap,
            "torque_grad_status_nonfinite_full": 1.0,
        }

    grads_t = torch.autograd.grad(
        float(w_cost) * torque_term_mean,
        active,
        retain_graph=True,
        allow_unused=True,
    )
    nt_sq = torque_term_mean.new_zeros((), dtype=torch.float64)
    rest_sq = torque_term_mean.new_zeros((), dtype=torch.float64)
    rest_torque_dot = torque_term_mean.new_zeros((), dtype=torch.float64)
    torque_finite = True
    for gf, gt in zip(g_full, grads_t):
        if gt is not None:
            torque_finite = torque_finite and bool(
                torch.isfinite(gt).all().item()
            )
        if gf is not None and gt is not None:
            rest = gf.double() - gt.detach().double()
            torque = gt.detach().double()
            nt_sq = nt_sq + torque.square().sum()
            rest_sq = rest_sq + rest.square().sum()
            rest_torque_dot = rest_torque_dot + (rest * torque).sum()
        elif gf is not None:
            rest_sq = rest_sq + gf.double().square().sum()
        elif gt is not None:
            torque = gt.detach().double()
            torque_sq = torque.square().sum()
            nt_sq = nt_sq + torque_sq
            rest_sq = rest_sq + torque_sq
            rest_torque_dot = rest_torque_dot - torque_sq
    scalar_finite = all(
        bool(torch.isfinite(value).item())
        for value in (nt_sq, rest_sq, rest_torque_dot)
    )
    if not torque_finite or not scalar_finite:
        return {
            "torque_grad_cap": cap,
            "torque_grad_status_nonfinite_torque": 1.0,
        }

    nf = math.sqrt(max(float(nf_sq.item()), 0.0))
    nt = math.sqrt(max(float(nt_sq.item()), 0.0))
    nr = math.sqrt(max(float(rest_sq.item()), 0.0))
    dot = float(rest_torque_dot.item())
    frac_before = nt / max(nf, eps)
    cosine = (
        max(-1.0, min(1.0, dot / (nr * nt)))
        if nr > 0.0 and nt > 0.0 else 0.0
    )

    scale = 1.0
    if frac_before > cap and nt > 0.0:
        # Squaring alpha*||t|| <= cap*||r + alpha*t|| gives
        #
        #   (1-cap^2)||t||^2 alpha^2
        #       - 2 cap^2 <r,t> alpha - cap^2||r||^2 <= 0.
        #
        # Its nonnegative root bounds the feasible interval.  Use the
        # rationalized form when dot < 0 to avoid cancellation near
        # anti-aligned gradients.
        c2 = cap * cap
        one_minus_c2 = 1.0 - c2
        norm_ratio = nr / nt
        sqrt_factor = math.sqrt(
            max(one_minus_c2 + c2 * cosine * cosine, 0.0)
        )
        if cosine >= 0.0:
            root = norm_ratio * (
                c2 * cosine + cap * sqrt_factor
            ) / one_minus_c2
        elif nr > 0.0:
            root = (
                norm_ratio * cap / (sqrt_factor - cap * cosine)
            )
        else:
            root = 0.0

        # When ||r + alpha*t|| < eps, the denominator is eps rather than
        # the vector norm.  This can extend the feasible interval up to
        # alpha = cap*eps/||t|| (notably when the remainder is zero).
        eps_root = cap * eps / nt
        eps_torque_norm = cap * eps
        eps_total_norm = math.hypot(
            nr + eps_torque_norm * cosine,
            eps_torque_norm * math.sqrt(
                max(1.0 - cosine * cosine, 0.0)
            ),
        )
        if eps_total_norm <= eps:
            root = max(root, eps_root)
        scale = max(0.0, min(1.0, root))

    if scale < 1.0:
        rebuilt: list[torch.Tensor | None] = []
        new_total_sq = 0.0
        new_torque_sq = 0.0
        for p, gf, gt in zip(active, g_full, grads_t):
            if gf is None and gt is None:
                rebuilt.append(None)
                continue
            if gt is None:
                new_grad = gf
                scaled_torque = None
            elif gf is None:
                torque64 = gt.detach().double()
                scaled_torque = (scale * torque64).to(dtype=gt.dtype)
                new_grad = (-torque64 + scale * torque64).to(dtype=p.dtype)
            else:
                torque64 = gt.detach().double()
                scaled_torque = (scale * torque64).to(dtype=gt.dtype)
                new_grad = (
                    gf.double() - torque64 + scale * torque64
                ).to(dtype=p.dtype)
            assert new_grad is not None
            if not bool(torch.isfinite(new_grad).all().item()):
                return {
                    "torque_grad_cap": cap,
                    "torque_grad_status_nonfinite_reconstruction": 1.0,
                }
            rebuilt.append(new_grad)
            new_total_sq += float(new_grad.double().square().sum().item())
            if scaled_torque is not None:
                new_torque_sq += float(
                    scaled_torque.double().square().sum().item()
                )
        for p, new_grad in zip(active, rebuilt):
            p.grad = new_grad
        nf_after = math.sqrt(new_total_sq)
        nt_after = math.sqrt(new_torque_sq)
        frac_after = nt_after / max(nf_after, eps)
    else:
        nf_after = nf
        nt_after = nt
        frac_after = frac_before

    return {
        "torque_grad_cap": cap,
        "torque_grad_frac_before": frac_before,
        "torque_grad_frac_after": frac_after,
        "torque_grad_full_norm_before": nf,
        "torque_grad_full_norm_after": nf_after,
        "torque_grad_torque_norm_before": nt,
        "torque_grad_torque_norm_after": nt_after,
        "torque_grad_rest_norm": nr,
        "torque_grad_rest_torque_cosine": cosine,
        "torque_grad_cap_scale": scale,
    }
