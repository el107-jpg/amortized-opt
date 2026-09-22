#!/usr/bin/env python
"""Measure same-row derivatives and finite-step cost profiles for curvature VJPs.

This probes the exact FusedCost trajectory gradient at a saved actor checkpoint.
It does not execute AdamW, SaLSa, or the actor's auxiliary training losses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A
from probe_row_grad_stratify import _load_al, _prepare_batch, _scenario_from_ckpt
from train_phase2 import _load_npz_buckets
from amortized_opt.phase2_data import Phase2Noise, Phase2Pool
from amortized_opt.train_cost import (
    _dls_rows,
    _index_batch_weights,
    _strike_rows,
    build_costs,
    curvature_precondition_vjps,
    make_encode_step,
    phase2_config_from_checkpoint,
)


GROUPS = {
    "terminal": ("terminal", "terminal_aim"),
    "torque": ("torque", "torque_static", "effort"),
    "collision": ("collision",),
    "joint_box": ("joint_box",),
    "velocity": ("velocity",),
    "smooth": ("smooth",),
}


def _mean_float(t: torch.Tensor) -> float:
    return float(t.detach().float().mean())


def _cost_snapshot(terms) -> dict[str, torch.Tensor]:
    return {
        "total": terms.total.detach(),
        "terminal": (
            terms.terms["terminal"] + terms.terms["terminal_aim"]
        ).detach(),
        "torque": (
            terms.terms["torque"] + terms.terms["torque_static"]
            + terms.terms["effort"]
        ).detach(),
        "collision": terms.terms["collision"].detach(),
        "joint_box": terms.terms["joint_box"].detach(),
        "contact_mm": 1000.0 * terms.diagnostics["contact_pos_err"].detach(),
    }


def _summarize_profile(
    base: dict[str, torch.Tensor],
    steps: list[float],
    snapshots: list[dict[str, torch.Tensor]],
) -> dict:
    total_delta = torch.stack([
        snap["total"] - base["total"] for snap in snapshots
    ], dim=0)
    best_idx = total_delta.argmin(dim=0)
    improves = total_delta < -1e-6
    radii = total_delta.new_tensor(steps).unsqueeze(1).expand_as(total_delta)
    largest = torch.where(improves, radii, 0.0).amax(dim=0)
    best_step = radii.gather(0, best_idx.unsqueeze(0)).squeeze(0)
    out = {
        "largest_improving_step_median": float(largest.median()),
        "largest_improving_step_mean": _mean_float(largest),
        "best_step_median": float(best_step.median()),
        "best_total_delta_mean": _mean_float(total_delta.gather(
            0, best_idx.unsqueeze(0)).squeeze(0),
        ),
        "by_step": [],
    }
    for step, snap, delta in zip(steps, snapshots, total_delta):
        out["by_step"].append({
            "step": step,
            "improved_frac": _mean_float((delta < -1e-6).float()),
            **{
                f"{key}_delta_mean": _mean_float(snap[key] - base[key])
                for key in base
            },
        })
    for key in ("terminal", "torque", "collision", "joint_box", "contact_mm"):
        deltas = torch.stack([
            snap[key] - base[key] for snap in snapshots
        ], dim=0)
        out[f"{key}_delta_at_best_mean"] = _mean_float(
            deltas.gather(0, best_idx.unsqueeze(0)).squeeze(0)
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pool-cache", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bucket", type=int, default=None)
    p.add_argument("--curvature-frac", type=float, default=0.25)
    p.add_argument("--curvature-mix", type=float, default=0.25)
    p.add_argument("--parameter-probe", action="store_true",
                   help="also measure the pre-optimizer actor-gradient pullback")
    p.add_argument("--line-profile", choices=("none", "trajectory", "parameter", "both"),
                   default="none", help="evaluate finite equal-distance steps")
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    torch.set_num_threads(4)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = phase2_config_from_checkpoint(ck, device=str(dev), compile_model=False)
    cfg.curvature_direct_frac = args.curvature_frac
    cfg.curvature_direct_mix = args.curvature_mix
    buckets, stats = _load_npz_buckets(str(args.pool_cache))
    pool = Phase2Pool(
        buckets, cfg=_scenario_from_ckpt(ck), noise=Phase2Noise(),
        device=str(dev), stats=stats,
    )
    bucket = int(args.bucket or max(pool.buckets))
    model = A.build_model(A.model_config_from_checkpoint(ck), dev)
    model.load_state_dict(ck["model"])
    model.train()
    costs = build_costs([bucket], cfg.cost, device=str(dev), dtype=torch.float32)
    cost = costs[bucket]
    al = _load_al(cfg, ck)
    gen = torch.Generator(device=str(dev)).manual_seed(args.seed + 7919)
    batch, enc, strike, lengths, weights, meta, _, _ = _prepare_batch(
        model, make_encode_step(model, compile_mode=None), pool, None, costs,
        cfg, al, bucket, int(ck.get("step") or 0), gen,
        np.random.default_rng(args.seed + 31), use_replay=False, base=cost,
        diagnostic_batch_size=args.batch_size,
    )
    dec = model(batch["init"], batch["state"], enc, batch["masks"])
    traj = dec.traj
    terms = cost.forward(
        batch["q0"], traj, batch["v0"], strike,
        lengths=lengths, weights=weights,
    )
    # Sum, rather than mean, preserves each row's local gradient magnitude.
    g = torch.autograd.grad(terms.total.sum(), traj, retain_graph=True)[0]
    rows = _dls_rows(args.batch_size, args.curvature_frac, dev)
    selected = g.index_select(0, rows)
    curved, stats = curvature_precondition_vjps(
        selected.detach().unsqueeze(-1),
        traj.detach().index_select(0, rows),
        batch["q0"].detach().index_select(0, rows),
        _strike_rows(strike, rows),
        lengths.index_select(0, rows),
        _index_batch_weights(weights, rows, args.batch_size),
        cost, cfg,
    )
    curved = curved[..., 0]
    # Compare unit-L2 descent directions so a smaller blended norm cannot
    # masquerade as a worse direction.
    raw_norm = selected.flatten(1).norm(dim=1).clamp_min(1e-20)
    curved_norm = curved.flatten(1).norm(dim=1).clamp_min(1e-20)
    raw_dir = -selected / raw_norm[:, None, None]
    curved_dir = -curved / curved_norm[:, None, None]
    result = {
        "checkpoint": str(args.ckpt), "step": int(ck.get("step") or 0),
        "seed": args.seed, "batch_size": args.batch_size,
        "curvature_frac": args.curvature_frac,
        "curvature_mix": args.curvature_mix,
        "bucket": bucket, "selected_rows": rows.tolist(),
        "scope": "fresh actor rows; FusedCost only; per-row unit-L2 trajectory directions",
        "direction_cosine": _mean_float((raw_dir * curved_dir).sum(dim=(1, 2))),
        "blended_norm_ratio": _mean_float(curved_norm / raw_norm),
        "fallback_frac": float(stats["curvature_direct_fallback_frac"]),
        "terms": {},
    }
    for name, keys in GROUPS.items():
        value = sum(terms.terms[k] for k in keys)
        grad = (
            torch.autograd.grad(value.sum(), traj, retain_graph=True)[0]
            if value.requires_grad else torch.zeros_like(traj)
        )
        sg = grad.index_select(0, rows)
        raw_change = (sg * raw_dir).sum(dim=(1, 2))
        curved_change = (sg * curved_dir).sum(dim=(1, 2))
        result["terms"][name] = {
            "value_mean": _mean_float(value.index_select(0, rows)),
            "raw_directional_change_mean": _mean_float(raw_change),
            "curved_directional_change_mean": _mean_float(curved_change),
            "curvature_worse_frac": _mean_float((curved_change > raw_change).float()),
            "raw_increase_frac": _mean_float((raw_change > 0).float()),
            "curved_increase_frac": _mean_float((curved_change > 0).float()),
        }
    # All remaining terms make the grouped sum exactly equal FusedCost.total.
    grouped = {key for keys in GROUPS.values() for key in keys}
    other = sum(v for k, v in terms.terms.items() if k not in grouped)
    og = (
        torch.autograd.grad(other.sum(), traj, retain_graph=True)[0]
        if other.requires_grad else torch.zeros_like(traj)
    ).index_select(0, rows)
    for name, grad in (("other", og), ("total", selected)):
        result["terms"][name] = {
            "raw_directional_change_mean": _mean_float((grad * raw_dir).sum(dim=(1, 2))),
            "curved_directional_change_mean": _mean_float((grad * curved_dir).sum(dim=(1, 2))),
        }
    if args.parameter_probe or args.line_profile in ("parameter", "both"):
        params = [p for p in model.parameters() if p.requires_grad]
        raw_pg = torch.autograd.grad(
            terms.total.mean(), params, retain_graph=True, allow_unused=True,
        )
        delta = torch.zeros_like(traj)
        delta.index_copy_(0, rows, curved - selected)
        correction_pg = torch.autograd.grad(
            traj, params, grad_outputs=delta / args.batch_size,
            retain_graph=True, allow_unused=True,
        )
        raw_pg = [torch.zeros_like(p) if g is None else g for p, g in zip(params, raw_pg)]
        curved_pg = [
            g + (torch.zeros_like(p) if d is None else d)
            for p, g, d in zip(params, raw_pg, correction_pg)
        ]
        pnorm_raw = sum((g * g).sum() for g in raw_pg).sqrt()
        pnorm_curved = sum((g * g).sum() for g in curved_pg).sqrt()
        pdot = sum((g * c).sum() for g, c in zip(raw_pg, curved_pg))
        pr = {
            "scope": "FusedCost actor gradient before gradient clipping, AdamW, or SaLSa",
            "cosine": float((pdot / (pnorm_raw * pnorm_curved)).detach()),
            "norm_ratio": float((pnorm_curved / pnorm_raw).detach()),
            "terms": {},
        }
        for name, keys in GROUPS.items():
            value = sum(terms.terms[k] for k in keys)
            if not value.requires_grad:
                continue
            tg = torch.autograd.grad(
                value.mean(), params, retain_graph=True, allow_unused=True,
            )
            raw_change = sum(
                (t * g).sum() for t, g in zip(tg, raw_pg) if t is not None
            )
            curved_change = sum(
                (t * g).sum() for t, g in zip(tg, curved_pg) if t is not None
            )
            pr["terms"][name] = {
                "raw_directional_change": float((-raw_change / pnorm_raw).detach()),
                "curved_directional_change": float((-curved_change / pnorm_curved).detach()),
            }
        result["parameter_space"] = pr
    if args.line_profile in ("trajectory", "both"):
        with torch.no_grad():
            q0_sel = batch["q0"].index_select(0, rows)
            v0_sel = batch["v0"].index_select(0, rows)
            strike_sel = _strike_rows(strike, rows)
            lengths_sel = lengths.index_select(0, rows)
            weights_sel = _index_batch_weights(weights, rows, args.batch_size)
            traj_sel = traj.detach().index_select(0, rows)
            base = _cost_snapshot(cost.forward(
                q0_sel, traj_sel, v0_sel, strike_sel,
                lengths=lengths_sel, weights=weights_sel,
            ))
            rms_steps = [0.0, 0.000001, 0.000002, 0.000005,
                         0.00001, 0.00002, 0.00005, 0.0001,
                         0.0002, 0.0005, 0.001, 0.002, 0.005]
            scale = float(traj_sel.shape[1] * traj_sel.shape[2]) ** 0.5
            prof = {}
            for name, direction in (("ordinary", raw_dir), ("curvature", curved_dir)):
                snaps = []
                for rms in rms_steps:
                    cand = traj_sel + (rms * scale) * direction
                    snaps.append(_cost_snapshot(cost.forward(
                        q0_sel, cand, v0_sel, strike_sel,
                        lengths=lengths_sel, weights=weights_sel,
                    )))
                prof[name] = _summarize_profile(base, rms_steps, snaps)
            result["trajectory_line_profile"] = {
                "scope": "FusedCost on selected rows; same per-row L2 step; step is RMS joint radians across H*D",
                "n_rows": int(rows.numel()),
                "profiles": prof,
            }
    if args.line_profile in ("parameter", "both"):
        # This measures a raw-gradient actor parameter step, not AdamW/SaLSa.
        # Re-encode at each trial because the encoder parameters also move.
        with torch.no_grad():
            base = _cost_snapshot(terms)
            saved = [p.detach().clone() for p in params]
            dirs = {
                "ordinary": [-g / pnorm_raw for g in raw_pg],
                "curvature": [-g / pnorm_curved for g in curved_pg],
            }
            steps = [0.0, 0.00001, 0.00002, 0.00005, 0.0001,
                     0.0002, 0.0005, 0.001, 0.002, 0.004,
                     0.008, 0.016]
            profiles = {}
            try:
                for name, direction in dirs.items():
                    snaps = []
                    for step in steps:
                        for p, p0, d in zip(params, saved, direction):
                            p.copy_(p0 + step * d)
                        trial_enc = model.encode_context(batch["context"])
                        trial = model(batch["init"], batch["state"],
                                      trial_enc, batch["masks"])
                        snaps.append(_cost_snapshot(cost.forward(
                            batch["q0"], trial.traj, batch["v0"], strike,
                            lengths=lengths, weights=weights,
                        )))
                    profiles[name] = _summarize_profile(base, steps, snaps)
            finally:
                for p, p0 in zip(params, saved):
                    p.copy_(p0)
            result["parameter_line_profile"] = {
                "scope": "FusedCost; same global actor-parameter L2 step; raw pre-optimizer gradient directions, not AdamW/SaLSa",
                "n_rows": args.batch_size,
                "profiles": profiles,
            }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
