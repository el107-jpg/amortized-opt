#!/usr/bin/env python
"""One-shot collision breakdown for a trained checkpoint.

Loads a checkpoint, runs it on the eval grid through ``FusedCost``, and
reports collision sub-types (drum body / under-drum / self) and time-bucketed
per-hand drum collision (far / mid / near / exempt / no-hit).

    python scripts/analyze_collision.py --ckpt runs/phase2d_waist/best.pt --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.constants import N_DOF  # noqa: E402
from amortized_opt.cost import FusedCost, StrikeContext  # noqa: E402
from amortized_opt.train_cost import phase2_cost_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pack_cases(cases, plans, max_horizon: int, dtype=torch.float32):
    """Pack case plans into ``(q0, waypoints, v0)`` tensors for FusedCost."""
    B = len(cases)
    q0 = torch.zeros(B, N_DOF, dtype=dtype)
    v0 = torch.zeros(B, N_DOF, dtype=dtype)
    wp = torch.zeros(B, max_horizon, N_DOF, dtype=dtype)
    for i, c in enumerate(cases):
        Q = plans[c.case_id]
        q0[i] = torch.as_tensor(Q[0], dtype=dtype)
        v0[i] = torch.as_tensor(c.qdot_start, dtype=dtype)
        N = min(c.N, max_horizon)
        wp[i, :N] = torch.as_tensor(Q[1 : N + 1], dtype=dtype)
        wp[i, N:] = torch.as_tensor(Q[N], dtype=dtype)
    return q0, wp, v0


def analyze_ckpt(ckpt_path: str, device: str = "cuda:1", grid_seed: int = 0):
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")

    # ---- load checkpoint ----
    print(f"[load] {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple(
        (ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100))
    )
    cfg = {"buckets": buckets, "num_drums": mcfg.num_drums, "num_future": mcfg.num_future}
    step = ck.get("step", "?")
    print(f"[ckpt] step={step} buckets={buckets}")

    # ---- build cost ----
    cost_cfg = phase2_cost_config()
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    drum_index = {name: i for i, name in enumerate(kit.names)}
    max_h = max(buckets)
    cost = FusedCost(cost_cfg, max_horizon=max_h, device=dev, dtype=torch.float32)

    # ---- run eval grid ----
    amp = torch.bfloat16  # match training
    dtype = torch.float32

    all_diag: dict[str, list[float]] = {}

    for grid in ("standard", "near_drum"):
        cases = A.generate_grid(seed=grid_seed, grid=grid, scorer=A.TrajectoryScorer(device="cpu"))
        if not cases:
            continue
        print(f"\n[grid] {grid}: {len(cases)} cases")

        plans_map, wall = A.batched_model_plans(
            model, cases, device=str(dev), kit=kit, library=library,
            buckets=buckets, num_drums=mcfg.num_drums,
            num_future=mcfg.num_future, iterations=1, init="quintic",
            amp_dtype=amp,
        )

        # Group by horizon for batched cost computation
        by_n: dict[int, list] = {}
        for c in cases:
            by_n.setdefault(c.N, []).append(c)

        for N, group in sorted(by_n.items()):
            if any(c.case_id not in plans_map for c in group):
                continue
            ctx = A.strike_context_from_cases(group, drum_index=drum_index, device=dev, dtype=dtype)
            q0, wp, v0 = pack_cases(group, plans_map, max_h, dtype=dtype)
            q0, wp, v0 = q0.to(dev), wp.to(dev), v0.to(dev)

            with torch.no_grad():
                ct = cost.forward(q0, wp, v0, ctx)

            for k, v in ct.diagnostics.items():
                vals = v.detach().cpu().numpy().tolist()
                if isinstance(vals, float):
                    vals = [vals]
                all_diag.setdefault(k, []).extend(vals)

    # ---- report ----
    report_collision_breakdown(all_diag)
    return all_diag


def report_collision_breakdown(all_diag: dict[str, list[float]]):
    print("\n" + "=" * 70)
    print("COLLISION BREAKDOWN")
    print("=" * 70)

    # Sub-type breakdown
    keys_sub = ["collision_drum", "collision_under", "collision_self"]
    if all(k in all_diag for k in keys_sub):
        print("\n--- Per-type collision (raw hinge, mean over batch) ---")
        for k in keys_sub:
            vals = all_diag[k]
            a = np.array(vals)
            print(f"  {k:25s}  mean={a.mean():.6f}  p50={np.median(a):.6f}  "
                  f"p90={np.percentile(a, 90):.6f}  max={a.max():.6f}  "
                  f"zero={float((a < 1e-8).mean()):.1%}")

    # Compute relative contribution
    if all(k in all_diag for k in keys_sub):
        drum = np.array(all_diag["collision_drum"])
        under = np.array(all_diag["collision_under"])
        sel = np.array(all_diag["collision_self"])
        total = drum + under + sel
        nonzero = total > 1e-10
        if nonzero.any():
            print(f"\n  Relative contribution (nonzero cases only, n={nonzero.sum()}):")
            print(f"    drum body:  {float(drum[nonzero].mean() / total[nonzero].mean()):.1%}")
            print(f"    under-drum: {float(under[nonzero].mean() / total[nonzero].mean()):.1%}")
            print(f"    self:       {float(sel[nonzero].mean() / total[nonzero].mean()):.1%}")

    # Time-bucketed (per hand)
    buckets = ["far", "mid", "near", "exempt", "nohit"]
    for hand in ("left", "right"):
        keys_t = [f"t_col_{b}_{hand}" for b in buckets]
        if not all(k in all_diag for k in keys_t):
            continue
        print(f"\n--- Time-bucketed drum collision: {hand} hand ---")
        print(f"    {'bucket':>8s}  {'mean':>10s}  {'p50':>10s}  {'p90':>10s}  {'nonzero':>8s}")
        for b, k in zip(buckets, keys_t):
            vals = np.array(all_diag[k])
            nz = float((vals > 1e-10).mean())
            print(f"    {b:>8s}  {vals.mean():10.6f}  {np.median(vals):10.6f}  "
                  f"{np.percentile(vals, 90):10.6f}  {nz:7.1%}")

    # Also show per-case min SDF and pen max (existing diagnostics)
    for k in ("collision_min_sdf", "collision_pen_max"):
        if k in all_diag:
            vals = np.array(all_diag[k])
            print(f"\n  {k}: mean={vals.mean():.6f} p50={np.median(vals):.6f} "
                  f"p90={np.percentile(vals, 90):.6f}")

    # Summary: what dominates?
    print("\n--- Summary ---")
    for hand in ("left", "right"):
        keys_t = [f"t_col_{b}_{hand}" for b in buckets]
        if all(k in all_diag for k in keys_t):
            means = {b: np.array(all_diag[f"t_col_{b}_{hand}"]).mean() for b in buckets}
            total_t = sum(means.values())
            if total_t > 1e-12:
                print(f"  {hand} hand time profile:", end="")
                for b in buckets:
                    print(f" {b}={means[b]/total_t:.1%}", end="")
                print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to checkpoint .pt file")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--grid-seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    analyze_ckpt(args.ckpt, device=args.device, grid_seed=args.grid_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
