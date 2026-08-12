#!/usr/bin/env python
"""Sweep PID integral gains over the closed-loop probe and report results.

The champion model is run through ``closed_loop_probe`` with a range of ``ki``
values.  Because the integral fights the reference lead (the lead intentionally
shifts the command ahead, so the plant's "error" against the led command is by
design), every ``ki > 0`` run sets ``cmd_lead_s = 0``.  The baseline (ki = 0)
is reported with the lead both ON (the champion's deployment config) and OFF,
so we can see what the lead alone buys and what the integral adds on top of a
leadless baseline.

Usage::

    python amortized/scripts/eval_pid_sweep.py

Output: a markdown table to stdout + a JSON file in ``runs/pid_sweep/``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.lead import LEAD_PRESETS  # noqa: E402
from amortized_opt.probe import closed_loop_probe  # noqa: E402
from amortized_opt.eval_harness import TrajectoryScorer  # noqa: E402
from amortized_opt.rhythm import DrumKit, load_rcc_pool  # noqa: E402


# ------------------------------------------------------------------ helpers --
def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    step = int(ck.get("step", 0))
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    return model, step, buckets, mcfg


def run_one(
    model, *, kit, scorer, clips, device, buckets,
    ki: float = 0.0, ki_windup: float = 0.0,
    cmd_lead_s=0.0, lead_s: float = 0.0,
    seed: int = 0, horizon: int = 50,
) -> dict:
    """One closed-loop probe configuration."""
    t0 = time.perf_counter()
    out = closed_loop_probe(
        model,
        kit=kit, scorer=scorer, clips=clips, device=device,
        num_drums=8,
        num_future=3,
        horizon=horizon,
        seed=seed,
        iterations=1,
        cmd_lead_s=cmd_lead_s,
        lead_s=lead_s,
        ki=ki,
        ki_windup=ki_windup,
        freeze_wp=0,
        freeze_waist=True,
        sticky_goal_point=True,
    )
    out["wall_s"] = time.perf_counter() - t0
    out["ki"] = ki
    out["ki_windup"] = ki_windup
    out["cmd_lead_s"] = str(cmd_lead_s)
    return out


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten one level of nesting for the table."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and not k.startswith("_"):
            for sk, sv in v.items():
                out[f"{prefix}{k}.{sk}"] = sv
        elif not k.startswith("_"):
            out[f"{prefix}{k}"] = v
    return out


# --------------------------------------------------------------------- main --
def main():
    parser = argparse.ArgumentParser(description="PID gain sweep for the amortized MPC.")
    parser.add_argument("--ckpt", default=None, help="Model checkpoint (default: champion).")
    parser.add_argument("--device", default="cuda:1", help="Torch device.")
    parser.add_argument("--seed", type=int, default=0, help="Start-sample seed.")
    parser.add_argument("--clips", type=int, default=0, nargs="+",
                      help="Eval clip indices (default: first 3).")
    parser.add_argument("--horizon", type=int, default=50, help="MPC horizon in waypoints.")
    parser.add_argument("--steps", type=int, default=250, help="Max steps per episode.")
    parser.add_argument("--output_dir", default=None,
                      help="Output directory (default: runs/pid_sweep/).")
    args = parser.parse_args()

    # --- device ----------------------------------------------------------
    dev = torch.device(args.device)
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")

    # --- checkpoint ------------------------------------------------------
    ckpt = (
        Path(args.ckpt)
        if args.ckpt
        else Path(__file__).resolve().parent.parent / "runs" / "phase2b_main" / "best.pt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model, step, buckets, mcfg = load_model(ckpt, dev)
    print(f"[ckpt] {ckpt} step={step} buckets={buckets}")
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # --- clips -----------------------------------------------------------
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(A.default_rcc_dir("eval"), kit)
    clip_idx = args.clips or list(range(min(3, len(clips))))
    eval_clips = [clips[i] for i in clip_idx]
    print(f"[clips] {[c.name for c in eval_clips]}")

    # --- scorer -----------------------------------------------------------
    scorer = TrajectoryScorer(device="cpu")

    # --- sweep -----------------------------------------------------------
    # ki values: 0 (baseline PD), then powers of ~sqrt(10) from 0.1 to 100.
    # The kp values are ~40 (arms) to ~200 (waist_yaw), so ki values in the
    # 1–50 range are the physically meaningful band.  We also include "lead
    # only" (the champion's config) as the reference.
    ki_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    windup_values = [0.0, 0.5, 1.0]  # rad·s of anti-windup clamping

    # Baseline: champion config (lead ON, ki=0)
    print("\n--- baseline: champion lead=fit, ki=0 ---")
    baseline_lead = run_one(
        model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
        buckets=buckets, ki=0.0, cmd_lead_s=LEAD_PRESETS["fit"],
        seed=args.seed, horizon=args.horizon, lead_s=0.0,
    )

    # Baseline: lead OFF, ki=0 (pure PD, no compensation)
    print("\n--- baseline: lead=off, ki=0 (pure PD) ---")
    baseline_pd = run_one(
        model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
        buckets=buckets, ki=0.0, cmd_lead_s=0.0,
        seed=args.seed, horizon=args.horizon, lead_s=0.0,
    )

    rows = []

    # PID sweeps (all with lead OFF)
    first_windup = True
    for wu in windup_values:
        for ki in ki_values:
            if ki == 0.0 and not first_windup:
                continue  # only one ki=0 per windup
            cfg_label = f"lead=off, ki={ki:.1f}"
            if wu > 0:
                cfg_label += f", windup={wu:.1f}"
            print(f"\n--- {cfg_label} ---")
            out = run_one(
                model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
                buckets=buckets, ki=ki, ki_windup=wu, cmd_lead_s=0.0,
                seed=args.seed, horizon=args.horizon, lead_s=0.0,
            )
            lat = out["latency"]
            ep = out["episodes"]
            pd_p50 = float(np.median([e["pd_track_p50_mrad"] for e in ep])) if ep else float("nan")
            rows.append({
                "config": cfg_label,
                "ki": ki,
                "windup": wu,
                "crossed_in_disk": lat["crossed_in_disk"],
                "miss_rate": f"{lat['miss_rate']*100:.1f}%",
                "|lat|_p50_ms": f"{lat['lat_abs_p50_ms']:.1f}",
                "|lat|_p90_ms": f"{lat['lat_abs_p90_ms']:.1f}",
                "within_10ms": lat["within_10ms"],
                "within_20ms": lat["within_20ms"],
                "within_40ms": lat["within_40ms"],
                "speed_in_band": lat.get("speed_in_band", "?"),
                "pd_track_p50_mrad": f"{pd_p50:.1f}",
                "diverged": out.get("diverged", "?"),
                "wall_s": f"{out.get('wall_s', 0):.1f}",
            })

    # --- table -----------------------------------------------------------
    # Add baselines at the top
    def bl_row(label, d):
        lat = d["latency"]
        ep = d["episodes"]
        pd_p50 = float(np.median([e["pd_track_p50_mrad"] for e in ep])) if ep else float("nan")
        return {
            "config": label,
            "ki": 0,
            "windup": 0,
            "crossed_in_disk": lat["crossed_in_disk"],
            "miss_rate": f"{lat['miss_rate']*100:.1f}%",
            "|lat|_p50_ms": f"{lat['lat_abs_p50_ms']:.1f}",
            "|lat|_p90_ms": f"{lat['lat_abs_p90_ms']:.1f}",
            "within_10ms": lat["within_10ms"],
            "within_20ms": lat["within_20ms"],
            "within_40ms": lat["within_40ms"],
            "speed_in_band": lat.get("speed_in_band", "?"),
            "pd_track_p50_mrad": f"{pd_p50:.1f}",
            "diverged": d.get("diverged", "?"),
            "wall_s": f"{d.get('wall_s', 0):.1f}",
        }

    all_rows = [
        bl_row("champion lead=fit, ki=0", baseline_lead),
        bl_row("lead=off, ki=0 (pure PD)", baseline_pd),
    ] + rows

    # Print markdown
    key_map = {
        "config": "config",
        "crossed_in_disk": "crossed",
        "miss_rate": "miss%",
        "|lat|_p50_ms": "|lat|p50",
        "|lat|_p90_ms": "|lat|p90",
        "within_10ms": "≤10ms",
        "within_20ms": "≤20ms",
        "within_40ms": "≤40ms",
        "speed_in_band": "spd_ok",
        "pd_track_p50_mrad": "trk_p50",
        "diverged": "div",
        "wall_s": "wall_s",
    }
    header = list(key_map.values())
    col_w = {h: len(h) for h in header}
    for r in all_rows:
        for h, k in zip(header, key_map):
            col_w[h] = max(col_w[h], len(str(r.get(k, ""))))
    fmt = " | ".join(f"{{:<{col_w[h]}}}" for h in header)
    print("\n" + fmt.format(*header))
    print(" | ".join("-" * col_w[h] for h in header))
    for r in all_rows:
        vals = []
        for h, k in zip(header, key_map):
            v = str(r.get(k, ""))
            vals.append(v[: col_w[h]])
        print(fmt.format(*vals))

    # --- save JSON -------------------------------------------------------
    out_dir = Path(args.output_dir or (
        Path(__file__).resolve().parent.parent / "runs" / "pid_sweep"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"pid_sweep_{stamp}.json"
    payload = {
        "ckpt": str(ckpt),
        "clips": [c.name for c in eval_clips],
        "seed": args.seed,
        "horizon": args.horizon,
        "baseline_lead": baseline_lead,
        "baseline_pd": baseline_pd,
        "rows": rows,
    }
    # Convert non-serializable values
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()
                    if not str(k).startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (torch.Tensor,)):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)
    with open(out_path, "w") as f:
        json.dump(sanitize(payload), f, indent=2, default=str)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
