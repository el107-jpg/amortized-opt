#!/usr/bin/env python
"""2D sweep: reference lead time × PID integral gain.

Tests combinations of scalar ``cmd_lead_s`` (seconds, applied uniformly to all
joints) and ``ki`` (integral gain) against the closed-loop probe.  The
champion's per-joint ``"fit"`` lead is included as a reference row.

Usage::

    python amortized/scripts/eval_pid_lead_sweep.py

Output: a markdown table to stdout + a JSON file in ``runs/pid_sweep/``.
"""

from __future__ import annotations

import argparse
import datetime
import json
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
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    step = int(ck.get("step", 0))
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    return model, step, buckets


def run_one(
    model, *, kit, scorer, clips, device,
    ki: float = 0.0, ki_windup: float = 0.0,
    cmd_lead_s=0.0, lead_s: float = 0.0,
    pid_with_lead: bool = False,
    seed: int = 0, horizon: int = 50,
) -> dict:
    """One closed-loop probe configuration."""
    t0 = time.perf_counter()
    out = closed_loop_probe(
        model,
        kit=kit, scorer=scorer, clips=clips, device=device,
        num_drums=8, num_future=3, horizon=horizon, seed=seed,
        iterations=1,
        cmd_lead_s=cmd_lead_s,
        lead_s=lead_s,
        ki=ki,
        ki_windup=ki_windup,
        pid_with_lead=pid_with_lead,
        freeze_wp=0, freeze_waist=True, sticky_goal_point=True,
    )
    out["wall_s"] = time.perf_counter() - t0
    out["ki"] = ki
    out["cmd_lead_s"] = str(cmd_lead_s)
    return out


def fmt_row(label, out):
    lat = out["latency"]
    ep = out["episodes"]
    pd_p50 = float(np.median([e["pd_track_p50_mrad"] for e in ep])) if ep else float("nan")
    return {
        "config": label,
        "crossed": lat["crossed_in_disk"],
        "miss%": f"{lat['miss_rate']*100:.1f}",
        "|lat|p50": f"{lat['lat_abs_p50_ms']:.1f}",
        "|lat|p90": f"{lat['lat_abs_p90_ms']:.1f}",
        "≤10ms": lat["within_10ms"],
        "≤20ms": lat["within_20ms"],
        "≤40ms": lat["within_40ms"],
        "spd_ok": lat.get("speed_in_band", "?"),
        "trk_p50": f"{pd_p50:.1f}",
        "div": out.get("diverged", "?"),
        "wall_s": f"{out.get('wall_s', 0):.1f}",
    }


# --------------------------------------------------------------------- main --
def main():
    parser = argparse.ArgumentParser(description="2D sweep: lead time × ki.")
    parser.add_argument("--ckpt", default=None, help="Model checkpoint.")
    parser.add_argument("--device", default="cuda:3", help="Torch device.")
    parser.add_argument("--seed", type=int, default=0, help="Start-sample seed.")
    parser.add_argument("--clips", type=int, default=0, nargs="+",
                      help="Eval clip indices (default: first clip only).")
    parser.add_argument("--horizon", type=int, default=50, help="MPC horizon.")
    parser.add_argument("--output_dir", default=None, help="Output directory.")
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
    model, step, buckets = load_model(ckpt, dev)
    print(f"[ckpt] {ckpt} step={step} buckets={buckets}")
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # --- clips -----------------------------------------------------------
    kit = DrumKit.from_layout()
    clips = load_rcc_pool(A.default_rcc_dir("eval"), kit)
    clip_idx = args.clips or [0]
    eval_clips = [clips[i] for i in clip_idx if i < len(clips)]
    print(f"[clips] {[c.name for c in eval_clips]}")

    scorer = TrajectoryScorer(device="cpu")

    # --- sweep grid ------------------------------------------------------
    # Scalar lead times (seconds, uniform across all joints) and ki values.
    # CONTROL_DT = 0.02 s, so lead_s = 0.01 = half a waypoint.
    lead_times = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]
    ki_values = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    # Baselines first
    print("\n--- champion lead=fit, ki=0 ---")
    bl_fit = run_one(model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
                     cmd_lead_s=LEAD_PRESETS["fit"], ki=0.0, seed=args.seed,
                     horizon=args.horizon)

    print("\n--- lead=off, ki=0 (pure PD) ---")
    bl_pd = run_one(model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
                    cmd_lead_s=0.0, ki=0.0, seed=args.seed, horizon=args.horizon)

    rows = []
    # 2D grid: PID with lead (pid_with_lead=True)
    for lead_s in lead_times:
        for ki in ki_values:
            if lead_s == 0.0 and ki == 0.0:
                continue  # already covered by bl_pd
            pid_with_lead = (lead_s > 0.0 and ki > 0.0)
            tag = f"lead={lead_s*1e3:.1f}ms, ki={ki:.1f}"
            if pid_with_lead:
                tag += " (+pid_with_lead)"
            print(f"\n--- {tag} ---")
            out = run_one(
                model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
                cmd_lead_s=float(lead_s), ki=ki, ki_windup=0.0,
                pid_with_lead=pid_with_lead,
                seed=args.seed, horizon=args.horizon,
            )
            rows.append(fmt_row(tag, out))
            rows[-1]["lead_s"] = lead_s
            rows[-1]["ki"] = ki

    # Also test per-joint "fit" lead with small ki (pid_with_lead=True)
    for ki in [1.0, 2.0, 5.0, 10.0]:
        tag = f"lead=fit, ki={ki:.1f} (+pid_with_lead)"
        print(f"\n--- {tag} ---")
        out = run_one(
            model, kit=kit, scorer=scorer, clips=eval_clips, device=dev,
            cmd_lead_s=LEAD_PRESETS["fit"], ki=ki, ki_windup=0.0,
            pid_with_lead=True,
            seed=args.seed, horizon=args.horizon,
        )
        rows.append(fmt_row(tag, out))
        rows[-1]["lead_s"] = "fit"
        rows[-1]["ki"] = ki

    # --- table -----------------------------------------------------------
    all_rows = [fmt_row("champion lead=fit, ki=0", bl_fit),
                fmt_row("lead=off, ki=0 (pure PD)", bl_pd)] + rows

    header = ["config", "crossed", "miss%", "|lat|p50", "|lat|p90",
              "≤10ms", "≤20ms", "≤40ms", "spd_ok", "trk_p50", "div", "wall_s"]
    col_w = {h: len(h) for h in header}
    key_map = {
        "config": "config", "crossed": "crossed", "miss%": "miss%",
        "|lat|p50": "|lat|p50", "|lat|p90": "|lat|p90",
        "≤10ms": "≤10ms", "≤20ms": "≤20ms", "≤40ms": "≤40ms",
        "spd_ok": "spd_ok", "trk_p50": "trk_p50", "div": "div", "wall_s": "wall_s",
    }
    for r in all_rows:
        for h in header:
            col_w[h] = max(col_w[h], len(str(r.get(h, ""))))
    fmt = " | ".join(f"{{:<{col_w[h]}}}" for h in header)
    print("\n" + fmt.format(*header))
    print(" | ".join("-" * col_w[h] for h in header))
    for r in all_rows:
        vals = [str(r.get(h, ""))[:col_w[h]] for h in header]
        print(fmt.format(*vals))

    # --- save JSON -------------------------------------------------------
    out_dir = Path(args.output_dir or (
        Path(__file__).resolve().parent.parent / "runs" / "pid_sweep"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"pid_lead_sweep_{stamp}.json"

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

    payload = {
        "ckpt": str(ckpt),
        "clips": [c.name for c in eval_clips],
        "seed": args.seed, "horizon": args.horizon,
        "lead_times": lead_times, "ki_values": ki_values,
        "baseline_fit": sanitize(bl_fit),
        "baseline_pd": sanitize(bl_pd),
        "rows": sanitize(rows),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
