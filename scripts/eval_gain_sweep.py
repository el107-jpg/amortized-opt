#!/usr/bin/env python
"""Closed-loop ablation over the servo gains, with the PID integral OFF.

    python scripts/eval_gain_sweep.py --device cuda:2 --clips 0 1

``ki`` is 0 in every row here by construction: the PID chapter's sweep found the
integral worth ~0 on top of a per-joint lead, and the offline plant has no torque
saturation to wind up against, so the integral's measured benefit does not
transfer.  What this script varies instead is the PLANT: ``kp_scale`` /
``kd_scale`` multiply the runtime stiffness / damping tables.

The point is to separate two explanations of the closed-loop miss:

* **delay**  -- the servo is behind the plan and a lead fixes it;
* **bandwidth** -- the servo cannot reach the commanded acceleration at all,
  in which case only more stiffness (or a plan that asks for less) helps.

Each gain configuration is run at three lead settings so the two effects
separate: ``off`` (no compensation), ``fit`` (the fixed per-joint preset --
NOTE it was fitted at kp_scale=1 and is therefore mis-specified elsewhere, which
is itself the measurement), and, if ``--schedule`` is given, the horizon-
dependent schedule from ``fit_lead_schedule.py``.
"""

from __future__ import annotations

import argparse
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


def load_model(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, int(ck.get("step", 0))


def run_one(model, *, kit, scorer, clips, device, cmd_lead_s, kp_scale, kd_scale,
            seed, horizon, steps):
    t0 = time.perf_counter()
    out = closed_loop_probe(
        model, kit=kit, scorer=scorer, clips=clips, device=device,
        num_drums=8, num_future=3, horizon=horizon, steps=steps, seed=seed,
        iterations=1, cmd_lead_s=cmd_lead_s, lead_s=0.0,
        ki=0.0, ki_windup=0.0, pid_with_lead=False,
        kp_scale=kp_scale, kd_scale=kd_scale,
        freeze_wp=0, freeze_waist=True, sticky_goal_point=True,
    )
    out["wall_s"] = time.perf_counter() - t0
    return out


def row(label, out):
    lat = out["latency"]
    ep = out["episodes"]
    trk = float(np.median([e["pd_track_p50_mrad"] for e in ep])) if ep else float("nan")
    return {
        "config": label,
        "crossed": lat["crossed_in_disk"],
        "n": lat.get("n_strikes", lat.get("n", "?")),
        "miss%": round(lat["miss_rate"] * 100, 1),
        "lat_p50": round(lat["lat_abs_p50_ms"], 1),
        "lat_p90": round(lat["lat_abs_p90_ms"], 1),
        "w10": lat["within_10ms"],
        "w20": lat["within_20ms"],
        "spd_ok": lat.get("speed_in_band", "?"),
        "trk_p50": round(trk, 1),
        "div": out.get("diverged", 0),
        "wall_s": round(out.get("wall_s", 0.0), 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--clips", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="repeat every row at these start-sample seeds; the probe "
                        "moves by +-7 crossings on the start draw alone, so a "
                        "single seed cannot rank two similar configs")
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--kp-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--kd-scales", type=float, nargs="+", default=[1.0])
    p.add_argument("--leads", nargs="+", default=["off", "fit"],
                   help="'off', a preset name, a scalar in seconds, or 'schedule'")
    p.add_argument("--schedule", default=None,
                   help="lead_schedule.json from fit_lead_schedule.py")
    p.add_argument("--out", default="runs/pid_sweep/gain_sweep.json")
    args = p.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")

    model, step = load_model(Path(args.ckpt), dev)
    kit = DrumKit.from_layout()
    clips_all = load_rcc_pool(A.default_rcc_dir("eval"), kit)
    clips = [clips_all[i] for i in args.clips if i < len(clips_all)]
    scorer = TrajectoryScorer(device="cpu")
    print(f"[ckpt] {args.ckpt} step={step}")
    print(f"[clips] {[c.name for c in clips]}  ki=0 in every row")

    sched = None
    if args.schedule:
        sched = json.loads(Path(args.schedule).read_text())["schedule_s"]

    def resolve(tag):
        if tag == "off":
            return 0.0
        if tag == "schedule":
            if sched is None:
                raise SystemExit("--schedule is required for the 'schedule' lead")
            return {"schedule": sched}
        if tag in LEAD_PRESETS:
            return LEAD_PRESETS[tag]
        return float(tag)

    seeds = args.seeds or [args.seed]
    rows = []
    for kp in args.kp_scales:
        for kd in args.kd_scales:
            for tag in args.leads:
                for sd in seeds:
                    label = f"kp x{kp:g}  kd x{kd:g}  lead={tag}  seed={sd}"
                    print(f"\n--- {label} ---", flush=True)
                    out = run_one(model, kit=kit, scorer=scorer, clips=clips,
                                  device=dev, cmd_lead_s=resolve(tag), kp_scale=kp,
                                  kd_scale=kd, seed=sd, horizon=args.horizon,
                                  steps=args.steps)
                    r = row(label, out)
                    r.update(kp_scale=kp, kd_scale=kd, lead=tag, seed=sd)
                    rows.append(r)
                    print("    " + "  ".join(f"{k}={v}" for k, v in r.items()
                                             if k not in ("config",)), flush=True)

    cols = ["config", "crossed", "n", "miss%", "lat_p50", "lat_p90", "w10", "w20",
            "spd_ok", "trk_p50", "div"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    if len(seeds) > 1:
        print("\n=== pooled over seeds ===")
        print("| config | crossed/n | miss% | lat_p50 (median) | lat_p90 (median) |")
        print("|---|---|---|---|---|")
        keys = sorted({(r["kp_scale"], r["kd_scale"], r["lead"]) for r in rows})
        for k in keys:
            sel = [r for r in rows if (r["kp_scale"], r["kd_scale"], r["lead"]) == k]
            c = sum(r["crossed"] for r in sel)
            tot = sum(int(r["n"]) for r in sel)
            print(f"| kp x{k[0]:g} kd x{k[1]:g} lead={k[2]} | {c}/{tot} | "
                  f"{100 * (1 - c / max(tot, 1)):.1f} | "
                  f"{np.median([r['lat_p50'] for r in sel]):.1f} | "
                  f"{np.median([r['lat_p90'] for r in sel]):.1f} |")

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"ckpt": args.ckpt, "clips": args.clips,
                               "seed": args.seed, "rows": rows}, indent=2))
    print(f"\n[out] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
