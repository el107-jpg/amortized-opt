#!/usr/bin/env python
"""Section 8 / C(iii) -- initialization robustness, consolidated.

The decoder is a refinement operator: it is handed an initialization and one
forward pass produces the plan.  Every deployment condition supplies a
DIFFERENT initialization, so "how much does the answer depend on the warm
start?" is a generalization question, not a curiosity:

* ``quintic``  -- the straight-line warm start the eval grid uses;
* ``hold``     -- a zero-motion hold at the start pose (the worst legal init,
                  and the one phase 1 collapsed from);
* ``quintic + OU noise`` at several sigmas -- a warm start that is wrong by a
  known amount, the phase-1 training channel;
* the closed loop's own **warm-MPC init** (previous plan shifted one waypoint)
  is measured by ``eval_section8.py``, not here -- it needs the plant.

Reported on the seed-7 TEST grid, standard + near_drum, one refinement
iteration (and 3, for the iteration question).

    python scripts/eval_init_robust.py --device cuda:1 --grid-seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.warmstart import InitNoise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    return model, {"buckets": buckets, "num_drums": mcfg.num_drums,
                   "num_future": mcfg.num_future}, int(ck["step"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--grid-seed", type=int, default=7)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.05, 0.15, 0.30])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="runs/final/init_robust.json")
    args = ap.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16

    model, cfg, step = load_model(ROOT / args.ckpt, dev)
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    scorer = A.TrajectoryScorer(device="cpu")

    variants = [("quintic_x1", "quintic", None, 1),
                ("quintic_x3", "quintic", None, 3),
                ("hold_x1", "hold", None, 1),
                ("hold_x3", "hold", None, 3)]
    for s in args.sigmas:
        variants.append((f"quintic_ou{s:g}_x1", "quintic",
                         InitNoise(sigma_corr=float(s), sigma_white=0.01), 1))

    report = {"ckpt": str(args.ckpt), "step": step, "grid_seed": args.grid_seed,
              "variants": {}}
    for label, init, noise, iters in variants:
        tot = {"accepted": 0, "n": 0, "ok_hit": 0, "ok_collision": 0,
               "ok_torque": 0, "short_accepted": 0, "short_n": 0}
        pos = []
        for grid in ("standard", "near_drum"):
            cases = A.generate_grid(seed=args.grid_seed, grid=grid, scorer=scorer)
            plans, wall = A.batched_model_plans(
                model, cases, device=str(dev), kit=kit, library=library,
                buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
                num_future=cfg["num_future"], iterations=iters, init=init,
                noise=noise, seed=args.grid_seed, amp_dtype=amp,
            )
            rows, _ = A.run_eval(
                A.precomputed_planner(plans, wall / max(len(cases), 1)),
                cases, scorer=scorer)
            agg = A.aggregate(rows)
            tot["n"] += agg["n"]
            for k in ("accepted", "ok_hit", "ok_collision", "ok_torque"):
                tot[k] += agg["counts"][k]["count"]
            sub = [r for r in rows if r.N <= 24]
            tot["short_n"] += len(sub)
            tot["short_accepted"] += sum(1 for r in sub if r.accepted)
            pos.append(agg["metrics"]["contact_pos_err_mm"]["p50"])
        tot["contact_pos_err_mm_p50"] = float(sum(pos) / len(pos))
        report["variants"][label] = tot
        print(f"{label:22} accepted {tot['accepted']:3d}/{tot['n']}  "
              f"hit {tot['ok_hit']:3d}  coll {tot['ok_collision']:3d}  "
              f"tau {tot['ok_torque']:3d}  short {tot['short_accepted']}/{tot['short_n']}  "
              f"pos {tot['contact_pos_err_mm_p50']:.1f} mm", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
