#!/usr/bin/env python
"""Score a phase-1 checkpoint on the section-2 eval grid.

    python scripts/eval_phase1.py --ckpt runs/phase1_<ts>/best.pt --device cuda:1

Runs three planners over the SAME seeded grid so the numbers are paired:

* ``quintic``          -- the raw quintic + waist-aim seed (what phase 1 imitates);
* ``model_quintic``    -- the transformer, one refinement iteration from that seed;
* ``model_hold``       -- the transformer from a zero-motion hold init.

The recorded trajopt baseline in ``baselines/`` is the other reference line;
it is scored on the same grid but on its own easier problem (single arm, waist
frozen for the idle side) -- see ``baselines/trajopt_hpipm_2026-08-10.md``.
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--grid", default="standard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--out", default=None, help="directory for CSV / summary (default: ckpt dir)")
    p.add_argument("--tag", default="phase1")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--planners", nargs="+",
                   default=["quintic", "model_quintic", "model_hold"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")

    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(model_cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple(ck.get("scenario_cfg", {}).get("buckets", (25, 50, 100)))
    print(f"[ckpt] {ckpt_path} step={ck['step']} buckets={buckets}", flush=True)

    out_dir = Path(args.out) if args.out else ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer = A.TrajectoryScorer(device="cpu")
    cases = A.generate_grid(seed=args.seed, grid=args.grid, scorer=scorer)
    print(f"[grid] {args.grid}: {len(cases)} cases", flush=True)

    amp = None if args.no_amp else torch.bfloat16
    library = A.GoalPoseLibrary()
    kit = A.DrumKit.from_layout()
    planners = {
        "quintic": lambda: A.quintic_case_planner(library=library),
        "model_quintic": lambda: A.ModelPlanner(
            model=model, device=str(dev), init="quintic", iterations=args.iterations,
            buckets=buckets, num_drums=model_cfg.num_drums,
            num_future=model_cfg.num_future, library=library, kit=kit, amp_dtype=amp,
        ),
        "model_hold": lambda: A.ModelPlanner(
            model=model, device=str(dev), init="hold", iterations=args.iterations,
            buckets=buckets, num_drums=model_cfg.num_drums,
            num_future=model_cfg.num_future, library=library, kit=kit, amp_dtype=amp,
        ),
    }

    summary: dict = {"ckpt": str(ckpt_path), "step": int(ck["step"]),
                     "grid": args.grid, "seed": args.seed, "n_cases": len(cases),
                     "iterations": args.iterations, "planners": {}}
    for name in args.planners:
        if name not in planners:
            raise SystemExit(f"unknown planner {name!r}")
        plan = planners[name]()
        t0 = time.perf_counter()
        rows, trajs = A.run_eval(plan, cases, scorer=scorer)
        wall = time.perf_counter() - t0
        txt = A.format_summary(rows, title=f"{args.tag} :: {name}")
        print(txt, flush=True)
        prefix = out_dir / f"{args.tag}_{name}"
        A.write_csv(prefix.with_name(prefix.name + "_per_case.csv"), rows)
        prefix.with_name(prefix.name + "_summary.txt").write_text(txt)
        agg = A.aggregate(rows)
        agg["wall_s"] = wall
        agg["ms_per_case"] = 1000.0 * wall / max(len(cases), 1)
        summary["planners"][name] = agg
        np.savez_compressed(
            prefix.with_name(prefix.name + "_trajectories.npz"),
            **{str(k): v for k, v in trajs.items()},
        )

    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))

    # --- compact paired table ------------------------------------------------
    keys = [
        ("contact_pos_err_mm", "p50"), ("contact_speed_err_mps", "p50"),
        ("coll_pen_max_m", "p90"), ("tau_excess_max_nm", "p90"),
        ("travel_l1_rad", "p50"), ("jerk_integral", "p50"), ("solve_s", "p50"),
    ]
    names = [n for n in args.planners if n in summary["planners"]]
    print("\n===== paired comparison =====")
    head = f"{'metric':32s}" + "".join(f"{n:>18s}" for n in names)
    print(head)
    print("-" * len(head))
    for attr, stat in keys:
        label = summary["planners"][names[0]]["metrics"][attr]["label"]
        row = f"{label + ' ' + stat:32s}"
        for n in names:
            row += f"{summary['planners'][n]['metrics'][attr][stat]:18.4f}"
        print(row)
    for flag in ("ok_hit", "ok_speed", "ok_collision", "ok_torque", "ok_velocity",
                 "ok_joint_limits", "elev_ok", "accepted"):
        row = f"{flag:32s}"
        for n in names:
            c = summary["planners"][n]["counts"][flag]
            row += f"{c['count']:>12d}/{c['n']:<5d}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
