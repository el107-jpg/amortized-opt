#!/usr/bin/env python
"""A/B a phase-1 APEX checkpoint against the straight-line ``phase1_main`` one.

    python scripts/eval_phase1_apex.py --device cuda:1 \
        --ckpt straight=runs/phase1_main/best.pt apex=runs/phase1_apex/best.pt \
        --grid standard --out runs/phase1_apex/eval_ab

Runs the SAME seeded grid through:

* ``teacher_quintic``  -- the straight-line quintic + waist-aim seed;
* ``teacher_apex``     -- the apex teacher (start -> apex -> terminal);
* for every checkpoint, the model from each requested init
  (``quintic`` = the trained regime for BOTH models, ``apex``, ``hold``).

Every planner is scored by the section-2 :class:`TrajectoryScorer`, so the
numbers line up cell for cell with ``runs/phase1_main/eval/``.
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
from amortized_opt.warmstart_apex import (  # noqa: E402
    ApexConfig,
    ApexModelPlanner,
    ApexSolver,
    apex_case_planner,
)

HEADLINE = [
    ("contact_pos_err_mm", "p50"),
    ("contact_speed_err_mps", "p50"),
    ("contact_dir_cos", "p50"),
    ("approach_deg", "p50"),
    ("coll_pen_max_m", "p50"),
    ("coll_pen_max_m", "p90"),
    ("coll_pen_max_all_m", "p90"),
    ("coll_min_sdf_m", "p50"),
    ("tau_excess_max_nm", "p90"),
    ("travel_l1_rad", "p50"),
    ("jerk_integral", "p50"),
    ("solve_s", "p50"),
]
FLAGS = ("ok_hit", "ok_speed", "ok_collision", "ok_collision_margin", "ok_torque",
         "ok_velocity", "ok_joint_limits", "elev_ok", "accepted")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--grid", default="standard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--inits", nargs="+", default=["quintic", "apex", "hold"])
    p.add_argument("--teachers", nargs="+", default=["quintic", "apex"])
    p.add_argument("--out", default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out) if args.out else Path("runs/phase1_apex/eval_ab")
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer = A.TrajectoryScorer(device="cpu")
    cases = A.generate_grid(seed=args.seed, grid=args.grid, scorer=scorer)
    print(f"[grid] {args.grid}: {len(cases)} cases", flush=True)

    amp = None if args.no_amp else torch.bfloat16
    library = A.GoalPoseLibrary()
    kit = A.DrumKit.from_layout()
    acfg = ApexConfig()
    solver = ApexSolver(cfg=acfg, dtype=torch.float64)

    planners: dict[str, object] = {}
    if "quintic" in args.teachers:
        planners["teacher_quintic"] = A.quintic_case_planner(library=library)
    if "apex" in args.teachers:
        planners["teacher_apex"] = apex_case_planner(
            library=library, apex_cfg=acfg, solver=solver
        )

    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        model_cfg = A.ModelConfig(**ck["model_cfg"])
        model = A.build_model(model_cfg, dev)
        model.load_state_dict(ck["model"])
        model.eval()
        buckets = tuple(ck.get("scenario_cfg", {}).get("buckets", (25, 50, 100)))
        print(f"[ckpt] {label}: {path} step={ck['step']}", flush=True)
        for init in args.inits:
            planners[f"{label}::{init}"] = ApexModelPlanner(
                model=model, device=str(dev), iterations=args.iterations, init=init,
                buckets=buckets, num_drums=model_cfg.num_drums,
                num_future=model_cfg.num_future, library=library, kit=kit,
                apex_cfg=acfg, solver=solver, amp_dtype=amp,
            )

    summary: dict = {"grid": args.grid, "seed": args.seed, "n_cases": len(cases),
                     "iterations": args.iterations, "planners": {}}
    for name, plan in planners.items():
        t0 = time.perf_counter()
        rows, trajs = A.run_eval(plan, cases, scorer=scorer)
        wall = time.perf_counter() - t0
        tag = name.replace("::", "_")
        txt = A.format_summary(rows, title=f"{args.grid} :: {name}")
        (out_dir / f"{args.grid}_{tag}_summary.txt").write_text(txt)
        A.write_csv(out_dir / f"{args.grid}_{tag}_per_case.csv", rows)
        np.savez_compressed(out_dir / f"{args.grid}_{tag}_trajectories.npz",
                            **{str(k): v for k, v in trajs.items()})
        agg = A.aggregate(rows)
        agg["wall_s"] = wall
        summary["planners"][name] = agg
        print(f"[eval] {name}: accepted {agg['counts']['accepted']['count']}/{agg['n']} "
              f"({wall:.0f} s)", flush=True)

    (out_dir / f"{args.grid}_ab_summary.json").write_text(json.dumps(summary, indent=2))

    names = list(summary["planners"])
    w = max(18, max(len(n) for n in names) + 2)
    head = f"{'metric':34s}" + "".join(f"{n:>{w}s}" for n in names)
    lines = ["", f"===== {args.grid}: A/B table =====", head, "-" * len(head)]
    for attr, stat in HEADLINE:
        label = summary["planners"][names[0]]["metrics"][attr]["label"]
        row = f"{label + ' ' + stat:34s}"
        for n in names:
            row += f"{summary['planners'][n]['metrics'][attr][stat]:{w}.4f}"
        lines.append(row)
    lines.append("-" * len(head))
    for flag in FLAGS:
        row = f"{flag:34s}"
        for n in names:
            c = summary["planners"][n]["counts"][flag]
            row += f"{str(c['count']) + '/' + str(c['n']):>{w}s}"
        lines.append(row)
    txt = "\n".join(lines)
    print(txt)
    (out_dir / f"{args.grid}_ab_table.txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
