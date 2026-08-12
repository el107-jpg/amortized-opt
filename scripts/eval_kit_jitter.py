#!/usr/bin/env python
"""Section 8 / C(i) -- generalization to UNSEEN kit randomizations.

Every open-loop number in the project is on the nominal drum kit.  Training
randomizes the kit (7b prerequisite fix B: the cost's drum head centres come
from the same context tokens the model is told about, so scene and context move
together by construction), but the acceptance grid never did.

This re-runs the seed-7 (test) grid under ``n_draws`` fresh
:class:`~amortized_opt.rhythm.KitRandomization` draws, moving **everything**
together:

* the layout the grid's cases are BUILT from (aim points, start poses, hit
  disks) -- a temporary YAML written from the perturbed positions, handed to
  ``generate_grid(layout_path=...)``;
* the layout the SCORER prices collisions and hit-disk membership against
  (``TrajectoryScorer(layout_path=...)``);
* the ``DrumKit`` the model's context tokens are built from
  (``batched_model_plans(kit=...)``).

Any one of the three left at nominal would measure a scene/context mismatch
instead of generalization.  Draw 0 is the NOMINAL kit, so the deltas are paired.

    python scripts/eval_kit_jitter.py --device cuda:1 --grid-seed 7 --draws 5
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.rhythm import KitRandomization  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def write_jittered_layout(kit, src: Path, dst: Path) -> Path:
    """A copy of the layout YAML with ``kit``'s (env-frame) positions."""
    layout = yaml.safe_load(src.read_text())
    for name, spec in layout["drums"].items():
        spec["position"] = [float(x) for x in kit.positions[kit.index(str(name))]]
    dst.write_text(yaml.safe_dump(layout, sort_keys=False))
    return dst


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    return model, {"buckets": buckets, "num_drums": mcfg.num_drums,
                   "num_future": mcfg.num_future}, int(ck["step"])


def summarize(rows):
    agg = A.aggregate(rows)
    sub = [r for r in rows if r.N <= 24]
    return {
        "n": agg["n"],
        "accepted": agg["counts"]["accepted"]["count"],
        "ok_hit": agg["counts"]["ok_hit"]["count"],
        "ok_speed": agg["counts"]["ok_speed"]["count"],
        "ok_collision": agg["counts"]["ok_collision"]["count"],
        "ok_torque": agg["counts"]["ok_torque"]["count"],
        "ok_joint_limits": agg["counts"]["ok_joint_limits"]["count"],
        "contact_pos_err_mm_p50": agg["metrics"]["contact_pos_err_mm"]["p50"],
        "speed_err_p50": agg["metrics"]["contact_speed_err_mps"]["p50"],
        "short_n": len(sub), "short_accepted": sum(1 for r in sub if r.accepted),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--grid-seed", type=int, default=7)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--jitter-seed", type=int, default=4242)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="runs/final/kit_jitter.json")
    args = ap.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16

    model, cfg, step = load_model(ROOT / args.ckpt, dev)
    library = A.GoalPoseLibrary()
    src = Path(A.default_layout_path())
    nominal = A.DrumKit.from_layout()
    rng = np.random.default_rng(args.jitter_seed)
    krand = KitRandomization()

    report = {"ckpt": str(args.ckpt), "step": step, "grid_seed": args.grid_seed,
              "jitter": {"per_drum_xy": krand.per_drum_xy,
                         "global_xy": krand.global_xy, "global_z": krand.global_z},
              "draws": []}

    tmpd = Path(tempfile.mkdtemp(prefix="kitjitter_"))
    for d in range(args.draws + 1):
        kit = nominal if d == 0 else nominal.randomized(rng, krand)
        shift = float(np.abs(kit.positions - nominal.positions).max())
        path = (src if d == 0
                else write_jittered_layout(kit, src, tmpd / f"layout_{d}.yaml"))
        scorer = A.TrajectoryScorer(device="cpu", layout_path=path)
        entry = {"draw": d, "max_shift_mm": 1000 * shift,
                 "rms_shift_mm": 1000 * float(np.sqrt(
                     ((kit.positions - nominal.positions) ** 2).sum(1).mean())),
                 "grids": {}}
        for grid in ("standard", "near_drum"):
            cases = A.generate_grid(seed=args.grid_seed, grid=grid, scorer=scorer,
                                    layout_path=path)
            plans, wall = A.batched_model_plans(
                model, cases, device=str(dev), kit=kit, library=library,
                buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
                num_future=cfg["num_future"], iterations=1, init="quintic",
                amp_dtype=amp,
            )
            rows, _ = A.run_eval(A.precomputed_planner(plans, wall / max(len(cases), 1)),
                                 cases, scorer=scorer)
            entry["grids"][grid] = summarize(rows)
        tot = sum(v["accepted"] for v in entry["grids"].values())
        n = sum(v["n"] for v in entry["grids"].values())
        entry["accepted_total"] = tot
        entry["n_total"] = n
        print(f"[draw {d}] shift max {entry['max_shift_mm']:.0f} mm rms "
              f"{entry['rms_shift_mm']:.0f} mm -> accepted {tot}/{n}  "
              f"std {entry['grids']['standard']['accepted']}/162  "
              f"near {entry['grids']['near_drum']['accepted']}/81  "
              f"short {entry['grids']['standard']['short_accepted']}/54", flush=True)
        report["draws"].append(entry)

    base = report["draws"][0]["accepted_total"]
    jit = [e["accepted_total"] for e in report["draws"][1:]]
    report["summary"] = {
        "nominal_accepted": base,
        "jittered_mean": float(np.mean(jit)) if jit else float("nan"),
        "jittered_min": int(min(jit)) if jit else -1,
        "jittered_max": int(max(jit)) if jit else -1,
        "delta_mean": float(np.mean(jit) - base) if jit else float("nan"),
    }
    print(f"\nnominal {base}  jittered mean {report['summary']['jittered_mean']:.1f} "
          f"[{report['summary']['jittered_min']}, {report['summary']['jittered_max']}] "
          f"delta {report['summary']['delta_mean']:+.1f}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
