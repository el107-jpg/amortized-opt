#!/usr/bin/env python
"""Final evaluation of a phase-2 checkpoint (section 7).

    python scripts/eval_phase2.py --ckpt runs/phase2_main/best.pt --device cuda:1

Four blocks:

1. **the grids** -- ``standard`` (162) and ``near_drum`` (81), both init modes
   (warm MPC-style quintic seed and a zero-motion hold), 1 and 3 refinement
   iterations, against the quintic planner on the same seeded cases;
2. **the short-horizon cell** -- the T = 0.44-0.45 s block trajopt v2b scores
   26/54 on, reported separately because it is the regime the MPC actually
   lives in;
3. **latency** -- batched-compiled per-tick solve time at batch 1 and at the
   grid batch;
4. **closed loop** -- three RCC clips driven through the FIXED
   ``MPCGenerator`` loop with the trained model as the planner: hit-in-disk
   rate, impact-speed band, collisions, PD tracking error and the
   temporal-consistency drift between successive plans.

The trajopt reference lives in ``baselines/`` and is not re-solved -- its
recorded per-case CSV is read straight off disk.
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
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator, PlanRequest  # noqa: E402
from amortized_opt.strike import shaft_hit_allowed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ============================================================== planners =====
def model_planner_factory(model, cfg, kit, library, dev, *, init, iterations, amp):
    def make(cases):
        plans, wall = A.batched_model_plans(
            model, cases, device=str(dev), kit=kit, library=library,
            buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
            num_future=cfg["num_future"], iterations=iterations, init=init,
            amp_dtype=amp,
        )
        return A.precomputed_planner(plans, wall / max(len(cases), 1)), plans, wall

    return make


def short_cell(rows, n_max: int = 24):
    sub = [r for r in rows if r.N <= n_max]
    return len(sub), sum(1 for r in sub if r.accepted)


def summarize(rows, label: str) -> dict:
    agg = A.aggregate(rows)
    n = max(agg["n"], 1)
    sn, sa = short_cell(rows)
    return {
        "label": label, "n": agg["n"],
        "accepted": agg["counts"]["accepted"]["count"],
        "ok_hit": agg["counts"]["ok_hit"]["count"],
        "ok_speed": agg["counts"]["ok_speed"]["count"],
        "ok_collision": agg["counts"]["ok_collision"]["count"],
        "ok_torque": agg["counts"]["ok_torque"]["count"],
        "ok_velocity": agg["counts"]["ok_velocity"]["count"],
        "ok_joint_limits": agg["counts"]["ok_joint_limits"]["count"],
        "elev_ok": agg["counts"]["elev_ok"]["count"],
        "contact_pos_err_mm_p50": agg["metrics"]["contact_pos_err_mm"]["p50"],
        "speed_err_p50": agg["metrics"]["contact_speed_err_mps"]["p50"],
        "pen_max_mm_p90": agg["metrics"]["coll_pen_max_m"]["p90"],
        "tau_excess_p90": agg["metrics"]["tau_excess_max_nm"]["p90"],
        "travel_l1_p50": agg["metrics"]["travel_l1_rad"]["p50"],
        "jerk_p50": agg["metrics"]["jerk_integral"]["p50"],
        "solve_ms_p50": agg["metrics"]["solve_s"]["p50"],
        "short_n": sn, "short_accepted": sa,
    }


def baseline_rows(prefix: str):
    """Read a recorded baseline CSV back into the summary shape."""
    import csv

    p = ROOT / "baselines" / f"{prefix}_per_case.csv"
    if not p.exists():
        return None
    with p.open() as fh:
        rows = list(csv.DictReader(fh))
    tb = lambda s: str(s).strip().lower() in ("true", "1")  # noqa: E731
    n = len(rows)
    sub = [r for r in rows if int(r["N"]) <= 24]
    f = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    return {
        "label": prefix, "n": n,
        "accepted": sum(tb(r["accepted"]) for r in rows),
        "ok_hit": sum(tb(r["ok_hit"]) for r in rows),
        "ok_speed": sum(tb(r["ok_speed"]) for r in rows),
        "ok_collision": sum(tb(r["ok_collision"]) for r in rows),
        "ok_torque": sum(tb(r["ok_torque"]) for r in rows),
        "ok_velocity": sum(tb(r["ok_velocity"]) for r in rows),
        "ok_joint_limits": sum(tb(r["ok_joint_limits"]) for r in rows),
        "elev_ok": sum(tb(r["elev_ok"]) for r in rows),
        "contact_pos_err_mm_p50": float(np.percentile(f("contact_pos_err_mm"), 50)),
        "speed_err_p50": float(np.percentile(f("contact_speed_err_mps"), 50)),
        "pen_max_mm_p90": float(np.percentile(f("coll_pen_max_m") * 1000, 90)),
        "tau_excess_p90": float(np.percentile(f("tau_excess_max_nm"), 90)),
        "travel_l1_p50": float(np.percentile(f("travel_l1_rad"), 50)),
        "jerk_p50": float(np.percentile(f("jerk_integral"), 50)),
        "solve_ms_p50": float(np.percentile(f("solve_s") * 1000, 50)),
        "short_n": len(sub), "short_accepted": sum(tb(r["accepted"]) for r in sub),
    }


# ============================================================ closed loop ====
def closed_loop(model, cfg, kit, dev, amp, *, clips, n_episodes=3, steps=250, seed=0,
                configs=None):
    """The closed loop, scored per executed strike in the drum's head frame.

    ``configs`` is a list of ``(label, kwargs)`` for
    :func:`~amortized_opt.probe.closed_loop_probe`; the default runs the bare
    loop plus the two deployment knobs section 7c measured (the lag lead and the
    final-approach freeze), because on a position-controlled servo those two
    account for the whole of the closed-loop miss that the kinematic grid cannot
    see.  See ``amortized_opt/probe.py``.
    """
    from amortized_opt.probe import closed_loop_probe

    scorer = A.TrajectoryScorer(device="cpu")
    clips = list(clips)[:n_episodes]
    from amortized_opt.lead import LEAD_PRESETS

    if configs is None:
        configs = [
            # the 7b protocol, for continuity
            ("bare", dict()),
            # THE RECOMMENDED DEPLOYMENT CONFIG: one mechanism, per-joint
            # reference lead, parameters identified offline against the servo
            # model by scripts/fit_cmd_lead.py
            ("lead_fit", dict(cmd_lead_s=LEAD_PRESETS["fit"])),
            # the two optional extras, layered, for the ablation
            ("lead_fit+freeze5+goal20",
             dict(cmd_lead_s=LEAD_PRESETS["fit"], freeze_wp=5, lead_s=0.02)),
        ]
    out = {}
    for label, kw in configs:
        r = closed_loop_probe(
            model, kit=kit, scorer=scorer, clips=clips, device=dev,
            num_drums=cfg["num_drums"], num_future=cfg["num_future"],
            horizon=cfg["buckets"][-1], steps=steps, seed=seed, amp_dtype=amp, **kw,
        )
        r.pop("_hits", None)
        out[label] = r
        L = r["latency"]
        print(f"  [{label}] ticks {r['ticks']} hits {r['hits']} "
              f"in-disk@k {r['in_disk']}  axial p50 {r['axial_p50_mm']:+.1f} mm "
              f"in-plane p50 {r['inplane_p50_mm']:+.1f} mm "
              f"diverged {r['diverged']}\n"
              f"        LATENCY crossed-in-disk {L['crossed_in_disk']}/{L['n']} "
              f"miss {100*L['miss_rate']:.0f}%  p50 {L['lat_p50_ms']:+.1f} ms "
              f"|p50| {L['lat_abs_p50_ms']:.1f} |p90| {L['lat_abs_p90_ms']:.1f} "
              f"<=10/20/40 {L['within_10ms']}/{L['within_20ms']}/{L['within_40ms']} "
              f"early {L['early']} speed-in-band {L['speed_in_band']}", flush=True)
    return out


# ================================================================ latency ====
@torch.no_grad()
def latency(model, cases, cfg, kit, library, dev, amp) -> dict:
    core = model.module if hasattr(model, "module") else model
    core.eval()
    out = {}
    for n, tag in ((1, "batch1"), (len(cases), "grid")):
        sub = cases[:n]
        for _ in range(2):  # warmup
            A.batched_model_plans(core, sub, device=str(dev), kit=kit, library=library,
                                  buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
                                  num_future=cfg["num_future"], amp_dtype=amp)
        if torch.device(dev).type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 20 if n == 1 else 3
        for _ in range(reps):
            A.batched_model_plans(core, sub, device=str(dev), kit=kit, library=library,
                                  buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
                                  num_future=cfg["num_future"], amp_dtype=amp)
        if torch.device(dev).type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        out[tag] = {"cases": n, "total_ms": 1000 * dt, "per_case_ms": 1000 * dt / max(n, 1)}
    return out


# =================================================================== main ====
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--phase1", default="runs/phase1_main/best.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-seed", type=int, default=None,
                   help="seed for generate_grid; defaults to --seed.  seed 0 is "
                        "the VALIDATION grid every phase-2 run selected on; use a "
                        "different one for an untouched test set.")
    p.add_argument("--out", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--skip-closed-loop", action="store_true")
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--closed-loop-steps", type=int, default=250)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple(
        (ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100))
    )
    cfg = {"buckets": buckets, "num_drums": mcfg.num_drums, "num_future": mcfg.num_future}
    print(f"[ckpt] {args.ckpt} step={ck['step']} buckets={buckets}", flush=True)

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "eval_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    scorer = A.TrajectoryScorer(device="cpu")
    grid_seed = args.seed if args.grid_seed is None else int(args.grid_seed)
    report: dict = {"ckpt": str(args.ckpt), "step": int(ck["step"]),
                    "grid_seed": grid_seed, "seed": int(args.seed), "grids": {}}

    phase1 = None
    if args.phase1 and Path(args.phase1).exists():
        c1 = torch.load(args.phase1, map_location="cpu", weights_only=False)
        phase1 = A.build_model(A.model_config_from_checkpoint(c1), dev)
        phase1.load_state_dict(c1["model"])
        phase1.eval()

    for grid in ("standard", "near_drum"):
        cases = A.generate_grid(seed=grid_seed, grid=grid, scorer=scorer)
        print(f"[grid] {grid}: {len(cases)} cases", flush=True)
        block: dict = {}

        rows, _ = A.run_eval(A.quintic_case_planner(library=library), cases, scorer=scorer)
        block["quintic"] = summarize(rows, "quintic")

        for init in ("quintic", "hold"):
            for iters in (1, 3):
                tag = f"model_{init}_x{iters}"
                mk = model_planner_factory(model, cfg, kit, library, dev,
                                           init=init, iterations=iters, amp=amp)
                planner, plans, wall = mk(cases)
                rows, _ = A.run_eval(planner, cases, scorer=scorer)
                block[tag] = summarize(rows, tag)
                block[tag]["batched_solve_s"] = wall
                A.write_csv(out_dir / f"{grid}_{tag}_per_case.csv", rows)
        if phase1 is not None:
            mk = model_planner_factory(phase1, cfg, kit, library, dev,
                                       init="quintic", iterations=1, amp=amp)
            planner, _p, _w = mk(cases)
            rows, _ = A.run_eval(planner, cases, scorer=scorer)
            block["phase1_quintic_x1"] = summarize(rows, "phase1_quintic_x1")

        report["grids"][grid] = block

    for pref in ("trajopt_hpipm_2026-08-10_v2b", "trajopt_hpipm_2026-08-10_v2",
                 "trajopt_hpipm_2026-08-10"):
        b = baseline_rows(pref)
        if b is not None:
            report.setdefault("baselines", {})[pref] = b

    cases = A.generate_grid(seed=grid_seed, grid="standard", scorer=scorer)
    report["latency"] = latency(model, cases, cfg, kit, library, dev, amp)

    if not args.skip_closed_loop:
        clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.clips]
        print(f"[closed loop] {[c.name for c in clips]}", flush=True)
        report["closed_loop"] = closed_loop(
            model, cfg, kit, dev, amp, clips=clips, n_episodes=len(clips),
            steps=args.closed_loop_steps, seed=args.seed,
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- printed tables -----------------------------------------------------
    cols = ["label", "n", "accepted", "ok_hit", "ok_speed", "ok_collision",
            "ok_torque", "ok_velocity", "ok_joint_limits",
            "contact_pos_err_mm_p50", "speed_err_p50", "pen_max_mm_p90",
            "tau_excess_p90", "jerk_p50", "solve_ms_p50", "short_accepted", "short_n"]
    for grid, block in report["grids"].items():
        print(f"\n===== {grid} =====")
        hdr = "  ".join(c[:14].rjust(14) for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for _k, v in block.items():
            print("  ".join(
                (f"{v[c]:14.4g}" if isinstance(v.get(c), float) else str(v.get(c, ""))[:14].rjust(14))
                for c in cols
            ))
        if grid == "standard" and "baselines" in report:
            for k, v in report["baselines"].items():
                print("  ".join(
                    (f"{v[c]:14.4g}" if isinstance(v.get(c), float)
                     else str(v.get(c, ""))[:14].rjust(14)) for c in cols
                ))
    print(f"\n[latency] {json.dumps(report['latency'])}")
    if "closed_loop" in report:
        print("\n===== closed loop =====")
        for label, r in report["closed_loop"].items():
            print(f"{label}: " + json.dumps({k: v for k, v in r.items()
                                             if k != "episodes"}))
            for e in r["episodes"]:
                print("   " + json.dumps(e))
    print(f"\n[out] {out_dir/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
