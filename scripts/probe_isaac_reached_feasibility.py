#!/usr/bin/env python3
"""Ask the existing DAgger trajopt expert to solve Isaac-reached miss states.

Input is ``play_isaac.py --hit_diagnostics`` JSON.  For each selected missed
beat, take the actual Isaac ``(q, qdot)`` snapshot nearest a requested lead
time and relabel the *exact scheduled strike point and velocity*.  This is a
feasibility probe only: it does not add rows to actor training.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--cohort", choices=("strong_bad", "broad_bad", "all_misses", "plausible"),
        default="strong_bad",
    )
    p.add_argument("--lead-s", type=float, default=0.35)
    p.add_argument("--max-iters", type=int, default=80)
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--retry-nonfeasible-from", type=Path, default=None,
        help="Restrict to cases marked non-feasible in a previous probe JSON.",
    )
    p.add_argument(
        "--g1-root", type=Path,
        default=Path("/data2/eric/g1-drummer-lab-rythmic"),
    )
    p.add_argument(
        "--layout", type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "drum_kit_layout.yaml",
    )
    return p.parse_args()


def selected(event: dict, cohort: str) -> bool:
    if event.get("outcome", {}).get("credited", False):
        return False
    p = event["proposed"]
    axial = abs(float(p["axial_mm"]))
    lateral = float(p["lateral_beyond_mm"])
    strong_bad = axial > 80.0 or lateral > 20.0
    broad_bad = axial > 40.0 or lateral > 0.0
    if cohort == "strong_bad":
        return strong_bad
    if cohort == "broad_bad":
        return broad_bad
    if cohort == "plausible":
        return not broad_bad
    return True


def main() -> int:
    args = parse_args()
    control = args.g1_root / "source" / "g1_drummer" / "g1_drummer" / "control"
    if str(control) not in sys.path:
        sys.path.insert(0, str(control))

    import imitation.dagger as dg
    from imitation.dagger import DaggerConfig, DaggerRunner, _drum_by_name, _drum_normal_env
    from imitation.rollout import RolloutSeed
    from teacher.pipeline import _strike_goal_for_drum
    from teacher.sampler import drum_id_map

    payload = json.loads(args.diagnostics.read_text())
    events = [e for e in payload["events"] if selected(e, args.cohort)]
    if args.retry_nonfeasible_from is not None:
        prior = json.loads(args.retry_nonfeasible_from.read_text())
        retry_times = {
            round(float(c["scheduled_time_s"]), 6)
            for c in prior["cases"] if c["status"] != "feasible"
        }
        events = [
            e for e in events
            if round(float(e["scheduled_time_s"]), 6) in retry_times
        ]
    if not events:
        raise SystemExit(f"no events selected for cohort {args.cohort!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.out.parent / f".{args.out.stem}_work"
    cfg = DaggerConfig(
        data_dir=work_dir / "_nodata",
        layout=args.layout,
        out_dir=work_dir,
        seed=int(args.seed),
        max_iters=int(args.max_iters),
        min_relabel_horizon_s=float(args.lead_s),
        jobs=int(args.jobs),
        rrt_iteration_bounded=True,
        waist_mode="frozen",
        seed_use_tube=False,
        tube_radius=0.0,
        obstacle_n_samples=0,
    )
    runner = DaggerRunner(cfg)
    ids = drum_id_map(runner.drums)
    rng = np.random.default_rng(int(args.seed))

    work = []
    meta = []
    for i, event in enumerate(events):
        hand = str(event["hand"])
        drum = _drum_by_name(runner.drums, str(event["drum"]))
        if drum is None:
            continue
        approach = event.get("approach", [])
        usable = [r for r in approach if "q" in r and "qd" in r]
        if not usable:
            continue
        target_ticks = float(args.lead_s) / 0.02
        snap = min(usable, key=lambda r: abs(float(r["ticks_to_hit"]) - target_ticks))
        horizon = float(snap["ticks_to_hit"]) * 0.02
        template = _strike_goal_for_drum(
            drum, hand, rng, horizon, horizon, 0.4, 1.2,
        )
        q = np.asarray(snap["q"], dtype=np.float64)
        qd = np.asarray(snap["qd"], dtype=np.float64)
        goal_pos = np.asarray(snap["goal_pos_env"], dtype=np.float64)
        goal_vel = np.asarray(snap["goal_velocity_env"], dtype=np.float64)
        seed = RolloutSeed(
            hand=hand,
            q=q,
            qd=qd,
            goal_pos_env=goal_pos,
            goal_vel_env=goal_vel,
            horizon_s=horizon,
            drum_id=int(ids[str(event["drum"])]),
            hit_radius=float(template.hit_radius) if template.hit_radius is not None else 0.08,
            goal_drum=str(event["drum"]),
            half_plane_dir=template.half_plane_dir,
            disk_center_offset=template.disk_center_offset,
            Q_ref=None,
            tube_radius=0.0,
            drum_normal_env=_drum_normal_env(drum),
            dt=cfg.dt,
        )
        shim = SimpleNamespace(
            q_last_ok=q,
            qd_last_ok=qd,
            t_remain_last_ok=horizon,
            q_fail=q,
            qd_fail=qd,
            message="isaac_mpc_miss",
        )
        work.append((i, seed, shim))
        meta.append({
            "scheduled_time_s": event["scheduled_time_s"],
            "hand": hand,
            "drum": event["drum"],
            "horizon_s": horizon,
            "proposed": event["proposed"],
            "tracking_rms_mrad": event["tracking_rms_mrad"],
        })

    jobs = dg._resolve_jobs(int(args.jobs), len(work))
    print(
        f"[feasibility] cohort={args.cohort} states={len(work)} "
        f"lead={args.lead_s:.2f}s jobs={jobs}", flush=True,
    )
    t0 = time.perf_counter()
    if jobs <= 1:
        results = [runner._relabel_failure(s, r, index=i) for i, s, r in work]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=jobs, initializer=dg._seed_worker_init, initargs=(cfg,),
        ) as pool:
            results = list(pool.map(dg._relabel_worker, work))

    counts = Counter()
    cases = []
    for info, (rows, _entry, avoid) in zip(meta, results):
        if rows:
            status = "feasible"
        elif avoid is not None:
            status = "infeasible"
        else:
            status = "timeout"
        counts[status] += 1
        cases.append({**info, "status": status, "teacher_rows": len(rows)})
    elapsed = time.perf_counter() - t0
    total = len(cases)
    summary = {
        "diagnostics": str(args.diagnostics),
        "cohort": args.cohort,
        "lead_s_requested": float(args.lead_s),
        "states": total,
        "feasible": counts["feasible"],
        "infeasible": counts["infeasible"],
        "timeout": counts["timeout"],
        "feasible_fraction": counts["feasible"] / max(total, 1),
        "elapsed_s": elapsed,
        "seconds_per_state": elapsed / max(total, 1),
        "by_drum": {},
    }
    for drum in sorted({c["drum"] for c in cases}):
        rows = [c for c in cases if c["drum"] == drum]
        summary["by_drum"][drum] = dict(Counter(c["status"] for c in rows))
    args.out.write_text(json.dumps({"summary": summary, "cases": cases}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[feasibility] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
