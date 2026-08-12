#!/usr/bin/env python
"""Markdown tables from a phase-2 run's summary + final-eval report.

    python scripts/report_phase2.py --run runs/phase2_main

Reads ``summary.json`` (the training curve and the rolling evals) and, when it
exists, ``eval_final/report.json`` (the grids, latency and closed loop), and
prints the tables the README chapter wants.  Nothing is recomputed here -- this
is a formatter, so a table in the README can always be traced to a recorded
number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def md(rows, keys, headers=None):
    headers = headers or keys
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/phase2_main")
    a = p.parse_args(argv)
    run = Path(a.run)

    s = json.loads((run / "summary.json").read_text())
    print(f"### training curve — {run}\n")
    print(f"* {s['steps']} steps in {s['wall_s']:.0f} s ({s['steps_per_s']:.2f} it/s), "
          f"stopped: {s['stopped']}, skipped (non-finite) {s.get('skipped_steps', 0)}")
    print(f"* best rolling acceptance {s['best_acceptance']:.4f} @ step {s['best_step']}")
    print(f"* final AL multipliers: "
          + ", ".join(f"{k} {v:.2f}" for k, v in s["lam"].items()))
    print(f"* replay at the end: {s.get('replay')}\n")
    rows = []
    for e in s.get("evals", []):
        rows.append({
            "step": e["step"],
            "accepted": f"{e['accepted']}/{e['n']}",
            "hit": f"{e['ok_hit']:.2f}", "speed": f"{e['ok_speed']:.2f}",
            "coll": f"{e['ok_collision']:.2f}", "torque": f"{e['ok_torque']:.2f}",
            "box": f"{e['ok_joint_limits']:.2f}",
            "elbow": f"{e.get('elbow_down', float('nan')):.2f}",
            "pos p50 mm": f"{e['contact_pos_err_mm_p50']:.1f}",
            "speed err p50": f"{e['speed_err_p50']:.3f}",
            "tau exc p90": f"{e['tau_excess_p90']:.1f}",
            "jerk p50": f"{e['jerk_p50']:.2e}",
            "short": f"{e['short_accepted']}/{e['short_n']}",
            "stage A": "yes" if e.get("stage_a") else "no",
        })
    print(md(rows, list(rows[0]) if rows else []))

    rep_p = run / "eval_final" / "report.json"
    if not rep_p.exists():
        return 0
    rep = json.loads(rep_p.read_text())
    keys = ["label", "accepted", "ok_hit", "ok_speed", "ok_collision", "ok_torque",
            "ok_joint_limits", "contact_pos_err_mm_p50", "speed_err_p50",
            "pen_max_mm_p90", "tau_excess_p90", "jerk_p50", "short"]
    for grid, block in rep["grids"].items():
        print(f"\n### {grid}\n")
        rows = []
        src = dict(block)
        if grid == "standard":
            src.update(rep.get("baselines", {}))
        for k, v in src.items():
            r = {kk: v.get(kk) for kk in keys}
            r["label"] = v.get("label", k)
            r["short"] = f"{v.get('short_accepted')}/{v.get('short_n')}"
            r["accepted"] = f"{v.get('accepted')}/{v.get('n')}"
            for f in ("contact_pos_err_mm_p50", "speed_err_p50", "pen_max_mm_p90",
                      "tau_excess_p90"):
                r[f] = f"{v[f]:.3g}" if v.get(f) is not None else ""
            r["jerk_p50"] = f"{v['jerk_p50']:.2e}" if v.get("jerk_p50") else ""
            rows.append(r)
        print(md(rows, keys))
    print("\n### latency\n")
    print(json.dumps(rep.get("latency", {}), indent=1))
    if "closed_loop" in rep:
        print("\n### closed loop\n")
        for e in rep["closed_loop"]["episodes"]:
            print("* " + json.dumps(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
