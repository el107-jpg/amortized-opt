#!/usr/bin/env python
"""Section 8 / D -- the training story, read off the recorded metrics.

No training and no GPU: this walks every ``runs/*/metrics.jsonl`` +
``summary.json`` and prints (a) the wall-clock / step budget of the whole
project and (b) per-phase cost, violation and augmented-Lagrangian
trajectories, so the "trains in hours" claim is checkable rather than asserted.

    python scripts/report_training.py --out runs/final/training.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: (label, run dir, note) in the order the project ran them.
RUNS = [
    ("phase 1 (warm start)", "phase1_main", "BC on the quintic warm start"),
    ("pilot race (4 x 1200)", "pilots_main", "lr / rho / aux-tc, paired on data"),
    ("phase 2 (cost descent)", "phase2_main", "lr_hi config, eager"),
    ("phase 2b aborted", "phase2b_lr3e4_aborted", "lr 3e-4 continuation, destroyed"),
    ("phase 2b (CHAMPION)", "phase2b_main", "lr 5e-5 + torque peak; legs 1-3"),
    ("phase 2c", "phase2c_main", "7c objective; not the champion"),
]


def read_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def phase_entry(label: str, name: str, note: str) -> dict:
    d = ROOT / "runs" / name
    recs = read_jsonl(d / "metrics.jsonl")
    # a pilot directory holds one sub-run per pilot
    subs = []
    if not recs:
        for sub in sorted(d.glob("*/metrics.jsonl")):
            subs.append((sub.parent.name, read_jsonl(sub)))
    entry = {"label": label, "run": name, "note": note, "exists": d.exists()}
    summ = {}
    for f in ("summary.json", "summary_leg1.json"):
        if (d / f).exists():
            summ[f] = json.loads((d / f).read_text())
    entry["summary"] = summ

    def legs(rs):
        """One record group per --resume leg (each writes a fresh 'start')."""
        out, cur = [], None
        for r in rs:
            if r.get("event") == "start":
                cur = {"train": [], "eval": [], "probe": [], "done": None,
                       "cfg": r.get("cfg", {})}
                out.append(cur)
            elif cur is None:
                continue
            elif r.get("event") in ("train", "eval", "probe"):
                cur[r["event"]].append(r)
            elif r.get("event") == "done":
                cur["done"] = r
        return out

    groups = legs(recs) if recs else []
    entry["legs"] = []
    for i, g in enumerate(groups):
        tr, ev = g["train"], g["eval"]
        L = {
            "leg": i,
            "steps": (tr[-1].get("step") if tr else None),
            "wall_s": (g["done"] or {}).get("wall_s",
                                            tr[-1].get("elapsed_s") if tr else None),
            "it_per_s": (g["done"] or {}).get("steps_per_s"),
            "lr0": tr[0].get("lr") if tr else None,
            "lr_end": tr[-1].get("lr") if tr else None,
            "cost_first": tr[0].get("cost") if tr else None,
            "cost_last": tr[-1].get("cost") if tr else None,
            "lam_last": tr[-1].get("lam") if tr else None,
            "grad_norm_p50": None,
            "skipped": (g["done"] or {}).get("skipped_steps"),
            "evals": [{k: e.get(k) for k in
                       ("step", "accepted", "n", "ok_hit", "ok_collision",
                        "ok_torque", "ok_speed", "elbow_down", "short_accepted",
                        "short_n", "contact_pos_err_mm_p50", "speed_err_p50",
                        "tau_excess_p90", "jerk_p50", "stage_a")} for e in ev],
            "probes": [{k: p.get(k) for k in ("step", "in_disk", "hits", "score")}
                       for p in g["probe"]],
        }
        gn = [r.get("grad_norm") for r in tr if r.get("grad_norm") is not None]
        if gn:
            gn = sorted(gn)
            L["grad_norm_p50"] = gn[len(gn) // 2]
        # the cost trajectory, thinned
        L["cost_curve"] = [{"step": r.get("step"), "cost": r.get("cost"),
                            "c_terminal": r.get("c_terminal"),
                            "c_torque": r.get("c_torque"),
                            "c_collision": r.get("c_collision"),
                            "d_pos_err": r.get("d_pos_err"),
                            "d_min_sdf": r.get("d_min_sdf"),
                            "d_tau_hinge_max": r.get("d_tau_hinge_max"),
                            "lam": r.get("lam"), "lr": r.get("lr")}
                           for r in tr]
        entry["legs"].append(L)
    for nm, rs in subs:
        g = legs(rs)
        if g and g[0]["train"]:
            tr = g[0]["train"]
            entry.setdefault("pilots", []).append({
                "name": nm, "steps": tr[-1].get("step"),
                "wall_s": tr[-1].get("elapsed_s"),
                "evals": [{k: e.get(k) for k in ("step", "accepted", "n", "ok_hit",
                                                 "ok_collision", "ok_torque")}
                          for e in g[0]["eval"]],
            })
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/final/training.json")
    args = ap.parse_args(argv)

    report = {"phases": [phase_entry(*r) for r in RUNS]}
    tot_s = 0.0
    tot_steps = 0
    print(f"{'phase':28} {'legs':>4} {'steps':>8} {'wall':>10} {'it/s':>6}  note")
    print("-" * 92)
    for p in report["phases"]:
        if not p["exists"]:
            print(f"{p['label']:28} (missing)")
            continue
        legs = p["legs"]
        st = sum(int(l["steps"] or 0) for l in legs)
        w = sum(float(l["wall_s"] or 0.0) for l in legs)
        if not legs and p.get("pilots"):
            st = sum(int(x["steps"] or 0) for x in p["pilots"])
            w = sum(float(x["wall_s"] or 0.0) for x in p["pilots"])
        tot_s += w
        tot_steps += st
        rate = st / w if w else float("nan")
        print(f"{p['label']:28} {max(len(legs), len(p.get('pilots', []))):>4} "
              f"{st:>8} {w/60:>8.1f} m {rate:>6.2f}  {p['note']}")
        p["total_steps"], p["total_wall_s"] = st, w
    print("-" * 92)
    print(f"{'TOTAL':28} {'':>4} {tot_steps:>8} {tot_s/3600:>8.2f} h")
    report["total_steps"] = tot_steps
    report["total_wall_s"] = tot_s

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
