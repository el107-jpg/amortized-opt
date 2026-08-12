#!/usr/bin/env python
"""Section 8 / A -- per-drum and per-cell breakdown of the open-loop grids.

Reads the per-case CSVs written by ``eval_phase2.py`` (model) and the recorded
expert CSVs in ``baselines/`` and prints acceptance / component gates split by
drum, by horizon and by commanded impact speed.  Pure CSV arithmetic; no GPU,
no re-solving -- the expert is never re-run (its SQP is not reproducible, see
``baselines/trajopt_hpipm_2026-08-10.md`` caveat 6).

    python scripts/report_breakdown.py --dirs runs/final/grid_seed0 \
        runs/final/grid_seed7 runs/final/grid_seed13 --out runs/final/breakdown.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATES = ("accepted", "ok_hit", "ok_speed", "ok_collision", "ok_torque",
         "ok_joint_limits", "elev_ok")


def tb(s) -> bool:
    return str(s).strip().lower() in ("true", "1")


def read(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def counts(rows) -> dict:
    out = {"n": len(rows)}
    for gname in GATES:
        if rows and gname in rows[0]:
            out[gname] = sum(tb(r[gname]) for r in rows)
    # both-arms collision, the metric the expert must be judged against
    for c in ("coll_pen_count_all", "coll_pen_count"):
        if rows and c in rows[0]:
            out[f"free_{c}"] = sum(1 for r in rows if float(r[c]) == 0.0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--tag", default="model_quintic_x1")
    ap.add_argument("--out", default="runs/final/breakdown.json")
    args = ap.parse_args(argv)

    report: dict = {"model": {}, "expert": {}}

    for d in args.dirs:
        dp = ROOT / d if not Path(d).is_absolute() else Path(d)
        rows = []
        for grid in ("standard", "near_drum"):
            p = dp / f"{grid}_{args.tag}_per_case.csv"
            if p.exists():
                rows.extend(read(p))
        if not rows:
            print(f"[skip] {d}: no CSVs for {args.tag}")
            continue
        name = dp.name
        blk = {"all": counts(rows)}
        blk["by_drum"] = {k: counts(v) for k, v in
                          sorted(group(rows, lambda r: r["drum"]).items())}
        blk["by_grid"] = {k: counts(v) for k, v in
                          sorted(group(rows, lambda r: r["grid"]).items())}
        blk["by_horizon"] = {
            f"{float(k):.2f}": counts(v) for k, v in
            sorted(group(rows, lambda r: round(float(r["horizon_s"]), 2)).items())
        }
        blk["by_speed"] = {
            f"{float(k):.1f}": counts(v) for k, v in
            sorted(group(rows, lambda r: float(r["strike_speed"])).items())
        }
        blk["by_hand"] = {k: counts(v) for k, v in
                          sorted(group(rows, lambda r: r["hand"]).items())}
        report["model"][name] = blk

    for pref in ("trajopt_hpipm_2026-08-10_v2b", "trajopt_hpipm_2026-08-10_v2",
                 "trajopt_hpipm_2026-08-10"):
        p = ROOT / "baselines" / f"{pref}_per_case.csv"
        if not p.exists():
            continue
        rows = read(p)
        blk = {"all": counts(rows)}
        blk["by_drum"] = {k: counts(v) for k, v in
                          sorted(group(rows, lambda r: r["drum"]).items())}
        blk["by_horizon"] = {
            f"{float(k):.2f}": counts(v) for k, v in
            sorted(group(rows, lambda r: round(float(r["horizon_s"]), 2)).items())
        }
        blk["by_speed"] = {
            f"{float(k):.1f}": counts(v) for k, v in
            sorted(group(rows, lambda r: float(r["strike_speed"])).items())
        }
        blk["by_hand"] = {k: counts(v) for k, v in
                          sorted(group(rows, lambda r: r["hand"]).items())}
        report["expert"][pref] = blk

    # ---- printed tables -----------------------------------------------------
    def line(label, c):
        acc = f"{c.get('accepted', 0)}/{c['n']}"
        return (f"  {label:16} {acc:>9}  hit {c.get('ok_hit',0):>3} "
                f"spd {c.get('ok_speed',0):>3} coll {c.get('ok_collision',0):>3} "
                f"tau {c.get('ok_torque',0):>3} "
                f"collALL {c.get('free_coll_pen_count_all','-'):>4}")

    for who in ("model", "expert"):
        for name, blk in report[who].items():
            print(f"\n===== {who}: {name} =====")
            print(line("ALL", blk["all"]))
            for sect in ("by_drum", "by_horizon", "by_speed", "by_hand", "by_grid"):
                if sect not in blk:
                    continue
                print(f" -- {sect} --")
                for k, c in blk[sect].items():
                    print(line(k, c))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
