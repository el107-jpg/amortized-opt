#!/usr/bin/env python3
"""Carve a small known-good fixture out of a recorded baseline run.

``tests/test_eval_harness.py`` needs real solver output to assert that the
harness scores a *good* trajectory as good.  Re-solving in the test would drag
the repo (and ~1 s/case) into the standalone test suite, so a handful of
accepted trajopt-HPIPM solutions are frozen into ``tests/data/`` instead.

Selection is deterministic: among the cases the recorded run marked
``accepted``, take up to ``--per-drum`` per drum, lowest case id first, so the
fixture covers every drum class the strike rules distinguish (pad, tom, cymbal,
hi-hat) rather than whatever happened to sort first.

    python scripts/make_test_fixture.py --run-dir baselines \\
        --prefix trajopt_hpipm_2026-08-10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from amortized_opt.eval_harness import load_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=_PKG / "baselines")
    ap.add_argument("--prefix", default="trajopt_hpipm_2026-08-10")
    ap.add_argument("--out-dir", type=Path, default=_PKG / "tests" / "data")
    ap.add_argument("--per-drum", type=int, default=2)
    ap.add_argument("--grid", default="standard")
    args = ap.parse_args()

    cases, trajs, meta = load_run(args.run_dir, prefix=args.prefix)
    csv_path = args.run_dir / f"{args.prefix}_per_case.csv"
    with csv_path.open() as fh:
        scored = {int(r["case_id"]): r for r in csv.DictReader(fh)}

    by_id = {int(c.case_id): c for c in cases}
    chosen: dict[str, list[int]] = {}
    for cid in sorted(by_id):
        c, row = by_id[cid], scored.get(cid)
        if row is None or row["accepted"] != "True" or c.grid != args.grid:
            continue
        if cid not in trajs:
            continue
        got = chosen.setdefault(c.drum, [])
        if len(got) < args.per_drum:
            got.append(cid)
    ids = sorted(i for v in chosen.values() for i in v)
    if not ids:
        raise SystemExit(f"no accepted {args.grid} cases in {csv_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p_npz = args.out_dir / "trajopt_hpipm_fixture.npz"
    p_json = args.out_dir / "trajopt_hpipm_fixture_cases.json"
    np.savez_compressed(
        p_npz, **{f"case_{i:05d}": np.asarray(trajs[i], float) for i in ids}
    )
    p_json.write_text(
        json.dumps(
            {
                "source": {
                    "prefix": args.prefix,
                    "planner": meta.get("planner"),
                    "git_head": meta.get("git_head"),
                    "utc": meta.get("utc"),
                },
                "cases": [by_id[i].to_dict() for i in ids],
            },
            indent=1,
        )
    )
    print(f"{len(ids)} cases over {len(chosen)} drums -> {p_npz.name}, {p_json.name}")
    for drum, v in sorted(chosen.items()):
        print(f"  {drum:14s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
