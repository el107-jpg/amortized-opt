#!/usr/bin/env python
"""Pilot race: 3-4 short phase-2 configs in parallel, pick one, then commit.

Resolved decision 6c -- "pilot-race 3-4 short (~5k-step) weight settings in
parallel on the two free GPUs before committing, to avoid mid-run restarts".

    python scripts/pilot_phase2.py --steps 5000 --devices cuda:1 cuda:3

Each pilot is a full ``train_phase2.py`` invocation differing only in the knobs
under test, sharing one cached scenario pool so the comparison is paired on
data as well as on seed.  They are scheduled round-robin over the given devices
(two at a time by default) and ranked by the rolling eval-grid acceptance,
tie-broken by the short-horizon cell.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: name -> extra CLI flags.  Everything not listed is the ``Phase2Config``
#: default, so a pilot differs from the reference in ONE dimension.
PILOTS: dict[str, list[str]] = {
    "base":      [],
    "lr_hi":     ["--lr", "3e-4"],
    "rho_hi":    ["--rho-scale", "4.0"],
    "aux_tc_hi": ["--w-aux", "20", "--w-tc", "200"],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:3"])
    p.add_argument("--pilots", nargs="+", default=list(PILOTS))
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--scenarios", type=int, default=250_000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--init", default="runs/phase1_main/best.pt")
    p.add_argument("--eval-every", type=int, default=1_000)
    p.add_argument("--eval-cases", type=int, default=81)
    # Scratch, not a deliverable: a ~GB scenario pool shared by every pilot in
    # the batch. Honours $TMPDIR so the folder is not tied to this box.
    p.add_argument(
        "--pool-cache",
        default=str(Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "p2pool_pilot.npz"),
    )
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--buckets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--out", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--extra", nargs="*", default=[],
                   help="extra flags forwarded to every pilot")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_cmd(name: str, args, device: str, run_dir: Path) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts" / "train_phase2.py"),
        "--device", device,
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--scenarios", str(args.scenarios),
        "--workers", str(args.workers),
        "--init", args.init,
        "--eval-every", str(args.eval_every),
        "--eval-cases", str(args.eval_cases),
        "--ckpt-every", str(max(args.steps // 2, 1)),
        "--pool-cache", args.pool_cache,
        # A pilot is a WEIGHT probe, not a curriculum probe: compress the ramp
        # so 5 k steps still see the fully-weighted objective, and keep the
        # chained stage out of it (it only ramps in over the final third of a
        # real run, which a pilot does not have).
        "--ramp-end", str(max(args.steps // 2, 1)),
        "--bptt-max-frac", "0.0",
        "--horizon", str(args.horizon),
        "--buckets", *[str(b) for b in args.buckets],
        "--out", str(run_dir),
        "--name", name,
    ] + (["--no-compile"] if args.no_compile else []) + list(args.extra) + PILOTS[name]


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out) if args.out else ROOT / "runs" / (
        f"pilots_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pilots] {out}", flush=True)

    # The pool is generated once, by the first pilot, and reused by the rest --
    # so the race is paired on data.  Warm it here so the parallel starts do
    # not all try to write the cache at once.
    cache = Path(args.pool_cache)
    if not cache.exists() and not args.dry_run:
        print("[pilots] warming the shared scenario pool ...", flush=True)
        warm = build_cmd(args.pilots[0], args, args.devices[0], out / "_warm")
        warm[warm.index("--steps") + 1] = "1"
        subprocess.run(warm, check=True, cwd=ROOT,
                       stdout=open(out / "_warm.log", "w"), stderr=subprocess.STDOUT)

    queue = list(args.pilots)
    running: list[tuple[str, subprocess.Popen, Path]] = []
    slots = list(args.devices)          # one slot per entry; repeat a device to
    busy: list[str | None] = [None] * len(slots)   # run two jobs on one GPU
    t0 = time.perf_counter()
    while queue or running:
        for si, dev in enumerate(slots):
            if busy[si] is None and queue:
                name = queue.pop(0)
                rd = out / name
                rd.mkdir(parents=True, exist_ok=True)
                cmd = build_cmd(name, args, dev, rd)
                (rd / "cmd.txt").write_text(" ".join(cmd))
                print(f"[pilots] start {name} on {dev}", flush=True)
                if args.dry_run:
                    continue
                proc = subprocess.Popen(
                    cmd, cwd=ROOT, stdout=open(rd / "train.log", "w"),
                    stderr=subprocess.STDOUT,
                )
                busy[si] = name
                running.append((name, proc, rd))
        if args.dry_run:
            break
        time.sleep(10)
        for item in list(running):
            name, proc, rd = item
            if proc.poll() is None:
                continue
            running.remove(item)
            for si, n in enumerate(busy):
                if n == name:
                    busy[si] = None
            print(f"[pilots] done  {name} rc={proc.returncode} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- collate ------------------------------------------------------------
    rows = []
    for name in args.pilots:
        s = out / name / "summary.json"
        if not s.exists():
            rows.append({"pilot": name, "status": "FAILED"})
            continue
        d = json.loads(s.read_text())
        evs = d.get("evals", [])
        last = evs[-1] if evs else {}
        best = max(evs, key=lambda e: e.get("acceptance", 0.0)) if evs else {}
        rows.append({
            "pilot": name,
            "steps": d.get("steps"),
            "it_s": round(d.get("steps_per_s", 0.0), 2),
            "best_acc": round(best.get("acceptance", 0.0), 4),
            "last_acc": round(last.get("acceptance", 0.0), 4),
            "hit": round(last.get("ok_hit", 0.0), 3),
            "coll": round(last.get("ok_collision", 0.0), 3),
            "tau": round(last.get("ok_torque", 0.0), 3),
            "elbow": round(last.get("elbow_down", 0.0), 3),
            "short": f"{last.get('short_accepted', 0)}/{last.get('short_n', 0)}",
            "pos_p50": round(last.get("contact_pos_err_mm_p50", 0.0), 1),
            "spd_p50": round(last.get("speed_err_p50", 0.0), 3),
            "skipped": d.get("skipped_steps", 0),
        })
    rows.sort(key=lambda r: (-(r.get("best_acc") or 0.0),
                             -(float((r.get("short") or "0/1").split("/")[0]))))
    (out / "pilots.json").write_text(json.dumps(rows, indent=2))
    keys = ["pilot", "steps", "it_s", "best_acc", "last_acc", "hit", "coll", "tau",
            "elbow", "short", "pos_p50", "spd_p50", "skipped"]
    w = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print("\n" + "  ".join(k.ljust(w[k]) for k in keys))
    print("-" * (sum(w.values()) + 2 * len(keys)))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(w[k]) for k in keys))
    if rows and rows[0].get("best_acc") is not None:
        print(f"\n[pilots] winner: {rows[0]['pilot']} "
              f"({PILOTS.get(rows[0]['pilot'], []) or 'defaults'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
