#!/usr/bin/env python3
"""Quick LS epsilon stability digest from a phase-2 metrics.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _finite(xs):
    return [x for x in xs if x is not None and math.isfinite(float(x))]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("metrics", type=Path, nargs="?",
                   default=Path("runs/rhythm_bellman_nf1_ls/metrics.jsonl"))
    p.add_argument("--eps-min", type=float, default=1e-10,
                   help="resolved ε_min (seed*1e-6 for seed=1e-4)")
    p.add_argument("--eps-max", type=float, default=1e-3,
                   help="resolved ε_max (seed*10 for seed=1e-4)")
    p.add_argument("--atol-frac", type=float, default=1e-3,
                   help="relative tolerance for 'at bound' count")
    args = p.parse_args()
    if not args.metrics.is_file():
        print(f"missing {args.metrics}", file=sys.stderr)
        return 1

    trains, fails = [], []
    for line in args.metrics.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = rec.get("event")
        if ev == "train":
            trains.append(rec)
        elif ev == "ls_selection_failed":
            fails.append(rec)

    print(f"file={args.metrics}  train_rows={len(trains)}  "
          f"selection_failures={len(fails)}")
    if not trains:
        return 0

    eps = _finite([r.get("ls_epsilon", r.get("lr")) for r in trains])
    trials = _finite([r.get("ls_trial_count") for r in trains])
    accepted = [int(r["ls_accepted"]) for r in trains if r.get("ls_accepted") is not None]
    searched = [int(r["ls_searched"]) for r in trains if r.get("ls_searched") is not None]
    Ks = _finite([r.get("ls_K") for r in trains])
    losses = _finite([r.get("loss") for r in trains])

    def pct(xs):
        xs = sorted(xs)
        n = len(xs)
        if not n:
            return None
        def q(f):
            i = min(n - 1, max(0, int(round(f * (n - 1)))))
            return xs[i]
        return dict(min=xs[0], p50=q(0.5), p90=q(0.9), max=xs[-1], n=n)

    print("epsilon:", pct(eps))
    print("lr field:", pct(_finite([r.get("lr") for r in trains])))
    print("trials:", pct(trials))
    print("K:", pct(Ks))
    if accepted:
        print(f"ls_accepted rate: {sum(accepted)/len(accepted):.3f} "
              f"(n={len(accepted)})")
    if searched:
        print(f"ls_searched rate: {sum(searched)/len(searched):.3f} "
              f"(n={len(searched)})")
    if losses:
        print(f"loss: first={losses[0]:.4g} last={losses[-1]:.4g} "
              f"min={min(losses):.4g}")

    emin, emax = float(args.eps_min), float(args.eps_max)
    if eps:
        near_min = sum(1 for e in eps if e <= emin * (1.0 + args.atol_frac)
                       or abs(e - emin) / max(emin, 1e-30) <= args.atol_frac)
        near_max = sum(1 for e in eps if e >= emax * (1.0 - args.atol_frac))
        print(f"at/near ε_min={emin:g}: {near_min}/{len(eps)} "
              f"({100*near_min/len(eps):.1f}%)")
        print(f"at/near ε_max={emax:g}: {near_max}/{len(eps)} "
              f"({100*near_max/len(eps):.1f}%)")
        # last 20
        tail = eps[-20:]
        print(f"last {len(tail)} ε: "
              + ", ".join(f"{e:.3g}" for e in tail))

    fail_reasons = {}
    for r in fails:
        k = r.get("ls_failure") or r.get("failure") or "?"
        fail_reasons[k] = fail_reasons.get(k, 0) + 1
    if fail_reasons:
        print("selection_failure reasons:", fail_reasons)

    # last train row snapshot
    last = trains[-1]
    print(f"last step={last.get('step')} u={last.get('ls_u')} "
          f"ε={last.get('ls_epsilon', last.get('lr'))} "
          f"trials={last.get('ls_trial_count')} "
          f"K={last.get('ls_K')} searched={last.get('ls_searched')} "
          f"accepted={last.get('ls_accepted')} "
          f"failure={last.get('ls_failure')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
