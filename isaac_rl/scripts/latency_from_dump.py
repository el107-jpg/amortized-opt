#!/usr/bin/env python
"""Hit-latency histogram from a smoke-test rollout dump, OUTSIDE Isaac.

``amortized_opt.eval_harness.TrajectoryScorer`` needs ``BatchedRNEA``, which
needs Pinocchio's kinematic tree once at construction -- and Isaac's process
cannot provide it (``isaac_rl.env.try_build_rnea`` explains why).  So the Isaac
run dumps ``(T, B, 17)`` measured poses plus the fired-strike events, and this
script measures them here, with the SAME ``amortized_opt.probe.hit_latency``
that produced `REPORT.md` §4's 63 % miss rate.  Same function, same window, same
sub-tick interpolation -- which is the whole point of the comparison.

    python isaac_rl/scripts/latency_from_dump.py runs/rl1/smoke.rollout.npz
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import amortized_opt as A                                        # noqa: E402
from amortized_opt.probe import hit_latency, summarize_latency   # noqa: E402
from isaac_rl.measure import EpisodeView                         # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--device", default="cpu")
    p.add_argument("--window-ticks", type=int, default=5)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    d = np.load(a.dump)
    Q = d["q"]                                    # (T, B, 17)
    kit = A.DrumKit.from_layout()
    scorer = A.TrajectoryScorer(device=a.device)
    ev: dict[int, list] = {}
    for t, b, h, dr in zip(d["ev_tick"], d["ev_env"], d["ev_hand"], d["ev_drum"]):
        ev.setdefault(int(b), []).append(
            (int(t), ("left", "right")[int(h)], int(dr)))

    rows = []
    for b, events in ev.items():
        qh = np.concatenate([Q[0:1, b], Q[:, b]], axis=0)
        for r in hit_latency(scorer, kit, EpisodeView(q_hist=qh, hit_steps=events),
                             window_ticks=a.window_ticks, dt=A.CONTROL_DT):
            r["env"] = b
            rows.append(r)

    out = {"dump": a.dump, "n_strikes": len(rows),
           "overall": summarize_latency(rows)}
    for key in ("hand", "drum"):
        g: dict[str, list] = {}
        for r in rows:
            g.setdefault(str(r[key]), []).append(r)
        out[f"by_{key}"] = {k: summarize_latency(v) for k, v in sorted(g.items())}
    o = out["overall"]
    print(f"ISAAC closed loop, {o['n']} scheduled strikes")
    print(f"  crossed in disk : {o['crossed_in_disk']}")
    print(f"  MISS RATE       : {100*o['miss_rate']:.0f} %      "
          f"(analytic plant, REPORT.md 4: 63 %)")
    print(f"  |lat| p50 / p90 : {o['lat_abs_p50_ms']:.1f} / {o['lat_abs_p90_ms']:.1f} ms"
          f"   (analytic: 9.5 / 23.2 ms)")
    print(f"  <=10/20/40 ms   : {o['within_10ms']} / {o['within_20ms']} / {o['within_40ms']}")
    print(f"  early           : {o['early']}")
    print(f"  speed in band   : {o['speed_in_band']}   p50 {o['speed_p50_mps']:.2f} m/s")
    print(f"  histogram (ms)  : {o['hist_ms']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
