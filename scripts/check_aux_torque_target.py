#!/usr/bin/env python
"""Compare the two auxiliary TORQUE targets the model's ``head_torque`` can learn.

    python scripts/check_aux_torque_target.py --n 400 --json out.json

``aux_targets(..., torque_mode=...)`` has two definitions:

* ``"rnea_segmax"`` -- the historical one, ``max(|tau_rnea[i-1]|, |tau_rnea[i]|)``:
  the OPEN-LOOP inverse dynamics of the plan itself;
* ``"tracking"`` -- ``|tau_rnea[i] + tau_fb[i]|``: the torque the PD servo
  actually DEMANDS to track the plan, which is what ``w_track_torque`` prices
  and what the 5 N*m wrist motors actually saturate on.

This prints, per joint, the RMS and p99 of each definition, their Pearson
correlation over the masked waypoints, and the RNEA share of the demand
(``rms_rnea / rms_tracking``).  The claim under test is that on the wrist
pitch/yaw joints the RNEA number is a small fraction of the real demand, so a
head trained on it is not predicting anything the objective charges for.

CPU only by default and cheap -- a few hundred scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.constants import G1_UPPER_BODY_JOINTS  # noqa: E402
from amortized_opt.limits import isaac_effort_limits  # noqa: E402
from amortized_opt.warmstart_data import (  # noqa: E402
    ScenarioConfig,
    WarmstartPool,
    generate_pool_parallel,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=400, help="scenarios to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split", default="train")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default=None, help="also dump the table here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    names = tuple(G1_UPPER_BODY_JOINTS)
    dev = torch.device(args.device)
    dtype = torch.float32
    rating = np.asarray(isaac_effort_limits(names), float)

    cfg = ScenarioConfig()
    raw = generate_pool_parallel(
        args.n, seed=args.seed, cfg=cfg, split=args.split, workers=args.workers
    )
    fk = A.ChainFK(device=dev, dtype=dtype)
    rnea = A.BatchedRNEA.from_urdf(
        joint_names=names, with_armature=True, device=dev, dtype=dtype
    )
    kernel = A.nominal_torque_kernels(
        rnea, joint_names=names, dt=float(cfg.dt), device=dev, dtype=dtype
    )
    print(f"[kernel] {tuple(kernel.shape)} taps, DC sum per joint "
          f"min {float(kernel.sum(1).min()):.3f} max {float(kernel.sum(1).max()):.3f}")

    # Two REAL pools over the identical scenarios -- same masks, same spline,
    # same chunking as training; only ``torque_mode`` differs.
    pools = {
        mode: WarmstartPool(
            raw["buckets"], cfg=cfg, device=dev, dtype=dtype, fk=fk, rnea=rnea,
            aux_torque_tracking=(mode == "tracking"),
        )
        for mode in ("rnea_segmax", "tracking")
    }
    p_seg, p_trk = pools["rnea_segmax"], pools["tracking"]

    cols: dict[str, list[np.ndarray]] = {"a": [], "b": [], "m": []}
    n_rows = 0
    for H in p_seg.buckets:
        em = p_seg.data[H]["elem_mask"].to(dtype)
        a = p_seg.data[H]["aux_torque"]
        b = p_trk.data[H]["aux_torque"]
        cols["a"].append(a.reshape(-1, len(names)).numpy())
        cols["b"].append(b.reshape(-1, len(names)).numpy())
        cols["m"].append(em.reshape(-1, len(names)).numpy() > 0.5)
        n_rows += int(em.shape[0])
        print(f"[bucket {H:3d}] {int(em.shape[0])} rows")

    A_ = np.concatenate(cols["a"]), np.concatenate(cols["b"]), np.concatenate(cols["m"])
    seg, trk, msk = A_
    print(f"\n{n_rows} scenarios, {int(msk.sum())} masked (waypoint, joint) elements")
    print(f"{'joint':<28}{'rate':>6}{'rms_seg':>9}{'rms_trk':>9}"
          f"{'p99_seg':>9}{'p99_trk':>9}{'corr':>7}{'seg/trk':>9}{'>rate%':>8}")
    rows = []
    for j, nm in enumerate(names):
        k = msk[:, j]
        if k.sum() < 8:
            continue
        x, y = seg[k, j], trk[k, j]
        rms_a = float(np.sqrt((x * x).mean()))
        rms_b = float(np.sqrt((y * y).mean()))
        c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        over = 100.0 * float((y > rating[j]).mean())
        rec = dict(
            joint=nm, rating=float(rating[j]), rms_segmax=rms_a, rms_tracking=rms_b,
            p99_segmax=float(np.quantile(x, 0.99)), p99_tracking=float(np.quantile(y, 0.99)),
            corr=c, rnea_share=rms_a / max(rms_b, 1e-12),
            pct_tracking_over_rating=over, n=int(k.sum()),
        )
        rows.append(rec)
        print(f"{nm:<28}{rating[j]:6.0f}{rms_a:9.2f}{rms_b:9.2f}"
              f"{rec['p99_segmax']:9.2f}{rec['p99_tracking']:9.2f}"
              f"{c:7.3f}{rec['rnea_share']:9.3f}{over:8.2f}")

    wr = [r for r in rows if "wrist_pitch" in r["joint"] or "wrist_yaw" in r["joint"]]
    if wr:
        sa = float(np.mean([r["rms_segmax"] for r in wr]))
        sb = float(np.mean([r["rms_tracking"] for r in wr]))
        print(f"\nWRIST pitch/yaw (4 joints, 5 N*m rating): rms_segmax {sa:.2f} N*m, "
              f"rms_tracking {sb:.2f} N*m -> RNEA is {100*sa/max(sb,1e-12):.1f}% of demand; "
              f"mean corr {np.mean([r['corr'] for r in wr]):.3f}; "
              f"{np.mean([r['pct_tracking_over_rating'] for r in wr]):.2f}% of tracking "
              f"waypoints exceed the rating (segmax: "
              f"{np.mean([100.0*float((seg[msk[:, j], j] > rating[j]).mean()) for j, n in enumerate(names) if 'wrist_pitch' in n or 'wrist_yaw' in n]):.2f}%)")

    if args.json:
        Path(args.json).write_text(json.dumps({"n_scenarios": n_rows, "rows": rows}, indent=2))
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
