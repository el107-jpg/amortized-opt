#!/usr/bin/env python
"""Build ``assets/idle_home_poses.json`` -- the idle arm's park posture.

    python scripts/build_idle_home.py [--margin 0.06] [--yaw-checks 129] [-v]

ONE torso-relative arm pose per hand.  The arm is mounted on the torso, so a
fixed vector of joint angles rotates with the body for free -- an idle arm asked
to hold it genuinely sits still.  The waist-yaw sweep is not an index, it is the
CONSTRAINT SET: the kit is world-fixed while the arm swings with the torso, so
the parked posture has to clear every drum at every yaw simultaneously.

Prints the report card: worst-case clearance over the whole sweep (the number
that matters), and how far the pose sits from the strike poses it departs from.
The ``ready`` row is the baseline -- what an idle arm held before this existed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos  # noqa: E402
from amortized_opt.fk import ChainFK  # noqa: E402
from amortized_opt.idle_home import (  # noqa: E402
    _FRAMES,
    IdleHomeConfig,
    build_idle_home_table,
    default_idle_home_path,
)
from amortized_opt.limits import urdf_position_limits  # noqa: E402
from amortized_opt.sdf import DrumSceneSpec, SdfPenalty  # noqa: E402
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice  # noqa: E402


def _sweep_clearance(fk, sdf, arm_q, yaws, names) -> np.ndarray:
    """(2, n_yaw) per-hand clearance of ``arm_q`` swept over ``yaws``."""
    q = np.repeat(ready_joint_pos(names)[None, :], len(yaws), axis=0)
    q[:, names.index("waist_yaw_joint")] = yaws
    for nm in ("waist_pitch_joint", "waist_roll_joint"):
        q[:, names.index(nm)] = 0.0
    for hi, hand in enumerate(("left", "right")):
        q[:, arm_slice(hand, names)] = arm_q[hi]
    with torch.no_grad():
        _p, det = sdf.forward(
            frames=fk.forward(torch.as_tensor(q, dtype=sdf.dtype)), return_details=True
        )
    return np.stack(
        [det.extras["min_sdf_left"].numpy(), det.extras["min_sdf_right"].numpy()]
    )


def _reach_rms(arm_q, yaws, lib, names) -> np.ndarray:
    """(2,) RMS joint distance from each park pose to that hand's strike poses."""
    out = []
    for hi, hand in enumerate(("left", "right")):
        acc = [
            ((arm_q[hi] - lib.select(hand, d, float(y)).arm_q) ** 2).mean()
            for y in yaws for d in lib.drums(hand)
        ]
        out.append(np.sqrt(np.mean(acc)))
    return np.asarray(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaw-checks", type=int, default=IdleHomeConfig.n_yaw_checks)
    ap.add_argument("--margin", type=float, default=IdleHomeConfig.margin)
    ap.add_argument("--iters", type=int, default=IdleHomeConfig.iters)
    ap.add_argument("--lr", type=float, default=IdleHomeConfig.lr)
    ap.add_argument("--w-reach", type=float, default=IdleHomeConfig.w_reach)
    ap.add_argument("--restarts", type=int, default=IdleHomeConfig.restarts)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = IdleHomeConfig(
        n_yaw_checks=args.yaw_checks, margin=args.margin, iters=args.iters,
        lr=args.lr, w_reach=args.w_reach, restarts=args.restarts,
    )
    names = tuple(G1_UPPER_BODY_JOINTS)
    fk = ChainFK(None, joint_names=names, frames=dict(_FRAMES), dtype=torch.float64)
    scene = DrumSceneSpec.from_layout(None, frame="pelvis")
    sdf = SdfPenalty(scene, fk=fk, margin=cfg.margin, power=2, dtype=torch.float64)
    lib = GoalPoseLibrary(joint_names=names)

    print(f"solving 2 park postures, each clear at {cfg.n_yaw_checks} waist yaws "
          f"({cfg.restarts} restarts x {cfg.iters} iters)...")
    tab = build_idle_home_table(cfg, scene=scene, fk=fk, library=lib,
                                verbose=args.verbose)
    yaws = tab.yaw_grid

    ready_arm = np.stack([ready_joint_pos(names)[arm_slice(h, names)]
                          for h in ("left", "right")])
    rows = (("ready", ready_arm), ("home", tab.q_arm))

    print(f"\nclearance swept over {len(yaws)} waist yaws, "
          f"safety margin = {cfg.margin * 1e3:.0f} mm")
    print(f"{'pose':>8} {'hand':>6} {'WORST':>9} {'p10':>9} {'p50':>9} {'in collision':>13}")
    for label, arm in rows:
        sw = _sweep_clearance(fk, sdf, arm, yaws, names)
        for hi, hand in enumerate(("left", "right")):
            v = sw[hi] * 1e3
            print(f"{label:>8} {hand:>6} {v.min():9.1f} {np.percentile(v, 10):9.1f} "
                  f"{np.percentile(v, 50):9.1f} {int((sw[hi] < 0).sum()):8d}/{len(yaws)}")

    print("\nRMS joint distance to the strike poses (rad, lower = readier):")
    for label, arm in rows:
        r = _reach_rms(arm, yaws, lib, names)
        print(f"{label:>8}  left {r[0]:.3f}   right {r[1]:.3f}")

    lo, hi = urdf_position_limits(names, posture_limits=True)
    print("\npark posture (rad):")
    for hi_, hand in enumerate(("left", "right")):
        idx = arm_slice(hand, names)
        inside = bool((tab.q_arm[hi_] >= lo[idx]).all() and (tab.q_arm[hi_] <= hi[idx]).all())
        print(f"  {hand:>5}: {np.round(tab.q_arm[hi_], 3).tolist()}  in-box={inside}")
        print(f"         tip (torso frame): {np.round(tab.tip_pos(hand), 3).tolist()}")

    out = tab.save(args.out or default_idle_home_path())
    print(f"\nworst clearance anywhere: {tab.worst_clearance * 1e3:.1f} mm")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
