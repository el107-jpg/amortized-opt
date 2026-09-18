#!/usr/bin/env python
"""Build ``assets/idle_cylinder_poses.json`` -- ~20 idle arm poses per hand.

    python scripts/build_idle_cylinder_library.py [--n-poses 20] [-v]

Shoulders stay near a tucked/ready mix; elbow/wrist come from GoalPoseLibrary
blends.  Every kept pose has its stick tip inside the waist-centered cylinder
from ``DrumKit.from_layout()``.
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
from amortized_opt.idle_cylinder import (  # noqa: E402
    IdleCylinder,
    IdleCylinderLibrary,
    default_idle_cylinder_path,
    derive_cylinder,
    nudge_into_cylinder,
    outside,
    tip_in_torso,
    tip_inside,
)
from amortized_opt.limits import urdf_position_limits  # noqa: E402
from amortized_opt.rhythm import DrumKit  # noqa: E402
from amortized_opt.warmstart import GoalPoseLibrary, arm_slice  # noqa: E402

_HANDS = ("left", "right")
_N_SHOULDER = 3


def _distal_pool(library: GoalPoseLibrary, hand: str, names: tuple[str, ...]) -> np.ndarray:
    """Collect elbow+wrist (and full arm) variants from the strike library."""
    rows = []
    for drum in library.drums(hand):
        for e in library.entries(hand, drum):
            rows.append(np.asarray(e.arm_q, float))
    if not rows:
        ready = ready_joint_pos(names)[arm_slice(hand, names)]
        return ready.reshape(1, -1)
    return np.stack(rows, axis=0)


def _candidate_arms(
    hand: str,
    names: tuple[str, ...],
    library: GoalPoseLibrary,
    *,
    n_target: int,
    seed: int,
) -> np.ndarray:
    """Shoulder≈ready plus tucked (inward) variants, distal from library blends."""
    ready = ready_joint_pos(names)[arm_slice(hand, names)].copy()
    pool = _distal_pool(library, hand, names)
    rng = np.random.default_rng(seed + (0 if hand == "left" else 17))
    # Waist-centered cylinder is ~0.6 m inward of the ready tip; ready shoulders
    # point at the kit, so also seed a more tucked proximal pose.
    tucked = ready.copy()
    sgn = 1.0 if hand == "left" else -1.0
    tucked[0] = ready[0] + 0.35          # less forward pitch
    tucked[1] = ready[1] + sgn * 0.45    # more inward roll
    tucked[3] = min(ready[3] + 0.45, 1.6)  # more elbow
    out = [ready.copy(), tucked]
    for proximal in (ready, tucked):
        for row in pool:
            q = proximal.copy()
            q[_N_SHOULDER:] = row[_N_SHOULDER:]
            out.append(q)
    if len(pool) >= 2:
        for _ in range(max(n_target * 2, 40)):
            i, j = rng.integers(0, len(pool), size=2)
            a = float(rng.uniform(0.0, 1.0))
            proximal = tucked if rng.random() < 0.5 else ready
            q = proximal.copy()
            q[_N_SHOULDER:] = (1.0 - a) * pool[i, _N_SHOULDER:] + a * pool[j, _N_SHOULDER:]
            q[:_N_SHOULDER] = proximal[:_N_SHOULDER] + rng.normal(0.0, 0.06, size=_N_SHOULDER)
            out.append(q)
    return np.stack(out, axis=0)


def _select_inside(
    hand: str,
    cands: np.ndarray,
    *,
    fk: ChainFK,
    cyl,
    names: tuple[str, ...],
    n_keep: int,
    q_bounds,
) -> tuple[np.ndarray, np.ndarray]:
    """Nudge into a slightly shrunken cylinder so kept tips are strictly inside."""
    lo, hi = q_bounds
    inner = IdleCylinder(
        center=cyl.center, axis=cyl.axis,
        radius=max(0.01, float(cyl.radius) - 0.008),
        half_height=max(0.01, float(cyl.half_height) - 0.008),
    )
    cols = arm_slice(hand, names)
    kept_q, kept_tip = [], []
    base = ready_joint_pos(names)
    for arm in cands:
        q = base.copy()
        q[cols] = np.clip(arm, lo[cols], hi[cols])
        q = nudge_into_cylinder(
            fk, q, hand, inner, steps=20, damping=0.05, shoulder_weight=2.0,
            joint_names=names, q_bounds=q_bounds, tol=1e-4,
        )
        q[cols] = np.clip(q[cols], lo[cols], hi[cols])
        with torch.no_grad():
            tip = tip_in_torso(
                fk, torch.as_tensor(q[None], dtype=fk.dtype), hand
            ).squeeze(0).cpu().numpy()
        if not tip_inside(cyl, tip, tol=1e-4):
            continue
        # dedupe by tip proximity
        if kept_tip and min(np.linalg.norm(tip - t) for t in kept_tip) < 0.02:
            continue
        kept_q.append(q[cols].copy())
        kept_tip.append(tip)
        if len(kept_q) >= n_keep:
            break
    if len(kept_q) < max(n_keep // 2, 4):
        raise RuntimeError(
            f"{hand}: only {len(kept_q)} poses inside the cylinder "
            f"(need ~{n_keep}); check cylinder placement / library"
        )
    ready_a = ready_joint_pos(names)[cols]
    order = np.argsort([np.sum((a[:_N_SHOULDER] - ready_a[:_N_SHOULDER]) ** 2) for a in kept_q])
    kept_q = [kept_q[i] for i in order]
    kept_tip = [kept_tip[i] for i in order]
    while len(kept_q) < n_keep:
        kept_q.append(kept_q[0].copy())
        kept_tip.append(kept_tip[0].copy())
    return np.stack(kept_q[:n_keep]), np.stack(kept_tip[:n_keep])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-poses", type=int, default=20)
    ap.add_argument("--n-yaw", type=int, default=129)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    names = tuple(G1_UPPER_BODY_JOINTS)
    fk = ChainFK(None, joint_names=names, dtype=torch.float64)
    kit = DrumKit.from_layout()
    goal_lib = GoalPoseLibrary(joint_names=names)
    q_ref = ready_joint_pos(names)
    cyl = derive_cylinder(kit, fk, q_ref, n_yaw=int(args.n_yaw), joint_names=names)
    q_bounds = urdf_position_limits(names, posture_limits=True)

    print(
        f"nominal cylinder: center={np.round(cyl.center, 3).tolist()}  "
        f"R={cyl.radius:.3f}  hh={cyl.half_height:.3f}"
    )

    q_arm = np.zeros((2, args.n_poses, 7))
    tip_torso = np.zeros((2, args.n_poses, 3))
    for hi, hand in enumerate(_HANDS):
        cands = _candidate_arms(
            hand, names, goal_lib, n_target=args.n_poses, seed=args.seed
        )
        if args.verbose:
            print(f"{hand}: {len(cands)} candidates before filter")
        qa, tips = _select_inside(
            hand, cands, fk=fk, cyl=cyl, names=names,
            n_keep=args.n_poses, q_bounds=q_bounds,
        )
        q_arm[hi] = qa
        tip_torso[hi] = tips
        outs = outside(cyl, tips)
        print(
            f"{hand}: kept {len(qa)} poses  "
            f"tip outside max={outs.max()*1e3:.2f} mm  "
            f"mean shoulder |dq| from ready="
            f"{np.mean(np.abs(qa[:, :_N_SHOULDER] - ready_joint_pos(names)[arm_slice(hand, names)][:_N_SHOULDER])):.3f} rad"
        )
        if args.verbose:
            for k in range(min(3, len(qa))):
                print(f"  [{k}] tip={np.round(tips[k], 3).tolist()}  q={np.round(qa[k], 3).tolist()}")

    lib = IdleCylinderLibrary(
        q_arm=q_arm, joint_names=names, cylinder=cyl, tip_torso=tip_torso
    )
    out = lib.save(args.out or default_idle_cylinder_path())
    print(f"wrote {out}  shape q_arm={tuple(q_arm.shape)}")


if __name__ == "__main__":
    main()
