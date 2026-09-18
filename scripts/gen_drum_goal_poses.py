#!/usr/bin/env python
"""Regenerate ``assets/drum_goal_poses.json`` after a stick-geometry change.

The library stores JOINT poses solved by the original repo's full planner
against the stick that was mounted at the time.  When
``amortized_opt/constants.py`` changes the stick (mount position / mount
rotation / length), every stored pose keeps its joints but its TIP moves, so
the library silently stops putting the tip where the author intended -- while
``GoalPoseLibrary(max_tip_err=...)`` keeps screening on the STORED ``tip_err``.

This script re-aims each entry at the point its author intended:

1.  The intended contact point of an entry is where its stored ``q`` put the
    tip under the OLD stick (hardcoded below -- the pre-2026-08 stick the
    shipped asset was solved against).  That bakes in whatever aim the original
    planner chose, including the cymbals' off-centre in-disk aims.
2.  Only the entry's own arm columns are DLS-corrected so the NEW stick's tip
    lands on that same point (waist frozen, joint box clamped) -- the same
    update rule as :func:`amortized_opt.ik.correct_tip`, and literally that
    function whenever the new geometry is the one in ``constants.py``.
3.  ``tip_err`` is re-stored with the original semantics: the planner's
    positional miss against its aim.  For membrane drums the aim is exactly
    ``strike_aim_on_head`` (the shipped asset matches it to 1e-10), so it is
    recomputed outright.  For the cymbals/hi-hat the planner aimed at its own
    in-disk point, which is not recoverable from the JSON; there the stored
    miss is carried over plus the (small) correction residual, an upper bound
    that can only make the ``max_tip_err`` filter stricter, never laxer.
    ``wrist_headroom`` / ``arm_travel`` are recomputed with the generator's
    formulas (verified bit-equal on the shipped asset).

The JSON schema is preserved key-for-key, so ``GoalPoseLibrary`` loads the
result unchanged.

Usage::

    # stats only, nothing written
    python scripts/gen_drum_goal_poses.py --check

    # regenerate in place (backs up the source first)
    python scripts/gen_drum_goal_poses.py

    # regenerate for the 2026-08-13 stick before constants.py carries it
    python scripts/gen_drum_goal_poses.py --geom aug13

The source of poses defaults to ``assets/drum_goal_poses.json.bak_oldstick``
when that backup exists (so re-runs never correct an already-corrected file),
else the live asset.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amortized_opt.constants import (  # noqa: E402
    ASSETS_DIR,
    G1_UPPER_BODY_JOINTS,
    PELVIS_IN_ENV,
    STICK_MOUNT_POS_LEFT,
    STICK_MOUNT_POS_RIGHT,
    STICK_MOUNT_ROT_LEFT_WXYZ,
    STICK_MOUNT_ROT_RIGHT_WXYZ,
    STICK_TIP_SIGMA,
    ready_joint_pos,
)
from amortized_opt.fk import ChainFK  # noqa: E402
from amortized_opt.ik import arm_joint_indices, correct_tip  # noqa: E402
from amortized_opt.limits import urdf_position_limits  # noqa: E402
from amortized_opt.rhythm import DrumKit  # noqa: E402
from amortized_opt.strike import strike_aim_on_head  # noqa: E402

NAMES = G1_UPPER_BODY_JOINTS
DT = torch.float64
_HANDS = ("left", "right")

# When |old-tip - strike_aim_on_head| reproduces the stored tip_err this
# tightly, the entry's aim is the canonical one and tip_err is recomputed
# exactly; otherwise (cymbals/hi-hat: planner-internal in-disk aim) the stored
# value is carried over plus the correction residual.
_AIM_MATCH_TOL = 1.0e-6


# ------------------------------------------------------------------ geometry --
def _quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _tip_local(mount_pos, mount_quat_wxyz, sigma: float) -> np.ndarray:
    """Stick-tip offset in the wrist-yaw frame for an explicit mount."""
    return np.asarray(mount_pos, float) + _quat_wxyz_to_R(mount_quat_wxyz) @ np.array(
        [0.0, 0.0, float(sigma)]
    )


#: The stick the shipped asset was solved against (constants.py before the
#: 2026-08 change: TIP_RADIUS 0.014, mount y +/-0.06, mount quats with the
#: +/-0.163718 x/z components).  TIP_RADIUS does not enter the tip point.
OLD_GEOM = {
    "left": _tip_local((0.2, -0.06, 0.0), (0.656636, 0.163718, 0.717789, -0.163718), 0.2),
    "right": _tip_local((0.2, 0.06, 0.0), (0.656636, -0.163718, 0.717789, 0.163718), 0.2),
}

#: The 2026-08-13 stick (TIP_RADIUS 0.010, mount y +/-0.03, x/z quat
#: components removed), spelled out so the asset can be regenerated for it
#: even while constants.py in the working tree still carries the old stick.
AUG13_GEOM = {
    "left": _tip_local((0.2, -0.03, 0.0), (0.656636, 0.0, 0.717789, 0.0), 0.2),
    "right": _tip_local((0.2, 0.03, 0.0), (0.656636, 0.0, 0.717789, 0.0), 0.2),
}


def _constants_geom() -> dict[str, np.ndarray]:
    return {
        "left": _tip_local(STICK_MOUNT_POS_LEFT, STICK_MOUNT_ROT_LEFT_WXYZ, STICK_TIP_SIGMA),
        "right": _tip_local(STICK_MOUNT_POS_RIGHT, STICK_MOUNT_ROT_RIGHT_WXYZ, STICK_TIP_SIGMA),
    }


def _geoms_equal(a: dict, b: dict, tol: float = 1e-9) -> bool:
    return all(float(np.linalg.norm(a[h] - b[h])) <= tol for h in _HANDS)


# ------------------------------------------------------------- FK / correction --
def _tip_env(fk: ChainFK, q: torch.Tensor, hand: str, local: np.ndarray) -> torch.Tensor:
    """(B, 3) env-frame tip for an EXPLICIT wrist-frame offset ``local``."""
    R, p = fk.frame(q, f"{hand}_wrist")
    off = torch.as_tensor(local.tolist(), dtype=q.dtype)
    env = torch.as_tensor(PELVIS_IN_ENV.tolist(), dtype=q.dtype)
    return p + torch.einsum("bij,j->bi", R, off) + env


def _correct_tip_geom(
    fk: ChainFK,
    q: torch.Tensor,
    aim: torch.Tensor,
    hand: str,
    local: np.ndarray,
    *,
    steps: int,
    damping: float,
    q_bounds,
) -> torch.Tensor:
    """``ik.correct_tip`` with an explicit tip offset instead of constants'.

    Same update rule step for step (DLS on the arm columns, clamped to the
    joint box); needed only while the target geometry is not the one
    ``constants.py`` describes.
    """
    cols = torch.as_tensor(arm_joint_indices(hand, NAMES))
    out = q.clone()
    lam2 = float(damping) ** 2
    eye = torch.eye(3, dtype=q.dtype)
    lo = torch.as_tensor(q_bounds[0], dtype=q.dtype).index_select(0, cols)
    hi = torch.as_tensor(q_bounds[1], dtype=q.dtype).index_select(0, cols)
    for _ in range(int(steps)):
        with torch.no_grad():
            err = aim - _tip_env(fk, out, hand, local)
        qg = out.detach().requires_grad_(True)
        pos = _tip_env(fk, qg, hand, local)
        rows = []
        for axis in range(3):
            (g,) = torch.autograd.grad(pos[:, axis].sum(), qg, retain_graph=axis < 2)
            rows.append(g)
        J = torch.stack(rows, dim=1).index_select(2, cols)
        JT = J.transpose(1, 2)
        dq = (JT @ torch.linalg.solve(J @ JT + lam2 * eye, err.unsqueeze(-1))).squeeze(-1)
        upd = torch.clamp(out.index_select(1, cols) + dq, lo, hi)
        out = out.index_copy(1, cols, upd)
    return out.detach()


# ------------------------------------------------------------------- library --
def _vec(qd: dict) -> np.ndarray:
    return np.array([float(qd.get(n, 0.0)) for n in NAMES])


def _write_back(qd: dict, q: np.ndarray) -> None:
    for i, n in enumerate(NAMES):
        if n in qd:
            qd[n] = float(q[i])


def _wrist_headroom(q: np.ndarray, hand: str, lo: np.ndarray, hi: np.ndarray) -> float:
    idx = [i for i, n in enumerate(NAMES) if n.startswith(f"{hand}_wrist")]
    return float(min(min(q[i] - lo[i], hi[i] - q[i]) for i in idx))


def _arm_travel(q: np.ndarray, hand: str, q_ready: np.ndarray) -> float:
    idx = [
        i
        for i, n in enumerate(NAMES)
        if n.startswith(hand) and ("shoulder" in n or "elbow" in n)
    ]
    return float(np.max(np.abs(q[idx] - q_ready[idx])))


def _aims(kit: DrumKit) -> dict[str, np.ndarray]:
    out = {}
    for i, name in enumerate(kit.names):
        aim, _r = strike_aim_on_head(
            kit.positions[i], kit.axes[i], float(kit.radii[i]), name=name
        )
        out[name] = aim
    return out


def _collect_jobs(data: dict) -> list[dict]:
    """Every stored pose, flat: yaw_grid rows, pair halves, base poses."""
    jobs = []
    for hand, per_drum in data.get("yaw_grid", {}).items():
        for drum, rows in per_drum.items():
            for k, row in enumerate(rows):
                jobs.append(
                    {"kind": "yaw_grid", "hand": hand, "drum": drum,
                     "q": _vec(row["q"]), "stored_err": float(row["tip_err"]),
                     "ref": (hand, drum, k)}
                )
    for key, pair in data.get("pairs", {}).items():
        left_drum, right_drum = key.split("|")
        per = {"left": left_drum, "right": right_drum}
        diag = data.get("pair_diagnostics", {}).get(key, {})
        for hand in _HANDS:
            if hand not in pair or not per[hand]:
                continue
            stored = diag.get("tip_err", {}).get(hand, float("nan"))
            jobs.append(
                {"kind": "pairs", "hand": hand, "drum": per[hand],
                 "q": _vec(pair[hand]), "stored_err": float(stored),
                 "ref": (key, hand)}
            )
    for hand, per_drum in data.get("poses", {}).items():
        for drum, qd in per_drum.items():
            jobs.append(
                {"kind": "poses", "hand": hand, "drum": drum,
                 "q": _vec(qd), "stored_err": float("nan"), "ref": (hand, drum)}
            )
    return jobs


def _pct(v: np.ndarray, p: float) -> float:
    return float(np.percentile(v, p)) if v.size else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="print before/after tip-error stats only; write nothing")
    ap.add_argument("--src", type=Path, default=None,
                    help="old-stick library to correct (default: the "
                         ".bak_oldstick backup if present, else the live asset)")
    ap.add_argument("--out", type=Path, default=ASSETS_DIR / "drum_goal_poses.json")
    ap.add_argument("--geom", choices=("constants", "aug13"), default="constants",
                    help="NEW stick geometry: 'constants' = whatever "
                         "amortized_opt.constants currently says (the normal, "
                         "tested path), 'aug13' = the 2026-08-13 stick spelled "
                         "out above, for regenerating ahead of constants.py")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--damping", type=float, default=0.05)
    args = ap.parse_args(argv)

    live = ASSETS_DIR / "drum_goal_poses.json"
    bak = live.with_name(live.name + ".bak_oldstick")
    src = args.src if args.src is not None else (bak if bak.exists() else live)
    data = json.loads(Path(src).read_text())
    print(f"[gen_drum_goal_poses] source: {src}")

    cgeom = _constants_geom()
    new_geom = cgeom if args.geom == "constants" else AUG13_GEOM
    use_library_correct = _geoms_equal(new_geom, cgeom)
    for h in _HANDS:
        print(f"[geom] {h:5s} old tip offset {np.round(OLD_GEOM[h], 4)} -> "
              f"new {np.round(new_geom[h], 4)} "
              f"(shift {np.linalg.norm(new_geom[h] - OLD_GEOM[h]):.4f} m)")
    if _geoms_equal(new_geom, OLD_GEOM):
        msg = ("the requested NEW geometry equals the OLD stick -- constants.py "
               "still carries the pre-change stick.  Regenerating would be a "
               "no-op; land the constants change or pass --geom aug13.")
        if args.check:
            print(f"[gen_drum_goal_poses] WARNING: {msg}")
        else:
            print(f"[gen_drum_goal_poses] REFUSING TO WRITE: {msg}", file=sys.stderr)
            return 2
    print(f"[geom] correction path: "
          f"{'ik.correct_tip (constants)' if use_library_correct else 'explicit-geometry DLS mirror'}")

    fk = ChainFK(device="cpu", dtype=DT)
    lo, hi = urdf_position_limits(NAMES)
    q_bounds = (lo, hi)
    q_ready = ready_joint_pos(NAMES)
    kit = DrumKit.from_layout()
    aims = _aims(kit)

    # Self-check: the explicit-geometry FK/DLS must agree with the library path
    # whenever it is describing the constants stick.
    probe = torch.as_tensor(np.tile(q_ready, (2, 1)), dtype=DT)
    from amortized_opt.ik import tip_position as _lib_tip
    for h in _HANDS:
        d = (_tip_env(fk, probe, h, cgeom[h]) -
             _lib_tip(fk, probe, h, frame="env")).norm(dim=-1).max()
        if float(d) > 1e-9:
            raise AssertionError(f"explicit-geometry FK disagrees with ik.tip_position ({h}: {float(d)})")

    jobs = _collect_jobs(data)
    out_data = copy.deepcopy(data)

    stats: dict[str, dict[str, list]] = {h: {"stale": [], "resid": [], "stored": [], "new_err": []}
                                         for h in _HANDS}
    n_carry = 0
    for hand in _HANDS:
        hj = [j for j in jobs if j["hand"] == hand]
        if not hj:
            continue
        Q = torch.as_tensor(np.stack([j["q"] for j in hj]), dtype=DT)
        with torch.no_grad():
            intended = _tip_env(fk, Q, hand, OLD_GEOM[hand])       # author's contact point
            stale = _tip_env(fk, Q, hand, new_geom[hand])          # new stick, stored q
        if use_library_correct:
            Qc = correct_tip(fk, Q, intended, hand, steps=args.steps,
                             damping=args.damping, q_bounds=q_bounds)
        else:
            Qc = _correct_tip_geom(fk, Q, intended, hand, new_geom[hand],
                                   steps=args.steps, damping=args.damping,
                                   q_bounds=q_bounds)
        with torch.no_grad():
            fixed = _tip_env(fk, Qc, hand, new_geom[hand])
        stale_miss = (stale - intended).norm(dim=-1).numpy()
        resid = (fixed - intended).norm(dim=-1).numpy()
        intended_np = intended.numpy()
        fixed_np = fixed.numpy()
        Qc_np = Qc.numpy()

        for i, j in enumerate(hj):
            aim = aims[j["drum"]]
            d_old = float(np.linalg.norm(intended_np[i] - aim))
            stored = j["stored_err"]
            if np.isfinite(stored) and abs(d_old - stored) <= _AIM_MATCH_TOL:
                new_err = float(np.linalg.norm(fixed_np[i] - aim))   # exact semantics
            elif np.isfinite(stored):
                new_err = stored + float(resid[i])                    # planner-internal aim
                n_carry += 1
            else:
                new_err = float("nan")
            j["q_new"], j["new_err"], j["resid"] = Qc_np[i], new_err, float(resid[i])
            stats[hand]["stale"].append(stale_miss[i])
            stats[hand]["resid"].append(resid[i])
            if np.isfinite(stored):
                stats[hand]["stored"].append(stored)
                stats[hand]["new_err"].append(new_err)

    # ------------------------------------------------------------- reporting --
    for hand in _HANDS:
        s = {k: np.asarray(v, float) for k, v in stats[hand].items()}
        if not s["stale"].size:
            continue
        print(f"\n[{hand}] {s['stale'].size} stored poses")
        print(f"  stale miss  (new stick @ stored q vs intended point): "
              f"p50 {_pct(s['stale'], 50):.4f}  p95 {_pct(s['stale'], 95):.4f}  "
              f"max {s['stale'].max():.4f} m")
        print(f"  post-corr residual (new tip vs intended point):       "
              f"p50 {_pct(s['resid'], 50):.4f}  p95 {_pct(s['resid'], 95):.4f}  "
              f"max {s['resid'].max():.4f} m")
        print(f"  tip_err stored  p50 {_pct(s['stored'], 50):.4f}  ->  "
              f"regenerated p50 {_pct(s['new_err'], 50):.4f} m")
    print(f"\n[tip_err] {n_carry} cymbal/hi-hat entries keep the stored miss "
          f"(+ residual); the rest are recomputed against strike_aim_on_head")

    # max_tip_err filter health, before vs after, per (hand, drum)
    max_tip_err = 0.12
    print(f"\n[filter] yaw_grid entries passing tip_err <= {max_tip_err}:")
    bad = False
    for hand, per_drum in data.get("yaw_grid", {}).items():
        for drum in per_drum:
            rows = [j for j in jobs
                    if j["kind"] == "yaw_grid" and j["hand"] == hand and j["drum"] == drum]
            was = sum(1 for j in rows if j["stored_err"] <= max_tip_err)
            now = sum(1 for j in rows if j["new_err"] <= max_tip_err)
            flag = ""
            if now == 0:
                flag = "  <-- NO VALID ENTRY LEFT"
                bad = True
            print(f"  {hand:5s} {drum:12s} {was:2d}/{len(rows)} -> {now:2d}/{len(rows)}{flag}")
    if bad:
        print("[filter] WARNING: some (hand, drum) pairs lost all sub-threshold "
              "entries; GoalPoseLibrary.select will fail for them", file=sys.stderr)

    if args.check:
        print("\n[gen_drum_goal_poses] --check: nothing written")
        return 0

    # ------------------------------------------------------------ write back --
    lo_np, hi_np = lo, hi
    for j in jobs:
        q_new = j["q_new"]
        if j["kind"] == "yaw_grid":
            hand, drum, k = j["ref"]
            row = out_data["yaw_grid"][hand][drum][k]
            _write_back(row["q"], q_new)
            row["tip_err"] = float(j["new_err"])
            row["wrist_headroom"] = _wrist_headroom(q_new, hand, lo_np, hi_np)
        elif j["kind"] == "pairs":
            key, hand = j["ref"]
            _write_back(out_data["pairs"][key][hand], q_new)
            diag = out_data.get("pair_diagnostics", {}).get(key)
            if diag is not None and hand in diag.get("tip_err", {}):
                diag["tip_err"][hand] = float(j["new_err"])
        else:  # poses
            hand, drum = j["ref"]
            _write_back(out_data["poses"][hand][drum], q_new)
    # pair-level scalars from the corrected poses
    for key, diag in out_data.get("pair_diagnostics", {}).items():
        pair = out_data["pairs"][key]
        hands = [h for h in _HANDS if h in pair]
        qs = {h: _vec(pair[h]) for h in hands}
        if "wrist_headroom" in diag:
            diag["wrist_headroom"] = min(_wrist_headroom(qs[h], h, lo_np, hi_np) for h in hands)
        if "arm_travel" in diag:
            diag["arm_travel"] = max(_arm_travel(qs[h], h, q_ready) for h in hands)

    out = Path(args.out)
    if out.resolve() == live.resolve() and not bak.exists():
        shutil.copy2(live, bak)
        print(f"\n[gen_drum_goal_poses] backed up old-stick asset -> {bak}")
    out.write_text(json.dumps(out_data, indent=2, sort_keys=True) + "\n")
    print(f"[gen_drum_goal_poses] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
