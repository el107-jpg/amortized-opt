#!/usr/bin/env python
"""Phase-1 sanity: waist set-point follow + arms moving toward the drum.

Not a phase-2 gate. Collision / torque are expected to be untrained.

    python scripts/inspect_phase1.py --ckpt runs/phase1_waist_follow/best.pt \\
        --device cuda:1 --out runs/phase1_waist_follow/inspect
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A
from amortized_opt.train_warmstart import waist_follow_probe
from amortized_opt.warmstart_data import NoiseConfig, ScenarioConfig, generate_pool


def _attr(obj, *names, default=None):
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return default
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


JOINTS = list(_attr(A, "G1_UPPER_BODY_JOINTS", "G1_UPPER_BODY_JOINTS"))
YAW = JOINTS.index("waist_yaw_joint") if "waist_yaw_joint" in JOINTS else JOINTS.index("waist_yaw_joint")
PITCH = next(i for i, n in enumerate(JOINTS) if "pitch" in n and "waist" in n)
ROLL = next(i for i, n in enumerate(JOINTS) if "roll" in n and "waist" in n)
PELVIS = _attr(A, "PELVIS_IN_ENV", "PELVIS_IN_ENV")
ChainFK = _attr(A, "ChainFK", "ChainFK")
tip_position = _attr(A, "tip_position", "tip_position")
aim_yaw = _attr(A, "aim_yaw", "aim_yaw")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--grid", default="standard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--probe-scenarios", type=int, default=400)
    return p.parse_args(argv)


def _pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def _mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def load_model(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_fn = _attr(A, "model_config_from_checkpoint", "model_config_from_checkpoint")
    model_cfg = cfg_fn(ck)
    model = A.build_model(model_cfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    scen = ck.get("scenario_cfg") or ck.get("scenario_cfg") or {}
    buckets = tuple(scen.get("buckets", (25, 50, 100)))
    ease = scen.get("waist_upright_ease_wp", scen.get("waist_upright_ease_wp"))
    return model, model_cfg, ck, buckets, (None if ease is None else int(ease))


def run_probe(model, ck, dev, args):
    scen_d = ck.get("scenario_cfg") or ck.get("scenario_cfg") or {}
    H = int(scen_d.get("horizon", scen_d.get("horizon", 50)))
    buckets = tuple(scen_d.get("buckets", (25, 50)))
    scen = ScenarioConfig(horizon=H, buckets=buckets)
    fk = ChainFK(device=str(dev), dtype=torch.float32)
    rnea = A.BatchedRNEA.from_urdf(with_armature=True, device=str(dev), dtype=torch.float32)
    pool = generate_pool(
        int(args.probe_scenarios), seed=int(args.seed) + 7, cfg=scen,
        noise=NoiseConfig(), split="eval", workers=8, device=str(dev),
        fk=fk, rnea=rnea,
    )
    return waist_follow_probe(model, pool, batch_size=64, n_batches=6, seed=int(args.seed))


def tip_xyz(fk, Q, hand):
    q = torch.as_tensor(np.asarray(Q, float), device=fk.device, dtype=fk.dtype)
    return tip_position(fk, q, hand, frame="pelvis").detach().cpu().numpy()


def analyze(cases, trajs, fk):
    rows = []
    for case in cases:
        cid = int(_attr(case, "case_id", "case_id"))
        Q = trajs.get(cid)
        if Q is None:
            continue
        Q = np.asarray(Q, float)
        N = int(_attr(case, "N", "N"))
        k = min(N, Q.shape[0] - 1)
        strike = _attr(case, "strike", "strike")
        goal = np.asarray(strike.position, float)
        hand = str(_attr(case, "hand", "hand"))
        tips = tip_xyz(fk, Q, hand)
        d0 = float(np.linalg.norm(tips[0] - goal))
        dN = float(np.linalg.norm(tips[k] - goal))
        q0, qN = Q[0], Q[k]
        aim = float(aim_yaw({hand: goal + PELVIS}, yaw_current=float(q0[YAW])))
        rows.append(dict(
            case_id=cid,
            drum=str(_attr(case, "drum", "drum")),
            hand=hand,
            d0_m=d0,
            dN_m=dN,
            approach_m=d0 - dN,
            yaw0=float(q0[YAW]),
            yawN=float(qN[YAW]),
            aim=aim,
            yaw_err_end=abs(float(qN[YAW]) - aim),
            yaw_err_start=abs(float(q0[YAW]) - aim),
            lean0=float(np.hypot(q0[PITCH], q0[ROLL])),
            leanN=float(np.hypot(qN[PITCH], qN[ROLL])),
            pitchN=float(qN[PITCH]),
            rollN=float(qN[ROLL]),
        ))
    return rows


def summarize(tag, rows):
    d0 = [r["d0_m"] for r in rows]
    dN = [r["dN_m"] for r in rows]
    app = [r["approach_m"] for r in rows]
    yaw_err = [r["yaw_err_end"] for r in rows]
    yaw_err0 = [r["yaw_err_start"] for r in rows]
    lean0 = [r["lean0"] for r in rows]
    leanN = [r["leanN"] for r in rows]
    toward = float(np.mean([a > 0.02 for a in app])) if app else float("nan")
    return {
        "tag": tag,
        "n": len(rows),
        "tip_start_p50_mm": 1000 * _pct(d0, 50),
        "tip_end_p50_mm": 1000 * _pct(dN, 50),
        "tip_end_p90_mm": 1000 * _pct(dN, 90),
        "approach_p50_mm": 1000 * _pct(app, 50),
        "frac_moved_toward_drum": toward,
        "frac_end_lt_150mm": float(np.mean([d < 0.15 for d in dN])) if dN else float("nan"),
        "frac_end_lt_80mm": float(np.mean([d < 0.08 for d in dN])) if dN else float("nan"),
        "yaw_err_start_p50_deg": float(np.degrees(_pct(yaw_err0, 50))),
        "yaw_err_end_p50_deg": float(np.degrees(_pct(yaw_err, 50))),
        "yaw_err_end_p90_deg": float(np.degrees(_pct(yaw_err, 90))),
        "lean_start_p50_deg": float(np.degrees(_pct(lean0, 50))),
        "lean_end_p50_deg": float(np.degrees(_pct(leanN, 50))),
        "lean_end_p90_deg": float(np.degrees(_pct(leanN, 90))),
        "mean_abs_pitch_end_deg": float(np.degrees(_mean([abs(r["pitchN"]) for r in rows]))),
        "mean_abs_roll_end_deg": float(np.degrees(_mean([abs(r["rollN"]) for r in rows]))),
    }


def plot(out_dir, by_planner):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for tag, rows in by_planner.items():
        app = 1000 * np.array([r["approach_m"] for r in rows])
        ax[0].hist(app, bins=24, alpha=0.45, label=tag)
        ax[1].scatter(
            np.degrees([r["aim"] for r in rows]),
            np.degrees([r["yawN"] for r in rows]),
            s=10, alpha=0.5, label=tag,
        )
        ax[2].scatter(
            np.degrees([r["lean0"] for r in rows]),
            np.degrees([r["leanN"] for r in rows]),
            s=10, alpha=0.5, label=tag,
        )
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("tip approach start→contact (mm)")
    ax[0].set_title("arm toward drum (>0 is closer)")
    lim = [-90, 90]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set_xlim(*lim)
    ax[1].set_ylim(*lim)
    ax[1].set_xlabel("aim yaw (deg)")
    ax[1].set_ylabel("plan yaw at contact (deg)")
    ax[1].set_title("waist yaw vs aim")
    ax[2].plot([0, 40], [0, 40], "k--", lw=0.8)
    ax[2].set_xlabel("lean at start (deg)")
    ax[2].set_ylabel("lean at contact (deg)")
    ax[2].set_title("pitch/roll ease to upright")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase1_waist_arms.png", dpi=140)
    plt.close(fig)


def fmt(s):
    return (
        f"  {s['tag']:16s}  n={s['n']:3d}\n"
        f"    tip start/end p50  {s['tip_start_p50_mm']:.0f} → {s['tip_end_p50_mm']:.0f} mm"
        f"  (p90 end {s['tip_end_p90_mm']:.0f})\n"
        f"    approach p50       {s['approach_p50_mm']:.0f} mm   "
        f"toward drum {100 * s['frac_moved_toward_drum']:.0f}%   "
        f"end<150mm {100 * s['frac_end_lt_150mm']:.0f}%   "
        f"end<80mm {100 * s['frac_end_lt_80mm']:.0f}%\n"
        f"    yaw err start→end  {s['yaw_err_start_p50_deg']:.1f} → {s['yaw_err_end_p50_deg']:.1f} deg"
        f"  (p90 {s['yaw_err_end_p90_deg']:.1f})\n"
        f"    lean start→end     {s['lean_start_p50_deg']:.1f} → {s['lean_end_p50_deg']:.1f} deg"
        f"  (p90 {s['lean_end_p90_deg']:.1f}; |pitch| {s['mean_abs_pitch_end_deg']:.1f}, "
        f"|roll| {s['mean_abs_roll_end_deg']:.1f})"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out) if args.out else ckpt_path.parent / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_cfg, ck, buckets, ease = load_model(ckpt_path, dev)
    print(f"[ckpt] {ckpt_path} step={ck.get('step')} buckets={buckets} ease_wp={ease}",
          flush=True)

    print("[probe] causal token-read ...", flush=True)
    probe = run_probe(model, ck, dev, args)
    print(json.dumps(probe, indent=2), flush=True)

    amp = None if args.no_amp else torch.bfloat16
    scorer = A.TrajectoryScorer(device="cpu")
    cases = A.generate_grid(seed=args.seed, grid=args.grid, scorer=scorer)
    print(f"[grid] {args.grid}: {len(cases)} cases", flush=True)
    library = A.GoalPoseLibrary()
    kit = A.DrumKit.from_layout()
    n_drums = _attr(model_cfg, "num_drums", "num_drums")
    n_future = _attr(model_cfg, "num_future", "num_future")
    mp_kw = dict(
        model=model, device=str(dev), iterations=1, buckets=buckets,
        library=library, kit=kit, amp_dtype=amp, waist_upright_ease_wp=ease,
    )
    mp_fields = {f.name for f in dataclasses.fields(A.ModelPlanner)}
    if "num_drums" in mp_fields:
        mp_kw["num_drums"] = n_drums
        mp_kw["num_future"] = n_future
    else:
        mp_kw["num_drums"] = n_drums
        mp_kw["num_future"] = n_future
    planners = {
        "quintic": A.quintic_case_planner(library=library, waist_upright_ease_wp=ease),
        "model_quintic": A.ModelPlanner(init="quintic", **mp_kw),
        "model_hold": A.ModelPlanner(init="hold", **mp_kw),
    }

    fk = ChainFK(device=str(dev), dtype=torch.float32)
    report = {"ckpt": str(ckpt_path), "step": int(ck.get("step", -1)), "probe": probe,
              "planners": {}}
    by_rows = {}
    for name, plan in planners.items():
        print(f"[plan] {name} ...", flush=True)
        rows, trajs = A.run_eval(plan, cases, scorer=scorer)
        stats = analyze(cases, trajs, fk)
        summ = summarize(name, stats)
        contact = [float(_attr(r, "contact_pos_err_mm", "contact_pos_err_mm", default=np.nan))
                   for r in rows]
        summ["contact_pos_err_p50_mm"] = _pct(contact, 50)
        summ["contact_pos_err_p90_mm"] = _pct(contact, 90)
        print(fmt(summ), flush=True)
        print(f"    scorer contact p50/p90  {summ['contact_pos_err_p50_mm']:.0f}/"
              f"{summ['contact_pos_err_p90_mm']:.0f} mm  "
              "(phase 1: match quintic, ignore torque/collision)", flush=True)
        report["planners"][name] = summ
        by_rows[name] = stats

    plot(out_dir, by_rows)
    (out_dir / "inspect.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {out_dir / 'inspect.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
