#!/usr/bin/env python
"""Section 7d -- how much waist does a checkpoint actually use?

The point of the waist-upright fine-tune is a posture claim, and a posture claim
needs a posture measurement.  This script reports, for one or more checkpoints,
the distribution of the three waist joints over real trajectories:

* **open loop** -- every plan the model produces on an acceptance grid, scored
  over the plan's own valid waypoints (waypoint 0..N, the window the cost sees);
* **closed loop** -- the measured plant state ``res.q_hist`` over full
  ``MPCGenerator`` episodes with the deployed per-joint reference lead, which is
  what the rendered videos show and what Eric was looking at.

Reported per joint: p50 / p90 / max of ``|q|`` pooled over all samples, plus
**travel** -- ``sum_t |q_t - q_{t-1}|`` per episode (open loop: per plan) -- which
separates "leans a long way once" from "waves back and forth".  Yaw is reported
alongside pitch/roll precisely because it is supposed to be UNCHANGED.

    python scripts/eval_waist.py --ckpt runs/phase2b_main/best.pt \
        --ckpt runs/phase2d_waist/best.pt --device cuda:1 \
        --out runs/final/waist.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.lead import LEAD_PRESETS  # noqa: E402
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator  # noqa: E402
from amortized_opt.probe import mpc_model_planner  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

WAIST = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")
WCOL = {n: A.G1_UPPER_BODY_JOINTS.index(n) for n in WAIST}
SHORT = {"waist_yaw_joint": "yaw", "waist_pitch_joint": "pitch",
         "waist_roll_joint": "roll"}


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    return model, {"buckets": buckets, "num_drums": mcfg.num_drums,
                   "num_future": mcfg.num_future}, int(ck["step"])


def waist_stats(series: list[np.ndarray]) -> dict:
    """``series`` is a list of ``(T, 17)`` trajectories -- one per plan / episode.

    Excursion percentiles pool every sample; travel is per trajectory, so its
    percentiles are over trajectories.  Both matter: a p90 of 0.4 rad with tiny
    travel is a static lean, the same p90 with large travel is a sway.
    """
    out: dict[str, float] = {"n_traj": len(series),
                             "n_samples": int(sum(len(s) for s in series))}
    for name, col in WCOL.items():
        tag = SHORT[name]
        allq = np.concatenate([np.abs(s[:, col]) for s in series]) if series else np.zeros(1)
        trav = np.array([np.abs(np.diff(s[:, col])).sum() for s in series if len(s) > 1])
        out[f"{tag}_abs_p50"] = float(np.percentile(allq, 50))
        out[f"{tag}_abs_p90"] = float(np.percentile(allq, 90))
        out[f"{tag}_abs_max"] = float(allq.max())
        out[f"{tag}_travel_p50"] = float(np.percentile(trav, 50)) if trav.size else float("nan")
        out[f"{tag}_travel_mean"] = float(trav.mean()) if trav.size else float("nan")
    # fraction of samples where the LEAN (pitch,roll) norm exceeds a threshold --
    # a single number for "is it upright", independent of which way it leans
    if series:
        pr = np.concatenate([
            np.hypot(s[:, WCOL["waist_pitch_joint"]], s[:, WCOL["waist_roll_joint"]])
            for s in series
        ])
        out["lean_norm_p50"] = float(np.percentile(pr, 50))
        out["lean_norm_p90"] = float(np.percentile(pr, 90))
        for th in (0.15, 0.20, 0.30):
            out[f"frac_lean_over_{th:g}"] = float((pr > th).mean())
    return out


# ------------------------------------------------------------------ open loop --
@torch.no_grad()
def open_loop(model, cfg, kit, library, scorer, dev, amp, *, grid, seed):
    cases = A.generate_grid(seed=seed, grid=grid, scorer=scorer)
    plans, _wall = A.batched_model_plans(
        model, cases, device=str(dev), kit=kit, library=library,
        buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
        num_future=cfg["num_future"], iterations=1, init="quintic", amp_dtype=amp,
    )
    # Only the VALID window: the plan array is padded to the bucket, and the
    # padding is an unpriced hold that would flatten every percentile.
    series = [np.asarray(plans[int(c.case_id)], float)[: int(c.N) + 1] for c in cases]
    return waist_stats(series), len(cases)


# ---------------------------------------------------------------- closed loop --
def closed_loop(model, cfg, kit, dev, amp, *, clips, probe_seeds, steps, cmd_lead):
    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    series, eps = [], []
    for ps in probe_seeds:
        starts = sampler.sample(len(clips), seed=int(ps))
        for ep, clip in enumerate(clips):
            gen = MPCGenerator(
                kit=kit,
                config=MPCGenConfig(horizon=cfg["buckets"][-1], cmd_lead_s=cmd_lead,
                                    sticky_goal_point=True),
            )
            plan, _st = mpc_model_planner(
                model, num_drums=cfg["num_drums"], num_future=cfg["num_future"],
                device=dev, amp_dtype=amp, iterations=1, record=None,
            )
            res = gen.run_episode(clip, starts.q[ep].cpu().numpy(),
                                  starts.qd[ep].cpu().numpy(),
                                  seed=1000 + 100 * int(ps) + ep, planner=plan,
                                  max_steps=int(steps))
            q = np.asarray(res.q_hist, float)
            series.append(q)
            eps.append({"clip": clip.name, "probe_seed": int(ps),
                        "ticks": int(res.cmd_hist.shape[0]),
                        "diverged": bool(res.diverged),
                        **{f"{SHORT[n]}_abs_max": float(np.abs(q[:, c]).max())
                           for n, c in WCOL.items()}})
            print(f"    {clip.name} s{ps}: ticks {eps[-1]['ticks']} "
                  f"|pitch|max {eps[-1]['pitch_abs_max']:.3f} "
                  f"|roll|max {eps[-1]['roll_abs_max']:.3f} "
                  f"|yaw|max {eps[-1]['yaw_abs_max']:.3f}", flush=True)
    out = waist_stats(series)
    out["episodes"] = eps
    return out


# ------------------------------------------------------------ where it enters --
def diagnose_lean(model, cfg, kit, dev, amp, *, clip, seed, steps, cmd_lead):
    """Split the closed-loop waist excursion into its three possible sources.

    At every tick we have four waist states for the SAME instant:

    ``plan_wp1``  what the planner asked for one control period ahead -- the
                  only thing ``FusedCost`` ever prices;
    ``cmd``       what ``run_episode`` actually commanded, i.e. ``plan_wp1``
                  after the reference LEAD, the slew limiter and the box clamp;
    ``q``         what the plant reached.

    ``cmd - plan_wp1`` is the lead (plus clamping); ``q - cmd`` is servo
    tracking error.  If both are small the lean is the PLAN's own -- receding
    -horizon drift, each tick planning a small lean from a state that already
    has one.
    """
    gen = MPCGenerator(
        kit=kit,
        config=MPCGenConfig(horizon=cfg["buckets"][-1], cmd_lead_s=cmd_lead,
                            sticky_goal_point=True),
    )
    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    starts = sampler.sample(1, seed=int(seed))
    record: list = []
    plan, _st = mpc_model_planner(
        model, num_drums=cfg["num_drums"], num_future=cfg["num_future"],
        device=dev, amp_dtype=amp, iterations=1, record=record,
    )
    res = gen.run_episode(clip, starts.q[0].cpu().numpy(), starts.qd[0].cpu().numpy(),
                          seed=1000, planner=plan, max_steps=int(steps))
    q = np.asarray(res.q_hist, float)
    cmd = np.asarray(res.cmd_hist, float)
    T = cmd.shape[0]
    wp1 = np.full((T, 17), np.nan)
    for t, r in enumerate(record[:T]):
        tr = np.asarray(r["traj"], float)
        if tr.ndim == 2 and tr.shape[0] >= 1:
            wp1[t] = tr[0]
    ok = ~np.isnan(wp1[:, 0])
    out = {"clip": clip.name, "ticks": int(T), "n_plans": int(ok.sum())}
    for name, col in WCOL.items():
        tag = SHORT[name]
        a, b, c = wp1[ok, col], cmd[ok, col], q[1:T + 1][ok, col]
        out[f"{tag}_plan_abs_p50"] = float(np.percentile(np.abs(a), 50))
        out[f"{tag}_plan_abs_p90"] = float(np.percentile(np.abs(a), 90))
        out[f"{tag}_cmd_abs_p50"] = float(np.percentile(np.abs(b), 50))
        out[f"{tag}_cmd_abs_p90"] = float(np.percentile(np.abs(b), 90))
        out[f"{tag}_meas_abs_p50"] = float(np.percentile(np.abs(c), 50))
        out[f"{tag}_meas_abs_p90"] = float(np.percentile(np.abs(c), 90))
        out[f"{tag}_lead_gap_p50"] = float(np.percentile(np.abs(b - a), 50))
        out[f"{tag}_track_gap_p50"] = float(np.percentile(np.abs(c - b), 50))
    return out


def _fmt(tag: str, s: dict) -> str:
    return (f"{tag:>26}  "
            + "  ".join(
                f"{SHORT[n]} {s[SHORT[n]+'_abs_p50']:.3f}/{s[SHORT[n]+'_abs_p90']:.3f}/"
                f"{s[SHORT[n]+'_abs_max']:.3f} tv {s[SHORT[n]+'_travel_p50']:.2f}"
                for n in WAIST)
            + f"  lean p90 {s['lean_norm_p90']:.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="repeatable: label=path, or just path")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--grid-seeds", type=int, nargs="+", default=[0, 7])
    ap.add_argument("--grids", nargs="+", default=["standard", "near_drum"])
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lead", default="fit", help="LEAD_PRESETS key, or a float")
    ap.add_argument("--skip-closed-loop", action="store_true")
    ap.add_argument("--skip-open-loop", action="store_true")
    ap.add_argument("--diagnose-lean", action="store_true",
                    help="split the closed-loop waist excursion into plan / "
                         "lead / servo-tracking contributions on one clip")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="runs/final/waist.json")
    args = ap.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16
    cmd_lead = LEAD_PRESETS[args.lead] if args.lead in LEAD_PRESETS else float(args.lead)

    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    scorer = A.TrajectoryScorer(device="cpu")
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.clips]

    report: dict = {"lead": args.lead, "grids": args.grids,
                    "grid_seeds": args.grid_seeds, "clips": [c.name for c in clips],
                    "probe_seeds": args.probe_seeds, "models": {}}
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        model, cfg, step = load_model(p, dev)
        print(f"\n=== {label}  ({path}, step {step}) ===", flush=True)
        block: dict = {"path": str(path), "step": step, "open_loop": {}}
        if not args.skip_open_loop:
            for gs in args.grid_seeds:
                for grid in args.grids:
                    s, n = open_loop(model, cfg, kit, library, scorer, dev, amp,
                                     grid=grid, seed=gs)
                    block["open_loop"][f"{grid}_seed{gs}"] = s
                    print("  " + _fmt(f"OL {grid} s{gs} ({n})", s), flush=True)
        if args.diagnose_lean:
            block["lean_sources"] = {}
            for tag, cl in (("lead_fit", cmd_lead), ("no_lead", 0.0)):
                d = diagnose_lean(model, cfg, kit, dev, amp, clip=clips[0],
                                  seed=0, steps=args.steps, cmd_lead=cl)
                block["lean_sources"][tag] = d
                for j in ("pitch", "roll", "yaw"):
                    print(f"    [{tag:8s}] {j:5s} |q| p50 plan {d[j+'_plan_abs_p50']:.3f}"
                          f" -> cmd {d[j+'_cmd_abs_p50']:.3f} -> meas {d[j+'_meas_abs_p50']:.3f}"
                          f"   (p90 {d[j+'_plan_abs_p90']:.3f}/{d[j+'_cmd_abs_p90']:.3f}"
                          f"/{d[j+'_meas_abs_p90']:.3f})"
                          f"  gaps lead {d[j+'_lead_gap_p50']:.4f} track"
                          f" {d[j+'_track_gap_p50']:.4f}", flush=True)
        if not args.skip_closed_loop:
            print("  closed loop:", flush=True)
            block["closed_loop"] = closed_loop(
                model, cfg, kit, dev, amp, clips=clips,
                probe_seeds=args.probe_seeds, steps=args.steps, cmd_lead=cmd_lead)
            print("  " + _fmt("CL (plant q_hist)", block["closed_loop"]), flush=True)
        report["models"][label] = block

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\n[out] {out}")

    print("\n===== waist usage:  |q| p50 / p90 / max  (rad), tv = travel p50 =====")
    for label, b in report["models"].items():
        for k, s in b["open_loop"].items():
            print(_fmt(f"{label} OL {k}", s))
        if "closed_loop" in b:
            print(_fmt(f"{label} CLOSED LOOP", b["closed_loop"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
