#!/usr/bin/env python
"""Section 8 -- the final closed-loop evaluation (deployment condition).

The 243-case grid scores a *plan*; this scores what the PLANT does.  Section 7c
established that the ranking metric is the **hit-latency distribution** -- for
every scheduled strike, when did the stick actually cross the head plane inside
the legal disk, to sub-tick resolution -- because "in the disk at tick k" is a
binary on one 20 ms sample of a moving stick and conflates *late* with *missed*.

What this adds over ``eval_phase2.py --closed-loop``:

* **all four** eval clips rather than three, and 3 start seeds per clip, so the
  probe noise 7c decision 6 warns about is bounded rather than assumed;
* the raw per-strike rows are kept, so latency can be broken down **per hand**
  and **per drum** (the grid's ``tom2`` weak spot has a closed-loop twin);
* collision is reported from the loop's own recorded ``min_sdf`` (the measured
  pose, with the strike / departure exemptions), which the probe summary drops.

Configs are ``(label, model_key, kwargs)``; the champion is the phase-2b
checkpoint with the per-joint reference lead and nothing else, per 7c decision 2.

    python scripts/eval_section8.py --device cuda:1 --out runs/final/closed_loop.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.lead import LEAD_PRESETS  # noqa: E402
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator  # noqa: E402
from amortized_opt.probe import (  # noqa: E402
    hit_latency,
    mpc_model_planner,
    score_hits,
    summarize_hits,
    summarize_latency,
)

ROOT = Path(__file__).resolve().parent.parent


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    return model, {"buckets": buckets, "num_drums": mcfg.num_drums,
                   "num_future": mcfg.num_future}, int(ck["step"])


def pct(v, q):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return float(np.percentile(v, q)) if v.size else float("nan")


def run_config(model, cfg, kit, dev, amp, clips, *, steps, probe_seeds, cmd_lead,
               freeze_wp=0, lead_s=0.0, iterations=1, label=""):
    """Every clip x every start seed.  Returns raw rows plus per-episode stats."""
    scorer = A.TrajectoryScorer(device="cpu")
    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    lead_ticks = int(round(float(lead_s) / A.CONTROL_DT))

    lat_rows, hit_rows, episodes = [], [], []
    t_all = time.perf_counter()
    for ps in probe_seeds:
        starts = sampler.sample(len(clips), seed=int(ps))
        for ep, clip in enumerate(clips):
            gen = MPCGenerator(
                kit=kit,
                config=MPCGenConfig(horizon=cfg["buckets"][-1],
                                    hit_lead_s=float(lead_s),
                                    cmd_lead_s=cmd_lead,
                                    freeze_wp=int(freeze_wp),
                                    sticky_goal_point=True),
            )
            record: list = []
            plan, st = mpc_model_planner(
                model, num_drums=cfg["num_drums"], num_future=cfg["num_future"],
                device=dev, amp_dtype=amp, iterations=iterations, record=record,
            )
            t0 = time.perf_counter()
            res = gen.run_episode(clip, starts.q[ep].cpu().numpy(),
                                  starts.qd[ep].cpu().numpy(),
                                  seed=1000 + 100 * int(ps) + ep, planner=plan,
                                  max_steps=int(steps))
            wall = time.perf_counter() - t0
            lr = hit_latency(scorer, kit, res, lead_ticks=lead_ticks)
            hr = score_hits(scorer, kit, res, record, lead_ticks=lead_ticks)
            for r in lr:
                r["clip"], r["probe_seed"] = clip.name, int(ps)
            lat_rows.extend(lr)
            hit_rows.extend(hr)
            track = (np.abs(res.q_hist[1:] - res.cmd_hist)
                     if len(res.cmd_hist) else np.zeros(1))
            sdf = np.asarray([s.min_sdf for s in res.samples], float)
            sdf = sdf[np.isfinite(sdf)]
            L = summarize_latency(lr)
            episodes.append({
                "clip": clip.name, "probe_seed": int(ps),
                "ticks": int(res.cmd_hist.shape[0]), "hits": len(lr),
                "diverged": bool(res.diverged),
                "crossed_in_disk": L["crossed_in_disk"],
                "lat_abs_p50_ms": L["lat_abs_p50_ms"],
                "pd_track_p50_mrad": float(np.percentile(track, 50) * 1000),
                "pd_track_p90_mrad": float(np.percentile(track, 90) * 1000),
                "tc_drift_p50_mrad": (float(np.percentile(st["drift"], 50) * 1000)
                                      if st["drift"] else float("nan")),
                "tc_drift_p90_mrad": (float(np.percentile(st["drift"], 90) * 1000)
                                      if st["drift"] else float("nan")),
                "min_sdf_p10_mm": 1000 * pct(sdf, 10) if sdf.size else float("nan"),
                "min_sdf_p50_mm": 1000 * pct(sdf, 50) if sdf.size else float("nan"),
                "ticks_penetrating": int((sdf < 0).sum()),
                "n_sdf": int(sdf.size),
                "ms_per_tick": 1000 * wall / max(res.cmd_hist.shape[0], 1),
            })
            print(f"    [{label}] {clip.name} s{ps}: ticks {episodes[-1]['ticks']} "
                  f"hits {len(lr)} crossed {L['crossed_in_disk']} "
                  f"|lat|p50 {L['lat_abs_p50_ms']:.1f} ms "
                  f"pen {episodes[-1]['ticks_penetrating']}/{episodes[-1]['n_sdf']}",
                  flush=True)

    out = {"label": label, "latency": summarize_latency(lat_rows),
           "at_tick": {k: v for k, v in summarize_hits(hit_rows).items()
                       if k != "planned_inplane_p50_by_h"},
           "planned_inplane_p50_by_h":
               summarize_hits(hit_rows)["planned_inplane_p50_by_h"],
           "episodes": episodes, "wall_s": time.perf_counter() - t_all}

    # ---- breakdowns ---------------------------------------------------------
    def sub(rows):
        return summarize_latency(rows)

    for key in ("hand", "drum", "clip", "probe_seed"):
        g = defaultdict(list)
        for r in lat_rows:
            g[r[key]].append(r)
        out[f"by_{key}"] = {str(k): sub(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}

    # aggregate plant health over episodes
    for f in ("pd_track_p50_mrad", "pd_track_p90_mrad", "tc_drift_p50_mrad",
              "min_sdf_p10_mm", "ms_per_tick"):
        out[f] = pct([e[f] for e in episodes], 50)
    out["ticks"] = int(sum(e["ticks"] for e in episodes))
    out["diverged"] = int(sum(e["diverged"] for e in episodes))
    out["ticks_penetrating"] = int(sum(e["ticks_penetrating"] for e in episodes))
    out["n_sdf"] = int(sum(e["n_sdf"] for e in episodes))
    out["_rows"] = lat_rows
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    ap.add_argument("--phase1", default="runs/phase1_main/best.pt")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="runs/final/closed_loop.json")
    args = ap.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16

    kit = A.DrumKit.from_layout()
    champ, cfg, step = load_model(ROOT / args.ckpt, dev)
    print(f"[champion] {args.ckpt} step={step} buckets={cfg['buckets']}", flush=True)
    p1, p1cfg, p1step = load_model(ROOT / args.phase1, dev)
    print(f"[phase1]   {args.phase1} step={p1step}", flush=True)

    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.clips]
    print(f"[clips] {[c.name for c in clips]}", flush=True)

    configs = [
        ("champion_lead", champ, cfg, dict(cmd_lead=LEAD_PRESETS["fit"])),
        ("champion_nolead", champ, cfg, dict(cmd_lead=0.0)),
        ("phase1_lead", p1, p1cfg, dict(cmd_lead=LEAD_PRESETS["fit"])),
    ]
    report = {"ckpt": str(args.ckpt), "step": step, "clips": [c.name for c in clips],
              "probe_seeds": list(args.probe_seeds), "steps": args.steps,
              "configs": {}}
    for label, m, c, kw in configs:
        print(f"\n=== {label} ===", flush=True)
        r = run_config(m, c, kit, dev, amp, clips, steps=args.steps,
                       probe_seeds=args.probe_seeds, label=label, **kw)
        rows = r.pop("_rows")
        report["configs"][label] = r
        L = r["latency"]
        print(f"  -> crossed {L['crossed_in_disk']}/{L['n']} miss {100*L['miss_rate']:.0f}% "
              f"|lat|p50 {L['lat_abs_p50_ms']:.1f} |lat|p90 {L['lat_abs_p90_ms']:.1f} "
              f"<=10/20/40 {L['within_10ms']}/{L['within_20ms']}/{L['within_40ms']} "
              f"speed-in-band {L['speed_in_band']}", flush=True)
        (Path(ROOT / args.out).parent).mkdir(parents=True, exist_ok=True)
        Path(ROOT / args.out).with_name(
            Path(args.out).stem + f"_{label}_rows.json").write_text(
            json.dumps(rows, indent=1, default=str))

    Path(ROOT / args.out).write_text(json.dumps(report, indent=1, default=str))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
