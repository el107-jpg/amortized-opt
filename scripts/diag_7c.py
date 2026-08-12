#!/usr/bin/env python
"""Section 7c decomposition: WHY is the closed-loop strike 40-56 mm short?

Three candidate mechanisms, measured separately:

* **indexing** -- the scorer's "scheduled hit tick" and the planner's terminal
  knot refer to different instants (off-by-one);
* **PD lag** -- the plan is right but the plant is one control period behind it,
  which at a 2-2.8 m/s approach is exactly 40-56 mm;
* **procrastination** -- each replan defers the approach into waypoints that are
  never executed, so the planned terminal itself drifts as the horizon shrinks.

Probes
------

``--probe index``     analytic + empirical check of the tick conventions.
``--probe openloop``  one full grid plan per case, executed through the PD sim
                      START TO FINISH with NO replanning.  Compares the plan's
                      own kinematic contact at the terminal knot with the
                      plant's, at knots k-2..k+2.  Isolates PD lag.
``--probe closed``    three eval clips through ``MPCGenerator``.  For every
                      executed hit: the measured contact at k-3..k+3 AND the
                      per-tick PLANNED terminal contact (the drift curve).
``--probe knobs``     the closed loop again with a hit LEAD (schedule the goal
                      early) and/or a FREEZE window (stop replanning inside the
                      last few waypoints), on the unchanged checkpoint.

Everything is reported in the head-plane frame of the drum that was aimed at:
``axial`` is the signed distance along the outward head normal (positive =
short of the head, negative = through it) and ``in_plane`` is the distance
beyond the legal disk edge (<= 0 = inside).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402
from amortized_opt.mpc_gen import MPCGenConfig, MPCGenerator, PlanRequest  # noqa: E402
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, frozen_minv  # noqa: E402
from amortized_opt.probe import (  # noqa: E402
    head_frame_error,
    hit_latency,
    mpc_model_planner,
    score_hits,
    summarize_latency,
)

ROOT = Path(__file__).resolve().parent.parent

#: Per-joint reference lead identified by ``scripts/fit_cmd_lead.py`` against
#: the exact servo model (strike window, 162 standard cases, randomized gains).
PER_JOINT_LEAD = (
    0.0025, 0.05, 0.05,                      # waist yaw / pitch / roll
    0.05, 0.0075, 0.0, 0.015, 0.0, 0.0025, 0.0,   # left arm
    0.0, 0.005, 0.0, 0.015, 0.0025, 0.0, 0.0,     # right arm
)
#: the same fit re-run over a wider delta grid: the narrow sweep's 50 ms was its
#: own upper bound for the waist and the left shoulder, so those are unbounded
#: fits, not minima.
PER_JOINT_LEAD_WIDE = (
    0.0025, 0.08, 0.10,
    0.10, 0.12, 0.0, 0.015, 0.0, 0.0025, 0.0,
    0.0, 0.005, 0.0, 0.015, 0.0025, 0.0, 0.0,
)
#: SPARSE: only the joints whose fitted lead buys >= 10 % of their tracking RMS,
#: capped at 50 ms.  Four numbers instead of seventeen, and every one of them is
#: a joint where the fit has a real minimum rather than a flat floor.
PER_JOINT_LEAD_SPARSE = (
    0.0, 0.05, 0.05,
    0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.015, 0.0, 0.0, 0.0,
)


def pct(x, q):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return float(np.percentile(x, q)) if x.size else float("nan")


def brief(name, vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not v.size:
        return f"{name}: (empty)"
    return (f"{name}: n={v.size} p10={np.percentile(v,10):+.1f} "
            f"p50={np.percentile(v,50):+.1f} p90={np.percentile(v,90):+.1f} "
            f"mean={v.mean():+.1f}")


# ---------------------------------------------------------------- the model --
def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = A.ModelConfig(**ck["model_cfg"])
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    buckets = tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50, 100)))
    cfg = {"buckets": buckets, "num_drums": mcfg.num_drums,
           "num_future": mcfg.num_future}
    return model, cfg, ck


# ============================================================ probe: index ====
def probe_index(args, model, cfg, kit, dev, amp):
    """Are the planner's terminal knot and the scorer's hit tick the same instant?

    Analytic chain, stated once so it can be checked against the code:

    * ``MPCGenerator._hit_index``: ``k = round(tth / dt)``, >= 1.
    * the mask is built from ``k * dt``, so waypoint index ``k - 1`` (0-based in
      the plan array, time ``k * dt``) is the terminal;
    * ``FusedCost._terminal`` prices ``sp.knot(k)`` and knot ``i`` is waypoint
      ``i - 1`` -- the same instant;
    * ``run_episode`` commands ``traj[0]``, appends ``q_hist[step + 1]``, then
      decrements ``hit_index``; a hand fires when it reaches 0, and records
      ``(step + 1, ...)``;
    * the scorer reads ``q[k] = q_hist[step + 1]`` -- the plant state one control
      period after the tick whose plan had ``hit_index == 1``.

    So the plan's terminal waypoint (time ``dt`` ahead of the tick with
    ``hit_index == 1``) and the scored plant state are the SAME instant.  The
    empirical half below checks it end to end on a real episode by asking, for
    each executed hit, at which tick offset the COMMAND (not the plant) is
    closest to the head.
    """
    scorer = A.TrajectoryScorer(device="cpu")
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[:1]
    gen = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen.library, fk=gen.fk, rnea=gen.rnea,
                             sdf=gen.sdf)
    starts = sampler.sample(1, seed=args.seed)
    plan, _st = model_planner(model, cfg, kit, dev, amp)
    res = gen.run_episode(clips[0], starts.q[0].cpu().numpy(), starts.qd[0].cpu().numpy(),
                          seed=1000, planner=plan, max_steps=args.steps)
    q, cmd = res.q_hist, res.cmd_hist
    rows = []
    for (k, hand, drum) in res.hit_steps:
        sigma = A.strike.default_contact_sigma(kit.names[drum])
        r = {"k": k, "hand": hand, "drum": kit.names[drum]}
        for off in (-2, -1, 0, 1, 2):
            i = k + off
            if not (0 <= i < len(q)):
                continue
            ax_q, ip_q, ok_q = head_frame_error(scorer, q[i], hand, kit, drum, sigma)
            r[f"q{off:+d}"] = 1000 * ax_q
            # the COMMAND at the same instant: cmd_hist[j] is the target for
            # q_hist[j + 1]
            j = i - 1
            if 0 <= j < len(cmd):
                ax_c, _ip, _ok = head_frame_error(scorer, cmd[j], hand, kit, drum, sigma)
                r[f"c{off:+d}"] = 1000 * ax_c
        rows.append(r)
    offs = [-2, -1, 0, 1, 2]
    print("\n--- probe: index (axial mm, + = short of the head) ---")
    print(f"{'hit':>4} {'hand':>6} {'drum':>13} " +
          " ".join(f"{'q'+f'{o:+d}':>8}" for o in offs) + "  |" +
          " ".join(f"{'cmd'+f'{o:+d}':>8}" for o in offs))
    for r in rows[: args.max_hits]:
        print(f"{r['k']:>4} {r['hand']:>6} {r['drum']:>13} " +
              " ".join(f"{r.get(f'q{o:+d}', float('nan')):8.1f}" for o in offs) + "  |" +
              " ".join(f"{r.get(f'c{o:+d}', float('nan')):8.1f}" for o in offs))
    out = {"rows": rows}
    for tag in ("q", "c"):
        best = []
        for r in rows:
            vals = {o: abs(r.get(f"{tag}{o:+d}", np.nan)) for o in offs}
            vals = {o: v for o, v in vals.items() if np.isfinite(v)}
            if vals:
                best.append(min(vals, key=vals.get))
        out[f"argmin_{tag}"] = {int(o): int(sum(1 for b in best if b == o)) for o in offs}
        print(f"argmin |axial| over offsets, {tag}: {out[f'argmin_{tag}']}")
    return out


# ========================================================= probe: openloop ====
def probe_openloop(args, model, cfg, kit, dev, amp):
    """A whole grid plan, executed through the PD sim with NO replanning."""
    scorer = A.TrajectoryScorer(device="cpu")
    library = A.GoalPoseLibrary()
    cases = A.generate_grid(seed=args.seed, grid="standard", scorer=scorer)
    if args.max_cases and len(cases) > args.max_cases:
        stride = max(len(cases) // args.max_cases, 1)
        cases = cases[::stride][: args.max_cases]
    plans, _wall = A.batched_model_plans(
        model, cases, device=str(dev), kit=kit, library=library,
        buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
        num_future=cfg["num_future"], iterations=1, init="quintic", amp_dtype=amp,
    )
    names = A.G1_UPPER_BODY_JOINTS
    q_lo, q_hi = A.urdf_position_limits(names)
    step_max = A.isaac_velocity_limits(names) * A.CONTROL_DT
    rnea = scorer.rnea
    sim = BatchedPDSim(PDSimConfig(dt=A.CONTROL_DT), device="cpu",
                       dtype=torch.float64, rnea=rnea)
    kin, pdo = {o: [] for o in (-2, -1, 0, 1, 2)}, {o: [] for o in (-2, -1, 0, 1, 2)}
    kin_ok, pd_ok, pd_ok_any = [], [], []
    rng = np.random.default_rng(args.seed)
    for c in cases:
        Q = np.asarray(plans[int(c.case_id)], float)     # (N+1, 17), row 0 = q_start
        N = int(c.N)
        # ---- the plant: one 20 ms control tick per waypoint, no replanning ---
        kp, kd, _ki = (sim.nominal_gains(1) if args.nominal_gains else
                  sim.sample_gains(1, generator=_tgen(rng)))
        q = torch.as_tensor(c.q_start, dtype=torch.float64).view(1, -1)
        qd = torch.as_tensor(c.qdot_start, dtype=torch.float64).view(1, -1)
        prev = q.clone()
        meas = [q[0].numpy().copy()]
        for i in range(1, N + 1):
            cmd = np.clip(
                prev[0].numpy() + np.clip(Q[i] - prev[0].numpy(), -step_max, step_max),
                q_lo, q_hi,
            )
            cmd_t = torch.as_tensor(cmd, dtype=torch.float64).view(1, -1)
            Minv = frozen_minv(rnea, q.unsqueeze(1))[:, 0]
            q, qd, _int = sim.step(prev, cmd_t, q, qd, kp=kp, kd=kd, Minv=Minv)
            prev = cmd_t
            meas.append(q[0].numpy().copy())
        meas = np.stack(meas)
        k = int(np.clip(c.strike.waypoint(c.dt), 1, N))
        sigma = c.strike.resolved_contact_sigma()
        drum = kit.index(str(c.drum))
        for o in kin:
            i = k + o
            if 0 <= i <= N:
                a1, _p1, _o1 = head_frame_error(scorer, Q[i], c.hand, kit, drum, sigma)
                a2, _p2, _o2 = head_frame_error(scorer, meas[i], c.hand, kit, drum, sigma)
                kin[o].append(1000 * a1)
                pdo[o].append(1000 * a2)
        _a, _p, ok1 = head_frame_error(scorer, Q[k], c.hand, kit, drum, sigma)
        _a, _p, ok2 = head_frame_error(scorer, meas[k], c.hand, kit, drum, sigma)
        kin_ok.append(ok1)
        pd_ok.append(ok2)
        oks = []
        for o in (-1, 0, 1, 2):
            i = k + o
            if 0 <= i <= N:
                oks.append(head_frame_error(scorer, meas[i], c.hand, kit, drum, sigma)[2])
        pd_ok_any.append(any(oks))
    print("\n--- probe: openloop (grid plan through the PD sim, NO replanning) ---")
    print(f"cases {len(cases)}  gains={'nominal' if args.nominal_gains else 'randomized'}")
    for o in sorted(kin):
        print(f"  knot k{o:+d}: " + brief("plan  axial mm", kin[o]))
        print(f"             " + brief("plant axial mm", pdo[o]))
    print(f"  in-disk at knot k: plan {sum(kin_ok)}/{len(kin_ok)}   "
          f"plant {sum(pd_ok)}/{len(pd_ok)}   plant-any(k-1..k+2) {sum(pd_ok_any)}/{len(pd_ok_any)}")
    lag = np.asarray(pdo[0]) - np.asarray(kin[0])
    print("  " + brief("PD-induced axial shortfall at k, mm", lag))
    return {"kin": {o: [pct(v, 50)] for o, v in kin.items()},
            "pd": {o: [pct(v, 50)] for o, v in pdo.items()},
            "kin_ok": int(sum(kin_ok)), "pd_ok": int(sum(pd_ok)),
            "pd_ok_any": int(sum(pd_ok_any)), "n": len(cases),
            "pd_minus_plan_p50_mm": pct(lag, 50)}


def _tgen(rng):
    g = torch.Generator()
    g.manual_seed(int(rng.integers(0, 2**31 - 1)))
    return g


# =========================================================== the MPC planner ==
def model_planner(model, cfg, kit, dev, amp, *, iterations=1, record=None):
    """``amortized_opt.probe.mpc_model_planner``, with this script's arg shape."""
    return mpc_model_planner(
        model, num_drums=cfg["num_drums"], num_future=cfg["num_future"],
        device=dev, amp_dtype=amp, iterations=iterations, record=record,
    )


# =========================================================== probe: closed ====
def run_clip(model, cfg, kit, dev, amp, clip, q0, qd0, *, steps, seed,
             lead_s=0.0, freeze_wp=0, iterations=1, sticky=True,
             freeze_waist=True, cmd_lead=0.0):
    gen = MPCGenerator(
        kit=kit,
        config=MPCGenConfig(horizon=cfg["buckets"][-1], hit_lead_s=float(lead_s),
                            freeze_wp=int(freeze_wp),
                            freeze_waist=bool(freeze_waist),
                            cmd_lead_s=cmd_lead,
                            sticky_goal_point=bool(sticky)),
    )
    record: list = []
    plan, st = model_planner(model, cfg, kit, dev, amp, iterations=iterations,
                             record=record)
    t0 = time.perf_counter()
    res = gen.run_episode(clip, q0, qd0, seed=seed, planner=plan, max_steps=steps)
    return res, record, st, time.perf_counter() - t0


def summarize_hits(hits, tag):
    n = len(hits)
    ax0 = [h["meas"].get(0, {}).get("axial_mm", np.nan) for h in hits]
    ip0 = [h["meas"].get(0, {}).get("inplane_mm", np.nan) for h in hits]
    ok0 = sum(1 for h in hits if h["meas"].get(0, {}).get("in_disk"))
    okany = sum(1 for h in hits
                if any(h["meas"].get(o, {}).get("in_disk") for o in (-1, 0, 1, 2)))
    axmin = [min((abs(h["meas"][o]["axial_mm"]) for o in h["meas"]), default=np.nan)
             for h in hits]
    per_off = {o: sum(1 for h in hits if h["meas"].get(o, {}).get("in_disk"))
               for o in (-3, -2, -1, 0, 1, 2, 3)}
    ax = [h["meas"][0]["axial_mm"] for h in hits if 0 in h["meas"]]
    early = sum(1 for x in ax if x < -20.0)      # already through the head
    late = sum(1 for x in ax if x > 20.0)        # still short of it
    print(f"   at the scheduled tick: {sum(1 for x in ax if abs(x)<=20)}/{len(ax)} "
          f"in the +/-20 mm band, {early} EARLY (through), {late} LATE (short)")
    print(f"  [{tag}] hits {n}  in-disk@k {ok0}/{n}  in-disk@any(k-1..k+2) {okany}/{n}")
    print(f"   in-disk by tick offset: " + " ".join(f"{o:+d}:{c}" for o, c in per_off.items()))
    ax_off = {o: pct([h["meas"][o]["axial_mm"] for h in hits if o in h["meas"]], 50)
              for o in (-3, -2, -1, 0, 1, 2, 3)}
    print(f"   axial p50 by offset (mm): " + " ".join(f"{o:+d}:{v:.0f}" for o, v in ax_off.items()))
    print("   " + brief("axial@k mm", ax0))
    print("   " + brief("inplane@k mm (>0 outside disk)", ip0))
    print("   " + brief("min|axial| over k-3..k+3 mm", axmin))
    return {"n": n, "in_disk_k": ok0, "in_disk_any": okany,
            "in_disk_by_offset": per_off, "axial_p50_by_offset": ax_off,
            "axial_k_p50": pct(ax0, 50), "inplane_k_p50": pct(ip0, 50),
            "min_axial_p50": pct(axmin, 50)}


def drift_table(hits):
    """Median planned-terminal axial error as a function of remaining horizon."""
    by_h: dict[int, list] = {}
    for h in hits:
        for p in h["planned"]:
            by_h.setdefault(p["h"], []).append(p["axial_mm"])
    print("   planned-terminal axial error vs remaining horizon h (mm):")
    ks = sorted(by_h)
    print("     h : " + " ".join(f"{k:>7d}" for k in ks))
    print("     p50: " + " ".join(f"{pct(by_h[k],50):>7.1f}" for k in ks))
    print("     n  : " + " ".join(f"{len(by_h[k]):>7d}" for k in ks))
    return {int(k): {"p50": pct(v, 50), "n": len(v)} for k, v in by_h.items()}


def probe_closed(args, model, cfg, kit, dev, amp, *, lead_s=0.0, freeze_wp=0,
                 sticky=True, freeze_waist=True, cmd_lead=0.0, label="closed"):
    scorer = A.TrajectoryScorer(device="cpu")
    clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.clips]
    gen0 = MPCGenerator(kit=kit, config=MPCGenConfig(horizon=cfg["buckets"][-1]))
    sampler = A.StartSampler(kit=kit, library=gen0.library, fk=gen0.fk,
                             rnea=gen0.rnea, sdf=gen0.sdf)
    starts = sampler.sample(len(clips), seed=args.seed)
    print(f"\n--- probe: {label} (lead={lead_s*1000:.0f} ms freeze={freeze_wp} wp "
          f"sticky={sticky}) ---")
    allhits, alllat, out = [], [], {"episodes": []}
    for ep, clip in enumerate(clips):
        res, record, st, wall = run_clip(
            model, cfg, kit, dev, amp, clip,
            starts.q[ep].cpu().numpy(), starts.qd[ep].cpu().numpy(),
            steps=args.steps, seed=1000 + ep, lead_s=lead_s, freeze_wp=freeze_wp,
            sticky=sticky, freeze_waist=freeze_waist, cmd_lead=cmd_lead,
        )
        lt = int(round(lead_s / A.CONTROL_DT))
        hits = score_hits(scorer, kit, res, record, lead_ticks=lt)
        allhits.extend(hits)
        alllat.extend(hit_latency(scorer, kit, res, lead_ticks=lt))
        track = np.abs(res.q_hist[1:] - res.cmd_hist) if len(res.cmd_hist) else np.zeros(1)
        out["episodes"].append({
            "clip": clip.name, "ticks": int(res.cmd_hist.shape[0]),
            "hits": len(hits), "diverged": bool(res.diverged),
            "frozen_ticks": int(getattr(res, "frozen_ticks", 0)),
            "pd_track_p50_mrad": float(np.percentile(track, 50) * 1000),
            "pd_track_p90_mrad": float(np.percentile(track, 90) * 1000),
            "tc_drift_p50_mrad": (float(np.percentile(st["drift"], 50) * 1000)
                                  if st["drift"] else float("nan")),
            "ms_per_tick": 1000 * wall / max(res.cmd_hist.shape[0], 1),
        })
        print(f"  {clip.name}: ticks {res.cmd_hist.shape[0]} hits {len(hits)} "
              f"frozen {getattr(res, 'frozen_ticks', 0)} diverged {res.diverged} "
              f"pd_p50 {out['episodes'][-1]['pd_track_p50_mrad']:.1f} mrad")
    L = summarize_latency(alllat)
    out["latency"] = L
    print(f"   LATENCY: crossed-in-disk {L['crossed_in_disk']}/{L['n']} "
          f"(miss {100*L['miss_rate']:.0f}%)  p50 {L['lat_p50_ms']:+.1f} ms  "
          f"|p50| {L['lat_abs_p50_ms']:.1f}  |p90| {L['lat_abs_p90_ms']:.1f}  "
          f"early {L['early']}  <=10/20/40 ms {L['within_10ms']}/{L['within_20ms']}"
          f"/{L['within_40ms']}  speed p50 {L['speed_p50_mps']:.2f} "
          f"in-band {L['speed_in_band']}")
    print("   hist: " + " ".join(f"{k}:{v}" for k, v in L["hist_ms"].items() if v))
    out["summary"] = summarize_hits(allhits, label)
    out["drift"] = drift_table(allhits)
    out["hits"] = [{k: v for k, v in h.items()} for h in allhits]
    return out



# ============================================================== probe: lag ====
def _roll_pd(sim, rnea, Q, q0, qd0, kp, kd, q_lo, q_hi, step_max, extra=0):
    """Execute a waypoint plan through the PD sim, one 20 ms tick per waypoint.

    ``extra`` extra ticks HOLD the last waypoint, so we can ask whether the
    plant merely lags the plan or fails to reach it at all.
    """
    q = torch.as_tensor(q0, dtype=torch.float64).view(1, -1)
    qd = torch.as_tensor(qd0, dtype=torch.float64).view(1, -1)
    prev = q.clone()
    meas, cmds = [q[0].numpy().copy()], []
    rows = list(Q[1:]) + [Q[-1]] * int(extra)
    for tgt in rows:
        cmd = np.clip(prev[0].numpy() + np.clip(tgt - prev[0].numpy(),
                                                -step_max, step_max), q_lo, q_hi)
        cmd_t = torch.as_tensor(cmd, dtype=torch.float64).view(1, -1)
        Minv = frozen_minv(rnea, q.unsqueeze(1))[:, 0]
        q, qd, _int = sim.step(prev, cmd_t, q, qd, kp=kp, kd=kd, Minv=Minv)
        prev = cmd_t
        meas.append(q[0].numpy().copy())
        cmds.append(cmd.copy())
    return np.stack(meas), np.stack(cmds)


def probe_lag(args, model, cfg, kit, dev, amp):
    """Is the plant merely LATE, or does it never reach the head?"""
    scorer = A.TrajectoryScorer(device="cpu")
    library = A.GoalPoseLibrary()
    cases = A.generate_grid(seed=args.seed, grid="standard", scorer=scorer)
    if args.max_cases and len(cases) > args.max_cases:
        stride = max(len(cases) // args.max_cases, 1)
        cases = cases[::stride][: args.max_cases]
    plans, _w = A.batched_model_plans(
        model, cases, device=str(dev), kit=kit, library=library,
        buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
        num_future=cfg["num_future"], iterations=1, init="quintic", amp_dtype=amp,
    )
    qplan = A.quintic_case_planner(library=library)
    names = A.G1_UPPER_BODY_JOINTS
    q_lo, q_hi = A.urdf_position_limits(names)
    step_max = A.isaac_velocity_limits(names) * A.CONTROL_DT
    sim = BatchedPDSim(PDSimConfig(dt=A.CONTROL_DT), device="cpu",
                       dtype=torch.float64, rnea=scorer.rnea)
    EXTRA = 8
    out = {}
    for tag in ("model", "quintic"):
        ax_plan, ax_k, argmin, ax_min, jerr, tipspd_plan, tipspd_pd = [], [], [], [], [], [], []
        ok_plan = ok_k = ok_win = 0
        rng = np.random.default_rng(args.seed)
        for c in cases:
            Q = (np.asarray(plans[int(c.case_id)], float) if tag == "model"
                 else np.asarray(qplan(c).Q, float))
            N = int(c.N)
            kp, kd, _ki = (sim.nominal_gains(1) if args.nominal_gains
                      else sim.sample_gains(1, generator=_tgen(rng)))
            meas, cmds = _roll_pd(sim, scorer.rnea, Q, c.q_start, c.qdot_start,
                                  kp, kd, q_lo, q_hi, step_max, extra=EXTRA)
            k = int(np.clip(c.strike.waypoint(c.dt), 1, N))
            sigma = c.strike.resolved_contact_sigma()
            drum = kit.index(str(c.drum))
            a_p, _i, o_p = head_frame_error(scorer, Q[k], c.hand, kit, drum, sigma)
            a_m, _i, o_m = head_frame_error(scorer, meas[k], c.hand, kit, drum, sigma)
            ax_plan.append(1000 * a_p)
            ax_k.append(1000 * a_m)
            ok_plan += int(o_p)
            ok_k += int(o_m)
            win = []
            for j in range(k, min(k + EXTRA, len(meas) - 1) + 1):
                aa, _ii, oo = head_frame_error(scorer, meas[j], c.hand, kit, drum, sigma)
                win.append((abs(aa), j - k, oo))
            ok_win += int(any(w[2] for w in win))
            best = min(win)
            argmin.append(best[1])
            ax_min.append(1000 * best[0])
            jerr.append(1000 * float(np.abs(meas[k] - cmds[k - 1]).max()))
            Jt = scorer.tip_jacobian(Q[k], c.hand, sigma)
            tipspd_plan.append(float(np.linalg.norm(Jt @ ((Q[k] - Q[k - 1]) / c.dt))))
            tipspd_pd.append(float(np.linalg.norm(Jt @ ((meas[k] - meas[k - 1]) / c.dt))))
        print(f"\n  == {tag} plans ==")
        print("   " + brief("plan  axial@k mm", ax_plan))
        print("   " + brief("plant axial@k mm", ax_k))
        print("   " + brief("plant min|axial| over k..k+8 mm", ax_min))
        print(f"   argmin offset histogram: "
              f"{ {int(o): int(argmin.count(o)) for o in sorted(set(argmin))} }")
        print("   " + brief("max joint |q-cmd| at k, mrad", jerr))
        print("   " + brief("tip speed at k, plan  (mm/s)", [1000 * v for v in tipspd_plan]))
        print("   " + brief("tip speed at k, plant (mm/s)", [1000 * v for v in tipspd_pd]))
        print(f"   in-disk: plan {ok_plan}/{len(cases)}  plant@k {ok_k}/{len(cases)} "
              f"plant-any(k..k+8) {ok_win}/{len(cases)}")
        out[tag] = {"ax_plan_p50": pct(ax_plan, 50), "ax_k_p50": pct(ax_k, 50),
                    "ax_min_p50": pct(ax_min, 50),
                    "argmin": {int(o): int(argmin.count(o)) for o in sorted(set(argmin))},
                    "jerr_p50_mrad": pct(jerr, 50),
                    "tip_plan_p50": pct(tipspd_plan, 50), "tip_pd_p50": pct(tipspd_pd, 50),
                    "ok_plan": ok_plan, "ok_k": ok_k, "ok_win": ok_win, "n": len(cases)}
    print("\n--- probe: lag done ---")
    return out


# ==================================================================== main ====
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--probe", nargs="+",
                   default=["index", "openloop", "closed", "knobs"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=60)
    p.add_argument("--max-hits", type=int, default=12)
    p.add_argument("--nominal-gains", action="store_true")
    p.add_argument("--leads", type=float, nargs="*", default=[0.010, 0.020])
    p.add_argument("--freezes", type=int, nargs="*", default=[5])
    p.add_argument("--knobs", nargs="*", default=None,
                   help="subset of the knob grid to run")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16
    kit = A.DrumKit.from_layout()
    model, cfg, ck = load_model(args.ckpt, dev)
    print(f"[ckpt] {args.ckpt} step={ck['step']} buckets={cfg['buckets']}", flush=True)

    report: dict = {"ckpt": str(args.ckpt), "step": int(ck["step"])}
    if "index" in args.probe:
        report["index"] = probe_index(args, model, cfg, kit, dev, amp)
    if "lag" in args.probe:
        print("\n--- probe: lag (plan vs plant through the PD sim, terminal held) ---")
        report["lag"] = probe_lag(args, model, cfg, kit, dev, amp)
    if "openloop" in args.probe:
        report["openloop"] = probe_openloop(args, model, cfg, kit, dev, amp)
    if "closed" in args.probe:
        report["closed_jitter"] = probe_closed(
            args, model, cfg, kit, dev, amp, sticky=False,
            label="closed / JITTERED goal (pre-7c behaviour)")
        report["closed"] = probe_closed(
            args, model, cfg, kit, dev, amp, sticky=True,
            label="closed / STICKY goal")
    if "knobs" in args.probe:
        report["knobs"] = {}
        grid = [
            ("base", dict()),
            ("lead20", dict(lead_s=0.02)),
            ("fz5arm", dict(freeze_wp=5, freeze_waist=False)),
            ("fz5aw", dict(freeze_wp=5, freeze_waist=True)),
            ("fz3aw", dict(freeze_wp=3, freeze_waist=True)),
            ("fz5aw_lead20", dict(freeze_wp=5, freeze_waist=True, lead_s=0.02)),
            ("fz3aw_lead20", dict(freeze_wp=3, freeze_waist=True, lead_s=0.02)),
            ("fz5aw_lead10", dict(freeze_wp=5, freeze_waist=True, lead_s=0.01)),
            ("lead40", dict(lead_s=0.04)),
            ("fz5aw_lead40", dict(freeze_wp=5, freeze_waist=True, lead_s=0.04)),
            ("fz5aw_lead60", dict(freeze_wp=5, freeze_waist=True, lead_s=0.06)),
            ("fz8aw_lead40", dict(freeze_wp=8, freeze_waist=True, lead_s=0.04)),
            ("cl5", dict(cmd_lead=0.005)),
            ("cl10", dict(cmd_lead=0.010)),
            ("cl15", dict(cmd_lead=0.015)),
            ("cl20", dict(cmd_lead=0.020)),
            ("cl25", dict(cmd_lead=0.025)),
            ("clpj", dict(cmd_lead=PER_JOINT_LEAD)),
            ("cl10_lead20", dict(cmd_lead=0.010, lead_s=0.02)),
            ("clpj_fz5aw_lead20", dict(cmd_lead=PER_JOINT_LEAD, freeze_wp=5,
                                       freeze_waist=True, lead_s=0.02)),
            ("clpj_wide", dict(cmd_lead=PER_JOINT_LEAD_WIDE)),
            ("clpj_sparse", dict(cmd_lead=PER_JOINT_LEAD_SPARSE)),
            ("clpj_fz5aw", dict(cmd_lead=PER_JOINT_LEAD, freeze_wp=5,
                                freeze_waist=True)),
            ("clpj_lead20", dict(cmd_lead=PER_JOINT_LEAD, lead_s=0.02)),
            ("clpjsp_fz5aw", dict(cmd_lead=PER_JOINT_LEAD_SPARSE, freeze_wp=5,
                                  freeze_waist=True)),
        ]
        want = set(args.knobs) if args.knobs else None
        for name, kw in grid:
            if want is not None and name not in want:
                continue
            report["knobs"][name] = probe_closed(
                args, model, cfg, kit, dev, amp, label=name, **kw)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
