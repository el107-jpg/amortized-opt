#!/usr/bin/env python
"""Is the optimal reference lead CONSTANT along the trajectory?  Measure it.

    python scripts/fit_lead_schedule.py --device cuda:2

``scripts/fit_cmd_lead.py`` fits ONE lead per joint by minimizing the tracking
RMS over a window around the strike.  That is a fair summary only if the lead
the servo wants is the same everywhere.  This script drops that assumption: it
buckets the tracking error by ``h``, the number of waypoints REMAINING to that
hand's hit, and fits a separate ``delta*`` per (joint, h).

Why one would expect a schedule.  For the FOH position servo the standing error
needed to produce the commanded acceleration is ``e ~ M(q) qdd / kp``, while a
time lead ``delta`` buys a command offset of ``qd * delta``.  Equating the two,

    delta*(t) ~ M(q) qdd(t) / (kp * qd(t))

-- a RATIO of acceleration to velocity, which is emphatically not constant over
a strike: the approach accelerates from rest (qd -> 0, so delta* -> inf, which
is the unbounded 80-100 ms the ``"wide"`` preset fit at the waist) and the final
descent is near-constant-velocity (qdd -> 0, so delta* -> 0).  ``--report-ratio``
prints the measured ``qdd/qd`` profile next to the fitted schedule so the two can
be compared directly.

Method, and its one approximation: each roll commands ONE constant delta for the
whole trajectory and we read the error off at each h.  A genuinely time-varying
command would give a slightly different plant state history (the plant has
memory).  This is the same approximation ``fit_cmd_lead.py`` already makes, and
it is the right one for identification -- the schedule it produces is then
A/B'd end-to-end by ``eval_gain_sweep.py --lead schedule``, which does apply it
time-varying.

Outputs ``--out`` JSON with ``schedule_s`` as an ``(H_BUCKETS, 17)`` table
indexed by ``h = 0 .. H_BUCKETS-1`` (h = 0 is the strike waypoint), directly
loadable by ``amortized_opt.lead.resolve_lead``.
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
from amortized_opt.lead import lead_waypoints  # noqa: E402
from amortized_opt.pd_sim import BatchedPDSim, PDSimConfig, frozen_minv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_cmd_lead import GROUPS, NAMES, _admissible, build_plans, pack  # noqa: E402


@torch.no_grad()
def sweep_by_horizon(Q, mask, hit, q0, v0, sim, rnea, q_lo, q_hi, step_max,
                     deltas, dt, n_h: int, kp_scale, kd_scale, nominal, seed):
    """``(n_delta, n_h, n_joint)`` tracking RMS, bucketed by waypoints-to-hit.

    ``h = 0`` is the strike waypoint itself, ``h = 1`` one waypoint before it.
    Also returns the reference ``|qd|`` / ``|qdd|`` profiles in the same buckets.
    """
    B, K1, n = Q.shape
    ref = Q[:, 1:]                                  # q_ref at t = dt .. K dt
    m_all = mask[:, 1:]
    kidx = torch.arange(K1 - 1, device=Q.device).view(1, -1)
    # ref index (hit - 1) IS the strike waypoint (ref[k] == knot k+1)
    h_of = (hit.view(-1, 1) - 1) - kidx             # (B, K)
    valid = m_all * (h_of >= 0).to(m_all.dtype) * (h_of < n_h).to(m_all.dtype)
    hb = h_of.clamp(0, n_h - 1)

    if nominal:
        kp, kd, _ = sim.nominal_gains(B)
    else:
        g = torch.Generator(device=Q.device)
        g.manual_seed(int(seed))
        kp, kd, _ = sim.sample_gains(B, generator=g)
    Minv = frozen_minv(rnea, Q[:, :-1])

    # one-hot bucket weights, (B, K, n_h)
    onehot = torch.zeros(B, K1 - 1, n_h, device=Q.device, dtype=Q.dtype)
    onehot.scatter_(2, hb.unsqueeze(-1), 1.0)
    onehot = onehot * valid.unsqueeze(-1)
    counts = onehot.sum((0, 1)).clamp_min(1.0)      # (n_h,)

    out = np.zeros((len(deltas), n_h, n))
    for di, d in enumerate(deltas):
        f = float(d) / float(dt)
        led = lead_waypoints(ref, torch.full((n,), f, device=Q.device, dtype=Q.dtype))
        cmd = _admissible(q0, led, q_lo, q_hi, step_max)
        x = torch.cat([q0.unsqueeze(1), cmd], dim=1)
        qa, _va, _ig = sim.rollout(x, qa0=q0, va0=v0, kp=kp, kd=kd, Minv=Minv)
        err = (qa[:, 1:] - ref) ** 2                            # (B, K, n)
        # sum over cases and knots into the h buckets: (n_h, n)
        s = torch.einsum("bkn,bkh->hn", err, onehot)
        out[di] = torch.sqrt(s / counts.unsqueeze(-1)).cpu().numpy()

    # reference kinematics in the same buckets (finite differences on the plan)
    qd = (ref - Q[:, :-1]) / float(dt)
    qdd = torch.zeros_like(qd)
    qdd[:, 1:] = (qd[:, 1:] - qd[:, :-1]) / float(dt)
    prof = {}
    for tag, sig in (("qd", qd.abs()), ("qdd", qdd.abs())):
        prof[tag] = (torch.einsum("bkn,bkh->hn", sig, onehot)
                     / counts.unsqueeze(-1)).cpu().numpy()
    return out, prof, counts.cpu().numpy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", default="standard")
    p.add_argument("--max-cases", type=int, default=162)
    p.add_argument("--deltas", type=float, nargs="*", default=None)
    p.add_argument("--n-h", type=int, default=16,
                   help="how many waypoints-to-hit buckets to fit (default 16)")
    p.add_argument("--kp-scale", type=float, default=1.0)
    p.add_argument("--kd-scale", type=float, default=1.0)
    p.add_argument("--sampled-gains", action="store_true",
                   help="use the randomized gain draw instead of the nominal")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default="runs/diag7c/lead_schedule.json")
    args = p.parse_args(argv)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    amp = None if args.no_amp else torch.bfloat16
    deltas = args.deltas or list(np.arange(0.0, 0.0501, 0.0025))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    cfg = {"buckets": tuple((ck.get("scenario_cfg") or {}).get("buckets", (25, 50))),
           "num_drums": mcfg.num_drums, "num_future": mcfg.num_future}

    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    scorer = A.TrajectoryScorer(device="cpu")
    cases = A.generate_grid(seed=args.seed, grid=args.grid, scorer=scorer)
    if args.max_cases and len(cases) > args.max_cases:
        cases = cases[:: max(len(cases) // args.max_cases, 1)][: args.max_cases]
    print(f"[fit] {len(cases)} cases, {len(deltas)} deltas, {args.n_h} h-buckets, "
          f"kp x{args.kp_scale} kd x{args.kd_scale}", flush=True)

    plans = build_plans(model, cases, kit, library, cfg, dev, amp)
    dtype = torch.float32
    Q, mask, hit = pack(cases, plans, dev, dtype)
    q0 = torch.as_tensor(np.stack([c.q_start for c in cases]), device=dev, dtype=dtype)
    v0 = torch.as_tensor(np.stack([c.qdot_start for c in cases]), device=dev, dtype=dtype)
    rnea = A.BatchedRNEA.from_urdf(device=dev).to(dev, dtype)
    sim = BatchedPDSim(
        PDSimConfig(dt=A.CONTROL_DT, kp_scale=args.kp_scale, kd_scale=args.kd_scale),
        device=dev, dtype=dtype, rnea=rnea,
    )
    q_lo, q_hi = A.urdf_position_limits(NAMES)
    q_lo = torch.as_tensor(q_lo, device=dev, dtype=dtype)
    q_hi = torch.as_tensor(q_hi, device=dev, dtype=dtype)
    step_max = torch.as_tensor(A.isaac_velocity_limits(NAMES) * A.CONTROL_DT,
                               device=dev, dtype=dtype)

    curves, prof, counts = sweep_by_horizon(
        Q, mask, hit, q0, v0, sim, rnea, q_lo, q_hi, step_max, deltas,
        A.CONTROL_DT, int(args.n_h), args.kp_scale, args.kd_scale,
        not args.sampled_gains, args.seed,
    )
    ds = np.asarray(deltas)
    best = ds[curves.argmin(axis=0)]                     # (n_h, n_joint)
    rms0 = curves[0]                                     # (n_h, n_joint)
    rmsb = curves.min(axis=0)

    # ---- per-group schedule ------------------------------------------------
    print(f"\n=== fitted lead (ms) by waypoints-to-hit, per joint group ===")
    print(f"{'h':>3}{'t_to_hit':>10}{'n':>7}", end="")
    for g in GROUPS:
        print(f"{g:>10}", end="")
    print(f"{'rms@0':>9}{'rms@fit':>9}{'gain%':>7}")
    for h in range(int(args.n_h)):
        print(f"{h:>3}{h * A.CONTROL_DT * 1e3:>9.0f}ms{int(counts[h]):>7}", end="")
        for g, idx in GROUPS.items():
            print(f"{1e3 * float(np.median(best[h, idx])):>10.1f}", end="")
        r0, rb = 1e3 * float(rms0[h].mean()), 1e3 * float(rmsb[h].mean())
        print(f"{r0:>9.1f}{rb:>9.1f}{100 * (1 - rb / max(r0, 1e-9)):>7.1f}")

    # ---- the analytic prediction: delta* ~ qdd / qd -------------------------
    print(f"\n=== reference kinematics in the same buckets (arm joints) ===")
    arm = [i for i, nm in enumerate(NAMES) if not nm.startswith("waist_")]
    print(f"{'h':>3}{'|qd| rad/s':>12}{'|qdd| rad/s2':>14}{'qdd/qd (s^-1)':>15}"
          f"{'predicted ms':>14}{'fitted ms':>11}")
    for h in range(int(args.n_h)):
        v = float(prof["qd"][h, arm].mean())
        a = float(prof["qdd"][h, arm].mean())
        ratio = a / max(v, 1e-6)
        # delta ~ M qdd / (kp qd); with the arm's effective inertia folded into
        # the fit this is only a SHAPE prediction, so report the raw ratio too.
        print(f"{h:>3}{v:>12.3f}{a:>14.2f}{ratio:>15.2f}"
              f"{1e3 * ratio / 40.0:>14.2f}"
              f"{1e3 * float(np.median(best[h, arm])):>11.1f}")

    out = {
        "ckpt": args.ckpt, "n_cases": len(cases), "grid": args.grid,
        "seed": args.seed, "deltas_s": [float(x) for x in ds],
        "n_h": int(args.n_h), "kp_scale": args.kp_scale, "kd_scale": args.kd_scale,
        "nominal_gains": not args.sampled_gains,
        "joints": list(NAMES),
        "schedule_s": [[float(x) for x in row] for row in best],
        "rms_at_0_mrad": [[float(1e3 * x) for x in row] for row in rms0],
        "rms_at_fit_mrad": [[float(1e3 * x) for x in row] for row in rmsb],
        "bucket_counts": [int(x) for x in counts],
        "ref_qd": [[float(x) for x in row] for row in prof["qd"]],
        "ref_qdd": [[float(x) for x in row] for row in prof["qdd"]],
    }
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\n[out] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
