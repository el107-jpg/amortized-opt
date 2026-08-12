#!/usr/bin/env python
"""Identify the per-joint reference lead against the exact servo model.

    python scripts/fit_cmd_lead.py --ckpt runs/phase2b_main/best.pt --device cuda:3

The plant is ``amortized_opt.pd_sim`` -- an FOH position ramp with velocity
feed-forward and the real coupled effective mass matrix, at the runtime Kp/Kd.
We own it exactly, so the lead does not have to be guessed: sweep ``delta``,
roll a batch of strike-like planned trajectories through the plant commanding
``q_ref(t + delta)``, and take the ``delta`` that minimizes ``|q_act - q_ref|``
PER JOINT.

Two windows are reported because they answer different questions:

* ``all``    -- RMS over the whole trajectory: the lead that makes the servo
  track the plan generally;
* ``strike`` -- RMS over the last ``--strike-window`` knots: the lead that makes
  it arrive, which is the one that decides ``ok_hit``.

The fitted vector is printed in ``G1_UPPER_BODY_JOINTS`` order, ready to paste
into ``MPCGenConfig.cmd_lead_s`` / ``Phase2Config.cmd_lead_s``, and written to
``--out`` as JSON.
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

NAMES = A.G1_UPPER_BODY_JOINTS
GROUPS = {
    "waist": [i for i, n in enumerate(NAMES) if n.startswith("waist_")],
    "shoulder": [i for i, n in enumerate(NAMES) if "shoulder" in n],
    "elbow": [i for i, n in enumerate(NAMES) if "elbow" in n],
    "wrist": [i for i, n in enumerate(NAMES) if "wrist" in n],
}


def build_plans(model, cases, kit, library, cfg, dev, amp):
    plans, _ = A.batched_model_plans(
        model, cases, device=str(dev), kit=kit, library=library,
        buckets=tuple(cfg["buckets"]), num_drums=cfg["num_drums"],
        num_future=cfg["num_future"], iterations=1, init="quintic", amp_dtype=amp,
    )
    return plans


def pack(cases, plans, dev, dtype):
    """``(B, K+1, n)`` padded reference trajectories + a ``(B, K+1)`` validity mask."""
    K = max(int(c.N) for c in cases)
    n = len(NAMES)
    Q = np.zeros((len(cases), K + 1, n))
    M = np.zeros((len(cases), K + 1))
    hit = np.zeros(len(cases), dtype=np.int64)
    for i, c in enumerate(cases):
        q = np.asarray(plans[int(c.case_id)], float)
        N = int(c.N)
        Q[i, : N + 1] = q
        Q[i, N + 1 :] = q[-1]              # hold past the terminal
        M[i, : N + 1] = 1.0
        hit[i] = int(np.clip(c.strike.waypoint(c.dt), 1, N))
    t = lambda a, d=dtype: torch.as_tensor(a, device=dev, dtype=d)  # noqa: E731
    return t(Q), t(M), torch.as_tensor(hit, device=dev)


def _admissible(q0, traj, q_lo, q_hi, step_max):
    prev = q0
    rows = []
    for i in range(traj.shape[1]):
        prev = torch.clamp(
            prev + torch.clamp(traj[:, i] - prev, -step_max, step_max), q_lo, q_hi
        )
        rows.append(prev)
    return torch.stack(rows, dim=1)


@torch.no_grad()
def sweep(Q, mask, hit, q0, v0, sim, rnea, q_lo, q_hi, step_max, deltas, dt,
          strike_window: int, nominal: bool, seed: int):
    """``{delta: (per-joint RMS all, per-joint RMS strike)}``."""
    B, K1, n = Q.shape
    ref = Q[:, 1:]                                   # q_ref at t = dt .. K dt
    m_all = mask[:, 1:]
    kidx = torch.arange(K1 - 1, device=Q.device).view(1, -1)
    m_str = m_all * (kidx >= (hit.view(-1, 1) - int(strike_window))).to(m_all.dtype) \
        * (kidx <= hit.view(-1, 1)).to(m_all.dtype)
    if nominal:
        kp, kd, _ki = sim.nominal_gains(B)
    else:
        g = torch.Generator(device=Q.device)
        g.manual_seed(int(seed))
        kp, kd, _ki = sim.sample_gains(B, generator=g)
    # frozen inverse effective mass, one per knot of the NOMINAL plan -- the
    # trajopt convention the rest of the package uses
    Minv = frozen_minv(rnea, Q[:, :-1])
    out = {}
    for d in deltas:
        f = float(d) / float(dt)
        led = lead_waypoints(ref, torch.full((n,), f, device=Q.device, dtype=Q.dtype))
        cmd = _admissible(q0, led, q_lo, q_hi, step_max)
        x = torch.cat([q0.unsqueeze(1), cmd], dim=1)
        qa, _va, _ig = sim.rollout(x, qa0=q0, va0=v0, kp=kp, kd=kd, Minv=Minv)
        err = (qa[:, 1:] - ref) ** 2
        rms_all = torch.sqrt((err * m_all.unsqueeze(-1)).sum((0, 1))
                             / m_all.sum().clamp_min(1.0))
        rms_str = torch.sqrt((err * m_str.unsqueeze(-1)).sum((0, 1))
                             / m_str.sum().clamp_min(1.0))
        out[float(d)] = (rms_all.cpu().numpy(), rms_str.cpu().numpy())
    return out


def argmin_lead(res, key: int):
    ds = np.array(sorted(res))
    curves = np.stack([res[d][key] for d in ds])       # (n_delta, n_joint)
    best = ds[curves.argmin(axis=0)]
    return best, curves, ds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="runs/phase2b_main/best.pt")
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", default="standard")
    p.add_argument("--max-cases", type=int, default=162)
    p.add_argument("--deltas", type=float, nargs="*", default=None,
                   help="seconds; default 0..0.05 in 2.5 ms steps")
    p.add_argument("--strike-window", type=int, default=6)
    p.add_argument("--kp-scale", type=float, default=1.0,
                   help="ablation multiplier ON TOP of the runtime Kp table")
    p.add_argument("--kd-scale", type=float, default=1.0,
                   help="ablation multiplier ON TOP of the runtime Kd table")
    p.add_argument("--nominal-gains", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out", default="runs/diag7c/cmd_lead.json")
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
    print(f"[fit] {len(cases)} cases, deltas {deltas[0]*1000:.1f}..{deltas[-1]*1000:.1f} ms",
          flush=True)

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

    res = sweep(Q, mask, hit, q0, v0, sim, rnea, q_lo, q_hi, step_max, deltas,
                A.CONTROL_DT, args.strike_window, args.nominal_gains, args.seed)

    out = {"ckpt": args.ckpt, "n_cases": len(cases), "deltas_s": list(map(float, deltas)),
           "strike_window": args.strike_window,
           "nominal_gains": bool(args.nominal_gains)}
    for tag, key in (("all", 0), ("strike", 1)):
        best, curves, ds = argmin_lead(res, key)
        out[f"best_{tag}_s"] = [float(x) for x in best]
        out[f"rms_at_0_{tag}_mrad"] = [float(1000 * x) for x in curves[0]]
        out[f"rms_at_best_{tag}_mrad"] = [
            float(1000 * curves[:, j].min()) for j in range(curves.shape[1])
        ]
        print(f"\n=== window: {tag} ===")
        print(f"{'joint':<28}{'delta ms':>9}{'rms@0 mrad':>12}{'rms@best':>10}"
              f"{'gain %':>8}")
        for j, nm in enumerate(NAMES):
            r0, rb = 1000 * curves[0, j], 1000 * curves[:, j].min()
            print(f"{nm:<28}{1000*best[j]:>9.1f}{r0:>12.1f}{rb:>10.1f}"
                  f"{100*(1 - rb / max(r0, 1e-9)):>8.0f}")
        print("  per GROUP (median of the group's joints):")
        for g, idx in GROUPS.items():
            print(f"    {g:<10} delta {1000*np.median(best[idx]):>5.1f} ms   "
                  f"rms {1000*np.median([curves[:, j].min() for j in idx]):>6.1f} mrad "
                  f"(from {1000*np.median(curves[0, idx]):.1f})")
        # the best UNIFORM delta, for the ablation table
        tot = np.sqrt((curves ** 2).mean(axis=1))
        print(f"  best UNIFORM delta: {1000*ds[tot.argmin()]:.1f} ms "
              f"(all-joint rms {1000*tot.min():.1f} vs {1000*tot[0]:.1f} at 0)")
        out[f"best_uniform_{tag}_s"] = float(ds[tot.argmin()])
        out[f"uniform_curve_{tag}_mrad"] = [float(1000 * x) for x in tot]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n[out] {args.out}")
    print("\ncmd_lead_s (strike window, joint order):")
    print("  " + repr([round(float(x), 4) for x in out["best_strike_s"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
