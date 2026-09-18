#!/usr/bin/env python
"""Go / no-go for the tracking-torque estimator: FIR estimate vs the real plant.

    python scripts/validate_track_torque.py                       # 81-case grid
    python scripts/validate_track_torque.py --ckpt runs/.../best.pt --json out.json

The cost's tracking-torque term prices ``tau_hat = tau_rnea + tau_fb``, where
``tau_fb`` is a per-joint FIR bank identified once against the plant
(:func:`amortized_opt.pd_sim.extract_torque_kernels`) and convolved with the
commanded waypoints.  The GROUND TRUTH is the demand the plant itself reports
now that ``simulate_pd(return_torque=True)`` stops throwing it away.  This script
puts the two side by side on real plans from a real checkpoint and reports, per
joint:

* **corr**       -- Pearson correlation of the two series, pooled over cases;
* **peak ratio** -- ``max|est| / max|sim|``, the number the hinge is a hinge on;
* **lag**        -- the shift (in control ticks) that maximizes the normalized
  cross-correlation, per case, reported as the median.  A biased estimator that
  fires a tick late is useless for a peak that lasts one tick.

PASS: the wrist peak ratios (the joints that actually saturate: 5 N*m ratings)
within +/-20 %, and |lag| <= 1 tick everywhere.

The estimator's two approximations are printed alongside so a failure can be
attributed rather than guessed at: the kernel is identified with a FROZEN
DIAGONAL mobility at the ready pose while the plant runs the full coupled
``Minv`` frozen per knot, and the per-tick reduction is a max over substeps,
which is linear only while the worst substep is the same one for every input.
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
from amortized_opt.constants import CONTROL_DT, PD_SUBSTEPS, ready_joint_pos  # noqa: E402
from amortized_opt.pd_sim import (  # noqa: E402
    BatchedPDSim,
    PDSimConfig,
    extract_torque_kernels,
    feedback_torque,
    frozen_minv,
    mass_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
F64 = torch.float64


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", default="runs/phase2_19m_grooves/best.pt")
    p.add_argument("--stride", type=int, default=3,
                   help="grid subsample: generate_grid()[::stride]")
    p.add_argument("--buckets", type=int, nargs="+", default=[25, 50])
    p.add_argument("--max-lag", type=int, default=4)
    p.add_argument("--peak-tol", type=float, default=0.20,
                   help="allowed |peak ratio - 1| on the wrists")
    p.add_argument("--json", default=None, help="write the full table here")
    return p.parse_args(argv)


def lag_of_peak_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Shift of ``a`` against ``b`` maximizing normalized cross-correlation.

    Positive = the estimate LEADS the plant.  NaN when either series is flat.
    """
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    best, best_l = -2.0, 0
    for l in range(-max_lag, max_lag + 1):
        if l >= 0:
            x, y = a[l:], b[: len(b) - l] if l else b
        else:
            x, y = a[: len(a) + l], b[-l:]
        if len(x) < 3:
            continue
        c = float(np.dot(x, y) / (na * nb))
        if c > best:
            best, best_l = c, l
    return float(best_l)


def main(argv=None) -> int:
    args = parse_args(argv)
    dev = "cpu"
    names = list(A.G1_UPPER_BODY_JOINTS)
    n = len(names)

    # ---- model + plans ------------------------------------------------------
    ck_path = Path(args.ckpt)
    if not ck_path.is_absolute():
        ck_path = ROOT / ck_path
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    mcfg = A.model_config_from_checkpoint(ck)
    model = A.build_model(mcfg, dev)
    model.load_state_dict(ck["model"])
    model.eval()

    scorer = A.TrajectoryScorer(device="cpu")
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()
    cases = A.generate_grid(seed=0, grid="standard", scorer=scorer)[:: args.stride]
    print(f"[cases] {len(cases)} (grid[::{args.stride}]), buckets {tuple(args.buckets)}",
          flush=True)
    plans, wall = A.batched_model_plans(
        model, cases, device=dev, kit=kit, library=library,
        buckets=tuple(args.buckets), num_drums=mcfg.num_drums,
        num_future=mcfg.num_future, iterations=1, init="quintic",
    )
    print(f"[plans] {len(plans)} in {wall:.1f} s", flush=True)

    # ---- the estimator's constants -----------------------------------------
    rnea = scorer.rnea                       # float64, armature on
    kp = torch.as_tensor(A.isaac_runtime_stiffness(names), dtype=F64)
    kd = torch.as_tensor(A.isaac_runtime_damping(names), dtype=F64)
    with torch.no_grad():
        M = mass_matrix(rnea, torch.as_tensor(ready_joint_pos(names), dtype=F64).view(1, -1))
        minv_diag = torch.diagonal(torch.linalg.inv(M)[0])
    kernel = extract_torque_kernels(kp, kd, minv_diag, dt=CONTROL_DT,
                                    substeps=PD_SUBSTEPS)
    L = int(kernel.shape[1])
    taps = torch.arange(L, dtype=F64)
    k_dc = kernel.sum(1).numpy()                       # sum_l k_l
    k_m1 = (kernel * taps).sum(1).numpy()              # sum_l l k_l
    k_m2 = (kernel * taps * taps).sum(1).numpy() / 2.0  # ~ the effective mass
    print(f"[kernel] {L} taps; |sum k| max {np.abs(k_dc).max():.3g} N*m/rad, "
          f"|sum l k| max {np.abs(k_m1).max():.3g}", flush=True)

    sim = BatchedPDSim(PDSimConfig(dt=CONTROL_DT), device=dev, dtype=F64, rnea=rnea)
    tau_rating = A.isaac_effort_limits(names)

    # ---- per case: estimate vs plant ---------------------------------------
    est_all = [[] for _ in range(n)]
    sim_all = [[] for _ in range(n)]
    lags = [[] for _ in range(n)]
    peak_e = np.zeros(n)
    peak_s = np.zeros(n)
    over_e = np.zeros(n)
    over_s = np.zeros(n)
    n_tick = 0
    for case in cases:
        Q = np.asarray(plans[int(case.case_id)], float)      # (N+1, 17)
        dt = float(case.dt)
        qd = np.empty_like(Q)
        qd[0] = case.qdot_start
        qd[1:] = np.diff(Q, axis=0) / dt
        qdd = np.empty_like(Q)
        qdd[:-1] = np.diff(qd, axis=0) / dt
        qdd[-1] = qdd[-2]

        Qt = torch.as_tensor(Q, dtype=F64).unsqueeze(0)      # (1, N+1, n)
        with torch.no_grad():
            # --- the estimate: RNEA of the plan + DC-removed FIR feedback ----
            tau_r = rnea.rnea(
                Qt[0], torch.as_tensor(qd, dtype=F64), torch.as_tensor(qdd, dtype=F64)
            )
            est = (tau_r + feedback_torque(Qt, kernel)[0]).numpy()   # (N+1, n)

            # --- the truth: the plant's own demanded-torque series -----------
            # Open loop over the whole plan, nominal gains, the full coupled
            # Minv frozen at each period's starting knot -- the same rollout the
            # torque audit was run with.
            Minv = frozen_minv(rnea, Qt[:, :-1])
            kp_b, kd_b, _ki = sim.nominal_gains(1)
            _qa, _va, _ig, tau_sim = sim.rollout(
                Qt, qa0=torch.as_tensor(case.q_start, dtype=F64).view(1, -1),
                va0=torch.as_tensor(case.qdot_start, dtype=F64).view(1, -1),
                kp=kp_b, kd=kd_b, Minv=Minv, return_torque=True,
            )
        gt = tau_sim[0].numpy()                              # (N, n), ticks 1..N
        eh = est[1:]                                         # aligned to those
        n_tick += gt.shape[0]
        for j in range(n):
            est_all[j].append(eh[:, j])
            sim_all[j].append(gt[:, j])
            lags[j].append(lag_of_peak_xcorr(eh[:, j], gt[:, j], args.max_lag))
            peak_e[j] = max(peak_e[j], np.abs(eh[:, j]).max())
            peak_s[j] = max(peak_s[j], np.abs(gt[:, j]).max())
            over_e[j] += float(np.sum(np.abs(eh[:, j]) > tau_rating[j]))
            over_s[j] += float(np.sum(np.abs(gt[:, j]) > tau_rating[j]))

    # ---- report -------------------------------------------------------------
    rows = []
    for j, nm in enumerate(names):
        a = np.concatenate(est_all[j])
        b = np.concatenate(sim_all[j])
        sa, sb = a.std(), b.std()
        corr = float(np.corrcoef(a, b)[0, 1]) if sa > 1e-12 and sb > 1e-12 else float("nan")
        lg = np.array([x for x in lags[j] if np.isfinite(x)])
        rows.append({
            "joint": nm,
            "rating_nm": float(tau_rating[j]),
            "corr": corr,
            "peak_sim_nm": float(peak_s[j]),
            "peak_est_nm": float(peak_e[j]),
            "peak_ratio": float(peak_e[j] / max(peak_s[j], 1e-12)),
            "lag_median": float(np.median(lg)) if lg.size else float("nan"),
            "lag_max_abs": float(np.abs(lg).max()) if lg.size else float("nan"),
            "frac_over_sim": float(over_s[j] / max(n_tick, 1)),
            "frac_over_est": float(over_e[j] / max(n_tick, 1)),
            "kernel_dc": float(k_dc[j]),
            "kernel_m1": float(k_m1[j]),
            "kernel_m2": float(k_m2[j]),
        })

    hdr = (f"{'joint':<28}{'rat':>6}{'corr':>8}{'pk_sim':>9}{'pk_est':>9}"
           f"{'ratio':>8}{'lag':>6}{'over_s':>8}{'over_e':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['joint']:<28}{r['rating_nm']:6.0f}{r['corr']:8.3f}"
              f"{r['peak_sim_nm']:9.2f}{r['peak_est_nm']:9.2f}{r['peak_ratio']:8.3f}"
              f"{r['lag_median']:6.1f}{r['frac_over_sim']:8.3f}{r['frac_over_est']:8.3f}")

    wrists = [r for r in rows if "wrist_pitch" in r["joint"] or "wrist_yaw" in r["joint"]]
    ok_peak = all(abs(r["peak_ratio"] - 1.0) <= args.peak_tol for r in wrists)
    ok_lag = all(abs(r["lag_median"]) <= 1.0 for r in rows if np.isfinite(r["lag_median"]))
    print(f"\nticks compared: {n_tick}")
    detail = ", ".join(
        "{}={:.3f}".format(r["joint"].split("_", 1)[1], r["peak_ratio"]) for r in wrists
    )
    print(f"PASS wrist peak ratio within +/-{args.peak_tol:.0%}: {ok_peak}  ({detail})")
    print(f"PASS |median lag| <= 1 tick on every joint: {ok_lag}")
    print(f"GO/NO-GO: {'GO' if (ok_peak and ok_lag) else 'NO-GO'}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"rows": rows, "n_ticks": n_tick, "kernel_taps": L,
             "pass_peak": ok_peak, "pass_lag": ok_lag}, indent=2))
    return 0 if (ok_peak and ok_lag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
