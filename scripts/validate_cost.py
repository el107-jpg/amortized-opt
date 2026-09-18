"""Validate the fused cost against the recorded trajopt baseline.

    python scripts/validate_cost.py --device cuda:1

Four checks, all from ``baselines/trajopt_hpipm_2026-08-10*``:

1. **Hinge separation** -- the distribution of every hinge term over cases the
   harness ACCEPTED vs cases it rejected, and the per-term AUROC against that
   term's own harness failure flag (``ok_torque``, ``ok_collision``, ...).  A
   cost whose terms do not separate the scorer's own verdicts is the wrong cost.
2. **Monotonicity under noise** -- growing joint-space perturbations (1 mm ->
   5 cm) applied to accepted trajectories must raise the total cost.
3. **Descent** -- 50 Adam steps on the waypoints, from a perturbed accepted
   trajectory, must move the HARNESS's numbers (contact error, violation
   counts) back toward the unperturbed solution.
4. **Finiteness** of the gradient at training scale.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402

BASE = Path(__file__).resolve().parent.parent / "baselines"
#: v2 is the reference line (waist-aim policy on); v1 is kept for provenance.
PREFIX = "trajopt_hpipm_2026-08-10_v2"


# ---------------------------------------------------------------- helpers --
def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(score(positive) > score(negative)), ties at 0.5.  NaN if degenerate."""
    pos = scores[positive]
    neg = scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def load_baseline(prefix: str = PREFIX):
    if not (BASE / f"{prefix}_cases.json").exists():
        prefix = "trajopt_hpipm_2026-08-10"
    cases, trajs, _meta = A.load_run(BASE, prefix=prefix)
    cases = [c for c in cases if c.case_id in trajs]
    rows = {
        int(r["case_id"]): r
        for r in csv.DictReader((BASE / f"{prefix}_per_case.csv").open())
    }
    return cases, trajs, rows


def pack(cases, trajs, H, device, dtype):
    B = len(cases)
    q0 = torch.zeros(B, 17, dtype=dtype)
    v0 = torch.zeros(B, 17, dtype=dtype)
    wp = torch.zeros(B, H, 17, dtype=dtype)
    for i, c in enumerate(cases):
        Q = trajs[c.case_id]
        q0[i] = torch.as_tensor(Q[0], dtype=dtype)
        v0[i] = torch.as_tensor(c.qdot_start, dtype=dtype)
        wp[i, : c.N] = torch.as_tensor(Q[1 : c.N + 1], dtype=dtype)
        wp[i, c.N :] = torch.as_tensor(Q[c.N], dtype=dtype)  # hold (masked anyway)
    return (q0.to(device), wp.to(device), v0.to(device))


def unpack(wp: torch.Tensor, q0: torch.Tensor, cases) -> dict[int, np.ndarray]:
    out = {}
    w = wp.detach().cpu().numpy()
    z = q0.detach().cpu().numpy()
    for i, c in enumerate(cases):
        out[c.case_id] = np.concatenate([z[i][None, :], w[i, : c.N]], axis=0)
    return out


# ------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n-sub", type=int, default=1)
    ap.add_argument("--physics-mode", default="knot_fd")
    ap.add_argument("--descent-cases", type=int, default=32)
    ap.add_argument("--descent-steps", type=int, default=300)
    ap.add_argument("--descent-lr", type=float, default=3e-3)
    ap.add_argument("--descent-noise", type=float, default=0.02)
    ap.add_argument("--prefix", default=PREFIX)
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float64 if dev.type == "cpu" else torch.float32

    cases, trajs, rows = load_baseline(args.prefix)
    H = max(c.N for c in cases)
    cfg = A.CostConfig(n_sub=args.n_sub, physics_mode=args.physics_mode)
    cost = A.FusedCost(cfg, max_horizon=H, device=dev, dtype=dtype)
    ctx = A.strike_context_from_cases(
        cases, drum_index=cost.drum_index, device=dev, dtype=dtype
    )
    q0, wp, v0 = pack(cases, trajs, H, dev, dtype)

    flag = lambda n: np.array([rows[c.case_id][n] == "True" for c in cases])
    accepted = flag("accepted")
    print(f"cases={len(cases)}  accepted={accepted.sum()} ({accepted.mean():.1%})  "
          f"H={H}  n_sub={args.n_sub}  physics_mode={args.physics_mode}  dtype={dtype}")

    out = cost.forward(q0, wp, v0, ctx)
    raw = {k: v.detach().cpu().numpy() for k, v in out.raw.items()}
    diag = {k: v.detach().cpu().numpy() for k, v in out.diagnostics.items()}
    total = out.total.detach().cpu().numpy()
    hand_coll = np.array(
        [diag[f"collision_hinge_{c.hand}"][i] for i, c in enumerate(cases)]
    )
    raw["collision_striking_hand"] = hand_coll

    # ---------------------------------------------------- 1. separation ----
    print("\n== 1. hinge term: accepted vs rejected (median [p90]) ==")
    hdr = f"{'term':26s} {'accepted':>22s} {'rejected':>22s} {'ratio':>8s} {'AUROC':>7s}"
    print(hdr)
    print("-" * len(hdr))
    pair = {
        "torque": "ok_torque",
        "velocity": "ok_velocity",
        "joint_box": "ok_joint_limits",
        "collision_striking_hand": "ok_collision",
        "terminal_region": "ok_hit",
        "terminal_velocity": "ok_speed",
    }
    for name in ("terminal_region", "terminal_velocity", "terminal_elev",
                 "collision_striking_hand", "collision_drum", "collision_under",
                 "collision_self", "torque", "velocity", "joint_box",
                 "smooth_acc", "smooth_jerk", "posture_mid", "posture_elbow"):
        v = raw[name]
        a, r = v[accepted], v[~accepted]
        ratio = (np.median(r) + 1e-12) / (np.median(a) + 1e-12)
        au = auroc(v, ~accepted)
        print(f"{name:26s} {np.median(a):9.3e}[{q(a,90):8.2e}] "
              f"{np.median(r):9.3e}[{q(r,90):8.2e}] {ratio:8.2f} {au:7.3f}")
    print(f"{'TOTAL':26s} {np.median(total[accepted]):9.3e}"
          f"[{q(total[accepted],90):8.2e}] {np.median(total[~accepted]):9.3e}"
          f"[{q(total[~accepted],90):8.2e}] "
          f"{(np.median(total[~accepted])+1e-12)/(np.median(total[accepted])+1e-12):8.2f} "
          f"{auroc(total, ~accepted):7.3f}")

    print("\n== 1b. per-term AUROC against that term's own harness flag ==")
    for term, fl in pair.items():
        bad = ~flag(fl)
        if bad.sum() == 0:
            print(f"{term:26s} vs !{fl:18s}  (no failures in the baseline)")
            continue
        print(f"{term:26s} vs !{fl:18s} AUROC={auroc(raw[term], bad):.3f}  "
              f"n_fail={int(bad.sum())}  "
              f"median fail/ok = {np.median(raw[term][bad]):.3e} / "
              f"{np.median(raw[term][~bad]):.3e}")

    # -------------------------------------------------- 2. monotonicity ----
    print("\n== 2. total cost vs joint-space noise on ACCEPTED cases ==")
    sel = np.flatnonzero(accepted)
    idx = torch.as_tensor(sel, device=dev)
    q0a, wpa, v0a = q0[idx], wp[idx], v0[idx]
    ctxa = A.StrikeContext(**{
        k: (None if v is None else v[idx]) for k, v in ctx.__dict__.items()
    })
    base_cost = cost.forward(q0a, wpa, v0a, ctxa).total.detach()
    print(f"{'sigma (rad)':>12s} {'mean total':>12s} {'delta':>12s} "
          f"{'frac raised':>12s}")
    prev = base_cost
    curve = []
    ok_mono = True
    for sigma in (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        g = torch.Generator(device="cpu").manual_seed(1234)
        acc = []
        for rep in range(4):
            n = torch.randn(wpa.shape, generator=g).to(device=dev, dtype=dtype) * sigma
            c = cost.forward(q0a, wpa + n, v0a, ctxa).total.detach()
            acc.append(c)
        cur = torch.stack(acc).mean(0)
        frac = float((cur >= prev - 1e-9).to(torch.float32).mean())
        curve.append((sigma, float(cur.mean()), float((cur - base_cost).mean()), frac))
        print(f"{sigma:12.4f} {float(cur.mean()):12.4f} "
              f"{float((cur-base_cost).mean()):12.4f} {frac:12.3f}")
        if sigma > 0 and float(cur.mean()) < float(prev.mean()):
            ok_mono = False
        prev = cur
    print("monotone in the mean:", ok_mono)

    # ------------------------------------------------------- 3. descent ----
    print(f"\n== 3. {args.descent_steps} Adam steps from a perturbed accepted "
          f"trajectory (sigma={args.descent_noise} rad) ==")
    take = sel[: args.descent_cases]
    dcases = [cases[i] for i in take]
    ti = torch.as_tensor(take, device=dev)
    q0d, wpd, v0d = q0[ti], wp[ti], v0[ti]
    ctxd = A.StrikeContext(**{
        k: (None if v is None else v[ti]) for k, v in ctx.__dict__.items()
    })
    g = torch.Generator(device="cpu").manual_seed(7)
    noise = torch.randn(wpd.shape, generator=g).to(device=dev, dtype=dtype)
    noise = noise * args.descent_noise
    start = (wpd + noise).clone().requires_grad_(True)
    opt = torch.optim.Adam([start], lr=args.descent_lr)
    c0 = float(cost.forward(q0d, start, v0d, ctxd).total.mean())

    scorer = A.TrajectoryScorer(device="cpu")
    def summarize(wp_t, label):
        Qs = unpack(wp_t, q0d, dcases)
        sc = [scorer.score(c, Qs[c.case_id]) for c in dcases]
        m = dict(
            contact_pos_err_mm=np.mean([s.contact_pos_err_mm for s in sc]),
            contact_speed_err=np.mean([s.contact_speed_err_mps for s in sc]),
            on_head=np.mean([s.on_head for s in sc]),
            coll_pen_count=np.mean([s.coll_pen_count for s in sc]),
            tau_excess=np.mean([s.tau_excess_max_nm for s in sc]),
            vel_excess=np.mean([s.vel_excess_max_rads for s in sc]),
            qpos_excess=np.mean([s.qpos_posture_excess_max_rad for s in sc]),
            accepted=np.mean([s.accepted for s in sc]),
            jerk=np.mean([s.jerk_integral for s in sc]),
        )
        print(f"  {label:12s} " + "  ".join(f"{k}={v:.4g}" for k, v in m.items()))
        return m

    m_ref = summarize(wpd, "trajopt")
    m_pert = summarize(wpd + noise, "perturbed")
    done = 0
    for target in sorted({50, args.descent_steps}):
        while done < target:
            opt.zero_grad(set_to_none=True)
            cost.forward(q0d, start, v0d, ctxd).total.sum().backward()
            opt.step()
            done += 1
        m_opt = summarize(start, f"adam {done}")
    c1 = float(cost.forward(q0d, start, v0d, ctxd).total.mean())
    print(f"  cost mean: {c0:.4f} -> {c1:.4f}")
    # Metrics the cost actually prices.  ``contact_pos_err_mm`` is distance to
    # the AIM point, which ``w_terminal_aim = 0`` deliberately does NOT price
    # (matching trajopt's ``validate()``); the cost prices hit-DISK membership,
    # i.e. ``on_head``.  It is reported above but excluded from the tally.
    lower_is_better = ("contact_speed_err", "coll_pen_count", "tau_excess",
                       "vel_excess", "qpos_excess", "jerk")
    better = [k for k in lower_is_better if m_opt[k] <= m_pert[k] + 1e-9]
    worse = [k for k in lower_is_better if k not in better]
    print(f"  priced metrics improved: {len(better)}/{len(lower_is_better)} "
          f"{better}" + (f"  NOT improved: {worse}" if worse else ""))
    print(f"  on_head {m_pert['on_head']:.3f} -> {m_opt['on_head']:.3f} "
          f"(trajopt {m_ref['on_head']:.3f}); "
          f"accepted {m_pert['accepted']:.2f} -> {m_opt['accepted']:.2f} "
          f"(trajopt {m_ref['accepted']:.2f})")
    print("  (contact_pos_err_mm is distance to the AIM point; "
          "CostConfig.w_terminal_aim = 0 by default, so validate() does not "
          "price it; phase2_cost_config turns on a weak in-plane pull)")

    # ---------------------------------------------------- 4. finiteness ----
    print("\n== 4. gradient finiteness at training scale ==")
    Bt, Ht = 256, 50
    c2 = A.FusedCost(A.CostConfig(n_sub=args.n_sub, physics_mode=args.physics_mode),
                     max_horizon=Ht, device=dev, dtype=torch.float32)
    gg = torch.Generator(device="cpu").manual_seed(0)
    z0 = (torch.randn(Bt, 17, generator=gg) * 0.2).to(dev, torch.float32)
    stp = (torch.randn(Bt, Ht, 17, generator=gg) * 0.03).to(dev, torch.float32)
    W = (z0.unsqueeze(1) + torch.cumsum(stp, 1)).requires_grad_(True)
    V0 = (torch.randn(Bt, 17, generator=gg) * 0.5).to(dev, torch.float32)
    kl = torch.randint(20, Ht + 1, (Bt,), generator=gg).to(dev)
    kr = torch.randint(20, Ht + 1, (Bt,), generator=gg).to(dev)
    D = c2.n_drums
    ones = torch.ones(Bt, 2, device=dev)
    ctx2 = A.StrikeContext(
        hit_index=torch.stack([kl, kr], 1), active=ones,
        target_drum=torch.randint(0, D, (Bt, 2), generator=gg).to(dev),
        start_drum=torch.full((Bt, 2), -1, device=dev, dtype=torch.long),
        p_star=torch.full((Bt, 2, 3), 0.3, device=dev),
        v_star=torch.full((Bt, 2, 3), -1.0, device=dev),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=dev).expand(Bt, 2, 3).contiguous(),
        hit_radius=torch.full((Bt, 2), 0.08, device=dev),
        disk_offset=torch.zeros(Bt, 2, 3, device=dev),
        half_plane_dir=torch.zeros(Bt, 2, 3, device=dev),
        has_half_plane=torch.zeros(Bt, 2, device=dev),
        contact_sigma=torch.full((Bt, 2), 0.2, device=dev),
        shaft_allowed=ones, start_shaft_allowed=torch.zeros(Bt, 2, device=dev),
        elev_required=ones,
    )
    o = c2.forward(z0, W, V0, ctx2)
    o.total.sum().backward()
    gnorm = float(W.grad.norm())
    print(f"total finite: {bool(torch.isfinite(o.total).all())}  "
          f"grad finite: {bool(torch.isfinite(W.grad).all())}  "
          f"|grad|={gnorm:.4g}  max|grad|={float(W.grad.abs().max()):.4g}")


if __name__ == "__main__":
    main()
