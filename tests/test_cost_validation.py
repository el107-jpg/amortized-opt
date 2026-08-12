"""Validate the fused cost against the RECORDED trajopt baseline.

The plan's explicit gate: *"validate it against trajopt's own cost on trajopt's
own solutions -- the QP's optimum should score well under the torch cost.  If it
doesn't, the cost is wrong."*  Three assertions here:

1. every constraint term separates the cases the harness FAILED on that same
   constraint from the ones it passed (AUROC);
2. growing joint-space noise on accepted trajectories monotonically raises the
   total cost;
3. gradient descent on the cost moves the HARNESS's numbers the right way.

Skips cleanly when ``baselines/`` has not been recorded.  The full report with
all the distributions is ``scripts/validate_cost.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import N_DOF

BASE = Path(__file__).resolve().parent.parent / "baselines"
PREFIX = "trajopt_hpipm_2026-08-10_v2"


def _auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = scores[positive], scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(order.size, float)
    ranks[order] = np.arange(1, order.size + 1)
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


@pytest.fixture(scope="module")
def baseline():
    prefix = PREFIX
    if not (BASE / f"{prefix}_cases.json").exists():
        prefix = "trajopt_hpipm_2026-08-10"
    if not (BASE / f"{prefix}_cases.json").exists():
        pytest.skip(f"no recorded baseline at {BASE}")
    cases, trajs, _ = A.load_run(BASE, prefix=prefix)
    cases = [c for c in cases if c.case_id in trajs]
    rows = {int(r["case_id"]): r
            for r in csv.DictReader((BASE / f"{prefix}_per_case.csv").open())}
    H = max(c.N for c in cases)
    cost = A.FusedCost(A.CostConfig(), max_horizon=H, device="cpu",
                       dtype=torch.float32)
    ctx = A.strike_context_from_cases(cases, drum_index=cost.drum_index,
                                      dtype=torch.float32)
    B = len(cases)
    q0 = torch.zeros(B, N_DOF)
    v0 = torch.zeros(B, N_DOF)
    wp = torch.zeros(B, H, N_DOF)
    for i, c in enumerate(cases):
        Q = trajs[c.case_id]
        q0[i] = torch.as_tensor(Q[0], dtype=torch.float32)
        v0[i] = torch.as_tensor(c.qdot_start, dtype=torch.float32)
        wp[i, : c.N] = torch.as_tensor(Q[1 : c.N + 1], dtype=torch.float32)
        wp[i, c.N :] = torch.as_tensor(Q[c.N], dtype=torch.float32)
    return dict(cases=cases, trajs=trajs, rows=rows, cost=cost, ctx=ctx,
                q0=q0, wp=wp, v0=v0, H=H)


@pytest.fixture(scope="module")
def scored(baseline):
    b = baseline
    return b["cost"].forward(b["q0"], b["wp"], b["v0"], b["ctx"])


def test_each_term_separates_its_own_harness_failure(baseline, scored):
    """AUROC of the cost term against the harness flag it is supposed to price."""
    b = baseline
    cases, rows = b["cases"], b["rows"]
    raw = {k: v.detach().numpy() for k, v in scored.raw.items()}
    diag = {k: v.numpy() for k, v in scored.diagnostics.items()}
    # The recorded baseline freezes its idle arm at URDF neutral, which is inside
    # a drum under-volume for EVERY case -- so ok_collision is the STRIKING
    # hand's verdict and must be compared against the striking hand's hinge.
    raw["collision_striking_hand"] = np.array(
        [diag[f"collision_hinge_{c.hand}"][i] for i, c in enumerate(cases)]
    )
    flag = lambda n: np.array([rows[c.case_id][n] == "True" for c in cases])
    expect = {
        "torque": ("ok_torque", 0.90),
        "collision_striking_hand": ("ok_collision", 0.85),
        "terminal_region": ("ok_hit", 0.85),
        "terminal_velocity": ("ok_speed", 0.80),
    }
    report = {}
    for term, (fl, floor) in expect.items():
        bad = ~flag(fl)
        if bad.sum() < 3:
            continue
        au = _auroc(raw[term], bad)
        report[term] = au
        assert au >= floor, f"{term} vs !{fl}: AUROC {au:.3f} < {floor}"
    print("AUROC:", {k: round(v, 3) for k, v in report.items()})
    assert len(report) >= 3


def test_accepted_cases_score_lower_than_rejected(baseline, scored):
    b = baseline
    acc = np.array([b["rows"][c.case_id]["accepted"] == "True" for c in b["cases"]])
    total = scored.total.detach().numpy()
    au = _auroc(total, ~acc)
    print(f"total-cost AUROC vs !accepted = {au:.3f}; "
          f"median accepted {np.median(total[acc]):.3f} "
          f"vs rejected {np.median(total[~acc]):.3f}")
    assert np.median(total[acc]) < np.median(total[~acc])
    assert au > 0.70


def test_cost_rises_monotonically_with_noise(baseline, scored):
    b = baseline
    acc = np.array([b["rows"][c.case_id]["accepted"] == "True" for c in b["cases"]])
    idx = torch.as_tensor(np.flatnonzero(acc))
    ctx = A.StrikeContext(**{k: (None if v is None else v[idx])
                             for k, v in b["ctx"].__dict__.items()})
    q0, wp, v0 = b["q0"][idx], b["wp"][idx], b["v0"][idx]
    cost = b["cost"]
    prev = cost.forward(q0, wp, v0, ctx).total.detach()
    curve = [float(prev.mean())]
    for sigma in (0.001, 0.005, 0.02, 0.05):
        g = torch.Generator().manual_seed(11)
        cur = torch.stack([
            cost.forward(q0, wp + torch.randn(wp.shape, generator=g) * sigma,
                         v0, ctx).total.detach()
            for _ in range(2)
        ]).mean(0)
        frac = float((cur > prev).to(torch.float32).mean())
        curve.append(float(cur.mean()))
        assert float(cur.mean()) > float(prev.mean()), sigma
        assert frac > 0.98, (sigma, frac)
        prev = cur
    print("mean total vs sigma[0, 1mm, 5mm, 2cm, 5cm]:",
          [round(c, 2) for c in curve])


def test_descent_improves_the_harness_score(baseline):
    b = baseline
    cost, cases = b["cost"], b["cases"]
    acc = np.array([b["rows"][c.case_id]["accepted"] == "True" for c in cases])
    sel = np.flatnonzero(acc)[:8]
    idx = torch.as_tensor(sel)
    dcases = [cases[i] for i in sel]
    ctx = A.StrikeContext(**{k: (None if v is None else v[idx])
                             for k, v in b["ctx"].__dict__.items()})
    q0, wp, v0 = b["q0"][idx], b["wp"][idx], b["v0"][idx]
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(wp.shape, generator=g) * 0.01
    start = (wp + noise).clone().requires_grad_(True)
    opt = torch.optim.Adam([start], lr=3e-3)
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        cost.forward(q0, start, v0, ctx).total.sum().backward()
        opt.step()

    scorer = A.TrajectoryScorer(device="cpu")

    def score(w):
        arr = w.detach().numpy()
        z = q0.numpy()
        out = []
        for i, c in enumerate(dcases):
            Q = np.concatenate([z[i][None, :], arr[i, : c.N]], axis=0)
            out.append(scorer.score(c, Q))
        return out

    pert = score(wp + noise)
    done = score(start)
    m = lambda rs, f: float(np.mean([f(r) for r in rs]))
    print("perturbed -> optimized: "
          f"pos {m(pert, lambda r: r.contact_pos_err_mm):.2f} -> "
          f"{m(done, lambda r: r.contact_pos_err_mm):.2f} mm; "
          f"speed {m(pert, lambda r: r.contact_speed_err_mps):.4f} -> "
          f"{m(done, lambda r: r.contact_speed_err_mps):.4f}; "
          f"tau_excess {m(pert, lambda r: r.tau_excess_max_nm):.3f} -> "
          f"{m(done, lambda r: r.tau_excess_max_nm):.3f}")
    assert (m(done, lambda r: r.contact_speed_err_mps)
            < m(pert, lambda r: r.contact_speed_err_mps))
    assert (m(done, lambda r: r.tau_excess_max_nm)
            <= m(pert, lambda r: r.tau_excess_max_nm) + 1e-9)
    assert (m(done, lambda r: r.jerk_integral)
            < m(pert, lambda r: r.jerk_integral))
