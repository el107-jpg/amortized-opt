#!/usr/bin/env python
"""Causal counterfactuals for trained terminal-pose QP layers.

Compare the deployed QP solution with the same learned quadratic after removing
the inequalities, and after removing both equality and inequality constraints.
This distinguishes an active solver from a useful task-space inductive bias.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amortized_opt.terminal_pose import (  # noqa: E402
    TerminalPoseConfig,
    TerminalPoseCost,
    TerminalPoseNet,
    collision_violation,
    sample_terminal_batch,
)


RUNS = {
    "qpout": "runs/terminal_pose/ablate_feasible_qpth_20260926/qpout_10k.pt",
    "mix32": "runs/terminal_pose/ablate_feasible_qpth_20260926/mix32_10k.pt",
    "mix128": "runs/terminal_pose/ablate_feasible_qpth_20260926/mix128_10k.pt",
}


def equality_solution(Q, p, A, b):
    """Minimize 1/2 z'Qz + p'z subject to Az=b."""
    qinv_p = torch.linalg.solve(Q, p.unsqueeze(-1)).squeeze(-1)
    qinv_at = torch.linalg.solve(Q, A.transpose(1, 2))
    gram = A @ qinv_at
    rhs = b + (A @ qinv_p.unsqueeze(-1)).squeeze(-1)
    dual = torch.linalg.solve(gram, rhs.unsqueeze(-1))
    return -qinv_p + (qinv_at @ dual).squeeze(-1)


def pose_from_latent(net, z, h2):
    if net.qp_out:
        return net._soft_clip(z.to(h2.dtype))
    skipped = h2[:, : net.skip_dim]
    mixed = F.silu(net.mix(torch.cat([z.to(h2.dtype), skipped], dim=-1)))
    raw = net.head(mixed)
    q = z.to(h2.dtype) + net.q_half.to(raw.dtype) * torch.tanh(raw)
    return net._soft_clip(q)


def metrics(q, batch, cost):
    with torch.no_grad():
        out = cost(q, batch, None)
    hit = out.aux["hit_dist"][batch.active]
    col = collision_violation(out.aux["min_sdf"])
    return {
        "hit_p50_mm": 1000.0 * float(hit.median()),
        "hit_lt_1cm": float((hit < 0.01).float().mean()),
        "collision_clean": float((col <= 0).float().mean()),
        "collision_p90_mm": 1000.0 * float(torch.quantile(col.float(), 0.9)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument(
        "--out", default="runs/terminal_pose/ablate_feasible_qpth_20260926/causal.json",
    )
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch = sample_terminal_batch(
        args.n, rng=np.random.default_rng(args.seed), device=device, dtype=torch.float32,
    )
    cost = TerminalPoseCost(
        TerminalPoseConfig(w_start=0.0, w_motion=0.0), device=device,
    )
    result = {"n": args.n, "seed": args.seed, "runs": {}}
    for name, path in RUNS.items():
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck["model"]
        in_dim = int(state["pre.weight"].shape[1])
        net = TerminalPoseNet(
            in_dim=in_dim,
            width=int(state["pre.weight"].shape[0]),
            optnet=True,
            n_constraints=int(ck.get("n_constraints", 4)),
            mix_width=ck.get("mix_width"),
            qp_out=bool(ck.get("qp_out", False)),
        ).to(device).eval()
        net.load_state_dict(state)
        feats = batch.features[:, :in_dim]
        poses = {k: [] for k in ("full_qp", "equality_only", "unconstrained")}
        latent = {k: [] for k in ("ineq_shift", "eq_shift", "full_shift")}
        active_rates = []
        for begin in range(0, args.n, 64):
            x = feats[begin : begin + 64]
            with torch.no_grad():
                h1 = F.silu(net.pre(x))
                h2 = F.silu(net.enc(h1))
                Q, p, A, b, G, lower, upper, _ = net.qp.qp_data(h2)
                Qd, pd, Ad, bd = Q.double(), p.double(), A.double(), b.double()
                z_unc = -torch.linalg.solve(Qd, pd.unsqueeze(-1)).squeeze(-1)
                z_eq = equality_solution(Qd, pd, Ad, bd)
                z_qp = net.qp(h2, net.q_lo, net.q_hi, net.q_ready).double()
                gz = (G.double() @ z_qp.unsqueeze(-1)).squeeze(-1)
                slack = torch.minimum(gz - lower.double(), upper.double() - gz)
                active_rates.append((slack < 1e-5).float().mean())
                poses["full_qp"].append(pose_from_latent(net, z_qp, h2))
                poses["equality_only"].append(pose_from_latent(net, z_eq, h2))
                poses["unconstrained"].append(pose_from_latent(net, z_unc, h2))
                latent["ineq_shift"].append((z_qp - z_eq).norm(dim=-1))
                latent["eq_shift"].append((z_eq - z_unc).norm(dim=-1))
                latent["full_shift"].append((z_qp - z_unc).norm(dim=-1))
        poses = {k: torch.cat(v) for k, v in poses.items()}
        latent = {k: torch.cat(v) for k, v in latent.items()}
        full = poses["full_qp"]
        record = {
            "active_inequality_fraction": float(torch.stack(active_rates).mean()),
            **{f"latent_{k}_p50_rad": float(v.median()) for k, v in latent.items()},
            "pose_eq_counterfactual_p50_rad": float(
                (full - poses["equality_only"]).norm(dim=-1).median()
            ),
            "pose_unconstrained_counterfactual_p50_rad": float(
                (full - poses["unconstrained"]).norm(dim=-1).median()
            ),
            "metrics": {k: metrics(v, batch, cost) for k, v in poses.items()},
        }
        result["runs"][name] = record
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
