#!/usr/bin/env python
"""Paired evaluation for the start-conditioned terminal-pose ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amortized_opt.starts import StartSampler  # noqa: E402
from amortized_opt.terminal_pose import (  # noqa: E402
    TerminalPoseConfig,
    TerminalPoseCost,
    TerminalPoseNet,
    collision_violation,
    sample_terminal_batch,
)


DEFAULT_RUNS = {
    "legacy_29k": "runs/terminal_pose/ablate_ncon/plain_10k.pt",
    "inputs_k1": "runs/terminal_pose/start_collision_20260926/control_inputs_k1_10k.pt",
    "start_k1": "runs/terminal_pose/start_collision_20260926/start_k1_10k.pt",
    "start_k8": "runs/terminal_pose/start_collision_20260926/start_k8_10k.pt",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument(
        "--out", default="runs/terminal_pose/start_collision_20260926/metrics.json",
    )
    return p.parse_args()


def make_start_bank(n: int, seed: int, device: torch.device):
    sampler = StartSampler(device=device, dtype=torch.float32)
    rng = np.random.default_rng(seed + 7919)
    chunks = [sampler.sample(min(512, n - i), rng=rng) for i in range(0, n, 512)]
    return torch.cat([x.q for x in chunks]), torch.cat([x.qd for x in chunks])


def load_net(path: str, device: torch.device) -> TerminalPoseNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"]
    optnet = bool(ck.get("optnet", False))
    first = "pre.weight" if optnet else "hidden.0.weight"
    in_dim = int(ck.get("in_dim", state[first].shape[1]))
    net = TerminalPoseNet(
        in_dim=in_dim,
        width=int(ck.get("width", state[first].shape[0])),
        depth=int(ck.get("depth", 2)),
        optnet=optnet,
        n_constraints=int(ck.get("n_constraints", 4)),
        centered_qp=bool(ck.get("centered_qp", False)),
        mix_width=ck.get("mix_width"),
        qp_out=bool(ck.get("qp_out", False)),
    ).to(device)
    net.load_state_dict(state)
    return net.eval()


def quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float(), q).item()) if x.numel() else float("nan")


def evaluate(net, batch, cost) -> dict[str, float]:
    with torch.no_grad():
        q = net(batch.features[:, : net.in_dim])
        out = cost(q, batch, None)
    hit = out.aux["hit_dist"][batch.active]
    col = collision_violation(out.aux["min_sdf"])
    start = out.aux["start_rmse_rad"]
    accel = out.aux["accel_ratio_rms"]
    return {
        "params": int(net.n_parameters()),
        "hit_p50_mm": 1000.0 * quantile(hit, 0.5),
        "hit_p90_mm": 1000.0 * quantile(hit, 0.9),
        "hit_lt_1cm": float((hit < 0.01).float().mean().item()),
        "collision_p50_mm": 1000.0 * quantile(col, 0.5),
        "collision_p90_mm": 1000.0 * quantile(col, 0.9),
        "collision_clean": float((col <= 0).float().mean().item()),
        "start_rmse_p50_rad": quantile(start, 0.5),
        "start_rmse_p90_rad": quantile(start, 0.9),
        "accel_ratio_p50": quantile(accel, 0.5),
        "accel_ratio_p90": quantile(accel, 0.9),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    q0, qd0 = make_start_bank(args.n, args.seed, device)
    batch = sample_terminal_batch(
        args.n,
        rng=np.random.default_rng(args.seed),
        device=device,
        dtype=torch.float32,
        q_start=q0,
        qd_start=qd0,
    )
    cost = TerminalPoseCost(TerminalPoseConfig(), device=device, dtype=torch.float32)
    result = {
        "n": int(args.n),
        "seed": int(args.seed),
        "runs": {
            name: evaluate(load_net(path, device), batch, cost)
            for name, path in DEFAULT_RUNS.items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
