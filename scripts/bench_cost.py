"""Throughput / sync benchmark for the fused cost (section 3).

    python scripts/bench_cost.py --device cuda:1 --batch 256 --horizon 50

Reports eager and ``torch.compile(mode="reduce-overhead")`` forward+backward
wall time at each sub-sample count, and checks the hot path is free of
device->host syncs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import amortized_opt as A  # noqa: E402


def make_batch(cost, B, H, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q0 = torch.randn(B, 17, generator=g).to(device=device, dtype=dtype) * 0.2
    v0 = torch.randn(B, 17, generator=g).to(device=device, dtype=dtype) * 0.5
    step = torch.randn(B, H, 17, generator=g).to(device=device, dtype=dtype) * 0.02
    wp = (q0.unsqueeze(1) + torch.cumsum(step, dim=1)).requires_grad_(True)

    kl = torch.randint(H // 2, H + 1, (B,), generator=g).to(device)
    kr = torch.randint(H // 2, H + 1, (B,), generator=g).to(device)
    hit = torch.stack([kl, kr], dim=1)
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    D = cost.n_drums
    tgt = torch.randint(0, D, (B, 2), generator=g).to(device)
    ctx = A.StrikeContext(
        hit_index=hit,
        active=ones,
        target_drum=tgt,
        start_drum=torch.full((B, 2), -1, device=device, dtype=torch.long),
        p_star=torch.zeros(B, 2, 3, device=device, dtype=dtype) + 0.3,
        v_star=torch.zeros(B, 2, 3, device=device, dtype=dtype) - 1.0,
        n_out=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, device=device, dtype=dtype),
        disk_offset=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        half_plane_dir=torch.zeros(B, 2, 3, device=device, dtype=dtype),
        has_half_plane=torch.zeros(B, 2, device=device, dtype=dtype),
        contact_sigma=torch.full((B, 2), 0.2, device=device, dtype=dtype),
        shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        start_shaft_allowed=torch.zeros(B, 2, device=device, dtype=dtype),
        elev_required=ones,
    )
    return q0, wp, v0, ctx


def timeit(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--n-sub", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--physics-mode", default="knot_fd")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)  # MANDATORY for CUDA-graph capture
    torch.set_float32_matmul_precision("high")

    B, H = args.batch, args.horizon
    print(f"device={dev} batch={B} horizon={H} physics_mode={args.physics_mode}")
    for n_sub in args.n_sub:
        cfg = A.CostConfig(n_sub=n_sub, physics_mode=args.physics_mode)
        cost = A.FusedCost(cfg, max_horizon=H, device=dev, dtype=torch.float32)
        q0, wp, v0, ctx = make_batch(cost, B, H, dev, torch.float32)

        def step():
            out = cost.forward(q0, wp, v0, ctx)
            out.total.sum().backward()
            wp.grad = None

        eager = timeit(step)

        def fn(q0_, wp_, v0_):
            return cost.forward(q0_, wp_, v0_, ctx).total

        compiled = torch.compile(fn, mode="reduce-overhead")

        def step_c():
            out = compiled(q0, wp, v0)
            out.sum().backward()
            wp.grad = None

        try:
            comp = timeit(step_c, warmup=8)
        except Exception as exc:  # pragma: no cover
            comp = float("nan")
            print(f"  compile failed: {type(exc).__name__}: {exc}")
        n_poses = B * (H * n_sub + 1)
        print(
            f"  n_sub={n_sub} ({n_poses:>7d} FK poses): "
            f"eager {eager:7.2f} ms ({1e3/eager:5.1f} st/s)   "
            f"compiled {comp:7.2f} ms ({1e3/comp:5.1f} st/s)"
        )

    # --------------------------------------------------------- sync check --
    cfg = A.CostConfig(n_sub=args.n_sub[0], physics_mode=args.physics_mode)
    cost = A.FusedCost(cfg, max_horizon=H, device=dev, dtype=torch.float32)
    q0, wp, v0, ctx = make_batch(cost, 8, H, dev, torch.float32)
    cost.forward(q0, wp, v0, ctx).total.sum().backward()
    wp.grad = None
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = cost.forward(q0, wp, v0, ctx)
        out.total.sum().backward()
        print("sync check: CLEAN (no device->host sync in fwd+bwd)")
    finally:
        torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    main()
