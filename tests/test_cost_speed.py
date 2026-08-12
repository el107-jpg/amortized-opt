"""Throughput guard for the fused cost.

Target from the section-3 brief: forward+backward at batch 256 x 50 in <= 20 ms
(>= 50 steps/s) with ``torch.compile(mode="reduce-overhead")``.  The full sweep
over sub-sample counts is ``scripts/bench_cost.py``.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import N_DOF

BATCH, HORIZON = 256, 50
BUDGET_MS = 20.0


def _batch(cost, B, H, device, dtype):
    g = torch.Generator().manual_seed(0)
    q0 = (torch.randn(B, N_DOF, generator=g) * 0.2).to(device=device, dtype=dtype)
    step = (torch.randn(B, H, N_DOF, generator=g) * 0.02).to(device=device, dtype=dtype)
    wp = (q0.unsqueeze(1) + torch.cumsum(step, 1)).requires_grad_(True)
    v0 = (torch.randn(B, N_DOF, generator=g) * 0.5).to(device=device, dtype=dtype)
    kl = torch.randint(H // 2, H + 1, (B,), generator=g).to(device)
    kr = torch.randint(H // 2, H + 1, (B,), generator=g).to(device)
    ones = torch.ones(B, 2, device=device, dtype=dtype)
    ctx = A.StrikeContext(
        hit_index=torch.stack([kl, kr], 1), active=ones,
        target_drum=torch.randint(0, cost.n_drums, (B, 2), generator=g).to(device),
        start_drum=torch.full((B, 2), -1, device=device, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.3, device=device, dtype=dtype),
        v_star=torch.full((B, 2, 3), -1.0, device=device, dtype=dtype),
        n_out=torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        .expand(B, 2, 3).contiguous(),
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


def test_compiled_throughput(cuda_device):
    if not os.environ.get("AMORTIZED_BENCH"):
        pytest.skip("set AMORTIZED_BENCH=1 to run the ~1 min torch.compile benchmark")
    torch.set_float32_matmul_precision("high")
    cost = A.FusedCost(A.CostConfig(n_sub=1), max_horizon=HORIZON,
                       device=cuda_device, dtype=torch.float32)
    q0, wp, v0, ctx = _batch(cost, BATCH, HORIZON, cuda_device, torch.float32)

    def fn(a, b, c):
        return cost.forward(a, b, c, ctx).total

    compiled = torch.compile(fn, mode="reduce-overhead")

    def step():
        compiled(q0, wp, v0).sum().backward()
        wp.grad = None

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"fused cost fwd+bwd, {BATCH}x{HORIZON}, compiled: "
          f"{ms:.2f} ms ({1e3/ms:.1f} steps/s)")
    assert ms < BUDGET_MS * 1.5, f"{ms:.2f} ms exceeds the {BUDGET_MS} ms budget"
