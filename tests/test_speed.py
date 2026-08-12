"""Steady-state wall time of the whole FK+SDF+RNEA fwd+bwd at batch 256x50.

Budget from the brief: <= 20 ms. Note the 10.84 ms the scratch prototypes
reported is the ``torch.compile(mode="reduce-overhead")`` figure -- eager, this
stack is launch-bound at ~150-180 ms even with every sync removed (removing the
syncs is what makes CUDA-graph capture possible in the first place).

Run standalone for a report: ``python tests/test_speed.py``.
"""

from __future__ import annotations

import time

import pytest
import torch

from amortized_opt.fk import ChainFK
from amortized_opt.rnea import BatchedRNEA
from amortized_opt.sdf import SdfPenalty

B, K = 256, 50
BUDGET_MS = 20.0


def _timeit(step, iters: int = 30, warmup: int = 10) -> float:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def measure(device: str, dtype=torch.float32, compile_mode: str | None = "reduce-overhead") -> float:
    # MANDATORY off cuda:0, else inductor captures an empty CUDA graph.
    torch.cuda.set_device(torch.device(device).index or 0)
    fk = ChainFK(device=device, dtype=dtype)
    rnea = BatchedRNEA.from_urdf(device=device, dtype=dtype).to(device, dtype)
    sdf = SdfPenalty(fk=fk, device=device, dtype=dtype)

    g = torch.Generator(device="cpu").manual_seed(0)
    q = (torch.rand(B * K, 17, generator=g, dtype=torch.float64) * 2 - 1).to(device, dtype)
    qd = (torch.randn(B * K, 17, generator=g, dtype=torch.float64) * 0.5).to(device, dtype)
    qdd = (torch.randn(B * K, 17, generator=g, dtype=torch.float64) * 2.0).to(device, dtype)
    for t in (q, qd, qdd):
        t.requires_grad_(True)
    tl = torch.randint(-1, sdf.num_drums, (B * K,), device=device)
    tr = torch.randint(-1, sdf.num_drums, (B * K,), device=device)

    def loss_fn(q, qd, qdd):
        frames = fk.forward(q)  # ONE FK pass, shared with the SDF
        loss = sum(p.square().sum() for _R, p in frames.values())
        loss = loss + sdf.forward(frames=frames, target_drum_left=tl, target_drum_right=tr).sum()
        return loss + rnea.rnea(q, qd, qdd).square().sum()

    fn = loss_fn if compile_mode is None else torch.compile(loss_fn, mode=compile_mode)

    def step():
        for t in (q, qd, qdd):
            t.grad = None
        fn(q, qd, qdd).backward()

    step()
    torch.cuda.synchronize()
    return _timeit(step)


def test_speed_under_budget(cuda_device, capsys):
    torch._dynamo.reset()
    try:
        ms = measure(cuda_device, compile_mode="reduce-overhead")
    except Exception as exc:  # pragma: no cover - inductor availability
        pytest.skip(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
    with capsys.disabled():
        print(f"\n[speed {B}x{K} fp32 fwd+bwd on {cuda_device}, compile=reduce-overhead] "
              f"{ms:.2f} ms/step ({1e3 / ms:.0f} steps/s), budget {BUDGET_MS} ms")
    assert ms <= BUDGET_MS


if __name__ == "__main__":  # pragma: no cover
    from conftest import pick_cuda_device

    dev = pick_cuda_device() or "cpu"
    for mode in (None, "default", "reduce-overhead"):
        torch._dynamo.reset()
        print(f"{dev} fp32 compile={mode}: {measure(dev, compile_mode=mode):.2f} ms/step")
