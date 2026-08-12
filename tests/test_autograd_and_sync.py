"""Autograd health + the CI sync-free guard the plan demands.

A single device->host sync inside these hot loops costs ~15x (see the GPU review
in docs/amortized_optimizer_plan.md) and nothing fails loudly, so
``torch.cuda.set_sync_debug_mode("error")`` is asserted here on purpose.
"""

from __future__ import annotations

import pytest
import torch

from amortized_opt.fk import ChainFK
from amortized_opt.limits import isaac_armature
from amortized_opt.rnea import BatchedRNEA
from amortized_opt.sdf import SdfPenalty

B, K = 256, 50  # one training minibatch at the plan's horizon: 12,800 poses


def _inputs(device, dtype=torch.float32, requires_grad=True):
    g = torch.Generator(device="cpu").manual_seed(0)
    q = (torch.rand(B * K, 17, generator=g, dtype=torch.float64) * 2.0 - 1.0).to(device, dtype)
    qd = (torch.randn(B * K, 17, generator=g, dtype=torch.float64) * 0.5).to(device, dtype)
    qdd = (torch.randn(B * K, 17, generator=g, dtype=torch.float64) * 2.0).to(device, dtype)
    if requires_grad:
        for t in (q, qd, qdd):
            t.requires_grad_(True)
    return q, qd, qdd


def _build(device, dtype=torch.float32):
    fk = ChainFK(device=device, dtype=dtype)
    rnea = BatchedRNEA.from_urdf(device=device, dtype=dtype).to(device, dtype)
    sdf = SdfPenalty(fk=fk, device=device, dtype=dtype)
    return fk, rnea, sdf


def _loss(fk, rnea, sdf, q, qd, qdd, tl, tr):
    frames = fk.forward(q)
    aux = sum(p.square().sum() for _R, p in frames.values())
    pen = sdf.forward(frames=frames, target_drum_left=tl, target_drum_right=tr).sum()
    tau = rnea.rnea(q, qd, qdd).square().sum()
    return aux + pen + tau


# ------------------------------------------------------------------ autograd --
@pytest.mark.parametrize("device", ["cpu"])
def test_backward_finite_cpu(device):
    fk, rnea, sdf = _build(device)
    q, qd, qdd = _inputs(device)
    n = 512  # cpu: keep it quick, the CUDA test covers the full 256x50
    q, qd, qdd = q[:n], qd[:n], qdd[:n]
    for t in (q, qd, qdd):
        t.retain_grad()
    tl = torch.randint(-1, sdf.num_drums, (n,))
    tr = torch.randint(-1, sdf.num_drums, (n,))
    _loss(fk, rnea, sdf, q, qd, qdd, tl, tr).backward()
    for t in (q, qd, qdd):
        assert torch.isfinite(t.grad).all()
    assert float(q.grad.abs().max()) > 0.0


def test_backward_finite_cuda_full_batch(cuda_device, capsys):
    fk, rnea, sdf = _build(cuda_device)
    q, qd, qdd = _inputs(cuda_device)
    tl = torch.randint(-1, sdf.num_drums, (B * K,), device=cuda_device)
    tr = torch.randint(-1, sdf.num_drums, (B * K,), device=cuda_device)
    _loss(fk, rnea, sdf, q, qd, qdd, tl, tr).backward()
    gmax = [float(t.grad.abs().max()) for t in (q, qd, qdd)]
    with capsys.disabled():
        print(f"\n[autograd {B}x{K} on {cuda_device}] |grad|max q/qd/qdd = "
              f"{gmax[0]:.3e} / {gmax[1]:.3e} / {gmax[2]:.3e}")
    for t in (q, qd, qdd):
        assert torch.isfinite(t.grad).all()
    assert all(g > 0.0 for g in gmax)


# ----------------------------------------------------------------- sync-free --
@pytest.mark.parametrize("what", ["fk", "rnea", "sdf", "all"])
def test_no_device_host_sync(cuda_device, what):
    """fwd+bwd of each module must perform zero device->host syncs."""
    fk, rnea, sdf = _build(cuda_device)
    q, qd, qdd = _inputs(cuda_device)
    tl = torch.randint(-1, sdf.num_drums, (B * K,), device=cuda_device)
    tr = torch.randint(-1, sdf.num_drums, (B * K,), device=cuda_device)
    # warm up outside the guard: lazy module init / autotune may legitimately sync
    _loss(fk, rnea, sdf, q, qd, qdd, tl, tr).backward()
    for t in (q, qd, qdd):
        t.grad = None
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        if what == "fk":
            loss = sum(p.square().sum() for _R, p in fk.forward(q).values())
        elif what == "rnea":
            loss = rnea.rnea(q, qd, qdd).square().sum()
        elif what == "sdf":
            loss = sdf.forward(q, target_drum_left=tl, target_drum_right=tr).sum()
        else:
            loss = _loss(fk, rnea, sdf, q, qd, qdd, tl, tr)
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.isfinite(q.grad).all()
