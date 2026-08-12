"""Transformer: shapes, mask exactness, context caching, determinism, grads,
``torch.compile(fullgraph=True)``, sync-freedom and parameter counts.

Run standalone for the latency report: ``python tests/test_model.py``.
"""

from __future__ import annotations

import time

import pytest
import torch

from amortized_opt.constants import CONTROL_DT, N_DOF
from amortized_opt.model import AmortizedOptimizer, ModelConfig, param_counts
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks, pack_state

H = 100


def make_model(device, **kw) -> AmortizedOptimizer:
    cfg = ModelConfig(max_horizon=H, **kw)
    m = AmortizedOptimizer(cfg).to(device)
    m.eval()
    return m


def make_batch(batch: int, device, *, seed: int = 0, hits=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = ContextInputs.zeros(batch, num_drums=8, num_future=3, device=device)
    ctx.drum_pos.copy_(torch.rand(batch, 8, 3, generator=g).to(device))
    ctx.drum_axis[..., 2] = 1.0
    ctx.drum_radius.fill_(0.15)
    ctx.drum_height.fill_(0.1)
    ctx.drum_valid[:, :7] = True
    ctx.goal_valid[:] = True
    # ...except the RIGHT hand of the last row, which is idle: that row carries
    # the idle-arm token instead, so ``encoder.idle_proj`` is on the graph and
    # the "every parameter gets a gradient" check below actually covers it.
    ctx.goal_valid[-1, 1] = False
    ctx.idle_valid[-1, 1] = True
    ctx.idle_home_q[-1, 1].copy_(torch.rand(7, generator=g).to(device))
    ctx.idle_home_tip[-1, 1].copy_(torch.rand(3, generator=g).to(device))
    ctx.goal_pos.copy_(torch.rand(batch, 2, 3, generator=g).to(device))
    ctx.goal_vel.copy_((torch.rand(batch, 2, 3, generator=g) - 0.5).to(device) * 6.0)
    ctx.goal_drum.copy_(torch.randint(0, 7, (batch, 2), generator=g).to(device))
    ctx.future_valid[:, :, :2] = True
    ctx.future_pos.copy_(torch.rand(batch, 2, 3, 3, generator=g).to(device))
    ctx.future_drum.copy_(torch.randint(0, 7, (batch, 2, 3), generator=g).to(device))

    if hits is None:
        t_l = 0.2 + torch.rand(batch, generator=g) * 1.2
        t_r = 0.2 + torch.rand(batch, generator=g) * 1.2
    else:
        t_l = torch.full((batch,), float(hits[0]))
        t_r = torch.full((batch,), float(hits[1]))
    t_l, t_r = t_l.to(device), t_r.to(device)
    ctx.goal_tth.copy_(torch.stack([t_l, t_r], dim=1))
    ctx.future_tth.copy_(torch.stack([t_l, t_r], dim=1).unsqueeze(-1) + 0.5)

    masks = build_waypoint_masks(t_l, t_r, max_horizon=H, dt=CONTROL_DT)
    init = (torch.rand(batch, H, N_DOF, generator=g).to(device) - 0.5) * 0.6
    state = pack_state(
        (torch.rand(batch, N_DOF, generator=g).to(device) - 0.5) * 0.6,
        (torch.rand(batch, N_DOF, generator=g).to(device) - 0.5) * 2.0,
    )
    return ctx, init, state, masks


# ------------------------------------------------------------- 1. shapes --
def test_shapes(cuda_device, capsys):
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(8, cuda_device)
    with torch.no_grad():
        enc = m.encode_context(ctx)
        out = m(init, state, enc, masks)
    assert enc.memory.shape == (8, m.cfg.n_context_tokens, m.cfg.d_ctx)
    assert enc.mask.shape == (8, m.cfg.n_context_tokens)
    assert out.traj.shape == (8, H, 17)
    for t in (out.vel, out.acc, out.torque):
        assert t.shape == (8, H, 17)
    assert out.fk_pos.shape == (8, H, 6, 3) and out.fk_vel.shape == (8, H, 6, 3)
    assert out.elem_mask.shape == (8, H, 17)
    assert out.point_mask.shape == (8, H, 6)
    assert out.token_mask.shape == (8, H)
    assert torch.isfinite(out.traj).all()
    with capsys.disabled():
        print(
            f"\n[shapes] context tokens={m.cfg.n_context_tokens} "
            f"(8 drums + 2 goals + 2x3 future), traj={tuple(out.traj.shape)}, "
            f"aux vel/acc/torque={tuple(out.vel.shape)}, fk={tuple(out.fk_pos.shape)}"
        )


def test_refine_iterates(cuda_device):
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(4, cuda_device)
    with torch.no_grad():
        enc = m.encode_context(ctx)
        one = m.refine(init, state, enc, masks, iterations=1)
        three = m.refine(init, state, enc, masks, iterations=3)
        manual = m(m(m(init, state, enc, masks).traj, state, enc, masks).traj, state, enc, masks)
    assert torch.equal(three.traj, manual.traj)
    assert not torch.equal(one.traj, three.traj)


# --------------------------------------------------------------- 2. masks --
def _fwd(m, init, state, enc, masks):
    with torch.no_grad():
        return m(init, state, enc, masks)


def test_masked_tokens_cannot_influence_anything(cuda_device, capsys):
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(4, cuda_device, hits=(0.4, 0.4))
    enc = m.encode_context(ctx)
    base = _fwd(m, init, state, enc, masks)

    dead = ~masks.token_mask  # tokens beyond both horizons
    pert = init.clone()
    pert[dead] += 5.0
    got = _fwd(m, pert, state, enc, masks)
    live = masks.token_mask.unsqueeze(-1).expand_as(base.traj)
    d = (got.traj - base.traj)[live].abs().max().item()
    aux = max(
        (getattr(got, k) - getattr(base, k)).abs().max().item()
        for k in ("vel", "acc", "torque", "fk_pos", "fk_vel")
    )
    with capsys.disabled():
        print(f"\n[mask a] dead-token perturbation -> live-output max diff {d:.3e}, aux {aux:.3e}")
    assert d == 0.0
    assert aux == 0.0


def test_masked_elements_cannot_influence_anything(cuda_device, capsys):
    """Right-arm elements past the (shorter) right horizon, left horizon longer."""
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(4, cuda_device, hits=(1.0, 0.4))
    enc = m.encode_context(ctx)
    base = _fwd(m, init, state, enc, masks)

    right_dead = (masks.elem_mask[..., 10:17] < 0.5) & masks.token_mask.unsqueeze(-1)
    assert right_dead.any()
    pert = init.clone()
    pert[..., 10:17][right_dead] += 7.0
    got = _fwd(m, pert, state, enc, masks)
    d = (got.traj - base.traj).abs().max().item()
    with capsys.disabled():
        print(f"[mask b] masked-element perturbation -> max output diff {d:.3e}")
    assert d == 0.0


def test_equal_horizons_ignore_everything_past_the_hit(cuda_device, capsys):
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(4, cuda_device, hits=(0.6, 0.6))
    enc = m.encode_context(ctx)
    base = _fwd(m, init, state, enc, masks)
    beyond = ~masks.token_mask
    assert int(masks.token_mask[0].sum().item()) == 30
    pert = init.clone()
    pert[beyond] = torch.randn_like(pert[beyond]) * 3.0
    got = _fwd(m, pert, state, enc, masks)
    live = masks.token_mask.unsqueeze(-1).expand_as(base.traj)
    d = (got.traj - base.traj)[live].abs().max().item()
    with capsys.disabled():
        print(f"[mask c] equal horizons, beyond-horizon noise -> max diff {d:.3e}")
    assert d == 0.0


def test_masked_outputs_are_zero(cuda_device):
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(4, cuda_device, hits=(1.0, 0.4))
    enc = m.encode_context(ctx)
    out = _fwd(m, init, state, enc, masks)
    for t in (out.traj, out.vel, out.acc, out.torque):
        assert torch.equal(t[masks.elem_mask < 0.5], torch.zeros_like(t[masks.elem_mask < 0.5]))
    dead_pts = masks.point_mask < 0.5
    assert out.fk_pos[dead_pts].abs().max().item() == 0.0


# ------------------------------------------------------------- 3. fourier --
def test_temporal_encoding_shifts_with_the_horizon(cuda_device):
    """Same init, horizon shifted by one waypoint -> the answer shifts too."""
    m = make_model(cuda_device)
    ctx, init, state, _ = make_batch(2, cuda_device, hits=(0.6, 0.6))
    enc = m.encode_context(ctx)
    b = init.shape[0]
    dev = init.device
    m0 = build_waypoint_masks(
        torch.full((b,), 0.6, device=dev), torch.full((b,), 0.6, device=dev),
        max_horizon=H, dt=CONTROL_DT,
    )
    m1 = build_waypoint_masks(
        torch.full((b,), 0.6 + CONTROL_DT, device=dev),
        torch.full((b,), 0.6 + CONTROL_DT, device=dev),
        max_horizon=H, dt=CONTROL_DT,
    )
    # the two problems differ only by the horizon: the codes are the shifted ones
    assert torch.allclose(m1.tau_left[:, 1:], m0.tau_left[:, :-1], atol=1e-6)
    o0 = _fwd(m, init, state, enc, m0)
    o1 = _fwd(m, init, state, enc, m1)
    assert not torch.allclose(o0.traj[:, :30], o1.traj[:, :30], atol=1e-4)
    assert int(m1.token_mask[0].sum().item()) == int(m0.token_mask[0].sum().item()) + 1


# ------------------------------------------------------ 4. context caching --
def test_cached_context_is_bit_identical(cuda_device, capsys):
    m = make_model(cuda_device)
    ctx, init, state, masks = make_batch(6, cuda_device)
    with torch.no_grad():
        cached = m.encode_context(ctx)
        worst = 0.0
        cur = init
        for _ in range(5):
            a = m(cur, state, cached, masks)  # reused encoding
            b = m(cur, state, m.encode_context(ctx), masks)  # re-encoded
            worst = max(worst, (a.traj - b.traj).abs().max().item())
            cur = a.traj
    with capsys.disabled():
        print(f"[cache] 5 decoder calls, cached vs re-encoded max diff {worst:.3e}")
    assert worst == 0.0


# ------------------------------------------------- 5. determinism + grads --
def test_determinism_and_finite_grads(cuda_device, capsys):
    m = make_model(cuda_device)
    m.train()
    ctx, init, state, masks = make_batch(4, cuda_device)
    enc = m.encode_context(ctx)
    o1 = m(init, state, enc, masks)
    o2 = m(init, state, enc, masks)
    d = max((a - b).abs().max().item() for a, b in zip(o1.as_tuple(), o2.as_tuple()))
    assert d == 0.0
    loss = sum(t.square().mean() for t in o1.as_tuple())
    loss.backward()
    n_grad = 0
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        n_grad += int(p.grad.abs().sum().item() > 0)
    with capsys.disabled():
        print(
            f"[determinism] repeat-forward max diff {d:.3e}; "
            f"{n_grad}/{len(list(m.parameters()))} params got a nonzero finite grad"
        )
    assert n_grad == len(list(m.parameters()))


# ------------------------------------------------------------- 6. compile --
def _timeit(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _decoder_args(m, batch, device, seed=0):
    ctx, init, state, masks = make_batch(batch, device, seed=seed)
    with torch.no_grad():
        enc = m.encode_context(ctx)
    return (
        init, state, masks.tau_left, masks.tau_right, masks.tau0_left, masks.tau0_right,
        masks.elem_mask, masks.token_mask, masks.point_mask, enc.memory, enc.mask,
    )


def measure_latency(device, batch, mode="reduce-overhead", tf32: bool = False):
    torch.cuda.set_device(torch.device(device).index or 0)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    try:
        m = make_model(device)
        args = _decoder_args(m, batch, device)
        with torch.no_grad():
            eager = _timeit(lambda: m.decoder(*args))
            torch._dynamo.reset()
            comp = torch.compile(m.decoder, fullgraph=True, mode=mode)
            comp(*args)
            compiled = _timeit(lambda: comp(*args))
    finally:
        torch.set_float32_matmul_precision(prev)
    return eager, compiled


def test_compile_fullgraph_and_latency(cuda_device, capsys):
    torch._dynamo.reset()
    try:
        rows = [
            (b, *measure_latency(cuda_device, b), measure_latency(cuda_device, b, tf32=True)[1])
            for b in (1, 256)
        ]
    except Exception as exc:  # pragma: no cover - inductor availability
        pytest.skip(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
    with capsys.disabled():
        print(f"\n[compile] decoder fwd, H={H}, fullgraph=True, mode=reduce-overhead, {cuda_device}")
        for b, eager, comp, tf32 in rows:
            print(f"  batch {b:>4}: eager {eager:7.3f} ms   compiled {comp:7.3f} ms"
                  f" ({eager / comp:.1f}x)   compiled+tf32 {tf32:7.3f} ms")
    assert all(c > 0 for _b, _e, c, _t in rows)


def test_compiled_matches_eager(cuda_device):
    torch._dynamo.reset()
    m = make_model(cuda_device)
    args = _decoder_args(m, 4, cuda_device, seed=3)
    with torch.no_grad():
        ref = m.decoder(*args)
        try:
            comp = torch.compile(m.decoder, fullgraph=True)
            got = comp(*args)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
    d = max((a - b).abs().max().item() for a, b in zip(ref, got))
    assert d < 1e-4, d


# ---------------------------------------------------------- 7. sync-free --
def test_no_device_host_sync(cuda_device):
    m = make_model(cuda_device)
    m.train()
    ctx, init, state, _ = make_batch(4, cuda_device)
    t_l = torch.full((4,), 0.8, device=cuda_device)
    t_r = torch.full((4,), 0.5, device=cuda_device)
    init.requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        masks = build_waypoint_masks(t_l, t_r, max_horizon=H, dt=CONTROL_DT)
        out = m(init, state, m.encode_context(ctx), masks)
        sum(t.square().sum() for t in out.as_tuple()).backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert init.grad is not None


# -------------------------------------------------------- 8. param counts --
def test_param_counts(cuda_device, capsys):
    m = make_model(cuda_device)
    c = param_counts(m)
    with capsys.disabled():
        print(
            f"\n[params] encoder {c['encoder'] / 1e6:.2f} M | decoder body "
            f"{c['decoder'] / 1e6:.2f} M | heads {c['heads'] / 1e6:.3f} M | "
            f"total {c['total'] / 1e6:.2f} M"
        )
    assert c["total"] == sum(p.numel() for p in m.parameters())
    assert 5e6 < c["total"] < 60e6


# ------------------------------------------------------------- extra: modes --
def test_residual_mode_starts_as_the_identity(cuda_device):
    m = make_model(cuda_device, output_mode="residual")
    ctx, init, state, masks = make_batch(4, cuda_device)
    out = _fwd(m, init, state, m.encode_context(ctx), masks)
    assert torch.allclose(out.traj, init * masks.elem_mask, atol=1e-6)


def test_config_is_configurable(cuda_device):
    m = make_model(cuda_device, d_model=256, dec_layers=5, dec_heads=8, ctx_layers=3, d_ctx=256)
    assert m.cfg.dec_head_dim == 32
    ctx, init, state, masks = make_batch(2, cuda_device)
    out = _fwd(m, init, state, m.encode_context(ctx), masks)
    assert out.traj.shape == (2, H, 17)


if __name__ == "__main__":  # pragma: no cover
    from conftest import pick_cuda_device

    dev = pick_cuda_device() or "cpu"
    torch.cuda.set_device(torch.device(dev).index or 0)
    mdl = make_model(dev)
    print(param_counts(mdl))
    for bsz in (1, 8, 64, 256):
        for mode in ("default", "reduce-overhead"):
            torch._dynamo.reset()
            e, c = measure_latency(dev, bsz, mode=mode)
            print(f"batch {bsz:>4} {mode:>16}: eager {e:7.3f} ms  compiled {c:7.3f} ms")
