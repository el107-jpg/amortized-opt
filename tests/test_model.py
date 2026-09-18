"""Transformer: shapes, mask exactness, context caching, determinism, grads,
``torch.compile(fullgraph=True)``, sync-freedom and parameter counts.

Run standalone for the latency report: ``python tests/test_model.py``.
"""

from __future__ import annotations

import time

import pytest
import torch

from amortized_opt.constants import CONTROL_DT, N_DOF
from amortized_opt.model import (
    AmortizedOptimizer, ModelConfig, hold_fill_waypoints, param_counts,
)
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
    # TYPE_IDLE payload is the torso-frame safety cylinder (8-D), not a joint home.
    ctx.idle_cyl[-1, 1].copy_(torch.rand(8, generator=g).to(device))
    ctx.idle_cyl[-1, 1, 3:6].copy_(
        torch.nn.functional.normalize(ctx.idle_cyl[-1, 1, 3:6], dim=0)
    )
    ctx.idle_cyl[-1, 1, 6:].clamp_(min=0.05)
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
    # The future tokens are HIT-RELATIVE (``ContextInputs.future_gap``): 0.5 s
    # of gap per slot AFTER the current goal's strike, not 0.5 s after "now".
    ctx.future_gap.copy_(
        0.5 * torch.arange(
            1, ctx.num_future + 1, device=device, dtype=ctx.future_gap.dtype
        ).view(1, 1, -1).expand_as(ctx.future_gap)
    )

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
        masks.hit_tth, masks.elem_mask, masks.token_mask, masks.point_mask,
        enc.memory, enc.mask,
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


@pytest.mark.parametrize("dec_rope", [False, True])
def test_compiled_matches_eager(cuda_device, dec_rope):
    torch._dynamo.reset()
    m = make_model(cuda_device, dec_rope=dec_rope)
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


def test_compiled_fp32_traj_head_matches_eager(cuda_device):
    torch._dynamo.reset()
    m = make_model(cuda_device, traj_head_fp32=True)
    args = _decoder_args(m, 4, cuda_device, seed=31)
    with torch.no_grad():
        ref = m.decoder(*args)
        try:
            got = torch.compile(m.decoder, fullgraph=True)(*args)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
    d = max((a - b).abs().max().item() for a, b in zip(ref, got))
    assert d < 2e-4, d


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


def test_traj_head_fp32_forward_backward_and_fp32_parity(cuda_device):
    """Only the trajectory readout escapes AMP; its gradients stay connected."""
    small = dict(
        d_model=32, dec_heads=2, dec_layers=1,
        d_ctx=32, ctx_heads=2, ctx_layers=1,
    )
    base = make_model(cuda_device, traj_head_fp32=False, **small)
    fp32_head = make_model(cuda_device, traj_head_fp32=True, **small)
    fp32_head.load_state_dict(base.state_dict())
    ctx, init, state, masks = make_batch(2, cuda_device, seed=29)

    # With autocast disabled the option is numerically identical.
    enc0 = base.encode_context(ctx)
    enc1 = fp32_head.encode_context(ctx)
    ref = base(init, state, enc0, masks).traj
    got = fp32_head(init, state, enc1, masks).traj
    assert torch.equal(got, ref)

    captured = {}
    def capture(name):
        def hook(_module, _args, output):
            captured[name] = output
        return hook

    hooks = [
        fp32_head.decoder.blocks[0].self_attn.q_proj.register_forward_hook(
            capture("amp_linear")
        ),
        fp32_head.decoder.norm.register_forward_hook(capture("h")),
    ]
    device_type = "cuda" if torch.device(cuda_device).type == "cuda" else "cpu"
    try:
        with torch.autocast(device_type, dtype=torch.bfloat16):
            out = fp32_head(
                init, state, fp32_head.encode_context(ctx), masks
            )
            loss = out.traj.square().mean()
    finally:
        for hook in hooks:
            hook.remove()
    assert captured["amp_linear"].dtype == torch.bfloat16
    assert out.traj.dtype == torch.float32
    expected = (
        fp32_head.decoder.q_center.float()
        + fp32_head.decoder.q_half.float()
        * torch.nn.functional.linear(
            captured["h"][:, 1:].float(),
            fp32_head.decoder.head_traj.weight.float(),
            fp32_head.decoder.head_traj.bias.float(),
        )
    ) * masks.elem_mask.float()
    assert torch.equal(out.traj, expected)
    loss.backward()
    grad = fp32_head.decoder.head_traj.weight.grad
    assert grad is not None and grad.dtype == torch.float32
    assert torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)


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


# =============================== rebuilding a model from a checkpoint ========
# ``ModelConfig(**ck["model_cfg"])`` fills a field the stored config lacks from
# the CURRENT default, so a feature that starts defaulting ON reconstructs an
# architecture the checkpoint never had and the strict load dies on a shape
# mismatch.  On 2026-08-12 that made every checkpoint in the tree unloadable by
# 21 of the 23 entry points at once, so these pin the recovery.
def _fake_ckpt(*, idle_token: bool, drop_cfg_key: bool, **cfg_kw):
    """A checkpoint dict as ``save_checkpoint`` writes one."""
    cfg = ModelConfig(max_horizon=25, idle_home_token=idle_token, **cfg_kw)
    sd = AmortizedOptimizer(cfg).state_dict()
    stored = dict(vars(cfg))
    if drop_cfg_key:
        stored.pop("idle_home_token")       # written before the field existed
    return {"model_cfg": stored, "model": sd}, cfg


@pytest.mark.parametrize("idle_token", [False, True])
def test_arch_flag_is_recovered_from_the_weights(idle_token):
    """A config that predates the field is rebuilt from what the weights have."""
    from amortized_opt.model import model_config_from_checkpoint

    ck, cfg = _fake_ckpt(idle_token=idle_token, drop_cfg_key=True)
    assert "idle_home_token" not in ck["model_cfg"]
    got = model_config_from_checkpoint(ck)
    assert got.idle_home_token is idle_token
    assert got.n_token_types == cfg.n_token_types


@pytest.mark.parametrize("idle_token", [False, True])
def test_recovered_config_loads_strictly(idle_token):
    """THE regression: build from the recovered config and load with strict=True."""
    from amortized_opt.model import model_config_from_checkpoint

    ck, _cfg = _fake_ckpt(idle_token=idle_token, drop_cfg_key=True)
    model = AmortizedOptimizer(model_config_from_checkpoint(ck))
    model.load_state_dict(ck["model"])      # strict -- raises if the shape moved
    assert (model.cfg.idle_home_token is idle_token)


def test_a_stored_flag_is_believed():
    """When the config DOES carry the field, it is used as written."""
    from amortized_opt.model import model_config_from_checkpoint

    for tok in (False, True):
        ck, _ = _fake_ckpt(idle_token=tok, drop_cfg_key=False)
        assert model_config_from_checkpoint(ck).idle_home_token is tok


def test_overrides_win_over_both():
    """``train_phase2.py`` switches the token ON over a warm start without it."""
    from amortized_opt.model import model_config_from_checkpoint

    ck, _ = _fake_ckpt(idle_token=False, drop_cfg_key=True)
    got = model_config_from_checkpoint(ck, idle_home_token=True, dec_layers=2)
    assert got.idle_home_token is True and got.dec_layers == 2


def test_a_field_removed_from_ModelConfig_is_dropped_not_raised(capsys):
    """A stale key must not turn into a TypeError in every entry point."""
    from amortized_opt.model import model_config_from_checkpoint

    ck, _ = _fake_ckpt(idle_token=True, drop_cfg_key=False)
    ck["model_cfg"]["a_field_that_no_longer_exists"] = 7
    got = model_config_from_checkpoint(ck)
    assert got.idle_home_token is True
    assert "no longer in ModelConfig" in capsys.readouterr().out


def test_other_shape_fields_still_round_trip():
    """The recovery must not disturb the fields that were always stored."""
    from amortized_opt.model import model_config_from_checkpoint

    ck, cfg = _fake_ckpt(idle_token=True, drop_cfg_key=True,
                         dec_layers=2, d_model=256, d_ctx=128, num_future=2)
    got = model_config_from_checkpoint(ck)
    for f in ("dec_layers", "d_model", "d_ctx", "num_future", "max_horizon"):
        assert getattr(got, f) == getattr(cfg, f), f
    AmortizedOptimizer(got).load_state_dict(ck["model"])


def test_traj_head_fp32_checkpoint_config_migration():
    """Missing old key means OFF; an opt-in checkpoint round-trips as ON."""
    from amortized_opt.model import model_config_from_checkpoint

    ck, _ = _fake_ckpt(idle_token=True, drop_cfg_key=False)
    ck["model_cfg"].pop("traj_head_fp32")
    assert model_config_from_checkpoint(ck).traj_head_fp32 is False

    ck_on, _ = _fake_ckpt(
        idle_token=True, drop_cfg_key=False, traj_head_fp32=True
    )
    restored = model_config_from_checkpoint(ck_on)
    assert restored.traj_head_fp32 is True
    AmortizedOptimizer(restored).load_state_dict(ck_on["model"])


# --------------------------------------------------------------- 8. RoPE --
def _rope_cis(dh: int, seq: int, base: float = 10000.0, shift: int = 0):
    from amortized_opt.model import rope_inv_freq
    inv = rope_inv_freq(dh, base)
    t = torch.arange(shift, shift + seq, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().view(1, 1, seq, -1), emb.sin().view(1, 1, seq, -1)


def test_rope_scores_are_shift_invariant():
    """Relative position only: sliding every index by 1 leaves QK scores identical."""
    from amortized_opt.model import apply_rope
    torch.manual_seed(0)
    B, h, S, d = 2, 2, 8, 8
    q = torch.randn(B, h, S, d)
    k = torch.randn(B, h, S, d)
    q0, k0 = apply_rope(q, k, *_rope_cis(d, S, shift=0))
    q1, k1 = apply_rope(q, k, *_rope_cis(d, S, shift=1))
    s0 = torch.matmul(q0, k0.transpose(-2, -1))
    s1 = torch.matmul(q1, k1.transpose(-2, -1))
    assert torch.allclose(s0, s1, atol=1e-5, rtol=1e-5)


def test_rope_distinguishes_later_from_earlier():
    """Signed relative: +1 knot is not the same rotation as -1 knot."""
    from amortized_opt.model import apply_rope
    torch.manual_seed(1)
    B, h, S, d = 1, 1, 7, 8
    q = torch.randn(B, h, S, d)
    k = torch.randn(B, h, S, d)
    qr, kr = apply_rope(q, k, *_rope_cis(d, S))
    scores = torch.matmul(qr, kr.transpose(-2, -1))[0, 0]
    i = 3
    assert not torch.allclose(scores[i, i + 1], scores[i, i - 1], atol=1e-5)


def test_dec_rope_off_is_the_default_and_matches_unrotated():
    """Flag off: same weights, same traj as a model that never heard of RoPE."""
    torch.manual_seed(0)
    off = make_model("cpu", d_model=64, dec_heads=4, dec_layers=1, d_ctx=32, ctx_heads=2, ctx_layers=1)
    assert off.cfg.dec_rope is False
    ctx, init, state, masks = make_batch(2, "cpu", seed=4)
    with torch.no_grad():
        enc = off.encode_context(ctx)
        a = off(init, state, enc, masks).traj
    torch.manual_seed(0)
    also = make_model("cpu", d_model=64, dec_heads=4, dec_layers=1, d_ctx=32, ctx_heads=2, ctx_layers=1,
                      dec_rope=False)
    with torch.no_grad():
        b = also(init, state, also.encode_context(ctx), masks).traj
    assert torch.equal(a, b)


def test_dec_rope_changes_the_plan_at_the_same_weights():
    """ON is a behaviour change, not a new parameter: strict load, different traj."""
    torch.manual_seed(0)
    off = make_model("cpu", d_model=64, dec_heads=4, dec_layers=1, d_ctx=32, ctx_heads=2, ctx_layers=1)
    on = make_model("cpu", d_model=64, dec_heads=4, dec_layers=1, d_ctx=32, ctx_heads=2, ctx_layers=1,
                    dec_rope=True)
    assert param_counts(off) == param_counts(on)
    on.load_state_dict(off.state_dict())
    ctx, init, state, masks = make_batch(2, "cpu", seed=5)
    with torch.no_grad():
        a = off(init, state, off.encode_context(ctx), masks).traj
        b = on(init, state, on.encode_context(ctx), masks).traj
    assert torch.isfinite(b).all()
    assert not torch.allclose(a, b, atol=1e-6)


def test_dec_rope_is_self_attn_only():
    m = make_model("cpu", d_model=64, dec_heads=4, dec_layers=2, d_ctx=32, ctx_heads=2, ctx_layers=2,
                   dec_rope=True)
    for blk in m.decoder.blocks:
        assert blk.self_attn.rope is True
        assert blk.cross_attn.rope is False
        assert "rope_inv_freq" not in blk.self_attn.state_dict()
    for blk in m.encoder.blocks:
        assert blk.attn.rope is False


def test_dec_rope_legacy_checkpoint_stays_off():
    """A config written before the field reconstructs OFF and loads strictly."""
    from amortized_opt.model import model_config_from_checkpoint
    ck, _ = _fake_ckpt(idle_token=True, drop_cfg_key=False, dec_rope=False)
    ck["model_cfg"].pop("dec_rope", None)
    ck["model_cfg"].pop("rope_base", None)
    got = model_config_from_checkpoint(ck)
    assert got.dec_rope is False
    AmortizedOptimizer(got).load_state_dict(ck["model"])


def test_dec_rope_stored_flag_is_believed():
    from amortized_opt.model import model_config_from_checkpoint
    ck, _ = _fake_ckpt(idle_token=True, drop_cfg_key=False, dec_rope=True)
    got = model_config_from_checkpoint(ck)
    assert got.dec_rope is True
    AmortizedOptimizer(got).load_state_dict(ck["model"])


# ----------------------------------------------------- plant features --
_SMALL = dict(d_model=64, dec_heads=4, dec_layers=1, d_ctx=32, ctx_heads=2, ctx_layers=1)


def test_hold_fill_keeps_last_valid_knot():
    q0 = torch.zeros(1, N_DOF)
    wp = torch.zeros(1, 8, N_DOF)
    wp[0, :3, 4] = torch.tensor([0.1, 0.2, 0.3])
    mask = torch.zeros(1, 8, N_DOF)
    mask[:, :3] = 1.0
    filled = hold_fill_waypoints(q0, wp, mask)
    assert filled.shape == (1, 9, N_DOF)
    assert float(filled[0, 3, 4]) == pytest.approx(0.3)
    assert torch.equal(filled[0, 4:, 4], filled[0, 3, 4].expand(5))


def test_dec_plant_feats_off_is_the_default():
    m = make_model("cpu", **_SMALL)
    assert m.cfg.dec_plant_feats is False
    assert not hasattr(m.decoder, "wp_plant_proj")


def test_dec_plant_feats_zero_init_is_a_noop():
    """Untrained plant projections add 0, so the plan matches a flag-off twin."""
    torch.manual_seed(0)
    off = make_model("cpu", **_SMALL)
    torch.manual_seed(0)
    on = make_model("cpu", **_SMALL, dec_plant_feats=True)
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert any("wp_plant_proj" in k for k in missing)
    assert not unexpected
    assert torch.count_nonzero(on.decoder.wp_plant_proj.weight) == 0
    ctx, init, state, masks = make_batch(2, "cpu", seed=11)
    with torch.no_grad():
        a = off(init, state, off.encode_context(ctx), masks).traj
        b = on(init, state, on.encode_context(ctx), masks).traj
    assert torch.allclose(a, b, atol=1e-5)
    on.decoder.wp_plant_proj.weight.data.normal_(0.0, 0.05)
    with torch.no_grad():
        c = on(init, state, on.encode_context(ctx), masks).traj
    assert not torch.allclose(a, c, atol=1e-5)


def test_dec_plant_feats_held_pose_is_constant_in_cartesian():
    """When every joint is held, world-frame tip/wrist/elbow stay put.

    A one-handed hit does NOT freeze Cartesian: the waist rides the other
    hand's horizon, so a parked arm still translates with the torso.  That
    is the point of hold-fill (keep the parked *joints*, not a world pose).
    """
    m = make_model("cpu", **_SMALL, dec_plant_feats=True)
    ctx, init, state, masks = make_batch(1, "cpu", seed=2, hits=(1.6, 1.6))
    em = masks.elem_mask.clone()
    em[:, 10:] = 0.0
    plant_wp, plant_st = m.decoder._plant_tokens(init, state, em)
    assert plant_wp.shape[-1] == m.cfg.plant_feat_dim
    assert plant_st.shape == (1, m.cfg.plant_feat_dim)
    assert not plant_wp.requires_grad
    assert torch.isfinite(plant_wp).all()
    cart = plant_wp[0, :, :18]  # 6 points x 3
    assert torch.allclose(cart[10:], cart[9], atol=1e-5)


def test_dec_plant_feats_legacy_checkpoint_stays_off():
    from amortized_opt.model import model_config_from_checkpoint
    ck, _ = _fake_ckpt(idle_token=True, drop_cfg_key=False)
    ck["model_cfg"].pop("dec_plant_feats", None)
    got = model_config_from_checkpoint(ck)
    assert got.dec_plant_feats is False
    AmortizedOptimizer(got).load_state_dict(ck["model"])


def test_dec_plant_feats_recovered_from_weights():
    from amortized_opt.model import model_config_from_checkpoint
    ck, _ = _fake_ckpt(idle_token=True, drop_cfg_key=False, dec_plant_feats=True)
    ck["model_cfg"].pop("dec_plant_feats")
    got = model_config_from_checkpoint(ck)
    assert got.dec_plant_feats is True
    AmortizedOptimizer(got).load_state_dict(ck["model"])


def test_dec_plant_feats_compiled_matches_eager(cuda_device):
    torch._dynamo.reset()
    m = make_model(cuda_device, **_SMALL, dec_plant_feats=True)
    args = _decoder_args(m, 2, cuda_device, seed=3)
    with torch.no_grad():
        ref = m.decoder(*args)
        try:
            comp = torch.compile(m.decoder, fullgraph=False)
            got = comp(*args)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile unavailable: {type(exc).__name__}: {exc}")
    d = max((a - b).abs().max().item() for a, b in zip(ref, got))
    assert d < 1e-4, d
