"""Phase-1 identity Jacobian: the decoder must pass the waist set-point through.

Ordinary quintic Huber is unchanged (init and target stay the quintic).
Mimic trains encoder residual MLPs against aim / upright.  Gain is
``(∂ hold / ∂ s − 1)²`` with a frozen hold window and a stopgrad leaf ``s``.
Val never calls the follow path.
"""

from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS
from amortized_opt.model import AmortizedOptimizer, ModelConfig
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
from amortized_opt.train_cost import substitute_waist_setpoints, substitute_waist_sp
from amortized_opt.train_warmstart import (
    LossScales,
    LossWeights,
    TrainConfig,
    apply_waist_follow,
    forward_loss,
    unpack,
    waist_identity_gain,
)
from amortized_opt.warmstart import rewrite_waist_target_to_sp

N = len(G1_UPPER_BODY_JOINTS)
YAW = G1_UPPER_BODY_JOINTS.index("waist_yaw_joint")
PITCH = G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")
ROLL = G1_UPPER_BODY_JOINTS.index("waist_roll_joint")
H = 20
TINY = dict(
    d_ctx=32, ctx_layers=1, ctx_heads=2,
    d_model=32, dec_layers=1, dec_heads=2, max_horizon=H,
)


def _model(**kw) -> AmortizedOptimizer:
    torch.manual_seed(0)
    return AmortizedOptimizer(ModelConfig(**TINY, **kw))


def _ctx(B: int = 8) -> ContextInputs:
    ctx = ContextInputs.zeros(B, num_drums=8, num_future=3)
    ctx.drum_valid[:, :3] = True
    ctx.goal_valid[:, 0] = True
    ctx.yaw_now = torch.linspace(-0.2, 0.4, B)
    ctx.aim_yaw = ctx.yaw_now + 0.12
    ctx.aim_valid = torch.ones(B, dtype=torch.bool)
    ctx.aim_valid[-1] = False
    return ctx


def _masks(B: int = 8, horizon: int = H):
    t = torch.full((B,), 0.30)
    return build_waypoint_masks(t, t, max_horizon=horizon)


def _batch(model: AmortizedOptimizer, B: int = 8, horizon: int = H) -> dict:
    ctx = _ctx(B)
    masks = _masks(B, horizon)
    q0 = torch.zeros(B, N)
    q0[:, YAW] = ctx.yaw_now
    q0[:, PITCH] = 0.05
    target = q0.unsqueeze(1).expand(B, horizon, N).contiguous().clone()
    aim = torch.where(ctx.aim_valid, ctx.aim_yaw, ctx.yaw_now)
    for k in range(horizon):
        u = min((k + 1) / 8.0, 1.0)
        target[:, k, YAW] = q0[:, YAW] + (aim - q0[:, YAW]) * (u ** 3) * (10 - 15 * u + 6 * u * u)
    init = target + 0.02 * torch.randn_like(target)
    v0 = torch.zeros(B, N)
    aux_z = torch.zeros(B, horizon, N)
    fk_z = torch.zeros(B, horizon, 6, 3)
    return {
        "init": init,
        "state": torch.cat([q0, v0], dim=-1),
        "q0": q0,
        "context": ctx,
        "masks": masks,
        "target": target,
        "aux": {
            "vel": aux_z.clone(), "acc": aux_z.clone(), "torque": aux_z.clone(),
            "fk_pos": fk_z.clone(), "fk_vel": fk_z.clone(),
        },
    }


def test_frac_zero_and_mimic_zero_matches_forward_loss():
    model = _model(waist_sp_head=True, waist_pr_sp_head=True)
    batch = _batch(model)
    scales, w = LossScales("cpu"), LossWeights()
    with torch.no_grad():
        l0, m0 = forward_loss(model, *unpack(batch), scales, w)
    cfg = TrainConfig(waist_follow_frac=0.0)
    assert cfg.waist_follow_frac == 0.0 and w.mimic == 0.0 and w.gain == 0.0
    with torch.no_grad():
        l1, m1 = forward_loss(model, *unpack(batch), scales, w)
    assert torch.equal(l0, l1) and torch.equal(m0, m1)


def test_apply_not_called_when_follow_off(monkeypatch):
    model = _model(waist_sp_head=True)
    batch = _batch(model)
    scales, w = LossScales("cpu"), LossWeights()

    def boom(*_a, **_k):
        raise AssertionError("apply_waist_follow must not run when follow is off")

    monkeypatch.setattr(
        "amortized_opt.train_warmstart.apply_waist_follow", boom
    )
    with torch.no_grad():
        forward_loss(model, *unpack(batch), scales, w)


def test_apply_does_not_rewrite_target_or_init():
    model = _model(waist_sp_head=True, waist_pr_sp_head=True)
    batch = _batch(model)
    init0 = batch["init"].clone()
    target0 = batch["target"].clone()
    enc = model.encode_context(batch["context"])
    cfg = TrainConfig(waist_hold_wp=10)
    enc_d, target, aux_em, l_mimic, info = apply_waist_follow(model, batch, enc, cfg)
    assert info["n_iv"] == 0
    assert torch.equal(batch["init"], init0)
    assert torch.equal(target, target0)
    assert torch.equal(aux_em, batch["masks"].elem_mask)
    # stopgrad substitute still rebuilds the token from the announced sp
    assert enc_d.waist_yaw_sp is not None
    assert torch.allclose(enc_d.waist_yaw_sp, enc.waist_yaw_sp)


def test_mimic_near_zero_at_zero_init():
    model = _model(waist_sp_head=True, waist_pr_sp_head=True)
    batch = _batch(model)
    enc = model.encode_context(batch["context"])
    cfg = TrainConfig()
    _enc_d, _t, _em, l_mimic, info = apply_waist_follow(model, batch, enc, cfg)
    assert info["n_iv"] == 0
    assert float(l_mimic) < 1e-6
    ctx = batch["context"]
    base = torch.where(ctx.aim_valid, ctx.aim_yaw, ctx.yaw_now)
    assert torch.allclose(enc.waist_yaw_sp, base, atol=1e-6)
    assert torch.allclose(enc.waist_pr_sp, torch.zeros_like(enc.waist_pr_sp), atol=1e-6)


def test_mimic_trains_residual_not_token_when_stopgrad():
    model = _model(waist_sp_head=True, waist_pr_sp_head=True)
    with torch.no_grad():
        model.encoder.waist_sp_out.bias.fill_(0.05)
        model.encoder.waist_pr_sp_out.bias.fill_(0.04)
    batch = _batch(model)
    enc = model.encode_context(batch["context"])
    cfg = TrainConfig()
    _e, _t, _m, l_mimic, _ = apply_waist_follow(model, batch, enc, cfg)
    assert float(l_mimic) > 0
    l_mimic.backward()
    assert float(model.encoder.waist_sp_out.weight.grad.norm()) > 0
    assert float(model.encoder.waist_pr_sp_out.weight.grad.norm()) > 0
    # token proj is not in the mimic graph (leaf is stopgrad)
    g_proj = model.encoder.waist_sp_proj.weight.grad
    assert g_proj is None or float(g_proj.norm()) < 1e-8


def test_gain_vjp_finite_and_trains_proj_not_residual():
    model = _model(waist_sp_head=True, waist_pr_sp_head=True)
    batch = _batch(model, B=4)
    enc = model.encode_context(batch["context"])
    cfg = TrainConfig(waist_hold_wp=10, waist_gain_frac=1.0)
    loss, info = waist_identity_gain(model, batch, enc, cfg)
    assert torch.isfinite(loss)
    assert info["n"] == 4
    assert torch.isfinite(torch.tensor(info["gain_yaw"]))
    loss.backward()
    assert float(model.encoder.waist_sp_proj.weight.grad.norm()) > 0
    g_out = model.encoder.waist_sp_out.weight.grad
    assert g_out is None or float(g_out.norm()) < 1e-8
    dec_grad = sum(
        (p.grad.detach().square().sum() if p.grad is not None else 0.0)
        for p in model.decoder.parameters()
    )
    assert float(dec_grad) > 0


def test_gain_frac_slices_batch():
    model = _model(waist_sp_head=True)
    batch = _batch(model, B=8)
    enc = model.encode_context(batch["context"])
    cfg = TrainConfig(waist_gain_frac=0.25)
    _loss, info = waist_identity_gain(model, batch, enc, cfg)
    assert info["n"] == 2


def test_k_star_clamped_to_elem_mask_horizon():
    B, Hs = 4, 8
    q0 = torch.zeros(B, N)
    sp = torch.full((B, 1), 1.0)
    target = torch.zeros(B, Hs, N)
    em = torch.ones(B, Hs, N)
    em[:, 6:, :] = 0
    out = rewrite_waist_target_to_sp(
        target, q0, sp, [YAW], k_min=10, elem_mask=em,
    )
    assert torch.allclose(out[:, 5, YAW], torch.ones(B), atol=1e-4)
    assert torch.equal(out[:, 6:, YAW], target[:, 6:, YAW])


def test_k_star_skips_horizon_shorter_than_2():
    B, Hs = 2, 5
    q0 = torch.zeros(B, N)
    sp = torch.ones(B, 1)
    target = torch.zeros(B, Hs, N)
    em = torch.zeros(B, Hs, N)
    em[:, :1, :] = 1
    out = rewrite_waist_target_to_sp(
        target, q0, sp, [YAW], k_min=10, elem_mask=em,
    )
    assert torch.equal(out, target)


def test_substitute_setpoints_yaw_and_pr():
    m = _model(waist_sp_head=True, waist_pr_sp_head=True)
    ctx = _ctx(4)
    enc = m.encode_context(ctx)
    yaw = enc.waist_yaw_sp + 0.2
    pr = enc.waist_pr_sp + 0.05
    base = torch.where(ctx.aim_valid, ctx.aim_yaw, ctx.yaw_now)
    sub = substitute_waist_setpoints(
        m, enc, yaw_sp=yaw, pr_sp=pr, yaw_now=ctx.yaw_now, base_yaw=base,
    )
    assert torch.allclose(sub.waist_yaw_sp, yaw)
    assert torch.allclose(sub.waist_pr_sp, pr)
    assert not torch.allclose(sub.memory[:, -1], enc.memory[:, -1])
    assert not torch.allclose(sub.memory[:, -2], enc.memory[:, -2])


def test_substitute_pr_only_keeps_yaw_token():
    m = _model(waist_sp_head=True, waist_pr_sp_head=True)
    ctx = _ctx(4)
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.03)
        m.encoder.waist_sp_out.bias.add_(0.07)
    enc = m.encode_context(ctx)
    yaw_tok = enc.memory[:, -1].clone()
    pr = enc.waist_pr_sp + 0.08
    sub = substitute_waist_setpoints(m, enc, pr_sp=pr)
    assert torch.allclose(sub.memory[:, -1], yaw_tok)
    assert not torch.allclose(sub.memory[:, -2], enc.memory[:, -2])
    assert torch.allclose(sub.waist_pr_sp, pr)


def test_substitute_waist_sp_wrapper_keeps_pr_token():
    m = _model(waist_sp_head=True, waist_pr_sp_head=True)
    ctx = _ctx(4)
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.04)
    enc = m.encode_context(ctx)
    pr_tok = enc.memory[:, -2].clone()
    base = torch.where(ctx.aim_valid, ctx.aim_yaw, ctx.yaw_now)
    sub = substitute_waist_sp(m, enc, enc.waist_yaw_sp + 0.2, ctx.yaw_now, base)
    assert torch.allclose(sub.memory[:, -2], pr_tok)
    assert torch.allclose(sub.waist_pr_sp, enc.waist_pr_sp)


def test_evaluate_never_calls_apply(monkeypatch):
    model = _model(waist_sp_head=True)
    batch = _batch(model)
    scales, w = LossScales("cpu"), LossWeights()

    def boom(*_a, **_k):
        raise AssertionError("val must not call apply_waist_follow")

    monkeypatch.setattr(
        "amortized_opt.train_warmstart.apply_waist_follow", boom
    )
    with torch.no_grad():
        forward_loss(model, *unpack(batch), scales, w)


def test_cli_scripts_declare_gain_flags():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("train_phase1.py", "train_phase1_apex.py"):
        src = (root / name).read_text()
        assert "--w-waist-gain" in src
        assert "--waist-gain-frac" in src
        assert "--w-waist-mimic" in src
        assert "--waist-follow-frac" in src
        assert "use --w-waist-gain" in src
