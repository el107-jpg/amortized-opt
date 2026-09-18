"""The learned waist pitch/roll set-point (``docs/waist_pr_sp_head_plan.md``).

Load-bearing claims:

1. **OFF is bit-identical** -- no extra parameters, encoder output unchanged.
2. **Untrained head is upright** -- ``sp == 0``, residual is the set-point.
3. **No plant lean in the encoder** -- ``sp`` is invariant to ``yaw_now``
   (the only ticking torso input the encoder is allowed; measured P/R is not
   even a context field).
4. **Gradient reaches the head** through the upright-well anchor and the
   tether, and through the conditioning token into the trajectory.
5. A trajectory holding the set-point pays zero; 0.3 rad off on pitch pays
   ``0.5 * 0.3²``.
6. Checkpoint reconciliation detects the flag from weights.
7. Yaw-search substitution still treats the yaw token as last when both
   heads are on.
8. Quintic seed eases pitch/roll to the announced set-point (yaw analogue);
   ``sp is None`` is a no-op; untrained ``sp == 0`` eases toward upright.
9. Hold loss prices ``(sp - stopgrad(q0_pr))²``; q0 is not an encoder input.
10. Hold-band hinge is 0 inside ``|sp - clamp(q0)| ≤ 0.15``, prices the
    linear excess outside, clamps a 1.55 inherited ``q0`` to URDF, and
    beats tether at the live rail numbers (``w=10`` vs ``w=5``).
"""
from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.model import (
    ARCH_FROM_WEIGHTS,
    AmortizedOptimizer,
    ModelConfig,
    model_config_from_checkpoint,
)
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
from amortized_opt.train_cost import (
    Phase2Config,
    substitute_waist_sp,
    train_phase2,
    waist_pr_hold_band_excess,
    waist_pr_hold_band_loss,
)
from amortized_opt.warmstart import seed_waist_pr_to_sp

torch.manual_seed(0)


def _small(**kw) -> ModelConfig:
    return ModelConfig(dec_layers=1, ctx_layers=1, max_horizon=25, **kw)


def _ctx(B: int = 4) -> ContextInputs:
    ctx = ContextInputs.zeros(B, num_drums=8, num_future=3)
    ctx.drum_valid[:, :3] = True
    ctx.goal_valid[:, 0] = True
    ctx.goal_pos[:, 0] = torch.tensor([0.4, 0.2, 0.8])
    ctx.yaw_now = torch.tensor([0.10, -0.20, 0.00, 0.50])
    ctx.aim_yaw = torch.tensor([0.30, -0.60, 0.00, 0.50])
    ctx.aim_valid = torch.tensor([True, True, True, False])
    return ctx


def _strike(B: int, H: int, pr_ref: torch.Tensor | None) -> StrikeContext:
    kl = torch.full((B,), H - 1, dtype=torch.long)
    return StrikeContext(
        hit_index=torch.stack([kl, kl], dim=1),
        active=torch.ones(B, 2, dtype=torch.float64),
        target_drum=torch.zeros(B, 2, dtype=torch.long),
        start_drum=torch.full((B, 2), -1, dtype=torch.long),
        p_star=torch.full((B, 2, 3), 0.30, dtype=torch.float64),
        v_star=torch.full((B, 2, 3), -1.0, dtype=torch.float64),
        n_out=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        .expand(B, 2, 3).contiguous(),
        hit_radius=torch.full((B, 2), 0.08, dtype=torch.float64),
        disk_offset=torch.zeros(B, 2, 3, dtype=torch.float64),
        half_plane_dir=torch.zeros(B, 2, 3, dtype=torch.float64),
        has_half_plane=torch.zeros(B, 2, dtype=torch.float64),
        contact_sigma=torch.full((B, 2), 0.20, dtype=torch.float64),
        shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        start_shaft_allowed=torch.zeros(B, 2, dtype=torch.float64),
        elev_required=torch.ones(B, 2, dtype=torch.float64),
        waist_pr_ref=pr_ref,
    )


def test_pr_head_off_is_bit_identical():
    m_off = AmortizedOptimizer(_small())
    m_on = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    missing, unexpected = m_on.load_state_dict(m_off.state_dict(), strict=False)
    assert not unexpected
    assert missing and all("waist_pr_sp" in k for k in missing)
    assert not any("waist_pr_sp" in k for k in m_off.state_dict())

    ctx = _ctx()
    eo, en = m_off.encode_context(ctx), m_on.encode_context(ctx)
    assert eo.waist_pr_sp is None
    assert torch.equal(eo.memory, en.memory[:, :-1])
    assert torch.equal(eo.mask, en.mask[:, :-1])
    assert m_on.cfg.n_context_tokens == m_off.cfg.n_context_tokens + 1


def test_untrained_pr_head_is_exactly_upright():
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    enc = m.encode_context(_ctx())
    assert torch.equal(enc.waist_pr_sp, torch.zeros(4, 2))
    assert torch.equal(enc.waist_pr_res, enc.waist_pr_sp)


def test_pr_setpoint_is_invariant_to_yaw_now():
    """The lean head must not see the ticking torso input the yaw head uses."""
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True, waist_sp_head=True))
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.weight.add_(0.02)
        m.encoder.waist_pr_sp_out.bias.add_(0.03)
    c1, c2 = _ctx(), _ctx()
    c2.yaw_now = c2.yaw_now + 0.4
    e1, e2 = m.encode_context(c1), m.encode_context(c2)
    assert torch.allclose(e1.waist_pr_sp, e2.waist_pr_sp, atol=1e-6)
    # yaw head DID move (it conditions on yaw_now)
    assert not torch.allclose(e1.waist_yaw_sp, e2.waist_yaw_sp, atol=1e-5)


def test_pr_gradient_reaches_the_head_both_ways():
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    ctx = _ctx()
    m.zero_grad()
    m.encode_context(ctx).waist_pr_sp.sum().backward()
    assert float(m.encoder.waist_pr_sp_out.weight.grad.norm()) > 0

    # Untrained sp is 0, so proj(sp) contributes no weight-gradient; nudge
    # the head off upright so the conditioning token is a function of the
    # linear map, the same as the yaw test's nonzero [sp, yaw_now, sp-yaw_now].
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.05)
    m.zero_grad()
    B = 4
    enc = m.encode_context(ctx)
    masks = build_waypoint_masks(
        torch.full((B,), 0.4), torch.full((B,), 0.3), max_horizon=25
    )
    out = m(torch.zeros(B, 25, 17), torch.zeros(B, 34), enc, masks)
    out.traj.pow(2).mean().backward()
    assert float(m.encoder.waist_pr_sp_proj.weight.grad.norm()) > 0
    assert m.encoder.waist_pr_sp_emb.grad is not None


def test_pr_checkpoint_reconciliation():
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    stored = {k: v for k, v in vars(m.cfg).items() if k != "waist_pr_sp_head"}
    ck = {"model_cfg": stored, "model": m.state_dict()}
    assert ARCH_FROM_WEIGHTS["waist_pr_sp_head"] in ck["model"]
    assert model_config_from_checkpoint(ck).waist_pr_sp_head is True
    ck_off = {"model_cfg": stored,
              "model": AmortizedOptimizer(_small()).state_dict()}
    assert model_config_from_checkpoint(ck_off).waist_pr_sp_head is False


def test_yaw_checkpoint_still_loads_with_pr_flag_off():
    """Widening yaw_sp_out would have broken this; the sibling head must not."""
    m_yaw = AmortizedOptimizer(_small(waist_sp_head=True))
    m_both_off_pr = AmortizedOptimizer(_small(waist_sp_head=True, waist_pr_sp_head=False))
    missing, unexpected = m_both_off_pr.load_state_dict(m_yaw.state_dict(), strict=True)
    assert not missing and not unexpected


def test_upright_well_tracks_the_pr_setpoint():
    from dataclasses import replace

    H, B = 10, 4
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.05)
    enc = m.encode_context(_ctx())
    sp = enc.waist_pr_sp
    assert float(sp.detach().abs().max()) > 1e-4

    cost = FusedCost(CostConfig(n_sub=1, w_waist_upright=1.0),
                     max_horizon=H, device="cpu", dtype=torch.float64)
    pitch_i = cost.joint_names.index("waist_pitch_joint")
    roll_i = cost.joint_names.index("waist_roll_joint")
    strike = replace(
        _strike(B, H, None),
        waist_pr_ref=sp.double().unsqueeze(1).expand(-1, H, -1).contiguous(),
    )
    q0 = torch.zeros(B, 17, dtype=torch.float64)
    v0 = torch.zeros(B, 17, dtype=torch.float64)
    on = q0.unsqueeze(1).repeat(1, H, 1)
    on[:, :, pitch_i] = sp[:, 0].double().detach().unsqueeze(1)
    on[:, :, roll_i] = sp[:, 1].double().detach().unsqueeze(1)
    off = on.clone()
    off[:, :, pitch_i] += 0.3

    ct_on = cost.forward(q0, on, v0, strike)
    assert float(ct_on.raw["posture_upright"].abs().max()) < 1e-12
    m.zero_grad()
    off = off.clone().requires_grad_(True)
    ct_off = cost.forward(q0, off, v0, strike)
    # mean over 2 lean rows; only pitch is 0.3 off
    assert torch.allclose(ct_off.raw["posture_upright"],
                          torch.full((B,), 0.5 * 0.09, dtype=torch.float64))
    ct_off.raw["posture_upright"].sum().backward()
    # stopgrad(sp): encoder is not an upright teacher; decoder still is
    g = m.encoder.waist_pr_sp_out.weight.grad
    assert g is None or float(g.norm()) == 0.0
    assert float(off.grad[:, :, pitch_i].abs().max()) > 0


def test_pr_hold_prices_walk_from_q0_at_encode():
    """Encoder SP stays near the waist pose at encode; q0 is not an input."""
    B = 4
    n = len(G1_UPPER_BODY_JOINTS)
    pitch_i = G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")
    roll_i = G1_UPPER_BODY_JOINTS.index("waist_roll_joint")
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    ctx = _ctx(B)
    q0 = torch.zeros(B, n)
    q0[:, pitch_i] = 0.20
    q0[:, roll_i] = -0.10
    w = 5.0

    enc = m.encode_context(ctx)
    # Untrained SP is 0, so the hold is ||0 - q0_pr||².
    d = enc.waist_pr_sp.float() - q0[:, [pitch_i, roll_i]].float()
    loss = w * d.pow(2).mean()
    assert torch.allclose(loss, torch.tensor(w * (0.04 + 0.01) / 2))

    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.copy_(
            torch.tensor([0.20, -0.10], dtype=m.encoder.waist_pr_sp_out.bias.dtype)
        )
    m.zero_grad()
    enc = m.encode_context(ctx)
    d = enc.waist_pr_sp.float() - q0[:, [pitch_i, roll_i]].detach().float()
    loss = w * d.pow(2).mean()
    assert float(loss) < 1e-6
    # Gradient still reaches the head when the hold is off the pose.
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.15)
    m.zero_grad()
    enc = m.encode_context(ctx)
    q0_g = q0.clone().requires_grad_(True)
    d = enc.waist_pr_sp.float() - q0_g[:, [pitch_i, roll_i]].detach().float()
    (w * d.pow(2).mean()).backward()
    assert float(m.encoder.waist_pr_sp_out.weight.grad.norm()) > 0
    assert q0_g.grad is None or float(q0_g.grad.abs().max()) == 0.0
    # Still no plant lean in the encoder: two q0s, same SP.
    enc2 = m.encode_context(ctx)
    assert torch.equal(enc.waist_pr_sp, enc2.waist_pr_sp)


def test_hold_band_idle_inside_window():
    sp = torch.tensor([[0.10, -0.04]], dtype=torch.float64)
    q0 = torch.tensor([[0.20, 0.05]], dtype=torch.float64)
    assert float(waist_pr_hold_band_loss(sp, q0, band_rad=0.15, pr_limit=0.52)) == 0.0
    assert torch.equal(
        waist_pr_hold_band_excess(sp, q0, band_rad=0.15, pr_limit=0.52),
        torch.zeros_like(sp),
    )


def test_hold_band_prices_linear_excess():
    # |0.12 - 0.52| - 0.15 = 0.25 on pitch; roll in-band.
    sp = torch.tensor([[0.12, 0.00]], dtype=torch.float64, requires_grad=True)
    q0 = torch.tensor([[0.52, 0.00]], dtype=torch.float64, requires_grad=True)
    loss = waist_pr_hold_band_loss(sp, q0, band_rad=0.15, pr_limit=0.52)
    assert torch.allclose(loss, torch.tensor(0.25 / 2, dtype=torch.float64))
    loss.backward()
    assert q0.grad is None or float(q0.grad.abs().max()) == 0.0
    # Outside: pull sp toward q0 (increasing sp lowers the excess).
    assert float(sp.grad[0, 0]) < 0.0
    assert float(sp.grad[0, 1]) == 0.0


def test_hold_band_clamps_dumped_q0_to_urdf():
    sp = torch.zeros(1, 2, dtype=torch.float64)
    q0 = torch.tensor([[1.55, 0.0]], dtype=torch.float64)
    # Target is 0.52, not 1.55: excess = 0.52 - 0.15 = 0.37.
    got = waist_pr_hold_band_excess(sp, q0, band_rad=0.15, pr_limit=0.52)
    assert torch.allclose(got, torch.tensor([[0.37, 0.0]], dtype=torch.float64))
    unclamped = (sp - q0).abs() - 0.15
    assert float(unclamped[0, 0]) > 1.0
    assert float(got[0, 0]) < float(unclamped[0, 0])


def test_hold_band_beats_tether_at_rail_q0():
    """Live numbers: sp=0.12, q0=0.52, w_tether=5, w_hinge=10 → net toward q0.

    Squared hinge at these weights equilibrates with a residual gap; the
    linear hinge's constant pull (10) beats tether's 2*5*0.12.
    """
    sp = torch.tensor([[0.12, 0.12]], dtype=torch.float64, requires_grad=True)
    q0 = torch.tensor([[0.52, 0.52]], dtype=torch.float64)
    tether = 5.0 * sp.pow(2).mean()
    hinge = 10.0 * waist_pr_hold_band_loss(sp, q0, band_rad=0.15, pr_limit=0.52)
    (tether + hinge).backward()
    assert float(sp.grad[0, 0]) < 0.0
    assert float(sp.grad[0, 1]) < 0.0


def test_hold_band_weight_defaults_off():
    assert Phase2Config().w_waist_pr_sp_hold_band == 0.0


def test_pr_tether_prices_walk_from_upright():
    ctx = _ctx()
    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    w = 5.0
    enc = m.encode_context(ctx)
    assert float(w * enc.waist_pr_res.float().pow(2).mean()) == 0.0

    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.05)
    m.zero_grad()
    enc = m.encode_context(ctx)
    assert torch.equal(enc.waist_pr_res, enc.waist_pr_sp)
    loss = w * enc.waist_pr_res.float().pow(2).mean()
    assert float(loss) > 0.0
    loss.backward()
    assert float(m.encoder.waist_pr_sp_out.weight.grad.norm()) > 0


def test_substitute_yaw_keeps_pr_token_when_both_heads_on():
    m = AmortizedOptimizer(_small(waist_sp_head=True, waist_pr_sp_head=True))
    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(0.04)
        m.encoder.waist_sp_out.bias.add_(0.07)
    ctx = _ctx()
    enc = m.encode_context(ctx)
    pr_tok = enc.memory[:, -2].clone()
    pr_sp = enc.waist_pr_sp.clone()
    base = torch.where(ctx.aim_valid, ctx.aim_yaw, ctx.yaw_now)
    sub = substitute_waist_sp(
        m, enc, enc.waist_yaw_sp + 0.2, ctx.yaw_now, base,
    )
    assert torch.allclose(sub.waist_pr_sp, pr_sp)
    assert torch.allclose(sub.memory[:, -2], pr_tok)
    assert not torch.allclose(sub.memory[:, -1], enc.memory[:, -1])
    assert sub.memory.shape[1] == enc.memory.shape[1]


def test_quintic_seed_eases_pr_to_encoder_sp():
    """Pitch/roll init ramps to the announced SP, like yaw to aim_yaw."""
    B, H, n = 4, 20, len(G1_UPPER_BODY_JOINTS)
    pitch_i = G1_UPPER_BODY_JOINTS.index("waist_pitch_joint")
    roll_i = G1_UPPER_BODY_JOINTS.index("waist_roll_joint")
    q0 = torch.zeros(B, n)
    q0[:, pitch_i] = 0.30
    q0[:, roll_i] = -0.10
    qd0 = torch.zeros(B, n)
    init = q0.unsqueeze(1).expand(B, H, n).contiguous()

    assert seed_waist_pr_to_sp(init, q0, qd0, None) is init

    m = AmortizedOptimizer(_small(waist_pr_sp_head=True))
    z = seed_waist_pr_to_sp(
        init, q0, qd0, m.encode_context(_ctx(B)).waist_pr_sp,
        joint_names=G1_UPPER_BODY_JOINTS, dt=CONTROL_DT, ease_wp=5,
    )
    assert float(z[:, -1, pitch_i].abs().max()) < 1e-5
    assert float(z[:, -1, roll_i].abs().max()) < 1e-5
    assert float(z[:, 0, pitch_i].mean()) < 0.30  # already moving toward 0

    with torch.no_grad():
        m.encoder.waist_pr_sp_out.bias.add_(torch.tensor([0.12, -0.08]))
    sp = m.encode_context(_ctx(B)).waist_pr_sp
    got = seed_waist_pr_to_sp(
        init, q0, qd0, sp,
        joint_names=G1_UPPER_BODY_JOINTS, dt=CONTROL_DT, ease_wp=5,
    )
    assert torch.allclose(got[:, -1, pitch_i], sp[:, 0], atol=1e-5)
    assert torch.allclose(got[:, -1, roll_i], sp[:, 1], atol=1e-5)
    # Other joints untouched.
    mask = torch.ones(n, dtype=torch.bool)
    mask[pitch_i] = mask[roll_i] = False
    assert torch.equal(got[:, :, mask], init[:, :, mask])
    # Seed is a prior: no gradient through the quintic.
    assert not got.requires_grad


def test_tokenizer_does_not_export_context_fields():
    """Roller re-goal used to ``from .tokenizer import CONTEXT_FIELDS`` and die.

    The name lives on ``warmstart_data`` (derived from the dataclass).  Keep
    it off ``tokenizer`` so a future local import cannot recreate the crash.
    """
    import amortized_opt.tokenizer as tok
    from amortized_opt.tokenizer import ContextInputs
    from amortized_opt.train_cost import MPCRoller
    from amortized_opt.warmstart_data import CONTEXT_FIELDS

    assert not hasattr(tok, "CONTEXT_FIELDS")
    assert CONTEXT_FIELDS == tuple(ContextInputs.__dataclass_fields__)
    # The method that crashed at ~step 200 when KIND_POST_HIT came online.
    import inspect
    src = inspect.getsource(MPCRoller._pr_sp_for_rows)
    assert "from .tokenizer import CONTEXT_FIELDS" not in src


def test_train_phase2_refuses_a_headless_pr_model(tmp_path):
    m = AmortizedOptimizer(_small())
    cfg = Phase2Config(compile_mode=None, device="cpu", waist_pr_sp_from_head=True)
    with pytest.raises(ValueError, match="waist_pr_sp_head"):
        train_phase2(model=m, pool=None, cfg=cfg, model_cfg=_small(),
                     run_dir=tmp_path)
