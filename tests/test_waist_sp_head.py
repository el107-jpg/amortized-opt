"""The learned waist-yaw set-point (``docs/waist_sp_head_plan.md``).

Four claims, each load-bearing for the design:

1. **OFF is bit-identical.**  ``waist_sp_head=False`` adds no parameters and
   the encoding is unchanged, so every existing checkpoint loads strictly and
   every existing run is unaffected.
2. **The untrained head IS the analytic law.**  ``waist_sp_out`` is zero-init,
   so at step 0 the set-point equals ``ContextInputs.aim_yaw`` exactly (and
   falls back to ``yaw_now`` where ``aim_valid`` is False).  This is the whole
   distillation -- there is no distillation loss to get wrong.
3. **Gradient reaches the head** through the set-point (the phase-2 anchor
   path) and the conditioning token (the phase-1 imitation path).
4. **The data path computes the same aim the head starts at**:
   ``build_context_inputs(yaw_now=...)`` agrees with a direct ``aim_yaw`` call
   in the pelvis frame, and a context built WITHOUT ``yaw_now`` (an old pool)
   still encodes, with the head falling back to "hold".

Plus the cost-side knobs: the ``waist_aim`` deadband prices only the excess
over the band, and ``waist_aim_yaw_only`` silences the pitch/roll rows.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.cost import CostConfig, FusedCost, StrikeContext
from amortized_opt.model import (
    ARCH_FROM_WEIGHTS,
    AmortizedOptimizer,
    ModelConfig,
    model_config_from_checkpoint,
)
from amortized_opt.rhythm import (
    PELVIS_IN_ENV,
    DrumKit,
    GoalSnapshot,
    HandGoal,
    build_context_inputs,
)
from amortized_opt.tokenizer import ContextInputs, build_waypoint_masks
from amortized_opt.waist_aim import aim_yaw

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


# ----------------------------------------------------------------- model --
def test_head_off_is_bit_identical():
    m_off = AmortizedOptimizer(_small())
    m_on = AmortizedOptimizer(_small(waist_sp_head=True))
    missing, unexpected = m_on.load_state_dict(m_off.state_dict(), strict=False)
    assert not unexpected
    assert missing and all("waist_sp" in k for k in missing)
    assert not any("waist_sp" in k for k in m_off.state_dict())

    ctx = _ctx()
    eo, en = m_off.encode_context(ctx), m_on.encode_context(ctx)
    assert eo.waist_yaw_sp is None
    assert torch.equal(eo.memory, en.memory[:, :-1])
    assert torch.equal(eo.mask, en.mask[:, :-1])


def test_untrained_head_is_exactly_the_analytic_aim():
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    enc = m.encode_context(_ctx())
    # rows 0-2: aim_valid -> sp == aim_yaw; row 3: invalid -> sp == yaw_now
    assert torch.equal(enc.waist_yaw_sp, torch.tensor([0.30, -0.60, 0.00, 0.50]))
    assert torch.equal(enc.waist_yaw_res, torch.zeros(4))


def test_gradient_reaches_the_head_both_ways():
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    ctx = _ctx()

    # the anchor path: a loss on the set-point itself
    m.zero_grad()
    m.encode_context(ctx).waist_yaw_sp.sum().backward()
    assert float(m.encoder.waist_sp_out.weight.grad.norm()) > 0

    # the conditioning path: a loss on the TRAJECTORY reaches the token params
    m.zero_grad()
    B = 4
    enc = m.encode_context(ctx)
    masks = build_waypoint_masks(
        torch.full((B,), 0.4), torch.full((B,), 0.3), max_horizon=25
    )
    out = m(torch.zeros(B, 25, 17), torch.zeros(B, 34), enc, masks)
    out.traj.pow(2).mean().backward()
    assert float(m.encoder.waist_sp_proj.weight.grad.norm()) > 0
    assert m.encoder.waist_sp_emb.grad is not None


def test_checkpoint_reconciliation_detects_the_head():
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    stored = {k: v for k, v in vars(m.cfg).items() if k != "waist_sp_head"}
    ck = {"model_cfg": stored, "model": m.state_dict()}
    assert ARCH_FROM_WEIGHTS["waist_sp_head"] in ck["model"]
    assert model_config_from_checkpoint(ck).waist_sp_head is True
    ck_off = {"model_cfg": stored,
              "model": AmortizedOptimizer(_small()).state_dict()}
    assert model_config_from_checkpoint(ck_off).waist_sp_head is False


# ------------------------------------------------------------ data path --
def test_build_context_inputs_matches_direct_aim_yaw():
    kit = DrumKit.from_layout()
    goals = {
        "left": HandGoal(
            drum=0,
            position=kit.positions_in("env")[0] + np.array([0.0, 0.0, 0.02]),
            velocity=np.array([0.0, 0.0, -2.0]),
            time_to_hit=0.4,
            active=True,
        ),
        "right": HandGoal.inactive(),
    }
    snap = GoalSnapshot(t_now=0.0, goal=goals, future={"left": [], "right": []})
    for yaw0 in (0.0, 0.25, -1.0):
        ctx = build_context_inputs(kit, snap, num_drums=8, num_future=3,
                                   yaw_now=yaw0, dtype=torch.float64)
        direct = aim_yaw(
            {"left": goals["left"].position - PELVIS_IN_ENV}, yaw_current=yaw0
        )
        assert bool(ctx.aim_valid[0])
        assert abs(float(ctx.yaw_now[0]) - yaw0) < 1e-12
        assert abs(float(ctx.aim_yaw[0]) - direct) < 1e-12

    # an old pool / a caller without robot state: fields default, still encodes
    ctx0 = build_context_inputs(kit, snap, num_drums=8, num_future=3)
    assert not bool(ctx0.aim_valid[0]) and float(ctx0.aim_yaw[0]) == 0.0
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    enc = m.encode_context(ctx0.to("cpu"))
    assert torch.equal(enc.waist_yaw_sp, ctx0.yaw_now.float())  # "hold" fallback


# ------------------------------------------------------------ cost knobs --
def _yaw_cost(cfg_kw: dict, yaw: float, ref: float) -> float:
    """raw['waist_aim'] for a constant-yaw trajectory against a constant ref."""
    H, B = 10, 1
    cost = FusedCost(CostConfig(n_sub=1, **cfg_kw), max_horizon=H,
                     device="cpu", dtype=torch.float64)
    yaw_i = cost.joint_names.index("waist_yaw_joint")
    q0 = torch.zeros(B, 17, dtype=torch.float64)
    q0[:, yaw_i] = yaw
    wp = q0.unsqueeze(1).expand(B, H, 17).contiguous()
    v0 = torch.zeros(B, 17, dtype=torch.float64)
    kl = torch.full((B,), H - 1, dtype=torch.long)
    ctx = StrikeContext(
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
        waist_yaw_ref=torch.full((B, H), ref, dtype=torch.float64),
    )
    return float(cost.forward(q0, wp, v0, ctx).raw["waist_aim"][0])


def test_waist_aim_deadband_prices_only_the_excess():
    base = dict(waist_aim_yaw_only=True, w_waist_aim=1.0)
    # |err| = 0.3: plain quadratic 0.09; with a 0.2 band, (0.3-0.2)^2 = 0.01
    plain = _yaw_cost(base, yaw=0.3, ref=0.0)
    banded = _yaw_cost({**base, "waist_aim_deadband": 0.2}, yaw=0.3, ref=0.0)
    inside = _yaw_cost({**base, "waist_aim_deadband": 0.2}, yaw=0.15, ref=0.0)
    assert abs(plain - 0.09) < 1e-9
    assert abs(banded - 0.01) < 1e-9
    assert inside == 0.0


def test_waist_aim_yaw_only_drops_pitch_roll():
    # yaw error 0, but the trajectory is priced against pitch=roll=0 -- put the
    # error in PITCH instead and check the switch silences it.
    H, B = 10, 1
    for yaw_only, expect_zero in ((True, True), (False, False)):
        cost = FusedCost(
            CostConfig(n_sub=1, w_waist_aim=1.0, waist_aim_yaw_only=yaw_only),
            max_horizon=H, device="cpu", dtype=torch.float64,
        )
        pitch_i = cost.joint_names.index("waist_pitch_joint")
        q0 = torch.zeros(B, 17, dtype=torch.float64)
        q0[:, pitch_i] = 0.3
        wp = q0.unsqueeze(1).expand(B, H, 17).contiguous()
        v0 = torch.zeros(B, 17, dtype=torch.float64)
        kl = torch.full((B,), H - 1, dtype=torch.long)
        ctx_kw = dict(
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
            waist_yaw_ref=torch.zeros(B, H, dtype=torch.float64),
        )
        val = float(cost.forward(q0, wp, v0, StrikeContext(**ctx_kw))
                    .raw["waist_aim"][0])
        assert (val == 0.0) == expect_zero, (yaw_only, val)


# ------------------------------------------- the phase-2 loop's three uses --
def _strike(B: int, H: int, ref: torch.Tensor) -> StrikeContext:
    """The minimal all-active strike context ``_yaw_cost`` builds, batched."""
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
        waist_yaw_ref=ref,
    )


def test_the_substituted_reference_prices_yaw_against_the_head():
    """The training loop's head-mode substitution, end to end: the encoder's
    set-point broadcast over the knots becomes ``waist_yaw_ref``, a trajectory
    HOLDING the set-point pays zero, one 0.3 rad off pays 0.09, and the
    gradient of the term reaches the head's own parameters (the anchor path)."""
    from dataclasses import replace

    H, B = 10, 4
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    with torch.no_grad():                  # a real LEARNED set-point, not aim
        m.encoder.waist_sp_out.weight.add_(0.01)
        m.encoder.waist_sp_out.bias.add_(0.05)
    enc = m.encode_context(_ctx())
    sp = enc.waist_yaw_sp
    assert float((sp.detach() - torch.tensor([0.30, -0.60, 0.00, 0.50]))
                 .abs().max()) > 1e-4      # the head really walked

    cost = FusedCost(CostConfig(n_sub=1, w_waist_aim=1.0, waist_aim_yaw_only=True),
                     max_horizon=H, device="cpu", dtype=torch.float64)
    yaw_i = cost.joint_names.index("waist_yaw_joint")
    # EXACTLY the loop's substitution (train_phase2's head-mode branch)
    strike = replace(_strike(B, H, None),
                     waist_yaw_ref=sp.double().unsqueeze(1).expand(-1, H))

    q0 = torch.zeros(B, 17, dtype=torch.float64)
    v0 = torch.zeros(B, 17, dtype=torch.float64)
    on = q0.unsqueeze(1).repeat(1, H, 1)
    on[:, :, yaw_i] = sp.double().detach().unsqueeze(1)
    off = on.clone()
    off[:, :, yaw_i] += 0.3

    ct_on = cost.forward(q0, on, v0, strike)
    assert float(ct_on.raw["waist_aim"].abs().max()) < 1e-12
    m.zero_grad()
    ct_off = cost.forward(q0, off, v0, strike)
    assert torch.allclose(ct_off.raw["waist_aim"],
                          torch.full((B,), 0.09, dtype=torch.float64))
    ct_off.raw["waist_aim"].sum().backward()
    assert float(m.encoder.waist_sp_out.weight.grad.norm()) > 0


def test_the_tether_prices_the_walk_from_the_analytic_law():
    """``w_waist_sp_tether * mean(res^2)`` with ``res = sp - aim_yaw`` by
    construction: exactly zero at init (zero-init head), positive as soon as
    the head walks, and its gradient reaches the head."""
    ctx = _ctx()
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    w = 5.0
    enc = m.encode_context(ctx)
    assert float(w * enc.waist_yaw_res.float().pow(2).mean()) == 0.0

    with torch.no_grad():
        m.encoder.waist_sp_out.weight.add_(0.01)
        m.encoder.waist_sp_out.bias.add_(0.05)
    m.zero_grad()
    enc = m.encode_context(ctx)
    # res IS sp - base, base = aim where valid, else "hold" (yaw_now)
    base = torch.where(ctx.aim_valid, ctx.aim_yaw, ctx.yaw_now)
    assert torch.allclose(enc.waist_yaw_sp.detach(),
                          base + enc.waist_yaw_res.detach(), atol=1e-6)
    loss = w * enc.waist_yaw_res.float().pow(2).mean()
    assert float(loss) > 0.0
    loss.backward()
    assert float(m.encoder.waist_sp_out.weight.grad.norm()) > 0


def test_train_phase2_refuses_a_headless_model(tmp_path):
    """``waist_sp_from_head`` without ``ModelConfig.waist_sp_head`` must be a
    loud ValueError, not a silent run whose anchor term prices nothing."""
    from amortized_opt.train_cost import Phase2Config, train_phase2

    m = AmortizedOptimizer(_small())            # no head
    cfg = Phase2Config(compile_mode=None, device="cpu", waist_sp_from_head=True)
    with pytest.raises(ValueError, match="waist_sp_head"):
        # the check fires before the pool is ever touched
        train_phase2(model=m, pool=None, cfg=cfg, model_cfg=_small(),
                     run_dir=tmp_path)
