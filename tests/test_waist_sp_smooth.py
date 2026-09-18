"""Cross-segment waist set-point smoothness (``Phase2Config.w_waist_sp_smooth``).

The encoder announces one held yaw per goal segment.  Closed loop re-encodes
at every hit, so intra-segment terms never see ``sp`` jump.  The term prices
``(sp_new - sp_old)²`` on a synthetic next-hit context: promote the soonest
current goal to ``future[0]``, refresh ``aim_yaw``, encode again.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from amortized_opt.constants import PELVIS_IN_ENV
from amortized_opt.model import AmortizedOptimizer, ModelConfig
from amortized_opt.rhythm import HANDS, DrumKit
from amortized_opt.tokenizer import ContextInputs
from amortized_opt.train_cost import (
    _waist_sp_smooth_term,
    promote_context_next_hit,
    refresh_context_aim,
    slice_context,
)
from amortized_opt.waist_aim import aim_yaw

LEFT, RIGHT = 0, 1


def _small(**kw) -> ModelConfig:
    return ModelConfig(dec_layers=1, ctx_layers=1, max_horizon=25,
                       num_drums=8, num_future=3, **kw)


def _kit_ctx(*, num_future: int = 2, B: int = 2) -> tuple[DrumKit, ContextInputs]:
    """Two-row context: left fires first, with a cross-body cymbal in future[0]."""
    kit = DrumKit.from_layout()
    pos = kit.positions_in("env")
    ctx = ContextInputs.zeros(B, num_drums=8, num_future=3)
    d = kit.num_drums
    ctx.drum_pos[:, :d] = torch.as_tensor(pos, dtype=ctx.drum_pos.dtype)
    ctx.drum_valid[:, :d] = True
    ctx.yaw_now[:] = 0.10

    # left snare (inside the reach cone) then left/cymbal_right (66 deg inward,
    # which opens the gate); right has a later tom so left fires first.
    lo, far, mid = 0, 5, 2
    ctx.goal_valid[:, LEFT] = True
    ctx.goal_drum[:, LEFT] = lo
    ctx.goal_pos[:, LEFT] = ctx.drum_pos[:, lo]
    ctx.goal_tth[:, LEFT] = 0.20
    ctx.goal_valid[:, RIGHT] = True
    ctx.goal_drum[:, RIGHT] = mid
    ctx.goal_pos[:, RIGHT] = ctx.drum_pos[:, mid]
    ctx.goal_tth[:, RIGHT] = 0.50

    n_f = min(int(num_future), 3)
    ctx.future_valid[:, LEFT, 0] = True
    ctx.future_drum[:, LEFT, 0] = far
    ctx.future_pos[:, LEFT, 0] = ctx.drum_pos[:, far]
    ctx.future_gap[:, LEFT, 0] = 0.30
    if n_f > 1:
        ctx.future_valid[:, LEFT, 1] = True
        ctx.future_drum[:, LEFT, 1] = lo
        ctx.future_pos[:, LEFT, 1] = ctx.drum_pos[:, lo]
        ctx.future_gap[:, LEFT, 1] = 0.60

    refresh_context_aim(ctx)
    return kit, ctx


def test_promote_leaves_the_other_hand_and_unchanged_rows_alone():
    _kit, ctx = _kit_ctx(num_future=2, B=2)
    ctx.goal_valid[1] = False
    ctx.future_valid[1] = False
    before = ctx.goal_pos[1].clone()
    left_future0 = ctx.future_pos[0, LEFT, 0].clone()
    right_pos = ctx.goal_pos[0, RIGHT].clone()
    old_right_tth = float(ctx.goal_tth[0, RIGHT])
    old_left_tth = float(ctx.goal_tth[0, LEFT])

    ctx_new, changed = promote_context_next_hit(ctx)
    assert bool(changed[0]) and not bool(changed[1])
    assert torch.equal(ctx_new.goal_pos[1], before)
    assert torch.equal(ctx_new.goal_pos[0, LEFT], left_future0)
    assert torch.equal(ctx_new.goal_pos[0, RIGHT], right_pos)
    # fire-time from-now of the promoted hit is the stored gap
    assert float(ctx_new.goal_tth[0, LEFT]) == pytest.approx(0.30)
    # other hand ages by the wait until the fire
    assert float(ctx_new.goal_tth[0, RIGHT]) == pytest.approx(
        old_right_tth - old_left_tth
    )
    # surviving future re-anchored: 0.60 - 0.30
    assert bool(ctx_new.future_valid[0, LEFT, 0])
    assert float(ctx_new.future_gap[0, LEFT, 0]) == pytest.approx(0.30)
    assert not bool(ctx_new.future_valid[0, LEFT, 1])
    # original not mutated
    assert float(ctx.goal_tth[0, LEFT]) == pytest.approx(0.20)


def test_promote_last_hit_goes_idle():
    _kit, ctx = _kit_ctx(num_future=1, B=1)
    ctx.future_valid[:] = False
    ctx_new, changed = promote_context_next_hit(ctx)
    assert bool(changed[0])
    assert not bool(ctx_new.goal_valid[0, LEFT])
    assert bool(ctx_new.goal_valid[0, RIGHT])  # other hand untouched
    assert bool(ctx_new.idle_valid[0, LEFT])
    assert not bool(ctx_new.idle_valid[0, RIGHT])


def test_refresh_aim_matches_direct_aim_yaw():
    kit, ctx = _kit_ctx(num_future=1, B=1)
    pelvis = np.asarray(PELVIS_IN_ENV, float)
    targets = {}
    tth = {}
    futures = {h: [] for h in HANDS}
    for hi, hand in enumerate(HANDS):
        if bool(ctx.goal_valid[0, hi]):
            targets[hand] = ctx.goal_pos[0, hi].numpy() - pelvis
            tth[hand] = float(ctx.goal_tth[0, hi])
        if bool(ctx.future_valid[0, hi, 0]):
            futures[hand].append((
                ctx.future_pos[0, hi, 0].numpy() - pelvis,
                float(ctx.future_gap[0, hi, 0] + ctx.goal_tth[0, hi]),
            ))
    direct = aim_yaw(
        targets, yaw_current=float(ctx.yaw_now[0]), goal_tth=tth, futures=futures,
    )
    assert abs(float(ctx.aim_yaw[0]) - direct) < 1e-8


def test_untrained_head_smooth_loss_is_the_analytic_aim_jump():
    _kit, ctx = _kit_ctx(num_future=1, B=2)
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    enc = m.encode_context(ctx)
    assert torch.allclose(enc.waist_yaw_sp, ctx.aim_yaw)

    ctx_new, changed = promote_context_next_hit(ctx)
    idx = changed.nonzero(as_tuple=True)[0]
    refresh_context_aim(ctx_new, rows=idx)
    enc_new = m.encode_context(slice_context(ctx_new, idx))
    d = enc_new.waist_yaw_sp - enc.waist_yaw_sp.index_select(0, idx)
    expect = d.pow(2).mean()

    loss, yaw_abs, pr_abs, n = _waist_sp_smooth_term(model=m, ctx=ctx, enc=enc)
    assert n == int(idx.numel())
    assert float(loss) == pytest.approx(float(expect), rel=1e-5, abs=1e-6)
    assert yaw_abs == pytest.approx(float(d.abs().mean()), rel=1e-5, abs=1e-6)
    assert pr_abs == 0.0
    # a real next-hit on this kit should move the analytic aim
    assert yaw_abs > 1e-4


def test_smooth_gradient_reaches_the_head():
    _kit, ctx = _kit_ctx(num_future=1, B=2)
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    enc = m.encode_context(ctx)
    m.zero_grad()
    loss, *_ = _waist_sp_smooth_term(model=m, ctx=ctx, enc=enc)
    loss.backward()
    assert m.encoder.waist_sp_out.weight.grad is not None
    assert float(m.encoder.waist_sp_out.weight.grad.norm()) > 0


def test_smooth_term_is_zero_when_nothing_promotes():
    ctx = ContextInputs.zeros(3, num_drums=8, num_future=3)
    ctx.yaw_now[:] = 0.1
    m = AmortizedOptimizer(_small(waist_sp_head=True))
    enc = m.encode_context(ctx)
    loss, yaw_abs, pr_abs, n = _waist_sp_smooth_term(model=m, ctx=ctx, enc=enc)
    assert n == 0
    assert float(loss) == 0.0
    assert yaw_abs == 0.0 and pr_abs == 0.0
