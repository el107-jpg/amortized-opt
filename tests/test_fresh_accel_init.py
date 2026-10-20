"""Decoder init is a fresh acceleration line to a frozen terminal pose."""

from __future__ import annotations

import numpy as np
import torch

import amortized_opt as A
from amortized_opt.warmstart import arm_slice
from isaac_rl.mpc import IsaacMPC, MPCConfig
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig

H = 50
B = 1


def _mpc(**cfg_kw):
    torch.manual_seed(0)
    kit = A.DrumKit.from_layout()
    clip = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[0]
    mcfg = A.ModelConfig(
        d_model=32, dec_heads=2, dec_layers=1, d_ctx=32, ctx_heads=2,
        ctx_layers=1, max_horizon=H, terminal_head=True,
    )
    model = A.build_model(mcfg, "cpu").eval()
    sched = BatchedSchedule(
        B, ScheduleConfig(horizon=H, num_drums=mcfg.num_drums, num_future=mcfg.num_future),
        device="cpu",
    )
    sched.reset(torch.arange(B), [clip], [kit], rng=np.random.default_rng(0))
    mpc = IsaacMPC(
        model, sched,
        MPCConfig(horizon=H, amp_dtype=None, fresh_accel_init=True, **cfg_kw),
        device="cpu", kit_names=tuple(kit.names),
    )
    return mpc, sched


def _arm_in_horizon(hit_index: torch.Tensor) -> tuple[int, int] | None:
    for h in (0, 1):
        k = int(hit_index[0, h])
        if 1 <= k <= H:
            return h, k
    return None


def test_each_tick_redraws_the_line_to_the_same_terminal():
    mpc, sched = _mpc()
    q = torch.zeros(B, A.N_DOF)
    qd = torch.zeros_like(q)
    mpc.reset(torch.arange(B), q)
    for _ in range(300):
        if _arm_in_horizon(sched.hit_index()) is not None:
            break
        sched.advance()
    else:
        raise AssertionError("eval clip never put a hit inside the horizon")

    mpc.plan(q, qd)
    terminal = mpc._frozen_terminal.clone()
    init = mpc.init.clone()
    hand, k = _arm_in_horizon(sched.hit_index())
    cols = arm_slice("left" if hand == 0 else "right")
    torch.testing.assert_close(init[0, k - 1, cols], terminal[0, cols])

    q2 = q.clone()
    q2[0, cols[0]] += 0.2
    mpc.plan(q2, qd)
    torch.testing.assert_close(mpc._frozen_terminal, terminal)
    assert not torch.allclose(mpc.init, init)
    torch.testing.assert_close(mpc.init[0, k - 1, cols], terminal[0, cols])

    held = mpc.init.clone()
    mpc.commit(q2, q2, torch.zeros(B, 2, dtype=torch.bool))
    torch.testing.assert_close(mpc.init, held)

    left = arm_slice("left")
    right = arm_slice("right")
    base = mpc._frozen_terminal.clone()
    real = mpc.model.terminal_initializer.terminal_pose

    def _shifted(ctx, state, masks):
        out = real(ctx, state, masks).clone()
        out[:, left] = out[:, left] + 0.05
        return out

    mpc.model.terminal_initializer.terminal_pose = _shifted
    mpc._frozen_goal_key[:, 0] = -1
    mpc.plan(q2, qd)
    torch.testing.assert_close(mpc._frozen_terminal[:, right], base[:, right])
    assert float((mpc._frozen_terminal[:, left] - base[:, left]).abs().mean()) > 1e-3


def test_every_tick_rebuilds_the_terminal_from_the_measured_pose():
    mpc, sched = _mpc(fresh_terminal_every_tick=True)
    q = torch.zeros(B, A.N_DOF)
    qd = torch.zeros_like(q)
    mpc.reset(torch.arange(B), q)
    for _ in range(300):
        if _arm_in_horizon(sched.hit_index()) is not None:
            break
        sched.advance()
    else:
        raise AssertionError("eval clip never put a hit inside the horizon")

    real = mpc.model.terminal_initializer.terminal_pose

    def _echo_q(ctx, state, masks):
        out = real(ctx, state, masks).clone()
        out[:, :] = state[:, : A.N_DOF]
        return out

    mpc.model.terminal_initializer.terminal_pose = _echo_q
    mpc.plan(q, qd)
    hand, k = _arm_in_horizon(sched.hit_index())
    cols = arm_slice("left" if hand == 0 else "right")
    torch.testing.assert_close(mpc._frozen_terminal, q)
    torch.testing.assert_close(mpc.init[0, k - 1, cols], q[0, cols])

    q2 = q.clone()
    q2[0, cols[0]] += 0.2
    mpc.plan(q2, qd)
    torch.testing.assert_close(mpc._frozen_terminal, q2)
    torch.testing.assert_close(mpc.init[0, k - 1, cols], q2[0, cols])

    held = mpc.init.clone()
    mpc.commit(q2, q2, torch.zeros(B, 2, dtype=torch.bool))
    torch.testing.assert_close(mpc.init, held)
