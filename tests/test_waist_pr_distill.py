"""Gated strike-knot P/R distill: encoder copies decoder waist only on good plans.

The 0.15 band is the decoder's exploration window.  This module is the
encoder teacher: ``‖sp - stopgrad(q_pr[hit])‖²`` on the conjunction
(active, on-head, in-box, no collision, in-band, |q_pr| in URDF).
Upright ``stopgrad(sp)`` is tested in ``test_waist_pr_band_al`` /
``test_waist_pr_sp_head``.
"""
from __future__ import annotations

import pytest
import torch

from amortized_opt.constants import N_DOF
from amortized_opt.train_cost import (
    PR_DISTILL_GATE_KEYS,
    waist_pr_distill_gate,
    waist_pr_distill_loss,
    waist_pr_hit_teacher,
)

B, H, N = 4, 10, N_DOF
PR_IDX = [0, 1]  # synthetic: first two columns are pitch/roll


def _traj(*, pitch, roll, dump_pitch=None, dump_knot=5):
    q = torch.zeros(B, H, N, dtype=torch.float64)
    q[:, :, 0] = float(pitch)
    q[:, :, 1] = float(roll)
    if dump_pitch is not None:
        q[:, int(dump_knot), 0] = float(dump_pitch)
    return q


def _hit(k=H - 1, active=(1.0, 1.0)):
    idx = torch.full((B, 2), int(k), dtype=torch.long)
    act = torch.tensor(active, dtype=torch.float64).view(1, 2).expand(B, 2).contiguous()
    return idx, act


def _ok_gate_inputs(q_hit, active, *, region=0.0, box=0.0, coll=0.0, band=0.0):
    z = q_hit.new_full((B,), 0.0)
    return waist_pr_distill_gate(
        z + region, z + box, z + coll, z + band, q_hit, active,
        pr_limit=0.52, hit_m=0.02,
    )


# =========================================================== 1. teacher =====
def test_teacher_is_strike_knot_mean_of_active_hands():
    q = _traj(pitch=0.0, roll=0.0)
    q[:, 3, 0] = 0.20
    q[:, 7, 0] = 0.40
    hit = torch.tensor([[3, 7]] * B, dtype=torch.long)
    act = torch.ones(B, 2, dtype=torch.float64)
    got = waist_pr_hit_teacher(q, hit, act, PR_IDX)
    assert got.shape == (B, 2)
    assert torch.allclose(got[:, 0], torch.full((B,), 0.30, dtype=torch.float64))
    assert torch.allclose(got[:, 1], torch.zeros(B, dtype=torch.float64))
    assert not got.requires_grad


def test_idle_hand_does_not_vote():
    q = _traj(pitch=0.0, roll=0.0)
    q[:, 3, 0] = 0.20
    q[:, 7, 0] = 0.40
    hit = torch.tensor([[3, 7]] * B, dtype=torch.long)
    act = torch.tensor([[1.0, 0.0]] * B, dtype=torch.float64)
    got = waist_pr_hit_teacher(q, hit, act, PR_IDX)
    assert torch.allclose(got[:, 0], torch.full((B,), 0.20, dtype=torch.float64))


def test_teacher_detaches_decoder_traj():
    q = _traj(pitch=0.10, roll=0.0).requires_grad_(True)
    hit, act = _hit()
    t = waist_pr_hit_teacher(q, hit, act, PR_IDX)
    assert not t.requires_grad
    assert t.grad_fn is None


# ============================================================== 2. gate =====
def test_good_plan_passes_and_dump_or_miss_fails():
    q = _traj(pitch=0.10, roll=0.0)
    hit, act = _hit()
    q_hit = waist_pr_hit_teacher(q, hit, act, PR_IDX)
    ok = _ok_gate_inputs(q_hit, act)
    assert torch.equal(ok, torch.ones(B, dtype=torch.float64))

    dump = _ok_gate_inputs(q_hit, act, band=1.75)
    assert torch.equal(dump, torch.zeros(B, dtype=torch.float64))
    miss = _ok_gate_inputs(q_hit, act, region=0.05)
    assert torch.equal(miss, torch.zeros(B, dtype=torch.float64))
    oob = _ok_gate_inputs(q_hit + 2.0, act)
    assert torch.equal(oob, torch.zeros(B, dtype=torch.float64))
    idle = _ok_gate_inputs(q_hit, torch.zeros_like(act))
    assert torch.equal(idle, torch.zeros(B, dtype=torch.float64))


def test_hit_margin_matches_frontier_default():
    q = _traj(pitch=0.10, roll=0.0)
    hit, act = _hit()
    q_hit = waist_pr_hit_teacher(q, hit, act, PR_IDX)
    # 15 mm is still on-head at the 2 cm grid margin; 25 mm is not
    inside = _ok_gate_inputs(q_hit, act, region=0.015)
    assert torch.equal(inside, torch.ones(B, dtype=torch.float64))
    outside = _ok_gate_inputs(q_hit, act, region=0.025)
    assert torch.equal(outside, torch.zeros(B, dtype=torch.float64))


# ============================================================== 3. loss =====
def test_loss_zero_when_gate_silent_nonzero_when_open():
    sp = torch.zeros(B, 2, dtype=torch.float64, requires_grad=True)
    q_hit = torch.full((B, 2), 0.10, dtype=torch.float64)
    silent = torch.zeros(B, dtype=torch.float64)
    z = waist_pr_distill_loss(sp, q_hit, silent)
    assert float(z) == 0.0
    z.backward()
    assert float(sp.grad.abs().max()) == 0.0

    sp2 = torch.zeros(B, 2, dtype=torch.float64, requires_grad=True)
    open_m = torch.ones(B, dtype=torch.float64)
    loss = waist_pr_distill_loss(sp2, q_hit, open_m)
    assert float(loss) == pytest.approx(0.01)
    loss.backward()
    assert float(sp2.grad.abs().max()) > 0.0


def test_loss_does_not_train_the_decoder():
    sp = torch.zeros(B, 2, dtype=torch.float64, requires_grad=True)
    q = _traj(pitch=0.10, roll=0.0).requires_grad_(True)
    hit, act = _hit()
    q_hit = waist_pr_hit_teacher(q, hit, act, PR_IDX)
    mask = torch.ones(B, dtype=torch.float64)
    waist_pr_distill_loss(sp, q_hit, mask).backward()
    assert q.grad is None or float(q.grad.abs().max()) == 0.0
    assert float(sp.grad.abs().max()) > 0.0


def test_gated_mean_does_not_dilute_with_silent_rows():
    sp = torch.zeros(B, 2, dtype=torch.float64)
    q_hit = torch.full((B, 2), 0.10, dtype=torch.float64)
    mask = torch.zeros(B, dtype=torch.float64)
    mask[0] = 1.0
    loss = waist_pr_distill_loss(sp, q_hit, mask)
    # one gated row, not 1/B of the batch
    assert float(loss) == pytest.approx(0.01)


def test_gate_key_order_matches_cost_tuple_stack():
    assert PR_DISTILL_GATE_KEYS == (
        "terminal_region_dist",
        "joint_box_excess_max",
        "collision_pen_max",
        "waist_pr_band",
    )
