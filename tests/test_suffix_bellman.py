"""The suffix-Bellman term (:func:`amortized_opt.train_cost.make_suffix_step`).

The horizon shrinks to the strike, so the tail of the H-plan and the
(H - k)-solve from the state it passes through price the SAME terminal event --
the principle of optimality holds exactly, and the term is well-posed.  What
these tests pin:

* offsets never move a strike into the past (``suffix_offsets``);
* the shifted plan, its validity mask and the pad convention (``shift_plan``);
* the child's element mask IS the shifted parent mask on the overlap -- the
  property the tie mask relies on;
* a child that echoes its init (== the shifted parent) ties at exactly zero;
* the gradient topology: the tie couples child to parent, the child's COST
  anchor does not reach the parent (the handoff is detached).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import amortized_opt as A
from amortized_opt.constants import CONTROL_DT, G1_UPPER_BODY_JOINTS
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.tokenizer import build_waypoint_masks
from amortized_opt.train_cost import (
    Phase2Config,
    _strike_rows,
    effective_hits,
    group_lengths,
    make_suffix_step,
    shift_plan,
    suffix_offsets,
)

H = 20
NAMES = G1_UPPER_BODY_JOINTS


def _masks(hit, active):
    """``Phase2Pool.masks_for`` without a pool."""
    t = hit.clamp_min(0).to(torch.float32) * CONTROL_DT
    return build_waypoint_masks(
        t[:, 0], t[:, 1], max_horizon=H, dt=CONTROL_DT,
        active_left=active[:, 0], active_right=active[:, 1],
        joint_names=NAMES,
    )


@pytest.fixture(scope="module")
def scene():
    torch.manual_seed(0)
    cost = FusedCost(CostConfig(), max_horizon=H, device="cpu",
                     dtype=torch.float32)
    scorer = A.TrajectoryScorer(device="cpu")
    case = next(c for c in A.generate_grid(seed=0, grid="standard",
                                           scorer=scorer) if c.drum == "snare")
    strike = A.strike_context_from_cases([case] * 4, drum_index=cost.drum_index,
                                         device="cpu", dtype=torch.float32)
    return cost, strike


def _parent(n=17):
    """4 rows: left strikes at 12 / 5 / 1, row 3 all-idle."""
    hit = torch.tensor([[12, -1], [5, -1], [1, -1], [-1, -1]])
    active = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 0]], dtype=torch.bool)
    q0 = torch.zeros(4, n)
    v0 = torch.zeros(4, n)
    torch.manual_seed(1)
    drift = 0.05 * torch.randn(4, 1, n)
    traj = q0.unsqueeze(1) + torch.linspace(0.05, 1.0, H).view(1, H, 1) * drift
    return hit, active, q0, v0, traj


def test_suffix_offsets_keep_strikes_ahead():
    torch.manual_seed(2)
    hit, active, *_ = _parent()
    hit = hit.repeat(64, 1)
    active = active.repeat(64, 1)
    offs = suffix_offsets(hit, active, H)
    assert offs.shape == (256,) and (offs >= 0).all()
    # every ACTIVE hand's shifted strike is still ahead (index >= 1)...
    shifted_hit = hit - offs.unsqueeze(1)
    assert (shifted_hit[active] >= 1).all()
    # ...an imminent strike (hit 1) forces k = 0, all-idle rows roam the grid
    assert (offs[2::4] == 0).all()
    assert (offs[3::4] <= H - 1).all() and offs[3::4].max() > 0


def test_shift_plan_matches_manual():
    torch.manual_seed(3)
    traj = torch.randn(3, 6, 2)
    offs = torch.tensor([0, 2, 5])
    shifted, valid = shift_plan(traj, offs)
    for b, k in enumerate(offs.tolist()):
        for j in range(6):
            src = min(j + k, 5)
            assert torch.equal(shifted[b, j], traj[b, src])
            assert valid[b, j].item() == (j < 6 - k)


def test_child_mask_is_shifted_parent_mask_on_overlap():
    hit, active, *_ = _parent()
    offs = torch.tensor([3, 2, 0, 4])
    hit_sf = torch.where(active, hit - offs.unsqueeze(1), hit)
    em_parent = _masks(hit, active).elem_mask
    em_child = _masks(hit_sf, active).elem_mask
    em_shift, valid = shift_plan(em_parent, offs)
    v = valid.unsqueeze(-1)
    assert torch.equal(em_child * v, em_shift * v)


def _run_fn(scene, delta: float):
    cost, strike = scene
    cfg = Phase2Config(compile_model=False)
    hit, active, q0, v0, traj = _parent()
    traj = traj.clone().requires_grad_(True)
    offs = torch.tensor([3, 2, 0, 4])
    hit_sf = torch.where(active, hit - offs.unsqueeze(1), hit)
    masks_p = _masks(hit, active)
    masks_s = _masks(hit_sf, active)
    hit_eff = effective_hits(hit_sf, active, H)
    strike_s = replace(_strike_rows(strike, slice(0, 4)), hit_index=hit_eff,
                       active=active.to(torch.float32), waist_yaw_ref=None)
    stub = lambda init, state, enc, masks: SimpleNamespace(traj=init + delta)
    fn = make_suffix_step(stub, cost, cfg, compile_mode=None)
    loss_cost, tie, gate, q0_s, v0_s, init_s = fn(
        traj, q0, v0, None, masks_p.elem_mask, masks_s,
        strike_s, group_lengths(hit_eff), None, offs,
    )
    assert gate.shape == (traj.shape[0], 8)
    assert q0_s.shape == q0.shape
    assert v0_s.shape == v0.shape
    assert init_s.shape == traj.shape
    assert not gate.requires_grad
    return loss_cost, tie, traj


def test_echo_child_ties_at_zero(scene):
    loss_cost, tie, _ = _run_fn(scene, delta=0.0)
    assert torch.isfinite(loss_cost)
    assert tie.item() == pytest.approx(0.0, abs=1e-12)


def test_gradient_topology(scene):
    """Tie couples child<->parent; the cost anchor must NOT reach the parent."""
    loss_cost, tie, traj = _run_fn(scene, delta=0.01)
    assert tie.item() > 0.0
    # The stub has no parameters, so the ONLY possible path from the child's
    # cost back to the graph is through its init/state handoff -- which is
    # detached.  A requires-grad cost here would mean the child drags the
    # parent through its inputs.
    assert not loss_cost.requires_grad
    g_tie = torch.autograd.grad(tie, traj, allow_unused=True)[0]
    assert g_tie is not None and g_tie.abs().sum() > 0.0
