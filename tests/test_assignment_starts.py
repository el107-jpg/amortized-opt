"""Randomized starts remain valid and use reproducible measured states."""
import numpy as np
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.cost import CostConfig, FusedCost
from amortized_opt.limits import urdf_position_limits
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance


def test_no_exemptions_and_cross_stick_clearance_are_checked():
    cost = FusedCost(CostConfig(), max_horizon=1, device="cpu", dtype=torch.float32)
    q = torch.tensor(ready_joint_pos()[None], dtype=torch.float32)
    distance = pose_clearance(cost, q)
    assert distance.shape == (1,) and torch.isfinite(distance).all()
    assert distance[0] > .003


def test_mixture_rejects_bad_poses_and_preserves_exact_ready_starts():
    ready = ready_joint_pos().astype(np.float32)
    anchors = np.tile(ready, (3, 1))
    anchors[1, 0] = .3
    anchors[2, 0] = 10  # Invalid joint limit, must never enter anchor pool.
    lo, hi = urdf_position_limits(posture_limits=False)
    def clearance(q):
        return np.where(q[:, 3] < ready[3]+.04, .01, -.01)
    sampler = StartingPoseSampler(anchors, ready, lo, hi, clearance=clearance,
                                 names=G1_UPPER_BODY_JOINTS)
    assert len(sampler.anchors) == 2
    rng = np.random.default_rng(71)
    state = rng.bit_generator.state
    q, stats = sampler.sample(rng, 2048)
    assert sampler.valid(q).all()
    assert .15 < stats["ready_fraction"] < .25
    assert .55 < stats["anchor_fraction"] < .65
    assert stats["rejection_fraction"] > 0
    assert np.any(np.all(q == ready, axis=1))
    assert q[:, 0].std() > .1
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    q2, stats2 = sampler.sample(restored, 2048)
    np.testing.assert_array_equal(q, q2)
    assert stats == stats2


def test_rejection_fallback_is_checked_anchor_not_invalid_candidate():
    ready = ready_joint_pos().astype(np.float32)
    lo, hi = urdf_position_limits(posture_limits=False)
    def clearance(q):
        return np.where((q == ready).all(1), .01, -.01)
    sampler = StartingPoseSampler(ready[None], ready, lo, hi, clearance=clearance,
                                 names=G1_UPPER_BODY_JOINTS, ready_fraction=0)
    q, stats = sampler.sample(np.random.default_rng(0), 8)
    assert stats["fallback_fraction"] == 1
    np.testing.assert_array_equal(q, np.tile(ready, (8, 1)))
