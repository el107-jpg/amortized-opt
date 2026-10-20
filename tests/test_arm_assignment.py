"""Assignment freedom, temporal credit and real PPO learning checks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from isaac_rl.arm_assignment import (
    AssignmentConfig, AssignmentPolicy, AssignmentQueue, decision_gae, ppo_update,
)
from isaac_rl.assignment_train import install_beats, sync_credit_hands, window_clip


def make_clip(kit, events):
    return RhythmClip("test", [Hit(t, t, kit.index(d), d, "right", 0.8) for t, d in events], 4.0)


def test_floor_left_hihat_right_and_immutable_simultaneous_pair():
    kit = DrumKit.from_layout()
    queue = AssignmentQueue(make_clip(kit, [(1, "floor_tom"), (1, "hihat"), (2, "hihat")]),
                            kit, AssignmentConfig())
    # Canonical drum order places hihat before floor tom: action one puts
    # hihat on the right AND floor tom on the left. Neither is eligibility filtered.
    first = queue.commit(1)
    assert {h.drum_name: h.hand for _, h in first} == {"hihat": "right", "floor_tom": "left"}
    queue.commit(0)
    assert queue.hands[0] == "right" and queue.hands[1] == "left"
    assert len(queue.assigned_clip().hits) == 3


def test_reject_three_hits_without_dropping_one():
    kit = DrumKit.from_layout()
    with pytest.raises(ValueError, match="more than two"):
        AssignmentQueue(make_clip(kit, [(1, "tom1"), (1, "hihat"), (1, "floor_tom")]),
                        kit, AssignmentConfig())


def test_observation_exposes_future_and_pending_deadlines():
    kit = DrumKit.from_layout()
    queue = AssignmentQueue(make_clip(kit, [(1, "tom1"), (1.04, "floor_tom")]), kit, AssignmentConfig())
    q = ready_joint_pos(G1_UPPER_BODY_JOINTS)
    before = queue.observation(0, q, np.zeros(17), np.zeros((2, 3)))
    queue.commit(0)
    after = queue.observation(0, q, np.zeros(17), np.zeros((2, 3)))
    assert before.shape == after.shape == (queue.cfg.obs_dim,)
    assert not np.array_equal(before, after)
    assert queue.hands[0] == "left"


def test_isaac_credit_receives_unrestricted_hands():
    kit = DrumKit.from_layout()
    queue = AssignmentQueue(make_clip(kit, [(1, "hihat"), (1.04, "floor_tom")]), kit, AssignmentConfig())
    d, b = kit.num_drums, 8
    u = SimpleNamespace(device="cpu", beat_valid=torch.zeros(1, b, dtype=torch.bool),
                        beat_time=torch.zeros(1, b), beat_hand=torch.zeros(1, b, dtype=torch.long),
                        beat_drum_id=torch.zeros(1, b, dtype=torch.long), beat_velocity=torch.zeros(1, b),
                        beat_pos=torch.zeros(1, b, 3), beat_drum_height=torch.zeros(1, b),
                        beat_dipole_offset=torch.zeros(1, b, 3), drum_pos_env=torch.tensor(kit.positions).unsqueeze(0),
                        drum_height=torch.tensor(kit.heights), drum_dipole_offset=torch.zeros(d, 3),
                        beat_is_left=torch.zeros(1, b, dtype=torch.bool), beat_is_right=torch.zeros(1, b, dtype=torch.bool))
    install_beats(u, queue)
    assert not u.beat_valid.any()
    queue.commit(1)  # right hihat
    queue.commit(0)  # left floor tom
    sync_credit_hands(u, queue)
    assert u.beat_hand[0, :2].tolist() == [1, 0]
    assert u.beat_valid.sum() == 2
    assert u.beat_is_right[0, 0] and u.beat_is_left[0, 1]


def test_bellman_returns_respect_elapsed_time_and_terminal_boundary():
    returns, _ = decision_gae([0, 0, 10], [2, 3, 4], [1, 0.5, 0], lam=1)
    torch.testing.assert_close(returns, torch.tensor([5., 5., 10.]))
    other, _ = decision_gae([7], [100], [0], lam=1)
    torch.testing.assert_close(other, torch.tensor([7.]))


def test_ppo_changes_arm_preference_from_executed_reward():
    torch.manual_seed(5)
    policy = AssignmentPolicy(AssignmentConfig(hidden=16))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    obs = np.zeros(policy.cfg.obs_dim, np.float32)
    with torch.no_grad():
        initial = float(policy(torch.tensor(obs))[0].probs[1])
    for _ in range(4):
        episodes = []
        with torch.no_grad():
            dist, val = policy(torch.tensor(obs))
            for a in [0, 1]*16:
                episodes.append([dict(obs=obs, action=a, log_prob=float(dist.log_prob(torch.tensor(a))),
                                      value=float(val), reward=1. if a else -1., discount=0.)])
        metrics = ppo_update(policy, optimizer, episodes, entropy_weight=0.02)
    assert float(policy(torch.tensor(obs))[0].probs[1]) > initial+0.1
    assert np.isfinite(metrics["value_bellman_loss"])


def test_training_windows_preserve_gaps_and_leave_a_scoring_tail():
    kit = DrumKit.from_layout()
    clip = make_clip(kit, [(0.4, "tom1"), (0.44, "floor_tom")])
    window = window_clip(clip, np.random.default_rng(0), span_s=5, tail_s=0.6)
    assert window.hits[0].time == 1
    assert window.hits[1].time-window.hits[0].time == pytest.approx(0.04)
    assert window.duration-window.hits[-1].time == pytest.approx(0.6)
