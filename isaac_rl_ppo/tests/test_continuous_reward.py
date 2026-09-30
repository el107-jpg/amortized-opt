from types import SimpleNamespace

import pytest
import torch

from isaac_rl_ppo.reward import ContactScore, ExecutedReward, RewardConfig


class _Scorer:
    dt = 0.02
    device = torch.device("cpu")

    def __init__(self, hit_error, speed, hits_any=None):
        self.hit_error = torch.as_tensor(hit_error, dtype=torch.float32)
        self.speed = torch.as_tensor(speed, dtype=torch.float32)
        self.hits_any = hits_any

    def score(self, q0, q1, ctx):
        shape = self.hit_error.shape
        return ContactScore(
            crossed=torch.zeros(shape, dtype=torch.bool),
            in_disk=torch.zeros(shape, dtype=torch.bool),
            frac=torch.zeros(shape),
            speed=self.speed,
            axial_m=torch.zeros(shape),
            inplane_m=torch.zeros(shape),
            hit_error_m=self.hit_error,
            drum=torch.zeros(shape, dtype=torch.long),
        )

    def any_drum_contact(self, *args, **kwargs):
        return self.hits_any


class _Cost:
    def __init__(self, collision):
        self.collision = float(collision)

    def __call__(self, q0, traj, v0, ctx, **kwargs):
        value = q0.new_full((q0.shape[0],), self.collision)
        return SimpleNamespace(
            total=value,
            terms={"collision": value},
            violations={},
        )


def _step(hit_error, speed, *, fired=True, collision=None):
    scorer = _Scorer([[hit_error, hit_error]], [[speed, speed]])
    reward = ExecutedReward(
        RewardConfig(), scorer=scorer,
        cost=None if collision is None else _Cost(collision), device="cpu",
    )
    q = torch.zeros(1, 17)
    ctx = SimpleNamespace(
        active=torch.tensor([[1.0, 0.0]]),
        v_star=torch.tensor([[[0.0, 0.0, -2.0], [0.0, 0.0, 0.0]]]),
        hit_index=torch.tensor([[1, -1]]),
    )
    traj = None if collision is None else torch.zeros(1, 4, 17)
    out, _ = reward.step(
        q_before=q, q_after=q, qd_after=q, ctx=ctx,
        hit_index=ctx.hit_index,
        fired=torch.tensor([[fired, False]], dtype=torch.bool),
        ref0=q, traj=traj,
    )
    return out


def test_hit_reward_is_continuous_and_monotonic_in_probe_error():
    at_target = _step(0.0, 2.0)
    at_scale = _step(0.05, 2.0)
    far = _step(0.10, 2.0)

    assert at_target.terms["hit_error"].item() == pytest.approx(10.0)
    assert at_scale.terms["hit_error"].item() == pytest.approx(5.0)
    assert far.terms["hit_error"].item() == pytest.approx(2.0)
    assert "hit" not in at_target.terms
    assert "miss" not in at_target.terms
    assert "track" not in at_target.terms


def test_speed_reward_is_continuous_and_gated_by_hit_quality():
    exact = _step(0.0, 2.0)
    half_speed_quality = _step(0.0, 2.5)
    same_speed_bad_geometry = _step(0.05, 2.0)

    assert exact.terms["impact_speed"].item() == pytest.approx(0.5)
    assert half_speed_quality.terms["impact_speed"].item() == pytest.approx(0.25)
    assert same_speed_bad_geometry.terms["impact_speed"].item() == pytest.approx(0.25)


def test_contact_is_scored_at_firing_and_collision_is_continuous():
    due = _step(0.05, 2.0, fired=True, collision=100.0)
    not_due = _step(0.05, 2.0, fired=False, collision=100.0)

    assert due.terms["hit_error"].item() == pytest.approx(5.0)
    assert not_due.terms["hit_error"].item() == 0.0
    assert due.terms["collision_avoidance"].item() == pytest.approx(-0.1)
    assert set(due.terms) == {
        "hit_error", "impact_speed", "collision_avoidance", "filter", "alive",
    }


def test_binary_mode_rewards_goal_contact_and_penalizes_wrong_drum_only():
    # Left hand contacts its goal (drum 1); right hand contacts drum 0 while
    # its goal is drum 2.  Both contacts occur inside the timing window.
    hits = torch.zeros(1, 2, 3, dtype=torch.bool)
    hits[0, 0, 1] = True
    hits[0, 1, 0] = True
    scorer = _Scorer([[0.0, 0.0]], [[2.0, 2.0]], hits_any=hits)
    reward = ExecutedReward(
        RewardConfig(reward_mode="binary_hits", w_correct_hit=3.0,
                     w_wrong_hit=2.0),
        scorer=scorer, cost=_Cost(1000.0), device="cpu",
    )
    q = torch.zeros(1, 17)
    ctx = SimpleNamespace(
        active=torch.ones(1, 2),
        v_star=torch.zeros(1, 2, 3),
        target_drum=torch.tensor([[1, 2]]),
        contact_sigma=torch.zeros(1, 2),
        shaft_allowed=torch.zeros(1, 2),
        hit_index=torch.tensor([[2, 2]]),
    )
    out, _ = reward.step(
        q_before=q, q_after=q, qd_after=q, ctx=ctx,
        hit_index=ctx.hit_index, fired=torch.zeros(1, 2, dtype=torch.bool),
        ref0=q, traj=torch.zeros(1, 4, 17),
        drum_heads=torch.zeros(1, 3, 3), drum_axes=torch.zeros(1, 3, 3),
        drum_radii=torch.ones(1, 3), drum_valid=torch.ones(1, 3,
                                                          dtype=torch.bool),
    )
    assert out.terms["correct_hit"].item() == pytest.approx(3.0)
    assert out.terms["wrong_hit"].item() == pytest.approx(-2.0)
    assert out.total.item() == pytest.approx(1.0)
    # The binary objective suppresses the analytic reward even if a cost was
    # supplied; the PPO value target therefore contains contacts only.
    assert out.terms["collision_avoidance"].item() == 0.0
