"""Play-isaac start pose and first-hit pad, without Isaac."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from amortized_opt.constants import G1_UPPER_BODY_JOINTS, ready_joint_pos
from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl_ppo.cfg import PpoRunConfig
from isaac_rl_ppo.start import (
    amortized_ready_q,
    describe_clip_pad,
    ensure_clip_gap,
    install_duration_done,
    install_train_ready,
    pad_clip_pool,
    reset_envs_to_ready,
)
from isaac_rl_ppo.train_ppo import build_cfg, parse as train_parse


def _clip(t0: float = 0.14) -> RhythmClip:
    return RhythmClip(
        name="toy",
        hits=[
            Hit(time=t0, raw_time=t0, drum=0, drum_name="snare",
                hand="left", velocity=1.0),
            Hit(time=t0 + 0.40, raw_time=t0 + 0.40, drum=0, drum_name="snare",
                hand="right", velocity=1.0),
        ],
        duration=2.0,
    )


class _FakeEnv:
    def __init__(self, n: int = 4, n_joints: int = 30):
        self.device = torch.device("cpu")
        self.num_envs = n
        self.joint_ids = list(range(17))
        self.ready_q = torch.full((n, n_joints), 0.3)
        self.joint_pos_target = self.ready_q.clone()
        self.reset_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.read_q = torch.zeros(n, 17)

    def reset_to(self, env_ids, q, qd):
        self.reset_calls.append((env_ids.clone(), q.clone()))
        self.read_q[env_ids] = q

    def read_state(self):
        return self.read_q.clone(), torch.zeros_like(self.read_q)


def test_amortized_ready_is_training_pose_not_unitree_standing():
    q = amortized_ready_q("cpu", torch.float32)
    want = torch.as_tensor(ready_joint_pos(G1_UPPER_BODY_JOINTS), dtype=torch.float32)
    assert torch.allclose(q, want)
    ip = G1_UPPER_BODY_JOINTS.index("left_shoulder_pitch_joint")
    assert float(q[ip]) < -0.5, "ready pitch should be ~-1.20, not Unitree +0.3"


def test_pad_clip_pool_puts_first_hit_at_one_second():
    out, stats = pad_clip_pool([_clip(0.14)], 1.0)
    assert stats["enabled"] and stats["n_shifted"] == 1
    times = sorted(h.time for h in out[0].hits)
    assert times[0] == pytest.approx(1.0, abs=0.02)
    assert times[1] - times[0] == pytest.approx(0.40, abs=0.02)


def test_pad_clip_pool_off_at_zero():
    clip = _clip(0.14)
    out, stats = pad_clip_pool([clip], 0.0)
    assert not stats["enabled"]
    assert min(h.time for h in out[0].hits) == clip.hits[0].time
    assert "OFF" in describe_clip_pad(stats)


def test_install_train_ready_teleports_every_env():
    env = _FakeEnv()
    install_train_ready(env)
    want = amortized_ready_q("cpu", env.ready_q.dtype)
    assert torch.allclose(env.ready_q[:, :17], want.unsqueeze(0).expand(4, -1))
    assert len(env.reset_calls) == 1
    ids, q = env.reset_calls[0]
    assert ids.tolist() == [0, 1, 2, 3]
    assert torch.allclose(q, want.unsqueeze(0).expand(4, -1))


def test_reset_envs_to_ready_uses_the_ready_buffer():
    env = _FakeEnv()
    env.ready_q[:, :17] = amortized_ready_q("cpu", env.ready_q.dtype)
    reset_envs_to_ready(env, torch.tensor([1, 3]))
    ids, q = env.reset_calls[-1]
    assert ids.tolist() == [1, 3]
    assert q.shape == (2, 17)
    assert torch.allclose(q[0], amortized_ready_q("cpu", torch.float32))


def test_ppo_defaults_match_play_isaac_start():
    cfg = PpoRunConfig()
    assert cfg.env.isaac_ready is False
    assert cfg.env.first_hit_s == 1.0
    assert cfg.env.clip_gap_s == 1.0
    built = build_cfg(train_parse([]))
    assert built.env.isaac_ready is False
    assert built.env.first_hit_s == 1.0
    assert built.env.clip_gap_s == 1.0


def test_cli_can_restore_unitree_start_and_native_clock():
    cfg = build_cfg(train_parse(["--isaac-ready", "--first-hit-s", "0"]))
    assert cfg.env.isaac_ready is True
    assert cfg.env.first_hit_s == 0.0


def test_negative_first_hit_s_is_rejected():
    cfg = PpoRunConfig()
    cfg.env.first_hit_s = -0.1
    with pytest.raises(ValueError, match="first_hit_s"):
        cfg.validate()


def test_on_done_keeps_leftover_q():
    from isaac_rl_ppo.start import make_leftover_on_done

    env = _FakeEnv()
    env.ready_q[:, :17] = amortized_ready_q("cpu", env.ready_q.dtype)

    class _Mpc:
        def __init__(self):
            self.q = None

        def reset(self, ids, q):
            self.q = q.clone()

    mpc = _Mpc()
    sched_ids = []

    def reset_sched(ids):
        sched_ids.append(ids.tolist())

    leftover = torch.arange(4 * 17, dtype=torch.float32).reshape(4, 17)
    q, _qd = make_leftover_on_done(env, reset_sched, mpc)(
        torch.tensor([0, 2]), leftover, leftover,
    )
    assert sched_ids == [[0, 2]]
    assert torch.allclose(q[0], leftover[0])
    assert torch.allclose(q[2], leftover[2])
    assert torch.allclose(mpc.q[0], leftover[0])


def test_ensure_clip_gap_extends_duration_past_last_hit():
    clip = replace(_clip(0.14), duration=0.54)
    last = max(h.time for h in clip.hits)
    out = ensure_clip_gap([clip], 1.0)
    assert out[0].duration == pytest.approx(last + 1.0)
    unchanged = ensure_clip_gap([_clip(0.14)], 1.0)
    assert unchanged[0].duration == pytest.approx(2.0)


def test_duration_done_waits_out_the_gap():
    class _Sched:
        def __init__(self):
            self.t_now = torch.tensor([0.5, 1.5])
            self.duration = torch.tensor([1.5, 1.5])

        def done(self):
            return torch.tensor([True, True])

    sched = _Sched()
    install_duration_done(sched)
    assert sched.done().tolist() == [False, True]


def test_cli_clip_gap_zero():
    cfg = build_cfg(train_parse(["--clip-gap-s", "0"]))
    assert cfg.env.clip_gap_s == 0.0
