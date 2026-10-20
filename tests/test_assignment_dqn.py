"""Temporal Bellman credit, warm-start migration and environment isolation."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import amortized_opt as A
from isaac_rl.arm_assignment import AssignmentConfig, AssignmentPolicy, AssignmentQueue
from isaac_rl.assignment_dqn import AssignmentQPolicy, DoubleDQN, Replay
from isaac_rl.assignment_parallel import ParallelCommittedMPC
from isaac_rl.assignment_parallel_train import load_with_tempo
from isaac_rl.mpc import MPCConfig
from isaac_rl.schedule import BatchedSchedule, ScheduleConfig


def test_expanded_lookahead_retains_learned_arm_preferences_and_value():
    torch.manual_seed(23)
    old = AssignmentPolicy(AssignmentConfig())
    obs = torch.randn(128, old.cfg.obs_dim)
    new = AssignmentQPolicy.from_ppo(dict(policy=old.state_dict(), policy_config=vars(old.cfg)))
    expanded = torch.cat((obs, torch.randn(128, new.cfg.obs_dim-old.cfg.obs_dim)), 1)
    dist, value = old(obs)
    q = new(expanded)
    assert torch.equal(q.argmax(1), dist.probs.argmax(1))
    torch.testing.assert_close(q.mean(1), value)
    assert new.cfg.lookahead_beats == 8 and new.cfg.future_hits == 32


def test_double_dqn_selects_online_action_and_respects_terminal_bootstrap():
    policy = AssignmentQPolicy(AssignmentConfig(hidden=4))
    for p in policy.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        policy.q_head.bias.copy_(torch.tensor([1., 3.]))
    learner = DoubleDQN(policy, lr=0, tau=0)
    with torch.no_grad():
        learner.target.q_head.bias.copy_(torch.tensor([100., 7.]))
    obs = torch.zeros(2, policy.cfg.obs_dim)
    # Online selects action 1, whose target value is 7 (not target max 100).
    # Terminal transition ignores both target values.
    replay = SimpleNamespace(size=2, sample=lambda b, d: (
        obs, torch.tensor([0, 0]), torch.tensor([2., 5.]), torch.tensor([.5, 0.]), obs))
    metrics = learner.update(replay, gradient_steps=1, batch=2)
    assert metrics["value_bellman_loss"] == pytest.approx(3.75)
    assert metrics["td_error"] == pytest.approx(4.25)


def test_replay_does_not_connect_episodes_and_resumes_after_wrap():
    replay = Replay(2, capacity=3, seed=10)
    def row(x):
        return dict(obs=np.full(2, x, np.float32), action=0, reward=x, discount=.8)
    replay.add_episode([row(1), row(2)])
    assert replay.discount[1] == 0 and np.all(replay.next_obs[1] == 0)
    replay.add_episode([row(3), row(4)])
    restored = Replay(2, capacity=3)
    restored.load_state_dict(replay.state_dict())
    assert restored.cursor == replay.cursor == 1
    for a, b in zip(replay.sample(8, "cpu"), restored.sample(8, "cpu")):
        torch.testing.assert_close(a, b)
    restored.add_episode([row(5)])
    assert restored.reward[1] == 5 and restored.discount[1] == 0


def test_beat_lookahead_scales_with_tempo_and_future_events_are_visible(tmp_path):
    import json
    path = A.default_rcc_dir("eval")
    src = sorted(path.glob("*.json"))[0]
    data = json.loads(src.read_text())
    data.setdefault("metadata", {})["original_bpm"] = 80
    dest = tmp_path/"clip.json"
    dest.write_text(json.dumps(data))
    kit = A.DrumKit.from_layout()
    cfg = AssignmentConfig(future_hits=32, lookahead_beats=8)
    clip, slow = load_with_tempo(dest, kit, cfg)
    assert slow.lookahead_s == 6
    data["metadata"]["original_bpm"] = 160
    dest.write_text(json.dumps(data))
    _, fast = load_with_tempo(dest, kit, cfg)
    assert fast.lookahead_s == 3
    queue = AssignmentQueue(clip, kit, slow)
    observation = queue.observation(0, np.zeros(17), np.zeros(17), np.zeros((2, 3)))
    future = observation[-32*(kit.num_drums+3):].reshape(32, -1)
    assert future[:, -1].sum() == min(32, len(clip.hits))


@dataclass
class _Encoding:
    memory: torch.Tensor


class _Decoder(torch.nn.Module):
    terminal_initializer = True

    def __init__(self):
        super().__init__()
        self.batches = []

    def initial_trajectory(self, ctx, state, masks):
        return state[:, None, :17].expand(-1, masks.elem_mask.shape[1], -1).clone()

    def refine(self, init, state, enc, masks, iterations):
        self.batches.append(len(state))
        return SimpleNamespace(traj=(init+.01)*masks.elem_mask)


def test_new_goal_in_one_environment_preserves_other_committed_plan():
    kit = A.DrumKit.from_layout()
    base = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[0]
    h = base.hits[0]
    clips = [replace(base, hits=[replace(h, time=.1), replace(h, time=.2)]),
             replace(base, hits=[replace(h, time=.3)])]
    sched = BatchedSchedule(2, ScheduleConfig(horizon=15), device="cpu")
    sched.reset(torch.arange(2), clips, [kit]*2, rng=None)
    decoder = _Decoder()
    cost = lambda q, *a, **kw: SimpleNamespace(violations={"collision": torch.zeros(len(q))})
    mpc = ParallelCommittedMPC(decoder, sched, MPCConfig(horizon=15, cmd_lead_s=None,
                                cmd_slew_eta=None, amp_dtype=None, fresh_terminal_every_tick=True),
                                device="cpu", descent_cost=cost)
    q = torch.zeros(2, 17)
    mpc.reset(torch.arange(2), q)
    def encoding(hit_index, state):
        mpc._ctx = sched.context(yaw_now=torch.zeros(2))
        return _Encoding(torch.zeros(2, 1)), True
    mpc._encoding = encoding
    mpc.plan(q, q)
    assert decoder.batches == [2]
    for _ in range(5):
        sched.advance()
        mpc.commit(q)
    before = mpc.reference[1].clone()
    mpc.plan(q+.2, q)
    assert decoder.batches == [2, 1]
    assert torch.equal(mpc.reference[1], before)
    assert mpc.decoder_passes.tolist() == [2, 1]


def _queue_clip(kit, times):
    names = ["snare", "hihat", "tom1", "floor_tom", "snare", "cymbal_right", "tom2", "tom1", "hihat"]
    return A.RhythmClip("queue", [A.Hit(t, t, kit.index(names[i%len(names)]), names[i%len(names)],
                                     "left", .8) for i, t in enumerate(times)], max(times)+.6)


def test_lookahead_is_visible_without_early_commitment():
    kit = A.DrumKit.from_layout()
    cfg = AssignmentConfig(future_hits=32, pending_per_arm=51, lookahead_s=4,
                           lookahead_beats=8, commit_lead_s=1)
    queue = AssignmentQueue(_queue_clip(kit, [1, 3, 5]), kit, cfg)
    q = np.zeros(17)
    obs = queue.observation(0, q, q, np.zeros((2, 3)))
    future = obs[-32*(kit.num_drums+3):].reshape(32, -1)
    assert future[:, -1].sum() == 2  # Includes the uncommitted 3-second hit.
    assert queue.due(0)
    queue.commit(0)
    assert queue.hands == {0: "left"}
    assert not queue.due(1.98) and queue.due(2)


def test_full_pending_state_distinguishes_previously_aliased_queues():
    kit = A.DrumKit.from_layout()
    clip = _queue_clip(kit, [1+i*.1 for i in range(9)])
    def pair(cfg):
        a, b = AssignmentQueue(clip, kit, cfg), AssignmentQueue(clip, kit, cfg)
        for x, y in zip([0, 0, 1, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1]):
            a.commit(x); b.commit(y)
        q, tips = np.zeros(17), np.zeros((2, 3))
        return a.observation(.8, q, q, tips), b.observation(.8, q, q, tips)
    legacy = AssignmentConfig(future_hits=32, lookahead_s=4)
    a, b = pair(legacy)
    assert np.array_equal(a, b)
    a, b = pair(replace(legacy, pending_per_arm=51, commit_lead_s=1))
    assert not np.array_equal(a, b)


def test_pending_capacity_never_silently_hides_assignments():
    kit = A.DrumKit.from_layout()
    queue = AssignmentQueue(_queue_clip(kit, [1+i*.02 for i in range(5)]), kit,
                            AssignmentConfig(commit_lead_s=1, pending_per_arm=2))
    for _ in range(3):
        queue.commit(0)
    with pytest.raises(ValueError, match="refusing to hide"):
        queue.observation(.1, np.zeros(17), np.zeros(17), np.zeros((2, 3)))


def test_queue_migration_keeps_semantic_weights_and_q_values():
    torch.manual_seed(11)
    old = AssignmentQPolicy(AssignmentConfig(future_hits=32, lookahead_beats=8))
    new = old.with_queue_config(commit_lead_s=1, pending_per_arm=51)
    obs = torch.randn(64, old.cfg.obs_dim)
    expanded = torch.randn(64, new.cfg.obs_dim)
    width, prefix = old.cfg.num_drums+3, 40+7*old.cfg.num_drums+2*(old.cfg.num_drums+2)
    expanded[:, :prefix] = obs[:, :prefix]
    for hand in (0, 1):
        a, b = prefix+hand*2*width, prefix+hand*51*width
        expanded[:, b:b+2*width] = obs[:, a:a+2*width]
    expanded[:, prefix+102*width:] = obs[:, prefix+4*width:]
    torch.testing.assert_close(new(expanded), old(obs), atol=1e-6, rtol=1e-5)
