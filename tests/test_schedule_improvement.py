from dataclasses import replace

import numpy as np
import torch

from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from isaac_rl.schedule_improvement import (
    EDITS, JointScheduleCritic, ScheduleNetConfig, cap_window, physical_table,
    discounted_returns, neighbors, valid_edits, teacher_edits, critic_search,
)


def sample_clip():
    kit = DrumKit.from_layout()
    return RhythmClip('test', [Hit(.1, .1, 0, kit.names[0], 'left', .5),
                              Hit(.2, .2, 1, kit.names[1], 'left', .5),
                              Hit(.2, .2, 2, kit.names[2], 'right', .5)], .2), kit


def sample_obs():
    clip, kit = sample_clip()
    table = physical_table([clip], kit, 'cpu')
    obs = {k: table[k] for k in ('events', 'hit_valid', 'group', 'member')}
    obs['state'] = torch.randn(1, 40)
    return obs


def test_cap_preserves_simultaneous_pair_and_does_not_hide_overlapping_requests():
    clip, kit = sample_clip()
    capped = cap_window(clip, max_hits=2)
    assert len(capped.hits) == 1
    assert len(cap_window(clip, max_hits=3).hits) == 3
    invalid = replace(clip, hits=clip.hits+[Hit(.2, .2, 3, kit.names[3], 'left', .5)])
    import pytest
    with pytest.raises(ValueError, match='at most two'):
        cap_window(invalid)


def test_discounted_plan_returns_obey_event_bellman_equation():
    score = torch.tensor([[2., 4., 0.]])
    times = torch.tensor([[.2, .7, .7]])
    live = torch.tensor([[True, True, False]])
    ret, reward, discount = discounted_returns(score, times, live, gamma=.9, scale=1)
    torch.testing.assert_close(ret[:, 0], reward[:, 0]+discount[:, 0]*ret[:, 1])
    torch.testing.assert_close(ret[:, 1], reward[:, 1])
    expected = .9**.2*2+.9**.7*4
    torch.testing.assert_close(ret[:, 0], torch.tensor([expected]))
    assert ret[:, 2] == 0


def test_randomized_first_deadlines_stay_on_control_grid_and_preserve_pairs():
    from scripts.train_schedule_improvement import sample_windows
    clip, _ = sample_clip()
    clip = replace(clip, hits=[replace(h, time=h.time+.06) for h in clip.hits], duration=.26)
    class Sampler:
        def sample(self, rng, count, **kwargs):
            return list(range(count)), [clip]*count
    windows = sample_windows(Sampler(), np.random.default_rng(12), 64)
    offsets = {c.hits[0].time for c in windows}
    assert len(offsets) > 4
    for c in windows:
        assert len(c.hits) == 3
        assert c.hits[1].time == c.hits[2].time
        assert all(abs(h.time/.02-round(h.time/.02)) < 1e-6 for h in c.hits)


def test_joint_critic_ignores_padding_but_conditions_on_proposed_hands():
    torch.manual_seed(9)
    obs = sample_obs()
    model = JointScheduleCritic(ScheduleNetConfig(width=16, heads=2, layers=1)).eval()
    a = torch.zeros(1, 10, dtype=torch.long)
    base = model(obs, a)
    other = {k: v.clone() for k, v in obs.items()}
    other['events'][:, 3:] = torch.randn_like(other['events'][:, 3:])*100
    torch.testing.assert_close(model(other, a), base)
    a[:, 0] = 1
    assert not torch.allclose(model(obs, a), base)


def test_local_search_can_make_coordinated_pair_edit_and_never_flips_padding():
    obs = sample_obs()
    live = valid_edits(obs)
    assert live.sum() == 4  # STOP, two singles, one pair
    class Critic(torch.nn.Module):
        def forward(self, obs, actions):
            return torch.where((actions[:, :2] == 1).all(1), 10.,
                               torch.where((actions[:, :2] == 0).all(1), 0., -5.))
    seed = torch.zeros(1, 10, dtype=torch.long)
    out = critic_search(Critic(), obs, seed, steps=3)
    assert out[0, :2].tolist() == [1, 1]
    assert out[0, 2:].sum() == 0
    bank = neighbors(seed)[0][live[0]].numpy()
    scores = Critic()(obs, torch.tensor(bank)).numpy()
    labels, gains = teacher_edits(bank, scores, 2)
    assert EDITS[labels[0]] == (0, 1)
    assert gains[0] == 10
