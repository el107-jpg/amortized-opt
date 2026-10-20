from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from amortized_opt.rhythm import Hit, RhythmClip
from isaac_rl.hand_plan_ranker import (ACTION_BANK, OBS_DIM, HandPlanRanker,
    HandPlanRankerConfig, encode_clip, legal_actions, build_ranker_batch,
    unpack_ranker_reports, ranking_loss, candidate_count, RankerReplay)
from isaac_rl.min_time_assignment import MinimumTimeChart


def eligibility():
    allowed = np.ones((2, 7, 7), bool)
    allowed[0, 6, :] = False; allowed[0, :, 6] = False
    return MinimumTimeChart(tuple(str(i) for i in range(7)),
        np.full((2, 7, 7), .2), np.zeros((2, 7, 7), int),
        np.zeros((2, 7, 7), int), allowed)


def clip(floor=False):
    hits = [Hit(1.+i*.2, 1.+i*.2, 6 if floor and i == 1 else i,
                str(6 if floor and i == 1 else i), 'left', .6) for i in range(4)]
    return RhythmClip('test', hits, 2.)


def test_action_semantics_and_floor_mask_keep_all_other_combinations():
    np.testing.assert_array_equal(ACTION_BANK, list(product((0, 1), repeat=4)))
    assert legal_actions(clip(), eligibility()).sum() == 16
    mask = legal_actions(clip(True), eligibility())
    assert mask.sum() == 8
    assert (ACTION_BANK[mask, 1] == 1).all()
    for h in clip().hits:
        single = RhythmClip('single', [replace(h, time=1.+i*.2) for i in range(4)], 2.)
        assert legal_actions(single, eligibility()).sum() == 16


def test_observation_is_canonical_sensitive_to_timing_state_and_both_targets():
    c = clip(); a = encode_clip(c)
    assert a.shape == (OBS_DIM,) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, encode_clip(replace(c, hits=list(reversed(c.hits)))))
    np.testing.assert_array_equal(a, encode_clip(replace(c, hits=[replace(h, hand='right') for h in c.hits])))
    assert not np.array_equal(a, encode_clip(replace(c, hits=[replace(h, time=h.time*1.1) for h in c.hits])))
    from amortized_opt.constants import ready_joint_pos
    assert not np.array_equal(a, encode_clip(c, q=ready_joint_pos()+.1))
    assert not np.array_equal(a, encode_clip(c, initial_drums=(0, 1)))
    doubled = replace(c, hits=[*c.hits, replace(c.hits[0], drum=4, drum_name='4')])
    assert not np.array_equal(a, encode_clip(doubled))


def test_candidate_packing_executes_unique_legal_plans_and_gathers_exact_actions():
    model = HandPlanRanker(HandPlanRankerConfig(hidden=16))
    batch = build_ranker_batch(model, [clip(), clip(True)], eligibility(), 16)
    assert len(batch['episodes']) == 24
    assert batch['observed'].sum(1).tolist() == [16, 8]
    assert len(set(map(tuple, batch['labels']))) == 24
    reports = [dict(reward=float(i), hit_rate=.5, hits=2, requested=4) for i in range(24)]
    outcomes = unpack_ranker_reports(batch, reports)
    for i, (row, action) in enumerate(batch['labels']):
        assert outcomes['rewards'][row, action] == i
        assert tuple(ACTION_BANK[action]) == tuple(
            int(h.hand == 'right') for h in batch['episodes'][i].hits)
    assert not outcomes['requested'][~batch['observed']].any()


def test_unexecuted_scores_and_missing_rewards_never_enter_ranking_loss():
    scores = torch.randn(2, 16, requires_grad=True)
    observed = torch.zeros(2, 16, dtype=torch.bool); observed[:, :4] = True
    rewards = torch.randn(2, 16); rewards[~observed] = torch.nan
    loss, _ = ranking_loss(scores, rewards, observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad[~observed].count_nonzero() == 0
    changed = scores.detach().clone(); changed[~observed] = 10000.
    changed_loss, _ = ranking_loss(changed, rewards, observed)
    torch.testing.assert_close(loss.detach(), changed_loss)


def test_near_tied_winners_share_target_and_loser_is_pushed_down():
    scores = torch.zeros(1, 16, requires_grad=True)
    rewards = torch.zeros_like(scores); rewards[0, :3] = torch.tensor([.8, .79, .1])
    observed = torch.zeros_like(scores, dtype=torch.bool); observed[0, :3] = True
    loss, _ = ranking_loss(scores, rewards, observed); loss.backward()
    assert scores.grad[0, 0] < 0 and scores.grad[0, 1] < 0
    assert scores.grad[0, 2] > 0


def test_ranker_can_learn_to_put_observed_best_plan_first():
    torch.set_num_threads(2); torch.manual_seed(12)
    model = HandPlanRanker(HandPlanRankerConfig(hidden=32))
    observations = torch.as_tensor(np.stack([encode_clip(clip())]*8))
    observed = torch.ones(8, 16, dtype=torch.bool)
    rewards = torch.zeros(8, 16); rewards[:, 11] = 1.
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    initial, _ = ranking_loss(model(observations), rewards, observed)
    for _ in range(40):
        opt.zero_grad(); loss, _ = ranking_loss(model(observations), rewards, observed)
        loss.backward(); opt.step()
    final, _ = ranking_loss(model(observations), rewards, observed)
    assert final < initial*.3
    assert (model(observations).argmax(1) == 11).all()


def test_schedule_exact_boundaries_and_replay_masks_survive_checkpoint(tmp_path):
    assert [candidate_count(i, [10, 20, 10]) for i in [1, 10, 11, 30, 31, 40]] == [16, 16, 8, 8, 4, 4]
    with pytest.raises(ValueError): candidate_count(41, [10, 20, 10])
    assert all(candidate_count(i,[40,0,0])==16 for i in range(1,41))
    assert candidate_count(1,[0,40,0])==8
    assert candidate_count(40,[0,0,40])==4
    for bad in ([0,0,0],[-1,20,10],[40,0]):
        with pytest.raises(ValueError): candidate_count(1,bad)
    model = HandPlanRanker(HandPlanRankerConfig(hidden=16))
    batch = build_ranker_batch(model, [clip(), clip(True)], eligibility(), 8)
    reports = [dict(reward=.5, hit_rate=.5, hits=2, requested=4) for _ in batch['episodes']]
    outcomes = unpack_ranker_reports(batch, reports)
    replay = RankerReplay(2); replay.add(batch['observations'], outcomes, batch['observed'])
    torch.save(dict(model=model.state_dict(), replay=replay), tmp_path/'checkpoint.pt')
    restored = torch.load(tmp_path/'checkpoint.pt', weights_only=False)
    np.testing.assert_array_equal(restored['replay'].data['observed'], batch['observed'])
    second = HandPlanRanker(model.config); second.load_state_dict(restored['model'])
    torch.testing.assert_close(model(torch.as_tensor(batch['observations'])), second(torch.as_tensor(batch['observations'])))


def test_fresh_neural_lr_schedule_warms_then_decays_without_stage_restart():
    from scripts.train_neural_hand_plans import learning_rate
    assert learning_rate(1, 4000) == pytest.approx(3e-6)
    assert learning_rate(100, 4000) == pytest.approx(3e-4)
    assert learning_rate(4000, 4000) == pytest.approx(3e-5)
    assert learning_rate(1001, 4000) < learning_rate(1000, 4000)
    assert learning_rate(3001, 4000) < learning_rate(3000, 4000)


def test_hidden_dropout_changes_training_only_and_preserves_legacy_weights():
    torch.manual_seed(77);legacy=HandPlanRanker(HandPlanRankerConfig(hidden=32))
    torch.manual_seed(77);regularized=HandPlanRanker(HandPlanRankerConfig(hidden=32,dropout=.15))
    assert legacy.state_dict().keys()==regularized.state_dict().keys()
    for name,value in legacy.state_dict().items():torch.testing.assert_close(value,regularized.state_dict()[name],rtol=0,atol=0)
    observations=torch.randn(64,OBS_DIM)
    regularized.train()
    assert not torch.equal(regularized(observations),regularized(observations))
    legacy.eval();regularized.eval()
    torch.testing.assert_close(legacy(observations),regularized(observations),rtol=0,atol=0)
    torch.testing.assert_close(regularized(observations),regularized(observations),rtol=0,atol=0)
    for bad in [-.1,1.,float('nan')]:
        with pytest.raises(ValueError):HandPlanRanker(HandPlanRankerConfig(dropout=bad))


def test_hand_selection_disables_dropout_even_if_caller_left_training_on():
    from isaac_rl.hand_plan_ranker import rank_clips
    model=HandPlanRanker(HandPlanRankerConfig(hidden=32,dropout=.15));model.train()
    before=torch.get_rng_state()
    a=rank_clips(model,[clip()],eligibility());b=rank_clips(model,[clip()],eligibility())
    assert model.training
    np.testing.assert_array_equal(a[2],b[2])
    torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)


def test_dropout_rng_optimizer_and_weights_resume_exactly(tmp_path):
    torch.manual_seed(88);model=HandPlanRanker(HandPlanRankerConfig(hidden=16,dropout=.15))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.3)
    observations=torch.randn(8,OBS_DIM);rewards=torch.randn(8,16);observed=torch.ones(8,16,dtype=torch.bool)
    def update(m,o):
        m.train();o.zero_grad();loss,_=ranking_loss(m(observations),rewards,observed)
        loss.backward();o.step()
    update(model,optimizer)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),tmp_path/'dropout.pt')
    update(model,optimizer)
    restored=HandPlanRanker(model.config);opt=torch.optim.AdamW(restored.parameters())
    saved=torch.load(tmp_path/'dropout.pt',weights_only=False)
    restored.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    update(restored,opt)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
