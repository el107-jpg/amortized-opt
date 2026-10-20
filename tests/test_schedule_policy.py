from types import SimpleNamespace

import torch

from isaac_rl.schedule_policy import (FourHitSchedulePolicy, entropy_and_kl,
    group_advantages, clipped_group_loss, expected_reward_loss)
from scripts.train_schedule_policy import rollout_transitions


def test_categorical_policy_masks_padding_and_handles_empty_horizons():
    torch.set_num_threads(1)
    policy = FourHitSchedulePolicy()
    x = torch.zeros(2, 86)
    x[0, 55] = 1.  # one live hit, then three padding slots
    lp = policy.log_probs(x); p = lp.exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    assert (p[0, 1:8] == 0).all() and (p[0, 9:] == 0).all()
    assert p[0, 0] > 0 and p[0, 8] > 0
    assert p[1, 0] == 1 and (p[1, 1:] == 0).all()
    entropy, kl = entropy_and_kl(lp, lp.detach())
    assert entropy[1] == 0 and torch.isfinite(entropy).all()
    torch.testing.assert_close(kl, torch.zeros(2))
    entropy.sum().backward()
    assert all(torch.isfinite(param.grad).all() for param in policy.parameters())


def test_group_advantages_center_each_case_and_dampen_near_ties():
    reward = torch.tensor([[1., 2., 3.], [100., 100., 100.], [0., .1, .2]], requires_grad=True)
    advantage = group_advantages(reward)
    torch.testing.assert_close(advantage.mean(-1), torch.zeros(3), atol=1e-7, rtol=0)
    torch.testing.assert_close(advantage[1], torch.zeros(3))
    torch.testing.assert_close(advantage[2], torch.tensor([-.1, 0., .1]))
    assert not advantage.requires_grad


def test_grpo_clips_both_signs_of_advantage():
    logits = torch.tensor([[2.1972246, 0.]], requires_grad=True)  # probabilities .9/.1
    lp = logits.log_softmax(-1)
    old = torch.full((1,2), .5).log()
    loss, fraction = clipped_group_loss(lp, torch.tensor([[0,1]]), old, torch.tensor([[1.,-1.]]))
    torch.testing.assert_close(loss, torch.tensor(-.2), atol=1e-6, rtol=0)
    assert fraction == 1
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_full_action_gradient_matches_enumerated_policy_gradient():
    logits = torch.tensor([[.2, -.4, .7]], requires_grad=True)
    rewards = torch.tensor([[20., -10., 50.]], requires_grad=True)
    lp = logits.log_softmax(-1)
    direct = expected_reward_loss(lp, rewards)
    grad = torch.autograd.grad(direct, logits, retain_graph=True)[0]
    centered = (rewards-rewards.mean(-1, keepdim=True)).detach()/20
    enumerated = -(lp.exp().detach()*centered*lp).sum(-1).mean()
    torch.testing.assert_close(grad, torch.autograd.grad(enumerated, logits)[0])
    shifted = expected_reward_loss(logits.log_softmax(-1), rewards+1000.)
    torch.testing.assert_close(direct, shifted)
    assert rewards.grad is None


def test_exact_kl_has_correct_direction_and_positive_penalty():
    lp = torch.tensor([[.8,.2]]).log()
    ref = torch.tensor([[.5,.5]]).log()
    _,kl = entropy_and_kl(lp,ref)
    expected = .8*torch.log(torch.tensor(1.6))+.2*torch.log(torch.tensor(.4))
    torch.testing.assert_close(kl[0], expected)
    assert kl[0] > 0


def test_sampled_transitions_keep_global_schedule_indices_and_mc_returns():
    class FakeScorer:
        reward_scale = 20.
        def features(self, table, q, v, fk):
            return q[:, :1].expand(-1, 86)

    table = dict(drums=torch.tensor([[[0,-1],[1,-1],[2,-1],[3,-1]]]),
        times=torch.tensor([[.1,.2,.3,.4]]), valid=torch.ones(1,4,dtype=torch.bool),
        speeds=torch.tensor([[[2.,0.]]*4]))
    scores = torch.arange(1.,17.).reshape(1,2,2,4)
    poses = torch.arange(1.,5.)[None,None,None,:,None].expand(1,2,2,4,17)
    out = dict(score=scores, poses=poses, post_velocity=torch.zeros_like(poses))
    replay = rollout_transitions(FakeScorer(), SimpleNamespace(cost=SimpleNamespace(fk=None)),
        table, torch.zeros(1,86), torch.tensor([[5,14]]), out)
    base = torch.tensor([5,5,14,14])
    for g in range(4):
        sl = slice(g*4,(g+1)*4)
        torch.testing.assert_close(replay['action'][sl], (base % (1<<(4-g))) << g)
        torch.testing.assert_close(replay['target'][sl], scores.reshape(4,4)[:,g:].sum(-1)/20)
        torch.testing.assert_close(replay['reward'][sl], scores.reshape(4,4)[:,g]/20)
        assert replay['done'][sl].all() if g==3 else not replay['done'][sl].any()
        torch.testing.assert_close(replay['obs'][sl,0], torch.full((4,),float(g)))


def test_policy_can_initialize_and_reload_a_wider_scorer(tmp_path):
    from isaac_rl.schedule_distillation import FourHitScheduleScorer
    scorer=FourHitScheduleScorer(hidden_dims=(48,64,32)).eval()
    source=tmp_path/'scorer.pt'
    torch.save(dict(format=scorer.format,model=scorer.state_dict(),reward_scale=scorer.reward_scale,
        hidden_dims=scorer.hidden_dims),source)
    policy=FourHitSchedulePolicy.from_scorer_checkpoint(source,temperature=1.5)
    path=tmp_path/'policy.pt'
    torch.save(dict(format=policy.format,model=policy.state_dict(),reward_scale=policy.reward_scale,
        hidden_dims=policy.hidden_dims,temperature=policy.temperature),path)
    restored=FourHitSchedulePolicy.from_checkpoint(path)
    assert restored.hidden_dims==(48,64,32) and restored.temperature==1.5
    x=torch.randn(2,86);x[:,[55,65,75,85]]=1.
    torch.testing.assert_close(restored.log_probs(x),policy.log_probs(x))
