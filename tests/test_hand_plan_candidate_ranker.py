from dataclasses import asdict
import numpy as np
import torch
from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,ACTION_BANK,encode_clip,ranking_loss
from tests.test_neural_hand_plans import clip


def model(dropout=0.):
    torch.manual_seed(18)
    return HandPlanRanker(HandPlanRankerConfig(architecture='shared_candidate',hidden=32,dropout=dropout))


def observations():
    return torch.as_tensor(np.stack([encode_clip(clip()),encode_clip(clip(),qd=np.ones(17))]))


def test_candidates_are_explicit_and_independent_of_list_position():
    m=model();m.eval();obs=observations()
    full=m(obs);assert full.shape==(2,16)
    permutation=np.random.default_rng(12).permutation(16)
    torch.testing.assert_close(m.score_candidates(obs,ACTION_BANK[permutation]),full[:,permutation])
    duplicate=np.array([ACTION_BANK[3],ACTION_BANK[3],ACTION_BANK[11]])
    scored=m.score_candidates(obs,duplicate)
    torch.testing.assert_close(scored,full[:,[3,3,11]])
    torch.testing.assert_close(m.score_candidates(obs,ACTION_BANK[[11]]),full[:,[11]])
    per_case=np.stack([ACTION_BANK[[2,4]],ACTION_BANK[[6,8]]])
    expected=torch.stack([full[0,[2,4]],full[1,[6,8]]])
    torch.testing.assert_close(m.score_candidates(obs,per_case),expected)


def test_state_and_deadlines_affect_scores_and_present_drums_share_embedding_gradients():
    m=model();m.eval();obs=observations()
    original=m(obs[:1]);changed=obs[:1].clone();changed[0,18]+=.1;changed[0,19]+=.4
    assert not torch.equal(original,m(changed))
    assert not torch.equal(original,m(obs[1:]))
    scores=m(obs);scores.retain_grad();observed=torch.zeros((2,16),dtype=torch.bool);observed[:,:8]=True
    rewards=torch.zeros((2,16));rewards[:,3]=1.;rewards[:,8:]=torch.nan
    loss,_=ranking_loss(scores,rewards,observed);loss.backward()
    assert scores.grad[:,8:].count_nonzero()==0
    gradient=m.candidate_scorer.drum_embedding.weight.grad
    assert gradient[:4].abs().sum()>0 and gradient[4:].count_nonzero()==0


def test_dropout_checkpoint_config_and_rng_reproduce_next_prediction(tmp_path):
    m=model(.15);m.train();obs=observations()
    first=m(obs);assert not torch.equal(first,m(obs))
    torch.save(dict(network=asdict(m.config),model=m.state_dict(),rng=torch.get_rng_state()),tmp_path/'candidate.pt')
    expected=m(obs)
    saved=torch.load(tmp_path/'candidate.pt',weights_only=False)
    restored=HandPlanRanker(HandPlanRankerConfig(**saved['network']));restored.load_state_dict(saved['model'])
    torch.set_rng_state(saved['rng']);torch.testing.assert_close(restored(obs),expected,rtol=0,atol=0)
    m.eval();restored.eval();torch.testing.assert_close(m(obs),restored(obs),rtol=0,atol=0)


def test_wider_shared_encoders_score_explicit_plans_and_roundtrip(tmp_path):
    torch.manual_seed(19)
    cfg=HandPlanRankerConfig(architecture='shared_candidate',hidden=448,
        group_width=128,state_width=128,dropout=.15)
    widened=HandPlanRanker(cfg).eval();obs=observations()
    scores=widened(obs)
    assert scores.shape==(2,16)
    assert widened.candidate_scorer.group_encoder[0].out_features==128
    assert widened.candidate_scorer.state_encoder[0].out_features==128
    torch.testing.assert_close(widened.score_candidates(obs,ACTION_BANK[[7,2,7]]),scores[:,[7,2,7]])
    torch.save(dict(network=asdict(cfg),model=widened.state_dict()),tmp_path/'wide.pt')
    saved=torch.load(tmp_path/'wide.pt',weights_only=False)
    restored=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).eval()
    restored.load_state_dict(saved['model'])
    torch.testing.assert_close(restored(obs),scores,rtol=0,atol=0)


def test_deeper_scorer_keeps_shared_candidate_semantics_and_roundtrips(tmp_path):
    torch.manual_seed(20)
    cfg=HandPlanRankerConfig(architecture='shared_candidate',hidden=448,
        group_width=128,state_width=128,scorer_depth=3,dropout=.15)
    deep=HandPlanRanker(cfg).eval();obs=observations()
    assert sum(parameter.numel() for parameter in deep.parameters())==607249
    scores=deep(obs)
    torch.testing.assert_close(deep.score_candidates(obs,ACTION_BANK[[9,1,9]]),scores[:,[9,1,9]])
    torch.save(dict(network=asdict(cfg),model=deep.state_dict()),tmp_path/'deep.pt')
    saved=torch.load(tmp_path/'deep.pt',weights_only=False)
    restored=HandPlanRanker(HandPlanRankerConfig(**saved['network'])).eval()
    restored.load_state_dict(saved['model'])
    torch.testing.assert_close(restored(obs),scores,rtol=0,atol=0)
