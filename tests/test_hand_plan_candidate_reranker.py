import torch

from isaac_rl.hand_plan_candidate_reranker import CandidateAwarePlanReranker
from isaac_rl.hand_plan_ranker import ACTION_BANK, HandPlanRankerConfig
from isaac_rl.hand_plan_value import DualRuntimeRewardModel


def make_model():
    torch.manual_seed(13)
    base=DualRuntimeRewardModel(HandPlanRankerConfig(architecture='shared_candidate',
        hidden=32,group_width=16,state_width=16,dropout=0.))
    return CandidateAwarePlanReranker(base,width=32,heads=4,layers=1,dropout=0.)


def test_candidate_attention_starts_at_base_and_is_order_equivariant():
    model=make_model().eval()
    obs=torch.randn(2,136)
    legal=torch.zeros(2,16,dtype=torch.bool)
    legal[0,[0,3,8,12]]=True;legal[1,7]=True
    initial=model(obs,legal=legal)
    torch.testing.assert_close(initial,model.base(obs))
    permutation=torch.randperm(16)
    reordered=model(obs,legal=legal[:,permutation],
        plans=torch.as_tensor(ACTION_BANK)[permutation])
    torch.testing.assert_close(reordered,initial[:,permutation],atol=1e-6,rtol=1e-6)
    with torch.no_grad():model.v4_correction.weight.fill_(.02)
    output,correction=model(obs,legal=legal,return_correction=True)
    assert correction[1,7].abs()<1e-7
    assert torch.all(correction[~legal]==0)
    assert torch.allclose((correction*legal).sum(1),torch.zeros(2),atol=1e-6)
    assert torch.isfinite(output).all()


def test_candidate_attention_updates_only_reranker_parameters():
    model=make_model().train()
    assert not model.base.training
    obs=torch.randn(2,136)
    legal=torch.zeros(2,16,dtype=torch.bool);legal[:,[0,1,3]]=True
    score=model(obs,legal=legal)
    loss=(score[:,0]-1).square().mean()
    loss.backward()
    assert model.v4_correction.weight.grad.abs().sum()>0
    assert all(parameter.grad is None for parameter in model.base.parameters())
