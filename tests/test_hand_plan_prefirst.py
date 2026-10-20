import torch
import numpy as np
import pytest

from isaac_rl.chart_cma_isaac import prefirst_snapshot_ticks
from isaac_rl.hand_plan_dataset import load_prefirst_factual_rows
from isaac_rl.hand_plan_ranker import ranking_loss
from isaac_rl.hand_plan_ranker import HandPlanRanker, HandPlanRankerConfig
from isaac_rl.hand_plan_value import (DualRuntimeRewardModel, score_rows,
    regression_metrics, reward_ranking_losses)


def test_prefirst_sampling_preserves_deadline_guard_and_deduplicates_fast_hits():
    assert prefirst_snapshot_ticks(.84,.02)==(10,21,32)
    assert prefirst_snapshot_ticks(.16,.02)==(2,)
    assert prefirst_snapshot_ticks(.12,.02)==()
    assert prefirst_snapshot_ticks(.02,.02)==()


def test_one_factual_plan_needs_pointwise_supervision():
    scores=torch.randn(2,16,requires_grad=True)
    observed=torch.zeros(2,16,dtype=torch.bool);observed[0,3]=True;observed[1,7]=True
    reward=torch.zeros(2,16)
    loss,_=ranking_loss(scores,reward,observed)
    loss.backward()
    assert scores.grad.abs().sum()==0


def test_factual_loader_keeps_case_split_and_executed_action(tmp_path):
    folder=tmp_path/'batches';folder.mkdir()
    observation=np.zeros((1,136),np.float32);observation[0,19]=.2
    observed=np.zeros((1,16),bool);observed[0,3]=True
    batch=dict(panel=[None],case_offset=0,observed=observed)
    factual=dict(observations=observation,actions=np.array([3]),rewards=np.array([.4]),
        hit_rates=np.array([.5]),case_ids=np.array([0]),elapsed_s=np.array([.2]),
        splits=np.array([0],np.int8))
    torch.save(dict(batch=batch,factual=factual),folder/'batch_00001.pt')
    torch.save(dict(train=np.array([0]),validation=np.array([],int),test=np.array([],int)),tmp_path/'splits.pt')
    assert len(load_prefirst_factual_rows(tmp_path)['actions'])==1
    factual['splits'][0]=1
    torch.save(dict(batch=batch,factual=factual),folder/'batch_00001.pt')
    with pytest.raises(ValueError,match='crossed'):
        load_prefirst_factual_rows(tmp_path)


def test_single_executed_plan_has_absolute_reward_gradient():
    torch.manual_seed(4)
    model=HandPlanRanker(HandPlanRankerConfig(architecture='shared_candidate',hidden=32,
        group_width=16,state_width=16))
    obs=np.zeros((2,136),np.float32)
    obs[:,8]=1.;obs[:,19]=.2
    actions=np.array([3,7])
    prediction=score_rows(model,obs,actions,chunk=1)
    expected=model(torch.as_tensor(obs))[torch.arange(2),torch.as_tensor(actions)]
    torch.testing.assert_close(prediction,expected)
    loss=(prediction-torch.tensor([.4,.7])).square().mean()
    loss.backward()
    assert float(model.candidate_scorer.scorer[-1].weight.grad.abs().sum())>0
    metrics=regression_metrics([.3,.5],[.4,.7])
    assert metrics['mse']==pytest.approx(.025)
    assert metrics['bias']==pytest.approx(-.15)


def test_reward_ranking_auxiliaries_correct_bad_order_without_rescaling_truth():
    truth=torch.tensor([[.1,.8,.4],[.6,0.,0.]])
    observed=torch.tensor([[True,True,False],[True,False,False]])
    correct=truth.clone().requires_grad_()
    pair,winner=reward_ranking_losses(correct,truth,observed)
    assert abs(float(pair))<1e-6
    assert abs(float(winner))<1e-6
    wrong=torch.tensor([[.8,.1,99.],[.2,99.,99.]],requires_grad=True)
    pair,winner=reward_ranking_losses(wrong,truth,observed)
    assert float(pair)>0 and float(winner)>0
    (pair+winner).backward()
    assert wrong.grad[0,0]>0 and wrong.grad[0,1]<0
    assert wrong.grad[0,2]==0
    assert torch.allclose(wrong.grad[1],torch.zeros(3))


def test_legacy_reward_head_cannot_change_v4_prediction_directly():
    torch.manual_seed(7)
    model=DualRuntimeRewardModel(HandPlanRankerConfig(architecture='shared_candidate',
        hidden=32,group_width=16,state_width=16))
    observation=torch.zeros(2,136)
    base=model(observation)
    torch.testing.assert_close(base,model(observation,runtime='v2'))
    with torch.no_grad():model.v2_scorer[-1].bias.add_(1.)
    torch.testing.assert_close(base,model(observation))
    torch.testing.assert_close(base+1,model(observation,runtime='v2'))
