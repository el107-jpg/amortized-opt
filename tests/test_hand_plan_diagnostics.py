import numpy as np
import pytest
import torch
from isaac_rl.hand_plan_diagnostics import prediction_diagnostics,weight_diagnostics


def test_entropy_handles_illegal_scores_ties_and_forced_choices():
    scores=np.array([[0.,0.,1e6],[0.,99.,99.]])
    observed=np.array([[True,True,False],[True,False,False]])
    rewards=np.array([[.2,.2,99.],[.5,99.,99.]])
    result=prediction_diagnostics(scores,observed,rewards)
    assert result['prediction_entropy_nats']==pytest.approx(np.log(2)/2)
    assert result['prediction_entropy_normalized']==pytest.approx(1.)
    assert result['entropy_choice_cases']==1
    assert result['near_best_probability_mass']==pytest.approx(1.)
    assert result['near_best_target_entropy_nats']==pytest.approx(np.log(2)/2)
    assert result['top_plan_probability']==pytest.approx(.75)


def test_confident_entropy_is_stable_and_does_not_mean_correctness():
    result=prediction_diagnostics(np.array([[1000.,-1000.]]),np.ones((1,2),bool),np.array([[0.,1.]]))
    assert result['prediction_entropy_nats']==0
    assert result['near_best_probability_mass']==0
    with pytest.raises(ValueError):prediction_diagnostics(np.zeros((1,2)),np.zeros((1,2),bool),np.zeros((1,2)))


def test_weight_entropy_separates_scale_from_concentration_and_handles_zero():
    model=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():model.weight.fill_(1.)
    first=weight_diagnostics(model)
    assert first['absolute_weight_entropy_normalized']==pytest.approx(1.)
    with torch.no_grad():model.weight.mul_(.1)
    scaled=weight_diagnostics(model)
    assert scaled['parameter_l2']==pytest.approx(first['parameter_l2']*.1)
    assert scaled['absolute_weight_entropy_normalized']==pytest.approx(1.)
    assert first['absolute_weight_statistic_raw']==0
    assert scaled['absolute_weight_statistic_raw']==pytest.approx(-4*.1*np.log(.1))
    with torch.no_grad():model.weight.zero_();model.weight[0,0]=1.
    assert weight_diagnostics(model)['absolute_weight_entropy_normalized']==0
    with torch.no_grad():model.weight.zero_()
    assert all(np.isfinite(v) for v in weight_diagnostics(model).values())
