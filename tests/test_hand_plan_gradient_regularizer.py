import math
import copy
import numpy as np
import torch
from torch import nn

from isaac_rl.hand_plan_gradient_regularizer import (InputGradientConfig, input_gradient_l1,
    physical_score_gradient, tip_jacobians)
from isaac_rl.hand_plan_ranker import encode_clip, cartesian_tips
from tests.test_neural_hand_plans import clip


class LinearPlans(nn.Module):
    def __init__(self, common=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))
        self.dropout = nn.Dropout(.9)
        self.common = common

    def forward(self, x):
        value = self.weight * self.dropout(x[:,19])
        # Huge derivative on an illegal score must not enter centering/penalty.
        return torch.stack((value, value * 0, value * 1000), 1) + self.common*x[:,18,None]


def inputs():
    obs = torch.tensor(np.stack([encode_clip(clip())]*3))
    legal = torch.tensor([[True,True,False]]*3)
    return obs, legal, torch.zeros(3,6,17)


def test_exact_linear_penalty_parameter_gradient_centering_and_illegal_mask():
    obs, legal, jac = inputs()
    model = LinearPlans(7.); model.train()
    rng = torch.Generator().manual_seed(14)
    before = torch.get_rng_state().clone()
    penalty, stats = input_gradient_l1(model, obs, legal, jac, generator=rng)
    torch.testing.assert_close(penalty, torch.tensor(.04))
    penalty.backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor(.02))
    assert model.training and model.dropout.training
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(stats['fraction_scaled_derivatives_below_1e_3'],torch.tensor(41/42))


def test_forced_singleton_has_zero_relative_gradient_and_rng_resumes():
    obs, legal, jac = inputs(); legal[:,1] = False
    model = LinearPlans(); rng = torch.Generator().manual_seed(99)
    saved = rng.get_state()
    first, stats = input_gradient_l1(model, obs, legal, jac, generator=rng)
    assert first == 0
    assert stats['gradient_probe_choice_cases']==0 and stats['fraction_scaled_derivatives_below_1e_3']==0
    rng.set_state(saved)
    second, _ = input_gradient_l1(model, obs, legal, jac, generator=rng)
    torch.testing.assert_close(first, second)


def test_physical_chain_rule_respects_redundant_deadlines_tips_and_absent_hits():
    obs, _, jac = inputs(); gradient = torch.zeros_like(obs)
    gradient[:,78] = 4  # Last absolute deadline, encoded in seconds /4.
    gradient[:,36] = 100  # Missing second hit of group1: no loudness derivative.
    gradient[:,80] = math.pi
    gradient[:,130] = 1
    jac[:,0,0] = 2
    gradient[:,97] = 10
    config = InputGradientConfig()
    result = physical_score_gradient(gradient,obs,jac,config)
    torch.testing.assert_close(result[:,:4],torch.full((3,4),.02))
    assert result[:,4:12].count_nonzero() == 0
    torch.testing.assert_close(result[:,12],torch.full((3,),3*config.position_scale_rad))
    torch.testing.assert_close(result[:,29],torch.full((3,),.1))


def test_fk_jacobian_matches_cartesian_finite_difference():
    obs = np.stack([encode_clip(clip())])
    jac = tip_jacobians(obs).numpy()[0]
    from amortized_opt.constants import ready_joint_pos
    q = np.asarray(ready_joint_pos(),np.float32)
    for index in [0,3,10]:
        delta = np.zeros(17,np.float32);delta[index]=.001
        finite = (cartesian_tips(q+delta)-cartesian_tips(q-delta)).flatten()/.002
        np.testing.assert_allclose(jac[:,index],finite,rtol=.003,atol=1e-4)


def test_shared_ranker_second_derivative_optimizer_and_sampler_resume_exactly():
    from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig,ranking_loss
    torch.manual_seed(20)
    model=HandPlanRanker(HandPlanRankerConfig(architecture='shared_candidate',hidden=32,dropout=.15))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    obs,_,jac=inputs();legal=torch.ones(3,16,dtype=torch.bool)
    rewards=torch.zeros(3,16);rewards[:,2]=1
    generator=torch.Generator().manual_seed(300)
    def step(m,opt,rng):
        m.train();ranking,_=ranking_loss(m(obs),rewards,legal)
        penalty,_=input_gradient_l1(m,obs,legal,jac,generator=rng)
        opt.zero_grad();(ranking+.1*penalty).backward();opt.step()
        assert torch.isfinite(penalty) and m.training
    step(model,optimizer,generator)
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),gradient_rng=generator.get_state()))
    step(model,optimizer,generator)
    restored=HandPlanRanker(model.config);restored.load_state_dict(saved['model'])
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.001)
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored_rng=torch.Generator();restored_rng.set_state(saved['gradient_rng'])
    torch.set_rng_state(saved['torch_rng']);step(restored,restored_optimizer,restored_rng)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_read_only_shared_model_probe_preserves_parameters_modes_and_training_rng():
    from isaac_rl.hand_plan_ranker import HandPlanRanker,HandPlanRankerConfig
    model=HandPlanRanker(HandPlanRankerConfig(architecture='shared_candidate',hidden=32,dropout=.15))
    model.train();obs,_,jac=inputs();legal=torch.ones(3,16,dtype=torch.bool)
    before=[p.detach().clone() for p in model.parameters()];rng=torch.get_rng_state().clone()
    input_gradient_l1(model,obs,legal,jac,generator=torch.Generator().manual_seed(400),create_graph=False)
    assert model.training and all(m.training for m in model.modules())
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in zip(before,model.parameters()):
        assert torch.equal(a,b) and b.grad is None
