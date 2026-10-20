import numpy as np
import torch
import pytest

from amortized_opt.rhythm import Hit,RhythmClip
from isaac_rl.hand_plan_collocation_priors import (heuristic_plan_values,
    collocation_prior_loss)


def clip(groups):
    hits=[]
    for index,drums in enumerate(groups):
        t=.2*(index+1)
        for slot,drum in enumerate(drums):
            hits.append(Hit(t,t,drum,str(drum),('left','right')[slot],.7))
    return RhythmClip('prior_test',hits,1.1)


def test_consecutive_snare_hihat_prefers_left_snare_right_hihat():
    rhythm=clip([(0,),(1,),(2,),(3,)])
    values,stats=heuristic_plan_values(rhythm,(-1,-1),motif_mode='consecutive',
        switch_margin=0.)
    assert stats['motifs']==1
    assert values[4]==0.  # bits 0100: snare left, hihat right.
    assert values[8]==pytest.approx(-.12)  # bits 1000 reverse both hands.


def test_simultaneous_snare_hihat_and_repeated_drum_bias():
    rhythm=clip([(0,1),(0,),(1,),(2,)])
    values,stats=heuristic_plan_values(rhythm,(-1,-1),motif_mode='simultaneous',
        switch_margin=0.)
    assert stats['motifs']==1
    assert values[0]>values[8]
    repeat=clip([(0,),(1,),(0,),(1,)])
    values,_=heuristic_plan_values(repeat,(-1,-1),motif_mode='both',motif_margin=0.)
    assert values[5]>values[0]  # Each hand stays on its own drum.


def test_prior_penalty_is_one_sided_and_ignores_illegal_plans():
    prior=torch.tensor([[0.,-.06,-.12]])
    legal=torch.tensor([[True,True,False]])
    correct=prior.clone().requires_grad_()
    loss,stats=collocation_prior_loss(correct,prior,legal)
    assert loss.item()==0 and stats['active_pairs']==1
    wrong=torch.tensor([[-.1,.1,100.]],requires_grad=True)
    loss,stats=collocation_prior_loss(wrong,prior,legal)
    assert loss.item()>0 and stats['violated_pairs']==1
    loss.backward()
    assert wrong.grad[0,0]<0 and wrong.grad[0,1]>0
    assert wrong.grad[0,2]==0
