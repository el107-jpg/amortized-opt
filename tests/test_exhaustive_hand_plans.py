import numpy as np

from scripts.evaluate_all_hand_plans import (assemble_trials, enumerate_context,
    observed_rank, summarize_contexts, make_panel)
from tests.test_transition_time_preferences import chart, clip


def test_all_sixteen_are_unique_legal_and_in_search_order():
    base=chart();context=enumerate_context(clip(),base,np.full((2,2),.15))
    assert context['legal_plans']==16
    assert len({tuple(row) for row in context['actions']})==16
    assert np.all(np.diff(context['scores'])>=0)


def test_rank_ties_and_top_two_missing_the_actual_best():
    assert observed_rank([.5,.5,.4])==1
    assert observed_rank([.4,.6,.6])==3
    contexts=[dict(legal_plans=4,graph_rank=2,scores=np.arange(4,dtype=float))]
    trial=dict(gather=[np.arange(4)],repeat_ids=[[]])
    reports=[dict(reward=x,hits=h,requested=5) for x,h in zip([.2,.4,.8,.6],[1,2,4,3])]
    summary,rows=summarize_contexts(contexts,trial,reports,np.array([0]))
    assert rows[0]['selected_actual_reward_rank']==4
    assert rows[0]['best_reward_predicted_rank']==3
    assert not rows[0]['predicted_top_two_contains_best_reward']
    assert summary['selected_pooled_hit_rate']==.2
    assert summary['oracle_pooled_hit_rate']==.8


def test_variable_legal_plan_counts_no_forbidden_padding_and_control_labels():
    contexts=[dict(legal_plans=2,episodes=['a','b']),dict(legal_plans=4,episodes=['c','d','e','f'])]
    trial=assemble_trials(contexts,8)
    assert trial['episodes'][:6]==['a','c','b','d','e','f']
    assert len(trial['episodes'])==8
    assert sum(label['repeat_control'] for label in trial['labels'])==2
    np.testing.assert_array_equal(trial['gather'][0],[0,2])
    np.testing.assert_array_equal(trial['gather'][1],[1,3,4,5])
