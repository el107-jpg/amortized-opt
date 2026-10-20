import numpy as np
import torch
from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from amortized_opt.constants import ready_joint_pos
from isaac_rl.arm_assignment import AssignmentConfig
from isaac_rl.assignment_search import candidate_sequences, pack_candidates, direct_q_targets


def phrase(n):
    kit = DrumKit.from_layout()
    hits = [Hit(.5+i*.2,.5+i*.2,i%kit.num_drums,kit.names[i%kit.num_drums],'left',.7) for i in range(n)]
    return kit, RhythmClip('test',hits,hits[-1].time)


def test_four_decisions_exhaustive_and_dense_phrases_preserved():
    kit, clip = phrase(4)
    groups, rows, _ = candidate_sequences(clip,kit,AssignmentConfig(),np.random.default_rng(7),ready_joint_pos())
    assert len(groups)==4 and rows.shape==(16,4)
    assert len(set(map(tuple, rows)))==16
    kit, clip = phrase(12)
    groups, rows, _ = candidate_sequences(clip,kit,AssignmentConfig(),np.random.default_rng(7),ready_joint_pos())
    assert len(groups)==12 and rows.shape==(16,12)
    assert set(rows[:,0])=={0,1}


def test_bounce_trials_are_paired_by_physical_hit_not_arm():
    kit, clip = phrase(4)
    p=pack_candidates([clip],kit,AssignmentConfig(),ready_joint_pos()[None],np.random.default_rng(4))
    for g, h in enumerate(clip.hits):
        values=[]
        for b in range(0,len(p['q0']),2):
            hand=int((p['drums'][b,g]==h.drum).nonzero()[0])
            values.append(p['efficiency'][b:b+2,g,hand])
        assert all(torch.equal(v,values[0]) for v in values)
    assert not torch.equal(values[0][0],values[0][1])


def test_return_targets_average_rebounds_before_choosing_sequence():
    p=dict(owner=np.zeros(6,int),first_action=np.array([0,0,0,0,1,1]),
           sequences=[(0,0),(0,0),(0,1),(0,1),(1,0),(1,0)])
    target,best=direct_q_targets(torch.tensor([100.,-100.,10.,10.,3.,5.]),p,1,requested=np.full(6,2))
    np.testing.assert_allclose(target,[[5.,2.]])
    np.testing.assert_array_equal(best,[2,3])


def test_graph_seed_failure_preserves_same_drum_simultaneous_strikes():
    kit, clip = phrase(5)
    clip.hits.insert(1, Hit(.5,.5,0,kit.names[0],'right',.7))
    packed=pack_candidates([clip],kit,AssignmentConfig(),ready_joint_pos()[None],np.random.default_rng(9))
    assert len(packed['problems'][0][0]) == 5
    assert packed['drums'].ge(0).sum((1,2)).eq(6).all()
    assert packed['drums'][:,0].eq(0).all()
    assert set(packed['first_action'])=={0,1}


def test_rejected_graph_matching_does_not_create_a_search_hand_mask(monkeypatch):
    from isaac_rl.arm_assignment import AssignmentQueue
    def fail(self):
        raise ValueError('graph teacher cannot assign this group')
    monkeypatch.setattr(AssignmentQueue,'teacher_action',fail)
    kit,clip=phrase(4)
    groups,rows,_=candidate_sequences(clip,kit,AssignmentConfig(),np.random.default_rng(9),ready_joint_pos())
    assert len(groups)==4 and rows.shape==(16,4)
    assert all(set(rows[:,g])=={0,1} for g in range(4))
