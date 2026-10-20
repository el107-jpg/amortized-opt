import numpy as np
import torch

from isaac_rl.schedule_ranking import select_preference_pairs, preference_loss


def test_pair_mining_excludes_padding_near_ties_and_unstable_rebound_preferences():
    predicted = np.array([[0., 2., 1., 999.]])
    samples = np.array([[[1., 1., 1., 1.], [0., 0., 0., 0.],
                         [3., 3., -3., -3.], [100., 100., 100., 100.]]])
    valid = np.array([[True, True, True, False]])
    pairs = select_preference_pairs(predicted, samples, valid, np.random.default_rng(3))
    assert pairs['owners'].tolist() == [0]
    assert pairs['candidate_a'].tolist() == [0]
    assert pairs['candidate_b'].tolist() == [1]
    assert pairs['delta'].tolist() == [1.]
    assert pairs['hard'].tolist() == [True]


def test_pairs_never_compare_scores_from_different_starting_states():
    predicted = np.array([[0., 2.], [100., 98.]])
    scores = np.array([[[1., 1.], [0., 0.]], [[101., 101.], [100., 100.]]])
    pairs = select_preference_pairs(predicted, scores, np.ones((2, 2), bool), np.random.default_rng(4))
    assert pairs['owners'].tolist() == [0, 1]
    assert pairs['delta'].tolist() == [1., 1.]
    assert pairs['hard'].tolist() == [True, False]


def test_preference_loss_corrects_ordering_in_both_directions_and_is_shift_invariant():
    a = torch.tensor([0., 1.], requires_grad=True)
    b = torch.tensor([1., 0.], requires_grad=True)
    delta = torch.tensor([1., -1.])
    hard = torch.tensor([True, False])
    loss = preference_loss(a, b, delta, hard)
    torch.testing.assert_close(loss, preference_loss(a+100, b+100, delta, hard))
    loss.backward()
    assert a.grad[0] < 0 and b.grad[0] > 0
    assert a.grad[1] > 0 and b.grad[1] < 0
    assert abs(a.grad[0]) > abs(a.grad[1])
