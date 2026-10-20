"""Controls for noise attribution on feasible MPC reference trajectories."""
import torch

from scripts.probe_mpc_expert_noise import corrupt_reference


def test_zero_noise_preserves_reference_and_does_not_modify_it():
    reference = torch.tensor([[[.1], [.2], [0.]]])
    mask = torch.tensor([[[1.], [1.], [0.]]])
    before = reference.clone()
    out = corrupt_reference(reference, mask, torch.ones_like(reference),
                            torch.ones_like(reference), 0, -1, 1, False)
    assert torch.equal(out, before)
    assert torch.equal(reference, before)
    out[0, 0] = .7
    assert torch.equal(reference, before)


def test_ou_white_corruption_clips_and_masks_without_moving_reference():
    reference = torch.zeros(1, 3, 2)
    mask = torch.ones_like(reference)
    mask[:, 2] = 0
    correlated = torch.tensor([[[1., -1.], [2., -2.], [3., -3.]]])
    white = torch.ones_like(reference)
    out = corrupt_reference(reference, mask, correlated, white, .5, -.7, .7, False)
    assert torch.allclose(out[0, 0], torch.tensor([.51, -.49]))
    assert torch.equal(out[0, 1], torch.tensor([.7, -.7]))
    assert torch.equal(out[0, 2], torch.zeros(2))
    assert torch.equal(reference, torch.zeros_like(reference))


def test_anchored_noise_matches_fresh_seed_envelope_after_clipping():
    reference = torch.zeros(1, 6, 1)
    mask = torch.ones_like(reference)
    out = corrupt_reference(reference, mask, torch.ones_like(reference),
                            torch.zeros_like(reference), 2., -1., 1., True)
    # Clip the noisy seed to 1 BEFORE interpolating with the clean reference.
    expected = torch.tensor([.104, .352, .648, .896, 1., 1.])
    assert torch.allclose(out[0, :, 0], expected)
