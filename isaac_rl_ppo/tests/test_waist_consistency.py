from types import SimpleNamespace

import pytest
import torch

from isaac_rl_ppo.actor import LoRALinear
from isaac_rl_ppo.ppo_loop import PpoUpdater, RolloutDriver, terminal_waist_reward


def test_lora_output_row_mask_is_structural():
    base = torch.nn.Linear(5, 4)
    layer = LoRALinear(base, rank=2, alpha=2.0)
    layer.restrict_output_rows((1,))
    with torch.no_grad():
        layer.lora_b.fill_(1.0)
    x = torch.randn(7, 5)
    delta = layer(x) - base(x)
    assert torch.equal(delta[:, 0], torch.zeros(7))
    assert torch.equal(delta[:, 2], torch.zeros(7))
    assert torch.equal(delta[:, 3], torch.zeros(7))
    assert bool((delta[:, 1].abs() > 0).any())


def test_first_plan_anchor_is_shifted_in_absolute_time():
    driver = RolloutDriver.__new__(RolloutDriver)
    driver.device = torch.device("cpu")
    driver.mpc = SimpleNamespace(B=1, H=4, dtype=torch.float32)
    driver._yaw_i = 0
    driver._waist_anchor = torch.zeros(1, 4)
    driver._waist_anchor_age = torch.full((1,), 4, dtype=torch.long)
    driver._waist_anchor_len = torch.full((1,), 4, dtype=torch.long)
    driver._waist_terminal_due = torch.zeros(1, dtype=torch.bool)
    driver._waist_terminal_target = torch.zeros(1)

    first = torch.zeros(1, 4, 17)
    first[0, :, 0] = torch.tensor([10.0, 11.0, 12.0, 13.0])
    mask = torch.ones_like(first)
    target0, valid0 = driver._aligned_waist_anchor(first, mask)
    assert torch.equal(target0, torch.tensor([[10.0, 11.0, 12.0, 13.0]]))
    assert bool(valid0.all())
    assert not bool(driver._waist_terminal_due[0])

    driver._waist_anchor_age.fill_(1)
    unrelated_replan = torch.full_like(first, -99.0)
    target1, valid1 = driver._aligned_waist_anchor(unrelated_replan, mask)
    assert torch.equal(target1, torch.tensor([[11.0, 12.0, 13.0, 13.0]]))
    assert torch.equal(valid1, torch.tensor([[True, True, True, False]]))

    driver._waist_anchor_age.fill_(3)
    driver._aligned_waist_anchor(unrelated_replan, mask)
    assert bool(driver._waist_terminal_due[0])
    assert float(driver._waist_terminal_target[0]) == pytest.approx(13.0)


def test_terminal_waist_reward_is_sparse_and_endpoint_based():
    q = torch.zeros(2, 17)
    q[:, 0] = torch.tensor([0.2, 0.0])
    bonus, error = terminal_waist_reward(
        q, torch.tensor([0.2, 0.2]), torch.tensor([True, False]),
        yaw_index=0, weight=10.0, scale_rad=0.1,
    )
    assert torch.equal(error, torch.tensor([0.0, 0.2]))
    assert torch.equal(bonus, torch.tensor([10.0, 0.0]))


def test_waist_consistency_masks_tail_and_uses_huber():
    updater = PpoUpdater.__new__(PpoUpdater)
    updater.device = torch.device("cpu")
    updater.yaw_i = 0
    updater.waist_consistency_beta = 0.02

    traj = torch.zeros(1, 3, 17)
    traj[0, :, 0] = torch.tensor([0.01, 0.04, 9.0])
    target = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    loss, mae = updater._waist_consistency(traj, target, mask)

    # Huber: 0.5*.01^2/.02=.0025; .04-.01=.03; mean=.01625.
    assert float(loss) == pytest.approx(0.01625)
    assert float(mae) == pytest.approx(0.025)
