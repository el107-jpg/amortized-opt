import pytest
import torch

from scripts.ablate_distributional_ood_critic import (
    GlobalTaskSampler,
    _selective_metrics,
)

from amortized_opt.distributional_critic import (
    calibrated_risk_threshold,
    DistributionalEnsemble,
    MANAGEABLE_BINS,
    categorical_ordinal_loss,
    distributional_risk,
    distributional_targets,
    high_entropy_loss,
    normalized_entropy,
    OnlineDistributionalCriticTrainer,
    probabilities,
)
from amortized_opt.violation_critic import ViolationCriticConfig


def test_bins_put_zero_on_manageable_side_and_append_worst_constraint():
    target = torch.tensor([
        [-3.0, -0.1, 0.0, 0.3],
        [-0.1, -0.2, -0.3, -0.4],
    ])
    bins = distributional_targets(target)
    assert bins.shape == (2, 5)
    assert bins[0, 2] == MANAGEABLE_BINS - 1
    assert bins[0, 4] == bins[0, :4].max()
    assert bins[1, 4] < MANAGEABLE_BINS


def test_softmax_and_isomax_ensemble_shapes_and_gradients():
    x = torch.randn(7, 12)
    target = distributional_targets(torch.randn(7, 4))
    for kind in ("softmax", "isomax"):
        model = DistributionalEnsemble(12, num_heads=3, kind=kind)
        logits, distance = model(x)
        assert logits.shape == (3, 7, 5, 10)
        if kind == "isomax":
            assert distance.shape == (3, 7, 5)
            assert (distance >= 0).all()
        else:
            assert distance is None
        loss = categorical_ordinal_loss(logits, target) + high_entropy_loss(logits)
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


def test_fused_softmax_heads_match_serial_forward_and_gradients():
    torch.manual_seed(13)
    model = DistributionalEnsemble(12, num_heads=3, kind="softmax")
    per_head = torch.randn(3, 7, 12)
    fused, distance = model.forward_per_head(per_head)
    serial = torch.stack([
        head(per_head[index])[0]
        for index, head in enumerate(model.heads)
    ])
    assert distance is None
    torch.testing.assert_close(fused, serial)

    fused.square().mean().backward()
    fused_grad = [parameter.grad.detach().clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    serial.square().mean().backward()
    serial_grad = [parameter.grad.detach().clone() for parameter in model.parameters()]
    for actual, expected in zip(fused_grad, serial_grad):
        torch.testing.assert_close(actual, expected)


def test_temperature_entropy_and_uniform_unknown_objective():
    logits = torch.zeros(2, 4, 5, 10, requires_grad=True)
    p = probabilities(logits, torch.ones(5))
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(2, 4, 5))
    torch.testing.assert_close(normalized_entropy(p), torch.ones(2, 4, 5))
    loss = high_entropy_loss(logits)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_distributional_risk_distinguishes_equal_entropy_shapes_and_heads():
    # Point masses have equal entropy.  A severe positive tail must still cost
    # more than a narrow acceptable prediction.
    central = torch.zeros(3, 1, 5, 10)
    severe = torch.zeros_like(central)
    central[..., 5] = 1.0
    severe[..., 9] = 1.0
    central_risk, _ = distributional_risk(central)
    severe_risk, _ = distributional_risk(severe)
    assert severe_risk.item() > central_risk.item()

    # Holding the ensemble mean roughly fixed, disagreement adds epistemic
    # cost beyond a set of identical heads.
    agreed = torch.zeros(3, 1, 5, 10)
    agreed[..., 4] = 0.5
    agreed[..., 8] = 0.5
    split = torch.zeros_like(agreed)
    split[0, ..., 4] = 1.0
    split[1:, ..., 8] = 1.0
    _, agreed_parts = distributional_risk(agreed)
    _, split_parts = distributional_risk(split)
    assert split_parts["epistemic"].item() > agreed_parts["epistemic"].item()


def test_vectorized_risk_calibration_matches_reference_sweep_with_ties():
    risk = torch.tensor([0.4, 0.1, 0.1, 0.8, 0.3, 0.4, 0.6])
    truth = torch.tensor([True, True, False, True, True, False, True])

    def reference(min_precision):
        best = None
        for threshold in risk.unique(sorted=True):
            selected = risk <= threshold
            count = int(selected.sum())
            true_count = int((selected & truth).sum())
            precision = true_count / max(count, 1)
            recall = true_count / max(int(truth.sum()), 1)
            if precision + 1.0e-12 < min_precision:
                continue
            candidate = (recall, count, float(threshold))
            if best is None or candidate > best[0]:
                best = (candidate, float(threshold), precision, recall)
        return None if best is None else best[1:]

    for min_precision in (0.5, 0.7, 0.95, 1.1):
        actual = calibrated_risk_threshold(
            risk, truth, min_precision=min_precision,
        )
        expected = reference(min_precision)
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected)


def test_selective_metrics_reward_entropy_that_identifies_errors():
    target = torch.zeros(4, dtype=torch.long)
    probability = torch.tensor([
        [0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7],
    ])
    entropy = torch.tensor([0.1, 0.2, 0.8, 0.9])
    metrics = _selective_metrics(probability, entropy, target)
    assert metrics["entropy_error_auc"] == 1.0
    assert metrics["coverage"]["0.5"]["bin_accuracy"] == 1.0
    assert metrics["coverage"]["1.0"]["bin_accuracy"] == 0.5


def test_global_task_sampler_emits_minimal_features_and_fresh_draws():
    sampler = GlobalTaskSampler()
    generator = torch.Generator().manual_seed(17)
    first = sampler.sample(512, generator=generator)
    second = sampler.sample(512, generator=generator)
    assert first.shape == (512, 42)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, second)
    assert (first[:, 40:42] >= 0.0).all()
    assert (first[:, 40:42] <= 1.0).all()
    goals = first[:, 34:40].reshape(-1, 2, 3)
    inactive = first[:, 40:42] == 0.0
    assert (goals[inactive] == 0.0).all()


def test_global_task_sampler_treats_idle_as_one_goal_choice_per_hand():
    sampler = GlobalTaskSampler()
    rows = sampler.sample(20_000, generator=torch.Generator().manual_seed(23))
    active = rows[:, 40:42] > 0.0
    # Each hand has six legal drums plus idle, so the expected idle rate is
    # 1/7.  This loose finite-sample bound also guards against special-casing
    # left-only or right-only modes in the sampler.
    idle_rate = (~active).float().mean(dim=0)
    assert ((idle_rate > 0.12) & (idle_rate < 0.17)).all()
    left_only = (active[:, 0] & ~active[:, 1]).float().mean()
    right_only = (~active[:, 0] & active[:, 1]).float().mean()
    assert abs(float(left_only - right_only)) < 0.02


def test_online_distributional_anchor_curriculum_opens_and_checkpoints():
    cfg = ViolationCriticConfig(
        mode="assist", representation="distributional", num_heads=3,
        batch_size=16, updates_per_step=1, recent_capacity=128,
        audit_capacity=16, risk_recalibrate_every=1,
        global_ood_fraction=0.0,
        anchor_curriculum=True, anchor_min_samples=32,
        anchor_min_close_rate=0.5,
    )
    trainer = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=7)
    features = torch.randn(40, 42)
    gate = torch.zeros(40, 8)
    gate[:, 4] = cfg.terminal_limit
    gate[:, 5] = cfg.collision_sdf_limit
    gate[:, 6] = cfg.joint_limit
    gate[:, 7] = cfg.torque_limit
    trainer.observe_gate(
        features, gate, actor_step=17,
        anchor_mask=torch.ones(40, dtype=torch.bool),
    )
    assert trainer.curriculum_open
    assert trainer.curriculum_open_actor_step == 17
    assert torch.isfinite(torch.tensor(trainer.risk_threshold))
    assert trainer.manageable(features).shape == (40,)
    metrics = trainer.metrics()
    assert 0.0 <= metrics["entropy_ema_train"] <= 1.0
    assert 0.0 <= metrics["entropy_ema_ood"] <= 1.0
    assert metrics["ensemble_mi_train"] >= 0.0
    assert metrics["ensemble_mi_ood"] >= 0.0
    assert 0.0 <= metrics["predicted_manageable_rate"] <= 1.0

    restored = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=8)
    restored.load_state_dict(trainer.state_dict())
    assert restored.curriculum_open
    assert restored.curriculum_open_actor_step == 17
    assert restored.anchor_samples == 40
    assert restored.risk_threshold == trainer.risk_threshold


def test_terminal_band_success_latch_unlocks_and_checkpoints():
    cfg = ViolationCriticConfig(
        representation="distributional", short_floor_wp=8,
        short_unlock_min_samples=8, short_unlock_success=0.5,
        short_success_ema=0.0,
    )
    trainer = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=3)
    assert not trainer.terminal_band_unlocked
    trainer.observe_terminal_band(torch.tensor([True, False, True, False]))
    assert not trainer.terminal_band_unlocked
    trainer.observe_terminal_band(torch.tensor([True, True, False, True]))
    assert trainer.terminal_band_unlocked
    assert trainer.terminal_band_samples == 8
    assert trainer.terminal_band_success_ema == pytest.approx(0.75)

    restored = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=4)
    restored.load_state_dict(trainer.state_dict())
    assert restored.terminal_band_unlocked
    assert restored.terminal_band_samples == 8


def test_boundary_manageability_controller_backs_off_and_readvances():
    cfg = ViolationCriticConfig(
        representation="distributional", short_floor_wp=8,
        coarse_step_knots=5, boundary_min_manageable=0.5,
        boundary_manageable_margin=0.1, boundary_rate_ema=0.0,
        boundary_floor_update_every=1,
    )
    trainer = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=5)
    trainer.observe_boundary(
        torch.tensor([True, False, False, False]), actor_step=10, horizon=50,
    )
    assert trainer.boundary_manageable_ema == pytest.approx(0.25)
    assert trainer.boundary_floor_wp == 13
    trainer.observe_boundary(
        torch.tensor([True, True, True, False]), actor_step=11, horizon=50,
    )
    assert trainer.boundary_manageable_ema == pytest.approx(0.75)
    assert trainer.boundary_floor_wp == 8

    restored = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=6)
    restored.load_state_dict(trainer.state_dict())
    assert restored.boundary_floor_wp == 8
    assert restored.boundary_manageable_ema == pytest.approx(0.75)


def test_anchor_curriculum_ramp_checkpoint_and_legacy_open_resume():
    cfg = ViolationCriticConfig(
        mode="assist", representation="distributional", num_heads=3,
        batch_size=16, updates_per_step=1, recent_capacity=128,
        audit_capacity=16, risk_recalibrate_every=1,
        global_ood_fraction=0.0,
        anchor_curriculum=True, anchor_min_samples=32,
        anchor_min_close_rate=0.5, anchor_curriculum_ramp_steps=100,
    )
    trainer = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=7)
    features = torch.randn(40, 42)
    gate = torch.zeros(40, 8)
    gate[:, 4] = cfg.terminal_limit
    gate[:, 5] = cfg.collision_sdf_limit
    gate[:, 6] = cfg.joint_limit
    gate[:, 7] = cfg.torque_limit
    trainer.observe_gate(
        features, gate, actor_step=40,
        anchor_mask=torch.ones(40, dtype=torch.bool),
    )
    assert trainer.curriculum_open_actor_step == 40
    assert trainer.curriculum_ramp_fraction(40) == 0.0
    assert trainer.curriculum_ramp_fraction(90) == 0.5
    assert trainer.curriculum_ramp_fraction(140) == 1.0

    restored = OnlineDistributionalCriticTrainer(42, cfg, device="cpu", seed=8)
    restored.load_state_dict(trainer.state_dict())
    assert restored.curriculum_ramp_fraction(90) == 0.5

    legacy = trainer.state_dict()
    legacy.pop("curriculum_open_actor_step")
    legacy_restored = OnlineDistributionalCriticTrainer(
        42, cfg, device="cpu", seed=9,
    )
    legacy_restored.load_state_dict(legacy)
    assert legacy_restored.curriculum_open
    assert legacy_restored.curriculum_open_actor_step is None
    assert legacy_restored.curriculum_ramp_fraction(40) == 1.0
