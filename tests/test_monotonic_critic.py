import torch

from amortized_opt.monotonic_critic import (
    ElasticAnchorMemory,
    MonotonicConfidenceConfig,
    MonotonicConfidenceTrainer,
)


def _config(**overrides):
    values = dict(
        num_heads=3,
        hidden_dim=16,
        depth=2,
        knots=4,
        batch_size=8,
        replay_capacity=32,
        anchor_capacity=0,
        background_fraction=0.0,
        randomized_prior_scale=0.1,
        inertia_warmup_updates=1,
        reference_refresh_updates=2,
    )
    values.update(overrides)
    return MonotonicConfidenceConfig(**values)


def test_selected_width_and_two_tier_anchor_policy_are_defaults():
    cfg = MonotonicConfidenceConfig()
    assert cfg.num_heads == 3
    assert cfg.hidden_dim == 256
    assert cfg.replay_capacity == 32_768
    assert cfg.anchor_capacity == 2_048
    assert cfg.anchor_warmup_rows == 18_432
    assert cfg.anchor_slow_fraction == 0.20
    assert cfg.anchor_half_life_rows == 55_296
    assert cfg.anchor_slow_half_life_multiplier == 5.0
    assert cfg.anchor_replacement


def test_initialization_is_pessimistic_low_confidence_and_heads_are_distinct():
    trainer = MonotonicConfidenceTrainer(16, _config(), device="cpu", seed=3)
    features = torch.randn(7, 16)
    features[:, -2:] = torch.rand(7, 2)
    prediction = trainer.predict(features, ema=False)
    confidence = trainer.confidence(
        features, ema=False, temper_by_support=False,
    )
    assert (prediction > 2.9).all()
    torch.testing.assert_close(
        confidence, torch.full_like(confidence, 0.01), atol=1e-6, rtol=0.0,
    )
    # Frozen derivative priors make time slopes distinct while preserving the
    # shared conservative baseline at zero TTH.
    assert prediction.std(dim=0).mean() > 0.0
    zero_time = features.clone()
    zero_time[:, -2:] = 0.0
    zero_prediction = trainer.predict(zero_time, ema=False)
    torch.testing.assert_close(
        zero_prediction, torch.full_like(zero_prediction, 3.0),
    )


def test_predictions_are_coordinatewise_nonincreasing_in_each_active_tth():
    trainer = MonotonicConfidenceTrainer(16, _config(), device="cpu", seed=5)
    low = torch.randn(9, 16)
    low[:, -2:] = torch.rand(9, 2) * 0.4 + 0.1
    high_left = low.clone()
    high_left[:, -2] += 0.3
    high_right = low.clone()
    high_right[:, -1] += 0.3
    assert torch.all(trainer.predict(high_left, ema=False) <= trainer.predict(low, ema=False))
    assert torch.all(trainer.predict(high_right, ema=False) <= trainer.predict(low, ema=False))


def test_local_expert_trunk_preserves_initialization_and_tth_monotonicity():
    cfg = _config(
        trunk_architecture="local_experts",
        expert_count=4,
        expert_top_k=2,
        expert_hidden_dim=8,
    )
    trainer = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=6)
    low = torch.randn(64, 16)
    low[:, -2:] = torch.rand(64, 2) * 0.4 + 0.1
    high = low.clone()
    high[:, -2:] += 0.3
    prediction = trainer.predict(low, ema=False)
    confidence = trainer.confidence(
        low, ema=False, temper_by_support=False,
    )
    assert (prediction > 2.9).all()
    torch.testing.assert_close(
        confidence, torch.full_like(confidence, 0.01), atol=1e-6, rtol=0.0,
    )
    assert torch.all(trainer.predict(high, ema=False) <= prediction)
    routing = trainer.routing_statistics(low)
    assert routing is not None
    assert routing["active_experts"] >= 2.0


def test_local_expert_router_trains_and_survives_state_round_trip():
    cfg = _config(
        trunk_architecture="local_experts",
        expert_count=4,
        expert_top_k=2,
        expert_hidden_dim=8,
        randomized_prior_scale=0.0,
    )
    trainer = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=8)
    before = [
        head.trunk.router.weight.detach().clone()
        for head in trainer.live.heads
    ]
    features = torch.randn(16, 16)
    features[:, -2:] = torch.rand(16, 2)
    stats = trainer.observe_targets(features, torch.zeros(16, 4), actor_step=1)
    assert torch.isfinite(torch.tensor(stats["loss_expert_routing"]))
    assert any(
        not torch.equal(old, head.trunk.router.weight)
        for old, head in zip(before, trainer.live.heads)
    )

    restored = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=99)
    restored.load_state_dict(trainer.state_dict())
    torch.testing.assert_close(restored.predict(features), trainer.predict(features))
    assert restored.routing_statistics(features) == trainer.routing_statistics(features)


def test_observable_confidence_targets_and_state_round_trip():
    cfg = _config(randomized_prior_scale=0.0)
    trainer = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=7)
    features = torch.randn(12, 16)
    features[:, -2:] = torch.rand(12, 2)
    # Initial prediction is approximately 3, so half the rows are observed
    # correct and half incorrect under the configured normalized tolerance.
    targets = torch.full((12, 4), 3.0)
    targets[6:] = 0.0
    stats = trainer.observe_targets(features, targets, actor_step=11)
    assert stats["correct_rate"] == 0.5
    assert (trainer.recent.calibration_head[:12] >= 0).all()
    assert (trainer.recent.calibration_head[:12] < cfg.num_heads).all()
    assert len(trainer.recent) == 12
    assert trainer.updates == 1

    state = trainer.state_dict()
    restored = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=99)
    restored.load_state_dict(state)
    query = torch.randn(4, 16)
    query[:, -2:] = torch.rand(4, 2)
    torch.testing.assert_close(restored.predict(query), trainer.predict(query))
    torch.testing.assert_close(
        restored.confidence(query), trainer.confidence(query),
    )


def test_support_tempers_confidence_away_from_memory():
    trainer = MonotonicConfidenceTrainer(
        16, _config(randomized_prior_scale=0.0), device="cpu", seed=9,
    )
    known = torch.zeros(8, 16)
    known[:, -2:] = 0.5
    trainer.observe_targets(known, torch.zeros(8, 4), actor_step=1)
    far = torch.full((2, 16), 20.0)
    far[:, -2:] = 0.5
    known_confidence = trainer.confidence(known).mean()
    far_confidence = trainer.confidence(far).mean()
    assert known_confidence > far_confidence


def test_elastic_anchor_locally_refreshes_an_old_exact_label():
    memory = ElasticAnchorMemory(
        1, 16, 3, slow_fraction=0.0, half_life_rows=1,
        slow_half_life_multiplier=5.0, match_radius=0.5,
        replacement=True,
    )
    feature = torch.zeros(1, 16)
    feature[:, -2:] = 0.5
    memory.seed(feature, torch.zeros(1, 4), torch.zeros(3, 1, 4))
    memory.rows_seen = 100
    memory.offer(
        feature, torch.ones(1, 4), torch.ones(3, 1, 4),
        scale=torch.ones(16), generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(memory.targets[0].float(), torch.ones(4))
    assert memory.local_replacements == 1


def test_anchor_losses_and_memory_survive_state_round_trip():
    cfg = _config(
        randomized_prior_scale=0.0,
        anchor_capacity=4,
        anchor_warmup_rows=0,
        anchor_batch_size=4,
        anchor_slow_fraction=0.25,
        anchor_half_life_rows=16,
    )
    trainer = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=13)
    features = torch.randn(8, 16)
    features[:, -2:] = 0.5
    stats = trainer.observe_targets(features, torch.zeros(8, 4), actor_step=1)
    assert trainer.anchors is not None
    assert len(trainer.anchors) == 4
    # The warmup-crossing observation finishes before anchors are seeded; the
    # newly seeded bank contributes beginning with the following update.
    assert stats["loss_anchor_exact"] == 0.0
    assert trainer._update_once()["loss_anchor_exact"] > 0.0

    state = trainer.state_dict()
    restored = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=99)
    restored.load_state_dict(state)
    assert restored.anchors is not None
    assert restored.anchors.diagnostics() == trainer.anchors.diagnostics()
    torch.testing.assert_close(
        restored.anchors.features[:4], trainer.anchors.features[:4],
    )


def test_anchor_loss_force_halves_after_one_half_life():
    cfg = _config(
        lr=0.0,
        inertia_weight=0.0,
        randomized_prior_scale=0.0,
        anchor_capacity=1,
        anchor_warmup_rows=0,
        anchor_batch_size=1,
        anchor_slow_fraction=0.0,
        anchor_half_life_rows=8,
    )
    trainer = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=21)
    features = torch.zeros(1, 16)
    features[:, -2:] = 0.5
    trainer.observe_targets(features, torch.zeros(1, 4), actor_step=1)
    assert trainer.anchors is not None

    fresh = trainer._update_once()["loss_anchor_exact"]
    trainer.anchors.rows_seen = (
        int(trainer.anchors.birth_row[0]) + cfg.anchor_half_life_rows
    )
    aged = trainer._update_once()["loss_anchor_exact"]
    torch.testing.assert_close(
        torch.tensor(aged), torch.tensor(0.5 * fresh), rtol=2e-3, atol=1e-6,
    )


def test_default_anchor_bank_waits_for_warmup_then_seeds_from_replay():
    cfg = _config(
        randomized_prior_scale=0.0,
        anchor_capacity=4,
        anchor_warmup_rows=6,
    )
    trainer = MonotonicConfidenceTrainer(16, cfg, device="cpu", seed=22)
    features = torch.randn(6, 16)
    features[:, -2:] = 0.5
    targets = torch.zeros(6, 4)

    trainer.observe_targets(features[:5], targets[:5], actor_step=1)
    assert trainer.anchors is not None
    assert len(trainer.anchors) == 0
    trainer.observe_targets(features[5:], targets[5:], actor_step=2)
    assert len(trainer.anchors) == 4
    assert trainer.rows_observed == 6
