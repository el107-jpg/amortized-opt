import pytest
import torch

from amortized_opt.violation_critic import (
    PriorityAuditCoreset,
    RecentCriticReplay,
    ViolationCriticConfig,
    ViolationCriticTrainer,
    build_critic_features,
    calibrate_manageability_rule,
    conservative_score_from_prediction,
    critic_loss_per_head_row,
    critic_feature_dim,
    manageability_metrics,
    residuals_from_gate,
    robust_inverse,
    robust_transform,
)


def _config(**kwargs):
    base = dict(
        mode="shadow",
        num_heads=5,
        hidden_dim=16,
        depth=2,
        batch_size=8,
        updates_per_step=1,
        recent_capacity=16,
        audit_capacity=4,
    )
    base.update(kwargs)
    return ViolationCriticConfig(**base)


def test_residuals_are_positive_bad_and_transform_round_trips():
    cfg = _config()
    gate = torch.zeros(2, 8)
    gate[:, 4:] = torch.tensor(
        [
            [0.030, -0.010, 0.030, 10.0],
            [0.010, 0.000, 0.000, 2.0],
        ]
    )
    residual = residuals_from_gate(gate, cfg)
    assert torch.all(residual[0] > 0.0)
    assert torch.all(residual[1] < 0.0)

    transformed = robust_transform(residual, cfg.scales)
    restored = robust_inverse(transformed, cfg.scales)
    torch.testing.assert_close(restored, residual)


def test_boundary_loss_prefers_correct_sign_at_equal_regression_error():
    cfg = _config(
        boundary_loss_weight=1.0,
        boundary_temperature=0.5,
        boundary_target_band=1.0,
        underprediction_multiplier=1.0,
    )
    target = torch.tensor([[0.10, -0.10, 0.10, -0.10]])
    correct = target.unsqueeze(0).clone()
    wrong = (-target).unsqueeze(0)
    correct_total, _, correct_boundary = critic_loss_per_head_row(
        correct, target, cfg,
    )
    wrong_total, _, wrong_boundary = critic_loss_per_head_row(wrong, target, cfg)
    assert torch.all(wrong_boundary > correct_boundary)
    assert torch.all(wrong_total > correct_total)


def test_underprediction_multiplier_penalizes_unsafe_direction():
    cfg = _config(
        boundary_loss_weight=0.0,
        underprediction_multiplier=2.0,
    )
    target = torch.full((1, 4), 0.20)
    under = torch.full((1, 1, 4), 0.10)
    over = torch.full((1, 1, 4), 0.30)
    under_total, _, _ = critic_loss_per_head_row(under, target, cfg)
    over_total, _, _ = critic_loss_per_head_row(over, target, cfg)
    torch.testing.assert_close(under_total, 2.0 * over_total)


def test_features_are_only_state_raw_goals_and_atomic_timing_pair():
    batch, n_dof = 3, 4
    q0 = torch.randn(batch, n_dof, requires_grad=True)
    qd0 = torch.randn(batch, n_dof, requires_grad=True)
    goal_pos = torch.randn(batch, 2, 3, requires_grad=True)
    hit_s = torch.tensor(
        [[0.10, 0.08], [0.06, 0.0], [0.0, 0.10]], requires_grad=True,
    )
    features = build_critic_features(
        q0=q0, qd0=qd0, goal_pos=goal_pos, goal_tth=hit_s,
    )
    assert features.shape == (batch, critic_feature_dim(n_dof))
    assert not features.requires_grad
    goal_offset = 2 * n_dof
    timing_offset = goal_offset + 6
    # Inactive slots are zeroed even if their context contains a stale goal.
    assert (features[1, goal_offset + 3: goal_offset + 6] == 0.0).all()
    assert (features[2, goal_offset: goal_offset + 3] == 0.0).all()
    torch.testing.assert_close(features[:, timing_offset: timing_offset + 2], hit_s)
    assert torch.isfinite(features).all()


def test_trainer_updates_only_critic_and_round_trips_state():
    cfg = _config(ema_decay=0.5)
    input_dim = 12
    trainer = ViolationCriticTrainer(input_dim, cfg, device="cpu", seed=9)
    actor_tensor = torch.randn(10, input_dim, requires_grad=True)
    features = actor_tensor.detach()
    gate = torch.zeros(10, 8)
    gate[:, 4:] = torch.tensor([0.03, -0.01, 0.03, 10.0])

    before = [p.detach().clone() for p in trainer.live.parameters()]
    stats = trainer.observe_gate(
        features, gate, actor_step=17,
    )
    assert actor_tensor.grad is None
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.live.parameters()))
    assert stats["recent_size"] == 10
    assert len(trainer.audit) <= cfg.audit_capacity

    query = torch.randn(4, input_dim)
    expected = trainer.predict(query)
    state = trainer.state_dict()
    assert "source" not in state["recent"]
    assert "source" not in state["audit"]
    restored = ViolationCriticTrainer(input_dim, cfg, device="cpu", seed=1234)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.predict(query), expected)
    assert len(restored.recent) == len(trainer.recent)
    assert len(restored.audit) == len(trainer.audit)
    assert restored.updates == trainer.updates

    legacy = dict(state, format_version=1)
    with pytest.raises(ValueError, match="unsupported timing critic checkpoint"):
        restored.load_state_dict(legacy)

    incompatible = ViolationCriticTrainer(
        input_dim, _config(torque_limit=4.0), device="cpu", seed=1,
    )
    with pytest.raises(ValueError, match="incompatible.*torque_limit"):
        incompatible.load_state_dict(state)


def test_independent_heads_and_conservative_scores():
    trainer = ViolationCriticTrainer(7, _config(), device="cpu", seed=4)
    pred = trainer.predict(torch.randn(5, 7), ema=False)
    assert pred.shape == (5, 5, 4)
    assert pred.std(dim=0).mean() > 0.0
    score = trainer.ema.conservative_score(torch.randn(5, 7))
    assert score.shape == (5,)


def test_randomized_priors_are_frozen_distinct_and_preserve_learned_initialization():
    plain = ViolationCriticTrainer(
        7, _config(randomized_prior_scale=0.0), device="cpu", seed=4,
    )
    prior = ViolationCriticTrainer(
        7, _config(randomized_prior_scale=0.25), device="cpu", seed=4,
    )
    for plain_head, prior_head in zip(plain.live.heads, prior.live.heads):
        for a, b in zip(plain_head.net.parameters(), prior_head.net.parameters()):
            torch.testing.assert_close(a, b)
        assert prior_head.prior is not None
        assert all(not p.requires_grad for p in prior_head.prior.parameters())
    query = torch.randn(5, 7)
    prior_outputs = torch.stack([
        head.prior(query) for head in prior.live.heads
    ])
    assert prior_outputs.std(dim=0).mean() > 0.0

    frozen_before = [
        p.clone() for head in prior.live.heads for p in head.prior.parameters()
    ]
    gate = torch.zeros(8, 8)
    prior.observe_gate(
        torch.randn(8, 7), gate, actor_step=1,
    )
    frozen_after = [
        p for head in prior.live.heads for p in head.prior.parameters()
    ]
    for before, after in zip(frozen_before, frozen_after):
        torch.testing.assert_close(before, after)
    expected = prior.predict(query)
    restored = ViolationCriticTrainer(
        7, _config(randomized_prior_scale=0.25), device="cpu", seed=99,
    )
    restored.load_state_dict(prior.state_dict())
    torch.testing.assert_close(restored.predict(query), expected)


def test_conservative_score_promotes_mixed_precision_before_quantile():
    trainer = ViolationCriticTrainer(7, _config(), device="cpu", seed=4)
    trainer.ema.to(dtype=torch.bfloat16)
    score = trainer.ema.conservative_score(
        torch.randn(5, 7, dtype=torch.bfloat16),
    )
    assert score.shape == (5,)
    assert score.dtype == torch.float32


def test_single_head_mc_dropout_is_stochastic_and_preserves_global_rng():
    cfg = _config(
        num_heads=1, dropout_probability=0.2, mc_dropout_samples=8,
        bootstrap_probability=1.0,
    )
    trainer = ViolationCriticTrainer(7, cfg, device="cpu", seed=4)
    query = torch.randn(5, 7)
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)
    prediction = trainer.predict(query)
    actual_next = torch.rand(4)
    assert prediction.shape == (8, 5, 4)
    assert prediction.std(dim=0).mean() > 0.0
    torch.testing.assert_close(actual_next, expected_next)
    assert trainer.conservative_score(query).shape == (5,)


def test_heterogeneous_heads_have_distinct_architectures_and_round_trip():
    cfg = _config(
        num_heads=3, hidden_dim=32, depth=2,
        dropout_probability=0.1, mc_dropout_samples=3,
        head_architecture_mix="heterogeneous",
    )
    trainer = ViolationCriticTrainer(17, cfg, device="cpu", seed=12)
    counts = [
        sum(p.numel() for p in head.parameters() if p.requires_grad)
        for head in trainer.live.heads
    ]
    assert len(set(repr(head.net) for head in trainer.live.heads)) == 3
    assert all(count > 0 for count in counts)
    query = torch.randn(6, 17)
    prediction = trainer.predict(query)
    assert prediction.shape == (9, 6, 4)
    state = trainer.state_dict()
    restored = ViolationCriticTrainer(17, cfg, device="cpu", seed=99)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.predict(query), trainer.predict(query))


def test_manageability_calibration_adds_only_pessimism_and_meets_budget():
    # All heads rank rows the same way, while increasing offsets make higher
    # quantiles more pessimistic.  Rows 2 and 4 are the unsafe false accepts at
    # the uncalibrated zero threshold.
    base = torch.tensor([-0.8, -0.6, -0.5, -0.3, -0.2, 0.2])
    prediction = torch.stack([
        base[:, None].expand(-1, 4) + offset
        for offset in (-0.10, -0.05, 0.0, 0.05, 0.10)
    ])
    target = torch.full((6, 4), -0.2)
    target[2] = 0.1
    target[4] = 0.1

    fit = calibrate_manageability_rule(
        prediction, target,
        quantiles=(0.5, 0.8, 1.0),
        max_false_manageable_rate=1.0 / 6.0,
    )
    selected = fit["selected"]
    assert selected["threshold"] <= 0.0
    assert selected["safety_margin"] >= 0.0
    assert selected["false_manageable_rate"] <= 1.0 / 6.0
    score = conservative_score_from_prediction(
        prediction, quantile=selected["quantile"],
    )
    metrics = manageability_metrics(score, target, threshold=selected["threshold"])
    assert metrics["fp"] == 1.0
    assert metrics["manageable_recall"] == pytest.approx(0.75)
    assert metrics["false_discovery_rate"] == pytest.approx(0.25)


def test_manageability_calibration_never_returns_positive_empty_fallback():
    prediction = torch.full((3, 4, 4), 0.25)
    target = torch.full((4, 4), -0.1)
    selected = calibrate_manageability_rule(
        prediction, target, quantiles=(0.8,),
    )["selected"]
    assert selected["threshold"] < 0.0
    assert selected["pred_manageable_rate"] == 0.0


def test_recent_ring_and_audit_coreset_are_bounded():
    replay = RecentCriticReplay(capacity=3, input_dim=2)
    generator = torch.Generator().manual_seed(1)
    replay.add(
        torch.arange(10, dtype=torch.float32).reshape(5, 2),
        torch.zeros(5, 4), actor_step=3,
    )
    assert len(replay) == 3
    x, y = replay.sample(20, generator=generator, device=torch.device("cpu"))
    assert x.shape == (3, 2)
    assert y.shape == (3, 4)

    audit = PriorityAuditCoreset(capacity=2, input_dim=2)
    audit.add(
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
        torch.zeros(3, 4), torch.tensor([1.0, 3.0, 2.0]),
        actor_step=4,
    )
    assert len(audit) == 2
    assert sorted(audit.priority[:2].tolist()) == pytest.approx([2.0, 3.0])


def test_replay_newest_fraction_and_online_timescale_update():
    replay = RecentCriticReplay(capacity=12, input_dim=2)
    for step in range(4):
        replay.add(
            torch.full((3, 2), float(step)), torch.zeros(3, 4),
            actor_step=step,
        )
    newest, _ = replay.sample(
        12, generator=torch.Generator().manual_seed(2),
        device=torch.device("cpu"), newest_fraction=0.25,
    )
    assert newest.shape[0] == 12
    assert (newest == 3.0).all()

    cfg = _config(
        num_heads=3, dropout_probability=0.1, mc_dropout_samples=2,
        replay_timescale_diversity=True,
    )
    trainer = ViolationCriticTrainer(2, cfg, device="cpu", seed=8)
    trainer.recent.load_state_dict(replay.state_dict())
    stats = trainer._update_once()
    assert stats["recent_size"] == 12
    assert trainer.updates == 1


def test_replay_sampling_and_audit_reserve_manageable_boundary_failure_strata():
    replay = RecentCriticReplay(capacity=30, input_dim=2)
    features = torch.arange(60, dtype=torch.float32).reshape(30, 2)
    targets = torch.full((30, 4), -1.0)
    targets[10:20, 0] = 0.25
    targets[20:, 0] = 2.0
    replay.add(
        features, targets, actor_step=1,
    )
    generator = torch.Generator().manual_seed(3)
    _x, manageable = replay.sample(
        30, generator=generator, device=torch.device("cpu"),
        manageable_fraction=1.0, near_failure_fraction=0.0,
        boundary_band=0.5,
    )
    assert (manageable.amax(dim=-1) <= 0.0).all()
    _x, near = replay.sample(
        30, generator=generator, device=torch.device("cpu"),
        manageable_fraction=0.0, near_failure_fraction=1.0,
        boundary_band=0.5,
    )
    assert ((near.amax(dim=-1) > 0.0) & (near.amax(dim=-1) <= 0.5)).all()

    audit = PriorityAuditCoreset(
        capacity=6, input_dim=2, boundary_band=0.5,
        fractions=(1 / 3, 1 / 3, 1 / 3),
    )
    # Fill with high-priority far failures, then verify lower-priority rows can
    # still claim their reserved manageable and near-boundary slots.
    order = torch.cat((torch.arange(20, 30), torch.arange(0, 20)))
    audit.add(
        features.index_select(0, order), targets.index_select(0, order),
        torch.linspace(10.0, 1.0, 30), actor_step=2,
    )
    score = audit.targets[: len(audit)].float().amax(dim=-1)
    counts = torch.tensor([
        (score <= 0.0).sum(),
        ((score > 0.0) & (score <= 0.5)).sum(),
        (score > 0.5).sum(),
    ])
    assert counts.tolist() == [2, 2, 2]


def test_invalid_config_and_gate_shape_fail_loudly():
    with pytest.raises(ValueError, match="mode"):
        ViolationCriticConfig(mode="maybe")
    with pytest.raises(ValueError, match="requires dropout"):
        ViolationCriticConfig(num_heads=1)
    with pytest.raises(ValueError, match="dropout_probability"):
        ViolationCriticConfig(dropout_probability=1.0)
    with pytest.raises(ValueError, match="head_architecture_mix"):
        ViolationCriticConfig(head_architecture_mix="random")
    with pytest.raises(ValueError, match="exactly three"):
        ViolationCriticConfig(
            num_heads=5, replay_timescale_diversity=True,
        )
    with pytest.raises(ValueError, match="boundary-loss"):
        ViolationCriticConfig(boundary_temperature=0.0)
    with pytest.raises(ValueError, match="boundary-loss"):
        ViolationCriticConfig(underprediction_multiplier=0.5)
    with pytest.raises(ValueError, match="decision_threshold"):
        ViolationCriticConfig(decision_threshold=0.1)
    with pytest.raises(ValueError, match="randomized_prior_scale"):
        ViolationCriticConfig(randomized_prior_scale=-0.1)
    with pytest.raises(ValueError, match="shape"):
        residuals_from_gate(torch.zeros(2, 4), _config())
