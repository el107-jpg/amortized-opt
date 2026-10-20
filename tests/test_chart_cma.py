from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cma")

from isaac_rl.chart_cma import (ChartCMA, plan_population, aggregate_population,
    plan_independent_population)
from isaac_rl.chart_cma_sampling import IndependentRhythmSampler
from isaac_rl.conditional_min_time import ConditionalMinimumTimeChart
from isaac_rl.min_time_assignment import MinimumTimeChart
from amortized_opt.rhythm import Hit, RhythmClip


def chart():
    times = np.full((2, 2, 2), .5)
    base = MinimumTimeChart(("snare", "hihat"), times, np.full_like(times, 8, int),
                            np.full_like(times, 8, int), np.ones_like(times, bool))
    return ConditionalMinimumTimeChart(base, [-1., 1.])


def panel():
    hits = [Hit(.3+i*.3, .3+i*.3, i % 2, ("snare", "hihat")[i % 2], "left", .5)
            for i in range(4)]
    return [RhythmClip("a", hits, 1.5), RhythmClip("b", hits, 1.5)]


def test_zero_residual_preserves_initialization_and_invalid_proposals_do_not_mutate():
    c = chart()
    opt = ChartCMA(c, population=4, seed=4)
    np.testing.assert_array_equal(opt.candidate_chart(np.zeros(24)).get_parameters(), c.get_parameters())
    with pytest.raises(ValueError):
        opt.candidate_chart(np.full(24, -10.))
    np.testing.assert_array_equal(opt.initial, c.get_parameters())
    np.testing.assert_array_equal(c.get_parameters(), np.full(24, .5))


def test_cma_tunes_initialized_unknown_cells_and_rejects_old_dimension_resume(tmp_path):
    old = chart()
    old.times[1, 0, 1] = np.inf
    old = ConditionalMinimumTimeChart(old, old.proxy_values)
    optimizer = ChartCMA(old, population=4, seed=8)
    optimizer.save(tmp_path/'old.pkl')
    expanded = old.with_initialized_unknown(seed=7)
    assert len(expanded.get_parameters()) == 24
    with pytest.raises(ValueError, match='match'):
        ChartCMA.load(tmp_path/'old.pkl', expanded)
    optimizer = ChartCMA(expanded, population=4, sigma=.05, seed=8)
    candidate = optimizer.candidate_chart(np.full(24, .1))
    assert candidate.minimum(1, 0, 1) > expanded.minimum(1, 0, 1)
    optimizer.ask()
    optimizer.save(tmp_path/'expanded.pkl')
    restored, _ = ChartCMA.load(tmp_path/'expanded.pkl', expanded)
    np.testing.assert_array_equal(restored.ask(), optimizer.ask())


def test_old_resume_does_not_enable_new_unknown_initialization(tmp_path):
    import json
    from scripts.train_chart_cma import parse_args
    (tmp_path/'config.json').write_text(json.dumps({'args': {'seed': 42}}))
    args = parse_args(['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl')])
    assert not args.initialize_unknown
    assert args.seed == 42
    assert parse_args(['--out', str(tmp_path/'fresh')]).initialize_unknown


def test_resume_after_ask_reuses_pending_candidates_and_private_rng(tmp_path):
    opt = ChartCMA(chart(), population=4, sigma=.05, seed=5)
    asked = opt.ask()
    opt.save(tmp_path/"last.pkl", panel_rng={"value": 3})
    restored, extra = ChartCMA.load(tmp_path/"last.pkl", chart())
    assert extra["panel_rng"] == {"value": 3}
    np.testing.assert_array_equal(restored.ask(), asked)
    rewards = -np.square(asked-.1).mean(-1)
    stats = opt.tell(rewards, -.01)
    resumed_stats = restored.tell(rewards, -.01)
    assert stats == resumed_stats
    # Environment-global NumPy randomness must not alter optimizer proposals.
    np.random.seed(100); np.random.randn(100)
    next_a = opt.ask()
    np.random.seed(999); np.random.randn(200)
    np.testing.assert_array_equal(restored.ask(), next_a)


def test_resume_after_completed_generation_and_overlay_guard(tmp_path):
    opt = ChartCMA(chart(), population=4, sigma=.05, seed=6)
    x = opt.ask(); opt.tell(-np.square(x).mean(-1), 0.)
    opt.save(tmp_path/"last.pkl")
    restored, _ = ChartCMA.load(tmp_path/"last.pkl", chart())
    assert restored.generation == 1
    np.testing.assert_array_equal(restored.ask(), opt.ask())
    changed = chart(); changed.set_parameters(changed.get_parameters()*1.1)
    with pytest.raises(ValueError, match="match"):
        ChartCMA.load(tmp_path/"last.pkl", changed)
    opt.es.mean[:] = -10.
    opt.save(tmp_path/"invalid.pkl")
    with pytest.raises(ValueError, match="search center"):
        ChartCMA.load(tmp_path/"invalid.pkl", chart())


def test_rejection_sampling_at_floor_and_resume_keeps_pending_population(tmp_path):
    c = chart()
    a = c.anchors
    a[0, 0, 0] = [.02, .02, .02]
    c.set_anchors(a)
    opt = ChartCMA(c, population=8, sigma=.05, seed=13, max_resamples=256)
    samples = opt.ask()
    assert opt.sampling_rejections > 0
    for x in samples:
        opt.candidate_chart(x)  # Every feasible candidate survives the vertex gate.
    opt.save(tmp_path/"last.pkl")
    resumed, _ = ChartCMA.load(tmp_path/"last.pkl", c)
    np.testing.assert_array_equal(resumed.ask(), samples)
    assert resumed.sampling_rejections == opt.sampling_rejections
    # pycma accepts the replacement ask(1) samples in the generation update.
    assert resumed.tell(np.arange(8, dtype=float), -1.)["valid_candidates"] == 8


def test_invalid_rewards_rejected_and_incumbent_never_promoted():
    opt = ChartCMA(chart(), population=4, seed=7)
    opt.ask()
    stats = opt.tell(np.full(4, np.nan), 0.)
    assert stats["rejected_candidates"] == 4
    assert not stats["incumbent_changed"]
    assert not stats["optimizer_updated"]
    assert opt.es.countiter == 0
    np.testing.assert_array_equal(opt.es.mean, np.zeros(24))
    np.testing.assert_array_equal(opt.incumbent, np.zeros(24))


def test_incumbent_compared_on_current_panel_not_historical_reward():
    opt = ChartCMA(chart(), population=4, sigma=.01, seed=8, regularization=0.)
    x = opt.ask()
    opt.tell([1, 2, 3, 4], 5)
    np.testing.assert_array_equal(opt.incumbent, np.zeros(24))
    x = opt.ask()
    opt.tell([1, 2, 3, 4], 0)
    np.testing.assert_array_equal(opt.incumbent, x[3])


def test_cma_improves_actual_black_box_objective():
    opt = ChartCMA(chart(), population=16, sigma=.1, seed=9, regularization=0.)
    target = np.linspace(-.1, .2, 24)
    initial = -np.square(target).sum()
    for _ in range(30):
        x = opt.ask()
        rewards = -np.square(x-target).sum(-1)
        incumbent_reward = -np.square(opt.incumbent-target).sum()
        opt.tell(rewards, incumbent_reward)
    assert -np.square(opt.incumbent-target).sum() > initial+.05
    assert opt.es.countiter == 30


def test_population_deduplication_preserves_candidate_and_scenario_scores():
    opt = ChartCMA(chart(), population=4, seed=10)
    candidates = np.zeros((4, 24))
    candidates[3] = -10.  # Invalid curve must never trigger physical evaluation.
    plan = plan_population(opt, candidates, panel(), np.full((2, 2), .2))
    assert (plan["gather"][3] == -1).all()
    np.testing.assert_array_equal(plan["gather"][0], plan["gather"][1])
    assert plan["unique_rollouts"] < plan["logical_rollouts"]
    # Identical rhythms on different scenario lanes are not conflated.
    assert plan["gather"][0, 0] != plan["gather"][0, 1]
    values = aggregate_population(plan, np.arange(plan["unique_rollouts"]))
    assert values[0] == values[1] == values[2] == values[-2]
    assert np.isnan(values[3])
    assert plan["distinct_schedules"] == 1


def test_plateau_expansion_and_input_errors():
    opt = ChartCMA(chart(), population=4, seed=11, plateau_growth=2.)
    opt.ask(); stats = opt.tell(np.ones(4), 1., distinct_schedules=1)
    assert stats["sigma"] > .15
    with pytest.raises(RuntimeError):
        opt.tell(np.ones(4), 1.)
    with pytest.raises(ValueError):
        ChartCMA(chart(), population=1)
    with pytest.raises(ValueError):
        aggregate_population({"unique_rollouts": 1}, [np.nan])


def test_episode_packing_preserves_hands_pairs_and_all_requests():
    from isaac_rl.chart_cma import assigned_clip
    c = panel()[0]
    c.hits.append(replace(c.hits[-1], drum=0, drum_name="snare"))
    selected = assigned_clip(c, [1, 0, 1, 0])
    assert len(selected.hits) == 5
    last = [h for h in selected.hits if h.time == c.hits[-1].time]
    assert {h.hand for h in last} == {"left", "right"}


def test_dry_run_has_no_isaac_launch(tmp_path, capsys):
    from scripts.train_chart_cma import main
    main(["--out", str(tmp_path/"unused"), "--population", "4", "--scenarios", "2",
          "--sampling", "matched", "--validation-scenarios", "8",
          "--groups", "4", "--envs", "8", "--dry-run"])
    assert not (tmp_path/"unused").exists()
    assert '"optimizer": "CMA-ES"' in capsys.readouterr().out


def test_independent_population_has_exactly_one_episode_per_offspring():
    opt = ChartCMA(chart(), population=4, seed=32)
    candidates = np.zeros((4, 24))
    one = panel()[0]
    plan = plan_independent_population(opt, candidates, [one]*4, np.full((2, 2), .2))
    assert plan['gather'].shape == (4, 1)
    assert plan['unique_rollouts'] == plan['logical_rollouts'] == 4
    assert plan['rollouts_per_offspring'] == 1
    np.testing.assert_array_equal(aggregate_population(plan, [.1, .2, .3, .4]), [.1, .2, .3, .4])
    with pytest.raises(ValueError, match='one rhythm'):
        plan_independent_population(opt, candidates, [one], np.full((2, 2), .2))


def test_independent_updates_export_mean_and_resume_exactly(tmp_path):
    opt = ChartCMA(chart(), population=8, sigma=.05, seed=33)
    samples = opt.ask()
    # Distinct scenario rewards have no old-incumbent promotion comparison.
    opt.tell_independent(np.arange(8, dtype=float))
    np.testing.assert_array_equal(opt.incumbent, opt.es.mean)
    assert not np.array_equal(opt.incumbent, samples[-1])
    opt.candidate_chart(opt.incumbent)
    opt.save(tmp_path/'independent.pkl', sampling='independent')
    resumed, extra = ChartCMA.load(tmp_path/'independent.pkl', chart())
    assert extra['sampling'] == 'independent'
    np.testing.assert_array_equal(resumed.ask(), opt.ask())
    before = resumed.es.mean.copy()
    stats = resumed.tell_independent(np.full(8, np.nan))
    assert not stats['optimizer_updated']
    np.testing.assert_array_equal(resumed.es.mean, before)


def test_independent_sampler_balances_exposure_preserves_groups_and_rng():
    c = chart()
    clips = panel()
    sampler = IndependentRhythmSampler(clips, 4, c)
    first = sampler.sample(20, np.random.default_rng(2))
    targets = sampler.last_targets.copy()
    second = sampler.sample(20, np.random.default_rng(2))
    assert first == second
    assert np.bincount(targets).tolist() == [10, 10]
    for clip, target in zip(first, targets):
        assert len(clip.hits) == 4
        assert clip.hits[0].time == 1.
        assert int(target) in [h.drum for h in clip.hits]
        np.testing.assert_allclose(np.diff([h.time for h in clip.hits]), [.3, .3, .3])


def test_independent_defaults_and_matched_resume_modes(tmp_path):
    import json
    from scripts.train_chart_cma import parse_args
    a = parse_args(['--out', str(tmp_path/'fresh')])
    assert a.sampling == 'independent' and a.population == a.envs == 4096
    assert a.scenarios == 1 and a.validation_scenarios == 0
    (tmp_path/'config.json').write_text(json.dumps({'args': dict(population=32,
        scenarios=16, validation_scenarios=8, sampling='matched')}))
    a = parse_args(['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl')])
    assert a.sampling == 'matched' and a.population == 32 and a.scenarios == 16


def test_model_rnea_context_honors_cache_only_and_restores_method(monkeypatch):
    import isaac_rl.rnea_cache as module
    from amortized_opt.rnea import BatchedRNEA
    monkeypatch.setenv("AMORTIZED_RNEA_CACHE_ONLY", "1")
    def forbidden(cls, *args, **kwargs):
        raise AssertionError("Live Pinocchio build in cache-only Isaac run")
    monkeypatch.setattr(BatchedRNEA, "from_urdf", classmethod(forbidden))
    marker = object()
    monkeypatch.setattr(module, "load_cached", lambda **kwargs: marker)
    with module.rnea_from_cache_if_needed():
        assert BatchedRNEA.from_urdf() is marker
    assert BatchedRNEA.from_urdf.__func__ is forbidden


def test_microbatch_controller_matches_existing_committed_controller():
    import torch
    import amortized_opt as A
    from isaac_rl.assignment_parallel import ParallelCommittedMPC
    from isaac_rl.chart_cma_isaac import ChunkedChartMPC
    from isaac_rl.mpc import MPCConfig
    from isaac_rl.schedule import BatchedSchedule, ScheduleConfig
    from isaac_rl.rnea_cache import load_cached
    torch.manual_seed(29)
    kit = A.DrumKit.from_layout()
    model = A.build_model(A.ModelConfig(max_horizon=25, terminal_head=True,
        d_model=32, d_ctx=32, dec_heads=4, ctx_heads=4, dec_layers=1,
        ctx_layers=1, num_future=1), "cpu").eval()
    cost = A.FusedCost(A.CostConfig(), max_horizon=25, device="cpu", rnea=load_cached())
    physical = RhythmClip("control", [Hit(.3, .3, 0, "snare", "left", .5),
                                    Hit(.4, .4, 2, "tom1", "right", .5)], .6)
    controllers = []
    for cls, fast in ((ParallelCommittedMPC, False), (ChunkedChartMPC, False), (ChunkedChartMPC, True)):
        schedule = BatchedSchedule(4, ScheduleConfig(horizon=25, num_future=1), device="cpu")
        schedule.reset(torch.arange(4), [physical]*4, [kit]*4)
        options = dict(decode_chunk=2, plan_diagnostics=not fast) if cls is ChunkedChartMPC else {}
        controllers.append(cls(model, schedule, MPCConfig(horizon=25, cmd_lead_s=None,
            fresh_accel_init=True, fresh_terminal_every_tick=True, amp_dtype=None),
            device="cpu", kit_names=kit.names, descent_cost=cost, **options))
    from amortized_opt.constants import ready_joint_pos
    q = torch.tensor(ready_joint_pos(), dtype=torch.float32)[None].repeat(4, 1)
    v = torch.zeros_like(q)
    with torch.no_grad():
        for control in controllers:
            control.reset(torch.arange(4), q)
        a, ai = controllers[0].plan(q, v)
        b, bi = controllers[1].plan(q, v)
        c, ci = controllers[2].plan(q, v)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ai, bi)
    torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ai, ci)
    torch.testing.assert_close(controllers[0].reference, controllers[1].reference, atol=1e-6, rtol=1e-6)
    assert controllers[1].decoder_passes.tolist() == [1, 1, 1, 1]
    torch.testing.assert_close(controllers[0].reference, controllers[2].reference, atol=1e-6, rtol=1e-6)


def test_execution_speed_flags_override_resume_without_changing_learning(tmp_path):
    import json
    from scripts.train_chart_cma import parse_args
    (tmp_path/'config.json').write_text(json.dumps({'args': dict(sampling='independent',
        decode_chunk=64, population=4096, envs=4096)}))
    a = parse_args(['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl'),
        '--decode-chunk', '512', '--collision-chunk', '1024', '--no-plan-diagnostics', '--planning-workers', '4',
        '--rollout-devices', 'cuda:0', 'cuda:2'])
    assert a.decode_chunk == 512 and a.collision_chunk == 1024 and a.planning_workers == 4
    assert not a.plan_diagnostics and a.population == 4096
    assert a.rollout_devices == ['cuda:0', 'cuda:2']
