#!/usr/bin/env python3
"""Learn16 hand-plan scores from matched Isaac trials, with16→8→4 collection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit, Hit, RhythmClip
from amortized_opt.constants import ready_joint_pos
from amortized_opt.limits import urdf_position_limits
from isaac_rl.min_time_assignment import MinimumTimeChart
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler, planned_transition_coverage
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.hand_plan_ranker import (ACTION_BANK, HandPlanRanker, HandPlanRankerConfig,
    RankerReplay, build_ranker_batch, unpack_ranker_reports, ranking_loss,
    ranking_metrics, candidate_count, encode_clip, rank_clips, OBSERVATION_VERSION)
from isaac_rl.hand_plan_start_states import PostHitStateBank
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler,load_dataset_batches,require_matching_rollout_runtime
from isaac_rl.assignment_starts import pose_clearance
from isaac_rl.decoder_scene import GEOMETRY_VERSION
from scripts.train_chart_cma import parse_args as runtime_args, write_json


def save_torch(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def learning_rate(step, total_steps, peak=3e-4, floor=3e-5, warmup=100):
    if step <= warmup:
        return peak*step/warmup
    progress = min(1., (step-warmup)/max(1, total_steps-warmup))
    return floor+.5*(peak-floor)*(1+math.cos(math.pi*progress))


def load_cached_panel(folder, expected_decoder_sha, expected_first_hit_policy=None):
    config = json.loads((folder/'config.json').read_text())
    if config['checkpoint_sha256'] != expected_decoder_sha:
        raise ValueError('Cached diagnostic panel uses a different decoder')
    if config.get('simulation_geometry_version') != GEOMETRY_VERSION:
        raise ValueError('Cached outcomes use old/unverified simulator geometry; regenerate them')
    actual_timing=config.get('first_hit_policy','fixed_one_second')
    if expected_first_hit_policy is not None and actual_timing!=expected_first_hit_policy:
        raise ValueError('Cached outcomes use a different first-hit timing policy; regenerate them')
    saved = pickle.loads((folder/'planned_trials.pkl').read_bytes())
    reports = json.loads((folder/'episodes.json').read_text())
    count = len(saved['panel'])
    rewards, hit_rates = np.zeros((count, 16), np.float32), np.zeros((count, 16), np.float32)
    hits, requested = np.zeros((count, 16), np.int64), np.zeros((count, 16), np.int64)
    observed = np.zeros((count, 16), bool)
    for row, (context, ids) in enumerate(zip(saved['contexts'], saved['trial']['gather'])):
        for action, report_id in zip(context['actions'], ids):
            code = int(np.asarray(action)@np.array([8, 4, 2, 1]))
            report = reports[int(report_id)]
            observed[row, code] = True
            rewards[row, code], hit_rates[row, code] = report['reward'], report['hit_rate']
            hits[row, code], requested[row, code] = report['hits'], report['requested']
    return dict(observations=np.stack([encode_clip(c) for c in saved['panel']]),
        rewards=rewards, observed=observed, hit_rates=hit_rates,
        hits=hits, requested=requested, panel=saved['panel'],rollout_runtime=config.get('rollout_runtime'))


@torch.inference_mode()
def evaluate_cached(model, panel):
    scores = model(torch.as_tensor(panel['observations'])).numpy()
    stats = ranking_metrics(scores, panel['rewards'], panel['observed'], panel['hit_rates'])
    choices = np.where(panel['observed'], scores, -np.inf).argmax(1)
    rows = np.arange(len(choices))
    requested = int(panel['requested'][rows, choices].sum())
    stats.update(pooled_hit_rate=float(panel['hits'][rows, choices].sum()/requested),
        total_hits=int(panel['hits'][rows, choices].sum()), requested=requested,
        contexts=len(choices), data=panel.get('scope','held_out_ready_cached_simulator_outcomes'),
        no_training_feedback=True)
    return stats


def plot_history(out):
    path = out/'metrics.jsonl'
    if not path.exists():
        return
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = [r['iteration'] for r in rows]
    axes[0].plot(x, [100*r['pre_fit']['mean_selected_hit_rate'] for r in rows], label='Fresh training rhythms, selected')
    monitored = [r for r in rows if r['held_out']]
    if monitored:
        monitor_label=('Fixed IID varied-start dataset test split' if
            monitored[-1]['held_out'].get('data')=='held_out_iid_varied_start_dataset_test_split' else
            'Fixed ready-start held-out rhythms, cached outcomes')
        axes[0].plot([r['iteration'] for r in monitored], [100*r['held_out']['pooled_hit_rate'] for r in monitored],
                     label=monitor_label)
    axes[0].plot(x, [100*r['pre_fit']['mean_best_observed_hit_rate'] for r in rows], ':', label='Training best observed')
    axes[0].set_ylabel('Hit rate (%)'); axes[0].set_xlabel('Collection update')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    if monitored:
        axes[1].plot([r['iteration'] for r in monitored], [r['held_out']['mean_best_reward_predicted_rank'] for r in monitored])
    axes[1].set_xlabel('Collection update'); axes[1].set_ylabel('Chart-free neural rank of best observed reward plan')
    axes[1].set_title('Fixed held-out full legal banks; lower is better'); axes[1].grid(alpha=.2)
    fig.suptitle('Neural hand-plan ranking: actual latest model, no evaluation-based selection')
    temporary = out/'training.tmp.png'; fig.savefig(temporary, dpi=160)
    temporary.replace(out/'training.png'); plt.close(fig)


def make_start_bank(geometry=None):
    import amortized_opt as A
    from isaac_rl.rnea_cache import rnea_from_cache_if_needed
    with rnea_from_cache_if_needed():
        cost = A.FusedCost(A.CostConfig(), device='cpu', max_horizon=2)
    if geometry is not None:
        # Clearance must price the same stick capsules as the simulator.
        for index, hand in enumerate(('left','right')):
            tip = np.asarray(geometry['tip_offsets_local_m'][index])
            mount = np.asarray(geometry['mount_positions_local_m'][index])
            butt = 2*mount-tip
            shaft = np.linspace(butt, tip, int(cost.cfg.shaft_samples))
            cost.c_points[hand] = torch.as_tensor(np.vstack((tip,shaft,np.zeros(3))),dtype=torch.float32)
    lower, upper = urdf_position_limits(posture_limits=False)
    def clearance(q):
        return np.concatenate([pose_clearance(cost, torch.as_tensor(q[i:i+256], dtype=torch.float32)).numpy()
                               for i in range(0, len(q), 256)])
    return PostHitStateBank(lower, upper, clearance, tip_geometry=geometry)


def bootstrap_post_hits(evaluator, eligibility, bank, rng, out, repeats=8, rounds=3):
    kit = DrumKit.from_layout()
    for round_index in range(rounds):
        clips = []
        for hand in (0, 1):
            for drum in range(7):
                if not eligibility.allowed[hand, drum, drum]:
                    continue
                for trial in range(repeats):
                    hits = [Hit(t, t, drum, kit.names[drum], ('left', 'right')[hand],
                                float(rng.uniform(.3, .9))) for t in (1., 1.6)]
                    clips.append(RhythmClip(f'bootstrap_{hand}_{drum}_{trial}', hits, 1.9))
        def progress(done,total,tick,ticks):
            write_json(out/'progress.json', dict(phase='bootstrap_measured_post_hit_states',
                bootstrap_round=round_index+1, episodes=total, complete=done, tick=tick,ticks=ticks))
            print(f'[post-hit-bootstrap] round={round_index+1} tick={tick}/{ticks}', flush=True)
        reports = evaluator.evaluate(clips, progress, collect_post_hits=True)
        states = [s for r in reports for s in r.get('post_hit_states', [])]
        metrics = bank.add(states, rng)
        write_json(out/f'bootstrap_{round_index+1:02d}.json', dict(metrics=metrics, reports=reports))
        save_torch(out/'post_hit_bank.pt', bank.state_dict())
        print('[post-hit-bootstrap-result] '+json.dumps(metrics), flush=True)
        if len(bank.states) == 13:
            break
    if not bank.states:
        raise RuntimeError('No valid measured post-hit states; inspect bootstrap reports')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT/'runs/transition_time_no_history_4096_20261007')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initialize-from',type=Path,
                        help='Supervised dataset checkpoint: preserve model/optimizer/RNG and train-only replay')
    parser.add_argument('--contexts', type=int, default=256)
    parser.add_argument('--stages', type=int, nargs=3, default=[10, 20, 10],
                        help='Updates at16/8/4 candidates; zero skips a stage, e.g.40 0 0')
    parser.add_argument('--fit-steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--replay-capacity', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min-lr', type=float, default=3e-5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--start-mode', choices=('varied', 'ready'), default='varied')
    parser.add_argument('--rhythm-sampling',choices=('iid','coverage'),default='iid')
    parser.add_argument('--first-hit-policy',choices=('iid_random','tier_random','fixed_one_second'),default=None,
                        help='Random first deadline from the same speed range as subsequent gaps')
    parser.add_argument('--bootstrap-repeats', type=int, default=8)
    parser.add_argument('--bootstrap-rounds', type=int, default=3)
    parser.add_argument('--cached-panel', type=Path, default=None,
                        help='Optional read-only exhaustive outcomes from decoder-matched geometry')
    parser.add_argument('--monitor-dataset',type=Path,help='Read-only test split of a supervised dataset')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    cli.first_hit_policy=cli.first_hit_policy or ('iid_random' if cli.rhythm_sampling=='iid' else 'tier_random')
    if ((cli.rhythm_sampling=='iid') != (cli.first_hit_policy=='iid_random')):
        parser.error('IID sampling requires iid_random deadlines; coverage sampling uses tier_random/fixed_one_second')
    if cli.resume and cli.initialize_from:parser.error('Choose resume or supervised initialization')
    if cli.cached_panel and cli.monitor_dataset:parser.error('Choose one held-out monitor')
    if (min(cli.contexts, cli.fit_steps, cli.batch, cli.replay_capacity) <= 0
            or min(cli.stages) < 0 or sum(cli.stages) <= 0 or not cli.devices):
        parser.error('Positive training budgets and nonnegative stage durations with positive total required')
    torch.set_num_threads(2); torch.manual_seed(cli.seed)
    rng = np.random.default_rng(cli.seed)
    source = json.loads((cli.source_run/'config.json').read_text())
    checkpoint = Path(source['checkpoint'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != source['checkpoint_sha256']:
        raise ValueError('Frozen decoder changed')
    eligibility = MinimumTimeChart.load(source['args']['table'])
    model = HandPlanRanker(HandPlanRankerConfig(hidden=cli.hidden))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cli.lr, weight_decay=1e-4)
    replay = RankerReplay(cli.replay_capacity)
    completed, gradient_steps, pending = 0, 0, None
    if cli.resume:
        saved = torch.load(cli.resume, map_location='cpu', weights_only=False)
        if saved['format'] != 'neural_hand_plan_ranker_v2' or saved['checkpoint_sha256'] != digest:
            raise ValueError('Incompatible neural hand-plan checkpoint')
        for key in ('contexts', 'stages', 'fit_steps', 'batch', 'replay_capacity', 'hidden',
                    'lr', 'min_lr', 'warmup', 'seed', 'start_mode','first_hit_policy','rhythm_sampling'):
            previous_value=saved['args'].get(key,'fixed_one_second' if key=='first_hit_policy' else 'coverage' if key=='rhythm_sampling' else None)
            if getattr(cli, key) != previous_value:
                raise ValueError(f'Resume must preserve {key} and the absolute schedule')
        model=HandPlanRanker(HandPlanRankerConfig(**saved['network']))
        optimizer=torch.optim.AdamW(model.parameters(),lr=cli.lr)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        replay = saved['replay']; rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        completed, gradient_steps, pending = saved['completed_updates'], saved['gradient_steps'], saved['pending']
    elif cli.initialize_from:
        saved=torch.load(cli.initialize_from,map_location='cpu',weights_only=False)
        if (saved['format']!='offline_hand_plan_ranker_v1' or saved['checkpoint_sha256']!=digest
                or saved['simulation_geometry_version']!=GEOMETRY_VERSION):
            raise ValueError('Incompatible supervised initialization')
        model=HandPlanRanker(HandPlanRankerConfig(**saved['network']));cli.hidden=model.config.hidden
        optimizer=torch.optim.AdamW(model.parameters(),lr=cli.lr)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer']);replay=saved['replay']
        gradient_steps=saved['gradient_steps'];rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'])
        if cli.lr!=cli.min_lr:raise ValueError('Supervised handoff uses a constant small rate; set lr=min-lr')
    total = sum(cli.stages)
    if cli.dry_run:
        print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
            observations=model.config.obs_dim, actions=16, stages=cli.stages,
            candidates=[16, 8, 4], contexts=cli.contexts, chart_used_for_scoring=False,
            decoder_checkpoint_sha256=digest, start_mode=cli.start_mode,
            observation_version=OBSERVATION_VERSION,first_hit_policy=cli.first_hit_policy), indent=2))
        return
    out = cli.out.resolve()
    if out.exists() and not cli.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = (StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility) if cli.rhythm_sampling=='iid' else
        ProceduralChartRhythmSampler(DrumKit.from_layout(), 4, eligibility,first_hit_policy=cli.first_hit_policy))
    seed_clip = out/'scene_seed.json'; write_json(seed_clip, sampler.seed_payload())
    cached = load_cached_panel(cli.cached_panel, digest,cli.first_hit_policy) if cli.cached_panel else None
    if cli.monitor_dataset:
        data_cfg=json.loads((cli.monitor_dataset/'config.json').read_text())
        if data_cfg['checkpoint_sha256']!=digest or data_cfg.get('first_hit_policy')!=cli.first_hit_policy:
            raise ValueError('Supervised test split uses a different decoder/timing policy')
        data=load_dataset_batches(cli.monitor_dataset)
        ids=torch.load(cli.monitor_dataset/'splits.pt',weights_only=False)['test']
        cached={key:value[ids] for key,value in data.items()};cached['panel']=None
        cached['rollout_runtime']=data_cfg.get('rollout_runtime')
        cached['scope']='held_out_iid_varied_start_dataset_test_split'
    config = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()},
        checkpoint=str(checkpoint), checkpoint_sha256=digest, network=asdict(model.config),
        neural_parameters=sum(p.numel() for p in model.parameters()), action_bank=ACTION_BANK.tolist(),
        action_semantics='Four binary decisions; bit1 means first drum (sorted by ID) assigned right; second simultaneous drum gets the other hand',
        algorithm='Neural16-action hand-plan ranker', candidate_schedule=dict(all16=cli.stages[0], top8=cli.stages[1], top4=cli.stages[2]),
        objective='2*best-observed-set cross entropy + confidence-weighted all-pair soft ranking',
        winner_tolerance=.02, reward_temperature=.05, illegal_action_policy='masked; LH floor excluded',
        unseen_action_policy='Unexecuted actions excluded from loss; never labelled misses',
        chart_used_for_scoring=False, eligibility_source=str(source['args']['table']),
        frozen_decoder=True, phase2_decoder_training=False, physics_hz=250, control_hz=50,
        starting_states=cli.start_mode, observation_version=OBSERVATION_VERSION,
        first_hit_policy=cli.first_hit_policy,rhythm_sampling=sampler.summary,
        first_hit_ranges_s=[[.02,.90]] if cli.rhythm_sampling=='iid' else [list(r) for r in sampler.speed_ranges],
        initialized_gradient_steps=gradient_steps,supervised_initialization=str(cli.initialize_from) if cli.initialize_from else None,
        initial_state_features='All17 joint positions and velocities (arms/waist), both Cartesian stick tips in kit/env frame, last hand targets',
        start_mixture=dict(post_hit=.80, random=.15, ready=.05),
        state_bank='Full measured q/qd/tips40ms after successful contact, max32 states/hand/drum; refreshed during training',
        candidate_start_matching='Every candidate of one rhythm receives the exact same physical q/qd',
        held_out_cached_panel=str(cli.cached_panel.resolve()) if cli.cached_panel else None, held_out_used_for_training=False,
        simulation_geometry_version=GEOMETRY_VERSION,
        checkpoint_selection='Latest model; held-out results never select or roll back weights',
        optimizer='AdamW', lr_schedule='100-step linear warmup, cosine to min_lr over absolute gradient-step budget')
    write_json(out/'config.json', config)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(out/'tb'), flush_secs=5)
    writer.add_text('setup/config', json.dumps(config), 0); writer.flush()
    if os.environ.get('TB_AUTO_REGISTER', '1') != '0':
        registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
        link = registry/out.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(out/'tb')
        if not link.is_symlink() or link.resolve() != out/'tb':
            raise ValueError('TensorBoard registration conflict')
        for _ in range(6):
            if out.name in json.load(urllib.request.urlopen('http://localhost:6006/data/runs', timeout=10)):
                break
            time.sleep(5)
        else:
            raise RuntimeError('Shared TensorBoard did not list neural run')
    def state():
        return dict(format='neural_hand_plan_ranker_v2', args=config['args'], network=asdict(model.config),
            checkpoint_sha256=digest, model=model.state_dict(), optimizer=optimizer.state_dict(),
            replay=replay, completed_updates=completed, gradient_steps=gradient_steps,
            rollout_runtime=config.get('rollout_runtime'),
            pending=pending, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            post_hit_bank=start_bank.state_dict() if start_bank is not None else None)
    def save_state():
        save_torch(out/'last.pt', state())
    start_bank = make_start_bank(model.config.tip_geometry) if cli.start_mode == 'varied' else None
    if (cli.resume or cli.initialize_from) and start_bank is not None and saved.get('post_hit_bank'):
        start_bank.load_state_dict(saved['post_hit_bank'])
    if not (out/'initial.pt').exists():
        if cli.resume:
            (out/'initial.pt').write_bytes((cli.resume.parent/'initial.pt').read_bytes())
        else:
            save_torch(out/'initial.pt', state())
    save_state()
    initial_held_out = evaluate_cached(model, cached) if cached is not None else {}
    if not (out/'initial_held_out.json').exists():
        if cli.resume:
            (out/'initial_held_out.json').write_bytes((cli.resume.parent/'initial_held_out.json').read_bytes())
        else:
            write_json(out/'initial_held_out.json', initial_held_out)
    evaluator, active_candidates = None, None
    write_json(out/'run.json', dict(status='starting', completed_updates=completed, stages=cli.stages))
    try:
        while completed < total:
            update = completed+1
            candidates = candidate_count(update, cli.stages)
            if candidates != active_candidates:
                if evaluator is not None:
                    evaluator.close()
                runtime = runtime_args(['--out', str(out), '--envs', str(cli.contexts*candidates),
                    '--device', cli.devices[0], '--rollout-devices', *cli.devices, '--physics-hz', '250',
                    '--decode-chunk', '512', '--collision-chunk', '1024', '--no-plan-diagnostics', '--clone-kit-from-source'])
                for key in ('torque_weight', 'joint_weight', 'collision_weight', 'collision_stride',
                            'near_miss_weight', 'near_miss_scale_m', 'near_miss_window_s'):
                    setattr(runtime, key, source['args'][key])
                runtime.groups = runtime.horizon = 4; runtime.seed = cli.seed
                evaluator = DistributedIsaacEvaluator(runtime, checkpoint, seed_clip)
                ready=json.loads((evaluator.workers[0][0]/'ready.json').read_text())
                actual_runtime={key:ready[key] for key in ('rollout_runtime_version','fixed_hit_scoring','decoder_amp','impact_speed_target')}
                if cli.resume or cli.initialize_from:require_matching_rollout_runtime(saved,{'rollout_runtime':actual_runtime})
                if cached is not None:require_matching_rollout_runtime(cached,{'rollout_runtime':actual_runtime})
                config['rollout_runtime']=actual_runtime
                active_candidates = candidates
                geometries = [json.loads((folder/'ready.json').read_text())['tip_geometry']
                              for folder,_ in evaluator.workers]
                if any(g != geometries[0] for g in geometries):
                    raise ValueError('Isaac workers disagree on stick/pelvis geometry')
                geometry = geometries[0]
                if model.config.tip_geometry is not None and model.config.tip_geometry != geometry:
                    raise ValueError('Simulator observation geometry changed on resume/stage switch')
                model.config.tip_geometry = geometry
                if start_bank is not None:
                    bank_data = start_bank.state_dict()
                    start_bank = make_start_bank(geometry); start_bank.load_state_dict(bank_data)
                if cached is not None and cached['panel'] is not None:
                    cached['observations'] = np.stack([encode_clip(c,tip_geometry=geometry) for c in cached['panel']])
                config['network'] = asdict(model.config); config['tip_geometry_source'] = 'Measured worker scene configuration'
                write_json(out/'config.json',config)
                if completed == 0 and not cli.resume:
                    save_torch(out/'initial.pt',state())
                    write_json(out/'initial_held_out.json',evaluate_cached(model,cached) if cached is not None else {})
                if start_bank is not None and not start_bank.states:
                    bootstrap_post_hits(evaluator, eligibility, start_bank, rng, out,
                        repeats=cli.bootstrap_repeats, rounds=cli.bootstrap_rounds)
                    save_state()
            started = time.monotonic()
            model.eval()
            if pending is None:
                panel = sampler.sample(cli.contexts, rng, minimum_plan_choices=1 if cli.rhythm_sampling=='iid' else 2)
                states, start_metrics = start_bank.sample(rng, cli.contexts) if start_bank is not None else (None, {})
                pending = build_ranker_batch(model, panel, eligibility, candidates, states)
                pending['tiers'] = sampler.last_strata.copy()
                pending['start_metrics'] = start_metrics
                save_state()
            def progress(done, total_episodes, tick, ticks):
                write_json(out/'progress.json', dict(iteration=update, candidates=candidates,
                    phase='neural_plan_physical_trials', episodes=total_episodes,
                    completed_episodes=done, tick=tick, ticks=ticks))
                print(f'[neural-plans] update={update} candidates={candidates} tick={tick}/{ticks}', flush=True)
            reports = evaluator.evaluate(pending['episodes'], progress,
                initial_states=pending['episode_states'], collect_post_hits=start_bank is not None)
            rollout_seconds = time.monotonic()-started
            outcomes = unpack_ranker_reports(pending, reports)
            bank_metrics = start_bank.add([s for r in reports for s in r.get('post_hit_states', [])], rng) if start_bank is not None else {}
            if start_bank is not None:
                save_torch(out/'post_hit_bank.pt', start_bank.state_dict())
            # Persist raw trials before fitting. Complete observed panels can
            # always be recovered after an interrupted optimizer step.
            save_torch(out/f'trials_{update:05d}.pt', dict(batch=pending, outcomes=outcomes))
            write_json(out/f'episodes_{update:05d}.json', reports)
            pre = ranking_metrics(pending['scores'], outcomes['rewards'], pending['observed'],
                outcomes['hit_rates'], pending['selected'])
            replay.add(pending['observations'], outcomes, pending['observed'])
            model.train(); losses, winner_losses, pair_losses = [], [], []
            fit_start = time.monotonic()
            before = {key: value.clone() for key, value in model.state_dict().items()}
            for _ in range(cli.fit_steps):
                gradient_steps += 1
                lr = learning_rate(gradient_steps, total*cli.fit_steps, cli.lr, cli.min_lr, cli.warmup)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                batch = replay.sample(rng, cli.batch)
                scores = model(batch['observations'])
                loss, components = ranking_loss(scores, batch['rewards'], batch['observed'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite neural plan loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach())); winner_losses.append(float(components['winner_loss']))
                pair_losses.append(float(components['pair_loss']))
            model.eval()
            with torch.no_grad():
                updated_scores = model(torch.as_tensor(pending['observations'])).numpy()
            post = ranking_metrics(updated_scores, outcomes['rewards'], pending['observed'], outcomes['hit_rates'])
            held_out = evaluate_cached(model, cached) if cached is not None else {}
            weight_changes = torch.cat([(value-before[key]).flatten() for key, value in model.state_dict().items()])
            metrics = dict(iteration=update, candidates=candidates, contexts=cli.contexts,
                physical_rollouts=len(reports), observed_pair_count=int(sum(n*(n-1)//2 for n in pending['observed'].sum(1))),
                replay_contexts=len(replay.data['observations']), gradient_steps=gradient_steps,
                learning_rate=lr, mean_loss=float(np.mean(losses)), winner_loss=float(np.mean(winner_losses)),
                pair_loss=float(np.mean(pair_losses)), weight_step_rms=float(weight_changes.square().mean().sqrt()),
                rollout_seconds=rollout_seconds, fit_seconds=time.monotonic()-fit_start,
                iteration_seconds=time.monotonic()-started, pre_fit=pre, post_fit_same_observed_bank=post,
                held_out=held_out, speed_tier_counts=np.bincount(pending['tiers'], minlength=3).tolist())
            metrics.update(start_states=pending['start_metrics'], post_hit_bank=bank_metrics)
            metrics.update(planned_transition_coverage(pending['episodes'], 7))
            completed = update; pending = None; save_state()
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics)+'\n')
            write_json(out/f'iteration_{update:05d}.json', metrics)
            for scope, values in (('train', metrics), ('pre_fit', pre), ('post_fit', post), ('held_out', held_out),
                                  ('start_states', metrics['start_states']), ('post_hit_bank', bank_metrics)):
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'{scope}/{key}', value, completed)
            writer.flush(); plot_history(out)
            write_json(out/'run.json', dict(status='running', completed_updates=completed,
                candidates=candidates, gradient_steps=gradient_steps, stages=cli.stages))
            print('[neural-plans-result] '+json.dumps({k: v for k, v in metrics.items()
                if not k.startswith('planned_hand_')}), flush=True)
        write_json(out/'run.json', dict(status='complete', completed_updates=completed,
            gradient_steps=gradient_steps, stages=cli.stages))
    except BaseException as exc:
        write_json(out/'run.json', dict(status='failed', completed_updates=completed, error=repr(exc)))
        raise
    finally:
        writer.close()
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    main()
