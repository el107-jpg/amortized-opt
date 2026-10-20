#!/usr/bin/env python3
"""Fresh matched Isaac comparison: neural ranker, its initial weights, graph."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from amortized_opt.rhythm import DrumKit
from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.chart_cma_sampling import ProceduralChartRhythmSampler
from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator
from isaac_rl.min_time_assignment import MinimumTimeChart, graph_actions
from isaac_rl.hand_plan_ranker import ACTION_BANK, HandPlanRanker, HandPlanRankerConfig, rank_clips
from scripts.train_chart_cma import parse_args as runtime_args, write_json
from scripts.train_neural_hand_plans import make_start_bank
from isaac_rl.decoder_scene import GEOMETRY_VERSION
from isaac_rl.hand_plan_dataset import StochasticHandRhythmSampler


def rollout_source_config(config):
    """Resolve controller recipe through neural/dataset run provenance."""
    path=config.get('args',{}).get('source_run',config.get('source_run'))
    seen=set()
    while path:
        key=str(Path(path).resolve())
        if key in seen:raise ValueError('Cyclic rollout source provenance')
        seen.add(key);source=json.loads((Path(path)/'config.json').read_text())
        if 'torque_weight' in source.get('args',{}):return source
        path=source.get('args',{}).get('source_run',source.get('source_run'))
    raise ValueError('No frozen-controller recipe in run provenance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--contexts', type=int, default=256)
    parser.add_argument('--envs', type=int, default=1024)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--reference-run',type=Path,
                        help='Include a previous neural run on these exact same physical starts/rhythms')
    parser.add_argument('--reference-label',default='curriculum_neural',
                        choices=('curriculum_neural','fixed_first_neural','supervised_neural'),
                        help='Label the reference according to its training recipe')
    parser.add_argument('--reuse-test-plans',type=Path,
                        help='Reuse frozen panel and initial states from a previous matched test')
    cli = parser.parse_args()
    if cli.out.exists():
        raise FileExistsError(cli.out)
    if min(cli.contexts, cli.envs) < 1:
        parser.error('Positive evaluation budgets required')
    cli.out.mkdir(parents=True)
    torch.set_num_threads(2)
    config = json.loads((cli.run/'config.json').read_text())
    if config.get('simulation_geometry_version') != GEOMETRY_VERSION:
        raise ValueError('Neural run uses old/unverified simulation geometry')
    source = rollout_source_config(config)
    checkpoint = Path(config['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config['checkpoint_sha256']:
        raise ValueError('Decoder changed')
    eligibility = MinimumTimeChart.load(config['eligibility_source'])
    models, saved_models = [], []
    for name in ('last.pt', 'initial.pt'):
        snapshot = cli.out/name; snapshot.write_bytes((cli.run/name).read_bytes())
        saved = torch.load(snapshot, map_location='cpu', weights_only=False)
        model = HandPlanRanker(HandPlanRankerConfig(**saved['network']))
        model.load_state_dict(saved['model']); model.eval(); models.append(model)
        saved_models.append(saved)
    names=['learned_neural','initial_neural']
    if cli.reference_run:
        reference_cfg=json.loads((cli.reference_run/'config.json').read_text())
        if (reference_cfg['checkpoint_sha256']!=config['checkpoint_sha256']
                or reference_cfg.get('simulation_geometry_version')!=GEOMETRY_VERSION):
            raise ValueError('Reference neural model uses incompatible decoder/geometry')
        snapshot=cli.out/'reference.pt';snapshot.write_bytes((cli.reference_run/'last.pt').read_bytes())
        saved=torch.load(snapshot,map_location='cpu',weights_only=False)
        model=HandPlanRanker(HandPlanRankerConfig(**saved['network']));model.load_state_dict(saved['model']);model.eval()
        models.append(model);names.append(cli.reference_label)
    first_hit_policy=config['args'].get('first_hit_policy',config.get('first_hit_policy','fixed_one_second'))
    sampler = (StochasticHandRhythmSampler(DrumKit.from_layout(),4,eligibility)
        if config['args'].get('rhythm_sampling','iid' if first_hit_policy=='iid_random' else 'coverage')=='iid' else
        ProceduralChartRhythmSampler(DrumKit.from_layout(),4,eligibility,first_hit_policy=first_hit_policy))
    rng = np.random.default_rng(cli.seed)
    if cli.reuse_test_plans:
        previous_cfg=json.loads((cli.reuse_test_plans.parent/'config.json').read_text())
        if (previous_cfg['checkpoint_sha256']!=config['checkpoint_sha256']
                or previous_cfg.get('simulation_geometry_version')!=GEOMETRY_VERSION
                or previous_cfg.get('first_hit_policy','fixed_one_second')!=first_hit_policy):
            raise ValueError('Reused test panel uses incompatible decoder/geometry')
        previous=torch.load(cli.reuse_test_plans,map_location='cpu',weights_only=False)
        panel,states=previous['panel'],previous.get('initial_states')
        if len(panel)!=cli.contexts:raise ValueError('Reused test context count differs')
        sampler.last_strata=previous['tiers']
        start_metrics=previous_cfg['starting_states']
    else:
        panel = sampler.sample(cli.contexts, rng, minimum_plan_choices=1 if first_hit_policy=='iid_random' else 2)
        states, start_metrics = None, {}
        if saved_models[0].get('post_hit_bank'):
            bank = make_start_bank(models[0].config.tip_geometry)
            bank.load_state_dict(saved_models[0]['post_hit_bank'])
            states, start_metrics = bank.sample(rng, cli.contexts)
    choices = []
    for model in models:
        _, legal, scores = rank_clips(model, panel, eligibility, states)
        choices.append(ACTION_BANK[np.where(legal, scores, -np.inf).argmax(1)])
    choices.append(np.stack([graph_actions(hit_groups(c), initial_drums=
        tuple(states[row].get('initial_drums',(-1,-1))) if states is not None else (-1,-1))[0]
        for row,c in enumerate(panel)]))
    # Preserve floor eligibility in the graph reference using its closest
    # legal assignment if the old graph ever requests a forbidden action.
    _, legal, _ = rank_clips(models[0], panel, eligibility, states)
    graph_repaired = 0
    for row, action in enumerate(choices[-1]):
        code = int(action@np.array([8, 4, 2, 1]))
        if not legal[row, code]:
            ids = np.flatnonzero(legal[row])
            code = int(ids[np.argmin((ACTION_BANK[ids] != action).sum(1))])
            choices[-1][row] = ACTION_BANK[code]; graph_repaired += 1
    names.append('graph')
    episodes = [assigned_clip(clip, choices[k][row]) for k in range(len(names)) for row, clip in enumerate(panel)]
    torch.save(dict(panel=panel, choices=choices, names=names, tiers=sampler.last_strata,
                   initial_states=states), cli.out/'plans.pt')
    write_json(cli.out/'config.json', dict(run=str(cli.run.resolve()), seed=cli.seed,
        contexts=cli.contexts, names=names, checkpoint_sha256=config['checkpoint_sha256'],
        physics_hz=250, control_hz=50, no_training_feedback=True,
        graph_eligibility_repairs=graph_repaired, starting_states=start_metrics,
        reference_run=str(cli.reference_run.resolve()) if cli.reference_run else None,
        reference_label=cli.reference_label if cli.reference_run else None,
        reused_test_plans=str(cli.reuse_test_plans.resolve()) if cli.reuse_test_plans else None,
        graph_initialization='Actual previous hand targets from matched physical starting state',
        first_hit_policy=first_hit_policy,
        simulation_geometry_version=GEOMETRY_VERSION))
    args = runtime_args(['--out', str(cli.out), '--envs', str(cli.envs), '--device', cli.devices[0],
        '--rollout-devices', *cli.devices, '--physics-hz', '250', '--decode-chunk', '512',
        '--collision-chunk', '1024', '--no-plan-diagnostics', '--clone-kit-from-source'])
    for key in ('torque_weight', 'joint_weight', 'collision_weight', 'collision_stride',
                'near_miss_weight', 'near_miss_scale_m', 'near_miss_window_s'):
        setattr(args, key, source['args'][key])
    args.groups = args.horizon = 4; args.seed = cli.seed
    seed_clip = cli.out/'scene_seed.json'; write_json(seed_clip, sampler.seed_payload())
    evaluator = None
    try:
        evaluator = DistributedIsaacEvaluator(args, checkpoint, seed_clip)
        started = time.monotonic()
        reports = evaluator.evaluate(episodes, lambda done,total,tick,ticks:
            write_json(cli.out/'progress.json', dict(complete=done,total=total,tick=tick,ticks=ticks)),
            initial_states=states*len(names) if states is not None else None)
        write_json(cli.out/'episodes.json', reports)
        summary = dict(status='complete', contexts=cli.contexts, physics_seconds=time.monotonic()-started,
            methods={}, matched=True, no_training_feedback=True)
        rewards = np.array([r['reward'] for r in reports]).reshape(len(names), cli.contexts)
        hits = np.array([r['hits'] for r in reports]).reshape(len(names), cli.contexts)
        requested = np.array([r['requested'] for r in reports]).reshape(len(names), cli.contexts)
        np.testing.assert_array_equal(requested, np.broadcast_to(requested[:1], requested.shape))
        for i, name in enumerate(names):
            summary['methods'][name] = dict(hits=int(hits[i].sum()), requested=int(requested[i].sum()),
                hit_rate=float(hits[i].sum()/requested[i].sum()), mean_reward=float(rewards[i].mean()))
        rng = np.random.default_rng(cli.seed+1)
        indices = rng.integers(cli.contexts, size=(2000, cli.contexts))
        for other in range(1,len(names)):
            gain = (hits[0, indices]-hits[other, indices]).sum(1)/requested[0, indices].sum(1)
            summary[f'learned_minus_{names[other]}'] = dict(
                hit_rate_gain=float((hits[0].sum()-hits[other].sum())/requested[0].sum()),
                rhythm_bootstrap_95_interval=np.quantile(gain, [.025,.975]).tolist(),
                mean_reward_gain=float((rewards[0]-rewards[other]).mean()))
        write_json(cli.out/'summary.json', summary); write_json(cli.out/'run.json', summary)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        values = [summary['methods'][name]['hit_rate']*100 for name in names]
        labels={'learned_neural':'Learned neural','initial_neural':'Initial neural',
                'curriculum_neural':'16→8→4 neural','fixed_first_neural':'Fixed-first neural',
                'supervised_neural':'Original supervised','graph':'Graph'}
        colors={'learned_neural':'#4c8dba','initial_neural':'#a5bbc9','curriculum_neural':'#bd8f51',
                'fixed_first_neural':'#bd8f51','supervised_neural':'#bd8f51','graph':'#888888'}
        ax.bar([labels[name] for name in names],values,color=[colors[name] for name in names])
        for i, value in enumerate(values):
            ax.text(i, value+1, f'{value:.1f}%', ha='center')
        ax.set_ylim(0, 100); ax.set_ylabel('Credited hits (%)')
        ax.set_title(f'Fresh matched four-group test: {cli.contexts} rhythms')
        fig.savefig(cli.out/'comparison.png', dpi=170); plt.close(fig)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(cli.out/'run.json', dict(status='failed', error=repr(exc)))
        raise
    finally:
        if evaluator is not None:
            evaluator.close()


if __name__ == '__main__':
    main()
