#!/usr/bin/env python3
"""Online analytic schedule RL: sampled GRPO or full-action expected reward."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
import amortized_opt as A
from amortized_opt.constants import ready_joint_pos
from isaac_rl.assignment_starts import StartingPoseSampler, pose_clearance
from isaac_rl.random_rhythms import RandomRhythmConfig, RandomRhythmSampler
from isaac_rl.schedule_distillation import FourHitScheduleScorer, schedule_bank, suffix_table
from isaac_rl.schedule_policy import (FourHitSchedulePolicy, entropy_and_kl,
    group_advantages, clipped_group_loss, expected_reward_loss, policy_metrics)
from isaac_rl.schedule_improvement import physical_table
from scripts.train_four_hit_schedule import setup, random_windows, load_split, log, write_json, physical_test


@torch.no_grad()
def rollout_transitions(scorer, env, table, features, indices, out):
    """Observed suffix transitions for the actual evaluated global schedules."""
    B, C, S, G = out['score'].shape
    owner = torch.arange(B, device=features.device).repeat_interleave(C*S)
    action = indices[:, :, None].expand(-1, -1, S).reshape(-1)
    scores = out['score'].reshape(-1, G)/scorer.reward_scale
    poses = out['poses'].reshape(-1, G, 17)
    velocities = out['post_velocity'].reshape(-1, G, 17)
    raw = {k:table[k][owner, :4] for k in ('drums', 'speeds', 'times', 'valid')}
    current = features[owner]; parts = []
    for g in range(G):
        nxt_table = suffix_table(raw, raw['times'][:, g], g+1)
        nxt = scorer.features(nxt_table, poses[:, g], velocities[:, g], env.cost.fk)
        parts.append(dict(obs=current, next_obs=nxt,
            action=(action % (1 << (4-g))) << g,
            next_action=(action % (1 << (3-g))) << (g+1),
            target=scores[:, g:].sum(-1), reward=scores[:, g],
            done=~nxt_table['valid'].any(-1)))
        current = nxt
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def make_samplers(env, kit, device):
    anchors = np.load(ROOT/'runs/assignment_start_poses_training_20261002.npz')
    assert tuple(anchors['joint_names']) == tuple(A.G1_UPPER_BODY_JOINTS)
    lo, hi = A.urdf_position_limits(posture_limits=False)
    starts = StartingPoseSampler(anchors['poses'], ready_joint_pos(), lo, hi,
        clearance=lambda q:pose_clearance(env.cost, torch.as_tensor(q, device=device)).cpu().numpy(),
        names=A.G1_UPPER_BODY_JOINTS, arm_std=.12, yaw_std=.25)
    sampler = RandomRhythmSampler(kit, RandomRhythmConfig(max_hits=4, pair_probability=0.))
    return starts, sampler


def fresh_cases(starts, sampler, kit, rng, count, device):
    clips = random_windows(sampler, rng, count)
    positions, stats = starts.sample(rng, count)
    std = rng.choice([.1, .3, .7], size=(count, 1), p=[.6, .3, .1])
    velocities = rng.normal(size=positions.shape)*std
    velocities[np.max(np.abs(positions-ready_joint_pos()), axis=1)<1e-6] = 0.
    q = torch.as_tensor(positions, device=device, dtype=torch.float32)
    v = torch.as_tensor(velocities, device=device, dtype=torch.float32)
    return clips, q, v, physical_table(clips, kit, device), stats


def fingerprint(q, v, table, draws):
    digest = hashlib.sha256()
    for tensor in (q, v, table['drums'], table['speeds'], table['times'], table['valid'], draws):
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, actor, critic, target, actor_optimizer, critic_optimizer, step, metrics, args):
    torch.save(dict(format=actor.format, model=actor.state_dict(), temperature=actor.temperature,
        hidden_dims=actor.hidden_dims,
        reward_scale=actor.reward_scale, critic_model=critic.state_dict(), target_model=target.state_dict(),
        actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
        step=step, actor_updates=step*args.epochs, critic=True, bellman=True,
        validation=metrics, source_checkpoint=str(args.scorer_checkpoint),
        generator_checkpoint=str(args.checkpoint), algorithm=args.algorithm,
        output_units='Policy logits; critic_model predicts score/20'), path)


def run(args):
    if args.out.exists():
        raise FileExistsError('Choose a fresh run directory')
    args.out.mkdir(parents=True)
    (args.out/'online').mkdir()
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.out/'tb'), flush_secs=5)
    registry = ROOT/'runs/tensorboard_active'; registry.mkdir(exist_ok=True)
    (registry/args.out.name).symlink_to((args.out/'tb').resolve())
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(critic=True, bellman=True, discount=1., generator_frozen=True,
        network='Separate actor and critic: each 86 -> '+' -> '.join(f'{w} LN GELU' for w in args.hidden_dims)+' -> 16',
        online_windows=args.rounds*args.windows,
        group_sampling='with replacement from old categorical policy' if args.algorithm=='grpo' else 'all 16 schedules, no action sampling',
        reference='Frozen best distillation scorer, interpreted as normalized policy logits',
        rewards='Measured full actual decoder spline, both hands and waist; original physics scoring',
        evaluation='Static four-hit schedule, perfect tracking and analytic bounce; no Isaac',
        critic_loss='Sampled root MC Huber + 0.5 suffix MC Huber + 0.1 finite-schedule Bellman Huber',
        policy_output_units='Logits, not calibrated schedule values', grpo_std_floor_raw_points=1.)
    write_json(args.out/'config.json', config)
    write_json(args.out/'run.json', dict(status='running', pid=os.getpid(), stage='initializing'))
    begin = time.monotonic()
    try:
        torch.set_num_threads(2); torch.manual_seed(args.seed)
        device = torch.device(args.device)
        actor = FourHitSchedulePolicy.from_scorer_checkpoint(args.scorer_checkpoint, device, args.temperature)
        reference = copy.deepcopy(actor).eval().requires_grad_(False)
        critic = FourHitScheduleScorer.from_checkpoint(args.scorer_checkpoint, device).requires_grad_(True)
        target = copy.deepcopy(critic).eval().requires_grad_(False)
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_learning_rate, weight_decay=1e-4)
        validation = load_split(args.eval_from/'data/validation', device)
        test = load_split(args.eval_from/'data/test', device)
        if set(validation['root_ids']) & set(test['root_ids']):
            raise AssertionError('Validation and test root overlap')
        baseline = policy_metrics(actor, reference, validation['features'], validation['values'], validation['graph_index'])
        best = baseline['regret_mean']; best_step = 0
        log(writer, 'validation', baseline, 0)
        save_checkpoint(args.out/'best.pt', actor, critic, target, actor_optimizer, critic_optimizer, 0, baseline, args)
        write_json(args.out/'baseline.json', dict(validation=baseline))
        env, kit = setup(args.checkpoint, device, args.chunk)
        starts, sampler = make_samplers(env, kit, device)
        write_json(args.out/'run.json', dict(status='running', pid=os.getpid(), stage='training',
            algorithm=args.algorithm, target_windows=args.rounds*args.windows))
        rng = np.random.default_rng(args.seed)
        # Action draws never advance the scenario RNG: both algorithms get the
        # same fresh starts, rhythms and matched physical bounce samples.
        action_rng = torch.Generator(device=device).manual_seed(args.seed+100000)
        bank = schedule_bank(device); total_counts = torch.zeros(7, dtype=torch.long, device=device)
        fingerprints = []; latest = baseline
        for step in range(1, args.rounds+1):
            clips, q, v, table, start_stats = fresh_cases(starts, sampler, kit, rng, args.windows, device)
            draws = torch.as_tensor(rng.uniform(.2, .8, (args.windows, args.bounce_draws, 4, 2)),
                device=device, dtype=torch.float32)
            features = actor.features(table, q, v, env.cost.fk)
            with torch.no_grad():
                old_lp = actor.log_probs(features)
                ref_lp = reference.log_probs(features)
                indices = (torch.multinomial(old_lp.exp(), args.group, replacement=True, generator=action_rng)
                    if args.algorithm=='grpo' else torch.arange(16, device=device)[None].expand(args.windows, -1))
                old_selected = old_lp.gather(1, indices)
            out = env.score(table, q, v, bank[indices], draws)
            rewards = out['score'].double().sum(-1).mean(-1).float().detach()
            replay = rollout_transitions(critic, env, table, features, indices, out)
            advantages = group_advantages(rewards) if args.algorithm=='grpo' else None
            case_hash = fingerprint(q, v, table, draws); fingerprints.append(case_hash)
            # Store actual outcomes and collection probabilities for audit.
            torch.save(dict(features=features.cpu(), q=q.cpu(), v=v.cpu(), clips=clips,
                draws=draws.cpu(), indices=indices.cpu(), rewards=rewards.cpu(), old_log_probs=old_lp.cpu(),
                start_stats=start_stats, scenario_fingerprint=case_hash,
                root_ids=[f'online:{args.seed}:{step}:{i}' for i in range(args.windows)],
                physics={k:out[k].cpu() for k in ('score','hits','strict_valid_hits','collision_m')}),
                args.out/'online'/f'batch_{step:04d}.pt')
            total_counts += torch.bincount(table['drums'][:, :4, 0].flatten(), minlength=7)
            collection_hits = float(out['hits'].sum()/out['requested'].sum())
            del out
            for epoch in range(args.epochs):
                actor.train(); lp = actor.log_probs(features)
                entropy, kl = entropy_and_kl(lp, ref_lp)
                if args.algorithm=='grpo':
                    policy_loss, clipped_fraction = clipped_group_loss(lp, indices, old_selected, advantages, clip=args.clip)
                else:
                    policy_loss = expected_reward_loss(lp, rewards, reward_scale=actor.reward_scale)
                    clipped_fraction = features.new_tensor(0.)
                actor_loss = policy_loss+args.kl_weight*kl.mean()-args.entropy_weight*entropy.mean()
                actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.); actor_optimizer.step()

                critic.train()
                root_value = critic(features).gather(1, indices)
                root_loss = F.smooth_l1_loss(root_value, rewards/critic.reward_scale)
                tid = torch.randint(len(replay['obs']), (args.critic_batch,), device=device)
                current = critic(replay['obs'][tid]).gather(1, replay['action'][tid,None]).squeeze(-1)
                with torch.no_grad():
                    nxt = target(replay['next_obs'][tid]).gather(1, replay['next_action'][tid,None]).squeeze(-1)
                    bellman_target = replay['reward'][tid]+nxt*(~replay['done'][tid])
                suffix_loss = F.smooth_l1_loss(current, replay['target'][tid])
                bellman_loss = F.smooth_l1_loss(current, bellman_target)
                critic_loss = root_loss+.5*suffix_loss+.1*bellman_loss
                critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.); critic_optimizer.step()
                with torch.no_grad():
                    for dst,src in zip(target.parameters(), critic.parameters()):
                        dst.lerp_(src, .01)
            actor.eval(); critic.eval()
            ordered = indices.sort(-1).values
            stats = dict(policy_loss=float(policy_loss), actor_loss=float(actor_loss), entropy_bits=float(entropy.mean()/np.log(2.)),
                reference_kl=float(kl.mean()), clipped_fraction=float(clipped_fraction),
                root_value_loss=float(root_loss), suffix_value_loss=float(suffix_loss), bellman_loss=float(bellman_loss),
                critic_loss=float(critic_loss), evaluated_reward_mean=float(rewards.mean()), collection_hit_rate=collection_hits,
                unique_sampled_schedules=float((1+(ordered[:,1:]!=ordered[:,:-1]).sum(-1)).float().mean()),
                completed_windows=step*args.windows, actor_updates=step*args.epochs,
                learning_rate=actor_optimizer.param_groups[0]['lr'])
            log(writer, 'train', stats, step)
            if step%args.validation_every==0 or step in (1, args.rounds):
                latest = policy_metrics(actor, reference, validation['features'], validation['values'], validation['graph_index'])
                log(writer, 'validation', latest, step)
                save_checkpoint(args.out/'last.pt', actor, critic, target, actor_optimizer, critic_optimizer, step, latest, args)
                if latest['regret_mean'] < best:
                    best = latest['regret_mean']; best_step = step
                    save_checkpoint(args.out/'best.pt', actor, critic, target, actor_optimizer, critic_optimizer, step, latest, args)
                print(json.dumps(dict(step=step, best_step=best_step, validation=latest, **stats)), flush=True)
            write_json(args.out/'progress.json', dict(stage='training', step=step, rounds=args.rounds, best_step=best_step,
                elapsed_s=time.monotonic()-begin, validation=latest, **stats))
            del replay, features, rewards, advantages
        write_json(args.out/'data_manifest.json', dict(fresh_training_windows=args.rounds*args.windows,
            seed=args.seed, scenario_fingerprints=fingerprints, drum_counts=total_counts.cpu().tolist(),
            bounce_draws=args.bounce_draws, schedules_per_window=args.group if args.algorithm=='grpo' else 16,
            validation_source=str(args.eval_from/'data/validation'), test_source=str(args.eval_from/'data/test')))
        actor = FourHitSchedulePolicy.from_checkpoint(args.out/'best.pt', device)
        selected = policy_metrics(actor, reference, test['features'], test['values'], test['graph_index'])
        log(writer, 'test', selected, best_step)
        write_json(args.out/'test_selection.json', dict(best_step=best_step, test_windows=len(test['features']), **selected))
        # Reuse the same saved states and fresh-bounce evaluation seed as all
        # preceding scorer runs. physical_test accepts either scorer or policy.
        (args.out/'data').symlink_to((args.eval_from/'data').resolve(), target_is_directory=True)
        eval_args = argparse.Namespace(out=args.out, devices=[args.device], checkpoint=args.checkpoint, chunk=args.chunk,
            seed=json.loads((args.eval_from/'config.json').read_text())['seed'])
        report = physical_test(eval_args, actor, writer)
        result = dict(status='complete', elapsed_s=time.monotonic()-begin, best_step=best_step,
            fresh_training_windows=args.rounds*args.windows, actor_updates=args.rounds*args.epochs,
            test_selection=selected, physical_test=report)
        write_json(args.out/'run.json', result); write_json(args.out/'progress.json', dict(stage='complete', **result))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(args.out/'run.json', dict(status='failed', error=str(exc))); raise
    finally:
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--algorithm', choices=('grpo','expected'), required=True)
    p.add_argument('--scorer-checkpoint', type=Path, default=ROOT/'runs/four_hit_schedule_distillation_constant_lr_20261004/best.pt')
    p.add_argument('--eval-from', type=Path, default=ROOT/'runs/four_hit_schedule_distillation_20261004')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--rounds', type=int, default=128)
    p.add_argument('--windows', type=int, default=128); p.add_argument('--group', type=int, default=8)
    p.add_argument('--bounce-draws', type=int, default=2); p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=4); p.add_argument('--critic-batch', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=1e-4); p.add_argument('--critic-learning-rate', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=1.); p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--kl-weight', type=float, default=.05); p.add_argument('--entropy-weight', type=float, default=.005)
    p.add_argument('--validation-every', type=int, default=16); p.add_argument('--seed', type=int, default=20261004017)
    args = p.parse_args()
    if min(args.rounds,args.windows,args.epochs,args.bounce_draws,args.critic_batch,args.validation_every,args.chunk) < 1 or args.group < 2:
        p.error('Positive dimensions required; group must be at least two')
    if min(args.learning_rate,args.critic_learning_rate,args.temperature)<=0 or not 0<args.clip<1 or min(args.kl_weight,args.entropy_weight)<0:
        p.error('Invalid optimization configuration')
    args.out=args.out.resolve(); args.eval_from=args.eval_from.resolve(); args.scorer_checkpoint=args.scorer_checkpoint.resolve()
    source = torch.load(args.scorer_checkpoint,map_location='cpu',weights_only=False)
    args.hidden_dims = source.get('hidden_dims', (256,256,128))
    args.checkpoint = Path(source['generator_checkpoint']).resolve()
    run(args)


if __name__=='__main__':
    main()
