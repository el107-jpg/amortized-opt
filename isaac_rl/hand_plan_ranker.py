"""Sixteen-action neural ranking of four-group hand assignment plans.

Scores predict preference directly; no transition-time chart enters selection.
Observed-subset losses never label unexecuted plans as failures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from amortized_opt.constants import N_DOF, ready_joint_pos, PELVIS_IN_ENV
from amortized_opt.fk import ChainFK, stick_tip_offset
from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip

GROUPS, NUM_DRUMS, NUM_ACTIONS = 4, 7, 16
ACTION_BANK = np.asarray(list(product((0, 1), repeat=GROUPS)), np.int64)
BASE_OBS_DIM = GROUPS*(2*(NUM_DRUMS+2)+2)+2*N_DOF+2*(NUM_DRUMS+1)
OBS_DIM = BASE_OBS_DIM+6
OBSERVATION_VERSION = 'joint_state_and_cartesian_tips_v2'


@lru_cache(maxsize=1)
def _tip_fk():
    return ChainFK(frames={f'{hand}_wrist': f'{hand}_wrist_yaw_joint' for hand in ('left', 'right')})


@torch.no_grad()
def cartesian_tips(q, geometry=None):
    values = np.asarray(q, np.float32)
    single = values.ndim == 1
    joints = torch.as_tensor(values[None] if single else values)
    tips = []
    for index, hand in enumerate(('left', 'right')):
        rotation, position = _tip_fk().frame(joints, f'{hand}_wrist')
        offset = torch.as_tensor(stick_tip_offset(hand) if geometry is None else geometry['tip_offsets_local_m'][index],
                                 dtype=joints.dtype)
        tips.append(position+torch.einsum('bij,j->bi', rotation, offset)
                    +torch.as_tensor(PELVIS_IN_ENV if geometry is None else geometry['pelvis_in_env_m'], dtype=joints.dtype))
    result = torch.stack(tips, 1).numpy()
    return result[0] if single else result


@dataclass
class HandPlanRankerConfig:
    hidden: int = 256
    obs_dim: int = OBS_DIM
    num_actions: int = NUM_ACTIONS
    tip_geometry: dict | None = None
    dropout: float = 0.
    architecture: str = 'flat'
    group_width: int = 64
    state_width: int = 64
    scorer_depth: int = 2


class HandPlanRanker(nn.Module):
    def __init__(self, config=HandPlanRankerConfig()):
        super().__init__()
        self.config = config
        if not 0. <= config.dropout < 1.:raise ValueError('Dropout must be in [0, 1)')
        if config.architecture=='shared_candidate':
            if config.obs_dim!=OBS_DIM or config.num_actions!=NUM_ACTIONS:
                raise ValueError('Shared candidate model requires136 inputs and16 binary plans')
            from isaac_rl.hand_plan_candidate_ranker import SharedCandidateScorer
            self.candidate_scorer=SharedCandidateScorer(config.hidden,config.dropout,
                config.group_width,config.state_width,config.scorer_depth)
            return
        if config.architecture!='flat':raise ValueError('Unknown ranker architecture')
        self.network = nn.Sequential(nn.Linear(config.obs_dim, config.hidden),
            nn.LayerNorm(config.hidden), nn.GELU(),
            nn.Linear(config.hidden, config.hidden), nn.GELU(),
            nn.Linear(config.hidden, config.num_actions))
        # Keep existing parameter keys and initialization identical for old models.
        self.hidden_dropout = nn.Dropout(config.dropout)
        # Small symmetric initial preferences, independent of the old chart.
        nn.init.normal_(self.network[-1].weight, std=.001)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observations):
        if self.config.architecture=='shared_candidate':return self.candidate_scorer(observations)
        value=observations
        for index,layer in enumerate(self.network):
            value=layer(value)
            if index in (2,4):value=self.hidden_dropout(value)
        return value

    def score_candidates(self,observations,plans):
        if self.config.architecture!='shared_candidate':raise ValueError('Explicit candidates require shared_candidate architecture')
        return self.candidate_scorer(observations,plans)


def encode_clip(clip, *, q=None, qd=None, initial_drums=(-1, -1), tips=None, tip_geometry=None,
                allow_partial=False):
    groups = hit_groups(clip)
    if len(groups) != GROUPS and not (allow_partial and 0 < len(groups) < GROUPS):
        raise ValueError('Sixteen-action ranker requires exactly four hit groups')
    if any(len({h.drum for h in group}) != len(group) for group in groups):
        raise ValueError('Duplicate simultaneous target')
    result = []
    previous = 0.
    for group in groups + [[] for _ in range(GROUPS-len(groups))]:
        for slot in range(2):
            one_hot = np.zeros(NUM_DRUMS, np.float32)
            if slot < len(group):
                hit = group[slot]
                if not 0 <= hit.drum < NUM_DRUMS:
                    raise ValueError('Target drum outside seven-drum action space')
                one_hot[hit.drum] = 1.
                result.extend([*one_hot, float(hit.velocity), 1.])
            else:
                result.extend([*one_hot, 0., 0.])
        time_s = float(group[0].time) if group else previous
        result.extend([time_s/4., time_s-previous])
        previous = time_s
    ready = np.asarray(ready_joint_pos(), np.float32)
    q = ready if q is None else np.asarray(q, np.float32)
    qd = np.zeros(N_DOF, np.float32) if qd is None else np.asarray(qd, np.float32)
    if q.shape != (N_DOF,) or qd.shape != (N_DOF,):
        raise ValueError('Wrong initial joint-state shape')
    result.extend((q-ready)/np.pi)
    result.extend(qd/10.)
    if len(initial_drums) != 2:
        raise ValueError('Two initial hand targets required')
    for drum in initial_drums:
        if not -1 <= drum < NUM_DRUMS:
            raise ValueError('Initial hand drum outside action space')
        one_hot = np.zeros(NUM_DRUMS+1, np.float32)
        one_hot[drum+1] = 1.
        result.extend(one_hot)
    tips = cartesian_tips(q, tip_geometry) if tips is None else np.asarray(tips, np.float32)
    if tips.shape != (2, 3) or not np.isfinite(tips).all():
        raise ValueError('Two Cartesian stick-tip positions required')
    result.extend(tips.reshape(-1))  # Kit/env frame, metres; unaffected by env grid origin.
    features = np.asarray(result, np.float32)
    if features.shape != (OBS_DIM,) or not np.isfinite(features).all():
        raise ValueError('Invalid neural plan observation')
    return features


def legal_actions(clip, eligibility):
    groups = hit_groups(clip)
    if len(groups) != GROUPS:
        raise ValueError('Four-group plans required')
    mask = np.ones(NUM_ACTIONS, bool)
    for group_index, group in enumerate(groups):
        mask &= np.isin(ACTION_BANK[:, group_index], eligibility.choices(group))
    if not mask.any():
        raise ValueError('No legal hand assignment')
    return mask


@torch.inference_mode()
def rank_clips(model, clips, eligibility, states=None):
    if states is not None and len(states) != len(clips):
        raise ValueError('One initial state per rhythm required')
    observations = np.stack([encode_clip(clip, tip_geometry=model.config.tip_geometry, **({} if states is None else
        {key: states[row][key] for key in ('q', 'qd', 'initial_drums', 'tips') if key in states[row]}))
                             for row, clip in enumerate(clips)])
    legal = np.stack([legal_actions(clip, eligibility) for clip in clips])
    device = next(model.parameters()).device
    was_training=model.training;model.eval()
    try:scores = model(torch.as_tensor(observations, device=device)).cpu().numpy()
    finally:model.train(was_training)
    if not np.isfinite(scores).all():
        raise FloatingPointError('Nonfinite neural assignment scores')
    return observations, legal, scores


def build_ranker_batch(model, panel, eligibility, candidates, states=None):
    if candidates not in (4, 8, 16):
        raise ValueError('Candidate count must be4,8 or16')
    observations, legal, scores = rank_clips(model, panel, eligibility, states)
    selected = np.full((len(panel), candidates), -1, np.int64)
    for row in range(len(panel)):
        ids = np.flatnonzero(legal[row])
        order = ids[np.lexsort((ids, -scores[row, ids]))]
        selected[row, :min(len(order), candidates)] = order[:candidates]
    observed = np.zeros_like(legal)
    episodes, labels, episode_states = [], [], []
    for rank in range(candidates):
        for row, clip in enumerate(panel):
            action = int(selected[row, rank])
            if action >= 0:
                episodes.append(assigned_clip(clip, ACTION_BANK[action]))
                labels.append((row, action))
                if states is not None:
                    episode_states.append(states[row])
                observed[row, action] = True
    return dict(observations=observations, legal=legal, scores=scores,
        selected=selected, observed=observed, episodes=episodes,
        labels=np.asarray(labels, np.int64), panel=panel, states=states,
        episode_states=episode_states if states is not None else None)


def unpack_ranker_reports(batch, reports):
    if len(reports) != len(batch['labels']):
        raise ValueError('Executed plan/report counts differ')
    shape = batch['observed'].shape
    rewards, hit_rates = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    hits, requested = np.zeros(shape, np.int64), np.zeros(shape, np.int64)
    for (row, action), report in zip(batch['labels'], reports):
        rewards[row, action] = report['reward']
        hit_rates[row, action] = report['hit_rate']
        hits[row, action], requested[row, action] = report['hits'], report['requested']
    if not np.isfinite(rewards[batch['observed']]).all():
        raise FloatingPointError('Nonfinite physical plan reward')
    return dict(rewards=rewards, hit_rates=hit_rates, hits=hits, requested=requested)


def ranking_loss(scores, rewards, observed, *, best_weight=2.,
                 reward_temperature=.05, best_tolerance=.02):
    """Best-set cross entropy plus confidence-weighted all-pair preferences.

    Both terms average within each rhythm. Only physically observed scores
    participate. Near-tied best outcomes share the winner target.
    """
    if scores.shape != rewards.shape or observed.shape != scores.shape:
        raise ValueError('Score/target/mask dimensions differ')
    if not observed.any(1).all():
        raise ValueError('Every training rhythm needs an observed outcome')
    observed = observed.bool()
    rewards = torch.where(observed, rewards, torch.zeros_like(rewards))
    masked_rewards = rewards.masked_fill(~observed, -torch.inf)
    best = masked_rewards.max(1, keepdim=True).values
    winners = observed & (rewards >= best-best_tolerance)
    target = winners.to(scores.dtype)/winners.sum(1, keepdim=True)
    log_probs = F.log_softmax(scores.masked_fill(~observed, -1e9), dim=1)
    winner_loss = -(target*log_probs).sum(1).mean()
    pairs = observed[:, :, None] & observed[:, None, :]
    pairs &= torch.ones(scores.shape[1], scores.shape[1], device=scores.device,
                        dtype=torch.bool).triu(1)[None]
    gaps = rewards[:, :, None]-rewards[:, None, :]
    targets = torch.sigmoid(gaps/reward_temperature)
    differences = scores[:, :, None]-scores[:, None, :]
    losses = F.binary_cross_entropy_with_logits(differences, targets, reduction='none')
    weights = (gaps.abs()/.1).clamp(min=.05, max=1.)*pairs
    pair_loss = ((losses*weights).sum((1, 2))/weights.sum((1, 2)).clamp_min(1e-8)).mean()
    total = best_weight*winner_loss+pair_loss
    return total, dict(winner_loss=winner_loss.detach(), pair_loss=pair_loss.detach())


def ranking_metrics(scores, rewards, observed, hit_rates=None, selected=None):
    scores, rewards, observed = np.asarray(scores), np.asarray(rewards), np.asarray(observed, bool)
    choices = np.where(observed, scores, -np.inf).argmax(1)
    rows = np.arange(len(scores))
    chosen = rewards[rows, choices]
    best = np.where(observed, rewards, -np.inf).max(1)
    ranks = 1+((rewards > chosen[:, None]+1e-8) & observed).sum(1)
    winner_ranks, pair_correct, pair_total = [], 0, 0
    for row, mask in enumerate(observed):
        ids = np.flatnonzero(mask)
        order = ids[np.lexsort((ids, -scores[row, ids]))]
        winner_ranks.append(int(np.flatnonzero(rewards[row, order] >= best[row]-1e-8)[0])+1)
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                if abs(rewards[row, a]-rewards[row, b]) > .05:
                    delta = scores[row, a]-scores[row, b]
                    pair_total += 1
                    pair_correct += .5 if abs(delta) < 1e-8 else int(delta*(rewards[row, a]-rewards[row, b]) > 0)
    result = dict(mean_selected_reward=float(chosen.mean()), mean_best_observed_reward=float(best.mean()),
        mean_reward_regret=float((best-chosen).mean()),
        selected_is_best_reward_fraction=float((chosen >= best-1e-8).mean()),
        mean_selected_actual_reward_rank=float(ranks.mean()),
        mean_best_reward_predicted_rank=float(np.mean(winner_ranks)),
        pair_accuracy=float(pair_correct/max(pair_total, 1)), strong_pairs=pair_total)
    if hit_rates is not None:
        hit_rates = np.asarray(hit_rates)
        result.update(mean_selected_hit_rate=float(hit_rates[rows, choices].mean()),
            mean_best_observed_hit_rate=float(np.where(observed, hit_rates, -np.inf).max(1).mean()))
    if selected is not None:
        first = np.asarray(selected)[:, 0]
        result['collection_first_matches_score_choice_fraction'] = float((first == choices).mean())
    return result


def candidate_count(update, stages):
    if update < 1 or len(stages) != 3 or min(stages) < 0 or sum(stages) <= 0:
        raise ValueError('One-based updates and three nonnegative stage durations with positive total required')
    end = 0
    for count, duration in zip((16, 8, 4), stages):
        end += duration
        if update <= end:
            return count
    raise ValueError('Update outside candidate schedule')


class RankerReplay:
    def __init__(self, capacity=4096):
        self.capacity = int(capacity)
        self.data = None

    def add(self, observations, outcomes, observed):
        values = dict(observations=np.asarray(observations, np.float32),
                      rewards=outcomes['rewards'], hit_rates=outcomes['hit_rates'],
                      observed=np.asarray(observed, bool))
        self.data = {key: (value.copy() if self.data is None else
            np.concatenate((self.data[key], value)))[-self.capacity:] for key, value in values.items()}

    def sample(self, rng, count):
        ids = rng.integers(len(self.data['observations']), size=count)
        return {key: torch.as_tensor(value[ids]) for key, value in self.data.items()}
