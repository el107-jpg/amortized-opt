"""Joint analytic schedule values and learned discrete local improvement.

No Isaac imports. A state is measured q/qd/tips plus <=10 future physical
hits. A binary decision assigns a whole simultaneous strike group; its second
hit uses the other arm. No drum eligibility or commitment queue is encoded.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import torch
from torch import nn

from amortized_opt.constants import N_DOF, PELVIS_IN_ENV, STICK_TIP_SIGMA
from amortized_opt.fk import stick_contact_offset
from isaac_rl.arm_assignment import AssignmentConfig, AssignmentQueue
from isaac_rl.assignment_search import hit_groups

MAX_HITS = 10
EDITS = [()] + [(i,) for i in range(MAX_HITS)] + list(combinations(range(MAX_HITS), 2))


def cap_window(clip, *, horizon=1., max_hits=MAX_HITS):
    """Keep earliest complete groups. Never split a simultaneous pair."""
    kept = []
    for group in hit_groups(clip):
        if group[0].time > horizon+1e-7 or len(kept)+len(group) > max_hits:
            break
        if group[0].time > 1e-7:
            kept.extend(group)
    if not kept:
        raise ValueError('empty capped schedule')
    return replace(clip, hits=kept, duration=kept[-1].time)


def graph_seed(clip, kit, *, parked=None):
    queue = AssignmentQueue(clip, kit, AssignmentConfig())
    if parked is not None:
        queue.parked.update(parked)
    actions = np.zeros(MAX_HITS, np.int64)
    fallback = 0
    while queue.next_group < len(queue.groups):
        try:
            action = queue.teacher_action()
        except ValueError:
            # Duplicate drum names can defeat the graph dictionary teacher;
            # retain both physical hits and record the fallback explicitly.
            action = queue.next_group % 2
            fallback += 1
        actions[queue.next_group] = action
        queue.commit(action)
    return actions, fallback


def physical_table(clips, kit, device):
    B = len(clips)
    drums = np.full((B, MAX_HITS, 2), -1, np.int64)
    speeds = np.zeros((B, MAX_HITS, 2), np.float32)
    times = np.zeros((B, MAX_HITS), np.float32)
    valid = np.zeros((B, MAX_HITS), bool)
    events = np.zeros((B, MAX_HITS, 16), np.float32)
    hit_valid = np.zeros((B, MAX_HITS), bool)
    group = np.zeros((B, MAX_HITS), np.int64)
    member = np.zeros((B, MAX_HITS), np.int64)
    for b, clip in enumerate(clips):
        groups = hit_groups(clip)
        if sum(map(len, groups)) > MAX_HITS or clip.hits[-1].time > 1+1e-6:
            raise ValueError('schedule exceeds ten hits or one second')
        times[b] = groups[-1][0].time
        slot = 0
        for g, rows in enumerate(groups):
            valid[b, g] = True
            times[b, g] = rows[0].time
            for j, hit in enumerate(rows):
                drums[b, g, j] = hit.drum
                speeds[b, g, j] = 1+2*hit.velocity
                events[b, slot, hit.drum] = 1
                events[b, slot, 7:10] = kit.positions[hit.drum]-PELVIS_IN_ENV
                events[b, slot, 10:13] = kit.axes[hit.drum]
                events[b, slot, 13:] = [kit.radii[hit.drum], hit.time, hit.velocity]
                hit_valid[b, slot] = True
                group[b, slot], member[b, slot] = g, j
                slot += 1
    arrays = dict(drums=drums, speeds=speeds, times=times, valid=valid,
                  events=events, hit_valid=hit_valid, group=group, member=member)
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def observation(table, q, v, fk):
    tips = []
    for hand in ('left', 'right'):
        R, p = fk.frame(q, hand+'_wrist')
        tips.append(p+(R @ q.new_tensor(stick_contact_offset(hand, STICK_TIP_SIGMA))[:, None]).squeeze(-1))
    state = torch.cat((q/np.pi, v/10, torch.stack(tips, 1).flatten(1)), -1)
    return dict(state=state, **{k: table[k] for k in ('events', 'hit_valid', 'group', 'member')})


def index_obs(obs, ids):
    return {k: v[ids] for k, v in obs.items()}


def group_mask(obs):
    mask = torch.zeros(len(obs['state']), MAX_HITS, dtype=torch.long, device=obs['state'].device)
    return mask.scatter_add(1, obs['group'], obs['hit_valid'].long()) > 0


def edit_masks(device):
    masks = torch.zeros(len(EDITS), MAX_HITS, dtype=torch.long, device=device)
    for i, edit in enumerate(EDITS):
        masks[i, list(edit)] = 1
    return masks


def valid_edits(obs):
    masks = edit_masks(obs['state'].device).bool()
    return ~(masks[None] & ~group_mask(obs)[:, None]).any(-1)


def neighbors(actions):
    return actions[:, None] ^ edit_masks(actions.device)[None]


def suffix_obs(obs, q, v, fk, elapsed, removed):
    nxt = observation(obs, q, v, fk)
    keep = obs['hit_valid'] & (obs['group'] >= removed)
    events = obs['events'].clone()
    events[..., 14] -= elapsed[:, None]
    nxt.update(events=events*keep[..., None], hit_valid=keep,
               group=(obs['group']-removed).clamp_min(0))
    return nxt


def suffix_actions(actions, removed):
    out = torch.zeros_like(actions)
    if removed < MAX_HITS:
        out[:, :-removed] = actions[:, removed:]
    return out


def discounted_returns(segment_scores, times, valid, *, gamma=.98, scale=20.):
    """Q of following the supplied finite plan; reward credited at each deadline."""
    B, G = times.shape
    start = torch.cat((times.new_zeros(B, 1), times[:, :-1]), 1)
    discounts = gamma**(times-start)
    rewards = discounts*segment_scores/scale*valid
    returns = torch.zeros_like(rewards)
    carry = times.new_zeros(B)
    for g in reversed(range(G)):
        carry = (rewards[:, g]+discounts[:, g]*carry)*valid[:, g]
        returns[:, g] = carry
    return returns, rewards, discounts


@dataclass
class ScheduleNetConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    max_hits: int = MAX_HITS


class ScheduleEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.state_embed = nn.Linear(40, cfg.width)
        self.hit_embed = nn.Linear(16, cfg.width)
        self.hand_embed = nn.Embedding(2, cfg.width)
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, dim_feedforward=2*cfg.width,
            dropout=0, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, obs, actions):
        hands = actions.gather(1, obs['group']) ^ obs['member']
        hit = self.hit_embed(obs['events'])+self.hand_embed(hands)
        tokens = torch.cat((self.state_embed(obs['state'])[:, None], hit), 1)
        padding = torch.cat((torch.zeros(len(hit), 1, dtype=torch.bool, device=hit.device),
                             ~obs['hit_valid']), 1)
        encoded = self.norm(self.transformer(tokens, src_key_padding_mask=padding))
        # Pool only live hits; terminal suffixes remain finite with no hits.
        pooled = (encoded[:, 1:]*obs['hit_valid'][..., None]).sum(1)/obs['hit_valid'].sum(1).clamp_min(1)[:, None]
        return encoded[:, 1:], torch.cat((encoded[:, 0], pooled), -1)


class JointScheduleCritic(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ScheduleNetConfig()
        self.encoder = ScheduleEncoder(self.cfg)
        self.value = nn.Sequential(nn.Linear(2*self.cfg.width, self.cfg.width), nn.GELU(),
                                   nn.Linear(self.cfg.width, 1))

    def forward(self, obs, actions):
        _, pooled = self.encoder(obs, actions)
        return self.value(pooled).squeeze(-1)


class ScheduleEditPolicy(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ScheduleNetConfig()
        self.encoder = ScheduleEncoder(self.cfg)
        w = self.cfg.width
        self.stop = nn.Linear(2*w, 1)
        self.single = nn.Sequential(nn.Linear(3*w, w), nn.GELU(), nn.Linear(w, 1))
        self.pair = nn.Sequential(nn.Linear(4*w, w), nn.GELU(), nn.Linear(w, 1))
        self.register_buffer('pairs', torch.tensor(list(combinations(range(MAX_HITS), 2))))

    def forward(self, obs, actions):
        hit, pooled = self.encoder(obs, actions)
        B, _, w = hit.shape
        groups = hit.new_zeros(B, MAX_HITS, w).scatter_add(
            1, obs['group'][..., None].expand(-1, -1, w), hit*obs['hit_valid'][..., None])
        counts = hit.new_zeros(B, MAX_HITS).scatter_add(1, obs['group'], obs['hit_valid'].float())
        groups /= counts.clamp_min(1)[..., None]
        single = self.single(torch.cat((groups, pooled[:, None].expand(-1, MAX_HITS, -1)), -1)).squeeze(-1)
        pair = self.pair(torch.cat((groups[:, self.pairs[:, 0]], groups[:, self.pairs[:, 1]],
                                   pooled[:, None].expand(-1, len(self.pairs), -1)), -1)).squeeze(-1)
        logits = torch.cat((self.stop(pooled), single, pair), -1)
        return logits.masked_fill(~valid_edits(obs), -1e9)


@torch.no_grad()
def critic_values(critic, obs, candidates, *, chunk=512):
    B, C, G = candidates.shape
    owners = torch.arange(B, device=candidates.device).repeat_interleave(C)
    flat = candidates.reshape(-1, G)
    values = []
    for begin in range(0, len(flat), chunk):
        values.append(critic(index_obs(obs, owners[begin:begin+chunk]), flat[begin:begin+chunk]))
    return torch.cat(values).reshape(B, C)


@torch.no_grad()
def critic_search(critic, obs, seed, *, steps=3):
    actions = seed.clone()
    mask = valid_edits(obs)
    for _ in range(steps):
        candidates = neighbors(actions)
        values = critic_values(critic, obs, candidates).masked_fill(~mask, -torch.inf)
        choice = values.argmax(1)
        actions = candidates[torch.arange(len(actions), device=actions.device), choice]
    return actions


@torch.no_grad()
def policy_search(policy, obs, seed, *, steps=3, critic=None):
    actions = seed.clone()
    alive = torch.ones(len(actions), dtype=torch.bool, device=actions.device)
    edits = edit_masks(actions.device)
    for _ in range(steps):
        choice = policy(obs, actions).argmax(1)
        proposed = actions ^ edits[choice]
        accept = alive & (choice != 0)
        if critic is not None:
            accept &= critic(obs, proposed) > critic(obs, actions)+1e-5
        actions = torch.where(accept[:, None], proposed, actions)
        alive &= accept
    return actions


def teacher_edits(bank_actions, returns, group_count):
    """Best observed single/pair edit in an analytically evaluated bank.

    These are bounded local-search labels, not global-optimum labels. STOP is
    relative to the evaluated neighborhood; ties preserve the current plan.
    """
    masks = edit_masks('cpu').numpy()
    labels = np.zeros(len(bank_actions), np.int64)
    gains = np.zeros(len(bank_actions), np.float32)
    lookup = {tuple(a): i for i, a in enumerate(bank_actions)}
    for i, action in enumerate(bank_actions):
        best = float(returns[i])
        for label, edit in enumerate(masks):
            if edit[group_count:].any():
                continue
            other = lookup.get(tuple(action ^ edit))
            if other is not None and returns[other] > best+1e-5:
                labels[i] = label
                best = float(returns[other])
        gains[i] = best-float(returns[i])
    return labels, gains
