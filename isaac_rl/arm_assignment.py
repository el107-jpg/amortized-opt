"""Small, unrestricted arm policy and event queue (no Isaac imports).

One categorical action per strike group: 0 gives its first drum to the left,
1 gives it to the right. A second simultaneous drum uses the other arm.
There are no drum eligibility masks, including floor tom and hihat.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from amortized_opt.constants import PELVIS_IN_ENV
from amortized_opt.hand_assign import (
    LIMB_LEFT, LIMB_RIGHT, assign_simultaneous, kit_graph_from_layout,
)
from amortized_opt.rhythm import DrumKit, Hit, RhythmClip


@dataclass
class AssignmentConfig:
    num_drums: int = 7
    future_hits: int = 12
    pending_per_arm: int = 2
    hidden: int = 64
    lookahead_s: float = 2.0
    lookahead_beats: float = 0.0
    time_scale_s: float = 2.0
    # None preserves older checkpoints' shared lookahead/commit window.
    commit_lead_s: float | None = None

    @property
    def obs_dim(self):
        # q, qd, tips; geometry; last assigned drum/time; pending; future.
        d = self.num_drums
        return 40 + 7*d + 2*(d+2) + (2*self.pending_per_arm+self.future_hits)*(d+3)


class AssignmentPolicy(nn.Module):
    def __init__(self, cfg: AssignmentConfig):
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(nn.Linear(cfg.obs_dim, cfg.hidden), nn.Tanh(),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh())
        self.actor = nn.Linear(cfg.hidden, 2)
        self.critic = nn.Linear(cfg.hidden, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        hidden = self.body(obs)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)


class AssignmentQueue:
    """Observe future groups and commit them as their execution window opens.

    Every decision is immutable. Keeping the original hit ordinal allows the
    planner table and Isaac's credit table to receive exactly the same hand.
    Groups are defined on the CONTROL grid, not by the MIDI file's hand labels.
    More than two simultaneous requested strikes are rejected, never dropped.
    """
    def __init__(self, clip: RhythmClip, kit: DrumKit, cfg: AssignmentConfig):
        if kit.num_drums != cfg.num_drums:
            raise ValueError("policy drum count differs from the kit")
        self.clip, self.kit, self.cfg = clip, kit, cfg
        self.hits = sorted(clip.hits, key=lambda h: (h.time, h.drum))
        self.groups = [list(items) for _, items in groupby(
            range(len(self.hits)), key=lambda i: self.hits[i].time)]
        if any(len(g) > 2 for g in self.groups):
            raise ValueError("more than two requested strikes on one control tick")
        self.hands: dict[int, str] = {}
        self.next_group = 0
        self.graph = kit_graph_from_layout()
        self.parked = {LIMB_LEFT: None, LIMB_RIGHT: None}

    def due(self, now: float):
        lead = self.cfg.lookahead_s if self.cfg.commit_lead_s is None else self.cfg.commit_lead_s
        return (self.next_group < len(self.groups)
                and self.hits[self.groups[self.next_group][0]].time <= now+lead+1e-8)

    def teacher_action(self):
        ids = self.groups[self.next_group]
        names = [self.hits[i].drum_name for i in ids]
        assignment = dict(assign_simultaneous(names, self.parked, self.graph))
        if len(assignment) != len(names):
            raise ValueError("graph teacher cannot assign this group")
        return int(assignment[names[0]] == LIMB_RIGHT)

    def commit(self, action: int):
        if action not in (0, 1):
            raise ValueError("assignment action must be 0 or 1")
        ids = self.groups[self.next_group]
        out = []
        for j, i in enumerate(ids):
            hand = ("left", "right")[(action+j) % 2]
            self.hands[i] = hand
            self.parked[LIMB_LEFT if hand == "left" else LIMB_RIGHT] = self.hits[i].drum_name
            out.append((i, replace(self.hits[i], hand=hand)))
        self.next_group += 1
        return out

    def assigned_clip(self):
        return replace(self.clip, hits=[replace(h, hand=self.hands[i])
                       for i, h in enumerate(self.hits) if i in self.hands], ineligible=0)

    def observation(self, now: float, q, qd, tips) -> np.ndarray:
        d = self.cfg.num_drums
        def event(h: Hit | None):
            x = np.zeros(d+3, np.float32)
            if h is not None:
                x[h.drum] = 1
                x[d:] = [np.clip((h.time-now)/self.cfg.time_scale_s, -2, 8), h.velocity, 1]
            return x
        blocks = [np.asarray(q, np.float32)/np.pi, np.asarray(qd, np.float32)/10,
                  np.asarray(tips, np.float32).reshape(-1),
                  np.concatenate([self.kit.positions-PELVIS_IN_ENV,
                                  self.kit.axes, self.kit.radii[:, None]], axis=1).reshape(-1)]
        for hand in ("left", "right"):
            assigned = [h for i, h in enumerate(self.hits) if self.hands.get(i) == hand]
            last = np.zeros(d+2, np.float32)
            if assigned:
                h = assigned[-1]
                last[h.drum] = 1
                last[d:] = [np.clip((h.time-now)/self.cfg.time_scale_s, -2, 8), 1]
            blocks.append(last)
        for hand in ("left", "right"):
            pending = [h for i, h in enumerate(self.hits)
                       if self.hands.get(i) == hand and h.time > now+1e-8]
            if self.cfg.commit_lead_s is not None and len(pending) > self.cfg.pending_per_arm:
                raise ValueError("pending assignment capacity exceeded; refusing to hide committed hits")
            blocks.extend(event(pending[j] if j < len(pending) else None)
                          for j in range(self.cfg.pending_per_arm))
        start = self.groups[self.next_group][0] if self.next_group < len(self.groups) else len(self.hits)
        upcoming = self.hits[start:]
        if self.cfg.commit_lead_s is not None:
            upcoming = [h for h in upcoming if h.time <= now+self.cfg.lookahead_s+1e-8]
        upcoming = upcoming[:self.cfg.future_hits]
        blocks.extend(event(upcoming[j] if j < len(upcoming) else None)
                      for j in range(self.cfg.future_hits))
        obs = np.concatenate(blocks).astype(np.float32)
        if obs.shape != (self.cfg.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError("invalid assignment observation")
        return obs


def decision_gae(rewards, values, discounts, *, lam=0.95):
    """SMDP GAE: interval rewards already discounted from decision time.

    Complete episodes only: terminal bootstrap is zero. `discounts[i]` is
    gamma_per_second ** elapsed_seconds until the next decision, or zero at
    the episode boundary. Multiple decisions on one tick have discount one.
    """
    rewards, values, discounts = map(lambda x: torch.as_tensor(x, dtype=torch.float32),
                                     (rewards, values, discounts))
    advantages = torch.zeros_like(rewards)
    carry = torch.tensor(0.0)
    for i in reversed(range(len(rewards))):
        next_value = values[i+1] if i+1 < len(values) else 0.0
        delta = rewards[i] + discounts[i]*next_value - values[i]
        carry = delta + discounts[i]*lam*carry
        advantages[i] = carry
    return advantages+values, advantages


def ppo_update(policy, optimizer, episodes, *, epochs=4, batch_size=128,
               clip=0.2, entropy_weight=0.02, value_weight=0.5, target_kl=0.02):
    device = next(policy.parameters()).device
    obs = torch.as_tensor(np.stack([r["obs"] for e in episodes for r in e]), device=device)
    actions = torch.tensor([r["action"] for e in episodes for r in e], device=device)
    old_lp = torch.tensor([r["log_prob"] for e in episodes for r in e], device=device)
    targets, advantages = [], []
    for episode in episodes:
        ret, adv = decision_gae([r["reward"] for r in episode], [r["value"] for r in episode],
                                [r["discount"] for r in episode])
        targets.append(ret); advantages.append(adv)
    returns = torch.cat(targets).to(device)
    adv = torch.cat(advantages).to(device)
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    logs = []
    for _ in range(epochs):
        order = torch.randperm(len(obs), device=device)
        for ids in order.split(batch_size):
            dist, values = policy(obs[ids])
            log_ratio = dist.log_prob(actions[ids])-old_lp[ids]
            ratio = log_ratio.exp()
            actor_loss = -torch.minimum(ratio*adv[ids], ratio.clamp(1-clip, 1+clip)*adv[ids]).mean()
            value_loss = (values-returns[ids]).square().mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + value_weight*value_loss - entropy_weight*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite assignment PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            kl = ((ratio-1)-log_ratio).mean().item()
            logs.append([actor_loss.item(), value_loss.item(), entropy.item(), kl])
        if kl > target_kl:
            break
    means = np.mean(logs, axis=0)
    return dict(zip(("policy_loss", "value_bellman_loss", "entropy", "approx_kl"), means.tolist()),
                decisions=len(obs))
