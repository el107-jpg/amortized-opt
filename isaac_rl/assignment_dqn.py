"""Double DQN with event-time discounts and an unrestricted two-action head."""
from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from isaac_rl.arm_assignment import AssignmentConfig, AssignmentPolicy


class AssignmentQPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(nn.Linear(cfg.obs_dim, cfg.hidden), nn.Tanh(),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh())
        self.q_head = nn.Linear(cfg.hidden, 2)

    def forward(self, obs):
        return self.q_head(self.body(obs))

    def with_queue_config(self, *, commit_lead_s, pending_per_arm):
        """Transfer semantic feature blocks when expanding pending slots.

        Right-hand pending events and future events move in the flat input;
        a prefix copy would incorrectly reinterpret their learned weights.
        """
        from dataclasses import replace
        cfg = replace(self.cfg, commit_lead_s=commit_lead_s, pending_per_arm=pending_per_arm)
        new = type(self)(cfg).to(next(self.parameters()).device)
        if cfg.num_drums != self.cfg.num_drums or pending_per_arm < self.cfg.pending_per_arm:
            raise ValueError("queue migration requires the same kit and expanded pending capacity")
        prefix = 40+7*cfg.num_drums+2*(cfg.num_drums+2)
        width = cfg.num_drums+3
        old_pending, new_pending = self.cfg.pending_per_arm*width, cfg.pending_per_arm*width
        with torch.no_grad():
            new.body[0].weight.zero_()
            new.body[0].weight[:, :prefix].copy_(self.body[0].weight[:, :prefix])
            for hand in (0, 1):
                old_start, new_start = prefix+hand*old_pending, prefix+hand*new_pending
                new.body[0].weight[:, new_start:new_start+old_pending].copy_(
                    self.body[0].weight[:, old_start:old_start+old_pending])
            old_future, new_future = prefix+2*old_pending, prefix+2*new_pending
            new.body[0].weight[:, new_future:].copy_(self.body[0].weight[:, old_future:])
            new.body[0].bias.copy_(self.body[0].bias)
            new.body[2].load_state_dict(self.body[2].state_dict())
            new.q_head.load_state_dict(self.q_head.state_dict())
        return new

    @classmethod
    def from_ppo(cls, blob, *, future_hits=32, lookahead_beats=8):
        old_cfg = AssignmentConfig(**blob["policy_config"])
        old = AssignmentPolicy(old_cfg)
        old.load_state_dict(blob["policy"])
        cfg = AssignmentConfig(**{**asdict(old_cfg), "future_hits": future_hits,
                                  "lookahead_beats": lookahead_beats})
        new = cls(cfg)
        with torch.no_grad():
            new.body[0].weight.zero_()
            new.body[0].weight[:, :old_cfg.obs_dim].copy_(old.body[0].weight)
            new.body[0].bias.copy_(old.body[0].bias)
            new.body[2].load_state_dict(old.body[2].state_dict())
            # Preserve the learned state value and greedy arm preference.
            # Actor logits give only a weak initial advantage; Q magnitudes
            # subsequently come entirely from executed-reward Bellman targets.
            new.q_head.weight.copy_(old.critic.weight.expand(2, -1)
                                   + 0.1*(old.actor.weight-old.actor.weight.mean(0, keepdim=True)))
            new.q_head.bias.copy_(old.critic.bias.expand(2)
                                 + 0.1*(old.actor.bias-old.actor.bias.mean()))
        return new


class Replay:
    def __init__(self, obs_dim, capacity=50000, seed=0):
        self.obs = np.empty((capacity, obs_dim), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.action = np.empty(capacity, np.int64)
        self.reward = np.empty(capacity, np.float32)
        self.discount = np.empty(capacity, np.float32)
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add_episode(self, records):
        for i, record in enumerate(records):
            j = self.cursor
            self.obs[j] = record["obs"]
            self.next_obs[j] = records[i+1]["obs"] if i+1 < len(records) else 0
            self.action[j] = record["action"]
            self.reward[j] = record["reward"]
            self.discount[j] = record["discount"] if i+1 < len(records) else 0
            self.cursor = (self.cursor+1) % self.capacity
            self.size = min(self.size+1, self.capacity)

    def sample(self, batch, device):
        ids = self.rng.integers(self.size, size=batch)
        return tuple(torch.as_tensor(x[ids], device=device) for x in
                     (self.obs, self.action, self.reward, self.discount, self.next_obs))

    def state_dict(self):
        return {"obs": self.obs[:self.size].copy(), "next_obs": self.next_obs[:self.size].copy(),
                "action": self.action[:self.size].copy(), "reward": self.reward[:self.size].copy(),
                "discount": self.discount[:self.size].copy(), "size": self.size, "cursor": self.cursor,
                "capacity": self.capacity,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if self.capacity != state.get("capacity", 50000):
            raise ValueError("replay capacity differs from checkpoint; use its original capacity when resuming")
        self.size, self.cursor = state["size"], state["cursor"]
        for key in ("obs", "next_obs", "action", "reward", "discount"):
            getattr(self, key)[:self.size] = state[key]
        self.rng.bit_generator.state = state["rng"]


class DoubleDQN:
    def __init__(self, policy, *, lr=3e-4, tau=0.01):
        self.policy = policy
        self.target = copy.deepcopy(policy).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.tau = tau
        self.updates = 0

    def update(self, replay, *, gradient_steps=32, batch=128):
        device = next(self.policy.parameters()).device
        losses, q_means, td_errors = [], [], []
        for _ in range(gradient_steps):
            obs, action, reward, discount, next_obs = replay.sample(batch, device)
            prediction = self.policy(obs).gather(1, action[:, None]).squeeze(1)
            with torch.no_grad():
                # Online network selects; lagged target network evaluates.
                next_action = self.policy(next_obs).argmax(1)
                target_q = self.target(next_obs).gather(1, next_action[:, None]).squeeze(1)
                target = reward+discount*target_q
            loss = F.smooth_l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Double DQN loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            with torch.no_grad():
                for dst, src in zip(self.target.parameters(), self.policy.parameters()):
                    dst.lerp_(src, self.tau)
            self.updates += 1
            losses.append(loss.item()); q_means.append(prediction.mean().item())
            td_errors.append((prediction-target).abs().mean().item())
        return {"value_bellman_loss": float(np.mean(losses)), "q_mean": float(np.mean(q_means)),
                "td_error": float(np.mean(td_errors)), "gradient_updates": self.updates,
                "replay_size": replay.size}


class AnchoredDoubleDQN(DoubleDQN):
    """Small Bellman updates with a fixed behavior anchor and rollback guard.

    The guard bounds disagreement on measured probe states; it is not a global
    guarantee on every possible observation. Common shifts in Q remain free.
    """
    def __init__(self, policy, *, lr=1e-5, tau=.005, anchor_weight=5., temperature=1.,
                 max_flip_fraction=.05, max_kl=.02):
        super().__init__(policy, lr=lr, tau=tau)
        self.reference = copy.deepcopy(policy).eval().requires_grad_(False)
        self.anchor_weight, self.temperature = anchor_weight, temperature
        self.max_flip_fraction, self.max_kl = max_flip_fraction, max_kl

    @torch.no_grad()
    def drift(self, observations):
        device = next(self.policy.parameters()).device
        obs = torch.as_tensor(observations, device=device, dtype=torch.float32)
        current, ref = self.policy(obs), self.reference(obs)
        p = F.softmax(ref/self.temperature, -1)
        kl = (p*(F.log_softmax(ref/self.temperature,-1)-F.log_softmax(current/self.temperature,-1))).sum(-1)
        return dict(flip_fraction=float((current.argmax(-1)!=ref.argmax(-1)).float().mean()),
                    mean_kl=float(kl.mean()),max_kl=float(kl.max()))

    def update(self, replay, *, gradient_steps=128, batch=128, guard_obs=None):
        if not replay.size: raise ValueError('Isaac replay is empty')
        if guard_obs is None or not len(guard_obs): raise ValueError('Measured guard observations are required')
        before = {name:copy.deepcopy(getattr(self,name).state_dict()) for name in ('policy','target','optimizer')}
        updates_before = self.updates
        device = next(self.policy.parameters()).device
        td_losses=[]; anchor_losses=[]
        for _ in range(gradient_steps):
            obs, action, reward, discount, nxt = replay.sample(batch, device)
            q = self.policy(obs)
            prediction = q.gather(1, action[:,None]).squeeze(1)
            with torch.no_grad():
                selected = self.policy(nxt).argmax(1)
                target = reward+discount*self.target(nxt).gather(1,selected[:,None]).squeeze(1)
                ref = self.reference(obs)
                p = F.softmax(ref/self.temperature,-1)
            td_loss=F.smooth_l1_loss(prediction,target)
            anchor=(p*(F.log_softmax(ref/self.temperature,-1)-F.log_softmax(q/self.temperature,-1))).sum(-1).mean()
            loss=td_loss+self.anchor_weight*anchor
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite anchored DQN loss')
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(),.25); self.optimizer.step()
            with torch.no_grad():
                for dst,src in zip(self.target.parameters(),self.policy.parameters()): dst.lerp_(src,self.tau)
            self.updates+=1; td_losses.append(float(td_loss.detach())); anchor_losses.append(float(anchor.detach()))
        proposed=self.drift(guard_obs)
        # The same behavior bound also covers the entire stored training replay.
        training_drift=self.drift(replay.obs[:replay.size])
        accepted = (max(proposed['flip_fraction'],training_drift['flip_fraction']) <= self.max_flip_fraction
                    and max(proposed['mean_kl'],training_drift['mean_kl']) <= self.max_kl)
        if not accepted:
            for name,state in before.items(): getattr(self,name).load_state_dict(state)
            self.updates=updates_before
            for group in self.optimizer.param_groups: group['lr']*=.5
        final=self.drift(guard_obs)
        return dict(value_bellman_loss=float(np.mean(td_losses)),anchor_kl=float(np.mean(anchor_losses)),
            gradient_updates=self.updates,replay_size=replay.size,update_accepted=int(accepted),
            proposed_guard_flip_fraction=proposed['flip_fraction'],proposed_guard_mean_kl=proposed['mean_kl'],
            proposed_training_flip_fraction=training_drift['flip_fraction'],
            guard_flip_fraction=final['flip_fraction'],guard_mean_kl=final['mean_kl'],
            learning_rate=self.optimizer.param_groups[0]['lr'])
