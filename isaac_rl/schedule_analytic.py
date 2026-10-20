"""Batched schedule evaluation in the existing analytic trajectory environment."""
from __future__ import annotations

import torch

from isaac_rl.schedule_improvement import discounted_returns


class AnalyticScheduleEnvironment:
    def __init__(self, generator, surrogate, *, chunk=128, gamma=.98, reward_scale=20.):
        self.generator, self.surrogate = generator, surrogate
        self.chunk, self.gamma, self.reward_scale = chunk, gamma, reward_scale

    def packed(self, table, q, v, actions, owners, draws):
        """Bounce draws belong to physical hit ordinals, not chosen hands."""
        G = int(table['valid'].sum(1).max())
        swap = actions[:, :G, None].bool()
        result = dict(q0=q[owners], v0=v[owners],
                      times=table['times'][owners, :G], valid=table['valid'][owners, :G])
        for field in ('drums', 'speeds'):
            raw = table[field][owners, :G]
            result[field] = torch.where(swap, raw.flip(-1), raw)
        result['efficiency'] = torch.where(swap, draws[:, :G].flip(-1), draws[:, :G])
        return result

    @torch.no_grad()
    def score(self, table, q, v, candidates, draws):
        B, C, _ = candidates.shape
        S = draws.shape[1]
        owners = torch.arange(B, device=q.device).repeat_interleave(C*S)
        actions = candidates[:, :, None].expand(-1, -1, S, -1).reshape(-1, candidates.shape[-1])
        eta = draws[:, None].expand(-1, C, -1, -1, -1).reshape(-1, draws.shape[-2], 2)
        parts = []
        collocations = 0
        keys = ('score', 'hits', 'requested', 'hit_mask', 'gap', 'torque_excess',
                'velocity_excess', 'collision_m', 'joint_excess_rad', 'segment_score',
                'poses', 'pre_velocity', 'post_velocity')
        for begin in range(0, len(actions), self.chunk):
            packed = self.packed(table, q, v, actions[begin:begin+self.chunk],
                                 owners[begin:begin+self.chunk], eta[begin:begin+self.chunk])
            result = self.surrogate.rollout(self.generator, packed, return_segments=True)
            if not torch.isfinite(result['score']).all():
                raise FloatingPointError('Nonfinite analytic schedule score')
            collocations += result['collocations']
            parts.append({k: result[k] for k in keys})
        merged = {k: torch.cat([p[k] for p in parts]) for k in keys}
        times = table['times'][owners, :merged['segment_score'].shape[1]]
        valid = table['valid'][owners, :times.shape[1]]
        returns, rewards, discounts = discounted_returns(merged['segment_score'], times, valid,
            gamma=self.gamma, scale=self.reward_scale)
        merged.update(returns=returns, rewards=rewards, discounts=discounts)
        merged = {k: value.reshape(B, C, S, *value.shape[1:]) for k, value in merged.items()}
        merged['collocations'] = collocations
        return merged

    @torch.no_grad()
    def refresh_rollout(self, table, q0, v0, draws, choose):
        """Refresh all proposed future assignments after each actual hit group.

        Motion still uses the frozen decoder, including both arms' next timed
        goals. No decision queue is irrevocably preassigned for a full second.
        """
        from amortized_opt.trajectory_surrogate import rebound_velocity
        B = len(q0)
        G = int(table['valid'].sum(1).max())
        q, v = q0.clone(), v0.clone()
        executed = torch.zeros(B, table['valid'].shape[1], dtype=torch.long, device=q.device)
        poses, velocities = [], []
        for g in range(G):
            elapsed = table['times'].new_zeros(B) if g == 0 else table['times'][:, g-1]
            proposed = choose(q, v, elapsed, g)
            executed[:, g:] = proposed[:, :executed.shape[1]-g]
            packed = self.packed(table, q, v, executed, torch.arange(B, device=q.device), draws)
            pose, pre = self.generator.predict(q, v, packed['drums'], packed['times'],
                packed['speeds'], packed['valid'], g)
            gap, J = self.surrogate.contact(pose[:, None], packed['drums'][:, g:g+1])
            approach = torch.einsum('bhd,bd->bh', J[:, 0], pre)
            hit = ((packed['drums'][:, g] >= 0) & packed['valid'][:, g, None]
                   & (gap[:, 0] <= 1e-5) & (approach < -.1))
            post = rebound_velocity(pre, J[:, 0], self.surrogate.inverse_mass,
                                    packed['efficiency'][:, g], hit)
            poses.append(pose); velocities.append(pre)
            q, v = pose, post
        packed = self.packed(table, q0, v0, executed, torch.arange(B, device=q.device), draws)
        poses, velocities = torch.stack(poses, 1), torch.stack(velocities, 1)
        result = self.surrogate.evaluate(q0, v0, poses, torch.zeros_like(velocities),
            packed['drums'], packed['times'], packed['speeds'], packed['valid'], packed['efficiency'],
            pre_velocity_override=velocities, return_segments=True)
        result['returns'], _, _ = discounted_returns(result['segment_score'], packed['times'],
            packed['valid'], gamma=self.gamma, scale=self.reward_scale)
        result['actions'] = executed
        return result
