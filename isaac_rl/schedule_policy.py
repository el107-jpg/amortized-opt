"""Categorical four-hit policies and finite-schedule policy objectives."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from isaac_rl.schedule_distillation import FourHitScheduleScorer


class FourHitSchedulePolicy(FourHitScheduleScorer):
    """Same observation and MLP as the scorer; outputs are policy logits."""
    format = 'four_hit_schedule_policy_v1'

    def __init__(self, *, reward_scale=20., temperature=1., hidden_dims=(256,256,128)):
        super().__init__(reward_scale=reward_scale, hidden_dims=hidden_dims)
        if temperature <= 0:
            raise ValueError('Policy temperature must be positive')
        self.temperature = temperature

    def log_probs(self, features):
        valid = features[:, 46:].reshape(-1, 4, 10)[..., -1].bool()
        allowed = ~((self.bank[None].bool() & ~valid[:, None]).any(-1))
        logits = (self(features)/self.temperature).masked_fill(~allowed, -torch.inf)
        return F.log_softmax(logits, -1)

    @classmethod
    def from_scorer_checkpoint(cls, path, device='cpu', temperature=1.):
        source = FourHitScheduleScorer.from_checkpoint(path, device)
        policy = cls(reward_scale=source.reward_scale, temperature=temperature, hidden_dims=source.hidden_dims).to(device)
        policy.load_state_dict(source.state_dict())
        return policy.eval()

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get('format') != cls.format:
            raise ValueError('Not a four-hit schedule policy checkpoint')
        policy = cls(reward_scale=saved['reward_scale'], temperature=saved['temperature'],
                     hidden_dims=saved.get('hidden_dims', (256,256,128))).to(device)
        policy.load_state_dict(saved['model'])
        return policy.eval().requires_grad_(False)


def entropy_and_kl(log_probs, reference_log_probs):
    """Exact categorical entropy and KL, including safely masked padding."""
    p = log_probs.exp()
    live = torch.isfinite(log_probs)
    own = log_probs.masked_fill(~live, 0.)
    ref = reference_log_probs.masked_fill(~live, 0.)
    entropy = -(p*own).sum(-1)
    kl = (p*(own-ref)).sum(-1)
    return entropy, kl


def group_advantages(rewards, *, std_floor=1.):
    """Population-standardized outcome advantages; rewards in raw points.

    A one-point standard-deviation floor dampens nearly tied/noisy groups.
    Detached advantages keep environment rewards out of the gradient graph.
    """
    if rewards.shape[-1] < 2 or std_floor <= 0:
        raise ValueError('Need at least two outcomes and a positive std floor')
    return ((rewards-rewards.mean(-1, keepdim=True))/
            rewards.std(-1, keepdim=True, unbiased=False).clamp_min(std_floor)).detach()


def clipped_group_loss(log_probs, actions, old_log_probs, advantages, *, clip=.2):
    log_ratio = log_probs.gather(1, actions)-old_log_probs.detach()
    ratio = log_ratio.clamp(-20, 20).exp()
    advantage = advantages.detach()
    objective = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    return -objective.mean(), ((ratio-1).abs()>clip).float().mean().detach()


def expected_reward_loss(log_probs, rewards, *, reward_scale=20.):
    """Sum over all actions, eliminating action-sampling variance.

    Dynamics rewards still use Monte Carlo bounce draws. A detached per-window
    baseline improves numerics without changing the expected-reward gradient.
    """
    centered = (rewards-rewards.mean(-1, keepdim=True)).detach()/reward_scale
    return -(log_probs.exp()*centered).sum(-1).mean()


@torch.no_grad()
def policy_metrics(policy, reference, features, rewards, graph_index):
    lp = policy.log_probs(features); ref = reference.log_probs(features)
    entropy, kl = entropy_and_kl(lp, ref)
    chosen = lp.argmax(-1); rows = torch.arange(len(features), device=features.device)
    best = rewards.max(-1).values; selected = rewards[rows, chosen]
    regret = best-selected; expected = (lp.exp()*rewards).sum(-1)
    return dict(regret_mean=float(regret.mean()), regret_p95=float(regret.quantile(.95)),
        search_agreement=float((chosen==rewards.argmax(-1)).float().mean()),
        within_one_score=float((regret<=1.).float().mean()), selected_score=float(selected.mean()),
        search_score=float(best.mean()), graph_score=float(rewards[rows, graph_index].mean()),
        stochastic_expected_score=float(expected.mean()), stochastic_regret=float((best-expected).mean()),
        entropy_bits=float(entropy.mean()/torch.log(entropy.new_tensor(2.))),
        reference_kl=float(kl.mean()),
        changed_from_reference=float((chosen!=ref.argmax(-1)).float().mean()))
