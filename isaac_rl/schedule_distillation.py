"""Four-hit joint schedule values distilled from exhaustive decoder search."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from amortized_opt.constants import STICK_TIP_SIGMA
from amortized_opt.fk import stick_contact_offset
from amortized_opt.limits import urdf_position_limits, isaac_velocity_limits
from amortized_opt.trajectory_surrogate import contact_jacobian


def schedule_bank(device='cpu'):
    return (torch.arange(16, device=device)[:, None] //
            torch.tensor([8, 4, 2, 1], device=device)) & 1


def suffix_table(table, elapsed, removed):
    """Drop completed hits, subtract elapsed time, and zero-pad to four slots."""
    B = len(elapsed); result = {}
    for name in ('drums', 'speeds', 'times', 'valid'):
        src = table[name][:, :4]
        fill = -1 if name == 'drums' else 0
        dst = torch.full_like(src, fill)
        if removed < 4:
            dst[:, :4-removed] = src[:, removed:]
        result[name] = dst
    result['times'] = torch.where(result['valid'], result['times']-elapsed[:, None], 0.)
    return result


class FourHitScheduleScorer(nn.Module):
    """86 normalized features -> 16 complete finite-schedule values.

    Output indices use binary L=0 / R=1 in chronological order. Value units
    are the undiscounted analytic score divided by ``reward_scale``.
    """
    format = 'four_hit_schedule_scorer_v1'
    input_dim = 86

    def __init__(self, *, reward_scale=20., hidden_dims=(256, 256, 128)):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims or any(not isinstance(w, int) or w < 1 for w in self.hidden_dims):
            raise ValueError('Hidden dimensions must be positive integers')
        lo, hi = urdf_position_limits(posture_limits=False)
        self.register_buffer('q_center', torch.tensor((lo+hi)/2, dtype=torch.float32))
        self.register_buffer('q_half', torch.tensor((hi-lo)/2, dtype=torch.float32).clamp_min(.01))
        self.register_buffer('v_scale', torch.tensor(isaac_velocity_limits(), dtype=torch.float32))
        self.register_buffer('bank', schedule_bank())
        self.reward_scale = reward_scale
        layers = []; previous = self.input_dim
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU()))
            previous = width
        layers.append(nn.Linear(previous, 16))
        self.body = nn.Sequential(*layers)

    @torch.no_grad()
    def features(self, table, q, v, fk):
        drums = table['drums'][:, :4]; valid = table['valid'][:, :4].bool()
        if drums.shape[1:] != (4, 2):
            raise ValueError('Expected four physical event slots, padded if necessary')
        if (((drums >= 0).sum(-1) != 1)&valid).any():
            raise ValueError('This first model supports single-hit events only')
        d = drums.amax(-1)
        if ((d >= 7)&valid).any():
            raise ValueError('This model requires the fixed seven-drum layout')
        tips = []; velocities = []
        for hand in ('left', 'right'):
            local = q.new_tensor(stick_contact_offset(hand, STICK_TIP_SIGMA)).expand(len(q), -1)
            p, J = contact_jacobian(fk, q, hand, local)
            tips.append(p); velocities.append(torch.einsum('bij,bj->bi', J, v))
        state = torch.cat(((q-self.q_center)/self.q_half, v/self.v_scale,
                           torch.cat(tips, -1), torch.cat(velocities, -1)/3.), -1)
        speed = (table['speeds'][:, :4]*(drums >= 0)).sum(-1)/3.
        hits = torch.cat((F.one_hot(d.clamp(0, 6), 7).to(q.dtype),
                          table['times'][:, :4, None], speed[..., None], valid[..., None]), -1)
        hits = hits*valid[..., None]
        return torch.cat((state, hits.flatten(1)), -1)

    def forward(self, features):
        values = self.body(features)
        # The empty finite horizon has exactly zero remaining return.
        live = features[:, 46:].reshape(-1, 4, 10)[..., -1].bool().any(-1)
        return values*live[:, None]

    @torch.no_grad()
    def choose(self, features):
        valid = features[:, 46:].reshape(-1, 4, 10)[..., -1].bool()
        # Canonicalize unused padding bits. There are no drum-to-hand masks.
        allowed = ~((self.bank[None].bool() & ~valid[:, None]).any(-1))
        values = self(features).masked_fill(~allowed, -torch.inf)
        choice = values.argmax(-1)
        return choice, self.bank[choice]

    @torch.no_grad()
    def assign(self, table, q, v, fk):
        return self.choose(self.features(table, q, v, fk))[1]

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get('format') != cls.format:
            raise ValueError('Not a four-hit schedule scorer checkpoint')
        model = cls(reward_scale=saved['reward_scale'], hidden_dims=saved.get('hidden_dims', (256,256,128))).to(device)
        model.load_state_dict(saved['model'])
        return model.eval().requires_grad_(False)


def relative_score_loss(predicted, target):
    """Regress differences within each window, independent of its baseline."""
    return F.smooth_l1_loss(predicted-predicted.mean(-1, keepdim=True),
                           target-target.mean(-1, keepdim=True))


def value_and_ranking_loss(predicted, target, *, minimum_gap=.25/20., temperature=.5,
                           top_k=0, gap_cap=4.):
    """Absolute regression and optional top-schedule, gap-weighted ranking.

    Scores, minimum_gap, temperature and gap_cap are in normalized value units.
    top_k=0 preserves the original uniform all-pairs ranking. Otherwise pairs
    must include one of the teacher's top_k schedules, and their contribution
    is weighted by the measured score gap (capped to limit extreme failures).
    The weighted mean is computed per window before averaging across windows.
    """
    if not 0 <= top_k <= predicted.shape[-1]:
        raise ValueError('top_k must be between zero and the number of schedules')
    if temperature <= 0 or gap_cap <= 0:
        raise ValueError('temperature and gap_cap must be positive')
    value = F.smooth_l1_loss(predicted, target)
    a, b = torch.triu_indices(predicted.shape[-1], predicted.shape[-1], offset=1, device=predicted.device)
    delta = target[:, a]-target[:, b]
    clear = delta.abs() >= minimum_gap
    ranking = F.softplus(-delta.sign()*(predicted[:, a]-predicted[:, b])/temperature)
    if top_k:
        top = torch.zeros_like(target, dtype=torch.bool).scatter_(1, target.topk(top_k, -1).indices, True)
        weights = delta.abs().clamp_max(gap_cap)*clear*(top[:, a]|top[:, b])
        ranking = ((ranking*weights).sum(-1)/weights.sum(-1).clamp_min(1e-12)).mean()
    else:
        ranking = (ranking*clear).sum()/clear.sum().clamp_min(1)
    return value, ranking


@torch.no_grad()
def selection_metrics(model, features, values, graph_index):
    predicted = model(features)*model.reward_scale
    choice, _ = model.choose(features)
    rows = torch.arange(len(values), device=values.device)
    best = values.max(-1).values
    selected = values[rows, choice]
    regret = best-selected
    graph = values[rows, graph_index]
    return dict(regret_mean=float(regret.mean()), regret_p95=float(torch.quantile(regret, .95)),
        search_agreement=float((choice == values.argmax(-1)).float().mean()),
        within_one_score=float((regret <= 1.).float().mean()),
        value_mae=float((predicted-values).abs().mean()), selected_score=float(selected.mean()),
        search_score=float(best.mean()), graph_score=float(graph.mean()),
        graph_regret=float((best-graph).mean()), gain_over_graph=float((selected-graph).mean()))
