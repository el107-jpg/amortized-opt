"""Common-state schedule preferences and mining of critic ranking errors."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from isaac_rl.schedule_improvement import critic_values


def select_preference_pairs(predicted, score_samples, valid, rng, *, limit=32,
                            minimum_gap=.25, minimum_agreement=.75):
    """Pairs share a root state and matched physical rebound samples.

    Prefer wrong orderings and comparisons of the predicted winner with graph
    and the measured winner. Uniform pairs preserve broad ranking coverage.
    Noise filtering is symmetric: average separation and sample sign agreement.
    """
    predicted = np.asarray(predicted)
    score_samples = np.asarray(score_samples)
    valid = np.asarray(valid, dtype=bool)
    if predicted.shape != valid.shape or score_samples.shape[:2] != valid.shape:
        raise ValueError('candidate scores and validity must have matching shapes')
    owners, aa, bb, differences, hard = [], [], [], [], []
    for owner in range(len(predicted)):
        ids = np.flatnonzero(valid[owner])
        if len(ids) < 2:
            continue
        i, j = np.triu_indices(len(ids), 1)
        i, j = ids[i], ids[j]
        draws = score_samples[owner, i]-score_samples[owner, j]
        delta = draws.mean(-1)
        agree = ((draws > 0) == (delta[:, None] > 0)).mean(-1)
        clear = (np.abs(delta) >= minimum_gap) & (agree >= minimum_agreement)
        i, j, delta = i[clear], j[clear], delta[clear]
        wrong = ((predicted[owner, i]-predicted[owner, j]) > 0) != (delta > 0)
        if not len(i):
            continue
        winner = ids[np.argmax(predicted[owner, ids])]
        actual_winner = ids[np.argmax(score_samples[owner, ids].mean(-1))]
        selected = []
        for other in (0, actual_winner):
            match = np.flatnonzero(((i == winner) & (j == other)) | ((j == winner) & (i == other)))
            selected.extend(match.tolist())
        critical = np.flatnonzero(wrong | (i == winner) | (j == winner))
        selected.extend(rng.permutation(critical).tolist()[:max(0, limit//2-len(selected))])
        selected.extend(rng.permutation(len(i)).tolist())
        selected = list(dict.fromkeys(selected))[:limit]
        owners.extend([owner]*len(selected))
        aa.extend(i[selected].tolist()); bb.extend(j[selected].tolist())
        differences.extend(delta[selected].tolist()); hard.extend(wrong[selected].tolist())
    return dict(owners=np.asarray(owners, np.int64), candidate_a=np.asarray(aa, np.int64),
        candidate_b=np.asarray(bb, np.int64), delta=np.asarray(differences, np.float32),
        hard=np.asarray(hard, bool))


def preference_loss(predicted_a, predicted_b, delta, hard, *, temperature=.5, hard_weight=3.):
    """Weighted logistic ranking; positive delta requires A to score above B."""
    if len(delta) == 0:
        return (predicted_a.sum()+predicted_b.sum())*0
    sign = delta.sign()
    weights = 1+(hard_weight-1)*hard.to(predicted_a.dtype)
    loss = F.softplus(-sign*(predicted_a-predicted_b)/temperature)
    return (loss*weights).sum()/weights.sum()


@torch.no_grad()
def ranking_stats(critic, panel, *, minimum_gap=.25):
    predicted = critic_values(critic, panel['obs'], panel['bank'])
    actual, valid = panel['returns'], panel['valid']
    B, C = actual.shape
    rows = torch.arange(B, device=actual.device)
    choice = predicted.masked_fill(~valid, -torch.inf).argmax(1)
    measured_gain = actual[rows, choice]-actual[:, 0]
    changed = choice != 0
    best = actual.masked_fill(~valid, -torch.inf).max(1).values
    actual_difference = actual[:, :, None]-actual[:, None, :]
    predicted_difference = predicted[:, :, None]-predicted[:, None, :]
    mask = (valid[:, :, None] & valid[:, None, :]
            & torch.ones(C, C, dtype=torch.bool, device=actual.device).triu(1)[None]
            & (actual_difference.abs() >= minimum_gap))
    good = ((actual_difference > 0) == (predicted_difference > 0)) & mask
    return dict(mae=float((predicted-actual).abs()[valid].mean()),
        bank_selection_regret=float((best-actual[rows, choice]).mean()),
        selected_return=float(actual[rows, choice].mean()), best_bank_return=float(best.mean()),
        ranking_accuracy=float(good.sum()/mask.sum().clamp_min(1)),
        ranking_comparisons=int(mask.sum()), changed_schedules=int(changed.sum()),
        alternatives_better=int((changed & (measured_gain > 1e-5)).sum()),
        alternatives_worse=int((changed & (measured_gain < -1e-5)).sum()),
        alternative_improvement_fraction=float((changed & (measured_gain > 1e-5)).sum()/changed.sum().clamp_min(1)),
        mean_gain_over_graph=float(measured_gain.mean()))
