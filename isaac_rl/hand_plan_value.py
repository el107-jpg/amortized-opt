"""Pointwise v4 hand-plan reward labels and absolute-reward diagnostics."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from isaac_rl.hand_plan_ranker import ACTION_BANK, OBS_DIM, HandPlanRanker
from isaac_rl.hand_plan_dataset import dataset_metrics, load_dataset_batches


SPLIT_NAMES = ('train', 'validation', 'test')


class DualRuntimeRewardModel(nn.Module):
    """Shared observations/plan encoding with distinct v4 and legacy scorers."""
    def __init__(self, config):
        super().__init__()
        if config.architecture != 'shared_candidate':
            raise ValueError('Runtime heads need explicit candidate encoding')
        self.v4 = HandPlanRanker(config)
        self.v2_scorer = deepcopy(self.v4.candidate_scorer.scorer)

    def forward(self, observations, *, runtime='v4', plans=None):
        if runtime == 'v4':
            return self.v4.candidate_scorer(observations, plans)
        if runtime == 'v2':
            encoded = self.v4.candidate_scorer.encode(observations, plans)
            return self.v2_scorer(encoded).squeeze(-1)
        raise ValueError('Unknown label runtime')

    def score_candidates(self, observations, plans):
        return self.forward(observations, runtime='v4', plans=plans)


def load_legacy_plan_cases(folder, expected_decoder_sha, expected_geometry):
    """Read v2 labels only as their own runtime, preserving whole-case splits."""
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    run = json.loads((folder/'run.json').read_text())
    if run['status'] != 'complete' or config['rollout_runtime']['rollout_runtime_version'] != 'matched_decoder_speed_contact_reset_v2':
        raise ValueError('Expected complete v2 contact-reset labels')
    if (config['checkpoint_sha256'] != expected_decoder_sha or
            config['simulation_geometry_version'] != expected_geometry):
        raise ValueError('Legacy decoder/geometry differs from v4')
    data = load_dataset_batches(folder)
    splits = torch.load(folder/'splits.pt', map_location='cpu', weights_only=False)
    ids = np.concatenate([np.asarray(splits[name], np.int64) for name in SPLIT_NAMES])
    if (len(data['observations']) != run['cases'] or
            not np.array_equal(np.sort(ids), np.arange(run['cases'])) or
            int(data['observed'].sum()) != run['rollouts']):
        raise ValueError('Legacy cases/splits/rollouts differ from manifest')
    if not np.isfinite(data['observations']).all() or not np.isfinite(data['rewards'][data['observed']]).all():
        raise ValueError('Invalid legacy reward labels')
    return dict(cases=data, splits=splits, config=config, run=run)


def load_value_dataset(folder):
    """Expand observed full plans and append factual pre-hit continuations.

    The split unit is the original rhythm/start case, including all its plans
    and all snapshots from those plans. Unexecuted plans are never labels.
    """
    folder = Path(folder)
    splits = torch.load(folder/'splits.pt', map_location='cpu', weights_only=False)
    counts = {name: len(splits[name]) for name in SPLIT_NAMES}
    case_split = np.full(sum(counts.values()), -1, np.int8)
    for code, name in enumerate(SPLIT_NAMES):
        ids = np.asarray(splits[name], np.int64)
        if (case_split[ids] >= 0).any(): raise ValueError('Overlapping case splits')
        case_split[ids] = code
    if (case_split < 0).any(): raise ValueError('Case splits do not cover all cases')

    cases = {key: [] for key in ('observations', 'rewards', 'observed', 'hits', 'requested')}
    points = {key: [] for key in ('observations', 'actions', 'rewards', 'case_ids', 'source')}
    offset = 0
    files = sorted((folder/'batches').glob('batch_*.pt'))
    if not files: raise ValueError('No plan batches found')
    for path in files:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        batch, outcome, factual = saved['batch'], saved['outcomes'], saved['factual']
        obs = np.asarray(batch['observations'], np.float32)
        valid = np.asarray(batch['observed'], bool)
        count = len(obs)
        if int(batch['case_offset']) != offset or valid.shape != (count, 16):
            raise ValueError('Noncontiguous cases or wrong plan mask')
        if not np.array_equal(valid, batch['legal']):
            raise ValueError('Missing legal-plan rollout label')
        if not np.array_equal(batch['split_codes'], case_split[offset:offset+count]):
            raise ValueError('Batch split differs from case split')
        for key in cases: cases[key].append(obs if key == 'observations' else np.asarray(
            valid if key == 'observed' else outcome[key]))
        row, action = np.nonzero(valid)
        points['observations'].append(obs[row])
        points['actions'].append(action.astype(np.int64))
        points['rewards'].append(np.asarray(outcome['rewards'][row, action], np.float32))
        points['case_ids'].append((row+offset).astype(np.int64))
        points['source'].append(np.zeros(len(row), np.int8))

        factual_case = np.asarray(factual['case_ids'], np.int64)
        factual_action = np.asarray(factual['actions'], np.int64)
        local = factual_case-offset
        if len(local) and ((local < 0).any() or (local >= count).any()):
            raise ValueError('Continuation row refers outside its source batch')
        if len(local) and not valid[local, factual_action].all():
            raise ValueError('Continuation row uses an unexecuted plan')
        if not np.array_equal(np.asarray(factual['splits']), case_split[factual_case]):
            raise ValueError('Continuation row crossed case split')
        if np.any(np.asarray(factual['observations'])[:, 19] <= 0):
            raise ValueError('Continuation has no remaining first-hit deadline')
        points['observations'].append(np.asarray(factual['observations'], np.float32))
        points['actions'].append(factual_action)
        points['rewards'].append(np.asarray(factual['rewards'], np.float32))
        points['case_ids'].append(factual_case)
        points['source'].append(np.ones(len(factual_case), np.int8))
        offset += count
    if offset != len(case_split): raise ValueError('Case count differs from split count')
    cases = {key: np.concatenate(value) for key, value in cases.items()}
    points = {key: np.concatenate(value) for key, value in points.items()}
    points['splits'] = case_split[points['case_ids']]
    if (points['observations'].shape[1] != OBS_DIM or
            not np.isfinite(points['observations']).all() or
            not np.isfinite(points['rewards']).all() or
            not np.isin(points['actions'], np.arange(16)).all()):
        raise ValueError('Invalid pointwise reward dataset')
    return dict(points=points, cases=cases, splits=splits, split_counts=counts)


def score_rows(model, observations, actions, *, chunk=4096):
    """One scalar prediction per executed candidate, without scoring all16."""
    device = next(model.parameters()).device
    bank = torch.as_tensor(ACTION_BANK, device=device)
    outputs = []
    for start in range(0, len(actions), chunk):
        stop = min(start+chunk, len(actions))
        obs = torch.as_tensor(observations[start:stop], device=device)
        plans = bank[torch.as_tensor(actions[start:stop], device=device)][:, None, :]
        outputs.append(model.score_candidates(obs, plans)[:, 0])
    return torch.cat(outputs) if outputs else torch.empty(0, device=device)


def regression_metrics(predicted, actual):
    predicted = np.asarray(predicted, np.float64)
    actual = np.asarray(actual, np.float64)
    error = predicted-actual
    mse = float(np.mean(error**2))
    variance = float(np.mean((actual-actual.mean())**2))
    return dict(rows=len(actual), mse=mse, rmse=float(np.sqrt(mse)),
        mae=float(np.mean(np.abs(error))), bias=float(error.mean()),
        r2=float(1-mse/variance) if variance > 0 else None,
        target_mean=float(actual.mean()), prediction_mean=float(predicted.mean()))


def reward_ranking_losses(predicted, actual, observed, *, pair_temperature=.1,
                          winner_temperature=.05):
    """Proper relative-reward losses for observed plans sharing one start.

    Both terms are zero when predicted rewards match actual rewards, so the
    auxiliary objective does not require inflating the absolute reward scale.
    Singleton banks contribute zero and unexecuted plans are masked out.
    """
    if predicted.shape != actual.shape or predicted.shape != observed.shape:
        raise ValueError('Plan scores, rewards, and observed mask must match')
    if predicted.ndim != 2 or not observed.any(1).all():
        raise ValueError('Each case needs at least one observed plan')
    if min(pair_temperature, winner_temperature) <= 0:
        raise ValueError('Ranking temperatures must be positive')
    observed = observed.bool()
    actual = torch.where(observed, actual, 0.)
    target_log = F.log_softmax((actual/winner_temperature).masked_fill(~observed, -1e9), dim=1)
    target = target_log.exp()
    predicted_log = F.log_softmax((predicted/winner_temperature).masked_fill(~observed, -1e9), dim=1)
    winner_kl = (target*(target_log-predicted_log)).sum(1).mean()

    gap = actual[:, :, None]-actual[:, None, :]
    predicted_gap = predicted[:, :, None]-predicted[:, None, :]
    target_pair = torch.sigmoid(gap/pair_temperature)
    pair_logit = predicted_gap/pair_temperature
    cross_entropy = F.binary_cross_entropy_with_logits(pair_logit, target_pair, reduction='none')
    safe_target = target_pair.clamp(1e-6, 1-1e-6)
    target_entropy = -(target_pair*safe_target.log()+(1-target_pair)*torch.log1p(-safe_target))
    triangle = torch.ones(predicted.shape[1], predicted.shape[1], dtype=torch.bool,
                          device=predicted.device).triu(1)
    pairs = observed[:, :, None]&observed[:, None, :] & triangle[None]
    weights = (gap.abs()/.1).clamp(.05, 1.)*pairs
    pair_kl = ((cross_entropy-target_entropy)*weights).sum((1, 2))/weights.sum((1, 2)).clamp_min(1)
    return pair_kl.mean(), winner_kl


@torch.inference_mode()
def evaluate_value_model(model, data, *, row_chunk=4096, case_chunk=512):
    model.eval()
    points, cases = data['points'], data['cases']
    result = {}
    for code, name in enumerate(SPLIT_NAMES):
        selected = np.flatnonzero(points['splits'] == code)
        predictions = []
        for start in range(0, len(selected), row_chunk):
            ids = selected[start:start+row_chunk]
            predictions.append(score_rows(model, points['observations'][ids], points['actions'][ids], chunk=row_chunk).cpu().numpy())
        predictions = np.concatenate(predictions)
        stats = regression_metrics(predictions, points['rewards'][selected])
        for source, source_name in ((0, 'full'), (1, 'continuation')):
            mask = points['source'][selected] == source
            values = regression_metrics(predictions[mask], points['rewards'][selected][mask])
            stats.update({f'{source_name}_{key}': value for key, value in values.items()})
        case_ids = np.asarray(data['splits'][name], np.int64)
        scores = []
        device = next(model.parameters()).device
        for start in range(0, len(case_ids), case_chunk):
            ids = case_ids[start:start+case_chunk]
            scores.append(model(torch.as_tensor(cases['observations'][ids], device=device)).cpu().numpy())
        scores = np.concatenate(scores)
        stats.update({f'selection_{key}': value for key, value in dataset_metrics(scores, cases, case_ids).items()})
        result[name] = stats
    return result


@torch.inference_mode()
def evaluate_legacy_runtime(model, legacy, *, case_chunk=512):
    """Read-only v2 diagnostics, kept separate from primary v4 metrics."""
    model.eval()
    cases = legacy['cases']
    result = {}
    device = next(model.parameters()).device
    for name in ('validation', 'test'):
        ids = np.asarray(legacy['splits'][name], np.int64)
        scores = []
        for start in range(0, len(ids), case_chunk):
            chunk = ids[start:start+case_chunk]
            scores.append(model(torch.as_tensor(cases['observations'][chunk], device=device),
                runtime='v2').cpu().numpy())
        scores = np.concatenate(scores)
        valid = cases['observed'][ids]
        stats = regression_metrics(scores[valid], cases['rewards'][ids][valid])
        stats.update({f'selection_{key}': value for key, value in
            dataset_metrics(scores, cases, ids).items()})
        result[name] = stats
    return result
