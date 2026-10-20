"""Matched plan preferences with a zero-initialized linear cost correction.

The full Laplace covariance belongs to cost coefficients, not timing minima.
Only feature differences influence preference fitting and Thompson exploration.
"""
from __future__ import annotations

from itertools import product
import pickle
from pathlib import Path
import hashlib
import json

import numpy as np

from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.min_time_assignment import graph_actions, window_scores


def normalize_plan_scores(scores):
    scores = np.asarray(scores, float)
    finite = np.isfinite(scores)
    result = np.full_like(scores, np.inf)
    if finite.any():
        low, high = scores[finite].min(), scores[finite].max()
        result[finite] = (scores[finite]-low)/max(high-low, 1.)
    return result


class PreferenceChart:
    """Chart-compatible inference adapter; base timing anchors stay frozen."""
    def __init__(self, base, coefficients):
        self.base = base
        self.coefficients = np.asarray(coefficients, float).copy()
        if self.coefficients.shape != base.get_parameters().shape:
            raise ValueError('One utility correction per chart anchor required')

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)

    def adjust_plan_scores(self, scores, features):
        return normalize_plan_scores(scores)+features@self.coefficients

    @classmethod
    def load(cls, path, base, base_path):
        with np.load(path, allow_pickle=False) as data:
            if str(data['format']) != 'plan_preference_cost_correction_v1':
                raise ValueError('Not a plan-preference correction')
            metadata = json.loads(str(data['metadata_json']))
            if hashlib.sha256(Path(base_path).read_bytes()).hexdigest() != metadata['base_sha256']:
                raise ValueError('Correction belongs to a different frozen base chart')
            result = cls(base, data['coefficients'])
            result.metadata = dict(base.metadata)
            result.metadata['parameter_search'] = dict(algorithm='Matched plan preferences',
                generation=metadata['iteration'], selection=metadata['selection'], correction=str(Path(path).resolve()))
            return result


def sigmoid(values):
    values = np.asarray(values, float)
    out = np.empty_like(values)
    live = values >= 0
    out[live] = 1/(1+np.exp(-values[live]))
    ez = np.exp(values[~live])
    out[~live] = ez/(1+ez)
    return out


class PlanPreferenceLearner:
    format = 'matched_plan_preferences_v1'

    def __init__(self, dimensions, *, prior_std=1., temperature=.25,
                 replay_capacity=16384, deadband=.05, gap_scale=.10,
                 reward_temperature=.05, indifference_weight=.05):
        if dimensions < 1 or min(prior_std, temperature, replay_capacity, gap_scale) <= 0 or deadband < 0:
            raise ValueError('Invalid preference dimensions, prior, temperature or replay settings')
        self.dimensions = int(dimensions)
        self.prior_std, self.temperature = float(prior_std), float(temperature)
        self.replay_capacity = int(replay_capacity)
        self.deadband, self.gap_scale = float(deadband), float(gap_scale)
        if reward_temperature <= 0 or not 0 <= indifference_weight <= 1:
            raise ValueError('Positive reward temperature and bounded indifference weight required')
        self.reward_temperature, self.indifference_weight = float(reward_temperature), float(indifference_weight)
        self.mean = np.zeros(dimensions)
        self.covariance = np.eye(dimensions)*prior_std**2
        self.factor = np.eye(dimensions)*prior_std
        self.features = np.empty((0, dimensions))
        self.offsets = np.empty(0)
        self.weights = np.empty(0)
        self.targets = np.empty(0)
        self.updates = 0

    def add_preferences(self, features, base_scores, rewards):
        features, base_scores, rewards = np.asarray(features), np.asarray(base_scores), np.asarray(rewards)
        if features.shape != (*rewards.shape, self.dimensions) or base_scores.shape != rewards.shape or rewards.ndim != 2:
            raise ValueError('Expected contexts x candidates x features and matching scores/rewards')
        if not all(np.isfinite(x).all() for x in (features, base_scores, rewards)):
            raise ValueError('Finite preference inputs required')
        x, offsets, weights, targets = [], [], [], []
        ties = unidentifiable = 0
        for f, b, r in zip(features, base_scores, rewards):
            cx, cb, cw, cy = [], [], [], []
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    gap = abs(r[i]-r[j])
                    if gap <= self.deadband:
                        ties += 1
                    difference = (f[j]-f[i])/self.temperature
                    if np.linalg.norm(difference) <= 1e-12:
                        unidentifiable += 1
                        continue
                    cx.append(difference)
                    cb.append((b[j]-b[i])/self.temperature)
                    cw.append(max(self.indifference_weight,
                                  min(1., (gap-self.deadband)/self.gap_scale)))
                    cy.append(float(sigmoid(np.array([(r[i]-r[j])/self.reward_temperature]))[0]))
            # Correlated pairs from one rhythm carry at most one context's
            # information, preventing artificial posterior overconfidence.
            if cw:
                denominator = max(sum(cw), 1.)
                x.extend(cx); offsets.extend(cb); weights.extend(w/denominator for w in cw); targets.extend(cy)
        if x:
            self.features = np.concatenate((self.features, x))[-self.replay_capacity:]
            self.offsets = np.r_[self.offsets, offsets][-self.replay_capacity:]
            self.weights = np.r_[self.weights, weights][-self.replay_capacity:]
            self.targets = np.r_[self.targets, targets][-self.replay_capacity:]
        return dict(new_pairs=len(x), near_tie_pairs=ties, unidentifiable_pairs=unidentifiable,
                    replay_pairs=len(self.weights), new_information_weight=float(sum(weights)))

    def objective(self, mean):
        logits = self.offsets+self.features@mean
        return float(.5*np.dot(mean, mean)/self.prior_std**2+
                     np.dot(self.weights, np.logaddexp(0, logits)-self.targets*logits))

    def derivatives(self, mean):
        p = sigmoid(self.offsets+self.features@mean)
        gradient = mean/self.prior_std**2+self.features.T@(self.weights*(p-self.targets))
        curvature = self.weights*p*(1-p)
        precision = (np.eye(self.dimensions)/self.prior_std**2+
                     (self.features.T*curvature)@self.features)
        return gradient, precision

    def fit(self, max_steps=8):
        before = self.objective(self.mean)
        steps = 0
        for _ in range(max_steps):
            gradient, precision = self.derivatives(self.mean)
            step = np.linalg.solve(precision, gradient)
            decrease = float(gradient@step)
            if decrease < 1e-12:
                break
            value = self.objective(self.mean)
            scale = 1.
            for _ in range(16):
                trial = self.mean-scale*step
                if self.objective(trial) <= value-1e-4*scale*decrease:
                    self.mean = trial
                    steps += 1
                    break
                scale *= .5
            else:
                break
        _, precision = self.derivatives(self.mean)
        self.covariance = np.linalg.solve(precision, np.eye(self.dimensions))
        self.covariance = (self.covariance+self.covariance.T)/2
        self.factor = np.linalg.cholesky(self.covariance)
        self.updates += 1
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        return dict(preference_loss_before=before, preference_loss_after=self.objective(self.mean),
            fit_steps=steps, covariance_min=float(eigenvalues.min()),
            covariance_median=float(np.median(eigenvalues)), covariance_max=float(eigenvalues.max()),
            coefficient_norm=float(np.linalg.norm(self.mean)))

    def sample(self, rng, exploration=1.):
        return self.mean+exploration*self.factor@rng.normal(size=self.dimensions)

    def uncertainty(self, differences):
        differences = np.asarray(differences)
        return np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', differences, self.covariance, differences), 0))

    def save(self, path, **extra):
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(pickle.dumps(dict(format=self.format, learner=self, extra=extra), protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        value = pickle.loads(Path(path).read_bytes())
        if value.get('format') != cls.format or not isinstance(value['learner'], cls):
            raise ValueError('Not a plan-preference checkpoint')
        return value['learner'], value['extra']


def select_context_plans(clip, chart, learner, ready, rng, candidates=2, exploration=1., selection='top_two'):
    """Full candidate windows from the same nominal ready state, committed."""
    groups = hit_groups(clip)
    bank = np.asarray(list(product(*[chart.choices(g) for g in groups])), int)
    scores = window_scores(groups, bank, chart, initial_drums=(-1, -1), last_times=(0., 0.),
        initial_minimum_s=ready, return_features=True)
    features = scores['features']
    base = normalize_plan_scores(scores['total'])
    corrected = base+features@learner.mean
    graph, _ = graph_actions(groups)
    distances = (bank != graph).sum(1)
    ranking = np.lexsort((np.arange(len(bank)), distances, corrected))
    greedy = int(ranking[0])
    graph_id = int(np.lexsort((np.arange(len(bank)), scores['total'], distances))[0])
    if selection == 'top_two':
        if candidates != 2:
            raise ValueError('Top-two selection requires exactly two candidates')
        if len(bank) < 2:
            raise ValueError('No distinct second-ranked plan exists for this window')
        chosen = ranking[:2].tolist()
    elif selection == 'thompson':
        chosen = [greedy]
        if graph_id != greedy:
            chosen.append(graph_id)
        while len(chosen) < min(candidates, len(bank)):
            sampled = base+features@learner.sample(rng, exploration)
            sampled[chosen] = np.inf
            chosen.append(int(np.argmin(sampled)))
    else:
        raise ValueError('Unknown plan selection method')
    # Rare windows with fewer eligible plans use explicit physical replicas.
    # Identical feature comparisons supply no training information.
    while len(chosen) < candidates:
        chosen.append(chosen[len(chosen) % min(candidates, len(bank))])
    selected = np.asarray(chosen)
    return dict(episodes=[assigned_clip(clip, bank[i]) for i in selected],
        actions=bank[selected], features=features[selected], base_scores=base[selected],
        corrected_scores=corrected[selected], greedy_slot=0,
        graph_slot=chosen.index(graph_id) if selection == 'thompson' else None,
        selection=selection, selected_search_ranks=[int(np.flatnonzero(ranking == i)[0])+1 for i in chosen],
        eligible_plans=len(bank), distinct_selected=len(set(chosen)),
        predictive_uncertainty=learner.uncertainty(features[selected]-features[greedy]))
