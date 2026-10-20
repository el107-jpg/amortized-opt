"""Paired plan preferences that update real conditional transition times.

All learned parameters and exported chart anchors are seconds. The scorer is
the original timing/slack scorer; a fixed-scaled pairwise score difference
defines the preference likelihood. There is no auxiliary plan-cost correction.
"""
from __future__ import annotations

import copy
from itertools import product
from pathlib import Path
import pickle

import numpy as np

from isaac_rl.assignment_search import hit_groups
from isaac_rl.chart_cma import assigned_clip
from isaac_rl.min_time_assignment import graph_actions, window_scores
from isaac_rl.transition_assignment import AssignmentScoreConfig
from isaac_rl.plan_preferences import sigmoid

DESIGN_FIELDS = ('indices', 'basis', 'span', 'fixed_cost', 'variable')
TIMING_SCORING_VERSION = 'timing_only_no_historical_penalties_v2'


def remove_legacy_calibration_costs(design):
    """Migrate cached v1 replay: retain finite ready timing costs, strip history.

    v1 ready-start trials always began at1 s, and finite ready calibration was
    at most260 ms. The exact1000 constant therefore identifies missing data,
    not a genuine timing deficit. Known trainable edges held only support cost
    in fixed_cost; their actual timing cost is recomputed from current anchors.
    """
    migrated = {key: value.copy() for key, value in design.items()}
    fixed, variable = migrated['fixed_cost'], migrated['variable']
    missing = ~variable & np.isclose(fixed, 1000., rtol=0., atol=1e-10)
    fixed[variable | missing] = 0.
    return migrated


def timing_scores_and_jacobian(parameters_s, design, cfg=AssignmentScoreConfig()):
    """Exact original scores and derivatives with respect to time anchors/s."""
    indices = design['indices']
    basis = design['basis']
    variable = design['variable']
    times = (np.asarray(parameters_s)[indices.clip(0)]*basis).sum(-1)
    slack = design['span']-times
    argument = (cfg.comfortable_slack_s-5*cfg.slack_scale_s-slack)/cfg.slack_scale_s
    shoulder = np.logaddexp(0, argument)
    deficit = np.maximum(-slack, 0)
    costs = cfg.slack_weight*shoulder**2+cfg.deficit_weight*(deficit/cfg.slack_scale_s)**2
    values = (design['fixed_cost']+np.where(variable, costs, 0)).sum(-1)
    derivative = (2*cfg.slack_weight*shoulder*sigmoid(argument)/cfg.slack_scale_s+
                  2*cfg.deficit_weight*deficit/cfg.slack_scale_s**2)
    leading = indices.shape[:-2]
    jacobian = np.zeros((*leading, len(parameters_s)))
    flat = jacobian.reshape(-1, len(parameters_s))
    rows = np.broadcast_to(np.arange(len(flat))[:, None, None],
        indices.reshape(len(flat), -1, 3).shape)
    live = indices.reshape(len(flat), -1, 3) >= 0
    increments = (derivative[..., None]*basis).reshape(len(flat), -1, 3)
    np.add.at(flat, (rows[live], indices.reshape(len(flat), -1, 3)[live]), increments[live])
    return values, jacobian


def pad_design(design, edge_capacity):
    result = {}
    for key in DESIGN_FIELDS:
        value = design[key]
        width = edge_capacity-value.shape[1]
        if width < 0:
            raise ValueError('Plan exceeds timing replay capacity')
        pads = [(0, 0), (0, width)]+[(0, 0)]*(value.ndim-2)
        result[key] = np.pad(value, pads, constant_values=-1 if key == 'indices' else 0)
    return result


def curve_minima(anchors):
    left, middle, right = np.asarray(anchors).T
    b, c = (right-left)/2, (left+right)/2-middle
    vertex = np.divide(-b, 2*c, out=np.zeros_like(c), where=c != 0)
    interior = (c > 0) & (np.abs(vertex) < 1)
    return np.minimum(np.minimum(left, right), np.where(interior,
        middle+b*vertex+c*vertex**2, np.inf))


class TransitionTimePreferenceLearner:
    format = 'direct_transition_time_preferences_v1'

    def __init__(self, chart, *, prior_std_ms=50., max_update_ms=50.,
                 temperature=.25, score_scale=10., reward_temperature=.05,
                 deadband=.05, gap_scale=.10, indifference_weight=.05,
                 replay_capacity=16384, groups=4):
        settings = [prior_std_ms, max_update_ms, temperature, score_scale,
                    reward_temperature, gap_scale, replay_capacity]
        if not np.isfinite(settings).all() or min(settings) <= 0:
            raise ValueError('Positive finite time/preference settings required')
        self.template = copy.deepcopy(chart)
        self.initial_s = chart.get_parameters()
        self.prior_std_s = float(prior_std_ms)/1000
        self.max_update_s = float(max_update_ms)/1000
        self.mean = np.zeros(len(self.initial_s))  # Standardized timing residual, not a utility.
        self.covariance = np.eye(len(self.mean))*self.prior_std_s**2
        self.temperature, self.score_scale = float(temperature), float(score_scale)
        self.reward_temperature = float(reward_temperature)
        self.deadband, self.gap_scale = float(deadband), float(gap_scale)
        self.indifference_weight = float(indifference_weight)
        self.replay_capacity, self.groups = int(replay_capacity), int(groups)
        self.design = None
        self.reference_scales = None
        self.targets, self.weights = np.empty(0), np.empty(0)
        self.updates = 0
        self.scoring_version = TIMING_SCORING_VERSION

    def set_max_update_ms(self, value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Maximum timing update must be positive finite milliseconds')
        self.max_update_s = float(value)/1000

    @property
    def parameters_s(self):
        return self.initial_s+self.prior_std_s*self.mean

    def current_chart(self):
        chart = copy.deepcopy(self.template)
        chart.set_parameters(self.parameters_s)
        return chart

    def add_preferences(self, design, rewards, pair_weights=None):
        rewards = np.asarray(rewards, float)
        if rewards.ndim != 2 or rewards.shape[1] != 2:
            raise ValueError('Only matched first/second ranked plan pairs supported')
        gap = rewards[:, 0]-rewards[:, 1]
        targets = sigmoid(gap/self.reward_temperature)
        weights = np.maximum(self.indifference_weight,
            np.minimum(1., (np.abs(gap)-self.deadband)/self.gap_scale))
        if pair_weights is not None:
            pair_weights = np.asarray(pair_weights, float)
            if pair_weights.shape != gap.shape or not np.isfinite(pair_weights).all() or (pair_weights < 0).any():
                raise ValueError('Invalid pair weights')
            weights *= pair_weights
        # Identical timing score functions cannot supply parameter direction.
        identifiable = np.zeros(len(rewards), bool)
        for key in DESIGN_FIELDS:
            difference = design[key][:, 0] != design[key][:, 1]
            identifiable |= difference.reshape(len(rewards), -1).any(1)
        weights[~identifiable] = 0
        for key in DESIGN_FIELDS:
            if design[key].shape[:2] != rewards.shape:
                raise ValueError('Timing design/reward pairing differs')
        self.design = ({k: v.copy() for k, v in design.items()} if self.design is None else
            {k: np.concatenate((self.design[k], design[k]))[-self.replay_capacity:] for k in DESIGN_FIELDS})
        self.targets = np.r_[self.targets, targets][-self.replay_capacity:]
        self.weights = np.r_[self.weights, weights][-self.replay_capacity:]
        self.reference_scales = None  # Rebuild fixed initial-chart pair scales for retained replay.
        return dict(new_pairs=len(rewards), replay_pairs=len(self.weights),
            near_tie_pairs=int((np.abs(gap) <= self.deadband).sum()),
            unidentifiable_pairs=int((~identifiable).sum()))

    def add_ranked_preferences(self, design, rewards, valid):
        """Use every observed legal pair, with equal total weight per rhythm.

        The K(K-1)/2 comparisons share K physical outcomes; averaging their
        loss prevents larger legal banks from being treated as independent
        extra rhythms. Invalid padded slots never enter replay.
        """
        rewards, valid = np.asarray(rewards, float), np.asarray(valid, bool)
        if rewards.shape != valid.shape or rewards.ndim != 2:
            raise ValueError('Ranked rewards/valid mask differ')
        if not np.isfinite(rewards[valid]).all():
            raise ValueError('Observed rewards must be finite')
        for key in DESIGN_FIELDS:
            if design[key].shape[:2] != rewards.shape:
                raise ValueError('Ranked timing design/rewards differ')
        rows, first, second, weights = [], [], [], []
        for row, mask in enumerate(valid):
            ids = np.flatnonzero(mask)
            pairs = [(int(a), int(b)) for i, a in enumerate(ids) for b in ids[i+1:]]
            for a, b in pairs:
                rows.append(row); first.append(a); second.append(b)
                weights.append(1./len(pairs))
        if not rows:
            raise ValueError('At least one observed candidate pair required')
        rows = np.asarray(rows)
        slots = np.stack((first, second), axis=1)
        paired_design = {key: design[key][rows[:, None], slots] for key in DESIGN_FIELDS}
        paired_rewards = rewards[rows[:, None], slots]
        metrics = self.add_preferences(paired_design, paired_rewards, weights)
        metrics['pair_weight_sum'] = float(np.sum(weights))
        metrics['pair_weighting'] = 'mean_pairwise_loss_per_rhythm'
        return metrics

    def logits_and_jacobian(self, mean):
        if self.design is None:
            return np.empty(0), np.empty((0, len(mean)))
        parameters = self.initial_s+self.prior_std_s*mean
        scores, jacobian = timing_scores_and_jacobian(parameters, self.design)
        if getattr(self, 'reference_scales', None) is None:
            initial_scores, _ = timing_scores_and_jacobian(self.initial_s, self.design)
            self.reference_scales = np.maximum(self.score_scale,
                np.abs(initial_scores[:, 1]-initial_scores[:, 0]))
        denominator = self.reference_scales*self.temperature
        difference = scores[:, 1]-scores[:, 0]
        # Shared edges/costs cancel exactly. The normalizer is fixed relative
        # to the initial chart and never gives shared parameters a fake gradient.
        difference_jacobian = (jacobian[:, 1]-jacobian[:, 0])*self.prior_std_s
        return difference/denominator, difference_jacobian/denominator[:, None]

    def objective(self, mean):
        logits, _ = self.logits_and_jacobian(mean)
        return float(.5*np.dot(mean, mean)+np.dot(self.weights,
            np.logaddexp(0, logits)-self.targets*logits))

    def derivatives(self, mean):
        logits, jacobian = self.logits_and_jacobian(mean)
        p = sigmoid(logits)
        gradient = mean+jacobian.T@(self.weights*(p-self.targets))
        curvature = self.weights*p*(1-p)
        precision = np.eye(len(mean))+(jacobian.T*curvature)@jacobian
        return gradient, precision

    def feasible_trial(self, proposed, before):
        lo, hi = before-self.max_update_s/self.prior_std_s, before+self.max_update_s/self.prior_std_s
        trial = np.clip(proposed, lo, hi)
        parameters = self.initial_s+self.prior_std_s*trial
        anchors = parameters.reshape(-1, 3)
        # A common shift of all three anchors shifts the entire parabola equally.
        shift = np.maximum(self.template.minimum_s-curve_minima(anchors), 0)
        anchors = anchors+shift[:, None]
        trial = (anchors.reshape(-1)-self.initial_s)/self.prior_std_s
        if (trial < lo-1e-10).any() or (trial > hi+1e-10).any():
            return None
        try:
            candidate = copy.deepcopy(self.template)
            candidate.set_parameters(self.initial_s+self.prior_std_s*trial)
        except ValueError:
            return None
        return trial

    def fit(self, max_steps=8):
        before = self.mean.copy()
        loss_before = self.objective(self.mean)
        steps = 0
        for _ in range(max_steps):
            gradient, precision = self.derivatives(self.mean)
            step = np.linalg.solve(precision, gradient)
            if float(gradient@step) < 1e-12:
                break
            loss = self.objective(self.mean)
            scale = 1.
            for _ in range(20):
                trial = self.feasible_trial(self.mean-scale*step, before)
                if trial is not None:
                    movement = trial-self.mean
                    slope = float(gradient@movement)
                    if slope < -1e-14 and self.objective(trial) <= loss+1e-4*slope:
                        self.mean = trial
                        steps += 1
                        break
                scale *= .5
            else:
                break
        _, precision = self.derivatives(self.mean)
        covariance = np.linalg.solve(precision, np.eye(len(self.mean)))*self.prior_std_s**2
        self.covariance = (covariance+covariance.T)/2
        self.current_chart()  # Exact interior-vertex floor validation.
        self.updates += 1
        change_ms = (self.mean-before)*self.prior_std_s*1000
        eigen = np.linalg.eigvalsh(self.covariance)
        return dict(preference_loss_before=loss_before, preference_loss_after=self.objective(self.mean),
            fit_steps=steps, time_step_rms_ms=float(np.sqrt(np.mean(change_ms**2))),
            time_step_median_abs_ms=float(np.median(np.abs(change_ms))),
            time_step_max_abs_ms=float(np.abs(change_ms).max()),
            changed_time_anchors=int((np.abs(change_ms) > 1e-9).sum()),
            posterior_std_min_ms=float(np.sqrt(eigen.min())*1000),
            posterior_std_median_ms=float(np.sqrt(np.median(eigen))*1000),
            posterior_std_max_ms=float(np.sqrt(eigen.max())*1000))

    def save(self, path, **extra):
        path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_bytes(pickle.dumps(dict(format=self.format, learner=self, extra=extra), protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        saved = pickle.loads(Path(path).read_bytes())
        if saved.get('format') != cls.format:
            raise ValueError('Not a direct transition-time learner checkpoint')
        learner, extra = saved['learner'], saved['extra']
        if getattr(learner, 'scoring_version', None) != TIMING_SCORING_VERSION:
            if learner.design is not None:
                learner.design = remove_legacy_calibration_costs(learner.design)
            learner.reference_scales = None
            learner.scoring_version = TIMING_SCORING_VERSION
            # Re-rank unfinished contexts with the new score instead of silently
            # executing candidates picked using the removed historical penalty.
            extra['pending'] = None
            extra['historical_penalties_removed_on_resume'] = True
        return learner, extra


def select_time_context(clip, chart, ready, candidates=2):
    if not isinstance(candidates, int) or candidates < 2:
        raise ValueError('At least two ranked candidates required')
    groups = hit_groups(clip)
    bank = np.asarray(list(product(*[chart.choices(g) for g in groups])))
    if len(bank) < 2:
        raise ValueError('A genuine second-ranked plan is required')
    scored = window_scores(groups, bank, chart, initial_drums=(-1, -1), last_times=(0., 0.),
        initial_minimum_s=ready, return_timing_design=True)
    graph, _ = graph_actions(groups)
    order = np.lexsort((np.arange(len(bank)), (bank != graph).sum(1), scored['total']))[:candidates]
    design = pad_design({k: v[order] for k, v in scored['timing_design'].items()}, 2*len(groups))
    return dict(episodes=[assigned_clip(clip, bank[i]) for i in order], actions=bank[order],
        scores=scored['total'][order], design=design, selected_search_ranks=list(range(1, len(order)+1)))


def build_time_batch(panel, chart, ready, candidates=2):
    contexts = [select_time_context(clip, chart, ready, candidates) for clip in panel]
    valid = np.zeros((len(contexts), candidates), bool)
    report_indices = np.full(valid.shape, -1, int)
    episodes = []
    for k in range(candidates):
        for row, context in enumerate(contexts):
            if k < len(context['episodes']):
                valid[row, k] = True
                report_indices[row, k] = len(episodes)
                episodes.append(context['episodes'][k])
    def padded(value):
        # Values are rectangular for cached scorer parity, but all padding is
        # masked and never simulated, learned from, or counted in diagnostics.
        return np.concatenate((value, np.repeat(value[-1:], candidates-len(value), axis=0)))
    return dict(episodes=episodes,
        actions=np.stack([padded(c['actions']) for c in contexts]),
        scores=np.stack([padded(c['scores']) for c in contexts]),
        design={key: np.stack([padded(c['design'][key]) for c in contexts]) for key in DESIGN_FIELDS},
        valid=valid, report_indices=report_indices, contexts=contexts)


def unpack_time_reports(batch, reports):
    """Restore context/rank layout after rank-major, variable-bank execution."""
    if 'report_indices' not in batch:  # Previously saved top-two pending trials.
        count = len(batch['scores'])
        indices = np.arange(2*count).reshape(2, count).T
        valid = np.ones(indices.shape, bool)
    else:
        indices, valid = batch['report_indices'], batch['valid']
    if int(valid.sum()) != len(reports):
        raise ValueError('Physical reports differ from legal trial count')
    rewards, hits = np.full(indices.shape, np.nan), np.full(indices.shape, np.nan)
    for name, result in (('reward', rewards), ('hit_rate', hits)):
        result[valid] = np.asarray([r[name] for r in reports])[indices[valid]]
    return rewards, hits, valid
