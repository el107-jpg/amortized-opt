"""Resumable CMA-ES for conditional planning anchors, independent of Isaac."""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from isaac_rl.conditional_min_time import chart_fingerprint, PROXY_RULE


def overlay_fingerprint(chart):
    digest = hashlib.sha256(chart_fingerprint(chart).encode())
    digest.update(PROXY_RULE.encode())
    digest.update(chart.proxy_values.tobytes())
    digest.update(np.asarray([chart.minimum_s, chart.idle_proxy]).tobytes())
    digest.update(chart.get_parameters().tobytes())
    # Preserve old fingerprints for overlays without initialized unknown cells.
    if chart.initialized_unknown_mask.any():
        digest.update(chart.initialized_unknown_mask.tobytes())
    return digest.hexdigest()


class ChartCMA:
    """Optimize normalized additive anchor residuals with a private RNG.

    Zero reproduces the supplied overlay. Invalid quadratics are rejected by
    the chart's existing validator and never sent to the simulator. The incumbent
    is compared against candidates on the *current* matched panel, not against
    historical scores from a different rhythm panel.
    """
    format = "conditional_chart_cma_v1"

    def __init__(self, chart, *, population=32, sigma=.15, seed=0,
                 regularization=.001, plateau_growth=1.5, sigma_max=1., max_resamples=128):
        try:
            import cma
        except ImportError as exc:
            raise ImportError("Install the assignment-search extra: pip install -e '.[assignment-search]'") from exc
        if population < 4 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Need population >=4 and finite positive sigma")
        if (not np.isfinite([regularization, plateau_growth, sigma_max]).all()
                or regularization < 0 or plateau_growth < 1 or sigma_max < sigma or max_resamples < 0):
            raise ValueError("Invalid regularization or exploration bounds")
        self.chart = copy.deepcopy(chart)
        self.fingerprint = overlay_fingerprint(chart)
        self.initial = chart.get_parameters()
        if len(self.initial) < 2:
            raise ValueError("Need at least two supported anchor parameters")
        self.scale = np.maximum(self.initial, .1)
        self.rng = np.random.RandomState(seed)
        self.es = cma.CMAEvolutionStrategy(np.zeros(len(self.initial)), sigma, {
            "popsize": population, "seed": seed, "randn": self.rng.randn,
            "CMA_mu": max(1, population//2),
            "verbose": -9, "verb_log": 0, "tolflatfitness": 1000000,
            "tolfun": 0., "tolfunhist": 0., "maxiter": 1000000,
        })
        self.population = population
        self.regularization = regularization
        self.plateau_growth = plateau_growth
        self.sigma_max = sigma_max
        self.max_resamples = max_resamples
        self.sampling_rejections = 0
        self.incumbent = np.zeros(len(self.initial))
        self.generation = 0
        self.pending = None

    def decode(self, residual):
        residual = np.asarray(residual, float)
        if residual.shape != self.initial.shape or not np.isfinite(residual).all():
            raise ValueError("Invalid normalized CMA candidate")
        return self.initial+self.scale*residual

    def candidate_chart(self, residual):
        result = copy.deepcopy(self.chart)
        result.set_parameters(self.decode(residual))
        return result

    def penalty(self, residual):
        return self.regularization*float(np.mean(np.asarray(residual)**2))

    def ask(self):
        # A checkpoint taken after ask can resume the same unfinished generation.
        if self.pending is None:
            self.pending = np.asarray(self.es.ask(), dtype=float)
            self.sampling_rejections = 0
            # Rejection sampling is important at a flat 20ms lower-bound curve:
            # shrinking sigma alone cannot change the probability of a negative
            # perturbation. Keep invalid proposals away from expensive physics.
            for i in range(self.population):
                for attempt in range(getattr(self, "max_resamples", 128)+1):
                    try:
                        self.candidate_chart(self.pending[i])
                        break
                    except ValueError:
                        self.sampling_rejections += 1
                        if attempt < getattr(self, "max_resamples", 128):
                            self.pending[i] = self.es.ask(1)[0]
        return self.pending.copy()

    def tell(self, rewards, incumbent_reward, *, distinct_schedules=None):
        if self.pending is None:
            raise RuntimeError("ask must precede tell")
        rewards = np.asarray(rewards, float)
        if rewards.shape != (self.population,) or not np.isfinite(incumbent_reward):
            raise ValueError("Need one reward per candidate and a finite incumbent reward")
        fitness = np.empty(self.population)
        valid = np.zeros(self.population, bool)
        for i, theta in enumerate(self.pending):
            try:
                self.candidate_chart(theta)
                valid[i] = np.isfinite(rewards[i])
            except ValueError:
                pass
            fitness[i] = (-rewards[i]+self.penalty(theta) if valid[i]
                          else 1e6+float(np.mean(theta**2)))
        old_fitness = -float(incumbent_reward)+self.penalty(self.incumbent)
        feasible = np.flatnonzero(valid)
        improved = False
        if len(feasible):
            best = int(feasible[np.argmin(fitness[feasible])])
            if fitness[best] < old_fitness:
                self.incumbent = self.pending[best].copy()
                improved = True
        # Never recombine invalid parents. If fewer than mu are feasible, keep
        # the existing mean/covariance and retry with a narrower distribution.
        updated = int(valid.sum()) >= max(1, self.population//2)
        if updated:
            self.es.tell(list(self.pending), fitness.tolist())
        # Discrete argmin plateaus can be hidden by the residual penalty.
        if not updated:
            self.es.sigma = max(1e-8, self.es.sigma*.5)
        elif distinct_schedules is not None and distinct_schedules <= 1:
            self.es.sigma = min(self.sigma_max, self.es.sigma*self.plateau_growth)
        self.es.sigma = min(self.sigma_max, self.es.sigma)
        self.generation += 1
        self.pending = None
        return dict(generation=self.generation, sigma=float(self.es.sigma),
                    sampling_rejections=getattr(self, "sampling_rejections", 0),
                    optimizer_updated=updated,
                    valid_candidates=int(valid.sum()), rejected_candidates=int((~valid).sum()),
                    incumbent_changed=improved, best_candidate_fitness=float(fitness.min()))

    def tell_independent(self, rewards, *, strata=None):
        """Rank one independently sampled episode per offspring; export the mean.

        Individual winners on different rhythms cannot be compared to an old
        incumbent score. Only the population updates CMA; the resulting mean
        is the current chart, without a validation/promotion gate.
        """
        if self.pending is None:
            raise RuntimeError('ask must precede tell')
        rewards = np.asarray(rewards, float)
        if rewards.shape != (self.population,):
            raise ValueError('Need exactly one independent reward per offspring')
        valid = np.zeros(self.population, bool)
        fitness = np.full(self.population, np.inf)
        for i, theta in enumerate(self.pending):
            try:
                self.candidate_chart(theta)
                valid[i] = np.isfinite(rewards[i])
            except ValueError:
                pass
            if valid[i]:
                fitness[i] = -rewards[i]+self.penalty(theta)
        updated = int(valid.sum()) >= max(1, self.population//2)
        # Invalid rows must rank after every finite, feasible parent.
        worst = float(fitness[valid].max()) if valid.any() else 0.
        raw_best = float(fitness[valid].min()) if valid.any() else worst+1e6
        if strata is not None:
            fitness = stratified_fitness_ranks(fitness, strata, valid)
            worst = float(fitness[valid].max()) if valid.any() else 0.
        fitness[~valid] = worst+1e6
        previous = self.incumbent.copy()
        if updated:
            self.es.tell(list(self.pending), fitness.tolist())
            # Positive recombination of feasible curves is feasible: the
            # time-floor constraints are linear for each fixed proxy value.
            self.candidate_chart(self.es.mean)
            self.incumbent = self.es.mean.copy()
        else:
            self.es.sigma = max(1e-8, self.es.sigma*.5)
        self.es.sigma = min(self.sigma_max, self.es.sigma)
        self.generation += 1
        self.pending = None
        return dict(generation=self.generation, sigma=float(self.es.sigma),
            sampling_rejections=getattr(self, 'sampling_rejections', 0),
            optimizer_updated=updated, valid_candidates=int(valid.sum()),
            rejected_candidates=int((~valid).sum()),
            incumbent_changed=not np.array_equal(self.incumbent, previous),
            best_candidate_fitness=raw_best,
            fitness_ranking='within_speed_tier' if strata is not None else 'global',
            training_chart_selection='cma_population_mean',
            mean_offspring_reward=float(rewards[valid].mean()) if valid.any() else None)

    def save(self, path, **extra):
        payload = dict(format=self.format, fingerprint=self.fingerprint,
            optimizer=self, extra=extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+".tmp")
        temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        temporary.replace(path)

    @classmethod
    def load(cls, path, chart):
        # Only load locally produced/trusted checkpoints, as with model .pt files.
        payload = pickle.loads(Path(path).read_bytes())
        if payload.get("format") != cls.format or payload["fingerprint"] != overlay_fingerprint(chart):
            raise ValueError("CMA checkpoint does not match the initial conditional overlay")
        optimizer = payload["optimizer"]
        if not isinstance(optimizer, cls) or optimizer.fingerprint != payload["fingerprint"]:
            raise ValueError("Invalid CMA checkpoint")
        try:
            optimizer.candidate_chart(optimizer.incumbent)
            optimizer.candidate_chart(optimizer.es.mean)
        except ValueError as exc:
            raise ValueError("CMA checkpoint contains an invalid incumbent or search center") from exc
        return optimizer, payload["extra"]


def assigned_clip(clip, actions):
    from dataclasses import replace
    from isaac_rl.assignment_search import hit_groups
    groups = hit_groups(clip)
    if len(actions) != len(groups):
        raise ValueError("One hand choice is required for every hit group")
    hits = [replace(h, hand=("left", "right")[(int(a)+j) % 2])
            for rows, a in zip(groups, actions) for j, h in enumerate(rows)]
    return replace(clip, hits=hits)


def schedule_key(clip, scenario):
    """Exact static assignments may share a rollout on a matched scenario."""
    return (scenario, tuple((h.time, h.drum, h.hand, h.velocity) for h in clip.hits), clip.duration)


def plan_population(optimizer, candidates, panel, initial_minimum_s, horizon=4):
    """Return unique physical episodes and candidate/scenario gather indexes.

    The final two rows are the current chart incumbent and original graph.
    Missing/rejected candidates are -1 and receive no physical rollout.
    """
    from isaac_rl.assignment_search import hit_groups
    from isaac_rl.min_time_assignment import receding_assignments, graph_actions
    K, P = len(candidates), len(panel)
    if P < 1:
        raise ValueError("Need a nonempty matched rhythm panel")
    gather = np.full((K+2, P), -1, int)
    episodes, keys = [], {}
    signatures = []
    for k in range(K+2):
        chart = None
        if k != K+1:
            try:
                chart = optimizer.candidate_chart(candidates[k] if k < K else optimizer.incumbent)
            except ValueError:
                signatures.append(None)
                continue
        signature = []
        for scenario, clip in enumerate(panel):
            actions = (graph_actions(hit_groups(clip))[0] if chart is None else
                       receding_assignments(clip, chart, horizon, initial_minimum_s)[0])
            physical = assigned_clip(clip, actions)
            key = schedule_key(physical, scenario)
            if key not in keys:
                keys[key] = len(episodes)
                episodes.append(physical)
            gather[k, scenario] = keys[key]
            signature.append(tuple(actions.tolist()))
        signatures.append(tuple(signature))
    distinct = len({s for s in signatures[:K+1] if s is not None})
    return dict(episodes=episodes, gather=gather, distinct_schedules=distinct,
                unique_rollouts=len(episodes), logical_rollouts=(K+2)*P)


def aggregate_population(plan, episode_rewards):
    rewards = np.asarray(episode_rewards, float)
    if rewards.shape != (plan["unique_rollouts"],) or not np.isfinite(rewards).all():
        raise ValueError("Every physical episode needs a finite reward")
    values = np.full(len(plan["gather"]), np.nan)
    for k, ids in enumerate(plan["gather"]):
        if (ids >= 0).all():
            values[k] = float(rewards[ids].mean())
    return values


def plan_independent_population(optimizer, candidates, panel, initial_minimum_s,
                                horizon=4, progress=None, executor=None):
    """Assign exactly one random episode to each offspring, without references.

    Different offspring never share a rollout even if their sampled rhythms
    happen to coincide. Gather has one column and no incumbent/graph rows.
    """
    from isaac_rl.min_time_assignment import receding_assignments
    if len(panel) != len(candidates) or not len(panel):
        raise ValueError('Independent mode needs one rhythm window per offspring')
    gather = np.full((len(candidates), 1), -1, dtype=int)
    episodes = []
    if executor is not None:
        jobs = [(optimizer.chart, optimizer.initial, optimizer.scale,
            candidates[start:start+128], panel[start:start+128], initial_minimum_s, horizon, start)
            for start in range(0, len(candidates), 128)]
        done = 0
        for results in executor.map(_independent_plan_job, jobs):
            for index, clip in results:
                if clip is not None:
                    gather[index, 0] = len(episodes)
                    episodes.append(clip)
            done += len(results)
            if progress:
                progress(done, len(candidates))
        return dict(episodes=episodes, gather=gather, unique_rollouts=len(episodes),
            logical_rollouts=len(candidates), sampling='independent', rollouts_per_offspring=1)
    for i, (theta, clip) in enumerate(zip(candidates, panel)):
        try:
            chart = optimizer.candidate_chart(theta)
        except ValueError:
            continue
        actions = receding_assignments(clip, chart, horizon, initial_minimum_s)[0]
        gather[i, 0] = len(episodes)
        episodes.append(assigned_clip(clip, actions))
        if progress and (i+1) % 128 == 0:
            progress(i+1, len(candidates))
    return dict(episodes=episodes, gather=gather, unique_rollouts=len(episodes),
        logical_rollouts=len(candidates), sampling='independent',
        rollouts_per_offspring=1)


def planning_worker_ready():
    return True


def _independent_plan_job(job):
    """CPU-only spawned worker; preserve case order and all hand decisions."""
    from isaac_rl.min_time_assignment import receding_assignments
    base, initial, scale, candidates, clips, ready, horizon, start = job
    results = []
    for i, (theta, clip) in enumerate(zip(candidates, clips)):
        chart = copy.deepcopy(base)
        try:
            chart.set_parameters(initial+scale*theta)
        except ValueError:
            results.append((start+i, None))
            continue
        actions = receding_assignments(clip, chart, horizon, ready)[0]
        results.append((start+i, assigned_clip(clip, actions)))
    return results


def stratified_fitness_ranks(fitness, strata, valid):
    """Comparable [0,1] average-tie ranks within each timing difficulty level."""
    fitness, strata, valid = np.asarray(fitness), np.asarray(strata), np.asarray(valid, bool)
    if fitness.shape != strata.shape or fitness.shape != valid.shape:
        raise ValueError('One fitness/stratum/valid flag required per offspring')
    result = np.full(fitness.shape, np.inf)
    for level in np.unique(strata):
        ids = np.flatnonzero(valid & (strata == level))
        if len(ids) == 1:
            result[ids] = .5
        elif len(ids):
            _, inverse, counts = np.unique(fitness[ids], return_inverse=True, return_counts=True)
            high = counts.cumsum()
            average_rank = (high-counts+high-1)/2.
            result[ids] = average_rank[inverse]/(len(ids)-1)
    return result
