"""Balance physical drum requests while preserving complete rhythm windows."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def capped_softmax(logits, cap):
    """Maximum-entropy softmax with an upper bound on each probability."""
    if cap*len(logits) < 1 or cap <= 0:
        raise ValueError("window probability cap cannot cover the candidate pool")
    probabilities = np.zeros(len(logits))
    active = np.ones(len(logits), dtype=bool)
    remaining = 1.
    while active.any():
        # Recompute in log space after saturation to avoid underflow when
        # strongly weighted rare patterns have all reached their cap.
        values = logits[active]
        probabilities[active] = remaining*np.exp(values-logsumexp(values))
        saturated = active & (probabilities >= cap)
        if not saturated.any():
            break
        probabilities[saturated] = cap
        active[saturated] = False
        remaining = max(0., 1-probabilities[~active].sum())
    return probabilities


class BalancedWindowSampler:
    """Entropy-regularized weighting of natural, unmodified training windows.

    Balance expected *hit counts*, including both members of simultaneous
    pairs. Uniformly choosing a target drum would not balance the accompanying
    hits in its window. A fixed candidate seed makes resume reproducible without
    consuming the rollout RNG. Each clip contributes up to 64 candidate starts.
    """

    def __init__(self, clips, num_drums, *, span_s=5., seed=7201,
                 candidates_per_clip=64, regularization=1e-4, deduplicate=False,
                 max_probability=None, spans_s=None, allow_missing_drums=False):
        if not clips or span_s <= 0 or candidates_per_clip < 1 or regularization <= 0:
            raise ValueError("nonempty clips and positive sampler settings required")
        self.clips = clips
        self.span_s = span_s
        self.spans = np.full(len(clips), span_s) if spans_s is None else np.asarray(spans_s)
        if self.spans.shape != (len(clips),) or np.any(self.spans <= 0):
            raise ValueError("one positive window span required per clip")
        candidates, counts, seen = [], [], set()
        rng = np.random.default_rng(seed)
        for index, clip in enumerate(clips):
            clip_span = float(self.spans[index])
            times = np.asarray([h.time for h in clip.hits])
            drums = np.asarray([h.drum for h in clip.hits], dtype=np.int64)
            if not len(times):
                continue
            upper = max(0., times.max()-clip_span)
            starts = rng.uniform(0, upper, candidates_per_clip) if upper else [0.]
            for start in starts:
                row = np.bincount(drums[(times >= start) & (times < start+clip_span)],
                                  minlength=num_drums)
                if row.sum():
                    if deduplicate:
                        selected = [h for h in clip.hits if start <= h.time < start+clip_span]
                        key = tuple((round(h.time-selected[0].time, 8), h.drum,
                                     round(h.velocity, 6)) for h in selected)
                        if key in seen:
                            continue
                        seen.add(key)
                    candidates.append((index, float(start)))
                    counts.append(row)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.candidates = candidates
        if not len(counts) or (not allow_missing_drums and np.any(self.counts.sum(0) == 0)):
            raise ValueError("every drum must occur in the training window pool")
        if max_probability is not None and max_probability*len(counts) < 1:
            raise ValueError("window probability cap cannot cover the candidate pool")
        present = self.counts.sum(0) > 0
        balance_counts = self.counts[:, present]
        features = (balance_counts-balance_counts.sum(1, keepdims=True)/present.sum())
        features /= self.counts.sum(1).mean()

        def objective(theta):
            logits = features @ theta
            if max_probability is None:
                norm = logsumexp(logits)
                probs = np.exp(logits-norm)
            else:
                probs = capped_softmax(logits, max_probability)
                norm = probs @ logits-np.sum(probs*np.log(np.maximum(probs, 1e-300)))
            return (norm-np.log(len(logits))+regularization*(theta @ theta)/2,
                    features.T @ probs+regularization*theta)

        result = minimize(objective, np.zeros(int(present.sum())), jac=True, method="BFGS",
                          options={"gtol": 1e-8})
        if not np.isfinite(result.fun) or np.linalg.norm(result.jac) > 1e-6:
            raise RuntimeError(f"drum balancing did not converge: {result.message}")
        logits = features @ result.x
        self.probabilities = (np.exp(logits-logsumexp(logits)) if max_probability is None else
                              capped_softmax(logits, max_probability))
        expected = self.probabilities @ self.counts
        self.expected_shares = expected/expected.sum()
        self.summary = dict(candidate_windows=len(counts), seed=seed,
                            present_drums=np.flatnonzero(present).tolist(),
                            candidates_per_clip=candidates_per_clip,
                            deduplicated=deduplicate,
                            max_window_probability=max_probability,
                            regularization=regularization,
                            expected_hit_shares=self.expected_shares.tolist(),
                            uniform_hit_shares=(self.counts.sum(0)/self.counts.sum()).tolist(),
                            effective_windows=float(1/(self.probabilities @ self.probabilities)))

    def sample(self, rng, size, *, tail_s=.6, first_hit_s=1.):
        from isaac_rl.assignment_train import window_clip
        choices = rng.choice(len(self.candidates), size=size, p=self.probabilities)
        indexes, windows = [], []
        for choice in choices:
            index, start = self.candidates[int(choice)]
            indexes.append(index)
            windows.append(window_clip(self.clips[index], rng, span_s=float(self.spans[index]),
                                       first_hit_s=first_hit_s, tail_s=tail_s, start_s=start))
        return indexes, windows
