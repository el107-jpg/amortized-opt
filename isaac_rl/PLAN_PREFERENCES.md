# Matched hand-plan preferences

`scripts/train_plan_preferences.py` learns a plan-cost correction from matched
Isaac trials. The latest completed CMA mean initializes the fixed timing chart;
the correction begins at zero. Its255 coefficients use the same85 eligible
hand/source/destination cells and three other-hand-proxy anchors. Left-hand
floor tom remains excluded. The frozen12000-update decoder, original timing
and ready matrices, nominal robot/drums,250 Hz physics and50 Hz control are
retained. This is assignment learning, with no neural phase-2 update.

## Current selection: deterministic top two

The active method ranks every eligible plan with the current MAP-corrected
search score and evaluates **ranks1 and2 only**. No graph candidate is forced
into the trials and no covariance/Thompson draw selects an alternative. Graph
proximity remains the existing deterministic tie-break rule for exactly tied
search scores. Fresh runs default to `--selection top_two --contexts 2048
--candidates 2`:4096 physical trials across four GPUs, with two worker shards
per candidate slot. Corresponding context indices have matching nominal origins.

Covariance/replay remain part of fitting and checkpoint diagnostics, but
uncertainty is not used for selection. Near ties retain weak indifference
learning. The procedural sampler redraws only an independent random group if
a window otherwise has one eligible plan, preserving forced49 request-pair and
21 simultaneous-pair coverage at all three timing levels. Every training
window therefore has a genuine runner-up; no duplicate plan substitutes for it.

Explicitly switch an existing four-plan checkpoint while retaining its model,
replay, covariance and RNG:

```
... scripts/train_plan_preferences.py --out runs/plan_preferences_top2_trial \
  --resume runs/plan_preferences_trial/last.pkl --selection top_two \
  --contexts 2048 --candidates 2 --iterations 20
```

The active continuation is `runs/plan_preferences_top2_4096_20261006/`, starting
from completed update10 and running through20. The previous four-plan method
below is retained as explicit `--selection thompson` for archived experiments;
its completed artifacts are preserved.

## Original four-plan experiment

The first experiment uses1024 fresh random four-hit-group contexts from the
nominal ready state, with four candidate plans each (4096 actual environments).
All plans are enumerated (at most16), using the established conditional-proxy
rules. Trials include the current greedy plan, the graph baseline (or its
nearest eligible sequence), and distinct alternatives selected with Thompson
sampling from the full correction covariance. Rare windows with fewer than
four eligible plans use explicitly repeated physical trials; identical-feature
comparisons supply no learning information.

Every candidate's entire window is committed and physically executed. Candidate
slots occupy separate GPUs, with matching context indices/world origins across
workers. This removes much of the random-rhythm difficulty and simulator-origin
variation from comparisons. Randomization changes rhythms only, at the same
three timing levels and all eligible drums. Mid-rhythm state branching is not
implemented: matching a scheduled prefix alone would not match physical states,
because the frozen decoder also sees future hits. Initial trials genuinely
share the nominal ready start.

The original plan score is normalized monotonically within each eligible bank
and gets an additive correction `features @ coefficients`. Zero coefficients
preserve the original ranking and graph tie rules. Transition features sum
the quadratic weights `[z(z-1)/2, 1-z², z(z+1)/2]` in their cell's three slots,
normalized by the window's requested hit count. Ready arrivals retain their
separate calibration and never become fictitious drum-source features.
Base minimum-time anchors/support data remain unchanged; corrections are in
normalized **plan-cost units**, not seconds or measured minimum times.

## Preference fitting and covariance

For plans i/j in the same context, form `x=(features_j-features_i)/temperature`
and the matching normalized base-score offset. The model predicts probability
i is preferable as `sigmoid(offset + x @ coefficients)`. A soft target derives
from the measured reward difference (temperature0.05), with capped reliability
weight based on its magnitude. Near ties receive weak indifference weight0.05,
so repeated neutral evidence can reduce uncertainty without an arbitrary winner.
All correlated comparisons in a context have total information weight at most1.
Actual contact, proximity and physical-penalty rewards are the existing ones.

MAP fitting uses a zero-mean Gaussian prior and a convex weighted logistic loss.
Newton updates with backtracking solve the255-parameter correction. Its full
Laplace covariance is

```
Sigma = inverse(prior_precision + sum(weight * p * (1-p) * x x.T))
```

This is an approximate posterior, not CMA's evolution covariance. It retains
parameter correlations and reduces uncertainty along observed comparison
directions. Unobserved directions retain prior uncertainty. Directions that
shift all plan scores equally cancel from feature differences and have no
effect on decision exploration. Predictive variance for competing plans is
`difference.T @ Sigma @ difference`; exploration draws a correction per
decision from `N(MAP, exploration² * Sigma)`.

The covariance is rebuilt from retained replay once per fit. Repeated Newton
steps do not count the same observations as fresh evidence. Default replay
capacity is16384 comparisons. Eviction/new data can widen some directions;
there is no forced global noise-decay schedule. Context omissions and noisy
physical outcomes still limit the posterior's calibration.

## Run, resume and evaluate

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 AMORTIZED_RNEA_CACHE_ONLY=1 \
/data2/eric/conda_envs/g1drum/bin/python scripts/train_plan_preferences.py \
  --out runs/plan_preferences_trial --selection top_two --contexts 2048 --candidates 2 --iterations 10
```

TensorBoard is registered under `runs/tensorboard_active`, retaining old runs.
The four simulator workers persist across updates. Checkpoints preserve MAP,
full covariance, replay, RNG, and the exact unfinished context/candidate batch.
Actor and base-chart checksums are validated on resume. Budgets are absolute:

```
... scripts/train_plan_preferences.py --out runs/plan_preferences_trial \
  --resume runs/plan_preferences_trial/last.pkl --iterations 20
```

`last_correction.npz` is the actual current MAP correction; no validation gate
selects it. `base_chart.npz` is the immutable starting chart. `iteration_*.npz`
stores selected features/actions, scores/rewards and before/after covariance.
`metrics.jsonl`, `progress.json` and `rollout_workers.json` expose live status.
Current training metrics compare first/second/oracle on each matched context; the best
observed candidate is an exploration oracle, not the learned policy's accuracy.

Explicitly load corrections for real recorded-rhythm inference:

```
... scripts/evaluate_chart_cma.py --run runs/plan_preferences_trial \
  --learned-overlay runs/plan_preferences_trial/base_chart.npz \
  --initial-overlay runs/plan_preferences_trial/base_chart.npz \
  --plan-correction runs/plan_preferences_trial/last_correction.npz \
  --panel-manifest runs/chart_balanced_recorded_panel_20261006/manifest.json \
  --out runs/plan_preferences_test --envs 64 --device cuda:2
```

The reserved test preserves its complete eight-group clips and uses receding
four-group inference, comparing correction/zero-correction/graph with the same
frozen actor. Test feedback does not affect fitting or training budget.

## Verification

Unit checks cover warm-start parity, analytical correction features, eligibility,
pair-weight bounds, correlated covariance contraction/null directions, neutral
labels, replay recount prevention and pending/RNG checkpoint restoration.
Two-update32-environment/four-GPU physics smoke completed successfully in
`runs/plan_preferences_smoke_20261006/`:94 informative comparisons,48+46 across
updates, preference objectives decreased and covariance's minimum eigenvalue
fell1→0.498→0.420. This smoke verifies execution and adaptation, not general
hit-rate improvement. The recorded evaluation loader also supports corrections.
