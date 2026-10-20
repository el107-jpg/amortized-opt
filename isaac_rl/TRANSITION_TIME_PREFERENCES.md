# Direct transition-time preference learning

This is the corrected implementation of paired plan learning. It updates the
actual255 timing anchors in `ConditionalMinimumTimeChart`, in **seconds**.
The previous separate plan-cost layer was an implementation mistake and its
run was stopped. Its coefficients/covariance/replay cannot be treated as timing
parameters. The corrected experiment starts from the last legitimate timing
chart: the final CMA generation53 chart, retaining the frozen12000-update
decoder, original ready calibration and underlying measurement provenance.

## Historical penalties removed

Assignment scoring no longer reads transition support/trial success fractions.
There is no historical reliability surcharge and no missing-calibration constant.
Unknown timing estimates remain marked unknown and contribute no fabricated
minimum time or arbitrary score penalty. Finite ready-start durations remain
actual timing estimates. Explicit mechanical hand eligibility, including the
left-hand floor-tom exclusion, is unchanged.

Cached older replay is migrated on resume: known-edge support surcharges and
missing-ready1000 constants are stripped while preserving finite ready timing
costs, time parameters and physical reward targets. The v1 ready-start data
always started at1 s; its finite ready durations were100–260 ms, so the1000
constant cannot be confused with a genuine ready timing deficit. Pending
candidates selected by the old score are replaced with a fresh top-two ranking.
Legacy counts remain in chart provenance but are not used by active scoring.

The completed old10-update trial retains its original reports. Current
continuation `runs/transition_time_no_history_4096_20261007/` resumes its real
time parameters, with penalties removed in selection and replay, and uses the
corrected shared-cost-invariant preference likelihood.2048 rhythms ×2 plans,
4096 physical environments,250 Hz physics,50 ms maximum timing step.76 tests
and the16-env physical smoke passed. This scoring change has not yet received
its final recorded-rhythm comparison.

## Current top-eight trials (2026-10-07)

The user replaced first/second-only comparisons with256 fresh random rhythms
per update and up to eight legal candidates per rhythm. Resume the update20
timing checkpoint, retaining parameters, prior, covariance, replay and RNG.
Execute ranks1–8 under the current timing-only chart; if fewer than8 legal
plans exist, execute each legal plan once. No duplicate padding is simulated.
The environment capacity is2048 across four GPUs (512 each); actual rollouts
are fewer when mechanical eligibility reduces the legal bank.

All observed candidate pairs supply the same reward-difference preference
likelihood as before. Eight plans give28 comparisons. Average pairwise loss
within each rhythm so shared rollouts do not give rhythms with more legal
plans28 times the weight of a two-plan rhythm. Padded slots are masked out of
replay and metrics. The reward, fixed decoder,250 Hz physics,50 ms timing cap,
prior and original score are unchanged. This remains actual timing learning.

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 AMORTIZED_RNEA_CACHE_ONLY=1 \
/data2/eric/conda_envs/g1drum/bin/python scripts/train_transition_time_preferences.py \
  --resume runs/transition_time_no_history_4096_20261007/last.pkl \
  --out runs/transition_time_top8_256_20261007 \
  --contexts 256 --candidates 8 --selection top_k --iterations 30
```

The iteration budget is absolute:30 means ten more updates after20. The
starting actual chart is also saved as `before_continuation_chart.npz`;
`initial_chart.npz` retains the original prior center/provenance.

## Previous paired trials

Generate2048 fresh four-hit-group rhythms and rank each eligible plan using the
chart's temporal slack score. Physically execute only ranks1
and2:4096 environments on four GPUs. No learned cost correction, graph candidate
insertion, posterior sampling or changed physics. The actual chart is used at
inference through the ordinary `minimum()` lookup. The original graph remains
only an exact-tie ordering convention and a separate recorded-test reference.
Each candidate window is committed from the nominal ready pose; matching
context slots have the same nominal simulator origin on paired GPUs. Dynamics
and drum-position randomization remain disabled.250 Hz physics /50 Hz control.

## Timing gradients and updates

For each fixed candidate, cache its actual transition indices, three quadratic
proxy basis weights, time available per edge, and fixed ready/support costs.
The cached score and analytical timing Jacobian reproduce the active timing-only
chart scorer, including the other-hand interval proxy. Unknown ready arrivals
stay in their separate calibration; they never become fake drum departures.
Every transition's estimated time is evaluated directly from learned anchors:

```
T(z) = z(z-1)/2 * L + (1-z²) * M + z(z+1)/2 * R
```

Matched physical reward differences supply confidence-weighted soft pairwise
preferences. The preference logit uses the raw pairwise cost difference divided by a fixed
scale computed from the initial chart's cost difference (with a configurable
minimum scale). Shared costs and timing gradients therefore cancel exactly.
The earlier separate `log1p(score/10)` transforms created spurious gradients
for shared costs; that likelihood was corrected after the first completed run.
This adds no trainable utility/cost parameters. Gradients propagate through
the original slack penalties into timing anchors. The inference scorer is
unchanged; previously recorded physical test results remain valid.

Optimization uses standardized timing residuals relative to the initial chart,
with a50 ms Gaussian prior, damped Gauss-Newton steps and backtracking against
the actual preference objective. The internal coordinate scale is explicitly
converted to seconds before chart scoring and export. Each training update's
net change is bounded to **50 ms per anchor**, across all fitting substeps.
Explicit `--max-update-ms` overrides the saved cap on resume without resetting
parameters, covariance, replay or RNG. The20 ms positive floor is checked over the entire parabola, including its
interior vertex. Infeasible trial curves get a common three-anchor floor shift
and must still pass the net-step bound and actual objective-decrease check.

The full covariance describes time uncertainty in **seconds²**; diagnostic
standard deviations are logged in milliseconds. It shapes fitting but does not
select alternatives. Retained replay is rebuilt at each fit without counting
optimizer substeps as extra observations. The rank-preference objective is
nonlinear in timing anchors; the Gauss-Newton covariance is an approximation.

## Artifacts and usage

`scripts/train_transition_time_preferences.py` is the explicit trainer.
The regular `scripts/train_plan_preferences.py` entrypoint also delegates to
this timing trainer by default. Reproducing the old wrong cost-layer experiment
requires the explicit archival flag `--legacy-plan-costs`.

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 AMORTIZED_RNEA_CACHE_ONLY=1 \
/data2/eric/conda_envs/g1drum/bin/python scripts/train_transition_time_preferences.py \
  --out runs/direct_time_trial --contexts 2048 --iterations 10 --max-update-ms 50
```

`initial_chart.npz` and `last_chart.npz` contain actual timing anchors in seconds,
compatible with existing chart inference/plotting. `last.pkl` preserves timing
parameters, prior, covariance, full transition-design replay, RNG and exact
pending matched episodes. Resume requires this direct timing checkpoint format;
unitless cost-layer checkpoints are rejected. `iteration_*.npz` records before/
after timing arrays, covariance in seconds², full transition designs and physical
rewards. TensorBoard logs include `time_step_rms_ms`, `time_step_median_abs_ms`
and `time_step_max_abs_ms`; registration preserves old runs.

Evaluate actual updated times on the complete reserved recorded clips:

```
... scripts/evaluate_chart_cma.py --run runs/direct_time_trial \
  --panel-manifest runs/chart_balanced_recorded_panel_20261006/manifest.json \
  --out runs/direct_time_test --envs 64 --device cuda:2
```

No `--plan-correction` is needed. Longer eight-group test clips retain every
request and use receding four-group timing-chart inference. Test results do not
select a chart or change the training budget.

## Verification

70 relevant checks passed, including cached/original score parity, every-anchor
finite-difference gradients, genuine time movement toward the observed winning
plan, net millisecond step bounds, interior-vertex positivity, ready separation
and actual timing-chart export/checkpoint restoration.16-env/four-GPU physics
smoke changed61 real time anchors:1.276 ms overall RMS, max10 ms, zero auxiliary
cost layer. Evidence and before/after millisecond plots:
`runs/transition_time_preferences_smoke_20261007/`.

The initial cap was10 ms. User requested larger updates; a controlled replay of
the same iteration2 preferences gave RMS9.23 /20.30 /34.05 /44.89 ms at caps
10 /25 /50 /100 ms respectively. The50 ms cap reduced fitting loss to2002.78
versus2347.99 at10 ms, with all curve constraints valid. Current default/cap
is50 ms; prior remains50 ms. This is optimizer-fit evidence, not a physical
hit-rate gain. Evidence: `runs/transition_time_step_probe_20261007/summary.json`.

Objective audit: `runs/transition_planner_objective_audit_20261007/`. The prior
separate-log likelihood introduced shared-direction gradients in935/2048
comparisons in update10 (45.65%;11.66% of squared gradient magnitude). Corrected
exact shared gradients are zero.73 tests passed including common-cost invariance
and zero updates for identical timing functions. A CPU-only shadow refit is
stored separately; it was not promoted and has no new physical evaluation.
