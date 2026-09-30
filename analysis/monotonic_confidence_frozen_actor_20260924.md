# Monotonic confidence critic on a frozen actor

## Setup

- Actor: `runs/rhythm_timing_com_linear_ropeplant_dls_critic4_gpu3_20k_20260923/last.pt`
- The actor produced 5,120 fresh exact-label rows across 80 batches with its
  learning rate set to zero.  All 131 actor tensors were bit-for-bit unchanged
  after collection (`max_abs_change = 0`).
- Chronological split by whole actor batch: 3,072 training rows, 1,024
  calibration rows, and 1,024 future holdout rows.
- The critic received 20 optimizer updates per 256 newly revealed training
  rows.  Every incoming batch was scored before its labels entered replay.
- Calibration used a 1% false-feasible budget, leaving room for degradation on
  the future holdout relative to the eventual 2% operating target.

The experimental critic has five independent heads, signed direct residuals,
piecewise-linear positive TTH derivatives, frozen randomized derivative priors,
cross-fitted feasibility-confidence targets, random-task confidence
suppression, nearest-replay support tempering, EMA predictions, and
confidence-weighted function-space inertia.

## Results

| Rule on future holdout | Coverage | Feasible recall | Precision | False-feasible rate |
|---|---:|---:|---:|---:|
| Existing regression critic, calibrated | 2.05% | 1.67% | 47.62% | 1.07% |
| Monotonic residual margin only, calibrated | 3.03% | 3.50% | 67.74% | 0.98% |
| Monotonic residual + learned confidence, calibrated | **7.42%** | **11.33%** | **89.47%** | **0.78%** |

The confidence-aware rule therefore admitted 2.45x as many rows as residual
calibration alone and 3.62x as many as the existing critic while remaining
inside the 2% held-out false-feasible target.

Confidence predicted per-constraint feasibility-sign correctness with:

- Brier score: 0.1186
- expected calibration error: 0.0747
- correct/incorrect ranking AUC: 0.8096
- mean confidence on held-out actor tasks: 0.7635
- mean confidence on broad random physical-domain tasks: 0.0402

The architectural monotonicity check found zero increasing residuals when
either active TTH was increased.  The largest measured change was negative
(`-0.00388` normalized residual), as required.

## Important failure found

The uncalibrated monotonic zero boundary was much too optimistic: it predicted
77.54% of holdout rows feasible and had a 28.03% false-feasible rate.  The
existing critic's normalized residual MAE was 0.6199 versus 0.6058 for the new
critic, so mean regression error alone did not reveal this boundary problem.
The confidence and conservative-margin calibration are essential in this
version.

Zeroing every weight also required substantially more critic optimization.  A
first trial with one update per incoming batch remained close to the initial
pessimistic constant.  Twenty small updates per batch overcame the expected
layer-by-layer gradient startup delay.

## Interpretation and next experiment

The observable confidence mechanism worked well enough to keep: cross-fitting
removed stale correctness labels, confidence generalized to future actor
batches, and random/support suppression kept confidence low away from replay.
The hard monotonic residual surface remains provisional because this dataset
contains randomized TTHs rather than matched-context TTH sweeps.  It may be
forcing collision, joint, and torque residuals into a shape that the frozen
actor does not actually follow.

Before connecting this critic to sampling, collect paired TTH sweeps for fixed
state/goal contexts and measure empirical monotonicity per constraint.  Then
compare the current all-constraint monotonic model against terminal-only
monotonicity and an unconstrained residual model, all using the same confidence
and calibration machinery.

Machine-readable metrics and the trained critic are in
`analysis/monotonic_confidence_frozen_actor_safe_20260924.json` and the adjacent
`.pt` file.
