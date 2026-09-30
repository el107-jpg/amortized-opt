# Monotonic confidence critic: 30k-example frozen-actor run

## Setup

- Actor checkpoint: `runs/rhythm_timing_com_linear_ropeplant_dls_critic4_gpu3_20k_20260923/last.pt`
- The frozen actor produced 30,720 fresh labeled rows over 480 chronological
  batches. Its 131 tensors remained bit-for-bit unchanged after collection
  (`max_abs_change = 0`).
- Split by whole actor batch: 18,432 training rows, 6,144 calibration rows, and
  6,144 later holdout rows.
- Both critics saw identical rows in identical order. Calibration targeted a
  1% false-feasible rate.
- The main run used 50 optimizer updates per 256 newly revealed rows, for 3,600
  critic updates. A 20-update arm used 1,440 updates to measure optimization
  sensitivity.

The proposed critic has five independent heads, signed direct residuals,
piecewise-positive TTH derivatives, frozen randomized derivative priors,
cross-fitted observable confidence targets, random-task confidence
suppression, replay-support tempering, EMA predictions, and
confidence-weighted function-space inertia.

## Main result

| Calibrated rule on future holdout | Coverage | Feasible recall | Precision | False-feasible rate |
|---|---:|---:|---:|---:|
| Existing regression critic | 5.55% | 11.45% | 89.15% | 0.60% |
| Monotonic residual margin only | 6.35% | 13.59% | **92.56%** | **0.47%** |
| Monotonic residual + confidence | **6.72%** | **14.38%** | 92.49% | 0.50% |

At 3,600 updates, the confidence-aware critic admitted 21.1% more holdout rows
and recovered 25.7% more truly feasible rows than the existing critic while
slightly reducing false-feasible admissions. The learned confidence gate also
improved coverage and recall over residual-margin calibration alone at nearly
the same error rate.

Confidence predicted whether all four residual signs were correct with:

- Brier score: 0.1513
- expected calibration error: 0.0789
- correct/incorrect ranking AUC: 0.7550
- mean confidence on held-out actor rows: 0.7528
- mean confidence on broad random-domain inputs: 0.0025

The TTH sweep check found no monotonicity violations. The largest observed
residual change after increasing TTH was `-0.000394` in normalized units.

## Optimization sensitivity

| Updates per stream batch | Total updates | Proposed coverage | Proposed recall | Proposed precision | Proposed false-feasible |
|---:|---:|---:|---:|---:|---:|
| 20 | 1,440 | 3.89% | 7.53% | 83.68% | 0.63% |
| 50 | 3,600 | 6.72% | 14.38% | 92.49% | 0.50% |

The existing critic beat the proposed model after 1,440 updates, with 5.00%
coverage and 10.54% recall. The ranking reversed after 3,600 updates. This is
consistent with the proposed model needing more optimization because all
trainable weights start at zero and the hidden biases provide symmetry
breaking.

## Stability issue

Normalized residual MAE improved from 0.7614 at 1,440 updates to 0.6783 at
3,600 updates, but a very small number of extreme normalized predictions made
the inverse `sinh` transform explode in physical units. Across all scalar
holdout residual predictions, the absolute-value quantiles were 0.711 at the
median, 1.740 at p90, 3.888 at p99, and 5.752 at p99.9; 0.228% exceeded 5 and
0.0163% exceeded 10. Accepted rows had a maximum absolute normalized residual
of 2.774, so the confidence/margin rule rejected the extreme cases and the
reported admission metrics were unaffected.

Before using direct physical predictions downstream, add an explicit output
magnitude or derivative penalty and repeat the run. Keep calibration in
normalized space.

## Scope of the conclusion

This run labels feasibility from the frozen actor's realized constraint
residuals. It shows that the critic can identify actor samples whose observed
constraints are feasible under stream drift. It does not yet establish the
quantity needed for AL training: that admitting a sample will avoid unstable
multiplier growth during subsequent optimization. That requires replaying or
simulating the downstream AL update and using multiplier growth or persistent
post-update feasibility as the observable target.

Machine-readable metrics and the trained critic are in
`analysis/monotonic_confidence_frozen_actor_30k_u50_20260924.json` and the
adjacent `.pt` file. The 1,440-update arm is stored under the same base name
without the `_u50` suffix.
