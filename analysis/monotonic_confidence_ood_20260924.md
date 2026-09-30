# Exact-labeled OOD evaluation of the monotonic confidence critic

## Setup

The 30,720-example, 3,600-update critic was frozen and evaluated without any
additional learning. The same frozen actor generated 4,096 exact constraint
labels for each evaluation pool. All 131 actor tensors remained bit-for-bit
unchanged in every collection.

The decision thresholds were retained from the original in-distribution
calibration set:

- conservative residual threshold: `-0.37243`
- minimum support-tempered row confidence: `0.81092`
- calibration false-feasible budget: 1% of all evaluated rows

The datasets were:

- **future ID**: the chronological future holdout from critic training;
- **same-generator control**: unseen tasks from an independently generated
  pool with the same eval-box task regime;
- **readiness/cross-hand shift**: a pool with altered ready-state and
  cross-hand task generation;
- **legacy task regime**: an older pool without the eval-box generation
  regime, producing a larger state/task shift and much harder actor outputs.

## Calibrated admissions

| Dataset | Rows | True feasible | Mean replay support | Coverage | Precision | Recall | False-feasible rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future ID | 6,144 | 43.23% | 0.909 | 6.72% | 92.49% | 14.38% | 0.50% |
| Same-generator control | 4,096 | 60.33% | 0.935 | 9.30% | 92.39% | 14.25% | 0.71% |
| Readiness/cross-hand shift | 4,096 | 54.83% | 0.933 | 8.54% | 90.57% | 14.11% | 0.81% |
| Legacy task regime | 4,096 | 37.08% | 0.881 | 0.00% | n/a | 0.00% | 0.00% |

The rule remained within its 1% population-level false-feasible budget on all
three fresh sets. The same-generator control closely reproduced ID precision
and recall. The readiness/cross-hand shift retained useful coverage with 33
false admissions among 4,096 rows. The critic abstained on every legacy-regime
row, including 1,519 truly feasible rows; this is conservative but not useful
generalization.

## Which conservative component mattered

The two actual shifted pools contain 8,192 rows. Applying successive gates at
the joint rule's residual threshold gives:

| Rule on shifted pools | Coverage | Precision | Recall | False-feasible rate |
|---|---:|---:|---:|---:|
| Raw zero residual boundary | 36.01% | 73.63% | 57.69% | 9.50% |
| Calibrated joint margin, no confidence | 12.96% | 84.84% | 23.93% | 1.97% |
| Margin + learned confidence, no support tempering | 7.89% | 86.84% | 14.90% | 1.04% |
| Margin + support-tempered confidence | 4.27% | **90.57%** | 8.42% | **0.40%** |

The confidence gate cut shifted-pool false admissions from 161 to 85. Replay
support tempering cut them again to 33, bringing the aggregate shifted result
well under budget. It also removed 244 true admissions, so the safety gain has
a substantial coverage cost.

The separately calibrated residual-only rule was also conservative on the two
shifted pools: 5.05% coverage, 89.13% precision, 9.80% recall, and a 0.55%
false-feasible rate. Thus learned confidence did not improve OOD coverage in
this test; it supplied a stricter abstention mechanism.

## OOD recognition and confidence quality

Replay distance recognized the legacy regime as shifted with an OOD-detection
AUC of 0.728. It did not recognize the readiness/cross-hand pool as OOD: its
mean support was higher than the future ID holdout and detection AUC was 0.431.
The same-generator control behaved similarly, as expected.

Even when replay distance did not detect the readiness/cross-hand shift,
observable confidence continued to rank sign-correct predictions reasonably:

| Dataset | Correctness AUC | Brier score | ECE |
|---|---:|---:|---:|
| Future ID | 0.755 | 0.151 | 0.079 |
| Same-generator control | 0.819 | 0.105 | 0.068 |
| Readiness/cross-hand shift | 0.792 | 0.115 | 0.044 |
| Legacy task regime | 0.754 | 0.191 | 0.103 |

## Interpretation

The current conservatism works for these labeled pool shifts: no evaluation
set exceeded the calibrated 1% false-feasible budget. It works through two
different paths. On the moderate readiness/cross-hand shift, prediction and
confidence generalize well enough to preserve useful admissions. On the
larger legacy shift, support tempering shuts admission down completely.

This also exposes the limitation of nearest-replay support as a general OOD
detector. A semantically changed task generator can appear closer than the
chronological ID holdout in the 42-dimensional feature metric. More diverse
labeled shifts and an actively searched set of confidently optimistic timing
points are needed before relying on this gate in AL training.

Machine-readable results are in
`analysis/monotonic_confidence_ood_20260924.json`. The reusable evaluator is
`scripts/evaluate_monotonic_critic_ood.py`.
