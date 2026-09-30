# Inertia and continual recalibration ablation

## Setup

Every arm started from the same 30,720-example critic checkpoint and consumed
the same 12,288 cached continuation rows in the same order, with 50 updates per
256-row arrival. The stream overwrote the oldest two-thirds of the original
18,432-row replay. Retention was evaluated on fixed 1,024-row anchors.

The sweep used inertia weights 0.02, 0.1, 0.5, and 2.0. At each regime boundary,
the decision rule was also refit on the protected 6,144-row original
calibration split, which was never added to critic replay.

## Inertia sweep

Final metrics on the evicted oldest anchor were:

| Inertia | Prediction drift MAE | Old-anchor MAE | Sign accuracy | New-anchor MAE | Fixed-rule false-feasible |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.1971 | 0.4672 | 88.89% | 0.5768 | 1.37% |
| 0.10 | 0.1942 | 0.4666 | 88.92% | **0.5766** | 1.37% |
| **0.50** | **0.1845** | **0.4657** | **89.01%** | 0.5768 | 1.37% |
| 2.00 | 0.1548 | 0.4682 | 88.96% | 0.5837 | 1.46% |

Weight 0.5 is the useful knee. Relative to 0.02 it reduced old-function drift
by 6.4%, slightly improved old-anchor error, and left new-anchor error
unchanged. Weight 2.0 reduced drift by 21.5% but began to impede adaptation,
raising new-anchor MAE by 1.2% and worsening the fixed decision rule.

Higher inertia alone did not fix calibration drift. Every weight retained an
old-anchor false-feasible rate above the original 1% budget when the original
fixed thresholds were reused.

## Selected recalibration policy

The selected policy uses inertia 0.5 and periodically refits the residual and
confidence thresholds on protected calibration memory with a 0.5%
false-feasible target. The tighter calibration target leaves transfer margin
when the rule is applied to a different old anchor.

| Stream point | Evicted-anchor coverage | Precision | Recall | False-feasible |
|---|---:|---:|---:|---:|
| Initial | 8.30% | 89.41% | 12.14% | 0.88% |
| After same-generator rows | 9.67% | 91.92% | 14.54% | 0.78% |
| After readiness/cross-hand rows | 8.20% | 95.24% | 12.78% | 0.39% |
| After legacy-regime rows | 8.98% | **95.65%** | 14.06% | **0.39%** |

The retained newest anchor also remained below 1% at every stage, finishing at
0.59%. Recalibration therefore solved the observed safety-budget drift in this
turnover test. It roughly halved final coverage relative to the stale rule
(17.87% to 8.98%), which is the cost of preserving the tighter guarantee.

## Reusable label cache

All actor-generated data used by these experiments now lives in one compact
cache:

- 30,720 original chronological rows;
- 4,096 same-generator control rows;
- 4,096 readiness/cross-hand rows;
- 4,096 legacy-regime rows.

The cache contains 43,008 feature/target pairs and actor-step indices in 4.2
MB. Re-running the OOD evaluator from the cache reproduced every admission
metric exactly. Future critic ablations should refer to datasets as
`analysis/monotonic_confidence_actor_labels_20260924.pt::DATASET`.

The selected run is in
`analysis/monotonic_confidence_inertia_selected_20260924.json`; the complete
sweep is in `analysis/monotonic_confidence_inertia_sweep_20260924.json`.
