# Continual-learning forgetting stress test

## Setup

The trained critic began with an 18,432-row replay at full capacity. It then
consumed 12,288 new exact-labeled rows in this order:

1. 4,096 same-generator control rows;
2. 4,096 readiness/cross-hand shifted rows;
3. 4,096 legacy-regime rows.

Each 256-row arrival received the production-like 50 optimizer updates, for
2,400 additional updates. Adding the new stream overwrote the oldest 12,288
original replay rows. Retention was measured on two fixed 1,024-row anchors:
the oldest original rows, which were evicted, and the newest original rows,
which remained in replay.

Two matched arms started from the identical trained checkpoint and random
generator state. One retained the configured function-space inertia weight of
0.02; the other set inertia to zero.

## Retention with configured inertia

| Anchor | Point | Normalized MAE | Sign accuracy | Mean row confidence | Prediction drift |
|---|---|---:|---:|---:|---:|
| Oldest, later evicted | Before new stream | 0.4706 | 88.82% | 0.708 | 0.000 |
| Oldest, evicted | After 12,288 rows | **0.4672** | **88.89%** | 0.719 | 0.197 |
| Newest, retained | Before new stream | 0.6296 | 83.52% | 0.643 | 0.000 |
| Newest, retained | After 12,288 rows | **0.5768** | **84.59%** | 0.645 | 0.218 |

The critic's outputs moved substantially, but accuracy on both old anchors
held steady or improved. The evicted oldest anchor therefore shows no
catastrophic forgetting over this stream.

The fixed calibrated admission rule on the evicted anchor changed from 10.94%
coverage, 91.96% precision, and 0.88% population false-feasible rate to 17.09%
coverage, 92.00% precision, and 1.37% false-feasible rate. Prediction quality
was retained, but the operating point drifted above the original 1% calibration
budget as coverage increased. Continual deployment should periodically update
the calibration threshold even when the regression model is retaining old
knowledge.

## Inertia ablation

| Final metric on evicted anchor | Inertia 0.02 | No inertia |
|---|---:|---:|
| Normalized MAE | 0.4672 | 0.4672 |
| Sign accuracy | 88.89% | 88.92% |
| Prediction drift MAE | 0.1971 | 0.1977 |
| Calibrated precision | 92.00% | 91.48% |
| False-feasible rate | 1.37% | 1.46% |

The current inertia penalty has only a small effect. It slightly reduced
function drift and false admissions but did not explain the retained
regression accuracy. Replay of the remaining original distribution and
transfer between the related task regimes did most of the work.

## Conclusion

The critic is not forgetting too quickly under one substantial replay-turnover
test. This does not yet establish indefinite retention: one-third of the
original replay remained, the new regimes share physical structure with the
old one, and the stream covered only 2,400 additional optimizer updates. The
next stronger test is complete replay replacement followed by multiple regime
cycles, with a small protected anchor memory if long-horizon retention begins
to degrade.

Machine-readable stage metrics are in
`analysis/monotonic_confidence_forgetting_20260924.json`. The reusable stress
test is `scripts/evaluate_monotonic_critic_forgetting.py`.
