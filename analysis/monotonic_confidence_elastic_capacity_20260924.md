# Decaying anchor memory and critic-capacity ablation

Date: 2026-09-24

## Question

Can a slowly decaying, locally replaceable functional-anchor bank reduce old
prediction drift in the monotonic confidence critic, and does increasing critic
width reduce continual-learning interference?

## Setup

The experiment reused the frozen-actor cache in
`analysis/monotonic_confidence_actor_labels_20260924.pt`; the actor was not
trained. The 43,008 exact labels were divided into:

- 18,432 original rows for initial critic training;
- 6,144 protected original rows for decision-rule calibration;
- 6,144 original holdout rows;
- three sequential 4,096-row streams: same generator, readiness/cross-hand
  shift, and legacy task-regime shift.

Each base critic received 3,600 updates. Each continuation received another
2,400 updates, for 6,000 total updates per arm. All arms used the same seed,
row order, five ensemble heads, replay settings, and calibration data. The
admission rule was recalibrated after every stream stage with a target maximum
false-feasible rate of 0.5%.

The anchor bank held 2,048 exact labels. Anchor force decayed as
`2^(-age / half_life)`, with a 55,296-row half-life. The two-tier policy kept
20% of anchors on a five-times-longer half-life. New observations could:

- refresh a nearby anchor stochastically, including an outdated slow anchor;
- replace a sampled old elastic anchor when the observation was novel;
- never replace the slow tier merely because a novel point arrived.

Age and boundary weights multiply the exact-label, observable-confidence, and
output-drift losses without renormalizing by their sum. Consequently the
absolute force of a uniformly old bank really decreases. A focused test checks
that one half-life halves the anchor regression force.

Raw corrected results are in
`analysis/monotonic_confidence_elastic_capacity_decayed_20260924.json`. The
earlier `analysis/monotonic_confidence_elastic_capacity_20260924.json` is a
diagnostic non-decaying run and should not be used as the main result.

## Anchor-policy result at width 256

Percentages below are row-level feasibility admission metrics. Drift is the
mean absolute change in normalized constraint predictions on the oldest 1,024
original training rows.

| Policy | Old drift | Drift p95 | Original holdout MAE | Holdout false feasible | Holdout coverage | Readiness false feasible | Readiness coverage | Same-gen false feasible | Same-gen coverage | Continuation time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 0.2010 | 0.6738 | **0.5980** | 0.147% | **5.78%** | 0.439% | **9.06%** | 0.317% | 11.01% | 141.5 s |
| Fixed | **0.1153** | **0.3660** | 0.6054 | 0.179% | 5.22% | **0.366%** | 8.50% | 0.342% | 11.45% | 178.3 s |
| Elastic | 0.1217 | 0.3828 | 0.6045 | **0.163%** | 5.31% | 0.391% | 8.42% | **0.317%** | 11.43% | 180.2 s |
| Two-tier | 0.1216 | 0.3814 | 0.6043 | 0.179% | 5.27% | 0.415% | 8.59% | 0.342% | **11.50%** | 179.2 s |

All three anchor policies reduced mean old-output drift by 39-43% and p95 drift
by 43-46% relative to no anchors. This cost about 27% more continuation time
and a small amount of current-distribution fit and coverage. The difference
among fixed, elastic, and two-tier anchors is too small to distinguish in this
single-seed, 12,288-row continuation.

Replacement was active rather than nominal. The fully elastic bank refreshed
255 nearby anchors and admitted 71 novel anchors (2.65% of offers). The
two-tier bank refreshed 229 nearby anchors and admitted 57 novel anchors
(2.33% of offers), while retaining its 20% slow tier. The fixed bank made no
replacements.

## Model-size result with two-tier anchors

| Width | Trainable / total parameters | Old drift | Original holdout MAE | Holdout false feasible | Holdout coverage | Confidence Brier | Readiness false feasible | Readiness coverage | Same-gen false feasible | Same-gen coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 177,160 / 266,600 | **0.1173** | 0.6412 | **0.130%** | 4.39% | 0.1464 | **0.244%** | 5.96% | **0.220%** | 7.86% |
| 256 | 517,640 / 696,040 | 0.1216 | 0.6043 | 0.179% | 5.27% | 0.1405 | 0.415% | 8.59% | 0.342% | 11.50% |
| 512 | 1,690,120 / 2,046,440 | 0.1223 | **0.5958** | 0.195% | **6.43%** | **0.1366** | 0.659% | **10.79%** | 0.708% | **13.67%** |

More capacity improved normalized error, confidence Brier score, and coverage,
but it did not reduce old-output drift. The 512-wide critic crossed the 0.5%
false-feasible target on readiness/cross-hand (0.659%) and same-generator
(0.708%) data. It is therefore too aggressive for the current safety gate even
though its average predictions are more accurate.

Every arm abstained completely on the legacy-regime set after final
calibration. That is conservative OOD behavior, but it provides no useful
coverage and shows that these results do not establish broad task-regime
generalization.

## Decision

Use width 256 for the next experimental integration. Enable the two-tier bank
if the goal is to preserve a small long-lived core while allowing stale local
labels to refresh; its observed behavior is essentially tied with the simpler
fully elastic bank at this horizon. Keep the feature configurable because this
run does not demonstrate that the 20% slow tier is better than fixed or fully
elastic storage.

Do not move to width 512 for feasibility admission yet. Its extra coverage
came with shifted false-feasible rates above the calibrated target, and extra
capacity did not reduce interference. The anchor mechanism directly addresses
drift more effectively than width in this experiment.

The next discriminating test should continue beyond at least one 55,296-row
elastic half-life with genuinely new actor labels and multiple seeds. The
present continuation covers only 0.22 half-lives, so it validates smooth decay
and replacement mechanics but cannot establish the long-run advantage of the
two-tier schedule.

## Default promotion

Following this result, `MonotonicConfidenceConfig` now defaults to three heads
at width 256, a 32,768-row replay, and the selected 2,048-entry two-tier anchor
policy. The anchor bank waits for 18,432 observed labels, lets the
warmup-crossing batch finish training, and then seeds from replay using the
updated EMA. This avoids anchoring the deliberately pessimistic initialization
and matches the train-then-seed order evaluated above. The capacity experiment
used five heads; reducing the default to three is a subsequent compute-saving
choice and still requires a matched head-count ablation.
