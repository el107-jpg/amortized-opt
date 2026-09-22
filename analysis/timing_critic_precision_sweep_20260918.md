# Timing critic precision sweep — 2026-09-18

## Setup

All arms forked the same step-1500 actor, actor optimizer, SaLSa selector,
distributional critic, critic optimizer, and replay state. Each fork ran to
step 1800 with the normal batch-512 timing curriculum. The only changed
hyperparameter was `timing_risk_min_precision`. The continuing 0.50 run is the
baseline; 0.70, 0.80, and 0.90 were matched online forks.

Metrics below aggregate steps 1550–1800, after each fork had recalibrated at
least once. “Genuine yield” excludes exact passes caused by falling back to
the unchanged one-second anchor. Proposal precision is exact true positives
divided by critic-selected manageable proposals.

| Minimum calibration precision | Genuine shorter rows/step | Boundary rows/step | Interior rows/step | Selected precision | False selections/step | Recent mean active TTH | Active TTH <= 0.8 s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 40.41 | 0.10 | 40.31 | 17.2% | 194.21 | 0.949 s | 10.4% |
| **0.70** | **59.20** | **17.77** | **41.43** | **49.5%** | **60.30** | **0.919 s** | **16.4%** |
| 0.80 | 42.24 | 13.28 | 28.96 | 60.0% | 28.12 | 0.941 s | 13.1% |
| 0.90 | 17.64 | 5.53 | 12.11 | 70.6% | 7.36 | 0.977 s | 3.5% |

## False-positive severity

The original 0.50 rule frequently ran timing rays to their most aggressive
endpoint. At the final snapshots, the median worst normalized violation among
still-predicted-manageable rejected proposals was 2.609, and 61.6% exceeded
2.0. The acceptance boundary is 0.2.

At 0.70, the median fell to 0.838 and only 2.7% exceeded 2.0. Thus 0.70 did
not merely increase the number of exact passes: it removed nearly all
catastrophic extrapolations while preserving enough recall to advance time.

The 0.80 and 0.90 arms improved nominal precision further, but rejected the
one-second anchor often enough to replace useful exploration with anchor
fallback. This is visible in their lower genuine yield and longer actor-batch
TTH distributions.

## Decision

Use `timing_risk_min_precision=0.70` as the production default. It delivered
46% more exact-verified shorter rows than 0.50, reduced false selections by
69%, and produced the shortest observed actor-training TTH distribution.

The training telemetry now separately records advancement, accepted
advancement, selected precision, and selected/accepted TTH summaries so that
future experiments cannot count an unchanged anchor fallback as curriculum
progress.
