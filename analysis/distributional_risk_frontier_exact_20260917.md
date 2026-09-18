# Distributional-risk timing frontier: exact CPU check

## Decision statistic

The risk search replaces the entropy/JS hard gate with a consequence computed
from the complete worst-violation categorical distribution.  For each head it
combines probability of exceeding the requested residual range, linear and
quadratic excess severity, then aggregates expected consequence, upper-tail
CVaR, and between-head variation.  A held-out exact-label split calibrates the
largest admissible risk at a requested proposal precision.

The requested close-feasibility range was worst signed-asinh-normalized
residual `<= 0.2`.  Relative to the existing critic limits and scales, this is
approximately no worse than 24 mm terminal-region distance, -7 mm collision
SDF, 0.024 rad joint excess, or 9.61 Nm torque excess.

## Frozen-actor result

Configuration: three-head Softmax/OE critic, 64 random horizon-50 Phase-2 pool
tasks, calibration precision 0.50, risk threshold 1.3071, and exact frozen actor
plus physical-cost verification on CPU.

| Selection | Proposed | Exact pass | Within 0.2 | Within 0.5 | Mean worst residual |
|---|---:|---:|---:|---:|---:|
| Full-distribution risk frontier | 16 | 0 | 1 | 2 | 1.596 |
| Legacy entropy/disagreement frontier | 8 | 0 | 1 | 1 | 2.107 |

The risk functional improves the average severity and doubles coverage, but
ordinary-row calibration does not transfer to optimized boundary points.  The
optimizer finds critic errors: frontier risk values occupy only 1.292--1.307,
while exact worst residuals vary from 0.148 to 3.050.

Stricter ordinary-row calibration did not solve the distribution shift:

| Requested precision | Proposals | Exact pass | Within 0.2 | Within 0.5 |
|---:|---:|---:|---:|---:|
| 0.50 | 16 | 0 | 1 | 2 |
| 0.60 | 8 | 1 | 1 | 2 |
| 0.70 | 2 | 0 | 0 | 0 |
| 0.80+ | 0 | 0 | 0 | 0 |

## Is more time corrective?

Moving the same 16 risk-selected tasks toward the one-second anchor generally
improved exact labels:

| Added knots per active hand | Exact pass | Within 0.2 | Within 0.5 | Mean worst residual |
|---:|---:|---:|---:|---:|
| 0 | 0/16 | 1/16 | 2/16 | 1.596 |
| 5 | 1/16 | 1/16 | 2/16 | 1.428 |
| 10 | 1/16 | 2/16 | 3/16 | 1.372 |
| 20 | 3/16 | 3/16 | 5/16 | 0.904 |
| one-second anchor | 4/16 | 5/16 | 8/16 | 0.531 |

Across all 64 tasks at the one-second anchor (combining the 16 explicit anchor
checks with the 48 fallback-anchor checks), only 9 were exact passes, 13 were
within 0.2, and 20 were within 0.5.  Thus actor anchor proficiency, not merely
critic conservatism, currently caps attainable proposal yield.

## Consequence

Use the full-distribution risk statistic, but bootstrap it with an exact
anchor-first acquisition loop.  Only search inward for tasks whose anchor is
within the accepted range.  Exact-check the proposed point, backtrack toward
the anchor on failure, and put every checked boundary/collar label into critic
replay.  Recalibrate on these selection-induced labels rather than ordinary
actor rows.  Until that dataset exists, the critic is useful for ranking and
reducing violation severity, not for certifying its own optimized contour.
