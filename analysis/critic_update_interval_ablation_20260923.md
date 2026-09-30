# Consolidated timing-critic updates (2026-09-23)

## Design

Actor labels are detached and cached on CPU for four outer updates.  On the
fourth update, all labels are inserted into the bounded replay/audit stores and
the distributional critic makes one optimizer update using a 1024-row replay
sample instead of four 256-row updates.  Timing proposals continue to run on
every actor step, using a decision critic that is at most three actor updates
stale.

The cache is included in checkpoints, and a partial cache is flushed at normal
training termination.  The live-to-decision EMA and anchor-rate EMA use
`decay ** interval` on consolidated updates to retain their actor-step time
constants.  Calibration also retains its actor-step cadence.

Two learning-rate policies were tested:

- `1x`: one 1024-row AdamW update at the ordinary critic LR.
- `4x`: the same update with a linear-scaled critic LR intended to approximate
  the parameter travel of four sequential updates.

## Matched 40-actor-update result

All arms resumed the same step-20000 checkpoint with actor batch 512 and the
same production loss/line-search recipe.

| critic policy | critic optimizer updates | wall time | actor updates/s | eval accepted | short accepted | critic loss | train entropy | OOD entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| every step, B256 | 40 | 86.4 s | 0.463 | 21/81 | 7/27 | 0.789 | 0.337 | 0.951 |
| every 4, B1024, 1x LR | 10 | 84.3 s | 0.475 | 21/81 | 7/27 | 0.860 | 0.366 | 0.965 |
| every 4, B1024, 4x LR | 10 | 85.0 s | 0.471 | 21/81 | 6/27 | 0.875 | 0.358 | 0.964 |

Run directories:

- `/tmp/rhythm_b512_control40_20260923`
- `/tmp/rhythm_critic_interval4_lr1_trial40_20260923`
- `/tmp/rhythm_critic_interval4_lr4_trial40_20260923`

## Conclusion

Consolidation works and preserved the short-run actor acceptance, but it only
improved end-to-end speed by about 2.5%.  The critic is not the main production
bottleneck; actor physics, auxiliary solves, and SaLSa probes dominate.  The 1x
LR policy is the safer of the two consolidated variants.  Its higher critic
loss means this should remain opt-in until a longer from-scratch run confirms
that the slower optimizer-update cadence does not delay opening or tracking the
timing curriculum.

## Fresh-critic tracking test (600 actor steps)

A matched fresh Phase-2 run was then started from the same phase-1 warm start as
the original 20k baseline.  Six hundred actor steps were sufficient to fill the
8192-row recent replay repeatedly and make 150 consolidated critic updates.

| metric at actor step 600 | every-step B256 | every-4 B1024 |
|---|---:|---:|
| elapsed wall time | 913.9 s | 853.8 s |
| actor steps/s | 0.676 | 0.723 |
| critic loss | 1.144 | 1.247 |
| calibrated precision | 0.700 | 0.700 |
| calibrated recall | 0.976 | 0.642 |
| true manageable rate | 0.594 | 0.587 |
| predicted manageable rate | 0.828 | 0.538 |
| anchor-close EMA | 0.3024 | 0.3024 |

The interval-4 run was 6.6% faster to step 600 and preserved the anchor-rate
tracking almost exactly, confirming that the powered EMA is working.  However,
one large AdamW update is not equivalent to four sequential updates: critic
loss remained 9% higher and feasible-region recall lagged by 33 percentage
points.  Both arms retained the requested 70% precision, so interval 4 is safe
but materially more conservative during early learning.  An interval of 2 is
the better next compromise if early frontier opening matters.
