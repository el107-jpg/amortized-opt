# Anchor-first distributional critic: 200-step CPU run

Date: 2026-09-17

## Setup

- Initialized the actor from `runs/phase1_rhythm_bellman_nf1_prsp/best.pt`.
- Ran 200 Phase-2 outer updates on CPU with 64 actor rows per update.
- Forced fresh rows to the 1.0 s anchor horizon until the safety curriculum opens.
- Trained a three-member categorical violation critic once per outer update with a
  256-row replay batch per member.
- Added 4x fresh global OOD rows per member at aggregate weight 0.0025.
- Required at least 2,048 anchor labels, close-rate EMA >= 0.50, and a calibrated
  risk threshold with precision >= 0.50 before allowing shorter-time proposals.

Artifacts are in `/tmp/distributional-anchor-cpu-200/` (`last.pt` and
`metrics.jsonl`).

## Runtime and actor result

- 200/200 updates completed in 460 s: 0.43 outer updates/s.
- Training loss fell from 5,579 at step 10 to 431 at step 200.
- The logged torque cost fell from 1,435 to 308.
- Training terminal miss moved from 59.1 mm to 37.2 mm.
- Training collision penetration moved from 35.8 mm to 5.6 mm.
- The final fixed 81-case evaluation accepted 0/81 trajectories. Individual
  pass rates were hit 42%, collision 56%, torque 6%, velocity 100%, joint limits
  59%, and elbow orientation 93%.

The fixed evaluation remains substantially harder than the current actor. This
run is evidence of rapid anchor improvement, not end-to-end feasibility yet.

## Curriculum result

- Exact labels observed: 12,800.
- Anchor close-rate EMA was zero through step 90, reached 7.1% at step 150, and
  ended at 10.8%.
- The exact close fraction over the final 40 actor steps was 11.1%.
- The required opening rate was 50%, so the curriculum stayed closed.
- No shorter-time critic proposals were admitted.

Final-40-step per-channel fractions with normalized violation <= 0.2:

| Constraint | Fraction close |
|---|---:|
| Terminal | 41.7% |
| Collision | 51.6% |
| Joint limits | 66.5% |
| Torque | 72.0% |

The worst channel was terminal on 39.1% of rows, collision on 30.5%, joint
limits on 15.4%, and torque on 15.0%. The joint all-channel pass rate is much
lower than any marginal pass rate, as expected.

## Critic and OOD behavior

Measured on the newest 40 actor steps (2,560 rows):

| Decision model | Feasible-vs-failure AUROC | Best recall at >=50% precision |
|---|---:|---:|
| Live critic | 0.593 | 0.000 |
| EMA critic (used for decisions) | 0.389 | 0.000 |

Mean normalized categorical entropy:

| Model | Actor rows | Fresh global OOD rows |
|---|---:|---:|
| Live critic | 0.765 | 0.839 |
| EMA critic | 0.966 | 0.976 |

The OOD objective is producing the desired entropy ordering, especially in the
live network. However, 200 updates are not enough for reliable admission: the
live risk ranking is only modestly informative and cannot isolate any subset at
50% precision. The decision EMA is much worse because decay 0.995 leaves it
dominated by its nearly uniform initialization during this short, rapidly
changing online run.

## Conclusion

The safety behavior is correct: a weak critic did not generate speculative
short collocations. The actor did begin creating near-feasible anchor examples,
but neither the anchor proficiency nor critic calibration was ready after 200
steps. Before a longer from-scratch experiment, the decision EMA should receive
an initialization-aware warmup (copy live weights early, or use a substantially
faster decay during warmup). That change targets the observed lag without
lowering the exact-label curriculum gate.
