# Anchor-first distributional timing curriculum

The production Phase-2 loop can now train the actor and a categorical timing
critic together from the ordinary Phase-1 initialization.

## State machine

1. **Anchor acquisition:** choose only the longest pool bucket and retime every
   fresh active hand to its horizon (50 knots = 1 second in the current pool).
2. Train the actor on those anchor trajectories and train three independent
   critic heads from the exact four-constraint diagnostics already computed by
   the actor cost pass.
3. Train every critic head from its own replay draw.  Each update also applies
   low-weight high-entropy exposure to fresh globally random physical tasks.
4. Recalibrate the full-distribution risk threshold on the newest exact-label
   window.  The risk includes expected exceedance/severity, upper-tail CVaR,
   and between-head variation.
5. Keep timing search closed until all of the following hold:
   - at least `anchor_min_samples` exact anchor labels exist;
   - the EMA fraction inside the configured close-feasibility range is at least
     `anchor_min_close_rate`; and
   - calibration can find a finite risk threshold meeting `risk_min_precision`.
6. Once open, physical tasks are grouped before timing expansion.  At batch
   512, 40 independently sampled `(q, qdot, drum goals)` tasks each generate
   two conservative directional Pareto-frontier timings and four random
   component-wise-looser timings inside the rectangle from that frontier point
   to `(1 s, 1 s)`.  This produces 80 boundary and 160 interior rows.
7. The remaining 272 primary rows come from the exact-confirmed shifted-plan
   generation produced by the preceding outer step.  The generation is fully
   replaced every step, rather than accumulated in an unbounded replay.  If it
   is empty, its slots use distinct 1-second anchors.  MPC ingress permits a
   terminal miss up to 40 mm and collision SDF down to -10 mm; joint and torque
   retain the configured normalized close-feasibility threshold.
8. Every proposed boundary/interior row receives a no-gradient actor/physics
   evaluation.  Only rows within the configured normalized violation range
   replace their anchor rows in actor training; accepted and rejected
   evaluations both train the critic.
9. Suffix-Bellman parents are drawn proportionally from every primary stratum,
   not only MPC rows.  Their exact child diagnostics train the same critic with
   the same per-example loss.  Temporal-consistency child solves are disabled
   in this regimen because suffix-Bellman already supplies shortened replans.

The actor, critic ensemble, replay, calibration threshold, anchor proficiency
EMA, and open/closed state are saved together in `last.pt`.

The gradient-updated critic also has a smoothed copy used for timing decisions.
Its default weight EMA decay is 0.95 so it can follow the rapidly changing actor
during anchor training; this is separate from the anchor proficiency EMA.

## Principal CLI flags

```text
--timing-critic-mode assist
--timing-critic-representation distributional
--timing-critic-heads 3
--timing-anchor-curriculum
--timing-anchor-min-samples 2048
--timing-anchor-min-close-rate 0.50
--timing-acceptable-normalized-violation 0.2
--timing-risk-min-precision 0.50
--timing-critic-assist-frac 0.05
--timing-global-ood-fraction 4.0
--timing-global-ood-weight 0.0025
--timing-primary-base-tasks-per-512 40
--timing-primary-boundary-per-task 2
--timing-primary-interior-per-task 4
--timing-primary-mpc-terminal-limit 0.040
--timing-primary-mpc-collision-sdf-limit -0.010
```

The small assist fraction makes expansion gradual.  The exact verification
barrier, rather than critic calibration alone, controls actor admission.

## CPU smoke results

A realistic one-step run from
`runs/phase1_rhythm_bellman_nf1_prsp/best.pt` produced 64 exact critic labels.
All active TTH values were exactly 1 second, anchor close-feasibility EMA was
zero, calibration produced no finite threshold, and the curriculum correctly
remained closed.

A separate intentionally permissive smoke (`acceptable_normalized_violation =
10`) forced the transition for plumbing validation.  Step 1 opened the
curriculum; step 2 proposed and exact-evaluated 32 shortened rows, committed all
32 under the artificial range, stored shortened labels down to one knot, and
checkpointed the open state.  This run is not a quality result—the permissive
range caused the actor loss to jump—and exists only to verify the complete
closed-to-open execution path.
