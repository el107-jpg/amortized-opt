# Phase-2 training-instability forensics

Scope: read-only analysis of:

- `runs/rhythm_bellman_nf1_ls_full_rnea_fresh`
- `runs/rhythm_bellman_nf1_ls_full_rnea_replay_dw`
- `runs/rhythm_bellman_nf1_ls_full_rnea_replay_dw_frontier`

Analysis date: 2026-09-11.

## Executive finding

The logs do **not** support one common "irrecoverable spike" across all three
runs.

1. `fresh` has a real, sustained model failure. It is a gradual torque runaway,
   not a one-step explosion: loss rises from 123.7 at step 4000 to 7045.2 at
   6100 and never returns below 5261.8. Waist torque violation rises from 0.485
   to 112 N·m-scale logged residual over the same interval. The fixed 81-case
   evaluation also degrades.
2. `replay_dw` has no comparable spike. It resumes the `fresh` **best** model at
   step 2000, not the failed late model, and remains finite with no skipped
   gradients. Its train loss drifts from a median 100.8 at 2500–6000 to 241.3
   at 24000–27325 while evaluation improves.
3. `frontier` has an immediate training-distribution/restart shock and a
   sustained rise in train loss, but no consistent mean evaluation degradation
   versus `replay_dw`.
   Across ten common evaluations (8000–26000), frontier minus replay has mean
   acceptance -0.0111, mean `ok_torque` indistinguishable to floating-point
   precision, and median contact error **0.82 mm better**. Individual paired
   changes are noisy and can reach about ±6–9 percentage points. Its 2–4x
   larger train losses are therefore mainly a changed training distribution,
   not an irrecoverable model collapse.

The leading mechanism is a nonstationary hard-example feedback loop. In
`fresh`, unfiltered recency replay is a medium-confidence candidate. In
`frontier`, a changed training-data process is strongly consistent with the
larger torque cost, but cloud/new-row sampling is confounded with reconstructed
replay/post-hit pools and new RNG/roller state. The logs do **not** directly
record the violation scores of rows actually sampled by `fresh`, and its replay
ring and RNG stream were not preserved across the comparison fork. "Replay
torque violations caused the fresh failure" is therefore plausible but not
proven.

The SaLSa line search is a credible amplifier. It never rejects a sampled
search in `fresh`, accepts strongly uphill probe evaluations because its
smoothed reference remains positive, and probes a different vector from the
momentum-bearing vector it executes. It does not by itself explain why only
`fresh` collapses.

AL changes add a small 100-step sawtooth but are not the spike trigger. The
dominant waist multiplier was already capped and its quadratic penalty was
zero before the failure; raw violations grew by orders of magnitude before
the remaining torque multipliers changed materially.

## Provenance and timing caveats

- Train records are emitted every 50 steps. Loss/term/violation fields are
  averages over that 50-step window. Most `ls_*` fields describe only the most
  recent update, not a window average. Spike timing is therefore resolved to
  about 50 steps.
- AL updates occur every 100 steps after the baseline metrics are accumulated.
  A record at step 5000 contains a 4951–5000 loss window measured entirely
  under the pre-5000 AL state but displays the newly updated `lam_i`/`pen`
  state.
- `fresh` has 313 records: 296 train, seven eval, seven probe, one each
  start/stop/done.
- `replay_dw` has 534 records: 506 train, twelve eval, twelve probe, two start,
  one stop, one done. Its first launch OOMed before a train record; the second
  launch resumed the same `fresh/best.pt`.
- `frontier` has 485 records: 457 train, eleven eval, eleven probe, two start,
  two `best_blocked`, one stop, one done. Its first launch OOMed before a train
  record; the second resumed the same `replay_dw/last.pt`.
- The checkpoints referenced by the launch logs are no longer present in any
  of the three run directories. Parameter tensors and optimizer moments could
  not be inspected directly.

## Run lineage and controlled comparisons

Effective config diff from `fresh` to `replay_dw`:

- `replay_viol_downweight`: 0 to 0.05.
- Non-causal bookkeeping/runtime differences: max steps 60000 to 40000,
  checkpoint retention, GPU, and disabling an unused auto-frontier threshold.
- Model config, scenario config, pool sizes (2630 rows at H=25 and 28729 at
  H=50), full-weight schedule, AL config, and line-search config match.
- Important confounds: `replay_dw` restarts the replay/roller and RNG stream
  from `fresh/best.pt` at step 2000. It does not fork the live `fresh` replay
  ring.

Effective config diff from `replay_dw` to `frontier`:

- Frontier enabled, idle/far cloud source enabled, and 64 cloud rows spliced
  into every batch.
- Replay proficiency sampling enabled with 40% proficient and 40% new rows,
  violation cap 8.
- `suffix_warmup` changes 1000 to 1, but both are already fully active at the
  step-6000 fork.
- Model, scenario, base pool, AL, optimizer, and line-search configs match.

All logged curriculum multipliers are identically 1.0 throughout all runs:
`alpha`, all four peak alphas, waist alpha, and full-RNEA alpha. `full_weights`
and cost-curriculum transitions did not cause these windows.

## Precise windows

### `fresh`: startup transient

- Step 200: loss 402.0.
- Step 250: 1898.0.
- Step 300: 5053.3.
- Local peak step 400: 6176.0.
- It recovers below 1000 at 850, below 500 at 1100, below 200 at 1600, and
  below 150 at 1850.
- AL changes are temporally mixed into this transient: the 250→300 jump
  coincides with a multiplier-vector change, but the 200→250 jump occurs with
  no AL change. This is recoverable and distinct from the later failure.

### `fresh`: sustained catastrophic window

Stable reference median at steps 2000–3750: **107.96**.

- First ten consecutive records above 1.25x reference: step 4050.
- Above 1.5x: 4150.
- Above 2x: 4550.
- Above 3x: 4850.
- Above 5x: 5000.
- Above 10x: 5150.
- Above 25x: 5350.
- Above 50x: 5750.
- Peak: **7820.69 at step 6800**.
- For steps 6200–14800, minimum 5261.75, median 5954.59, and final-20-record
  median 5725.07. It is irrecoverable over the observed continuation.

Selected trajectory:

- 4000: loss 123.7, torque cost 25.35, waist torque violation 0.485,
  torque-peak violation 0.668.
- 4500: 210.8, 95.0, 2.203, 2.522.
- 5000: 716.6, 494.8, 12.90, 13.50.
- 5500: 3688.3, 2990.4, 72.61, 73.92.
- 6000: 6594.6, 5621.7, 111.10, 112.30.
- 7000: 7163.4, 6116.0, 90.19, 91.64.

The rise is already visible by 3750–4050 and evolves over roughly 2000 steps.
There is no logged single parameter/output discontinuity.

### `replay_dw`: drift, not an irrecoverable spike

- Median loss 2500–6000: 100.81 (p90 114.98).
- 6000–12000: median 152.86.
- 12000–18000: 211.87.
- 18000–24000: 253.52.
- 24000–27325: 241.28.
- Maximum 392.67 at step 22450. Its local post-window median does not jump;
  the largest rows are ordinary batch variability on top of a slow drift.
- First ten consecutive records above 1.5x the early reference occur at 9450;
  2x occurs only at 21750; 3x never occurs.

### `frontier`: restart/config shock plus sustained hard-batch drift

- First frontier train record, step 6050: loss 420.5 and torque cost 246.4.
  The non-frontier fork at the same step reports 113.4 and 16.0.
- Frontier returns to loss 104.5 by step 6200 without a restart.
- Reference median 6200–7950: 132.06.
- Ten consecutive records exceed 1.5x at 9300, 2x at 10200, 3x at 12750,
  and 4x at 17300.
- Median loss 10000–16000: 464.13; 16000–22000: 593.28;
  22000–28882: 626.00.
- Maximum 993.85 at step 23250.
- This is sustained in the train metric but not in the fixed-grid evaluation.

At common 2000-step snapshots from 10000 through 26000, frontier train loss is
1.54x–3.96x replay loss and frontier torque cost is much larger. Across the
same fixed-grid evaluations, mean differences in acceptance, hit, collision,
and torque rates are all at or below about 1.1 percentage points, but signed
per-evaluation differences are noisier.

## Ranked hypotheses

### 1. Nonstationary training-data distribution / replay feedback — medium for specific-source causality

Evidence for:

- From the same `fresh` best checkpoint at step 2000, the no-downweight
  continuation and `replay_dw` are close through about 3500, then separate:
  at 4050 losses are 142.4 vs 102.8; at 5050, 809.9 vs 94.8; at 6050,
  7045.3 vs 113.4. At 6050 waist torque violations are 111.67 vs 0.25.
- The effective configuration difference is almost entirely replay violation
  downweighting. The weight is `exp(-0.05 * score)`.
- In `replay_dw`, mean unweighted **harvested** score rises from median 17.7
  early to 61.3 mid-run and 72.2 late. Evaluating the unnormalized
  `exp(-0.05 * score)` term at those illustrative scores gives 0.41, 0.047,
  and 0.027. These are not observed mean sampling probabilities: actual
  normalized sampling also contains recency, horizon weighting, and—in
  frontier—proficiency strata. The model remains finite and fixed-grid torque
  acceptance improves, which is consistent with, but does not prove,
  downweighting breaking a feedback loop.
- The step-6000 replay/frontier comparison is strong evidence that the changed
  training-data process changes the train metric. Frontier adds cloud and
  new-row sampling; it immediately sees a 3.7x loss shock and later 2–4x train
  loss while paired evaluation has no consistent mean degradation.
- Frontier child counts rise from 152 to 103,969 at H=25 and from 2,428 to the
  131,072 cap at H=50. Raw child-count/loss correlations are 0.787 and 0.543.

Evidence against / limitations:

- `fresh` did not compute or log `viol_score`, so there is no direct record of
  the torque scores of its sampled replay rows.
- The reported score in the downweighted runs is the score of newly harvested
  current-policy rows, not the score distribution of sampled training rows.
  It is as much an output-health readout as a causal input.
- Raw score/loss correlations are trend-confounded. First-difference score
  versus first-difference loss is only 0.015 contemporaneously in replay and
  0.033 in frontier. Best forward correlations of the differences are weak
  (0.055 and -0.151). No clear score lead is demonstrated.
- `replay_dw` reconstructs empty replay/post-hit pools, starts a newly seeded
  roller, resets the batch RNG stream, and selects a much smaller epsilon after
  resume. It is not a clean one-variable fork.
- Frontier likewise reconstructs rather than restores dynamic pools. At its
  first train record, replay sizes are 6,572/6,228 and post-hit sizes 38/44,
  while the continuing replay comparator has full 40,000/40,000 replay and
  8,192/8,192 post-hit pools. The first record also follows 50 updates. Thus
  the 3.7x shock does not isolate cloud/frontier rows from replay maturity,
  roller/RNG reset, or early adaptation.
- Frontier child-count first differences do not predict loss differences
  (correlations -0.097 at H=25 and -0.018 at H=50); cumulative correlations
  mostly encode time.

Conclusion: a changed training-data process is strongly consistent with
frontier's train-loss separation, but the responsible source is not isolated.
For fresh, distribution feedback is the leading candidate but only
medium-confidence: "high replay score leads spike" remains unproven without
sampled-row telemetry, matched RNG, and a replay-ring snapshot.

### 2. Apparent train instability from evaluation-distribution mismatch — high for frontier

Evidence for:

- Ten paired replay/frontier evals show mean frontier-minus-replay acceptance
  -0.0111, hit approximately 0, torque approximately 0, collision -0.00864.
  There is no consistent signed degradation, but individual acceptance deltas
  range -0.0741 to +0.0864 and torque deltas -0.0617 to +0.0494. At step
  26000, frontier acceptance is 0.2469 versus replay 0.2963.
- Frontier median contact error is 0.82 mm better, jerk about 2.0e6 lower, and
  travel 0.30 lower despite much higher train loss.
- Replay train loss roughly doubles while eval torque violation p90 reaches
  zero from step 14000 onward and acceptance peaks at 0.296 at 26000.

Evidence against:

- `fresh` is not merely a mismatch: eval acceptance falls 0.148 at 4000 to
  0.0617 at 6000, `ok_torque` 0.765 to 0.519, and hit rate eventually falls
  to 0.383 at 14000. However, fixed-grid torque itself partly recovers:
  `ok_torque` is 0.716 at 8000 and 0.753 at 14000, near its step-4000 value.
  Persistent fixed-grid failure is increasingly hit-owned rather than a
  sustained torque-only collapse.
- The one-clip closed-loop probe is noisy and can disagree with the grid:
  `fresh` probe score is 0.0795 at 6000 during the train/eval failure, then
  returns to zero. It is too small a sample to adjudicate stability alone.

Conclusion: most of frontier's "instability" is hard-batch exposure, whereas
fresh contains a real policy degradation.

### 3. Smoothed line-search acceptance and probe/executor mismatch — medium as amplifier

Evidence for:

- Sampled searched rows are almost never rejected:
  - `fresh`: 152 searched, 152 accepted, zero floor commits.
  - `replay_dw`: 165 searched, 164 accepted, one floor commit.
  - `frontier`: 138 searched, 138 accepted.
- Searches usually accept on trial one. `fresh` has 34/152 accepted searches
  where the instantaneous probe loss **increases**.
- At step 5700, during the surge, `F0=7972.78` and `Fprobe=9609.08`
  (instantaneous decrease -1636.29). The smoothed state still has
  `A_prev=51.49`, `A_cand=34.61`, and `B_cand≈1`; with epsilon 5.2e-8 the
  smoothed threshold is negligible, so the grossly uphill probe is accepted.
  A second large uphill probe (-430.05) is accepted at 6100.
- The search probes `p_probe` but executes Adam-EMA-bearing `p_exec`.
  At step 4800, immediately before acceleration, the probe reports a +32.75
  decrease while the logged local executed slope is **+2829.56** (ascent),
  almost entirely from the logged `p_momentum` component. In default gradient
  mode, that component uses `m_next`, so it mixes the current gradient with
  carried history; it is not pure historical momentum. Strict executed-step
  checking and the actual-update diagnostic are both disabled.
- The stable replay fork selects a much smaller rate during the critical
  comparison. At steps 4000–5000, replay is near 2.9e-9 while fresh is
  commonly 0.8e-8–2.2e-8. The major executed component's logged norm
  (`ls_u_mom_norm`) has median 3.10e-4 replay versus 1.46e-3 fresh.
- The advertised `ls_u_total_norm` and fresh/momentum fractions contain a
  diagnostic bug in default gradient mode. Execution is
  `epsilon * (p_momentum + p_decay)`, but `step_vector_breakdown()` reports
  `epsilon * (p_probe + p_momentum + p_decay)`. These fields must not be
  interpreted as an actual executed-step decomposition.

Evidence against:

- There is no abrupt epsilon explosion. `fresh` epsilon median is 6.52e-9
  pre-window, 1.47e-8 at onset, 2.15e-8 during the surge; global max is
  7.18e-8, far below the 1e-5 selector ceiling.
- Instantaneously uphill probes also occur in the stable run
  (`replay_dw`: 33/165) and frontier (18/138).
- Executed local ascent slopes are rare: 4/152, 5/165, and 2/138 sampled
  searches.
- No `ls_selection_failed` events or nonfinite executed parameters occur.

Conclusion: line search does not enforce the descent property its name may
imply and likely amplifies the bad distribution/output state. It is not unique
enough to be the sole root cause.

### 4. Decoder torque runaway and gradient dominance — high as symptom, medium as independent cause

Evidence for:

- `fresh` waist residual and torque-peak residual rise by about 200x from 4000
  to 6000. Waist is the dominant actuator segment.
- Correct fresh-forward per-term gradient attribution shows
  `grad_torque_frac` 0.795 at 3500, 0.940 at 4500, 0.974 at 5500, and 0.993 at
  6000. After 6200 its median is 0.995. Torque gradient norm rises from about
  5e3 to 1.8e5.
- Evaluation torque and hit behavior degrade, proving this is not only AL
  repricing.

Evidence against:

- This does not identify what starts the drift; torque dominance can be the
  correct response to already-bad replay rows.
- Pairwise torque-vs-strike/shape/feasibility/waist cosine diagnostics switch
  signs across probes rather than showing one persistent conflict.
- The configured torque gradient cap is not strongly active on average:
  median logged cap scale is 0.967 at onset, 0.956 during the surge, and 0.968
  afterward. It does not appear to suppress torque correction by orders of
  magnitude.
- Near-unit `grad_torque_frac` also occurs frequently in frontier while paired
  evaluation remains sound.

Conclusion: the decoder really enters a waist-torque failure basin in `fresh`,
but the logs favor data/optimizer interaction over an isolated decoder output
head discontinuity.

### 5. Nonfinite backward/AMP gradients — medium as recovery blocker, low as trigger

Evidence for:

- `fresh` reports 767 skipped optimizer steps out of 14843 (5.17%).
- Successful-update deltas imply skip rates of 3.49% pre-window, 2.74% onset,
  3.91% surge, and 6.40% after 6200.
- `grad_norm` is NaN on 22 sampled train rows. The torque-cap fraction metrics
  are NaN in 263/296 50-step records because a single nonfinite backward in a
  window poisons their average.
- The training loop skips the optimizer update on nonfinite `gnorm` but still
  accumulates finite baseline violation metrics for AL. This can make recovery
  harder once violations are large.
- Replay and frontier have zero skipped steps under the same model/AMP setup,
  linking nonfinites to the bad data/output regime.

Evidence against:

- The finite baseline loss never produces a `skip` event; no parameter NaNs,
  nonfinite executed parameters, or nonfinite eval outputs are logged.
- Skip rate does not jump before onset; it is lower at onset than in the stable
  pre-window and rises after the torque runaway.
- The many probe latency NaNs are expected undefined statistics when there is
  no in-disk crossing, not numerical model failures.

Conclusion: nonfinite gradients are mainly a consequence and persistence
factor, not the initiating event.

### 6. AL multiplier/penalty discontinuity — low

Evidence for:

- AL does change every 100 steps without forcing a line-search history reset.
- Mean next-window loss delta after an AL boundary is modestly positive:
  +26.3 fresh, +1.45 replay, +6.04 frontier, versus +4.42, -1.34, and -4.66
  on the alternate half-cycle. This is a measurable repricing sawtooth.
- The startup 250→300 jump overlaps a large multiplier change.

Evidence against:

- At catastrophic onset, dominant `lam_i.torque_waist` and left-shoulder
  multipliers are already capped at 20. They remain constant.
- From 4000 to 5000, waist raw violation rises 0.485→12.90 and torque cost
  25.35→494.8 while torque-peak lambda only rises 5.903→6.142 (4%) and
  right-shoulder lambda 11.89→12.57. The residual moves first and much more.
- Dominant torque quadratic penalties are zero for waist/left shoulder and
  only 1e-6 for other torque segments; no torque PHR growth occurs.
- Correlation of multiplier-vector change with the next loss delta is 0.076
  fresh, 0.008 replay, and 0.026 frontier. Penalty-change correlations are
  similarly negligible.
- Several largest loss jumps have no AL change. Fresh 200→250, 300→350,
  5300→5350, 5400→5450, 5500→5550, and 5700→5750 are examples.
- All cost schedules are fixed at full strength, so there is no curriculum
  discontinuity masquerading as an AL event.

Conclusion: AL repricing changes the displayed objective and can amplify an
existing violation, but it does not explain catastrophic onset.

### 7. Optimizer moment pathology — unresolved, low to medium

Evidence for:

- The step-4800 local executed ascent is in the Adam EMA execution component,
  proving `p_exec` can reverse the direction validated using `p_probe`.
- Restored moments influence selector behavior, and actual executed-step
  objective checks are absent.

Evidence against:

- Frontier restores the same optimizer state as replay at step 6000 and sees a
  large loss difference immediately with epsilon fixed at 3.18e-9 and a small
  logged EMA-component norm, pointing to batch source rather than a new moment
  discontinuity.
- Replay restores optimizer moments and remains finite. No checkpoint tensor
  evidence is available to show corrupt `exp_avg_sq`.
- The logged component fractions cannot rule moment pathology in or out:
  `p_momentum` contains current-gradient and carried-history contributions,
  while the reported total incorrectly adds the non-executed probe vector.

Conclusion: a probe/executor mismatch is real, but generalized moment
corruption is neither evidenced nor fully ruled out by the available
diagnostics.

### 8. Schedule/LR/resume, bucket mix, or parameter NaNs — very low

- Every cost alpha is 1.0 for every train record.
- Under line search, the selector owns LR; the cosine `lr_horizon=30000` is not
  written to optimizer groups. Absolute-step cosine resume artifacts do not
  apply to these runs.
- Batch size remains 512. Bucket-25 fraction has negligible same-time loss
  correlation: 0.009 replay and -0.095 frontier. It does not explain drift.
- The duplicate starts are pre-update CUDA OOMs on occupied GPUs, not model
  OOMs during a spike. Relaunches use the original source checkpoints.
- Losses and executed parameters stay finite. Only backward gradients become
  nonfinite in `fresh`.

## Falsification experiments

These should start from one immutable checkpoint and use a serialized RNG,
replay-ring snapshot, and frozen canary batches. The current run directories
do not retain the necessary checkpoints/rings.

1. **Replay causality, same ring**
   - Fork the same step-3500 or step-4000 model/optimizer/AL/LS state and exact
     replay ring.
   - Arms: beta 0; beta 0.05; no replay; beta 0 with hard score cap/rejection.
   - Log the **sampled** row score mean/p50/p90/max, source, age, bucket, and
     effective sampling weight every step. Keep line-search epsilon fixed.
   - Replay hypothesis is falsified if beta/no-replay arms have the same
     frozen-canary torque trajectory with matched epsilon.

2. **Separate moving-data loss from model drift**
   - Save 3–5 immutable canary batches: fresh H25/H50, low-score replay,
     high-score replay, frontier seed, and frontier child.
   - Evaluate each before every update and every 50 steps without mutation.
   - A rising online loss with flat canary losses proves distribution drift;
     rising canary torque proves parameter drift.

3. **Line-search executor safety**
   - From the same pre-onset state compare: current smoothed probe;
     current-batch unsmoothed Armijo; probing the actual `p_exec`; and strict
     actual-update veto.
   - Set `actual_update_diagnostic_every=1` at least through 3500–6500.
   - The line-search hypothesis is falsified if actual `F(theta_exec)` remains
     decreasing on canaries while the failure still occurs.
   - Log the true norm of `epsilon * p_exec` and separately decompose `m_next`
     into current-gradient and carried-`m_old` contributions; do not use the
     current `ls_u_total_norm` for this purpose.

4. **Rate versus replay content**
   - Cross beta `{0, 0.05}` with fixed epsilon `{2.5e-9, 1.5e-8}` and restored
     versus zeroed moments.
   - This isolates downweighting from the 5–8x smaller rate selected in the
     stable fork.

5. **AL freeze**
   - Freeze all lambda/pen states at step 4000 in one arm; live AL in another;
     a third resets SaLSa A/B history after every AL change.
   - If the same raw waist violation runaway occurs under frozen AL, moving AL
     is falsified as the root. Compare raw `v_*`, not only priced `c_*`.

6. **Frontier 2x2**
   - From replay step 6000, cross cloud splice `{0, 64}` with replay
     proficiency `{off, on}`. Keep suffix settings and epsilon identical.
     Restore the same mature replay/post-hit pools and all RNG/roller state in
     every arm.
   - Log source-conditioned losses. The current evidence predicts immediate
     train-loss separation but similar fixed-grid evals.

7. **Nonfinite-gradient origin**
   - Repeat the short pre/onset window with AMP off, finite hooks per top-level
     cost term, and per-source row-score logging.
   - Record whether the first nonfinite originates in torque VJP, another term,
     or recombination after torque-cap surgery. Treat nonfinite windows as a
     lead only if their rate rises before canary torque does.
   - Add an arm where nonfinite-gradient rows do not contribute violations to
     AL, isolating whether the existing side-effect cadence blocks recovery.

8. **Per-bucket/source evaluation**
   - Report loss and raw violation separately for H25/H50 and every source.
     This will falsify residual bucket-mix explanations and reveal whether
     frontier H50 children own the apparent drift.
   - Before the first frontier update, evaluate replay, cloud-seed, and child
     batches at the identical step-6000 parameters.

9. **Evaluation robustness**
   - Repeat the fixed-grid evaluation across multiple fixed grids/seeds and
     report per-case paired transitions. Current frontier means contain
     substantial signed cancellation.

## Artifact inventory

- `analysis/forensics_logs_phase2_instability/analyze_metrics.py`
  - Standalone parser, robust spike ranking, nonfinite inventory, lead/lag
    correlations, context windows, and CSV export.
- `analysis/forensics_logs_phase2_instability/metrics_forensics.json`
  - Machine-readable event counts, spike windows, key inventory, correlations,
    and nonfinite counts.
- `analysis/forensics_logs_phase2_instability/rhythm_bellman_nf1_ls_full_rnea_fresh_train.csv`
- `analysis/forensics_logs_phase2_instability/rhythm_bellman_nf1_ls_full_rnea_replay_dw_train.csv`
- `analysis/forensics_logs_phase2_instability/rhythm_bellman_nf1_ls_full_rnea_replay_dw_frontier_train.csv`
  - Flattened numeric train records for independent plotting/statistics.

