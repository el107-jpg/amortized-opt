# Task-Time Frontier Critic Implementation Plan

> **Production amendment (2026-09-18):** This document records the original
> guarded rollout plan.  The measured production curriculum now uses the
> calibrated distributional critic to select timing pairs without a separate
> actor/physics pre-admission evaluation.  Forty permanent 1-second anchor rows
> remain in each 512-row batch; 40 critic-selected frontier and 160 looser
> interior rows enter the normal actor pass directly, alongside 272
> previous-step MPC rows.  That single normal pass supplies the exact critic
> labels.  Accordingly, references below to an "exact verification barrier"
> describe the superseded rollout design, not the current admission behavior.

## Implementation status

The initial implementation now includes the Phase 0/1 contracts, shadow
critic, bounded replay/audit storage, checkpointing, two-dimensional frontier
primitives, atomic retiming, and an exact-verified directional sampler for
`assist` and `control` modes. Critic replay and the priority audit are stratified
across manageable, near-outside, and far-failure targets. Shadow warmup evaluates
only trajectories already evaluated by the actor objective: the ordinary cost
batch and the already-computed suffix-Bellman children. No critic-only actor or
physics passes are scheduled. The
direct boundary proposal model, periodic raw audit relabeling, and a dedicated
offline full-grid calibration command remain later optimizations; the violation
critic and exact verification path do not depend on them.

## Objective

Replace combinatorial spatial-voxel expansion with a learned model of the actor's task-time feasibility boundary. The new system should:

- sample broad starting states and goals without enumerating a high-dimensional voxel cloud;
- find aggressive but plausible two-hand time-to-hit pairs;
- use every exact rollout, successful or not, as continuous critic supervision;
- prevent uncertain critic predictions from admitting dangerous actor-training examples;
- retain proficiency without storing every historical collocation point; and
- preserve the current voxel system as a fallback until the replacement is measured and trusted.

The critic is a proposal and prioritization mechanism. The current actor plus exact physical-cost calculation remains the authoritative feasibility oracle.

## Core representation

### A task does not have one unique time-to-hit

For a fixed starting state and pair of goals, there is generally a curve of useful time pairs rather than one correct scalar time. The two active hands must therefore use an atomic pair:

```text
T = (t_left, t_right)
```

The two values must never be selected independently from different trajectories. A pair can be jointly infeasible even when each component appeared feasible in another pairing. A task with one idle hand is treated as a one-dimensional special case by retaining the existing idle-hand sentinel and mask semantics.

For synthetic exploration, a random state and goal are supplemented by:

- a generous timing anchor `A = (A_left, A_right)`, initially around `(50, 50)` control knots at `CONTROL_DT = 0.02`, with longer fallbacks if required; and
- a positive search direction `d`, which expresses the desired tradeoff between shortening the left- and right-hand times.

The search then follows

```text
T(alpha) = A - alpha * d
```

until it reaches the earliest failure connected to the known-manageable anchor. Real rhythm tasks continue to provide their requested timing pair directly.

### Critic inputs

The production critic deliberately uses a minimal physical task description:

- initial joint state `q0` and velocity `qd0`;
- ordered left/right Cartesian drum-goal positions; and
- the complete two-hand time-to-hit pair.

An inactive hand has zero time-to-hit and its goal slot is zeroed. The critic
does not receive the actor's context embedding, future-task tokens, initializer
statistics, horizon, active masks, MPC flags, or example provenance. Raw goal
coordinates avoid a nonstationary dependency on the actor encoder and preserve
generalization to goal jitter; the critic MLP itself is their encoder. This is
an intentional Markov-style approximation: different initializer routes for
the same physical task are treated as observations of the same conditional
violation distribution rather than separate inputs.

### Critic targets

Use the exact per-row diagnostics already computed by `_gate_from_ct` in `train_cost.py`:

- terminal-region distance;
- minimum collision signed distance;
- maximum joint-box excess; and
- maximum torque excess.

Convert all four to positive-means-bad residuals:

```text
r_terminal = terminal_region_dist - terminal_limit
r_collision = collision_sdf_limit - collision_min_sdf
r_joint    = joint_box_excess_max - joint_limit
r_torque   = torque_excess_max - torque_limit
```

Initial limits should match the current loose proficiency gate where possible: 20 mm terminal distance, -5 mm collision SDF, the configured torque cap (currently commonly 8 Nm), and a newly explicit joint-excess cap.

Normalize each target with a reversible robust transform such as scaled `asinh` or signed `log1p`. Train primarily with asymmetric Huber regression, charging underprediction more than overprediction, and add a per-constraint logistic sign loss concentrated near the zero crossing. The sign loss uses the same signed scalar output and exact label rather than a separate classification head, so feasibility cannot contradict the predicted magnitude. Keep the physical-unit values available for reporting. Every exact evaluation supplies a useful four-component label, including severe failures; no result is reduced to only a binary success label.

## Critic ensemble and uncertainty

Start with five independently initialized small MLP heads. Each head sees the same detached input features but should initially have its own learned parameters rather than sharing a learned critic trunk. Train heads with independent bootstrap masks or Poisson bootstrap weights.

For head `m`, define a scalar manageability score at a candidate pair as the worst normalized violation:

```text
S_m(T) = max_j normalized_residual_m,j(T)
```

The nominal prediction remains the unbiased ensemble mean. Pessimism is applied only when making a timing decision, using either:

- the 0.8 quantile of the five head scores; or
- an equivalent rule requiring at least four of five heads to classify the pair as manageable.

The implemented decision rule is `quantile(S_m) <= decision_threshold`, where
`decision_threshold` is nonpositive and measured in robust normalized residual
space.  Offline calibration splits future labels by whole actor step, selects
the quantile and threshold that retain the most feasible rows under a requested
false-manageable budget, and reports false-positive and false-discovery rates
separately.  Calibration is checkpoint-specific; zero remains the safe default
until a held-out report supplies a threshold.

Maintain exponential-moving-average copies of all heads for timing selection. Live heads train on exact labels; EMA heads make less volatile proposals. If head disagreement later proves overconfident outside the training distribution, add randomized prior functions or an explicit support-distance feature, but do not make those part of the first implementation.

Randomized prior functions are now available as an ablation: each learned head
can add a differently initialized, permanently frozen MLP scaled by
`randomized_prior_scale`.  Priors are attached only after all learned heads are
initialized, keeping learned initial weights identical in seeded prior/no-prior
comparisons.  The default scale remains zero until uncertainty-error ranking
and held-out calibration demonstrate an improvement.

The first static-replay ablation used 8,192 identical training rows, 4,000
updates per arm, and 1,600 identical future rows.  A 0.25 prior alone increased
mean head-score disagreement but did not improve error ranking.  Combining the
prior with bootstrap probability 0.5 improved the held-out calibrated operating
point, but disagreement remained anti-correlated with decision errors.  The
prior therefore remains opt-in pending a directional-frontier/OOD ablation.

The subsequent frozen-actor directional ablation evaluated 1,280 identical ray
proposals per arm.  The ordinary-row calibrated thresholds rejected the
generous anchor on more than 99% of rays and produced no true positives among
the handful of predicted-manageable points.  At the uncalibrated zero threshold,
predicted-manageable precision was 1.3% for baseline, 2.1% for the 0.25 prior,
and 1.2% for prior plus 0.5 bootstrap.  Randomized priors therefore do not fix
the distribution gap by themselves.  Frontier-selected exact labels must first
enter the critic's training/calibration stream; prior scale remains zero by
default.

## The two-dimensional timing frontier

For scenario `z`, define the conservative predicted feasible region:

```text
F(z) = {T : quantile_0.8(S_1(T), ..., S_5(T)) <= 0}
```

The boundary of interest is the Pareto-minimal portion of this region under componentwise time ordering. It is generally a one-dimensional curve in the two-dimensional time plane, not a single answer.

### Full-grid calibration path

For shadow-mode evaluation and debugging:

1. Evaluate the critic on a coarse two-dimensional grid, initially every five timing knots.
2. Retain only the manageable connected component containing the generous anchor.
3. Extract its Pareto-minimal contour.
4. Refine cells near that contour at one-knot resolution.
5. Compare a sampled subset with exact actor-plus-physics evaluations.

A `50 x 50` critic grid is only 2,500 small network evaluations, so it is acceptable for diagnostics. It is not acceptable to run the expensive actor and physical-cost oracle at every grid point during normal training.

### Directional runtime path

Normal training uses directional searches from the generous anchor:

1. Draw a direction from a mixture of left-axis, right-axis, balanced, and random positive directions.
2. Evaluate progressively shorter pairs along `T(alpha)`.
3. Locate the first predicted exit from the anchor-connected manageable component.
4. Refine locally to adjacent unique integer-knot pairs.
5. Select a point just inside, on, or just outside the predicted boundary.
6. Run the exact actor and physical-cost oracle before deciding how to use it.

The search must be sequential rather than assuming global monotonicity. The actor can behave non-monotonically, so the relevant boundary is the first failure reached from the verified generous anchor.

Each exact directional search should retain an interval:

```text
[alpha_pass, alpha_fail)
```

where the first endpoint is exactly manageable and the second is exactly unmanageable.

For stochastic frontier coverage, sample directions with an initial mixture such as 80% random and 10% for each axis. Uniform angles do not produce uniform arc-length samples; if coverage becomes visibly uneven, estimate the contour at about 32 directions and sample directions using arc-length weights while deduplicating integer timing pairs.

## Exact verification barrier

No critic-only result may enter actor training. Every proposed pair crosses an exact verification barrier under `no_grad`:

- **Manageable:** admit to actor training and train the critic on its exact residual vector.
- **Moderately infeasible but repairable:** send to an isolated active-learning/repair batch with conservative caps, and train the critic.
- **Severely infeasible:** train only the critic, exclude it from actor and line-search batches, and back off toward the anchor.

This makes false positives consume some evaluation compute without destabilizing actor training. False negatives are countered by maintaining critic-independent random exploration and long-anchor probes; those streams must never decay to zero.

At cold start, the critic operates only in shadow mode. Boundary acquisition uses the existing actor and exact oracle:

1. Evaluate a generous anchor pair.
2. If it passes, shorten along a sampled direction in coarse steps until the first failure.
3. Refine the resulting bracket to adjacent unique timing knots.
4. If the anchor fails, try a longer fallback, a teacher/MPC initializer, or defer the scenario.

The critic does not control sampling until calibration gates have been met.

## Optional direct boundary model

After the violation critic is useful, add an optional proposal model that predicts a boundary location directly:

```text
(scenario, anchor A, direction d) -> alpha_star
```

This is the appropriate VBOC-like formulation for this problem. Direction is required because the Pareto frontier contains many valid time pairs; a scenario-only model cannot identify which tradeoff the caller wants.

Train it from exact interval supervision rather than an arbitrary point target:

```text
loss = relu(alpha_pass - alpha_hat)^2
     + relu(alpha_hat - alpha_fail)^2
```

An ensemble of five boundary heads can again supply uncertainty. Since larger `alpha` is more aggressive, use a conservative lower quantile when proposing a time. The boundary model is an acceleration layer; the violation critic remains the diagnostic model and exact verification remains mandatory.

## Exploration sources

The curriculum should mix these scenario sources:

- existing collision-screened Phase 2 box, far-fetched, and static scenarios;
- roller, MPC, and post-hit states;
- verified bounce or post-contact states paired with a new random goal;
- critic-independent random state/goal samples; and
- audit scenarios retained because they exposed regressions or rare behavior.

For a random starting state and goal pair:

1. Construct goal/IK terminal data independently of timing.
2. Choose a generous atomic timing anchor and a direction preference.
3. Retime the canonical initializer for every tested candidate pair.
4. Rebuild hit indices, masks, goal time-to-hit fields, strike/terminal windows, and the actor initializer consistently.
5. Evaluate the actor and exact residuals.

Extract the existing voxel retiming logic near `train_cost.py`'s Phase 2 voxel sampling into one reusable, tested helper. `Phase2Pool.masks_for` should remain the canonical source for time-dependent masks.

MPC and Bellman rollouts remain especially valuable: their suffixes provide dense examples at shorter times. Their labels share the same minimal `(q0, qd0, goal, TTH)` representation; initializer route is intentionally marginalized out.

## Continual learning and bounded memory

The measured target is actor-conditioned and therefore changes as the actor changes. Historical scenario inputs remain useful, but their old violation labels gradually become stale. Use three bounded mechanisms:

### Recent replay ring

Store compact critic inputs, exact targets, actor step/version, source, and sampling stratum. This buffer supplies most critic minibatches and is overwritten FIFO within each stratum.

### Audit coreset

Keep a much smaller raw scenario coreset selected for:

- previous success followed by regression;
- high prediction error;
- high ensemble disagreement;
- rare source or geometry strata;
- feature-space diversity; and
- important boundary brackets.

Periodically rerun the current actor and exact oracle on this coreset to refresh its labels. Store full trajectories only for a tiny witness/repair subset; most entries should contain compact scenario data from which initialization can be rebuilt.

### Procedural resampling

Continuously draw broad fresh scenarios independently of the critic. This is the primary defense against slow self-confirming drift: a forgotten region can still be rediscovered even if the critic currently rejects it.

Prioritize replay with a mixture of prediction error, head disagreement, stratum rarity, novelty, and detected regression. Actor version should be an age/weighting signal, not a claim that stale labels remain correct.

## Proposed code organization

### New modules

`amortized_opt/violation_critic.py`

- `ViolationCriticConfig`
- feature construction and detached actor-context adapter
- physical-unit target definitions and robust reversible transforms
- five-head ensemble and EMA handling
- bootstrap loss and optimizer step
- recent replay ring and audit coreset
- prediction, uncertainty, and calibration utilities

`amortized_opt/timing_curriculum.py`

- `TimingPair` and active/idle-hand invariants
- generous-anchor selection and fallback policy
- two-dimensional grid and connected-component frontier extraction
- directional search and bracket refinement
- stochastic frontier/collar sampling
- reusable pair-retiming helper
- exact-verification classification and actor-batch admission
- optional direct boundary ensemble

### Existing files to modify

`amortized_opt/model.py`

- keep critic parameters outside the actor model and optimizer.

`amortized_opt/train_cost.py`

- extract exact residual targets from the existing `cost_tuple`/gate path;
- update the critic after actor line-search side effects are complete;
- insert shadow, assist, and control curriculum modes;
- enforce the exact verification barrier;
- save critic state under checkpoint `extra`; and
- emit calibration and frontier metrics.

`amortized_opt/phase2_data.py`

- expose canonical timing-mask construction and compact scenario reconstruction;
- add bounded scenario references suitable for audit relabeling; and
- ensure an atomic pair is preserved through batching.

`scripts/train_phase2.py` and `scripts/_run_rhythm_bellman.sh`

- add configuration and command-line controls;
- default the new system to `off` initially, then `shadow` for experiments; and
- retain the current voxel defaults for reproducible A/B runs.

`amortized_opt/__init__.py`

- export only the public configuration and curriculum interfaces.

### Configuration

Add a nested `TimingCriticConfig` to `Phase2Config` rather than adding many unrelated flat fields. Update `phase2_config_from_checkpoint` to reconstruct it explicitly, matching the repository's current nested-config checkpoint behavior.

Minimum configuration fields:

```text
mode: off | shadow | assist | control
num_heads: 5
ema_decay
target_scales[4]
terminal_limit
collision_sdf_limit
joint_limit
torque_limit
recent_capacity
audit_capacity
critic_updates_per_exact_eval
bootstrap_probability or poisson_rate
anchor_pairs and fallback policy
coarse_step_knots
refine_step_knots
direction_mixture
collar_offsets
moderate_failure_caps
independent_probe_fraction
assist_fraction
```

Use four critic replay-gradient steps per outer actor step by default as the
first online increase to test. These steps reuse stored features and exact
labels; they do not run the actor or physics again. Effective reuse depends on
label influx: with a 256-row critic minibatch, four updates give about 12.8
presentations per label for the 64+16 diagnostic actor batch, 3.2 for an actor
batch of 256 plus 64 suffixes, and 1.6 for the default actor batch of 512 plus
128 suffixes. A frozen-final-replay forward-holdout sweep—not an exact online
equivalent—showed that increasing total critic updates from 1,001 to 4,001
improved manageable precision from 32.8% to 40.7%, recall from 57.4% to 59.6%,
and reduced the overall false-manageable rate from 23.3% to 17.2%.

Checkpoint critic state separately in `extra["timing_critic"]` with a format version, live and EMA weights, critic optimizer, normalization state, replay metadata, audit coreset, RNG state, mode, and trust/calibration state. A legacy checkpoint without this key must resume with a freshly initialized critic. The actor model state remains strict and unchanged.

Use a separate RNG stream and optimizer for the critic so critic updates do not perturb actor batch order, line-search reproducibility, or actor optimizer state.

## Rollout phases

### Phase 0: Contracts and target extraction

Implement configuration, `TimingPair`, exact target extraction, robust transforms, and the reusable retiming helper.

Definition of done:

- all time-dependent fields are rebuilt consistently from an atomic pair;
- target signs and physical thresholds exactly match the current gate;
- idle-hand behavior is covered; and
- no existing training behavior changes when mode is `off`.

### Phase 1: Shadow critic

Train the five critic heads on exact evaluations already produced by Phase 2. Add recent replay, audit coreset plumbing, EMA heads, checkpointing, and metrics. Predictions are logged but ignored by the sampler and actor.

Definition of done:

- legacy and new checkpoints round-trip;
- critic gradients cannot reach actor parameters;
- all exact samples, including severe failures, train the critic; and
- actor results are bitwise or tolerance-equivalent to `off` under a fixed actor RNG stream.

### Phase 2: Exact frontier calibration

Add an offline/shadow evaluator that selects representative scenarios, calculates exact two-dimensional timing grids or dense directional brackets, and compares them with conservative critic frontiers.

Definition of done:

- reports include Pareto-contour distance rather than only scalar time error;
- results are stratified by source, geometry, timing direction, and initializer;
- false-manageable calibration is measured; and
- non-monotonic and disconnected predicted regions are visible.

### Phase 3: Assist mode

Use the critic to propose timing-boundary samples, but exact-verify all proposals. Initially replace only the existing 5% voxel-unknown probe stream. Keep current voxel, MPC, Bellman, and random exploration active.

Definition of done:

- severe false positives never enter actor or active-learning batches;
- verified boundary proposals produce more useful repair/manageable samples per exact evaluation than voxel probes;
- no material regression appears in retained audit tasks; and
- assist fraction can increase to 20–30% without destabilizing training.

### Phase 4: Broad random and drum-to-drum exploration

Draw broad random starting states/goals, attach long anchors and directional timing searches, and incorporate verified post-hit states for drum-to-drum continuation tasks.

Definition of done:

- anchor success and fallback rates are monitored;
- every candidate is consistently retimed before evaluation;
- critic-independent broad probing remains nonzero; and
- memory remains bounded over long training runs.

### Phase 5: Optional direct boundary proposal model

Train the directional `alpha_star` ensemble from exact pass/fail brackets and use it to seed critic searches. Continue exact verification and violation-critic supervision.

Definition of done:

- proposal search requires fewer critic evaluations without increasing exact false-manageable rate;
- uncertainty is conservative in sparse strata; and
- disabling the boundary model cleanly falls back to critic ray search.

### Phase 6: Control mode and voxel retirement

After sustained A/B evidence, stop creating new spatial voxel probes in `control` mode. Preserve legacy voxel code and checkpoint compatibility for rollback until several full training runs confirm the replacement.

Definition of done:

- coverage and proficiency meet or exceed the voxel baseline;
- exact evaluations per accepted useful sample improve materially;
- peak and steady-state memory are bounded and lower than the voxel system; and
- rollback to `assist` or the legacy voxel curriculum requires only a configuration change.

## Tests

Add `tests/test_violation_critic.py` for:

- target sign, threshold, transform, and inverse-transform correctness;
- output shapes and finite loss on extreme violations;
- independent bootstrap behavior and head diversity;
- EMA update and conservative quantile selection;
- absence of gradients in actor parameters; and
- replay capacity, stratification, aging, and audit replacement.

Add `tests/test_timing_curriculum.py` for:

- atomic timing-pair semantics and idle-hand specialization;
- a case where independently feasible hand times are jointly infeasible;
- connected-component and Pareto-frontier extraction on nonrectangular regions;
- first-exit behavior on a non-monotonic ray;
- exact bracket refinement and no-feasible-anchor fallback;
- 4-of-5/quantile conservative selection;
- retiming consistency across hits, masks, goal fields, terminal/strike windows, and actor initialization; and
- exclusion of severe false positives from actor and repair batches.

Add integration coverage for:

- critic checkpoint round-trip and explicit fresh initialization when a legacy
  feature schema is encountered;
- deterministic actor RNG behavior between `off` and `shadow`;
- a short shadow run that trains all heads;
- a short assist run that exact-verifies every proposal; and
- reporting predicted versus exact two-dimensional frontiers.

## Metrics and promotion gates

Log at least:

- per-residual physical-unit MAE and normalized MAE;
- head disagreement and empirical quantile coverage;
- false-manageable and false-unmanageable rates;
- exact verification acceptance, repair, and severe-rejection rates;
- predicted-to-exact Pareto-contour distance;
- coverage split across left-heavy, balanced, and right-heavy directions;
- anchor failure and fallback rates;
- selected timing-pair and collar-offset histograms;
- replay/audit age and source distributions;
- retained-proficiency regression rate; and
- critic, actor, and physical-oracle compute time.

Initial promotion gates from `shadow` to `assist` can be:

- false-manageable rate below 2% on the audit frontier set;
- empirical coverage of the conservative 0.8 quantile at or above 80%;
- median Pareto-contour distance no greater than two timing knots; and
- minimum sample counts met for every important scenario/initializer stratum.

These are starting criteria, not permanent constants. Tune them from shadow data, and require exact verification even after promotion.

## Recommended first implementation slice

The first reviewable change should include only:

1. `TimingCriticConfig` and checkpoint-compatible `off|shadow` modes.
2. Exact four-residual extraction with unit-tested robust transforms.
3. A detached five-head critic ensemble with EMA copies.
4. A bounded recent replay ring and small audit coreset format.
5. Shadow training on evaluations the existing Phase 2 loop already performs.
6. Calibration logging, with no critic influence on actor batches.

This slice validates whether violations are predictable from the available features before changing exploration. The next change can extract pair retiming and add exact directional frontier acquisition, followed by the guarded 5% assist experiment.

## Non-goals for the first version

- Do not attempt to learn an analytic PDE solution or assume the feasible set has a closed form.
- Do not make the critic the authority on physical feasibility.
- Do not train a single scalar minimum-time target for a two-hand task.
- Do not retain every collocation point or full trajectory.
- Do not remove the current voxel path before controlled A/B evidence exists.
- Do not assume feasibility is globally monotonic in either time coordinate.
