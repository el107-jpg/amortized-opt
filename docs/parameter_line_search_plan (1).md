# Implementation plan: SaLSa-style learning-rate selection with curvature-informed trials

Implement a configurable learning-rate selector for the existing transformer training loop. The model produces humanoid drumming trajectories, and the objective includes augmented-Lagrangian constraints, torque and joint limits, collision avoidance, drum targets, smoothness, and existing auxiliary terms. Preserve the trajectory representation and physics calculations.

Revision: adopt downhill-probe learning-rate transfer, smoothed acceptance, and adaptive searches every 1–10 optimizer updates with early-search triggers. Retain finite-step curvature interpolation and learning-rate expansion. Disable cosine scheduling entirely in line-search mode; the selector is the sole owner of the effective learning rate. Define the acceptance EMA timescale in training updates rather than search calls. Require all five efficiency measures: an ordinary optimizer path between searches, cached search quantities, reusable device buffers, consolidated GPU acceptance work, and static physics caching. This is a hybrid specification, not an exact reproduction of the SaLSa package.

No repository has been inspected while preparing this plan. The implementing agent must locate actual modules and follow repository instructions. Numeric defaults are initial engineering choices to benchmark.

## 1. Intended behavior and changed contract

Each training iteration has two different parameter-space directions:

- Probe direction: the negative, diagonally preconditioned gradient of the current scalar objective. This direction is downhill when that gradient is nonzero.
- Executed direction: the normal staged Adam/AdamW update, including its existing momentum, gradient transformations, and configured decay.

On a search update, select a scalar learning rate using forward evaluations along the probe direction and a smoothed sufficient-decrease test. Then restore the baseline parameters and apply that same scalar learning rate to the executed direction. Between searches, reuse the last selected rate for ordinary momentum updates on fresh minibatches, with no extra probe forwards. The learning rate remains outside the Adam moment recurrences.

The default mode does NOT require the executed momentum direction to be downhill and does NOT line-search it again. Remove the previous default fallback that bypassed momentum when its slope was nonnegative. Requiring a final loss-decrease check on the momentum update would undo the user's requested transfer design.

Smoothing can allow a probe whose current loss increases, transferring the rate can produce an executed update whose current loss increases, and skipped searches reuse a rate that may become stale. These are permitted outcomes. This method does not guarantee monotonic batch loss, feasibility, or accurate torque linearization throughout a step. A scalar rate does not remove directional anisotropy.

Retain an explicitly separate `strict_actual_update` mode only as an optional ablation if useful; it must not be the default or silently activate after a transferred update increases loss. Nonfinite parameters remain an invalid numerical result, not an accepted optimization tradeoff.

## 2. Inspect the repository

Locate the training step, complete scalar objective, reduction convention, available per-term gradients, PCGrad/clipping, Adam variant and parameter groups, weight decay, schedulers, multiplier/penalty/weight update cadence, MPC caches, checkpointing, and logging.

Check gradient accumulation, distributed execution, AMP/scalers, master parameters, dropout/random sampling, mutable model buffers, and physics routines that internally require autodiff. Identify any custom backward or stop-gradient rule that means the raw recorded gradient differs from the derivative of the tested scalar function.

Expose a replayable objective evaluator and an optimizer adapter that stages proposals without committing state. Reuse existing optimizer code where possible. Do not silently approximate unsupported optimizer variants.

## 3. Definitions and the two directions

Freeze batch B and external objective controls C for one search:

    F(theta) = complete_training_objective(model(theta, B), C)
    theta0   = parameters before this training update
    F0       = F(theta0)
    g        = true gradient of F at theta0, before PCGrad or clipping
    q        = gradient supplied to the existing optimizer after its transformations

On search updates, preserve g before destructive gradient transformations, or recover its exact sum from existing weighted group gradients. Decide whether to search before overwriting g wherever possible. If a trigger depends on transformed q, preserve the information needed to recover g until that decision. Avoid an unconditional full-gradient clone or second backward on ordinary updates. Unscale AMP gradients before these operations.

On search updates, stage the next Adam moments once from q and the pre-search optimizer state. Ordinary reuse updates use the existing optimizer directly as specified in Section 9. For a standard Adam group:

    m_next = beta1*m_old + (1-beta1)*q
    v_next = beta2*v_old + (1-beta2)*q^2
    m_hat, v_hat = bias-corrected staged moments

Use the configured optimizer's actual denominator, including enabled variants. Let eta_ref be a positive reference group learning rate, and r_group = lr_group/eta_ref. Fold this fixed ratio into a diagonal metric:

    D_j = r_group / (sqrt(v_hat_j) + adam_eps)

For the basic Adam/AdamW adapter:

    p_probe = -D*g
    p_momentum = -D*m_hat
    p_decay_j = -r_group*weight_decay_group*theta0_j  # AdamW decay parameters only
    p_exec = p_momentum + p_decay

No momentum or decoupled decay is included in p_probe. D may come from transformed-gradient statistics; its positivity still ensures descent when applied to the true g. Preserve inactive, frozen, unused-parameter, and per-group semantics. More complex optimizer variants require an adapter matching their actual displacement.

Compute:

    a        = -dot(g, p_probe) = dot(g, D*g)
    s_probe  = -a
    P2       = dot(p_probe, p_probe)
    s_exec   = dot(g, p_exec)                 # diagnostic only

On search updates, require finite a > 0 and P2 > 0 over the parameters eligible to move. A negative, zero, or positive s_exec does not decide rate selection or block an otherwise valid transferred step. If no usable probe exists, skip with a clear zero-gradient/invalid-direction status; do not divide by zero or claim that a finite rate was selected.

For any epsilon:

    theta_probe = theta0 + epsilon*p_probe
    theta_exec  = theta0 + epsilon*p_exec

At epsilon = eta_ref, theta_exec must reproduce the existing optimizer proposal, including AdamW decay exactly once. These points generally differ. Never apply the momentum update on top of theta_probe.

## 4. Smoothed acceptance statistics

Use distinct names for the scalar acceptance statistics and Adam's vector momentum. Persist:

    A_prev = EMA of selected probe loss decreases
    B_prev = EMA of positive probe slope magnitudes a
    n_history = number of committed search samples
    last_history_update = optimizer-update index of the last committed sample
    retention_product = product of effective retention factors since reset

Initialize A_prev = B_prev = 0, last_history_update = None, and retention_product = 1. Set beta_accept_per_update = 0.99 initially. Let u count successfully executed optimizer updates and j = u+1 be the proposed update index. At a search, compute:

    gap = 1 if last_history_update is None else j-last_history_update
    beta_effective = beta_accept_per_update ** gap
    B_candidate = beta_effective*B_prev + (1-beta_effective)*a

Reuse this same beta_effective throughout that search. At an interval of 10, beta_effective is approximately 0.904, rather than 0.99 per search. This gives old samples an approximately 100-update decay timescale while still observing only the searched minibatches. It does not reconstruct the missing probe results or equal an EMA of every unobserved update. This is a deliberate adaptation of SaLSa's smoothing.

For each candidate epsilon, evaluate the SAME frozen batch objective at theta_probe and calculate:

    Fprobe  = F(theta0 + epsilon*p_probe)
    decrease = F0 - Fprobe
    A_candidate(epsilon) = beta_effective*A_prev + (1-beta_effective)*decrease

The selection criterion is:

    A_candidate(epsilon) >= c1*epsilon*B_candidate

Use c1 = 0.5 initially, preserving the prior plan's setting; beta_accept_per_update and c1 are configurable. An ablation may compare c1 = 0.3. Do not silently relax them on failure.

Every retry must form A_candidate from the same A_prev. Rejected trials do not become EMA samples. B_candidate is formed once from B_prev, not repeatedly smoothed during backtracking. Commit both scalar statistics exactly once only after a searched update is successfully executed. On that commit set last_history_update = j, increment n_history, and multiply retention_product by beta_effective. Ordinary updates between searches advance Adam state but do not change either acceptance EMA, its sample count, or retention_product; the next search accounts for elapsed updates through gap.

With zero initial history, the common (1-beta_effective) factor cancels, so the first criterion reduces to ordinary sufficient decrease along p_probe. With beta_accept_per_update = 0, every searched criterion is unsmoothed. Raw EMAs are sufficient because both sides have matching initialization and sampling. If reporting bias-corrected EMAs, divide both by 1-retention_product; using 1-beta_accept_per_update**n_history is wrong for variable intervals.

The history contains selected PROBE decreases, not changes measured along the executed momentum direction and not differences between unrelated minibatch baseline losses. Do not smooth the full residual `decrease - c1*epsilon*a` instead: that is a different rule from smoothing decrease and slope separately.

Positive history can admit a negative current decrease. This is intended. There is no additional Fprobe <= F0 requirement by default.

## 5. Retain the finite-step curvature proposal

Fit curvature using the CURRENT unsmoothed probe evaluation and its CURRENT true slope:

    R = Fprobe - F0 - epsilon*s_probe
      = epsilon*a - decrease
    kappa = 2*R / epsilon^2
    L_measured = max(0, kappa/P2)

When R is positive, finite, and numerically resolved:

    epsilon_quad = -s_probe*epsilon^2 / (2*R)
                 = a*epsilon^2 / (2*R)

This uses two loss values plus the starting derivative. No additional half-step probe is necessary. A half-step experiment must use its actual displacement (epsilon/2)*p_probe consistently in every formula.

Use epsilon_quad only to propose another trial after failure. It minimizes a fitted CURRENT probe-loss quadratic; it does not exactly solve the history-dependent acceptance inequality, does not certify a local Lipschitz bound, and is not a curvature estimate for p_exec. Never substitute A_candidate or B_candidate for R or a in this fit.

For rejected finite trials, clamp a valid epsilon_quad to [0.1*epsilon, 0.5*epsilon]. If interpolation is invalid, halve epsilon. Nonfinite Fprobe also triggers halving. Always evaluate the new probe and apply the smoothed test. Growth remains available at the next scheduled or triggered search, never on an unchecked update.

This curvature-informed proposal is the retained custom addition; SaLSa's standard multiplicative shrinking remains an optional `proposal_rule=geometric` ablation.

## 6. Search control and initial settings

| Configuration | Initial choice |
| --- | --- |
| mode | salsa_transfer |
| acceptance | smoothed_probe |
| beta_accept_per_update | 0.99 |
| c1 | 0.5 |
| proposal_rule | quadratic_with_safeguards |
| expansion_factor | 1.1 in adaptive-interval mode; configurable |
| backtrack_min_factor / max_factor | 0.1 / 0.5 |
| fallback_shrink | 0.5 |
| max_trials | 6, excluding baseline |
| epsilon_seed | Explicit constant; default to the configured initial reference learning rate, never cfg.lr_at(step) |
| lr_scheduler | Disabled in line-search mode; no cosine multiplier, cap, or scheduled seed |
| epsilon_max | Initially 10 times epsilon_seed, configurable |
| epsilon_min | Initially 1e-6 times epsilon_seed, configurable |
| search_interval_mode | adaptive; fixed=1 available for profiling/ablation |
| initial_interval / min_interval / max_interval | 1 / 1 / 10 |
| warmup_updates | 10 successfully executed updates, searching each time |
| stable_searches_to_increase | 3 consecutive stable searches |
| stable_rate_ratio_bound | 1.15 |
| stable_max_trials | 2 |
| substantial_rate_drop_ratio | 0.75 |
| costly_search_trials | 3 or more |
| actual_update_diagnostic_every | Configurable, off by default outside debugging |
| grouped_scaling | Off |

First search: use epsilon_seed. At a normal later search, start at min(expansion_factor*last_selected_epsilon, epsilon_max). Between searches execute exactly last_selected_epsilon, with no growth, interpolation, or EMA updates. The adaptive default reduces the previous 1.5 growth factor to 1.1 so deliberate trial expansion does not constantly defeat the rate-stability check. After budget exhaustion start the next search from the pending smaller retry_seed, without expansion. Keep last_selected_epsilon and retry_seed separate. Early searches triggered by a problem start at the retained rate without expansion; objective-convention changes use a reset seed. Resume normal expansion after recovery.

Every trial starts from theta0 with the same controls and staged moments. Stop without an update when trials are exhausted, the next rate falls below epsilon_min, or no effective parameter displacement can be represented. No forced acceptance of an untested epsilon_min.

Smoothed history need not admit a sufficiently small step: for example, beta_effective*A_prev < 0 persists as epsilon approaches zero. On a complete selection failure, discard tentative statistics and schedule an explicit history reset for the NEXT iteration. Log this reset; do not erase history mid-search simply to make the current test pass.

### Adaptive interval controller

Persist successful_update_count u, last_search_update, last_history_update, interval K, stable_search_streak, last_selected_epsilon, retry_seed, and pending trigger/reset reasons. The scheduling and EMA clocks are distinct even if they usually coincide.

At the beginning of proposed update j = u+1, search if there has never been a successful search, a retry/reset/early trigger is pending, the warmup is not finished, or j-last_search_update >= K. This means K=1 searches every update; K=10 executes nine ordinary updates between searched updates. A failed or overflow-skipped update does not advance u, and keeps a search pending for the next usable batch. Each ordinary successful update uses a new minibatch and advances Adam moments once.

After a successful searched AND executed update, compare epsilon_new with the previous committed selected rate, not the expanded initial trial:

    ratio = epsilon_new / epsilon_previous
    stable = max(ratio, 1/ratio) <= 1.15 and trial_count <= 2

When warmup is complete and no recovery/reset trigger applies, increment the stable streak on stable searches. After three such searches set K = min(K+1, 10) and clear the streak. A rate drop to 0.75 or less of its prior value, three or more trials, selection exhaustion, or a numerical failure sets K=1 and clears the streak. For other nonstable successful searches, shorten K to max(1, floor(K/2)) and clear the streak. The first search after initialization or an objective reset does not establish stability. During warmup keep K=1 and do not accumulate a promotion streak.

This controller is a transparent application heuristic rather than SaLSa's exact fast/slow learning-rate-EMA ratio formula. Stability of selected rates is evidence for reuse, not proof of a safe future rate.

### Cheap early-search triggers

Check signals from the CURRENT baseline forward/backward before applying the current update, comparing with reference statistics captured before incorporating the current batch:

- Explicit multiplier/penalty changes, objective redefinition, or changed parameter-group ratios force a search and K=1. Apply the paired history-reset policy in Section 8.
- A marked rise in normalized torque/collision/joint-limit violation metrics forces a search and K=1. An initial configurable rule is more than 2 times its reference AND a meaningful absolute increase in normalized violation units. Set the absolute threshold from the existing constraint normalization/tolerances; do not invent one raw-unit threshold for all constraints.
- A gradient norm more than 3 times its moving reference, with a configured nonzero reference floor, forces a search and K=1. Use raw g if already available; otherwise name and consistently monitor the existing optimizer-input q. This is a change detector, not a curvature estimate.
- Nonfinite baseline losses/gradients use the existing invalid-batch handler and schedule a search at the next valid update. Search does not repair an invalid baseline.

Ordinary minibatch variation can trigger measurements, but does not automatically imply divergence. Finite metric spikes alone do not reset acceptance history or veto execution; objective-definition changes and exhausted searches follow their explicit reset policy. Reference EMAs update once per valid logical baseline, not per trial; document initialization and use only reliable initialized references. Use a common trigger decision across distributed ranks.

On an update already committed, a later diagnostic can trigger a future search but cannot retroactively protect the past step. If controllers change after commit, latch a trigger before the next update. If objective controls change every update, search may remain at K=1; report that honestly rather than suppressing triggers or changing controller cadence without an explicit algorithm decision.

## 7. State transaction, momentum transfer, and decay

Provide a clear interface:

    stage_optimizer(q, optimizer0) -> staged_state, D, p_exec
    build_probe(g, D) -> p_probe, a
    select_rate(theta0, p_probe, frozen_context, EMA_snapshot) -> epsilon, probe_stats
    execute(theta0, epsilon, p_exec)
    commit(staged_state, accepted_EMA, search_state)

Moment estimates and optimizer counters advance once per executed update, never per trial. Acceptance EMAs advance only on successfully executed search updates. On ordinary reuse updates, use the existing compatible optimizer directly with the selected scalar rate and fixed group ratios; verify equivalence with the staged search adapter. Reserve the staged path for searches and explicitly enabled transactional diagnostics. After rate selection, discard probe parameter/buffer mutations, construct theta_exec from theta0, check parameter finiteness, then commit exactly that displacement and the staged optimizer state once. Do not run optimizer.step again and accidentally create a second moment update.

Keep Adam's standard gradient-history recurrence. Do not insert epsilon into m_next or v_next, and do not add LaProp or accepted-displacement momentum as part of this revision.

For AdamW, include decay in p_exec, scale it by the selected epsilon, and apply it once. Do not decay parameters before calculating F0 or before probes. The probe selection intentionally does not account for decay or guarantee its effect on F. Non-descent s_exec is logged, not a reason to remove decay or reset momentum in default mode.

A finite executed update is not reevaluated for acceptance by default. Optional same-batch evaluation of theta_exec is diagnostic only; a finite increase is logged and retained. On diagnostic updates, prepare the required transaction before execution. If the diagnostic finds a nonfinite objective, use the existing invalid-update handler, roll back that transaction, and do not commit probe EMAs. Ordinary fast-path updates retain native pre-step AMP/gradient checks and the existing numerical recovery policy; exact post-step rollback requires an explicitly enabled snapshot or staged diagnostic path and its measured cost. Never continue training with known corrupted parameters or optimizer state. Clearly separate numerical failure from permitted nonmonotonicity.

### Learning-rate ownership: no cosine in line-search mode

Disable cosine scheduling entirely while line search is enabled. Bypass the existing per-step `lr = cfg.lr_at(step)` assignment and any scheduler callbacks that could write optimizer group rates. Cosine must not supply a multiplier, cap, search seed, or recovery rate. Keep the existing scheduler behavior only when the line-search feature is disabled.

Initialize epsilon_seed from an explicit constant, defaulting to the configured initial reference learning rate. epsilon_min and epsilon_max are fixed configurable bounds, not functions of training step. Capture parameter-group ratios from the configured initial rates, not the current cosine-scheduled rates. All probe and execution directions exclude the scalar learning rate; apply the selected epsilon exactly once. On ordinary reuse updates set each group rate to `last_selected_epsilon * fixed_group_ratio` and call the native optimizer once. On staged search updates commit the selected displacement and moments once, without another optimizer step.

Checkpoint and restore the selected rate, fixed group ratios, bounds, and selector state without letting a restored scheduler overwrite them. When enabling this feature from a legacy checkpoint, initialize selector state explicitly from its configured constant seed and record the mode transition; do not infer a new seed from the checkpoint's cosine-reduced group rates. An explicit user-provided seed remains allowed. Reset warm-start conventions and force a search if group ratios or parameterization deliberately change. Search-interval scheduling remains active and is separate from the disabled learning-rate scheduler.

## 8. Frozen evaluations, buffers, caches, and history resets

Freeze batch, targets, trajectory initialization, stochastic samples, dropout draws, detached weights, multipliers, and penalty coefficients within each search. Recalculate parameter-dependent masks, torques, collisions, and residuals at every probe. Preserve the baseline model mode and replay the same randomness; rejected probes must not consume future training RNG draws.

All probes start from the same pre-forward mutable-buffer snapshot. Discard their buffer/cache writes. By default retain the baseline forward's single logical buffer update, as in ordinary training; do not commit buffers from a probe whose parameters will not be executed. Diagnostic executed-point forwards must be isolated too. If the codebase requires a fresh output/cache at theta_exec, evaluate that point explicitly and document its extra cost. Never populate the MPC cache or multiplier controller with a surrogate probe trajectory.

Trial evaluation does not require a parameter backward pass. Permit local autodiff inside physics routines where needed; blanket no_grad around such a routine may break its computation.

Keep controller updates at the existing outer-loop cadence, outside the search. Use baseline or actual executed metrics according to the existing controller contract, never selected-probe metrics by accident. Do not advance update-dependent controllers on skipped updates.

EMA history should survive ordinary minibatch changes: that is the purpose of smoothing. Reset both A and B together, n_history, last_history_update=None, and retention_product=1 when the scalar objective's definition changes discontinuously: explicit augmented-Lagrangian multiplier/penalty updates, manual loss-weight changes, loss-normalization changes, or changed direction conventions. Record the reason.

If detached dynamic weights change every iteration, blanket resets would effectively disable smoothing. Detect and report this during integration. Define an explicit policy before enabling: preferably keep acceptance-objective controls fixed over documented blocks if compatible with the current algorithm; otherwise expose a deliberate policy for smoothing across gradual weight changes and label its history approximate. Do not silently change the user's constraint controller to make the optimizer convenient. Fresh minibatch residual-dependent masks are not objective-definition resets.

## 9. Required computational-efficiency implementation

Implement all five measures below. They combine the chosen SaLSa-style algorithm with workload-specific engineering; do not attribute all five to the published implementation. Preserve the acceptance and optimizer semantics above.

### 1. Ordinary fast path between searches

On an update that reuses the selected rate, run the existing baseline forward/backward, cheap trigger checks, and the existing optimizer step. Preserve its supported fused/foreach path. Do not build p_probe, D, or a separate p_exec; stage duplicate moments; clone the full parameter/optimizer state; or run a trial evaluator on this branch. Set group learning rates from the retained scalar and fixed ratios without changing the moment recurrence. Release the baseline autograd graph once all required gradient work finishes.

Triggers can arise after the baseline backward, so distinguish cheap replay preparation from a full transaction. Before the baseline forward, retain the RNG state and only the mutable buffers needed to replay that forward, using reusable storage or a verified stateless-buffer approach. Some replay bookkeeping is unavoidable when a late trigger must search the same batch. Preserve the baseline's logical buffer result separately when a search is actually needed. Allocate/copy parameter snapshots and stage optimizer proposals only after deciding to search, while parameters and optimizer state still have their pre-step values. Arrange raw-gradient preservation according to Section 3 so late triggers do not accidentally require a second backward.

An ordinary fused optimizer step cannot also promise exact rollback of an arbitrary post-step numerical failure without retaining old state. Keep native overflow handling on the fast path. Route explicitly scheduled transactional diagnostics through the staged path, and document any additional recovery mechanism the codebase requires. Do not hide full-state snapshots inside the normal fast path.

### 2. Cache quantities that are fixed throughout one search

Reuse F0 and the gradient from the baseline training computation. Compute staged moments, bias corrections, D, p_exec, p_probe, a, P2, beta_effective, and B_candidate once per search. Rejected trials change only epsilon, candidate parameters, forward-dependent outputs, and scalar acceptance/interpolation calculations. They do not rerun backward or rebuild optimizer directions. Compute optional execution-direction diagnostics once when enabled.

The slope needed for interpolation is already a; the curvature residual uses the existing trial loss. Do not add an epsilon/2 evaluation, Hessian calculation, or JVP for this selector. Invalidate cached search quantities when the search ends; a new minibatch or update requires new gradients and staged statistics.

### 3. Reuse device storage and reduce parameter-copy overhead

Lazily allocate reusable storage for theta0, staged state, candidates, and necessary mutable buffers. Retain allocations across searches when shapes, devices, dtypes, and parameter membership match; refresh their contents each search. Rebuild storage when those properties change. Avoid model deepcopy, optimizer deepcopy, and fresh full-size allocations inside the trial loop.

Construct every candidate from theta0 + epsilon*p_probe using safe in-place writes into dedicated storage and batched tensor operations where supported. Avoid incremental perturb/undo arithmetic that can accumulate roundoff. Preserve parameter identities, aliases/tied weights, and optimizer references. A functional evaluator or temporarily overwritten live parameters is acceptable after verifying repository compatibility. If live parameters are overwritten, restore them on failure and construct the executed candidate directly from theta0 on success. Reusable buffers reduce allocation overhead, not the bandwidth needed to write candidate parameters; benchmark both memory and time.

### 4. Keep trial arithmetic on device and consolidate synchronization

Keep loss reductions, a, P2, acceptance EMAs, the curvature residual, and interpolation arithmetic on the accelerator in stable precision. Consolidate the accept/reject, finite-value, and next-trial information needed by the host into one small transfer per trial where practical. Avoid separate .item() calls for every loss component, norm, and logging field. Buffer diagnostics and transfer them together at the existing logging cadence. Distributed reductions and decisions must still follow Section 11.

Compile the repeated model/physics objective evaluator separately from the dynamic search controller when the repository supports it. Reuse the compiled evaluator for stable shapes and parameter storage, avoiding recompilation for each epsilon. Do not require compilation of the Python backtracking loop. A host-controlled loop still needs a decision synchronization; this optimization reduces redundant synchronization rather than eliminating it. Keep a correct eager fallback and measure compile warmup separately from steady-state throughput.

### 5. Cache static physics inputs and batch independent evaluations

Cache parameter- and trajectory-independent inputs such as robot topology, fixed inertial data, fixed link transforms, trajectory differentiation operators, fixed targets, and reusable collision geometry. For per-batch fixed inputs, scope the cache to the frozen batch/context. Record invalidation conditions for changes to the robot, time grid, geometry, normalization, targets, or device/dtype. Learned or otherwise trainable quantities cannot be treated as static.

Every candidate must recompute trajectory-dependent kinematics, velocities, accelerations, mass/dynamics terms, torques, distances, residuals, and active masks. A previous candidate's activations or dynamics values are not a valid cache simply because epsilon changed only slightly. Batch/vectorize independent waypoint inverse-dynamics and collision work where supported. Preserve temporal dependencies in rollouts and controllers; do not parallelize a sequential state evolution as if waypoints were independent. Candidate trials remain sequential because each rejection chooses the next epsilon.

## 10. Core pseudocode

```text
# Line-search mode: never call cfg.lr_at(step) or advance an LR scheduler.
# Initialize fixed seed/bounds/group ratios once, or restore selector state.
# Lightweight/minimal replay bookkeeping supports triggers discovered after backward.
replay = prepare_minimal_replay_state_using_reusable_storage()
F0, raw_gradient_info, baseline_metrics = baseline_forward_and_gradient()
if invalid_baseline_or_gradients:
    run_invalid_batch_handler_and_latch_search()
    return

j = successful_update_count + 1
# Inspect raw-gradient triggers before destructive transforms where possible.
# If a trigger uses q, retain/recover raw g from existing gradient information.
q, triggers = transform_gradients_and_decide_triggers_preserving_needed_raw_g(...)
if invalid(q):
    run_invalid_batch_handler_and_latch_search()
    return
run_search = warmup_or_due_or_triggered(j, interval, last_search_update, triggers)
transactional_diagnostic = diagnostic_due(j)

if not run_search and not transactional_diagnostic:
    set_group_rates(last_selected_epsilon, fixed_group_ratios)
    result = existing_optimizer_step_with_native_overflow_handling(q)
    if not result.executed:
        latch_recovery_search_if_required()
        return
    successful_update_count = j
    # No probes, duplicate optimizer staging, or full-state snapshots.
    # A/B/count/retention and the selected rate remain unchanged.
    advance_update_dependent_controllers_once_with_valid_metrics()
    latch_search_if_controls_changed_after_commit()
    return

# Only searches or explicitly enabled transactional diagnostics reach this setup.
theta0, optimizer0, baseline_logical_buffers = refresh_reusable_transaction_storage()
staged = stage_optimizer_once(q, optimizer0)
p_exec = staged.actual_direction_including_decay

if not run_search:
    epsilon = last_selected_epsilon
else:
    g = recover_preserved_true_gradient_without_another_backward(...)
    p_probe = build_once(-staged.D * g)
    a, P2 = cached_direction_reductions(g, p_probe)
    if no_finite_nonzero_downhill_probe(a, P2):
        rollback_and_keep_search_pending()
        return

    A0, B0 = history_after_applying_scheduled_reset()
    gap = 1 if last_history_update is None else j-last_history_update
    b = beta_accept_per_update ** gap
    Bc = b*B0 + (1-b)*a
    epsilon = choose_normal_expansion_or_trigger_seed_or_pending_retry()
    selected = false
    for trial in range(max_trials):
        write_candidate_into_reusable_storage(theta0, epsilon, p_probe)
        Fp = evaluate_replayed_candidate(replay, cached_static_physics_inputs)
        # Device work uses current trial loss and fixed search quantities.
        decision = acceptance_and_safeguarded_interpolation_on_device(
            Fp, F0, epsilon, a, P2, A0, Bc, b, c1, ...)
        accept, stop, next_epsilon = consolidated_host_decision(decision)
        if accept:
            selected = true
            Ac = decision.A_candidate
            break
        if stop:
            break
        epsilon = next_epsilon
    discard_all_probe_mutations()
    if not selected:
        rollback_transaction()
        set_interval_one_and_remember_retry_and_schedule_history_reset()
        return selection_failed

candidate_exec = write_candidate_into_reusable_storage(theta0, epsilon, p_exec)
if not finite_parameters(candidate_exec):
    rollback_transaction_and_latch_recovery_search()
    return
if transactional_diagnostic:
    # Isolated objective evaluation. A finite increase is retained.
    if not finite(evaluate_actual_candidate_without_persistent_side_effects(...)):
        rollback_transaction_and_latch_recovery_search()
        return
commit(candidate_exec, staged.state, baseline_logical_buffers)
successful_update_count = j
if run_search:
    commit_history(Ac, Bc, n_history+1, last_history_update=j,
                   retention_product=retention_product*b)
    update_interval_from_rate_change_trials_and_triggers(...)
    last_search_update = j
    remember_selected_rate(epsilon)
advance_update_dependent_controllers_once_with_valid_metrics()
latch_search_if_controls_changed_after_commit()
```

Exception handling in the transactional branch must restore persistent state, including RNG and mutable buffers according to Section 8. Do not copy optimizer state per trial. On the ordinary branch retain native optimizer/error semantics and use the declared numerical recovery policy.

## 11. Numerical behavior, distributed training, and verification

Use stable precision for loss reductions, a, P2, R, and EMAs; at least float32, with float64 scalar accumulation where practical. Detect cancellation in the curvature residual and fall back to geometric shrinking. Zero observed loss change is not automatically a successful search; apply the specified smoothed criterion. Detect numerical stalls explicitly.

Respect master-parameter updates, AMP unscaling and overflow handling. Update scalers once per logical iteration. For distributed training, use the global objective reduction for baseline/probes and the matching global gradient. Share interval/trigger decisions as well as trial decisions and numerical failures across ranks. Do not sum a replicated gradient dot product a second time and multiply it by world size. Accumulated gradients require reevaluating the corresponding effective batch, not just its last microbatch.

Log F0, selected Fprobe, optional Fexec, all cost/violation groups, epsilon seed/selection, trial count, a, s_exec and separate momentum/decay slopes, direction alignment, displacement norms, current decrease, A_prev/Ac, B_prev/Bc, R/kappa when resolved, growth success, history resets, failure reasons, search-versus-reuse status, K, elapsed updates, trigger reasons, stability streak, beta_effective, and timing. Label probe and executed losses distinctly. Never report Fprobe as the post-update training loss.

Checkpoint the two acceptance EMAs, sample count, retention_product, last_history_update, successful_update_count, last_search_update, interval/streak, selected-rate/retry seeds, metric reference statistics, pending trigger/reset reasons, direction convention, controller identity, and configuration alongside the optimizer. Restore them consistently or explicitly reset incompatible history.

Focused tests:

1. Known quadratic: p_probe has the correct slope and fitted directional curvature. First-step or beta_accept_per_update=0 selection reduces to the unsmoothed probe criterion.
2. Uphill momentum: construct s_exec > 0 and a > 0; verify selection uses p_probe and execution uses p_exec, with no silent descent fallback or veto of a finite increase.
3. Smoothing: create positive past improvement that permits a negative current probe decrease; verify intended acceptance. Rejected trials must not repeatedly advance A or B.
4. Transfer and decay: at the selected rate, the executed update matches the original optimizer proposal; probes are fully restored, decay occurs once, and moments/counters advance once.
5. Expansion, checkpoint continuation, objective-change resets, and negative-history/budget failure behavior.
6. Repeatable dropout/buffers/caches and collective distributed decisions where applicable; nonfinite probe/diagnostic evaluations trigger the documented transactional failure path; ordinary-step overflow/recovery follows its declared native policy.
7. K=10 performs nine ordinary updates between searched updates; reused rates do not expand, no probe closures run, Adam advances normally, and A/B remain unchanged. The next search uses gap=10 and beta_effective=0.99**10.
8. Promotion and demotion boundaries, early triggers before execution, warmup, failure without update-clock advancement, and resume from a checkpoint midway through a reuse interval.
9. Both EMA sides use the same elapsed-update retention and rejected candidates do not compound it; optional bias correction uses the retention product. Objective resets force a new search even if the interval has not elapsed.
10. Scheduler ownership: verify cosine cannot overwrite, multiply, cap, or reseed the selected rate on searched or reuse updates, including checkpoint resume. Compare the same selector state and frozen input at different legacy cosine step indices; proposed and executed rates must match. Verify legacy scheduling still works with line search disabled.
11. Instrument the fast path to verify no trial evaluations, duplicate moment staging, full-state snapshots, or extra backward passes. Verify optimizer equivalence, including supported fused execution and AMP skips. Trigger a search after the baseline backward and verify that replay and raw-gradient recovery remain correct.
12. Across rejected trials, verify fixed quantities are computed once and reusable storage is retained. Compare cached/compiled objective results with the uncached eager evaluator, including changed trajectories and cache-invalidating inputs. Test only applicable modes; compilation is optional, objective correctness is required.

On a representative robotics batch, verify the probe slope with a finite-difference debugging check. Benchmark existing AdamW, this hybrid with fixed K=1, and this hybrid with adaptive K in [1,10] over a bounded run. Compare geometric-only proposals if interpolation cost or retry quality remains uncertain. Optional strict actual-update search is a separate ablation. Compare progress and physical violation metrics against wall-clock time, and measure how well the probe-selected rate transfers. Report average K, fraction of updates searched, trials per search, trigger frequency, probe time, peak device memory, and time spent in replay preparation, transaction setup, candidate writes, optimizer staging, and synchronization. Separate compilation warmup from steady-state timings. Verify that ordinary-update timing stays close to the existing optimizer path and identify any unavoidable replay overhead. Approximate overhead as search_fraction * mean_trials * probe_forward_time / baseline_step_time plus adapter overhead; validate with measured wall time. Do not assume the paper's reported overhead applies to this physics workload. No monotonic-loss assertion belongs in the default integration test.

## 12. Optional per-cost scaling extension

Keep grouped scaling off during this revision's initial rollout. The prior idea remains a later extension: individual group scales require an explicit mapping into a combined momentum update. One shared Adam moment vector cannot simply be split into independent Adam updates.

For that extension, define paired probe/execution directions per group and separate acceptance histories. Individual probes alone do not validate their sum. A combined surrogate probe check may select a final scalar correction; transferring that correction to the grouped momentum direction still does not guarantee actual decrease. Mixing weights can also make the combined surrogate direction non-descent on the total objective even if every individual group probe is downhill on its own cost.

Do not automatically add a mandatory strict actual-update check to the default transfer mode in order to support grouping. Treat grouped combination semantics as a separate feature with its own configuration, tests, and benchmark before enabling. Preserve the repository's current loss weights and gradient transformations in the core.

## 13. Deliverables and source boundaries

Deliver the optimizer proposal adapter, replayable evaluator, smoothed rate selector with optional curvature interpolation, execution/commit integration, configuration, checkpoints/logging, focused tests, all five efficiency measures in Section 9, and a bounded benchmark report. Use actual repository paths in the implementation summary.

Completion requires correct separation of probe and execution paths; upward rate recovery; once-per-executed-update Adam bookkeeping and once-per-successful-search acceptance-EMA bookkeeping; no accidental persistence of probe trajectories; bounded failure handling; correct interval/trigger/checkpoint behavior; and measured overhead and transfer quality. Intermittent reuse is an empirical stability/efficiency tradeoff, not a guarantee for every skipped update. Explicitly document unsupported execution modes.

References:

- [SaLSa paper](https://arxiv.org/html/2407.20650v1): motivation for a preconditioned current-gradient probe, transfer to momentum, smoothed acceptance statistics, and adaptive search-frequency motivation. Our controller thresholds and elapsed-update EMA weighting are explicit custom choices.
- [Current SaLSa source](https://github.com/TheMody/No-learning-rates-needed-Introducing-SALSA-Stable-Armijo-Line-Search-Adaptation/blob/main/src/salsa/SaLSA.py): implementation reference only. Its flags and search/decay behavior are not automatically this specification; in particular always execute the staged momentum direction after selection here.
- [Adaptive Backtracking Line Search](https://arxiv.org/html/2408.13150v2): Armijo/descent-lemma background. Our interpolation safeguards, transactional integration, reset policy, and defaults are application design choices.

- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): implementation guidance for avoiding unnecessary gradient tracking, reducing synchronization, and compiling suitable compute regions.
- [PyTorch optimizer documentation](https://docs.pytorch.org/docs/stable/optim.html): supported optimizer implementations and execution options; preserve repository-compatible fused/foreach behavior where available.
