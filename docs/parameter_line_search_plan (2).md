# Implementation plan: SaLSa-style learning-rate selection with curvature-informed trials

Implement a configurable learning-rate selector for the existing transformer training loop. The model produces humanoid drumming trajectories, and the objective includes augmented-Lagrangian constraints, torque and joint limits, collision avoidance, drum targets, smoothness, and existing auxiliary terms. Preserve the trajectory representation and physics calculations.

Revision: adopt downhill-probe learning-rate transfer, smoothed acceptance, and adaptive searches every 1–10 optimizer updates with early-search triggers. Retain finite-step curvature interpolation and learning-rate expansion. Disable cosine scheduling entirely in line-search mode; the selector is the sole owner of the effective learning rate. Define the acceptance EMA timescale in training updates rather than search calls. Require all five efficiency measures: an ordinary optimizer path between searches, cached search quantities, reusable device buffers, consolidated GPU acceptance work, and static physics caching. This is a hybrid specification, not an exact reproduction of the SaLSa package.

**Hard rule for this revision:** when line search is enabled, **force `grad_balance=""`** (no PCGrad, no GradNorm, no output-space surgery, no surrogate group-EMA path). The training step uses one complete `total.backward()` so the true objective gradient is the Adam input (up to global norm clipping). Do not build a second surgical `q`. Non-line-search runs may keep `al_pcgrad_output`; line-search mode is an explicit recipe fork and must log that surgery is off.

This revision incorporates the user-provided codebase findings for auxiliary objectives, replay, curricula, checkpointing, and the decision to drop PCGrad under line search. The plan editor has not independently inspected the repository. The implementing agent must verify these reported paths against actual modules and follow repository instructions. Numeric defaults are initial engineering choices to benchmark.

## 1. Intended behavior and changed contract

Each training iteration has two different parameter-space directions:

- Probe direction: the negative, diagonally preconditioned **true** gradient of the current scalar objective. This direction is downhill when that gradient is nonzero.
- Executed direction: the staged Adam/AdamW update from that **same** gradient (after the usual clip), including momentum and configured decay.

On a search update, select a scalar learning rate using forward evaluations along the probe direction and a smoothed sufficient-decrease test. Then restore the baseline parameters and apply that same scalar learning rate to the executed direction. Between searches, reuse the last selected rate for ordinary momentum updates on fresh minibatches, with no extra probe forwards. The learning rate remains outside the Adam moment recurrences.

Because PCGrad/GradNorm are off, there is no separate “transformed gradient” family. Clipping may still rescale the vector Adam consumes; see Section 3.

The default mode does NOT require the executed momentum direction to be downhill and does NOT line-search it again. Remove the previous default fallback that bypassed momentum when its slope was nonnegative. Requiring a final loss-decrease check on the momentum update would undo the user's requested transfer design.

Smoothing can allow a probe whose current loss increases, transferring the rate can produce an executed update whose current loss increases, and skipped searches reuse a rate that may become stale. These are permitted outcomes. This method does not guarantee monotonic batch loss, feasibility, or accurate torque linearization throughout a step. A scalar rate does not remove directional anisotropy.

Retain an explicitly separate `strict_actual_update` mode only as an optional ablation if useful; it must not be the default or silently activate after a transferred update increases loss. Nonfinite parameters remain an invalid numerical result, not an accepted optimization tradeoff.

## 2. Inspect the repository

Locate the training step, complete scalar objective, reduction convention, clipping, Adam variant and parameter groups, weight decay, schedulers, multiplier/penalty/weight update cadence, MPC caches, checkpointing, and logging.

The reported live path uses bf16 and explicit `isfinite` skips, without GradScaler. Preserve those handlers; do not introduce scaler bookkeeping. DDP is not an initial integration requirement unless an active distributed path is found. Check gradient accumulation, dropout/random sampling, mutable model buffers, physics-local autodiff, and all stop-gradient rules. Compiled functions are split by bucket (`step_fns[b]`, `tc_fns[b]`, and others); preserve that dispatch structure.

Expose a replayable objective evaluator and an optimizer adapter that stages proposals without committing state. Reuse existing optimizer code where possible. Do not silently approximate unsupported optimizer variants.

When enabling line search, **override or reject** any launcher `GRAD_BALANCE` / `--grad-balance` other than empty. Do not leave `al_pcgrad_output` active “for comparability” inside LS mode. Document the recipe fork in logs and checkpoints (`grad_balance=""`, `line_search=1`).

Implementation priority: first establish exact scalar-objective coverage and replay with plain `total.backward()`; next implement legacy-resume migration; then AL/curriculum history and efficient scheduling. Rate-selection tuning is premature until objective/slope parity passes. Do **not** implement force-refresh group VJPs, dual `g`/`q` buffers, or late-trigger surgery rebuilds for this revision.

## 3. Definitions and the two directions

Freeze an objective replay context `replay_ctx` and controls `C` for one search:

    F(theta; replay_ctx, C) = loss(theta; replay_ctx, C) + extra(theta; replay_ctx, C)
    theta0   = parameters before this training update
    F0       = F(theta0)
    g        = true gradient of F at theta0 (complete total.backward / autograd.grad)
    q        = g after the existing global `clip_grad_norm_` (Adam input)

With surgery off, `g` and `q` are the **same vector family**: either identical (no clip) or `q = g * min(1, clip/||g||)` under global norm clipping. There is no PCGrad projection and no GradNorm τ-rescaling.

### Bind F to the complete live scalar

Factor a shared objective assembly that returns exactly the live `loss + extra`, with the same coefficients, reductions, and detach policy. Inventory every contribution: base/AL terms, aux, `w_tc`, `w_bptt`, `w_mpc_init_tie`, suffix terms with `sfx_ramp`, SP extras, and every other enabled extra. Add each term exactly once wherever the code currently places it. Former GradNorm/PCGrad coefficients are **not** part of F and must not appear in the LS path.

Replaying only `step_fns[bucket]` is insufficient. Baseline and every trial call the same complete wrapper, including enabled `tc_fns[b]` and second-batch BPTT work. `replay_ctx` includes both source batches, bucket/branch choices, external inputs, sampled randomness (including Python `rng`), and `C` includes the frozen curriculum values. A term-coverage ledger must map each scalar term to its forward call, coefficient, derivative route, and replay inputs.

### Single backward: g from complete F

On every update (search or ordinary), form

    total = loss + extra
    total.backward()

into parameter `.grad`. That tensor **is** `g` before clipping. Then apply the existing `clip_grad_norm_` to produce `q` for Adam. Optionally stash a pre-clip copy of `g` (or its norm) for logging and for the probe slope; do not run a second backward.

Default `true_gradient_mode = plain_total_backward`. Do not implement `force_refresh_preserve_unscaled_groups` or dual-buffer PCGrad reconstruction in this revision. Those remain documented only as a possible **future** extension if someone re-enables surgery under LS (out of scope).

Trial forwards after this setup require no further parameter backward passes.

If a late trigger decides to search after the baseline backward already ran, replay `theta0` under `replay_ctx` only as needed for consistent F0/metrics; the existing `.grad` is already true `g` (surgery was never applied). Discard any tentative optimizer step that has not committed. Measure late-trigger cost; prefer deciding known searches before the forward when possible.

On search updates, stage the next Adam moments once from `q` and the pre-search optimizer state. Ordinary reuse updates use the existing optimizer directly as specified in Section 9. For a standard Adam group:

    m_next = beta1*m_old + (1-beta1)*q
    v_next = beta2*v_old + (1-beta2)*q^2
    m_hat, v_hat = bias-corrected staged moments

Use the configured optimizer's actual denominator, including enabled variants. Let `eta_ref` be the positive reference rate resolved at initialization/resume in Section 7, and `r_group = resolved_group_lr/eta_ref`. Retain these group ratios independently of later epsilon values. Fold this fixed ratio into a diagonal metric:

    D_j = r_group / (sqrt(v_hat_j) + adam_eps)

For the basic Adam/AdamW adapter:

    p_probe = -D*g_preclip    # use pre-clip g so a is the true directional slope of F
    p_momentum = -D*m_hat     # moments from clipped q
    p_decay_j = -r_group*weight_decay_group*theta0_j  # AdamW decay parameters only
    p_exec = p_momentum + p_decay

No momentum or decoupled decay is included in `p_probe`. With global norm clipping, `g_preclip` and `q` share a direction, so probe/exec alignment is far healthier than under PCGrad. Preserve inactive, frozen, unused-parameter, and per-group semantics. More complex optimizer variants require an adapter matching their actual displacement.

### Clipping (only remaining g vs q difference)

Record `||g||`, post-clip `||q||`, and the clipping factor on searched updates. Fixed epsilon bounds and numerical-failure handling still apply. No diagnostic creates a finite-loss-increase veto.

Compute:

    a        = -dot(g_preclip, p_probe) = dot(g_preclip, D*g_preclip)
    s_probe  = -a
    P2       = dot(p_probe, p_probe)
    s_exec   = dot(g_preclip, p_exec)   # diagnostic only

On search updates, require finite `a > 0` and `P2 > 0` over the parameters eligible to move. A negative, zero, or positive `s_exec` does not decide rate selection or block an otherwise valid transferred step. If no usable probe exists, skip with a clear zero-gradient/invalid-direction status; do not divide by zero or claim that a finite rate was selected.

For any epsilon:

    theta_probe = theta0 + epsilon*p_probe
    theta_exec  = theta0 + epsilon*p_exec

At `epsilon = eta_ref`, `theta_exec` must reproduce the existing optimizer proposal (surgery off), including AdamW decay exactly once. These points generally differ because of momentum and decay. Never apply the momentum update on top of `theta_probe`.

## 4. Smoothed acceptance statistics

Use distinct names for the scalar acceptance statistics and Adam's vector momentum. Persist:

    A_prev = EMA of selected probe loss decreases
    B_prev = EMA of positive probe slope magnitudes a
    n_history = number of committed search samples
    last_history_update = optimizer-update index of the last committed sample
    retention_product = product of effective retention factors since reset

Initialize `A_prev = B_prev = 0`, `last_history_update = None`, and `retention_product = 1`. Set `beta_accept_per_update = 0.99` initially. Let `u` count successfully executed optimizer updates and `j = u+1` be the proposed update index. At a search, compute:

    gap = 1 if last_history_update is None else j-last_history_update
    beta_effective = beta_accept_per_update ** gap
    B_candidate = beta_effective*B_prev + (1-beta_effective)*a

Reuse this same `beta_effective` throughout that search. At an interval of 10, `beta_effective` is approximately 0.904, rather than 0.99 per search. This gives old samples an approximately 100-update decay timescale while still observing only the searched minibatches. It does not reconstruct the missing probe results or equal an EMA of every unobserved update. With the reported `al_every=100` and resets on actual AL changes, history is confined to an AL block of roughly the same duration as this decay timescale; it is not long-term smoothing across AL blocks. This is a deliberate adaptation of SaLSa's smoothing.

For each candidate epsilon, evaluate the SAME frozen batch objective at `theta_probe` and calculate:

    Fprobe  = F(theta0 + epsilon*p_probe)
    decrease = F0 - Fprobe
    A_candidate(epsilon) = beta_effective*A_prev + (1-beta_effective)*decrease

The selection criterion is:

    A_candidate(epsilon) >= c1*epsilon*B_candidate

Use `c1 = 0.5` initially, preserving the prior plan's setting; `beta_accept_per_update` and `c1` are configurable. An ablation may compare `c1 = 0.3`. Do not silently relax them on failure.

Every retry must form `A_candidate` from the same `A_prev`. Rejected trials do not become EMA samples. `B_candidate` is formed once from `B_prev`, not repeatedly smoothed during backtracking. Commit both scalar statistics exactly once only after a searched update is successfully executed. On that commit set `last_history_update = j`, increment `n_history`, and multiply `retention_product` by `beta_effective`. Ordinary updates between searches advance Adam state but do not change either acceptance EMA, its sample count, or `retention_product`; the next search accounts for elapsed updates through `gap`.

With zero initial history, the common `(1-beta_effective)` factor cancels, so the first criterion reduces to ordinary sufficient decrease along `p_probe`. With `beta_accept_per_update = 0`, every searched criterion is unsmoothed. Raw EMAs are sufficient because both sides have matching initialization and sampling. If reporting bias-corrected EMAs, divide both by `1-retention_product`; using `1-beta_accept_per_update**n_history` is wrong for variable intervals.

The history contains selected PROBE decreases, not changes measured along the executed momentum direction and not differences between unrelated minibatch baseline losses. Do not smooth the full residual `decrease - c1*epsilon*a` instead: that is a different rule from smoothing decrease and slope separately.

Positive history can admit a negative current decrease. This is intended. There is no additional `Fprobe <= F0` requirement by default.

## 5. Retain the finite-step curvature proposal

Fit curvature using the CURRENT unsmoothed probe evaluation and its CURRENT true slope:

    R = Fprobe - F0 - epsilon*s_probe
      = epsilon*a - decrease
    kappa = 2*R / epsilon^2
    L_measured = max(0, kappa/P2)

When R is positive, finite, and numerically resolved:

    epsilon_quad = -s_probe*epsilon^2 / (2*R)
                 = a*epsilon^2 / (2*R)

This uses two loss values plus the starting derivative. No additional half-step probe is necessary. A half-step experiment must use its actual displacement `(epsilon/2)*p_probe` consistently in every formula.

Use `epsilon_quad` only to propose another trial after failure. It minimizes a fitted CURRENT probe-loss quadratic; it does not exactly solve the history-dependent acceptance inequality, does not certify a local Lipschitz bound, and is not a curvature estimate for `p_exec`. Never substitute `A_candidate` or `B_candidate` for `R` or `a` in this fit.

For rejected finite trials, clamp a valid `epsilon_quad` to `[0.1*epsilon, 0.5*epsilon]`. If interpolation is invalid, halve epsilon. Nonfinite `Fprobe` also triggers halving. Always evaluate the new probe and apply the smoothed test. Growth remains available at the next scheduled or triggered search, never on an unchecked update.

This curvature-informed proposal is the retained custom addition; SaLSa's standard multiplicative shrinking remains an optional `proposal_rule=geometric` ablation.

## 6. Search control and initial settings

| Configuration | Initial choice |
| --- | --- |
| mode | salsa_transfer |
| acceptance | smoothed_probe |
| grad_balance | **forced `""` when line search enabled** |
| beta_accept_per_update | 0.99 |
| c1 | 0.5 |
| proposal_rule | quadratic_with_safeguards |
| expansion_factor | 1.1 in adaptive-interval mode; configurable |
| backtrack_min_factor / max_factor | 0.1 / 0.5 |
| fallback_shrink | 0.5 |
| max_trials | 6, excluding baseline |
| epsilon_seed | Fresh run: configured initial rate. Legacy resume: checkpoint last effective rate, unless explicitly overridden; see Section 7 |
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
| true_gradient_mode | plain_total_backward |
| curriculum_history_policy | approximate_gradual; reset at discrete objective changes and term activation/deactivation |
| legacy_resume_seed | Checkpoint last effective reference LR, or explicit override |
| grouped_scaling | Off |

First search: use `epsilon_seed`. At a normal later search, start at `min(expansion_factor*last_selected_epsilon, epsilon_max)`. Between searches execute exactly `last_selected_epsilon`, with no growth, interpolation, or EMA updates. The adaptive default reduces the previous 1.5 growth factor to 1.1 so deliberate trial expansion does not constantly defeat the rate-stability check. After budget exhaustion start the next search from the pending smaller `retry_seed`, without expansion. Keep `last_selected_epsilon` and `retry_seed` separate. Early searches triggered by a problem start at the retained rate without expansion. AL/curriculum history resets also retain the last selected rate as their first trial, rather than jumping back to a fresh-run rate; changed direction/group conventions require an explicit seed migration. Resume normal expansion after recovery.

Every trial starts from `theta0` with the same controls and staged moments. Stop without an update when trials are exhausted, the next rate falls below `epsilon_min`, or no effective parameter displacement can be represented. No forced acceptance of an untested `epsilon_min`.

Smoothed history need not admit a sufficiently small step: for example, `beta_effective*A_prev < 0` persists as epsilon approaches zero. On a complete selection failure, discard tentative statistics and schedule an explicit history reset for the NEXT iteration. Log this reset; do not erase history mid-search simply to make the current test pass.

### Adaptive interval controller

Persist `successful_update_count u`, `last_search_update`, `last_history_update`, interval `K`, `stable_search_streak`, `last_selected_epsilon`, `retry_seed`, and pending trigger/reset reasons. The scheduling and EMA clocks are distinct even if they usually coincide.

At the beginning of proposed update `j = u+1`, search if there has never been a successful search, a retry/reset/early trigger is pending, the warmup is not finished, or `j-last_search_update >= K`. This means `K=1` searches every update; `K=10` executes nine ordinary updates between searched updates. A failed or isfinite-skipped update does not advance `u`, and keeps a search pending for the next usable batch. Each ordinary successful update uses a new minibatch and advances Adam moments once.

After a successful searched AND executed update, compare `epsilon_new` with the previous committed selected rate, not the expanded initial trial:

    ratio = epsilon_new / epsilon_previous
    stable = max(ratio, 1/ratio) <= 1.15 and trial_count <= 2

When warmup is complete and no recovery/reset trigger applies, increment the stable streak on stable searches. After three such searches set `K = min(K+1, 10)` and clear the streak. A rate drop to 0.75 or less of its prior value, three or more trials, selection exhaustion, or a numerical failure sets `K=1` and clears the streak. For other nonstable successful searches, shorten `K` to `max(1, floor(K/2))` and clear the streak. The first search after initialization or an objective reset does not establish stability. During warmup keep `K=1` and do not accumulate a promotion streak.

This controller is a transparent application heuristic rather than SaLSa's exact fast/slow learning-rate-EMA ratio formula. Stability of selected rates is evidence for reuse, not proof of a safe future rate.

### Cheap early-search triggers

Check signals from the CURRENT baseline forward/backward before applying the current update, comparing with reference statistics captured before incorporating the current batch:

- Explicit multiplier/penalty changes, objective redefinition, or changed parameter-group ratios force a search and `K=1`. Apply the paired history-reset policy in Section 8.
- A marked rise in normalized torque/collision/joint-limit violation metrics forces a search and `K=1`. An initial configurable rule is more than 2 times its reference AND a meaningful absolute increase in normalized violation units. Set the absolute threshold from the existing constraint normalization/tolerances; do not invent one raw-unit threshold for all constraints.
- A gradient norm more than 3 times its moving reference, with a configured nonzero reference floor, forces a search and `K=1`. Monitor the pre-clip `||g||` consistently on ordinary and searched updates (clipping must not hide a spike). This is a change detector, not a curvature estimate.
- Nonfinite baseline losses/gradients use the existing invalid-batch handler and schedule a search at the next valid update. Search does not repair an invalid baseline.

Ordinary minibatch variation can trigger measurements, but does not automatically imply divergence. Finite metric spikes alone do not reset acceptance history or veto execution; objective-definition changes and exhausted searches follow their explicit reset policy. Reference EMAs update once per valid logical baseline, not per trial; document initialization and use only reliable initialized references.

On an update already committed, a later diagnostic can trigger a future search but cannot retroactively protect the past step. If controllers change after commit, latch a trigger before the next update. Gradual curricula follow the explicit approximate-history policy in Section 8, so their routine increments do not force `K=1`. Discrete objective changes still trigger immediate searches. Do not change controller cadence to make rate selection convenient.

## 7. State transaction, momentum transfer, and decay

Provide a clear interface:

    stage_optimizer(q, optimizer0) -> staged_state, D, p_exec
    build_probe(g_preclip, D) -> p_probe, a
    select_rate(theta0, p_probe, frozen_context, EMA_snapshot) -> epsilon, probe_stats
    execute(theta0, epsilon, p_exec)
    commit(staged_state, accepted_EMA, search_state)

Moment estimates and optimizer counters advance once per executed update, never per trial. Acceptance EMAs advance only on successfully executed search updates. On ordinary reuse updates, use the existing compatible optimizer directly with the selected scalar rate and fixed group ratios; verify equivalence with the staged search adapter. Reserve the staged path for searches and explicitly enabled transactional diagnostics. After rate selection, discard probe parameter/buffer mutations, construct `theta_exec` from `theta0`, check parameter finiteness, then commit exactly that displacement and the staged optimizer state once. Do not run `optimizer.step` again and accidentally create a second moment update.

Keep Adam's standard gradient-history recurrence. Do not insert epsilon into `m_next` or `v_next`, and do not add LaProp or accepted-displacement momentum as part of this revision.

For AdamW, include decay in `p_exec`, scale it by the selected epsilon, and apply it once. Do not decay parameters before calculating `F0` or before probes. The probe selection intentionally does not account for decay or guarantee its effect on F. Non-descent `s_exec` is logged, not a reason to remove decay or reset momentum in default mode.

A finite executed update is not reevaluated for acceptance by default. Optional same-batch evaluation of `theta_exec` is diagnostic only; a finite increase is logged and retained. On diagnostic updates, prepare the required transaction before execution. If the diagnostic finds a nonfinite objective, use the existing invalid-update handler, roll back that transaction, and do not commit probe EMAs. Ordinary fast-path updates retain native bf16/`isfinite` pre-step checks and the existing numerical recovery policy; exact post-step rollback requires an explicitly enabled snapshot or staged diagnostic path and its measured cost. Never continue training with known corrupted parameters or optimizer state. Clearly separate numerical failure from permitted nonmonotonicity.

### Learning-rate ownership: no cosine in line-search mode

Disable cosine scheduling entirely while line search is enabled. Bypass the existing per-step `lr = cfg.lr_at(step)` assignment and any scheduler callbacks that could write optimizer group rates. The running cosine schedule must not supply a multiplier, cap, search seed, or recovery rate. One-time inheritance of a legacy checkpoint's last applied LR follows the migration policy below. Keep the existing scheduler behavior only when the line-search feature is disabled.

All probe and execution directions exclude the scalar learning rate; apply the selected epsilon exactly once. On ordinary reuse updates set each group rate to `last_selected_epsilon * fixed_group_ratio` and call the native optimizer once. On staged search updates commit the selected displacement and moments once, without another optimizer step.

Use this explicit initialization/resume policy:

- Fresh run: use an explicit seed override or the configured initial reference LR. Derive fixed group ratios from the configured initial group rates.
- Selector checkpoint: restore epsilon, bounds, group ratios, histories, clocks, interval, and pending state. No scheduler may overwrite them.
- Legacy checkpoint with Adam state but no selector: before scheduler initialization or any LR assignment, read the last effective reference LR used with the restored optimizer. Prefer an explicit saved effective-LR field; otherwise use saved optimizer-group LRs only after confirming checkpoint save ordering means they describe the last applied update. An explicit resume-seed override takes precedence. Derive compatible group ratios from those saved effective rates. Initialize empty A/B, `K=1`, and perform the first search without expansion at that inherited rate. Preserve the restored Adam moments. Never blindly jump to `cfg.lr=1e-4` just because selector state is absent.
- If a legacy checkpoint does not identify a positive finite last effective rate (or compatible group ratios), require an explicit resume seed/group mapping. Fail initialization clearly rather than guessing a larger fresh-run rate. Zero-rate/frozen groups need the existing explicit frozen-group policy.

Using the checkpoint's last applied rate once is state migration, not continuing cosine scheduling. Do not evaluate `cfg.lr_at(step)` to supply future seeds, caps, or multipliers. Fixed `epsilon_min`/`epsilon_max` default to the configured multiples of the resolved initial or migration seed and are then checkpointed; explicit bounds must contain that seed. Log seed provenance and any deliberate override. Reset warm-start conventions and force a search if group ratios or parameterization deliberately change. Search-interval scheduling remains active and is separate from the disabled learning-rate scheduler.

## 8. Full-objective replay, state, and history policy

### Replay both batches and every random source

Introduce an objective replay record before the first relevant random draw or mutable input rewrite. Retain the actual primary batch and, when enabled, BPTT's second batch, including source trajectory initializations, targets, masks/selections, suffix payload, jitter, and other sampled inputs. Retain references to immutable batch storage instead of cloning tensors unnecessarily; copy only mutable inputs that baseline execution would overwrite. Never obtain a fresh second batch inside a candidate evaluation.

Cover all reported random streams: `gen`, `sfx_gen`, `jit_gen`, the phase-2
**`numpy.random.Generator` named `rng`** (bucket choice, replay coin, BPTT
include coin — not `random.Random`), and global torch CPU/CUDA generators.
`Phase2Roller.gen` is advanced in `roller.tick()` before the minibatch and is
outside F once the batch is frozen; register it for checkpoint parity only —
never tick inside probes. Verify concrete types and capture each through its
native state API (see `amortized_opt/objective_replay.py`). Prefer recording
realized external samples in the replay record so candidates consume the same
samples even when parameter-dependent computation changes. Note: `_bptt_term`
takes an `rng` argument but currently draws only via `gen`; the BPTT coin flip
lives in `train_phase2`. For dropout or other internal sampling, replay
pre-forward RNG states around a fixed call structure; where data-dependent draw
counts change, supply recorded randomness explicitly or reject the unsupported
branch. Merely restoring torch's global RNG is insufficient.

Record the baseline post-forward/training RNG positions and preserve the one logical baseline's stream advancement. Every probe and any late-trigger baseline replay runs in an isolated RNG/buffer scope and restores those live positions afterward.

### Re-encode and repeat parameter-dependent rewrites

Freeze external source inputs, not all derived initialization/reference tensors. Every candidate must run the normal encode path and re-apply, in their original live order:

1. encoder → `strike` waist reference rewrites (when those heads are enabled)
2. `seed_waist_pr_to_sp` (detached SP inside the seed — F values can depend on θ while that path contributes no grad; document this)
3. `correct_quintic_init_arms` on fresh (non-replay) rows when the live loop would run it

Start from fresh views/copies of the unmodified source payload so in-place rewrites cannot leak between baseline and probes. Repeat these operations for all relevant BPTT branches. Recompute normal `clamp_traj_to_box`, dynamics, collision quantities, active masks, and all extras. Do not cache baseline encoder outputs or parameter-dependent strike references as static physics inputs.

Audit derivative parity through these rewrites: a reference recomputed from theta but detached during backward can make autograd's recorded derivative differ from the derivative of the replayed scalar. For the default re-encode policy, the probe gradient must include all differentiable parameter-dependent paths in F. If the existing detach semantics prevent this, explicitly implement a shared frozen-derived-reference policy for BOTH baseline/probe F and `g`, or expose the missing derivatives for `g` while preserving intended training semantics. Do not claim true slope until the full-objective directional check passes. Any frozen-reference alternative is a separately named objective convention, documented with its effect on rate transfer and a history reset; it cannot silently replace the default.

All probes start from the same pre-forward mutable-buffer snapshot and discard their writes. Retain the baseline forward's single logical buffer result. Do not populate MPC caches, multiplier controllers, or training metrics with candidate outputs. Actual-point diagnostic forwards are isolated too. If downstream code needs fresh outputs at `theta_exec`, evaluate them explicitly and account for that cost.

With surgery off, there is no GradNorm/surrogate EMA to poison on probes. Still never write frontier/replay/post_hit from candidate trajectories.

Search setup uses the single complete `total.backward()` from Section 3. Trial evaluations do not backward through parameters. Permit local autodiff required by physics; preserve the normal bf16/evaluation behavior and use stable scalar reductions. A disabled parameter backward is not a license to omit differentiable physics computation from F.

### AL and curriculum history

The reported AL/PID controls are held between `al_every=100` updates. Preserve that cadence and its actual training clock; search retries must never call `al.update`. On an actual change to lambda or penalty, reset A/B, sample count, `last_history_update=None`, `retention_product=1`, stability streak, and `K=1`, and force a search before the next update under the changed objective. An `al.update` call that leaves controls identical need not reset history. Smoothing therefore occurs within each AL block, not across the discontinuity. Keep the last selected rate as the recovery trial seed without expansion.

**Dual clocks:** selector `u` counts successfully executed Adam updates; AL / `sfx_ramp` / `bptt_frac` use absolute training `step`. A failed search or isfinite skip that does not advance `u` may still advance absolute `step` (and thus curriculum/AL schedule). Log both clocks; do not silently retarget AL to `u`. Decide explicitly whether nonfinite-`gnorm` paths that today still accumulate `viol_acc` / call `al.update` keep that behavior under LS; default is preserve existing skip semantics and document them.

Default `curriculum_history_policy=approximate_gradual`: keep the existing per-step `sfx_ramp` (approximately the first 1k steps) and later `bptt_frac` evolution. Freeze their CURRENT values, term weights, and realized branch/batch choices for all baseline/probe evaluations within a search. Retain A/B across small scheduled increments and label this as approximate history from changing objectives. Do not reset or force a search on every routine ramp increment and do not freeze the training curriculum across updates.

Reset both histories and force a search on discrete term activation/deactivation, objective branch/stage changes, manual weight/normalization changes, and actual AL jumps. For a smooth numeric ramp with unchanged active terms, including reaching its endpoint, keep approximate history. Log curriculum values and the amount of drift since the last search. If strict stationary-objective history is desired as an ablation, reset on every change and report the resulting loss of smoothing/search overhead; it is not the default. Fresh minibatch variation and parameter-dependent residual masks are not resets.

Replay completeness is a required enablement gate. If an enabled extra, RNG source, BPTT payload, or rewrite cannot be replayed, do not silently omit it or switch to an approximate scalar. Finish that adapter before enabling this mode; return an actionable unsupported-path error if needed. Late triggers must be handled by complete replay before the current update, not by accepting an unvalidated same-batch slope.

## 9. Required computational-efficiency implementation

Implement all five measures below. They combine the chosen SaLSa-style algorithm with workload-specific engineering; do not attribute all five to the published implementation. Preserve the acceptance and optimizer semantics above.

### 1. Ordinary fast path between searches

On an update that reuses the selected rate, run the existing baseline forward, **plain** `total.backward()`, cheap trigger checks, clip, and the existing optimizer step. Preserve its supported fused/foreach path. Do not build `p_probe`, stage duplicate moments, clone the full parameter/optimizer state, or run a trial evaluator on this branch. Set group learning rates from the retained scalar and fixed ratios without changing the moment recurrence. Release the baseline autograd graph once all required gradient work finishes.

Decide scheduled searches and known AL/branch triggers before baseline execution when practical. On every visit retain the Section 8 replay record for all active streams and both batches through the trigger decision. This includes small RNG-state captures and references to realized payloads, not a second network graph or a full optimizer copy. Measure payload retention and required mutable-input copy costs; do not call them free. For a late trigger, replay the full baseline at `theta0` if F0/context must be regenerated; do **not** rebuild surgical gradients. Allocate/copy parameter snapshots and stage optimizer proposals only when a search or diagnostic needs them.

An ordinary fused optimizer step cannot also promise exact rollback of an arbitrary post-step numerical failure without retaining old state. Keep existing bf16/`isfinite` skip handling on the fast path. Route explicitly scheduled transactional diagnostics through the staged path, and document any additional recovery mechanism the codebase requires. Do not hide full-state snapshots inside the normal fast path.

### 2. Cache quantities that are fixed throughout one search

Reuse the complete `F0`. Build `g` once via the baseline `total.backward()`; after clip, stage Adam from `q`. Compute staged moments, bias corrections, `D`, `p_exec`, `p_probe`, `a`, `P2`, `beta_effective`, and `B_candidate` once per search. Rejected trials change only epsilon, candidate parameters, forward-dependent outputs, and scalar acceptance/interpolation calculations. They do not rerun backward or rebuild optimizer directions. Compute optional execution-direction diagnostics once when enabled.

The slope needed for interpolation is already `a`; the curvature residual uses the existing trial loss. Do not add an epsilon/2 evaluation, Hessian calculation, or JVP for this selector. Invalidate cached search quantities when the search ends; a new minibatch or update requires new gradients and staged statistics.

### 3. Reuse device storage and reduce parameter-copy overhead

Lazily allocate reusable storage for `theta0`, staged state, candidates, and necessary mutable buffers. Retain allocations across searches when shapes, devices, dtypes, and parameter membership match; refresh their contents each search. Rebuild storage when those properties change. Avoid model deepcopy, optimizer deepcopy, and fresh full-size allocations inside the trial loop.

Construct every candidate from `theta0 + epsilon*p_probe` using safe in-place writes into dedicated storage and batched tensor operations where supported. Avoid incremental perturb/undo arithmetic that can accumulate roundoff. Preserve parameter identities, aliases/tied weights, and optimizer references. A functional evaluator or temporarily overwritten live parameters is acceptable after verifying repository compatibility. If live parameters are overwritten, restore them on failure and construct the executed candidate directly from `theta0` on success. Reusable buffers reduce allocation overhead, not the bandwidth needed to write candidate parameters; benchmark both memory and time.

### 4. Keep trial arithmetic on device and consolidate synchronization

Keep loss reductions, `a`, `P2`, acceptance EMAs, the curvature residual, and interpolation arithmetic on the accelerator in stable precision. Consolidate the accept/reject, finite-value, and next-trial information needed by the host into one small transfer per trial where practical. Avoid separate `.item()` calls for every loss component, norm, and logging field. Buffer diagnostics and transfer them together at the existing logging cadence. Distributed reductions and decisions must still follow Section 11.

Keep the complete objective wrapper separate from the dynamic search controller and dispatch through the existing bucket-specific compiled functions (`step_fns[b]`, `tc_fns[b]`, and relevant extra/BPTT functions). The second batch may require its own bucket dispatch. Reuse each compatible compiled function for stable shapes and parameter storage, avoiding recompilation for each epsilon; do not replace this split with an assumed single compiled evaluator. Do not require compilation of the Python backtracking loop. A host-controlled loop still needs a decision synchronization; this optimization reduces redundant synchronization rather than eliminating it. Keep a correct eager fallback and measure compile warmup separately from steady-state throughput.

### 5. Cache static physics inputs and batch independent evaluations

Cache parameter- and trajectory-independent inputs such as robot topology, fixed inertial data, fixed link transforms, trajectory differentiation operators, fixed targets, and reusable collision geometry. For per-batch fixed inputs, scope the cache to the frozen batch/context. Record invalidation conditions for changes to the robot, time grid, geometry, normalization, targets, or device/dtype. Learned or otherwise trainable quantities cannot be treated as static.

Every candidate must recompute trajectory-dependent kinematics, velocities, accelerations, mass/dynamics terms, torques, distances, residuals, and active masks. A previous candidate's activations or dynamics values are not a valid cache simply because epsilon changed only slightly. Batch/vectorize independent waypoint inverse-dynamics and collision work where supported. Preserve temporal dependencies in rollouts and controllers; do not parallelize a sequential state evolution as if waypoints were independent. Candidate trials remain sequential because each rejection chooses the next epsilon.

## 10. Core pseudocode

```text
# No cosine callbacks. Resolve fresh/selector/legacy resume state per Section 7.
# Force grad_balance=""; refuse PCGrad/GradNorm while line search is on.
j = successful_update_count + 1
known_search = warmup_or_due_or_pending_control_change(j, ...)
transactional_diagnostic = diagnostic_due(j)
replay = begin_full_objective_record_before_draws_and_rewrites()
# Record primary + optional BPTT second batch, all active RNG streams
# (incl. Python rng), and controls C.

baseline = full_loss_plus_extra_forward(replay)   # encode, rewrites, loss+extra
if invalid_baseline(baseline):
    existing_isfinite_skip_handler_and_latch_search()
    return

total = baseline.loss + baseline.extra
total.backward()                                  # g = true ∇F
g_preclip = snapshot_grads_for_probe()            # before clip
gnorm = clip_grad_norm_(params, grad_clip)        # q = Adam input in .grad
if invalid(gnorm) or grads_nonfinite:
    existing_isfinite_skip_handler_and_latch_search()
    return

triggers = inspect_metrics_and_preclip_g_norm_before_reference_update(...)
run_search = known_search or triggers.require_search

if run_search and not known_search:
    # Late trigger: replay only if F0/context must be regenerated.
    # No surgical rebuild — grads already are true ∇F.
    maybe_isolated_full_baseline_replay_for_consistent_F0(replay)
    # Commit only one logical baseline's buffer/RNG effects.

F0 = baseline.loss + baseline.extra
commit_baseline_reference_metrics_once(...)
if not run_search and not transactional_diagnostic:
    set_group_rates(last_selected_epsilon, fixed_group_ratios)
    result = existing_optimizer_step_once_after_isfinite_checks()  # uses q
    if not result.executed:
        latch_recovery_search_if_required()
        return
    successful_update_count = j
    # No probes, duplicate moments, or full-state snapshots here.
    # A/B/count/retention and selected epsilon remain unchanged.
    advance_existing_update_controllers_once_and_latch_discrete_changes()
    return

theta0, optimizer0, baseline_logical_buffers = refresh_reusable_transaction_storage()
staged = stage_optimizer_once(q=.grad, optimizer0)   # moments from clipped q
p_exec = staged.actual_direction_including_decay

if not run_search:
    epsilon = last_selected_epsilon
else:
    p_probe = build_once(-staged.D * g_preclip)
    a, P2 = cached_direction_reductions(g_preclip, p_probe)
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
        Fp = full_loss_plus_extra_replayed_candidate(
            replay, reencode_and_rewrite=true, static_physics_cache=...)
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

Use stable precision for loss reductions, `a`, `P2`, `R`, and EMAs; at least float32, with float64 scalar accumulation where practical. Detect cancellation in the curvature residual and fall back to geometric shrinking. Zero observed loss change is not automatically a successful search; apply the specified smoothed criterion. Detect numerical stalls explicitly.

Follow the reported bf16 path and existing `isfinite` skip/recovery handlers. There is no GradScaler to unscale or advance. Keep the successful optimizer-update clock distinct from any absolute-step curriculum/AL clock; preserve existing skip behavior for those outer controls and record their actual values. DDP is outside the initial scope unless active in the repository. If supported later, baseline/probe reductions and `g` must match globally, with shared decisions and no double-summing replicated dot products. Accumulated gradients require replay of the corresponding effective batch, not only its last microbatch.

Log `F0`, selected `Fprobe`, optional `Fexec`, all cost/violation groups, epsilon seed/selection, trial count, `a`, `s_exec` and separate momentum/decay slopes, direction alignment, displacement norms, current decrease, `A_prev`/`Ac`, `B_prev`/`Bc`, `R`/`kappa` when resolved, growth success, history resets, failure reasons, search-versus-reuse status, `K`, elapsed updates, trigger reasons, stability streak, `beta_effective`, `grad_balance=""`, and timing. Label probe and executed losses distinctly. Never report `Fprobe` as the post-update training loss. Log selector `lr` / epsilon separately from any disabled cosine value.

The reported checkpoint currently stores model/optimizer/AL and `torch_rng` only. Add a versioned selector state and complete active-generator state registry (`gen`, `sfx_gen`, `jit_gen`, Python `rng`, and relevant global CPU/CUDA state); do not serialize ephemeral graphs or candidate buffers. Legacy checkpoints missing those streams cannot promise bitwise replay of the previous run: initialize missing streams by the existing seed policy and log this limitation. Preserve both training/AL/curriculum clocks and the new successful-update clock. Checkpoint the two acceptance EMAs, sample count, `retention_product`, `last_history_update`, `successful_update_count`, `last_search_update`, interval/streak, selected-rate/retry seeds, metric reference statistics, pending trigger/reset reasons, direction convention, controller identity, curriculum values/active-stage identity, fixed group ratios/bounds, resume-seed provenance, last applied effective group LRs, `grad_balance=""`, and configuration alongside the optimizer. Restore them consistently or explicitly reset incompatible history.

Focused tests:

1. Known quadratic: `p_probe` has the correct slope and fitted directional curvature. First-step or `beta_accept_per_update=0` selection reduces to the unsmoothed probe criterion.
2. Uphill momentum: construct `s_exec > 0` and `a > 0` (e.g. via decay dominance); verify selection uses `p_probe` and execution uses `p_exec`, with no silent descent fallback or veto of a finite increase.
3. Smoothing: create positive past improvement that permits a negative current probe decrease; verify intended acceptance. Rejected trials must not repeatedly advance A or B.
4. Transfer and decay: at the selected rate, the executed update matches the original optimizer proposal with surgery off; probes are fully restored, decay occurs once, and moments/counters advance once.
5. Expansion, checkpoint continuation, objective-change resets, and negative-history/budget failure behavior.
6. Repeatable dropout/buffers/caches; nonfinite probe/diagnostic evaluations trigger the documented transactional failure path; ordinary-step isfinite skip/recovery follows its declared native policy.
7. `K=10` performs nine ordinary updates between searched updates; reused rates do not expand, no probe closures run, Adam advances normally, and A/B remain unchanged. The next search uses `gap=10` and `beta_effective=0.99**10`.
8. Promotion and demotion boundaries, early triggers before execution, warmup, failure without update-clock advancement, and resume from a checkpoint midway through a reuse interval.
9. Both EMA sides use the same elapsed-update retention and rejected candidates do not compound it; optional bias correction uses the retention product. Discrete objective resets force a new search even if the interval has not elapsed; gradual curriculum changes follow the explicit approximate-history policy.
10. Scheduler ownership: verify cosine cannot overwrite, multiply, cap, or reseed the selected rate on searched or reuse updates, including checkpoint resume. Compare the same selector state and frozen input at different legacy cosine step indices; proposed and executed rates must match. Verify legacy scheduling still works with line search disabled.
11. Instrument the fast path to verify no trial evaluations, duplicate moment staging, full-state snapshots, or PCGrad/GradNorm calls. Verify optimizer equivalence, including supported fused execution and bf16/isfinite skips. Verify enabling LS forces `grad_balance=""` even if the launcher passed `al_pcgrad_output`.
12. Across rejected trials, verify fixed quantities are computed once and reusable storage is retained. Compare cached/compiled objective results with the uncached eager evaluator, including changed trajectories and cache-invalidating inputs. Test only applicable modes; compilation is optional, objective correctness is required.

Additional required integration checks for the reported code paths:

13. On a small deterministic batch, verify `g` from `total.backward()` matches `autograd.grad(loss+extra)` term-by-term, with clipping applied only afterward. Check the full-objective directional slope against a finite-difference probe, including rewritten references; do not accept autograd parity alone when detach semantics hide a missing derivative.
14. Enable TC, BPTT, MPC-init tie, suffix, and SP branches individually and together. Verify baseline/trial-at-epsilon-zero scalar and per-term parity, frozen second-batch identity, multi-stream random replay (including Python `rng`), and candidate-dependent re-encoding / waist / strike / `correct_quintic_init_arms` rewrites. Repeated rejected trials must leave RNG, input payloads, controllers, and caches unchanged relative to one logical baseline.
15. Resume late-run Adam moments from a legacy checkpoint with a small last applied LR; verify the first search starts there without expansion, never at `cfg.lr` unless explicitly overridden. Cover selector checkpoints, LR-field provenance, and missing-LR failure/override handling.
16. Exercise actual `al_every=100` boundaries, unchanged AL calls, continuous `sfx_ramp`/`bptt_frac` increments, discrete term activation, and curriculum resume. Verify history persists within blocks and across gradual changes as documented, resets on discontinuities, and does not accidentally reset every ramp step. Log dual-clock behavior on selection failures.
17. (Optional future / out of scope for default LS) Re-enabling PCGrad under LS would require a separate design for dual `g`/`q` and is not part of this revision's acceptance tests.

On a representative robotics batch, verify the probe slope with a finite-difference debugging check. Benchmark existing AdamW **with** `al_pcgrad_output`, this hybrid with surgery off and fixed `K=1`, and this hybrid with adaptive `K` in `[1,10]` over a bounded run. Compare geometric-only proposals if interpolation cost or retry quality remains uncertain. Optional strict actual-update search is a separate ablation. Compare progress and physical violation metrics against wall-clock time, and measure how well the probe-selected rate transfers. Report average `K`, fraction of updates searched, trials per search, trigger frequency, probe time, peak device memory, and time spent in multi-stream replay preparation and second-batch retention, transaction setup, candidate writes, optimizer staging, and synchronization. Separate compilation warmup from steady-state timings. Verify that ordinary-update timing stays close to the existing plain-backward optimizer path and identify any unavoidable replay overhead. Approximate overhead as `[search_fraction * (mean_trials * full_probe_time + staging_time) + late_trigger_fraction * baseline_replay_time + per_update_replay_bookkeeping_time] / baseline_step_time`, counting each measured phase once; validate with measured wall time. Do not assume the paper's reported overhead applies to this physics workload. No monotonic-loss assertion belongs in the default integration test.

## 12. Optional per-cost scaling extension

Keep grouped scaling off during this revision's initial rollout. The prior idea remains a later extension: individual group scales require an explicit mapping into a combined momentum update. One shared Adam moment vector cannot simply be split into independent Adam updates.

For that extension, define paired probe/execution directions per group and separate acceptance histories. Individual probes alone do not validate their sum. A combined surrogate probe check may select a final scalar correction; transferring that correction to the grouped momentum direction still does not guarantee actual decrease. Mixing weights can also make the combined surrogate direction non-descent on the total objective even if every individual group probe is downhill on its own cost.

Do not automatically add a mandatory strict actual-update check to the default transfer mode in order to support grouping. Treat grouped combination semantics as a separate feature with its own configuration, tests, and benchmark before enabling. Do not reintroduce PCGrad as a prerequisite for grouping.

## 13. Deliverables and source boundaries

Deliver the optimizer proposal adapter, replayable evaluator, smoothed rate selector with optional curvature interpolation, execution/commit integration, configuration (including **forced `grad_balance=""`** when LS is on), checkpoints/logging, focused tests, all five efficiency measures in Section 9, and a bounded benchmark report. Use actual repository paths in the implementation summary.

Completion first requires full `loss + extra` replay parity, plain `total.backward()` gradient construction with surgery disabled, explicit derivative treatment of encoder/reference rewrites (including `correct_quintic_init_arms`), and safe legacy LR migration. It also requires correct separation of probe and execution paths; upward rate recovery; once-per-executed-update Adam bookkeeping and once-per-successful-search acceptance-EMA bookkeeping; no accidental persistence of probe trajectories; bounded failure handling; correct interval/trigger/checkpoint behavior; and measured overhead and transfer quality. Intermittent reuse is an empirical stability/efficiency tradeoff, not a guarantee for every skipped update. Explicitly document unsupported execution modes. Re-enabling `al_pcgrad_output` under line search is out of scope for this revision.

## 14. Integration findings from codebase inspection

Concrete hooks in `amortized_opt/train_cost.py` (verified against the live loop). Implementers must follow these, not only the abstract pseudocode.

### Force surgery off before compiled step graphs

Override `cfg.grad_balance = ""` at the **entry** of `train_phase2`, **before** `make_cost_step` (~5459). Return arity and `cost_grad_balance_tuple` vs `cost_tuple` are baked at build time (`3411–3496`). Clearing balance only inside the step loop is too late when `compile_cost=True`. Launcher default remains `GRAD_BALANCE=al_pcgrad_output` in `_run_rhythm_bellman.sh` — LS must override it and log the fork.

### Bypass cosine every step

Gate `lr = cfg.lr_at(step)` / `param_groups["lr"] = lr` (~5704–5706). Under LS set rates from `last_selected_epsilon * fixed_group_ratios` only.

### Replay begin / pre-rewrite snapshot

- Roller harvest (~5708–5726) is **outside** F (runs every absolute step; do not re-tick on probes).
- Begin the objective replay record after the roller and **before** `rng.choice` (~5729), or immediately after the batch is materialized if recording realized tensors.
- **CRITICAL:** clone pre-rewrite `batch["init"]`, `context` (esp. `yaw_now`), and `strike` **before** yaw jitter (~5823), SP→strike refs, `seed_waist_pr_to_sp` (~5853), and `correct_quintic_init_arms` (~5867). In-place overwrites make “start from source payload” impossible without that snapshot.

RNG registry: `gen`, `sfx_gen`, `jit_gen`, numpy `rng` (`numpy.random.Generator`), global torch RNG; optional `roller.gen` for checkpoint bit-parity only. Prefer recording realized TC gains (`kp,kd`), suffix `offs`, and BPTT second-batch payloads. BPTT always uses `bptt_bucket = min(pool.buckets)`, which may differ from the main batch bucket — freeze that path separately (`encode(model, …)` inside `_bptt_term`, not `encode_step`). Helpers live in `amortized_opt/objective_replay.py`.

### Backward / search splice

Plain path: `total.backward()` (~6270) → snapshot `g_preclip` → `clip_grad_norm_` (~6272) → ordinary `opt.step` or staged search commit (**no** second `opt.step`). Known-search decisions preferably before the forward; late `‖g‖` / violation triggers after clip, before commit.

### Commit gate for side effects (explicit policy)

Today, if `total` was finite, **frontier EMA** (~6279) and **`viol_acc`** (~6289) still run even when `opt.step` is skipped for nonfinite `gnorm`. Under LS:

| Event | Roller | Frontier EMA / `viol_acc` / `al.update` | Selector `u` / A,B |
|---|---|---|---|
| Ordinary or search **success** | already ran | advance from **baseline** `_traj` / metrics only | advance; commit A,B on search |
| Selection **failure** | already ran (cannot undo) | **skip** EMA and `viol_acc` for that step (plan rollback); log dual-clock drift vs `al_every` | no `u`; retry_seed; no A,B |
| Nonfinite `total` | already ran | skip (`continue`) | no `u` |

Never write frontier/replay/post_hit from probe trajectories. After `al.update`, compare λ/pen before/after — reset selector history only on **actual** change (`AugmentedLagrangian.update` always mutates internal EMAs even when λ looks flat).

### Resume / checkpoint helpers (landed)

Use :mod:`amortized_opt.line_search_resume` — do not reimplement:

- ``resolve_epsilon_seed(...)`` — pass CLI ``--line-search-epsilon-seed`` as
  ``explicit_override``; fresh path otherwise prefers ``cfg.lr``. After resolve,
  **overwrite** ``line_search_cfg.epsilon_seed`` (and bounds) before the first
  ``choose_search_seed`` or the search still starts at the dataclass default
  ``1e-4``.
- ``pack_line_search_checkpoint_extra`` / ``unpack_...`` — merge into
  ``save_phase2_checkpoint(..., extra=)`` under ``extra["line_search"]``.
- Legacy LR migration requires non-empty Adam ``state``; a ckpt saved before
  any ``opt.step`` looks “fresh”.

See ``tests/test_line_search_resume.py``.

### Status (implementation)

`train_phase2` line-search path is wired (cosine off, search/commit, §14
commit-gate, replay for main+TC+suffix, checkpoint `extra["line_search"]`).
**Deferred:** BPTT second-batch probe replay — a search step with BPTT included
raises ``ReplayIncompleteError``. Until that lands, keep BPTT off under LS
(launcher warns when ``LINE_SEARCH=1``).

References:

- [SaLSa paper](https://arxiv.org/html/2407.20650v1): motivation for a preconditioned current-gradient probe, transfer to momentum, smoothed acceptance statistics, and adaptive search-frequency motivation. Our controller thresholds and elapsed-update EMA weighting are explicit custom choices.
- [Current SaLSa source](https://github.com/TheMody/No-learning-rates-needed-Introducing-SALSA-Stable-Armijo-Line-Search-Adaptation/blob/main/src/salsa/SaLSA.py): implementation reference only. Its flags and search/decay behavior are not automatically this specification; in particular always execute the staged momentum direction after selection here.
- [Adaptive Backtracking Line Search](https://arxiv.org/html/2408.13150v2): Armijo/descent-lemma background. Our interpolation safeguards, transactional integration, reset policy, and defaults are application design choices.

- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): implementation guidance for avoiding unnecessary gradient tracking, reducing synchronization, and compiling suitable compute regions.
- [PyTorch optimizer documentation](https://docs.pytorch.org/docs/stable/optim.html): supported optimizer implementations and execution options; preserve repository-compatible fused/foreach behavior where available.
