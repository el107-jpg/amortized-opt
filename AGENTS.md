# Project instructions

## Isaac geometry parity

Registered Isaac drum tasks must use the local `g1_scene.g1` asset and shared
`amortized_opt.constants` stick/body geometry, matching the decoder. Bind the
registered task's asset factory and shaft probes through `isaac_rl.decoder_scene`
before constructing configs, then configure the tip offsets. Do not fall back
to the external6cm/inward-flared asset. Verify USD collider transforms and
measured FK/tip parity at startup. Old simulator outcomes/state banks with the
external mount are incompatible; generate fresh labels for new experiments.

Isaac assignment rollouts must clear cached wrist contact-force reports after
teleport/reset and before the first physics substep. Pre-reset reports otherwise
trigger false contact recovery and change next-step PD gains despite identical
q/qd. New runtime defaults enable `--clear-reset-contacts --freeze-hit-scoring`;
fixed scoring preserves fresh-scene sigma0.4s, speed shape1 and done quality0.25,
so labels do not depend on PPO curriculum history. Decoder bfloat16 remains the
default; FP32 is an explicit diagnostic option. Old legacy resumes preserve
their archived runtime unless explicitly overridden, and legacy labels/replay
must not be mixed with corrected runtime labels. Record `rollout_runtime` in
datasets/checkpoints; regenerate labels for further corrected-controller fitting.
For local continuity tests hold each deadline in the same physical slot: sweep
order otherwise aliases with world grid coordinates. Residual contact-edge
variation is possible even after the reset fix; report repeatability controls.

Isaac assignment hit grading must use the decoder's requested physical impact
speed, from the same `strike_speed` mapping used by `strike_velocity` and the
scheduler (default1+2×normalized loudness m/s). Keep beat_velocity normalized;
install a separate beat_target_speed_m_s tensor for the registered task's
speed-quality calculation. New rollouts enable `--match-decoder-speed` and
record `matched_decoder_speed_contact_reset_v2` plus the speed bounds. Prior
legacy or contact-reset-v1 labels are not compatible with this scoring change;
collect fresh labels before further fitting. Timing sigma/contact thresholds
are separate criteria and are not changed by speed matching.

For contact-mode Isaac assignment rollouts, the current cymbal/hi-hat physical
cylinders remain10 cm thick. Do not make
them1 cm thick merely to narrow hit credit: at250 Hz a fast stick can cross a
thin collider between physics samples, and CCD is not assumed available on the
GPU path. Bind the local drum layout/USD and `isaac_rl.cymbal_contact_scoring`
before constructing registered Isaac configs. Cymbal/hi-hat top-face entry may
qualify through the body depth during its contact episode, but vertical-side
contact qualifies only within5 mm of the top. Top entry is certified per stick
probe so one probe cannot validate a different probe's side contact. The archived
runtime is `thick_cymbal_top_entry_side5mm_contact_v4`; record its drum-USD/layout
hashes. The v2 runtime above describes still older labels; new runs use v5.
Legacy/v2/v3 outcome labels must not be mixed into v4 fitting; collect
fresh labels before retraining or claiming v4 ranker performance. A v2 starting
state can be reused only after verifying the unchanged physical geometry and
installed q/qd/tips; v3 thin-collider states require a new provenance audit.
Controlled v4 tracker probes show side geometry qualifies from the top down to
5 mm below it (and up to2 mm above), but a horizontal side swipe still fails
the unchanged incidence gate: tangential speed must be at most1.5 times the
downward normal speed. A downward-diagonal upper-side strike with real force
can create an entry; side contact at6 mm or deeper cannot. Final beat credit
also requires sufficient impact-speed/timing quality. In a16-plan ready-start
hi-hat/snare audit, one0.42-mm-deep upper-side entry had15.4 N force but only
0.164 m/s normal speed against2.4 m/s requested and quality0.172<0.25, so it
did not count;10 top-entry hi-hat events with about1.85–2.41 m/s did count.
The tracker counter named `rejected_deep_side_substeps` counts all force-bearing
edge contacts lacking top or upper-side qualification, not exclusively literal
deep-side strikes. Use the contact-debug events and requested-beat quality to
distinguish geometry qualification, tracker entry, and final credited hit.

## Current continuous capsule contact scoring (v5)

New Isaac contact rollouts use
`distal20cm_capsule_top_side5mm_contact_v5`. The physical cymbal/hi-hat
cylinders remain 10 cm thick. On edge drums, the eligible stick is the full
continuous distal 20 cm playing shaft (8 mm radius) plus the 10 mm tip bead.
Keep the v4 top-face, upper-5-mm side, real-force, downward-incidence,
requested-speed, timing, and re-arm rules. A witness at any material fraction
of that shaft may qualify; fixed probe spacing must not create blind gaps.
Track a certified top-entry material fraction within its contact episode.
Membrane drums remain tip-only. Preserve the archived v4 scorer for old labels.

Fresh phase-2 and terminal-pose training use `contact_scoring_mode="isaac_v5"`.
The trajectory loss uses continuous shaft witnesses and a body-SDF proxy for
force; the terminal-pose role uses static capsule geometry only. Archived
`legacy`/`isaac_v4` resumes retain their saved objective. A changed objective
requires `--init` from weights with fresh optimizer/AL/critic state, never a
same-objective `--resume`. v4 and v5 Isaac outcomes are incompatible: collect
fresh v5 labels with `scripts/collect_v5_prefirst_dataset.py` before fitting
or claiming a v5 hand-assigner improvement. Record the v5 runtime and
drum-USD/layout hashes. v2 starting states still require the existing physical
geometry and installed q/qd/tip audit.
The current v4 reward predictor is uncalibrated under v5; keep its checkpoint
and labels archived until it can be evaluated or retrained on fresh v5 outcomes.

## Differentiable physics geometry v2

Fresh phase-2 costs also use `physics_loss_version="geometry_v2"`. This is
independent of the Isaac contact-scoring version. It installs the batched
convex capsule/cylinder signed-distance query, shared upper-arm/forearm
proxies, cross-arm checks, and collision coverage for physically present held
limbs after their shorter command horizon. The default geometry path is the
50 Hz first-order-hold command sampled every 4 ms (`n_sub=5`), with conservative
between-sample clearance bounds. `d_sampled_sdf` reports sampled clearance;
`d_sweep_bound_gap` reports the additional conservative tightening, not measured
PhysX penetration. Optional `--geometry-motion-mode pd` uses the saturated
nominal coupled PD surrogate; it has no contact impulses.

Terminal feasibility uses the hard minimum of complete contact-candidate
residuals. Geometry/incidence and impact-speed quality must come from the same
probe or material witness. Unused candidates must not reduce the residual by
moving farther away. Pose-only costs retain static geometry, without speed or
force inferred from a pose. Arm radii are shared conservative proxies, not new
Isaac colliders. Archived configs missing `physics_loss_version` resolve to
`legacy`; changing the version, geometry motion, sampling, or temporal bound
on `--resume` is rejected. Use a new `--init` run with fresh training state.

## Archived differentiable v4 contact losses

Archived v4 phase-2 training used `CostConfig.contact_scoring_mode="isaac_v4"` via
the earlier `phase2_cost_config` and `scripts/train_phase2.py`. Archived phase-2 resumes
whose saved cost lacks that field resolve to `legacy` automatically; changing
contact mode on `--resume` is rejected because its optimizer/AL prices refer
to the old objective. Initialize a new training run from weights instead.
At that time fresh terminal-pose runs also defaulted to v4; archived terminal-pose
resumes still keep their saved contact mode. When a
v4 phase-2 run loaded a legacy terminal-role checkpoint, its pose contact
geometry is upgraded to v4 and its old hit/collision AL prices are reset;
restoring a legacy role AL state into that new objective is forbidden.

The archived v4 trajectory FusedCost used the same physical tip plus four shaft probes and
their individual radii as the collision model. For cymbal/hi-hat target-drum
exemptions it requires either a per-probe top-face sweep over the full physical
disk, latched only within its proxy contact episode, or a downward upper-side
contact in the first5 mm. A pre-contact probe approach must satisfy
`v_tangent <= 1.5 * v_normal`; target collision remains priced outside the
near-hit window, on deeper side contact, or after side re-entry without top
certification. Because differentiable kinematics has no PhysX forces, body SDF
within3 mm proxies contact; this is a surrogate, not a claim of exact physics.
At the terminal knot, soft-min distances to the union of top and side regions
provide position gradients, and normal impact speed uses the same shape-1
Gamma quality `x*exp(1-x)` with `x=v_normal/v_requested`, adding a deficit
below quality0.25. Far-side top-face contact is legal in the v4 hit-region
term; the old near-side aim remains only a soft preference. The separate
pose-only terminal-role cost uses static probe top/side geometry, without
inventing a force, incidence, or speed from a single pose.

## Current v4 hand-plan reward prediction

For live neural hand-assignment playback, commit all choices for four
consecutive hit groups and retain them through the fourth group deadline.
Re-score only for the next non-overlapping block; do not revise provisional
future hands after each hit. Use `--hand_plan_commit_mode block` by default.
The decoder/PID still execute each goal normally. Preserve the prior receding
playback as an archived comparison, not the active policy.

The current hand assigner predicts the **absolute Isaac reward** for an explicit
four-hit-group hand plan. Score each mechanically legal plan with the shared
candidate network and choose the highest predicted reward. Its completed first
fit used pointwise mean squared error only. The current follow-up adds
calibrated pairwise and soft-winner ranking terms while retaining reward MSE.
One executed plan supplies a valid training label even when the other plans for
that case have not been run. Never treat an unexecuted plan as a failed plan.

The completed v4 dataset is
`runs/hand_plan_dataset_v4_prefirst_200k_20261009`: 6,511 whole rhythm/start
cases, 61,019 executed plan rollouts, and 138,981 factual pre-first-hit
continuation rows, exactly 200,000 plan-specific rows. Each continuation row
uses the measured q/qd/tips before first credit, shifted deadlines, the same
executed assignment, and remaining reward after prefix costs. Keep all rows
from one source case in the same 80/10/10 split; only training cases refresh
the starting-state bank. Do not mix v2/v3 outcome labels into this dataset.

The completed first absolute-value fit uses the depth-two shared-candidate scorer with
hidden width 448, group/state widths 128, dropout 0.15, AdamW decay 0.1,
batch 256, seed 20261010, and 100,000 updates. Warm up for 100 updates to
3e-4, then cosine decay to 3e-5. Sample observed full-plan and continuation
rows uniformly; minimize scalar reward MSE. Log full/continuation reward RMSE,
bias, and selected-plan hit rate/regret on fixed whole-case splits. Held-out
outcomes are read-only and never select weights or control fitting; report the
latest fixed-budget weights. The decoder stays frozen and this is not phase-2
decoder training.

The follow-up dataset manifest is
`runs/hand_plan_value_v4_plus_v2_1152947_20261009/manifest.json`: it includes
the 200,000 v4 rows plus 952,947 earlier v2 executed-plan labels (937,662
legacy training rows). These v2 rewards have different cymbal/hi-hat grading.
Use a v2 auxiliary reward scorer with shared drum/hit/group/state encoders and
a separate v4 scorer for inference; never fit the v4 reward head directly to
v2 targets. Keep each source's whole-case validation/test labels read-only.
The new run initializes from the completed v4 value weights and uses fresh
AdamW state, 20,000 updates, batch256 with half v4 and half v2 pointwise rows,
64 whole training cases per update split equally across runtimes, and100-step
warmup to1e-4 followed by cosine to1e-5. Optimize equal-weight runtime MSE
plus0.05×calibrated pairwise KL and0.01×soft-winner KL (temperatures0.1 and
0.05 reward units). The ranking terms apply only to observed plans of the
same rhythm/start case; singleton cases contribute zero. These KL terms vanish
when predicted rewards match actual rewards, preserving the absolute scale.
Evaluate v4 and v2 separately; only v4 scores choose hands. Do not use held-out
traces to select weights or alter the fixed fitting budget.

The archived candidate-aware reranker trial started from the completed20,000-update
mixed-runtime value checkpoint. Freeze that entire value model. Embed each
legal candidate's frozen representation, reward estimate, and four assignment
bits, then apply two128-wide, four-head self-attention layers with dropout0.1.
Mask illegal candidates; do not give the unordered candidate list positional
encodings (the beat order is already represented inside each candidate).
Separate v4/v2 correction heads output mean-centered legal-plan residuals,
both initialized to exactly zero so update0 reproduces the frozen model.
Use the same mixed manifest, train-only row/case sampling, pairwise0.05 and
winner0.01 auxiliary losses, batch256, 64 ranking cases, and20,000-update
warmup/cosine schedule as the mixed value run. Evaluate factual continuation
rows with the full legal candidate set, not as singleton attention inputs.
Compare actual latest v4 held-out reward RMSE, plan regret and hit rate against
the exact frozen initialization; held-out data must not alter fitting.
This trial was stopped at the user's direction after its15,000-update saved
evaluation (16,380 updates attempted); do not use its weights as the base for
the next prior experiment.

The archived human-prior experiment returned to the completed mixed-runtime value
model, without candidate attention. Sample fresh IID four-group rhythms and
pair them with independently chosen **training-only** v4 physical starts on
every update. Half the collocations may have an inserted snare/hi-hat motif,
either consecutive single-drum groups or a simultaneous pair. A soft heuristic
prefers left-snare/right-hi-hat by0.06 reward units per wrong assignment and
fewer changes of each hand's drum by0.006 reward units per switch, including
the known starting hand targets. Only legal plans participate. Penalize squared
one-sided violations of the resulting plan reward-gap margins; satisfied
margins and equal-prior pairs add no loss. Apply this to the v4 head only, with
dropout off but autograd on. Use an independent collocation RNG on resume so
the supervised v4/v2 sampling stream is unchanged. Coefficient0.0015 was set
from a training-only gradient probe to contribute roughly6% of the v4 MSE
gradient; preserve the mixed-runtime supervised MSE/pairwise/winner terms,
architecture, optimizer schedule, and fixed whole-case held-out splits.
This absolute-reward prior run was stopped at the user's direction, with its
last saved checkpoint at7,500 updates; the no-prior control completed.

The most recent trial returned to the **two-layer shared-candidate ranking model** on
`runs/hand_plan_dataset_iid_train100k_fresh_20261008` (952,947 observed v2
plan rollouts; 100,000 train and819 each validation/test rhythm/start cases).
The trial resumed the archived `neural_hand_plans_100k_wide_wd01_dropout15_20261008`
66,000-update weights/optimizer/RNG to the original100,000-update cosine clock.
Keep its winner-plus-pairwise ranking objective, architecture, dropout0.15,
AdamW decay0.1 and read-only archived reference evaluation. Apply the same
human priors to fresh IID collocation rhythms paired with train-only physical
starts, but in **unitless ranking-score margins**:0.5 per wrong canonical
snare/hi-hat hand and0.05 per hand drum switch, coefficient0.005 after a
training-only probe showed about6% of the normal ranking gradient. Both
simultaneous and consecutive motifs count. Use independent prior RNG and
dropout-off prior forward to preserve the supervised sampling/RNG stream.
It also ran a matched no-prior continuation from the same checkpoint and
compared both at equal saved update counts. The v4 all-plan dataset is read-only for evaluating how
archived v2 ranking transfers under the newer cymbal scoring. Never train the
v2 ranker on v4 held-out outcomes or select weights using them. The frozen
archived66,000-update ranker scored83.1% clear-pair accuracy on its v2 test and
76.8% on the v4 test, with38.4% v4 selected-plan hit rate.
At the user's explicit stop, the prior run's last saved weights were at85,000
updates and the no-prior control had completed100,000. Do not resume either
without a new user request. In a read-only v4 test of653 cases the prior85k
had38.21% hit rate and0.06275 reward regret versus control100k38.39% and
0.06095; the different update counts limit causal interpretation. On a fixed
hi-hat/snare/hi-hat/snare rhythm with256 train-bank starts, prior and control
selected different plans70 times; matched-slot v4 Isaac outcomes were701/1024
hits for prior and685/1024 for control. Identical assignments still varied in
hit count2/186 times across fresh scenes. The ready-start version led all
three rankers to LRLR (3/4 hits); among16 legal plans LRRR reached4/4 while
fully canonical RLRL reached1/4. Preserve these limitations when reporting.

## Neural hand-plan ranking (archived experiments)

The active assignment-learning method is a neural network with16 output scores
for all binary four-hit-group hand plans. The user explicitly replaced chart
parameter fitting with direct neural ranking; neural scores need not be times.
Default trial:256 fresh random rhythms/update,10 updates executing all16 legal
plans,20 executing the neural top8,10 executing the neural top4. Execute each
legal selected plan once; LH-floor remains forbidden. Use best-observed-plan
emphasis plus observed-pair ranking. Unexecuted plans are missing labels, never
failed plans. Keep the decoder frozen and physics250 Hz. Inputs include all17
arm/waist joint positions and velocities, both Cartesian tip positions in the
kit/env frame, and previous hand targets. Use80% measured post-hit starts,
15% collision-clear randomized arm/waist poses,5% ready controls. Save q/qd/tips
40 ms after successful contact and refresh a bounded per-hand/drum state bank.
Every compared candidate of a rhythm must receive exactly the same initial
position/velocity. Use simulator-reported stick geometry for tip observations
and starting-pose clearance; verify installed q/qd and Cartesian tips match.
Register TensorBoard. Report actual latest neural weights; held-out outcomes
must never select weights or enter replay. This is supervised assignment
ranking, not phase-2 decoder training.

Neural assignment training/test first-hit deadlines are randomized, not fixed
at1s: draw the first delay from the same timing tier as the following gaps
(20–120ms fast,120–320ms medium,320–900ms slow,20ms grid). Use
`--first-hit-policy tier_random`. Preserve the timing policy on resume and use
matching held-out outcomes; fixed1s label caches cannot monitor random-first
training as its primary test. Bootstrap hits used only to collect physical
starting states may retain a generous lead; those are not ranker labels.

The current all16 ablation is explicitly requested to use40 updates of every
legal plan throughout (`--stages 40 0 0`), with the same seed, varied-state mix
and optimizer budget as the completed curriculum. Zero-duration stages skip
that candidate count. Compare both latest trained models on matched physical
starting states/rhythms; do not resume from the curriculum's learned weights.

The current dataset-first experiment uses IID stochastic four-group windows:
uniformly draw each of28 single/two-distinct-drum sets and independently draw
each interval (including the first) from20–900ms in20ms ticks. Do not force
transition-pair templates or balance whole-rhythm timing tiers. User selected
32 batches of256 cases (8,192 cases), testing every legal assignment. Split
whole cases80/10/10; all assignments of a case stay in one split. Fit the fixed
training split for20,000 supervised optimizer steps, preserving untouched
validation/test labels. Then initialize small top8/top4 refinement from latest
supervised weights and optimizer, with train-only replay. This supersedes the
unlaunched tier-random restart plan; prior completed runs stay archived.

The current more-data comparison adds another32×256 IID cases as training-only
data, preserving the original819 validation and819 test cases exactly. Combined
dataset16,384 cases has14,746 training cases. Fit a fresh same-seed model with
the same20,000-step/batch256 optimizer budget to isolate data quantity; old
held-out observations must not occur in the added training pool.

The active stronger-regularization experiment supersedes the queued20,000-step
more-data fit: use the completed16,384-case dataset (14,746 train; original819
validation and819 test unchanged), fresh same-seed weights, AdamW weight decay
0.1, batch256,100,000 supervised steps,100-step warmup to3e-4 then cosine to3e-5.
Keep this dataset fixed throughout; online refinement is not part of this run.
Every1,000 steps log legal-plan softmax entropy (temperature1), entropy divided
by log(legal plan count) excluding forced singleton cases, absolute matrix-weight
entropy, parameter L2/RMS, split losses, ordering, hit rate and reward regret.
Use latest fixed-budget weights; held-out traces do not select checkpoints or
control training. Grokking requires evidence of delayed held-out improvement
after training fit stabilizes; an entropy change alone does not establish it.

The next user-requested regularization trial increases AdamW weight decay to0.3
and adds15% dropout after each of the two hidden GELU activations. Start fresh
with seed20261010 on the same fixed16,384-case dataset and100,000-step/batch256
schedule. Disable dropout for evaluation and hand selection; preserve its
configuration and RNG on resume. Retain the completed decay0.1/no-dropout run
as a reference. Log both normalized absolute-weight entropy and the paper's
unnormalized -sum |w| log |w| statistic, per matrix and total, alongside the
existing diagnostics. Held-out data remain read-only and never select weights.

The user next requested shared hit-group encoding and explicit candidate-plan
inputs. Train a fresh `shared_candidate` ranker on the same cached136-value
observations: shared16-dimensional drum embeddings and shared per-hit encoder,
then arrange hits as left/right using each candidate's four assignment bits.
Apply a shared64-dimensional group encoder to all four groups and a64-dimensional
initial-state encoder retaining all56 state features. One shared scalar scorer
over the ordered four group encodings plus state uses hidden widths224/112;
evaluate all16 candidates, preserving existing masks and ranking objective.
106,641 parameters versus105,488 for flat reference. Same fixed dataset/splits,
seed, decay0.3/dropout0.15 and100k-step/batch256 warmup/cosine budget. CUDA training
is allowed; preserve CUDA RNG/device on offline resume. Keep the flat trial
running/completed as reference and compare read-only traces at matched steps.

## Current input-gradient regularization trial

The user explicitly requested using the existing16,384-case dataset for this
offline regularization comparison, overriding fresh-label collection for this
trial only. Keep the legacy labels/splits unchanged and preserve legacy runtime
provenance; this does not authorize mixing corrected-runtime labels or claiming
corrected-controller performance. Fresh same-seed shared_candidate weights,
hidden224, decay0.3/dropout0.15,100k steps/batch256 with the existing100-step
warmup/cosine schedule. Add coefficient0.1 times input-Jacobian L1 of scores
centered over legal assignments. Uniformly sample one legal score row per case
and multiply its derivative L1 by legal count; use64 training cases/update.
Disable dropout for the regularizer without disabling autograd. Differentiate
physical continuous coordinates only: four intervals (20ms scale), present
loudness slots (0.02),17 q (1degree),17 qd (0.1rad/s). Chain interval changes
through all subsequent deadlines and joint changes through Cartesian tip FK.
Cache input-only FK Jacobians; preserve the independent plan-sampler RNG on
resume. Log ranking/gradient terms, per-coordinate-group sensitivity and fixed
read-only split probes. Exclude absent-hit slots and forced singleton cases from
the near-zero-derivative diagnostic. Compare with the existing shared-candidate
reference; held-out observations never train, choose weights or control fitting.

## Current model-size × gradient-penalty ablation

The user requested a larger model and a capacity ablation after the input-
gradient trial. Use the same existing16,384-case legacy dataset/splits. Cross
shared_candidate scorer widths224/112 (106,641 parameters) and448/224
(254,481) with input-gradient coefficients0 and0.1. Reuse the completed small
unregularized run and the active small gradient run; train both large conditions
from fresh seed20261010 weights for100k updates/batch256. All optimizer, dropout,
weight-decay, physical derivative scales and64-case penalty sampling settings
stay matched. Run read-only input-gradient diagnostics even when its coefficient
is zero; these must not change training parameters, RNG or dropout behavior.
Keep the small run active. Capacity references must match gradient configuration.
Register both large runs in TensorBoard. Compare all four at common update
budgets, separately retaining each run's actual latest metrics; never select
checkpoints or change fitting using held-out results. The existing-label override
extends to this offline ablation only; no corrected-controller labels are mixed.

## Fresh100k hand-assigner fit

The user requested a new hand-assigner run after completing fresh data collection.
Use `hand_plan_dataset_iid_train100k_fresh_20261008`:100000 training cases and
819 validation/819 test inputs freshly relabeled with corrected runtime. Keep
all splits fixed. Fresh shared_candidate scorer448/224 (254481 parameters),
seed20261010, decay0.3/dropout0.15, gradient L1 coefficient0.1 and64-case
sampling,100k updates/batch256,warmup100 to3e-4 cosine to3e-5. Use fresh weights
and optimizer; archived legacy labels/replay/optimizer must not enter this fit.
Decoder remains frozen; this is supervised assignment ranking, not phase2.
Benchmark the previous large-gradient model's prescribed latest100k weights
on the SAME fresh held-out labels, verifying identical held-out observations
and legal masks. This is read-only policy transfer: record previous training
runtime separately from evaluation runtime. Do not compare old-runtime hit
rates directly with new-label rates. Frozen reference only supplies plotted
metrics; held-out data never control updates, checkpoints or budget. More data
and runtime corrections both changed, so report their combined effect. Enable
TensorBoard and preserve reference snapshot/hash plus all RNG state on resume.

## Transition-chart learning

When learning transition-chart parameters, update the actual three timing anchors
per eligible hand/drum transition in seconds. Report update sizes in milliseconds
and export the changed timing chart. Do not replace timing updates with a separate
unitless plan-cost layer unless the user explicitly requests that representation.
Transition-chart preference trials currently use 256 fresh random rhythms per
update, executing the top eight legal assignments under the current chart
(all legal assignments when fewer than eight exist). Compare every observed
candidate pair, averaging pairwise loss within each rhythm; never execute
duplicate padding or mechanically forbidden assignments.
Hand-assignment scoring uses timing estimates and available slack only. Do not
add historical support/trial confidence or missing-calibration penalties.

## Training dashboards

Use one shared TensorBoard server on port 6006. Reuse it for every run; do not
start per-run servers or fall back to other ports. If 6006 is unavailable,
diagnose the shared server rather than choosing another port.

Whenever launching a training run, enable TensorBoard output, register its
event directory under `runs/tensorboard_active`, and verify that the shared
TensorBoard server on port 6006 lists the run. Keep existing registered runs
visible unless the user explicitly asks to remove them. Disposable startup
smokes may opt out with `TB_AUTO_REGISTER=0`.

For all phase-2 training runs, use the critic with Bellman enabled.
Phase 1 uses supervised imitation without critic/Bellman.

## Default phase-2 optimizer schedules

In this project, **"phase-2 from scratch"** means initialize the decoder from
the completed supervised phase-1 weights and start a new phase-2 run with
fresh optimizer, AL, critic, Bellman replay, RNG, and learning-rate schedule.
It does not mean random decoder weights. Use random initialization only when
the user explicitly requests it. Do not substitute an already-trained
phase-2 checkpoint for a from-scratch run.

For every new phase-2 objective, unless the user specifies otherwise, use
AdamW with this schedule. This includes runs initialized from already-trained
decoder weights when the contact, physics, or other training objective changes;
the checkpoint source alone does not make such a run a fine-tune:

- Updates 1–300: linear learning-rate warmup to `1e-4`.
- After warmup through update 10,000: cosine decay to `5e-6`.
- After update 10,000: SaLSa transfer line search takes sole control of the
  learning rate, starting from the final cosine rate.

Enable `--line-search --line-search-cosine-before-activation`, with
`--warmup-steps 300 --lr 1e-4 --min-lr-frac 0.05 --lr-horizon 10000
--line-search-activate-after-updates 10000`. Use `salsa_transfer`, fixed search
interval 3, acceptance beta `0.99`, Armijo c1 `0.01`, and at most 6 trials per
search. Select the rate from the current gradient-only probe. Reject a
current-probe increase and skip updates when the probe budget is exhausted,
but do not veto a finite AdamW momentum update because its batch loss rises.
An executed-step loss may be logged as a diagnostic only.

Only for a same-objective fine-tune of an already-trained phase-2 decoder,
start SaLSa transfer line search on the **first optimizer update**. Set
`--line-search --line-search-mode salsa_transfer
--line-search-activate-after-updates 0
--no-line-search-cosine-before-activation
--line-search-warmup-updates 10
--line-search-retry-retained-rate`; give the selector a logged initial epsilon
seed and retain the fixed interval 3, beta `0.99`, c1 `0.01`, at most 6 trials,
rejection of current-probe increases, and skipping updates when the probe budget
is exhausted. Keep `--no-line-search-strict-actual-update`: SaLSa selects the
rate on the gradient-only probe, and AdamW momentum may make the executed
step increase the current batch loss. Do not run the 300-update learning-
rate warmup or 10,000-update cosine before activating line search in a
fine-tune. Verify that the first update actually searches, records its selected
rate, and does not use the executed-step loss as an acceptance veto. If one
stochastic batch exhausts its probe budget, skip that batch and retry the next
from the last accepted rate; do not turn a failed tiny trial into the global
starting rate for future batches. The 20261011 v4 probe-only run still accepted
near-no-op rates by update450 despite this retry rule. Monitor the selected
epsilon and actual parameter movement; stop and diagnose a numerical-floor
collapse rather than counting subsequent attempted updates as learning.

If the contact, physics, or other training objective changes, treat it as new
training rather than fine-tuning: launch a new run with `--init` from the
selected weights and use the warmup → cosine → line-search schedule above.
Optimizer, AL, critic, Bellman replay, RNG, and line-search selector start
fresh under that objective. Changing contact mode on `--resume` is forbidden.
For a same-objective resume, preserve
the model, optimizer, AL, critic, Bellman replay, RNG, and selector state, and
keep its absolute schedule clock. Fresh-training resumes must not repeat warmup
or restart the cosine; verify cosine no longer overwrites the selector after
its scheduled handoff.
