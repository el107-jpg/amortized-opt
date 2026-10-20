# Learned arm assignment

`scripts/train_arm_assignment.py` trains a small MLP above the frozen
trajectory generator. Every drum, including floor tom and hihat, can go to
either arm. There are **no drum eligibility masks or fixed assignments**.

## Policy and execution

### Rhythm and starting-pose variety

`arm_assignment_balanced_diverse_20261002` resumes balanced batch 7, including
Q/target weights, Adam and the balanced replay. It rebuilds its candidate window
pool with an iteration-specific seed each batch. Windows with identical drum,
relative-time and velocity sequences are deduplicated across source clips.
Entropy fitting caps each unique candidate's probability at 0.002 (about four
expected draws in 2,048 environments). Exact equality can be infeasible under
this diversity cap; the initial five-second check gives expected drum shares of
13.7–14.6%. `sampling/unique_source_clips`, `sampling/unique_note_patterns` and
`sampling/effective_candidate_windows` expose actual concentration. Notes and
timing inside each window remain unchanged.

Starting positions are sampled independently from the upcoming rhythm:

- 20% exact nominal ready pose;
- 20% perturbed ready pose;
- 60% perturbed poses from actual training execution states.

The initial anchor pool contains 10,144 poses from the late-commit training
replay, selected with equal candidate budgets per next-drum label and deduplicated
on a 0.04 rad joint grid. Validation observations are not used. Position noise
is Gaussian with standard deviations 0.08 rad for arm joints, 0.15 rad for
waist yaw, and 0.02 rad for waist pitch/roll, clipped at 2.5 standard deviations.
Candidates must satisfy physical joint bounds, pitch/roll within 0.08 rad and
at least 3 mm clearance under the existing analytic collision proxies, including
cross-stick clearance and without strike/departure exemptions. Rejected noise
is resampled; exhaustion falls back to the checked anchor, not an unchecked pose.
Starting velocities are zero. Later decisions observe actual Isaac velocities.

The original task's reset receives each pose through `ready_joint_pos`, so its
PID/FOH histories, actuator targets and strike trackers reset consistently.
Nominal evaluations restore the original ready pose. `starting_pose/*` reports
mixture proportions, angle spread, rejection and fallback rates.

Every ten batches, a single Isaac evaluation pass splits environments into
three cohorts: the original held-out rhythm at nominal ready, that rhythm with
the start-pose mixture, and uniformly sampled windows from eligible eval-split
rhythms with the start-pose mixture. A fixed independent evaluation RNG makes
checkpoints comparable without changing training draws. These trajectories do
not enter replay. `dqn_eval/learned_hit_fraction` keeps the original nominal
rhythm metric; `dqn_eval/random_start_hit_fraction` and
`dqn_eval/varied_rhythm_random_start_hit_fraction` measure broader coverage.
Per-drum metrics and `validation_group` labels accompany reports.

```bash
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /data2/eric/conda_envs/g1drum/bin/python -u \
  scripts/train_arm_assignment_parallel.py --headless --device cuda:0 \
  --config runs/arm_assignment_balanced_20261002/config.json \
  --resume runs/arm_assignment_balanced_20261002/dqn_before_diversity.pt \
  --run-name arm_assignment_balanced_diverse_20261002 --envs 2048 \
  --window-sampling balanced --refresh-windows --deduplicate-windows \
  --max-window-probability .002 \
  --start-pose-pool runs/assignment_start_poses_training_20261002.npz \
  --gradient-steps 1024
```

Normal resume uses this new run's `config.json` and latest checkpoint. Position
and rhythm augmentation configurations are stored there and in DQN checkpoints.
The frozen trajectory planner, executed-contact credit, constraints and Bellman
critic remain active. This coverage expansion must still be assessed on the
held-out cohorts; it does not establish generalization to arbitrary rhythms.

### Balanced drum exposure

`arm_assignment_balanced_20261002` continues the late-commit Q network, target
and Adam state with entropy-regularized sampling of intact five-second windows
from the accepted training split. The candidate pool contains up to 64 seeded
starts per clip. Weights target equal **physical requested hit counts**, including
both drums in simultaneous pairs, rather than equal numbers of clips containing
a drum. Notes, velocities, simultaneous hits and inter-hit gaps remain intact.
The held-out rhythm is not part of this pool.

The initial corpus check gives expected shares of 14.2–14.4% for each drum,
instead of approximately 50% hi-hat and 1.75% floor tom. `window_sampling.json`
records expected shares and pool diversity; TensorBoard `sampling/<drum>_share`
and `sampling/<drum>_requested` report each actual batch before collection.
Metrics also record actual `requested_<drum>` counts after execution.

The continuation clears the saved, imbalanced replay so Bellman updates immediately
use the new distribution. The old run and its replay checkpoint are preserved.
Q weights, target, optimizer, update counters and window RNG resume normally;
an exploration offset preserves epsilon when replay is cleared. All subsequent
resumes restore the balanced replay normally (omit `--reset-replay`).

```bash
CUDA_VISIBLE_DEVICES=3 /data2/eric/conda_envs/g1drum/bin/python -u \
  scripts/train_arm_assignment_parallel.py --headless --device cuda:0 \
  --config runs/arm_assignment_late_commit_20261002/config.json \
  --run-name arm_assignment_balanced_20261002 --envs 2048 \
  --window-sampling balanced --reset-replay --gradient-steps 1024
```

### Correction: separate commitment from lookahead

`arm_assignment_late_commit_20261002` keeps eight beats of rhythm context but
chooses immutable arms only when a hit is **one second away**. Pending choices
are visible in 51 slots per arm, covering the maximum possible count on the
20 ms control grid. Overflow raises an error instead of silently dropping
committed choices from the observation. Future context has up to 32 unassigned
hits within eight beats. All drums remain available to both arms.

The old run is preserved through collection batch 46 in
`dqn_before_late_commit.pt`. Migration copies the common state/geometry/history
weights and relocates learned right-arm pending and future input weights to
their new feature offsets. Extra pending inputs start with zero weights.
The network remains two 64-unit tanh layers (96,962 parameters with this kit).
The incompatible old replay and optimizer are reset; the learned Q network is
retained and the target is initialized from it. No graph warm start is repeated.

This continuation collects 2,048 environments per batch, uses 1,024 Bellman
updates per batch, a 500,000-transition replay buffer, and decays exploration
from 0.30 to 0.05 over 250,000 stored decisions. TensorBoard additionally logs
same-instant, zero-reward and initial-state decision fractions to verify that
the excessive early assignment chains are reduced. Its event directory is
registered separately, preserving the earlier run for comparison.

```bash
CUDA_VISIBLE_DEVICES=3 /data2/eric/conda_envs/g1drum/bin/python -u \
  scripts/train_arm_assignment_parallel.py --headless --device cuda:0 \
  --resume runs/arm_assignment_mlp_20261002/dqn_before_late_commit.pt \
  --run-name arm_assignment_late_commit_20261002 --envs 2048 \
  --commit-lead-s 1 --gradient-steps 1024 --replay-capacity 500000
```

After its first launch, resume using `--config
runs/arm_assignment_late_commit_20261002/config.json`; the initial or latest
checkpoint is selected automatically. Normal resume restores replay, target
and optimizer. Saved policy configuration also applies the later commitment
window during playback.

### Earlier continuation: Double DQN and early eight-beat commitment

`scripts/train_arm_assignment_parallel.py` resumes the existing learned
assignment policy with Double DQN and 2,048 simultaneous Isaac environments.
The two 64-unit layers feed an unrestricted two-action Q critic. The expanded
input has 32 future strike slots and a queue horizon of **eight beats**,
converted to seconds using each RCC clip's `metadata.original_bpm` (120 BPM
fallback). This is two bars in 4/4, 3.33 seconds at 144 BPM. Joint state,
geometry and pending assignments remain part of the observation. Temporal
features retain their original two-second normalization so transferred weights
keep the same meaning. New future-input weights initially equal zero.

The PPO checkpoint is preserved in `ppo_before_parallel_dqn.pt`. Conversion
copies the shared trunk and initializes Q with the old value estimate plus a
small centered actor-logit advantage, preserving greedy arm preference.
These initial Q magnitudes are an initialization, not calibrated returns.
The original 20-step graph warm start is not repeated.

Replay holds up to 50,000 event transitions. Each collection iteration gathers
2,048 five-second training windows concurrently, then runs **16,384 Bellman updates**
at batch size 128. Double DQN uses the online network to select a next action
and a lagged target network to evaluate it. Targets use the same discounted
executed interval rewards, elapsed-time discounts and zero terminal bootstrap.
Adam learning rate is 0.0003, Huber loss, gradient norm limit 1, target update
fraction 0.01. Epsilon exploration declines from 0.30 to 0.05 over 20,000
collected decisions. No drum eligibility masks or imitation penalties apply.
The default update count is eight per environment, preserving the replay
training ratio when collection is scaled. `--gradient-steps` overrides it.

Each environment retains its own committed reference. Only rows with a new
goal enter the frozen trajectory decoder, with one pass and a fresh clean
initialization. The original task still runs the physics, contacts, strike
trackers and hit-credit computation. Unused task policy observations and
playback diagnostic traces are skipped. PID, waist reaction feedforward and
contact compliance match the existing player. Validation uses the held-out
punk rhythm every ten collection iterations.

```bash
CUDA_VISIBLE_DEVICES=3 /data2/eric/conda_envs/g1drum/bin/python -u \
  scripts/train_arm_assignment_parallel.py --headless --device cuda:0 \
  --envs 2048 --lookahead-beats 8 --future-hits 32 --iterations 100
```

This automatically resumes `dqn_last.pt` if present, otherwise the preserved
PPO checkpoint. New artifacts are `dqn_config.json`, `dqn_metrics.jsonl`,
`dqn_last.pt` (including target, optimizer, replay and random states),
`dqn_last_rollouts.json` and `dqn_eval_*.json`. TensorBoard continues in the
same registered run with separate `dqn/*` and `dqn_eval/*` tags. `iteration`
counts collection batches; `gradient_updates` counts actual Bellman optimizer
steps. `collection/control_tick` and `collection/iteration` refresh during
collection. Playback accepts `--assignment_checkpoint .../dqn_last.pt`.

The 2,048-env feasibility check used varied one-second rhythm windows and
the existing Isaac physics and executed-contact hit detector. It completed
233,524 environment steps in 156.42 seconds (1,492.92 steps/s), versus
163.30 steps/s in the same 64-env benchmark. Sampled total GPU usage peaked
at approximately 8.4 GB; PyTorch peak reserved memory was 3.91 GiB and host
RSS 6.19 GiB. No unexpected resets or nonfinite sampled joint states occurred.
This short check does not establish long-run hit performance.

Startup takes several minutes because the original scene spawns each drum kit
separately. Per-hit host geometry scoring took 57.5 seconds of the benchmark.
Larger batches improve experience throughput while taking longer per collection
iteration. For a disposable scaling benchmark, use `--benchmark-output
/absolute/result.json` with `--envs N` and optional `--benchmark-window-s 1`.
It collects varied training episodes without optimizing or overwriting the
policy. Training logs `dqn/peak_torch_allocated_gib` and
`dqn/peak_torch_reserved_gib`; these exclude Isaac GPU allocations outside
PyTorch.

The sections below document the original PPO run and warm start.

The actor and value critic share two 64-unit tanh layers (21,507 parameters
with the current seven-drum kit). The observation contains measured joint
positions/velocities, FK stick tips in the pelvis frame, drum geometry,
previous assignments, two pending hits per arm, and twelve upcoming hits with
deadlines and velocities. Assignments happen as groups enter a two-second
lookahead queue. Decisions remain fixed through execution.

An action selects the arm for the first drum in a group; a second simultaneous
drum uses the other arm. This represents the two physical arms, without
restricting which drums they play. Groups are built on the 20 ms control grid.
Clips with more than two strikes on a single tick are excluded and listed in
`config.json`; no strikes are silently dropped. The current train split has
490 accepted and 72 excluded clips.

The queue writes identical choices into `BatchedSchedule` and the original
Isaac environment's beat-credit table. It does not pass the choices through
`load_rcc`'s default eligibility reassignment. Context encodings are invalidated
when new futures are assigned; the committed physical plan is retained until
its normal goal boundary.

## Training

1. Exactly **20 optimizer updates** of graph behavior cloning, batch size 256,
   learning rate 0.001. Labels use `hand_assign.assign_simultaneous` with its
   existing graph preferences. They are an initial bias, not a runtime rule.
2. Start a fresh Adam optimizer for PPO at 0.0003. No imitation loss or
   persistent penalty for departing from the graph. Entropy weight 0.02 keeps
   exploration active. The separate assignment value head is trained with
   discounted Bellman/GAE targets.
3. Four complete Isaac episodes per PPO update, sampled from random five-second
   windows of training clips, with a one-second ready lead and a 0.6-second
   scoring tail. The generator, PID, feedforward and contact compliance remain
   frozen. Sampling windows gives varied measured states at later assignment
   decisions; the warm-start labels themselves use the ready pose.

Main reward: +10 per strike credited by the existing Isaac detector, -5 per
expired/unresolved strike, -2 per net stray strike. Small shaping is +0.2 times
bounded legal-region proximity, once at each deadline. Joint excursions,
unsaturated torque demand above actuator ratings, and new plans' analytic
collision violations receive smaller penalties. Physics losses and actuator
limits in the existing controller are unchanged. Collision penalties here are
analytic plan diagnostics, not a newly implemented physical body-contact sensor.

Returns account for variable time between decisions using
`gamma_per_second ** elapsed_seconds` (gamma 0.99). Multiple decisions at the
same tick have unit discount. Episodes finish after all scoring windows, with
zero terminal bootstrap. This prevents a truncated tail from evading misses.

The existing punk eval clip is held out from optimization. Graph and warm-start
policies are evaluated before PPO; deterministic learned policies are evaluated
every ten updates. Saved reports include per-drum success and actual assignments.
`best_training_return.pt` selects training return only, not validation performance.

## Run

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_arm_assignment.py \
  --name arm_assignment_mlp_20261002 --device cuda:0 --iterations 100
```

Defaults use `runs/phase2_clean_lines_fromp1_20261001/last.pt`, one decoder pass
per goal segment, fresh clean terminal initialization, PID Ki 5, windup 0.25,
`fit_kd4` reference lead, waist reaction feedforward and contact compliance.
`--dry-run` resolves the setup without creating a run or launching Isaac;
`--pretrain-only` saves the 20-update warm start without launching Isaac.

Outputs: `runs/<name>/config.json`, `pretrain.json`, `pretrained.pt`,
`metrics.jsonl`, `last.pt`, periodic checkpoints and `eval_*.json`. TensorBoard
events are registered under `runs/tensorboard_active`, preserving other runs.
Useful tags: `rl/hit_fraction`, `rl/value_bellman_loss`, `rl/entropy`,
`assignment/<drum>_left_fraction`, `hits/<drum>_fraction`, and
`eval/learned_hit_fraction`.

## Playback

Add the learned policy to the same player command used for the frozen planner:

```bash
python scripts/play_isaac.py --headless --device cuda:0 --model_device cuda:0 \
  --ckpt runs/phase2_clean_lines_fromp1_20261001/last.pt \
  --assignment_checkpoint runs/arm_assignment_mlp_20261002/last.pt \
  --clip_index 10 --first_hit_s 1 --steps 650 \
  --commit_plan --plan_max_passes 1 --plan_acceptance isaac \
  --fresh-terminal-every-tick --cmd_lead_s fit_kd4 --ki 5 --ki_windup 0.25 \
  --contact_compliance --no_overlay
```

The frozen generator was trained with restricted assignment coverage, and its
pose library omits left-floor/right-hihat entries. Missing library entries fall
back to measured-state seeds; the fresh terminal head then generates the
initialization. There is no action rejection. Learning may therefore reveal a
need to expand the generator's training coverage for those combinations.

## Four-beat model search (2026-10-02)

`scripts/train_arm_assignment_model.py` replaces Isaac collection with finite
model rollouts. The default generator reuses the pretrained terminal-pose MLP stored in the
current clean-line checkpoint. It directly predicts **one next-hit 17-joint pose
per call**. No transformer encoder or trajectory decoder runs. An analytic
minimum-energy contact-Jacobian solve provides the requested approach velocity;
that velocity is clipped to actuator velocity limits. Each subsequent pose call
starts at the predicted pose with sampled rebound velocity. Simultaneous hits
share one full-body pose. Inactive arms retain their measured joint positions.
The four-beat search supplies lookahead; the pose MLP itself sees the current
hit group. `--generator decoder` retains the original full-decoder extraction
baseline for comparison.

Search enumerates all 16 assignments when the window has four binary decisions.
Dense four-beat windows retain every strike and use at most 16 complete candidate
sequences, including graph initialization and randomized alternatives. A severe
simultaneous torso-frame crossover can be pruned; no individual drum is restricted
to a hand. All competing assignments use paired rebound draws for the same
physical hit. Two draws per sequence are averaged before selecting a continuation.

The small 64-by-64 Q MLP learns both root action values from measured complete
returns. It retains the previous assignment network's body, expands future
capacity to 128 hits, starts a fresh value head/optimizer/replay, and uses no
Bellman bootstrap. This overrides the general Bellman-on project default at the
user's request to evaluate short complete sequences directly. The pose
network remains frozen; only the assignment Q critic is trained.

The environment assumes perfect tracking of cubic Hermite connectors between
predicted hit poses. The pose MLP does not produce intermediate waypoints.
Existing FK contact geometry credits only an approaching contact inside the legal
hit region. A frictionless normal impulse uses a fixed nominal robot mass matrix
and energy retention sampled uniformly from 0.2 to 0.8. Collision, joint, velocity,
and RNEA torque diagnostics cover every interval at no more than 10 ms spacing.
These are surrogate results; they do not establish Isaac hit performance.

Training windows come from the 490 accepted natural training clips, with their
original tempo setting the four-beat duration, a 0.5 s first-hit preparation time,
and no trailing scored interval. Entropy weighting approximately balances actual
drum request counts. Windows are deduplicated, their probabilities capped at
0.002, and the candidate pool refreshed every ten batches. Starts mix nominal
ready, perturbed ready, and checked recorded training poses.

A default collection uses 64 different rhythm/start problems, up to 16 candidate
sequences each, and two rebound trials: up to **2,048 candidate rollouts**, not
2,048 independent rhythm problems. The `--chunk` option controls simultaneous
rollouts to fit GPU memory. TensorBoard logs `search/*`, `heldout/*`, `starts/*`,
and per-drum request/share tags. `heldout/hit_fraction` measures the best searched
sequence in the surrogate; `heldout/q_search_regret` measures the Q network's
root decision against that search. It is not a full Q-policy execution hit rate.

```bash
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  /data2/eric/conda_envs/g1drum/bin/python scripts/train_arm_assignment_model.py \
  --name arm_assignment_pose_search4_20261002 --states 64 --chunk 2048 --steps 1000
```

Outputs include `config.json`, `progress.json`, `heldout.json`, `last.pt`, and
registered TensorBoard events. Checkpoints use `arm_assignment_search_q_v1` to
identify the changed value target semantics.

`feasible_path_fraction` requires the whole searched window to stay within
1% of rated torque/velocity limits, 5 mm collision penetration, and 0.0001 rad
joint-bound tolerance. `feasible_hit_fraction` credits only geometric hits in
those complete feasible windows. These tags use explicit tolerances and are
separate from `hit_fraction`. Resume optimizer, replay and RNG with `--resume`.

## Anchored Isaac fine-tune and paired execution tests

`--close-finetune` on `scripts/train_arm_assignment_parallel.py` accepts the
`arm_assignment_search_q_v1` assignment checkpoint and creates a separate
physical-environment run. The source policy and trajectory weights are saved and
kept fixed as the behavior reference. Surrogate replay/optimizer state is reset;
only actual Isaac transitions enter the new Bellman replay.

The experiment compares graph assignments and the learned policy with both
controllers: direct single-pose MLP plus a committed Hermite reference, and the
existing one-pass decoder plus its committed reference. Both retain the existing
PID integral, reference lead, slew and joint limits, waist reaction feedforward,
low stiffness contact compliance, and original Isaac contact hit requirements.
Actual measured velocities enter subsequent pose calls. No synthetic bounce is
used in Isaac. A new future assignment does not interrupt the earliest-hit
committed pose reference.

A fixed panel of 64 natural held-out rhythm windows is replayed from the same
starting poses for every comparison. Half use nominal ready; half sample the
checked varied-start distribution. Inter-hit times stay unchanged, the first hit
is moved to 1 s, and 0.6 s remains for contact credit after the last requested hit.
Graph-unsupported windows are reported when excluded. The fine-tune uses only
training clips, with entropy-weighted physical drum counts, four-beat windows,
and randomized checked starting poses.

Fine-tuning uses Double DQN, learning rate 1e-5, 128 gradient steps per physical
collection, five collections, 2% epsilon exploration, and a KL penalty to the
frozen reference's two-action distribution (temperature 1, weight 5). An update
is rolled back, including target/optimizer state, if first-arm disagreement
exceeds 5% or mean KL exceeds 0.02 on either the fixed measured Isaac guard states
or the entire stored training replay. The learning rate is halved after a rejected
update. This bounds observed behavior drift on those states, rather than promising
identical behavior for every possible state.

`isaac_baseline_comparison.json` and `isaac_final_comparison.json` report physical
hit counts, per-drum performance, and the measured final policy drift. Final
checkpoint selection uses the last accepted training update; held-out hit rewards
do not select a checkpoint. Held-out observations participate only in the
behavior-preservation guard. TensorBoard tags include `isaac/<controller>_<policy>`
and `close_dqn/*`. Generator-call counts retain the older `decoder_passes` report
field, including direct MLP calls.
# Analytic joint-schedule improvement trial (2026-10-03)

`scripts/train_schedule_improvement.py` collects from the existing analytic
trajectory surrogate, trains a joint return critic first, then trains a
Transformer edit policy on analytically evaluated local-search moves.

- Observation: q/qd/tips (40 features), at most ten future physical hits within
  one second (16 features per hit). There is no committed-hit queue.
- Each simultaneous group is one binary hand decision; its second hit uses the
  opposite arm. No drum-to-hand eligibility restrictions are imposed.
- The default frozen decoder adapter includes the other arm's later goals. Its
  endpoint poses and velocities are connected by perfectly tracked Hermite
  curves. These are surrogate connectors, not the decoder's full generated
  path and not Isaac execution.
- Critic: two attention layers, width 64, four heads; proposed hand assignments
  are inputs; output is one complete finite-plan return.
- Critic training combines sampled complete-plan returns with target-network
  Bellman consistency between successive suffixes. Bellman is enabled. This is
  finite-plan value estimation, not a claim of an optimal continuing Q-value.
- Actor: shared single/pair edit heads and STOP; imitate bounded analytic search
  from graph, nearby, and random starting schedules. Teacher STOP is relative
  to the evaluated candidate bank, not a proof of local/global optimality.
- Rewards use the existing geometric hit and physical constraint checks,
  accumulated by event interval with gamma=0.98 per second and reward scale 20.
  Competing plans share bounce energy draws by physical hit ordinal.
- Fresh evaluation compares graph seeds, learned edits, critic-guarded edits,
  critic local search, and the best analytically evaluated candidate bank. A
  separate probe refreshes assignments after every hit, from actual surrogate
  positions and rebound velocities. All refresh methods receive the same
  physical hit times and bounce draws.

Run artifacts include `analytic_replay.pt`, `critic.pt`, `actor.pt`,
`final_report.json`, and TensorBoard events. Existing runs remain registered.

## Procedural rhythm critic continuation

`scripts/train_random_rhythm_critic.py` continues the joint critic with newly
generated one-second rhythms from `isaac_rl/random_rhythms.py`. Training uses no
recorded rhythm windows. It mixes independent drum choices, repeats and fills;
varies tempo (55–220 BPM), subdivision, swing, irregular spacing, simultaneous
distinct-drum pairs and strike strength; and retains at most ten physical hits
on the 20 ms control grid. Drum identities are exchangeable, with equal expected
coverage across all seven drums. Generation does not certify feasibility.

Complete hand schedules are sampled uniformly, augmented by graph and mirror
seeds. Critic labels are measured analytic returns and Bellman suffix targets.
Starts mix collision-clear randomized arm/waist poses and velocities with clear
moving states reached on previous analytic connectors. The frozen planner and
critic architecture are unchanged. This continuation does not train the actor.

Validation logs prediction error and bounded candidate-bank selection regret
every 500 updates, using independently generated rhythms and the prior run's
recorded evaluation panel. Checkpoint selection uses the last training update,
not validation rewards. Final causal probes compare graph assignments, the old
critic and the continued critic under identical bounce draws and starting states.
Artifacts include reproducible validation panels, `validation_history.json`,
`collection_summary.json`, `analytic_replay.pt`, `critic.pt`, and `final_report.json`.

## Schedule ranking and mistake mining

`scripts/train_schedule_ranking.py` continues the procedural critic with MC
return regression, Bellman suffix consistency, and an additional pairwise
logistic ranking loss. Preferences compare complete schedules from identical
q/qd and future requests, sharing four rebound draws. Pairs require at least
0.25 normalized return separation and at least 75% rebound sign agreement.
Misranked pairs receive extra sampling priority and weight; other pairs retain
general ranking coverage. Measured suffix returns from different resulting
states remain pointwise/Bellman examples, never same-state preference labels.

Each of four rounds collects new training states, then performs 2,000 updates.
The candidate bank includes the current critic's local-search winner and nearby
proposals, graph, graph neighbors, and random alternatives. Collection records
when the predicted winner is measurably worse than graph. Half of subsequent
roots use a state actually reached by a selected plan, keeping its remaining
requests and shifted deadlines. New cold roots mix 80% generated / 20% balanced
recorded training rhythms. Value replay sampling is 40% new, 40% previous
generated training, and 20% original recorded training data. Evaluation and
training clip sources remain separate; validation errors are not mined.

Both previous saved validation panels are monitored every 500 updates. A fresh
generated test panel is created after all training ends. Final probes compare
graph, the previous critic and the ranking critic with matched starts and bounce
draws, refreshing assignments after every hit. The actor and frozen trajectory
generator are unchanged. TensorBoard records ranking accuracy, regret, harmful
winner counts, pair loss, Bellman loss, and physical probe metrics.

## Transition-table assignment

`scripts/calibrate_transition_assignment.py` calibrates the frozen decoder,
without network training, using random near-head departure states and outward
velocities. All hand/drum combinations are allowed, including same-drum
recovery. Four waist-yaw centers produce 392 transition groups. Each of eight
starting states now checks exactly one second first, then every 20 ms down to
80 ms, followed by 1.2/1.4/1.6/1.8 second anchors. `--duration-search random`
reproduces the original sixteen-trial search. Every deadline is checked
independently; neural feasibility is not assumed monotonic in duration.
Each group's estimate is the mean of
the per-start minimum successful deadline, with support counts and standard
deviations retained. Unsupported starts do not become fabricated finite times.

Calibration checks actual decoder trajectories using the existing physics
cost, four collocation samples per segment, enabled velocity diagnostics,
and collision checks for the held idle arm throughout the motion. Success
requires legal hit-region membership, 5–45 degree stick elevation, impact
direction cosine >=0.5, speed error <=0.75 m/s at a requested 2 m/s strike,
and no collision, torque, velocity or joint-box excess beyond numerical
tolerance. Head membership now uses the exact closest legal tip/shaft contact;
the training loss's soft-weighted contact residual is logged separately.
The original random and first descending runs used the soft residual as a
gate, which can reject a cymbal shaft that already intersects the legal region.
By default the legal disk/depth bounds stay unchanged. This is an empirical model-specific estimate, not a physical
minimum-time certificate or an Isaac result.

A separate random static IK/clearance search chooses a preferred yaw for each
ordered `(left next drum, right next drum)` pair, including idle. Unsupported
pairs are retained with a large scoring penalty, without hand eligibility
masks. This table does not certify simultaneous dynamic feasibility.

`isaac_rl.transition_assignment.search_assignments` enumerates up to ten binary
hit-group assignments (1024 schedules). Vectorized table scoring constructs one
shared timeline of both hands' next goals, then duration-weights its yaw for
each hand transition. Time available spans consecutive hits of THAT hand,
including intervening hits by its partner. Waist turn and required average
turn rate are penalized once on the shared timeline, including the initial
turn from measured yaw. Unknown initial drums require first-edge estimates
computed from actual q/qd; ready poses are never treated as drum departures.

Slack cost is squared softplus, `10*softplus(-slack/0.02)^2`, plus
`100*max(-slack/0.02,0)^2`, with configurable weights/scales. It rises steeply
around the estimated minimum and is nearly flat by 100 ms extra time.
Low transition support adds a risk penalty; missing edges/pairs cost 1000.
Every winner reports its slack, yaw, support and missing-evidence diagnostics.

Example calibration:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/calibrate_transition_assignment.py \
  --out runs/transition_table_my_run \
  --samples 8 --duration-trials 16 --pair-candidates 24 --probe-windows 32
```

Artifacts include `transition_table.npz`, raw searched paths' diagnostics and
starting states under `samples/`, `transition_statistics.npz`, and fresh paired
graph/table probes. The probe uses the existing endpoint/Hermite analytic
environment; its paths differ from the actual decoder paths used to calibrate
the table. Assignment weights are initial defaults, not tuned on the probe.

To reuse the original saved starting states and audit every deadline across
four GPUs:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/audit_transition_deadlines.py \
  --source runs/transition_table_random_v2_20261004 \
  --out runs/transition_table_deadline_audit \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Workers partition saved starting states; each state still gets the entire
deadline sweep with a fresh checkpoint initialization at every duration.
The one-second report is written first. The resulting table joins new grid
witnesses with prior random-search witnesses, retaining unsupported states.
Failed one-second references do not exclude shorter successful trajectories.

Add `--deadline-search full` to the audit to test every supported deadline
from 20 ms through 2 seconds in 20 ms steps, including every intermediate
duration above one second. Each geometrically valid saved start receives all
100 independent decoder trials regardless of its one-second result. The
100-waypoint generator cannot represent a longer horizon without changing
the model setup. Saved starts outside the checkpoint's posture bounds must
be distinguished from admissible departures when interpreting coverage.

For a relaxed arrival-time calibration, append `--hit-depth-mm 70` and
`--ignore-hit-speed` to either calibration command. The legal depth then spans
the physical head surface through 70 mm inward, with the original outer face
and disk radius preserved. Impact speed remains logged, and the generator is
still asked for 2 m/s, but impact-speed error does not gate table success.
Direction, stick elevation, collision, torque, joint velocity and joint bounds
still apply. These options change only calibration acceptance; checkpoint
losses, training and runtime/Isaac hit definitions remain unchanged. The fresh
rhythm probe still uses its existing endpoint/Hermite hit definition.
Depth at the deadline is a proxy for early arrival, not a swept-contact-time
test: an early path can still fail the retained collision or other checks.

`--allow-early-target-top-entry` replaces the target's blanket final-two-waypoint
capsule exemption with a geometric entry rule for calibration. It accepts a
downward crossing through the physical head at any time, then exempts shallow
target-body contact within the selected depth (70 mm with `--hit-depth-mm 70`).
Side entry or underside entry before head contact, side re-entry after
withdrawal, excessive depth, wrists, other drums and body/cross-hand collisions
stay checked. Within the legal head-depth volume, subsequent target body/under
overlap is accepted as the intended early-hit proxy (including thin cymbals);
a physical stick would instead impact and rebound. The smaller endpoint goal
disk and playable half-plane still apply to terminal hit acceptance. Collision
entry uses the whole physical stick capsule; the bounded depth and terminal
checks retain the existing goal-contact arc (tip for pads, eligible shaft for
cymbals/hi-hat). Shaft top contact is recognized even when the tip is outside
the head. The contact path is sampled
at collocation resolution, with interpolation of the top-plane crossing;
this does not simulate an impact/bounce. Source departure exemptions stay as
before. The old timed rule could admit a late side entry, so its witnesses are
not merged into a table calibrated with this flag. Training/runtime use the
checkpoint rule unless this optional `StrikeContext` field is explicitly set.

### Short trajectory QP repair

`scripts/probe_transition_qp.py` compares a frozen decoder output with a small
budget of classical SQP repair. Each iteration solves an OSQP convex problem
over every active-arm and waist waypoint, using the true plant's autograd
linearization. The measured start pose/velocity, idle-arm joints, time grid,
and deadline stay fixed. Waist corrections are allowed: yaw labels describe
the initial yaw region, rather than an enforced constant yaw during repair.
No additional neural refinement passes or training are performed.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/probe_transition_qp.py \
  --audit runs/transition_table_full_range_top_entry_v6_full_idle_20261004/audit \
  --out runs/transition_qp_five_full_range_20261004 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --full-range \
  --iterations 5 --chunk 16 --jacobian-chunk 256
```

Omit `--full-range` for the complete one-second cohort. `--smoke` reproduces
the four pictured failures and saves both paths. Already feasible seeds are
retained; failed paths receive up to the requested number of QP steps, with
nonlinear backtracking and early termination when no step improves the merit.
The QP uses a 0.12 rad step bound, correction-curvature regularization and
elastic inequality slacks. It relinearizes the worst point for each actuator
and each collision primitive/obstacle; torque includes every physics knot,
velocity uses analytic segment extrema, and collisions include all four
samples per segment and both arms. Actual nonlinear acceptance rechecks the
entire path, not just the current linearized worst points. The 70 mm hit band,
verified early target-top entry and no required impact speed match this audit.

An OSQP iteration limit can still yield a bounded candidate; the solver
history explicitly distinguishes that from a converged QP, and the candidate
has to pass the same nonlinear check. A remaining failure is a local optimizer
failure within this budget, **not a physical infeasibility certificate**.
The source may be outside the generator manifold, the initial pose may
already violate a bound, or a different seed/longer optimization could work.

Workers partition starting states, each receiving all 100 independent
deadlines. Reports and TensorBoard progress update during execution; the
merged `deadline_050.json` is the complete one-second comparison. Completed
deadline files support resuming in the same output directory. Final outputs
include per-start minima, solver histories and `transition_table_qp.npz`.
This new table excludes departures outside checkpoint posture bounds;
all-start and within-bound metrics are reported separately. Original decoder
acceptance is compared against the saved audit to expose reproduction drift.

The v6 reference audit corrects an idle-arm horizon indexing error in earlier
calibration: spline groups are `(waist, left, right)`, while the old assignment
used a hand index and left the idle arm checked for only the first 20 ms.
Both physical arms are now checked through the entire deadline. The saved v5
top-entry sweep is retained with `SUPERSEDED.json`; its target-entry geometry
is unchanged, but its acceptance counts are not the corrected baseline.

### Rebuilding a database while the QP sweep runs

`scripts/rebuild_transition_database.py` consumes the complete corrected
decoder audit plus completed QP worker shards. It retains every certified raw
decoder minimum and incorporates new/shorter repaired witnesses. It does not
filter starts by their one-second result. A group remains unsupported (`inf`)
when none of its admissible starts has a successful witness; an optimizer
failure is not relabeled as physical infeasibility.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/rebuild_transition_database.py \
  --out runs/transition_database_qp_20261004 --watch
```

The current rebuilt database lives in `runs/transition_database_qp_20261004`:

- `transition_table_qp.npz`: corrected decoder witnesses plus QP improvements;
- `transition_table_decoder.npz`: corrected decoder-only comparison;
- `transition_table.csv`: all 392 groups, with means, spread and support;
- `per_start_minima.csv` / `.npz`: minima and refinement requirements per start;
- `summary.json`: support counts, deadline coverage and provisional/completed status.

Both tables load through `TransitionTable.load`. Each mean averages the
minimum successful deadline for each supported admissible starting case,
rather than averaging all successful trials. Support denominators include
all admissible starts, including failures. Geometry-invalid starts and starts
outside checkpoint posture bounds are excluded from that denominator and
reported separately. A mean can increase when previously unsupported, harder
starting cases become supported, even though each individual sampled minimum
can only decrease.

The publisher checks the recorded nonlinear gates for every repaired witness.
It writes immutable snapshot bundles, then atomically switches `current` so
table readers see a complete snapshot. Stable filenames link through that
pointer and update without restarting the trajectory sweep. The table remains
marked provisional until QP evaluation covers all requested state/deadline
pairs. The static drum-pair yaw table and assignment/slack objective are reused.
A selected minimum tagged as requiring QP assumes the same refinement will
be available when generating the executed plan. These are sampled achievable
times under the analytic acceptance rules, not certified physical lower bounds
or an Isaac execution guarantee.

## Four-hit exhaustive-search distillation

`isaac_rl.schedule_distillation.FourHitScheduleScorer` predicts all 16 complete
four-hit assignment values in one forward pass. Its MLP is
`86 -> 256 LN GELU -> 256 LN GELU -> 128 LN GELU -> 16` (124,304 parameters).
Features are normalized joint positions (17), joint velocities (17), pelvis-frame
stick-tip positions (6), tip Cartesian velocities divided by 3 m/s (6), and
four hit slots with seven-way drum one-hot, time until hit, requested normal
speed divided by 3 m/s, and a valid flag (40). The fixed seven-drum layout is
assumed. Tip velocities use the analytic contact-point Jacobian times measured
joint velocity. Hit timestamps must be relative to the current state, and
invalid slots are zero-padded.

Output indices are chronological binary schedules, L=0 and R=1, from LLLL
through RRRR. Values are undiscounted actual-spline schedule scores divided by
20. `choose` selects the best complete schedule; unused padding bits are
canonicalized to zero. There are no drum-to-hand eligibility masks. This first
version supports at most four single-hit events; simultaneous groups are
explicitly rejected rather than flattened or dropped.

```python
from isaac_rl.schedule_distillation import FourHitScheduleScorer

policy = FourHitScheduleScorer.from_checkpoint(
    "runs/four_hit_schedule_distillation_20261004/best.pt", device="cuda:0")
# physical_table's first four slots, deadlines relative to now; measured q/v.
actions = policy.assign(table, q, v, cost.fk)  # (batch, 4), L=0/R=1
```

`scripts/train_four_hit_schedule.py` collects all 16 assignments with two shared
physical-hit bounce draws using the frozen generator's complete original spline
prefixes. Collision, torque, velocity and joint bounds include both arms and
waist. Fresh procedural rhythms and collision-clear starting pose mixtures use
separate RNG streams for training, validation and test. All derivatives of a
root case stay in its split. Two candidate schedules per root (search winner
and random schedule) supply executed and post-rebound states for suffix value
regression and target-network Bellman consistency. The Bellman target follows
that finite schedule, with discount 1 and zero value after its last hit.

The primary loss regresses all 16 measured root scores; auxiliary losses are
pairwise ranking, suffix Monte Carlo return and Bellman consistency. The
generator stays frozen. Best checkpoints are selected by validation selection
regret, never test metrics. Run and data manifests, TensorBoard scalars, and
fresh-bounce physical comparisons are saved alongside `best.pt` and `last.pt`.

The first run used 16,384 training, 2,048 validation and 512 test root cases,
262,144 finite-plan transition records and 3,000 optimization updates. On its
test set it matched 33% of exact search winners and reduced mean selection
regret from the graph baseline's 55.4 to 32.2 score points. Fresh four-draw
analytic evaluation yielded hit rates 23.5% graph, 22.4% MLP, 32.4% search
teacher. The MLP improved aggregate score and collision depth but did not
improve hit rate in this first run. Evaluation fixes one four-hit schedule per
window and assumes perfect spline tracking plus analytic rebound; it is not
an Isaac or event-refresh controller result. Single-window feature extraction
and selection measured about 6 ms on an RTX 5090.

To continue from that run's best weights on exactly the same data, with a
constant learning rate and a separate TensorBoard registration:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_four_hit_schedule.py \
  --out runs/four_hit_schedule_distillation_constant_lr_20261004 \
  --continue-from runs/four_hit_schedule_distillation_20261004 \
  --updates 5000 --learning-rate 0.0001 --constant-lr --devices cuda:0
```

The source data is linked, not regenerated. The generator checkpoint and
evaluation seed are inherited for matched physical comparisons. AdamW starts
fresh because the source saved model weights only; the Bellman target network
starts from the restored weights. TensorBoard update indices continue from
the source checkpoint's step (2,900 here). The initial checkpoint remains a
candidate for best validation regret, even if subsequent training worsens.

This 5,000-update continuation completed, selecting step 7,300 by validation
regret (28.13 versus 30.11 initially). On the same 512 test roots, selection
regret improved from 32.20 to 27.90 and exact search agreement from 33.0% to
35.5%. Matched four-draw analytic evaluation improved MLP hit rate from 22.36%
to 23.28% (graph 23.49%, search 32.43%), strict valid hit rate from 7.07% to
8.17% (graph 7.64%), and mean score from -111.37 to -107.08. The graph and
search controls reproduced exactly. More optimization helped modestly; the
remaining search gap is substantial. `comparison.json` records paired run
metrics; `best.pt` is the selected continuation checkpoint.

### Focused ranking and relative-score regression

The scorer's original absolute-value and Bellman losses remain active. The
optional training improvements use:

- `--ranking-top-k 4`: compare pairs containing at least one of the teacher's
  top four schedules. Weight pairs by the measured absolute score gap, capped
  at four normalized units (80 raw score points), and average weighted ranking
  within each window before averaging windows. Gaps below 0.25 raw points are
  ignored. `--ranking-top-k 0` retains the original all-pairs ranking.
- `--ranking-weight 0.5`: increase ranking emphasis from the original 0.1.
- `--relative-weight 1.0`: add Huber regression on predicted and target values
  after subtracting their respective per-window means. This teaches differences
  between assignments without requiring the scenario's common score baseline
  to be correct. Absolute root regression still has weight 1, suffix regression
  0.5, and finite-plan Bellman consistency 0.1.

The matched comparison starts both runs at the preceding continuation's
validation-selected step 7,300, using the same data and batch RNG sequence,
5,000 updates, constant learning rate 0.0001 and fresh AdamW/target networks:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_four_hit_schedule.py \
  --out runs/four_hit_schedule_focused_relative_20261004 \
  --continue-from runs/four_hit_schedule_distillation_constant_lr_20261004 \
  --updates 5000 --learning-rate 0.0001 --constant-lr --devices cuda:0 \
  --ranking-top-k 4 --ranking-weight 0.5 --relative-weight 1.0
```

`runs/four_hit_schedule_matched_control_20261004` uses the same command with
the original loss defaults and `--devices cuda:1`. Both have separate shared
TensorBoard registrations; each selects its best weights by validation regret.

Both 5,000-update runs completed. The focused/relative run reduced measured
training-subset regret from about 22.0 to 16.6, compared with the control's
18.3. It did not improve validation: its best new regret was 28.30 and final
regret 29.57; the control's best new regret was 28.22 and final regret 29.24.
The starting checkpoint's 28.14 remained better, so both `best.pt` files retain
that checkpoint, with test regret 27.90 and analytic hit rate 23.28%. Their
`last.pt` files contain the newly trained weights at step 12,300. These are
continuation experiments on the existing dataset, not evidence about training
the new objective from scratch or on more data. `objective_comparison.json`
in the focused/relative run records the comparison and matched physical tests.

### Fresh initialization at constant learning rate 0.0003

`--data-from` reuses a run's teacher data and generator/evaluation configuration
without loading its scorer weights. It is mutually exclusive with
`--continue-from`. Fresh initialization is saved as `initial.pt` at step zero.
For a fresh training comparison, both runs use identical initial weights and
batch RNG streams, 8,000 updates and constant learning rate 0.0003 (three times
the continuation rate). The original first run also started at 0.0003, but
decayed it to 0.00003 by update 3,000.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_four_hit_schedule.py \
  --out runs/four_hit_schedule_focused_relative_scratch_lr3e4_20261004 \
  --data-from runs/four_hit_schedule_distillation_20261004 \
  --updates 8000 --learning-rate 0.0003 --constant-lr --devices cuda:0 \
  --ranking-top-k 4 --ranking-weight 0.5 --relative-weight 1.0
```

`runs/four_hit_schedule_scratch_control_lr3e4_20261004` uses the original loss
defaults on `cuda:1`. Both keep critic and Bellman enabled, reuse the fixed
train/validation/test splits, register TensorBoard, and select their best
checkpoint by validation regret before matched analytic physical evaluation.

Both fresh runs completed. The focused/relative run selected step 5,800 with
validation regret 28.35, test regret 31.47, search agreement 32.2%, and analytic
hit rate 22.45%. The original-loss control selected step 6,000 with validation
regret 28.25, test regret 27.85, agreement 35.2%, and hit rate 22.61%. The
previous constant-0.0001 continuation remains preferable by validation regret
(28.14) and had test regret 27.90/hit rate 23.28%. Thus fresh initialization at
the higher constant rate did not establish an improvement, and the combined
new losses performed worse on the heldout test roots at the tested weights.
Graph and search physical controls reproduced exactly. These are single-seed
comparisons; `scratch_comparison.json` in the focused/relative run records
checkpoints, curves' final metrics and physical results. Existing loss defaults
and previous checkpoint paths remain as before this comparison.

## Online categorical schedule RL

`scripts/train_schedule_policy.py` trains a separate actor and value critic,
each using the existing 86-input/16-output MLP. Both start from
`runs/four_hit_schedule_distillation_constant_lr_20261004/best.pt`. Actor outputs
become categorical logits, not physical score estimates; policy checkpoints
use a distinct `four_hit_schedule_policy_v1` format and load through
`isaac_rl.schedule_policy.FourHitSchedulePolicy.from_checkpoint`. The critic
continues predicting score/20. Inference uses the actor's maximum logit to
choose a complete four-hit schedule, with padded action bits canonicalized.

Each training round generates fresh procedural rhythms, collision-clear pose
mixtures and starting velocities. The frozen decoder executes each candidate
with perfect spline tracking, full both-hand/waist physics scoring, and two
matched analytic bounce draws. No predicted reward is used as a training
outcome. There are two algorithms:

- `grpo`: sample eight complete schedules with replacement from the current
  softmax policy, temperature 1 (initially equivalent to 20 raw score points).
  Compute detached per-window group advantages using population standard
  deviation with a one-raw-point floor. Take four epochs of PPO-clipped updates
  with ratio bounds 0.8/1.2. Duplicate sampled schedules share the same bounce
  draws, so identical choices have identical measured rewards.
- `expected`: evaluate all 16 actions, then differentiate the full categorical
  expected reward, scaled by 20. A detached reward baseline improves numerical
  stability without changing the policy gradient. This eliminates action
  sampling variance; physical bounce rewards are still Monte Carlo estimates.

Both use constant actor/critic learning rates 0.0001, exact categorical KL
penalty 0.05 toward the frozen starting policy, entropy bonus 0.005, and
gradient clipping at 1. Policy gradients use actual outcomes; the separate
critic trains on measured root and suffix Monte Carlo returns and finite-plan
Bellman consistency, with zero terminal value and target EMA 0.01. The actor
does not take gradients through that critic or the frozen generator.

The initial paired experiment uses 128 rounds of 128 fresh windows (16,384
windows and 512 actor/critic optimizer steps per run), on separate GPUs:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_schedule_policy.py \
  --out runs/four_hit_schedule_grpo_online_20261004 --algorithm grpo --device cuda:0
/data2/eric/conda_envs/g1drum/bin/python scripts/train_schedule_policy.py \
  --out runs/four_hit_schedule_expected_online_20261004 --algorithm expected --device cuda:1
```

Action draws use a separate torch RNG, so both algorithms receive the same
scenario RNG stream, with fresh seed 20261004017. Each round saves actual
inputs, categorical collection probabilities, chosen global action indices,
outcomes and a scenario fingerprint under `online/`. `data_manifest.json`
records drum counts and fingerprints for matching the comparisons. The `data`
symlink refers to the existing evaluation dataset; RL never trains on its
training shards. Best actor checkpoints are selected by greedy validation
regret on the original fixed validation roots, retaining the initial actor if
it remains best. Final test and four-draw physical evaluations use the same
heldout roots/bounce draws as preceding experiments. TensorBoard registers
both runs and logs policy entropy, KL, stochastic and greedy regret, value
losses, Bellman loss, fresh-window counts and optimizer-update counts.

Both initial online runs completed: GRPO took 286 seconds and the all-action
control 519 seconds. Every one of the 128 scenario fingerprints matched, drum
counts were balanced, and measured rewards for identical evaluated actions
matched exactly. Twelve policy/scorer tests passed, including clipped positive
and negative advantages, masked categorical probabilities, full-action policy
gradients, and global sampled-schedule suffix indexing.

Neither run beat the initial actor's validation regret 28.14, so `best.pt`
retains the starting actor in both runs. Evaluations of the actually trained
`last.pt` actors are saved separately under `last_evaluation/` and TensorBoard's
`last_evaluation/` tag prefix. Their test results were:

| Policy | Greedy regret | Sampling expected regret | Entropy (bits) | Analytic greedy hit rate |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 27.90 | 39.67 | 2.20 | 23.28% |
| Final GRPO actor | 28.55 | 31.40 | 0.91 | 22.83% |
| Final all-action actor | 29.51 | 31.81 | 0.70 | 22.02% |

The actors learned more concentrated distributions and improved expected
reward when sampling, but did not improve greedy selection or hit rate in
this first trial. The final GRPO actor changed 32.4% of greedy assignments
relative to the source; the all-action actor changed 36.7%. Evaluation is still
static four-hit perfect tracking and analytic rebound, not Isaac execution.
`rl_comparison.json` in the GRPO run records both algorithms, the selected and
final actors, and the matched data/reward audit. Policy logits after RL should
not be interpreted as calibrated schedule ratings.

## Supervised model-size comparison on combined data

`scripts/prepare_schedule_training_data.py` combines the original 16,384
exhaustive-search training windows with the online all-action control's 16,384
fresh windows. The latter contain all 16 outcomes measured using the same
frozen generator and physics scoring. Partial sampled-GRPO labels are not
included. Validation/test shards remain the original 2,048/512 heldout roots.
The original 262,144 observed suffix transitions supply MC/Bellman training;
the fresh online source saved root outcomes but not suffix poses. Root IDs are
checked for uniqueness and split overlap, and all-action labels are verified
against the saved per-event rewards.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/prepare_schedule_training_data.py \
  --out runs/four_hit_schedule_merged_data_20261005
```

The MLP accepts `--hidden-dims`, with every hidden layer retaining LayerNorm and
GELU. Width metadata is saved in scorer and policy checkpoints; legacy files
without width metadata still load as 256/256/128. In this comparison, all four
sizes start fresh, use the same 32,768 root labels, original loss (absolute
value + 0.1 all-pair ranking + 0.5 suffix MC + 0.1 Bellman), batch 512, 10,000
updates and cosine learning rate 0.0003 to 0.00003. A separate minibatch RNG
seed 2026100501 makes root/transition samples identical across architectures,
independent of each model's initialization draw count.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_four_hit_schedule.py \
  --out runs/four_hit_schedule_size_large_20261005 \
  --data-from runs/four_hit_schedule_merged_data_20261005 \
  --hidden-dims 1024 1024 512 --batch-seed 2026100501 \
  --updates 10000 --learning-rate 0.0003 --devices cuda:2
```

The four runs and their validation-selected results:

| Run suffix | Hidden widths | Parameters | Validation regret | Test regret | Analytic hit rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `size_small_20261005` | 256/256/128 | 124,304 | 26.17 | 27.86 | 22.85% |
| `size_medium_20261005` | 512/512/256 | 445,200 | 25.04 | 26.36 | 23.17% |
| `size_large_20261005` | 1024/1024/512 | 1,676,816 | 25.09 | 25.51 | 23.11% |
| `size_xlarge_20261005` | 2048/2048/1024 | 6,499,344 | 23.63 | 26.41 | 22.69% |

Run names begin with `four_hit_schedule_`; each is registered in the shared
TensorBoard. Every run finished all 10,000 updates and selected its best
checkpoint by validation regret. The largest model won on validation; the
1.68M model had the lowest test regret. Larger models improved score prediction
and schedule selection regret, but hit rates stayed around 23%. This is a
single-seed comparison. Analytic evaluation fixes four-hit schedules and uses
perfect tracking plus rebound, with matched graph/search controls; it does not
measure Isaac execution. End-to-end feature extraction/selection remained
approximately 6 ms across sizes. Fifteen scorer/policy tests passed, including
variable-width and legacy checkpoint loading and mixed-shard replay loading.
`runs/four_hit_schedule_size_comparison_20261005/summary.json` records the
comparison, and each run's `full_training_fit.json` separates original and
fresh root fit at the selected checkpoint.

### Minimum-time chart without waist regions

`scripts/build_unrestricted_transition_chart.py` builds a `(hand, source,
destination)` table from the frozen clean-straight-line generator and iterative
SQP. There is no yaw axis or additional yaw window/penalty. All three waist
joints remain optimization variables within the checkpoint's physical limits.
The left hand excludes floor tom as both source and destination; the right
hand includes every drum, including hi-hat and left cymbal. This gives 85
ordered pairs, including same-drum recovery.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/build_unrestricted_transition_chart.py \
  --out runs/transition_qp_unrestricted_20261005 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --samples 8 \
  --qp-starts-per-pair 2 --qp-steps 30 --chunk 16 --verify-substeps 32
```

Each source supplies up to eight legal near-head departure states with random
outward velocities. Initial yaw is sampled over the existing joint box and
filtered by source reachability, rather than divided into regions. Both the
normal goal-derived decoder hint and the current-yaw hint are tried; these
initialization hints impose no restriction on the repaired trajectory.

The decoder is checked at every 20 ms deadline from 20 ms to two seconds.
For each pair, SQP refines the two starts with the shortest verified decoder
times, or the lowest constraint merit if no decoder witness exists. Every
shorter deadline is checked for selected starts with a witness. Starts without
a witness receive a coarse recovery sweep. A failed deadline does not prune
shorter deadlines once a witness exists. Each accepted result is verified with
32 spline collocations per 20 ms segment; failed dense checks receive another
SQP repair at that density.

Acceptance retains the transition-calibration conventions: 70 mm head-depth
band, no impact-speed gate, target top entry permitted, and idle-arm violations
ignored. Moving-arm and shared-waist torque, velocity, joint, collision and
contact orientation/elevation constraints remain active. Signed head-region
constraints retain useful gradients inside the strike region; exact head
membership still determines acceptance.

The output directory contains `minimum_times.png`/`.svg`,
`mean_minimum_times.png`/`.svg`, `transition_table.csv`,
`transition_table.npz`, and `summary.json`. The primary minimum is the shortest
dense-verified witness across sampled starts. Means, medians and p90 values
describe observed successful per-start minima; their differing search effort
and the successful/valid-start counts are recorded. An unsupported entry is
unknown, rather than a proof that the transition is impossible. These are
finite-search estimates, not certified physical minima or simultaneous
two-hand execution guarantees. Each worker also saves departure states,
verified witness waypoints, raw outcomes and repair-attempt counts.

Charts and tables update after the initial decoder sweep and during SQP.
`supported_searching` CSV rows and `searching` chart annotations distinguish
verified provisional times from finished searches; `summary.json` remains
`running` until all workers finish. To resume an interrupted calibration with
the same settings, add `--resume` to the command. Completed source searches are
reused and partial source searches restart deterministically. A live launcher
or worker prevents a second launch into the same directory.

`transition_statistics.png`/`.svg` shows mean, median and p90 in six panels
(three statistics × two hands), using successful per-start minima only. Each
cell retains its support count and provisional-search label. New chart runs
produce it directly; an existing running producer can be followed without
restarting its workers:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/plot_transition_statistics.py \
  --out runs/transition_qp_unrestricted_20261005 --watch
```

The watcher reads the atomic table snapshots, updates PNG/SVG exports, and
stops after the producer finishes. It does not modify calibration results.
It also writes `standard_deviation_times.png`/`.svg` and
`transition_standard_deviations.csv`. These use sample standard deviation
(`ddof=1`) of successful per-start minima. Fewer than two successful starts
leaves the deviation undefined; a single success is labelled `need ≥2` rather
than shown as zero spread. The live watcher derives deviations from saved
worker snapshots for producers launched before this statistic was added.
New producers also include `std_minimum_s` in their main NPZ/CSV tables.

`scripts/plot_transition_starting_poses.py` draws the current sampled departure
states under `starting_pose_viz/`: `overview.png`, one PNG per drum, and the
self-contained `starting_poses.html` with drum/hand selectors and a sample
slider. Arm joint origins and stick geometry use the actual URDF FK. The
torso box is schematic, transformed with the actual waist frame. All coordinates
include the environment pelvis offset exactly once. Static examples span the
saved waist-yaw range; initial waist pitch/roll are zero. Missing future source
cohorts are reconstructed with the calibration's exact seed and sampler budget
and labelled as pending; they are saved separately from running worker inputs.
The contact sampler's 2–12 mm outward offset is relative to the strike plane,
which is inset 10 mm below the physical head. The viewer reports both heights.

`scripts/plot_transition_start_relationships.py` reads per-start snapshots and
writes `starting_state_relationships/relationships.html`, an overview PNG,
one PNG per hand/source, `per_start.csv`, and `pair_linear_fits.csv`:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/plot_transition_start_relationships.py \
  --run runs/transition_qp_unrestricted_20261005 --watch
```

The self-contained viewer selects any allowed pair and compares starting yaw
and actual outward contact speed against either the best known duration or
the decoder-only duration. Its third panel plots yaw against departure speed
for all starts. Unknown searches appear in a separate strip, never as numeric
durations. Markers distinguish starts selected for QP; all other starts only
receive the decoder search. Fits exclude unknown times, require at least three
observations and a varying predictor, and prediction error uses leave-one-out
validation against a constant baseline (at least four observations). Quadratic
prediction error is also reported at five or more observations. Lines are drawn
only at four or more successes. With eight starts per source, differing arm
poses, success-conditioned sampling, and only two QP starts per pair, these are
exploratory associations rather than evidence of physical linearity. The
watcher updates exports as source files change without modifying the search;
reopen or reload the HTML for the latest snapshot.

### Hi-hat failure diagnosis

`scripts/audit_hihat_transitions.py --run runs/transition_qp_unrestricted_20261005`
replays all 104 saved hi-hat destination starts across the full decoder grid
with both yaw hints, then verifies each start's closest candidate at 32
substeps. It writes separate reports and candidate paths in `hihat_diagnostic/`
without changing the completed table. Closest means smallest four-substep
constraint merit with elevation omitted; it is not the best dense candidate
over all deadlines. Both original and angle-exempt coarse support counts are
reported explicitly as coarse counts.

The 2026-10-05 audit found an unconditional elevation gate in SQP despite
`strike_elevation_required` exempting cymbals, including hi-hat. Omitting that
gate alone did not add any coarse decoder successes (10/104 either way).
Four inspected left-hand arrivals had valid terminal hit membership but
roughly 10 mm overlap with the target hi-hat body on the dense grid. Their
initial overlap preceded certified top entry by 1.25–2.5 ms; the entry mask
rejects the whole contact episode once it began without a top certificate.
`HIHAT_RIM_FRACTION=1.0` aims at the rim, whereas the certificate uses the
physical head radius minus the 10 mm stick radius. This rim-first contact can
be missed by the coarser optimization grid. The measured normalized collision
gradient maxima were only 2.7e-6–7.7e-6 per radian in those four candidates,
giving the local QP little guidance to repair their approach. `entry_examples.png`,
`entry_inspection.json`, and `collision_gradients.json` show these cases.

Right-hand candidates from other drums also had much larger terminal misses
(roughly 0.14–0.29 m in the closest examples). The normal pose library lacks
`right/hihat`, and the phase-2 row builder rejects goals absent from that
library. Those failures must not be interpreted as certified physical
unreachability. The existing top-entry policy still rejects side entry; this
audit did not relax it, change training, or rebuild the transition table.

### Hi-hat fixes and follow-up (2026-10-05)

The top-entry certificate now tests first downward skin contact against the
**physical head radius**. It previously used `radius - stick_radius`, which
incorrectly required the entire capsule cross-section to fit inside the head.
Near-rim top contact may overhang the rim. Side entry and side re-entry still
fail, and wrist/other-drum collisions retain their checks. Tests cover all three
cases, tilted heads, and shaft-only top contact. An explicit top-entry distance
supplies a repair gradient where a capped-cylinder intersection is flat;
nonlinear physical acceptance is still required after every SQP step.

The hi-hat aim is inset by 20 mm; the world legal hit disk and near half-plane
are unchanged. SQP, transition calibration and violation profiling now apply
the elevation band to each active **pad** hand; hi-hat/cymbal hands are exempt
as specified by strike geometry. Collision cylinders also use per-row drum
normals from randomized training inputs, matching the terminal geometry.

Both hi-hat library entries were regenerated at nine waist yaws inside the
checkpoint's +/-80-degree hard box. Each pose satisfies <=3 mm nominal shoulder
contact error and active-arm static clearance. `right/hihat` is now eligible in
the training sampler and library. Old frozen terminal heads get a cheap
right-hi-hat library initialization; the experimental four-step runtime IK was
removed after it failed to recover decoder plans and added 14-18 ms.

`runs/hihat_physical_top_dense_scan_20261005` checks all coarse-passing raw
candidates on the dense grid until it finds a witness for each start. The
unchanged checkpoint has seven supported starts in three left-hand source
pairs. `runs/hihat_physical_top_qp_20261005/verified_witnesses.csv` adds strict
repaired witnesses for snare and tom2: left snare->hi-hat 440 ms, tom1->hi-hat
160 ms, tom2->hi-hat 760 ms, cymbal_right->hi-hat 920 ms, and hi-hat->hi-hat
200 ms. These are tested feasible durations, **not certified minima**. One
closest candidate per source was repaired; the cymbal_left example retains
about 0.95 mm overlap, and none of the selected right-hand examples passes.
The completed original 85-pair table remains a baseline, not a corrected table.

The first continuation, `phase2_hihat_repair_20261005`, was stopped after its
100-update dense scan lost the baseline valid left-hand plans. Its falling
training loss was not a sufficient success criterion. The replacement,
`phase2_hihat_repair_retained_20261005`, starts from the original step-32000
checkpoint with rank-8 decoder LoRA and a frozen starting-actor trajectory tie
on goals outside right-hi-hat coverage. It keeps the distributional control
critic, Bellman suffix replay and waist stabilization; uses eight collocations
per knot and the physical top-entry training contract; and completed 1000 additional
updates. Both the 500-update and final checks preserve all seven raw dense-verified
starts; right-hand median closest-candidate miss is still 228 mm versus the
baseline 235 mm. The resumed pool includes 529 active right-hand hi-hat goals
and 47 right-hand hi-hat future goals. Focused single-hand clips cover every
eligible source with 400/600/800 ms gaps and repeated source->hi-hat->source
sequences, so the on-policy roller can harvest right-hand post-hit states and
bounce velocities. The original frozen checkpoint remains the retention
teacher across resumes. TensorBoard is registered under that run name.
`watch_hihat_repair_probe.py` automatically checks all saved departure starts
at 500 updates and completion, writes `hihat/raw_dense_*` TensorBoard metrics,
and runs dense-verified QP checks at completion. Right-hand recovery must be
established by these probes before exporting it as an improvement. This first
continuation produced no valid raw or selected QP right-hand arrivals.

A second experiment uses the existing four-step terminal DLS teacher only
on right-hand hi-hat rows, with a full-physics lookahead cost and an imitation
tie. Inference remains one decoder application. The adaptive line-search
pilot reached its learning-rate floor; `phase2_hihat_terminal_teacher_adam_20261005`
continues in FP32 with AdamW at 1e-4 and the original retention teacher. Its
step-33300 held-out scan passes 19/56 right-hand departure starts and 8/48
left-hand starts, retaining all seven original left-hand witnesses. Right-hand
raw support covers snare, tom1, tom2, cymbal_left and cymbal_right. The general
81-case evaluation regressed (40/81 at final step 33600 versus 57/81 for the
earlier continuation), so this checkpoint remains experimental and does not
replace the default generator. The final held-out scan passes 14/56 right
starts and 7/48 left starts, retaining six of the original seven left starts.
The step-33300 snapshot is the preferred hi-hat experiment: 19 right starts,
8 left starts and all seven original left witnesses retained. Neither is
promoted as a general-purpose replacement. Final ordinary QP checks also
recover a right cymbal_right arrival, giving six supported right source
pairs; floor_tom remains unresolved. The experiment completed 466 AdamW
updates, with critic and Bellman active throughout.

The offline SQP probe also exposed competing torque peaks: linearizing only
one maximum per actuator can repair one time point while breaking another.
`QPConfig.torque_time_bins` optionally keeps a maximum per actuator in each
time interval. Eight intervals plus smaller backtracking steps repair the
terminal-corrected right tom1->hi-hat witness. With the original frozen
generator and these offline seeds, all six eligible left sources and six of
seven right sources now have strict dense-verified hi-hat arrival witnesses;
right floor_tom->hi-hat is still unresolved. The witnesses are recorded
separately under `runs/hihat_fixes_probe_20261005`; they are not minimum-time
certificates. The main chart builder exposes `--qp-torque-time-bins` and
`--hihat-terminal-repair` to reproduce these repair options, with resume
checks preventing mixed search settings. Defaults preserve the original
search method, and the original completed table remains intact.

### Comparing 60 ms duration steps with 20 ms steps

`scripts/probe_coarse_transition_qp.py` keeps the full chart search running
and launches a separate six-case matched comparison. Both searches start from
the identical dense-verified raw decoder trajectory and measured departure
state. Only deadline spacing changes: spline knots remain 20 ms apart, SQP
still has 30 iterations per seed, and accepted paths still receive 32
collocations per knot interval. The two decoder initialization hints,
continuation/fresh seed order, trust region and physical gates are unchanged.

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/probe_coarse_transition_qp.py \
  --out runs/transition_qp_coarse60_comparison_20261005 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

The baseline checks every shorter 20 ms deadline. The comparison first checks
every shorter 60 ms deadline without pruning after failures, then checks the
20 ms grid in a 120 ms window below its shortest verified witness. The fine
window advances if a shorter witness is found. Coarse failures can be retried
with an improved continuation seed. An isolated feasible fine-grid island
outside that window can still be missed, so agreement must be measured rather
than assumed. Both methods share precomputed fresh decoder seeds, whose
preparation time is reported separately. Method order alternates across cases.
Reports include minimum times, dense constraint diagnostics, repair-call
counts, actual OSQP solve/iteration counts, and elapsed time. Timing is measured
while sharing GPUs with the existing chart search, rather than as an isolated
throughput benchmark. This probe does not change the full chart's search policy.

If interrupted, rerun with `--resume`. Frozen departure states are retained;
completed method measurements are reused and missing methods restart. Each
new method saves its witness NPZ before marking its JSON result complete.
The original probe hit a witness-save keyword collision after two method
measurements finished; that collision is fixed. Those dense-verified JSON
measurements remain reusable, with `witness_saved=false` when the associated
path archive was not written. Timing comparisons across resumed processes
should be read alongside actual QP solve counts.
