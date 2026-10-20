# Neural four-group hand-plan ranking

## All16 throughout comparison (2026-10-07)

User requested another run retaining every legal assignment for all40 collection
updates. `--stages 40 0 0` skips8/4 stages; zero durations are now supported and
tested. Same seed20261010, fresh neural initialization,136-input105,488-parameter
MLP,256 new random rhythms/update,80/15/5 post-hit/random/ready mixture, frozen
decoder and matched Isaac geometry, identical100 fit steps/update and4000-step
warmup/cosine schedule. Capacity4096 across4 GPUs; legal banks may have fewer
than16 dueLH-floor exclusion. This run changes candidate collection only; it
does not warm-start from learned curriculum weights.

Run `runs/neural_hand_plans_all16_256_20261007/`; supervisor script in
`runs/neural_hand_plans_all16_inputs_20261007/`. Reuse verified aligned-geometry
held-out outcomes for monitoring, never training. At completion, execute both
trained models, initial neural and graph on the exact frozen256 rhythms and
initial q/qd/tips from the corrected prior fresh test. New evaluator flags
`--reference-run` and `--reuse-test-plans` freeze this comparison, retain graph
initial target history and check decoder/geometry compatibility. Final artifacts
`runs/neural_hand_plans_all16_matched_test_20261007/`. Full-budget parameter
comparison is matched; the all16 experiment deliberately uses more simulator
rollouts. Fresh rollout-derived start banks can diverge after the schedules
diverge, even with the same RNG seed.

## Decoder-matched Isaac geometry is now authoritative (2026-10-07)

User chose to match Isaac to the decoder. Registered-task configuration and
shaft probes now bind to local `g1_scene.g1` via `isaac_rl.decoder_scene`, using
the same `amortized_opt.constants` as the analytic FK. Tip sensors are explicitly
configured to those offsets. Robot USD, tip/shaft collision transforms and
radii are audited at worker startup; the composed asset cache invalidates on
constants edits and serializes concurrent rebuilds. Playback and parallel
assignment entrypoints use the same binding. Frozen decoder weights are
unchanged; no decoder retraining is required for this geometry correction.

Corrected varied-state smoke:
`runs/neural_hand_plans_decoder_geometry_smoke_20261007/`. First64 measured
states agree with decoder FK within1.70 micrometres and with neural Cartesian
inputs within1.91 micrometres; q/qd install exactly. The actual USD collider
tip transform differs from the analytic offset by only1.37e-8 m.32 relevant
checks passed, including binding, stale cache invalidation and old-label
rejection. New simulation version `decoder_shared_constants_v1`.

Production pipeline in `runs/neural_hand_plans_decoder_inputs_20261007/`:
regenerate all legal plan outcomes on256 fixed held-out ready-start rhythms
at `runs/exhaustive_decoder_geometry_256_20261007/`, then start the fresh136-input
105,488-parameter varied-state network at
`runs/neural_hand_plans_decoder_geometry_256_20261007/`, preserving the user’s
10 /20 /10 (16→8→4) curriculum and256 rhythms/update. Initial states are
80% measured post-hit,15% collision-clear random arms/waist,5% ready. Neither
old-scene label cache nor old post-hit bank is reused. Trainer refuses caches
without the new geometry version. Held-out labels remain monitoring only.
After40 updates, run a fresh matched varied-start test against the initial
neural model and graph at
`runs/neural_hand_plans_decoder_geometry_fresh_test_20261007/`.

Use `pipeline.json` and current process state for launch/progress; launchPID
alone is not evidence of continued activity. TensorBoard registers when the
training child starts; data regeneration is read-only evaluation.

## Varied starting states and geometry finding (2026-10-07)

User added actual initial arm/waist q/qd and both Cartesian tip positions.
Observations are now136 inputs; MLP105,488 parameters. Joint state was already
present; six tip-coordinate inputs were added. Both tips are in kit/env metres,
excluding the replicated environment's grid origin. Model checkpoint metadata
stores simulator-reported wrist-local tip/mount geometry and pelvis origin.
Root FK uses that geometry for random/ready tips; post-hit tips are measured.

Collect whole measured q/qd/tips40 ms after successful contact, preserving both
arms and waist and last successful targets. Keep up to32 actual states per
hand/drum via reservoir replacement. Bootstrap with real decoder-driven hits;
never fabricate successful post-hit states for missing combinations. Refresh
the bank from later training hits. Mixture80% post-hit /15% collision-clear
randomized arm/waist /5% ready. Every candidate of a rhythm shares its exact
initial position/velocity. Reset writes qd inside the original reset hook,
before simulator FK refresh; servo/strike trackers still reset normally.
Runtime guards verify installed q/qd within1e-5 and Cartesian inputs within1mm.
Snapshots, replay and RNG include exact chosen physical starting states.

The original ready-start production launch was stopped before any completed
update, retaining its checkpoint. New varied-start production has not launched
while the user decides authoritative geometry. Tests20 passed. Initial smoke
found simulator tip/FK disagreement up to0.121905 m despite identical q/qd.
Amortized constants use mounts(0.2, ±0.03, 0) and mostly forward sticks; the
registered external g1_drummer asset uses(0.2, ±0.06, 0) with inward28-degree
flare. These are different physical definitions, not PD tracking lag.
Using worker-reported simulator geometry reduces FK vs simulator error to
2.07e-6 m across128 recorded initial states. Corrected physical smoke runs
`runs/neural_hand_plans_varied_geometry_smoke_20261007/`, and its first96
initial states agree with observations within2.44e-6 m. This fixes the ranker's
observations/start clearance, not the frozen decoder's geometry mismatch.
The requested authoritative-geometry choice is pending; modifying simulation
versus retraining the decoder has different implications. Cached old-scene
physical labels must be replaced if simulator geometry changes.

New full-checkpoint format is `neural_hand_plan_ranker_v2`; old130-input
ready-only checkpoints are deliberately not silently resumed. Standalone
fresh evaluator now applies the same matched varied-start mixture. Cached
exhaustive monitoring remains a ready-start-only measure.

The user stopped timing-chart fitting at actual timing update29 and requested
a normal neural network with16 actions, evaluating alternatives in Isaac and
learning their ordering. The chart checkpoint remains archived; its learner
and automatic recorded-test pipeline were stopped.

## Representation and selection

`isaac_rl/hand_plan_ranker.py` implements a103,952-parameter MLP:130 inputs,
256-wide hidden layers (LayerNorm/GELU then GELU),16 output scores. Each action
is a four-bit sequence. Groups are ordered by time; within a simultaneous
group targets are sorted by drum ID. Bit0 assigns the first target left,
bit1 assigns it right; the second target gets the opposite hand. This covers
all16 binary four-group plans. LH-floor eligibility masks forbidden outputs;
all other13 hand/drum combinations, including RH-hi-hat and LH-right-cymbal,
remain allowed. Legal banks can contain2,4,8 or16 plans.

Inputs retain both drums, velocity and presence per group, absolute relative
time and each gap, q relative to ready, qd, and each hand's last target/ready
one-hot. There is no average-other-hand proxy, minimum-time lookup, historical
support penalty or chart-derived ranking prior. The current training and
evaluation starts are nominal ready with zero velocity and no previous target.
The API accepts state features, but competence at other starting states has
not been trained or verified. Kit positions/dynamics remain nominal.

## Candidate curriculum

User confirmed10 /20 /10 collection updates,256 fresh procedural random rhythms
each. Updates1–10 execute all legal plans (up to16);11–30 execute neural top8;
31–40 execute neural top4. Select using the current model alone, mask mechanical
illegality, and execute each selected action once. Simulator capacity shrinks
from4096 to2048 to1024; persistent workers restart at each stage boundary.
All7 requested drums and49 ordered requests /21 simultaneous pairs remain
stratified across fast, medium and slow timing. Unselected legal actions carry
no outcome label and are excluded from training loss, rather than assigned0.

Physics250 Hz, control50 Hz, same frozen actor-step12000 all13-goal decoder:
SHA256953d1f31438c02dbbe522d9319fa552085386a4c6a1ec90be662f9dd99eda2ce.
Physical reward remains credited-hit fraction plus bounded missed-hit proximity,
minus demand torque excess, joint excursion and measured penetration penalties.

## Learning objective

Replay stores whole observed candidate panels, preserving shared rhythm context
and missing-action masks. The objective is2×best-set cross entropy plus weighted
soft pairwise ranking. Best outcomes within0.02 reward of the observed maximum
share the winner target to reduce arbitrary near-tie fitting. Pair targets are
sigmoid((rewardA−rewardB)/0.05), with gap-based confidence. Each loss averages
within a rhythm before averaging across the batch, so larger legal banks don't
implicitly get more weight. No chart Jacobian or timing-feasibility penalty
limits the neural score gradients.

Each collection update performs100 AdamW gradient steps, minibatch128, replay
capacity4096 rhythms. LR warms linearly over100 absolute gradient steps to3e-4,
then cosines to3e-5 over4000 total steps. Later candidate stages do not restart
the clock, so the top4 stage also uses smaller optimizer steps. This is a
supervised assignment ranker; phase-2 decoder critic/Bellman rules do not apply.

## Monitoring and checkpoints

`scripts/train_neural_hand_plans.py` saves `last.pt`: network, optimizer, full
observed replay, NumPy/Torch RNG, gradient/update clocks, exact pending panel
and frozen-decoder checksum. Resume preserves them and verifies schedule
compatibility. `initial.pt` retains the original neural initialization.
`trials_*.pt`, `episodes_*.json` and `iteration_*.json` retain raw labels and
pre-fit/post-fit metrics. TensorBoard auto-registers under
`runs/tensorboard_active` and verifies visibility on6006.

The previous256-rhythm exhaustive diagnostic is a read-only held-out panel,
containing all legal actions' cached physical outcomes. It supplies a stable
monitor of ranking quality, never replay, mining, checkpoint selection or
rollback. Metrics are explicitly cached outcomes from one execution per plan;
they are not new simulation or a repeat-averaged expected performance estimate.
`training.png` refreshes after every completed collection update.

After40 updates, `scripts/evaluate_neural_hand_plans.py` can execute a fresh
matched256-rhythm comparison of latest neural, initial neural and old graph,
with rhythm-bootstrap intervals. It uses no timing chart or recorded clip
truncation. Longer receding inference and varied-state data remain future work.

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 AMORTIZED_RNEA_CACHE_ONLY=1 \
/data2/eric/conda_envs/g1drum/bin/python scripts/train_neural_hand_plans.py \
  --out runs/neural_hand_plans_256_20261007 --contexts 256 --stages 10 20 10
```
