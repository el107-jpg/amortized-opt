# IID supervised hand-plan dataset

## Fresh100k supervised fit and frozen-policy comparison

Run `runs/neural_hand_plans_iid_train100k_large_gradient_l1_20261008/` trains a
fresh shared_candidate scorer448/224 (254481 parameters) on100000 corrected-
runtime training cases; validation/test remain819 each. Same seed20261010,
decay0.3/dropout0.15, input-gradient coefficient0.1 /batch64,100k updates with
batch256,100-step warmup3e-4 followed by cosine3e-5. No old labels, weights,
optimizer or replay initialize training; motion decoder is frozen and unused
by the offline fit. TensorBoard label matches the run directory.

`evaluate_cached_hand_plan_reference.py` freezes the previous large-gradient
model's latest100k checkpoint and verifies that both new held-out observation
arrays and legal masks exactly match its original held-outs. Scores are then
graded against fresh labels. Baseline test43.6892%, validation42.0%; baseline
training provenance stays legacy, evaluation provenance is fresh runtimev2.
Snapshot and metrics under `runs/hand_plan_train100k_fit_inputs_20261008/`.
Trainer `--reference-evaluation` validates dataset/evaluation runtime, split
counts, read-only flags and immutable checkpoint hash, and only plots/logs
baseline metrics. It never loads reference replay/optimizer into fitting.
Reference content hash is saved and enforced on resume. This comparison tests
both increased training data and corrected label generation, not quantity alone.
Live learning.png/entropy.png compare latest weights with the frozen baseline;
input_gradients.png retains fixed read-only split probes.10 targeted tests and
200-step CUDA smoke on the full new dataset passed before the full launch.

## Fresh100k collection and actual human-hand data research

`collect_large_hand_plan_dataset.py` is actively collecting exactly100000 IID
training cases plus freshly relabeled original819 validation/819 test inputs.
Output `runs/hand_plan_dataset_iid_train100k_fresh_20261008/`. Fresh processes /
scenes per batch, one physical wave, exact dynamic capacity, three GPUs0/1/3,
250Hz physics, BF16 and matched-speed/contact-reset/fixed-scoring runtimev2.
Added scene_reset_protocol fingerprint prevents mixing these labels with old
collections. Bank refresh uses training post-hits only. Batches117–119s imply
about13h for the collection; benchmark and progress in run.json/metrics.jsonl.

User clarified online search should find measured HUMAN hand choices. No
recorded-rhythm supplement was launched; Groove MIDI and graph-generated RCC
limbs must not be called human hand annotations. Strongest public lead is
[MMIP](https://mmip.cs.ucy.ac.cy/), with [official project description](https://www.theodoroskyriakou.com/projects/mmip/index.html).
Its current public catalog lists14 drum takes (~43.7min), synchronized MIDI,
60FPS FBX body/hand mocap and multi-angle video. Verified MIDI/FBX download
HEAD responses; no media downloaded. Per-hit L/R would require extraction from
measured hand motion and review of ambiguous notes. CC BY-NC-SA4.0 dataset.
The Snare Drum Motion Capture Dataset2015 describes L/R CSVs for40 rudiments,
but [the checked archive](https://zenodo.org/records/1179168) hosts only the PDF.
Disney2026 drummer-motion research reports3.5h of120FPS optical stick mocap /
MIDI; no public dataset link found, and its paper lacks hand-per-hit labels.
Human datasets have not entered training/replay or replaced the active IID run.

## Scorer-size × input-gradient penalty ablation

User requested a larger model and controlled ablation. The2×2 conditions are
scorer224/112 (106,641 parameters) and448/224 (254,481), each with input-gradient
coefficient0 or0.1. Shared hit/group/state encoders keep their original widths;
only the final candidate scorer is widened. Both new large runs start fresh
with seed20261010 on the same16384 cases/splits,100k updates/batch256, decay0.3,
dropout0.15 and the existing warmup/cosine schedule. The small unregularized
reference is completed; the small gradient trial continues. Both large models
had identical initial weights. Diagnostics run with dropout off and do not
change training RNG/parameters; zero coefficient performs no penalty update.

Capacity references validate matching gradient coefficient, sampling batch and
physical scales, alongside data/splits and optimizer settings. All new runs
preserve legacy label provenance. This extends the user's existing-dataset
choice to the offline ablation; no fresh simulator labels or actor training.
The read-only reporter `report_hand_plan_size_ablation.py --run
runs/hand_plan_size_ablation_20261008 --watch` publishes ablation.png,
comparison.json and matched_metrics.csv at common evaluated budgets. Latest
metrics/steps per run are retained separately; held-out results never select
weights or affect training. TensorBoard labels:
`neural_hand_plans_iid_large_gradient_l1_20261008` and
`neural_hand_plans_iid_large_nograd_20261008`.

## Input-gradient L1 trial on the existing dataset

The user explicitly selected the existing16,384-case legacy dataset for this
offline experiment. Its14,746 training /819 validation /819 test cases remain
unchanged; no corrected-runtime labels are mixed in. New checkpoint provenance
retains `legacy_contact_reset_v0`. This is a regularization comparison on the
archived labels, not a new assessment of corrected-controller hit performance.

`train_hand_plan_dataset.py --gradient-l1 0.1 --gradient-batch 64` adds input-
Jacobian L1 to the unchanged winner/pair ranking objective. For each of64 cases,
sample one legal assignment uniformly, center its score by the legal-score mean,
differentiate with respect to physical continuous inputs, sum absolute scaled
derivatives and multiply by legal count. This estimates the full legal-score
Jacobian L1 without computing16 backward passes. The extra evaluation-mode
forward disables dropout while preserving autograd; the ranking pass keeps its
existing dropout. The independent sampler's RNG is checkpointed/restored.

Physical scales:20ms per interval,1degree per joint,0.1rad/s per velocity,0.02
per present loudness. Interval derivatives include subsequent absolute deadlines;
joint derivatives include both normalized q features and Cartesian tips via
input-only FK Jacobians. Drum IDs, presence and previous-target onehots are
fixed. Cached Jacobians are fingerprinted against observations and geometry.
The near-zero derivative statistic excludes padded loudness and singleton cases.
Read-only fixed256-case probes per split log gradient magnitude/group terms,
alongside entropy, ranking, hit rate and regret; they never control fitting.

Run `runs/neural_hand_plans_iid_gradient_l1_20261008/`: fresh same-seed
shared_candidate hidden224,106,641 parameters, decay0.3/dropout0.15,
100k steps/batch256,100-step warmup to3e-4 then cosine to3e-5. Reference is the
completed shared-candidate no-gradient-penalty run. Physics/decoder remain idle
and frozen during this offline fit. TensorBoard label matches the run folder.

## Decoder/Isaac impact-speed agreement

The installed drum task previously graded against2×normalized loudness m/s,
while the frozen decoder scheduler requested1+2×loudness. It now supports an
explicit per-beat physical target tensor. Assignment evaluator `_install`
fills `beat_target_speed_m_s` using the same `amortized_opt.rhythm.strike_speed`
helper as decoder `strike_velocity`, with the scheduler's actual min/max bounds.
The normalized beat_velocity remains unchanged for observations/telemetry.
The installed task `_beat_target_speeds()` uses explicit targets when supplied;
ordinary PPO tasks with no override retain their own speed convention.

Fresh assignment rollouts default to `--match-decoder-speed`, alongside cleared
contact resets/fixed initial scoring. Version `matched_decoder_speed_contact_reset_v2`
and `impact_speed_target` bounds are recorded in worker/dataset/checkpoint
provenance. Old resumes default to their archived mapping unless explicitly
changed; collector resumes, replay handoffs and dataset composers reject mixes
of old and new outcomes. Timing sigma.4s, speed shape1, quality threshold.25 and
contact qualification remain separate criteria. No actor/ranker retraining or
large label collection is part of this correction.

Task extension is in the installed checkout:
`/data2/eric/g1-drummer-lab-rythmic/source/g1_drummer/g1_drummer/tasks/direct/g1_drum/g1_drummer_env.py`.
Unit tests extract its actual target/reward expressions without requiring GUI
bindings, and a fresh physical collection smoke exercises the loaded task.

## Contact reset correction and label provenance

Investigation found stale PhysX wrist-force reports surviving teleport/reset.
The controller read those reports before a new physics step and softened its
contact-dependent PD gains. Commands/q/qd were initially identical across
repeats, but all62 control cases had different first-step contact flags; state
diverged by the next control tick. `clear_reset_contact_buffers()` primes and
zeros the lazy force caches after reset; real subsequent physics repopulates
them. Fresh rollout defaults now clear these caches and freeze initial contact
completion criteria (timing sigma0.4s, speed shape1, threshold0.25). Scoring drift
was not the observed failure cause in the tested runs; freezing prevents future
label drift. FP32 alone did not cure hit-count variation and stays opt-in.

Three-repeat62-plan ablation: legacy8/62 hit counts varied; corrected bfloat16
1/62 varied.95thpercentile per-plan reward spread0.1223→0.00631. Residual edge
variation remains. A centered, single-slot FP32 timing rerun removed the prior
alternating0/1-hit teeth: all+11..+20ms offsets gave1hit in repeats.

New datasets record `rollout_runtime` from worker ready.json. Collections that
predate the correction cannot resume into corrected outcomes; composers reject
runtime mismatches. Offline checkpoints carry provenance, and online handoff/
resume/held-out monitors reject mismatched legacy replay/labels. Physical q/qd
banks are initialization states rather than reusable outcome labels. Existing
datasets/weights stay archived; no full relabeling/training run was launched by
the diagnostic. A4-case48-rollout collection smoke verified the corrected
default and metadata. Fresh labels are needed before further corrected-runtime
training. Detailed diagnostics under `runs/rollout_repeatability_audit_20261007`
and `runs/hand_plan_fixed_slot_timing_20261007`.

## Shared encoders and explicit candidate scoring

`--architecture shared_candidate --hidden 224` implements the next requested
representation experiment using the unchanged136-value dataset format.
Shared drum embeddings7×16 and an18→32 hit encoder operate on all eight slots.
Each candidate's four bits arrange those slots as left/right per group. A
shared67→64 group encoder receives both32-value hit features, deadline/gap and
the assignment bit. A shared56→64 state encoder retains q/qd, previous targets
and Cartesian tips. The ordered four group features plus state form320 values;
one320→224→112→1 scorer evaluates every candidate. Full four-group interactions
and both-hand/waist coupling remain possible; no additive-transition assumption.
106,641 parameters, close to105,488 flat reference. Dropout0.15 after hidden
activations, disabled for evaluation/selection. Invalid plans are masked by the
existing objective/selector; no missing labels become failures. Candidate-list
order has no effect on evaluation scores, and arbitrary candidate subsets can
be scored with `model.score_candidates(observations, plans)`.

Fresh fixed-dataset trial `neural_hand_plans_iid_shared_candidate_20261007`,
same14,746 train/819val/819test, seed20261010, decay0.3,100k/batch256 schedule.
Uses CUDA2; checkpoints retain CPU/CUDA RNG and require the same device on
offline resume. All older model configs default to architecture `flat` and
remain loadable. Evaluator and online handoff instantiate saved architectures.
`--reference-run` validates matching dataset/splits and optimizer budget, reads
flat-run traces for dashed test curves/matched-step `comparison.json`, and never
selects weights or alters fitting. This changes representation at approximately
matched capacity; CUDA vsCPU also changes numerical execution.

## Weight decay plus hidden dropout trial

User requested more weight decay and hidden-layer dropout. Fresh trial
`neural_hand_plans_iid_decay03_dropout15_20261007` uses AdamW decay0.3 and dropout
probability0.15 after both hidden GELUs. Same dataset/splits, seed20261010,
105,488 trainable parameters, objective,100k steps/batch256 and warmup/cosine
schedule as the completed decay0.1 experiment. It tests the combined change.
Dropout is disabled by model.eval() during all assessments and hand selection.
Parameter state-dict keys and seeded initial tensors match older no-dropout
models exactly. Configuration and Torch RNG are preserved on resume/handoff.

CLI `scripts/train_hand_plan_dataset.py --weight-decay 0.3 --dropout 0.15`.
Entropy diagnostics now include the paper's unnormalized -sum |w| log |w|
statistic per matrix and summed, alongside normalized magnitude entropy and
prediction entropy. They are distinct quantities; the raw statistic depends
on weight scale. TensorBoard retains older labels. No fresh physics collection
or online refinement is required for this fixed-dataset experiment.

## Stronger regularization and entropy monitoring

The completed expanded dataset has16,384 cases /153,244 physical plan rollouts,
with14,746 training cases and the original819 validation/819 test unchanged.
User requested stronger regularization after collection. The automatic weak-
regularization more-data fit was stopped at startup; its weights are not reused.
Fresh run `neural_hand_plans_iid_strongreg_20261007` uses seed20261010, the same
105,488-parameter architecture and ranking objective, batch256, AdamW decay0.1
(previous1e-4),100,000 steps,100-step warmup to3e-4 and cosine to3e-5. No online
data/refinement, early stopping, held-out checkpoint selection or weight rollback.
This changes both regularization and training duration; it is not the originally
planned controlled20k-step data-quantity comparison.

Every1,000 steps record train/validation/test loss, hit rate, pair ordering,
reward regret, prediction entropy in nats, normalized entropy, top probability,
near-best target entropy and probability mass, plus parameter L2/RMS and per-
matrix absolute-weight entropy. Prediction softmax uses temperature1 and only
legal labeled actions; normalized entropy excludes forced single-plan cases.
These diagnostic probabilities are not calibrated hit probabilities. Absolute-
weight entropy normalizes magnitudes, so uniform shrinkage does not change it;
norms separately track scale. No entropy bonus is added to the ranking loss.

`learning.png`, `entropy.png`, `diagnostics.json`, `evaluations.jsonl` and the
same-named TensorBoard run show the actual latest model. Entropy alone cannot
identify grokking: inspect delayed held-out generalization after training fit
has stabilized. Launch metadata/log:
`runs/neural_hand_strongreg_inputs_20261007/{launch.json,training.log}`.

## Training-pool expansion

User suggested more training data after observed overfitting. Current comparison
collects another32 panels of256 with seed20261015 and `--training-only`.
New cases are independent IID draws with the same frozen plant/distributions.
`scripts/extend_hand_plan_dataset.py` builds a manifest over old and new panels
without duplicating raw files. Original validation/test indices and observations
remain fixed; every new case enters training. Duplicate held-out observations
and incompatible decoder/geometry/sampling metadata are rejected. Counts:
16,384 total /14,746 train /819 validation /819 test.

Fresh fitting keeps the same initialization seed, network, loss and20,000-step
batch256 schedule; increased data is the variable being tested. This lowers
equivalent passes from781 to347. No held-out selection or rollback. Final fresh
matched physics test includes original supervised model and graph, reusing the
earlier final test's rhythm/start-state panel when available.

Pipeline `runs/neural_hand_dataset_moredata_inputs_20261007/run_pipeline.py`;
extra collection `hand_plan_dataset_iid_extra8192_20261007`; combined manifest
`hand_plan_dataset_iid_16384_20261007`; model
`neural_hand_plans_iid_moredata_20261007`; final
`neural_hand_plans_iid_moredata_test_20261007`. Collection usesGPUs0/1/3 because
GPU2 was nearly full; original refinement/test continues separately.

User replaced coupled simulator/fitting rounds with dataset-first learning.
Collect32 panels ×256 independently sampled cases =8,192 cases. Each case is
a four-hit-group window. Each group independently selects one of28 equally
likely sets (7 single drums,21 distinct two-drum sets); all four time intervals,
including first lead, are independently uniform over20–900ms in20ms ticks.
Velocities are uniform0.2–1.0. No forced request anchors, assigned hands or
balanced tempo tiers. This covers every legal four-group request pattern with
positive probability, while keeping the existing16-action representation.

Starting states remain80% real measured post-hit,15% collision-clear random
arm/waist,5% ready. A reusable verified post-hit bank seeds collection; only
physical q/qd/tips are reused, not previous neural weights or outcome labels.
The bank refreshes from new successful contacts. Decoder and geometry stay
fixed,250Hz physics. Every legal plan is executed once per case, including
single-plan cases. Enumeration is canonical, independent of a learned policy.

`scripts/collect_hand_plan_dataset.py` writes atomic per-panel .pt batches and
JSON physical reports, with exact initial states, all legal-plan masks and
per-hit contact credits/proximity. `collector_state.pt` preserves NumPy RNG,
post-hit bank, exact pending panel and collection clock for resume. Dataset
splits are by whole rhythm/start-state case:6,554 train /819 validation /819 test.
Every plan of one case stays in one split. Held-out labels never enter replay,
choose a checkpoint or guide simulator sampling.

`scripts/train_hand_plan_dataset.py` starts the same136-input105,488-parameter
MLP from scratch, with the existing2×near-best CE + weighted pairwise loss.
Train20,000 AdamW steps, batch256, warmup100→3e-4 then cosine→3e-5. At8,192 cases
this is about781 equivalent training passes. Validation/test metrics are
read-only; latest fixed-budget weights are used. Full model/optimizer/RNG
checkpoints and train-only replay seed are saved. `learning.png` and TensorBoard
show fitting and generalization separately.

Online refinement initializes model, optimizer, RNG, physical start bank and
train-only replay from the supervised checkpoint. Ten top8 updates then ten
top4 updates,256 IID fresh cases/update,50 optimizer steps/update at constant
3e-5, absolute optimizer clock20,000→21,000. No warmup restart. Test split stays
read-only. A final fresh256-case IID matched physics test compares refined
model, its offline initialization, previous fixed-first all16 model and graph.

Pipeline: `runs/neural_hand_dataset_iid_inputs_20261007/run_pipeline.py`.
Artifacts/TensorBoard run labels:
- `hand_plan_dataset_iid_8192_20261007`: physics collection/progress.
- `neural_hand_plans_iid_supervised_20261007`: offline fitting.
- `neural_hand_plans_iid_finetune_20261007`: small online refinement.
- `neural_hand_plans_iid_final_test_20261007`: fresh comparison.

The old procedural generator remains explicitly available as `coverage` for
reproducing archived experiments. New neural defaults are `iid`/`iid_random`.
There was a temporary tier-random-first prototype; that full pipeline was
never launched and is superseded by this independent sampling workflow.
