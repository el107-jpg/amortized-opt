# CMA-ES conditional-chart search

`scripts/train_chart_cma.py` optimizes the conditional chart's eligible
time anchors using pycma and actual Isaac contact credit. The trajectory
decoder, PD/PID, waist feedforward and contact-compliance controller are frozen.
This is a separate assignment-parameter search, not phase-2 decoder training;
the existing phase-2 critic/Bellman training is unaffected.

Install the optional dependency:

```bash
/data2/eric/conda_envs/g1drum/bin/python -m pip install -e '.[assignment-search]'
```

Resolve the setup without starting Isaac or writing a training run:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_chart_cma.py \
  --out runs/chart_cma_trial --dry-run
```

Start the default search: 4,096 offspring, each with one randomized rhythm:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/data2/eric/conda_envs/g1drum/bin/python scripts/train_chart_cma.py \
  --out runs/chart_cma_trial --device cuda:1 --sampling independent \
  --population 4096 --scenarios 1 --validation-scenarios 0 \
  --envs 4096 --decode-chunk 512 --collision-chunk 1024 \
  --no-plan-diagnostics --planning-workers 4 --groups 8 --horizon 4 \
  --sigma 0.15 --generations 100
```

The existing base chart, flat conditional overlay and matching measured ready
arrival calibration are CLI defaults. `--table`, `--overlay` and `--ready-initial`
can select compatible alternatives. The frozen decoder is taken from the base
chart's config and its checksum is verified. Only nominal ready starts are
currently supported; varied starts need their own measured arrival calibration.

An explicitly requested newer decoder can use the calculated matrices via
`--decoder-checkpoint PATH`. Joint order, drum count, horizon and control time
grid are checked. Timing/ready matrices retain their original decoder
provenance, both decoder hashes are recorded, and resume requires the exact
frozen rollout decoder. This does not regenerate the matrices.

## Optimization and evaluation

New searches initialize unknown eligible cells by default: each of the 22
unknown entries receives a seeded uniform 250–350 ms value, shared across its
three anchors. All 85 eligible cells can then be tuned (255 parameters). Use
`--no-initialize-unknown` to retain the original 189-parameter search or configure
`--unknown-center-s`, `--unknown-half-width-s`, and `--unknown-seed`.
Existing learned anchors can be supplied with `--overlay`; only remaining
unknown cells are initialized. Guesses carry an explicit mask and preserve the
base support/trials, so they are never reported as measured physical minima.

The initial search mean is zero. Normalized residual `x` becomes
`initial_anchor + max(initial_anchor, 0.1 s) * x`; zero therefore reproduces the
supplied overlay. The full covariance and exploration width are updated by
pycma. Ineligible cells and uninitialized unknown cells retain their base values.

Curves must satisfy the conditional overlay's positive floor over the full
proxy domain, including interior vertices. Invalid proposals are resampled up
to a bounded budget before physics evaluation. If too few feasible parents are
available, the covariance/mean update is skipped and exploration narrows. A
discrete assignment plateau widens exploration within a bound instead of letting
the small residual regularizer disguise identical behavior as useful learning.

The default independent mode pairs each parameter offspring with one fresh
**procedurally generated** rhythm (`--rhythm-source procedural`). No recorded
training clips are loaded. Eight complete groups are made from fresh random
single drums and distinct two-drum combinations. Each batch balances all49
ordered drum-request pairs and21 simultaneous combinations across three timing
levels:20–120 ms,120–320 ms,320–900 ms inter-group gaps. A random adjacent
single-hit anchor and a random pair slot guarantee coverage; remaining groups
are independently random, with35% pair probability. First hit is1 s from rest,
and loudness is uniform0.2–1.0. Times remain on the20 ms control grid.

The chart chooses all hands; request coverage does not force the selected
hands. Per-generation metrics log actual hand/drum and hand-transition counts,
so missing chart-cell exposure remains visible. Left-hand floor tom stays
excluded. A small generated seed JSON only allocates scene beat buffers and is
replaced by each offspring's own generated rhythm before simulation.

`--fitness-ranking speed` compares percentile fitness ranks within each timing
level before CMA recombination, allowing hard fast cases to contribute alongside
easy slow ones. The raw hit/proximity/physical-penalty reward stays unchanged;
raw reward metrics are separate from the ranks passed to CMA. Legacy recorded
mode is explicit (`--rhythm-source recorded --fitness-ranking global`) and
samples complete unmodified excerpts with balanced target-drum exposure.
Drum geometry, dynamics and the ready start remain nominal.

Each candidate has exactly one physical episode per generation, rather than a
cross product of offspring and shared rhythm scenarios. Independent offspring
are never deduplicated into the same physical lane. This gives noisy single
episode fitness, which is handled by population ranking and CMA recombination.
The exported current chart is **CMA's updated population mean**. An individual
high scorer on a different rhythm cannot be compared to an incumbent score or
called a better chart merely because its assigned rhythm was easier. There is
no validation/promotion gate or additional incumbent/graph rollout in this mode.

The old matched mode remains available explicitly:

```bash
... scripts/train_chart_cma.py --out runs/chart_cma_matched --sampling matched \
  --population 32 --scenarios 16 --validation-scenarios 8 --envs 64
```

Both modes preserve all hits within complete groups and search the next 4 or 6
groups. Each
candidate holds its parameters throughout an episode. Timing state carries
scheduled drum history, matching the earlier chart/graph comparison. The selected
hands are installed into both the MPC schedule and original Isaac hit-credit
labels. No analytic/synthetic rebound replaces PhysX contacts.

In matched mode, exactly identical static assignments share a rollout *within the same matched
scenario*. The incumbent chart and original graph are evaluated on that same
panel. Reward is the mean per-episode credited-hit fraction minus:

- Average summed positive **computed torque demand** excess, relative to the
  velocity-dependent available torque (default weight 0.01, scale 5 Nm).
- Average maximum joint-limit excursion (weight 0.1, scale 5 degrees).
- Geometry penetration evaluated from **measured** Isaac joint states
  (weight 0.05, scale 5 mm, sampled every 5 control ticks by default).

The small normalized residual penalty is applied by the optimizer in both modes. Applied
actuator torque can be saturated; computed demand is intentionally reported
separately. Collision fitness is measured-state FK geometry, not the player's
uninstrumented collision counter, and is not a collision-free certificate.
`--torque-weight`, `--joint-weight`, `--collision-weight` and
`--collision-stride` configure these terms.

New searches also use bounded partial reward for missed hits:

```
per-hit quality = 1                         if Isaac credited the hit
                 0.25 * exp(-distance/0.05) otherwise
```

Distance is the closest measured assigned-hand stick-tip distance to the legal
head disk/half-disk, sampled every20 ms within±80 ms of that hit's scheduled
time. Geometry uses simulator body-derived tips in the environment frame;
there is no planned-trajectory or wrong-hand credit. A pass outside the window
earns no partial credit. Half-disk cuts and out-of-plane error are included.
Actual contact credit is unchanged, and a credited hit never also receives
partial credit. The capped bonus is averaged over all requested hits and added
to the previous hit-fraction reward; existing physical/parameter penalties stay.

`--near-miss-weight` (default0.25), `--near-miss-scale-m` (0.05), and
`--near-miss-window-s` (0.08) configure shaping. Zero weight disables it.
Legacy resumes retain weight0 unless an explicit override is supplied. A changed
objective replays the complete pending generation under the new reward while
preserving parameters, covariance, paths, sampling RNG and rhythm pairing.
Logs separate actual `mean_offspring_hit_rate`, `mean_offspring_near_miss_bonus`,
and combined `mean_offspring_hit_quality`; combined quality is not hit accuracy.
Per-episode reports include the median closest distance of missed requests.
Paired8-episode physics smoke reproduced28/70 real contacts exactly while adding
an average0.065 proximity bonus;49 relevant checks passed.

Training windows come from the training RCC split. In matched mode, the validation panel
comes from the evaluation split. A candidate replaces the exported incumbent
only by beating the previous exported chart on the **same** fixed validation
panel, accounting for the residual penalty. Historical scores from different
training panels are never compared as if they were matched. The graph remains
a reported reference; a best chart does not imply it beats the graph.

## Environment capacity

`--sampling independent --population 4096 --scenarios 1 --envs 4096` requests
exactly4,096 offspring lanes and one episode per offspring. Matched mode instead
uses the population/scenario cross product, with additional reference episodes.
The runner processes evaluations in waves if
the logical population exceeds physical scene capacity. Extra scene slots stay
idle. `--decode-chunk` bounds decoder and trajectory-cost work, while the frozen
model/PID are shared across candidates.

`scripts/benchmark_chart_cma_capacity.py` measures an all-lanes-active scene
without CMA updates or training fitness. Its10-tick512-environment probe used
about3.8 GiB of total GPU memory. Full capacity results are in
`runs/chart_cma_capacity_4096_20261006/`; short probes do not establish sustained
full-episode peak memory or training speed.
The4096-lane probe completed at about6.2 GiB total GPU usage. For large nominal
scenes, `--clone-kit-from-source` avoids serial USD spawning of identical kits.
At512 lanes, scene creation dropped from55.8 s to2.0 s with the same memory and
decoder-pass count; an8-episode full contact smoke exactly reproduced all
per-episode/per-drum hit counts.4096 replicated kits were also verified.

## Physics frequency

`--physics-hz` configures chart-search simulation while keeping control and the
frozen decoder on the20 ms grid. Default/legacy250 Hz means5 substeps per control
update;100 Hz means2. Rates must be positive integer multiples of50 Hz. Saved
rates persist on resume; an explicit flag overrides them. Evaluation inherits
and records that rate; replay uses the test artifact's rate (legacy250 Hz).

CPU planning can scale separately: a matched4096-case benchmark took23.20 s
with4 workers versus6.97 s with16, with identical decisions. The active run uses
`--planning-workers 16` while retaining250 Hz physics.

The current controller's matched32-case100 Hz test degraded true hits from
132/339 to23/339 and average torque-demand excess from5.06 to171.07 Nm, despite
reducing rollout93.98→54.20 s. Live training retains250 Hz;100 Hz requires
controller/integration stabilization first. Evidence:
`runs/chart_cma_100hz_inputs_20261006/physics_verification.json`.

## Checkpoints and monitoring

For multiple GPUs, add `--rollout-devices cuda:0 cuda:1 cuda:2 cuda:3`.
`--envs 4096` is the total requested capacity, giving1024 environments per
worker on four GPUs. Each worker keeps its own Isaac scene, frozen decoder and
controller alive across generations. The coordinator distributes contiguous
offspring shards, gathers reports in original offspring order, and performs
one global CMA update after all results arrive. It does not average separate
optimizers or evaluate any offspring on extra rhythms. Capacities that do not
divide evenly round each worker's allocation up; extra slots stay idle.

Only the coordinator writes optimizer checkpoints and TensorBoard events.
`rollout_workers.json` records worker PIDs, GPUs and log folders; the session's
`progress.json` records per-worker rollout progress. Worker failure aborts the
generation before the optimizer update, preserving the pending checkpoint for
resume. Explicit `--rollout-devices` overrides saved devices on resume while
preserving the pending offspring and rhythm panel. Scenes start in isolated
subprocesses; the coordinator does not initialize Isaac/CUDA. Physical random
positions and dynamics remain disabled. Scene capacity/layout changes can
introduce GPU/PhysX floating-point differences, so bit-exact physical results
are not a resume guarantee. The first full four-GPU generation took192.65 s
versus322.69 s on one GPU, with4096 valid offspring; rollout167.50 s versus
296.52 s. This consecutive-generation comparison excludes startup and uses
different fresh procedural panels. Evidence:
`runs/chart_cma_multigpu_inputs_20261006/production_timing.json`.

For large populations, use `--decode-chunk 512 --collision-chunk 1024
--no-plan-diagnostics --planning-workers 4`. Decoder batching and measured-state
collision batching are independent. Disabling plan diagnostics omits a full
physics-cost evaluation that only fills an unused planned-collision buffer;
it preserves measured collision fitness, contact credit and control. Spawned
CPU workers parallelize independent hand searches and preserve offspring order.
These execution flags can explicitly override saved settings on resume without
changing the pending candidates, rhythms, optimizer or reward.

A 4096-lane ten-tick probe reduced decoder/plan work from12.04 to5.48 seconds
and ten measured collision calls from14.67 to0.97 seconds. That probe checked
collisions every tick, whereas training defaults to every fifth tick; its total
speedup is not a full-generation estimate. CPU hand search on1024 cases fell
from21.74 to6.33 seconds with identical decisions. A paired64-case full physics
smoke reproduced every per-episode report exactly (211/696 contacts), including
proximity and physical penalties. Generation metrics now include
`planning_seconds`, `rollout_seconds` and `generation_seconds` to measure the
actual benefit at training capacity. Evidence is under
`runs/chart_cma_speed_inputs_20261006/verification.json` and the corresponding
speed benchmark directories. The first accelerated full4096-offspring generation
took326.98 seconds versus564.08 seconds for the preceding generation (1.73×
faster); planning25.92 seconds and rollout299.40 seconds. These consecutive
generations used fresh different procedural rhythms; startup is excluded.

TensorBoard events are written to `tb/`, registered under
`runs/tensorboard_active`, and checked against the shared port-6006 server.
Existing registrations are preserved. Relevant scalars are under `cma/`.

Artifacts:

- `last.pkl`: complete optimizer, private sampling RNG, pending population,
  panel RNG/pending rhythm panel, validated parameters and frozen-decoder checksum.
- `best_chart.npz`: matched mode only, best validation-selected overlay, usable with
  `probe_min_time_chart_choices.py --conditional-chart`.
- `initial_chart.npz`: exact starting overlay including randomized guesses.
- `last_chart.npz`: current training chart; population mean in independent mode,
  matched training incumbent in matched mode.
- `generation_*.json`, `metrics.jsonl`, `progress.json`, `run.json`: fitness,
  rejected/resampled proposals, schedule reuse and rollout measurements.

Display and evaluation use the current training chart (`last_chart.npz`) by
default, as requested by the user. `best_chart.npz` is a separate validation
diagnostic. `evaluate_chart_cma.py --chart-selection training` is the default;
`--chart-selection validation` must be explicit to use that diagnostic.
`--learned-overlay` pins an immutable snapshot of either chart. Reports and
figures record and label the actual chart selection; performance must be
measured on the same snapshot being shown.

Resume locally produced/trusted checkpoints with an absolute generation budget:

```bash
/data2/eric/conda_envs/g1drum/bin/python scripts/train_chart_cma.py \
  --out runs/chart_cma_trial --resume runs/chart_cma_trial/last.pkl \
  --generations 200 --envs 128 --device cuda:1
```

Saved settings are restored except output directory, generation budget, device
and physical environment capacity. Overlay compatibility and frozen-decoder
checksums are validated. A checkpoint after `ask()` reuses the same unfinished
population and panel rather than drawing replacement candidates. Algorithmic
RNG state is preserved; GPU/PhysX episode outcomes need not be bit-exact.
Legacy checkpoints retain their original unknown mask and parameter dimension;
the new initialization default does not change them on resume. Changing the
mask/dimension requires a fresh search instead of reusing the old covariance.
Legacy configurations without a sampling mode resume as matched. Independent
checkpoints preserve the pending offspring-to-rhythm pairing and private RNGs.
Legacy configurations without `rhythm_source` resume as recorded. Switching
explicitly to procedural preserves CMA/parameter/RNG state but replaces an
unfinished recorded panel. Procedural pending panels include their timing-level
labels, preserved exactly on normal resume. Recorded rhythms remain an
independent end-of-run test; they do not gate the actual training chart.
Population size and sampling mode are fixed within a resumed run; a new run can
warm-start from existing learned curves when changing either.
Physics/asset/corpus files should stay fixed during a search.

The fingerprinted RNEA cache is pinned for both controller and decoder features.
On Linux, SIGUSR1 writes a diagnostic stack to the run log if setup is slow.

## Verification completed

Unit checks cover optimizer improvement on a controlled black-box objective,
private-RNG and pending/completed-generation resume, constraint rejection and
resampling, static-schedule deduplication, matched aggregation, and microbatch
controller parity. Actual Isaac smoke and resume artifacts:
`runs/chart_cma_isaac_smoke_feasible_20261006/`.
Both generations evaluated four feasible candidates and performed CMA updates.
This tiny smoke validates the implementation, not learned performance: all
sampled charts selected identical schedules on its very small panels and the
exported incumbent remained at its initialization.
