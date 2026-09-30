# `amortized/` — the G1 drummer amortized trajectory optimizer

> ## → **[`REPORT.md`](REPORT.md) is the final evaluation report.**
>
> Start there. It is the standalone write-up of the finished system: the
> headline numbers, the method in one page, every open-loop / closed-loop /
> generalization table, the honest limitations, and the RL-readiness
> recommendation. This README is the **build log** — one chapter per section of
> `docs/amortized_optimizer_plan.md`, in the order the work happened, with the
> DECISIONS lists that `REPORT.md` cites.
>
> The short version: **champion `runs/phase2b_main/best.pt` + the per-joint
> reference lead.** Open loop it beats the trajopt HPIPM expert outright —
> 175 / 180 / 178 of 243 on grid seeds 0 / 7 / 13 against the expert's 149 —
> at 17.9 ms a tick against 6 514 ms. Closed loop it lands 76 of 204 scheduled
> strikes (63 % miss) with a 9.5 ms median timing error. The gap between those
> two sentences is §7 of the report and is the reason the next stage is the
> IsaacSim RL fine-tune rather than more offline descent.

Section 1 of `docs/amortized_optimizer_plan.md`: a batched, differentiable,
Isaac-free cost-function substrate for a 17-DoF G1 upper body (3 waist + 2×7
arm) playing a drum kit.

**This folder is self-contained.** Nothing under `amortized_opt/` imports the
main repo, Isaac Lab, or `UNITREE_MODEL_DIR`. The G1 URDF, its meshes and the
drum-kit assets are vendored under `assets/`. Copy the folder anywhere with a
suitable conda env and it runs. (Only the *conformance tests* import the repo
originals, and they skip cleanly when the repo is absent.)

As of 2026-08-12 that holds for the **Isaac layer** too: `g1_scene/` vendors the
G1-with-sticks articulation cfg and the drum-kit/layout paths that
`isaac_rl*/env.py` used to import from `g1_drummer`, so the RL env builds
without the repo on the path. `pyproject.toml` makes the folder
`pip install -e .`-able, and `CLAUDE.md` carries the environment preamble.
Two files still need the original checkout, both benchmarking/playback tools
rather than parts of the system, both listed in `tests/test_standalone.py` and
both failing with an explanation rather than a traceback:
`scripts/repo_adapters/trajopt_hpipm_planner.py` (the trajopt baseline) and
`scripts/play_isaac.py` (drives the repo's own 6 374-line `G1DrummerEnv`).
See [`EXTRACT.md`](EXTRACT.md) for how to lift the folder out into its own repo.

## Contents

```
assets/
  g1_29dof_rev_1_0.urdf     # G1 29-DoF URDF (identical to the repo's copy)
  meshes/*.STL              # all 35 meshes the URDF references (19 MB)
  drum_kit_layout.yaml      # drum positions / radii / heights / normals
  drum_goal_poses.json      # per-drum terminal joint poses (phase-1 seed library)
amortized_opt/
  constants.py   control grid (50 Hz waypoints / 250 Hz PD), joint order,
                 stick + torso geometry, asset paths
  limits.py      effort/velocity/stiffness/damping/armature tables, the URDF
                 joint box, and derive_accel_limits
  fk.py          ChainFK — sync-free batched differentiable FK, five frames
                 (left/right wrist, left/right elbow, torso) + stick helpers
  rnea.py        BatchedRNEA — sync-free, autograd-safe, armature-correct
  sdf.py         SdfPenalty — analytic drum / under-drum / self-collision SDF,
                 branch-free target-drum exemption, optional shared FK input
  spline.py      quintic seed machinery (path spline + timing law)
  tokenizer.py   token / context builder: dual terminal-anchored Fourier time
                 codes, per-element horizon masks, drum / goal / future payloads
  model.py       context encoder + trajectory decoder + auxiliary heads
g1_scene/        Isaac scene assets, vendored from the repo (2026-08-12)
  g1.py          get_g1_cfg — composes the G1-with-sticks USD; stick and
                 body-collider geometry IMPORTED from amortized_opt.constants
                 so PhysX and the analytic SDF cannot drift apart
  layout.py      drum-kit USD / layout paths + the id-contiguity check the
                 contact-sensor filters depend on
isaac_rl_ppo/    the current Isaac RL env + PPO loop
isaac_rl/        the older RL tree (PSN / safety experiments)
tests/           conformance, autograd, sync-free, standalone and speed tests
pyproject.toml   pip install -e . (editable only — assets/ sits beside the
                 packages, so a built wheel would not carry it)
```

## Environment

On this box:

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export TMPDIR=/data2/eric/tmp
```

Elsewhere, the package needs `torch`, `numpy`, `scipy`, `pyyaml`, and
`pinocchio` (Pinocchio is a **build-time** dependency only: it is used to
extract inertias for `BatchedRNEA.from_urdf` and to compute
`derive_accel_limits`; evaluation is pure torch).

**GPU note.** `torch.cuda.set_device(idx)` is mandatory at the top of every
entry point. Without it, inductor's CUDA-graph capture silently no-ops when the
tensors live off `cuda:0` (`UserWarning: The CUDA Graph is empty`) and the
entire speedup disappears. On this box GPUs 0 and 2 are held by concurrent
sessions — use `cuda:1` or `cuda:3`.

## Running the tests

```bash
cd amortized
python -m pytest tests -q
```

Override the GPU with `AMORTIZED_TEST_DEVICE=cuda:3`. Tests that compare against
the in-repo originals skip automatically if `../source/g1_drummer` is missing.

## Quick start

```python
import torch
import amortized_opt as A

dev = "cuda:1"
torch.cuda.set_device(1)                       # mandatory, see above
fk   = A.ChainFK(device=dev)
rnea = A.BatchedRNEA.from_urdf(device=dev).to(dev, torch.float32)
sdf  = A.SdfPenalty(fk=fk, device=dev)         # shares the FK instance

q = torch.zeros(12_800, 17, device=dev, requires_grad=True)
frames = fk.forward(q)                          # ONE FK pass...
pen    = sdf.forward(frames=frames)             # ...reused by the SDF
tau    = rnea.rnea(q, q, q)                     # includes diag(armature) @ qdd
(pen.sum() + tau.square().sum()).backward()
```

`frames` keys: `left_wrist`, `right_wrist`, `left_elbow`, `right_elbow`,
`torso`. Stick tip / shaft world points come off a wrist frame with
`A.stick_tip_offset(hand)` / `A.stick_shaft_offsets(hand, n)` and
`ChainFK.points_world`.

## What was fixed relative to the repo originals

| | defect | fix |
| --- | --- | --- |
| FK | ~27 device→host syncs per forward (`q[:, cuda_scalar]`) | topology hoisted to Python ints + a cached device index tensor; Rodrigues `K`, `K²` precomputed; constants pre-cast per dtype |
| FK | only 3 frames (wrists + torso) | elbow frames added for both arms |
| RNEA | backward raised (in-place index assignment) | per-link Python lists rebound, then stacked |
| RNEA | ~60–70 syncs per call (`int(self.parent[i].item())`) | topology as Python ints; `index_select` for the opt↔pin maps |
| RNEA | **armature entirely absent** — torques optimistic exactly during the strike | `tau += diag(isaac_armature) @ qdd` (3.0 N·m at 300 rad/s² against a 5 N·m wrist limit) |
| SDF | `if bool(valid.any())` sync in the hand loop | branch-free `F.one_hot(tgt.clamp(min=0), D) * (tgt >= 0)` |
| SDF | recomputed its own FK | accepts an externally-computed `frames` dict |

A CI guard (`tests/test_autograd_and_sync.py::test_no_device_host_sync`) asserts
zero syncs under `torch.cuda.set_sync_debug_mode("error")` for FK, RNEA, SDF and
the fused stack — the regression it prevents cost 15× and failed silently.

## Measured (RTX 5090, `cuda:1`, batch 256 × 50 = 12 800 poses, fp32)

Whole FK + SDF + RNEA forward **and backward**, one shared FK pass:

| | ms/step | steps/s |
| --- | --- | --- |
| eager | 175.8 | 5.7 |
| `torch.compile(mode="default")` | 62.0 | 16 |
| `torch.compile(mode="reduce-overhead")` | **13.2–13.5** | **74** |

Removing the syncs is what makes the `reduce-overhead` path (CUDA-graph capture)
reachable at all; capture is forbidden in the presence of a sync.

## The transformer (`model.py` + `tokenizer.py`)

```python
import torch, amortized_opt as A

torch.cuda.set_device(1)
dev = "cuda:1"
model = A.AmortizedOptimizer(A.ModelConfig(max_horizon=100)).to(dev)

ctx = A.ContextInputs.zeros(B, num_drums=8, num_future=3, device=dev)
ctx.__dict__.update(A.drum_tensors_from_scene(scene, batch=B, num_drums=8, device=dev))
...                                        # fill goal_* / future_* per arm
enc = model.encode_context(ctx)            # ONCE per context change, then cached

masks = A.build_waypoint_masks(t_hit_left, t_hit_right, max_horizon=100)
out = model(init_traj, A.pack_state(q, qd), enc, masks)   # one MPC iteration
out = model(out.traj, A.pack_state(q, qd), enc, masks)    # ...refinement is a re-feed
```

* **Tokens.** One position-context token (17 q + 17 qd), `max_horizon` waypoint
  tokens (17-DoF joint position each, spaced `CONTROL_DT`), and a cached context
  of `num_drums` geometry tokens + one goal token per arm + `num_future` per arm
  (16 by default), with type / arm / drum-id / future-rank embeddings.
* **Temporal code.** Two Fourier encodings per waypoint — time-to-left-hit and
  time-to-right-hit, each anchored so a token *on* that hand's hit gets the
  zero-phase code (`[0]*K + [1]*K + [0]`). The code follows the horizon, not the
  token index.
* **Per-element masking** (resolved decision 2): left-arm elements masked by the
  left horizon, right-arm by the right, waist by the longer of the two. Masked
  elements are zeroed *before* the input projection and in every output;
  fully-masked tokens are additionally dropped from attention. `out.elem_mask`
  and `out.point_mask` are returned so the loss can exclude them.
* **Output.** ONE 17-DoF trajectory (`out.traj`), plus per-waypoint aux heads:
  `vel`, `acc`, `torque` (17 each) and `fk_pos` / `fk_vel` `(B, H, 6, 3)` over
  `FK_POINT_NAMES` = tip / wrist / elbow for both arms. `output_mode="residual"`
  switches to init + correction (zero-init, so it starts as the identity).
* **Sizes** (defaults): encoder 2 × d=256, decoder **3** × d=384 / 8 heads /
  head_dim 48 / SwiGLU 4× / pre-LN RMSNorm. **10.76 M** params (encoder 2.11 M,
  decoder body 8.61 M, heads 0.04 M). The default dropped from 6 decoder layers
  / 19.32 M on 2026-08-12 — see §8's model-size ablation. Override with
  `--dec-layers 6` on `train_phase1.py` / `train_phase1_apex.py` /
  `train_phase2.py` to rebuild the old size. **Existing checkpoints are
  unaffected**: every eval, deploy and resume entry point rebuilds `ModelConfig`
  from the checkpoint's stored `model_cfg`, so the 6-layer champion
  (`runs/phase2b_main/best.pt`) still loads `strict=True`.

Measured on `cuda:1`, one decoder pass, H = 100, `fullgraph=True`,
`mode="reduce-overhead"`:

| batch | eager | compiled | compiled + TF32 |
| --- | --- | --- | --- |
| 1 (one MPC tick) | 6.30 ms | 1.15 ms | **0.53 ms** |
| 256 (training) | 28.7 ms | 26.6 ms | 14.8 ms |

TF32 is not enabled by the package — call
`torch.set_float32_matmul_precision("high")` in the entry point to get the third
column (bf16 autocast measures 0.45 ms / 8.1 ms).

## The evaluation harness (`strike.py` + `eval_harness.py`)

Section 2 of the plan: one grid and one scorer that any trajectory-producing
callable can be measured on, so the amortized model's numbers are directly
comparable to the trajopt expert's.

```
amortized_opt/
  strike.py        drum-strike geometry ported from trajopt/problem.py --
                   aim point, hit disk, half-plane cut, contact sigma
                   (cymbal/hi-hat shaft hits), elevation band, impact-velocity
                   weight. Also feeds the section-3 terminal cost.
  eval_harness.py  EvalCase / PlanResult / TrajectoryScorer / generate_grid /
                   run_eval / save_run / load_run
scripts/
  run_eval.py             CLI: solve (optionally pooled) -> score -> persist
  make_test_fixture.py    carve a known-good fixture out of a recorded run
  repo_adapters/          THE ONLY repo-importing code under amortized/
    trajopt_hpipm_planner.py
baselines/                recorded reference runs (CSV + NPZ + summary)
```

A **planner** is any `plan(case) -> PlanResult | (T,17) array` on the 50 Hz
grid, row 0 the start pose and row `case.N` the strike waypoint.

```python
import amortized_opt as A

scorer = A.TrajectoryScorer(device="cpu")       # cpu is fastest at these sizes
cases  = A.generate_grid(seed=0, grid="standard", scorer=scorer)   # 162 cases
rows, trajs = A.run_eval(my_planner, cases, scorer=scorer)
print(A.format_summary(rows))
```

**Grids.** `standard` reproduces the repo's 162-case acceptance grid cell for
cell — 6 drums × 3 horizons (0.45/0.60/0.80 s) × 3 impact speeds (1/2/3 m/s) ×
3 jittered starts, three drums per hand. `near_drum` is the training
distribution: starts *on* another drum (that drum's pose from
`drum_goal_poses.json`, jittered) departing with a noised outward tip velocity
mapped through a damped pseudoinverse. Same seed → byte-identical cases.

**What is scored** (all 17 DoF, pelvis frame, vendored physics only): contact
position/speed/direction at the strike — at the `contact_sigma` point, so a
cymbal is scored at the stick shoulder, not the bead — hit-disk membership,
stick elevation, collision on the *interpolated* path with a target-drum
exemption over the last 2 waypoints, RNEA torque with armature against the
Isaac effort limits, velocity and joint-box excess, joint travel, and jerk /
accel integrals.

**Re-scoring.** Runs persist their trajectories, so a metric change is re-run
against stored paths rather than by re-solving:

```bash
python scripts/run_eval.py --rescore baselines --prefix trajopt_hpipm_2026-08-10
```

This matters because the repo SQP is not reproducible run-to-run (about a third
of cases move), so re-solving would confound a metric change with solver noise.

**The baseline.** `baselines/trajopt_hpipm_2026-08-10.md` records the repo's
HPIPM SQP over the same 243 cases three times, differing only in how the waist
is driven (all three share the apex-in-cylinder seed):

| | waist | accepted | coll-free striking / both |
| --- | --- | ---: | ---: |
| `_v1_` | free QP variable | 151 | 160 / 7 |
| `_v2_` | repo aim law, pinned | 126 | 148 / **83** |
| **`_v2b_`** | **+ teacher `refine_yaw`** | **149** | **162** / 66 |

**`_v2b_*` is the primary reference line** — the strongest correctly-specified
expert. `_v2_` is kept as the live-pipeline-convention datapoint and is the
better run on both-arms collision-freedom; `_v1_` is kept for provenance and
for work already pinned to its NPZ. None of them is fast enough for MPC, which
is the point.

Read the caveats before trusting any amortized number — in particular, trajopt
is scored on its *native, easier* problem (one arm plus the waist; the idle arm
is frozen and invisible to its own collision model).

## The fused cost (`tspline.py` + `cost.py` + `lagrangian.py`)

Section 3 of the plan: the differentiable objective the transformer is trained
against. One FK pass over the sub-sampled spline feeds every geometric term, one
RNEA pass gives the torque, and the spline coefficients give the velocity maxima
and the smoothness integrals in closed form.

```
amortized_opt/
  tspline.py     TrajectorySplineBuilder / TrajectorySpline — batched C2 cubic
                 over the 50 Hz grid, clamped to v_start, natural at the strike;
                 per-hand horizons; closed-form ∫v²/∫a²/∫j²; exact per-segment
                 |q̇|max; the scorer's knot finite differences
  cost.py        CostConfig / StrikeContext / FusedCost / CostTerms
  lagrangian.py  AugmentedLagrangian — one multiplier per constraint group
scripts/
  validate_cost.py  the whole validation report against the recorded baseline
  bench_cost.py     eager / compiled throughput + a sync-debug check
```

```python
import torch, amortized_opt as A

torch.cuda.set_device(1)
cost = A.FusedCost(A.CostConfig(), max_horizon=50, device="cuda:1")
al   = A.AugmentedLagrangian()

out = cost.forward(q0, waypoints, v0, ctx, weights=al.weights())
out.total.sum().backward()
al.update(out.violations)      # after the optimizer step
```

**The spline.** A C2 cubic on the uniform time grid — the same rule
`spline.joint_path_spline` uses (`make_interp_spline(..., k=3)`), conformed to
scipy at ≤1e-6 for clamped/natural and not-a-knot ends. Solving for the knot
derivatives is a linear system whose matrix depends only on the segment count on
a uniform grid, so the gains `A⁻¹B` are tabulated per horizon at construction and
the hot path is one gather plus one batched matmul. Rows and columns past a
horizon are exactly zero, which is what makes the per-hand masking bit-exact.

**Cost terms.** `terminal` (contact point inside the legal hit region, impact
velocity, stick-elevation band — per hand, at that hand's own hit index, with a
soft-min over the legal stick span so a cymbal may be sounded anywhere on the
last 15 cm), `collision` (drums + under-volumes + torso/head, with the scorer's
target-drum and departure exemptions), `torque`, `velocity`, `joint_box`,
`smooth`, `posture` (mid-range well, elbow-down, anti-axial-twist). Every hinge
is `softplus((x−lim)/β)·β` against a conservative limit (torque 80 %, velocity
70 %, box 80 %); every term reduces to a mean over valid elements and reports its
element count, so a training loop can renormalize across horizon lengths.

**Augmented Lagrangian.** The five constraint groups also report a *raw*
violation — zero when feasible, unlike the softplus — and `AugmentedLagrangian`
integrates it into a per-group multiplier that scales that group's penalty.
Save/loadable JSON state.

**Measured** (RTX 5090, `cuda:1`, batch 256 × 50, fp32, fwd+bwd):

| n_sub | FK poses | eager | compiled (`reduce-overhead`) |
| --- | --- | --- | --- |
| **1** (knots) | 13 056 | 222 ms | **14.4 ms — 69 steps/s** |
| 2 | 25 856 | 215 ms | 21.3 ms — 47 steps/s |
| 4 | 51 456 | 227 ms | 35.3 ms — 28 steps/s |

`n_sub = 1` is the default and the only setting inside the 20 ms budget. The
price is collision resolution: at `n_sub = 4` the cost's `min_sdf` matches the
scorer's 4-substep number to 1.4 cm (residual = spline vs linear interpolant);
at `n_sub = 1` the worst-case gap is 4.6 cm. `tests/test_cost.py::
test_no_device_host_sync` asserts the hot path stays sync-free.

**Validated against the recorded baseline** (`scripts/validate_cost.py`, run on
`_v2`, 243 cases, 126 accepted). `tests/test_cost_validation.py` asserts the
thresholds.

*Conformance to the scorer* — `torque_max` / `torque_excess_max` agree to
**0.0** (exactly; same RNEA, same finite-difference convention), contact
position to **6e-11 m**, stick elevation to **3e-14°**.

*Each term separates the scorer's own verdict* (AUROC of the term against the
harness flag it prices):

| term | flag | AUROC | median fail / ok |
| --- | --- | --- | --- |
| `torque` | `!ok_torque` (56) | **0.983** | 2.6e-2 / 7.4e-3 |
| `velocity` | `!ok_velocity` (1) | **1.000** | 2.9e-1 / 2.3e-10 |
| `collision` (striking hand) | `!ok_collision` (95) | **0.944** | 3.7e-5 / 1.3e-9 |
| `terminal_region` | `!ok_hit` (5) | **0.954** | 1.2e-1 / 3.5e-3 |
| `terminal_velocity` | `!ok_speed` (2) | **0.836** | 1.7e-1 / 4.4e-2 |

Total cost, accepted vs rejected: median 8.86 vs 17.08, AUROC 0.767.

*Monotone in noise* — mean total on accepted cases at σ = 0 / 1 / 2 / 5 / 10 /
20 / 50 mrad: **10.2 → 14.3 → 26.5 → 112.8 → 420.5 → 1649 → 10237**, and
**100 %** of individual cases rise at every step.

*Descendable* — 32 accepted cases perturbed by σ = 20 mrad i.i.d. per waypoint
(acceptance 1.00 → 0.00), then Adam on the waypoints under the cost alone:

| | cost | speed err | τ excess | on head | jerk ∫ | accepted |
| --- | --- | --- | --- | --- | --- | --- |
| trajopt | — | 0.055 | 0.0 | 1.00 | 1.31e7 | 1.00 |
| perturbed | 1639 | 0.904 | 220.8 | 0.75 | 1.26e9 | 0.00 |
| + 50 steps | — | 0.131 | 14.1 | 0.84 | 1.90e7 | 0.06 |
| **+ 300 steps** | **2.47** | **0.109** | **1.12** | **0.88** | **4.65e6** | **0.75** |

All six priced metrics improve. `contact_pos_err` (distance to the *aim* point)
does not, by design: `w_terminal_aim = 0`, matching `validate()`, so the cost
prices hit-*disk* membership (`on_head`), not the aim.

### Cost decisions needing review

1. **`physics_mode = "knot_fd"`.** The torque term is evaluated at the knots
   with the scorer's finite differences, not with the C2 spline's own qd/qdd.
   Rationale: the plant is a 250 Hz PD servo tracking first-order-hold ramps, so
   the second difference is what it is actually commanded to produce, and the
   scorer uses it too. Measured cost of the alternative: on the baseline the
   spline's |q̈| is ~2.3× the finite difference wherever the QP left a one-knot
   zig-zag (537 vs 234 rad/s² at `waist_roll`), and `physics_mode="spline"`
   reports a **median 1.1 N·m** torque overshoot (p90 18.3) on trajectories the
   scorer calls clean — which would have destroyed the torque separation.
   Reversible with one config field.
2. **`velocity_max_mode = "analytic"`** rather than the review's 5 sub-samples.
   Per-segment velocity is a *parabola*, so its extremum is one clamped
   division — exact and cheaper. Measured: 5 probes under-estimate the true
   per-segment |q̇|max by up to **7.4 %**, 9 probes by 2.3 %. (This is not the
   shelved *torque* locator, which would need an RNEA per candidate.)
3. **No velocity-dependent torque ceiling.** The G1's actuator groups are
   `ImplicitActuatorCfg` / `IdealPDActuatorCfg` (`unitree_rl_lab`'s
   `UNITREE_G1_29DOF_CFG`); neither models back-EMF derating and there is no
   `saturation_effort` anywhere to calibrate a linear model from. `DCMotorCfg`
   is not used by this robot. A `torque_derate="linear"` path exists and is OFF.
4. **`terminal_a_tangential = 0.14`, not 0.** `validate()` zeroes trajopt's
   tangential impact-velocity weight; kept here at the *declared* ratio
   (7 / 50) because the harness scores speed magnitude and direction cosine,
   both of which a zero tangential weight leaves unpriced.
5. **`n_sub = 1`** (collision/FK at the knots only). Throughput forces it, but
   it costs collision resolution: worst-case `min_sdf` gap to the scorer is
   4.6 cm at `n_sub=1` vs 1.4 cm at `n_sub=4`. Consider `n_sub=2` (21 ms) if
   between-waypoint tunnelling shows up in training.
6. **Spline end condition is natural** (zero acceleration at the strike). The
   terminal *velocity* stays free so the impact-speed term can price it; the
   alternative (clamping q̇ at the hit) would double-count.
7. **Impact velocity is a chord**, `(p(q_k) − p(q_{k−1}))/dt`, not `J(q_k) q̇_k`.
   The two speeds track closely but their *errors* correlate at only 0.79.
8. **Compiled vs eager gradients differ by up to 2 % relative** on individual
   elements (values agree to 3e-7), from TF32 + reassociation under
   `reduce-overhead`. Fine for training; do not use the compiled path for
   gradient conformance checks.

## Conventions

* Joint order is `constants.G1_UPPER_BODY_JOINTS` (17 entries) everywhere.
* FK frames are **pelvis-rooted**; add `constants.PELVIS_IN_ENV` for env coords.
  `DrumSceneSpec.from_layout(frame=...)` handles this for the drum scene.
* Control grid: `CONTROL_DT = 0.02` (50 Hz waypoints), `PD_DT = 1/250`,
  `PD_SUBSTEPS = 5` — resolved decision 1 of the plan. This folder only; the
  main repo keeps its 60 Hz control period.
* Limit constants are the **runtime** values. In particular
  `isaac_runtime_damping` is 2× the table in `CLAUDE.md` (arms 2.0, waist 10.0),
  because the env scales every actuator group's damping by 2 at load time.

## Data generation (`pd_sim` · `waist_aim` · `ik` · `starts` · `rhythm` · `warmstart` · `mpc_gen`)

Section 5 of the plan: everything that produces training inputs. No labels are
produced — phase 2 back-propagates through the cost — so this stack's job is to
put the model in front of the *right distribution of situations*.

```
amortized_opt/
  pd_sim.py     BatchedPDSim / simulate_pd — the noised PD servo model,
                batched + differentiable; mass_matrix / frozen_minv
  waist_aim.py  VENDORED control/waist_aim.py + aim_targets_from_layout
  ik.py         tip_jacobian, damped_pinv, inertia_weighted_pinv,
                solve_tip_velocity
  starts.py     StartSampler — 90% drum-departure / 10% neutral start states
  rhythm.py     DrumKit (+ domain randomization), RCC loading, hit
                quantization, goal snapshots -> ContextInputs
  warmstart.py  GoalPoseLibrary, warmstart_trajectory (the quintic seed),
                InitNoise (OU + white)
  mpc_gen.py    MPCGenerator — the simulated-MPC loop; ReplayStore; collate
assets/rcc/
  train/*.json  10 clips carved out of data/rcc/generated_long_train
  eval/*.json   4 clips from generated_long_eval
```

```python
import amortized_opt as A

kit    = A.DrumKit.from_layout()
clips  = A.load_rcc_pool(A.default_rcc_dir("train"), kit)   # hits on the 20 ms grid
gen    = A.MPCGenerator(kit=kit)
starts = A.StartSampler(kit=kit, library=gen.library, fk=gen.fk,
                        rnea=gen.rnea, sdf=gen.sdf)

store = gen.generate(clips, starts.sample(256, seed=0), n_episodes=8, seed=0)
batch = store.sample(64, np.random.default_rng(0))          # recency-biased
state, init, masks, ctx, extras = A.collate(batch, device="cuda:1")
```

Swap the planner (`gen.generate(..., planner=my_model_planner)`) and the same
loop becomes DAgger against the trained decoder; `A.quintic_planner()` is the
phase-1 stand-in.

### The PD model

`simulate_pd` is `control/trajopt/closed_loop.py` batched and made
differentiable. Per PD substep of control period *i*:

```
q_tgt  = x_{i-1} + (s/S) * (x_i - x_{i-1})     # the env's FOH position ramp
qd_cmd = (x_i - x_{i-1}) / dt                  # held across the whole period
acc    = Minv_i @ (kp*(q_tgt - qa) + kd*(ff*qd_cmd - va))
```

At `S = 1` the ramp lands on `x_i` and the recursion *is* the repo's, bit for
bit (measured `1.4e-16` / `2.0e-15` on q / v in fp64) — so there is no separate
"conformance mode". The default is `S = PD_SUBSTEPS = 5`, resolved decision 1.
That is not a cosmetic refinement: at 1024×50 the terminal velocity moves by
0.87 rad/s between `S = 1` and the converged answer, and the error falls
monotonically with `S` (8.7e-1, 3.9e-1, 1.4e-1, 6.9e-2, 2.5e-2 for
S = 1, 2, 5, 10, 25).

`Minv` is `inv(M(q) + diag(armature))` frozen per knot; `mass_matrix` gets `M`
from RNEA with unit accelerations, stacked into ONE batched call (matching
Pinocchio CRBA + armature to `5.8e-15`). Kp/Kd are randomized per joint and per
rollout with a log-normal multiplier, so a large sigma cannot flip a gain
negative.

| 1024 rollouts × 50 knots, 5 substeps, `cuda:1` | ms |
| --- | --- |
| eager | 39.7 |
| `torch.compile(mode="reduce-overhead")` | **1.34** |

### Start distribution

90 % of starts put **both** hands on a drum they are allowed to play — the pose
comes from `drum_goal_poses.json`'s waist-yaw grid, jittered — with a departure
tip velocity along the head normal, spread by a Gaussian cone. The two hands'
3-row tip Jacobians are stacked into one 6-row task and inverted with the
inertia-weighted (dynamically consistent) pseudoinverse, so the shared waist is
split by the metric instead of by an arbitrary rule. Measured over 1000 starts:
0 joint-box violations, 0 collisions, 90.4 % drum-proximal, 1.43 draws per
accepted sample.

Inertia weighting moves 21 % → 36 % of the joint speed into the wrists for the
same tip velocity (all 17 DoF), which is the point: the wrists have the least
inertia and the strike is theirs.

### Known rough edges

* The quintic seed is **obstacle-unaware and velocity-unlimited** by
  construction. Over a 120-tick episode ~4–7 % of ticks command a joint past
  its speed rating, and ~15–28 % of states report a negative SDF clearance
  (dominated by the stick-vs-torso self-collision term, an order of magnitude
  above the drum term). Both are what the phase-2 cost curriculum exists to
  fix; the generator records them rather than hiding them.
* `assets/drum_goal_poses.json` predates the `waist_aim` damping change
  0.7 → 0.3. Its per-drum `yaw_grid` (what `GoalPoseLibrary` reads) is indexed
  by yaw directly and is therefore unaffected; only the `pairs` block is stale
  (2 of 33 move).
* `MPCGenerator.run_episode` is single-episode (batch 1) at ~19–23 ms/tick.
  Batching episodes would amortize the RNEA and SDF calls, but it complicates
  the per-hand asymmetric reset, so it is deliberately left for later.
* **`MPCGenerator` used to wind the waist up over long episodes** -- measured
  with the stand-in `quintic_planner` over a 300-tick clip, `waist_yaw` drifted
  monotonically (1.5 -> 2.8 rad over ~90 ticks), left the joint box, and the aim
  then flipped sign, commanding a 5.2 rad step in one 20 ms period (a 262 rad/s
  velocity feed-forward that blew the PD state up to 1e11 rad). **Fixed** as
  phase-2 prerequisite A: the yaw ramp's boundary condition now comes from the
  *planned* state rather than the measured rate, pitch and roll are frozen at a
  planned setpoint rather than at the measured value, and both the seed and the
  executed command are slew-limited. Same clip, same start, after the fix: 300
  ticks complete, `waist_yaw` stays in +0.05 .. +0.80 rad and no command step
  exceeds 0.90x the joint's speed rating. See
  `tests/test_phase2_prereq.py::test_mpc_generator_waist_does_not_wind_up`.

## Phase-1 warm start (`warmstart_data` · `train_warmstart` · `planner`)

Section 6 of the plan, under **resolved decision 5** (phase 1 is a *short* warm
start, 20 k steps hard cap, accuracy is a diagnostic not a gate) and **resolved
decision 6** (bf16, batch ≥ 512, `torch.compile`, horizon bucketing).

```
amortized_opt/
  warmstart_data.py   ScenarioConfig / NoiseConfig / WarmstartPool;
                      generate_scenarios (+ a fork-pool driver), the auxiliary
                      label pass, the OU noise kernel, horizon bucketing
  train_warmstart.py  LossScales / LossWeights / forward_loss / make_step /
                      batch_sampler / train / evaluate / imitation_report /
                      save_checkpoint / load_checkpoint
  planner.py          quintic_case_planner + ModelPlanner (eval-grid planners)
scripts/
  train_phase1.py     generate the pools, run the loop, dump diagnostics
  eval_phase1.py      score a checkpoint on the section-2 grid
tests/test_train_warmstart.py
runs/                 (gitignored) checkpoints + metrics.jsonl + summary.json
```

```bash
python scripts/train_phase1.py --device cuda:1 --steps 20000 --batch 1024 \
    --train-scenarios 450000 --val-scenarios 40000 --workers 64
python scripts/eval_phase1.py --ckpt runs/<run>/best.pt --device cuda:1
```

### The data

Current strike timing (2026-09-23): phase-1 standard targets and fresh strike
initializations use `amortized_opt/strike_seed.py`. Each joint starts at its
measured velocity with zero acceleration, ramps acceleration linearly over
the first 10% of its hand's strike window, then holds acceleration through
contact. With `r = 0.1`, `T = hit_steps * dt`, and `d = q_goal - q0`, the
plateau is `(d - v0*T) / (T*T*(1/2 - r/2 + r*r/6))`. Arrival velocity is free;
an excessive or oppositely directed initial velocity can require braking or
an initial move away from the goal. From rest the joint-space path stays
straight; an off-path initial velocity necessarily bends it. Peak acceleration
from rest is about 62% below the old rest-to-rest quintic for the same move.
This is an acceleration comparison, not a torque-feasibility guarantee.

The post-hit padding tapers acceleration to zero over another `r*T`, then
coasts; it is masked out beyond each hand's strike and is not a rebound model.
Existing joint-box/slew clamps still apply where enabled and can alter the
analytic boundary conditions. Waist settling and idle-arm parking retain
their quintics. Apex teachers retain their apex paths; their standard init
and no-apex fallback use the new strike seed. Historical measurements below
describe the earlier quintic teacher; regenerate pools and retrain phase 1 to
learn the new timing. Existing checkpoints are not modified.

One scenario = a `StartSampler` start state (90 % drum-departure, both hands
placed on a drum) + a `goal_snapshot` taken at a random instant of a random RCC
clip against a **randomized** kit + the `warmstart_trajectory` target (per-arm
quintic to the goal-pose library, waist-aim yaw ramp). Scenarios are i.i.d.,
**not** rolled through `MPCGenerator`: in phase 1 the init is a noised quintic by
construction, so a simulated MPC buys nothing, costs 19 ms/tick, and currently
winds the waist up (see "Known rough edges"). Generation is a `fork` pool —
450 k requested scenarios in **65 s** on 64 workers, including the GPU auxiliary
label pass.

**Physically-impossible transitions are dropped**: any target whose commanded
per-joint peak `|q̇|` exceeds 1.5 × the rating goes in the bin. On the RCC
grooves that is a **30.7 % drop rate** (138 189 of 450 000 velocity rejects,
9 682 idle-both-hands); the surviving p50 demand is 0.95 × the rating and the
p99 is 1.4 ×. Without the filter the tail reaches 200 × — imitating those poisons BC.

The **input init** is the target plus OU (τ = 5 waypoints) + white noise, drawn
**fresh on the GPU every time a sample is served** (one `(H, H)` triangular
matmul, `ou_kernel`), with `sigma_corr` log-uniform in [0.02, 0.30] rad per
sample so the model sees warm starts from nearly-clean to badly wrong.

Auxiliary labels are computed **once** per pool entry, on the GPU, from the
target: spline `vel`/`acc` at each waypoint, `torque` = `max(|τ_{i-1}|, |τ_i|)`
through `cost.py`'s `knot_fd` + `BatchedRNEA`, and `fk_pos`/`fk_vel` for
tip/wrist/elbow of both arms. `tests/test_train_warmstart.py` asserts the torque
labels reduce **exactly** (0.0) to `FusedCost`'s `torque_max` diagnostic.

### Losses

Huber, not MSE (`beta = 0.1`), on the raw joint error — the library terminal
poses jump between waist-yaw grid entries, so a few samples per batch
legitimately need a ~1 rad correction and would dominate an MSE gradient. Each
auxiliary head is a Huber on the residual **normalized by a physical scale**
(velocity rating / 30 rad s⁻² / effort limit / 10 cm / 1 m s⁻¹), so one weight
(`aux = 0.1`) is meaningful across all five. Everything is masked by
`build_waypoint_masks` and reduced per *valid element*, so a 25-waypoint sample
weighs the same as a 100-waypoint one.

### Measured (RTX 5090, `cuda:1`, batch 1024, bf16, `torch.compile("default")`)

| | |
| --- | --- |
| data generation | 65 s (450 k requested → 302 129 accepted train, 28 860 val) |
| training | **20 000 steps in 1 231 s (20.5 min), 16.3 it/s** |
| bucket mix (train pool) | 25: 64 % · 50: 27 % · 100: 9 % |
| steps drawn per bucket | 12 724 / 5 495 / 1 781 |
| peak GPU memory | 14.9 GB (step) + ~7 GB (pool) |
| compiled shapes | 3 (one per bucket), **0 recompiles after warmup** |
| best val loss | 0.0062 at step 19 500 (no plateau; the 20 k cap ended it) |

Compiled vs eager at batch 512: 42–52 ms/step vs 88–95 ms/step. Batch 1024 is
nearly free on the two short buckets (launch-bound) and 2× on H = 100, so it
doubles the data seen for +13 % wall clock.

### Imitation and auxiliary accuracy (held-out clips **and** held-out starts)

Val is held out twice over: the four `assets/rcc/eval/` clips, and a disjoint
start-sampler seed stream.

| | noised-quintic init | zero-motion hold init |
| --- | --- | --- |
| masked RMSE, waist | **4.8 mrad** | 144 mrad |
| masked RMSE, left arm | **8.5 mrad** | 447 mrad |
| masked RMSE, right arm | **11.5 mrad** | 274 mrad |
| terminal-waypoint RMSE | **6.0 mrad** (max 227) | 537 mrad (max 2274) |
| spline vel | 0.20 rad/s (**7.5 %** rel) | 1.60 (59 %) |
| spline acc | 3.74 rad/s² (**4.7 %** rel) | 50.0 (62 %) |
| torque | 1.97 N·m (**10.5 %** rel) | 6.67 (36 %) |
| FK position | **4.4 mm** (1.6 % rel) | 77.5 mm (29 %) |
| FK velocity | 39 mm/s (**3.1 %** rel) | 612 mm/s (48 %) |

The model is a *refiner*, exactly as trained: it never saw a hold init, and from
one it recovers only the coarse shape.

The output is also **smooth** in distribution — over 20 val batches, model vs
target: jerk integral 1.05e9 vs 9.05e8 (+16 %), accel integral 1.08e5 vs 1.02e5
(+6 %), peak RNEA torque 145.6 vs 139.4 N·m, joint-box excess 1.8 mrad vs 0.

### Eval grid: model vs the quintic planner it imitates

`scripts/eval_phase1.py`, seed 0, one refinement iteration.

`standard` (162 cases):

| metric | quintic | model (quintic init) | model (hold init) |
| --- | --- | --- | --- |
| contact pos err p50 (mm) | 154.2 | 173.5 | 214.0 |
| impact speed err p50 (m/s) | 1.985 | 1.542 | 1.471 |
| max penetration p90 (mm) | 89.7 | 90.8 | 87.5 |
| torque excess p90 (N·m) | 0.0 | 25.2 | 27.0 |
| joint travel L1 p50 (rad) | 5.28 | 7.79 | 6.39 |
| jerk integral p50 | 9.7e4 | 7.7e7 | 8.0e7 |
| solve p50 (ms) | 2.3 | 15.6 | 13.3 |
| on the hit disk | 21/162 | 16/162 | 6/162 |
| no torque excess | 153/162 | 26/162 | 35/162 |
| inside the joint box | 162/162 | 131/162 | 162/162 |
| **accepted** | **0/162** | **0/162** | **0/162** |

`near_drum` (81 cases) tells the same story: 115.0 → 193.9 mm contact error,
2.2e5 → 5.6e7 jerk, 38/81 → 19/81 on the hit disk.

Two things are true at once and both matter:

1. **The teacher itself is not acceptable.** The quintic seed ends *at rest*
   (`sd1 = 0`), so its impact-speed error is ~2 m/s against a 1–3 m/s command,
   its impact direction cosine is negative, and it is obstacle-unaware
   (90 mm p90 penetration). It scores **0/243 accepted** on the same grid the
   trajopt SQP scores 126/243 on. Phase 1 cannot beat that ceiling — reaching it
   is the whole objective, and the model does reach it on its own distribution.
2. **The grid is off-distribution for phase 1.** `standard` starts are
   zero-velocity poses drawn from a scaled joint box, with a **single** active
   hand and no future-hit tokens; phase-1 training is 98 % dual-hand,
   drum-departure starts with 0.3–2 m/s tip velocity, and its targets carry
   ~1e9 of jerk. The model reproduces the *energy* of its training distribution
   and so emits a ~1e9-jerk trajectory even for a gentle grid case — a 700–800×
   jerk gap that torque excess then inherits (per-group medians on `near_drum`:
   waist 1.6e8, active arm 9.9e8, idle arm 1.1e9, versus 1.5e5 / 8.7e5 / 0 for
   the quintic).

Neither is a phase-1 failure under resolved decision 5 (trajectories reach their
terminals, every aux head is within 1.6–10.5 % on distribution), but (2) is a
concrete instruction for phase 2: **mix eval-grid-like contexts (rest starts,
single active hand, 0.45–0.8 s horizons) into the phase-2 data generator**, or
the acceptance grid will keep measuring distribution shift rather than
optimization quality.

## Phase 2+3 — direct cost descent (`phase2_data` · `train_cost`)

Section 7 of the plan, merged per resolved decision 6: one run, ramped weights,
no hard stage boundary, chained BPTT folded into the tail.

```
amortized_opt/
  phase2_data.py  Phase2ScenarioConfig / Phase2Pool -- the input distribution
                  (70 % plan-native, 30 % eval-like), the strike-geometry
                  columns the cost prices, and the drum centres it prices them
                  against
  train_cost.py   WeightSchedule / phase2_cost_config / phase2_al_config /
                  MPCRoller / ReplayPool / make_cost_step / make_tc_step /
                  make_bptt_step / bounce / train_phase2 / rolling_eval
scripts/
  train_phase2.py  one run
  pilot_phase2.py  the pilot race
  eval_phase2.py   the final grids + latency + closed-loop MPC
tests/test_phase2_prereq.py   fixes A and B
tests/test_train_cost.py      the training machinery
```

### Prerequisite fixes

**A — the MPC generator's waist wind-up.** The loop measured-yaw-rate →
`waist_reference_block(yaw_rate_start=…)` → FOH command → measured-yaw-rate is
positive feedback. Three guards, all in `warmstart_trajectory`:

* `yaw_state=(yaw, rate)` takes the boundary condition from the **planned**
  state; `MPCGenerator` carries `WarmStart.yaw_state_after(1)` forward;
* `waist_setpoint` freezes pitch/roll at a planned value — freezing them at the
  *measured* value is the same zero-stiffness loop and walked `waist_pitch`
  0.79 rad outside its ±0.52 rad box in 300 ticks;
* the yaw column is clipped into the joint box and slew-limited, and
  `slew_limit_eta` (opt-in) does the same for every joint. `MPCGenConfig` also
  slew-limits the *executed* command, whatever planner produced it.

Measured, 300 ticks on `gmd_train_000002_45s` with the quintic stand-in:

| | before | after |
| --- | --- | --- |
| ticks completed | 208 (diverged) | **300** |
| `waist_yaw` range (rad) | −2.82 … +2.90 (outside ±2.618) | **+0.05 … +0.80** |
| worst command rate / rating | 2.01× | **0.90×** |
| PD state finite | yes, but 1e11-scale | yes |

**B — kit randomization, end to end.** `FusedCost` now takes per-sample drum
head centres (`StrikeContext.drum_head_centers`), and the canonical way to build
them is `drum_heads_from_context(ctx.drum_pos)` — from the context *tokens*, so
the scene the cost prices and the scene the model was told about are the same
array by construction. `StartSampler.sample(..., kit=…)` screens against the
randomized kit too. `tests/test_phase2_prereq.py` asserts that passing the
nominal centres reproduces the baked-in scene exactly, that a 5 cm shift moves
both the collision and the terminal optimum, and that the tokens equal
`kit.positions_in("pelvis")` to 1e-12.

**C — eval-like contexts.** 30 % of scenarios are rest starts (box-sampled
waist + one arm, idle arm at URDF neutral, zero velocity), a single active hand,
0.44–0.80 s horizons and no future-hit tokens. Measured pool composition at
40 k requested: 30 571 accepted (22 % dropped by the impossible-transition
filter, 1.5 % both-hands-idle), 39 % eval-like, 55 % dual-arm of which 87 %
asymmetric.

### The objective, and why its weights are not the section-3 weights

`CostConfig`'s declared weights are the *validation* weights and are pinned by
`tests/test_cost_validation.py`. They are not a usable training objective,
because every term is a mean over its valid elements and the counts differ by
three orders of magnitude. Measured on the phase-1 checkpoint (bucket 50):

| group | value at declared weights | while the diagnostic says |
| --- | ---: | --- |
| `smooth` | 173 (1227 at bucket 25) | jerk integral 7.5e8 |
| `terminal` | 144 | 93 mm outside the legal region, 1.9 m/s speed error |
| `torque` | 0.71 | `torque_excess_max` **45 N·m** |
| `collision` | 0.046 | `min_sdf` **−55 mm** |

`phase2_cost_config()` undoes the averaging (collision ×1000, torque/velocity
×20, joint box ×5, smoothness ÷10, posture ×10) so the groups start within a
decade of each other; `phase2_al_config()` then moves multipliers by ones rather
than by thousands, capped at 100.

The full objective per step, on one bucket-homogeneous minibatch:

```
L = w_cost · mean FusedCost(q0, model(init, state, ctx), v0)          # no labels
  + w_aux  · Huber(aux heads, labels recomputed from the model's OWN traj)
  + w_tc   · masked disagreement between solve@H and solve@H−1 on 25 % of the batch
  + w_bptt · FusedCost of a SECOND solve, after a differentiable PD roll through
             the first hit and a bounce                    (ramped in at the tail)
```

* **No imitation term.** `init` is a warm start, never a target.
* **Auxiliary heads stay supervised throughout** (the plan's "the auxiliary terms
  ... should have continued throughout the entire thing"), but the labels now come
  from the model's own output through `tspline` / `ChainFK` / `knot_fd` + RNEA.
  They are computed under `no_grad`, so they train the heads and the shared trunk
  without letting the heads steer the trajectory.
* **Weight ramp** (`WeightSchedule`): collision, posture, terminal-*position* and
  smoothness are at full strength from step 0; the torque / velocity / joint-box
  hinges (from 0.05) and the strike-quality terms — impact speed and stick
  elevation (from 0.10) — ramp to 1.0 on a smoothstep. The augmented Lagrangian
  updates every 100 steps from the rolling mean of the *raw* violations.
* **Temporal consistency** (plan lines 102–107): solve at H, advance the noised PD
  one control tick with the plan's own first waypoint, solve at H−1 from where the
  servo got to, penalize disagreement on the overlap. Gradient flows through both
  solves and the plant. Exactly zero for a consistent pair (asserted).
* **Chained BPTT**: solve hit 1, roll the differentiable PD through the whole
  plan, gather the state at that hand's hit knot, bounce, solve hit 2 from there
  and price it. The bounce reflects the tip-*normal* velocity with restitution
  U(0.3, 0.6) and leaves the tangential component exactly untouched, then adds
  Gaussian tip-velocity / position / velocity noise — a documented placeholder for
  Isaac's contact solver.

### The data engine

* a **fresh pool** of i.i.d. scenarios, generated by 48 fork workers (40 k
  requested in 6.5 s), impossible-transition filter on;
* a **replay** of MPC contexts from `MPCRoller`, a batched on-GPU receding-horizon
  loop driven by the *current* weights: plan → command one admissible waypoint →
  noised PD tick → shift → re-plan. Slots that reach their hit respawn from the
  pool, carrying their post-bounce state with probability `chain_frac`, seeded by
  a quintic-eased ramp (`ramp_seed`) to the new scenario's terminal. Recency-biased
  sampling; the replay share ramps to 90 %.

Two deliberate differences from `mpc_gen`'s loop, both for batching: the mass
matrix is frozen at the tick's own pose rather than refreshed per knot inside a
rollout, and a finished slot respawns rather than re-goaling in place. Cross-hit
chaining is what the BPTT term trains explicitly, so it is not lost.

### `torch.compile` does not work in this environment

**The env's Triton and the bundled torch's inductor disagree.** `torch` comes
from Isaac's `pip_prebundle` (2.7.0+cu128); `triton` comes from the conda env
(3.3.0). Inductor emits kernels this Triton cannot parse:

```
triton.compiler.errors.CompilationError: at 34:31
AttributeError: type object 'constexpr' has no attribute 'is_block'
```

It fires on the `triton_poi_fused_stack_*` kernels — the ones inductor produces
for `ChainFK`'s Rodrigues stacking and `BatchedRNEA`'s per-link 3-element
spatial blocks — and then inductor **retries every autotune config**, so a
compile that should take ~2 minutes ran >13 minutes without producing a first
step. Measured on the cost alone at batch 512 × 50: 143 s to compile, and only
145 → 66.8 ms (2.2×) rather than the 15× the section-3 bench recorded.

Consequences, and they are the headline deviation of this section:

* the merged run is **eager** (`--no-compile`), at **2.5 it/s** at batch 256;
* resolved decision 6's 60–100 k steps is ~7 hours at that rate, so the run
  actually executed is far shorter — see the training-curve table;
* `Phase2Config.compile_model` / `compile_cost` / `compile_aux` exist so the
  moment the toolchain is fixed (pin `triton` to whatever the prebundled torch
  was built against, or use a torch from the same wheel as the triton) the same
  script gets the speedup with no code change;
* `exclude_rnea_from_compile()` is kept as the narrower workaround — it removes
  the RNEA frames from the graph — but it does not help, because FK trips the
  same codegen path.

### Unit tests

`tests/test_phase2_prereq.py` (7) and `tests/test_train_cost.py` (12). The whole
suite is **247 passed, 1 skipped** on `cuda:3` (6 m 35 s).

* a single Adam step on the fused cost lowers **the cost** on the same fixed
  batch (eager and, where the toolchain allows, compiled);
* the temporal-consistency term is exactly 0.0 for a perfectly consistent pair
  and > 0 for a one-waypoint disagreement;
* the chained-BPTT second-solve cost alone produces a finite, non-zero gradient
  in the *first* solve's decoder trunk — i.e. it back-propagates through the
  plant and the bounce;
* the bounce conserves tangential tip velocity exactly (fp64) and its
  joint-space realization tracks it to 2 mm/s;
* AL save → load → one more update is bit-identical, and the phase-2 checkpoint
  carries the multipliers;
* the MPC roller runs, respawns, and its harvested contexts collate into batches
  of the same shape as the fresh pool's;
* fix A's 300-tick regression and fix B's scene-consistency checks.

The compiled parametrization of the descent test **skips** rather than fails when
inductor cannot codegen (`"constexpr" in str(exc)`) -- that is a toolchain fault,
not a fault in the step, and the eager parametrization covers the behaviour. On
CPU, where inductor uses the C++ backend, the compiled path passes.

### The pilot race

Four configs, 1 200 steps each, batch 256, eager, two at a time on `cuda:3`,
sharing one cached scenario pool so the race is paired on data as well as on
seed. The ramp is compressed to `ramp_end = steps/2` so a short pilot still sees
the fully-weighted objective, and the chained stage is off (it only ramps in over
the final third of a real run).

`scripts/pilot_phase2.py --steps 1200 --devices cuda:3 cuda:3 --no-compile --batch 256`

| pilot | change | it/s | acc@1200 | best acc | hit | coll | tau | elbow | pos p50 (mm) | speed p50 | skipped |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | — | 1.90 | 0/81 | 0/81 | 0.83 | 0.67 | 0.00 | 0.98 | 35.9 | 0.097 | 17 |
| **`lr_hi`** | **lr 1e-4 → 3e-4** | 1.90 | **1/81** | **1/81** | **0.89** | 0.69 | **0.01** | 0.98 | 37.1 | **0.079** | 10 |
| `rho_hi` | AL rho ×4 | 1.87 | 0/81 | 1/81 @800 | 0.88 | **0.73** | 0.00 | 0.98 | **35.2** | 0.093 | 21 |
| `aux_tc_hi` | aux 5→20, tc 50→200 | 1.85 | 1/81 | 1/81 | 0.81 | 0.62 | 0.01 | **1.00** | 35.1 | 0.096 | 10 |

All four move the same way and fast: at step 400 the hit-disk rate is 0.15–0.64
and by 1 200 it is 0.81–0.89, contact error falls 154 mm (quintic) → ~36 mm and
impact-speed error 1.6 m/s (phase 1) → 0.08–0.10 m/s. **`lr_hi` wins** on
last-eval acceptance and impact speed; `rho_hi` is the best on collision (0.73)
and produced the only short-horizon acceptance, at step 800. `aux_tc_hi` shows
the cost descent is slowed by a heavy auxiliary/consistency weight without
buying a gate. The two winners were **not** combined — the committed run is
`lr_hi`'s measured configuration.

The pilots also identify the binding constraint unambiguously: **`ok_torque` is
~0.00–0.01 in every config**, and every other gate is at or near its target.
The harness's torque check is zero-tolerance against the *hard* effort limits,
and the models are asking for 60–70 N·m over them at p90 — worse than phase 1's
25 N·m, because a model that now actually reaches the drum on time has to
accelerate to do it.

### The training run

`scripts/train_phase2.py` writes `metrics.jsonl` (one record per log / eval /
skip / event), `best.pt` (highest rolling grid acceptance), `last.pt` and
`summary.json` into `runs/<name>/`. Checkpoints carry the augmented-Lagrangian
state, so a `--resume` continues the multipliers rather than restarting them.

Rolling eval every `--eval-every` steps on a stride-subsampled acceptance grid,
solved in one batched pass and scored by the section-2 harness, reporting the
plan's stage-A gates (hit-disk ≥ 85 %, striking-arm collision-free ≥ 66 %,
elbow-down ≥ 95 %) plus the short-horizon cell.

Fresh-init composition (coordinator's instruction, after the apex A/B): 50 % of
scenarios request an **apex three-point spline** as their initialization and
50 % the straight-line quintic. An apex construction that has no room for an
interior waypoint, or that fails the same impossible-transition filter, falls
back to the straight line **for that scenario** rather than being discarded, so
the apex path can never reshape the scenario distribution — only the
initialization. Measured on 300 requested scenarios: 156 apex requested, 107
used, 49 fell back, giving **47.6 % apex / 52.4 % straight-line** among the
accepted. Apex inits are inits, not references: ~43 % of raw apex poses are
self-collision-tainted (stick-vs-torso) and the apex target's torque excess is
2.7× the straight line's — the cost prices both away, which is the point of
initializing from it rather than imitating it. Generation cost rises from
6.5 s to ~6 min for 40 k scenarios (48 workers) because the apex IK is solved
per scenario rather than per chunk.

### Where the wall clock goes

Measured at batch 256 x H = 50, eager, bf16, `cuda:1`, 12 reps after warmup:

| component | ms | share |
| --- | ---: | ---: |
| core step: model + `FusedCost` + aux labels, fwd+bwd | 271.2 | 47 % |
| temporal-consistency term (64 samples: 2 solves + PD + `inv(M)`) | 175.5 | 31 % |
| MPC roller tick (256 slots, no-grad) | 126.0 | 22 % |
| **total** | **572.7** | 1.75 it/s |

Which settles the obvious question: the MPC data generator is **22 %** of the
step, so moving it fully async would buy at most 1.28x. The 10x gap to resolved
decision 6's throughput is inductor, not the data engine. The two contained
levers, if a run has to be fast without fixing the toolchain, are `tc_frac`
0.25 -> 0.10 (-20 %) and running the roller every second step (-11 %).

### Results

`runs/phase2_main`, `lr_hi`'s configuration, 7 500 steps in **4 190 s (1.79 it/s)**,
24 non-finite steps skipped (0.3 %), single GPU, eager. Starting point:
`runs/phase1_main/best.pt`.

| step | accepted | hit | speed | coll | torque | box | elbow | pos p50 (mm) | speed err p50 | τ exc p90 | jerk p50 | short | stage A |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1500 | 0/81 | 0.73 | 0.96 | 0.67 | 0.00 | 0.75 | 0.99 | 62.3 | 0.090 | 92.4 | 7.4e7 | 0/27 | no |
| 3000 | 0/81 | 1.00 | 0.90 | 0.58 | 0.00 | 0.75 | 0.99 | 51.7 | 0.185 | 57.0 | 9.4e7 | 0/27 | no |
| 4500 | 10/81 | 0.99 | 0.99 | 0.62 | 0.17 | 1.00 | 0.99 | 53.2 | 0.084 | 11.7 | 4.3e7 | 2/27 | no |
| **6000** | **33/81** | 1.00 | 1.00 | 0.72 | 0.52 | 1.00 | 0.99 | 35.9 | 0.051 | 3.5 | 4.2e7 | 11/27 | **yes** |
| 7500 | 31/81 | 1.00 | 1.00 | 0.72 | 0.53 | 1.00 | 0.99 | 35.1 | 0.069 | **1.8** | 3.6e7 | 11/27 | yes |

AL trajectory over the run: `terminal` 1 → 26.5, `joint_box` 1 → 24.6,
`torque` 1 → 7.2, `collision` 1 → 4.8, `velocity` 1 → 3.6. The multipliers are
what moved `ok_torque` off zero — it sat at 0.00 through step 3 000 while λ_torque
was still ~2, then went 0.17 → 0.52 as λ passed 4. `best.pt` is step 6 000.

#### The grids (seed 0, one refinement iteration unless noted)

`standard`, 162 cases:

| planner | accepted | hit | speed | coll | torque | box | pos p50 (mm) | speed err p50 | pen p90 (mm) | τ exc p90 | jerk p50 | short (T = 0.45 s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quintic | 0/162 | 21 | 0 | 14 | 153 | 162 | 154.2 | 1.985 | 89.7 | 0.0 | 9.7e4 | 0/54 |
| phase 1 | 0/162 | 15 | 16 | 9 | 29 | 127 | 173.5 | 1.581 | 91.1 | 24.8 | 7.0e7 | 0/54 |
| **phase 2, quintic init ×1** | **73/162** | **162** | **162** | 125 | 90 | 162 | **35.7** | 0.065 | 49.6 | 2.75 | 4.1e7 | **21/54** |
| phase 2, quintic init ×3 | 67/162 | 162 | 162 | 124 | 84 | 162 | 32.3 | 0.076 | 51.1 | 3.80 | 4.1e7 | 21/54 |
| phase 2, hold init ×1 | 69/162 | 162 | 162 | 125 | 84 | 162 | 29.5 | 0.061 | 50.2 | 2.79 | 4.2e7 | 20/54 |
| phase 2, hold init ×3 | 65/162 | 162 | 162 | 123 | 81 | 162 | 31.8 | 0.066 | 51.4 | 3.64 | 4.1e7 | 20/54 |

`near_drum`, 81 cases:

| planner | accepted | hit | speed | coll | torque | pos p50 (mm) | speed err p50 | τ exc p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quintic | 0/81 | 38 | 0 | 7 | 59 | 115.0 | 1.992 | 4.41 |
| phase 1 | 0/81 | 19 | 3 | 6 | 29 | 194.2 | 1.704 | 15.4 |
| **phase 2, quintic init ×1** | **51/81** | **81** | **81** | **77** | 53 | 37.3 | 0.063 | 7.29 |
| phase 2, hold init ×3 | 44/81 | 81 | 81 | 77 | 45 | 35.4 | **0.043** | 4.02 |

#### Versus the expert, on the same 243 cases

| | phase 2 (×1, quintic init) | trajopt v2b | trajopt v2 |
| --- | ---: | ---: | ---: |
| **accepted** | **124/243 (51.0 %)** | 149/243 (61.3 %) | 126/243 (51.9 %) |
| contact inside the hit disk | **243/243** | 241/243 | 238/243 |
| impact speed in band | **243/243** | 241/243 | 241/243 |
| collision-free (striking arm) | **202/243 (83 %)** | 162/243 (67 %) | 148/243 (61 %) |
| **no torque excess** | 143/243 (59 %) | **209/243 (86 %)** | 187/243 (77 %) |
| inside the joint box | 243/243 | 243/243 | 243/243 |
| contact pos err p50 | **35.7 mm** | 66.9 mm | 70.2 mm |
| impact speed err p50 | 0.065 m/s | **0.044 m/s** | 0.046 m/s |
| jerk integral p50 | 4.1e7 | **1.35e7** | 1.63e7 |
| short-horizon cell (T = 0.45 s) | 21/54 | **26/54** | 27/54 |
| solve p50 | **17.9 ms** (2.7 ms batched) | 6 514 ms | 2 231 ms |

Read plainly: after 7 500 eager steps the amortized optimizer **matches v2's
overall acceptance and beats both experts on four of the six gates** — it lands
every strike inside the legal hit disk, hits every impact-speed band, is 16-22
points better on striking-arm collision-freedom, and halves the contact-position
error — at **360× v2b's solve time**. The entire remaining gap is **torque**:
59 % vs 86 %. Nothing else is close to binding.

Two secondary findings from the same table:

* **One refinement iteration beats three** (73 vs 67 accepted). The operator is
  not contractive toward a better solution on this grid; iterating drifts the
  terminal and re-inflates the torque tail. The plan's "very good results by
  iteration 2-3" does not hold yet.
* **The model is now nearly initialization-insensitive** (73 quintic vs 69 hold),
  where phase 1 collapsed from a hold init (16 → 6 on the hit disk). Direct cost
  descent removed the refiner's dependence on a good warm start.

#### Latency

| | eager, bf16, `cuda:3` |
| --- | ---: |
| one MPC tick, batch 1 (encode + 1 refinement) | **17.9 ms** |
| the 162-case grid in one batched pass | 431.9 ms → **2.67 ms/case** |

17.9 ms fits inside the 20 ms control period, but only just, and it is eager;
section 4 measured the same decoder at 0.53 ms compiled.

#### Closed-loop MPC — this does not work yet

Three eval clips through the **fixed** `MPCGenerator` loop (the trained model as
the planner, noised PD, per-hand post-hit re-goal, 250-tick cap):

| clip | ticks | hits | hit in disk | speed in band | PD track p50 / p90 (mrad) | TC drift p50 / p90 (mrad) | ms/tick | ended |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gmd_eval_000001_30s` | 109 | 11 | 2 | 7 | 8.6 / 28.5 | 26.0 / 64.2 | 33.5 | diverged |
| `gmd_eval_000003_45s` | 25 | 1 | 0 | 1 | 15.8 / 60.6 | 31.6 / 39.4 | 33.7 | diverged |
| `gmd_eval_000006_30s` | 59 | 3 | 0 | 0 | 8.4 / 25.4 | 25.6 / 34.2 | 33.1 | diverged |

All three episodes hit the divergence guard (the measured pose left the joint box
by more than 0.5 rad) after 25-109 ticks, and only 2 of 15 executed strikes
landed in the disk. PD tracking itself is fine (8-16 mrad at the median) and the
temporal-consistency drift is 26-32 mrad — the successive plans agree about as
well as the TC loss was trained to enforce. What fails is the **goal switch**:
the roller the model trained against *respawns* a finished slot from the pool,
so the model never saw `MPCGenerator`'s in-place per-hand re-goal, where one arm
gets a new drum mid-flight while the other keeps counting down on the shared
waist. Single-shot grid performance is excellent and closed-loop performance is
not; that is the honest state of it, and closing it means training the roller's
respawn as a re-goal (see decision 6).

### Phase-2 decisions needing review

1. **`torch.compile` is off.** Not a design choice — the toolchain is broken
   (above). Every speed number in this chapter is eager and the run is ~10×
   shorter than resolved decision 6 asks for. Fix the Triton/torch pairing and
   re-run before drawing conclusions about the *ceiling* of this method.
2. **No DDP.** GPUs 0, 1 and 2 were held by concurrent sessions for the whole
   window (cuda:1 acquired a 27 GB job mid-session), so the run is single-GPU on
   `cuda:3`. `train_phase2` takes `rank` / `world_size` but no DDP wrapper is
   installed; with horizon bucketing both ranks must also choose the *same*
   bucket each step or the all-reduce deadlocks.
3. **The closed-loop MPC diverges and the single-shot grid does not.** Three of
   three eval clips hit the divergence guard within 25-109 ticks with only 2 of
   15 executed strikes on the disk, while the same checkpoint is 124/243 on the
   grid. PD tracking (8-16 mrad p50) and temporal-consistency drift (26-32 mrad
   p50) are both healthy, so the failure is the **goal switch**: the training
   roller respawns a finished slot from the pool, so the model has never seen
   `MPCGenerator`'s in-place per-hand re-goal, where one arm takes a new drum
   mid-flight while the other keeps counting down on the shared waist. Making
   the roller re-goal in place (promoting the next `future_*` token, which the
   context already carries) is the change; the tensorized strike table needed
   for it is already in `phase2_data._strike_row`'s inputs.
4. **One refinement iteration beats three** (73 vs 67 accepted). Feeding the
   output back in drifts the terminal and re-inflates the torque tail, so the
   plan's "very good results by iteration 2-3" is not yet true. Nothing in the
   objective currently *asks* iteration 2 to improve on iteration 1 -- training
   only ever back-propagates a single pass.
5. **The stage-A gate does not gate anything.** The plan wants the stage-A
   metrics to unlock the full ramp; the ramp here is on step count and the gate
   is only *reported*. A gate that never opens would freeze the curriculum
   forever, and on a run this short that risk outweighs the benefit. The eval
   record carries `stage_a` per checkpoint so the coupling can be switched on.
6. **The collision multiplier does not come back down.** `lam_collision` rises
   while the mean raw violation is positive and decays at 0.999 per update, so
   once clearance is positive the multiplier keeps a large price on it: measured,
   `min_sdf` keeps growing (−40 mm → +27 mm over 1 200 steps) and the contact
   error stops improving. A tolerated violation (`ALConfig.target > 0`) or a
   faster decay is the obvious fix and is one field.
7. **`ok_torque` is the binding constraint, and the cost cannot fix it as
   written.** The torque term is a **mean** over ``H × 17`` hinge elements; the
   harness's gate is a **max** with zero tolerance against the *hard* limit. One
   knot of one joint over the limit anywhere in the trajectory fails the case,
   and a single bad element moves a mean of 850 by ~0.1 %. That is why every
   pilot sits at `ok_torque` 0.00-0.01 while `torque_excess_max` p90 is 60-70 N·m
   and every *other* gate is at target. The fix is structural, not a weight: add
   a soft-max (or an explicit per-sample max) component to the torque term
   alongside the mean — measured on the recorded eval CSV, only **0.5 % of
   (knot, joint) elements** exceed the limit at the median case (p90 1.1 %,
   worst 2.3 %), so the offending elements carry 0.5 % of a mean over 850 — and/or restore `w_smooth_jerk` — the second difference of
   a one-knot zig-zag is what produces these spikes, and this chapter divided
   the section-3 smoothness weight by 10. **This is the single highest-value
   next change** and it was identified too late in the window to re-run.
8. **The roller respawns instead of re-goaling.** A slot that reaches its hit
   takes a *new* scenario (carrying its post-bounce state with probability 0.35)
   rather than promoting the next future-hit token in place. The chained-BPTT
   term is what trains cross-hit continuity; the roller only supplies
   receding-horizon *inits*.
9. **The mass matrix is frozen at the tick's pose** inside the roller and the
   BPTT roll, not refreshed per knot as `closed_loop` does. Batching decision.
10. **`best.pt` is selected on grid acceptance, which is degenerate at 0.**
   Every rolling eval scored 0 accepted, so the first eval wins the tie and
   `best.pt` is an early checkpoint even though the later ones are much better
   on every component gate (hit-disk 0.73 -> 1.00 between steps 1500 and 3000).
   The final evaluation therefore reports **both** `best.pt` and `last.pt`. A
   selection metric that degrades gracefully -- e.g. the mean of the six gate
   rates -- would fix this and is a two-line change.
11. **Bucket 100 was dropped** for the committed run (`--buckets 25 50
   --horizon 50`): the eval grid's horizons are 22–40 waypoints and bucket 100
   was 5 % of the pool, and each extra bucket is another set of compiled shapes
   and another 4× activation footprint.

---

## Section 7b — targeted fixes and the continuation run

Four fixes, in descending order of measured impact, then a continuation of
`runs/phase2_main` under the new objective.  Everything in this chapter
supersedes the "Phase-2 decisions needing review" items it names.

### 1. `torch.compile` DOES work here — the chapter above was wrong

Decision 1 recorded the toolchain as broken.  Re-measured: inductor compiles
both the decoder **and** `FusedCost` on this exact env, and the failure the
chapter saw was the **autotune retry storm**, not the codegen.

`scripts/train_phase2.configure_inductor()` is the whole fix — turn autotuning
and inductor's own cudagraphs off explicitly, and raise the dynamo cache limit
so ten compiled shapes (two buckets × cost / tc / bptt) do not silently spill
back to eager:

```python
torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_pointwise = False
torch._inductor.config.max_autotune_gemm = False
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.triton.cudagraphs = False
torch._dynamo.config.cache_size_limit = 64
```

Measured on `cuda:1`, batch 256 × H = 50, bf16 autocast, fwd+bwd, with
`suppress_errors` **off**:

| unit | eager | compiled | compile time | speedup |
| --- | ---: | ---: | ---: | ---: |
| decoder (`model(init, state, enc, masks)`) | 42.9 ms | **12.5 ms** | 19 s | 3.43× |
| `cost_tuple` (`FusedCost`, the FK/RNEA stack the chapter blamed) | 212.5 ms | **64.3 ms** | 157 s | 3.30× |
| `FusedCost` with the new peak term | 136.4 ms | **45.7 ms** | 135 s | 2.99× |
| **whole training step** | 1.79 it/s | **4.3 it/s** (3.1 with the roller) | ~700 s cold, cached after | **2.4×** |

`AttributeError: type object 'constexpr' has no attribute 'is_block'` still
appears in the log — hundreds of times.  It is emitted from
`triton_heuristics.py` while the autotuner compiles *candidate* configs and
discards the ones that fail; at least one config always survives, so the kernel
compiles.  Reading those lines as a compile failure is what cost the previous
run 2.4× throughput.

`backend="cudagraphs"` was also tried and does **not** work: its outputs are
overwritten by the next invocation and the eager backward then raises
`accessing tensor output of CUDAGraphs that has been overwritten`, even with
`torch.compiler.cudagraph_mark_step_begin()` before every call.

### 2. The closed-loop divergence was a missing command clamp, not the goal switch

Decision 3 attributed all three diverged eval clips to the training roller's
respawn.  Instrumenting the divergence says otherwise.  In every clip the
**command** left the joint box a tick *before* the measured pose did, at 2–3 %
of the joint's speed rating — so neither the slew limiter nor the PD tracking
was involved:

| clip | worst joint | q excess | cmd excess | box | first excess |
| --- | --- | ---: | ---: | --- | ---: |
| `gmd_eval_000001_30s` | `waist_roll` | 0.504 rad | 0.500 rad | ±0.52 | cmd @ 10, q @ 11 |
| `gmd_eval_000003_45s` | `waist_pitch` | 0.525 rad | 0.483 rad | ±0.52 | cmd @ 15, q @ 16 |
| `gmd_eval_000006_30s` | `waist_roll` | 0.507 rad | 0.520 rad | ±0.52 | cmd @ 36, q @ 37 |

`MPCGenerator.run_episode` slew-limited the executed command but never clamped
it into the joint box.  `MPCRoller` — the loop the model is *trained* against —
has always clamped, and so does the plant: Isaac's implicit actuators saturate
a position target at the joint limit, as does hardware.  So this loop was the
only place in the stack where a planner could integrate its own command out of
the box, and a slow unpriced waist drift did exactly that.

`MPCGenConfig.clamp_cmd_to_box` (default **True**) is the fix — one `np.clip`.
On the unchanged `runs/phase2_main/last.pt` checkpoint:

| clip | ticks before | ticks after | hits before | hits after |
| --- | ---: | ---: | ---: | ---: |
| `gmd_eval_000001_30s` | 89–109 (diverged) | **250** | 11 | **23** |
| `gmd_eval_000003_45s` | 25–30 (diverged) | **250** | 1 | **19** |
| `gmd_eval_000006_30s` | 59–72 (diverged) | **250** | 3 | **13** |

`tests/test_mpc_cmd_clamp.py` pins it: a deliberately runaway planner (0.03 rad
per 20 ms on one joint, a few percent of its rating) cannot leave the box with
the clamp on, *does* leave it and trip the divergence guard with the clamp off,
and an already-admissible planner gets a bit-identical episode either way.

### 3. The torque term now sees the max, not just the mean

Decision 7's diagnosis is confirmed on the recorded eval CSVs.  Of the 100
`standard`-grid cases that fail `ok_torque`, the median has **1 knot of 30**
over the limit — 0.27 % of the (knot, joint) elements (p90 0.51 %, worst
1.5 %) — by a median 0.87 N·m.  `FusedCost._torque` reduced the softplus hinge
with a **mean** over ~850 elements, so the element that decides acceptance
moved the loss by ~0.3 %.

`smooth_max(h, mask, mode, p, temp)` (exported from `cost.py`) adds the missing
half.  `"pnorm"` (default, `p = 12`) computes `(Σ hᵢᵖ)^(1/p)` in the
detached-max normalized form so it cannot overflow and its gradient stays
exact; `"lse"` is `T·logsumexp(h/T)` and is a **diagnostic only** — its
`T·log n` bias is additive, so a feasible row with a 0.039 N·m true max reads
0.305.  `CostConfig` gains `w_torque_peak` (0.0 by default, so the section-3
validation weights are untouched), `torque_peak_mode`, `torque_peak_p`,
`torque_peak_temp`; `FusedCost.forward` gains a `"torque_peak"` sub-multiplier
so a curriculum can ramp it, and the AL's group multiplier scales both halves.

Measured, single violator among 850 elements:

| | value | ratio to the true max |
| --- | ---: | ---: |
| true max | 3.000000 | 1.0 |
| `pnorm`, p = 12 | 3.000000 | **1.0000** |
| `lse`, T = 0.05 | 3.000000 | 1.0000 |
| the existing **mean** | 3.53e-03 | **0.001176 = 1/850** |

Gradient concentration on that one element: **100.0 %** for both modes (pnorm's
next-largest weight is `(0.02/3)^11 ≈ 6e-25`) against 0.118 % for the mean.
On the 243 recorded expert cases the new statistic separates the gate **26×**
(`torque_hinge_max` 0.298 on `ok_torque` PASS vs 7.78 on FAIL) where the mean
separates it 3.4×.  Cost: **+0.26 %** of a fwd+bwd step (0.61 ms of ~130 ms).
29 tests in `tests/test_torque_peak.py`.

`w_smooth_jerk` is also restored, 1e-5 → **1e-4** — the section-3 value the
phase-2 chapter divided by 10.  A one-knot zig-zag is a second difference, and
it is exactly what produces these single-element spikes.  `w_smooth_acc` stays
at 1e-4, so trajopt's 10:1 acc:jerk ratio is deliberately broken; see DECISIONS.

### 4. The roller re-goals per hand, in place

Decisions 3 and 8.  `MPCRoller.tick()` now mirrors `MPCGenerator.run_episode`'s
reset logic instead of respawning a finished slot:

1. `fired = active & (hit <= 0)`; `stale` = an idle hand whose next hit has
   ticked into the horizon;
2. fired hands set `start_drum`/`start_shaft` from the drum just struck (the
   departure exemption, = `MPCGenerator`'s `recent_drum`);
3. `future[:, :, 0]` is promoted into the goal tokens and the future list
   shifts left; a promotion that is out of horizon range, invalid, or
   library-impossible for that hand idles the hand instead;
4. the promoted hand's strike-cost row is re-priced from `strike_geometry_table`
   against **that slot's own randomized kit** — never another scenario's;
5. `hit = round(tth/dt).clamp(2, H)`, and only that hand's arm columns of the
   plan are re-seeded (`ramp_seed` to a `(2, 8, 129, 7)` GPU table of
   `GoalPoseLibrary.select(hand, drum, yaw).arm_q`).  The other hand's plan,
   horizon and strike row are bit-identical to the shifted plan;
6. the waist is re-seeded only when **both** hands re-goal on the same tick;
7. the PD state (`q0`, `qd0`, `prev_cmd`, `kp`, `kd`) is untouched.  A slot
   respawns only when no hand is active, `age > 400`, or it diverges.

`phase2_data.strike_geometry_table` is the batched port of
`rhythm.strike_region` + the `strike.py` name predicates (constant per drum
index, since randomization only moves positions).  Validated against the numpy
path over 32 randomized kits × every drum: **max error 2.2e-16** on every field
(`aim`, `disk_center`, `hit_radius`, `hp_dir`, `has_hp`, `disk_off`; exactly 0
on `sigma`/`shaft`/`elev`).

**A second, independent bug fell out of this**: the roller never aged
`goal_tth` / `future_tth`, so every harvested replay context told the model a
stale time-to-hit while the mask's Fourier τ encodings — built from `hit` — told
it the truth.  `MPCGenerator._context_row` rebuilds them from a fresh snapshot
every tick, so the replay distribution simply disagreed with deployment.
`_tick_tth()` fixes it.  (`future_tth` has since become `future_gap`, measured
from the current goal's strike rather than from now — see the tokenizer's field
doc.  It is segment-constant, so `_tick_tth()` now ages it only on the hands
whose gaps fall back to a from-now anchor, i.e. those with no active goal, and
`_regoal` re-anchors what survives a promotion.)

Measured at 256 slots on `cuda:1`, driven by the trained checkpoint, 50 timed
ticks:

| | ms/tick | resets/tick | re-goals/tick | idle/tick | mean slot age | diverged/tick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| respawn (old) | 153.9 | 11.35 | 0 | 0 | 13.2 | 0.52 |
| **re-goal (new)** | **108.4** | 3.65 | **12.35** | 6.31 | **44.9** | **0.00** |

The re-goal path is 32 % *faster* end to end, because it respawns 3.1× less
often and `_spawn` costs a host-side `multinomial` plus a device sync.  The
machinery's own cost, isolated, is +1.5 %.  11 tests in
`tests/test_roller_regoal.py`.

### The continuation run, and the learning rate that killed the first attempt

`runs/phase2b_lr3e4_aborted` is kept deliberately.  It resumed
`runs/phase2_main/last.pt` (model + optimizer moments + AL multipliers) under
the new objective at **lr 3e-4** — "the same winning config" the pilot race
chose — and destroyed the model in 1 300 steps:

| step | lr | cost | contact miss (mm) | `torque_hinge_max` |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 5.0e-5 | 238 | 12.5 | 13.9 |
| 250 | 2.5e-4 | 365 | 15.9 | 11.8 |
| 350 | 3.0e-4 | 1 109 | 22.8 | 16.8 |
| 850 | 3.0e-4 | 2 810 | 34.3 | 39.1 |
| 1 350 | 3.0e-4 | 2 260 | 37.5 | 46.2 |

`eval@1300: acc 0/81, hit 0.83, tau 0.00` — from 33/81 at the resume point.
The cost rises monotonically from the first LR-ramp step and the torque peak it
is being punished for gets *worse*, which is an unstable optimizer, not an
over-weighted term (zero non-finite steps were skipped).

The reading, in hindsight obvious from the phase-2 curve itself: **`lr_hi` was
the right LR for a model that could not yet reach a drum, not for a continuation
of one that can.** `runs/phase2_main` scored 0/81 at every eval while its cosine
was above ~1.2e-4 and only produced acceptance once it fell below it
(10/81 at lr 1.18e-4, 33/81 at lr 4.3e-5).  Gradient norms run 8e3–4e5 against
`grad_clip = 1.0`, so every step is a pure direction at exactly `lr` in
parameter space — the LR *is* the step size, and a 20× warm restart of it is a
20× step.

The committed run, `runs/phase2b_main`, is the same continuation at
**lr 5e-5** (3.3× the LR the previous run finished at, not 20×) with the torque
peak weight **halved to 1.0** and ramped 0 → 1 over 2 000 steps, so the
augmented Lagrangian — resumed at `lam_torque = 7.23` and multiplying both
halves — is not shocked.  For scale, at step 650 the two runs sit at:

| | lr 3e-4 (aborted) | lr 5e-5 (committed) |
| --- | ---: | ---: |
| cost | 1 635 | **250** |
| terminal term | 387 | **115** |
| torque term | 538 | **46** |
| contact miss | 33.2 mm | **12.4 mm** |
| min SDF | −9 mm | **+11 mm** |

Other run settings: `--ramp-start -1 --ramp-end 0` so the *limit* ramp is
already complete at step 0 (a resume must not drop the limit hinges back to
`lo_limits = 0.05`), `--stage-a-gate --stage-a-open`, `--compile-cost`, the
same `--buckets 25 50 --horizon 50` and the cached apex pool.

### The stage-A gate

Decision 5 said the gate is only reported.  `WeightSchedule.gate_stage_a` now
holds `alpha` at 0 until `stage_a_pass` has fired once, and `train_phase2`
takes `stage_a_init` so a resume seeds it from a checkpoint that already
passed.  It is enabled for this run and it is **honestly vacuous**: the
resumed checkpoint reached stage A at step 6 000 of `phase2_main`, the gate is
seeded open, and the limit ramp is complete at step 0 anyway.  Left unseeded it
would be actively harmful — the limit hinges would sit at 5 % for the first
`eval_every` steps of a continuation whose binding constraint is torque.

### Resume now restores the optimizer

`train_phase2(opt_state=…)` and `scripts/train_phase2.py --resume` load
`ck["optimizer"]`, which the previous chapter's resume path silently dropped.
Adam's second moment sets the effective per-parameter step size; restarting it
makes the first few hundred steps of a continuation behave like a fresh run at
a much larger LR.  `--max-hours` stops the loop cleanly on wall clock, which is
the real budget on this box.

### Results

The continuation reached the plan's acceptance target **in 1 000 steps** and
`train_phase2` stopped on it (`target_acceptance = 0.60`, now overridable with
`--target-acceptance`).  Rolling grid (81 stride-subsampled cases):

| | resume point (phase 2, step 7 500) | phase 2b, step 1 000 |
| --- | ---: | ---: |
| accepted | 31/81 | **54/81** |
| hit disk | 1.00 | 1.00 |
| collision-free | 0.72 | 0.72 |
| **no torque excess** | **0.53** | **0.93** |
| elbow down | 0.99 | 0.99 |
| short-horizon cell | 11/27 | **16/27** |

#### The full 243-case grid, versus phase 2 and the expert

`standard` (162) + `near_drum` (81), seed 0, one refinement iteration, quintic
init:

| | phase **2b** | phase 2 | trajopt **v2b** | trajopt v2 |
| --- | ---: | ---: | ---: | ---: |
| **accepted** | **175/243 (72.0 %)** | 124/243 (51.0 %) | 149/243 (61.3 %) | 126/243 (51.9 %) |
| contact inside the hit disk | 242/243 | **243/243** | 241/243 | 238/243 |
| impact speed in band | **243/243** | **243/243** | 241/243 | 241/243 |
| collision-free (striking arm) | 193/243 (79 %) | **202/243 (83 %)** | 162/243 (67 %) | 148/243 (61 %) |
| **no torque excess** | **220/243 (90.5 %)** | 143/243 (59 %) | 209/243 (86 %) | 187/243 (77 %) |
| inside the joint box | 243/243 | 243/243 | 243/243 | 243/243 |
| contact pos err p50 | 39.2 mm | **35.7 mm** | 66.9 mm | 70.2 mm |
| impact speed err p50 | 0.054 m/s | 0.065 m/s | **0.044 m/s** | 0.046 m/s |
| τ excess p90 (`standard`) | **0.00 N·m** | 2.75 | 1.18 | 3.53 |
| jerk integral p50 | 1.83e7 | 4.13e7 | **1.35e7** | 1.63e7 |
| short-horizon cell (T = 0.45 s) | **32/54** | 21/54 | 26/54 | 27/54 |
| solve p50 | **17.3 ms** (2.65 ms batched) | 17.9 ms | 6 514 ms | 2 231 ms |

**The amortized optimizer now beats the trajopt expert outright**: 175/243
against v2b's 149/243, and it does it while also beating v2b on five of the six
component gates and on the short-horizon cell that the MPC actually lives in,
at 380× the speed.  The torque gap that was the entire story of the previous
chapter is closed and then some — 90.5 % vs 86 % — and `tau_excess_max` p90 on
`standard` is now exactly **0.00 N·m**.  The two things given up are 3.5 mm of
contact-position error and 9 cases of striking-arm collision-freedom (still 12
points ahead of v2b).

Initialization insensitivity is now complete: quintic init 112/162, hold init
112/162 on `standard` — identical.  One refinement iteration still ties or beats
three (112 vs 111 quintic, 112 vs 113 hold), so decision 4 stands.

#### Closed loop — the divergence is fixed, the strike accuracy is not

Three eval clips through `MPCGenerator` with `clamp_cmd_to_box=True`:

| | phase 2 (before the clamp) | phase 2 (clamp only) | phase 2b |
| --- | ---: | ---: | ---: |
| clips completing 250 ticks | **0/3** | **3/3** | **3/3** |
| ticks | 109 / 25 / 59 | 250 / 250 / 250 | 250 / 250 / 250 |
| strikes executed | 15 | 55 | 55 |
| strikes in the hit disk | 2/15 | 4/55 | 2/55 |
| impact speed in band | 8/15 | 38/55 | 24/55 |
| PD tracking p50 | 8–16 mrad | 6–7 mrad | 5–9 mrad |
| TC drift p50 | 26–32 mrad | 16–20 mrad | 21–26 mrad |
| ms/tick | 33 | 34 | 33 |

**The residual failure is axial, and it is not the goal switch.**  Decomposing
the 55 executed strikes into the two halves of the hit test:

| | phase 2 | phase 2b |
| --- | ---: | ---: |
| in-plane distance beyond the disk edge, p50 | **−2.2 mm** (inside) | +52.7 mm |
| in-plane inside the disk | 30/55 | 10/55 |
| **axial distance from the head plane, p50** | **39.3 mm** | **56.2 mm** |
| **axial within the ±20 mm band** | **7/55** | **9/55** |
| both | 4/55 | 2/55 |

The aim is roughly right and the stick is **40–56 mm short of the head** at the
tick the schedule calls the hit.  That is receding-horizon procrastination: each
re-plan is free to leave the final approach in waypoints it will never execute,
because only the first waypoint is ever committed and the next tick re-plans
from scratch.  The temporal-consistency term penalizes plan-to-plan
*disagreement*, which two consistently-lazy plans satisfy perfectly; nothing in
the objective requires the executed **prefix** to make terminal progress.  It is
invisible on the single-shot grid, where the whole plan is executed.

### Section 7b decisions needing review

1. **The learning rate, not the objective, is what a continuation must be tuned
   on.** `lr_hi` (3e-4) was the pilot race's winner for a model that could not
   reach a drum, and it destroyed a model that could, in 1 300 steps
   (`runs/phase2b_lr3e4_aborted`, kept).  Gradient norms are 8e3–4e5 against
   `grad_clip = 1.0`, so every step is a unit direction scaled by exactly `lr`.
   Any future continuation should start at or below the LR the previous run
   *finished* at, times a small factor.
2. **The torque peak weight has a narrow window and it is below 1.0.** At
   `w_torque_peak = 2.0` with the AL's `lam_torque` at 7.2 the term reached 50 %
   of the objective and the model got *worse* on the very statistic being
   punished.  At an effective ~0.5 (the value the winning step-1 000 checkpoint
   was trained at) `ok_torque` went 0.53 → 0.93.  At a full 1.0 a further 1 200
   steps regressed it to 0.81 and collision 0.72 → 0.64.  Both halves are
   multiplied by the same AL multiplier, which is calibrated on the *mean*
   violation — giving the peak its own multiplier is the obvious fix and is not
   done here.
3. **The committed checkpoint is step 1 000 of the continuation.** Everything
   after it, at either peak weight tried, is worse on the rolling grid.  A
   1 000-step continuation of a 7 500-step run is not a converged result; it is
   the best point of a search that ran out of window.
4. **The chained-BPTT stage never ran.** `bptt_start_frac = 0.55` of a
   21 000-step budget puts it at step 11 550, and the winning checkpoint is at
   step 1 000.  Every number in this chapter is therefore *without* the phase-3
   term.
5. **Restoring `w_smooth_jerk` breaks trajopt's 10:1 acc:jerk ratio.**
   `w_smooth_acc` stays at 1e-4 and `w_smooth_jerk` goes back to 1e-4, so the
   ratio is now 1:1.  It works — jerk integral p50 halves, 4.13e7 → 1.83e7,
   which is most of the way to v2b's 1.35e7 — but it was not A/B'd against
   restoring both, and it is confounded with the peak term in this run.
6. **The closed-loop strike is 40–56 mm short on the head axis** (above).
   Recommended next lever, in order: (a) price the *executed prefix*, not just
   the plan — e.g. a term on the terminal progress achieved by waypoint 1, or
   train on the concatenation of first waypoints rather than on whole plans;
   (b) evaluate and train the 1–5 waypoint horizon explicitly, which neither the
   243-case grid (shortest N = 22) nor the pool covers densely, and which the
   MPC is in for every strike; (c) only then revisit the goal switch.
7. **The roller's re-goal did not measurably help the closed loop yet.** It is
   correct, it is exercised (12.35 re-goals per tick per 256 slots), it made the
   roller 32 % faster and its divergence rate zero, and it fixed a real
   context/mask disagreement — but with only 1 000 committed steps its effect on
   closed-loop hit accuracy is not separable from the noise, and the axial
   finding above suggests it was not the binding cause.
8. **`roller_regoal=False` also disables the time-to-hit ageing**, so the A/B
   measures the re-goal and the `tth` fix together.  The `tth` fix is an
   independent, unambiguous bug fix and should probably be unconditional.
9. **The waist seed on a both-hands re-goal is a hold**, not
   `waist_reference_block`.  That aim law is host-side numpy; porting it to
   batched torch is the faithful version, and it is the one piece of the
   "one arm re-goals over a *shared waist*" story that is approximated.
10. **`best.pt` is per-leg.** Each `--resume` restarts the best-so-far at −1, so
   a continuation that never beats its predecessor still overwrites `best.pt`
   with something worse.  The winning checkpoint is preserved by hand as
   `runs/phase2b_main/leg1_best.pt`.

#### What is left of the torque gap

23 of 243 cases still fail `ok_torque`, and they have exactly the shape the
peak term was built for: the median failure is **1 knot of 30** over the limit
(0.14–0.19 % of the elements, worst 2 knots), by 3.7 N·m on `standard` and
9.9 N·m on `near_drum`.  The mechanism is unchanged, so the lever is unchanged —
but the two continuations above show that simply turning the peak weight up
destabilizes the whole objective.  The reason is structural: the AL multiplier
`lam_torque` is driven by the **mean** raw violation and multiplies both halves,
so raising the peak price also amplifies a multiplier that has no idea the peak
exists.  **Giving `torque_peak` its own constraint group in
`lagrangian.CONSTRAINT_GROUPS`, driven by `diag["torque_hinge_max"]`, is the
recommended next change** — it is a handful of lines, and it is what lets the
peak price rise while the mean price relaxes.

#### Continuation legs, for the record

| leg | resume from | lr | `w_torque_peak` | best rolling acc |
| --- | --- | ---: | ---: | ---: |
| aborted | `phase2_main/last.pt` | 3e-4 | 2.0, ramp 2 k | **0/81** @ 1 300 |
| **1 (committed)** | `phase2_main/last.pt` | 5e-5 | 1.0, ramp 2 k | **54/81** @ 1 000 |
| 2 | `leg1_best.pt` | 5e-5 | 1.0, full | 42/81 @ 1 200 |
| 3 | `leg1_best.pt` | 5e-5 | 0.5, full | 34/81 @ 800 (hit disk 0.67) |

Legs 2 and 3 both regress from leg 1's peak at *either* peak weight, and leg 3
loses the hit disk itself (1.00 → 0.67).  Step 1 000 is a real optimum of this
objective, not a lucky eval, and pushing past it needs the multiplier change
above rather than more steps.

---

## Section 7c — the closed-loop miss is a SERVO problem, not a planning one

Section 7b left one blocker: the amortized optimizer beat the trajopt expert
outright on the 243-case grid (175/243 vs 149/243) and landed **2 of 55**
strikes in the hit disk in closed loop, arriving 40–56 mm short on the head
axis.  7b's reading was *receding-horizon procrastination* — each replan
deferring the approach into waypoints it never executes.

That reading is **wrong in the axial direction and right in the in-plane one**,
and neither half is the dominant term.  The decomposition below is the whole
chapter; everything after it follows from it.

### The decomposition

`scripts/diag_7c.py` runs four probes on the **unchanged** champion.

**1. There is no off-by-one.**  Analytically the chain closes:
`_hit_index` gives `k = round(tth / dt) >= 1`; the mask is built from `k * dt`,
so plan waypoint `k - 1` is the terminal; `FusedCost._terminal` prices
`sp.knot(k)`, and knot `i` IS waypoint `i - 1`; `run_episode` commands
`traj[0]`, appends `q_hist[step + 1]`, decrements, and records `(step + 1, …)`
when a hand reaches 0; the scorer reads `q[k] = q_hist[step + 1]`.  Same
instant.  Empirically, the tick offset minimising `|axial|` over a real episode
is spread across −2…+2 with no mode (7/4/4/7/1) — a systematic index error would
put every case on one offset.  **Ruled out.**

**2. The planner is right; the plant is behind it.**  Logging, at every tick of
a closed-loop episode, what that tick's plan puts at its OWN terminal waypoint,
as a function of the remaining horizon `h`:

| `h` (waypoints to the hit) | 13 | 10 | 8 | 5 | 4 | 3 | 2 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| planned terminal, **axial** (mm) | +8 | +6 | +3 | +1 | 0 | −2 | −0 | **+10** |
| planned terminal, **in-plane** (mm) | −28 | −13 | −9 | −2 | +8 | **+38** | **+62** | **+61** |
| measured at the hit tick | | | | | | | | axial **+36**, in-plane **+53** |

Read it as two independent failures:

* **axially the plan is on the head at every horizon** (±10 mm) and the *plant*
  is 36 mm short of it.  There is no axial procrastination at all.
* **in-plane the plan is inside the hit disk down to `h = 5`** (−2 mm) and then
  walks 60 mm OUT of it over the last three refinements.  That is real
  procrastination — but confined to `h <= 4`, which is a short-horizon
  *competence* gap, not a deferral the temporal-consistency term rewards.

**3. Open loop through the PD sim is far worse, which is the proof.**  Take a
grid plan the harness accepts and execute it start to finish through
`BatchedPDSim` with **no replanning** (54 `standard` cases, randomized gains):

| | plan (kinematic) | plant (through the PD) |
| --- | ---: | ---: |
| axial at the terminal knot, p50 | +9.2 mm | **+143.0 mm** |
| in the hit disk | **54/54** | **1/54** |
| max joint tracking error at the strike | — | **152 mrad** |
| tip speed at the strike, p50 | 2.01 m/s | 1.61 m/s |

The plant reaches `min |axial| = 9.7 mm` only if the terminal command is HELD,
and even after 8 extra ticks 30 of 54 are still converging.  This is not a
one-tick transport delay.  It is the standing position error a **position-
controlled** servo must carry to produce torque: the only torque source is
`kp (q_cmd − q) + kd (qd_cmd − qd)`, so accelerating the arm REQUIRES
`e ≈ M(q) q̈ / kp`, and with `kp = 40` on every arm joint a 2 m/s strike costs
0.15 rad of it.  Closed-loop replanning recovers most of that (143 → 36 mm)
because a one-waypoint step from the measured state is a much smaller demand.

**The uncomfortable corollary, and the most important thing in this chapter:
the 243-case acceptance grid — and `FusedCost` itself — are PURELY KINEMATIC.**
Both score the waypoints.  Neither has ever asked whether the plant can be
there.  A plan that is kinematically on the head executes ~140 mm short; a plan
that executes on the head is kinematically ~140 mm *through* it, which the grid
scores as a miss AND a collision.  The two metrics are in direct conflict on a
`kp = 40` servo, and every number in sections 3–7b is on the kinematic side of
it.

**4. A second, independent bug fell out: the closed-loop goal was re-randomized
every tick.**  `rhythm.goal_snapshot` draws the aim point with
`sample_hit_point`, which jitters uniformly in a disk of `hit_radius_frac`
(0.6) × the legal radius; `run_episode` re-snapshots every tick, so ONE
scheduled hit had its aim redrawn every 20 ms — a target teleporting by up to
~0.2 m on a snare.  Training has no such thing (a phase-2 scenario draws its aim
once, and `MPCRoller` carries `goal_pos` forward), so this was a pure
eval/deployment mismatch.  `MPCGenConfig.sticky_goal_point` (default **True**)
caches the draw per `(hand, drum, absolute hit time)`.  Worth fixing; it is
**not** the story — 2/55 → 3/55.

### The three deployment knobs, and which one to ship

All three are on `MPCGenConfig`, all measured on the unchanged champion over 3
eval clips / 55 executed strikes.  `axial` positive = short of the head;
`in-plane` positive = outside the disk edge; `early` = arrived through the head
at the scheduled tick, `late` = still short of it.

| config | mech | params | in-disk | ±20 mm | early | late | axial p50 | in-plane p50 | diverged | PD p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (the 7b protocol) | 0 | 0 | 3/55 | 13 | 11 | 31 | +35.5 | +53.1 | 0 | 6.8 |
| goal-shift 20 ms | 1 | 1 | 5/55 | 15 | 15 | 25 | **+7.7** | +52.2 | 0 | 6.7 |
| freeze 5 wp (arm+waist) | 1 | 2 | 7/55 | 10 | 2 | 43 | +47.8 | **−24.4** | 0 | 6.9 |
| freeze + goal-shift | 2 | 3 | 8/55 | 10 | 3 | 42 | +35.0 | −16.0 | 0 | 7.1 |
| ref-lead 5 ms uniform | 1 | 1 | 10/55 | 17 | 8 | 30 | +25.5 | +13.9 | 0 | 7.1 |
| ref-lead 10 ms uniform | 1 | 1 | 7/37 | 10 | 8 | 19 | +26.0 | −0.2 | **1** | 8.5 |
| ref-lead 15 ms uniform | 1 | 1 | 5/55 | 13 | 13 | 29 | +21.4 | −5.1 | 0 | 10.8 |
| ref-lead 20 ms uniform | 1 | 1 | 5/55 | 17 | 14 | 24 | +4.8 | −0.8 | 0 | 12.8 |
| ref-lead 25 ms uniform | 1 | 1 | 8/55 | 13 | 21 | 21 | −7.5 | −15.1 | 0 | 13.5 |
| **ref-lead PER-JOINT (`"fit"`)** | **1** | 17 | **21/55** | **28** | 9 | 18 | +10.2 | −44.4 | 0 | 8.3 |
| ref-lead per-joint (`"sparse"`, 4) | 1 | 4 | 10/55 | 16 | 20 | 19 | +1.1 | +15.7 | 0 | 7.5 |
| ref-lead per-joint (`"wide"`) | 1 | 17 | 5/36 | 9 | 17 | 10 | −14.4 | −15.9 | **1** | 12.4 |
| per-joint + freeze | 2 | 19 | 12/55 | 19 | 6 | 30 | +23.7 | −29.3 | 0 | 8.9 |
| per-joint(sparse) + freeze | 2 | 6 | 14/55 | 15 | 9 | 31 | +23.7 | −48.2 | 0 | 8.6 |
| per-joint + goal-shift | 2 | 18 | 20/55 | **29** | 15 | 11 | **−5.6** | −32.3 | 0 | 8.6 |
| ref-lead 10 ms + goal-shift | 2 | 2 | 0/37 | 6 | 14 | 17 | +13.8 | +9.1 | **1** | 8.4 |
| per-joint + freeze + goal-shift | 3 | 20 | **27/55** | 29 | 8 | 18 | +13.5 | −35.8 | 0 | 8.8 |

**Recommendation: the per-joint reference lead, alone.**  One mechanism,
`MPCGenConfig.cmd_lead_s = LEAD_PRESETS["fit"]`, 21/55 against the baseline's
3/55 — 75 % of the best achievable (27/55) at a fifth of the complexity, the
lowest PD tracking error of any lead configuration, and no divergence.  The
extra +6 strikes the three-mechanism stack buys is inside 1.7 sigma at n = 55.
The freeze and the goal shift stay in the config, **defaulted off**, and are
worth layering only against a measured residual: the freeze is the only thing
that fixes in-plane aim when the lead is absent, and the goal shift is the only
thing that takes the axial median to zero.

Three things the table says that are worth stating separately:

* **the two knob families act on different axes.**  The freeze fixes AIM
  (in-plane +53 → −24) and leaves 43/55 short; the goal shift and the uniform
  lead fix REACH (axial +35 → +5…+8) and leave the aim untouched.  Only the
  per-joint lead moves both, because it is the only one that improves tracking
  over the *whole* approach rather than at the terminal.
* **a uniform delta cannot substitute for the per-joint one.**  5–25 ms all land
  at 5–10/55 and 10 ms diverges a clip.  The fitted spread is the reason: 0 ms
  at the wrists, 15 ms at the elbows, 50 ms at waist pitch/roll.  Averaging that
  into one number gives every joint the wrong lead.
* **overcompensation is real and visible in the `early` column.**  At 25 ms
  uniform, 21/55 arrive THROUGH the head; the `"wide"` fit (waist 80–100 ms,
  left shoulder 100–120 ms) diverges a clip.  The narrow fit is the safe end of
  the identification.

### `amortized_opt.lead` — how the leads are identified

`q_cmd(t) = q_ref(t + Δ)`, linearly interpolated between plan waypoints, so
sub-tick Δ is meaningful.  Applied to the plan BEFORE the slew/box clamp, so the
command stays admissible, and to every tick including frozen ones.  The velocity
feed-forward follows for free: the FOH plant derives its rate target from
`(x_i − x_{i-1}) / dt` over the EXECUTED commands, so leading every command
leads the rate target by the same Δ.

Two semantics matter:

* **the lead may only read INSIDE the plan.**  Past a hand's terminal knot the
  decoder's output is unpriced (the element mask zeroes it and no cost term
  touches it), so the reference is HELD there.  A consequence: *at the strike
  tick itself the hand's horizon is one waypoint and the lead is a no-op*.  A
  reference lead compensates delay in the MIDDLE of a trajectory; it
  structurally cannot advance its END, because there is no trajectory beyond the
  end to advance into.  That is exactly what `hit_lead_s` is for, and why the
  two are complementary rather than redundant.
* **it must match what the TRAINING roller executes.**  `Phase2Config.cmd_lead_s`
  drives the roller, the temporal-consistency plant and the chained-BPTT plant;
  a lead present in deployment and absent in training is a distribution
  mismatch, and a model that has learned its own lead on top of a control-side
  lead strikes early.

`scripts/fit_cmd_lead.py` identifies Δ numerically against the exact servo model
(the package owns it, so this is not a guess): sweep Δ, roll 162 grid plans
through `BatchedPDSim` commanding `q_ref(t + Δ)`, take the per-joint argmin of
`|q_act − q_ref|` RMS over the 6 knots around the strike.

| group | fitted Δ (median) | strike-window RMS at Δ = 0 | at Δ* | gain |
| --- | ---: | ---: | ---: | ---: |
| waist (pitch/roll) | 50 ms (80–100 ms unbounded) | 56.9 mrad | 39.4 | 31 % |
| shoulder | 2.5 ms | 50.4 | 50.1 | 1 % |
| elbow | 15 ms | 32.1 | 17.7 | 45 % |
| wrist | 0 ms | 8.2 | 8.2 | 0 % |
| best UNIFORM Δ | 5.0 ms | 49.0 (all joints) | 47.9 | **2 %** |

The spread is the informative part, and so is the last row: **on the
tracking-RMS objective a uniform lead buys 2 %**, which is why the axial-only
knobs plateau.  Per joint, the elbows and the waist have genuine minima (45 % /
31 %) while the shoulders and wrists are flat — the shoulders' error is
attenuation, not phase, and a lead cannot touch it.

**Known limit, stated plainly:** a lead cancels *delay*, not *bandwidth*.  Of
the 152 mrad strike transient it recovers the part that is a pure phase shift;
the rest needs the planner to stop asking for motions this servo cannot
produce, which is a training problem, not a control one.

### Dynamics-based lead (Kp⁻¹·M·q̈) — experiment

The fixed time lead has two limitations: it needs per-joint fitting against the PD
model, and it borrows future waypoints that the cost never priced at the lead
offset.  A dynamics-based alternative computes the exact position offset the
P-gain needs to generate the required accelerating torque:

```
q_cmd = q_ref + Kp⁻¹ · M(q) · q̈_ref
```

where `M(q)·q̈` comes from two RNEA calls (`rnea(q, 0, q̈) − rnea(q, 0, 0)`) and
`Kp` is the per-joint runtime stiffness.  The offset is clamped to
`±τ_lim / Kp` — the offset that fully saturates the actuator — so the command
never asks for more torque than the servo can deliver.

Implemented in `play_isaac.py --dynamics_lead` (which also sets `cmd_lead_s=None`,
removing all timing adjustment), measured against the baseline (`--cmd_lead_s
fit`) on 4 eval clips, 249 control steps each:

**Joint limit violations:** none in either configuration.  The box clamp
(`torch.clamp(cmd, q_lo, q_hi)`) works perfectly — the model's planned trajectory
stays within bounds and the dynamics offset does not push it out.

**Torque violations** (computed torque exceeding the effort rating):

| | dynamics lead | fixed time lead (`fit`) |
| --- | ---: | ---: |
| clip 0 | 26/249 (10.4 %) peak 165 % | 5/249 (2.0 %) peak 36 % |
| clip 1 | 28/249 (11.2 %) peak 190 % | 2/249 (0.8 %) peak 87 % |
| clip 2 | 12/249 (4.8 %) peak 173 % | 3/249 (1.2 %) peak 101 % |
| clip 3 | 12/249 (4.8 %) peak 173 % | — |

The dynamics lead produces 4–5× more torque-violation steps than the fixed time
lead even with the offset clamp.  The mechanism is clear: `M·q̈ / Kp` computes
exactly the acceleration demand, but when the model's acceleration head is noisy
or the plan asks for motions the actuators cannot sustain, the offset saturates
at the effort limit and the servo still cannot track.

**Peak lead offset magnitude:** max observed was 108 mrad (6.2°) on the
shoulder — well within the 625 mrad budget at Kp=40, τ_lim=25 N·m.

**Verdict:** the dynamics lead is physically correct and needs no per-joint
fitting, but the model's acceleration head was trained as an auxiliary output
(never used to drive the command), so `q̈_ref` carries the full noise of a
supervised head rather than the precision of a control quantity.  The fixed time
lead remains the deployment recommendation.  The dynamics path is opt-in via
`--dynamics_lead` and the offset clamp is in place; revisit when the acceleration
head is trained with a control-aware objective or when the fixed lead's borrowing
from unpriced future waypoints causes a measurable problem.


### `sticky_goal_point`, `hit_lead_s`, `freeze_wp` — implementation notes

* **`sticky_goal_point`** (default True) caches the jittered aim per
  `(hand, drum, absolute hit time)`.
* **`hit_lead_s`** queries the rhythm at `t_now + hit_lead_s` rather than
  post-shifting the goals' `time_to_hit`.  The first version did the latter and
  every hit fired `lead/dt + 1` times, because the hit stays in the clip query
  until its TRUE time — 110 recorded strikes on three clips that contain 55.
  `_consumed` (one float per hand) is a second guard: `_hit_index` pulls
  anything under half a period up to `k = 1`, so a lead landing inside the final
  half tick would otherwise hand the same hit back next tick.
  `tests/test_mpc_deploy_knobs.py` pins exactly-once firing at six leads.
* **`freeze_wp` / `freeze_waist`** replace the striking arm's columns (and,
  with `freeze_waist`, the waist's) with the previously committed plan shifted
  one waypoint.  The planner is still CALLED every tick — the plan doc uses the
  window to prepare the next context, which only works if the loop keeps
  running.  Freezing the whole row instead diverged a clip; freezing the arm
  but replanning the waist recovers only 53 → 40 mm of the in-plane miss
  against −24 mm for arm+waist, because the waist carries the arm.

### Final-approach oversampling

The roller harvests every slot on every tick, so its horizon distribution is
roughly uniform over each goal's initial horizon.  Measured at 256 slots driven
by the champion:

| fraction of rows with nearest active hit ≤ | 1 | 3 | 5 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| replay, bucket 25 | 0.076 | 0.216 | 0.343 | 0.512 | 0.672 |
| replay, bucket 50 | 0.033 | 0.086 | 0.188 | 0.310 | 0.442 |
| fresh pool, bucket 25 | 0.031 | 0.158 | 0.362 | 0.542 | 0.680 |
| fresh pool, bucket 50 | 0.007 | 0.041 | 0.111 | 0.176 | 0.225 |

`Phase2Config.short_wp` / `short_boost` multiply the sampling weight of rows
inside the band, in the fresh pool AND in the recency-biased replay (the two
weights multiply).  At `short_wp = 8, short_boost = 3.0` the mix the optimizer
actually sees moves from the buffer's own distribution to

| sampled batch, `k_min` ≤ | 3 | 5 | 8 |
| --- | ---: | ---: | ---: |
| boost 1.0 (= the buffer) | ~0.15 | ~0.27 | ~0.41 |
| **boost 3.0** | **0.19** | **0.40** | **0.62** |

`train_phase2` logs it every `log_every` steps as `batch_short`, so the mix is
recorded rather than assumed.

### Shelved with evidence: the executed-prefix / approach-envelope cost term

7b recommended pricing the *executed prefix* — a hinge on the planned
distance-to-contact staying inside a feasible approach envelope, so a lazy plan
is caught early instead of at the terminal.  It is **not implemented**, and the
drift table above is why: the planned terminal is axially correct at EVERY
horizon (±10 mm from `h = 13` down to `h = 1`), so there is no axial deferral
for an envelope to catch.  The only procrastination-shaped residual is in-plane
and confined to `h <= 4`, which the freeze window removes outright without
retraining and which final-approach oversampling attacks at the root.

Two design notes for whoever revisits it:

* the envelope must be a **reachability cone**, `dist_j <= ‖v*‖ (k − j) dt`, not
  a linear or spline-eased schedule.  A schedule that forces monotone progress
  contradicts the impact-speed term directly: the last tick has to cover
  `v* dt` ≈ 40 mm at 2 m/s, so a linear schedule that puts the tip 0.75 cm out
  at `j = k − 1` makes the commanded impact speed unreachable.  The cone is also
  a closed-loop invariant — a plan satisfying it at every knot leaves the next
  tick's state exactly on its own cone — which the "reach at the terminal knot"
  objective is not, and that invariance is the actual anti-procrastination
  property.
* it must be a **static weight, not an AL group**.  On closed-loop states the
  cone is sometimes unsatisfiable, and the repo has already paid for that once:
  an unsatisfiable shortfall Lagrangian pinned its multiplier at `lam_max` and
  starved a 2 197-iteration run.

### The idle arm (implemented 2026-08-12)

An arm with no goal in the context lookahead has **no terminal objective** —
only collision, limits, smoothness and the mid-range posture well price it — so
its behaviour is underdetermined, and the known failure is the waist (serving
the active hand) sweeping the idle arm into the kit.

**The fix is to give it the same thing a striking hand has: a terminal.**  A
hand with a hit is priced on where its terminal knot lands (`w_terminal_region`)
and how fast it is going there (`w_terminal_velocity`).  A hand with no hit now
gets exactly those two, aimed at a resting posture instead of a drum head:

* `w_idle_home` — `|q_arm(K) − home|²` at that hand's terminal knot;
* `w_idle_vel` — `|qd_arm(K)|²`, i.e. **arrive stopped**.

That is the whole objective change.  Everything else was already there and is
**identical for both hands, with no special case**: collision, self-collision,
torque, velocity and joint-box all run over the idle arm's full horizon exactly
as they do over a striking arm's (verified term by term in
`tests/test_idle_home.py`).  So the idle terms do not have to keep the arm out
of the kit — they only say where it should end up, and the rest of the objective
shapes how it gets there.  Mid-horizon the arm is free; only its terminal is
priced.

Both are static weights, not AL groups — an arm mid-flight from a strike cannot
be at home, and this repo has already paid once for pricing an unsatisfiable
condition with a multiplier (the shortfall Lagrangian pinned at `lam_max`
starved a 2 197-iteration run).  `phase2_cost_config` carries
`w_idle_home = 1.0`, `w_idle_vel = 0.02`; both are exactly zero, with exactly
zero gradient, until a posture is installed.

**`home` — `amortized_opt/idle_home.py`, `assets/idle_home_poses.json`.**
**One 7-DoF posture per hand.**  Arm joint angles are already torso-relative, so
a fixed vector rides the waist for free: an idle arm holding it keeps its
posture relative to the body however the torso turns, and genuinely **sits
still** rather than tracking a moving target.  The term is therefore exactly
yaw-independent — no waist column enters it at all.

The posture is solved offline for clearance against the kit at all 129 waist
yaws simultaneously (the kit is world-fixed while the arm swings with the torso,
so the parked posture sweeps a circle that should miss every drum) plus a pull
toward that hand's strike poses and an elbow-down hinge, projected onto the
posture box.  This is a *better constant*, not a second collision system — the
cost's collision terms are what actually keep the arm safe.

* worst-case clearance **87 mm** over every yaw and both hands, 0 in collision;
  the URDF ready pose an idle arm used to hold is in collision at **10/129** yaws
* park → every reachable strike pose in **0.275 s** worst case against a 1 s
  budget (3.6× margin), which is the acceptance bar
* a yaw-INDEXED table was built first and rejected: it clears more tightly but
  asks a *resting* arm to keep moving as the torso turns, and made the target
  discontinuous in yaw (46/128 adjacent 2.3° bins jumped >0.15 rad)
* the clearance hinge must be **linear, not squared** — a squared hinge's
  gradient vanishes as the violation shrinks and the solve parked 9 mm *inside*
  the torso capsule and stopped

**The seed.**  A hand with no goal eases to the park posture over
`idle_settle_wp` waypoints (stretched to stay inside the joint speed ratings),
arrives at rest, and holds — in `warmstart_trajectory`, in
`MPCRoller._seed_idle` (every tick) and in `MPCGenerator`
(`park_idle_every_tick`).  Like the strike seed it is only an initialization;
the cost refines it.

**The token.**  `ModelConfig.idle_home_token` (on by default) gives the idle
hand its own context row per hand — the park posture, its tip, and the wait
until that hand's next hit — valid exactly on hands with no strike, so a hand
never carries both a strike target and a park target.  Off restores the previous
encoder exactly, including the `type_emb` row count, so old checkpoints load
strictly.

**Data.**  `MPCRoller` used to respawn a slot the instant `~active.any(dim=1)`,
deleting rest stretches; it now keeps a both-idle slot for
`roller_rest_ticks` (default 25 = 0.5 s) while it still has a future hit.
`Phase2ScenarioConfig.rest_frac` keeps that fraction of naturally both-idle
draws.  Single-hand idle was always ~40 % of the pool — the old note's claim that
`require_active_hand` drops it is wrong, it only drops both-idle.

**Train/eval parity.**  The acceptance grid is entirely single-hand, so it is
made of exactly these states, and `best.pt` is gated on it.
`planner.batched_model_plans`, `planner.ModelPlanner` and the closed-loop probe
now park the idle arm and fill its token by default, matching the roller; before
this they held it in place and emitted no token.

**Not yet measured:** the closed-loop 3-clip before/after on idle-arm collision
count and min-SDF p10, and the re-engagement split — those need a training run.
`SdfPenaltyDetails.extras["min_sdf_left"/"min_sdf_right"]` is the per-hand
clearance that split needs.

```
python scripts/build_idle_home.py     # once, writes the asset
python scripts/train_phase2.py ...    # --no-idle-home / --no-idle-token to disable
```

### The metric that actually ranks the knobs: hit latency

"In the disk at tick k" is a binary on one 20 ms sample of a moving stick, and
the offset tables show the distribution is **asymmetric-late** — almost nothing
arrives early, most strikes cross the head plane one to three ticks behind the
beat.  Those are hits; they are behind time.  `probe.hit_latency` measures that
directly: scan the plant over ±5 ticks around the scheduled instant, find the
first crossing of the head plane that is inside the legal disk, and locate it to
**sub-tick** resolution by interpolating the axial coordinate (a 20 ms grid is
too coarse for a ±10 ms question).  The in-plane test and the impact speed are
evaluated at the interpolated pose.  A strike that never crosses in-disk in the
window is a genuine **miss**, which the at-tick binary conflates with "late".

Same 3 clips / 55 strikes, champion checkpoint:

| config | crossed in disk | miss | lat p50 | \|lat\| p50 | \|lat\| p90 | ≤10 / ≤20 / ≤40 ms | early | speed in band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4/55 | 93 % | +23.5 | 23.5 | 74.4 | 1 / 2 / 3 | 0 | 3 |
| goal-shift 20 ms | 4/55 | 93 % | −16.8 | 24.7 | 67.5 | 0 / 2 / 3 | 3 | 2 |
| freeze 5 wp | 11/55 | 80 % | +10.5 | 16.6 | 39.8 | 3 / 7 / 10 | 4 | 4 |
| ref-lead 5 ms uniform | 9/55 | 84 % | +13.0 | 15.3 | 39.2 | 0 / 7 / 8 | 3 | 6 |
| ref-lead per-joint (sparse) | 16/55 | 71 % | −2.7 | 9.9 | 40.5 | 8 / 11 / 14 | 9 | 13 |
| **ref-lead PER-JOINT (`"fit"`)** | **25/55** | **55 %** | +4.8 | **10.7** | **23.2** | **12 / 21 / 24** | 7 | **16** |
| per-joint + goal-shift | 23/55 | 58 % | −12.2 | 12.2 | 25.9 | 9 / 18 / 23 | **23** | 12 |
| per-joint + freeze + goal-shift | 19/55 | 65 % | −3.5 | 16.8 | 42.7 | 5 / 12 / 17 | 11 | 8 |

This **reverses** the at-tick ranking and confirms the recommendation harder.
The three-mechanism stack won the binary (27/55 vs 21/55) purely by happening to
be in the right place on the sampled tick; on latency it is worse than the
per-joint lead alone on every column that matters (miss 65 % vs 55 %, p90 43 ms
vs 23 ms, speed-in-band 8 vs 16).  And `per-joint + goal-shift` is visibly
**over-compensating**: all 23 of its crossings are early.  The per-joint lead
alone takes the miss rate from 93 % to 55 %, the timing p90 from 74 ms to 23 ms,
and the number of strikes landing at a legal impact speed from 3 to 16.

Selection is now on this: `probe.closed_loop_probe`'s `score` is
`(1 − miss_rate) + 0.5·within20 + 0.25·within40 + 0.25·in_disk@k`, all
normalized, and `train_phase2` gates `best.pt` on it.

### The 7c continuation, `runs/phase2c_main` — and why it is NOT the new champion

Resumed from `runs/phase2b_main/best.pt` (model + optimizer + AL) at lr 5e-5,
with everything section 7c added: the `torque_peak` AL group at an effective
price matched to the resume point (`w_torque_peak = 0.5`, ramp 1, `lam` seeded
from `lam_torque = 7.26`), final-approach oversampling
(`short_wp = 8, short_boost = 3.0`), chained BPTT live from step 0 at
`bptt_max_frac = 0.15`, the roller executing the same per-joint lead the
deployed loop does, a rolling grid every 300 steps and a 2-clip closed-loop
probe every 600.  It reached step 7 150 at 2.6–3.4 it/s before the harness
killed it (~25 min short of its 1.5 h budget); the checkpoints are intact.

| | rolling grid (81) | closed-loop probe (42) |
| --- | ---: | ---: |
| resume point (phase 2b) | 54/81 | **15/42** |
| best rolling grid | **62/81** @ 2 700 | 2/42 at that step |
| `best.pt` (gated) | 60/81 @ 2 100 | 7/42 |
| best probe | 57/81 @ 3 600 | 8/26 (a clip diverged) |
| step 7 000+ | 47–53/81 | 4–6/42 |

**The probe gate fired and was right.**  At step 2 700 the rolling grid hit
62/81 — the best of the run — and `best.pt` was NOT overwritten, because that
checkpoint's closed-loop probe had collapsed to 2/42 against the 7/42 recorded
with the incumbent.  Section 7b shipped exactly that trade blind.

#### The full grids, and the fresh seed

`standard` (162) + `near_drum` (81), one refinement iteration.  Seed 0 is the
**validation** grid — it is what every phase-2 rolling eval selected on.  Seed 7
had never been looked at.

| | phase 2c (step 2 100) | phase 2b | trajopt v2b |
| --- | ---: | ---: | ---: |
| **accepted, seed 0 (validation)** | **182/243** (hold 184) | 175/243 | 149/243 |
| **accepted, seed 7 (fresh)** | **172/243** (hold 176) | **180/243** (hold 184) | — |
| both seeds pooled | 354/486 | **355/486** | — |
| hit disk, seed 0 | **243/243** | 242/243 | 241/243 |
| collision-free, seed 0 | **198/243** | 193/243 | 162/243 |
| no torque excess, seed 0 | **225/243** | 220/243 | 209/243 |
| short-horizon cell, seed 0 | **36/54** | 32/54 | 26/54 |
| short-horizon cell, seed 7 | 39/54 | **42/54** | — |

Read the last two rows of the first block together: **phase 2c's +7 on seed 0 is
validation-grid overfitting.**  It gains 7 cases on the grid its own rolling eval
selected against and loses 8 on a grid it has never seen; pooled over both, the
two checkpoints are indistinguishable (354 vs 355 of 486).  The component gates
do improve on seed 0 — hit disk 243/243, collision +5, torque +5, short cell
+4 — but they do not transfer.  This is the first untouched-test-set measurement
in the project, and it should be repeated for phase 2b's own headline number:
175/243 was also a seed-0 number.

#### Closed loop, phase 2c vs phase 2b, on the recommended config

Same 3 clips, 55 strikes, `latency` as the primary metric:

| | phase 2b bare | phase 2b **+ per-joint lead** | phase 2c bare | phase 2c + per-joint lead |
| --- | ---: | ---: | ---: | ---: |
| crossed in disk | 4/55 | **25/55** | 4/55 | 8/55 |
| miss rate | 93 % | **55 %** | 93 % | 85 % |
| \|latency\| p50 | 23.5 ms | 10.7 ms | 20.5 ms | **6.1 ms** |
| \|latency\| p90 | 74.4 ms | **23.2 ms** | 59.3 ms | 35.9 ms |
| within 20 ms | 2 | **21** | 2 | 6 |
| impact speed in band | 3 | **16** | 2 | 2 |
| ticks completed | 750/750 | 750/750 | 750/750 | 750/750 |

**phase 2c is worse in closed loop than the checkpoint it continued from** — 8/55
against 25/55 with the same compensation — which is exactly what its own
in-training probes said at every one of the eleven times they ran (best 10/42,
against the incumbent's 15/42).  Its timing when it does connect is tighter
(\|p50\| 6.1 ms), but it connects a third as often.

**`runs/phase2b_main/best.pt` therefore remains the champion.**  What the
continuation produced is not a better model; it is the evidence that *more
descent on a purely kinematic cost does not buy closed-loop strikes*, measured
against a probe that was not available before.  The infrastructure it exercised
— the decoupled `torque_peak` multiplier, the oversampling, the chained BPTT,
the gated selection — is all live and all tested, and is what a plant-aware
objective would be built on.

### Section 7c decisions needing review

1. **The acceptance grid and `FusedCost` are kinematic, and that is now the
   binding limitation of the whole project.**  Both score waypoints; neither
   asks whether a `kp = 40` servo can be there.  A plan that is kinematically on
   the head executes ~140 mm short.  Every headline number in sections 3–7b
   lives on the kinematic side of that gap.  The principled fix is to price the
   PD-EXECUTED state: roll the plan through the differentiable
   `BatchedPDSim` (the chained-BPTT step already does exactly this roll) and
   evaluate the terminal geometry at the reached pose.  It is NOT implemented
   here — it is a real objective change with a real conflict (the exec terminal
   wants the plan to overshoot the head; the plan terminal and the grid want it
   on the head), and it deserves its own chapter and its own A/B.
2. **The recommended deployment config is one mechanism: the per-joint reference
   lead.**  `MPCGenConfig.cmd_lead_s = LEAD_PRESETS["fit"]`, everything else
   off.  `hit_lead_s` and `freeze_wp` remain implemented, tested and defaulted
   OFF; layer them only against a measured residual.  On the latency metric the
   3-mechanism stack is worse than the lead alone.
3. **The per-joint lead's 17 numbers are a fit, and the fit is only as good as
   the servo model.**  They come from `scripts/fit_cmd_lead.py` against
   `BatchedPDSim` at the runtime Kp/Kd with the real coupled mass matrix.  If
   Isaac's actuator model differs — or if the RL fine-tune changes the effective
   stiffness — the vector must be re-fitted, not carried over.  The script is
   ~15 s of GPU time; re-run it.
4. **`sticky_goal_point` defaults to True and changes phase-1/MPC data
   generation**, not just evaluation: `MPCGenerator.generate` now emits samples
   whose goal token does not jitter within an episode.  That is the correct
   semantics and it matches what the phase-2 pool and the roller have always
   done, but any dataset regenerated after 2026-08-11 differs from one before.
5. **The rolling eval and the final grid are the SAME seed-0 grid.**  Every
   phase-2 checkpoint has been selected on it, and the seed-7 measurement above
   shows that selection is worth ~8 cases of transfer.  The rolling eval should
   move to a held-out seed.
6. **The closed-loop probe is 2 clips / 42 strikes and it is noisy.**  The
   eleven probes of the continuation span 1/42 to 10/42 with no trend, and one
   of them ran on a diverged clip (26 hits) and scored highest.  It is good
   enough to have blocked a bad checkpoint at step 2 700; it is not good enough
   to rank two similar ones.  A useful probe is probably 4–6 clips.
7. **The continuation degraded, and the degradation is visible in the cost, not
   just the eval.**  By step 7 000 the collision term had gone 12 → 38 and the
   torque term 58 → 180 while `lam_joint_box` climbed 25 → 55.  With chained
   BPTT live from step 0 (it never ran at all in 7b) this is confounded: the
   chained term contributes ~45 of the ~590 total at that point and it was not
   A/B'd.  Whether BPTT-from-step-0 is the cause or a bystander is unmeasured.
8. **`w_torque_peak` is now priced by `lam_torque_peak`, and I chose the resume
   value to make the objective continuous, not because it is optimal.**
   `lam` is seeded from `lam_torque` (7.26) and `w_torque_peak = 0.5` reproduces
   the exact effective price the winning 7b checkpoint was trained at.  The
   point of the split is that the peak price can now rise while the mean price
   relaxes; nothing in this chapter tested whether it should.
9. **The freeze window is per-hand on the ARM columns plus the shared waist.**
   Freezing the whole row diverged; freezing the arm alone recovers only half
   the in-plane gain.  The middle ground means the OTHER arm is replanned
   against a waist that is being held for the striking arm — a coupling nobody
   has priced.
10. **`train_phase2`'s probe gate compares a scalar `score`, and that score's
    formula changed mid-chapter** (from an at-tick composite to the
    latency-based one).  The continuation's recorded probe scores are on the
    OLD formula and are not comparable to anything measured after it.

## PID integral controller — lead × ki sweep (2026-08-11)

The per-joint reference lead (`cmd_lead_s = LEAD_PRESETS["fit"]`) is the
deployment config that ships with 7c.  It cancels servo *delay* by commanding a
point ahead on the planned trajectory.  What it cannot do is cancel
*attenuation* — a `kp = 40` servo that rolls off below the strike's frequency
content reaches a smaller peak velocity no matter when it is asked.

A PID controller adds an integral term `ki * ∫(q_ref − q) dt` that accumulates
steady-state tracking error and adds `(ki/kp) * integral` to the position
command.  In the offline PD sim the torque equation becomes:

```
acc = Minv @ (kp·err + ki·integral + kd·(ff·qd_cmd − va))
integral += err · h                    # clamped to ±ki_windup
```

**The lead and the integral fight each other.**  The lead intentionally shifts
the command *ahead*, so the plant's error against the led command is by design,
not a tracking failure.  Running both makes the integral accumulate against
that intentional offset.  A `pid_with_lead` flag exists for sweep experiments;
the default guard disables the integral whenever the lead is active.

**2D offline sweep** (`scripts/eval_pid_lead_sweep.py`) — scalar lead times
(0–20 ms) × ki (0–50), champion model on `gmd_eval_000001_30s` (23 hits):

| config | crossed | miss% | \|lat\| p50 | \|lat\| p90 | ≤10ms | ≤20ms |
|---|---:|---:|---:|---:|---:|---:|
| **champion lead=fit, ki=0** | **15** | 34.8 % | 8.0 ms | 40.5 ms | 8 | 12 |
| lead=off, ki=0 (pure PD) | 7 | 69.6 % | 13.1 ms | 38.2 ms | 1 | 6 |
| lead=off, ki=5 | 10 | 56.5 % | 8.8 ms | 22.8 ms | 6 | 9 |
| lead=10ms, ki=2 (+pid_with_lead) | 14 | 39.1 % | 12.2 ms | 31.8 ms | 6 | 12 |
| lead=15ms, ki=5 (+pid_with_lead) | 14 | 39.1 % | 11.0 ms | 28.5 ms | 6 | 12 |
| **lead=20ms, ki=2 (+pid_with_lead)** | **15** | 34.8 % | **7.0 ms** | **31.4 ms** | **10** | **13** |
| lead=20ms, ki=5 (+pid_with_lead) | 15 | 34.8 % | 12.0 ms | 32.2 ms | 7 | 13 |
| lead=20ms, ki=10 (+pid_with_lead) | 15 | 34.8 % | 10.5 ms | 32.8 ms | 7 | 13 |
| lead=fit, ki=1 (+pid_with_lead) | 14 | 39.1 % | 10.0 ms | 41.8 ms | 7 | 10 |

**Findings:**
- **Scalar lead=20ms + ki=2 ties the champion** (15 hits) with better timing
  consistency (p90 31.4 ms vs 40.5 ms, +2 hits within 10 ms).  One full control
  period of scalar lead plus a small integral gain is the winning combination.
- **Lead carries the weight** — going from ki=0 to ki=2 at lead=20ms adds 5
  hits (10→15); going from lead=0 to lead=20ms at ki=0 adds 3 hits (7→10).
  Together they are additive.
- **Integral hurts the per-joint "fit" lead** — adding ki to the champion's
  fit lead drops hits 15→14→13→12, because the per-joint lead is already doing
  the job the integral would do.
- **Too much ki always hurts** (ki ≥ 20 degrades everywhere), but the sweet
  spot shifts upward with a lead present (ki=2–10 works well with lead, vs
  ki=5 alone without it).
- **Anti-windup doesn't matter** on 250-tick episodes — the integral doesn't
  saturate enough to need clamping.

**Code.**  Integral support lives in `pd_sim.py` (`PDSimConfig.ki` /
`ki_windup` / `ki_sigma`; `simulate_pd` returns an `integral` tensor;
`BatchedPDSim.sample_gains` / `step` / `rollout` carry it through).  The MPC
loop in `mpc_gen.py` manages integral state per episode (reset on hit,
`pid_with_lead` guard).  The Isaac player (`play_isaac.py`) implements a PID
outer loop that adds `(ki/kp) * integral` to the position command, with
`--cmd_lead_s`, `--ki` and `--ki_windup` CLI args defaulting to **lead=20ms
+ ki=5.0**.  Unit tests: `tests/test_pid.py` (13 tests).

## Section 7d — waist-stabilisation ablation

The waist is three joints (yaw / pitch / roll) shared by both arms.  Pitch and
roll should stay near zero; yaw should follow `waist_aim`'s target (a damped
azimuth law that turns the torso toward the drum enough to give the arm room,
without squaring up to it).  Three configurations were compared, all continuing
from `runs/phase2b_main/best.pt`:

| config | mechanism | w | box | waist_aim IL |
|---|---|---|---|---|
| **phase2b** (baseline) | none | — | full URDF (±0.52) | — |
| **phase2d_waist** | `w_waist_upright` + `waist_pr_box` | 200 | ±0.20 | — |
| **phase2e_waist_aim** | `w_waist_aim` (3-DOF IL) | — | full URDF (±0.52) | 500 |

### Training evals (rolling grid, 81 cases, every 250 steps)

Each resumed from `phase2b_main/best.pt` (step 1000, acc 54/81 = 0.667).

| run | steps | best acc | tau_ok | coll_ok | elbow |
|---|---|---|---|---|---|
| phase2b (baseline) | 1000 | 54/81 (0.667) | 0.93 | 0.72 | 0.99 |
| phase2d_waist | 2500 | 59/81 (0.728) | **1.00** | 0.73 | **1.00** |
| **phase2e_waist_aim** | **5000** | **64/81 (0.790)** | **1.00** | **0.79** | 0.99 |

### Waist movement (max absolute pitch/roll and yaw over the trajectory)

Measured from the per-step diagnostics in `metrics.jsonl`, all steps.

| run | waist_pr p50 | waist_pr p90 | waist_yaw p50 | waist_yaw p90 |
|---|---|---|---|---|
| phase2d_waist | **0.34 rad (19°)** | 0.45 rad | **0.80 rad (46°)** | 0.83 rad |
| phase2e_waist_aim | 0.65 rad (37°) | 0.70 rad | 1.09 rad (62°) | 1.17 rad |

### Collision breakdown (from `analyze_collision.py`, 243 eval cases)

| run | drum body | under-drum | self |
|---|---|---|---|
| phase2b (baseline) | 65.7 % | 34.0 % | 0.3 % |
| phase2d_waist | 64.6 % | 35.2 % | 0.2 % |
| phase2e_waist_aim | 64.6 % | 35.0 % | 0.4 % |

Self-collision is negligible across all three runs (\< 0.5 %).  The collision
bottleneck is drum-body collisions (~65 %) during the mid-trajectory approach
(>10 waypoints before the hit).  The exemption windows around the target drum
work correctly — the `near` and `exempt` time buckets are clean.

### What happened

1. **The tight box (phase2d) worked exactly as designed** — pitch/roll movement
   dropped to p50 0.34 rad, half the URDF limit, and torque compliance hit
   1.00.  Elbow posture also reached 1.00 (elbows never flip up).  Acceptance
   improved from 0.667 → 0.728.

2. **The waist-aim IL (phase2e) traded stillness for reach** — the IL penalty
   is a soft objective competing with the terminal (strike) cost.  At
   `w_waist_aim = 500` the waist uses ~2× the pitch/roll range and ~1.4× the
   yaw range of phase2d, but acceptance jumps to **0.790** (+6.2 pp over
   phase2d, +12.3 pp over baseline).  Torque compliance and collision improved
   with the extra freedom.

3. **The IL weight is too low for the stated goal.**  500 rad⁻² means a 0.5 rad
   waist deviation costs 125 in the raw term — comparable to the terminal cost
   (100–200).  The model is willing to deviate if it improves the strike.  A
   weight of 2000–5000 would enforce stricter stillness, or the tight box from
   phase2d could be layered back on as a hard constraint alongside the soft IL.

4. **Both phase2d and phase2e outrun phase2c** (the section 7c continuation,
   best 0.741 with torque degradation after step 3000).  The waist terms are
   the difference — they are what kept the run stable past 3000 steps when
   phase2c collapsed.

### Remaining open questions

- **Does the increased waist movement in phase2e hurt closed-loop transfer?**
  The grid only scores kinematics; a 1.09 rad yaw swing may be harder for the
  PD servo to track than the 0.80 rad swing from phase2d.
- **Can a combined config (tight box + lighter IL) get the best of both?**
  The box prevents runaway and the IL shapes *where* the yaw goes, rather than
  just how far.  Likely the next thing to try.
- **The collision bottleneck (64 % drum body, far-from-hit) is unchanged**
  across all three runs.  Non-target-drum avoidance during the approach is the
  next binding constraint after waist stability.  Section 7c's `w_collision`
  sweep did not fix it; a spatial path term (explicitly penalising the hand's
  proximity to non-target drums) is the structural lever.

---

## Section 9 — `collision_peak`, the max-shaped half of the collision term

Collision is the **dominant** acceptance failure and has been for every run in
this record.  On the champion's three eval seeds it costs 50 / 49 / 54 of 243
cases against torque's 23 / 20 / 19 — roughly 2.4x — and torque is the one that
got a dedicated peak term.

### Why the mean cannot close it

`raw["collision_*"]` is a mean over `n_seen = 2 x S x P x D` = **~2 100**
(bucket 25) to **~4 200** (bucket 50) `(knot, point, drum)` pairs, the
overwhelming majority of them stick points nowhere near the drum they are being
scored against.  `ok_collision` is zero-tolerance on *any* penetration.  The
loss integrates one statistic and the gate reads another, `n_seen` apart — the
same structure section 7's `torque_peak` was added for.

Measured on the champion (`phase2b_main/best.pt`, 243-case grid, active-hand
gate):

| reduction | pass | fail | separation |
|---|---|---|---|
| mean hinge (what the cost priced) | — | — | 3.9x |
| max **hinge** (implemented, then rejected) | 0.01104 | 0.03247 | 2.9x |
| max **penetration** (what shipped) | 0.00537 | 0.02735 | **5.1x** |

**The middle row is why this section is worth reading.**  The obvious build —
mirror `torque_peak` exactly, i.e. `smooth_max` the same margin-softplus the
means reduce — was implemented, measured, and thrown away: it separates the
gate *worse than the mean it was added to fix*.  The softplus floors at
0.0066 m for an element at zero clearance and 0.0035 m at the 5 mm margin, every
clean sample pays it, and a max over a floor is mostly floor.  Reducing the raw
penetration instead is identically zero on a clear element, and it is what
`ok_collision` actually thresholds — `pen_count == 0`, no margin, no smoothing.
The margin and the smoothing belong to the mean, whose job is dense shaping on
the safe side.  The peak's job is the gate.

Residual on the PASS row: 5.37 mm, not zero, because the peak is a max over
**both hands** and a case that passes the striking-hand gate routinely has the
idle arm in a drum.  That caps the achievable separation and is the same idle-arm
problem flagged at the end of this section.

**The analogy to `torque_peak` is real but weaker, and the number says so.**
Torque violations are genuinely sparse — ~0.27 % of ~850 elements — and the
peak separated that gate 26x against the mean's 3.4x.  Collision violations run
a median of 10 penetrating elements at a median 35 mm depth, so the mean does
partially see them.  Expect a smaller win.  This is why the weight ships at
**0** and `phase2_cost_config()` does not turn it on: the term is an A/B to be
run, not a change to the committed objective.

### Two hypotheses tested and rejected

1. **Between-waypoint tunnelling.**  `n_sub = 1` evaluates collision at the
   knots only, and section 3 records a 4.6 cm worst-case gap to the scorer —
   suspiciously close to the 3.5 cm median failure depth.  Measured on the same
   plans, varying only `n_sub`: the cost already sees **158 / 243** cases
   penetrating at the knots (median −9.1 mm), and `n_sub = 4` adds only 20 more
   (median hidden depth 5.1 mm).  The training gradient barely moves —
   mean hinge 1.168e-05 at `n_sub=1` vs 1.155e-05 at `n_sub=4`.  Worth doing
   eventually for those 20 cases; it is not the bottleneck.
2. **Simply underweighting.**  Already refuted by section 7c's `w_collision`
   sweep, and the trace says why: over 15 k steps the batch-mean *minimum*
   clearance reaches **+22 mm** against a 5 mm margin while the per-sample
   *max* penetration plateaus at 5–9 mm from step 2 000 on.  Raising a
   mean-shaped price buys standoff the trajectory does not need and does not
   touch the tail — and `c_terminal` nearly doubles (35 → 65) over the same
   span paying for it.

### What shipped

`CostConfig.w_collision_peak` (0.0), `collision_peak_mode` / `_p` / `_temp`,
and a `collision_peak` augmented-Lagrangian group appended last.  The peak is a
`smooth_max` over the per-element **penetration** `max(-sdf, 0)` — both hands,
all three collision types, concatenated so one reduction sees the union rather
than a max of per-hand p-norms.  Measured p-norm overestimate at `p = 12`:
**1.0018x** the true max, so `raw["collision_peak"]` is
`diag["collision_pen_max"]` to within 0.2 %.

Cost of running it, batch 256 x horizon 50, fwd+bwd eager: **−0.2 % wall time**
(inside noise) and **+27 MB**.  At the default weight the concatenation is never
built, so `tests/test_cost_speed.py`'s 20 ms budget is untouched.

Three deliberate asymmetries against `torque_peak`, all pinned by
`tests/test_collision_peak.py`:

* **`viol["collision_peak"]` is the same tensor as `viol["collision"]`.**  That
  group's violation has always been a max, not a mean, so there is no second
  quantity to hand the new group.  The two multipliers will track the same
  signal; they are still worth splitting because the *penalties* they scale
  differ by `n_seen` and each carries its own `rho` (the peak's is 1/10 of the
  mean's, a conservative starting point and not a measured optimum).  Making
  `viol["collision"]` a mean is the more correct fix and is a separate,
  deliberately-not-bit-identical change.
* **`migrate_al_state` does not seed `lam_collision_peak`.**  Unlike
  `torque_peak` this is not a regrouping of a price already being paid, so
  `lam_init` is correct and continuity is the ramp's job.
* **It reduces the penetration, not the hinge** — see the rejected middle row
  of the table above.  `torque_peak` peaks its hinge and is right to: the
  torque hinge sits at ~0 on a feasible element, so it has no floor to
  contaminate a max.

At `w_collision_peak = 0` the smooth-max is not merely unpriced, it is not
computed: the concatenation never runs and the term is bit-identical to the
mean-only form.

### Not yet measured

The A/B itself.  Nothing in this section trains a model — the separation
numbers are on existing checkpoints and the term has never been in a loss.

Suggested start: **`--w-collision-peak 500`**, ramped over the default 1 000
steps.  Measured on the champion at `lambda = 1`, that puts the peak half at
**13.7 on a failing case and 2.7 on a passing one**, against a `terminal` of
~65 and the collision mean's ~1.0 on the same batch — i.e. the term is a real
price on the cases that fail the gate and close to free on the ones that do
not.  The full curve, `w x raw` at `lambda = 1`:

| `w` | PASS median | FAIL median |
|---:|---:|---:|
| 250 | 1.3 | 6.8 |
| **500** | **2.7** | **13.7** |
| 1000 | 5.4 | 27.4 |
| 2000 | 10.7 | 54.7 |

**The larger unexploited lever is still the idle arm**, and it is not this
term.  `ok_collision` scores the *striking hand only*: 62 of 243 cases fail it,
but **159** have a hand inside a drum once the idle arm is counted.  The cost
prices both hands at full price, so a large share of the collision gradient is
spent on violations acceptance never reads.  The `w_idle_home` work is aimed
exactly there and its effect on collision is still unmeasured.

---

## Section 8 — model-size ablations (2026-08-11–12)

Four experiments run overnight to answer: *can we shrink the decoder, do we
need the phase-1 warm start, and what happens without waist pitch/roll?*

All runs used `--horizon 50 --buckets 25 50 --no-compile --batch 512
--pool-cache /data2/eric/tmp/p2pool_apex.npz` on RTX 5090s.
`train_phase2.py` gained `--d-model`, `--dec-layers`, `--d-ctx`, `--ctx-layers`
and `--no-aux-heads` flags; when an architecture override changes the decoder
depth or width, `load_state_dict(strict=False)` loads matching layers from the
phase-1 checkpoint and randomly initialises the rest.

### The configurations

| run | decoder | total params | warmstart | other |
|---|---|---|---|---|
| `phase2b_main` | 6L d=384 (17.2M) | 19.3M | full (6/6 layers) | baseline |
| **`abl_10m_decoder`** | **3L d=384 (8.7M)** | 10.8M | partial (3/6 layers) | — |
| **`abl_5m_decoder`** | **2L d=384 (5.8M)** | 7.9M | partial (2/6 layers) | — |
| **`no_warmstart_test`** | 6L d=384 (17.2M) | 19.3M | **none** (random init) | — |
| `abl_no_waist_pr` | 6L d=384 (17.2M) | 19.3M | full | `waist_pr_box=0.03`, `w_waist_upright=20`, `w_waist_aim=0` |

**Caveat.** The 10M and 5M decoders started from **partial** warmstarts — only
the first 2–3 of 6 layers were pre-trained, the rest random.  The 19M baselines
(phase2b/d/e) had **full** warmstarts.  The partial init is a harder starting
point, so the smaller models' numbers understate their potential relative to a
full phase-1 pretraining at the smaller size.

### Inference time

Measured on `cuda:1`, batch=1, H=100, float32, eager (no compile):

| decoder | params | ms/forward | vs 19M | training it/s |
|---|---|---|---|---|
| 6L d=384 | 17.2M | 5.27 | 1.00× | 1.85 |
| **3L d=384** | **8.7M** | **3.09** | **1.71× faster** | 1.95 |
| **2L d=384** | **5.8M** | **2.34** | **2.25× faster** | 2.30 |

Scaling is nearly linear in layer count.  At the MPC's 20 ms control period all
three are well inside budget; the real win is 69–76 % fewer FLOPs during
training.

### Eval acceptance trajectories

All evals on the 81-case rolling grid.

| step | 19M (full warmstart, §7d) | 10M (3L, partial init) | 5M (2L, partial init) |
|---|---:|---:|---:|---:|
| 2k | — | 3/81 (4 %) | 5/81 (6 %) |
| 10k | — | 14/81 (17 %) | ~15/81 (19 %) |
| 20k | — | 25/81 (31 %) | ~24/81 (30 %) |
| 30k | — | 32/81 (40 %) | ~32/81 (40 %) |
| 38–40k | **73 %** (at 2.5k steps) | **51/81 (63 %)** | **47/81 (58 %)** |

The 19M numbers are from `phase2d_waist` / `phase2e_waist_aim` which used full
warmstart + stage-A gate already open — not directly comparable to the 10M/5M
runs which started from a much weaker init.  With full phase-1 pretraining at
the smaller sizes the gaps would close.

### Is the warmstart necessary?  Yes.

`no_warmstart_test` (19.3M, random init, 5 997 steps, 54 min):

| step | loss | eval |
|---:|---:|---|
| 50 | 203 884 | — |
| 450 | 7 993 | — |
| 1000 | 4 139 | **acc 0/81**, hit 0.07, coll 0.26, tau 0.00 |
| 5000 | 1 444 | acc 0/81 |
| 5997 | 1 425 | **best 0/81** |

The loss dropped 143× in 6k steps — the cost signal works from random init.
But **zero cases ever passed all gates**, and the torque gate never budged off
0.00.  The model learned the coarse shape of a trajectory but never produced a
constraint-satisfying one.  Phase 1 behaviour cloning is essential to put the
model in the basin of attraction.

### Signs of underfitting?

**10M (3L): no clear underfitting.**  Steady monotonic improvement from 4 % →
63 % over 38k steps.  Final gate profile: hit=0.98, coll=0.63, tau=1.00,
elbow=0.99 — essentially identical per-gate quality to the 19M baseline once the
torque multiplier caught up.  Was still climbing when it hit the 63 % target and
stopped.  Likely reaches 70 %+ with more steps or a full warmstart.

**5M (2L): mild underfitting.**  ~5 percentage points behind the 10M (58 % vs
63 %), and the learning curve is noisier — oscillated 35 → 29 → 47 in the last
6k steps.  The smaller latent space makes optimisation less smooth and
convergence less stable.  The model still has enough capacity to solve the task;
a full phase-1 pretraining at this size would likely close most of the gap.

The 5M crash (`ValueError: too many values to unpack` in `_bptt_term`) was a
pre-existing rollout-API mismatch (`sim.rollout` returns 3 values, caller
unpacked 2), not caused by the architecture change.  Fixed in the
`train_cost.py` call site; the run was resumed from `last.pt`.

### No waist pitch/roll

Launched on GPU 2 but **CUDA OOM** (Isaac Sim held 20.5 GB).  Relaunched on
GPU 1 (2026-08-12); results pending.

### Key takeaways

1. **Phase 1 warmstart is essential.**  Direct cost descent from random init
   never produced a single accepted trajectory despite 143× loss reduction.
2. **The 10M decoder (3L d=384) is the sweet spot.**  63 % acceptance at
   1.71× faster inference, with no sign of underfitting.  With a full warmstart
   it would likely match the 19M.  **`ModelConfig.dec_layers` therefore defaults
   to 3 as of 2026-08-12** (10.76 M total).  The default governs NEW runs only —
   the 6-layer champion and every other stored checkpoint load from their own
   `model_cfg` — and `--dec-layers 6` restores the old size.  The open item is
   that no 10M model has yet had a *full* phase-1 warmstart at its own size;
   `train_phase1.py` with the new default is that run, and it is the clean
   comparison the ablation's partial-init caveat asks for.
3. **The 5M decoder (2L d=384) is viable but noisier.**  58 % acceptance at
   2.25× faster inference, mild underfitting (slightly lower peak, less stable
   convergence).  Worth revisiting with full phase-1 pretraining.
4. **Inference scales linearly with layer count** — 2.3–5.3 ms per forward pass
   across the 5M–19M range, all well within the 20 ms control budget.
5. **Partial warmstart is a valid ablation tool.**  `strict=False` loading
   works: matching layers get pre-trained weights, new architecture dimensions
   start random.  The training loop handles the mixed initialization cleanly.

---

## Section 9 — deployment control: is the lag constant, and do the gains matter? (2026-08-12)

Two questions, one ablation each, both with the PID integral **off** (`ki = 0`)
throughout.  The short version: **the lag is emphatically not constant along the
approach, and scheduling it anyway does not buy a single strike** — while the
servo *gains*, which nobody had swept, move the miss rate more than any
compensation knob ever has.

### What was added

```
amortized_opt/
  pd_sim.py    PDSimConfig.kp_scale / kd_scale -- a DETERMINISTIC per-joint or
               scalar multiplier on the runtime Kp/Kd tables.  A plant ablation,
               applied to the nominal before sample_gains' log-normal draw, so it
               composes with the usual gain randomization instead of replacing it
  mpc_gen.py   MPCGenConfig.kp_scale / kd_scale; cmd_lead_s now also accepts
               {"schedule": (n_h, 17)} for a HORIZON-DEPENDENT lead, re-read
               every tick from the remaining horizon
  lead.py      is_schedule / schedule_fraction
  probe.py     closed_loop_probe(kp_scale=, kd_scale=)
scripts/
  fit_lead_schedule.py  fit delta* per (joint, waypoints-to-hit)
  eval_gain_sweep.py    closed-loop sweep over kp/kd x lead, ki=0
tests/test_lead.py      +4 (a CONSTANT schedule reproduces the fixed lead exactly)
```

### 1. The lag is not constant — it varies by an order of magnitude

`fit_cmd_lead.py` fits ONE delta per joint by minimizing tracking RMS over a
6-knot window around the strike.  That is only a fair summary if the lead the
servo wants is the same everywhere.  `fit_lead_schedule.py` drops the assumption:
it buckets the error by `h`, waypoints REMAINING to the hit, and fits a separate
delta per (joint, h).  162 `standard` cases, nominal gains, delta swept 0-50 ms.

| joint group | fixed `"fit"` | h = 1-2 | h = 4-7 | h = 9-11 |
| --- | ---: | ---: | ---: | ---: |
| shoulders | 0-7.5 ms | **50 ms** | 0-8 ms | 5-15 ms |
| elbows | 15 ms | 12-30 ms | 5-10 ms | 10-30 ms |
| wrists | 0-2.5 ms | 0-8 ms | 0 ms | 0-8 ms |
| waist pitch/roll | 50 ms | 50 ms | 50 ms | 0-48 ms |

Over the deployable region (h = 1-11) the schedule cuts tracking RMS
**26.5 -> 20.4 mrad (23 %)**; the best UNIFORM delta buys 2 % on the same kind of
objective (7c).  The single fitted number per joint is an average taken across
exactly the region where the demand swings hardest.

**Why it should vary, in closed form.** The standing error the servo needs to
produce the commanded acceleration is `e ~ M(q) qdd / kp`; a time lead buys a
command offset of `qd * delta`.  Equating them,

    delta*(t) ~ M(q) qdd(t) / (kp * qd(t))

— an acceleration-to-velocity RATIO, which cannot be constant over an approach
that starts from rest (`qd -> 0`, so `delta* -> inf`: this is the unbounded
80-100 ms the `"wide"` preset fit at the waist, and the censored 50 ms at h = 0)
and ends near constant velocity (`qdd -> 0`, so `delta* -> 0`).

**The h = 0 row is censored and unrealizable.**  Every group pins to the 50 ms
sweep boundary at the strike waypoint itself, and the deployed loop cannot apply
it: past a hand's terminal the reference is HELD, so the lead is structurally a
no-op there (7c).  That row is a restatement of the bandwidth limit, not a lead
that anyone can command.

### 2. ...and scheduling it anyway does not work

`{"schedule": table}` is applied per tick: row `min(h, n_h-1)` for each hand's arm
columns, and the MORE IMMINENT hand's row for the shared waist.  A constant table
reproduces the fixed lead waypoint-for-waypoint (asserted in `test_lead.py`).

4 eval clips x 3 start seeds, 225 scheduled strikes, `ki = 0`:

| config | crossed / 225 | miss % | \|lat\| p50 | \|lat\| p90 | per-seed |
| --- | ---: | ---: | ---: | ---: | --- |
| lead off | 49 | 78.2 | 13.8 | 30.0 | 18 / 15 / 16 |
| **fixed `"fit"`** | **84** | **62.7** | **8.2** | **20.1** | 25 / 30 / 29 |
| schedule | 82 | 63.6 | 10.8 | 31.1 | **21 / 18 / 43** |

**The schedule ties the fixed preset and is far noisier.**  A single-clip probe
had said 21/23 against the preset's 15/23 — clip 0 alone is optimistic and that
number is withdrawn.  What survives is that any lead beats none (49 -> 84).

Two reasons it plausibly fails, neither measured:

1. **The schedule steps discontinuously.**  The shoulders go 50 ms -> 0 between
   h = 2 and h = 3, a ~2.5-waypoint jump in what the command reads — a step
   disturbance injected two ticks before the strike.  A constant lead cannot do
   that.  The blown p90 (81.5 ms on seed 0) and the 18-to-43 spread are what that
   would look like.  Slew-limiting delta itself is the obvious fix.
2. **The identification is local, and the objective is global.**  Each roll
   commands ONE constant delta and the error is read off at each h, so the
   per-bucket argmin is contaminated by the earlier commands' lead.  A real
   schedule fit would optimize the whole table against the closed-loop metric.

**Decision: the fixed per-joint lead ships, unchanged.**  The schedule machinery
stays in the tree, defaulted off, on the same terms as `freeze_wp` and
`hit_lead_s`: implemented, tested, and only worth layering against a measured
residual.

### 3. The gain ablation — damping is the lever, not stiffness

2 clips x 2 seeds, 98 strikes per cell, `ki = 0`, `kp_scale` / `kd_scale`
multiplying the runtime tables (nominal = arms Kp 40 / Kd 2.0, waist Kp 200 /
Kd 10.0).  `spd_ok` counts strikes whose impact speed lands in the commanded
band — the metric that decides whether a crossing is musically usable.

| kp x | kd x | lead | crossed/98 | miss % | \|lat\| p50 | spd_ok | trk p50 | div |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.5 | fit | 7/79 | 91.1 | 17.3 | 2 | 16.9 | 1 |
| 0.5 | 1 | fit | 19 | 80.6 | 14.9 | 8 | 10.0 | 0 |
| 0.5 | 2 | fit | 49 | 50.0 | 8.7 | 46 | 6.5 | 0 |
| 1 | 0.5 | fit | 20/78 | 74.4 | 15.9 | 13 | 15.4 | 1 |
| **1** | **1** | **fit** *(shipped)* | **36** | **63.3** | **4.0** | **21** | **7.9** | **0** |
| 1 | 2 | fit | 60 | 38.8 | 10.8 | 50 | 6.0 | 0 |
| 2 | 0.5 | fit | 39/76 | 48.7 | 8.4 | 31 | 13.4 | 1 |
| 2 | 1 | fit | 51 | 48.0 | 8.9 | 46 | 7.7 | 0 |
| **2** | **2** | fit | **62** | **36.7** | 14.9 | **55** | 5.6 | 0 |
| 4 | 0.5 | fit | 52/76 | 31.6 | 17.6 | 8 | 12.6 | 1 |
| **4** | **1** | fit | **75** | **23.5** | 21.8 | 13 | 7.8 | 0 |
| 4 | 2 | fit | 60 | 38.8 | 22.7 | 33 | 5.8 | 0 |

(The `lead=off` half of every cell is in `runs/pid_sweep/gain_sweep.json`.  The
shipped cell's 63.3 % miss reproduces the report's headline 63 %.)

Five things it says:

1. **`kd x2` at the nominal stiffness nearly doubles crossings, 36 -> 60, and
   more than doubles usable strikes, 21 -> 50.**  It is the cheapest change on
   the table and it is a config edit, not a training run.  The mechanism is
   direct: the plant's torque is `kp*e + kd*(ff*qd_cmd - qd)` with
   `ff_scale = 1.0`, so **kd multiplies the velocity feed-forward**.  Doubling it
   doubles the feed-forward torque authority at the strike — which is exactly the
   *attenuation* term a lead cannot touch and an integral cannot either.
2. **`kp x4 / kd x1` maximizes crossings (75/98) but WRECKS impact speed**
   (spd_ok 13 vs the shipped 21 and `kd x2`'s 50).  Stiffness pulls the tip onto
   the disk without giving it the commanded velocity: a louder-scoring metric and
   a quieter drum.  On usable strikes the ranking is `kp x2 / kd x2` (55) >
   `kp x1 / kd x2` (50) > `kp x2 / kd x1` (46) >> `kp x4 / kd x1` (13).
3. **`kd x0.5` is unusable** — it is the only setting that DIVERGES, in all four
   of its lead rows.  Under-damping plus a reference lead is an instability.
4. **The fixed lead helps in all 12 cells**, despite having been fitted at
   `kp_scale = 1` and therefore being mis-specified in 11 of them.  That is the
   strongest evidence in this chapter that the fixed lead is the robust choice.
5. **Tracking error falls monotonically with kd** (16.9 -> 6.5 mrad at kp x0.5;
   15.4 -> 6.0 at kp x1), which is the same statement as (1) from the servo's
   side.

**The `kp x4` row is the least trustworthy number here and must not be shipped on
this evidence.**  `simulate_pd` has **no torque saturation** (`acc = Minv @ (kp*e
+ ki*integral + kd*(...))`, unclamped) and no gravity.  Quadrupling kp quadruples
the torque demanded for the same error, and Isaac clamps at 25 N.m on the arms /
5 N.m on the wrists; the dynamics-lead experiment already measured *computed*
torque at 165-190 % of the rating on 5-11 % of steps at the NOMINAL gains.  The
`kd` result is the trustworthy half of this table — velocity feed-forward is
cheap in torque and its signature (impact speed entering the band) is directly
physical.  **Adding an effort clamp to `simulate_pd` is the prerequisite for
believing any high-stiffness row**, and it is one `torch.clamp` against the
effort table.

### 4. Deployment defaults, changed

`play_isaac.py` shipped `--cmd_lead_s 0.02 --ki 5.0` — a scalar lead plus an
integral, which is neither the 7c recommendation nor the PID chapter's own winner
(`ki = 2`).  It now defaults to **`--cmd_lead_s fit --ki 0.0`**, matching 7c
decision 2, the 225-strike measurement above, and `isaac_rl/env_cfg.py`, which
already defaulted to `"fit"`.  The PID chapter's numbers stand as measured but
are superseded as a recommendation: they are 23 strikes from one clip, and the
plant they were swept in cannot saturate, which is precisely where an integral
fails.

### 5. `BatchedPDSim.step` / `.rollout` grew a third return value; callers did not

Three call sites were still unpacking two, all found today, all silent until
executed:

* `scripts/fit_cmd_lead.py` — **the script the README tells you to re-run before
  any Isaac deployment**, and it raised on line 1 of its sweep;
* `tests/test_roller_regoal.py` — its monkeypatch spy passed only `(q, v)` back
  to the roller, breaking all 5 roller tests (the roller itself was already
  correct);
* `train_cost.py`'s BPTT call site — found earlier, during the 5M ablation crash.

All three are fixed.  The lesson is worth recording: `step`/`rollout` returning
an optional third value that most callers discard is an API that drifts silently.

### Section 9 decisions needing review

1. **The fixed per-joint lead remains the deployment config** and is now the
   default in `play_isaac.py`.  The horizon schedule is measured, implemented,
   tested and OFF.
2. **`kd x2` is the most promising untaken change in the project right now** —
   +24 crossings and +29 usable strikes out of 98, from an actuator config edit.
   It has NOT been tried in Isaac, and the offline plant that produced it cannot
   saturate.  Verify against the real actuator model before believing it.
3. **The offline plant has no torque limit and no gravity.**  Every conclusion in
   this chapter and in the PID chapter inherits that.  The effort clamp is the
   single highest-value fix to the plant model.
4. **The lead was not re-fitted per gain cell.**  Every off-nominal row runs a
   lead fitted at `kp_scale = 1`, which biases the sweep AGAINST the off-nominal
   gains.  `fit_cmd_lead.py` is ~15 s; re-fitting would strengthen (2), not
   weaken it.
5. **`Phase2Config.cmd_lead_s` does not accept a schedule.**  If a schedule is
   ever revived it must be applied in training too, or it is the exact
   train/deploy lead mismatch 7c warns about.
