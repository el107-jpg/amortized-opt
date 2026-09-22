# Trajectory Teaching Studio

## What it does

This localhost app creates full trajectory-level imitation targets without
using the actor as the teacher:

1. Sample a production-format random start state, randomized drum kit, two
   goal drums, and two time-to-hit values.
2. Feed the existing analytic quintic to the current amortized actor as its
   normal hidden initialization input.
3. Display the actor's resulting `50 x 17` proposal, then refine that waypoint
   tensor directly through the differentiable `FusedCost` physics objective.
4. Inspect every waypoint in an orbitable 3D view. Add hard or soft partial
   joint keyframes at any active waypoint and re-run the direct solve.
5. Save the complete optimized tensor as an IL teacher.
6. Run a quick, isolated decoder-LoRA fit and compare Base, Teacher, and
   Adapted trajectories on exactly the same frozen scenario.

The actor supplies the initial editable proposal but is never differentiated
through by the direct teacher solve. The actor is loaded again by the explicit
IL job only after a feasible teacher has been saved.

## Run it

Use the environment with the working Pinocchio installation:

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
SESSION=my_trajectory_session PORT=6010 scripts/run_strike_pose_editor.sh
```

Open `http://localhost:6010`. The default actor checkpoint is
`runs/rhythm_timing_direct_admission_gpu1_20k_20260918/step_020000.pt`; set
`CHECKPOINT=/absolute/path/to/step.pt` before launching to choose another one.

The server stays in the foreground and stops with Ctrl-C. Session data is
written only below `runs/strike_pose_editor/<session>/`:

- `teachers/<scenario-id>/teacher.pt`: exact tensor-only actor inputs and target
- `teachers/<scenario-id>/metadata.json`: teaching audit metadata
- `jobs/<job-id>/adapter.pt`: decoder-only LoRA adapter
- `jobs/<job-id>/metrics.jsonl`: per-step IL losses
- `jobs/<job-id>/predictions.pt`: Base/Teacher/Adapted comparison tensors
- `jobs/<job-id>/status.json` and `trainer.log`: job state and diagnostics

## Teaching workflow

- **Sample random scenario** generates a dual-hand row with independent target
  drums and time-to-hit values, runs the current actor once, and displays that
  actor proposal. The analytic quintic is not shown as a trajectory. The
  display always has 50 rows; inactive padded rows after the later hit are not
  actor supervision.
- Use the timeline or Play button to inspect the full path. Drag to orbit and
  use the mouse wheel to zoom.
- At the selected hand's hit, edit contact, stick direction, and waist, then
  solve IK. **Optimize** automatically persists that displayed task-space
  target and seeds the terminal row from the IK result; it is not necessary to
  convert the target into a joint keyframe.
- Optional joint keyframes provide additional posture guidance. Soft keyframes
  add a weighted differentiable objective; hard keyframes are projected exactly
  after every optimizer update.
- **Optimize** runs 60 direct Adam/augmented-Lagrangian refinement steps from
  the actor proposal; **Reoptimize** runs another 30 from the current best
  trajectory. The actor stays frozen and outside this objective.
- **Save teacher trajectory** is enabled only after the refined path passes the
  displayed terminal, impact-speed, collision, torque, velocity, joint-box,
  and signed approach-corridor checks; it then snapshots the path and context.
- **Quick IL train** launches a separate CPU child process for 100, 500, or
  1000 steps. The base checkpoint remains read-only. When the job completes,
  the app overlays the original actor and adapted actor outputs against the
  teacher on the same scenario.

## IL objective

The quick adaptation path freezes the full phase-2 actor and inserts rank-4,
alpha-8 LoRA matrices only in decoder `o_proj`, `w12`, and `w3` layers. It uses:

- masked, joint-range-normalized Huber loss against the optimized teacher;
- a small base-output distillation loss on the same exact scenarios;
- explicit gradient clipping;
- adapter-only output with a hash of the immutable source checkpoint.

This small in-app experiment is meant to show whether the correction moves the
model in the desired direction. A later durable training run should mix saved
teachers with broad ordinary actor rows to measure retention and generalization.

## Isolation and API

The browser ships as local HTML/CSS/JavaScript with no CDN or build step. The
server binds only to `127.0.0.1`, uses one CPU thread, requires a per-session
token plus same-origin checks for mutations, and accepts no filesystem path or
shell command from HTTP clients. The launcher hides CUDA, so both direct
optimization and quick IL use CPU and cannot interfere with GPU training runs.

Trajectory endpoints are:

- `POST /api/scenarios/sample`
- `GET /api/scenarios/{id}`
- `POST /api/scenarios/{id}/keyframes`
- `POST /api/scenarios/{id}/optimize`
- `POST /api/scenarios/{id}/save`
- `POST /api/train`
- `GET /api/jobs/{id}`

The older terminal-pose endpoints remain available for compatibility, but the
browser workflow no longer uses them.

## Verification

Focused tests cover deterministic production scenario sampling, exact hard
keyframe projection, direct cost reduction, API sample/keyframe/optimize/save,
session-path isolation, safe production-checkpoint loading, adapter-only
training, Base/Teacher/Adapted inference, static resource isolation, and a real
loopback server smoke test.
