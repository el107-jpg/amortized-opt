# Development snapshot: 2026-10-20

This snapshot saves the accumulated drummer development under the private
`el107-jpg/amortized-opt` repository. It retains the existing Git history and
the original `blafy02/drummer-pinn` remote. The authenticated account has read
access to that upstream and write access to the new private repository.

The saved changes include:

- Differentiable collision geometry v2, continuous capsule contact v5, terminal
  pose costs, trajectory surrogates and their shared geometry conventions.
- Phase1/phase2 training, warm starts, critic/Bellman support, line-search and
  resume changes, plus the related simulator/playback adapters.
- Hand-assignment search, transition calibration, timing tables, neural hand-plan
  ranking/value models, data collection, diagnostics and experiment reports.
- Shared robot/drum assets, project instructions, package configuration, scripts
  and regression tests already present in the working tree.

Archived and corrected runtimes remain distinct. Follow `AGENTS.md` for contact,
geometry and dataset compatibility; this snapshot does not merge their labels
or restart any experiment. The separate point-cloud planning code and completed
learning-rate ablation results live in `el107-jpg/pointcloud-motion-planning`.

Existing ignore rules keep `runs/`, training checkpoints/logs under that
directory, generated robot USD caches and environment/session caches local.
Previously tracked small analysis assets remain part of the Git history.

Validation before saving:

- Changed-file syntax/configuration checks passed:294 Python files,2 TOML files,
  1 JSON file,1 YAML file and2 shell scripts. Git whitespace checks passed.
- The focused regression run used the configured `g1drum` environment, one
  PyTorch CPU thread and CUDA device0:176 passed,3 failed,1 skipped.
- A broader suite attempt did not complete after its multiprocessing workers
  stalled; the verification processes were stopped and cleaned up. The full
  suite is not certified passing by this snapshot.

Known focused-test failures:

1. `tests/test_assignment_close.py::test_direct_pose_controller_does_not_interrupt_current_hit_for_a_future_assignment`:
   the trajectory surrogate's collision context lacks `drum_axes`, which
   `FusedCost._collision` reads.
2. `tests/test_trajectory_surrogate.py::test_analytic_schedule_collision_context_works_without_top_entry_override`:
   the same missing collision-context attribute.
3. `tests/test_terminal_pose.py::test_optnet_gpu_coordinate_equalities_match_cvxpy`:
   `solve_feasible_qp` raises `IndexError` concatenating one-dimensional bounds
   along dimension1.

Detailed local check logs are in `runs/git_sync_checks/`. Runtime source was
preserved while saving; these failures remain recorded follow-up work.
