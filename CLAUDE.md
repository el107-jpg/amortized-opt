Amortized trajectory optimizer for a two-stick **Unitree G1** drummer: a learned
warm-start + cost model that replaces an online SQP, trained and evaluated
against a batched analytic model of the plant (FK, RNEA, SDF, PD/FOH sim).

This folder is **standalone**. It was extracted from `g1-drummer-lab-rythmic`
and carries its own URDF, meshes, drum layout and rhythm clips under `assets/`.

## Environment — run this before anything

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export UNITREE_MODEL_DIR=/data2/eric/workspace/unitree_model
export TMPDIR=/data2/eric/tmp
```

The core (`amortized_opt`) needs none of this — it is pure torch/numpy and runs
on a bare `pip install -e .`. The preamble is for the Isaac layer: Isaac Sim,
Isaac Lab, `unitree_rl_lab`, and the upstream Unitree G1 USD that
`$UNITREE_MODEL_DIR` points at.

## Layout

| path | what | may import Isaac? |
|---|---|---|
| `amortized_opt/` | the core: FK, RNEA, SDF, cost, planner, MPC, tokenizer, warmstart, training | **no — enforced** |
| `g1_scene/` | the G1-with-sticks articulation cfg + drum-kit/layout paths | yes |
| `isaac_rl_ppo/` | the current Isaac RL env + PPO loop | yes |
| `isaac_rl/` | the older RL tree (PSN/safety experiments) | yes |
| `assets/` | URDF, meshes, drum layout, goal poses, RCC clips | — |
| `scripts/` | training, eval, ablation and report entry points | some |
| `runs/` | checkpoints and logs (gitignored) | — |

The dependency arrow runs one way: `g1_scene` and `isaac_rl*` import geometry
constants from `amortized_opt`, never the reverse. `tests/test_standalone.py`
enforces this statically and in a clean subprocess — if you add an
`import isaaclab` anywhere under `amortized_opt/`, that test fails.

## Two files still need the original repo

Both are benchmarking/playback tools, not part of the system, and both fail with
a clear message rather than a traceback when the repo is absent:

* `scripts/repo_adapters/trajopt_hpipm_planner.py` — the trajopt SQP baseline
  the amortized optimizer is measured against.
* `scripts/play_isaac.py` — drives the *repo's* `G1DrummerEnv`
  (`Isaac-G1-Drum-Direct-v0`), a 13k-line task with its own hit detector and
  reward. The standalone env is `isaac_rl_ppo/env.py`; use its own scripts
  (`isaac_rl_ppo/scripts/smoke_rl.py`) for playback that does not need the repo.

Point them at a checkout with `--add-dir` / `REPO_ROOT` if you want them.

## Gotchas

* **`simplified_cost` is the default** and `validate()` silently zeroes ~12
  weights, including the impact-speed floor — the declared defaults are not what
  the solver sees. Check `amortized_opt/train_cost.py` before trusting a weight.
* **GPUs 0 and 2 are often held** by other sessions on this box; the test
  conftest picks 1 or 3. Override with `AMORTIZED_TEST_DEVICE=cuda:3` or `cpu`.
  A "NoneType has no attribute create_articulation_view" crash in Isaac is
  really VRAM exhaustion on *physical* GPU 0 — use `CUDA_VISIBLE_DEVICES`,
  `--device` does not move the allocation.
* **The composed G1-with-sticks USD is a cache**, rebuilt under
  `assets/robots/_generated/` whenever `g1_scene/g1.py` or the upstream USD is
  newer. Redirect it with `$AMORTIZED_USD_CACHE` if the checkout is read-only.
* **Stick geometry has exactly one source**: `amortized_opt/constants.py`.
  `g1_scene/g1.py` imports it to author the PhysX colliders, so the simulated
  stick and the tip the cost function scores cannot drift.
  `tests/test_g1_scene.py` pins them to machine epsilon.

## Testing

```bash
pytest -q                      # repo-dependent tests SKIP, they do not fail
pytest -q tests/test_standalone.py tests/test_g1_scene.py
```
