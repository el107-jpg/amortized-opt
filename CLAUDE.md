Amortized trajectory optimizer for a two-stick **Unitree G1** drummer: a learned
warm-start + cost model that replaces an online SQP, trained and evaluated
against a batched analytic model of the plant (FK, RNEA, SDF, PD/FOH sim).

Current-iteration index (input space, train recipe, scripts, live run): **`quick.md`**.

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

## One file still needs the original repo

A benchmarking tool, not part of the system; it fails with a clear message
rather than a traceback when the repo is absent:

* `scripts/repo_adapters/trajopt_hpipm_planner.py` — the trajopt SQP baseline
  the amortized optimizer is measured against.

Point it at a checkout with `--add-dir` / `REPO_ROOT` if you want it.
(`scripts/play_isaac.py` used to be the second such file, but the `g1drum`
conda env from the preamble ships `g1_drummer` pip-installed from the
checkout, so it runs there as-is.)

## Gotchas

* **The composed G1-with-sticks USD is a cache**, rebuilt under
  `assets/robots/_generated/` whenever `g1_scene/g1.py` or the upstream USD is
  newer. Redirect it with `$AMORTIZED_USD_CACHE` if the checkout is read-only.
* **Stick geometry has exactly one source**: `amortized_opt/constants.py`.
  `g1_scene/g1.py` imports it to author the PhysX colliders, so the simulated
  stick and the tip the cost function scores cannot drift.
  `tests/test_g1_scene.py` pins them to machine epsilon.
* **Resume LR is absolute step**, not “fresh cosine from this job.”
  `Phase2Config.lr_at(step)` decays against `--lr-horizon` using the
  checkpoint’s step. A 59k resume with `LR_HORIZON=60000` is already at the
  ~5e-6 floor. Encoder-only terms (gated P/R distill, hold-band hinge `hB`)
  cannot walk the head there — `pr` stays glued and `hB` plateaus even when
  the loss is on. **When adding or changing those terms, reset the LR
  schedule** (raise `--lr-horizon` so the cosine is not at the floor). Do not
  reuse the parent run’s leftover decay. `hB` in the train line is mean
  `relu(|sp − clamp(q0)| − band)` mrad; the encoder still does not see `q0`.

## Testing

```bash
pytest -q                      # repo-dependent tests SKIP, they do not fail
pytest -q tests/test_standalone.py tests/test_g1_scene.py
```

When you run your things in shell, clean up the shell after you are done. Do not leave dead things running and taking up memory and compute