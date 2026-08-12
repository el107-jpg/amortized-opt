# Extracting `amortized/` into its own repo

Audit + vendoring done 2026-08-12, verified against this checkout.

## Status: ready to move

Everything the system needs now lives in this folder. Verified, not assumed:

* **Core** — `amortized_opt` runs with `UNITREE_MODEL_DIR` unset and `PYTHONPATH`
  empty, no `g1_drummer` module in `sys.modules`, URDF resolving from `assets/`.
  `tests/test_standalone.py` enforces it statically and in a clean subprocess.
* **Isaac layer** — `g1_scene/` (new) vendors what `isaac_rl*/env.py` used to
  import from the repo. Exercised under a live headless Kit app: it composes
  `g1_with_sticks_frozen_legs.usd` (3 085 bytes, byte-size identical to the
  repo's), yields the three expected actuator groups (`N7520-14.3` 88 N·m,
  `N5020-16` 25 N·m, `W4010-25` 5 N·m) as `IdealPDActuatorCfg`, with
  `fix_root_link` and self-collisions on and no leg patterns surviving the
  `freeze_legs` prune.
* **Packaging** — `pyproject.toml` (editable metadata builds; `pip install -e .`
  dry-run passes) and `CLAUDE.md` with the environment preamble.

Two files are deliberately left repo-only; see the last section.

## What was vendored

| new file | replaces | note |
|---|---|---|
| `g1_scene/g1.py` | `g1_drummer/assets/g1.py` (772 lines) | geometry now *imported* from `amortized_opt.constants` |
| `g1_scene/layout.py` | `DRUM_KIT_USD_PATH`/`DRUM_LAYOUT_PATH` + `DrumLayout` (105 lines) | only the id-contiguity check was actually used |
| `assets/drums/simple_drum_kit_6drums.usd` | repo `assets/drums/` | 4 KB, byte-identical |

Three deliberate changes from the repo originals:

1. **One source for stick geometry.** The repo restated `STICK_LENGTH`,
   `STICK_RADIUS`, `TIP_RADIUS`, the mount pose and `TORSO_CAPSULE`/`HEAD_SPHERE`
   in `assets/g1.py`, keeping them in sync with `amortized_opt.constants` by
   hand. `g1_scene/g1.py` imports them, so the colliders PhysX simulates and the
   tip the cost function scores are built from the same numbers.

   This surfaced a real (if tiny) defect. `constants` stores the mount
   quaternion rounded to 6 dp, which leaves `|q| = 0.9999995`.
   `amortized_opt.fk` re-normalizes before building its rotation matrix; the
   repo's `_quat_rotate` applies the raw Hamilton formula, which scales the
   result by `|q|²`. Imported naively, the collider tip and the analytic tip
   landed **2.1e-7 m apart**. Normalizing once on the way in (`_unit_quat`,
   preserved from the repo for exactly this reason) takes the gap to 3e-18.
   `tests/test_g1_scene.py` pins tip and all four shaft probes, both hands.

2. **Isaac imports deferred.** `import isaaclab.sim` and `unitree_rl_lab` moved
   off module scope into the functions that need them, matching
   `isaac_rl_ppo.env._isaac_imports`. The geometry half of the module is now
   importable — and testable — with no Kit app running.

3. **Composed-USD cache** moved to `assets/robots/_generated/` (gitignored,
   redirectable with `$AMORTIZED_USD_CACHE`) instead of the repo root.

Also fixed: `scripts/pilot_phase2.py` no longer hardcodes
`/data2/eric/tmp/p2pool_pilot.npz` — it honours `$TMPDIR`.

## Known remaining issues (not blockers)

* **`isaac_rl/` and `isaac_rl_ppo/` are diverged forks.** `mpc.py` and
  `env_cfg.py` are identical; `env.py`, `loop.py`, `reward.py`, `measure.py`,
  `schedule.py` and `perturb.py` all differ, and `isaac_rl_ppo` imports
  `isaac_rl`. Both were updated to use `g1_scene`. Pick a canonical one before
  the move or you carry both forever.
* **`runs/` is 12 GB** and gitignored, so no git-based extraction carries it.
  `scripts/play_isaac.py` and `isaac_rl/tests/test_env_isaac.py` default to
  `runs/phase2b_main/best.pt`.
* **`tests/conftest.py` pins CUDA to devices 1/3** ("held by concurrent sessions
  on this box"). Override with `AMORTIZED_TEST_DEVICE`.
* **No `LICENSE`.** Not something to pick on your behalf; add one before the
  folder goes anywhere public.

## Move procedure

148 tracked files (25 MB) plus **~80 untracked** ones — all of `isaac_rl/`,
`isaac_rl_ppo/`, `g1_scene/`, `REPORT.md`, and three `amortized_opt` modules.
`git subtree split` only sees committed files, so commit first.

### Option A — keep git history (recommended)

```bash
cd /data2/eric/g1-drummer-lab-rythmic

# 1. get the untracked work into history first
git add amortized && git commit -m "amortized: vendor Isaac scene assets, add packaging"

# 2. split the subtree onto its own branch (rewrites paths, keeps history)
git subtree split --prefix=amortized -b amortized-only

# 3. new repo
mkdir -p /data2/eric/amortized-opt && cd /data2/eric/amortized-opt
git init -b main
git pull /data2/eric/g1-drummer-lab-rythmic amortized-only

# 4. carry the gitignored artifacts across separately (12 GB — prune first)
rsync -a --info=progress2 \
  /data2/eric/g1-drummer-lab-rythmic/amortized/runs/ \
  /data2/eric/amortized-opt/runs/
```

Step 1 commits into a branch that also holds unrelated uncommitted work (9
modified files outside `amortized/`, in `docs/`, `scripts/` and `source/`) —
`git add amortized` scopes the commit to this folder only, so the rest stays
dirty. `git subtree` is present in this git install; `git-filter-repo` is not.

### Option B — clean copy, no history

```bash
rsync -a --exclude __pycache__ --exclude .pytest_cache --exclude runs/ \
  /data2/eric/g1-drummer-lab-rythmic/amortized/ /data2/eric/amortized-opt/
cd /data2/eric/amortized-opt && git init -b main && git add -A && git commit -m "initial"
```

### Verify after the move

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export UNITREE_MODEL_DIR=/data2/eric/workspace/unitree_model
export TMPDIR=/data2/eric/tmp

pip install -e .                                   # editable only, see pyproject
pytest -q tests/test_standalone.py tests/test_g1_scene.py
pytest -q                                          # repo-only tests SKIP
```

`test_standalone.py` will fail loudly if anything outside the two known
exceptions starts importing `g1_drummer` again.

## Giving Claude Code access to the new folder

Working directory is the access boundary:

```bash
cd /data2/eric/amortized-opt && claude
```

`CLAUDE.md` already exists at this folder's root and moves with it — it carries
the conda/`setup_conda_env.sh`/`UNITREE_MODEL_DIR` preamble (without which
nothing runs), the layout table, the one-way `amortized_opt` ← `isaac_rl*`
dependency rule, and the gotchas worth knowing (`simplified_cost` zeroing
weights, the GPU-0 VRAM crash, the USD cache).

To still reach the old repo — for the trajopt baseline, `play_isaac.py`, or old
runs — any of:

* `claude --add-dir /data2/eric/g1-drummer-lab-rythmic`
* `/add-dir /data2/eric/g1-drummer-lab-rythmic` mid-session
* persist it in `.claude/settings.json`:

```json
{ "permissions": { "additionalDirectories": ["/data2/eric/g1-drummer-lab-rythmic"] } }
```

The parent repo has no `.claude/settings.json`, so there is no project allowlist
to carry over — your user-level `~/.claude/settings.json` applies automatically.

## The two files that stay repo-only

Neither is part of the system; both now fail with an explanation rather than a
traceback, and `tests/test_standalone.py` holds the list so it cannot grow by
accident.

* `scripts/repo_adapters/trajopt_hpipm_planner.py` — the trajopt SQP baseline
  the amortized optimizer is benchmarked against.
* `scripts/play_isaac.py` — plays into the repo's registered task
  `Isaac-G1-Drum-Direct-v0`, reading ~25 attributes off `G1DrummerEnv`
  (hit detector, beat telemetry, waist pin), several of them private. That env
  is 6 374 lines plus a 3 333-line cfg; porting it is not worth it when
  `isaac_rl_ppo/env.py` already carries the plant the optimizer was identified
  against. Use `isaac_rl_ppo/scripts/` for repo-free playback, or run it with
  `PYTHONPATH=<repo>/source/g1_drummer`.
