# Agent index — current iteration

This is a **map**, not a dump. Read the files it points at. Do not treat
`README.md` / `REPORT.md` champion numbers as the live system — they describe
an earlier phase-2b checkpoint.

Env, layout, Isaac isolation, USD cache: **`CLAUDE.md`**. Source that conda
preamble before GPU work; do not leave processes on the GPU.

---

## Live recipe and run

| what | where |
|---|---|
| The script that defines *this* iteration | `scripts/_run_rhythm_bellman.sh` — voxel curriculum by default, current-goal-only context, direct decoder torso with no encoder SP heads, 50/30/15/5 MPC/pool/boundary/probe mix |
| Latest stability A/Bs | `runs/diagnostic_clean_uniform_fp32_300/`, `diagnostic_clean_softdw_fp32_300/`, `diagnostic_clean_downweight_fp32_300/`, and the `diagnostic_clean_dw_gamma{05,10}_fp32_300/` floor arms |
| Optimizer defaults | SaLSa with raw-objective directional derivatives, strict actual-step validation, rejection instead of floor commits, AL-change reset; gradient surgery remains off |
| Legacy gradient balancing / compile details | `docs/cost_curriculum_and_grad_balance.md` — `al_pcgrad_output`, `gb_output_space`, compile environment and throughput |
| How to launch it | header comment of that script. Default tag: `rhythm_bellman_nf1_direct` |
| Frontier collocation (default **off**) | `FRONTIER=1` on the wrapper after `--resume` last.pt; optional `FULL_WEIGHTS=1` to snap suffix warmup too |
| Phase-2 CLI (every knob) | `scripts/train_phase2.py` `parse_args` |
| Phase-2 loop / mix / AL / eval | `amortized_opt/train_cost.py` — `Phase2Config`, `WeightSchedule`, `train_phase2` |
| Phase-1 (warm start) CLI | `scripts/train_phase1_apex.py` |
| Phase-1 loss (identity Jacobian, not follow-frac) | `amortized_opt/train_warmstart.py` |
| Checkpoints / log | `runs/<tag>/{last,best,best_probe}.pt`, `launch.log`, `metrics.jsonl` |
| Pool cache (must match kit DR + tau in the name) | `runs/p2pool_*.npz`; built by the same wrapper or inline in `train_phase2` |

Do not start from `scripts/_run_ablation.sh` — it still points at `amortized/scripts/` (pre-extract path).

---

## Input space (what the encoder sees)

| what | where |
|---|---|
| Tensor inventory + shapes + idle / stick / waist fields | `amortized_opt/tokenizer.py` — `ContextInputs` |
| How a rhythm snapshot becomes those tensors | `amortized_opt/rhythm.py` — `build_context_inputs` (env-frame goals; pelvis-frame `aim_yaw`) |
| Token types / packing | `amortized_opt/model.py` — `TYPE_*`, `ContextEncoder` |
| Decoder waypoint tokens + plant feats | `amortized_opt/model.py` — `TrajectoryDecoder`, `ModelConfig` (`dec_rope`, `dec_plant_feats`) |
| Analytic waist yaw (gate, `az_pref`, λ, futures) | `amortized_opt/waist_aim.py` module docstring + `WaistAimConfig` |
| Learned yaw / P/R set-point heads | `docs/waist_sp_head_plan.md`, `docs/waist_pr_sp_head_plan.md`; impl in `ContextEncoder` |
| Joint order, dt, stick geom (single source) | `amortized_opt/constants.py` |
| Kit layout / radii / heights | `assets/drum_kit_layout.yaml` |
| Idle-arm cylinder | `amortized_opt/idle_cylinder.py`; library `assets/idle_cylinder_poses.json` |

`future_gap` is **hit-relative**, not time-to-hit from now — see the `ContextInputs.future_gap` comment. Closed-loop re-encode rules: `isaac_rl/README.md` DECISION 2 / `isaac_rl/INTERFACES.md`.

---

## Cost, gates, plant

| what | where |
|---|---|
| Term inventory (collision, torque, upright, waist_aim, …) | `amortized_opt/cost.py` header + `CostConfig` + `FusedCost` |
| AL groups / multipliers | `amortized_opt/lagrangian.py` — `CONSTRAINT_GROUPS`, `ALConfig` |
| `ok_hit` / `ok_collision` / acceptance | `amortized_opt/eval_harness.py` — `TrajectoryScorer`, `AcceptanceThresholds` |
| 81-grid construction (box starts, extra drums, z-floor) | `eval_harness.generate_grid`; subsample `train_cost.eval_subset` |
| Hit region (shaft on cymbals, bead on pads) | `amortized_opt/strike.py` |
| FK / RNEA / SDF | `fk.py`, `rnea.py`, `sdf.py` |
| PD / FOH sim (BPTT, roller) | `amortized_opt/pd_sim.py`, `train_cost.MPCRoller` |

Rolling eval during train does **not** pass a custom `WaistAimConfig` — default `future_tau=1.0` matches the wrapper’s `AIM_TAU`.

---

## Scripts you actually run

**Train**

```bash
# this iteration (voxel mode is the wrapper default)
DEV=cuda:1 VOXEL_POOL=1 SKIP_P1=1 INIT=runs/phase1_waist_gain/best.pt \
  bash scripts/_run_rhythm_bellman.sh

# small integration smoke: reduce pool/bootstrap/update sizes, keep LS off
DEV=cpu VOXEL_POOL=1 LINE_SEARCH=0 SKIP_P1=1 \
  INIT=runs/diagnostic_safe_uniform_300/last.pt \
  POOL=runs/p2pool_diagnostic_safe_uniform_300_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \
  VOXEL_BOOTSTRAP_ROWS=300 VOXEL_BOOTSTRAP_STRIDE=25 STEPS=1 BATCH=8 \
  COMPILE_MODEL=0 EVAL_EVERY=1 CKPT_EVERY=1 \
  bash scripts/_run_rhythm_bellman.sh

# phase 1 only
python scripts/train_phase1_apex.py --help
```

**Eval / probe / collisions**

| script | use |
|---|---|
| `scripts/diagnose_loss_spike.py` | fixed-batch checkpoint deltas under own and frozen-common objectives |
| `scripts/probe_row_grad_stratify.py` | vector-summed replay torque-gradient attribution, cancellation and tail exclusion |
| `scripts/probe_replay_boundary.py` | shifted-seed/current-decode boundary and long-age roller diagnosis |
| `scripts/audit_full_rnea_numerics.py` | fp32/fp64 RNEA, finite-difference and AMP-boundary audit |
| `scripts/plot_collision_traj.py` | 81-grid collide/miss plots + `summary.json` (the collision viz) |
| `scripts/analyze_collision.py` | FusedCost-side collision buckets |
| `scripts/eval_phase2.py` | full grids + closed-loop MPCGenerator |
| `scripts/eval_waist.py` | waist-specific eval |
| `scripts/probe_waist_follow.py` | phase-1 causal probe: does the decoder read `sp`? |
| `scripts/mpc_smoke_apex.py` | short MPC smoke |
| `scripts/play_isaac.py` | Isaac playback (needs the `CLAUDE.md` Isaac preamble) |

Collision plots for the live ckpt live under `runs/rhythm_bellman_nf1_waist_gain/collision_viz/`.

**Do not** re-derive collision types by hand — use `plot_collision_traj.py`.

---

## This iteration — where the live diagnostics are

Not the paper trail; the artifacts from the current job:

| topic | look here |
|---|---|
| Yaw encoder vs aim vs `q0` (median plot is misleading; use scatter) | `runs/rhythm_bellman_nf1_waist_gain/waist_diag/` |
| Collision cases / overview | `runs/rhythm_bellman_nf1_waist_gain/collision_viz/` |
| Train curve / eval every 2k | `runs/rhythm_bellman_nf1_waist_gain/{launch.log,metrics.jsonl}` |
| Phase-1 waist gain (identity Jacobian) | `runs/phase1_waist_gain/`; tests `tests/test_phase1_waist_follow.py` |
| P/R head import pin | `tests/test_waist_pr_sp_head.py` (`CONTEXT_FIELDS` lives in `warmstart_data`, not `tokenizer`) |

**Torso (default recipe):** decoder direct movements — `w_waist_upright`, `w_waist_yaw_vel`, `w_waist_pr_vel`, `w_waist_pr_terminal`; no encoder SP heads, no SP→strike/init/loss path, `w_waist_aim=0`. Voxel batches retain the checkpoint-compatible `num_future=1` shape but set `future_valid=False` for every row. `VOXEL_POOL=0` selects the legacy A/B path. See `.cursor/rules/phase2-direct-movement.mdc`.

**Voxel exploration:** each certified boundary witness proposes its same-timing positional faces and pairwise diagonals plus a separate atomic timing stencil: one knot shorter and four knots longer by default. The 256-entry unknown buffer rotates fresh candidates every 10 steps and retries least-attempted candidates first. Set `VOXEL_PROBE_REFRESH_EVERY`, `VOXEL_PROBE_SHORTER_LAYERS`, `VOXEL_PROBE_LONGER_LAYERS`, `VOXEL_PROBE_TIMING_STEP`, or `VOXEL_PROBE_TIMING_FRAC` to tune it. Failed probes remain unknown and never update AL.

Encoder waist-head design (legacy): `docs/waist_sp_head_plan.md`, `Phase2Config.w_waist_sp_*` in `train_cost.py`.

---

## Tests / standalone rule

```bash
pytest -q tests/test_standalone.py tests/test_g1_scene.py
```

`amortized_opt/` must not import Isaac (`test_standalone.py`). Geometry constants stay in `constants.py`.

---

## Older material (read last)

- `README.md` — build log of sections, not the live recipe.
- `REPORT.md` — evaluation write-up of the earlier champion.
- `isaac_rl_ppo/` — current Isaac RL env; not this amortized phase-2 job.
- `scripts/repo_adapters/trajopt_hpipm_planner.py` — SQP baseline; needs the original repo.
