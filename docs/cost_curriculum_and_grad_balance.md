# Cost curriculum, balancing, terms, and gradient surgery

Reference for phase-2 training in **amortized-opt**. The live recipe is
`scripts/_run_rhythm_bellman.sh` (tag `rhythm_bellman_nf1_direct`). Code lives in
`amortized_opt/cost.py` (what each term prices),
`amortized_opt/train_cost.py` (schedule, loop, AL wiring), and
`amortized_opt/grad_balance.py` (gradient surgery).

---

## 1. What the optimizer minimizes

Each training step on a horizon-homogeneous minibatch:

```
L = w_cost · mean(FusedCost)
  + w_aux  · Huber(aux heads, labels from the model's own traj)
  + w_mpc_init_tie · Huber(output, shifted-parent init)  [all sampled MPC rows]
  + w_tc   · online temporal-consistency (small subset)
  + w_bptt · FusedCost(second solve after PD roll + bounce)   [phased in]
  + sfx_ramp · (w_suffix_cost · FusedCost(child) + w_suffix_tie · Huber tie)  [phased in]
  + encoder SP extras (tether / hold / distill — all **off** in the Bellman default)
```

There is no general imitation term.  Fresh/quintic ``init`` is only a warm
start.  On roller-derived rows specifically, ``init`` is the previous decoder
plan shifted one tick and is used as a detached temporal-consistency target.

| piece | default weight | phased? |
|---|---|---|
| `w_cost` | 1.0 | no |
| `w_aux` | 5.0 (~2% of objective) | no |
| `w_mpc_init_tie` | 50.0 on every sampled MPC row | no |
| `w_tc` | 50.0 on 5% of batch | no |
| `w_bptt` | 1.0 on up to 25% of steps after step 40k | yes (`bptt_start_frac`) |
| suffix Bellman | cost 1.0, tie 50.0 on 25% of batch | yes (`suffix_warmup`) |

The **cost** itself is a weighted sum of ~16 logged term groups plus AL-priced
constraint residuals. λ (multipliers) adjusts from violations; explicit weight
ramps are optional (A/B only in the current recipe).

---

## 2. Two layers of “balancing”

### 2a. Cost rebalancing (`phase2_cost_config`)

`CostConfig`'s raw defaults are **validation** weights (section 3 of the plan).
They are not usable for training: every term is a **mean over valid elements**,
and element counts differ by orders of magnitude (collision ~2000 pairs vs a
handful violated; jerk integral ~1e9).

`phase2_cost_config()` rescales so groups start within ~1 decade on a phase-1
checkpoint. Examples:

| group | training weight | why |
|---|---|---|
| `w_collision_*` | 1e5 each | undo per-pair mean dilution |
| `w_torque` | 20 | 1 N·m characteristic excess |
| `w_velocity` | 40 | 0.5 rad/s characteristic |
| `w_joint_box` | 400 | 0.05 rad characteristic |
| `w_torque_peak` | 2 | max-shaped gate signal (separate AL group) |
| `w_velocity_peak` | 2 | same |
| `w_joint_box_peak` | 40 | smooth-max ReLU on waypoints (p-norm default) |
| `w_smooth_*` | 1e-4 | jerk no longer 10× weaker than acc |
| `w_posture_mid` / `w_elbow_down` | 0.1 / 10 | real preference, not rounding error |
| `w_terminal_aim` | 10 | weak in-plane pull toward aim |
| `w_waist_pr_band` | 1 | AL group for decoder vs encoder SP |

The Bellman script **overrides** several of these again (collision peak, stick-stick,
waist terms) — see §6.

### 2b. Augmented Lagrangian (`phase2_al_config`)

Fixed weights cannot balance constraint groups whose units and typical magnitudes
move across training (N·m, rad/s, rad, metres). **λ** integrates batch-mean raw
violations and decays when feasible.

Each constraint group gets:

1. **Softplus warning** — fixed `CostConfig.w_*`, never × λ; gradient on the
   safe side of limits.
2. **PHR hard wall** — `λ_used · [g]₊ + (pen/2) · [g]₊²` on raw ReLU excess
   (zero when feasible).
3. **Peak half** (torque / velocity / collision / joint_box) — **smooth max**
   of ReLU priced at `λ_peak · w_*_peak · smooth_max(ReLU)` (default p-norm
   p=12) because the acceptance gate is a **max**, not a mean. AL/eval still
   use hard `amax`.

λ update (every `al_every=100` cost steps):

```
e = ema(V_g) - target_g
λ ← clip(decay · λ + rho_g · e, λ_min, λ_max)
λ_used = clip(λ + Kp·e + Kd·Δe, ...)   # held between updates
```

Notable defaults:

| knob | value | note |
|---|---|---|
| `lam_init` | 1.0 | |
| `lam_max` | 100 (20 for all `torque_*` + `torque_peak` via `AMORTIZED_LAM_TORQUE_MAX`) | |
| `rho` torque | 0.5 | dual step per N·m class-max |
| `rho` joint_box | 15 | per rad class-max |
| `pen` torque | **0** (unless `AMORTIZED_PEN_TORQUE=1`) | PHR off on torque — dominated grads |
| `pen` collision | 1e4 | |
| `target` torque | 0.05 N·m class-max | slack so λ doesn't wind on noise |

**Per-actuator-class limits:** torque, velocity, and joint_box each split into
7 segments (waist, L/R shoulder, L/R elbow, L/R wrist) so a hot left wrist
cannot raise λ on the right shoulder.

**Groups:** 21 limit segment keys + `collision`, `terminal`, four `*_peak` keys,
`waist_pr_band`, `velocity_peak` — see `CONSTRAINT_GROUPS` in
`lagrangian.py`.

---

## 3. Cost curriculum (`WeightSchedule`)

### Current default: full from step 0

```python
default_weight_schedule()  # == WeightSchedule().snap_full()
```

All smoothstep ramps complete at step 0. **λ adjusts** instead of opening
low-weight windows that let the model exploit unpriced DoFs (measured: waist
P/R momentum migrated into joints left at 5% limit price during the old ramp).

### What ramps exist (for A/B / continuations)

| ramp | keys affected | default (pre-`snap_full`) |
|---|---|---|
| `alpha` | limit hinges (`torque`, `velocity`, `joint_box`), strike sub-terms | 5% → 100% over steps 2k–20k |
| `torque_peak_alpha` | `weights["torque_peak"]` | 0 → 1 over 1k steps |
| `velocity_peak_alpha` | `weights["velocity_peak"]` | same |
| `collision_peak_alpha` | `weights["collision_peak"]` | same |
| `joint_box_peak_alpha` | `weights["joint_box_peak"]` | same |
| `waist_alpha` | `waist_box`, `posture_upright`, all waist rate/aim/terminal ramps | 0 → 1 over 300 steps |
| `effort_alpha` | `weights["effort"]` | 300 steps |
| `idle_alpha` | `weights["posture_idle"]` | 300 steps |
| `tip_crossover_alpha` | `weights["tip_crossover"]` | 300 steps |

Strike sub-terms (`terminal_velocity`, `terminal_elev`) use `lo_strike=0.10`
blended with `alpha`.

### Per-joint limit exemption

`full_limit_joints = ("waist_pitch_joint", "waist_roll_joint")`: those two
joints get **full** limit hinge price from step 0 while others follow `alpha`.
Rationale: waist pitch/roll actuators are already at ~52% of torque rating
standing upright — the arm-reach curriculum window is a lie for them. AL λ
updates are **unchanged** (ramps only scaled the penalty, not violations).

Emitted as `weights["limit_alpha"]` — a `(D,)` vector on softplus hinges only,
never on ReLU `live`.

### Stage-A gate (legacy, off by default)

`gate_stage_a=True` holds `alpha=0` until `stage_a_pass` (≥30/81 accepted,
≥66% collision-free, ≥95% elbow-down). **Off** in the Bellman recipe: a gate
that never opens freezes limits forever on short runs.

`--full-weights` / `FULL_WEIGHTS=1`: re-applies `snap_full()` and sets
`suffix_warmup=1` for continuations.

---

## 4. Every cost term

Terms reduce to **means over valid elements** unless noted. Logged keys are
`TERM_KEYS` in `train_cost.py`. Constraint terms also have `v_*` violation
columns.

### 4.1 Terminal (`terminal`, `terminal_aim`)

**What:** per active hand at its hit index — contact inside legal hit region
(shaft capsule on cymbals/hi-hat, bead on pads), impact velocity vs command,
stick elevation band.

**Pricing:** `w_terminal_region`, `w_terminal_velocity`, `w_terminal_elev` (from
`CostConfig`; strike sub-weights ramp with `lo_strike` if ramps on).

**AL:** `terminal` group on region miss (metres).

**Extra:** `w_terminal_aim=10` — weak 2 cm deadband pull toward aim so the
region hinge doesn't settle on the rim.

### 4.2 Collision (`collision`)

**What:** stick capsule (butt→tip) + wrist sphere vs every drum / under-volume /
torso capsule / head. Target-drum and departure exemptions on stick-vs-drum only;
wrist never exempt.

**Pricing:** `w_collision_drum`, `w_collision_under`, `w_collision_self` — training
default 1e5 each.

**AL:** `collision` on max penetration (metres); `collision_peak` on
smooth-max ReLU penetration (Bellman: `w_collision_peak=2000`).

**Sub-types logged:** drum / under / self / cross-hand (`collision_stick`).

### 4.3 Stick–stick (`w_stick_stick`)

**What:** cross-hand capsule–capsule clearance (same geometry as per-hand
collision). Distinct from `w_tip_crossover` (lateral ordering).

**Default in library:** 0 (bit-identical). **Bellman:** 100.

**AL:** joins `collision_peak` / violations when weight > 0.

### 4.4 Tip crossover (`tip_crossover`)

**What:** plain objective — softplus on `margin - (tip_L.y - tip_R.y)` in torso
frame; penalizes crossed tips even if capsules don't intersect.

**Bellman:** `w_tip_crossover=50`. Ramped via `weights["tip_crossover"]` if
curriculum on.

**Not AL.**

### 4.5 Torque (`torque`)

**What:** RNEA (armature ON) at subsampled knots vs `eta_torque ×` Isaac effort
limits.

**Two halves:**

| half | shape | priced by |
|---|---|---|
| warning | mean softplus | fixed `w_torque=20` |
| peak | **smooth max** ReLU hard excess | `λ_peak × w_torque_peak × smooth_max` (Bellman: 2.0) |

**AL:** per-segment `torque_{waist,left_shoulder,...}` on class-max ReLU;
`torque_peak` on horizon max.

**Tracking torque (off in Bellman):** `w_track_torque` / `w_track_torque_peak`
price `|τ_rnea + τ_fb|` (PD demand). When on, the `torque` group's raw
violation switches to tracking excess. Default gate is **knot RNEA**, not PD
demand — avoids scoring unclipped first-substep Kd spikes.

**Env:** `AMORTIZED_LAM_TORQUE_MAX=20` caps segment + peak λ (wrapper default).

### 4.6 Velocity (`velocity`)

**What:** per-segment max `|qd|` vs `eta_velocity ×` speed ratings.

Same warning + peak split as torque. Library `phase2_cost_config`:
`w_velocity_peak=2.0`. **Bellman launcher default: both off**
(`W_VELOCITY=0`, `W_VELOCITY_PEAK=0`) — rarely binding; skips the analytic
locator when both weights are zero.

### 4.7 Joint box (`joint_box`, `joint_box_peak`)

**What:** decoder **waypoints** vs posture-narrowed URDF box (`eta_position=0.8`).
Waist pitch/roll rows optionally shrunk to `±waist_pr_box` (Bellman: 0.52 rad)
via `weights["waist_box"]` linear blend — **not** a separate multiplier.

**Peak:** smooth-max ReLU per (waypoint, joint) — Bellman `w_joint_box_peak=40`.

### 4.8 Smooth (`smooth`)

**What:** closed-form ∫ v² / a² / j² per spline segment.

**Training:** `w_smooth_acc = w_smooth_jerk = 1e-4` (10:1 acc:jerk ratio from
trajopt deliberately broken — jerk fights torque spikes).

### 4.9 Posture (`posture`)

**What:** mid-range well, elbow-down hinge, optional axial twist.

**Training:** `w_posture_mid=0.1`, `w_elbow_down=10`.

### 4.10 Effort (`effort`)

**What:** strike-masked mean `τ²` energy proxy on RNEA torques. Faded to zero
over last `effort_fade_wp=5` waypoints before each hit so it doesn't fight
impact speed.

**Plain objective**, no λ. Default `w_effort=0` (off unless enabled).

### 4.11 Waist — direct decoder movement (Bellman default)

Encoder set-point heads and `w_waist_aim` are **off**. Torso is shaped by:

| term | weight (Bellman) | what |
|---|---|---|
| `waist_upright` | 100 | `mean(q_pitch² + q_roll²)` vs 0 (yaw excluded) |
| `waist_yaw_vel` | 80 | `(rad/s)⁻²` on `qd_yaw²` — anti wind-up |
| `waist_pr_vel` | 80 | `(rad/s)⁻²` on `qd_pitch² + qd_roll²` |
| `waist_pr_terminal` | 200 (CLI default when P/R head off) | V_f: P/R² at **nearer** active hit vs 0 |
| `waist_pr_box` | 0.52 rad half-width | linear blend on joint-box rows |
| `waist_pr_band` | 1.0 (via `phase2_cost_config`) | sum-ReLU decoder vs `stopgrad(encoder sp)`; **off** when no SP head |
| `waist_aim` | **0** | yaw IL vs analytic ramp — off |

All waist terms share `waist_alpha` ramp when curriculum is on (full at step 0
in default schedule).

**Legacy encoder path (off):** `waist_sp_from_head`, `waist_pr_sp_from_head`,
tether / hold / hold-band / distill, `w_waist_sp_*` — see
`docs/waist_sp_head_plan.md` and `docs/waist_pr_sp_head_plan.md`.

### 4.12 Idle arm (`posture_idle` → cylinder + vel)

When an idle-cylinder library is loaded (`assets/idle_cylinder_poses.json`):

| term | default | what |
|---|---|---|
| `w_idle_cylinder` | 1.0 | tip outside torso-frame safety cylinder |
| `w_idle_vel` | 0.02 | parked arm stopped at terminal knot |

**Not AL** (unsatisfiable on mid-flight states). Ramped via `weights["posture_idle"]`.

---

## 5. Auxiliary losses (not cost curriculum)

These have their **own** phase-in; cost weights stay full.

### Replay init tie (`w_mpc_init_tie=50`, all sampled MPC rows)

The roller advances the plant, shifts the previous output plan, and stores that
shifted plan as the next row's ``init``.  On every sampled roller/replay row,
masked Huber ties the current output to this stored parent tail, excluding the
repeated final padding waypoint.  Fresh/quintic and frontier rows are masked
out.  This is detached, recency-biased TC: no extra solve or plant step.

Metrics: ``mpc_init_tie`` is the raw Huber value and ``mpc_init_frac`` is the
fraction of sampled rows carrying the MPC target.

### Online temporal consistency (`w_tc=50`, 5% of batch)

Advance the noised PD plant one tick; Huber disagreement between re-plan and
shifted parent.  Unlike the replay tie, gradients flow through the current
parent solve and plant step.  The small online slice remains as a current-policy
reference while the cheaper replay approximation supplies the dense signal.

### BPTT (`w_bptt=1`, Bellman)

- Starts at step `0.667 × 60000 = 40000` (`bptt_start_frac`).
- Ramps to 25% of steps doing a second solve after PD roll + randomized bounce.
- Uses same `FusedCost` and weights; waist_aim ref suppressed in sub-loss.

### Suffix Bellman (`suffix_frac=0.25`, Bellman)

On a random tail offset `k`, re-solve from parent state with shifted parent as
init:

```
w_suffix_cost · FusedCost(child) + w_suffix_tie · Huber(child, parent tail)
```

- `suffix_warmup=1000` smoothstep on both weights.
- Stays on the **plan manifold** (no sim) — trains shrinking-horizon optimality.
- Distinct from TC.

---

## 6. Bellman script defaults (`_run_rhythm_bellman.sh`)

Quick reference for flags passed to `train_phase2.py`:

```bash
# Gradient surgery
--grad-balance al_pcgrad
--grad-balance-tau-torque-cap 0.30
--grad-balance-waist-floor 2.0
--grad-term-every 50

# Limits / peaks
--w-torque-peak 2.0
--w-velocity 0 --w-velocity-peak 0   # Bellman default: velocity limits off
# restore with --w-velocity 40 --w-velocity-peak 2.0
--w-collision-peak 2000
--w-joint-box-peak 40
--w-track-torque 0
--w-track-torque-peak 0
--soft-wrist-effort-scale 1
--clamp-traj-to-box

# Geometry / ordering
--w-stick-stick 100
--w-tip-crossover 50
--w-idle-cylinder 1.0

# Temporal consistency
--w-mpc-init-tie 50       # every sampled roller/replay row
--tc-frac 0.05            # online differentiable TC reference

# Waist (decoder-owned)
--w-waist-aim 0
--w-waist-upright 100
--waist-pr-box 0.52
--w-waist-yaw-vel 80
--w-waist-pr-vel 80
--w-waist-pr-terminal 200   # Phase2Config default when no P/R head

# Aux
--suffix-frac 0.25
--w-suffix-cost 1.0 --w-suffix-tie 50.0 --suffix-warmup 1000
--bptt-start-frac 0.6667

# Schedule: implicit snap_full() — no ramp CLI
# Env: AMORTIZED_LAM_TORQUE_MAX=20
```

**Off unless explicitly enabled:** encoder yaw/P/R SP heads, `w_waist_aim`,
tether/smooth/hold/distill, `WAIST_PR_EMA`, tracking torque, frontier
collocation, stage-A gate, explicit limit/strike ramps.

**Pool mix:** 30% rhythm idle / 60% post-hit / 10% far-fetched; kit DR
(tilt, radius, jitter) baked into cache key.

---

## 7. Gradient surgery (`grad_balance`)

When `--grad-balance al_pcgrad` (Bellman default), the **cost** backward is
replaced; aux / TC / BPTT / suffix still use normal `backward()` afterward.

### Problem

RNEA inflates raw torque gradient norms (~850 dense hinge elements). A single
`total.backward()` lets torque monopolize the step even when strike or waist
terms need movement.

### Groups (`GRAD_BALANCE_GROUPS`)

| group | cost terms summed | AL keys for priority |
|---|---|---|
| `torque` | `torque` | `torque_peak`, all `torque_*` segments |
| `feas_other` | `velocity`, `joint_box`, `collision` | peaks + segment keys + `collision` |
| `strike` | `terminal`, `terminal_aim` | `terminal` |
| `shape` | `smooth`, `posture`, `effort`, `tip_crossover` | (none → shape floor) |
| `waist` | all `waist_*` terms | `waist_pr_band` |

### Algorithm

1. **Priority** `p_i = γ_torque · λ_i · max(0, V_i - target_i)^β` (torque
   damped by `grad_balance_torque_gamma=0.35`; waist floored at
   `grad_balance_waist_floor=2.0` for `al_pcgrad`).
2. **Budget** `τ_i = p_i / Σp`, then **cap** `τ_torque ≤ 0.30` and redistribute.
3. **Per-group VJP** on `w_cost · Σ terms_in_group`.
   `al_pcgrad_output` (Bellman default) stops these VJPs at the emitted
   trajectory; `al_pcgrad` retains the exact, slower parameter-space version.
4. **Protected PCGrad:** compute the dedicated yaw/pitch/roll
   `joint_box_waist` warning + AL direction. If torque conflicts with it,
   project only the torque gradient off the box direction. The protected box
   and all other groups are unchanged; the reference is not a sixth budget
   group and is therefore not double-counted.
5. **Combine** `g = Σ_i τ_i · (g_i / ||g_i||)`. Output-space mode sends the
   combined trajectory gradient through the transformer once; parameter-space
   mode writes the combined parameter gradients directly.
6. Then `(w_aux · loss_aux).backward()` and `extra.backward()` for TC/BPTT/suffix/SP extras.

Alternative: `al_gradnorm` — same without PCGrad / waist floor.

The Bellman launcher uses `grad_balance_refresh=4`: norms are measured exactly
every fourth visit to each horizon bucket and updated with a 0.9 EMA. The three
intervening visits use `Σ_i τ_i C_i / EMA(||g_i||)` for GradNorm, while a
batched torque/waist-box reference VJP preserves the one-sided protection on
every optimizer step. A non-finite refresh is rejected rather than poisoning
the EMA; the first valid visit after resume initializes each bucket.

### Logged diagnostics (`gb_stats` → `metrics.jsonl`)

- `gb_tau_*`, `gb_pri_*`, `gb_norm_*` — allocation and raw norms
- `gb_batched_vjp` — `1` on an exact eager batched VJP, `0` on EMA steps
- `gb_output_space` — `1` when norms/projections are in trajectory space
- `gb_refresh`, `gb_refresh_failed`, `gb_norm_ema_*` — periodic refresh state
- `cos_torque_joint_box_waist`, `gb_pcgrad_torque_box_conflict` — protected
  waist-box relationship and whether surgery activated
- `cos_torque_strike`, `cos_torque_waist`, … — pairwise cosine conflicts
- `cos_min`, `cos_mean` — summary conflict

Priority, norm, dot-product, PCGrad projection, and combine math stays on GPU;
only emitted log rows convert diagnostics to host scalars. Pairwise cosines are
also computed only on log rows rather than every optimizer step.

### Compilation

The Isaac environment is pinned to PyTorch 2.7 / Triton 3.3 and hits the
upstream Inductor `constexpr.is_block` bug.  A standalone training environment
at `/data2/eric/conda_envs/amortized_compile` uses PyTorch 2.11 / Triton 3.6:

```bash
AMORTIZED_CONDA_ENV=/data2/eric/conda_envs/amortized_compile \
COMPILE_MODEL=1 COMPILE_COST=0 \
bash scripts/_run_rhythm_bellman.sh
```

The context encoder is compiled as its own unit, separately from the decoder.
Measured on RTX 5090: standalone `FusedCost` was 274 ms eager vs 100 ms compiled
at batch 256 × horizon 50, but the compiled physics **backward** produced NaN
group gradients on finite-loss batches and is therefore disabled. With eager
physics and compiled encoder/decoder, refresh-4 measured 0.697 it/s versus
0.624 for exact-every-step under the same diagnostic interval (12% faster);
the resumed live process reached 0.83 it/s after warmup.

Negative cosines on sustained steps indicate **conflict** (candidate for grad
surgery); mere dominance is an λ / weight / PHR knob problem first.

---

## 8. Gradient attribution probes (`grad_term_every`)

Separate from grad surgery: **diagnostic** per-term norms via
`measure_cost_term_grad_norms` — one extra forward + one `autograd.grad` per
`TERM_KEYS` entry every N steps (`--grad-term-every 250` in Bellman).

Logs `grad_<term>`, `grad_torque_frac`, etc. on train rows.

The same 250-step probe evaluates torque curvature on 8 rows. It logs the
autograd directional HVP, a ±1 mrad finite-difference curvature, and Taylor
reduction ratios from 0.25–5 mrad maximum waypoint displacement. On the 24k
checkpoint the point HVP underestimated finite-difference curvature by
833–2,590×: hard-max/ReLU active-set switches dominate the local second
derivative. Measured `rho` stayed about 0.83–0.91 at 1 mrad, fell to 0.69–0.73
at 2 mrad, and reached 0.33–0.55 at 5 mrad. Therefore the raw HVP must remain a
diagnostic; it is not a safe torque-step controller.

**Do not** tune from loss share `c_*` or λ alone — see
`.cursor/skills/gradient-attribution-training/SKILL.md`.

Correct probe recipe:

1. Fresh forward on the same batch context (not after `total.backward()`).
2. `autograd.grad(w_cost · mean(term), params)` per key.
3. Report L2 norms and fractions.

---

## 9. Wiring: `_wtensors`

Each step, `WeightSchedule.multipliers(step)` × `AugmentedLagrangian.weights()`
become the `weights` dict passed to `FusedCost.forward`:

- **λ only** on constraint ReLU residuals (segment keys expanded from family).
- **Peak keys:** `alpha_peak × λ_peak` (curriculum alpha on the product).
- **Non-λ curriculum:** `limit_alpha` vector, `waist_box`, strike ramps, etc.
- **PHR:** `pen_<group>` tensors when non-zero.

Same dict is forwarded to BPTT and suffix sub-steps.

---

## 10. Acceptance gates (eval, not training loss)

Rolling eval uses `TrajectoryScorer` / `AcceptanceThresholds`:

- `ok_hit`, `ok_collision` (zero-tolerance max penetration)
- `ok_torque` (max RNEA excess — **not** tracking torque in Bellman default)
- joint limits, elbow-down, etc.

Target: `--target-acceptance 0.95` on 81-grid eval every 2k steps.

Training optimizes the **differentiable** cost + AL; gates measure deployability.

---

## 11. Tuning order (when something dominates or conflicts)

1. **λ / AL** — `AMORTIZED_LAM_TORQUE_MAX`, targets, separate peak vs segment
   groups; avoid raising `w_torque_peak` against λ calibrated on the mean.
2. **Residual shape** — smooth-max ReLU peak + class-max violations (already default).
3. **Upstream** — `clamp_traj_to_box`, `limit_alpha`, waist full limits on P/R.
4. **PHR** — keep `pen_torque=0`; chronic groups already have frozen `pen_grow=1`.
5. **Grad surgery** — only if probes show sustained **conflict** (negative
   cosines), not mere torque dominance.
6. **Re-probe** after each change; `c_*` ratios do not predict `grad_*`.

---

## 12. Key file map

| topic | file |
|---|---|
| Term math | `amortized_opt/cost.py` |
| Schedule, loop, probes | `amortized_opt/train_cost.py` |
| λ dynamics | `amortized_opt/lagrangian.py` |
| Grad surgery | `amortized_opt/grad_balance.py` |
| CLI / config assembly | `scripts/train_phase2.py` |
| Live launch recipe | `scripts/_run_rhythm_bellman.sh` |
| Agent index | `quick.md` |
| Direct-movement rule | `.cursor/rules/phase2-direct-movement.mdc` |
