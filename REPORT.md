# Amortized trajectory optimizer for the G1 drummer — final evaluation

**Date:** 2026-08-11 · **Champion:** `runs/phase2b_main/best.pt` (phase 2b, step 1 000)
· **Deployment config:** per-joint reference lead (`MPCGenConfig.cmd_lead_s =
LEAD_PRESETS["fit"]`), everything else off · **Hardware:** one RTX 5090
(`cuda:1` / `cuda:3`), Isaac-free.

This is the standalone final report for the amortized-optimizer project
(`docs/amortized_optimizer_plan.md`, sections 1–7). It is an **evaluation
report**: nothing was retrained for it. Method detail lives in
[`README.md`](README.md); the expert reference lives in
[`baselines/trajopt_hpipm_2026-08-10.md`](baselines/trajopt_hpipm_2026-08-10.md).

---

## 1. Executive summary

A 19.3 M-parameter transformer replaces a 10-DoF HPIPM SQP as the drummer's
trajectory planner. It plans **17 DoF (both arms + waist) in one forward pass**,
and on the project's own acceptance grid it **beats the expert outright while
being 360–2 500× faster**. It does *not* yet play the drum reliably in closed
loop: 63 % of scheduled strikes never reach the head inside the legal disk.

### The numbers

| | amortized (phase 2b) | trajopt HPIPM v2b (expert) | quintic seed |
| --- | ---: | ---: | ---: |
| **accepted, 243-case grid, seed 0** (validation) | **175 / 243 (72.0 %)** | 149 / 243 (61.3 %) | 0 / 243 |
| **accepted, seed 7** (held-out test) | **180 / 243 (74.1 %)** | — | 0 / 243 |
| **accepted, seed 13** (fresh test, never used) | **178 / 243 (73.3 %)** | — | 0 / 243 |
| contact inside the legal hit disk | 240–243 / 243 | 241 / 243 | 49–59 / 243 |
| impact speed in band | **243 / 243** | 241 / 243 | 0 / 243 |
| collision-free, striking arm | **189–194 / 243** | 162 / 243 | 9–21 / 243 |
| collision-free, **both arms** | 57–68 / 243 | 66 / 243 (v2: **83**) | — |
| no torque excess | **220–224 / 243** | 209 / 243 | 211–213 / 243 |
| stick elevation in band | **243 / 243** | 230 / 243 | — |
| short-horizon cell (T = 0.44 s) | **32–42 / 54** | 26 / 54 | 0 / 54 |
| contact position error p50 (`standard`) | **39.2 mm** | 66.9 mm | 154 mm |
| solve time p50, one MPC tick | **17.9 ms** eager / **2.65 ms** batched | 6 514 ms | 2.4 ms |

**Fresh-seed transfer is clean.** The grid the model was selected on (seed 0)
gives 175/243; two grids it has never seen give **180** and **178**. There is no
validation-grid overfitting in the champion — unlike its own continuation
(§7c: phase 2c gained +7 on seed 0 and lost −8 on seed 7).

**Closed loop (the deployment condition)** — champion + per-joint lead, 4
held-out rhythm clips × 3 start seeds = 12 episodes, 3 000 ticks, **204 executed
strikes**, scored by sub-tick hit latency:

| | champion + lead | champion, no lead | phase 1 + lead |
| --- | ---: | ---: | ---: |
| crossed the head plane inside the disk | **76 / 204 (37 %)** | 54 / 204 (26 %) | 0 / 59 |
| **miss rate** | **63 %** | 74 % | 100 % |
| \|latency\| p50 | **9.5 ms** | 14.2 ms | — |
| \|latency\| p90 | **18.1 ms** | 48.4 ms | — |
| within 10 / 20 / 40 ms | **42 / 72 / 73** | 20 / 44 / 46 | 0 / 0 / 0 |
| impact speed in band at crossing | **57 / 204 (28 %)** | 32 / 204 | 0 |
| episodes completing 250 ticks | **12 / 12** | 12 / 12 | 0 / 12 (all diverged) |
| PD tracking p50 / p90 | 8.0 / 37.3 mrad | 5.3 / 18.6 mrad | 18.6 / 139.8 mrad |

**What a listener would hear.** Roughly one beat in three lands, and the ones
that land are tight (p50 9.5 ms, p90 18.1 ms — inside a human drummer's timing
spread). The other two in three are silent: the stick arrives near the head but
never crosses it inside the disk within ±100 ms. This is not a groove yet.

**Training cost: 4.14 GPU-hours, single GPU, no DDP** — 20 000 phase-1 steps
(20.5 min) + 44 200 total optimizer steps across every phase, pilot race and
abandoned continuation included. The "trains in hours" claim holds; the champion
itself is reachable in ~2.2 h from cold.

### Verdict

**Ready for the IsaacSim RL fine-tune, with one precondition.** The planner is
fast enough (17.9 ms eager < 20 ms control period; 0.57 ms compiled), stable
(0/12 divergences, initialization-insensitive), and kinematically better than
the expert it replaces. The remaining failure is *not* a planning failure — it
is the gap between a kinematically-correct waypoint and what a `kp = 40`
position servo can be at that instant (§7c decision 1). That gap is exactly what
an in-simulator fine-tune with real contact is for, so RL is the right next
stage rather than more offline descent — the phase-2c continuation is the
control experiment that proves more kinematic descent does not buy strikes.
The precondition: **the RL objective must score the executed state, not the
plan.** Details in §8.

---

## 2. Method recap

One page; every claim points into a `README.md` chapter.

**The problem.** A receding-horizon MPC at 50 Hz plans a joint-space trajectory
to the next drum hit, executes one waypoint, and re-plans. The horizon *shrinks*
toward each strike. The incumbent planner is an HPIPM SQP that takes 1.5–6.5 s
per solve — 73–326× the 20 ms control period.

**The substrate** (README *"standalone physics core"*). A vendored, Isaac-free,
batched, differentiable 17-DoF model: `ChainFK` (5 frames), `BatchedRNEA`
(armature-correct — the repo's original omitted armature entirely, a 27× torque
underestimate at the wrists), an analytic drum / self-collision `SdfPenalty`, and
the Isaac limit tables. All sync-free, so `torch.compile(reduce-overhead)`
reaches 74 fwd+bwd steps/s at 12 800 poses. A CI test asserts zero
device→host syncs.

**The scorer** (README *"the evaluation harness"*). `generate_grid` reproduces
the repo's 162-case acceptance grid cell for cell, plus an 81-case `near_drum`
departure grid. `TrajectoryScorer` prices contact position / speed / direction at
the strike's `contact_sigma` point, hit-disk membership, stick elevation,
collision on the interpolated path with target-drum and departure exemptions,
RNEA torque with armature against the Isaac effort limits, and velocity /
joint-box excess. **Every number in this report — expert and model — comes from
this one scorer.**

**The expert** (`baselines/`). The repo's HPIPM SQP over the same 243 cases,
three waist policies; `v2b` is primary (149/243). Read its caveats before any
comparison: trajopt plans **10 DoF** and its idle arm is invisible to its own
collision model (caveat 1); the scorer's SDF is stricter than trajopt's hpp-fcl
(caveat 2); the SQP is not reproducible run to run (caveat 6), which is why the
expert is **re-scored, never re-solved**.

**The model** (README *"the transformer"*). Encoder 2 × d=256 over drum-geometry
/ goal / future-hit tokens; decoder 6 × d=384 over one state token and
`max_horizon` waypoint tokens, with dual terminal-anchored Fourier time codes
(time-to-left-hit and time-to-right-hit) and per-element horizon masks. Outputs
one 17-DoF trajectory plus supervised `vel` / `acc` / `torque` / `fk_pos` /
`fk_vel` heads. 19.32 M params.

**The objective** (README *"the fused cost"* + *§7b.3*). `FusedCost`: terminal
(hit region, impact velocity, stick elevation), collision, torque (mean hinge
**and**, since 7b, a `pnorm` smooth-max peak), velocity, joint box, smoothness,
posture — each a softplus hinge against a conservative limit, reduced per valid
element. Five (now six) augmented-Lagrangian groups scale the penalties from raw
violations. Validated against the recorded expert: torque agrees with the scorer
to **0.0**, contact position to 6e-11 m; every term separates the gate it prices
(AUROC 0.84–1.00).

**Training** (README *phase 1 · phase 2 · 7b · 7c*).
Phase 1 is a 20 k-step behaviour clone of a quintic warm start — a *diagnostic*,
not a gate; it scores 0/243. Phase 2+ is **direct descent on the differentiable
cost with no labels**: `L = cost(model(init, state, ctx)) + aux(own output) +
temporal-consistency(solve@H vs solve@H−1 through the plant) + chained BPTT`.
Data is 30 % eval-like fresh scenarios plus a receding-horizon `MPCRoller` replay
that re-goals per hand in place. The champion is step 1 000 of the 7b
continuation (`lr 5e-5`, torque-peak weight 1.0 ramped over 2 000 steps).

**Deployment** (README *§7c*). The plan is executed through a 250 Hz PD servo
with FOH ramps. `MPCGenConfig.cmd_lead_s` commands `q_ref(t + Δ)` with a
**per-joint** Δ identified offline against that exact servo model
(`scripts/fit_cmd_lead.py`): 0 ms at the wrists, 15 ms at the elbows, 50 ms at
waist pitch/roll. This is the only deployment mechanism enabled; `hit_lead_s`
(goal shift) and `freeze_wp` remain implemented, tested and **off**.

---

## 3. A — Open-loop grids (kinematic)

243 cases = `standard` (162) + `near_drum` (81). One refinement iteration,
quintic init unless noted. Commands in §9.

### 3.1 Headline, three seeds

`accepted` is on the **striking-arm** collision basis, matching how the expert is
scored.

| | seed 0 (validation) | seed 7 (test) | seed 13 (fresh test) | v2b | v2 | v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **accepted** | **175 / 243** | **180 / 243** | **178 / 243** | 149 | 126 | 151 |
| contact in the hit disk | 242 | 240 | **243** | 241 | 238 | 240 |
| impact speed in band | **243** | **243** | **243** | 241 | 241 | 233 |
| collision-free, striking arm | 193 | 194 | 189 | 162 | 148 | 160 |
| collision-free, **both arms** | 60 | 57 | 68 | 66 | **83** | 7 |
| no torque excess | 220 | 223 | **224** | 209 | 187 | 208 |
| inside the joint box | 243 | 243 | 243 | 243 | 243 | 243 |
| stick elevation in band | **243** | **243** | **243** | 230 | 226 | 232 |
| short-horizon cell T = 0.44 s | 32 / 54 | 42 / 54 | 42 / 54 | 26 / 54 | 27 / 54 | 17 / 54 |

Continuous metrics, `standard` grid p50 (the expert column is its seed-0 run):

| | seed 0 | seed 7 | seed 13 | v2b |
| --- | ---: | ---: | ---: | ---: |
| contact position error (mm) | 39.2 | 40.1 | 40.0 | 66.9 |
| impact speed error (m/s) | 0.054 | 0.052 | 0.059 | **0.044** |
| max penetration p90 (mm) | 49.7 | 33.7 | 30.3 | **41.8** |
| torque excess p90 (N·m) | **0.00** | **0.00** | **0.00** | 1.18 |
| jerk integral | 1.83e7 | 1.83e7 | 1.83e7 | **1.35e7** |

**Reading it.** The model beats v2b on acceptance (+26 to +31 cases), on
striking-arm collision-freedom (+27 to +32), on torque (+11 to +15), on hit-disk
membership, on impact-speed band, on stick elevation, on contact accuracy
(−28 mm), and on the short-horizon cell the MPC actually lives in (+6 to +16) —
at 1/364 of the solve time. It gives back impact-speed precision (0.054 vs
0.044 m/s), jerk (1.83e7 vs 1.35e7) and **both-arms collision-freedom**.

**The both-arms result is a negative finding and it matters.** The baseline
document predicted this row "should be a rout": the amortized model plans both
arms, so it is not subject to the expert's single-arm blind spot. Measured, it is
**not** a rout — 57–68/243 against v2b's 66 and v2's **83**, i.e. the model is no
better than an expert that freezes its idle arm at URDF neutral, and clearly
worse than v2. The mechanism is §7c's *design note: the idle arm*: an arm with no
goal in the context lookahead **has no terminal objective**, so only collision,
limits, smoothness and a mid-range posture well price it, and its behaviour is
underdetermined. The specified fix (a yaw-parameterized home-pose term,
`w_idle_home`) is designed but not implemented.

### 3.2 Per-drum breakdown (pooled over the three seeds, 729 cases)

| drum | amortized | v2b | v2 |
| --- | ---: | ---: | ---: |
| cymbal_right | **96 / 117 (82 %)** | 23 / 39 (59 %) | 24 / 39 |
| snare | **93 / 117 (79 %)** | 29 / 39 (74 %) | 18 / 39 |
| cymbal_left | **93 / 126 (74 %)** | 22 / 42 (52 %) | 20 / 42 |
| tom1 | 87 / 117 (74 %) | 30 / 39 (77 %) | 24 / 39 |
| **tom2** | **84 / 126 (67 %)** | 13 / 42 (**31 %**) | 11 / 42 |
| **hihat** | **80 / 126 (63 %)** | 32 / 42 (76 %) | 29 / 42 |

* **`tom2`, the expert's standing weak spot, is fixed.** v2b manages 31 % there
  and it was worst on every expert run; the model reaches 67 %. It is still the
  model's second-weakest drum and the least stable across seeds (30, 23, 31 of
  42), so it is worth watching — but it is no longer a hole.
* **`hihat` is the model's weak spot and the one drum the expert beats it on**
  (63 % vs 76 %), consistently across all three seeds (25, 27, 28 of 42). The
  hi-hat is a shaft-hit, edge-hit drum with a tight 15 cm radius sitting high and
  far to the left; it is also the drum that dominates the closed-loop clips.
* Both cymbals are the model's strongest cells and among the expert's weakest —
  the shaft-hit `contact_sigma` geometry rewards a planner that can place the
  whole stick, which the dual-arm model does and the single-arm QP does not.

### 3.3 Per-cell breakdown (seed 7, the test grid)

| cell | n | accepted | coll (striking) | no torque excess | coll (both) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **T = 0.44 s** | 54 | **42** (v2b 26) | 42 | 51 | 14 |
| T = 0.60 s | 135 | 102 (v2b 84) | 113 | 118 | 21 |
| T = 0.80 s | 54 | 36 (v2b 39) | 39 | 54 | 22 |
| 1 m/s | 54 | 34 (v2b 32) | 37 | 54 | 24 |
| 2 m/s | 135 | 100 (v2b 93) | 111 | 115 | 14 |
| 3 m/s | 54 | **46** (v2b 24) | 46 | 54 | 19 |
| left hand | 126 | 81 | 95 | 106 | 29 |
| right hand | 117 | **99** | 99 | 117 | 28 |
| `standard` | 162 | 120 | 123 | 159 | 53 |
| `near_drum` | 81 | 60 | 71 | 64 | 4 |

Three things to note. **(a) The short horizon is the model's best relative
cell** — 42/54 against the expert's 26/54, and 32–42/54 across seeds. This is the
regime the MPC is in for every strike and the one the expert is worst at (with
only 22 waypoints there is no time for the SQP to negotiate a waist path *and* a
strike). **(b) The 3 m/s cell reverses completely**: the expert collapses to
24/54 while the model reaches 46/54 — the amortized planner is better precisely
where the dynamics bite. **(c) The left hand trails the right** (81 vs 99 on
seed 7, and the gap is present at every seed), which is the same
hihat / cymbal_left geometry as §3.2.

`near_drum` both-arms collision-freedom is 1–6 / 81 — near-total failure. Those
cases start with *one* stick resting on a source drum and the other arm wherever
the sampler put it; the idle arm has no reason to leave the kit and nothing pulls
it out. Same root cause as §3.1.

### 3.4 Initialization and iteration robustness

Seed 7, 243 cases (this is generalization item C(iii); it is reported here
because it is an open-loop measurement):

| init | ×1 accepted | ×3 accepted | hit disk | coll | torque |
| --- | ---: | ---: | ---: | ---: | ---: |
| quintic warm start | **180 / 243** | 176 | 240 | 194 | 223 |
| zero-motion hold | **184 / 243** | 178 | 243 | 194 | 224 |
| quintic + OU noise σ = 0.05 rad | **186 / 243** | — | 243 | 194 | 226 |
| quintic + OU noise σ = 0.15 rad | 177 / 243 | — | 241 | 193 | 221 |
| quintic + OU noise σ = 0.30 rad | 173 / 243 | — | 241 | 193 | 217 |

**Initialization insensitivity is complete.** The spread over five very different
initializations — including a 0.30 rad OU-noised warm start and a zero-motion
hold, which phase 1 collapsed from (16 → 6 on the hit disk) — is **173 to 186 of
243**, a 5 % band. Direct cost descent removed the refiner's dependence on a good
warm start entirely. The closed loop supplies a third initialization family (the
previous plan shifted one waypoint) and is measured in §4.

**One refinement iteration ties or beats three** on every seed and every init
(180 vs 176, 184 vs 178). §7b decision 4 stands: nothing in the objective asks
iteration 2 to improve on iteration 1 — training only ever back-propagates a
single pass.

### 3.5 Solve latency

| | measured | vs the 20 ms control period |
| --- | ---: | --- |
| one MPC tick, batch 1, **eager**, bf16 (3 runs: 19.7 / 17.9 / 17.6 ms) | **17.9 ms** p50 | fits, but only just |
| the 162-case grid in one batched pass | 424–433 ms → **2.65 ms/case** | 7.5× headroom |
| decoder alone, batch 1, `torch.compile(reduce-overhead)` | 1.26 ms | |
| decoder alone, batch 1, compiled **+ TF32** | **0.565 ms** | 35× headroom |
| trajopt HPIPM v2b, p50 | 6 514 ms | 326× over budget |
| trajopt HPIPM v2 / v1, p50 | 2 231 / 1 451 ms | 112× / 73× over |

**364× the expert at the honest end-to-end eager number** (17.9 vs 6 514 ms),
2 458× batched. The 17.9 ms figure includes context encoding, host-side
tokenization and the numpy round-trip in the deployed loop, none of which is
compiled; the 0.565 ms decoder measurement is what the same forward pass costs
once the toolchain is used (§7b.1 showed `configure_inductor()` makes compilation
work in this environment). Closed-loop wall clock is 29.7 ms/tick including the
scorer and the host-side MPC loop, which is an *evaluation-harness* cost, not a
controller cost.

---

## 4. B — Closed-loop MPC (the deployment condition)

Champion + per-joint reference lead, **all four** held-out eval clips
(`gmd_eval_000001_30s`, `000003_45s`, `000006_30s`, `000007_45s`), **3 start
seeds each** = 12 episodes × 250 ticks = 3 000 ticks, **204 executed strikes**.
Comparators: the same checkpoint with no lead, and phase 1 with the lead.

Primary metric is `probe.hit_latency`: scan the plant ±5 ticks around the
scheduled instant, find the first crossing of the head plane that is inside the
legal disk, locate it to **sub-tick** resolution by interpolating the axial
coordinate, and evaluate the in-plane test and the impact speed at the
interpolated pose. A strike that never crosses in-disk in the window is a genuine
**miss** — the statistic the at-tick binary conflates with "late".

### 4.1 Headline

| | champion **+ lead** | champion, no lead | phase 1 + lead |
| --- | ---: | ---: | ---: |
| executed strikes | 204 | 204 | 59 |
| **crossed the head plane in-disk** | **76 (37 %)** | 54 (26 %) | 0 |
| **miss rate** | **63 %** | 74 % | 100 % |
| latency p50 (signed; + = late) | +6.9 ms | +11.3 ms | — |
| **\|latency\| p50** | **9.5 ms** | 14.2 ms | — |
| **\|latency\| p90** | **18.1 ms** | 48.4 ms | — |
| within 10 ms | **42** | 20 | 0 |
| within 20 ms | **72** | 44 | 0 |
| within 40 ms | **73** | 46 | 0 |
| arrived early | 23 | 16 | 0 |
| **impact speed in band (1–3.5 m/s) at crossing** | **57** | 32 | 0 |
| impact speed p50 at crossing | 1.34 m/s | 1.09 m/s | — |
| in the disk at the sampled tick (the old binary) | 72 / 204 | 43 / 204 | 1 / 59 |
| axial p50 at the scheduled tick (+ = short) | +14.0 mm | +16.0 mm | +161.9 mm |
| in-plane p50 (+ = outside the disk edge) | **−47.0 mm** | +1.9 mm | +240.0 mm |
| episodes completing 250 ticks | **12 / 12** | 12 / 12 | **0 / 12** |
| PD tracking p50 / p90 | 8.0 / 37.3 mrad | 5.3 / 18.6 mrad | 18.6 / 139.8 mrad |
| temporal-consistency drift p50 | 18.8 mrad | 20.0 mrad | 24.2 mrad |
| recorded states with negative clearance | 582 / 3831 (15 %) | 1049 / 3831 (27 %) | 260 / 1079 (24 %) |
| min-SDF p10 (median over episodes) | −7.4 mm | −24.5 mm | −38.8 mm |
| wall clock | 29.7 ms/tick | 32.8 | 26.6 |

### 4.2 Latency histogram (76 and 54 crossings respectively)

| bin (ms) | [−100,−60) | [−60,−40) | [−40,−20) | [−20,−10) | [−10,0) | [0,10) | [10,20) | [20,40) | ≥40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **+ lead** | 3 | 0 | 1 | 3 | 16 | **26** | **27** | 0 | 0 |
| no lead | 1 | 1 | 0 | 2 | 12 | 8 | 22 | 2 | 6 |

The lead does two distinct things. It **removes the late tail entirely** — no
crossing later than 20 ms, against 8 beyond 20 ms without it — and it **doubles
the mass in the ±10 ms core** (42 vs 20). Both are consistent with §7c's
identification: the fitted Δ cancels the servo's *phase* lag over the whole
approach, not just at the terminal.

Two non-obvious wins:

* **the lead improves clearance**, not just timing: negative-clearance states
  fall 27 % → 15 % and min-SDF p10 goes −24.5 → −7.4 mm. Commanding ahead makes
  the plant track the *planned* (collision-priced) path more closely, so less of
  the trajectory is spent off-plan.
* **PD tracking error rises** (5.3 → 8.0 mrad p50) while everything else
  improves. That is correct and expected: the error is now measured against a
  *led* command, so a well-compensated servo is deliberately behind its command
  and on its reference. PD error is a diagnostic here, not an objective.

### 4.3 Per-hand, per-drum, per-clip, per-seed (champion + lead)

**The four eval clips contain only two of the six drums** — `hihat` (159 strikes,
left hand) and `snare` (45, right hand). The closed-loop per-drum breakdown is
therefore thin and does *not* exercise the tom or cymbal cells at all. This is a
property of the RCC grooves, not a choice; it does mean the closed loop is
weighted onto the model's **worst** open-loop drum.

| split | crossed / n | \|lat\| p50 |
| --- | ---: | ---: |
| left hand / hihat | 60 / 159 (38 %) | 9.8 ms |
| right hand / snare | 16 / 45 (36 %) | 7.4 ms |
| `gmd_eval_000001_30s` (69 strikes) | 20 / 69 (29 %) | — |
| `gmd_eval_000003_45s` (57) | 17 / 57 (30 %) | — |
| `gmd_eval_000006_30s` (39) | 20 / 39 (**51 %**) | — |
| `gmd_eval_000007_45s` (39) | 19 / 39 (49 %) | — |
| start seed 0 | 26 / 68 (38 %) | 6.6 ms |
| start seed 1 | 24 / 68 (35 %) | 8.2 ms |
| start seed 2 | 26 / 68 (38 %) | 11.1 ms |

**Strike density is what separates the good clips from the bad ones**, not the
drum: at ~3.5 strikes/s (69 and 57 strikes in 250 ticks) the horizons are short
enough that consecutive hits overlap and the shared waist is contested; at
~1.6 strikes/s the model lands half its strikes.

### 4.4 Probe noise (7c decision 6)

The three start seeds agree closely in aggregate (26 / 24 / 26 of 68), so **the
204-strike headline is stable to about ±2 %**. Individual episodes are not.

A cleaner demonstration, and a caution for anyone comparing against §7c: the
exact 7c protocol (3 clips, seed 0) reproduces **bit-for-bit** — 25/55 crossed,
miss 55 %, |lat| p50 10.7 ms, p90 23.2 ms, 12/21/24 within 10/20/40 ms, 16 in
the speed band (README §7c table, verbatim), and the no-lead baseline reproduces
at 4/55 / 93 % / 23.5 ms / 74.4 ms. The *same three clips at the same seed*
inside the 4-clip run give **18/55**, because `StartSampler.sample(4, seed=0)`
and `sample(3, seed=0)` draw different start poses. **A 3-clip / 55-strike probe
is worth ±7 crossings of noise on the start draw alone.** 7c decision 6 asked for
4–6 clips; this report uses 4 clips × 3 seeds and the headline miss rate is
accordingly 63 %, not 55 %.

### 4.5 What cost training bought, in closed loop

Phase 1 — the behaviour clone of the quintic warm start, with the same deployment
lead — **cannot run the loop at all**: all 12 episodes tripped the divergence
guard (838 of 3 000 ticks completed), 0 of 59 strikes crossed in-disk, the stick
sits 162 mm short of the head and 240 mm outside the disk at the scheduled tick,
and PD tracking is 139.8 mrad at p90. Direct cost descent is the difference
between a planner that emits plausible-looking splines and one that can be closed
around.

---

## 5. C — Generalization

### 5.1 Unseen kit randomizations

The seed-7 test grid re-run under **5 fresh `KitRandomization` draws** (per-drum
±4 cm horizontal, global ±7.5 cm horizontal / ±3.5 cm vertical — the plan's
numbers), with the layout the cases are *built* from, the layout the *scorer*
prices against, and the `DrumKit` the model's *context tokens* come from all
moved together (the 7b fix-B machinery; any one left at nominal would measure a
scene/context mismatch instead of generalization). Draw 0 is nominal, so the
deltas are paired.

| draw | max drum shift | rms shift | accepted / 243 | `standard` | `near_drum` | short cell |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (nominal) | 0 | 0 | **180** | 120 / 162 | 60 / 81 | 42 / 54 |
| 1 | 87 mm | 74 mm | 129 | 89 | 40 | 16 |
| 2 | 83 mm | 69 mm | 142 | 89 | 53 | 21 |
| 3 | 60 mm | 40 mm | 177 | 119 | 58 | 46 |
| 4 | 75 mm | 72 mm | **194** | 138 | 56 | 43 |
| 5 | 84 mm | 75 mm | 152 | 99 | 53 | 29 |
| **jittered mean** | | | **158.8** | | | |

**Acceptance falls 180 → 158.8 (−21.2 cases, −8.7 points) on average, with a
large spread: 129 to 194.** The model does not break under kit randomization — it
never loses the hit disk (135–162 of 162 on `standard`) and torque stays clean
(151–160 of 162) — but the acceptance margin is kit-dependent, and one draw is
*better* than nominal. The components that move are **collision** (92 to 142 of
162) and the **short-horizon cell** (16 to 46 of 54): a kit shifted toward the
robot leaves less room to swing, and the short horizon has no time to route
around it.

The variance is the honest headline: with only 5 draws the mean is worth roughly
±10 cases, so read this as "kit jitter costs somewhere between 0 and 20 % of
acceptance depending on the draw", not as a point estimate. It also suggests the
training kit randomization is doing its job on *reaching* but not on *clearance*.

### 5.2 Held-out rhythm clips

**Every closed-loop number in §4 is on held-out clips.** The four eval clips live
in `assets/rcc/eval/` and come from `data/rcc/generated_long_eval`; the ten
training clips come from `generated_long_train`. No eval clip was ever used to
generate a training scenario, a replay context or a rolling-eval metric. Phase
1's validation set was held out twice over — the same four clips *and* a disjoint
start-sampler seed stream.

### 5.3 Initialization robustness

Consolidated in §3.4. Four families measured — straight-line quintic warm start,
zero-motion hold, OU-noised quintic at three σ, and (in §4) the closed loop's own
warm-MPC init, the previous plan shifted one waypoint. Spread over the open-loop
families: **173–186 of 243**. The model is a refinement operator that no longer
*needs* a good refinement target.

---

## 6. D — Training cost and fit

From the recorded `metrics.jsonl` / `summary.json` only; nothing re-run.

### 6.1 The whole budget

| phase | legs | steps | wall clock | it/s | note |
| --- | ---: | ---: | ---: | ---: | --- |
| phase 1 (warm start) | 1 | 20 000 | **20.5 min** | 16.3 | BC on the quintic warm start, bf16 + compile |
| pilot race | 4 | 4 800 | 42.2 min | 1.89 | lr / AL-ρ / aux-tc, paired on data and seed |
| phase 2 (cost descent) | 1 | 7 500 | 69.8 min | 1.79 | `lr_hi`, eager (the toolchain was believed broken) |
| phase 2b aborted | 1 | 1 550 | 14.5 min | 1.78 | lr 3e-4 continuation — destroyed the model |
| **phase 2b (champion)** | 3 | 3 200 | **41.8 min** | 1.28 | lr 5e-5 + torque peak; **champion is leg 1, step 1 000** |
| phase 2c | 1 | 7 150 | 59.6 min | 2.00 | 7c objective; not the champion |
| **TOTAL** | | **44 200** | **4.14 h** | | one GPU, no DDP |

Data generation is on top and negligible: 450 k phase-1 scenarios in **65 s** on
64 fork workers; 40 k phase-2 scenarios in 6.5 s (straight-line init) or ~6 min
(with the apex IK). **The champion is reachable in ~2.2 h of GPU time** from cold
(phase 1 + phase 2 + phase 2b leg 1).

### 6.2 Phase 1 — the behaviour clone

| step | val loss | trajectory RMSE | torque RMSE | FK position RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 0.3545 | 144.6 mrad | 21.4 N·m | 72.9 mm |
| 4 500 | 0.0359 | 27.4 | 5.6 | 13.9 |
| 8 500 | 0.0183 | 17.2 | 4.1 | 8.0 |
| 12 500 | 0.0108 | 11.3 | 3.6 | 5.4 |
| 16 500 | 0.0077 | 8.8 | 3.4 | 4.3 |
| **20 000** | **0.0062** | **6.8** | **3.3** | **3.3** |

No plateau — the 20 k cap ended it (resolved decision 5 caps phase 1
deliberately). On distribution the model is a 4.8–11.5 mrad refiner and every
auxiliary head is within 1.6–10.5 % relative. On the acceptance grid it scores
**0/243**, which is not a failure: its teacher, the quintic seed, also scores
0/243 (the seed ends at rest, so its impact-speed error is ~2 m/s).

### 6.3 Phase 2 — direct cost descent, and the augmented-Lagrangian story

| step | accepted (rolling grid, 81) | hit | coll | **torque** | short cell |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 500 | 0 / 81 | 0.73 | 0.67 | 0.00 | 0 / 27 |
| 3 000 | 0 / 81 | 1.00 | 0.58 | 0.00 | 0 / 27 |
| 4 500 | 10 / 81 | 0.99 | 0.62 | 0.17 | 2 / 27 |
| 6 000 | 33 / 81 | 1.00 | 0.72 | 0.52 | 11 / 27 |
| 7 500 | 31 / 81 | 1.00 | 0.72 | 0.53 | 11 / 27 |
| **2b @ 1 000 (champion)** | **54 / 81** | 1.00 | 0.72 | **0.93** | **16 / 27** |

The priced cost *rises* over phase 2 (66 → 591) while everything it measures
improves. That is the augmented Lagrangian doing its job: the multipliers
integrate the raw violations, so the price of a constraint climbs until the
constraint is satisfied and the priced total climbs with it. Raw diagnostics on
the training distribution over the same 7 500 steps:

| | step 50 | step 7 500 |
| --- | ---: | ---: |
| contact position error | 173 mm | 120 mm |
| min SDF (clearance) | **−32.1 mm** | **+20.4 mm** |
| λ terminal / joint_box / torque / collision / velocity | 1 / 1 / 1 / 1 / 1 | 26.5 / 24.6 / **7.2** / 4.8 / 3.6 |

**λ_torque is the whole story of phase 2.** `ok_torque` sat at exactly 0.00
through step 3 000 while λ_torque was ~2, then went 0.17 → 0.52 as λ passed 4. It
reached **0.93** only in phase 2b, after 7b added a `pnorm` smooth-max **peak**
term alongside the mean — the harness gate is a zero-tolerance *max* over ~850
(knot, joint) elements and the median failing case has **one knot of 30** over the
limit, which moves a mean by 0.3 %.

The champion's own leg, for the record (lr 5e-5, torque peak ramped 0→1 over
2 000 steps): cost 254 → 283, contact 101 → 99 mm, min SDF +2.7 → +14.2 mm,
`torque_hinge_max` 13.6 → 16.6, λ essentially flat — a *stable* continuation, in
contrast to the aborted lr 3e-4 leg where the cost went 238 → 2 494 monotonically
and `torque_hinge_max` 13.9 → 83.0 (the model got worse on the very statistic it
was being punished for — an unstable optimizer, not an over-weighted term; zero
non-finite steps were skipped).

### 6.4 Phase 2c — the control experiment

Resumed from the champion with everything 7c added (decoupled `torque_peak` AL
group, final-approach oversampling, chained BPTT live from step 0, the roller
executing the same lead). 7 150 steps. Its rolling grid peaked at **62/81**
(better than the champion's 54/81) and its closed-loop probe peaked at 10/42
against the incumbent's 15/42. On the full grids it gained **+7 on seed 0** and
lost **−8 on seed 7**; in closed loop it landed 8/55 against the champion's 25/55
on the same protocol.

**This is the most useful negative result in the project.** It is a clean
demonstration that *more descent on a purely kinematic cost does not buy
closed-loop strikes* — and the probe gate caught it: at step 2 700, the best
rolling grid of the whole run, `best.pt` was **not** overwritten, because that
checkpoint's closed-loop probe had collapsed to 2/42. Section 7b shipped exactly
that trade blind.

---

## 7. E — Honest limitations

**1. The objective and the acceptance grid are purely KINEMATIC, and that is the
binding limitation of the whole project** (§7c decision 1). Both score waypoints.
Neither has ever asked whether a `kp = 40` position servo can *be* there.
Measured on the champion, 54 `standard` plans rolled through the exact PD model
with no replanning:

| | plan (kinematic) | plant (through the PD) |
| --- | ---: | ---: |
| axial at the terminal knot, p50 | +9.2 mm | **+143.0 mm** |
| in the hit disk | **54 / 54** | **1 / 54** |
| max joint tracking error at the strike | — | **152 mrad** |
| tip speed at the strike, p50 | 2.01 m/s | 1.61 m/s |

A plan that is kinematically on the head executes ~140 mm short; a plan that
executes on the head is kinematically ~140 mm *through* it, which the grid scores
as both a miss and a collision. **The two metrics are in direct conflict on this
servo**, and every number in §3 lives on the kinematic side of it. The plant
reaches `min |axial| = 9.7 mm` only if the terminal command is held, and 30 of 54
are still converging 8 ticks later — a bandwidth limit, not a transport delay.
Closed-loop replanning recovers most of it (143 mm → +14.0 mm at the scheduled
tick with the lead), because a one-waypoint step from the measured state is a
much smaller demand.

**2. The closed-loop miss rate is 63 %, and 57 of 204 strikes land at a legal
impact speed.** As a listener would hear it: **about one beat in three sounds,
and about one in four sounds at a musically usable velocity.** The ones that do
sound are well-timed (p50 9.5 ms, p90 18.1 ms) and none is more than 20 ms late.
This is not yet a drummer. §7c reported the same system as 55 % miss / 16-of-55
in band on a 3-clip probe; the honest number over 4 clips × 3 seeds is worse.

**3. The both-arms collision result is a regression against expectation, not
merely a shortfall.** 57–68/243 against v2's 83/243, from a planner that moves
both arms and should not have to choose. Root cause identified (the idle arm has
no terminal objective, §7c design note), fix specified, **not implemented**. On
`near_drum` it is 1–6 / 81. Any deployment that cares about the idle arm should
treat this as open.

**4. Probe noise.** A 3-clip / 55-strike closed-loop probe moves by ±7 crossings
on the start draw alone (§4.4). Two similar checkpoints cannot be ranked with it;
it is good enough to block a bad one. The 12-episode / 204-strike measurement in
this report is stable to ~±2 % across start seeds, but its per-clip and per-drum
splits are not.

**5. Chained BPTT (phase 3) was never A/B'd.** `bptt_start_frac = 0.55` of a
21 000-step budget put it at step 11 550 and the champion is at step 1 000, so
**every number in §3 and §4 is from a model trained without the phase-3 term.**
Phase 2c ran it live from step 0 and degraded, but that run changed four things
at once (§7c decision 7) — whether BPTT-from-step-0 is a cause or a bystander is
unmeasured. The machinery is implemented and tested (the second solve's cost
produces a finite, non-zero gradient in the first solve's decoder trunk) and
unused.

**6. The idle arm is a design note only.** `w_idle_home`, `idle_home_table`, the
yaw-indexed home-pose construction and its verification protocol are specified in
README §7c but not built. The roller also *deletes* both-arms-idle stretches (it
respawns a slot when no hand is active) and the fresh pool drops single-hand-idle
scenarios, so the term would currently train off-distribution; a rest-fraction
knob is a prerequisite.

**7. The per-joint lead is a fit to one servo model** (§7c decision 3). Its 17
numbers come from `scripts/fit_cmd_lead.py` against `BatchedPDSim` at the runtime
Kp/Kd with the real coupled mass matrix. If Isaac's actuator model differs, or if
the RL fine-tune changes the effective stiffness, **the vector must be re-fitted,
not carried over** — it is ~15 s of GPU time. A lead cancels *delay*, not
*bandwidth*; on the tracking-RMS objective a uniform lead buys 2 %, which is why
every uniform-delta knob plateaus at 5–10 of 55.

**8. The champion is not converged.** It is step 1 000 of a continuation of a
7 500-step run (§7b decision 3), selected as the best point of a search that ran
out of window. Legs 2 and 3 both regressed at either torque-peak weight, and the
recommended structural change — giving `torque_peak` its own AL group — was
implemented in 7c but only exercised inside a run that degraded for other
reasons.

**9. `best.pt` selection has a history.** Phase 2's `best.pt` was selected on a
grid acceptance that was degenerate at 0 (decision 10); phase 2b's is per-leg and
can be overwritten by a worse continuation (7b decision 10 — the champion is
preserved by hand as `leg1_best.pt`, byte-identical to `best.pt`, md5-verified
for this report). Phase 2c's probe gate uses a `score` whose formula changed
mid-chapter, so its recorded probe scores are not comparable across the run (7c
decision 10).

**10. Smaller things, carried forward.** `n_sub = 1` in the cost means the
collision term's `min_sdf` can differ from the scorer's by up to 4.6 cm (cost
decision 5). `physics_mode = "knot_fd"` prices torque at the knots with finite
differences rather than the spline's own accelerations (cost decision 1).
Restoring `w_smooth_jerk` broke trajopt's 10:1 acc:jerk ratio and was not A/B'd
(7b decision 5). The waist seed on a both-hands re-goal is a hold, not the aim
law (7b decision 9). The freeze window holds the waist for the striking arm while
the other arm replans against it — a coupling nobody has priced (7c decision 9).
`sticky_goal_point` changed data generation as well as evaluation, so any dataset
regenerated after 2026-08-11 differs from one before (7c decision 4). No DDP
anywhere (§7b/2). The eval clips exercise only 2 of 6 drums (§4.3).

**11. Nothing here has touched Isaac.** The PD model, the bounce (restitution
U(0.3, 0.6), tangential tip velocity exactly preserved, plus Gaussian noise) and
the contact are all *documented placeholders* for Isaac's solver.

---

## 8. Recommendation — is this ready for the IsaacSim RL fine-tune?

**Yes, with one precondition and a specific priority order.**

The case for going: the planner is fast enough by a wide margin, it is stable
(12/12 episodes, 0 divergences, kit-jitter-robust on reaching,
initialization-insensitive over a 5 % band), and it is *better than the expert it
replaces* on the offline objective. The residual failure has been localized — and
localized to the one thing an offline kinematic pipeline structurally cannot fix.
Phase 2c is the control experiment: 7 150 more steps of the same kind of descent
moved the validation grid +7, the test grid −8, and closed-loop strikes 25/55 →
8/55. **There is no more to extract from a kinematic offline objective.**

### The precondition: the RL objective must score the EXECUTED state

Cite §7c decision 1. The plan already asks for this implicitly ("a term… for
reducing the pd controller error… keeping in mind that we have added a lag
compensation"), but the stronger statement is required: **the reward must
evaluate the terminal geometry at the pose the plant reached, not the pose the
plan nominated.** In simulation this is free — Isaac *is* the plant. Offline the
same change is a genuine objective conflict (the executed terminal wants the plan
to overshoot the head; the plan terminal and the grid want it on the head) and
deserves its own chapter and A/B; in Isaac the conflict simply disappears,
because there is only one terminal and it is the real one.

Concretely: keep every existing analytic term (torque, velocity, joint box,
collision, smoothness, posture, and the auxiliary heads — the plan is explicit
that these "should have continued throughout the entire thing") as a
differentiable loss on the *plan*, and add the hit reward on the *executed*
contact. The differentiable half keeps the un-executed tail of the trajectory
valid; the RL half carries the signal backprop cannot see.

### Priority order for the RL stage

1. **Executed-state hit reward, mixed with the existing analytic cost.** This is
   the precondition above and it is where the expected gain is. Target: the §4
   miss rate, 63 %, measured the same way (sub-tick latency, not the at-tick
   binary). Anything that does not move that number is not progress.

2. **Contact and bounce, for real.** The bounce in `train_cost.bounce` is a
   restitution placeholder. Real contact changes the problem qualitatively: the
   stick's post-hit state is what the *next* plan starts from, and the
   chained-BPTT term — implemented, tested, never actually trained (§7 E.5) — is
   the offline stand-in for exactly this. Put contact parameters (restitution,
   friction, head compliance) **in the domain randomization**, as the plan's own
   failure-mode list requires, or distillation overfits to simulated bounce
   quirks.

3. **Lag randomization.** The plan is explicit: "a lag will also be added to
   IsaacSim as a randomized domain variable for lag robustness… this will not be
   added to our trajectory planner as it obviously doesn't know the actual amount
   of lag." The per-joint lead is the *control-side* compensation and it is a fit
   to one servo model (§7 E.7). Two things must hold: the lead the training loop
   executes must equal the lead the deployed loop executes
   (`Phase2Config.cmd_lead_s` already drives the roller, the temporal-consistency
   plant and the BPTT plant for exactly this reason), and the *residual* lag
   around it must be randomized so the planner learns lag-robust trajectories
   rather than leaning on an exact Δ. Re-fit the vector against Isaac's actuator
   before the run — 15 s of GPU time.

4. **The idle arm.** Cheap, specified, and the one place the model is *worse*
   than the expert (§3.1). Either implement `w_idle_home` before the RL run, or
   accept that the fine-tune inherits a 57/243 both-arms number and add an
   idle-arm clearance term to the reward instead. The roller's rest-fraction
   problem must be fixed either way, or the term trains off-distribution.

5. **Short-horizon competence, `h ≤ 4`.** The one genuinely
   procrastination-shaped residual: the plan is inside the disk down to `h = 5`
   and walks 60 mm out over the last three refinements. Final-approach
   oversampling (`short_wp = 8, short_boost = 3.0`) exists and is logged as
   `batch_short`; it should be on for the fine-tune, and the 1–5-waypoint horizon
   should be evaluated explicitly, which neither the 243-case grid (shortest
   N = 22) nor the pool covers densely.

6. **Exploration: parameter-space noise, as the plan specifies** — low-rank
   (LoRA / FiLM) deltas on the decoder only, one ε per hit segment, adaptive σ
   measured as tip deviation at the hit, self-imitation of the winning
   perturbations with the analytic cost mixed in. §3.4 is direct evidence for the
   plan's own argument against output-space noise: the model is now so
   initialization-insensitive (173–186/243 over a 6× range of init noise) that
   additive trajectory noise would be denoised away within one refinement.

**What NOT to prioritize.** More kinematic descent (phase 2c settles it). More
refinement iterations (×1 ties or beats ×3 at every seed and every init). The
goal-switch hypothesis (7b fixed the roller's re-goal and it did not move the
closed loop; 7c showed the residual is axial lag plus a short-horizon in-plane
gap, not a goal switch). The executed-prefix / approach-envelope cost term
(shelved with evidence in §7c — the planned terminal is axially correct at
*every* horizon, so there is no axial deferral for an envelope to catch).

**Two hygiene items before starting.** Move the rolling eval off seed 0 (§7c
decision 5 — selection on it is worth ~8 cases of transfer, as phase 2c
demonstrated), and select `best.pt` on a metric that degrades gracefully and is
gated on the closed-loop probe (already implemented — use it).

---

## 9. Reproduction

Environment (mandatory on this box):

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export UNITREE_MODEL_DIR=/data2/eric/workspace/unitree_model
export TMPDIR=/data2/eric/tmp
cd amortized
```

```bash
# A -- open-loop grids, one per seed (0 = validation, 7 / 13 = test)
for s in 0 7 13; do
  python scripts/eval_phase2.py --ckpt runs/phase2b_main/best.pt --device cuda:1 \
      --grid-seed $s --skip-closed-loop --out runs/final/grid_seed$s
done
python scripts/report_breakdown.py --dirs runs/final/grid_seed0 \
    runs/final/grid_seed7 runs/final/grid_seed13 --out runs/final/breakdown.json

# B -- closed loop, 4 clips x 3 start seeds, champion / no-lead / phase 1
python scripts/eval_section8.py --device cuda:1 --steps 250 --clips 4 \
    --probe-seeds 0 1 2 --out runs/final/closed_loop.json
#    the 7c-comparable 3-clip / seed-0 protocol, for continuity
python scripts/eval_section8.py --device cuda:1 --clips 3 --probe-seeds 0 \
    --out runs/final/closed_loop_7c_repro.json

# C -- generalization
python scripts/eval_kit_jitter.py  --device cuda:1 --grid-seed 7 --draws 5 \
    --out runs/final/kit_jitter.json
python scripts/eval_init_robust.py --device cuda:3 --grid-seed 7 \
    --out runs/final/init_robust.json

# D -- the training story, from the recorded metrics only
python scripts/report_training.py --out runs/final/training.json

# F -- videos
python scripts/render_mpc.py --ckpt runs/phase2b_main/best.pt --clip-index 2 \
    --steps 250 --lead perjoint --out runs/render/champion_lead_best_clip6.mp4

# the test suite
python -m pytest tests -q        # 329 passed, 1 skipped (12 m 41 s, cuda:3)
```

### Artifacts

| what | where |
| --- | --- |
| open-loop grids, per seed | `runs/final/grid_seed{0,7,13}/report.json` + per-case CSVs |
| per-drum / per-cell breakdown | `runs/final/breakdown.json` |
| closed loop, 204 strikes | `runs/final/closed_loop.json` + `closed_loop_*_rows.json` (raw per-strike) |
| closed loop, 7c protocol | `runs/final/closed_loop_7c_repro.json` |
| kit randomization | `runs/final/kit_jitter.json` |
| initialization robustness | `runs/final/init_robust.json` |
| training story | `runs/final/training.json` |
| test-suite log | `runs/final/pytest.log` |
| **videos** | `runs/render/champion_lead_best_clip6.mp4` · `champion_lead_worst_clip1.mp4` · `champion_nolead_clip6.mp4` |
| champion checkpoint | `runs/phase2b_main/best.pt` (= `leg1_best.pt`, md5 verified) |
| expert reference (read-only) | `baselines/trajopt_hpipm_2026-08-10*` |

New scripts added for this evaluation: `scripts/eval_section8.py` (closed loop
with raw per-strike rows and breakdowns), `scripts/eval_kit_jitter.py`,
`scripts/eval_init_robust.py`, `scripts/report_breakdown.py`,
`scripts/report_training.py`. Nothing under `amortized_opt/`, `tests/` or
`baselines/` was modified.

The expert is **never re-solved** — its recorded per-case CSVs are read off disk.
Re-solving would confound a metric change with solver noise (about a third of its
cases move run to run) and costs 45 min; the fresh-seed transfer claim in §3.1 is
tested on the *model*, which is deterministic, and that is why the expert has no
seed-7/13 column.
