# `isaac_rl` — the IsaacSim RL fine-tune, stage RL-1 (scaffold, no training)

> **The public interface contract lives in [`INTERFACES.md`](INTERFACES.md).**
> Read that first if you are consuming these modules. This file is the build
> log: what was built, what was carried over from the main repo's env and what
> was dropped, what was measured, and what still needs a decision.

This is the **only** part of `amortized/` allowed to import Isaac Lab or
`source/g1_drummer`. The standalone core (`amortized_opt/`) stays Isaac-free and
was **not modified**; `isaac_rl` imports *from* it (model, tokenizer, cost,
lead, strike geometry, probe).

```
isaac_rl/
  env_cfg.py    the rate constants (250 Hz / dec 5) + the knob dataclass   [Isaac-free]
  env.py        the Isaac plant: scene, contact sensors, FOH, ff, plant()  [Isaac]
  schedule.py   batched GPU rhythm scheduler -> ContextInputs + StrikeContext
  mpc.py        the model as the MPC: batched plan, lead, clamp, re-seed
  reward.py     executed-state reward + the analytic half + tracking
  perturb.py    per-env FiLM / LoRA parameter-space noise + adaptive sigma
  safety.py     batched command projection with intervention counting
  loop.py       one control tick, assembled; the plant is injected
  measure.py    hit latency, via amortized_opt.probe.hit_latency verbatim
  scripts/      smoke_rl.py · ref_loop.py · fit_lead_isaac.py
  tests/        the Isaac-marked suite (not collected by pytest.ini)
```

Everything except `env.py` is Isaac-free and unit-tested on CPU.

---

## 1. Why this is not a subclass of `G1DrummerEnv`

`G1DrummerEnv` is 6 374 lines. Almost all of it serves an MLP policy whose
action is 17 scalars: the observation assembly (base block, seam block, torque
budget block, per-hand lookahead / field-gradient / credit), AMP, the curriculum
machinery, the action decoder with `absolute` / `delta_vel` modes and its soft
clip, and twenty reward terms. Here the policy is the amortized optimizer and
its action is a whole 50 × 17 trajectory, so none of that applies, and
inheriting it would mean carrying six thousand lines of dead configuration that
can still fire.

What *is* carried is everything about the **plant**, because the per-joint
reference lead and the entire phase-2 training distribution were identified
against it.

### Env-variant knob table

| knob | this env | main repo | disposition |
| --- | --- | --- | --- |
| `sim.dt` | **1/250** | 1/120 | **CHANGED** — resolved decision 1 |
| `decimation` | **5** | 2 | **CHANGED** — resolved decision 1 |
| `step_dt` | **0.02 (50 Hz)** | 1/60 | **CHANGED** — now exactly the waypoint grid |
| rate constants live in | `isaac_rl/env_cfg.py` | `control/control_rate.py` | **FORKED** (see §2) |
| `PhysxCfg(enable_ccd)` | True (inert, see §3) | True | carried |
| physics material | `restitution=0`, combine `max`, friction `max` | same | carried |
| solver iterations | 8 pos / 4 vel (upstream) | same | carried |
| `robot_cfg` | `get_g1_cfg(ideal_pd, Kp×1, Kd×2, frozen legs)` | same | carried |
| drum kit | `simple_drum_kit_6drums.usd`, per-env spawn loop | same | carried |
| contact sensors | 2 per-hand, per-drum filters in drum-id order | same | carried |
| `pin_pelvis` | True, every substep | True | carried |
| FOH position ramp + velocity target | yes | yes | carried |
| gravity / Coriolis feedforward | yes, vendored `BatchedRNEA`, `qdd = 0` | RNEA with optional inertial term | **carried, narrowed** (§5) |
| `foh_vel_target_ema` | none (raw `(cmd−prev)/dt`) | 0.5 | **DROPPED** (§5) |
| action space | trajectory (50 × 17) | 17 scalars | **REPLACED** |
| `action_scale` / `action_delta_scale` / `_soft_clip` / `dr_action_latency` | none | present | **DROPPED** |
| observation space | `ContextInputs` (encoder tokens) | 68 + blocks | **REPLACED** |
| lookahead / field-grad / credit obs | none | present | **DROPPED** |
| hit detection | analytic sub-tick crossing (`hit_mode="geometric"`) | PhysX contact + geometric band | **REPLACED** — DECISION 1 |
| extra-hit punishment | `reward.extra_hit` + `early_contact` | `extra_reward` with telescoping refund + escalation | **carried, reshaped** |
| timing window | `exp(−(latency/20 ms)²)` on the sub-tick crossing | `exp(−(dt/σ)²)`, σ annealed 0.40→0.01 s | **REPLACED** (§4) |
| speed band | `[1.0, 3.5] m/s` indicator | gamma kernel about `beat_velocity × 2.0` | **REPLACED** (§4) |
| analytic cost terms | `FusedCost` on the plan, negated | `safety_sdf` + `joint_limit` hinges | **REPLACED** — plan doc requires the full set |
| QP safety filter | always on, interventions counted | eval/`play.py` only | **PROMOTED** (§6) |
| curriculum / AMP / PPO | none | present | **DROPPED** |
| `regularizer_norm_steps` (300) | n/a | present | **DROPPED** — see hazard (8) below |
| `reward_action_rate` | n/a | −5.25 | **DROPPED** — see hazard (9) below |

The two DROPPED regularizers are worth naming explicitly because they are the
main repo's most rate-sensitive terms: at 50 Hz a given physical motion needs
~44 % larger per-step action deltas while the fixed divisor stays 300, so
`reward_action_rate` would have been silently re-weighted by ~1.7× against
`reward_correct`. This stack has neither term (smoothness is priced by
`FusedCost`, in physical units, per unit *time*), so the hazard does not apply —
but it is exactly the kind of thing that would have made a naive rate change
look like a training bug.

---

## 2. The rate, and why the constants are forked

`control/control_rate.py` is imported by the offline AMP/teacher/BC generators
*and* by `qp_filter.SafetyFilter`'s default `dt`. Mutating it to 250 Hz would
silently reshard every dataset in the main repo. So `isaac_rl/env_cfg.py` owns
its own `SIM_DT = 1/250`, `DECIMATION = 5`, and guards the fork with
`assert_rate_consistency()`, which pins `SIM_DT * DECIMATION` to
`amortized_opt.CONTROL_DT`. That identity — **one waypoint = one control tick** —
is assumed by every horizon in the stack and would raise nowhere if broken.

---

## 3. 250 Hz in Isaac: what actually changed

Isaac reports the requested rate exactly:

```
Physics step-size     : 0.004
Rendering step-size   : 0.02
Environment step-size : 0.02
```

Two findings from the first live run:

* **`enable_ccd=True` is inert on the GPU pipeline.** Isaac logs
  `CCD is not supported on GPU, ignoring request to enable it`. So the main
  repo's CCD-instead-of-a-finer-timestep argument was never actually in force
  on GPU, and at 1/250 a 3 m/s tip travels 12 mm — inside the tip's own 14 mm
  radius — so the tunneling motivation is gone regardless. Left declared, so
  the CPU pipeline behaves the same.
* the `hit_cooldown_steps`, `dr_action_latency_max_steps` and
  `perturb_hold_steps_*` tick-denominated buffers of the main repo are all
  DROPPED here, so their 20 % wall-clock reinterpretation is moot.

---

## 4. DECISIONS NEEDING REVIEW

### DECISION 1 — the two hit detectors disagree, and I did not silently pick one

The brief anticipated this: *"if the existing hit detector's semantics conflict
with the analytic strike geometry"*. They do, in three places:

| | repo `GeometricHitDetector` + `ContactStrikeTracker` | `amortized_opt.strike` (what the cost prices) |
| --- | --- | --- |
| legal radius | `radius − 0.02` (membrane) / `radius + 0.01` (cymbal edge) | `strike_aim_on_head` with `rim_buffer = 0.01`, plus a **disk-centre offset** on cymbals and toms |
| half-plane | **none** | `strike_half_plane_dir` — a tom/cymbal strike is only legal on the near semicircle |
| axial | band `[−0.02, +0.017]` around the head, plus a PhysX contact force `> 1 N` and an incidence gate (`tangential ≤ 1.5 × speed`) | a plane crossing, located sub-tick |
| shaft | probes 1.. allowed only on `shaft_hit` drums, 4 samples over 0.20 m | `contact_sigma` as a continuous arc coordinate, with the speed requirement applied **at the contact point** |

They are not reconcilable by tuning. **What I built:** `hit_mode="geometric"`
(default) scores the executed crossing with the *analytic* geometry, so the RL
half and the differentiable half agree about what a hit is, and so the Isaac
numbers are directly comparable to `REPORT.md` §4. `hit_mode="contact"` exists
and gates the same crossing on the PhysX contact sensors.
**Needs a decision:** which is the training objective. My recommendation is
`geometric` for the reward and `contact` logged alongside as a diagnostic, with
the disagreement rate reported — because the analytic geometry is what the
half of the objective that has gradients prices, and an objective whose two
halves disagree about the target is the failure mode §7c already paid for once
(the kinematic grid and the executed terminal in direct conflict).

### DECISION 2 — context re-encoding every tick vs caching per hit segment

The brief asks for *"context-encoder caching per hit segment"*. That was not
free here: `goal_tth` and `future_tth` were **inputs to the encoder** and both
ticked down every 20 ms, so caching fed the decoder a stale time-to-hit. This is
precisely the bug `MPCRoller._tick_tth` exists to prevent
(*"without this every harvested replay row tells the model a stale time-to-hit
while the mask says something else — a silent distribution bug"*). I defaulted
to `encode_every_tick=True` (correct) and left the cache as a flag. The encoder
is 2 layers at d=256 over ~16 tokens against a 6-layer d=384 decoder over 51
tokens, so the saving is small; the measured share is in the report.

**Resolved by making the encoder's inputs segment-constant instead.** The
per-tick re-encode was not merely a cost: it is what let the robot's own state
echo back into the encoder every tick and run the waist away in closed loop.
`future_tth` is now `future_gap` — the gap from the CURRENT goal's strike to
each future hit, which does not move inside a goal segment (and is the
musically meaningful quantity: how long the hand has to relocate after this
hit). `goal_tth` was the remaining from-now input and has now moved to the
DECODER: it rides `WaypointMasks.hit_tth` into the state token
(`ModelConfig.dec_tth_feats`) and the goal token stops embedding it
(`enc_goal_tth=False`). Old checkpoints and pools are incompatible by design;
the pool cache's `CONTEXT_FIELDS` schema marker refuses them loudly.

With that, every encoder input is constant inside a goal segment by
construction, so `MPCConfig.encode_every_tick` now defaults to **False** and
`IsaacMPC._encode_key` rebuilds the encoding on a segment boundary (per-hand
cursor + goal activity, so a re-goal and a horizon-cap promotion both count), on
a reset, or when the `yaw_cond` conditioning yaw has moved past `enc_yaw_tol`
(0.05 rad) — the one remaining slowly-varying input. `True` stays as the A/B
escape hatch.

### DECISION 3 — the inertial feedforward term

The repo's env can add an inertial feedforward, `RNEA(q, q̇, q̈_cmd)`, driven by
the commanded acceleration. Here the command *is* a trajectory, so `q̈_cmd` is
available and better-conditioned than the repo's EMA. I left it **out**
(`q̈ = 0`, gravity + Coriolis only), because including it makes the Isaac plant
a different operator from `amortized_opt.pd_sim` — the model the per-joint lead
was fitted against and the plant the whole phase-2 temporal-consistency term was
trained against. **Needs a decision:** an A/B, not a default.

### DECISION 4 — the executed-state reward drops the kinematic terminal

`RewardConfig.plan_terms` excludes `terminal_*` from the analytic half. §7c is
explicit that a plan which executes on the head is kinematically ~140 mm
*through* it, which the kinematic terminal scores as a miss **and** a collision.
Keeping both terminals re-creates that conflict inside a single reward.
**Needs a decision:** confirm, or keep a small terminal weight as a regularizer.

---

## 5. Deviations from the brief, stated plainly

* **`foh_vel_target_ema` is not carried.** The repo EMAs the velocity target
  with `a = 0.5`, a pole tuned to give one-step group delay at 60 Hz; at 50 Hz
  the equivalent is `a ≈ 0.545`. `amortized_opt.pd_sim` — the plant the lead was
  fitted against — models the *unfiltered* FOH rate target, so carrying the EMA
  would introduce a discrepancy the offline stack does not have. Dropped, and
  flagged here rather than silently retuned.
* **The QP filter is promoted from eval-only to always-on.** The plan doc says
  the QP correction *"will not be included in the IsaacSim RL training as it's
  supposed to be an option of last resort"*. The brief overrides this
  (*"the training backstop with intervention counting"*), so `safety.py` ships
  a fast batched **box + SDF** projection as the training default and the real
  OSQP QP as an opt-in mode; the plan's intent is preserved by making the
  expensive last-resort solver opt-in rather than by removing the backstop.
* **`num_drums`**: the layout has **7** drums (`snare, hihat, tom1, tom2,
  cymbal_left, cymbal_right, floor_tom`), padded to the model's 8.

---

## 6. Running it

```bash
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export UNITREE_MODEL_DIR=/data2/eric/workspace/unitree_model
export TMPDIR=/data2/eric/tmp
cd amortized

# the RL-1 smoke test (Isaac)
python -u isaac_rl/scripts/smoke_rl.py --device cuda:1 --envs 64 --steps 250 \
    --out runs/rl1/smoke.json

# the control experiment: the SAME loop against the analytic plant
python isaac_rl/scripts/ref_loop.py --device cuda:1 --envs 256 --steps 250 \
    --out runs/rl1/ref.json

# re-fit the per-joint lead against Isaac's actuator (RL priority 3)
python isaac_rl/scripts/fit_lead_isaac.py --device cuda:1 \
    --out runs/rl1/lead_isaac.json

# tests
python -m pytest tests -q                          # the standalone suite, unchanged
python -m pytest isaac_rl/tests -q -m isaac --isaac-device cuda:1
```

`torch.cuda.set_device` is called in every entry point (the package rule).
GPUs 0 and 2 are held by concurrent sessions on this box; this work uses
**`cuda:1`**.

---

## 7. RL-1 smoke-test results (2026-08-11, `cuda:1`)

`runs/rl1/` holds every artifact: `smoke.log` / `smoke.json`,
`smoke.rollout.npz`, `latency_isaac.json`, `deviation_safety.json`,
`ref_256.json`, `repo250.log`, `cfgdiff.log`.

### The headline: Isaac hit latency vs the analytic 63 %

32 envs x 200 ticks, champion + per-joint lead, sigma = 0, **528 scheduled
strikes** — measured by `amortized_opt.probe.hit_latency`, the same function
that produced `REPORT.md` §4.

| | analytic plant (`REPORT.md` §4) | **ISAAC (this run)** |
| --- | ---: | ---: |
| scheduled strikes | 204 | **528** |
| crossed in disk | 76 | **158** |
| **miss rate** | **63 %** | **70 %** |
| \|lat\| p50 | 9.5 ms | **17.1 ms** |
| \|lat\| p90 | 23.2 ms | **55.5 ms** |
| ≤10 / ≤20 / ≤40 ms | — | 32 / 112 / 142 |
| early | 7/25 | **158 / 158** |
| speed in band | 16/25 | **144 / 158**, p50 2.05 m/s |
| histogram (ms) | asymmetric-LATE | `[-100,-60):16 [-40,-20):30 [-20,-10):80 [-10,0):32`, **nothing ≥ 0** |

**Read the `early` row, not the miss rate.** On the analytic plant the
distribution was *asymmetric-late* — that is the fact §7c was built on, and the
per-joint lead was fitted to cancel exactly that lag. In Isaac **every single
one of the 158 crossings is early**, and the histogram has no mass at or after
the beat at all. The lead is not under-compensating here, it is
**over-compensating**: the Isaac actuator is faster than
`amortized_opt.pd_sim`, so a Δ fitted against the analytic servo throws the
strike ahead of the beat. §7c already showed what over-compensation looks like
(`per-joint + goal-shift`: all 23 crossings early, miss rate *up*).

So the +7 points of miss rate is **not** evidence that Isaac is harder. It is
evidence that the shipped lead is the wrong lead for this plant, which is
precisely what RL priority 3 predicted and asked to be fixed first
(*"Re-fit the vector against Isaac's actuator before the run"*). The speed
column is the corroboration: 144 of 158 crossings land in the legal impact
band at p50 2.05 m/s, against 16 of 25 analytically — the strikes are
*physically good*, they are just early. `isaac_rl/scripts/fit_lead_isaac.py`
is written and is the first thing to run in the training stage.

**Caveat, stated up front:** this run had the gravity/Coriolis feedforward OFF
(Pinocchio cannot build the RNEA inside the Isaac process — see
`env.try_build_rnea`), so the servo carried gravity with position error. That
makes the plant *slower*, not faster, so it does not explain the earliness; but
the number should be re-taken with the feedforward on before it is quoted as
final.

### Throughput

| plant | envs | env-steps/s | realtime | plan / plant / reward share |
| --- | ---: | ---: | ---: | --- |
| Isaac (250 Hz, dec 5) | 32 | 219 | 0.14x | 23 % / 31 % / 25 % |
| analytic `BatchedPDSim` | 256 | 3 737 | 0.29x | 32 % / 45 % / 23 % |
| repo env, forced 250 Hz | 4 | 9 | 0.04x | — |

**The ≥256-envs-at-realtime target is NOT met** and the model is not the
reason: inference is under a quarter of the tick at 32 envs and would amortize
further with batch. The cost is PhysX at 5 substeps of 4 ms plus per-tick
Python. The repo's own env at the same rate is *slower still* (9 env-steps/s at
4 envs), so this is the rate's cost, not this layer's. Levers not yet pulled:
more envs (the shares say it should scale close to linearly), `torch.compile`
on the decoder, and removing the remaining host syncs in `_reseed` / the freeze
branch.

### Deviation diagnostic — **the subspace is NOT where the plan predicted**

Clean vs perturbed plan from the *same* state and context (32 envs):

| mode | σ | rms | waist | shoulder | elbow | wrist | contact-adjacent | roughness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FiLM | 0.02 | 23.0 mrad | 6 % | **50 %** | 6 % | 38 % | **11 %** | x1.00 |
| FiLM | 0.05 | 57.4 mrad | 6 % | **50 %** | 6 % | 38 % | **11 %** | x1.01 |
| LoRA r=8 | 0.02 | 53.6 mrad | 11 % | 40 % | 13 % | 35 % | 9 % | x1.02 |

Two of the three things the plan asked for hold, one does not.

* **Smooth: yes.** Roughness (mean \|second difference\| along the waypoint
  axis) is x1.00–1.02 of the clean plan at every σ. Parameter-space noise does
  what the plan claimed additive output noise could not — a perturbed θ′ still
  emits smooth, PD-trackable trajectories.
* **Diverse: yes.** Per-env spread 0.63 rad at σ=0.02, one shared forward pass.
* **Wrist-heavy and contact-adjacent: NO.** The plan predicted the achievable
  deviation should concentrate at the wrists and near contact. It is
  **shoulder-dominated** (50 %), and the contact-adjacent share (11 %) is
  indistinguishable from the fraction of the grid that window occupies (12 %) —
  i.e. the deviation is **uniform along the trajectory**, exactly the
  "all mid-swing" failure the plan named. LoRA r=8 spreads slightly toward the
  waist/elbow but does not fix the temporal concentration.

The plan's own remedy is *"widen rank if the deviation diagnostic looks
wrong"*, but the diagnostic says the problem is **temporal**, not rank: a
per-layer or per-site perturbation is time-invariant by construction, so it
cannot preferentially express itself near contact. See DECISION 5.

### Safety filter

`mode="box"`, B=32: engaged on **100 %** of injected bad plans (both the
out-of-box half and the huge-slew half), **0 %** on a clean command, output
provably inside the joint box and finite. Mechanism counters separate box (32)
from slew (32) from SDF (0). **0 interventions** across the 200-tick champion
rollout — the planner's own clamp chain already keeps it admissible, which is
the right baseline for reading intervention counts as a penalty later.
Measured 8.9–9.4 ms/call at B=256 with the 2 compiled clearance steps,
0.59 ms with `sdf_steps=0`.

### DECISION 5 — the perturbation subspace is time-invariant

The deviation diagnostic above says the FiLM/LoRA subspace spreads its effect
evenly over the horizon while the residual the RL stage must fix is
contact-local. Options, none of which I picked unilaterally:
(a) accept it — episodic returns still credit the whole segment;
(b) gate the FiLM vector on the decoder's *time-to-hit* Fourier feature, so the
perturbation is a function of τ and can concentrate near contact;
(c) perturb only the last decoder blocks (`NoiseConfig.blocks`), which the
diagnostic can test in one run.
(b) is the one that matches the plan's intent, and it is a ~20-line change to
`FiLMPerturbation`. It should be decided before σ is tuned, because σ
adaptation on a subspace that cannot express the right thing will simply grow
σ until the trajectory degrades.

---

## 8. RL-1b — the RNEA cache, the Isaac-fitted lead, and the re-measurement

`runs/rl1b/` holds everything: `fit_lead.log`, `lead_isaac.json`,
`smoke_ffon_analytic.{log,json,rollout.npz}`,
`smoke_ffon_isaaclead.{log,json,rollout.npz}`.

### 8.1 The RNEA cache closes the gap RL-1 left open

`isaac_rl/rnea_cache.py` freezes the kinematic tree to a committed 5 KiB
`.npz`, so Isaac no longer needs Pinocchio. **Verified bit-exact**: max
`|tau_live − tau_cached|` = **0.0e+00** over 512 random states, and 0.0e+00 for
the gravity-only term. Confirmed live in-env:

```
[isaac_rl] RNEA from cache (RuntimeError on the live build)
[isaac_rl] gravity/Coriolis feedforward ON (BatchedRNEA available in-process)
```

Three things RL-1 could not do now work inside Isaac: the gravity/Coriolis
feedforward, `FusedCost` (the `plan_cost` reward term is live at −0.043), and
`TrajectoryScorer` — so the hit-latency histogram is computed **in-process**
instead of on a dump.

A second defect fell out on the way: handing `FusedCost` a shared `ChainFK`
raises `KeyError: 'left_shoulder'` — its posture term needs shoulder frames the
default frame set does not carry, and it only fires when a cost is *evaluated*,
not when it is constructed. Both are pinned by
`tests/test_isaac_rl_rnea_cache.py` (11 tests).

### 8.2 The Isaac-fitted lead, and how far it is from the analytic one

`fit_lead_isaac.py`, 162-case grid, ff ON, 0–50 ms @ 2.5 ms, strike window 6:

| group | analytic Δ (median) | **Isaac Δ (median)** | Isaac RMS @0 | @Δ* | gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| waist | 50.0 ms | **45.0 ms** | 79–85 mrad | 64–69 | 13–25 % |
| shoulder | 2.5 ms | **16.2 ms** | 30–356 mrad | 30–268 | 2–25 % |
| elbow | 15.0 ms | **7.5 ms** | 66–153 mrad | 49–153 | 0–26 % |
| **wrist** | **0.0 ms** | **38.8 ms** | 49–307 mrad | 47–247 | 3–32 % |

**The wrists are the story and they invert §7c.** The analytic fit put them at
0 ms and README §7c concluded *"the shoulders and wrists are flat — the
shoulders' error is attenuation, not phase, and a lead cannot touch it."* On
PhysX the wrists want a **38.8 ms median** and buy 20–32 % of their tracking
RMS. §7c's per-group reading was a property of `pd_sim`, not of the robot.

The other half of the table is just as important: Isaac's strike-window
tracking RMS at Δ=0 is **127 mrad** against the analytic model's **49 mrad**,
and the best uniform Δ only takes it to 112. The Isaac plant tracks the plan
roughly **2.6× worse** than the model the whole offline stack was tuned
against. Vector published at `runs/rl1b/lead_isaac.json[best_strike_s]`.

### 8.3 The re-measured histogram

32 envs × 200 ticks, 576 scheduled strikes each, same `probe.hit_latency`:

| config | in disk | miss | \|lat\| p50 | p90 | ≤20 ms | early | speed in band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RL-1: ff **OFF**, analytic lead | 158 | 70 % | 17.1 | 55.5 | 112 | **158/158** | 144 |
| ff **ON**, analytic lead | 44 | 92 % | **7.9** | 86.2 | 33 | 28/44 | 33 |
| ff **ON**, **Isaac** lead | **77** | **87 %** | 20.5 | 103.5 | 38 | 54/77 | **60** |

| bin (ms) | ff OFF, analytic | ff ON, analytic | ff ON, Isaac |
| --- | ---: | ---: | ---: |
| [−100,−60) | 16 | 9 | 15 |
| [−40,−20) | 30 | 1 | 0 |
| [−20,−10) | 80 | 2 | 5 |
| [−10,0) | 32 | 16 | 24 |
| [0,10) | 0 | 10 | 3 |
| [10,20) | 0 | 5 | 6 |
| [20,40) | 0 | 0 | 14 |

Three conclusions, in order of confidence.

1. **The all-early 158/158 of RL-1 was a feedforward artifact, not a lead
   artifact.** With the feedforward off the servo cannot hold the arm against
   gravity, so the stick sags *toward* the heads and crosses them early and
   often. That inflated both the crossing count and the speed-in-band column.
   The RL-1 reading ("the lead over-compensates") was **wrong**, and the
   honest baseline is the ff-ON row. Turning the feedforward on *raises* the
   measured miss rate from 70 % to 92 % — the earlier number was flattering.
2. **The Isaac-fitted lead beats the analytic one on the same plant**, which is
   the comparison RL priority 3 actually asked for: crossings 44 → **77**
   (+75 %), miss 92 % → **87 %**, speed-in-band 33 → **60** (+82 %). It is not
   a large win, and it moves p50 the wrong way (7.9 → 20.5 ms) while
   redistributing mass out of the ±10 ms bins into [20,40). Read together with
   §8.2's 127 mrad: a lead cancels delay, and on this plant delay is not the
   dominant error term.
3. **The gap to close is larger than §4 suggested.** 87 % miss on the real
   plant against 63 % on the analytic one is the honest starting point for the
   RL stage.

**Confound, stated rather than buried:** with the feedforward on, the safety
filter engages on **2 152 / 6 400 env-ticks (34 %)** with the analytic lead and
**1 657 (26 %)** with the Isaac lead — against **0** in the RL-1 (ff-OFF) run.
The filter modifies the executed command, so part of the ff-ON degradation may
be the filter rather than the plant. Isolating that (run with `--safety off`
and compare) is the first thing to do in RL-2, before any conclusion about the
lead is treated as settled.

### 8.4 DECISION 5, revisited — the subspace IS contact-concentrated

The RL-1 diagnostic said the FiLM deviation was uniform along the trajectory
(11 % of energy in 11 % of the grid). With the feedforward on, from realistic
states, it is **65–66 % of the energy in 11 % of the grid — a ~6× contact
concentration**, at both σ and both mechanisms:

| run | rms | waist | shoulder | elbow | wrist | contact-adjacent | roughness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ff OFF | 23.0 mrad | 6 % | 50 % | 6 % | 38 % | 11 % (of 12 %) | ×1.00 |
| ff ON, analytic lead | 22.6 mrad | 5 % | 48 % | 5 % | **41 %** | **66 %** (of 11 %) | ×1.00 |
| ff ON, Isaac lead | 22.7 mrad | 5 % | 48 % | 6 % | **41 %** | **65 %** (of 11 %) | ×1.00 |

So **half of DECISION 5 dissolves**: the perturbation *is* contact-adjacent, and
the RL-1 finding was an artifact of probing from a gravity-sagged state. What
remains true is that it is shoulder-dominated (48 %) rather than wrist-heavy
(41 %) — but 41 % at the wrists is not the "all mid-swing" failure the plan
warned about. **Recommendation: do not rebuild the perturbation as
time-gated.** Keep FiLM as-is, revisit only if σ tuning stalls.
