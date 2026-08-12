# Reference baseline — trajopt HPIPM SQP, 2026-08-10

The numbers the amortized optimizer has to beat, measured with the section-2
harness (`amortized_opt/eval_harness.py`) on the seeded grid it will also be
scored on.

> ## **v2b is the primary reference line.**
>
> Three runs over the **same 243 cases, case-for-case identical**, differing
> only in how the waist is driven. All three use the same apex-in-cylinder seed.
>
> | | waist policy | accepted | coll-free (striking) | coll-free (both arms) | solve p50 |
> | --- | --- | ---: | ---: | ---: | ---: |
> | v1 | free QP variable, no reference | 151 | 160 | 7 | 1451 ms |
> | v2 | aim law → pinned `waist_reference` | 126 | 148 | **83** | 2231 ms |
> | **v2b** | **… + teacher `refine_yaw` sweep** | **149** | **162** | 66 | 6514 ms |
>
> **v2b is primary** because it is the strongest *correctly specified* expert:
> it recovers 23 of the 25 acceptances v2 lost (92 %), and it is the best of the
> three on striking-arm collision, torque, hit-disk membership, impact speed,
> velocity limits, joint travel and jerk — while keeping 59 of v2's 76 extra
> both-arms-collision-free cases (a 9.4× improvement over v1). Solve time is
> disqualifying for MPC use in all three, which is the premise of the project,
> so it is not a selection criterion for an offline reference.
>
> **v2 remains the live-pipeline-convention datapoint** and is the better run on
> the one metric v2b gives ground on: both-arms collision-freedom (83 vs 66).
> If that single number is what you care about, use v2 — and see
> *Why v2b trades idle-arm clearance for arm quality* below, because the
> mechanism is clean and directly informative for section 3.
>
> **v1 is retained** as the "free waist" datapoint and because other work is
> already pinned to its NPZ. Nothing about v1 was mis-seeded — see
> *What v1 actually used*.

| | v1 | v2 | **v2b (reference)** |
| --- | --- | --- | --- |
| file prefix | `trajopt_hpipm_2026-08-10_` | `…_v2_` | **`…_v2b_`** |
| `--planner-args` | `{"backend":"hpipm"}` | `+ "waist_aim":true` | `+ "waist_aim":true,"refine_yaw":true` |
| waist | free, no reference | aim law, pinned | aim law + IK-scored yaw sweep, pinned |
| seed | apex-in-cylinder, apex-first | same | same |
| aim gate opened on | n/a | 83 / 243 | **222 / 243** |
| \|Δyaw\| p50 / p90 (rad) | n/a | 0.000 / 0.946 | 0.388 / 0.852 |
| batch | one serial run, 496.8 s | one serial run, 812.5 s | one serial run, 2832 s |
| recorded (UTC) | 09:45:52 | 10:42:38 | 11:38:29 |
| repo HEAD | `f806232` | `f806232` | `f806232` |

Reproduce, or re-score the stored paths after a metric change:

```bash
python scripts/run_eval.py --planner repo_adapters.trajopt_hpipm_planner:make_planner \
    --planner-args '{"backend":"hpipm","waist_aim":true,"refine_yaw":true}' \
    --grid standard+near_drum --seed 0 --jobs 1 \
    --out-dir baselines --prefix trajopt_hpipm_2026-08-10_v2b
python scripts/run_eval.py --rescore baselines --prefix trajopt_hpipm_2026-08-10_v2b
```

**Re-score, do not re-solve.** The repo SQP is not reproducible run to run
(roughly a third of cases move), so comparing a new metric against a fresh solve
would confound the metric change with solver noise. The NPZ exists precisely so
that stored trajectories can be re-scored with evolving cost code.

---

## What v1 actually used (and why v2 / v2b exist)

Two things were checked against the recorded v1 metadata and the solver source,
not assumed:

**Seed — v1 was already correct.** v1 passed no seed overrides, so it ran the
`TrajOptConfig` defaults `seed_mode="apex"`, `seed_apex_first=True`,
`seed_apex_height=0.30`, `seed_apex_radius=0.15`, `seed_apex_half_height=0.10`.
`validate()` touches no `seed_*` field (asserted by
`tests/test_repo_adapter.py::test_validate_silently_zeroes_weights`). That flag
combination is exactly the requested initialization:
`trajopt_hpipm/sqp.py::_resolve_apex_first_goal` solves the apex against the
**start pose alone** — tip inside a vertical cylinder 30 cm above the strike
position, closest in joint space — and then takes the terminal strike pose
**nearest that apex** (`_strike_goal_q(q_start, q_near=apex)`), writing it into
`StrikeSpec.cached_goal_q`. So "closest point in a cylinder above the drum, then
the closest point on the drum to that point" is what all three runs did.
**No change was needed here.**

**Waist — v1 was differently specified, not frozen.** The adapter inherited the
`bench_trajopt_hpipm.py` / `test_trajopt_hpipm_acceptance.py` convention
(`_make_arm_robot(..., freeze_waist=False)`): the 3 waist joints are ordinary
optimization variables inside the full URDF box, starting wherever the case's
start pose puts them, with no reference and no aim. It is *not* the
"frozen at mid-range" convention — `bench_trajopt_offline.py` defines
`_make_arm_robot` twice and the second, non-freezing definition shadows the
first, so that script's docstring is stale and its waist is free too.

Neither is what the repo's own pipelines do, and the consequence was visible in
v1: the waist is in the solved model but the **idle arm is not**, so the QP
freely spent yaw that swung an invisible hanging arm through the kit. Only
**7/243** v1 trajectories were collision-free once both arms were counted.

### The v2 waist policy

Mirrors `teacher/adapters_trajopt.py::_waist_reference` / `_with_waist_bc` and
`play_control_theory.py::_update_waist_aim`:

1. `waist_aim.aim_yaw({hand: strike.position}, cfg, yaw_current=q0_yaw)` — a
   gated, damped **partial** turn (λ = 0.3, so ~77 % of the demand), clamped to
   ±100°, with a 5° deadband and a 40°-in / 40°-out reach cone measured from the
   torso's *current* yaw.
2. `waist_aim.waist_reference_block(...)` — a leading quintic yaw ramp that
   commits within `max_move_s = 0.2 s` and then **holds**, settled
   `settle_wp = round(0.06/dt) = 3` waypoints before contact.
3. handed to the solver as `TrajOptConfig.waist_reference`, which Block 2 pins
   hard (`trajopt_hpipm/ocp.py:1124`, `sqp.py::_apply_waist_reference`).
4. `v_start`'s waist rows sourced from `waist_aim.waist_start_velocity(ref, dt)`
   — Block 1 pins `(q1-q0)/dt == v_start` as a hard equality, so waist rows that
   disagree with the reference make the QP primal-infeasible on iteration 1.

One deliberate departure, so all three runs stay paired: the teacher passes
`setpoint=zeros` for waist pitch/roll because its rollout states always have
zero pitch/roll, while this grid samples them. v2/v2b pass `setpoint =`
`q_start`'s own pitch/roll instead. "Pitch and roll stay frozen" is therefore
frozen *at the start pose* rather than at zero — which keeps
`ref[0] == q_start[waist]` exactly (the Block 1 / Block 2 consistency above) and
leaves the 243 evaluation cases **bit-identical across v1, v2 and v2b**.

### What v2b adds

`pipeline.refine_yaw`, the teacher/expert-data path's extra step: the analytic
aim law only *proposes*. Each of 0 / 0.5× / 1× / 1.5× / 2× the proposal is
scored with the real strike-IK oracle (worst arm-joint travel to the terminal
pose at that yaw) and the best is taken; **0 is always a candidate**, so the
aimed plan is never worse than the unaimed one by that measure. Reimplemented
against the HPIPM `DrumTrajOpt` rather than reused from `TrajOptArmTeacher`,
which imports the OSQP backend unconditionally (feasibility-review Gap 9) and
would have silently made this an OSQP number.

It costs a lot: `solve_wall` 812 s → 2832 s, p50 2231 → 6514 ms, p90 6.3 → 29.7 s.
Justified for an offline reference, useless for MPC — but so is 2231 ms.

---

## v2b headline (243 cases) — and the three-way delta

| check | **v2b** | v2 | v1 | v2→v2b |
| --- | ---: | ---: | ---: | ---: |
| contact inside the legal hit disk | **241 / 243** | 238 | 240 | **+3** |
| … and within 2 cm of the aim point | 12 / 243 | 9 | 12 | +3 |
| impact speed in band (±0.25 m/s or ±25 %) | **241 / 243** | 241 | 233 | 0 |
| no penetration, striking arm | **162 / 243** | 148 | 160 | **+14** |
| keeps the 5 mm margin, striking arm | 149 / 243 | 141 | 150 | +8 |
| **no penetration, BOTH arms** | 66 / 243 | **83** | 7 | **−17** |
| only the idle arm penetrates | 96 / 243 | 65 | 153 | +31 |
| no torque-limit excess (100 % of the effort limit) | **209 / 243** | 187 | 208 | **+22** |
| no velocity-limit excess | **243 / 243** | 242 | 243 | +1 |
| inside the joint box | 243 / 243 | 243 | 243 | 0 |
| stick elevation in the 5–45° band | 230 / 243 | 226 | 232 | +4 |
| solver reported `converged` | **15 / 243** | 12 | 11 | +3 |
| solver reported failure | **0 / 243** | 0 | 0 | 0 |
| **ACCEPTED** (striking-arm basis) | **149 / 243 (61 %)** | 126 | 151 | **+23** |

Continuous metrics, p50 (p90):

| metric | **v2b** | v2 | v1 |
| --- | ---: | ---: | ---: |
| contact position error (mm) | **66.88** (119.3) | 70.22 (132.6) | 66.91 (107.1) |
| impact speed error (m/s) | **0.044** (0.137) | 0.046 (0.167) | 0.050 (0.178) |
| max penetration, striking arm (mm) | 0.00 (**41.8**) | 0.00 (52.1) | 0.00 (45.4) |
| max penetration, both arms (mm) | **19.88** (54.4) | 20.58 (53.9) | 49.85 (62.7) |
| torque excess (N·m) | 0.00 (**1.18**) | 0.00 (3.53) | 0.00 (0.64) |
| peak torque / effort limit | 0.652 (**1.068**) | 0.696 (1.239) | 0.634 (1.107) |
| joint travel L1 (rad) | **7.22** (10.91) | 7.79 (12.64) | 7.89 (13.43) |
| wrist travel (rad) | 2.70 (4.20) | 2.88 (5.24) | **2.25** (3.96) |
| jerk integral (×10⁶) | **13.5** (38.6) | 16.3 (67.5) | 13.3 (43.0) |
| solve time (ms) | 6514 (29713) | 2231 (6318) | **1451** (2429) |

## Full v2b per-metric aggregate

| metric | mean | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| contact position error (mm) | 70.29 | 66.88 | 119.33 | 171.07 |
| tip position error (mm) | 110.64 | 103.91 | 187.39 | 250.14 |
| impact speed error (m/s) | 0.063 | 0.044 | 0.137 | 0.461 |
| impact direction cosine | 0.980 | 0.987 | 0.997 | 1.000 |
| approach angle off normal (deg) | 10.00 | 9.22 | 16.48 | 39.17 |
| min clearance, striking arm (mm) | 14.53 | 17.34 | 65.52 | 111.30 |
| max penetration, striking arm (mm) | 9.92 | 0.00 | 41.77 | 109.04 |
| max penetration, both arms (mm) | 23.45 | 19.88 | 54.36 | 109.04 |
| torque excess (N·m) | 2.91 | 0.00 | 1.18 | 184.90 |
| peak torque / effort limit | 0.811 | 0.652 | 1.068 | 8.40 |
| velocity excess (rad/s) | 0.000 | 0.000 | 0.000 | 0.000 |
| joint-box excess (rad) | 0.000 | 0.000 | 0.000 | 0.000 |
| joint travel L1 (rad) | 7.48 | 7.22 | 10.91 | 18.15 |
| travel waste (L1 / net) | 2.25 | 2.15 | 3.09 | 5.96 |
| wrist travel (rad) | 2.77 | 2.70 | 4.20 | 8.16 |
| jerk integral (Σ‖q⃛‖²·dt) | 2.94e7 | 1.35e7 | 3.86e7 | 7.80e8 |
| accel integral (Σ‖q̈‖²·dt) | 12337 | 8216 | 23095 | 126997 |
| PD tracking error needed (rad) | 0.469 | 0.329 | 0.784 | 5.20 |
| **solve time (ms)** | **11653** | **6514** | **29713** | **65560** |

`contact position error` is at the strike's `contact_sigma` point (the bead on
pads, the stick shoulder 10 cm back on cymbals and the hi-hat); `tip position
error` is the bead everywhere, which is what the repo benches report.

**Distance to the aim is not a miss.** Only 12/243 land within 2 cm of the aim,
yet 241/243 are inside the legal hit disk. `validate()` zeroes
`w_strike_position` outright under `simplified_cost` (see Caveat 5), so *nothing*
pulls the contact to the aim — only `mu_hit = 5e7` keeps it on the disk. Any
comparison should use `ok_hit` (disk membership), with `contact_pos_err_mm` as a
tie-breaker for aim quality, never as pass/fail. v2b also eliminated v2's three
catastrophic outliers (529 / 412 / 268 mm, all `cymbal_left` at T = 0.60 s that
ran out of SQP budget): v2b's worst is 171 mm.

⚠ **Solve times across the three runs are not comparable.** They are separate
batches on a box whose load average sat between 54 and 68 (several concurrent
agent sessions). The v2b increase is certainly real in part — five extra
strike-IK solves plus an `OcpWorkspace` allocation each — but the split between
that and contention is not measurable from these runs. Do not quote the ratios.

**Against a 20 ms control period (`CONTROL_DT`, resolved decision 1) the expert
is ~326× too slow at the v2b median.** It was ~73× at v1. Either way it is
disqualifying, which is the premise of the project.

## Why v2b trades idle-arm clearance for arm quality

The whole delta is explained by one number: **the aim gate opened on 83/243
cases in v2 and 222/243 in v2b** (|Δyaw| p50 0.000 → 0.388 rad). The analytic
law is conservative — its 40° reach cone keeps the torso still whenever the drum
is already reachable — whereas the IK oracle asks the different and greedier
question "would turning reduce arm travel?", and the answer is usually yes.

So v2b turns the torso on nearly every beat. That is *better for the arm*
(travel, torque, hit quality, convergence all improve, and v2b beats even the
free-waist v1 on striking-arm collision) and *worse for the arm that is not in
the model*: more yaw means more idle-arm sweep, so both-arms collision-freedom
falls 83 → 66 and "only the idle arm penetrates" rises 65 → 96.

This is not an argument against `refine_yaw`; it is a direct measurement of how
much the single-arm formulation is costing. **A dual-arm planner scores the idle
arm's collisions in its own objective and would take the yaw the oracle wants
without paying for it.** Expect the amortized model to beat *both* v2 and v2b on
the both-arms metric, and treat 66/243 as the number to beat only because v2b is
otherwise the strongest expert.

## Per-cell (accepted / striking-arm collision-free / both-arms collision-free)

Paired on `case_id`. Format `v1 → v2 → v2b`.

| cell | n | accepted | coll (striking) | coll (both) | torque OK |
| --- | ---: | ---: | ---: | ---: | ---: |
| **T = 0.44 s** | 54 | 17 → **27** → 26 | 19 → **29** → 26 | 4 → **26** → 16 | 41 → 44 → **51** |
| T = 0.60 s | 54 | 35 → 19 → **32** | 37 → 19 → **32** | 0 → **10** → 9 | 46 → 41 → **46** |
| T = 0.80 s | 54 | 34 → 32 → **39** | 38 → 36 → **39** | 0 → **21** → 16 | 46 → 46 → **52** |
| standard | 162 | 86 → 78 → **97** | 94 → 84 → **97** | 4 → **57** → 41 | 133 → 131 → **149** |
| near_drum | 81 | **65** → 48 → 52 | **66** → 64 → 65 | 3 → **26** → 25 | **75** → 56 → 60 |
| 1 m/s | 54 | 30 → 24 → — | 34 → 29 → — | 3 → **20** → — | — |
| 2 m/s | 54 | 33 → 37 → — | 36 → 38 → — | 1 → **27** → — | — |
| 3 m/s | 54 | 23 → 17 → — | 24 → 17 → — | 0 → **10** → — | — |

| drum | n | accepted | coll (striking) | coll (both) |
| --- | ---: | ---: | ---: | ---: |
| snare | 39 | 29 → 18 → **29** | 29 → 24 → **30** | 3 → 6 → **9** |
| hihat | 42 | 24 → 29 → **32** | 25 → 30 → **33** | 0 → **20** → 18 |
| tom1 | 39 | 29 → 24 → **30** | 31 → 26 → **31** | 3 → **17** → 9 |
| tom2 | 42 | **23** → 11 → 13 | **25** → 14 → 15 | 0 → **8** → 6 |
| cymbal_left | 42 | **27** → 20 → 22 | **29** → 28 → 28 | 1 → **18** → 12 |
| cymbal_right | 39 | 19 → **24** → 23 | 21 → **26** → 25 | 0 → **14** → 12 |

Findings worth carrying into section 3:

1. **The short horizon is where the waist policy matters most.** T = 0.44 s was
   v1's worst cell by a wide margin (17/54 accepted, 19/54 collision-free). Both
   waist-aim runs roughly double it. With only 22 waypoints there is no time to
   negotiate a waist path *and* a strike; a prescribed ramp that commits in
   0.2 s and then holds beats asking the QP to invent one. This is the regime
   the MPC actually lives in.
2. **`refine_yaw` is what rescues the mid and long horizons.** T = 0.60 s went
   35 → 19 under plain aim and comes back to 32 with the oracle; T = 0.80 s
   reaches 39/54, the best any run achieves in any standard cell. The analytic
   law's fixed 40° cone is simply wrong about when turning pays once there is
   time to use the turn.
3. **`tom2` is the standing weak spot** — 13/42 accepted at best, worst on every
   run. Worth a look before it poisons training data.
4. **`near_drum` is the one family where v1 still wins** (65 → 48 → 52). Departure
   starts come from `drum_goal_poses.json`, which were generated at *zero* waist
   yaw; imposing an aim ramp from a pose optimized for a centred torso is a
   mismatch. Regenerating that atlas per waist-yaw pair would likely close it
   (the file already carries 33 waist-yaw pair poses that this harness does not
   use).

---

## Caveats — read before comparing anything to these tables

### 1. The expert is still solving an easier problem (feasibility-review Gap 7)

`trajopt` plans **10 DoF**: 3 waist joints + the 7 joints of the striking arm.
The idle arm is not in its Pinocchio model at all — it is locked at URDF
neutral, and the solver's collision model cannot see it. The adapter embeds the
solution into 17 DoF with the idle arm held at that neutral pose, and the
harness scores all 17.

| | v1 | v2 | v2b |
| --- | ---: | ---: | ---: |
| striking arm clean, idle arm penetrating | 153 / 243 | **65 / 243** | 96 / 243 |
| both-arms penetration p50 | 49.9 mm | 20.6 mm | **19.9 mm** |
| collision-free, both arms | 7 / 243 | **83 / 243** | 66 / 243 |

The acceptance column uses the **striking-arm** metric, so trajopt is judged on
the problem it actually solved. `coll_*_all_*` columns in the CSV carry the
both-arms figures.

**When the amortized model (which moves both arms) is scored, it must be judged
on `coll_pen_count_all`, not `coll_pen_count`, or the comparison flatters it.**
On that basis the line to beat is **66 / 243** (v2b) — or **83 / 243** if you
want the harder target that v2 sets.

### 2. The collision model is the RL env's SDF, not trajopt's hpp-fcl

The harness scores collisions with the vendored analytic SDF (sphere samples at
tip / 4 shaft points / wrist against solid drum cylinders, under-drum no-go
volumes and the torso capsule + head sphere) — i.e. the model section 3 will
train against and the model the RL env uses. `trajopt` optimizes against
hpp-fcl capsule geometry. They disagree, and the SDF is the stricter of the two.

Measured on 27 stored v1 standard-grid trajectories, min clearance over the path:

| | p10 | p50 | p90 | calls the path clean |
| --- | ---: | ---: | ---: | ---: |
| trajopt's hpp-fcl (`rob.distances`) | 0.9 mm | 40.0 mm | 84.1 mm | 24 / 27 |
| vendored SDF (this harness) | −51.3 mm | 0.3 mm | 42.9 mm | 14 / 27 |
| difference (SDF − hpp-fcl) | −83.7 mm | −30.4 mm | −0.0 mm | |

So **do not compare the collision counts here with
`docs/trajopt_acceptance_findings.md`** (128/162 pass-through-free for hpipm).
Both are internally consistent; they measure different geometry. What matters is
that the amortized model will be scored with the *same* SDF.

### 3. Target-drum exemption is local, and there is a mirror-image departure one

You cannot both avoid a drum and hit it. `robot.strike_ignore_pairs()` drops the
striking stick ↔ target-drum pair for the **whole** trajectory. The harness only
exempts the last `strike_exempt_waypoints = 2` waypoints, matching the plan's
"stay away from all drums until the last 1–2 waypoints". Sensitivity, re-scored
on the same stored v1 paths:

| `strike_exempt_waypoints` | collision-free | accepted | min clearance p50 |
| ---: | ---: | ---: | ---: |
| 0 | 2 / 243 | 2 / 243 | −28.7 mm |
| **2 (default)** | **160 / 243** | **151 / 243** | **+19.4 mm** |
| 5 | 164 / 243 | 154 / 243 | +25.9 mm |
| 10 | 167 / 243 | 155 / 243 | +26.2 mm |
| ∞ (repo semantics) | 177 / 243 | 163 / 243 | +33.4 mm |

The `near_drum` grid needed the mirror image: its start poses come from
`drum_goal_poses.json`, i.e. the stick is *on* the source head at waypoint 0 by
construction. Without a departure exemption every such case scored as a
collision at its own boundary condition (23/81 collision-free); with the first
2 waypoints exempted from the **source** drum it is 66/81. Same reasoning as the
repo's own departure exemption. This is `start_exempt_waypoints`, defaulting to
the strike value.

### 4. Two limit bands are tighter inside the solver than in the scorer

Both discrepancies are conservative — the solver is held to *less* than the
harness checks — and both are asserted by `tests/test_repo_adapter.py`:

| | solver | harness |
| --- | --- | --- |
| joint box | URDF ∓ `limit_margin = 0.02` rad, then posture-narrowed | URDF, then posture-narrowed |
| torque | `torque_scale = 0.85` × Isaac effort limit | 100 % of the Isaac effort limit |

So `243/243 inside the joint box` is guaranteed by construction, and the 34 v2b
torque-violating cases exceed **100 %** of the limit despite having been asked
for 85 % — a real over-run. The worst is 184.9 N·m of excess (8.4× the limit).

### 5. Configuration actually in force — declared ≠ effective (Gap 8)

Everything except the problem definition and the waist reference is the current
`TrajOptConfig` dataclass default, with `validate()` free to do its silent
zeroing under `simplified_cost=True`. `describe()` records **both**; the run
metadata carries `trajopt_declared_defaults` and
`trajopt_effective_after_validate`. What `validate()` changes:

| weight | declared | **effective** |
| --- | ---: | ---: |
| `w_v` | 1.0 | **0.0** |
| `w_posture` | 0.3 | **0.0** |
| `w_strike_posture` | 5.0 | **0.0** |
| `w_strike_position` | 10.0 | **0.0** |
| `w_strike_velocity_tangential` | 7.0 | **0.0** |
| `w_stick_elev` | 500.0 | **0.0** |
| `w_drum_proximity` | 1000.0 | **0.0** |
| `strike_min_velocity_fraction` | 0.2 | **0.0** |
| `hit_tight_radius` | 0.03 | **0.0** |

(An early draft of this document reported only the declared column and was
misleading about `w_strike_position` in particular — that zero is *why* the
contact lands ~67 mm from the aim.) Surviving and in force:

```
seed_mode = apex          seed_apex_first = True      seed_apex_height = 0.30
seed_apex_radius = 0.15   seed_apex_half_height = 0.10   seed_apex_weight = 1000
start_velocity_mode = auto  (== "dynamic" on HPIPM)   max_iterations = 40
w_a = 1e-4  w_j = 1e-5  w_ke = 0
w_strike_velocity = 25000  (normal 50 / tangential 0 after validate)
enable_torque_limits = True   torque_scale = 0.85
enable_accel_limits = False   accel_limit = 30.0 (diagnostic only — never imposed)
collision_margin = 0.0    d_on = 0.08  d_off = 0.10   acm_samples = 300
limit_margin = 0.02       waist_pin_tol = 0.0
allow_rrt = False (no RRT seed cascade, matching the repo's HPIPM benches)
waist aim: damping 0.3, yaw limit 100 deg, gate 40 deg in / 40 deg out,
           az_pref 20 deg, deadband 5 deg, settle 0.06 s, max_move 0.2 s
v2b only: refine_yaw over 0 / 0.5x / 1x / 1.5x / 2x, scored by strike-IK travel
```

These differ from the repo *bench scripts*, which override `w_posture` to 0.02,
`w_strike_posture` to 1.0, `acm_samples` to 60, `d_on` to 0.06 and
`collision_margin` to 0.005. This baseline deliberately uses the library
defaults, not the bench overrides.

### 6. Non-determinism

`docs/trajopt_hpipm.md` and the project memory both record that this bench is
not reproducible across runs — about a third of cases move. Everything is
persisted: cases, per-case scores, raw trajectories, and the planner's own
`converged` / `iterations` / `message` / `yaw_target_rad` per case. Treat
single-case differences as noise; treat distribution shifts as signal, and
prefer paired comparisons on `case_id` (which is what every table above is). A
slice of ~50 cases is not enough to separate a 5-case effect from resampling
noise.

---

## What "beating this" means

Against **v2b**, on the same grid and the same scorer:

| | trajopt HPIPM v2b | amortized target |
| --- | --- | --- |
| inference time p50 | 6514 ms | **< 5 ms** (the actual point) |
| contact in the legal hit disk | 241 / 243 | ≥ 241 |
| impact speed in band | 241 / 243 | ≥ 241 |
| collision-free, striking arm | 162 / 243 | ≥ 162 |
| **collision-free, both arms** | **66 / 243** (v2: 83) | **≫ 83 — this should be a rout** |
| no torque excess | 209 / 243 | ≥ 209 |
| accepted (striking-arm basis) | 149 / 243 | > 149 |
| **accepted @ T = 0.44 s** | **26 / 54** | **the metric that matters most for MPC** |

The both-arms row is where the amortized model should win outright rather than
merely match: it plans both arms and the waist together, so it is not subject to
Caveat 1 at all — and Caveat 1 is what makes the expert choose between arm
quality (v2b) and idle-arm clearance (v2) in the first place. A dual-arm planner
should not have to choose.

---

## Decisions needing review

1. **v2b is primary; v2 is better on one metric.** v2b wins on 9 of 10
   acceptance checks and every continuous quality metric except wrist travel and
   solve time, but v2 is better on both-arms collision-freedom (83 vs 66). The
   choice made here follows "strongest correctly-specified expert". If the
   intended reference is instead "the expert whose *whole robot* is safest",
   that is v2 and the two lines are one flag apart.
2. **Pitch/roll setpoint is `q_start`, not zeros.** Deliberate, to keep the
   cases paired across all three runs. If the grid is regenerated with zero
   waist pitch/roll the teacher's exact convention becomes available — and it
   would probably also fix the `near_drum` regression (finding 4 above), since
   `drum_goal_poses.json` was generated at zero yaw.
3. **`near_drum` is the one family where v1 still wins** (65 → 48 → 52). Either
   regenerate the departure-pose atlas per waist yaw, or accept that departure
   cases prefer a free waist and note it when reading the amortized model's
   numbers on that half of the grid.
4. **Solve-time numbers are not comparable across the three runs** (box load
   54–68 throughout). Needs a back-to-back re-time on an idle box if the number
   ever matters — which, at 300×+ over budget, it probably does not.
