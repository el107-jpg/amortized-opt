## Verdict

**No mechanism beat the control on the strike, and the reason is that all three
pay for torso stillness out of the arms' budget.** Every one of them stabilises
the waist — convincingly, and the ratchet is genuinely cured — but every one
also costs acceptance, and the cost scales with how hard the waist is
constrained.

### 1. The ratchet is real, and an absolute anchor cures it

The control's waist does not oscillate around upright; it **walks to its joint
limit and sits there**. Over a 4 s episode, measured on the plant:

| arm | pitch slope | pitch net | roll net | measured pitch p50 |
|---|---|---|---|---|
| `base` | +0.254 rad/s | **+0.511** | **+0.641** | 0.520 |
| `aim5x` | +0.037 | −0.091 | +0.205 | 0.040 |
| `aim10x` | +0.041 | −0.019 | +0.133 | 0.023 |
| `direct` | +0.102 | −0.006 | +0.144 | 0.057 |
| `freeze` | +0.009 | −0.008 | +0.138 | 0.010 |

The URDF box on waist pitch/roll is **±0.52 rad** and the control's measured p50
is **0.520** — saturated. Every arm carrying an absolute anchor drives net drift
to ≈ 0. This is the ratchet, and it is fixed.

It is also **the plan's own doing, not the servo's**: lead gaps are 0.001–0.025
rad and tracking gaps 0.002–0.021 rad, one to two orders of magnitude below the
0.52 rad excursion. The planner asks for the lean, every tick, from a state that
already has one. That is the regime a cost weight can reach — and the results
confirm a weight reaches it.

Note this is precisely the failure a temporal-consistency price **cannot** fix.
`w_tc = 50` was active on all five arms, including the control that ratcheted to
saturation: a monotone march is highly self-consistent tick to tick, so it pays
almost nothing. Stopping drift needs an absolute anchor; `w_tc` only smooths it.

### 2. But the arms pay for it, and that is the real finding

Open-loop acceptance is a conjunction of six gates. Decomposing which gate
rejects (`standard` grid, both seeds):

| arm | acceptance | collision | joint box | torque | acc if collision were perfect |
|---|---|---|---|---|---|
| `base` | **0.475 / 0.512** | 0.531 / 0.556 | 0.994 / 1.000 | 0.975 / 0.957 | **0.903 / 0.892** |
| `aim5x` | 0.284 / 0.185 | 0.525 / 0.512 | **0.741 / 0.722** | 0.864 / 0.821 | 0.526 / 0.487 |
| `aim10x` | 0.315 / 0.185 | 0.512 / 0.494 | **0.735 / 0.636** | 0.944 / 0.809 | 0.617 / 0.463 |
| `direct` | 0.383 / 0.370 | 0.488 / 0.481 | 0.938 / 0.957 | 0.975 / 0.957 | 0.774 / 0.785 |
| `freeze` | 0.080 / 0.049 | 0.574 / 0.630 | 0.660 / 0.580 | **0.235 / 0.210** | 0.124 / 0.104 |

Two separate bottlenecks, and only one of them belongs to this study:

* **Collision (~0.5) is pre-existing and untouched.** It is flat across every
  arm, including the control, and is the same non-target drum-body contact
  during the mid-trajectory approach that §7d already documented. No waist term
  helps or hurts it. It is worth ~0.4 of acceptance on its own and is the single
  biggest available win — but it needs a spatial path term, not a waist term.
* **Joint box and torque degradation are CAUSED by the waist constraints**, and
  scale with severity. `direct` (soft well + 0.15 box) barely dents the box
  (0.94–0.96); the IL arms at 5–10x drop it to 0.64–0.74; `freeze` destroys
  torque outright (0.21–0.24 against the control's 0.96–0.98).

The mechanism is coherent. The waist is a shared DOF that buys reach and impact
speed for both arms. Price it or remove it and the arms must cover the same
targets from a worse posture, so they run into their own position limits — or,
with no waist lean to trade at all, into their torque ceiling.

### 3. Ranking

**`direct` (technique 2) is the one to keep.** It is the only mechanism that
stabilises the torso while leaving the arms' budget intact:

* torque compliance **unchanged** from the control (0.975 / 0.957);
* joint box barely touched (0.938 / 0.957);
* the **lowest yaw rate of any arm** (yawd p90 1.257 vs 1.858–3.479), which is
  what the `w_waist_yaw_vel = 20` was for;
* closed-loop yaw travel **1.95 vs the control's 4.05** — less than half the
  torso motion the servo has to track;
* closed-loop hit rate 0.298 vs the control's 0.323 — a 2.5 pp cost, near free.

Acceptance still costs ~10 pp open loop (0.383/0.370 vs 0.475/0.512), so this is
a trade, not a free win. But it buys a torso that no longer ratchets to its
limit and moves half as much, for almost nothing in strikes landed.

**The IL arms (technique 1) are dominated by `direct`**: they stabilise
pitch/roll harder (lean p90 0.44–0.47 vs 0.66) but cost far more acceptance and
push the joint box to 0.64–0.74. They also **move the yaw more, not less** —
yaw p90 0.65 and yawd p90 2.78–3.05 against the control's 0.53 / 1.86. Pinning
pitch and roll displaces the motion into yaw rather than removing it. Raising
5x → 10x bought a little stillness and no acceptance, so the weight is not the
lever.

**`freeze` (technique 3) should not be pursued as configured.** It produces by
far the stillest torso (lean p90 0.258, pitch p90 0.078) and the cleanest
ratchet numbers, but torque compliance collapses to 0.21–0.24, open-loop
acceptance falls to 0.049–0.080, closed-loop hit rate to 0.109, and it diverged
on one of six episodes. Removing the DOF does not remove the work — it moves it
onto actuators that cannot supply it.

### 4. Caveats

* **15 000 steps is under-trained, and not neutrally so.** `ramp_end = 20000`
  against `max_steps = 15000`, so the constraint curriculum never completed:
  `alpha` ended at **0.811**, leaving torque / velocity / joint-box limits at
  ~83 % strength — while the acceptance grid scores at **full** strength. Every
  arm was judged under constraints it never trained against, and the gates that
  separated the arms (box, torque) are exactly the ones affected. BPTT likewise
  ran for only the last 4 950 steps, and the project's reference length for this
  config is 60 000 (`_run_ablation.sh`, the section-8 runs).

  The curves make the asymmetry concrete: `direct`, `aim5x` and `aim10x` all hit
  their maximum on the FINAL eval (0.35 / 0.27 / 0.31) while `base` had
  plateaued and oscillated after its step-10000 peak. The control is therefore
  shown converged and the stabilised arms mid-climb, so **the ~10 pp acceptance
  penalty attributed to `direct` is likely an overestimate.**

  `lam["joint_box"]` also ended **pinned at its 100.0 ceiling** — the multiplier
  saturated with the constraint still violated, the same failure shape as the
  shortfall-Lagrangian episode.

  What survives: the ratchet result is a direct measurement of plant behaviour,
  not the endpoint of a converging curve, and the ranking is common-mode across
  a shared schedule. What does not: the absolute acceptance numbers, and the
  size (not the sign) of `direct`'s cost.
* **One seed, one run per arm.** Differences of a few points are not resolvable;
  the ordering `freeze` << IL < `direct` < `base` is large enough to trust, the
  5x-vs-10x gap is not.
* **`best.pt` is selected on rolling acceptance, so arms are compared at
  different training steps** (`base` 10000, `freeze` 11000, others 15000). The
  control's peak-then-decay (0.457 @ 10000 → 0.395 final) means it is being
  shown at its best while `aim5x`/`direct` are still at their final step.
* Arms trained against `model.py` as of 06:29 on 2026-08-12, before the
  `idle_home_token` encoder change landed at 07:19. All five share that code, so
  the comparison is internally valid; absolute numbers may shift on the current
  model.
* Closed-loop hit rate and open-loop acceptance are different measurements over
  different populations and are not complements — see the note on that table.
