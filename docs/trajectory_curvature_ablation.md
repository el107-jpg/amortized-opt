# Analytic trajectory-curvature lookahead

This opt-in phase-2 ablation replaces the arm-only DLS *teacher*, not Adam,
SaLSa, the actor architecture, or the actor's ordinary full-physics loss.
The default remains `--terminal-lookahead-mode dls`.

For each selected actor trajectory, the teacher differentiates the complete
`FusedCost` with respect to its waypoints. It applies a positive curvature
model consisting of the **exact** Hermite-spline acceleration/jerk Hessian,
terminal contact-position and finite-difference impact-velocity Gauss--Newton
blocks for both hands (including shared waist joints), and diagonal damping.
The small contact system is solved through Woodbury after per-joint Cholesky
solves. A row is admitted only if a bounded candidate decreases its *actual*
full `FusedCost`; up to five shrink factors are tried. Accepted corrections
are detached teachers for the same full-cost straight-through lookahead and
Huber tie used by DLS. Consequently, this is **not** a differentiable BFGS
layer, nor a guarantee of lower exact aim error: the full cost can prefer a
legal off-centre contact.

To compare with the existing DLS recipe, keep the same initialization, pool,
sample stream and DLS lookahead weights; set:

```bash
SKIP_P1=1 INIT=runs/phase1_19m_rope_plant/best.pt \
OUT=runs/trajectory_curvature_pilot \
POOL=runs/p2pool_diagnostic_safe_uniform_300_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz \
VOXEL_POOL=0 LINE_SEARCH=1 TIMING_CRITIC_MODE=assist \
TERMINAL_LOOKAHEAD_MODE=curvature DLS_TERMINAL_FRAC=0.05 \
SCENARIOS=80000 STEPS=500 \
  bash scripts/_run_rhythm_bellman.sh
```

The 0.05 fraction is a conservative pilot setting because this teacher adds
one full-cost gradient and one to five full-cost candidate evaluations for
its selected rows. The launcher also exposes `CURVATURE_DAMPING`,
`CURVATURE_POSITION_WEIGHT`, `CURVATURE_VELOCITY_WEIGHT`, and
`CURVATURE_MAX_STEP_RAD`. The mode and these weights enter the SaLSa objective
fingerprint so resuming with different values resets selector history, while
the default DLS fingerprint remains unchanged.

Inspect `terminal_curvature/accept_frac`,
`terminal_curvature/cost_improvement`, `terminal_dls/err_before_m`,
`terminal_dls/err_after_m`, train/eval terminal acceptance and miss, collision,
torque and joint-box violations, and step time. A useful outcome requires
actor improvement over equal compute—not merely teacher cost decrease.

## Initial CPU check (2026-09-21)

The analytic spline Hessian matched autograd on both full and shortened
horizons. The focused tests passed, and matched two-update CPU actor runs
completed with the curvature teacher and the default DLS teacher (same
initial checkpoint, pool, seed, batch size 64, and SaLSa settings). The GPU
driver was unavailable, so this is a smoke test rather than a learning
ablation.

| Teacher | Selected-row exact contact error, update 1 | Update 2 | Runtime |
| --- | --- | --- | --- |
| Curvature | 138.9 to 139.4 mm | 89.3 to 91.5 mm | 19 s |
| DLS | 138.9 to 34.8 mm | 107.6 to 16.6 mm | 17 s |

The curvature teacher accepted all selected rows because its full cost fell
by 129.5 and 99.9 units on the two updates. That did **not** make its exact
contact error fall. The two runs diverged after their first update, so their
update-2 starting errors are not paired measurements. Both had zero eval
acceptance after two updates. This is a reason to retain DLS as the default
and test any curvature variant against longer, equal-compute training before
adopting it, not evidence that curvature cannot help.

### No-lookahead AdamW/SaLSa control

A third run used the same checkpoint, pool, seed, batch of 64, two actor
updates, and AdamW/SaLSa settings, but set `DLS_TERMINAL_FRAC=0`. This is the
clean comparison to the ordinary actor update: no trajectory-space teacher
was computed. The actor's first-update loss was identical in all three runs
(5466.33), as expected before the updated models diverged.

| After update 2, fixed 81-case eval | No lookahead | DLS | Curvature |
| --- | ---: | ---: | ---: |
| Accepted | 0/81 | 0/81 | 0/81 |
| Hit pass rate | 7.4% | 0.0% | 4.9% |
| Collision pass rate | 64.2% | 54.3% | 54.3% |
| Torque pass rate | 0.0% | 0.0% | 0.0% |
| Median contact-position error | 114.7 mm | 113.7 mm | 98.5 mm |

These results are mixed and far too early to establish learning performance.
The no-lookahead launch also ran a final song probe whereas the earlier two
launches disabled probes, so **wall-clock totals are not comparable**. The
probe occurred after the two actor updates. This test compares the
trajectory teacher to an ordinary AdamW/SaLSa update; it does not compare
different parameter optimizers, since both variants still use AdamW/SaLSa.

## Direct trajectory-gradient preconditioning (2026-09-21)

`CURVATURE_DIRECT_FRAC>0` is a separate opt-in experiment. It disables the
lookahead teacher (`DLS_TERMINAL_FRAC=0`) and applies the same analytic metric
directly to the actor trajectory VJP. The ordinary scalar physics objective
is unchanged. Per-row gradient norms are matched before mixing, so this is a
direction test rather than an accidental learning-rate rescale. The true
parameter gradient is saved for SaLSa's directional test *before* the
preconditioned VJP is added to the live AdamW gradient. The torque budget uses
a matching preconditioned torque proxy on refresh steps.

For example, set `CURVATURE_DIRECT_FRAC=0.25` and
`CURVATURE_DIRECT_MIX=0.25`, while setting `DLS_TERMINAL_FRAC=0`, on the
launcher command above. `curvature_direct/cosine` reports the cosine between
the ordinary and transmitted trajectory VJPs; a near-zero cosine means a
strong rotation, even though both directions have the same norm.

The pure direct pilot (`FRAC=0.25`, `MIX=1`) rotated selected trajectory
gradients almost orthogonally (cosine 0.05–0.07) and gave 0/81 hits and
231.6 mm median contact error after two updates. This is a direction warning,
not a learning curve.

Matched 20-update CPU runs (same warm start, pool, seed, batch 64, SaLSa,
fixed 81-case eval; probes off) gave:

| Metric at update 20 | No preconditioning | Direct, 25% rows / 25% mix |
| --- | ---: | ---: |
| Accepted | 0/81 | 0/81 |
| Hit pass rate | 49.4% | 39.5% |
| Collision pass rate | 69.1% | 70.4% |
| Joint-limit pass rate | 59.3% | 85.2% |
| Torque pass rate | 0% | 0% |
| Torque excess p90 | 105.2 | 88.6 |
| Contact-position error p50 | 73.1 mm | 85.1 mm |
| CPU wall time | 95.5 s | 110.8 s |

The mixed VJP remained close to the ordinary direction (cosine 0.951 on
logged updates). It improved some constraint metrics but regressed contact
accuracy; neither run was feasible overall. CPU wall times are indicative
only, and the GPU driver was unavailable. Do not promote this recipe without
a longer equal-compute experiment and a terminal-preserving curvature metric.

### Continuation to 100 updates (2026-09-22)

Both 20-update runs were resumed from their own `last.pt` to absolute update
100 on CPU with the same pool, batch 64, 25-update eval cadence, and no song
probe. Model and Adam state resumed; **both** SaLSa selectors reset their
acceptance history on an objective-fingerprint change. Neither run skipped an
update. Results below are on the same fixed 81-case eval:

| Metric at update 100 | No preconditioning | Direct, 25% rows / 25% mix |
| --- | ---: | ---: |
| Accepted | 1/81 | 1/81 |
| Hit pass rate | 77.8% | 72.8% |
| Collision pass rate | 74.1% | 72.8% |
| Joint-limit pass rate | 87.7% | 92.6% |
| Torque pass rate | 12.3% | 18.5% |
| Torque excess p90 | 50.2 | 45.5 |
| Contact-position error p50 | 56.1 mm | 65.5 mm |
| Continuation wall time, 80 updates | 343.9 s | 394.2 s |

The control reached one accepted case at update 75; direct curvature reached
one at update 50, but both still had only one at 100. Direct curvature kept
its torque/joint-box advantage while retaining a contact-accuracy deficit and
taking about 15% longer on this CPU run. This single-seed pilot is not
evidence of a net improvement. GPU behavior remains unmeasured.
