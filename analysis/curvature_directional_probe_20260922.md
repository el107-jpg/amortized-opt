# Direct curvature: matched directional-derivative probe (2026-09-22)

This is a read-only local derivative probe of the 100-update direct-curvature
actor checkpoint. It asks whether the implemented 25%-row, 25%-mix metric
preferentially improves some components of **FusedCost** on the same rows.
The reproducible probe is `scripts/probe_curvature_term_tradeoff.py`.

Method: three fresh fixed-pool draws (seeds 0, 1, 2), 64 rows each, 16
curvature-selected rows per draw, horizon 50. For each row, compare the
ordinary negative trajectory gradient to the curvature-adjusted negative
gradient, with **both directions normalized to equal trajectory L2 norm**.
The table gives mean local directional change per unit L2 trajectory step;
negative is predicted improvement. Each cost group is weighted exactly as in
the checkpoint's FusedCost, including its active AL terms. It is a derivative,
not a finite actor update or a claim about final evaluation.

| Cost group | Ordinary | Curvature | Change in improvement |
| --- | ---: | ---: | ---: |
| Terminal + aim | -507.0 | -470.4 | 7.2% less |
| Torque + static + effort | -20,162.2 | -19,139.4 | 5.1% less |
| Collision | -164.1 | -187.8 | 14.4% more |
| Joint box | +2.1 | +2.1 | virtually unchanged; both locally worsen |
| Smoothness | -857.9 | -796.7 | 7.1% less |
| All FusedCost | -21,835.2 | -20,755.6 | 4.9% less |

Across the 48 selected rows, the curvature direction gave less terminal
improvement on 91.7% of rows and less torque improvement on 95.8% of rows.
Mean trajectory-direction cosine was 0.951; before the equal-L2 renormalization
used for this comparison, the implemented blended gradient had 0.808 times
the ordinary selected-row norm. The same seed-0 test at the no-curvature
100-update checkpoint also weakened terminal and torque improvement (6.9%
and 5.1%) while strengthening collision improvement. Thus, the earlier
suggestion that this metric locally shifts priority from terminal to torque
is **not supported** by this probe. It mostly reduces the local projected
improvement of both while redirecting some descent toward collision/other
costs.

To see how the actor changes this interpretation, the probe also pulls the
FusedCost gradients through the model parameters, before clipping, AdamW, or
SaLSa. The terminal directional change for ordinary vs corrected *parameter*
gradients was:

| Fresh draw | Ordinary | Curvature | Actor-gradient cosine |
| --- | ---: | ---: | ---: |
| Seed 0 | -442.7 | -694.9 | 0.960 |
| Seed 1 | -176.2 | +1,001.5 | 0.408 |
| Seed 2 | -923.3 | -627.7 | 0.880 |

The sign flip on seed 1 shows that a ~0.95 cosine in trajectory space does
not imply a similarly stable terminal direction after the actor Jacobian.
The shared parameter pullback can amplify and cancel contributions from
different rows. This is a stronger concrete concern than simple local
terminal-vs-torque reweighting, but these three batches do not prove it caused
the 100-update eval regression.

Scope limits: these are fresh fixed-pool samples, not a replay/MPC/Bellman or
critic-generated training batch. Only FusedCost is differentiated; auxiliary
training losses, torque-gradient capping, clipping, AdamW moment state, and
SaLSa are excluded. The parameter probe is a **pre-optimizer gradient**, not
an executed update. A definitive causal test would instrument the full live
batch and compare per-term changes under each actually executed SaLSa step.

## Finite-step line profiles

The follow-up probe evaluates the **actual FusedCost** at a grid of equal-L2
steps along each direction, rather than just the derivative at step zero.
For trajectories, one RMS step unit means one radian RMS over all 50 × 17
waypoint-joint coordinates in a selected row. For the actor, one step unit
means one global parameter-vector L2 unit. In both cases, the two directions
have exactly the same Euclidean length at a given grid point. "Largest
improving" is the largest sampled step with lower FusedCost than at zero;
"best reduction" is the greatest sampled FusedCost decrease per row. These
are grid-limited estimates, not continuous line-search optima.

Three fresh B=64 batches (seeds 0–2) were evaluated at each of the 100-update
checkpoints. The trajectory profile concerns the 16 selected rows per batch;
the actor-parameter profile concerns the full 64-row batch, because shared
model parameters move all rows. Values below average over the three draws.

| Checkpoint and direction space | Largest improving step, ordinary → curvature | Best cost reduction, ordinary → curvature |
| --- | ---: | ---: |
| No-curvature checkpoint, trajectory | 0.000691 → 0.000751 RMS rad (+8.7%) | 125.4 → 128.6 (+2.5%) |
| Curvature checkpoint, trajectory | 0.000659 → 0.000686 RMS rad (+4.1%) | 137.3 → 139.5 (+1.6%) |
| No-curvature checkpoint, actor parameters | 0.000936 → 0.000829 L2 (-11.4%) | 70.3 → 73.5 (+4.6%) |
| Curvature checkpoint, actor parameters | 0.000627 → 0.000828 L2 (+32.2%) | 92.2 → 78.5 (-14.9%) |

So the curvature direction **does** allow a somewhat longer improving move
in trajectory space on these rows, as intended. It does not reliably grant
the same advantage after the actor Jacobian. At the curvature-trained
checkpoint, the parameter direction allowed a longer mean sampled step but
achieved less best-case cost reduction. At the control checkpoint, the mean
improving parameter range was shorter.

Terminal behavior is also mixed. At each row's best FusedCost step, mean
contact-position error change at the curvature-trained checkpoint was
-1.12 mm (ordinary parameter direction) versus **+0.33 mm** (curvature
parameter direction). That is a local best-*total-cost* comparison, not a
terminal-constrained line search. It shows how taking a longer improving
step can still move contact accuracy the wrong way. The trajectory-only
best steps, by contrast, generally held or improved terminal error.

For reproducibility, run `scripts/probe_curvature_term_tradeoff.py` with
`--line-profile both` and `--seed 0`, `1`, or `2` on either saved checkpoint.
The trajectory grid is 0–0.005 RMS rad and the parameter grid is 0–0.016 L2.
The actor profile recomputes the encoder and decoder for every candidate and
never changes the saved checkpoint or starts a training run. It is still a
**raw FusedCost gradient** profile, not the real full training loss or an
executed AdamW/SaLSa update; those are the next diagnostic if we want to
explain the observed training difference causally.
