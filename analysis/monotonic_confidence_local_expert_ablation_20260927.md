# Local-expert monotonic critic ablation

Date: 2026-09-27

## Question

Does replacing the ordinary MLP trunk with input-routed local experts reduce
continual-learning interference while retaining conservative feasibility
admission?

## Architecture

The local critic keeps the same three-head ensemble, randomized derivative
prior, exact monotonic TTH basis, observable confidence target, replay, inertia,
and two-tier anchor memory as the default critic. Each head replaces its MLP
trunk with:

- a shared `42 -> 128` context projection;
- eight `128 -> 80` experts;
- a learned router selecting two experts per row;
- a weighted combination of the selected expert representations;
- small load-balancing and router-logit regularizers.

The router sees state/task context and active-hand indicators, but not the
continuous TTH coordinates. Changing a positive TTH therefore cannot change
the selected experts, and the existing coordinatewise monotonic guarantee is
preserved. Residual and confidence output weights remain zero initialized, so
both architectures begin with the same maximum-violation prediction and 1%
confidence.

The comparison is approximately parameter matched:

| Architecture | Trainable parameters | Total parameters |
|---|---:|---:|
| MLP | 310,584 | 417,624 |
| Local experts | 290,496 | 397,536 |

## Protocol

Three seeds (`17`, `29`, and `43`) used the same cached actor labels and data
order. For every seed and architecture:

- 18,432 original examples trained the base critic for 3,600 updates;
- 2,048 two-tier anchors were then seeded from the base data;
- same-generator, readiness/cross-hand, and legacy-regime streams contributed
  4,096 rows each and 2,400 further updates;
- a protected 6,144-row original split recalibrated the admission rule after
  every stage to a 0.5% false-feasible target;
- another 6,144 original rows remained holdout data.

The runs were isolated on GPU 2. Raw per-seed and per-stage results are in
`analysis/monotonic_confidence_local_expert_ablation_20260927.json`.

## Results

Values are mean ± population standard deviation across three seeds.

| Metric | MLP | Local experts | Interpretation |
|---|---:|---:|---|
| Oldest prediction drift, mean | 0.1209 ± 0.0037 | **0.1186 ± 0.0017** | 1.8% lower |
| Oldest prediction drift, p95 | **0.3567 ± 0.0044** | 0.3656 ± 0.0098 | 2.5% worse tail |
| Original holdout MAE | 0.6105 ± 0.0039 | **0.5622 ± 0.0044** | 7.9% lower |
| Original confidence AUC | 0.7664 ± 0.0004 | **0.7773 ± 0.0022** | Better error ranking |
| Original false feasible | 0.174% ± 0.015% | **0.119% ± 0.008%** | Both safe |
| Original coverage | 5.31% ± 0.18% | **6.21% ± 0.17%** | 17.0% relative gain |
| Same-generator false feasible | **0.260% ± 0.050%** | 0.513% ± 0.000% | Experts exceed target |
| Same-generator coverage | 11.16% ± 0.17% | **13.88% ± 0.40%** | 24.4% relative gain |
| Readiness false feasible | 0.480% ± 0.064% | **0.464% ± 0.087%** | Both near target |
| Readiness coverage | 8.19% ± 0.06% | **11.30% ± 0.31%** | 37.8% relative gain |

Both architectures abstained completely on the legacy-regime shift. The local
expert critic improved its legacy MAE from roughly 0.647 to 0.598, but this did
not create any safe admissions.

The extra same-generator admissions were mostly correct: local-expert precision
was 96.2-96.5% and recall was 21.6-23.1%, compared with MLP precision of
97.2-98.2% and recall of 17.9-18.4%. Nevertheless, the population false-feasible
rate was 0.513% for every local-expert seed, just above the specified 0.5%
budget.

## Routing and cost

Routing remained distributed after the full stream:

- all eight experts were selected in every seed;
- the busiest expert averaged 19.4% of selections;
- the least-used expert averaged 6.7%;
- mean router entropy was 1.873 nats, versus a maximum of `log(8) = 2.079`.

Thus the result was not caused by expert collapse. The current dense batched
implementation computes all narrow expert matrices before gathering the top
two. Base training took 216.6 seconds versus 106.3 seconds for the MLP, and
continuation took 243.8 versus 138.7 seconds: approximately 1.88x total runtime.

## Decision

Keep the local-expert trunk available as an experimental option, but retain the
MLP default. The experts learn a more accurate function, rank confidence better,
and admit substantially more feasible points with fewer parameters. They do not
materially improve mean forgetting, worsen tail drift, slightly exceed the
same-generator safety target, and nearly double training time in the current
implementation.

The most useful follow-up would test a stricter calibration transfer margin and
true sparse dispatch. That could recover the coverage gain within the safety
budget and remove much of the runtime cost. It would still not establish local
experts as a forgetting solution unless p95 drift improves.
