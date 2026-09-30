# Actor superbatch ablation (2026-09-23)

## Question

Does drawing 4096 timing-curriculum rows at once and consuming them as 1024-row
actor minibatches improve production training throughput without degrading the
training signal?

One implementation detail matters: the distributional timing critic is updated
after every actor optimizer step, not every four actor steps.  Accordingly, the
superbatch caches only the randomly drawn state/velocity/drum-goal and MPC task
rows.  Frontier time proposals, actor labels, and the critic update remain live
for every actor minibatch.

## Implementation

- `Phase2Config.actor_superbatch_factor` draws `batch_size * factor` rows,
  shuffles them, and consumes non-overlapping slices over subsequent updates.
- Boundary/interior timing rows belonging to one physical task are kept in the
  same slice so that interior rows reuse the correct selected frontier.
- Cached slices are held on CPU; only the active slice occupies GPU memory.
- `Phase2Config.suffix_microbatch` evaluates the differentiable suffix objective
  in row chunks while preserving its exact weighted batch mean and labels.
- Scalar SaLSa line-search probes run under `torch.no_grad()`; the curvature
  pair path still explicitly enables gradients when it needs them.

The 1024-row production configuration otherwise exceeded the 32 GiB GPU budget.
To compare it fairly with the 512-row baseline, the absolute auxiliary row
counts were preserved: suffix and DLS fractions were changed from 0.25 to 0.125
(128 rows each), and the suffix calculation was microbatched in groups of 32.

## Matched 40-update smoke comparison

All runs started from the same step-20000 checkpoint and used the full production
loss, SaLSa line search, timing critic, and final evaluation.  The streams are
randomized, so the evaluation is a smoke comparison rather than a statistically
powered learning ablation.

| actor batch | superbatch | wall time | updates/s | primary rows/s | eval accepted | hit pass | collision pass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 512 (factor 1) | 86 s | 0.46 | 238 | 21/81 | 0.93 | 0.86 |
| 1024 | 4096 (factor 4) | 169 s | 0.24 | 242 | 22/81 | 0.91 | 0.86 |
| 1024 | 1024 (factor 1) | 130 s | 0.31 | 315 | 21/81 | 0.91 | 0.86 |
| 512 | 4096 (factor 8) | 83 s | 0.48 | 247 | 21/81 | 0.93 | 0.86 |

Run directories:

- `/tmp/rhythm_b512_control40_20260923`
- `/tmp/rhythm_sb4096_b1024_trial40_20260923`
- `/tmp/rhythm_b1024_factor1_trial40_20260923`
- `/tmp/rhythm_sb4096_b512_factor8_trial40_20260923`

## Interpretation

The exact 4096-to-1024 proposal works, but does not make production training
faster.  It roughly preserves primary-row throughput while halving the number
of parameter updates per unit time.  Actor-core GPU utilization improves with a
larger batch in isolation, but the production suffix/DLS objectives, line-search
probes, critic work, and CPU-offloaded cache consume the gain.

The superbatch itself is not harmful.  Drawing 4096 rows and consuming eight
512-row minibatches matched the baseline evaluation and was about 3.5% faster in
this short run.  That is therefore the recommended opt-in configuration if the
goal is to amortize random task generation while retaining frequent stochastic
updates.  Keep 1024 as a future option for a memory/auxiliary-loss redesign, not
as the current production default.

Also, random examples do not make a larger minibatch more stochastic: averaging
more independently sampled rows lowers gradient variance.  This can improve the
gradient estimate, but it reduces optimizer noise.  The earlier frozen-batch
ablation and this production smoke both favor retaining frequent fresh updates.
