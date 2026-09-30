# Frozen-batch optimizer-step ablation (2026-09-22)

## Question and setup

Does taking 2, 3, 5, or 10 optimizer steps on a frozen actor collocation
batch improve early Phase-2 training relative to one step, without trajectory
curvature preconditioning?

`scripts/ablate_frozen_batch_updates.py` restores the same early checkpoint
(`/tmp/trajectory_no_lookahead_cpu20_20260921/last.pt`, step 20), model weights,
and **separate deep copies** of its AdamW moments for each arm. It samples a
bank of eight independently drawn, noisy Phase-2 batches of 64 horizon-50
rows and a disjoint 64-row held-out batch. Each run saves the exact sampled
bank, including corrected initial trajectories. All arms within a seed use
the same bank, same order, same start, fixed learning rate `1e-6`, same
clipping, and frozen checkpoint AL/schedule weights. Curvature, DLS, torque
gradient cap, line search, and additional curriculum losses are off. CPU only;
the GPU driver was unavailable during this experiment.

This isolates optimizer-step grouping in the primary actor physics-plus-aux
loss. It is **not** a full production training run: critic-selected fresh
rows, MPC/replay, suffix/Bellman, TC/BPTT, SaLSa, and changing AL multipliers
are not included. The fixed 81-case grid evaluation is the same for all arms.

An initial run was discarded because `optimizer.load_state_dict` aliased CPU
checkpoint Adam moment tensors, contaminating later arms. The script now
deep-copies that state per arm; only the corrected results below are used.

## Same unique rows, increasing compute (seed 0)

Each arm sees the eight unique batches once. Therefore K steps per batch also
means 8K total updates; this measures the gain from spending more compute on
the existing collocations, **not** the gain from grouping at equal compute.

| Steps/batch | Updates | Held-out cost | Held-out terminal term | Held-out contact mean | Grid hit checks |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 730.4 | 37.10 | 99.3 mm | 50.6% |
| 2 | 16 | 692.6 | 35.82 | 96.9 mm | 53.1% |
| 3 | 24 | 666.8 | 34.88 | 94.6 mm | 55.6% |
| 5 | 40 | 633.8 | 33.44 | 92.4 mm | 63.0% |
| 10 | 80 | 558.1 | 30.56 | 87.4 mm | 70.4% |

Extra updates clearly help at this early stage. The terminal term falls with
them, but this table alone cannot tell whether repeated-batch updates are
preferable to spreading the same updates across batches.

## Matched 80-update, matched-row-exposure control

Every arm makes exactly 80 optimizer updates and presents **each of the same
eight batches ten times**. K changes only grouping: for example, K=1 cycles
through the bank ten times, while K=10 completes ten steps on batch 1 before
moving to batch 2. K=3 uses 3+3+3+1 steps per batch. Held-out rows are never
optimized. Seed 1 uses a second independently sampled bank, again shared
exactly within its five arms.

| Seed | Steps/batch | Held-out cost | Held-out contact mean | Grid hit checks | Grid contact p50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | **533.5** | 86.5 mm | 70.4% | 56.5 mm |
| 0 | 2 | 535.7 | 86.2 mm | 70.4% | 56.7 mm |
| 0 | 3 | 542.0 | 86.1 mm | 70.4% | 58.9 mm |
| 0 | 5 | 547.8 | 86.0 mm | 66.7% | 56.8 mm |
| 0 | 10 | 558.1 | 87.4 mm | 70.4% | 56.7 mm |
| 1 | 1 | **602.2** | 91.0 mm | 74.1% | 63.7 mm |
| 1 | 2 | 607.1 | 90.7 mm | 74.1% | 62.7 mm |
| 1 | 3 | 619.9 | 91.6 mm | 74.1% | 65.6 mm |
| 1 | 5 | 625.3 | 95.4 mm | 74.1% | 67.0 mm |
| 1 | 10 | 629.0 | 91.3 mm | 72.8% | 59.6 mm |

All arms had 0/81 fully accepted cases: these are still early-stage models,
and hit-check percentages or median contact errors should not be mistaken for
overall feasibility. Per-arm training time for 80 CPU updates was about
118-125 seconds, excluding collocation-bank generation.

**Interpretation:** repeating a batch is useful if the alternative is simply
*fewer total updates*. At equal compute and data exposure, however, one-step
cycling has the lowest held-out cost on both banks, and two steps is nearly
tied. There is no evidence here that 3-10 consecutive steps transmit terminal
corrections through the trajectory more effectively than revisiting the same
rows after other batches. This narrow test supports keeping one step as the
default; two may be worth trying in the live loop if fresh-row selection is
expensive. It does not justify 5-10 as a production default.

## Late-checkpoint matched control

The matched 80-update experiment was repeated from the completed step-20,000
checkpoint `runs/rhythm_timing_com_linear_ropeplant_gpu1_20k_20260921/last.pt`
with a newly sampled seed-0 bank. All other isolation and matching rules above
were retained.

| Steps/batch | Held-out cost | Accepted | Grid hit checks | Grid contact p50 | Torque checks |
|---:|---:|---:|---:|---:|---:|
| 1 | **79.23** | 22/81 | 87.7% | 64.6 mm | 97.5% |
| 2 | 81.06 | 22/81 | 90.1% | **59.7 mm** | 97.5% |
| 3 | 81.69 | 22/81 | **92.6%** | 62.1 mm | 97.5% |
| 5 | 84.97 | 20/81 | **92.6%** | 61.8 mm | 88.9% |
| 10 | 82.90 | 22/81 | 87.7% | 70.6 mm | 100.0% |

Late training therefore makes 2-3 repeats somewhat more attractive for
hit-specific refinement, but neither improves complete acceptance in this
seed. Five repeats sacrifices torque feasibility and two accepted cases; ten
loses the contact improvement. One step still has the best held-out aggregate
cost. Training time for 80 CPU updates remains approximately flat (113-122 s),
so repeats do not make an optimizer update cheaper; their only systems-level
speed benefit is amortizing expensive collocation generation.

## Late checkpoint: 1,000-update control (2026-09-23)

The late-checkpoint matched control was extended to 1,000 updates for K = 1,
2, 3, and 5. Every arm saw each of the same eight 64-row batches exactly 125
times. Two independently sampled banks were tested. The four arms ran
concurrently on separate RTX 5090s; the GPU implementation uses the same loss,
checkpoint, AdamW state-copy isolation, fixed learning rate, and disabled
curvature settings as the CPU control.

| Bank | Steps/batch | Held-out cost | Accepted | Grid hit checks | Contact p50 | Wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | **72.47** | **33/81** | **92.6%** | 63.6 mm | 353 s |
| 0 | 2 | 72.86 | 30/81 | 91.4% | **59.8 mm** | 356 s |
| 0 | 3 | 74.27 | 30/81 | 91.4% | 61.2 mm | 361 s |
| 0 | 5 | 72.91 | 32/81 | 91.4% | 65.6 mm | 354 s |
| 1 | 1 | **50.13** | **25/81** | 96.3% | 57.9 mm | 351 s |
| 1 | 2 | 52.25 | **25/81** | 96.3% | 56.6 mm | 360 s |
| 1 | 3 | 51.47 | 24/81 | **97.5%** | 56.8 mm | 354 s |
| 1 | 5 | 51.90 | 23/81 | 93.8% | **56.3 mm** | 344 s |

Two-bank means:

| Steps/batch | Held-out cost | Accepted | Grid hit checks | Contact p50 | Wall time |
|---:|---:|---:|---:|---:|---:|
| 1 | **61.30** | **29.0/81** | **94.4%** | 60.7 mm | 352 s |
| 2 | 62.55 | 27.5/81 | 93.8% | **58.2 mm** | 358 s |
| 3 | 62.87 | 27.0/81 | **94.4%** | 59.0 mm | 357 s |
| 5 | 62.40 | 27.5/81 | 92.6% | 61.0 mm | 349 s |

At this longer horizon, one-step cycling is the most reliable policy: it has
the lowest held-out cost on both banks and the best mean complete acceptance.
Two steps improves median contact error by about 2.5 mm but loses 1.5 accepted
cases on average, largely through slightly worse collision/speed interactions.
Three and five do not catch up with additional training. All arms take about
5.8-6.0 minutes for 1,000 updates, so consecutive repeats provide no optimizer
throughput improvement on GPU either.

## Artifacts

- Ablation script: `scripts/ablate_frozen_batch_updates.py`
- Seed-0 bank: `/tmp/frozen_batch_repeat_b64_u8_20260922.bank.pt`
- Seed-0 matched control: `/tmp/frozen_batch_equal80_corrected_b64_u8_20260922.json`
- Seed-1 matched control: `/tmp/frozen_batch_equal80_corrected_seed1_b64_u8_20260922.json`
- Seed-0 same-unique-row sweep: `/tmp/frozen_batch_same8_corrected_seed0_20260922.json`
- Late-checkpoint matched control: `/tmp/frozen_batch_late_equal80_seed0_20260922.json`
- Late 1,000-update bank-0 arms: `/tmp/frozen_batch_late_equal1000_k{1,2,3,5}_gpu_20260923.json`
- Late 1,000-update bank-1 arms: `/tmp/frozen_batch_late_equal1000_seed1_k{1,2,3,5}_gpu_20260923.json`

The `/tmp` artifacts are ephemeral; rerun the script to regenerate them.
