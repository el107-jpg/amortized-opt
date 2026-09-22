#!/usr/bin/env python
"""Merged phase-2/3 run: direct cost descent + chained BPTT (section 7).

    python scripts/train_phase2.py --device cuda:1 --steps 60000 \
        --init runs/phase1_main/best.pt --scenarios 300000

Everything lands in ``amortized/runs/<name>/``: ``metrics.jsonl`` (one record
per log / eval / event), ``best.pt`` (highest rolling grid acceptance),
``last.pt`` and ``summary.json``.  Checkpoints carry the augmented-Lagrangian
state, so a resume continues the multipliers rather than restarting them.

Resolved decision 6: bf16 autocast, batch >= 512, ``torch.compile``, horizon
bucketing, rolling eval every 2 k steps with the plan's stage-A gates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def configure_inductor() -> None:
    """Settings under which inductor DOES compile this stack on this toolchain.

    The phase-2 chapter recorded ``torch.compile`` as unusable here: inductor
    emitted kernels the conda Triton (3.3.0) could not parse against the
    prebundled Isaac torch (2.7.0+cu128) -- ``'constexpr' has no attribute
    'is_block'`` on the ``triton_poi_fused_stack_*`` kernels -- and then retried
    **every autotune config**, so a compile that should take ~2 minutes ran past
    13 with no first step.

    Re-measured 2026-08-10 with autotuning and inductor's own cudagraphs turned
    off explicitly: the decoder compiles in 19 s for a 3.4x fwd+bwd speedup
    (42.9 -> 12.5 ms at batch 256 x H = 50) and ``FusedCost`` -- the FK/RNEA
    stack the chapter blamed -- compiles in 157 s for 3.3x (212 -> 64 ms), under
    bf16 autocast, with ``suppress_errors`` OFF.  So the failure was the autotune
    retry storm, not the codegen; these are the settings that avoid it.

    ``backend="cudagraphs"`` was also tried and does NOT work: its outputs are
    overwritten by the next invocation and the eager backward then raises, even
    with ``cudagraph_mark_step_begin`` before every call.
    """
    import torch

    torch._inductor.config.max_autotune = False
    torch._inductor.config.max_autotune_pointwise = False
    torch._inductor.config.max_autotune_gemm = False
    torch._inductor.config.coordinate_descent_tuning = False
    torch._inductor.config.triton.cudagraphs = False
    # One compiled unit per (bucket, batch shape) across the cost / tc / bptt
    # steps is already ~10 graphs; the default limit of 8 silently drops back to
    # eager partway through the run.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256


import amortized_opt as A  # noqa: E402
from amortized_opt.lead import resolve_lead  # noqa: E402
from amortized_opt.rhythm import KitRandomization  # noqa: E402
from amortized_opt.phase2_data import (  # noqa: E402
    Phase2Noise,
    Phase2ScenarioConfig,
    generate_cost_pool,
)
from amortized_opt.train_cost import (  # noqa: E402
    LineSearchConfig,
    Phase2Config,
    ViolationCriticConfig,
    WeightSchedule,
    eval_subset,
    load_phase2_checkpoint,
    phase2_al_config,
    phase2_cost_config,
    train_phase2,
    waist_sp_offsets,
)


def _resume_training_states(
    checkpoint: dict,
    *,
    reset_optimizer: bool,
    restore_line_search: bool,
) -> tuple[dict | None, dict | None, int, dict | None]:
    """Select independently resumable state without modifying the checkpoint."""
    extra = checkpoint.get("extra") or {}
    line_search_state = extra.get("line_search") if restore_line_search else None
    if not isinstance(line_search_state, dict):
        line_search_state = None
    return (
        checkpoint.get("al"),
        None if reset_optimizer else checkpoint.get("optimizer"),
        int(checkpoint.get("step") or 0),
        line_search_state,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--min-lr-frac", type=float, default=0.05)
    p.add_argument("--init", default="runs/phase1_main/best.pt",
                   help="phase-1 checkpoint to start from ('' = random init)")
    p.add_argument("--resume", default=None, help="phase-2 checkpoint to continue")
    p.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="on --resume, retain model/AL/absolute step but start with fresh "
             "optimizer state (combine with --line-search-fresh to reset the "
             "selector from current epsilon seed/bounds too)",
    )
    p.add_argument("--reset-al", nargs="*", default=None, metavar="GROUP",
                   help="on --resume, reset these augmented-Lagrangian "
                        "multipliers to lam_init; no names = ALL of them.  "
                        "Reset a group whose CONSTRAINT DEFINITION changed: its "
                        "multiplier is an integrator calibrated against the old "
                        "one, and with decay=0.999 every al_every=100 steps a "
                        "lambda at the 100 ceiling needs ~3e5 steps to fall to "
                        "5, so it never fades on its own.  Name the groups -- "
                        "resetting ALL is usually wrong: on the first fine-tune "
                        "of runs/phase2_19m_waist_sp_d it took lam_terminal "
                        "38.7 -> 1.0, a 38x cut in the price of hitting the "
                        "drum (c_terminal was 23.6 %% of the objective), and "
                        "acceptance fell 56/81 -> 0/81 by step 2 000")
    p.add_argument("--scenarios", type=int, default=250_000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--buckets", type=int, nargs="+", default=[25, 50, 100])
    p.add_argument("--num-future", type=int, default=1,
                   help="future-hit tokens per arm (rhythm-derived default: 1)")
    p.add_argument("--rhythm-idle-frac", type=float, default=0.30,
                   help="fresh-mix share of KIND_RHYTHM_IDLE (Phase A)")
    p.add_argument("--post-hit-frac", type=float, default=0.60,
                   help="fresh-mix share of KIND_POST_HIT (Phase B). Drawn from "
                        "on-policy roller harvests after a real hit (live "
                        "buffer); --post-hit-cache is a legacy overlay only")
    p.add_argument("--far-fetched-frac", type=float, default=0.10,
                   help="fresh-mix share of KIND_FAR_FETCHED (eval-style box "
                        "grid: rest, one strike, no future; not the seed-0 "
                        "eval poses)")
    p.add_argument("--eval-like-frac", type=float, default=0.0,
                   help="fresh-mix share of eval-like (0 = off; rhythm default)")
    p.add_argument("--lead-s", type=float, default=1.0,
                   help="Phase A lead time before the indexed hit (seconds)")
    p.add_argument("--synth-transition-frac", type=float, default=0.25,
                   help="within Phase A, fraction drawn from synth dual clips")
    p.add_argument("--tip-speed-max", type=float, default=4.0)
    p.add_argument("--post-hit-cache", default=None,
                   help="optional legacy npz of distilled KIND_POST_HIT rows "
                        "(scripts/distill_post_hit.py). Prefer the live roller "
                        "buffer; this merge is only a bootstrap while it fills")
    p.add_argument("--legacy-pool", action="store_true",
                   help="use the old plan-native/eval-like/cold generate_cost_pool "
                        "path instead of rhythm idle + far-fetched")
    p.add_argument(
        "--voxel-pool", action="store_true",
        help="train from the positive-witness voxel curriculum: hard-certified "
             "static bootstrap, MPC/feasible/boundary/unknown-probe mix, and "
             "no AL multiplier updates from probes",
    )
    p.add_argument("--voxel-q-cell-rad", type=float, default=0.02)
    p.add_argument(
        "--voxel-bootstrap-rows", type=int, default=10_000,
        help="static rows to hard-certify at startup (0 = all)",
    )
    p.add_argument("--voxel-bootstrap-batch", type=int, default=256)
    p.add_argument(
        "--voxel-bootstrap-stride", type=int, default=4,
        help="knot stride for ordinary along-trajectory bootstrap slices",
    )
    p.add_argument("--voxel-posthit-anchor-tth-s", type=float, default=1.0)
    p.add_argument("--voxel-rest-gap-s", type=float, default=1.0)
    p.add_argument(
        "--voxel-cache", default=None,
        help="optional hard-gate-fingerprinted VoxelPool npz cache",
    )
    p.add_argument("--voxel-probe-capacity", type=int, default=256)
    p.add_argument("--voxel-probe-refresh-every", type=int, default=10)
    p.add_argument("--voxel-probe-shorter-layers", type=int, default=1)
    p.add_argument("--voxel-probe-longer-layers", type=int, default=4)
    p.add_argument(
        "--voxel-probe-timing-step", type=int, default=1,
        help="knots per asymmetric timing-probe layer",
    )
    p.add_argument("--voxel-probe-timing-frac", type=float, default=0.5)
    p.add_argument("--voxel-strike-speed-min", type=float, default=1.0)
    p.add_argument("--voxel-strike-speed-max", type=float, default=3.0)
    p.add_argument("--voxel-frac-mpc", type=float, default=0.50)
    p.add_argument("--voxel-frac-pool", type=float, default=0.30)
    p.add_argument("--voxel-frac-boundary", type=float, default=0.15)
    p.add_argument("--voxel-frac-probe", type=float, default=0.05)
    p.add_argument("--voxel-probe-penalty", type=float, default=1.0)
    p.add_argument(
        "--timing-critic-mode",
        choices=("off", "shadow", "assist", "control"),
        default="off",
        help="task-time violation critic rollout mode; assist retimes a guarded "
             "slice and control retimes all eligible rows after shadow warmup",
    )
    p.add_argument("--timing-critic-heads", type=int, default=5)
    p.add_argument("--timing-critic-hidden-dim", type=int, default=128)
    p.add_argument("--timing-critic-lr", type=float, default=1e-3)
    p.add_argument(
        "--timing-critic-ema-decay", type=float, default=0.95,
        help="EMA decay for the stable critic copy used by timing decisions",
    )
    p.add_argument("--timing-critic-batch", type=int, default=256)
    p.add_argument("--timing-critic-updates", type=int, default=1)
    p.add_argument("--timing-critic-bootstrap-prob", type=float, default=0.8)
    p.add_argument("--timing-critic-prior-scale", type=float, default=0.0)
    p.add_argument("--timing-critic-dropout", type=float, default=0.0)
    p.add_argument("--timing-critic-mc-samples", type=int, default=1)
    p.add_argument(
        "--timing-critic-head-architecture",
        choices=("homogeneous", "heterogeneous"), default="homogeneous",
    )
    p.add_argument("--timing-critic-replay-timescales", action="store_true")
    p.add_argument("--timing-critic-recent-capacity", type=int, default=8192)
    p.add_argument("--timing-critic-audit-capacity", type=int, default=1024)
    p.add_argument("--timing-critic-manageable-frac", type=float, default=0.40)
    p.add_argument("--timing-critic-near-failure-frac", type=float, default=0.30)
    p.add_argument("--timing-critic-boundary-band", type=float, default=0.50)
    p.add_argument("--timing-critic-boundary-loss", type=float, default=0.25)
    p.add_argument("--timing-critic-boundary-temperature", type=float, default=0.50)
    p.add_argument("--timing-critic-boundary-target-band", type=float, default=0.50)
    p.add_argument("--timing-critic-underprediction-mult", type=float, default=2.0)
    p.add_argument("--timing-critic-quantile", type=float, default=0.8)
    p.add_argument(
        "--timing-critic-threshold", type=float, default=0.0,
        help="calibrated nonpositive manageable-score threshold in normalized space",
    )
    p.add_argument("--timing-critic-assist-frac", type=float, default=0.05)
    p.add_argument("--timing-critic-min-assist-samples", type=int, default=2048)
    p.add_argument("--timing-critic-coarse-step", type=int, default=5)
    p.add_argument("--timing-primary-base-tasks-per-512", type=int, default=40)
    p.add_argument("--timing-primary-anchor-per-task", type=int, default=1)
    p.add_argument("--timing-primary-boundary-per-task", type=int, default=1)
    p.add_argument("--timing-primary-interior-per-task", type=int, default=4)
    p.add_argument("--timing-primary-mpc-terminal-limit", type=float, default=0.040)
    p.add_argument(
        "--timing-primary-mpc-collision-sdf-limit", type=float, default=-0.010,
    )
    p.add_argument(
        "--timing-critic-representation",
        choices=("regression", "distributional"), default="regression",
    )
    p.add_argument("--timing-acceptable-normalized-violation", type=float, default=0.2)
    p.add_argument("--timing-risk-min-precision", type=float, default=0.7)
    p.add_argument("--timing-risk-recalibrate-every", type=int, default=25)
    p.add_argument("--timing-global-ood-fraction", type=float, default=4.0)
    p.add_argument("--timing-global-ood-weight", type=float, default=0.0025)
    p.add_argument("--timing-anchor-curriculum", action="store_true")
    p.add_argument("--timing-anchor-min-samples", type=int, default=2048)
    p.add_argument("--timing-anchor-min-close-rate", type=float, default=0.50)
    p.add_argument("--timing-terminal-limit", type=float, default=0.020)
    p.add_argument("--timing-collision-sdf-limit", type=float, default=-0.005)
    p.add_argument("--timing-joint-limit", type=float, default=0.020)
    p.add_argument("--timing-torque-limit", type=float, default=8.0)
    p.add_argument("--hit-time-jitter-s", type=float, default=0.05,
                   help="uniform +- jitter on each clip hit's time at snapshot "
                        "time, seconds; independent per hit under the rhythm "
                        "mix.  BAKED INTO THE POOL -- same cache caveat as "
                        "--drum-tilt-deg")
    p.add_argument("--hit-loudness-jitter", type=float, default=0.0,
                   help="uniform +- jitter on each hit's 0-1 loudness before "
                        "the 1-3 m/s speed mapping (per hit, clipped).  BAKED "
                        "INTO THE POOL -- same cache caveat as --drum-tilt-deg")
    p.add_argument("--rare-drum-boost", type=float, default=0.0,
                   help="fraction of plan-native draws whose clip instant is "
                        "TARGETED at a rare drum (inverse-frequency weighted; "
                        "floor_tom/cymbals) instead of uniform.  BAKED INTO "
                        "THE POOL -- same cache caveat as --drum-tilt-deg")
    p.add_argument("--pair-frac", type=float, default=0.0,
                   help="fraction of scenarios from the PAIR family: both "
                        "hands striking, (left, right) drum combination drawn "
                        "UNIFORMLY over the 21 distinct pairs + 5 same-drum "
                        "ones, hands assigned by lateral position so the arms "
                        "never cross.  The rhythm clips cover only 5 of the 31 "
                        "reachable pairs (hihat+snare alone is 83.6 %%) and "
                        "never put floor_tom or cymbal_left in a two-handed "
                        "row at all, so without this the model sees one "
                        "two-hand configuration and nothing else")
    p.add_argument("--cold-start-frac", type=float, default=0.0,
                   help="fraction of scenarios whose waist yaw is drawn "
                        "independently of the goals (legacy pool only; 0 under "
                        "the rhythm-derived default mix)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--compile-cost", action="store_true",
                   help="also compile the physics (broken on this Triton -- see "
                        "Phase2Config.compile_cost)")
    # model architecture overrides
    p.add_argument("--d-model", type=int, default=None, help="decoder width (default 384)")
    p.add_argument("--dec-layers", type=int, default=None, help="decoder depth (default 3)")
    p.add_argument("--d-ctx", type=int, default=None, help="encoder width (default 256)")
    p.add_argument("--ctx-layers", type=int, default=None, help="encoder depth (default 2)")
    p.add_argument("--dec-ffn-mult", type=float, default=None,
                   help="decoder FFN multiplier (default 4.0)")
    p.add_argument("--dec-rope", action="store_true",
                   help="rotary position on decoder SELF-attention (Llama RoPE "
                        "on the [state]+waypoint grid).  No extra parameters; "
                        "cross-attention and the encoder stay unrotated.  "
                        "Off by default so existing checkpoints stay "
                        "bit-identical; pass this to turn it on over a warm "
                        "start (strict load still works)")
    p.add_argument("--dec-plant-feats", action="store_true",
                   help="analytic Cartesian (tip/wrist/elbow) + knot-RNEA of "
                        "init_traj on decoder tokens (ModelConfig.dec_plant_feats). "
                        "Zero-init residual projections, so a warm start that "
                        "predates the flag loads with the plant channels as a "
                        "no-op (missing keys).  Off by default")
    p.add_argument("--traj-head-fp32", action="store_true",
                   help="run the trajectory output head, physical-unit "
                        "unprojection and trajectory clamp in explicit fp32 "
                        "while transformer blocks remain under AMP. "
                        "Behaviour-only and off by default")
    p.add_argument("--no-aux-heads", action="store_true",
                   help="disable auxiliary heads (vel/acc/torque/fk)")
    p.add_argument("--aux-torque-tracking", action="store_true",
                   help="regress head_torque on the DEMANDED PD tracking torque "
                        "|tau_rnea + tau_fb| -- the same array w_track_torque "
                        "prices -- instead of the RNEA segment max. OFF by "
                        "default: it redefines a live objective, and the "
                        "phase-1 head was fit to the other quantity.")
    # objective knobs (the pilot race varies these)
    p.add_argument("--w-aux", type=float, default=None)
    p.add_argument("--w-tc", type=float, default=None)
    p.add_argument("--tc-frac", type=float, default=None)
    p.add_argument("--w-mpc-init-tie", type=float, default=None,
                   help="masked Huber weight tying each sampled MPC replay "
                        "output to its stored shifted-parent initialization; "
                        "fresh/quintic rows are excluded (default 0)")
    p.add_argument("--suffix-frac", type=float, default=None,
                   help="fraction of the batch that also solves a random "
                        "suffix of the main solve's own plan (suffix-Bellman "
                        "term; 0 = off, the default)")
    p.add_argument("--w-suffix-cost", type=float, default=None,
                   help="FusedCost weight on the suffix solve (default 1.0)")
    p.add_argument("--w-suffix-tie", type=float, default=None,
                   help="Huber weight tying the suffix solve to the parent "
                        "plan's tail (default 50, the w_tc scale)")
    p.add_argument("--suffix-warmup", type=int, default=None,
                   help="smoothstep ramp of both suffix weights, steps "
                        "(default 1000)")
    p.add_argument("--dls-terminal-frac", type=float, default=None,
                   help="deterministic batch fraction receiving an arm-only "
                        "DLS terminal lookahead (0 disables it)")
    p.add_argument("--w-dls-terminal-cost", type=float, default=None,
                   help="full-physics cost weight on the DLS-corrected "
                        "straight-through trajectory")
    p.add_argument("--w-dls-terminal-tie", type=float, default=None,
                   help="Huber weight teaching the raw actor trajectory to "
                        "internalize its detached DLS correction")
    p.add_argument("--dls-terminal-steps", type=int, default=None,
                   help="DLS iterations per active hand (default 1)")
    p.add_argument("--dls-terminal-window", type=int, default=None,
                   help="final approach knots over which to spread the "
                        "terminal arm correction (default 16)")
    p.add_argument("--dls-terminal-damping", type=float, default=None,
                   help="DLS damping coefficient (default 0.05)")
    p.add_argument("--dls-terminal-max-step-rad", type=float, default=None,
                   help="per-iteration, per-joint DLS trust region in radians "
                        "(default 0.35)")
    p.add_argument("--terminal-lookahead-mode", choices=("dls", "curvature"),
                   default=None, help="terminal teacher: current arm-only DLS "
                        "or full-cost analytic trajectory-curvature step")
    p.add_argument("--curvature-damping", type=float, default=None,
                   help="positive diagonal regularizer for the trajectory solve")
    p.add_argument("--curvature-position-weight", type=float, default=None,
                   help="terminal contact-position Gauss-Newton curvature weight")
    p.add_argument("--curvature-velocity-weight", type=float, default=None,
                   help="terminal contact-velocity Gauss-Newton curvature weight")
    p.add_argument("--curvature-max-step-rad", type=float, default=None,
                   help="maximum absolute waypoint change in the curvature teacher")
    p.add_argument("--curvature-direct-frac", type=float, default=None,
                   help="fraction of actor rows whose trajectory-space backward "
                        "gradient is directly curvature-preconditioned (default 0)")
    p.add_argument("--curvature-direct-mix", type=float, default=None,
                   help="mix between ordinary and norm-matched preconditioned "
                        "trajectory gradient (default 1)")
    p.add_argument("--yaw-now-jitter", type=float, default=None,
                   help="std-dev (rad) of Gaussian jitter on the yaw_now "
                        "conditioning token, teaching the waist-sp head to "
                        "anchor at the aim instead of tracking yaw_now "
                        "(default 0.3; 0 = pre-fix behaviour)")
    p.add_argument("--rho-scale", type=float, default=1.0)
    p.add_argument("--ramp-start", type=int, default=None)
    p.add_argument("--ramp-end", type=int, default=None)
    p.add_argument("--torque-physics", default=None,
                   choices=["full", "diag_grav", "diag_grav_frozen"],
                   help="torque hinge plant: full RNEA (default) or level-A "
                        "diag(M)+gravity proxy")
    p.add_argument("--torque-full-start", type=int, default=None,
                   help="absolute step to begin ramping torque_full_blend 0->1 "
                        "(with --torque-physics diag_*).  0 + default ramp=0 "
                        "means full RNEA from step 0")
    p.add_argument("--torque-full-ramp", type=int, default=None,
                   help="smoothstep span for blending level-A torque into full "
                        "RNEA (default 0)")
    p.add_argument("--bptt-start-frac", type=float, default=None)
    p.add_argument("--bptt-max-frac", type=float, default=None)
    p.add_argument("--n-sub", type=int, default=1)
    p.add_argument("--w-torque", type=float, default=None,
                   help="override phase2_cost_config's torque price "
                        "(default 20; characteristic 1 N·m)")
    p.add_argument("--w-velocity", type=float, default=None,
                   help="override phase2_cost_config's velocity price "
                        "(default 40; characteristic 0.5 rad/s)")
    p.add_argument("--w-joint-box", type=float, default=None,
                   help="override phase2_cost_config's joint-box price "
                        "(default 400; characteristic 0.05 rad).  Scaled so "
                        "a 0.05 rad mean excess costs the same as 1 N·m of "
                        "torque at λ=1")
    p.add_argument("--w-torque-peak", type=float, default=None,
                   help="price on the SUM of ReLU hard torque excess "
                        "(dense grad; AL integrates max).  Default 2.0")
    p.add_argument("--torque-peak-ramp", type=int, default=None,
                   help="steps over which the peak half ramps 0 -> 1 "
                        "(default: off, full from step 0)")
    p.add_argument("--w-velocity-peak", type=float, default=None,
                   help="price on the SUM of ReLU hard velocity excess "
                        "(dense grad; AL integrates max).  Default 2.0")
    p.add_argument("--velocity-peak-ramp", type=int, default=None,
                   help="steps over which the velocity peak half ramps 0 -> 1 "
                        "(default: off, full from step 0)")
    p.add_argument("--w-torque-static", type=float, default=None,
                   help="price on gravity-only static torque: mean softplus on "
                        "|RNEA(q,0,0)| against effort limits (no AL multiplier)")
    p.add_argument("--torque-static-only", action="store_true",
                   help="skip coupled RNEA torque hinge/AL; use with "
                        "--w-torque-static and w_torque=0")
    p.add_argument("--torque-al-off", action="store_true",
                   help="keep the torque hinge but skip all torque AL groups "
                        "(level-A / simplified-torque experiments)")
    p.add_argument("--torque-static-al", action="store_true",
                   help="with --torque-static-only: drive torque_* AL groups "
                        "from |RNEA(q,0,0)| excess (requires w_torque_static>0)")
    p.add_argument("--static-torque-tighten", default=None,
                   help="per-joint static limit tightening mode "
                        "(decoder_measured = divide limits by measured "
                        "dyn/static RMS ratios; off = uniform)")
    p.add_argument("--eta-torque", type=float, default=None,
                   help="soft torque hinge fraction of hard rating (default 0.80; "
                        "0.72 = 10%% tighter)")
    p.add_argument("--w-track-torque", type=float, default=None,
                   help="price on the MEAN of the TRACKING-torque hinge: the "
                        "torque the PD controller demands while tracking the "
                        "plan (RNEA + estimated PD feedback), not the plan's "
                        "RNEA alone.  Positive => it also becomes the 'torque' "
                        "AL group's violation, AND the rolling eval's ok_torque "
                        "gate switches to the same quantity (default off)")
    p.add_argument("--w-track-torque-peak", type=float, default=None,
                   help="fixed price on the SMOOTH-MAX of the same hinge; rides "
                        "the per-joint limit_alpha ramp, no AL multiplier")
    p.add_argument("--soft-wrist-effort-scale", type=float, default=None,
                   help="scoring-only multiplier on wrist pitch/yaw effort "
                        "ratings in the cost hinge and ok_torque gate "
                        "(default 1.0 = Isaac 5 N*m; try 2.0 for brief "
                        "reflected-Kd spikes).  Does NOT change IdealPD clamp")
    p.add_argument("--plant-saturate-effort", action="store_true",
                   help="clamp the training plant's demanded torque to the "
                        "joint's usable effort inside every PD substep (roller, "
                        "TC, BPTT and the closed-loop probe), as Isaac does")
    p.add_argument("--w-smooth-jerk", type=float, default=None,
                   help="override the restored jerk weight (default 1e-4)")
    p.add_argument("--w-collision", type=float, default=None)
    p.add_argument("--w-stick-stick", type=float, default=None,
                   help="cross-hand stick/wrist capsule-capsule collision "
                        "weight (same margin-softplus units as "
                        "w_collision_self, whose shipped value is 100).  "
                        "Default None keeps CostConfig's 0.0 = OFF")
    p.add_argument("--w-tip-crossover", type=float, default=None,
                   help="plain (non-AL) tip lateral-ordering hinge: softplus "
                        "on margin-(tip_L.y-tip_R.y) in the torso frame.  "
                        "Discourages left tip sitting right of right tip.  "
                        "Default None keeps CostConfig's 0.0 = OFF")
    p.add_argument("--w-terminal-aim", type=float, default=None,
                   help="plain (non-AL) in-plane pull toward the AIM after a "
                        "configurable deadband. phase2_cost_config default is 40; "
                        "pass 0 to restore CostConfig's off switch.  Does not "
                        "ride lam_terminal")
    p.add_argument("--terminal-aim-deadband-m", type=float, default=None,
                   help="in-plane terminal AIM deadband in metres; "
                        "phase2_cost_config default is 0.005")
    p.add_argument("--tip-crossover-ramp", type=int, default=None,
                   help="steps over which tip_crossover ramps 0 -> 1 "
                        "(default: off, full from step 0)")
    p.add_argument("--w-collision-peak", type=float, default=None,
                   help="price on the SMOOTH-MAX half of the collision term -- "
                        "the reduction ok_collision's zero-tolerance check "
                        "actually corresponds to.  The mean is diluted by "
                        "~2100-4200 (knot, point, drum) pairs.  OFF by default; "
                        "try ~500 (the hinge is in metres, ~0.03 at a 27 mm "
                        "penetration).  Priced by its own AL group")
    p.add_argument("--collision-peak-ramp", type=int, default=None,
                   help="steps over which the collision peak half ramps 0 -> 1 "
                        "(default: off, full from step 0)")
    p.add_argument("--w-joint-box-peak", type=float, default=None,
                   help="price on the SUM of per-waypoint ReLU hard-box excess "
                        "(dense grad on every violating h,j; scales with count). "
                        "AL integrates the horizon max.  Default 40 in phase2_cost_config")
    p.add_argument("--joint-box-peak-ramp", type=int, default=None,
                   help="steps over which the joint-box peak half ramps 0 -> 1 "
                        "(default: off, full from step 0)")
    # --- section 7d: keep the waist upright / torso box stable --
    p.add_argument("--waist-pr-box", type=float, default=None,
                   help="shrink the joint box on waist_pitch/waist_roll to this "
                        "half-width in rad (URDF is 0.52).  When enabled, the "
                        "same existing joint_box path also narrows waist_yaw to "
                        "a fixed +-80 deg guardrail.  Off by default.")
    p.add_argument("--w-waist-upright", type=float, default=None,
                   help="quadratic upright well on waist pitch/roll, rad^-2 "
                        "(CostConfig.w_waist_upright).  Yaw is excluded")
    p.add_argument("--w-arm-compact", type=float, default=None,
                   help="plain objective weight on active-arm COM-proxy height "
                        "and horizontal reach beyond their deadbands (m^-2). "
                        "Measured in the torso frame; q0 is excluded")
    p.add_argument("--arm-compact-mode", choices=("linear", "hinge_sq"),
                   default=None,
                   help="linear continuously prices COM-proxy height+radius; "
                        "hinge_sq preserves the old squared-ReLU deadbands")
    p.add_argument("--arm-compact-height-max", type=float, default=None,
                   help="free active-arm COM-proxy height in torso frame (m)")
    p.add_argument("--arm-compact-radius-max", type=float, default=None,
                   help="free active-arm COM-proxy horizontal radius from torso (m)")
    p.add_argument("--w-waist-pr-terminal", type=float, default=None,
                   help="economic-MPC V_f on waist P/R at the NEARER active hit, "
                        "rad^-2 (CostConfig.w_waist_pr_terminal).  Always vs 0. "
                        "The later hand's hit only lengthens the waist window; "
                        "this term is the strike the committed prefix is for")
    p.add_argument("--waist-ramp", type=int, default=None,
                   help="steps over which BOTH waist terms ramp 0 -> 1 "
                        "(default: off, full from step 0; A/B only)")
    p.add_argument("--waist-full", action="store_true",
                   help="waist terms at FULL weight from step 0 (sets "
                        "weight_schedule.waist_start = -1, waist_ramp = 1).  "
                        "Use when the warm start was already trained with waist "
                        "terms, so there is no multiplier shock to absorb")
    p.add_argument("--full-weights", action="store_true",
                   help="Re-apply schedule.snap_full() and set suffix_warmup=1 "
                        "from step 0.  Cost weights are already full by default; "
                        "use on continuation when aux ramps should not reopen.")
    p.add_argument("--no-waist-pr-full-limits", action="store_true",
                   help="OFF SWITCH for the default: normally the waist "
                        "pitch/roll torque/velocity/joint-box hinges are priced "
                        "at FULL strength from step 0 while every other joint "
                        "follows the lo_limits ramp (weight_schedule."
                        "full_limit_joints).  Measured on phase2_19m_ctxv2, the "
                        "5 %% window let pitch/roll excursion grow 0.24 -> 0.64 "
                        "rad while the priced yaw rate fell -- and 0.64 rad is "
                        "28.4 N*m static against a 25 N*m rating.  Pass this "
                        "only to A/B against the old uniform ramp")
    p.add_argument("--w-waist-aim", type=float, default=None,
                   help="waist-aim IL penalty weight, rad^-2 "
                        "(CostConfig.w_waist_aim).  Penalises yaw deviation from "
                        "the waist_aim reference trajectory, and pitch/roll "
                        "deviation from zero, over the WHOLE horizon on every "
                        "batch (fresh pool + replay), not just in the seed")
    p.add_argument("--w-waist-yaw-vel", type=float, default=None,
                   help="price on mean waist-yaw RATE^2, (rad/s)^-2 "
                        "(CostConfig.w_waist_yaw_vel).  Stops the planner buying "
                        "impact speed by winding up the torso.  A steady 1 rad/s "
                        "turn pays exactly this weight")
    p.add_argument("--waist-upright-ease-wp", type=int, default=None,
                   help="pool seeds' pitch/roll ease from the sampled lean to "
                        "upright over this many waypoints "
                        "(ScenarioConfig.waist_upright_ease_wp; try 5, and "
                        "match the phase-1 warm start's setting).  Default "
                        "None holds the measured lean, the pre-existing "
                        "behaviour")
    p.add_argument("--waist-sp-head", action="store_true",
                   help="LEARNED waist-yaw set-point (docs/waist_sp_head_plan.md): "
                        "an encoder head, zero-init residual on the analytic "
                        "aim_yaw, becomes the --w-waist-aim reference (one held "
                        "yaw per goal segment) and gradient flows through the "
                        "term into the head.  Sets ModelConfig.waist_sp_head, "
                        "Phase2Config.waist_sp_from_head and "
                        "CostConfig.waist_aim_yaw_only; give --w-waist-aim a "
                        "non-zero weight or the anchor prices nothing")
    p.add_argument("--waist-aim-deadband", type=float, default=None,
                   help="free zone of the waist-aim YAW error, rad "
                        "(CostConfig.waist_aim_deadband; try 0.1-0.2 with the "
                        "head).  Only the excess over the band is squared")
    p.add_argument("--waist-aim-future-tau", type=float, default=1.0,
                   help="time constant (s) of the FUTURE-aware waist aim law "
                        "(WaistAimConfig.future_tau).  The aim is taken over "
                        "each hand's active goal AND its next hits, weighted by "
                        "time to hit, which is what stops the commanded yaw "
                        "square-waving as a goal crosses the horizon cap "
                        "(measured 131 deg p2p on gmd_eval_000006).  0 restores "
                        "the current-goals-only law bit for bit.  BAKED INTO "
                        "THE POOL (the aim_yaw column): change it and the pool "
                        "cache must change name too -- scripts/_run_19m_waist.sh "
                        "keys POOL_TAG on it")
    p.add_argument("--w-waist-sp-tether", type=float, default=None,
                   help="tether on the head's residual (sp - aim_yaw), rad^-2 "
                        "(Phase2Config.w_waist_sp_tether).  Bounds how far the "
                        "learned aim may walk from the analytic law the pool's "
                        "inits still ramp to.  Watch sp_res in the train log: "
                        "saturation means this weight is the binding constraint")
    p.add_argument("--waist-pr-sp-head", action="store_true",
                   help="LEARNED waist pitch/roll set-point "
                        "(docs/waist_pr_sp_head_plan.md): encoder head, "
                        "zero-init residual on upright, becomes the "
                        "--w-waist-upright reference (one held lean per goal "
                        "segment).  Does NOT put measured lean in the encoder.  "
                        "Sets ModelConfig.waist_pr_sp_head and "
                        "Phase2Config.waist_pr_sp_from_head; give "
                        "--w-waist-upright a non-zero weight or the anchor "
                        "prices nothing.  Default tether 5.0 (override with "
                        "--w-waist-pr-sp-tether)")
    p.add_argument("--w-waist-pr-sp-tether", type=float, default=None,
                   help="tether on the P/R set-point mean(sp²), rad^-2 "
                        "(Phase2Config.w_waist_pr_sp_tether).  Default 5.0 when "
                        "--waist-pr-sp-head is on, 0 otherwise.  Watch pr in "
                        "the train log: saturation at ~520 mrad is the rail")
    p.add_argument("--w-waist-pr-sp-hold", type=float, default=None,
                   help="hold the encode-time lean: mean((sp_pr - q0_pr)²), "
                        "rad^-2 (Phase2Config.w_waist_pr_sp_hold).  q0 is "
                        "stopgrad and is NOT an encoder input.  Default 5.0 "
                        "when --waist-pr-sp-head is on, 0 otherwise.  Watch "
                        "hold in the train log (mrad |sp - q0|)")
    p.add_argument("--w-waist-pr-sp-hold-band", type=float, default=None,
                   help="hinge relu(|sp - clamp(q0)| - band) on the encoder "
                        "(Phase2Config.w_waist_pr_sp_hold_band).  Same band "
                        "as --waist-pr-band-rad.  Makes |sp-q0| > band "
                        "illegal so the decoder window contains the start; "
                        "equal tether+hold cannot (fixed point is q0/2).  "
                        "Default 10 when --waist-pr-sp-head is on, 0 "
                        "otherwise.  Watch hB in the train log (mrad excess)")
    p.add_argument("--w-waist-pr-sp-distill", type=float, default=None,
                   help="gated strike-knot distill of decoder waist P/R onto "
                        "the encoder set-point "
                        "(Phase2Config.w_waist_pr_sp_distill).  Encoder copies "
                        "q_pr at hit only when the plan is on-head, in-box, "
                        "collision-free and still in the 0.15 band.  0 "
                        "(default) is off.  Try 10 with --waist-pr-sp-head; "
                        "watch pr_distill_frac / pr_distill_delta")
    p.add_argument("--pr-distill-hit-m", type=float, default=None,
                   help="on-head slack for that gate, metres "
                        "(Phase2Config.pr_distill_hit_m).  Default 0.02 "
                        "matches the 81-grid hit margin")
    p.add_argument("--waist-pr-ema", action="store_true",
                   help="closed-loop leaky EMA of the CLIPPED P/R command as "
                        "the upright-well reference (StrikeContext.waist_pr_ref). "
                        "No encoder P/R head; snapshot / 81-grid keep the well "
                        "vs 0.  Conflicts with --waist-pr-sp-head and "
                        "--freeze-waist-pr.  Peak waist_pr_band is forced off "
                        "unless --w-waist-pr-band is set")
    p.add_argument("--waist-pr-ema-alpha", type=float, default=0.1,
                   help="EMA mix on the last clipped P/R cmd (Phase2Config."
                        "waist_pr_ema_alpha).  1 = lag-1, 0 = freeze the "
                        "spawn q0")
    p.add_argument("--waist-pr-ema-leak", type=float, default=0.02,
                   help="per-tick leak of that EMA toward 0 (Phase2Config."
                        "waist_pr_ema_leak).  The upright attractor; without "
                        "it EMA is only a longer time constant")
    p.add_argument("--w-waist-sp-hold", type=float, default=None,
                   help="same hold on the yaw head, mean((sp_yaw - q0_yaw)²) "
                        "(Phase2Config.w_waist_sp_hold).  Default 0: yaw is "
                        "supposed to walk toward aim_yaw")
    p.add_argument("--w-waist-sp-smooth", type=float, default=None,
                   help="cross-segment set-point smoothness "
                        "(Phase2Config.w_waist_sp_smooth): mean((sp_new - "
                        "sp_old)²) where sp_new is a second encode after "
                        "promoting the soonest current goal to future[0].  "
                        "Prices the jump intra-segment waist terms cannot see "
                        "(closed loop re-encodes every hit).  Also prices P/R "
                        "when that head is on.  0 (default) is off, exactly")
    # --- zeroth-order search distillation for the set-point head -------------
    p.add_argument("--waist-sp-search-frac", type=float, default=None,
                   help="ZEROTH-ORDER SEARCH DISTILLATION of the waist set-point "
                        "head (Phase2Config.waist_sp_search_frac).  Fraction of "
                        "each BATCH (its leading rows) on which the envelope "
                        "V(psi) = cost after the arms re-plan around torso yaw "
                        "psi is evaluated at K candidate yaws and the head is "
                        "regressed onto the softmin winner.  Fixes the "
                        "consistency-term co-drift: --w-waist-aim prices "
                        "||traj_yaw - sp||, whose gradient grades the set-point "
                        "against its own plan, so the pair walked to ~1 rad off "
                        "the analytic aim on 2026-08-14.  Needs "
                        "--w-waist-sp-search too.  0 (default) is off, exactly")
    p.add_argument("--waist-sp-search-k", type=int, default=None,
                   help="TOTAL candidate yaws per row including the mandatory "
                        "analytic-aim column (Phase2Config.waist_sp_search_k); "
                        "the rest is a quadratic symmetric ladder on the head's "
                        "own set-point.  Odd values round down.  Cost is linear "
                        "in K x --waist-sp-search-frac")
    p.add_argument("--waist-sp-search-span", type=float, default=None,
                   help="half-width of that ladder, rad "
                        "(Phase2Config.waist_sp_search_span).  The distillation "
                        "target is a convex mix of the candidates, so this is a "
                        "HARD bound on how far one step can move the set-point")
    p.add_argument("--waist-sp-search-temp", type=float, default=None,
                   help="softmin temperature (Phase2Config.waist_sp_search_temp), "
                        "dimensionless -- candidate scores are min-max "
                        "normalized per row first.  -> 0 is hard argmin, large "
                        "is the candidate mean")
    p.add_argument("--w-waist-sp-search", type=float, default=None,
                   help="price on Huber(sp - detach(sp*)) "
                        "(Phase2Config.w_waist_sp_search).  0 (default) means "
                        "the search never runs whatever the frac is.  Watch "
                        "srch V in the train log: a V-spread near 0 says the "
                        "set-point does not change the outcome at all")
    p.add_argument("--freeze-waist-pr", action="store_true",
                   help="remove waist_pitch / waist_roll from the CONTROLLER: "
                        "the decoder's columns for them are pinned to the ready "
                        "pose (0 rad) and their aux rows zeroed, so no cost "
                        "weight can trade them away.  waist_yaw stays free.  "
                        "Stored in model_cfg, so eval / deploy inherit it")
    # --- section 7d leg 2: the strike-masked tau^2 effort term ----------------
    p.add_argument("--w-effort", type=float, default=None,
                   help="price on mean tau^2 (N^2 m^2), faded out around each "
                        "hit.  An objective, not an AL group.  Expect 1e-4..1e-2")
    p.add_argument("--effort-fade-wp", type=int, default=None,
                   help="waypoints before a hit over which the effort term is "
                        "exactly OFF (it ramps back over the same span again)")
    p.add_argument("--effort-ramp", type=int, default=None,
                   help="steps over which w_effort ramps 0 -> 1 "
                        "(default: off, full from step 0)")
    # ---- the idle arm ------------------------------------------------------
    p.add_argument("--no-idle-cylinder", action="store_true",
                   help="disable the idle-cylinder park entirely: no park seed, "
                        "no rest stretches in the roller, and w_idle_cylinder / "
                        "w_idle_vel stay exact zeros")
    p.add_argument("--idle-cylinder", default=None,
                   help="path to an idle-cylinder library (default: "
                        "assets/idle_cylinder_poses.json when it exists)")
    p.add_argument("--no-idle-home", action="store_true",
                   help="deprecated alias for --no-idle-cylinder")
    p.add_argument("--idle-home", default=None,
                   help="deprecated alias for --idle-cylinder")
    p.add_argument("--no-idle-token", action="store_true",
                   help="drop the idle arm's own CONTEXT token (the model then "
                        "learns the park from the seed alone).  Also the "
                        "way to load a pre-2026-08-12 checkpoint strictly")
    p.add_argument("--w-idle-cylinder", type=float, default=None,
                   help="price on tip-outside-cylinder hinge for a hand with "
                        "no strike (default 1.0; needs a library to do anything)")
    p.add_argument("--w-idle-home", type=float, default=None,
                   help="deprecated alias for --w-idle-cylinder")
    p.add_argument("--w-idle-vel", type=float, default=None,
                   help="price on the idle arm's terminal-weighted qd^2 -- the "
                        "'arrive stopped and stay stopped' half (default 0.02)")
    p.add_argument("--idle-ramp", type=int, default=None,
                   help="steps over which both idle weights ramp in "
                        "(default: off, full from step 0)")
    p.add_argument("--idle-settle-wp", type=int, default=10,
                   help="waypoints the retraction eases over before holding "
                        "inside the cylinder")
    p.add_argument("--rest-ticks", type=int, default=25,
                   help="ticks a BOTH-hands-idle roller slot keeps simulating "
                        "before respawn (0 = delete rests, the old behaviour)")
    p.add_argument("--rest-frac", type=float, default=0.25,
                   help="fraction of naturally both-idle scenario draws kept in "
                        "the fresh pool instead of rejected")
    p.add_argument("--w-waist-pr-vel", type=float, default=None,
                   help="price on the waist PITCH/ROLL rate, (rad/s)^-2 "
                        "(CostConfig.w_waist_pr_vel).  The lean complement of "
                        "--w-waist-yaw-vel: the weight reads directly as a "
                        "target rate, so a torso leaning at a steady 1 rad/s "
                        "pays it exactly.  Keeps the lean POSITIONAL -- settle "
                        "into it and strike, rather than swing through it")
    p.add_argument("--w-waist-pr-band", type=float, default=None,
                   help="AL weight on decoder waist P/R vs stopgrad(encoder "
                        "sp) (CostConfig.w_waist_pr_band).  Decoder pitch/roll "
                        "must stay within --waist-pr-band-rad of the encoder "
                        "set-point; peak ReLU excess, group waist_pr_band.  "
                        "Not the encoder-to-upright tether "
                        "(--w-waist-pr-sp-tether).  phase2_cost_config "
                        "defaults this to 1.0; pass 0 for A/B")
    p.add_argument("--waist-pr-band-rad", type=float, default=None,
                   help="deadband in rad around stopgrad(encoder P/R sp) "
                        "(CostConfig.waist_pr_band_rad, default 0.15).  "
                        "0 is a legal hinge at |q-sp|")
    p.add_argument("--clamp-traj-to-box", action="store_true",
                   help="clamp the emitted trajectory to the URDF position box "
                        "with a STRAIGHT-THROUGH gradient "
                        "(ModelConfig.clamp_traj_to_box).  Without it nothing "
                        "structurally bounds the plan and the decoder can (and "
                        "does) emit joint targets the plant will saturate")
    p.add_argument("--drum-tilt-deg", type=float, default=0.0,
                   help="per-drum TILT domain randomization, DEGREES of "
                        "half-angle on the head normal "
                        "(KitRandomization.per_drum_tilt, which is radians -- "
                        "this flag converts).  0 (the default) reproduces the "
                        "flat-normal kit every run before 2026-08-13 used.  "
                        "BAKED INTO THE POOL: the --pool-cache fingerprint "
                        "refuses a mismatch (tilt is part of the content key)")
    p.add_argument("--drum-radius-frac", type=float, default=0.0,
                   help="per-drum RADIUS domain randomization, as a FRACTION "
                        "(KitRandomization.per_drum_radius): each drum's radius "
                        "is scaled by an independent uniform draw in "
                        "[1-f, 1+f].  0 (the default) reproduces the "
                        "nominal-radius kit every run before 2026-08-14 used.  "
                        "Radius is the last drum feature that never varied, so "
                        "the weights reading its token were unidentified -- the "
                        "yaw_now failure class.  BAKED INTO THE POOL: the "
                        "--pool-cache fingerprint refuses a mismatch")
    p.add_argument("--lr-horizon", type=int, default=None,
                   help="denominator of the LR cosine, in steps (default: "
                        "--steps).  Set it to the PREDECESSOR run's budget to "
                        "continue on its LR trajectory under a shorter step cap")
    p.add_argument("--no-roller-regoal", action="store_true",
                   help="reproduce the old respawn-on-hit roller (no in-place "
                        "per-hand re-goal, no time-to-hit ageing)")
    p.add_argument("--stage-a-gate", action="store_true",
                   help="hold the weight ramp at lo_limits until the stage-A "
                        "metrics pass (see WeightSchedule.gate_stage_a)")
    p.add_argument("--stage-a-open", action="store_true",
                   help="seed the stage-A gate OPEN (a resume from a checkpoint "
                        "that already passed it)")
    p.add_argument("--short-wp", type=int, default=8,
                   help="waypoints defining the final-approach band that is "
                        "oversampled in the fresh pool and the MPC replay")
    p.add_argument("--short-boost", type=float, default=1.0,
                   help="sampling-weight multiplier for that band (1 = uniform)")
    p.add_argument("--probe-every", type=int, default=0,
                   help="steps between CLOSED-LOOP probes; 0 = off.  best.pt is "
                        "gated on the probe not regressing (see train_phase2)")
    p.add_argument("--probe-clips", type=int, default=1)
    p.add_argument("--probe-steps", type=int, default=250)
    p.add_argument("--probe-lead", type=float, default=0.0,
                   help="MPCGenConfig.hit_lead_s used by the probe")
    p.add_argument("--cmd-lead", default="0.0",
                   help="reference lead executed by the training roller / TC / "
                        "BPTT plant: a float in seconds, or a preset name from "
                        "amortized_opt.lead.LEAD_PRESETS ('fit'/'sparse'/'wide'). "
                        "MUST match the deployed MPCGenConfig.cmd_lead_s")
    p.add_argument("--probe-cmd-lead", default=None,
                   help="reference lead used by the closed-loop probe; defaults "
                        "to --cmd-lead")
    p.add_argument("--probe-freeze", type=int, default=5,
                   help="MPCGenConfig.freeze_wp used by the probe "
                        "(5 = ~100 ms stop-refining window; 0 = off)")
    p.add_argument("--no-probe-bounce", action="store_true",
                   help="disable the BPTT placeholder bounce on probe fires "
                        "(default ON: next-hit quintic is seeded from the "
                        "post-impulse plant, not a stick still inside the drum)")
    p.add_argument("--plateau-evals", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget; stops cleanly at the next step")
    p.add_argument("--target-acceptance", type=float, default=None,
                   help="rolling-eval acceptance that ends the run early "
                        "(default 0.60; raise it to spend the whole budget)")
    # bookkeeping
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="write TensorBoard scalars under <out>/tb (default on)")
    p.add_argument("--eval-cases", type=int, default=81)
    p.add_argument("--eval-start-above-drums", nargs="*", default=["snare", "floor_tom"],
                   help="reject rolling-eval box starts whose tip z is below "
                        "the highest of these drum heads (pelvis frame), park "
                        "the idle arm at ready instead of URDF zero, and drop "
                        "any start whose tip/shaft/wrist sits under any drum "
                        "head (cymbals included).  Pass with no names to disable.")
    p.add_argument("--eval-start-clearance", type=float, default=0.02,
                   help="min SDF (m) required at q_start on the rolling eval "
                        "grid.  Default 2 cm; 0 restores the old min_sdf>0 "
                        "kissing starts.  Ignored when --eval-start-above-drums "
                        "is passed empty (historical hanging grid).")
    p.add_argument("--ckpt-every", type=int, default=2_000,
                   help="full checkpoint bundle cadence (last.pt + frontier)")
    p.add_argument("--ckpt-last-every", type=int, default=500,
                   help="last.pt-only cadence between full bundles (0=off)")
    p.add_argument("--ckpt-keep-every", type=int, default=500,
                   help="retained step_NNNNNN.pt cadence (0=off; not overwritten)")
    p.add_argument("--grad-term-every", type=int, default=250,
                   help="per-cost-term decoder grad-norm probe every N steps "
                        "(0=off)")
    p.add_argument("--torque-curvature-rows", type=int, default=8,
                   help="rows used for torque HVP/Taylor diagnostics on each "
                        "grad-term probe (0=off)")
    p.add_argument("--grad-balance", default="",
                   choices=["", "al_gradnorm", "al_pcgrad", "al_pcgrad_output"],
                   help="cost gradient combiner: al_gradnorm = AL-prioritized "
                        "GradNorm with torque cap; al_pcgrad = same + PCGrad on "
                        "torque/waist in parameter space; al_pcgrad_output = "
                        "same balancing at the trajectory before one model "
                        "backward (default off)")
    p.add_argument("--grad-balance-tau-torque-cap", type=float, default=0.30)
    p.add_argument("--grad-balance-torque-gamma", type=float, default=0.35,
                   help="RNEA dampening on torque priority (0-1]")
    p.add_argument("--grad-balance-beta", type=float, default=0.5,
                   help="violation excess exponent in AL priority")
    p.add_argument("--grad-balance-waist-floor", type=float, default=2.0,
                   help="minimum AL priority for the waist group (al_pcgrad)")
    p.add_argument("--grad-balance-refresh", type=int, default=1,
                   help="exact output-space GradNorm every N visits per "
                        "horizon bucket; intervening visits use EMA norms")
    p.add_argument("--grad-balance-norm-ema", type=float, default=0.9,
                   help="EMA decay for norms measured on exact refreshes")
    p.add_argument("--torque-grad-scale", type=float, default=1.0,
                   help="multiply GradNorm-combined torque group gradient "
                        "(output or param space); 0.1 ≈ tenth torque step")
    p.add_argument("--torque-grad-cap", type=float, default=0.0,
                   help="after total.backward(), cap ||g_torque||/||g|| to this "
                        "fraction (0=off).  Uses terms['torque'] incl. AL/PHR; "
                        "works with --line-search")
    p.add_argument("--torque-grad-cap-refresh", type=int, default=1,
                   help="exact torque-gradient cap VJP every N outer steps; "
                        "reuse its EMA scale between refreshes")
    p.add_argument("--torque-grad-cap-scale-ema", type=float, default=0.8,
                   help="EMA decay for torque cap scales measured on refresh")
    p.add_argument("--phase-timing-every", type=int, default=0,
                   help="CUDA-event outer-step phase profile cadence (0=off)")
    p.add_argument("--replay-viol-downweight", type=float, default=0.0,
                   help="replay sampling beta: weight *= exp(-beta * harvested "
                        "torque violation); does not scale loss (0=off)")
    p.add_argument("--replay-viol-uniform-mix", type=float, default=0.0,
                   help="gamma in p=(1-gamma)*normalized replay sampling "
                        "weights + gamma*uniform over each eligible subset; "
                        "range [0,1], default 0")
    p.add_argument("--replay-proficiency", action="store_true",
                   help="EMA hit+coll+torque ok on replay ring; stratified sampling")
    p.add_argument("--replay-frac-prof", type=float, default=0.40,
                   help="fraction of replay batch from proficient stratum")
    p.add_argument("--replay-frac-new", type=float, default=0.40,
                   help="fraction of replay batch from new (untrusted EMA) stratum")
    p.add_argument("--proficiency-viol-cap", type=float, default=8.0,
                   help="max torque_excess_max (N·m) for replay/frontier EMA ok")
    p.add_argument("--rho-terminal", type=float, default=None,
                   help="AL dual step rho for the terminal group (m^-1 per mean "
                        "region miss); default 10, or AMORTIZED_RHO_TERMINAL")
    # ---- SaLSa line search (config only; executor not wired yet) ----------
    # When enabled, Phase2Config.__post_init__ (+ train_phase2 entry) forces
    # grad_balance="" before make_cost_step (plan §1–2 / §14). Cosine LR
    # ownership is gated later in the loop (TODO plan §7). Commit-gate /
    # viol_acc policy for selection failure is plan §14 — not CLI-owned.
    p.add_argument("--line-search", action="store_true",
                   help="enable SaLSa-style LR line search; forces "
                        "--grad-balance off (baked before make_cost_step, "
                        "plan §14) and takes LR ownership from cosine once "
                        "the executor is wired (plan §7)")
    p.add_argument("--line-search-mode", default=None,
                   choices=["salsa_transfer", "fixed_epsilon"],
                   help="salsa_transfer = Armijo probe search (default); "
                        "fixed_epsilon = constant ε, no probe trials")
    p.add_argument("--line-search-c1", type=float, default=None,
                   help="Armijo c1 for smoothed probe acceptance (default 0.5)")
    p.add_argument("--line-search-beta", type=float, default=None,
                   help="beta_accept_per_update EMA retention (default 0.99)")
    p.add_argument("--line-search-max-trials", type=int, default=None,
                   help="max probe trials per search, excl. baseline (default 6)")
    p.add_argument("--line-search-interval", default=None,
                   choices=["adaptive", "fixed"],
                   help="search_interval_mode: adaptive K∈[1,10] or fixed "
                        "(default adaptive)")
    p.add_argument("--line-search-initial-interval", type=int, default=None,
                   help="initial/fixed successful updates between searches")
    p.add_argument("--line-search-disable-early-triggers", action="store_true",
                   help="do not let gradient/violation spikes search before "
                        "the scheduled interval")
    p.add_argument("--line-search-expansion", type=float, default=None,
                   help="expansion_factor on last selected epsilon (default 1.1)")
    p.add_argument("--line-search-warmup-updates", type=int, default=None,
                   help="search every update for N successful commits "
                        "(default 10)")
    p.add_argument("--line-search-epsilon-seed", type=float, default=None,
                   help="explicit epsilon seed; default uses --lr / resume "
                        "migration")
    p.add_argument("--line-search-epsilon-min", type=float, default=None,
                   help="absolute ε floor (default epsilon_min_mult × seed)")
    p.add_argument("--line-search-epsilon-max", type=float, default=None,
                   help="absolute ε ceiling (default epsilon_max_mult × seed)")
    p.add_argument("--line-search-fresh", action="store_true",
                   help="do not restore SaLSa selector state from checkpoint "
                        "(use cfg seed/bounds instead)")
    p.add_argument("--line-search-reseed-epsilon-on-objective-change",
                   action="store_true",
                   help="on forced objective jumps, reseed ε (100× prev) instead "
                        "of keeping last selected rate for SaLSa to adapt")
    p.add_argument("--line-search-force-al-reset", action="store_true",
                   help="on each actual al.update λ/pen change, reset SaLSa A/B, "
                        "K=1, and force search (plan §8; off by default)")
    p.add_argument("--line-search-moment-on", default=None,
                   choices=["gradient", "step"],
                   help="Adam EMA on clipped q (SaLSa) or executed step u "
                        "(default gradient)")
    p.add_argument("--line-search-actual-diagnostic-every", type=int, default=None,
                   help="evaluate the fully replayed actual executed candidate "
                        "every N successful-update slots (0 disables; default 0)")
    p.add_argument("--line-search-strict-actual-update", action="store_true",
                   help="require a finite non-increasing replayed objective "
                        "along the actual executed candidate; backtrack or skip")
    p.add_argument("--line-search-reject-current-probe-increase",
                   action="store_true",
                   help="reject a current probe increase even when its smoothed "
                        "acceptance EMA would pass")
    p.add_argument("--line-search-skip-on-probe-exhaustion",
                   action="store_true",
                   help="skip the update instead of floor-committing when no "
                        "probe candidate is accepted")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "lbfgs"],
                   help="line-search direction: AdamW preconditioner or L-BFGS")
    p.add_argument("--lbfgs-history", type=int, default=10,
                   help="L-BFGS history size (pairs) when --optimizer=lbfgs")
    p.add_argument("--roller-slots", type=int, default=256)
    p.add_argument("--roller-horizon", type=int, default=50)
    p.add_argument("--roller-start", type=int, default=200)
    p.add_argument("--roller-ticks-per-step", type=int, default=1)
    p.add_argument("--out", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--pool-cache", default=None,
                   help="npz of a previously generated pool (reused if present "
                        "and the content fingerprint matches live kit/mix/"
                        "seed/lead settings)")
    p.add_argument("--gen-pool-only", action="store_true",
                   help="generate the scenario pool into --pool-cache and exit "
                        "without training.  CPU-only by construction, so it can "
                        "run CONCURRENTLY with phase-1 training and the "
                        "follow-up phase-2 launch finds the cache ready "
                        "(see scripts/_run_19m_waist.sh)")
    # frontier collocation -- later generalization / continuation.  Every
    # --frontier-* knob is a no-op unless --frontier is set.
    p.add_argument("--frontier", action="store_true",
                   help="later generalization / continuation stage on an "
                        "already-trained phase-2 checkpoint (--resume last.pt).  "
                        "The collocation cloud is seeded from the existing pool "
                        "npz (all already-feasible idle/far rows), not an empty "
                        "start.  Off by default; every --frontier-* knob is a "
                        "no-op unless this flag is set")
    p.add_argument("--frontier-proposals", type=int, default=64,
                   help="proposals per frontier expand (no-op unless --frontier)")
    p.add_argument("--frontier-chart-cap", type=int, default=2048,
                   help="child-ring depth per chart; total children = this * 64 "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-fresh-frac", type=float, default=0.5,
                   help="idle/far fraction whose q0 is stolen from the cloud "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-spawn-frac", type=float, default=0.25,
                   help="spawn-path fraction patched from same-chart parents "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-sigma-q", type=float, default=0.05,
                   help="q0 ball radius (rad) before per-joint weights: "
                        "waist 0.25, shoulders 0.5, wrists 1.0 "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-sigma-qd", type=float, default=0.10,
                   help="qd0 ball radius (rad/s), same per-joint weights "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-sigma-tth", type=float, default=0.05,
                   help="TTH jitter as a fraction of remaining time-to-hit "
                        "(0.05 = 5 percent), not seconds "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-ema-alpha", type=float, default=0.25,
                   help="EMA recency on hit/collision scores; 0.25 means two "
                        "hits from the 0.5 prior clear 0.70 (no-op unless "
                        "--frontier)")
    p.add_argument("--frontier-ema-nmin", type=int, default=2,
                   help="min EMA observations before a parent leaves the 0.5 "
                        "prior in spawn sampling (no-op unless --frontier)")
    p.add_argument("--frontier-ema-prof", type=float, default=0.70,
                   help="EMA value treated as proficient for landmarks "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-landmark-k", type=int, default=32,
                   help="farthest-point landmarks per chart (no-op unless "
                        "--frontier)")
    p.add_argument("--frontier-landmark-tau", type=float, default=2.0,
                   help="min proposal-sigma distance to add a landmark or keep "
                        "a proposed child (no-op unless --frontier)")
    p.add_argument("--frontier-propose-candidates", type=int, default=256,
                   help="parents scored against landmarks per propose "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-hull-ema", type=float, default=0.55,
                   help="min EMA for hull parent sampling; 0 disables "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-train-inv-frac", type=float, default=0.5,
                   help="train-batch fraction inverse-density vs uniform "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-train-inv-pool", type=int, default=2048,
                   help="uniform candidate pool for inverse-density half "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-sample-k", type=int, default=512,
                   help="cloud collocations per train step (the minibatch "
                        "when this is >= batch size). Mix of uniform and "
                        "inverse-density over the expanding start set "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-traj-stride", type=int, default=4,
                   help="along each already-feasible pool trajectory, also "
                        "seed collocations at knot k, k+stride, ... (knot 0 "
                        "always kept).  Remaining TTH = hit-k; the suffix of "
                        "the traj is a valid shorter problem.  Used at "
                        "continuation start with the existing pool (no-op "
                        "unless --frontier)")
    p.add_argument("--frontier-include-miss", action="store_true",
                   help="admit miss rows into the cloud (no-op unless "
                        "--frontier)")
    p.add_argument("--frontier-idle-far-cloud", action="store_true",
                   help="idle/far fresh rows from the cloud only (seed + "
                        "ball children); post-hit stays the live roller. "
                        "Disables whole-batch cloud takeover, q0-steal, "
                        "and post-hit harvest into the cloud. Replay still "
                        "splices --frontier-splice-k cloud rows "
                        "(no-op unless --frontier)")
    p.add_argument("--frontier-splice-k", type=int, default=64,
                   help="cloud rows spliced into every replay/fresh batch "
                        "under --frontier-idle-far-cloud so ball children "
                        "get EMA looks before wrap. 0 disables (no-op "
                        "unless --frontier-idle-far-cloud)")
    p.add_argument("--frontier-auto-miss-mm", type=float, default=0.0,
                   help="auto-enable frontier when probe axial_abs_p50_mm "
                        "drops below this (0=off).  Also honours "
                        "<run_dir>/.frontier_activate from the monitor.")
    p.add_argument("--no-frontier-auto-full-weights", action="store_true",
                   help="do not snap suffix warmup when auto-frontier fires")
    p.add_argument("--frontier-auto-viol-cap", type=float, default=8.0,
                   help="absolute train v_* cap for auto-frontier spike guard")
    p.add_argument("--frontier-auto-viol-ratio", type=float, default=3.0,
                   help="spike ratio vs prior-train median for auto-frontier")
    p.add_argument("--frontier-auto-viol-window", type=int, default=10,
                   help="train rows in the auto-frontier violation window")
    return p.parse_args(argv)


def _validate_voxel_cli(args) -> None:
    """Reject legacy context/head options that violate the voxel contract."""
    if not args.voxel_pool:
        return
    if int(args.num_future) != 1:
        raise SystemExit("--voxel-pool requires --num-future 1 (masked invalid)")
    forbidden_flags = {
        "--waist-sp-head": args.waist_sp_head,
        "--waist-pr-sp-head": args.waist_pr_sp_head,
        "--waist-pr-ema": args.waist_pr_ema,
        "--line-search": args.line_search,
        "--frontier": args.frontier,
    }
    forbidden_values = {
        "--w-waist-aim": args.w_waist_aim,
        "--w-waist-sp-tether": args.w_waist_sp_tether,
        "--w-waist-pr-sp-tether": args.w_waist_pr_sp_tether,
        "--w-waist-pr-sp-hold": args.w_waist_pr_sp_hold,
        "--w-waist-pr-sp-hold-band": args.w_waist_pr_sp_hold_band,
        "--w-waist-pr-sp-distill": args.w_waist_pr_sp_distill,
        "--w-waist-sp-hold": args.w_waist_sp_hold,
        "--w-waist-sp-smooth": args.w_waist_sp_smooth,
        "--waist-sp-search-frac": args.waist_sp_search_frac,
        "--w-waist-sp-search": args.w_waist_sp_search,
    }
    enabled = [name for name, value in forbidden_flags.items() if bool(value)]
    enabled += [
        name for name, value in forbidden_values.items()
        if value is not None and float(value) != 0.0
    ]
    if enabled:
        raise SystemExit(
            "--voxel-pool uses current-goal-only context and decoder-owned "
            "torso motion; incompatible options: " + ", ".join(enabled)
        )


def _configure_voxel_model(model_cfg, *, enabled: bool) -> bool:
    """Strip encoder SP modules before constructing a voxel-mode model.

    Returns whether the architecture changed, so checkpoint loading takes the
    reported non-strict path and treats old head tensors as unused.
    """
    if not enabled:
        return False
    changed = bool(
        getattr(model_cfg, "waist_sp_head", False)
        or getattr(model_cfg, "waist_pr_sp_head", False)
    )
    model_cfg.waist_sp_head = False
    model_cfg.waist_pr_sp_head = False
    return changed


def _pool_schema() -> str:
    """The cache's schema marker: the CONTEXT_FIELDS names, in order.

    A cache written before a context field existed (e.g. ``aim_yaw`` /
    ``aim_valid`` / ``stick_valid``) reloads without error and silently zeros
    the new tokens on every row -- under ``--waist-sp-head`` that turns the
    anchor into "pin yaw to 0" while the ``sp_res`` log reads a healthy 0.0.
    Refusing a marker mismatch is what turns that silent failure into a loud
    one.
    """
    from amortized_opt.warmstart_data import CONTEXT_FIELDS

    return json.dumps(list(CONTEXT_FIELDS))


def _schedule_schema() -> str:
    from amortized_opt.rhythm_chain_data import SCHEDULE_SCHEMA

    return SCHEDULE_SCHEMA


#: Bump when the fingerprint *key set* or semantics change.  Value mismatches
#: (tilt, seed, …) fail without a bump; this is for adding/renaming keys.
POOL_FINGERPRINT_VERSION = 1


def _pool_fingerprint(args, scen) -> str:
    """Content identity for a phase-2 pool npz.

    The column ``_schema`` only names CONTEXT_FIELDS.  Kit DR, rhythm mix,
    seed, lead, etc. are baked into the arrays at generation time and used to
    be keyed only by the bash ``POOL_TAG`` filename -- reuse of the same path
    with different flags silently trained on the wrong distribution.  This
    marker is checked on load and must match the live CLI/config.
    """
    kr = scen.kit_randomization
    waist = scen.waist_cfg
    payload = {
        "v": POOL_FINGERPRINT_VERSION,
        "n": int(args.scenarios),
        "seed": int(args.seed),
        "horizon": int(scen.horizon),
        "buckets": [int(b) for b in scen.buckets],
        "num_future": int(scen.num_future),
        "legacy": bool(getattr(args, "legacy_pool", False)),
        "rhythm_idle_frac": float(scen.rhythm_idle_frac),
        "post_hit_frac": float(scen.post_hit_frac),
        "far_fetched_frac": float(scen.far_fetched_frac),
        "eval_like_frac": float(scen.eval_like_frac),
        "cold_start_frac": float(scen.cold_start_frac),
        "pair_frac": float(scen.pair_frac),
        "rest_frac": float(scen.rest_frac),
        "lead_s": float(scen.lead_s),
        "synth_transition_frac": float(scen.synth_transition_frac),
        "tip_speed_max": float(scen.tip_speed_max),
        "hit_time_jitter_s": float(scen.hit_time_jitter_s),
        "hit_loudness_jitter": float(scen.hit_loudness_jitter),
        "rare_drum_boost": float(scen.rare_drum_boost),
        "kit_per_drum_xy": float(kr.per_drum_xy),
        "kit_global_xy": float(kr.global_xy),
        "kit_global_z": float(kr.global_z),
        "kit_per_drum_tilt": float(kr.per_drum_tilt),
        "kit_per_drum_radius": float(kr.per_drum_radius),
        "waist_aim_future_tau": float(
            0.0 if waist is None else waist.future_tau
        ),
        "idle_cylinder": (
            None if scen.idle_cylinder_path is None else str(scen.idle_cylinder_path)
        ),
        "idle_settle_wp": int(scen.idle_settle_wp),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_npz_buckets(
    cache: Path, *, fingerprint: str | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    z = np.load(cache, allow_pickle=True)
    want = _pool_schema()
    got = str(z["_schema"]) if "_schema" in z.files else None
    if got != want:
        raise RuntimeError(
            f"pool cache {cache} "
            + ("has no schema marker" if got is None else
               f"was written under a different context schema\n"
               f"  cache: {got}\n  code:  {want}")
            + " -- it predates the current CONTEXT_FIELDS and would reload "
              "with the missing tokens silently zeroed on every row. "
              "Delete the cache (or point --pool-cache elsewhere) to "
              "regenerate it."
        )
    # Rhythm-derived pools carry schedule/provenance; refuse a marker mismatch
    # so an old cache cannot silently drop those columns.
    if "_schedule_schema" in z.files:
        want_s = _schedule_schema()
        got_s = str(z["_schedule_schema"])
        if got_s != want_s:
            raise RuntimeError(
                f"pool cache {cache} schedule schema mismatch\n"
                f"  cache: {got_s}\n  code:  {want_s}\n"
                "Delete the cache (or point --pool-cache elsewhere) to "
                "regenerate it."
            )
    elif any(
        k.endswith("/sched_valid") or k.endswith("/clip_i") for k in z.files
    ):
        raise RuntimeError(
            f"pool cache {cache} has schedule/provenance columns but no "
            f"_schedule_schema marker -- regenerate it."
        )
    if fingerprint is not None:
        got_fp = str(z["_fingerprint"]) if "_fingerprint" in z.files else None
        if got_fp != fingerprint:
            raise RuntimeError(
                f"pool cache {cache} "
                + ("has no content fingerprint" if got_fp is None else
                   "was generated under different pool settings\n"
                   f"  cache: {got_fp}\n  code:  {fingerprint}")
                + " -- kit DR / mix / seed / lead / etc. are baked into the "
                  "arrays.  Delete the cache (or point --pool-cache elsewhere) "
                  "to regenerate it."
            )
    buckets: dict[int, dict[str, np.ndarray]] = {}
    for k in z.files:
        if k.startswith("_"):
            continue
        b, name = k.split("/", 1)
        buckets.setdefault(int(b), {})[name] = z[k]
    stats = json.loads(str(z["_stats"])) if "_stats" in z.files else {}
    return buckets, stats


def _save_npz_buckets(
    cache: Path, pool, *, schedule: bool, fingerprint: str | None = None,
) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for b, d in pool.data.items():
        for k, v in d.items():
            flat[f"{b}/{k}"] = v.detach().cpu().numpy()
    stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in pool.stats.items()}
    kw = dict(
        _stats=np.array(json.dumps(stats)),
        _schema=np.array(_pool_schema()),
        **flat,
    )
    if fingerprint is not None:
        kw["_fingerprint"] = np.array(fingerprint)
    if schedule:
        kw["_schedule_schema"] = np.array(_schedule_schema())
    np.savez(cache, **kw)


def _make_rhythm_pool(args, scen, noise, device):
    """Generate Phase A + far-fetched (+ optional post-hit cache) and wrap."""
    from amortized_opt.phase2_data import Phase2Pool, _merge
    from amortized_opt.rhythm import default_rcc_dir, load_rcc_pool
    from amortized_opt.rhythm_chain_data import (
        generate_far_fetched_scenarios,
        generate_rhythm_idle_scenarios,
    )

    n_idle = max(0, int(round(float(args.scenarios) * float(args.rhythm_idle_frac))))
    n_far = max(0, int(round(float(args.scenarios) * float(args.far_fetched_frac))))
    if n_idle + n_far <= 0:
        raise SystemExit(
            "rhythm pool needs --rhythm-idle-frac and/or --far-fetched-frac > 0 "
            "(or pass --legacy-pool)"
        )
    kit = A.DrumKit.from_layout()
    clips = load_rcc_pool(default_rcc_dir("train"), kit)
    print(f"[data] rhythm pool: idle={n_idle}, far={n_far} "
          f"(RCC clips={len(clips)})", flush=True)
    t0 = time.perf_counter()
    parts = []
    if n_idle > 0:
        print(f"[data] generating {n_idle} rhythm_idle scenarios...", flush=True)
        parts.append(generate_rhythm_idle_scenarios(
            n_idle, clips, seed=int(args.seed), cfg=scen, kit=kit,
        ))
    if n_far > 0:
        print(f"[data] generating {n_far} far_fetched scenarios...", flush=True)
        parts.append(generate_far_fetched_scenarios(
            n_far, clips, seed=int(args.seed) + 17, cfg=scen, kit=kit,
        ))
    ph = Path(args.post_hit_cache) if args.post_hit_cache else None
    if ph is not None and ph.exists():
        print(f"[data] merging post-hit cache {ph}", flush=True)
        bkt, st = _load_npz_buckets(ph)
        parts.append({"buckets": bkt, "stats": st})
    elif args.post_hit_cache:
        print(f"[data] --post-hit-cache {args.post_hit_cache} missing; "
              f"N_B=0 until distill", flush=True)
    raw = _merge(parts) if len(parts) > 1 else parts[0]
    pool = Phase2Pool(
        raw["buckets"], cfg=scen, noise=noise, device=device, stats=raw["stats"],
    )
    print(f"[data] {time.perf_counter() - t0:.1f} s -> {len(pool)} samples "
          f"{pool.distribution()}; kinds={pool.kind_counts()}", flush=True)
    return pool, True


def load_or_make_pool(args, scen, noise, device):
    cache = Path(args.pool_cache) if args.pool_cache else None
    use_rhythm = (not args.legacy_pool) and (
        float(args.rhythm_idle_frac) > 0.0 or float(args.far_fetched_frac) > 0.0
    )
    fingerprint = _pool_fingerprint(args, scen)
    if cache is not None and cache.exists():
        from amortized_opt.phase2_data import Phase2Pool

        buckets, stats = _load_npz_buckets(cache, fingerprint=fingerprint)
        print(f"[data] reusing {cache}", flush=True)
        pool = Phase2Pool(buckets, cfg=scen, noise=noise, device=device, stats=stats)
        # Optional post-hit merge on top of a cached A+F pool.
        ph = Path(args.post_hit_cache) if args.post_hit_cache else None
        if ph is not None and ph.exists():
            from amortized_opt.phase2_data import _merge

            # Post-hit overlays are schema-checked only (their generator does
            # not share the main pool's mix/seed fingerprint).
            bkt_b, st_b = _load_npz_buckets(ph)
            # Rebuild host arrays from the live pool, merge, re-wrap.
            host = {
                b: {k: v.detach().cpu().numpy() for k, v in d.items()}
                for b, d in pool.data.items()
            }
            raw = _merge([
                {"buckets": host, "stats": dict(pool.stats)},
                {"buckets": bkt_b, "stats": st_b},
            ])
            pool = Phase2Pool(
                raw["buckets"], cfg=scen, noise=noise, device=device,
                stats=raw["stats"],
            )
            print(f"[data] merged post-hit -> {len(pool)} samples "
                  f"kinds={pool.kind_counts()}", flush=True)
        return pool

    if use_rhythm:
        pool, schedule = _make_rhythm_pool(args, scen, noise, device)
    else:
        t0 = time.perf_counter()
        print(f"[data] generating {args.scenarios} legacy scenarios "
              f"({args.workers} workers)...", flush=True)
        pool = generate_cost_pool(
            args.scenarios, seed=args.seed, cfg=scen, noise=noise, split="train",
            workers=args.workers, device=device,
        )
        schedule = False
        print(f"[data] {time.perf_counter() - t0:.1f} s -> {len(pool)} samples "
              f"{pool.distribution()}", flush=True)
        s = pool.stats
        pr = np.asarray(s.get("peak_rate_ratio", []), float)
        print(f"[data] accepted {s['accepted']}/{s['requested']} "
              f"(|qd| reject {s['reject_velocity']}, idle {s['reject_idle']}, "
              f"box-start {s.get('reject_box_start', 0)}); "
              f"eval-like {s['eval_like']}, "
              f"cold {s.get('cold', 0)}/{s.get('cold_requested', 0)}, "
              f"dual-arm {s['dual_active']} "
              f"(asym {s['asymmetric']}), single-idle {s.get('single_idle', 0)}, "
              f"rest {s.get('rest', 0)}; peak|qd|/limit p50={np.percentile(pr, 50):.2f} "
              f"p99={np.percentile(pr, 99):.2f}", flush=True)
    if cache is not None:
        _save_npz_buckets(
            cache, pool, schedule=schedule or use_rhythm, fingerprint=fingerprint,
        )
    return pool


def main(argv=None) -> int:
    args = parse_args(argv)
    _validate_voxel_cli(args)
    if args.gen_pool_only:
        # Pool generation is CPU-bound and independent of the phase-1
        # checkpoint (the on-policy half of the data is the ROLLER replay,
        # built at train time), so it can run concurrently with phase-1
        # training on a box with spare cores.  Force CPU so this process
        # never touches the GPU phase 1 is on; the cache is saved from host
        # arrays either way.
        if not args.pool_cache:
            raise SystemExit("--gen-pool-only requires --pool-cache")
        args.device = "cpu"
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index or 0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if not args.no_compile:
        configure_inductor()

    root = Path(__file__).resolve().parent.parent
    name = args.name or f"phase2_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out) if args.out else root / "runs" / name
    if not args.gen_pool_only:   # a gen-only invocation owns no run dir
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] {run_dir}", flush=True)

    # The idle-cylinder library is resolved ONCE and threaded to all three
    # consumers (the pool's seeds, the roller, the cost) so they cannot disagree.
    no_idle = bool(args.no_idle_cylinder or args.no_idle_home)
    idle_cylinder_path = None if no_idle else (
        args.idle_cylinder or args.idle_home or "default"
    )
    w_idle_cyl = args.w_idle_cylinder
    if w_idle_cyl is None and args.w_idle_home is not None:
        w_idle_cyl = args.w_idle_home
    # ``per_drum_tilt`` and ``per_drum_radius`` are the KitRandomization knobs
    # whose features the encoder has never had a reason to read: ``drum_features``
    # carries the head normal (3), the scalar tilt angle (1) and the radius (1),
    # 5 of DRUM_FEAT_DIM = 9, and every run before 2026-08-13 / -08-14 held them
    # constant.  Turning either on is therefore a change to the INPUT
    # distribution, not just the geometry -- and a radius that never varies is
    # perfectly confounded with drum identity, which leaves the weights reading
    # it unidentified exactly as ``yaw_now`` was.
    kit_rand = KitRandomization(
        per_drum_tilt=math.radians(float(args.drum_tilt_deg)),
        per_drum_radius=float(args.drum_radius_frac),
    )
    # ONE waist-aim law for the whole run: the pool's seeds and ``aim_yaw``
    # tokens, the roller's refresh and its waist reference all read
    # ``ScenarioConfig.waist_cfg``, so setting it here is what keeps the seed's
    # ramp and the head's anchor on the same target.
    waist_cfg = A.WaistAimConfig(
        future_tau=float(args.waist_aim_future_tau),
        future_max=int(args.num_future),
    )
    use_rhythm = (not args.legacy_pool) and (
        float(args.rhythm_idle_frac) > 0.0 or float(args.far_fetched_frac) > 0.0
    )
    scen_kw = dict(
        waist_cfg=waist_cfg,
        horizon=int(args.horizon), buckets=tuple(args.buckets),
        num_future=int(args.num_future),
        eval_like_frac=float(args.eval_like_frac),
        cold_start_frac=float(args.cold_start_frac),
        pair_frac=float(args.pair_frac),
        idle_cylinder_path=idle_cylinder_path,
        idle_settle_wp=int(args.idle_settle_wp),
        rest_frac=float(args.rest_frac),
        waist_upright_ease_wp=args.waist_upright_ease_wp,
        kit_randomization=kit_rand,
        hit_time_jitter_s=float(args.hit_time_jitter_s),
        hit_loudness_jitter=float(args.hit_loudness_jitter),
        rare_drum_boost=float(args.rare_drum_boost),
        lead_s=float(args.lead_s),
        tip_speed_max=float(args.tip_speed_max),
        synth_transition_frac=float(args.synth_transition_frac),
        rhythm_idle_frac=float(args.rhythm_idle_frac),
        post_hit_frac=float(args.post_hit_frac),
        far_fetched_frac=float(args.far_fetched_frac),
    )
    if use_rhythm:
        scen_kw.update(
            eval_like_frac=0.0,
            cold_start_frac=0.0,
            pair_frac=0.0,
        )
    scen = Phase2ScenarioConfig(**scen_kw)
    print(f"[data] kit randomization: {kit_rand} "
          f"(tilt {args.drum_tilt_deg:.2f} deg, "
          f"radius +/-{100.0 * args.drum_radius_frac:.1f} %)", flush=True)
    print(f"[data] waist aim: future_tau={waist_cfg.future_tau} s "
          f"(0 = current goals only), future_max={waist_cfg.future_max}",
          flush=True)
    print(f"[data] fresh mix fracs: idle={args.rhythm_idle_frac:.2f} "
          f"post_hit={args.post_hit_frac:.2f} far={args.far_fetched_frac:.2f} "
          f"eval_like={args.eval_like_frac:.2f} "
          f"({'rhythm' if use_rhythm else 'legacy'} pool)", flush=True)
    noise = Phase2Noise()
    pool = load_or_make_pool(args, scen, noise, dev)
    if args.gen_pool_only:
        print(f"[data] pool cached at {args.pool_cache}; exiting (--gen-pool-only)",
              flush=True)
        return 0

    # ---- model -------------------------------------------------------------
    # ``model_config_from_checkpoint`` reconstructs the architecture the weights
    # ACTUALLY have, not the one today's defaults would build -- so this is the
    # checkpoint as trained.  The deliberate change (switching the idle token on
    # over a warm start that predates it) is applied below, explicitly, and the
    # tensors are reconciled with it.
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_cfg = A.model_config_from_checkpoint(ck)
    elif args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model_cfg = A.model_config_from_checkpoint(ck)
    else:
        ck = None
        model_cfg = A.ModelConfig(max_horizon=int(args.horizon),
                                  num_drums=scen.num_drums, num_future=scen.num_future)
    # apply CLI overrides
    overridden = False
    for attr in ("d_model", "dec_layers", "d_ctx", "ctx_layers", "dec_ffn_mult"):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(model_cfg, attr, val)
            overridden = True
    if getattr(args, "no_aux_heads", False):
        model_cfg.predict_aux = False
        overridden = True
    stripped_voxel_heads = _configure_voxel_model(
        model_cfg, enabled=bool(args.voxel_pool),
    )
    if stripped_voxel_heads:
        overridden = True
        print(
            "[model] voxel mode: dropped legacy waist yaw/P-R set-point "
            "modules before model construction",
            flush=True,
        )
        if args.resume and not args.reset_optimizer:
            args.reset_optimizer = True
            print(
                "[model] voxel mode: resetting optimizer because the resumed "
                "checkpoint contained removed SP parameters",
                flush=True,
            )
    # The idle token is a context-encoder shape, so it has to be decided BEFORE
    # the model is built -- and it is meaningless without a home table.
    want_token = (not args.no_idle_token) and idle_cylinder_path is not None
    if want_token != getattr(model_cfg, "idle_home_token", False):
        model_cfg.idle_home_token = want_token
        overridden = True
    if args.freeze_waist_pr:
        # Pins decoder P/R columns to the ready-pose ramp.  If the warm start
        # still carries ``waist_pr_sp_head``, drop it: that token would keep
        # conditioning the arms after the joints are gone from the action
        # space.  Unexpected head keys are forgiven below (``overridden``).
        model_cfg.frozen_joints = ("waist_pitch_joint", "waist_roll_joint")
        print(f"[model] frozen joints: {model_cfg.frozen_joints}", flush=True)
    drop_yaw_head = (
        not args.waist_sp_head
        and getattr(model_cfg, "waist_sp_head", False)
        and (args.init and not args.resume)
    )
    if drop_yaw_head:
        model_cfg.waist_sp_head = False
        overridden = True
        print("[model] waist_sp_head OFF (init without --waist-sp-head; "
              "dropping encoder head keys)", flush=True)
    drop_pr_head = (
        not args.waist_pr_sp_head
        and getattr(model_cfg, "waist_pr_sp_head", False)
        and (args.freeze_waist_pr or args.waist_pr_ema
             or (args.init and not args.resume))
    )
    if drop_pr_head:
        model_cfg.waist_pr_sp_head = False
        overridden = True
        why = (
            "freeze owns P/R" if args.freeze_waist_pr else
            "EMA owns closed-loop P/R" if args.waist_pr_ema else
            "init without --waist-pr-sp-head"
        )
        print(f"[model] waist_pr_sp_head OFF ({why}; dropping encoder head keys)",
              flush=True)
    if args.dec_rope and not getattr(model_cfg, "dec_rope", False):
        # Behaviour-only: RoPE is a rotate of Q/K, no parameter shape moves,
        # so the warm start still loads with strict=True.
        model_cfg.dec_rope = True
        print("[model] decoder self-attention RoPE ON", flush=True)
    if args.dec_plant_feats and not getattr(model_cfg, "dec_plant_feats", False):
        # New zero-init Linears (wp_plant_proj / state_plant_proj): missing
        # keys on a pre-flag warm start, no existing tensor changes shape, so
        # the slice-copy path below is what picks them up as random/zero init.
        model_cfg.dec_plant_feats = True
        overridden = True
        print("[model] decoder plant features ON "
              "(Cartesian + knot-RNEA of init_traj, zero-init no-op)", flush=True)
    if args.traj_head_fp32 and not getattr(model_cfg, "traj_head_fp32", False):
        # Behaviour-only: no tensor shape or state-dict key changes.
        model_cfg.traj_head_fp32 = True
        print("[model] trajectory head/unprojection fp32 ON "
              "(transformer and auxiliary heads keep AMP)", flush=True)
    if args.waist_sp_head and not getattr(model_cfg, "waist_sp_head", False):
        # Switching the head ON over a warm start that predates it: the new
        # parameters are pure MISSING keys (no shape moves), and the zero-init
        # of ``waist_sp_out`` survives the partial load -- the head starts as
        # exactly the analytic aim law.  A warm start already trained WITH the
        # head enables this through ``model_config_from_checkpoint`` instead.
        model_cfg.waist_sp_head = True
        overridden = True
        print("[model] waist_sp_head ON (fresh head, starts at aim_yaw)",
              flush=True)
    if args.waist_pr_sp_head and not getattr(model_cfg, "waist_pr_sp_head", False):
        model_cfg.waist_pr_sp_head = True
        overridden = True
        print("[model] waist_pr_sp_head ON (fresh head, starts at upright)",
              flush=True)
    # Rhythm-derived pools lock ``num_future=1``; phase-1 warm starts are still
    # F=3.  Shrinking is a real architecture change (``future_rank_emb``), so
    # mark ``overridden`` and let the slice-copy load below keep row 0 of the
    # rank embedding. Growing F is refused -- the pool would need regenerating.
    if int(args.num_future) != int(model_cfg.num_future):
        if int(args.num_future) > int(model_cfg.num_future):
            raise SystemExit(
                f"--num-future {args.num_future} > checkpoint "
                f"num_future={model_cfg.num_future}: cannot invent future-rank "
                f"embeddings; retrain phase 1 or regenerate the pool to match"
            )
        print(f"[model] num_future {model_cfg.num_future} -> {args.num_future} "
              f"(warm-start future_rank_emb row 0 kept; higher ranks dropped)",
              flush=True)
        model_cfg.num_future = int(args.num_future)
        overridden = True
    if args.clamp_traj_to_box:
        # Behaviour-only and parameter-free, so it can be switched on over any
        # warm start without touching a single tensor.
        model_cfg.clamp_traj_to_box = True
        print("[model] clamp_traj_to_box ON (straight-through, URDF box)",
              flush=True)
    if args.line_search and args.grad_balance:
        # Prefer warn + force-empty over rejecting the launcher (plan §1–2).
        warnings.warn(
            f"--line-search forces --grad-balance '' "
            f"(was {args.grad_balance!r}); surgery off before make_cost_step "
            f"(parameter_line_search_plan §14)",
            UserWarning,
            stacklevel=1,
        )
        print(f"[cfg] line_search=1 forces grad_balance='' "
              f"(launcher had {args.grad_balance!r})", flush=True)
        args.grad_balance = ""
    if args.grad_balance:
        print(f"[cfg] grad_balance={args.grad_balance} "
              f"tau_torque_cap={args.grad_balance_tau_torque_cap:g} "
              f"torque_gamma={args.grad_balance_torque_gamma:g} "
              f"beta={args.grad_balance_beta:g} "
              f"refresh={args.grad_balance_refresh} "
              f"norm_ema={args.grad_balance_norm_ema:g} "
              f"torque_grad_scale={args.torque_grad_scale:g}"
              + (f" waist_floor={args.grad_balance_waist_floor:g}"
                 if args.grad_balance in ("al_pcgrad", "al_pcgrad_output") else ""),
              flush=True)
    if args.line_search:
        print("[cfg] line_search=1 (cosine LR ownership gated once executor "
              "is wired — plan §7; commit-gate/viol_acc: plan §14)",
              flush=True)
    if float(args.torque_grad_cap) > 0.0:
        print(f"[cfg] torque_grad_cap={float(args.torque_grad_cap):g} "
              f"(LS-compatible; no PCGrad)", flush=True)
    if args.rho_terminal is not None:
        print(f"[cfg] rho_terminal={float(args.rho_terminal):g}", flush=True)
    # A checkpoint written before the idle token existed has neither
    # ``encoder.idle_proj`` nor the 4th ``type_emb`` row, so a strict load fails
    # on a shape mismatch.  Detect it and fall back to a REPORTED non-strict
    # load rather than silently disabling the token or silently random-initing.
    if ck is not None and model_cfg.idle_home_token:
        te = ck["model"].get("encoder.type_emb.weight")
        if te is not None and te.shape[0] != model_cfg.n_token_types:
            print(f"[init] checkpoint predates the idle-home token "
                  f"(type_emb {tuple(te.shape)}); its 2 new context params start "
                  f"random -- use --no-idle-token to load it strictly", flush=True)
            overridden = True

    model = A.build_model(model_cfg, dev)
    if ck is not None:
        if overridden:
            # ``strict=False`` forgives MISSING and UNEXPECTED keys but NOT a
            # shape mismatch on a key that exists in both -- it still raises.
            # ``encoder.type_emb.weight`` is exactly that case when the idle
            # token is switched on over a checkpoint that predates it: (3, 256)
            # against (4, 256).  Copy the overlapping block (the DRUM / GOAL /
            # FUTURE rows, which ARE trained) and leave the rest at its random
            # init, rather than dropping the whole tensor and throwing three
            # trained embeddings away with the one new row.
            sd = dict(ck["model"])
            own = model.state_dict()
            for k, v in list(sd.items()):
                w = own.get(k)
                if w is None or w.shape == v.shape:
                    continue
                if w.dim() != v.dim():
                    print(f"[init] {k}: rank {tuple(v.shape)} != {tuple(w.shape)}, "
                          f"dropped (random init)", flush=True)
                    sd.pop(k)
                    continue
                sl = tuple(slice(0, min(a, b)) for a, b in zip(w.shape, v.shape))
                new = w.clone()
                new[sl] = v[sl].to(new.dtype)
                sd[k] = new
                print(f"[init] {k}: {tuple(v.shape)} -> {tuple(w.shape)}, copied "
                      f"{tuple(s.stop for s in sl)} (remainder random)", flush=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[init] {len(missing)} missing keys (random init): "
                      f"{missing[:3]}...", flush=True)
            if unexpected:
                print(f"[init] {len(unexpected)} unused ckpt keys: "
                      f"{unexpected[:3]}...", flush=True)
        else:
            model.load_state_dict(ck["model"])
        print(f"[init] {args.resume or args.init} @ step {ck.get('step')}", flush=True)
    print(f"[model] {A.param_counts(model)}", flush=True)

    cfg = Phase2Config(
        max_steps=int(args.steps), batch_size=int(args.batch), lr=float(args.lr),
        warmup_steps=int(args.warmup_steps),
        min_lr_frac=float(args.min_lr_frac),
        amp_dtype="float32" if args.no_amp else "bfloat16",
        compile_mode=None if args.no_compile else args.compile_mode,
        compile_cost=bool(args.compile_cost), compile_aux=bool(args.compile_cost),
        voxel_pool=bool(args.voxel_pool),
        voxel_q_cell_rad=float(args.voxel_q_cell_rad),
        voxel_bootstrap_rows=int(args.voxel_bootstrap_rows),
        voxel_bootstrap_batch=int(args.voxel_bootstrap_batch),
        voxel_bootstrap_stride=int(args.voxel_bootstrap_stride),
        voxel_posthit_anchor_tth_s=float(args.voxel_posthit_anchor_tth_s),
        voxel_rest_gap_s=float(args.voxel_rest_gap_s),
        voxel_cache=args.voxel_cache,
        voxel_probe_capacity=int(args.voxel_probe_capacity),
        voxel_probe_refresh_every=int(args.voxel_probe_refresh_every),
        voxel_probe_shorter_layers=int(args.voxel_probe_shorter_layers),
        voxel_probe_longer_layers=int(args.voxel_probe_longer_layers),
        voxel_probe_timing_step=int(args.voxel_probe_timing_step),
        voxel_probe_timing_frac=float(args.voxel_probe_timing_frac),
        voxel_strike_speed_min=float(args.voxel_strike_speed_min),
        voxel_strike_speed_max=float(args.voxel_strike_speed_max),
        voxel_frac_mpc=float(args.voxel_frac_mpc),
        voxel_frac_pool=float(args.voxel_frac_pool),
        voxel_frac_boundary=float(args.voxel_frac_boundary),
        voxel_frac_probe=float(args.voxel_frac_probe),
        voxel_probe_penalty=float(args.voxel_probe_penalty),
        timing_critic=ViolationCriticConfig(
            mode=str(args.timing_critic_mode),
            num_heads=int(args.timing_critic_heads),
            hidden_dim=int(args.timing_critic_hidden_dim),
            lr=float(args.timing_critic_lr),
            ema_decay=float(args.timing_critic_ema_decay),
            batch_size=int(args.timing_critic_batch),
            updates_per_step=int(args.timing_critic_updates),
            bootstrap_probability=float(args.timing_critic_bootstrap_prob),
            randomized_prior_scale=float(args.timing_critic_prior_scale),
            dropout_probability=float(args.timing_critic_dropout),
            mc_dropout_samples=int(args.timing_critic_mc_samples),
            head_architecture_mix=str(args.timing_critic_head_architecture),
            replay_timescale_diversity=bool(
                args.timing_critic_replay_timescales
            ),
            recent_capacity=int(args.timing_critic_recent_capacity),
            audit_capacity=int(args.timing_critic_audit_capacity),
            replay_manageable_fraction=float(args.timing_critic_manageable_frac),
            replay_near_failure_fraction=float(
                args.timing_critic_near_failure_frac
            ),
            replay_boundary_band=float(args.timing_critic_boundary_band),
            boundary_loss_weight=float(args.timing_critic_boundary_loss),
            boundary_temperature=float(args.timing_critic_boundary_temperature),
            boundary_target_band=float(args.timing_critic_boundary_target_band),
            underprediction_multiplier=float(
                args.timing_critic_underprediction_mult
            ),
            conservative_quantile=float(args.timing_critic_quantile),
            decision_threshold=float(args.timing_critic_threshold),
            assist_fraction=float(args.timing_critic_assist_frac),
            min_assist_samples=int(args.timing_critic_min_assist_samples),
            coarse_step_knots=int(args.timing_critic_coarse_step),
            primary_base_tasks_per_512=int(
                args.timing_primary_base_tasks_per_512
            ),
            primary_anchor_per_task=int(
                args.timing_primary_anchor_per_task
            ),
            primary_boundary_per_task=int(
                args.timing_primary_boundary_per_task
            ),
            primary_interior_per_task=int(
                args.timing_primary_interior_per_task
            ),
            primary_mpc_terminal_limit=float(
                args.timing_primary_mpc_terminal_limit
            ),
            primary_mpc_collision_sdf_limit=float(
                args.timing_primary_mpc_collision_sdf_limit
            ),
            representation=str(args.timing_critic_representation),
            acceptable_normalized_violation=float(
                args.timing_acceptable_normalized_violation
            ),
            risk_min_precision=float(args.timing_risk_min_precision),
            risk_recalibrate_every=int(args.timing_risk_recalibrate_every),
            global_ood_fraction=float(args.timing_global_ood_fraction),
            global_ood_weight=float(args.timing_global_ood_weight),
            anchor_curriculum=bool(args.timing_anchor_curriculum),
            anchor_min_samples=int(args.timing_anchor_min_samples),
            anchor_min_close_rate=float(args.timing_anchor_min_close_rate),
            terminal_limit=float(args.timing_terminal_limit),
            collision_sdf_limit=float(args.timing_collision_sdf_limit),
            joint_limit=float(args.timing_joint_limit),
            torque_limit=float(args.timing_torque_limit),
        ),
        al=phase2_al_config(
            rho_scale=float(args.rho_scale),
            **({} if args.rho_terminal is None
               else {"rho_terminal": float(args.rho_terminal)}),
        ),
        roller_regoal=not args.no_roller_regoal,
        cost=phase2_cost_config(
            n_sub=int(args.n_sub),
            **({} if args.w_torque is None else {"w_torque": float(args.w_torque)}),
            **({} if args.w_velocity is None
               else {"w_velocity": float(args.w_velocity)}),
            **({} if args.w_joint_box is None
               else {"w_joint_box": float(args.w_joint_box)}),
            **({} if args.w_torque_peak is None
               else {"w_torque_peak": float(args.w_torque_peak)}),
            **({} if args.w_velocity_peak is None
               else {"w_velocity_peak": float(args.w_velocity_peak)}),
            **({} if args.w_track_torque is None
               else {"w_track_torque": float(args.w_track_torque)}),
            **({} if args.w_track_torque_peak is None
               else {"w_track_torque_peak": float(args.w_track_torque_peak)}),
            **({} if args.soft_wrist_effort_scale is None
               else {"soft_wrist_effort_scale": float(args.soft_wrist_effort_scale)}),
            **({} if args.w_smooth_jerk is None
               else {"w_smooth_jerk": float(args.w_smooth_jerk)}),
            **({} if args.w_collision is None else {
                "w_collision_drum": float(args.w_collision),
                "w_collision_under": float(args.w_collision),
                "w_collision_self": float(args.w_collision)}),
            **({} if args.w_collision_peak is None
               else {"w_collision_peak": float(args.w_collision_peak)}),
            **({} if args.w_joint_box_peak is None
               else {"w_joint_box_peak": float(args.w_joint_box_peak)}),
            **({} if args.w_stick_stick is None
               else {"w_stick_stick": float(args.w_stick_stick)}),
            **({} if args.w_tip_crossover is None
               else {"w_tip_crossover": float(args.w_tip_crossover)}),
            **({} if args.w_terminal_aim is None
               else {"w_terminal_aim": float(args.w_terminal_aim)}),
            **({} if args.terminal_aim_deadband_m is None
               else {"terminal_aim_deadband_m": float(
                   args.terminal_aim_deadband_m
               )}),
            **({} if args.torque_physics is None
               else {"torque_physics": str(args.torque_physics)}),
            **({} if args.waist_pr_box is None
               else {"waist_pr_box": float(args.waist_pr_box)}),
            **({} if args.w_waist_upright is None
               else {"w_waist_upright": float(args.w_waist_upright)}),
            **({} if args.w_arm_compact is None
               else {"w_arm_compact": float(args.w_arm_compact)}),
            **({} if args.arm_compact_mode is None
               else {"arm_compact_mode": str(args.arm_compact_mode)}),
            **({} if args.arm_compact_height_max is None
               else {"arm_compact_height_max": float(args.arm_compact_height_max)}),
            **({} if args.arm_compact_radius_max is None
               else {"arm_compact_radius_max": float(args.arm_compact_radius_max)}),
            **({} if args.w_waist_pr_terminal is None
               else {"w_waist_pr_terminal": float(args.w_waist_pr_terminal)}),
            **({} if args.w_waist_aim is None
               else {"w_waist_aim": float(args.w_waist_aim)}),
            **({} if args.w_waist_yaw_vel is None
               else {"w_waist_yaw_vel": float(args.w_waist_yaw_vel)}),
            **({} if args.w_waist_pr_vel is None
               else {"w_waist_pr_vel": float(args.w_waist_pr_vel)}),
            **(
                {"w_waist_pr_band": 0.0}
                if args.waist_pr_ema and args.w_waist_pr_band is None
                else ({} if args.w_waist_pr_band is None
                      else {"w_waist_pr_band": float(args.w_waist_pr_band)})
            ),
            **({} if args.waist_pr_band_rad is None
               else {"waist_pr_band_rad": float(args.waist_pr_band_rad)}),
            **({} if args.waist_aim_deadband is None
               else {"waist_aim_deadband": float(args.waist_aim_deadband)}),
            # With the learned set-point the upright well owns pitch/roll; the
            # aim term prices yaw alone.
            **({"waist_aim_yaw_only": True} if args.waist_sp_head else {}),
            **({} if args.w_effort is None else {"w_effort": float(args.w_effort)}),
            **({} if args.w_torque_static is None
               else {"w_torque_static": float(args.w_torque_static)}),
            **({"torque_static_only": True} if args.torque_static_only else {}),
            **({"torque_al_off": True} if args.torque_al_off else {}),
            **({"torque_static_al": True} if args.torque_static_al else {}),
            **({} if args.static_torque_tighten is None
               else {"static_torque_tighten": str(args.static_torque_tighten)}),
            **({} if args.eta_torque is None
               else {"eta_torque": float(args.eta_torque)}),
            **({} if args.effort_fade_wp is None
               else {"effort_fade_wp": int(args.effort_fade_wp)}),
            **({} if w_idle_cyl is None
               else {"w_idle_cylinder": float(w_idle_cyl)}),
            **({} if args.w_idle_vel is None
               else {"w_idle_vel": float(args.w_idle_vel)}),
        ),
        idle_cylinder_path=idle_cylinder_path,
        idle_settle_wp=int(args.idle_settle_wp),
        roller_rest_ticks=int(args.rest_ticks),
        log_every=int(args.log_every),
        eval_every=int(args.eval_every), ckpt_every=int(args.ckpt_every),
        ckpt_last_every=int(args.ckpt_last_every),
        ckpt_keep_every=int(args.ckpt_keep_every),
        tensorboard=bool(args.tensorboard),
        grad_term_every=int(args.grad_term_every),
        torque_curvature_rows=int(args.torque_curvature_rows),
        grad_balance=str(args.grad_balance),
        grad_balance_tau_torque_cap=float(args.grad_balance_tau_torque_cap),
        grad_balance_torque_gamma=float(args.grad_balance_torque_gamma),
        grad_balance_beta=float(args.grad_balance_beta),
        grad_balance_waist_floor=float(args.grad_balance_waist_floor),
        grad_balance_refresh=int(args.grad_balance_refresh),
        grad_balance_norm_ema=float(args.grad_balance_norm_ema),
        torque_grad_scale=float(args.torque_grad_scale),
        torque_grad_cap=float(args.torque_grad_cap),
        torque_grad_cap_refresh=int(args.torque_grad_cap_refresh),
        torque_grad_cap_scale_ema=float(args.torque_grad_cap_scale_ema),
        phase_timing_every=int(args.phase_timing_every),
        replay_viol_downweight=float(args.replay_viol_downweight),
        replay_viol_uniform_mix=float(args.replay_viol_uniform_mix),
        replay_proficiency=bool(args.replay_proficiency),
        replay_frac_prof=float(args.replay_frac_prof),
        replay_frac_new=float(args.replay_frac_new),
        proficiency_viol_cap=float(args.proficiency_viol_cap),
        line_search=bool(args.line_search),
        line_search_cfg=LineSearchConfig(
            **({} if args.line_search_mode is None
               else {"mode": str(args.line_search_mode)}),
            **({} if args.line_search_c1 is None
               else {"c1": float(args.line_search_c1)}),
            **({} if args.line_search_beta is None
               else {"beta_accept_per_update": float(args.line_search_beta)}),
            **({} if args.line_search_max_trials is None
               else {"max_trials": int(args.line_search_max_trials)}),
            **({} if args.line_search_interval is None
               else {"search_interval_mode": str(args.line_search_interval)}),
            **({} if args.line_search_initial_interval is None
               else {
                   "initial_interval": int(args.line_search_initial_interval),
                   "min_interval": int(args.line_search_initial_interval),
                   "max_interval": int(args.line_search_initial_interval),
               }),
            **({} if not args.line_search_disable_early_triggers
               else {"search_on_early_triggers": False}),
            **({} if args.line_search_expansion is None
               else {"expansion_factor": float(args.line_search_expansion)}),
            **({} if args.line_search_warmup_updates is None
               else {"warmup_updates": int(args.line_search_warmup_updates)}),
            **({} if args.line_search_epsilon_seed is None
               else {"epsilon_seed": float(args.line_search_epsilon_seed)}),
            **({} if args.line_search_epsilon_min is None
               else {"epsilon_min": float(args.line_search_epsilon_min)}),
            **({} if args.line_search_epsilon_max is None
               else {"epsilon_max": float(args.line_search_epsilon_max)}),
            **({} if not args.line_search_reseed_epsilon_on_objective_change
               else {"reseed_epsilon_on_objective_change": True}),
            **({} if not args.line_search_force_al_reset
               else {"force_search_on_al_change": True}),
            **({} if args.line_search_moment_on is None
               else {"moment_on": str(args.line_search_moment_on)}),
            **({} if args.line_search_actual_diagnostic_every is None
               else {"actual_update_diagnostic_every":
                     int(args.line_search_actual_diagnostic_every)}),
            **({} if not args.line_search_strict_actual_update
               else {"strict_actual_update": True}),
            **({} if not args.line_search_reject_current_probe_increase
               else {"reject_current_probe_increase": True}),
            **({} if not args.line_search_skip_on_probe_exhaustion
               else {"skip_on_probe_exhaustion": True}),
        ),
        optimizer=str(args.optimizer),
        lbfgs_history=int(args.lbfgs_history),
        short_wp=int(args.short_wp), short_boost=float(args.short_boost),
        rhythm_idle_frac=float(args.rhythm_idle_frac),
        post_hit_frac=float(args.post_hit_frac),
        far_fetched_frac=float(args.far_fetched_frac),
        eval_like_mix_frac=float(args.eval_like_frac),
        cmd_lead_s=resolve_lead(args.cmd_lead),
        probe_every=int(args.probe_every),
        roller_slots=int(args.roller_slots), roller_horizon=int(args.roller_horizon),
        roller_start=int(args.roller_start),
        roller_ticks_per_step=int(args.roller_ticks_per_step),
        plant_saturate_effort=bool(args.plant_saturate_effort),
        seed=int(args.seed), device=str(args.device),
        aux_torque_tracking=bool(args.aux_torque_tracking),
    )
    # Frontier knobs: assigned after construct so a parallel agent adding the
    # Phase2Config fields does not fight argparse.  No-ops in the trainer
    # unless cfg.frontier.
    cfg.frontier = bool(args.frontier)
    cfg.frontier_proposals = int(args.frontier_proposals)
    cfg.frontier_chart_cap = int(args.frontier_chart_cap)
    cfg.frontier_fresh_frac = float(args.frontier_fresh_frac)
    cfg.frontier_spawn_frac = float(args.frontier_spawn_frac)
    cfg.frontier_sigma_q = float(args.frontier_sigma_q)
    cfg.frontier_sigma_qd = float(args.frontier_sigma_qd)
    cfg.frontier_sigma_tth = float(args.frontier_sigma_tth)
    cfg.frontier_ema_alpha = float(args.frontier_ema_alpha)
    cfg.frontier_ema_nmin = int(args.frontier_ema_nmin)
    cfg.frontier_ema_prof = float(args.frontier_ema_prof)
    cfg.frontier_landmark_k = int(args.frontier_landmark_k)
    cfg.frontier_landmark_tau = float(args.frontier_landmark_tau)
    cfg.frontier_propose_candidates = int(args.frontier_propose_candidates)
    cfg.frontier_hull_ema = float(args.frontier_hull_ema)
    cfg.frontier_train_inv_frac = float(args.frontier_train_inv_frac)
    cfg.frontier_train_inv_pool = int(args.frontier_train_inv_pool)
    cfg.frontier_sample_k = int(args.frontier_sample_k)
    cfg.frontier_traj_stride = int(args.frontier_traj_stride)
    cfg.frontier_include_miss = bool(args.frontier_include_miss)
    cfg.frontier_idle_far_cloud = bool(args.frontier_idle_far_cloud)
    cfg.frontier_splice_k = int(args.frontier_splice_k)
    cfg.frontier_auto_miss_mm = float(args.frontier_auto_miss_mm)
    cfg.frontier_auto_full_weights = not bool(args.no_frontier_auto_full_weights)
    cfg.frontier_auto_viol_cap = float(args.frontier_auto_viol_cap)
    cfg.frontier_auto_viol_ratio = float(args.frontier_auto_viol_ratio)
    cfg.frontier_auto_viol_window = int(args.frontier_auto_viol_window)
    if args.w_aux is not None:
        cfg.w_aux = float(args.w_aux)
    if args.w_tc is not None:
        cfg.w_tc = float(args.w_tc)
    if args.tc_frac is not None:
        cfg.tc_frac = float(args.tc_frac)
    if args.w_mpc_init_tie is not None:
        cfg.w_mpc_init_tie = float(args.w_mpc_init_tie)
    if args.suffix_frac is not None:
        cfg.suffix_frac = float(args.suffix_frac)
    if args.w_suffix_cost is not None:
        cfg.w_suffix_cost = float(args.w_suffix_cost)
    if args.w_suffix_tie is not None:
        cfg.w_suffix_tie = float(args.w_suffix_tie)
    if args.suffix_warmup is not None:
        cfg.suffix_warmup = int(args.suffix_warmup)
    if args.dls_terminal_frac is not None:
        cfg.dls_terminal_frac = float(args.dls_terminal_frac)
    if args.w_dls_terminal_cost is not None:
        cfg.w_dls_terminal_cost = float(args.w_dls_terminal_cost)
    if args.w_dls_terminal_tie is not None:
        cfg.w_dls_terminal_tie = float(args.w_dls_terminal_tie)
    if args.dls_terminal_steps is not None:
        cfg.dls_terminal_steps = int(args.dls_terminal_steps)
    if args.dls_terminal_window is not None:
        cfg.dls_terminal_window = int(args.dls_terminal_window)
    if args.dls_terminal_damping is not None:
        cfg.dls_terminal_damping = float(args.dls_terminal_damping)
    if args.dls_terminal_max_step_rad is not None:
        cfg.dls_terminal_max_step_rad = float(args.dls_terminal_max_step_rad)
    if args.terminal_lookahead_mode is not None:
        cfg.terminal_lookahead_mode = str(args.terminal_lookahead_mode)
    if args.curvature_damping is not None:
        cfg.curvature_damping = float(args.curvature_damping)
    if args.curvature_position_weight is not None:
        cfg.curvature_position_weight = float(args.curvature_position_weight)
    if args.curvature_velocity_weight is not None:
        cfg.curvature_velocity_weight = float(args.curvature_velocity_weight)
    if args.curvature_max_step_rad is not None:
        cfg.curvature_max_step_rad = float(args.curvature_max_step_rad)
    if args.curvature_direct_frac is not None:
        cfg.curvature_direct_frac = float(args.curvature_direct_frac)
    if args.curvature_direct_mix is not None:
        cfg.curvature_direct_mix = float(args.curvature_direct_mix)
    if args.yaw_now_jitter is not None:
        cfg.yaw_now_jitter = float(args.yaw_now_jitter)
    if args.ramp_start is not None:
        cfg.schedule = replace(cfg.schedule, ramp_start=int(args.ramp_start))
    if args.ramp_end is not None:
        cfg.schedule = replace(cfg.schedule, ramp_end=int(args.ramp_end))
    if args.target_acceptance is not None:
        cfg.target_acceptance = float(args.target_acceptance)
    if args.plateau_evals is not None:
        cfg.plateau_evals = int(args.plateau_evals)
    if args.lr_horizon is not None:
        cfg.lr_horizon = int(args.lr_horizon)
    sched_kw: dict = {
        "gate_stage_a": bool(args.stage_a_gate),
        "full_limit_joints": (
            () if args.no_waist_pr_full_limits
            else cfg.schedule.full_limit_joints
        ),
    }
    if args.torque_full_start is not None:
        sched_kw["torque_full_start"] = int(args.torque_full_start)
    if args.torque_full_ramp is not None:
        sched_kw["torque_full_ramp"] = int(args.torque_full_ramp)
    # Curriculum ramps are opt-in.  argparse used to default these to 300/1000
    # and overwrite Phase2Config's snap_full() schedule, which reopened the
    # waist-rate / peak window the AL is supposed to own.
    _ramp = (
        ("torque_peak_ramp", args.torque_peak_ramp, "torque_peak_start"),
        ("velocity_peak_ramp", args.velocity_peak_ramp, "velocity_peak_start"),
        ("collision_peak_ramp", args.collision_peak_ramp, "collision_peak_start"),
        ("joint_box_peak_ramp", args.joint_box_peak_ramp, "joint_box_peak_start"),
        ("effort_ramp", args.effort_ramp, "effort_start"),
        ("idle_ramp", args.idle_ramp, "idle_start"),
        ("tip_crossover_ramp", args.tip_crossover_ramp, "tip_crossover_start"),
    )
    for ramp_f, val, start_f in _ramp:
        if val is not None:
            sched_kw[ramp_f] = int(val)
            sched_kw[start_f] = 0
    if args.waist_full:
        sched_kw["waist_start"] = -1
        sched_kw["waist_ramp"] = 1
    elif args.waist_ramp is not None:
        sched_kw["waist_ramp"] = int(args.waist_ramp)
        sched_kw["waist_start"] = 0
    cfg.schedule = replace(cfg.schedule, **sched_kw)
    if (
        cfg.cost.torque_static_only
        or cfg.cost.torque_al_off
        or cfg.cost.torque_static_al
        or cfg.cost.w_torque_static > 0.0
        or cfg.cost.torque_physics != "full"
    ):
        print(
            f"[cfg] torque: physics={cfg.cost.torque_physics} "
            f"static_only={cfg.cost.torque_static_only} "
            f"static_al={cfg.cost.torque_static_al} al_off={cfg.cost.torque_al_off} "
            f"w_torque_static={cfg.cost.w_torque_static} "
            f"w_torque={cfg.cost.w_torque} w_torque_peak={cfg.cost.w_torque_peak} "
            f"eta_torque={cfg.cost.eta_torque} "
            f"static_tighten={cfg.cost.static_torque_tighten} "
            f"blend@0={cfg.schedule.torque_full_blend(0):.3f}",
            flush=True,
        )
    if args.full_weights:
        # Applied AFTER the per-ramp CLI so --full-weights wins over
        # --ramp-start / --waist-ramp / --suffix-warmup / etc.
        cfg.schedule = cfg.schedule.snap_full()
        cfg.suffix_warmup = 1
        m0 = cfg.schedule.multipliers(0)
        print("[cfg] full-weights: all ramps at 1.0 from step 0 "
              f"(torque={m0['torque']:.2f} strike={m0['terminal_velocity']:.2f} "
              f"waist={m0['waist_box']:.2f} peak={m0['torque_peak']:.2f}/"
              f"{m0['velocity_peak']:.2f}/{m0['collision_peak']:.2f}/"
              f"{m0['joint_box_peak']:.2f} "
              f"idle={m0['posture_idle']:.2f} "
              f"crossover={m0['tip_crossover']:.2f} "
              f"suffix_warmup={cfg.suffix_warmup})",
              flush=True)
    if args.bptt_start_frac is not None:
        cfg.bptt_start_frac = float(args.bptt_start_frac)
    if args.bptt_max_frac is not None:
        cfg.bptt_max_frac = float(args.bptt_max_frac)
    if args.waist_sp_head:
        cfg.waist_sp_from_head = True
        if args.w_waist_aim is None or float(args.w_waist_aim) <= 0.0:
            print("[cfg] WARNING: --waist-sp-head without a non-zero "
                  "--w-waist-aim -- the anchor term prices nothing", flush=True)
    if args.waist_pr_sp_head:
        cfg.waist_pr_sp_from_head = True
        if args.w_waist_upright is None or float(args.w_waist_upright) <= 0.0:
            print("[cfg] WARNING: --waist-pr-sp-head without a non-zero "
                  "--w-waist-upright -- the lean anchor prices nothing", flush=True)
        if args.freeze_waist_pr:
            raise SystemExit(
                "--waist-pr-sp-head and --freeze-waist-pr conflict: the head "
                "has nothing to control if pitch/roll are pinned"
            )
        if args.waist_pr_ema:
            raise SystemExit(
                "--waist-pr-sp-head and --waist-pr-ema conflict: both want "
                "StrikeContext.waist_pr_ref"
            )
    if args.waist_pr_ema:
        if args.freeze_waist_pr:
            raise SystemExit(
                "--waist-pr-ema and --freeze-waist-pr conflict: freeze removes "
                "P/R from the action space the EMA would track"
            )
        cfg.waist_pr_ema = True
        cfg.waist_pr_ema_alpha = float(args.waist_pr_ema_alpha)
        cfg.waist_pr_ema_leak = float(args.waist_pr_ema_leak)
        if args.w_waist_upright is None or float(args.w_waist_upright) <= 0.0:
            print("[cfg] WARNING: --waist-pr-ema without a non-zero "
                  "--w-waist-upright -- the leaked EMA has nothing to pull",
                  flush=True)
        print(f"[cfg] waist-pr EMA ON: alpha={cfg.waist_pr_ema_alpha:g} "
              f"leak={cfg.waist_pr_ema_leak:g} (replay well vs leaked cmd; "
              f"snapshot vs 0; peak band "
              f"{'off' if args.w_waist_pr_band is None else args.w_waist_pr_band})",
              flush=True)
    if args.w_waist_pr_terminal is not None and float(args.w_waist_pr_terminal) > 0.0:
        print(f"[cfg] waist-pr terminal V_f ON: w={float(args.w_waist_pr_terminal):g} "
              f"(nearer active hit vs 0)", flush=True)
    if args.w_waist_sp_tether is not None:
        cfg.w_waist_sp_tether = float(args.w_waist_sp_tether)
    if args.w_waist_pr_sp_tether is not None:
        cfg.w_waist_pr_sp_tether = float(args.w_waist_pr_sp_tether)
    elif args.waist_pr_sp_head:
        cfg.w_waist_pr_sp_tether = 5.0
        print("[cfg] w_waist_pr_sp_tether defaulted to 5.0 "
              "(pass --w-waist-pr-sp-tether to override)", flush=True)
    if args.w_waist_pr_sp_hold is not None:
        cfg.w_waist_pr_sp_hold = float(args.w_waist_pr_sp_hold)
    elif args.waist_pr_sp_head:
        cfg.w_waist_pr_sp_hold = 5.0
        print("[cfg] w_waist_pr_sp_hold defaulted to 5.0 "
              "(pass --w-waist-pr-sp-hold to override)", flush=True)
    if args.w_waist_pr_sp_hold_band is not None:
        cfg.w_waist_pr_sp_hold_band = float(args.w_waist_pr_sp_hold_band)
    elif args.waist_pr_sp_head:
        cfg.w_waist_pr_sp_hold_band = 10.0
        print("[cfg] w_waist_pr_sp_hold_band defaulted to 10.0 "
              f"(band={cfg.cost.waist_pr_band_rad:g} rad; "
              "pass --w-waist-pr-sp-hold-band to override)", flush=True)
    if args.pr_distill_hit_m is not None:
        cfg.pr_distill_hit_m = float(args.pr_distill_hit_m)
    if args.w_waist_pr_sp_distill is not None:
        cfg.w_waist_pr_sp_distill = float(args.w_waist_pr_sp_distill)
        if cfg.w_waist_pr_sp_distill > 0.0:
            print(f"[cfg] waist-pr distill ON: w={cfg.w_waist_pr_sp_distill:g} "
                  "gated strike-knot P/R -> encoder SP "
                  f"(hit_m={cfg.pr_distill_hit_m:g} m)",
                  flush=True)
    if args.w_waist_sp_hold is not None:
        cfg.w_waist_sp_hold = float(args.w_waist_sp_hold)
    if args.w_waist_sp_smooth is not None:
        cfg.w_waist_sp_smooth = float(args.w_waist_sp_smooth)
        if cfg.w_waist_sp_smooth > 0.0:
            print(f"[cfg] waist-sp smooth ON: w={cfg.w_waist_sp_smooth:g} "
                  "on next-hit re-encode (sp_new - sp_old)²", flush=True)
    if args.waist_sp_search_frac is not None:
        cfg.waist_sp_search_frac = float(args.waist_sp_search_frac)
    if args.waist_sp_search_k is not None:
        cfg.waist_sp_search_k = int(args.waist_sp_search_k)
    if args.waist_sp_search_span is not None:
        cfg.waist_sp_search_span = float(args.waist_sp_search_span)
    if args.waist_sp_search_temp is not None:
        cfg.waist_sp_search_temp = float(args.waist_sp_search_temp)
    if args.w_waist_sp_search is not None:
        cfg.w_waist_sp_search = float(args.w_waist_sp_search)
    if cfg.waist_sp_search_frac > 0.0 and cfg.w_waist_sp_search > 0.0:
        # BOTH knobs gate the path, and the head is what the loss regresses --
        # say so at start-up rather than let a run silently pay nothing.
        if not (args.waist_sp_head or getattr(model_cfg, "waist_sp_head", False)):
            raise SystemExit(
                "--waist-sp-search-frac needs the set-point head "
                "(--waist-sp-head, or a checkpoint already trained with it)"
            )
        n_cand = len(waist_sp_offsets(cfg.waist_sp_search_k,
                                      cfg.waist_sp_search_span)) + 1
        print(f"[cfg] waist-sp search distillation ON: {n_cand} candidates "
              f"(span +-{cfg.waist_sp_search_span:.2f} rad) on the leading "
              f"{cfg.waist_sp_search_frac:.0%} of each batch, temp "
              f"{cfg.waist_sp_search_temp:g}, w {cfg.w_waist_sp_search:g}",
              flush=True)
    elif cfg.waist_sp_search_frac > 0.0 or cfg.w_waist_sp_search > 0.0:
        print("[cfg] WARNING: waist-sp search needs BOTH "
              "--waist-sp-search-frac and --w-waist-sp-search non-zero -- "
              "it is OFF", flush=True)
    if cfg.waist_sp_from_head:
        # The head's zero-init residual only equals the analytic law where the
        # pool carries a solved ``aim_yaw`` (``aim_valid``); a pool without one
        # (pre-aim data, or a cache the schema check somehow missed) degrades
        # every anchor to "hold" -- or worse, to yaw = 0 -- while ``sp_res``
        # logs 0.0.  Independent of the cache schema marker on purpose: this
        # also covers freshly generated pools.
        n_valid = n_rows = 0
        for b in pool.buckets:
            d = pool.data[b]
            n_rows += pool.size(b)
            if "aim_valid" in d:
                n_valid += int(d["aim_valid"].to(torch.bool).sum())
        if n_rows == 0 or n_valid == 0:
            raise RuntimeError(
                "--waist-sp-head needs a pool that carries solved waist-aim "
                f"tokens, but aim_valid is True on 0 of {n_rows} rows -- "
                "regenerate the scenario pool (and any --pool-cache) with the "
                "current code"
            )
        print(f"[data] waist-aim tokens: aim_valid on {n_valid}/{n_rows} rows",
              flush=True)

    # ---- rolling-eval grid --------------------------------------------------
    # ONE switch, DERIVED from the objective that is actually configured: the
    # rolling eval scores the torque the loss is optimizing, or a mixed state is
    # configurable in which training minimizes the PD demand while ``accepted``
    # still gates on the plan's RNEA.  Read off ``cfg.cost`` rather than off
    # ``args`` so a config path that sets the weights some other way cannot
    # split the two.
    track_eval = bool(
        cfg.cost.w_track_torque > 0.0 or cfg.cost.w_track_torque_peak > 0.0
    )
    wrist_scale = float(cfg.cost.soft_wrist_effort_scale)
    if track_eval:
        print("[eval] ok_torque / accepted scored on TRACKING torque "
              "(RNEA + PD feedback estimate) -- not comparable with earlier runs",
              flush=True)
    else:
        print("[eval] ok_torque / accepted scored on KNOT RNEA "
              "(plan inverse dynamics vs rating; PD-FIR demand is off)",
              flush=True)
    if wrist_scale != 1.0:
        print(f"[eval] soft-wrist effort scale={wrist_scale:g}x on pitch/yaw "
              f"(scoring/gate only; IdealPD clamp stays at Isaac ratings)",
              flush=True)
    scorer = A.TrajectoryScorer(
        device="cpu", track_torque=track_eval,
        soft_wrist_effort_scale=wrist_scale,
    )
    print(f"[eval] on_head hit_margin="
          f"{scorer.thresholds.hit_margin_m*1000:.0f} mm "
          f"(MPC slack on the 81-grid; probe stays tight)", flush=True)
    # ``include_extra_drums`` adds FLOOR_TOM.  The repo grid left it out and the
    # port inherited that, so 1 of the 7 drums in the layout -- the largest tom,
    # the longest reach, and present in both the training pool and the collision
    # scene -- was never once evaluated.  Every acceptance number before
    # 2026-08-14 is blind to it.
    above = tuple(n for n in (args.eval_start_above_drums or ()) if n)
    clearance = float(args.eval_start_clearance) if above else 0.0
    cases = eval_subset(
        A.generate_grid(seed=0, grid="standard", scorer=scorer,
                        include_extra_drums=True,
                        start_above_drums=above,
                        start_not_under_drums=bool(above),
                        start_clearance_m=clearance),
        int(args.eval_cases),
    )
    import collections as _c
    print(f"[eval] drums: {dict(_c.Counter(c.drum for c in cases))}", flush=True)
    print(f"[eval] rolling grid: {len(cases)} cases", flush=True)
    if above:
        print(f"[eval] start tip z >= max head-z of {list(above)}; "
              f"idle at ready; no stick (tip/shaft/wrist) under any drum "
              f"(cymbals included); q0 SDF > {clearance*1000:.0f} mm",
              flush=True)
    else:
        print("[eval] start filter OFF (historical hanging URDF-zero idles)",
              flush=True)
    kit = A.DrumKit.from_layout()
    library = A.GoalPoseLibrary()

    # ---- closed-loop probe ---------------------------------------------------
    probe_fn = None
    if args.probe_every > 0:
        from amortized_opt.probe import closed_loop_probe

        probe_clips = A.load_rcc_pool(A.default_rcc_dir("eval"), kit)[: args.probe_clips]
        probe_scorer = A.TrajectoryScorer(
            device="cpu", track_torque=track_eval,
            soft_wrist_effort_scale=wrist_scale,
            thresholds=A.AcceptanceThresholds(hit_margin_m=0.0),
        )
        print(f"[probe] every {args.probe_every} steps on "
              f"{[c.name for c in probe_clips]} "
              f"(lead {args.probe_lead*1000:.0f} ms, freeze {args.probe_freeze} wp, "
              f"bounce {'off' if args.no_probe_bounce else 'on'}, "
              f"tight on_head)",
              flush=True)

        def probe_fn(m, _clips=probe_clips, _sc=probe_scorer):  # noqa: F811
            r = closed_loop_probe(
                m, kit=kit, scorer=_sc, clips=_clips, device=dev,
                num_drums=model_cfg.num_drums, num_future=model_cfg.num_future,
                horizon=int(args.buckets[-1]), steps=int(args.probe_steps),
                seed=0, amp_dtype=None if args.no_amp else torch.bfloat16,
                lead_s=float(args.probe_lead), freeze_wp=int(args.probe_freeze),
                cmd_lead_s=resolve_lead(
                    args.probe_cmd_lead if args.probe_cmd_lead is not None
                    else args.cmd_lead),
                waist_upright_ease_wp=args.waist_upright_ease_wp,
                saturate_effort=bool(args.plant_saturate_effort),
                bounce=not bool(args.no_probe_bounce),
                record_drift=False,
                waist_pr_ema=bool(args.waist_pr_ema),
                waist_pr_ema_alpha=float(args.waist_pr_ema_alpha),
                waist_pr_ema_leak=float(args.waist_pr_ema_leak),
            )
            r.pop("_hits", None)
            return r

    al_state = opt_state = line_search_state = voxel_train_state = None
    timing_critic_state = None
    start_step = 0
    if args.reset_optimizer and not args.resume:
        raise SystemExit("--reset-optimizer requires --resume")
    if args.resume:
        if cfg.timing_critic.enabled:
            timing_critic_state = (ck.get("extra") or {}).get("timing_critic")
            if (
                timing_critic_state
                and (
                    timing_critic_state.get("representation", "regression")
                    != cfg.timing_critic.representation
                    or (
                        cfg.timing_critic.representation == "regression"
                        and int(timing_critic_state.get("format_version", -1))
                        != int(A.CRITIC_FORMAT_VERSION)
                    )
                )
            ):
                print(
                    "[resume] timing critic feature schema changed; "
                    "discarding legacy critic state and initializing fresh",
                    flush=True,
                )
                timing_critic_state = None
            print(
                "[resume] timing critic state "
                + ("restored" if timing_critic_state else "missing; initialize fresh"),
                flush=True,
            )
        if cfg.voxel_pool:
            voxel_train_state = (ck.get("extra") or {}).get("voxel_training")
            print(
                "[resume] voxel curriculum state "
                + ("restored" if voxel_train_state else "missing; re-bootstrap"),
                flush=True,
            )
        al_state, opt_state, start_step, line_search_state = (
            _resume_training_states(
                ck,
                reset_optimizer=bool(args.reset_optimizer),
                restore_line_search=bool(
                    cfg.line_search and not args.line_search_fresh
                ),
            )
        )
        if args.reset_optimizer:
            print(
                "[resume] optimizer reset requested: retaining model/AL/"
                f"start_step={start_step}; Adam moments and checkpoint group "
                "LRs will not be restored",
                flush=True,
            )
        if args.reset_al is not None and al_state:
            lam = dict(al_state.get("lam") or {})
            groups = list(args.reset_al) or sorted(lam)
            unknown = [g for g in groups if g not in lam]
            if unknown:
                raise SystemExit(
                    f"--reset-al: unknown group(s) {unknown}; "
                    f"checkpoint has {sorted(lam)}"
                )
            init = float(phase2_al_config().lam_init)
            for g in groups:
                print(f"[resume] AL reset {g}: {lam[g]:.2f} -> {init:.2f}",
                      flush=True)
                lam[g] = init
            kept = {k: v for k, v in lam.items() if k not in groups}
            print("[resume] AL kept "
                  + ", ".join(f"{k}={v:.1f}" for k, v in sorted(kept.items())),
                  flush=True)
            al_state = {**al_state, "lam": lam}
        print(f"[resume] AL {'yes' if al_state else 'no'}, "
              f"optimizer {'yes' if opt_state else 'no'}, "
              f"start_step={start_step}", flush=True)

    if args.resume and cfg.line_search and not args.line_search_fresh:
        if line_search_state is not None:
            print("[resume] line_search selector state restored from checkpoint",
                  flush=True)
    elif args.resume and cfg.line_search and args.line_search_fresh:
        print("[resume] line_search_fresh=1 — SaLSa selector reset from cfg "
              f"(seed={cfg.line_search_cfg.epsilon_seed:.3g}, "
              f"bounds=[{cfg.line_search_cfg.resolve_bounds()[0]:.3g}, "
              f"{cfg.line_search_cfg.resolve_bounds()[1]:.3g}])",
              flush=True)

    summary = train_phase2(
        model=model, pool=pool, cfg=cfg, model_cfg=model_cfg, run_dir=run_dir,
        scenario_cfg=scen, eval_cases=cases, scorer=scorer, kit=kit, library=library,
        al_state=al_state, opt_state=opt_state, probe_fn=probe_fn,
        stage_a_init=bool(args.stage_a_open), max_hours=args.max_hours,
        start_step=start_step,
        line_search_state=line_search_state,
        voxel_train_state=voxel_train_state,
        timing_critic_state=timing_critic_state,
    )
    summary["args"] = vars(args)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] absolute step {summary['steps']}; "
          f"{summary['updates_attempted']} updates attempted, "
          f"{summary['updates_executed']} executed in {summary['wall_s']:.0f} s "
          f"({summary['steps_per_s']:.2f} attempted it/s, "
          f"{summary['executed_steps_per_s']:.2f} executed it/s), best acceptance "
          f"{summary['best_acceptance']:.3f} @ {summary['best_step']} "
          f"({summary['stopped']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
