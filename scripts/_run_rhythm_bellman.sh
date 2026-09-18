#!/usr/bin/env bash
# Rhythm-derived Phase-2 run: peak torque + suffix-Bellman + stick-stick +
# collision peak.  Torso is DECODER-OWNED (direct movements): upright well vs
# 0, yaw/P/R rate wells, V_f at the nearer hit — no encoder waist set-point
# heads and no w_waist_aim IL.  Cost weights are FULL from step 0; AL λ adjusts.
# Default launch uses the positive-witness voxel curriculum.  It keeps the
# checkpoint-compatible future slot masked invalid and strips legacy encoder
# waist SP heads before model construction.  Set VOXEL_POOL=0 to run the old
# replay/SaLSa A/B path.
#
# Cost/eval hit scoring is the playing-shaft capsule on cymbals/hi-hat and
# the bead on pads; collision is stick capsule + wrist vs drums /
# under-volume / torso.
#
# Torque hinge / ok_torque are KNOT RNEA vs the hardware rating (can a
# clipped motor fly this spline). Tracking-torque (RNEA + PD-FIR demand)
# stays off -- that gate was scoring unclipped first-substep Kd spikes,
# not inverse dynamics.
#
# Kit DR is ON end-to-end: default KitRandomization xy/z, plus per-drum tilt
# and radius (baked into the pool -- must key the cache).
#
# Default recipe fork (SaLSa line search): LINE_SEARCH=1, surgery off
# (grad_balance=""), BPTT never starts (BPTT_START_FRAC=2).  Probe replay
# does not yet freeze BPTT's second batch.  Restore the prior PCGrad+BPTT
# arm with: LINE_SEARCH=0 TAG=rhythm_bellman_nf1_direct
#
# Usage:
#   DEV=cuda:1 VOXEL_POOL=1 bash scripts/_run_rhythm_bellman.sh
#   # phase 2 only, reuse a finished phase-1 warm start:
#   SKIP_P1=1 INIT=runs/phase1_rhythm_bellman_nf1_prsp/best.pt \
#     bash scripts/_run_rhythm_bellman.sh
#   # prior PCGrad + BPTT-onset recipe:
#   LINE_SEARCH=0 TAG=rhythm_bellman_nf1_direct bash scripts/_run_rhythm_bellman.sh
# Resume weights/AL/absolute step with fresh optimizer and selector state:
#   RESET_OPTIMIZER=1 LINE_SEARCH_FRESH=1 RESUME=runs/.../last.pt \
#     bash scripts/_run_rhythm_bellman.sh
# continuation / generalization (after a trained phase-2 ckpt):
#   SKIP_P1=1 RESUME=runs/rhythm_bellman_nf1_waist_gain/last.pt FRONTIER=1 \
#     FULL_WEIGHTS=1 TAG=rhythm_bellman_nf1_frontier bash scripts/_run_rhythm_bellman.sh
set -euo pipefail

echo "[run] starting _run_rhythm_bellman.sh" >&2
cd "$(dirname "$0")/.."
echo "[run] cwd=$(pwd)" >&2

# Prefer activating without relying on interactive conda hooks (those can exit 0
# with no output under `set -e` when already in base).
if [[ -f /data2/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /data2/miniconda3/etc/profile.d/conda.sh
  AMORTIZED_CONDA_ENV="${AMORTIZED_CONDA_ENV:-/data2/eric/conda_envs/amortized_compile}"
  conda activate "$AMORTIZED_CONDA_ENV"
  echo "[run] conda env=$CONDA_PREFIX" >&2
else
  echo "[run] missing conda.sh" >&2
  exit 1
fi
# Isaac/Unitree preamble is NOT sourced here. Under `set -e` a failure inside
# that script aborts this wrapper with no message (and the old `2>/dev/null`
# hid the reason). train_phase2 only needs the g1drum conda env.
export UNITREE_MODEL_DIR="${UNITREE_MODEL_DIR:-/data2/eric/workspace/unitree_model}"
export TMPDIR="${TMPDIR:-/data2/eric/tmp}"
# Torque grad recipe: cap segment+peak λ; PHR on torque off unless overridden.
# Cosine conflict diagnostics land on every train log row via ``gb_stats``.
LAM_TORQUE_MAX="${LAM_TORQUE_MAX:-20}"
export AMORTIZED_LAM_TORQUE_MAX="$LAM_TORQUE_MAX"
LAM_TORQUE_WAIST_MAX="${LAM_TORQUE_WAIST_MAX:-}"
if [[ -n "$LAM_TORQUE_WAIST_MAX" ]]; then
  export AMORTIZED_LAM_TORQUE_WAIST_MAX="$LAM_TORQUE_WAIST_MAX"
fi
# Pool seed + roller P/R ease window (waypoints).  Default 1 (upright ref
# essentially immediately after knot 0); ``none`` omits the flag (hold measured lean).
WAIST_UPRIGHT_EASE_WP="${WAIST_UPRIGHT_EASE_WP:-1}"
WAIST_EASE_FLAG=()
if [[ -n "$WAIST_UPRIGHT_EASE_WP" && "$WAIST_UPRIGHT_EASE_WP" != "none" ]]; then
  WAIST_EASE_FLAG=(--waist-upright-ease-wp "$WAIST_UPRIGHT_EASE_WP")
fi
mkdir -p "$TMPDIR" runs

DEV="${DEV:-cuda:1}"
# The distributional assist/control path replaces voxel storage with its
# bounded previous-generation MPC ring and critic-selected timing tables.
TIMING_CRITIC_MODE="${TIMING_CRITIC_MODE:-off}"
if [[ "$TIMING_CRITIC_MODE" == "assist" || "$TIMING_CRITIC_MODE" == "control" ]]; then
  VOXEL_POOL="${VOXEL_POOL:-0}"
else
  # Mixed voxel rows are not line-search replay-frozen yet.  Voxel remains the
  # default when the neural timing curriculum is off.
  VOXEL_POOL="${VOXEL_POOL:-1}"
fi
if [[ -z "${LINE_SEARCH+x}" && "$VOXEL_POOL" == "1" ]]; then
  LINE_SEARCH="${LINE_SEARCH:-0}"
elif [[ -z "${LINE_SEARCH+x}" ]]; then
  LINE_SEARCH="${LINE_SEARCH:-1}"
fi

# Learned task-time feasibility model.  Shadow mode consumes diagnostics from
# the normal cost pass but cannot change actor batches.
TIMING_CRITIC_FLAGS=(--timing-critic-mode "$TIMING_CRITIC_MODE")
if [[ "$TIMING_CRITIC_MODE" == "assist" || "$TIMING_CRITIC_MODE" == "control" ]]; then
  TIMING_CRITIC_FLAGS+=(
    --timing-critic-representation distributional
    --timing-critic-heads "${TIMING_CRITIC_HEADS:-3}"
    --timing-critic-ema-decay "${TIMING_CRITIC_EMA_DECAY:-0.95}"
    --timing-anchor-curriculum
    --timing-anchor-min-samples "${TIMING_ANCHOR_MIN_SAMPLES:-2048}"
    --timing-anchor-min-close-rate "${TIMING_ANCHOR_MIN_CLOSE_RATE:-0.50}"
    --timing-acceptable-normalized-violation "${TIMING_ACCEPTABLE_NORMALIZED_VIOLATION:-0.2}"
    --timing-risk-min-precision "${TIMING_RISK_MIN_PRECISION:-0.50}"
    --timing-primary-base-tasks-per-512 "${TIMING_PRIMARY_BASE_TASKS_PER_512:-40}"
    --timing-primary-boundary-per-task "${TIMING_PRIMARY_BOUNDARY_PER_TASK:-2}"
    --timing-primary-interior-per-task "${TIMING_PRIMARY_INTERIOR_PER_TASK:-4}"
    --timing-primary-mpc-terminal-limit "${TIMING_PRIMARY_MPC_TERMINAL_LIMIT:-0.040}"
    --timing-primary-mpc-collision-sdf-limit "${TIMING_PRIMARY_MPC_COLLISION_SDF_LIMIT:--0.010}"
  )
fi
if [[ -n "${TIMING_CRITIC_RECENT_CAPACITY:-}" ]]; then
  TIMING_CRITIC_FLAGS+=(--timing-critic-recent-capacity "$TIMING_CRITIC_RECENT_CAPACITY")
fi
if [[ -n "${TIMING_CRITIC_AUDIT_CAPACITY:-}" ]]; then
  TIMING_CRITIC_FLAGS+=(--timing-critic-audit-capacity "$TIMING_CRITIC_AUDIT_CAPACITY")
fi
if [[ -n "${TIMING_CRITIC_ASSIST_FRAC:-}" ]]; then
  TIMING_CRITIC_FLAGS+=(--timing-critic-assist-frac "$TIMING_CRITIC_ASSIST_FRAC")
fi
if [[ -n "${TIMING_CRITIC_MIN_ASSIST_SAMPLES:-}" ]]; then
  TIMING_CRITIC_FLAGS+=(--timing-critic-min-assist-samples "$TIMING_CRITIC_MIN_ASSIST_SAMPLES")
fi
if [[ "$VOXEL_POOL" == "1" && "$LINE_SEARCH" == "1" ]]; then
  echo "VOXEL_POOL=1 requires LINE_SEARCH=0" >&2
  exit 1
fi
if [[ "$VOXEL_POOL" == "1" && ( "$TIMING_CRITIC_MODE" == "assist" || "$TIMING_CRITIC_MODE" == "control" ) ]]; then
  echo "distributional timing assist/control replaces VOXEL_POOL; set VOXEL_POOL=0" >&2
  exit 1
fi
if [[ "$LINE_SEARCH" == "1" ]]; then
  TAG="${TAG:-rhythm_bellman_nf1_ls}"
elif [[ "$VOXEL_POOL" == "1" ]]; then
  TAG="${TAG:-rhythm_bellman_nf1_voxel}"
elif [[ "$TIMING_CRITIC_MODE" == "assist" || "$TIMING_CRITIC_MODE" == "control" ]]; then
  TAG="${TAG:-rhythm_bellman_nf1_timing}"
else
  TAG="${TAG:-rhythm_bellman_nf1_direct}"
fi
P1="${P1:-runs/phase1_${TAG}}"
OUT="${OUT:-runs/${TAG}}"
# SKIP_P1=1 reuses "$P1/best.pt" (or INIT) and jumps to phase 2.
SKIP_P1="${SKIP_P1:-0}"
INIT="${INIT:-}"
RESUME="${RESUME:-}"
# Resume optimizer moments/group LRs by default.  Set to 1 to retain the
# checkpoint model/AL/absolute step while starting a fresh optimizer.
RESET_OPTIMIZER="${RESET_OPTIMIZER:-0}"
SCENARIOS="${SCENARIOS:-80000}"
WORKERS="${WORKERS:-32}"
STEPS="${STEPS:-60000}"
# CPU training benefits from shorter actor updates; retain the production GPU
# batch unless the caller explicitly overrides BATCH.
if [[ -z "${BATCH+x}" ]]; then
  if [[ "$DEV" == "cpu" ]]; then
    BATCH=128
  else
    BATCH=512
  fi
fi
D_MODEL="${D_MODEL:-384}"
DEC_LAYERS="${DEC_LAYERS:-6}"
# Compile transformer solves only when requested.  LS probe replay mutates live
# params between compiled forwards.  CPU gradient balancing also uses retained
# output-space VJPs, which conflict with TorchInductor's donated buffers, so
# both paths default to eager execution.
if [[ "$LINE_SEARCH" == "1" || "$DEV" == "cpu" ]]; then
  COMPILE_MODEL="${COMPILE_MODEL:-0}"
else
  COMPILE_MODEL="${COMPILE_MODEL:-1}"
fi
COMPILE_FLAG=()
if [[ "$COMPILE_MODEL" != "1" ]]; then
  COMPILE_FLAG=(--no-compile)
fi
COMPILE_COST="${COMPILE_COST:-0}"
COMPILE_COST_FLAG=()
if [[ "$COMPILE_COST" == "1" ]]; then
  COMPILE_COST_FLAG=(--compile-cost)
fi
# The neural timing recipe uses the true scalar-objective gradient. Keep
# gradient surgery as an explicit opt-in; legacy non-LS recipes retain their
# prior trajectory-space PCGrad default. Line search also forces empty below.
if [[ -z "${GRAD_BALANCE+x}" ]]; then
  if [[ "$TIMING_CRITIC_MODE" == "assist" || "$TIMING_CRITIC_MODE" == "control" ]]; then
    GRAD_BALANCE=""
  else
    GRAD_BALANCE="al_pcgrad_output"
  fi
fi
GRAD_BALANCE_REFRESH="${GRAD_BALANCE_REFRESH:-4}"
GRAD_BALANCE_NORM_EMA="${GRAD_BALANCE_NORM_EMA:-0.9}"
# Extra scale on GradNorm-combined torque group (1=unchanged; 0.1≈tenth step).
TORQUE_GRAD_SCALE="${TORQUE_GRAD_SCALE:-1.0}"
# Cap ||g_torque||/||g|| after total.backward(); works with LINE_SEARCH=1.
TORQUE_GRAD_CAP="${TORQUE_GRAD_CAP:-0.7}"
# Replay sampling beta for high-torque-violation harvest rows (no loss scaling).
# A 5% uniform mixture keeps every eligible recovery row sampleable instead of
# letting exp(-beta*score) numerically censor the long-age tail.
REPLAY_VIOL_DOWNWEIGHT="${REPLAY_VIOL_DOWNWEIGHT:-0.05}"
REPLAY_VIOL_UNIFORM_MIX="${REPLAY_VIOL_UNIFORM_MIX:-0.05}"
# Replay: EMA proficiency stratification (prof/new/flex sampling).
REPLAY_PROFICIENCY="${REPLAY_PROFICIENCY:-0}"
REPLAY_FRAC_PROF="${REPLAY_FRAC_PROF:-0.40}"
REPLAY_FRAC_NEW="${REPLAY_FRAC_NEW:-0.40}"
PROFICIENCY_VIOL_CAP="${PROFICIENCY_VIOL_CAP:-8.0}"
REPLAY_PROFICIENCY_FLAGS=()
if [[ "$REPLAY_PROFICIENCY" != "0" ]]; then
  REPLAY_PROFICIENCY_FLAGS=(
    --replay-proficiency
    --replay-frac-prof "$REPLAY_FRAC_PROF"
    --replay-frac-new "$REPLAY_FRAC_NEW"
    --proficiency-viol-cap "$PROFICIENCY_VIOL_CAP"
  )
fi
# AL dual step on terminal region miss (default recipe 10; 30 = 3x faster λ).
RHO_TERMINAL="${RHO_TERMINAL:-30}"
GRAD_TERM_EVERY="${GRAD_TERM_EVERY:-250}"
TORQUE_CURVATURE_ROWS="${TORQUE_CURVATURE_ROWS:-8}"
LINE_SEARCH_C1="${LINE_SEARCH_C1:-0.5}"
LINE_SEARCH_BETA="${LINE_SEARCH_BETA:-0.99}"
LINE_SEARCH_MAX_TRIALS="${LINE_SEARCH_MAX_TRIALS:-6}"
LINE_SEARCH_INTERVAL="${LINE_SEARCH_INTERVAL:-adaptive}"
LINE_SEARCH_MODE="${LINE_SEARCH_MODE:-salsa_transfer}"
LINE_SEARCH_MOMENT_ON="${LINE_SEARCH_MOMENT_ON:-gradient}"
# Initial SaLSa ε (below --lr); selector expands on stable success, shrinks on reject.
LINE_SEARCH_EPSILON_SEED="${LINE_SEARCH_EPSILON_SEED:-1e-6}"
LINE_SEARCH_EPSILON_MIN="${LINE_SEARCH_EPSILON_MIN:-}"
LINE_SEARCH_EPSILON_MAX="${LINE_SEARCH_EPSILON_MAX:-}"
LINE_SEARCH_FRESH="${LINE_SEARCH_FRESH:-0}"
# Production SaLSa-transfer defaults from the completed ablation: select on
# the smoothed probe, always commit the finite staged Adam update, and let AL
# move without forcing a selector reset.  The stricter transactional checks
# remain opt-in diagnostics; enabling skip-on-exhaustion can otherwise freeze
# both parameters and Adam moments on a changing batch stream.
LINE_SEARCH_STRICT_ACTUAL_UPDATE="${LINE_SEARCH_STRICT_ACTUAL_UPDATE:-0}"
LINE_SEARCH_REJECT_CURRENT_PROBE_INCREASE="${LINE_SEARCH_REJECT_CURRENT_PROBE_INCREASE:-0}"
LINE_SEARCH_SKIP_ON_PROBE_EXHAUSTION="${LINE_SEARCH_SKIP_ON_PROBE_EXHAUSTION:-0}"
LINE_SEARCH_FORCE_AL_RESET="${LINE_SEARCH_FORCE_AL_RESET:-0}"
LINE_SEARCH_ACTUAL_DIAGNOSTIC_EVERY="${LINE_SEARCH_ACTUAL_DIAGNOSTIC_EVERY:-0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
LBFGS_HISTORY="${LBFGS_HISTORY:-10}"
OPTIMIZER_FLAGS=()
if [[ "$OPTIMIZER" != "adamw" ]]; then
  OPTIMIZER_FLAGS=(--optimizer "$OPTIMIZER" --lbfgs-history "$LBFGS_HISTORY")
fi
# Level-A torque curriculum (diag(M)+gravity -> full RNEA).  Defaults off.
TORQUE_PHYSICS="${TORQUE_PHYSICS:-full}"
TORQUE_FULL_START="${TORQUE_FULL_START:-}"
TORQUE_FULL_RAMP="${TORQUE_FULL_RAMP:-}"
TORQUE_PHYSICS_FLAGS=()
if [[ "$TORQUE_PHYSICS" != "full" ]]; then
  TORQUE_PHYSICS_FLAGS=(--torque-physics "$TORQUE_PHYSICS")
  if [[ -n "$TORQUE_FULL_START" ]]; then
    TORQUE_PHYSICS_FLAGS+=(--torque-full-start "$TORQUE_FULL_START")
  fi
  if [[ -n "$TORQUE_FULL_RAMP" ]]; then
    TORQUE_PHYSICS_FLAGS+=(--torque-full-ramp "$TORQUE_FULL_RAMP")
  fi
  echo "[run] torque_physics=$TORQUE_PHYSICS start=${TORQUE_FULL_START:-off} " \
       "ramp=${TORQUE_FULL_RAMP:-off}" >&2
fi
# Gravity-only static torque (RNEA(q,0,0)) with coupled torque AL off.
TORQUE_STATIC_ONLY="${TORQUE_STATIC_ONLY:-0}"
W_TORQUE_STATIC="${W_TORQUE_STATIC:-}"
TORQUE_STATIC_FLAGS=()
if [[ "$TORQUE_STATIC_ONLY" == "1" ]]; then
  W_TORQUE_STATIC="${W_TORQUE_STATIC:-20}"
  W_TORQUE="${W_TORQUE:-0}"
  W_TORQUE_PEAK="${W_TORQUE_PEAK:-0}"
  TORQUE_STATIC_AL="${TORQUE_STATIC_AL:-0}"
  if [[ "$TORQUE_STATIC_AL" != "1" ]]; then
    export AMORTIZED_LAM_TORQUE_MAX=0
    LAM_TORQUE_MAX=0
  fi
  TORQUE_STATIC_FLAGS=(
    --torque-static-only
    --w-torque-static "$W_TORQUE_STATIC"
    --w-torque "$W_TORQUE"
    --w-torque-peak "$W_TORQUE_PEAK"
  )
  if [[ "$TORQUE_STATIC_AL" == "1" ]]; then
    TORQUE_STATIC_FLAGS+=(--torque-static-al)
  fi
  STATIC_TORQUE_TIGHTEN="${STATIC_TORQUE_TIGHTEN:-}"
  if [[ -n "$STATIC_TORQUE_TIGHTEN" && "$STATIC_TORQUE_TIGHTEN" != "0" ]]; then
    TORQUE_STATIC_FLAGS+=(--static-torque-tighten "$STATIC_TORQUE_TIGHTEN")
  fi
  echo "[run] torque_static_only w_torque_static=$W_TORQUE_STATIC " \
       "w_torque=$W_TORQUE w_torque_peak=$W_TORQUE_PEAK " \
       "torque_static_al=$TORQUE_STATIC_AL " \
       "static_tighten=${STATIC_TORQUE_TIGHTEN:-off} " \
       "lam_torque_max=${LAM_TORQUE_MAX:-20}" >&2
fi
# Level-A / simplified torque hinge with torque AL off (not gravity-only static).
TORQUE_AL_OFF="${TORQUE_AL_OFF:-0}"
ETA_TORQUE="${ETA_TORQUE:-}"
TORQUE_SIMPLE_FLAGS=()
if [[ "$TORQUE_AL_OFF" == "1" ]]; then
  W_TORQUE="${W_TORQUE:-20}"
  W_TORQUE_PEAK="${W_TORQUE_PEAK:-0}"
  export AMORTIZED_LAM_TORQUE_MAX=0
  LAM_TORQUE_MAX=0
  TORQUE_SIMPLE_FLAGS=(
    --torque-al-off
    --w-torque "$W_TORQUE"
    --w-torque-peak "$W_TORQUE_PEAK"
  )
  if [[ -n "$ETA_TORQUE" ]]; then
    TORQUE_SIMPLE_FLAGS+=(--eta-torque "$ETA_TORQUE")
  fi
  echo "[run] torque_al_off physics=${TORQUE_PHYSICS} w_torque=$W_TORQUE " \
       "w_torque_peak=$W_TORQUE_PEAK eta_torque=${ETA_TORQUE:-default} " \
       "lam_torque_max=$AMORTIZED_LAM_TORQUE_MAX" >&2
fi
TORQUE_PEAK_FLAG=()
if [[ "$TORQUE_STATIC_ONLY" != "1" && "$TORQUE_AL_OFF" != "1" ]]; then
  TORQUE_PEAK_FLAG=(--w-torque-peak "${W_TORQUE_PEAK:-2.0}")
fi
LINE_SEARCH_FLAGS=()
# Use ``--grad-balance=VALUE`` so an empty surgery flag never eats the next argv.
GRAD_BALANCE_FLAG=(--grad-balance="$GRAD_BALANCE")
if [[ "$LINE_SEARCH" == "1" ]]; then
  LINE_SEARCH_FLAGS=(
    --line-search
    --line-search-c1 "$LINE_SEARCH_C1"
    --line-search-beta "$LINE_SEARCH_BETA"
    --line-search-max-trials "$LINE_SEARCH_MAX_TRIALS"
    --line-search-interval "$LINE_SEARCH_INTERVAL"
    --line-search-mode "$LINE_SEARCH_MODE"
    --line-search-moment-on "$LINE_SEARCH_MOMENT_ON"
    --line-search-epsilon-seed "$LINE_SEARCH_EPSILON_SEED"
    --line-search-actual-diagnostic-every "$LINE_SEARCH_ACTUAL_DIAGNOSTIC_EVERY"
  )
  if [[ -n "$LINE_SEARCH_EPSILON_MIN" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-epsilon-min "$LINE_SEARCH_EPSILON_MIN")
  fi
  if [[ -n "$LINE_SEARCH_EPSILON_MAX" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-epsilon-max "$LINE_SEARCH_EPSILON_MAX")
  fi
  if [[ "$LINE_SEARCH_FRESH" == "1" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-fresh)
  fi
  if [[ "$LINE_SEARCH_FORCE_AL_RESET" == "1" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-force-al-reset)
  fi
  if [[ "$LINE_SEARCH_STRICT_ACTUAL_UPDATE" == "1" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-strict-actual-update)
  fi
  if [[ "$LINE_SEARCH_REJECT_CURRENT_PROBE_INCREASE" == "1" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-reject-current-probe-increase)
  fi
  if [[ "$LINE_SEARCH_SKIP_ON_PROBE_EXHAUSTION" == "1" ]]; then
    LINE_SEARCH_FLAGS+=(--line-search-skip-on-probe-exhaustion)
  fi
  GRAD_BALANCE=""
  GRAD_BALANCE_FLAG=(--grad-balance=)
fi
if [[ "$VOXEL_POOL" == "1" ]]; then
  # Probe rows intentionally carry zero dual price.  The current gradient
  # surgery path is not row-mask aware, so voxel mode must leave it disabled.
  GRAD_BALANCE=""
  GRAD_BALANCE_FLAG=(--grad-balance=)
fi
# Cosine LR denominator. Wrapper default 30000 matches the 60k recipe (floor
# at step 30k). Under LS, cosine ownership is bypassed (selector owns ε / group
# LRs); LR still seeds ε. Override with LR_HORIZON=...
LR="${LR:-1e-4}"
LR_HORIZON="${LR_HORIZON:-30000}"
MIN_LR_FRAC="${MIN_LR_FRAC:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
PROBE_EVERY="${PROBE_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-50}"
# Keep BPTT onset at the same *absolute* step when STEPS is raised on a resume
# (0.6667 * 60000 = 40000).  Under LS MVP, default never starts (frac=2).
if [[ "$LINE_SEARCH" == "1" ]]; then
  BPTT_START_FRAC="${BPTT_START_FRAC:-2}"
  echo "[run] line_search=1 — grad_balance off, BPTT_START_FRAC=$BPTT_START_FRAC " \
       "(second-batch probe replay not wired yet)" >&2
else
  BPTT_START_FRAC="${BPTT_START_FRAC:-0.6667}"
fi
# Empty / 0 = no wall-clock stop.  Default 14h matches the old overnight runs.
MAX_HOURS="${MAX_HOURS:-14}"
# Empty = Phase2Config default (5 evals).  Set high to not early-stop on plateau.
PLATEAU_EVALS="${PLATEAU_EVALS:-}"

# Kit + rhythm randomizations BAKED INTO THE POOL.  The npz content
# fingerprint (scripts/train_phase2.py::_pool_fingerprint) refuses a mismatch
# on load; POOL_TAG still keys the path so different settings do not clobber
# each other on disk.  Defaults match the 19m tilt/radius production arms.
TILT_DEG="${TILT_DEG:-3}"
RADIUS_FRAC="${RADIUS_FRAC:-0.15}"
AIM_TAU="${AIM_TAU:-1.0}"
JIT_T="${JIT_T:-0.05}"
JIT_V="${JIT_V:-0.2}"
RARE_BOOST="${RARE_BOOST:-0.25}"

# evalbox = far family is the rolling-eval box grid (rest, one strike, no
# future), not ready+yaw+arm-noise with a required future hit.
POOL_TAG="${TAG}_mix306010_evalbox"
if [[ "$TILT_DEG" != "0" ]]; then POOL_TAG="${POOL_TAG}_tilt${TILT_DEG}"; fi
if [[ "$RADIUS_FRAC" != "0" ]]; then POOL_TAG="${POOL_TAG}_rad${RADIUS_FRAC}"; fi
if [[ "$JIT_T" != "0" || "$JIT_V" != "0" ]]; then POOL_TAG="${POOL_TAG}_jit${JIT_T}x${JIT_V}"; fi
if [[ "$RARE_BOOST" != "0" ]]; then POOL_TAG="${POOL_TAG}_rb${RARE_BOOST}"; fi
POOL_TAG="${POOL_TAG}_tau${AIM_TAU}"
POOL="${POOL:-runs/p2pool_${POOL_TAG}.npz}"

VOXEL_BOOTSTRAP_ROWS="${VOXEL_BOOTSTRAP_ROWS:-10000}"
VOXEL_BOOTSTRAP_BATCH="${VOXEL_BOOTSTRAP_BATCH:-256}"
VOXEL_BOOTSTRAP_STRIDE="${VOXEL_BOOTSTRAP_STRIDE:-4}"
VOXEL_POSTHIT_ANCHOR_TTH_S="${VOXEL_POSTHIT_ANCHOR_TTH_S:-1.0}"
VOXEL_REST_GAP_S="${VOXEL_REST_GAP_S:-1.0}"
VOXEL_CACHE="${VOXEL_CACHE:-}"
VOXEL_PROBE_CAPACITY="${VOXEL_PROBE_CAPACITY:-256}"
VOXEL_PROBE_REFRESH_EVERY="${VOXEL_PROBE_REFRESH_EVERY:-10}"
VOXEL_PROBE_SHORTER_LAYERS="${VOXEL_PROBE_SHORTER_LAYERS:-1}"
VOXEL_PROBE_LONGER_LAYERS="${VOXEL_PROBE_LONGER_LAYERS:-4}"
VOXEL_PROBE_TIMING_STEP="${VOXEL_PROBE_TIMING_STEP:-1}"
VOXEL_PROBE_TIMING_FRAC="${VOXEL_PROBE_TIMING_FRAC:-0.5}"
VOXEL_FLAGS=()
if [[ "$VOXEL_POOL" == "1" ]]; then
  VOXEL_FLAGS=(
    --voxel-pool
    --voxel-bootstrap-rows "$VOXEL_BOOTSTRAP_ROWS"
    --voxel-bootstrap-batch "$VOXEL_BOOTSTRAP_BATCH"
    --voxel-bootstrap-stride "$VOXEL_BOOTSTRAP_STRIDE"
    --voxel-posthit-anchor-tth-s "$VOXEL_POSTHIT_ANCHOR_TTH_S"
    --voxel-rest-gap-s "$VOXEL_REST_GAP_S"
    --voxel-probe-capacity "$VOXEL_PROBE_CAPACITY"
    --voxel-probe-refresh-every "$VOXEL_PROBE_REFRESH_EVERY"
    --voxel-probe-shorter-layers "$VOXEL_PROBE_SHORTER_LAYERS"
    --voxel-probe-longer-layers "$VOXEL_PROBE_LONGER_LAYERS"
    --voxel-probe-timing-step "$VOXEL_PROBE_TIMING_STEP"
    --voxel-probe-timing-frac "$VOXEL_PROBE_TIMING_FRAC"
    --voxel-frac-mpc 0.50
    --voxel-frac-pool 0.30
    --voxel-frac-boundary 0.15
    --voxel-frac-probe 0.05
  )
  if [[ -n "$VOXEL_CACHE" ]]; then
    VOXEL_FLAGS+=(--voxel-cache "$VOXEL_CACHE")
  fi
fi

# Knot RNEA vs rating (default OFF for the PD-demand FIR). Set these
# non-zero only to restore the old tracking-torque hinge + ok_torque.
W_TRACK_TORQUE="${W_TRACK_TORQUE:-0}"
W_TRACK_TORQUE_PEAK="${W_TRACK_TORQUE_PEAK:-0}"
# Joint SPEED limits (mean softplus + peak).  Default OFF: rarely active in
# practice and still pays for per-segment analytic |qd| max + AL groups.
# Restore with W_VELOCITY=40 W_VELOCITY_PEAK=2 (phase2_cost_config defaults).
W_VELOCITY="${W_VELOCITY:-0}"
W_VELOCITY_PEAK="${W_VELOCITY_PEAK:-0}"
# Scoring-only multiplier on wrist pitch/yaw effort in the RNEA hinge /
# ok_torque. 1 = Isaac 5 N·m. 2 was for FIR Kd spikes; leave at 1 now
# that the gate is inverse dynamics (RNEA on these joints stays << 5).
SOFT_WRIST_SCALE="${SOFT_WRIST_SCALE:-1}"
# Idle tip safety cylinder (full replace of joint-home park).  Default asset
# assets/idle_cylinder_poses.json; tip-outside hinge on all idle knots.
W_IDLE_CYLINDER="${W_IDLE_CYLINDER:-1.0}"
# Waist momentum well (rad/s)^-2 -- torso wind-up is the cheap torque loophole.
W_WAIST_YAW_VEL="${W_WAIST_YAW_VEL:-80}"
# Waist P/R rate well (rad/s)^-2 -- the lean complement of yaw rate.
W_WAIST_PR_VEL="${W_WAIST_PR_VEL:-80}"
# Decoder P/R band around stopgrad(ref) / upright.  Off by default: the
# tightened joint box (WAIST_PR_BOX) already caps lean; the soft band was
# contributing negligible gradient under GradNorm/PCGrad.
W_WAIST_PR_BAND="${W_WAIST_PR_BAND:-0}"
# Band half-width in rad (only matters when W_WAIST_PR_BAND > 0).
WAIST_PR_BAND_RAD="${WAIST_PR_BAND_RAD:-0.3}"
# Soft/hard joint-box half-width for waist pitch/roll (rad).  Yaw stays at
# the hardcoded +-80 deg guardrail inside CostModel._joint_box_hard_limits.
WAIST_PR_BOX="${WAIST_PR_BOX:-0.2}"
# Cheap temporal consistency on every sampled MPC replay row: output B is tied
# to its stored shifted-parent init.  Keep a small online TC slice as a current-
# policy, plant-differentiable reference while the replay approximation proves
# itself.
W_MPC_INIT_TIE="${W_MPC_INIT_TIE:-50}"
TC_FRAC="${TC_FRAC:-0.05}"
# Plain tip lateral-ordering hinge (non-AL); discourages crossed tips.
W_TIP_CROSSOVER="${W_TIP_CROSSOVER:-50}"
# 1 = snap suffix_warmup to 1.0 from step 0 (continuation only; cost weights
# are already full by default via WeightSchedule.snap_full).
FULL_WEIGHTS="${FULL_WEIGHTS:-0}"
FULL_WEIGHTS_FLAG=()
if [[ "$FULL_WEIGHTS" == "1" ]]; then
  FULL_WEIGHTS_FLAG=(--full-weights)
fi
# Frontier collocation (later generalization / continuation).  Default OFF.
# Seeded from the existing pool npz, not an empty start.  Pair with --resume
# last.pt and FULL_WEIGHTS=1 so ramps do not reopen.  Optional knobs:
# FRONTIER_PROPOSALS, FRONTIER_CHART_CAP, FRONTIER_FRESH_FRAC,
# FRONTIER_SPAWN_FRAC, FRONTIER_SIGMA_Q, FRONTIER_SIGMA_QD, FRONTIER_SIGMA_TTH,
# FRONTIER_EMA_ALPHA, FRONTIER_EMA_NMIN, FRONTIER_EMA_PROF,
# FRONTIER_LANDMARK_K, FRONTIER_LANDMARK_TAU, FRONTIER_PROPOSE_CANDIDATES,
# FRONTIER_HULL_EMA, FRONTIER_TRAIN_INV_FRAC, FRONTIER_TRAIN_INV_POOL,
# FRONTIER_SAMPLE_K, FRONTIER_SPLICE_K, FRONTIER_TRAJ_STRIDE,
# FRONTIER_INCLUDE_MISS=1,
# FRONTIER_IDLE_FAR_CLOUD=1 (idle/far = cloud only; keep live post-hit;
#   still splices FRONTIER_SPLICE_K cloud rows into replay).
FRONTIER="${FRONTIER:-0}"
if [[ "$VOXEL_POOL" == "1" && "$FRONTIER" == "1" ]]; then
  echo "VOXEL_POOL=1 and FRONTIER=1 are mutually exclusive" >&2
  exit 1
fi
FRONTIER_FLAG=()
if [[ "$FRONTIER" == "1" ]]; then
  FRONTIER_FLAG=(--frontier)
  if [[ -n "${FRONTIER_PROPOSALS:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-proposals "$FRONTIER_PROPOSALS")
  fi
  if [[ -n "${FRONTIER_CHART_CAP:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-chart-cap "$FRONTIER_CHART_CAP")
  fi
  if [[ -n "${FRONTIER_FRESH_FRAC:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-fresh-frac "$FRONTIER_FRESH_FRAC")
  fi
  if [[ -n "${FRONTIER_SPAWN_FRAC:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-spawn-frac "$FRONTIER_SPAWN_FRAC")
  fi
  if [[ -n "${FRONTIER_SIGMA_Q:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-sigma-q "$FRONTIER_SIGMA_Q")
  fi
  if [[ -n "${FRONTIER_SIGMA_QD:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-sigma-qd "$FRONTIER_SIGMA_QD")
  fi
  if [[ -n "${FRONTIER_SIGMA_TTH:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-sigma-tth "$FRONTIER_SIGMA_TTH")
  fi
  if [[ -n "${FRONTIER_EMA_ALPHA:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-ema-alpha "$FRONTIER_EMA_ALPHA")
  fi
  if [[ -n "${FRONTIER_EMA_NMIN:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-ema-nmin "$FRONTIER_EMA_NMIN")
  fi
  if [[ -n "${FRONTIER_EMA_PROF:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-ema-prof "$FRONTIER_EMA_PROF")
  fi
  if [[ -n "${FRONTIER_LANDMARK_K:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-landmark-k "$FRONTIER_LANDMARK_K")
  fi
  if [[ -n "${FRONTIER_LANDMARK_TAU:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-landmark-tau "$FRONTIER_LANDMARK_TAU")
  fi
  if [[ -n "${FRONTIER_PROPOSE_CANDIDATES:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-propose-candidates "$FRONTIER_PROPOSE_CANDIDATES")
  fi
  if [[ -n "${FRONTIER_HULL_EMA:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-hull-ema "$FRONTIER_HULL_EMA")
  fi
  if [[ -n "${FRONTIER_TRAIN_INV_FRAC:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-train-inv-frac "$FRONTIER_TRAIN_INV_FRAC")
  fi
  if [[ -n "${FRONTIER_TRAIN_INV_POOL:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-train-inv-pool "$FRONTIER_TRAIN_INV_POOL")
  fi
  if [[ -n "${FRONTIER_SAMPLE_K:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-sample-k "$FRONTIER_SAMPLE_K")
  fi
  if [[ -n "${FRONTIER_SPLICE_K:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-splice-k "$FRONTIER_SPLICE_K")
  fi
  if [[ -n "${FRONTIER_TRAJ_STRIDE:-}" ]]; then
    FRONTIER_FLAG+=(--frontier-traj-stride "$FRONTIER_TRAJ_STRIDE")
  fi
  if [[ "${FRONTIER_INCLUDE_MISS:-0}" == "1" ]]; then
    FRONTIER_FLAG+=(--frontier-include-miss)
  fi
  if [[ "${FRONTIER_IDLE_FAR_CLOUD:-0}" == "1" ]]; then
    FRONTIER_FLAG+=(--frontier-idle-far-cloud)
  fi
fi
# Auto-frontier: trainer enables collocation when probe miss drops (mm).
# ``scripts/frontier_auto_monitor.py`` polls metrics every 30 min and writes
# ``<out>/.frontier_activate`` as a backup trigger.
FRONTIER_AUTO_MISS_MM="${FRONTIER_AUTO_MISS_MM:-0}"
FRONTIER_AUTO_FLAG=()
if [[ -n "${FRONTIER_AUTO_MISS_MM}" && "${FRONTIER_AUTO_MISS_MM}" != "0" ]]; then
  FRONTIER_AUTO_FLAG=(--frontier-auto-miss-mm "$FRONTIER_AUTO_MISS_MM")
  if [[ -n "${FRONTIER_AUTO_VIOL_CAP:-}" ]]; then
    FRONTIER_AUTO_FLAG+=(--frontier-auto-viol-cap "$FRONTIER_AUTO_VIOL_CAP")
  fi
  if [[ -n "${FRONTIER_AUTO_VIOL_RATIO:-}" ]]; then
    FRONTIER_AUTO_FLAG+=(--frontier-auto-viol-ratio "$FRONTIER_AUTO_VIOL_RATIO")
  fi
  if [[ -n "${FRONTIER_AUTO_VIOL_WINDOW:-}" ]]; then
    FRONTIER_AUTO_FLAG+=(--frontier-auto-viol-window "$FRONTIER_AUTO_VIOL_WINDOW")
  fi
  if [[ "${FRONTIER_AUTO_FULL_WEIGHTS:-1}" == "0" ]]; then
    FRONTIER_AUTO_FLAG+=(--no-frontier-auto-full-weights)
  fi
fi
# Learned waist pitch/roll set-point (encoder residual on upright).  0 = off
# (decoder owns P/R; dense well vs 0 + V_f at the nearer hit).  Default recipe
# keeps both yaw and P/R encoder heads OFF — direct decoder movements only.
# WAIST_PR_EMA=1 harvests a leaky EMA of the clipped cmd into waist_pr_ref
# (walking well).  Conflicts with WAIST_PR_SP and FREEZE_WAIST_PR.
# FREEZE_WAIST_PR=1 pins decoder pitch/roll to the ready-pose ramp (yaw stays
# free) and forces WAIST_PR_SP=0 -- the two flags conflict in train_phase2.
WAIST_PR_SP="${WAIST_PR_SP:-0}"
WAIST_PR_EMA="${WAIST_PR_EMA:-0}"
FREEZE_WAIST_PR="${FREEZE_WAIST_PR:-0}"
W_WAIST_PR_TERMINAL="${W_WAIST_PR_TERMINAL:-200}"
if [[ "$FREEZE_WAIST_PR" == "1" && "$WAIST_PR_SP" == "1" ]]; then
  echo "[run] FREEZE_WAIST_PR=1 forces WAIST_PR_SP=0" >&2
  WAIST_PR_SP=0
fi
if [[ "$WAIST_PR_EMA" == "1" && "$WAIST_PR_SP" == "1" ]]; then
  echo "[run] WAIST_PR_EMA and WAIST_PR_SP both on -- pick one" >&2
  exit 1
fi
if [[ "$WAIST_PR_EMA" == "1" && "$FREEZE_WAIST_PR" == "1" ]]; then
  echo "[run] WAIST_PR_EMA and FREEZE_WAIST_PR conflict" >&2
  exit 1
fi
if [[ "$VOXEL_POOL" == "1" && ( "$WAIST_PR_SP" == "1" || "$WAIST_PR_EMA" == "1" ) ]]; then
  echo "VOXEL_POOL=1 forbids encoder waist SP heads and waist reference EMA" >&2
  exit 1
fi
WAIST_PR_FLAG=()
if [[ "$WAIST_PR_SP" == "1" ]]; then
  WAIST_PR_FLAG=(--waist-pr-sp-head)
fi
if [[ "$WAIST_PR_EMA" == "1" ]]; then
  WAIST_PR_FLAG+=(--waist-pr-ema)
  if [[ -n "${WAIST_PR_EMA_ALPHA:-}" ]]; then
    WAIST_PR_FLAG+=(--waist-pr-ema-alpha "$WAIST_PR_EMA_ALPHA")
  fi
  if [[ -n "${WAIST_PR_EMA_LEAK:-}" ]]; then
    WAIST_PR_FLAG+=(--waist-pr-ema-leak "$WAIST_PR_EMA_LEAK")
  fi
fi
if [[ "$FREEZE_WAIST_PR" == "1" ]]; then
  WAIST_PR_FLAG+=(--freeze-waist-pr)
fi
if [[ -n "${W_WAIST_PR_TERMINAL}" ]]; then
  WAIST_PR_FLAG+=(--w-waist-pr-terminal "$W_WAIST_PR_TERMINAL")
fi
# Decoder P/R band cost weight + radius.  In the direct recipe (no SP head)
# the ref is treated as upright (0), so this constrains lean to +-WAIST_PR_BAND_RAD.
if [[ -n "${W_WAIST_PR_BAND:-}" ]]; then
  WAIST_PR_FLAG+=(--w-waist-pr-band "$W_WAIST_PR_BAND")
fi
if [[ -n "${WAIST_PR_BAND_RAD:-}" ]]; then
  WAIST_PR_FLAG+=(--waist-pr-band-rad "$WAIST_PR_BAND_RAD")
fi
# Gated strike-knot P/R distill onto the encoder.  Empty = off (0).
if [[ -n "${W_WAIST_PR_DISTILL:-}" ]]; then
  WAIST_PR_FLAG+=(--w-waist-pr-sp-distill "$W_WAIST_PR_DISTILL")
fi
if [[ -n "${PR_DISTILL_HIT_M:-}" ]]; then
  WAIST_PR_FLAG+=(--pr-distill-hit-m "$PR_DISTILL_HIT_M")
fi
# Encoder hinge |sp - clamp(q0)| ≤ decoder band.  Empty = train_phase2
# default (10 when the P/R head is on).
if [[ -n "${W_WAIST_PR_HOLD_BAND:-}" ]]; then
  WAIST_PR_FLAG+=(--w-waist-pr-sp-hold-band "$W_WAIST_PR_HOLD_BAND")
fi
CKPT_EVERY="${CKPT_EVERY:-2000}"
CKPT_LAST_EVERY="${CKPT_LAST_EVERY:-500}"
CKPT_KEEP_EVERY="${CKPT_KEEP_EVERY:-500}"
# Rotary position on decoder self-attention (no extra params).  1 = on.
DEC_ROPE="${DEC_ROPE:-1}"
DEC_ROPE_FLAG=()
if [[ "$DEC_ROPE" == "1" ]]; then
  DEC_ROPE_FLAG=(--dec-rope)
fi
# Analytic Cartesian + knot-RNEA of init_traj on decoder tokens.  1 = on.
DEC_PLANT_FEATS="${DEC_PLANT_FEATS:-1}"
DEC_PLANT_FLAG=()
if [[ "$DEC_PLANT_FEATS" == "1" ]]; then
  DEC_PLANT_FLAG=(--dec-plant-feats)
fi
# Explicit-fp32 trajectory readout while transformer/aux heads retain AMP.
# Full-RNEA replay probes measured a 65.5% -> 0.239% torque-gradient error
# reduction; set 0 only for the legacy all-autocast A/B arm.
TRAJ_HEAD_FP32="${TRAJ_HEAD_FP32:-1}"
TRAJ_HEAD_FP32_FLAG=()
if [[ "$TRAJ_HEAD_FP32" == "1" ]]; then
  TRAJ_HEAD_FP32_FLAG=(--traj-head-fp32)
fi

MAX_HOURS_FLAG=()
if [[ -n "$MAX_HOURS" && "$MAX_HOURS" != "0" ]]; then
  MAX_HOURS_FLAG=(--max-hours "$MAX_HOURS")
fi
PLATEAU_FLAG=()
if [[ -n "$PLATEAU_EVALS" ]]; then
  PLATEAU_FLAG=(--plateau-evals "$PLATEAU_EVALS")
fi
if [[ "$RESET_OPTIMIZER" == "1" && -z "$RESUME" ]]; then
  echo "RESET_OPTIMIZER=1 requires RESUME=<phase2 checkpoint>" >&2
  exit 1
fi
if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "missing resume ckpt: $RESUME" >&2
    exit 1
  fi
  LOAD_FLAGS=(--resume "$RESUME")
  if [[ "$RESET_OPTIMIZER" == "1" ]]; then
    LOAD_FLAGS+=(--reset-optimizer)
  fi
  SKIP_P1=1
elif [[ "$SKIP_P1" == "1" ]]; then
  INIT="${INIT:-$P1/best.pt}"
  if [[ ! -f "$INIT" ]]; then
    echo "missing phase-1 warm start: $INIT" >&2
    exit 1
  fi
  LOAD_FLAGS=(--init "$INIT")
else
  LOAD_FLAGS=(--init "$P1/best.pt")
fi

echo "[run] DEV=$DEV SKIP_P1=$SKIP_P1 P1=$P1 INIT=${INIT:-$P1/best.pt} RESUME=${RESUME:-} OUT=$OUT POOL=$POOL BATCH=$BATCH D_MODEL=$D_MODEL DEC_LAYERS=$DEC_LAYERS VOXEL_POOL=$VOXEL_POOL" >&2
echo "[run] resume controls: reset_optimizer=${RESET_OPTIMIZER} " \
     "line_search_fresh=${LINE_SEARCH_FRESH} load_flags=${LOAD_FLAGS[*]}" >&2
echo "[run] kit DR: tilt=${TILT_DEG}deg radius_frac=${RADIUS_FRAC} " \
     "jit_t=${JIT_T}s jit_v=${JIT_V} rare_boost=${RARE_BOOST} tau=${AIM_TAU}" >&2
echo "[run] soft_wrist_effort_scale=${SOFT_WRIST_SCALE} w_idle_cylinder=${W_IDLE_CYLINDER} " \
     "w_tip_crossover=${W_TIP_CROSSOVER}" >&2
echo "[run] torque gate: knot RNEA  w_track_torque=${W_TRACK_TORQUE} " \
     "w_track_torque_peak=${W_TRACK_TORQUE_PEAK}" >&2
echo "[run] velocity limits: w_velocity=${W_VELOCITY} w_velocity_peak=${W_VELOCITY_PEAK}" >&2
echo "[run] line-search safety: strict_actual=${LINE_SEARCH_STRICT_ACTUAL_UPDATE} " \
     "reject_probe_increase=${LINE_SEARCH_REJECT_CURRENT_PROBE_INCREASE} " \
     "skip_probe_exhaustion=${LINE_SEARCH_SKIP_ON_PROBE_EXHAUSTION} " \
     "force_al_reset=${LINE_SEARCH_FORCE_AL_RESET} " \
     "actual_diagnostic_every=${LINE_SEARCH_ACTUAL_DIAGNOSTIC_EVERY}" >&2
if [[ "$LINE_SEARCH" == "1" ]]; then
  printf '[run] line-search CLI:' >&2
  printf ' %q' "${LINE_SEARCH_FLAGS[@]}" >&2
  printf '\n' >&2
else
  echo "[run] line-search CLI: (disabled; LINE_SEARCH=0 arm preserved)" >&2
fi
echo "[run] lam_torque_max=${LAM_TORQUE_MAX} lam_torque_waist_max=${LAM_TORQUE_WAIST_MAX:-(same)} " \
     "waist_upright_ease_wp=${WAIST_UPRIGHT_EASE_WP}" >&2
echo "[run] full_weights=${FULL_WEIGHTS} dec_rope=${DEC_ROPE} " \
     "dec_plant_feats=${DEC_PLANT_FEATS} traj_head_fp32=${TRAJ_HEAD_FP32} compile_model=${COMPILE_MODEL} compile_cost=${COMPILE_COST} grad_balance=${GRAD_BALANCE:-(off)} refresh=${GRAD_BALANCE_REFRESH} norm_ema=${GRAD_BALANCE_NORM_EMA} torque_grad_scale=${TORQUE_GRAD_SCALE} torque_grad_cap=${TORQUE_GRAD_CAP} replay_viol_sampling_beta=${REPLAY_VIOL_DOWNWEIGHT} replay_viol_uniform_mix=${REPLAY_VIOL_UNIFORM_MIX} replay_prof=${REPLAY_PROFICIENCY} replay_frac_prof=${REPLAY_FRAC_PROF} replay_frac_new=${REPLAY_FRAC_NEW} prof_viol_cap=${PROFICIENCY_VIOL_CAP} rho_terminal=${RHO_TERMINAL} grad_term_every=${GRAD_TERM_EVERY} torque_curve_rows=${TORQUE_CURVATURE_ROWS} line_search=${LINE_SEARCH} waist_pr_sp=${WAIST_PR_SP} " \
     "waist_pr_ema=${WAIST_PR_EMA} freeze_waist_pr=${FREEZE_WAIST_PR} " \
     "w_waist_pr_terminal=${W_WAIST_PR_TERMINAL} " \
     "w_waist_pr_band=${W_WAIST_PR_BAND:-default} waist_pr_band_rad=${WAIST_PR_BAND_RAD:-0.15} " \
     "waist_pr_box=${WAIST_PR_BOX} " \
     "w_waist_yaw_vel=${W_WAIST_YAW_VEL} " \
     "w_waist_pr_vel=${W_WAIST_PR_VEL} " \
     "w_mpc_init_tie=${W_MPC_INIT_TIE} tc_frac=${TC_FRAC} " \
     "w_waist_pr_distill=${W_WAIST_PR_DISTILL:-0} " \
     "w_waist_pr_hold_band=${W_WAIST_PR_HOLD_BAND:-default} " \
     "frontier=${FRONTIER} timing_critic=${TIMING_CRITIC_MODE} steps=$STEPS lr=$LR lr_horizon=$LR_HORIZON " \
     "warmup=${WARMUP_STEPS} min_lr_frac=${MIN_LR_FRAC} log_every=${LOG_EVERY} eval_every=${EVAL_EVERY} probe_every=${PROBE_EVERY} " \
     "max_hours=${MAX_HOURS:-off} bptt_start_frac=$BPTT_START_FRAC " \
     "plateau=${PLATEAU_EVALS:-default}" >&2
if [[ "$VOXEL_POOL" == "1" ]]; then
  echo "[run] voxel timing stencil: -${VOXEL_PROBE_SHORTER_LAYERS}/+${VOXEL_PROBE_LONGER_LAYERS} layers x ${VOXEL_PROBE_TIMING_STEP} knot(s), fresh timing fraction=${VOXEL_PROBE_TIMING_FRAC}" >&2
fi
echo "[run] (xy/z kit jitter is always on via KitRandomization defaults)" >&2

POOL_GEN_PID=""
if [[ ! -f "$POOL" ]]; then
  echo "=== generating rhythm pool -> $POOL (background) ===" >&2
  python scripts/train_phase2.py \
      --gen-pool-only \
      --pool-cache "$POOL" \
      --scenarios "$SCENARIOS" \
      --workers "$WORKERS" \
      --horizon 50 --buckets 25 50 \
      --rhythm-idle-frac 0.30 \
      --far-fetched-frac 0.10 \
      --post-hit-frac 0.60 \
      --eval-like-frac 0 \
      --num-future 1 \
      --hit-time-jitter-s "$JIT_T" \
      --hit-loudness-jitter "$JIT_V" \
      --rare-drum-boost "$RARE_BOOST" \
      --drum-tilt-deg "$TILT_DEG" \
      --drum-radius-frac "$RADIUS_FRAC" \
      --synth-transition-frac 0.25 \
      --lead-s 1.0 \
      --waist-aim-future-tau "$AIM_TAU" \
      "${WAIST_EASE_FLAG[@]}" \
      > "runs/poolgen_${TAG}.log" 2>&1 &
  POOL_GEN_PID=$!
  echo "[run] pool gen pid=$POOL_GEN_PID" >&2
else
  echo "[run] reusing pool $POOL" >&2
fi

if [[ "$SKIP_P1" != "1" ]]; then
  echo "=== [1/2] phase 1 apex warm start -> $P1 ===" >&2
  P1_HEAD_FLAGS=()
  if [[ "$WAIST_PR_SP" == "1" ]]; then
    P1_HEAD_FLAGS+=(--waist-pr-sp-head)
  fi
  python scripts/train_phase1_apex.py \
      --device "$DEV" \
      --d-model "$D_MODEL" \
      --dec-layers "$DEC_LAYERS" \
      "${DEC_ROPE_FLAG[@]}" \
      "${DEC_PLANT_FLAG[@]}" \
      --steps 20000 \
      --batch 1024 \
      --horizon 50 --buckets 25 50 \
      --train-scenarios 500000 --val-scenarios 40000 \
      --workers 48 \
      --init-mode mix --init-mix 0.5 \
      "${P1_HEAD_FLAGS[@]}" \
      "${WAIST_EASE_FLAG[@]}" \
      --waist-aim-future-tau "$AIM_TAU" \
      --hit-time-jitter-s "$JIT_T" \
      --hit-loudness-jitter "$JIT_V" \
      --out "$P1"
  test -f "$P1/best.pt" || { echo "phase 1 did not write $P1/best.pt" >&2; exit 1; }
fi

if [[ -n "$POOL_GEN_PID" ]]; then
  echo "=== waiting for pool gen (pid $POOL_GEN_PID) ===" >&2
  wait "$POOL_GEN_PID" || {
    echo "pool gen FAILED (see runs/poolgen_${TAG}.log); phase 2 will regenerate inline" >&2
  }
  if [[ -f "$POOL" ]]; then
    echo "pool done ($(wc -c < "$POOL") bytes)" >&2
  fi
fi

echo "=== [2/2] train_phase2 ($TAG) on $DEV ===" >&2
# Direct-movement torso: decoder upright + rate wells + V_f; no encoder SP heads.
# Cost weights full from step 0 (AL adjusts λ).  BPTT/suffix still phased in.
# Fingerprint on the npz also guards kit/mix/seed; keep gen flags identical
# so a hit reuses the cache instead of regenerating.
python scripts/train_phase2.py \
    --device "$DEV" \
    "${LOAD_FLAGS[@]}" \
    --out "$OUT" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --d-model "$D_MODEL" \
    --dec-layers "$DEC_LAYERS" \
    --lr "$LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --min-lr-frac "$MIN_LR_FRAC" \
    --lr-horizon "$LR_HORIZON" \
    --horizon 50 --buckets 25 50 \
    --pool-cache "$POOL" \
    --scenarios "$SCENARIOS" \
    --rhythm-idle-frac 0.30 \
    --far-fetched-frac 0.10 \
    --post-hit-frac 0.60 \
    --eval-like-frac 0 \
    --num-future 1 \
    "${VOXEL_FLAGS[@]}" \
    "${TIMING_CRITIC_FLAGS[@]}" \
    --hit-time-jitter-s "$JIT_T" \
    --hit-loudness-jitter "$JIT_V" \
    --rare-drum-boost "$RARE_BOOST" \
    --drum-tilt-deg "$TILT_DEG" \
    --drum-radius-frac "$RADIUS_FRAC" \
    "${COMPILE_FLAG[@]}" \
    "${COMPILE_COST_FLAG[@]}" \
    --clamp-traj-to-box \
    "${GRAD_BALANCE_FLAG[@]}" \
    --grad-balance-refresh "$GRAD_BALANCE_REFRESH" \
    --grad-balance-norm-ema "$GRAD_BALANCE_NORM_EMA" \
    --grad-balance-tau-torque-cap 0.30 \
    --grad-balance-waist-floor 2.0 \
    --torque-grad-scale "$TORQUE_GRAD_SCALE" \
    --torque-grad-cap "$TORQUE_GRAD_CAP" \
    --replay-viol-downweight "$REPLAY_VIOL_DOWNWEIGHT" \
    --replay-viol-uniform-mix "$REPLAY_VIOL_UNIFORM_MIX" \
    "${REPLAY_PROFICIENCY_FLAGS[@]}" \
    --rho-terminal "$RHO_TERMINAL" \
    --grad-term-every "$GRAD_TERM_EVERY" \
    --torque-curvature-rows "$TORQUE_CURVATURE_ROWS" \
    "${LINE_SEARCH_FLAGS[@]}" \
    "${OPTIMIZER_FLAGS[@]}" \
    "${TORQUE_PHYSICS_FLAGS[@]}" \
    "${TORQUE_STATIC_FLAGS[@]}" \
    "${TORQUE_SIMPLE_FLAGS[@]}" \
    "${TORQUE_PEAK_FLAG[@]}" \
    --w-velocity "$W_VELOCITY" \
    --w-velocity-peak "$W_VELOCITY_PEAK" \
    --w-track-torque "$W_TRACK_TORQUE" \
    --w-track-torque-peak "$W_TRACK_TORQUE_PEAK" \
    --soft-wrist-effort-scale "$SOFT_WRIST_SCALE" \
    --w-idle-cylinder "$W_IDLE_CYLINDER" \
    --w-collision-peak 2000 \
    --w-joint-box-peak 40 \
    --w-stick-stick 100 \
    --w-tip-crossover "$W_TIP_CROSSOVER" \
    --w-mpc-init-tie "$W_MPC_INIT_TIE" --tc-frac "$TC_FRAC" \
    --suffix-frac 0.25 \
    --w-suffix-cost 1.0 --w-suffix-tie 50.0 --suffix-warmup 1000 \
    --w-waist-aim 0 \
    "${WAIST_PR_FLAG[@]}" \
    --waist-aim-future-tau "$AIM_TAU" \
    --w-waist-upright 100 --waist-pr-box "$WAIST_PR_BOX" --w-waist-yaw-vel "$W_WAIST_YAW_VEL" \
    --w-waist-pr-vel "$W_WAIST_PR_VEL" \
    "${WAIST_EASE_FLAG[@]}" \
    "${FULL_WEIGHTS_FLAG[@]}" \
    "${FRONTIER_FLAG[@]}" \
    "${FRONTIER_AUTO_FLAG[@]}" \
    "${DEC_ROPE_FLAG[@]}" \
    "${DEC_PLANT_FLAG[@]}" \
    "${TRAJ_HEAD_FP32_FLAG[@]}" \
    --bptt-start-frac "$BPTT_START_FRAC" \
    --cmd-lead fit_kd4 \
    --log-every "$LOG_EVERY" \
    --eval-every "$EVAL_EVERY" --eval-cases 81 --ckpt-every "$CKPT_EVERY" \
    --ckpt-last-every "$CKPT_LAST_EVERY" --ckpt-keep-every "$CKPT_KEEP_EVERY" \
    --probe-every "$PROBE_EVERY" --probe-cmd-lead fit_kd4 \
    --target-acceptance 0.95 \
    "${PLATEAU_FLAG[@]}" \
    "${MAX_HOURS_FLAG[@]}"

echo "=== done ==="
echo "phase 1: $P1"
echo "phase 2: $OUT"
echo "pool: $POOL"
