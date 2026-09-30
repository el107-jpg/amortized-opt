#!/usr/bin/env bash
# Vetted architecture-search adapter for the production Phase-2 physics recipe.
set -euo pipefail

DEVICE=""
OUT=""
INIT=""
RESUME=""
SEED=0
STEPS=10000
BATCH=512
D_MODEL=""
DEC_LAYERS=""
D_CTX=""
CTX_LAYERS=""

while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --dec-layers) DEC_LAYERS="$2"; shift 2 ;;
    --d-ctx) D_CTX="$2"; shift 2 ;;
    --ctx-layers) CTX_LAYERS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in DEVICE OUT INIT D_MODEL DEC_LAYERS D_CTX CTX_LAYERS; do
  if [[ -z "${!value}" ]]; then
    echo "missing required --${value,,}" >&2
    exit 2
  fi
done

# train_phase2.py expects a torch device string.  Accept a bare GPU index at
# this adapter boundary because architecture-search schedulers commonly assign
# devices as integers.
if [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
  DEVICE="cuda:$DEVICE"
fi
if [[ ! -f "$INIT" && -z "$RESUME" ]]; then
  echo "missing Phase-1 checkpoint: $INIT" >&2
  exit 2
fi

# d_ctx/ctx_layers are reconstructed from the Phase-1 checkpoint.  They are
# accepted here so the search manifest records the complete intended shape.
echo "[arch-trial] device=$DEVICE out=$OUT init=$INIT resume=${RESUME:-none}" >&2
echo "[arch-trial] encoder=${CTX_LAYERS}x${D_CTX} decoder=${DEC_LAYERS}x${D_MODEL}" >&2

LOAD_ENV=(SKIP_P1=1 INIT="$INIT")
if [[ -n "$RESUME" ]]; then
  LOAD_ENV=(SKIP_P1=1 RESUME="$RESUME")
fi

exec env \
  DEV="$DEVICE" \
  OUT="$OUT" \
  TAG="$(basename "$OUT")" \
  STEPS="$STEPS" \
  BATCH="$BATCH" \
  D_MODEL="$D_MODEL" \
  DEC_LAYERS="$DEC_LAYERS" \
  SEED="$SEED" \
  POOL="runs/p2pool_diagnostic_safe_uniform_300_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz" \
  VOXEL_POOL=0 \
  LINE_SEARCH=1 \
  TIMING_CRITIC_MODE=assist \
  TIMING_CRITIC_UPDATES=1 \
  LR_HORIZON=30000 \
  EVAL_EVERY=2000 \
  PROBE_EVERY=2000 \
  MAX_HOURS=20 \
  "${LOAD_ENV[@]}" \
  bash scripts/_run_rhythm_bellman.sh
