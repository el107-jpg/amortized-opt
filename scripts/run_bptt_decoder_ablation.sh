#!/usr/bin/env bash
set -euo pipefail
cd /data2/eric/amortized-opt

if [[ $# -lt 4 ]]; then
  echo "usage: $0 PASSES DEVICE OUT FINAL_STEP [REGISTER=1]" >&2
  exit 2
fi

PASSES="$1"
DEVICE="$2"
RUN_OUT="$3"
FINAL_STEP="$4"
REGISTER="${5:-1}"

export AMORTIZED_CONDA_ENV=/data2/eric/conda_envs/amortized_compile
export DEV="$DEVICE"
export SKIP_P1=1
unset INIT
export RESUME="${BPTT_RESUME:-runs/phase2_terminal_critic_4p69m_initnoise_c1p01_floor1e7cos_lsafter10k_parentpool_live_20260928/step_028000.pt}"
export OUT="$RUN_OUT"
export TAG="$(basename "$RUN_OUT")"
export POOL=runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz

export D_MODEL=336
export DEC_LAYERS=2
export TIMING_CRITIC_MODE=control
export VOXEL_POOL=0
export TERMINAL_INIT_NOISE=1
export TC_FRAC=0
export TIMING_MPC_Q_NOISE=0.01
export TIMING_MPC_QD_NOISE=0.05
export TIMING_MPC_PARENT_CAPACITY=2048

export LINE_SEARCH=0
export RESET_OPTIMIZER="${BPTT_RESET_OPTIMIZER:-1}"
export DECODER_ONLY=1
export BPTT_START_FRAC=0
export BPTT_MAX_FRAC=1
export BPTT_BATCH="${BPTT_BATCH:-32}"
export BPTT_DECODER_PASSES="$PASSES"
export W_BPTT=1

export LR="${BPTT_LR:-1e-7}"
export LR_HORIZON=60000
export MIN_LR_FRAC=0.05
export WARMUP_STEPS=300
export STEPS="$FINAL_STEP"
export MAX_HOURS="${MAX_HOURS:-12}"
export LOG_EVERY="${LOG_EVERY:-25}"
export EVAL_EVERY="${EVAL_EVERY:-500}"
export PROBE_EVERY="${PROBE_EVERY:-500}"
export TB_AUTO_REGISTER="$REGISTER"

exec bash scripts/_run_rhythm_bellman.sh
