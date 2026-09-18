#!/usr/bin/env bash
# Adam + SaLSa Armijo: fresh phase-2 from phase-1 init (no resume).
#
# Usage:
#   DEV=cuda:2 bash scripts/_run_rhythm_bellman_adam_armijo.sh
# Resume a prior fresh run:
#   RESUME=runs/rhythm_bellman_nf1_adam_waist_fresh/last.pt TAG=... bash ...
set -euo pipefail

export SKIP_P1=1
export LINE_SEARCH=1
export LINE_SEARCH_MODE=salsa_transfer
export OPTIMIZER=adamw
export TAG="${TAG:-rhythm_bellman_nf1_adam_waist_fresh}"
export INIT="${INIT:-runs/phase1_rhythm_bellman_nf1_prsp/best.pt}"
export RESUME="${RESUME:-}"
export STEPS="${STEPS:-60000}"
export PROBE_EVERY="${PROBE_EVERY:-1000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export POOL="${POOL:-runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz}"
export LINE_SEARCH_EPSILON_SEED="${LINE_SEARCH_EPSILON_SEED:-1e-6}"
# Fresh run: new SaLSa ε/selector (set LINE_SEARCH_FRESH=0 when RESUME is set).
export LINE_SEARCH_FRESH="${LINE_SEARCH_FRESH:-$([[ -n "$RESUME" ]] && echo 0 || echo 1)}"
export LAM_TORQUE_WAIST_MAX="${LAM_TORQUE_WAIST_MAX:-100}"
export WAIST_UPRIGHT_EASE_WP="${WAIST_UPRIGHT_EASE_WP:-1}"
export FRONTIER_AUTO_MISS_MM="${FRONTIER_AUTO_MISS_MM:-30}"

echo "[adam_armijo] tag=${TAG} dev=${DEV:-cuda:1} init=${INIT} resume=${RESUME:-(none)} " \
     "ls_fresh=${LINE_SEARCH_FRESH} ε_seed=${LINE_SEARCH_EPSILON_SEED} " \
     "lam_torque_waist_max=${LAM_TORQUE_WAIST_MAX} waist_ease_wp=${WAIST_UPRIGHT_EASE_WP}" >&2
exec bash scripts/_run_rhythm_bellman.sh
