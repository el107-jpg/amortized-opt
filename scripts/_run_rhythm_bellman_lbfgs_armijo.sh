#!/usr/bin/env bash
# L-BFGS + SaLSa Armijo line search (100× ε scale for O(1) directions).
#
# Usage:
#   DEV=cuda:1 bash scripts/_run_rhythm_bellman_lbfgs_armijo.sh
set -euo pipefail

export SKIP_P1=1
export LINE_SEARCH=1
export LINE_SEARCH_MODE=salsa_transfer
export LINE_SEARCH_FRESH=1
export OPTIMIZER=lbfgs
export LBFGS_HISTORY="${LBFGS_HISTORY:-10}"
export TAG="${TAG:-rhythm_bellman_nf1_lbfgs_armijo}"
export RESUME="${RESUME:-runs/rhythm_bellman_nf1_ls_c1_05/last.pt}"
export STEPS="${STEPS:-38200}"
export PROBE_EVERY="${PROBE_EVERY:-1000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export POOL="${POOL:-runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz}"
export LINE_SEARCH_EPSILON_SEED="${LINE_SEARCH_EPSILON_SEED:-1e-4}"
export LINE_SEARCH_EPSILON_MIN="${LINE_SEARCH_EPSILON_MIN:-1e-8}"
export LINE_SEARCH_EPSILON_MAX="${LINE_SEARCH_EPSILON_MAX:-1e-3}"

echo "[lbfgs_armijo] ε_seed=${LINE_SEARCH_EPSILON_SEED} bounds=[${LINE_SEARCH_EPSILON_MIN},${LINE_SEARCH_EPSILON_MAX}] tag=${TAG} dev=${DEV:-cuda:1}" >&2
exec bash scripts/_run_rhythm_bellman.sh
