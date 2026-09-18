#!/usr/bin/env bash
# Adam + fixed scalar ε (no Armijo probe search).
#
# Usage:
#   FIXED_EPS=1e-10 DEV=cuda:0 bash scripts/_run_rhythm_bellman_adam_fixed.sh
set -euo pipefail

export SKIP_P1=1
export LINE_SEARCH=1
export LINE_SEARCH_MODE=fixed_epsilon
export LINE_SEARCH_FRESH=1
export OPTIMIZER=adamw
export RESUME="${RESUME:-runs/rhythm_bellman_nf1_ls_c1_05/last.pt}"
export STEPS="${STEPS:-38200}"
export PROBE_EVERY="${PROBE_EVERY:-1000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export POOL="${POOL:-runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz}"

FIXED_EPS="${FIXED_EPS:-1e-10}"
export LINE_SEARCH_EPSILON_SEED="$FIXED_EPS"
export LINE_SEARCH_EPSILON_MIN="${LINE_SEARCH_EPSILON_MIN:-1e-10}"
export LINE_SEARCH_EPSILON_MAX="${LINE_SEARCH_EPSILON_MAX:-1e-9}"

_eps_tag="${FIXED_EPS//./p}"
export TAG="${TAG:-rhythm_bellman_nf1_adam_fix${_eps_tag}}"

echo "[adam_fixed] ε=${FIXED_EPS} tag=${TAG} dev=${DEV:-cuda:1}" >&2
exec bash scripts/_run_rhythm_bellman.sh
