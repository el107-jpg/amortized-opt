#!/usr/bin/env bash
# L-BFGS with a fixed scalar ε (no Armijo probe search).
#
# Usage:
#   FIXED_EPS=1e-4 DEV=cuda:1 bash scripts/_run_rhythm_bellman_lbfgs_fixed.sh
#   bash scripts/_sweep_lbfgs_fixed_lr.sh
set -euo pipefail

export SKIP_P1=1
export LINE_SEARCH=1
export LINE_SEARCH_MODE=fixed_epsilon
export LINE_SEARCH_FRESH=1
export OPTIMIZER=lbfgs
export LBFGS_HISTORY="${LBFGS_HISTORY:-10}"
export RESUME="${RESUME:-runs/rhythm_bellman_nf1_ls_c1_05/last.pt}"
# last.pt @ ~38k; default +2k polish steps.
export STEPS="${STEPS:-40000}"
export PROBE_EVERY="${PROBE_EVERY:-1000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export POOL="${POOL:-runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz}"

FIXED_EPS="${FIXED_EPS:-1e-4}"
export LINE_SEARCH_EPSILON_SEED="$FIXED_EPS"
export LINE_SEARCH_EPSILON_MIN="${LINE_SEARCH_EPSILON_MIN:-1e-8}"
export LINE_SEARCH_EPSILON_MAX="${LINE_SEARCH_EPSILON_MAX:-1e-3}"

# Tag encodes ε for easy comparison (1e-4 -> fix1e-4).
_eps_tag="${FIXED_EPS//./p}"
export TAG="${TAG:-rhythm_bellman_nf1_lbfgs_fix${_eps_tag}}"

echo "[lbfgs_fixed] ε=${FIXED_EPS} tag=${TAG} dev=${DEV:-cuda:1}" >&2
exec bash scripts/_run_rhythm_bellman.sh
