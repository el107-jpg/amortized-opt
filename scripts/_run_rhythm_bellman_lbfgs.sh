#!/usr/bin/env bash
# L-BFGS + SaLSa line search polish from an Adam-plateau checkpoint.
#
# ε scale is 100× Adam defaults (O(1) L-BFGS directions vs huge Adam D·g).
#   seed 1e-4, bounds [1e-8, 1e-3]  (was 1e-6 / [1e-10, 1e-5])
#
# Usage:
#   DEV=cuda:2 bash scripts/_run_rhythm_bellman_lbfgs.sh
#   RESUME=runs/rhythm_bellman_nf1_ls_c1_05/best_probe.pt STEPS=35000 \
#     DEV=cuda:1 bash scripts/_run_rhythm_bellman_lbfgs.sh
set -euo pipefail

export SKIP_P1=1
export LINE_SEARCH=1
export OPTIMIZER=lbfgs
export LBFGS_HISTORY="${LBFGS_HISTORY:-10}"
export TAG="${TAG:-rhythm_bellman_nf1_lbfgs_e4}"
export RESUME="${RESUME:-runs/rhythm_bellman_nf1_ls_c1_05/last.pt}"
# 5k polish steps after the ~30k Adam plateau ckpt (override STEPS as needed).
export STEPS="${STEPS:-35000}"
export LINE_SEARCH_EPSILON_SEED="${LINE_SEARCH_EPSILON_SEED:-1e-4}"
export LINE_SEARCH_EPSILON_MIN="${LINE_SEARCH_EPSILON_MIN:-1e-8}"
export LINE_SEARCH_EPSILON_MAX="${LINE_SEARCH_EPSILON_MAX:-1e-3}"
export LINE_SEARCH_FRESH=1
# Reuse the Adam+LS pool cache (same recipe as c1_05).
export POOL="${POOL:-runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz}"

exec bash scripts/_run_rhythm_bellman.sh
