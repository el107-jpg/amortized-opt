#!/usr/bin/env bash
# Continue grav-static pretrain into full coupled knot-RNEA torque limits.
#
# Resumes weights/optimizer/AL from the static run; switches OFF
# ``--torque-static-only`` and restores default ``w_torque`` + ``w_torque_peak``.
# Rolling eval ``ok_torque`` stays on knot RNEA (unchanged from static phase).
#
# Usage:
#   DEV=cuda:1 bash scripts/_run_rhythm_bellman_rnea_from_grav.sh
#
# Custom resume / tag:
#   RESUME=runs/rhythm_bellman_nf1_ls_grav_static_tight/best.pt \
#     TAG=my_rnea_cont bash scripts/_run_rhythm_bellman_rnea_from_grav.sh
set -euo pipefail

export SKIP_P1=1
export LINE_SEARCH=1
export LINE_SEARCH_MODE=salsa_transfer
export OPTIMIZER=adamw
export TAG="${TAG:-rhythm_bellman_nf1_ls_rnea_from_grav}"
export RESUME="${RESUME:-runs/rhythm_bellman_nf1_ls_grav_static_tight/last.pt}"
export STEPS="${STEPS:-40000}"
export PROBE_EVERY="${PROBE_EVERY:-2000}"
export EVAL_EVERY="${EVAL_EVERY:-2000}"
export POOL="${POOL:-runs/p2pool_rhythm_bellman_nf1_ls_mix306010_evalbox_tilt3_rad0.15_jit0.05x0.2_rb0.25_tau1.0.npz}"
export LINE_SEARCH_FRESH=0
export LINE_SEARCH_EPSILON_SEED="${LINE_SEARCH_EPSILON_SEED:-1e-6}"
# Objective switch: fresh SaLSa acceptance stats; keep AL λ from static pretrain.
export LINE_SEARCH_FORCE_AL_RESET="${LINE_SEARCH_FORCE_AL_RESET:-0}"
export WAIST_UPRIGHT_EASE_WP="${WAIST_UPRIGHT_EASE_WP:-1}"
export FRONTIER_AUTO_MISS_MM="${FRONTIER_AUTO_MISS_MM:-30}"

echo "[rnea_from_grav] tag=${TAG} dev=${DEV:-cuda:1} resume=${RESUME} " \
     "steps=${STEPS} pool=${POOL} ls_fresh=${LINE_SEARCH_FRESH} " \
     "ls_al_reset=${LINE_SEARCH_FORCE_AL_RESET}" >&2

exec bash "$(dirname "$0")/_run_rhythm_bellman.sh"
