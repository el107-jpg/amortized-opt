#!/usr/bin/env bash
# Gravity-only static torque experiment on the rhythm Bellman recipe.
#
# Prices |RNEA(q,0,0)| with the usual softplus hinge (w_torque_static=20 by
# default) plus AL on the static excess (torque_* groups).  Coupled RNEA off.
#
# Usage:
#   DEV=cuda:1 SKIP_P1=1 INIT=runs/phase1_rhythm_bellman_nf1_prsp/best.pt \
#     bash scripts/_run_rhythm_bellman_grav_static.sh
#
# Pilot:
#   STEPS=8000 DEV=cuda:1 SKIP_P1=1 INIT=runs/phase1_rhythm_bellman_nf1_prsp/best.pt \
#     bash scripts/_run_rhythm_bellman_grav_static.sh
set -euo pipefail

export TAG="${TAG:-rhythm_bellman_nf1_ls_grav_static_tight}"
export TORQUE_STATIC_ONLY=1
export TORQUE_STATIC_AL="${TORQUE_STATIC_AL:-1}"
export STATIC_TORQUE_TIGHTEN="${STATIC_TORQUE_TIGHTEN:-decoder_measured}"
export W_TORQUE_STATIC="${W_TORQUE_STATIC:-20}"
export STEPS="${STEPS:-20000}"
export LINE_SEARCH="${LINE_SEARCH:-1}"

exec bash "$(dirname "$0")/_run_rhythm_bellman.sh" "$@"
