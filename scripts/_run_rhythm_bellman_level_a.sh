#!/usr/bin/env bash
# Level-A torque curriculum on the rhythm Bellman recipe.
#
# Phase 1 (steps < TORQUE_FULL_START): collision + terminal + decoupled torque
#   τ ≈ g(q) + diag(M(q_ready)) @ qdd  (no inertial coupling / Coriolis).
# Phase 2: smoothstep blend into full coupled knot RNEA over TORQUE_FULL_RAMP.
#
# Usage:
#   DEV=cuda:1 SKIP_P1=1 INIT=runs/phase1_rhythm_bellman_nf1_prsp/best.pt \
#     bash scripts/_run_rhythm_bellman_level_a.sh
#
# Pilot (shorter):
#   STEPS=8000 TORQUE_FULL_START=4000 TORQUE_FULL_RAMP=800 \
#     DEV=cuda:1 SKIP_P1=1 INIT=runs/phase1_rhythm_bellman_nf1_prsp/best.pt \
#     bash scripts/_run_rhythm_bellman_level_a.sh
set -euo pipefail

export TAG="${TAG:-rhythm_bellman_nf1_ls_level_a}"
export TORQUE_PHYSICS="${TORQUE_PHYSICS:-diag_grav_frozen}"
export TORQUE_FULL_START="${TORQUE_FULL_START:-8000}"
export TORQUE_FULL_RAMP="${TORQUE_FULL_RAMP:-2000}"
export STEPS="${STEPS:-20000}"
export LINE_SEARCH="${LINE_SEARCH:-1}"
# Reset SaLSa A/B when full RNEA ramp begins (objective discontinuity).
export LINE_SEARCH_FORCE_AL_RESET="${LINE_SEARCH_FORCE_AL_RESET:-1}"

exec bash "$(dirname "$0")/_run_rhythm_bellman.sh" "$@"
