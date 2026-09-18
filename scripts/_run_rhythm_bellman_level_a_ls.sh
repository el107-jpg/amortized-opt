#!/usr/bin/env bash
# Level-A torque (g(q) + diag(M)@qdd) with line-search Adam, torque AL off.
#
# Stays on level A forever (no blend to full RNEA).  Hinge at eta_torque=0.72
# (10% tighter than the default 0.80) to offset the looser plant proxy.
#
# Usage:
#   DEV=cuda:1 SKIP_P1=1 INIT=runs/phase1_rhythm_bellman_nf1_prsp/best.pt \
#     bash scripts/_run_rhythm_bellman_level_a_ls.sh
set -euo pipefail

export TAG="${TAG:-rhythm_bellman_nf1_ls_level_a_simple}"
export TORQUE_PHYSICS="${TORQUE_PHYSICS:-diag_grav_frozen}"
export TORQUE_FULL_START="${TORQUE_FULL_START:--1}"
export TORQUE_AL_OFF=1
export ETA_TORQUE="${ETA_TORQUE:-0.72}"
export STEPS="${STEPS:-20000}"
export LINE_SEARCH="${LINE_SEARCH:-1}"

exec bash "$(dirname "$0")/_run_rhythm_bellman.sh" "$@"
