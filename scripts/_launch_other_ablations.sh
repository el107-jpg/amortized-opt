#!/usr/bin/env bash
# Adam+Armijo, L-BFGS+Armijo, Adam fixed ε=1e-10 (200-step polish from last.pt).
#
#   bash scripts/_launch_other_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pkill -f "train_phase2.*lbfgs_fix" 2>/dev/null || true
sleep 2

mkdir -p runs/rhythm_bellman_nf1_adam_armijo \
         runs/rhythm_bellman_nf1_lbfgs_armijo \
         runs/rhythm_bellman_nf1_adam_fix1e-10

echo "[ablations] stopping fixed-L-BFGS sweep; launching 3 arms (+200 steps from 38k)" >&2

DEV=cuda:0 bash scripts/_run_rhythm_bellman_adam_armijo.sh \
  > runs/rhythm_bellman_nf1_adam_armijo/launch.log 2>&1 &
pid_a=$!
DEV=cuda:1 bash scripts/_run_rhythm_bellman_lbfgs_armijo.sh \
  > runs/rhythm_bellman_nf1_lbfgs_armijo/launch.log 2>&1 &
pid_l=$!
FIXED_EPS=1e-10 DEV=cuda:2 bash scripts/_run_rhythm_bellman_adam_fixed.sh \
  > runs/rhythm_bellman_nf1_adam_fix1e-10/launch.log 2>&1 &
pid_f=$!

echo "[ablations] PIDs adam_armijo=$pid_a lbfgs_armijo=$pid_l adam_fix1e-10=$pid_f" >&2
echo "[ablations] tail -f runs/rhythm_bellman_nf1_{adam_armijo,lbfgs_armijo,adam_fix1e-10}/launch.log" >&2
