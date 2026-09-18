#!/usr/bin/env bash
# Launch Adam+Armijo and L-BFGS+Armijo test arms (waits for free GPUs).
#
#   bash scripts/_launch_armijo_compare.sh
#   DEV_ADAM=cuda:0 DEV_LBFGS=cuda:1 bash scripts/_launch_armijo_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-8000}"
POLL_S="${POLL_S:-30}"

wait_gpu() {
  local label="$1"
  local exclude="${2:-}"
  while true; do
    for i in 0 1 2 3; do
      if [[ -n "$exclude" && "$exclude" == "cuda:$i" ]]; then
        continue
      fi
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$i" 2>/dev/null || echo 99999)
      if (( used < MEM_THRESHOLD_MB )); then
        echo "[armijo_compare] $label -> cuda:$i (${used} MiB used)" >&2
        echo "cuda:$i"
        return
      fi
    done
    echo "[armijo_compare] waiting for GPU < ${MEM_THRESHOLD_MB} MiB (exclude=${exclude:-none}) ..." >&2
    sleep "$POLL_S"
  done
}

if [[ -n "${DEV_ADAM:-}" && -n "${DEV_LBFGS:-}" ]]; then
  dev_a="$DEV_ADAM"
  dev_l="$DEV_LBFGS"
else
  dev_a=$(wait_gpu "adam_armijo")
  dev_l=$(wait_gpu "lbfgs_armijo" "$dev_a")
fi

mkdir -p runs/rhythm_bellman_nf1_adam_armijo runs/rhythm_bellman_nf1_lbfgs_armijo

echo "[armijo_compare] launching adam on $dev_a, lbfgs on $dev_l" >&2
DEV="$dev_a" bash scripts/_run_rhythm_bellman_adam_armijo.sh \
  > runs/rhythm_bellman_nf1_adam_armijo/launch.log 2>&1 &
pid_a=$!
DEV="$dev_l" bash scripts/_run_rhythm_bellman_lbfgs_armijo.sh \
  > runs/rhythm_bellman_nf1_lbfgs_armijo/launch.log 2>&1 &
pid_l=$!

echo "[armijo_compare] PIDs adam=$pid_a lbfgs=$pid_l" >&2
echo "[armijo_compare] tail -f runs/rhythm_bellman_nf1_{adam,lbfgs}_armijo/launch.log" >&2
