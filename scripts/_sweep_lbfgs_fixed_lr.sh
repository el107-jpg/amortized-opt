#!/usr/bin/env bash
# Launch fixed-ε L-BFGS polish arms in parallel (one GPU each).
#
#   bash scripts/_sweep_lbfgs_fixed_lr.sh
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" CANDIDATES="1e-7 1e-6 1e-5 1e-4" \
#     bash scripts/_sweep_lbfgs_fixed_lr.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CANDIDATES=(${CANDIDATES:-1e-7 1e-6 1e-5 1e-4})
DEVICES=(${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3})

if ((${#CANDIDATES[@]} > ${#DEVICES[@]})); then
  echo "need at least one GPU per candidate (${#CANDIDATES[@]} vs ${#DEVICES[@]})" >&2
  exit 1
fi

mkdir -p runs/_sweep_lbfgs_fixed
echo "[sweep] candidates: ${CANDIDATES[*]}" >&2
echo "[sweep] devices:    ${DEVICES[*]}" >&2

pids=()
for i in "${!CANDIDATES[@]}"; do
  eps="${CANDIDATES[$i]}"
  dev="${DEVICES[$i]}"
  _eps_tag="${eps//./p}"
  out="runs/rhythm_bellman_nf1_lbfgs_fix${_eps_tag}"
  mkdir -p "$out"
  echo "[sweep] $eps on $dev -> $out" >&2
  FIXED_EPS="$eps" DEV="$dev" \
    bash scripts/_run_rhythm_bellman_lbfgs_fixed.sh \
    > "$out/launch.log" 2>&1 &
  pids+=($!)
done

echo "[sweep] launched PIDs: ${pids[*]}" >&2
echo "[sweep] tail e.g. tail -f runs/rhythm_bellman_nf1_lbfgs_fix1e-4/launch.log" >&2
