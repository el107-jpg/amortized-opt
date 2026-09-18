#!/usr/bin/env bash
# TensorBoard for recent phase-2 runs (avoids scanning all of runs/).
#
# TensorBoard 2.21 defaults to --host localhost and --load_fast=auto. The
# localhost bind is unreachable without a tunnel; the Rust data server often
# serves an empty Time Series dashboard. Force bind_all + the Python loader.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-6008}"
MAX_RUNS="${MAX_RUNS:-8}"
MAX_AGE_HOURS="${MAX_AGE_HOURS:-48}"
LOAD_FAST="${LOAD_FAST:-false}"
TB_BIN="${TB_BIN:-}"
if [[ -z "$TB_BIN" ]]; then
  if [[ -x /data2/eric/conda_envs/amortized_compile/bin/tensorboard ]]; then
    TB_BIN=/data2/eric/conda_envs/amortized_compile/bin/tensorboard
  else
    TB_BIN=$(command -v tensorboard || true)
  fi
fi
if [[ -z "$TB_BIN" || ! -x "$TB_BIN" ]]; then
  echo "tensorboard not found. pip install 'tensorboard>=2.14' or pip install -e '.[monitoring]'" >&2
  exit 1
fi

VIEW_DIR="${VIEW_DIR:-runs/.tb_view}"
rm -rf "$VIEW_DIR"
mkdir -p "$VIEW_DIR"

link_run() {
  local name="$1" dir="$2"
  if [[ ! -d "$dir" ]]; then
    echo "missing tb dir: $dir" >&2
    exit 1
  fi
  ln -sfn "$(cd "$dir" && pwd)" "$VIEW_DIR/$name"
}

if [[ -n "${LOGDIR_SPEC:-}" ]]; then
  IFS=',' read -ra parts <<< "$LOGDIR_SPEC"
  for item in "${parts[@]}"; do
    link_run "${item%%:*}" "${item#*:}"
  done
else
  cutoff=$(date -d "${MAX_AGE_HOURS} hours ago" +%s)
  n=0
  while IFS= read -r dir; do
    [[ -z "$dir" ]] && continue
    mtime=$(stat -c %Y "$dir")
    (( mtime < cutoff )) && continue
    link_run "$(basename "$(dirname "$dir")")" "$dir"
    n=$((n + 1))
    (( n >= MAX_RUNS )) && break
  done < <(
    find runs -mindepth 2 -maxdepth 2 -type d -name tb -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | awk '{print $2}'
  )
  if [[ "$n" -eq 0 ]]; then
    echo "No tb logs newer than ${MAX_AGE_HOURS}h under runs/*/tb" >&2
    exit 1
  fi
fi

echo "$TB_BIN --logdir ${VIEW_DIR} --port ${PORT} --bind_all --load_fast=${LOAD_FAST}" >&2
echo "runs: $(ls -1 "$VIEW_DIR" | tr '\n' ' ')" >&2
echo "Open http://localhost:${PORT}  (or this host:${PORT}; 6006/6007 are often taken)" >&2
exec "$TB_BIN" --logdir "$VIEW_DIR" --port "${PORT}" --bind_all --load_fast="${LOAD_FAST}"
