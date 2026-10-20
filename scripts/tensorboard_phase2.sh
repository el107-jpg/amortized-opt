#!/usr/bin/env bash
# Register recent runs in the single shared TensorBoard on port 6006.
#
# TensorBoard 2.21 defaults to --host localhost and --load_fast=auto. The
# localhost bind is unreachable without a tunnel; the Rust data server often
# serves an empty Time Series dashboard. Force bind_all + the Python loader.
set -euo pipefail
cd "$(dirname "$0")/.."

TB_PORT=6006
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

REGISTRY_DIR="$(pwd)/runs/tensorboard_active"
mkdir -p "$REGISTRY_DIR"

link_run() {
  local name="$1" dir="$2"
  if [[ ! -d "$dir" ]]; then
    echo "missing tb dir: $dir" >&2
    exit 1
  fi
  local target entry
  target="$(cd "$dir" && pwd)"
  entry="$REGISTRY_DIR/$name"
  if [[ -e "$entry" || -L "$entry" ]]; then
    if [[ "$(readlink -f "$entry")" != "$target" ]]; then
      echo "registration already exists for $name; preserving $entry" >&2
      return 1
    fi
  else
    ln -s "$target" "$entry"
  fi
}

# Preserve runs previously exposed by the old separate-port view.
for entry in runs/.tb_view/*; do
  [[ -d "$entry" ]] || continue
  link_run "$(basename "$entry")" "$entry"
done

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
    echo "No recent logs to add; keeping existing registrations." >&2
  fi
fi

if curl -fsS --max-time 5 "http://127.0.0.1:${TB_PORT}/data/runs" >/dev/null 2>&1; then
  echo "Using the existing shared TensorBoard: http://localhost:${TB_PORT}" >&2
  exit 0
fi

# Prevent concurrent launchers from creating another server. Never fall back
# to an alternative port when the shared port is occupied or starting up.
exec 9>runs/.tensorboard6006.lock
if ! flock -n 9; then
  echo "Shared TensorBoard is already starting or running on ${TB_PORT}." >&2
  exit 0
fi
if ss -H -ltn "sport = :${TB_PORT}" | rg -q .; then
  echo "Port ${TB_PORT} is occupied but TensorBoard is not responding; no alternate server started." >&2
  exit 1
fi
echo "Starting shared TensorBoard: http://localhost:${TB_PORT}" >&2
exec "$TB_BIN" --logdir "$REGISTRY_DIR" --port "$TB_PORT" --bind_all --load_fast="$LOAD_FAST"
