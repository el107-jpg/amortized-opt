#!/usr/bin/env bash
# Local, CPU-only launcher for the isolated strike-pose teaching app.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-6010}"
SESSION="${SESSION:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-runs/rhythm_timing_direct_admission_gpu1_20k_20260918/step_020000.pt}"

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PORT must be an integer in [1, 65535], got: $PORT" >&2
  exit 2
fi
if (( ${#SESSION} > 64 )) || ! [[ "$SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "SESSION must be at most 64 letters, digits, '_' and '-'" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Actor checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

# These variables apply only to the exec'd editor process. The editor never
# needs CUDA, and one CPU thread prevents it from competing with active runs.
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export NUMEXPR_NUM_THREADS="1"

echo "[strike-pose-editor] session=$SESSION url=http://127.0.0.1:$PORT" >&2
echo "[strike-pose-editor] data=runs/strike_pose_editor/$SESSION" >&2
echo "[strike-pose-editor] checkpoint=$CHECKPOINT training=cpu" >&2
exec "$PYTHON_BIN" -m tools.strike_pose_editor.server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --session "$SESSION" \
  --checkpoint "$CHECKPOINT" \
  --train-device cpu
