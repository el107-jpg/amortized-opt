#!/usr/bin/env bash
# Launch a phase2 training ablation.
# Usage: ./_run_ablation.sh <name> <gpu> <max_hours> <extra_args...>
set -euo pipefail

NAME="$1"
GPU="$2"
HOURS="$3"
shift 3

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/.."  # repo root

source /data2/miniconda3/etc/profile.d/conda.sh
conda activate /data2/eric/conda_envs/g1drum
source /data2/tiger/beyond_mimic_workspace/software/setup_conda_env.sh
export UNITREE_MODEL_DIR=/data2/eric/workspace/unitree_model
export TMPDIR=/data2/eric/tmp

LOGDIR="$ROOT/runs/$NAME"
mkdir -p "$LOGDIR"

echo "=== $(date): starting $NAME on $GPU, max $HOURS h ===" | tee -a "$LOGDIR/run.log"
echo "extra args: $@" | tee -a "$LOGDIR/run.log"

python amortized/scripts/train_phase2.py \
    --device "$GPU" \
    --steps 60000 \
    --max-hours "$HOURS" \
    --scenarios 250000 \
    --workers 48 \
    --name "$NAME" \
    "$@" 2>&1 | tee -a "$LOGDIR/run.log"

echo "=== $(date): $NAME finished with exit code $? ===" | tee -a "$LOGDIR/run.log"
