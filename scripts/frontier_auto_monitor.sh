#!/usr/bin/env bash
# 30-minute frontier auto-monitor for the two active rhythm Bellman runs.
#
# Writes ``runs/<tag>/.frontier_activate`` when probe miss <= 30 mm and train
# violations are not spiking.  Trainers with ``FRONTIER_AUTO_MISS_MM=30`` pick
# this up in-process; use ``--restart-fallback`` for older processes.
#
# Usage:
#   bash scripts/frontier_auto_monitor.sh          # one poll + status
#   bash scripts/frontier_auto_monitor.sh --daemon # background loop (30 min)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONITOR_DIR="$ROOT/runs/.frontier_monitor"
mkdir -p "$MONITOR_DIR"
LOG="$MONITOR_DIR/monitor.log"
PIDFILE="$MONITOR_DIR/monitor.pid"

if [[ "${1:-}" == "--daemon" ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "monitor already running (pid $(cat "$PIDFILE"))" >&2
    exit 0
  fi
  nohup python3 "$ROOT/scripts/frontier_auto_monitor.py" --daemon --interval 1800 \
    >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "frontier monitor started pid=$(cat "$PIDFILE") log=$LOG"
  exit 0
fi

exec python3 "$ROOT/scripts/frontier_auto_monitor.py" "$@"
