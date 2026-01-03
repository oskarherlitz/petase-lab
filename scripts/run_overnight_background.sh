#!/usr/bin/env bash
# Run overnight optimization in background without tmux
# Usage: bash scripts/run_overnight_background.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="runs/overnight_$(date +%Y%m%d_%H%M%S).log"

echo "Starting overnight optimization..."
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps aux | grep overnight_fastdesign"
echo ""

# Run in background with nohup
nohup bash "$SCRIPT_DIR/overnight_fastdesign.sh" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "PID saved to: runs/overnight.pid"
echo "$PID" > runs/overnight.pid

echo ""
echo "You can now close this terminal. The process will continue running."
echo ""
echo "To stop it later:"
echo "  kill \$(cat runs/overnight.pid)"

