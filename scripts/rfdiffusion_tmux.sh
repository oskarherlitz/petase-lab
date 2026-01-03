#!/bin/bash
# Run RFdiffusion in tmux session for long-running jobs
# Usage: bash scripts/rfdiffusion_tmux.sh [conservative|aggressive|test]

set -e

SESSION_NAME="rfdiffusion_${1:-test}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Session '${SESSION_NAME}' already exists!"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    echo "Or kill it first: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

# Create new tmux session
echo "Creating tmux session: ${SESSION_NAME}"
echo ""

# Determine which script to run
case "${1}" in
    conservative)
        SCRIPT="scripts/rfdiffusion_conservative.sh"
        ;;
    aggressive)
        SCRIPT="scripts/rfdiffusion_aggressive.sh"
        ;;
    test|*)
        SCRIPT="scripts/rfdiffusion_direct.sh"
        NUM_DESIGNS="${2:-5}"
        ;;
esac

# Create session and run script
if [ "${1}" == "test" ] || [ -z "${1}" ]; then
    tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}" \
        "bash ${SCRIPT} data/structures/7SH6/raw/7SH6.pdb ${NUM_DESIGNS} 2>&1 | tee runs/rfdiffusion_${SESSION_NAME}.log"
else
    tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}" \
        "bash ${SCRIPT} 2>&1 | tee runs/rfdiffusion_${SESSION_NAME}.log"
fi

echo "✓ Started RFdiffusion in tmux session: ${SESSION_NAME}"
echo ""
echo "Commands:"
echo "  Attach to session:    tmux attach -t ${SESSION_NAME}"
echo "  Detach (while inside): Ctrl+B, then D"
echo "  View logs:            tail -f runs/rfdiffusion_${SESSION_NAME}.log"
echo "  Kill session:         tmux kill-session -t ${SESSION_NAME}"
echo "  List sessions:        tmux ls"
echo ""

