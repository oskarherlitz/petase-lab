#!/bin/bash
# Run RFdiffusion in tmux session for long-running jobs
# Usage: bash scripts/rfdiffusion_tmux.sh [conservative|aggressive|test]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="rfdiffusion_${1:-test}"

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

# Create output directory for logs
mkdir -p "${PROJECT_ROOT}/runs"
LOG_FILE="${PROJECT_ROOT}/runs/rfdiffusion_${SESSION_NAME}.log"

# Determine which script to run
case "${1}" in
    conservative)
        SCRIPT="${PROJECT_ROOT}/scripts/rfdiffusion_conservative.sh"
        ;;
    aggressive)
        SCRIPT="${PROJECT_ROOT}/scripts/rfdiffusion_aggressive.sh"
        ;;
    test|*)
        SCRIPT="${PROJECT_ROOT}/scripts/rfdiffusion_direct.sh"
        NUM_DESIGNS="${2:-5}"
        ;;
esac

# Create a wrapper script that will run in tmux
WRAPPER_SCRIPT="${PROJECT_ROOT}/runs/tmux_wrapper_${SESSION_NAME}.sh"
cat > "${WRAPPER_SCRIPT}" << EOF
#!/bin/bash
cd "${PROJECT_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "RFdiffusion tmux session: ${SESSION_NAME}"
echo "Started: \$(date)"
echo "=========================================="
echo ""

if [ "${1}" == "test" ] || [ -z "${1}" ]; then
    echo "Running: ${SCRIPT} data/structures/7SH6/raw/7SH6.pdb ${NUM_DESIGNS}"
    echo ""
    bash "${SCRIPT}" "${PROJECT_ROOT}/data/structures/7SH6/raw/7SH6.pdb" "${NUM_DESIGNS}"
else
    echo "Running: ${SCRIPT}"
    echo ""
    bash "${SCRIPT}"
fi

EXIT_CODE=\$?
echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ Script completed successfully!"
else
    echo "✗ Script failed with exit code: \$EXIT_CODE"
fi
echo "Ended: \$(date)"
echo "=========================================="
echo ""
echo "Log saved to: ${LOG_FILE}"
echo "Press Ctrl+C to close this window"
sleep 3600
EOF

chmod +x "${WRAPPER_SCRIPT}"

# Create tmux session
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}" "bash ${WRAPPER_SCRIPT}"

# Wait a moment and check if session is still running
sleep 2
if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "⚠ Session exited immediately. Check the log:"
    echo "  cat ${LOG_FILE}"
    if [ -f "${LOG_FILE}" ]; then
        echo ""
        echo "Last 30 lines of log:"
        tail -30 "${LOG_FILE}"
    else
        echo ""
        echo "Log file not created. Possible issues:"
        echo "  1. RFdiffusion not installed - run: bash scripts/install_rfdiffusion_quick.sh"
        echo "  2. Input PDB missing - check: ls ${PROJECT_ROOT}/data/structures/7SH6/raw/7SH6.pdb"
        echo "  3. Script not found - check: ls ${SCRIPT}"
    fi
    exit 1
fi

echo "✓ Started RFdiffusion in tmux session: ${SESSION_NAME}"
echo ""
echo "Commands:"
echo "  Attach to session:    tmux attach -t ${SESSION_NAME}"
echo "  Detach (while inside): Ctrl+B, then D"
echo "  View logs:            tail -f ${LOG_FILE}"
echo "  Kill session:         tmux kill-session -t ${SESSION_NAME}"
echo "  List sessions:        tmux ls"
echo ""
