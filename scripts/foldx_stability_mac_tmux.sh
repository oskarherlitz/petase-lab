#!/bin/bash
# Run FoldX stability scoring in tmux on Mac
# Usage: bash scripts/foldx_stability_mac_tmux.sh [results_dir] [num_jobs]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-${PROJECT_ROOT}/runs/2026-01-03_rfdiffusion_conservative}"
NUM_JOBS="${2:-$(sysctl -n hw.ncpu)}"
SESSION_NAME="foldx_stability"

cd "${PROJECT_ROOT}"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux not found. Install with:"
    echo "  brew install tmux"
    exit 1
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
LOG_FILE="${PROJECT_ROOT}/runs/foldx_stability.log"

SCRIPT="${PROJECT_ROOT}/scripts/foldx_stability_mac.sh"

# Create a wrapper script that will run in tmux
WRAPPER_SCRIPT="${PROJECT_ROOT}/runs/tmux_wrapper_foldx.sh"
cat > "${WRAPPER_SCRIPT}" << EOF
#!/bin/bash
cd "${PROJECT_ROOT}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "FoldX Stability Scoring (tmux)"
echo "Started: \$(date)"
echo "Directory: ${RESULTS_DIR}"
echo "Parallel jobs: ${NUM_JOBS}"
echo "=========================================="
echo ""

bash "${SCRIPT}" "${RESULTS_DIR}" "${NUM_JOBS}"

EXIT_CODE=\$?
echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ FoldX scoring completed successfully!"
else
    echo "✗ FoldX scoring failed with exit code: \$EXIT_CODE"
fi
echo "Ended: \$(date)"
echo "=========================================="
echo ""
echo "Log saved to: ${LOG_FILE}"
echo "Results: ${RESULTS_DIR}/foldx_scores/foldx_scores.csv"
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
    fi
    exit 1
fi

echo "✓ Started FoldX scoring in tmux session: ${SESSION_NAME}"
echo ""
echo "Commands:"
echo "  Attach to session:    tmux attach -t ${SESSION_NAME}"
echo "  Detach (while inside): Ctrl+B, then D"
echo "  View logs:            tail -f ${LOG_FILE}"
echo "  Kill session:         tmux kill-session -t ${SESSION_NAME}"
echo "  List sessions:        tmux ls"
echo ""
echo "Estimated time: ~$((300 * 2 / NUM_JOBS / 60)) hours for 300 designs"
echo "Results will be in: ${RESULTS_DIR}/foldx_scores/foldx_scores.csv"
echo ""

