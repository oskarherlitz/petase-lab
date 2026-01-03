#!/bin/bash
# Run RFdiffusion aggressive mask in tmux on RunPod GPU
# Usage: bash scripts/rfdiffusion_aggressive_tmux.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="rfdiffusion_aggressive"

cd "${PROJECT_ROOT}"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux 2>/dev/null || {
        echo "Error: Could not install tmux. Install manually:"
        echo "  apt-get install -y tmux"
        exit 1
    }
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
LOG_FILE="${PROJECT_ROOT}/runs/rfdiffusion_aggressive.log"

SCRIPT="${PROJECT_ROOT}/scripts/rfdiffusion_aggressive.sh"

# Create a wrapper script that will run in tmux
WRAPPER_SCRIPT="${PROJECT_ROOT}/runs/tmux_wrapper_aggressive.sh"
cat > "${WRAPPER_SCRIPT}" << EOF
#!/bin/bash
cd ${PROJECT_ROOT}

# Fix CUDA library path for DGL
CUDA_LIB_PATHS=(
    "/usr/local/nvidia/lib64"
    "/usr/local/nvidia/lib"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-11.8/targets/x86_64-linux/lib"
    "/usr/local/cuda-11.8/lib64"
    "/usr/local/cuda-11.6/lib64"
    "/usr/local/cuda-12.4/lib64"
    "/usr/lib/x86_64-linux-gnu"
)
for path in "\${CUDA_LIB_PATHS[@]}"; do
    if [ -d "\${path}" ] && find "\${path}" -name "libcudart.so*" 2>/dev/null | grep -q .; then
        export LD_LIBRARY_PATH="\${path}:\${LD_LIBRARY_PATH}"
    fi
done
# Fallback: search for libcudart
if [ -z "\${LD_LIBRARY_PATH##*cuda*}" ] && [ -z "\${LD_LIBRARY_PATH##*nvidia*}" ]; then
    CUDA_LIB=\$(find /usr /usr/local -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -n "\${CUDA_LIB}" ]; then
        export LD_LIBRARY_PATH="\${CUDA_LIB}:\${LD_LIBRARY_PATH}"
    fi
fi

exec > >(tee -a ${LOG_FILE}) 2>&1

echo "=========================================="
echo "RFdiffusion Aggressive Mask (tmux)"
echo "Started: \$(date)"
echo "LD_LIBRARY_PATH: \${LD_LIBRARY_PATH}"
echo "=========================================="
echo ""

echo "Running: ${SCRIPT}"
echo ""

bash ${SCRIPT}

EXIT_CODE=\$?
echo ""
echo "=========================================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✓ Aggressive mask run completed successfully!"
else
    echo "✗ Aggressive mask run failed with exit code: \$EXIT_CODE"
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
    fi
    exit 1
fi

echo "✓ Started RFdiffusion aggressive mask in tmux session: ${SESSION_NAME}"
echo ""
echo "Commands:"
echo "  Attach to session:    tmux attach -t ${SESSION_NAME}"
echo "  Detach (while inside): Ctrl+B, then D"
echo "  View logs:            tail -f ${LOG_FILE}"
echo "  Kill session:         tmux kill-session -t ${SESSION_NAME}"
echo "  List sessions:        tmux ls"
echo ""
echo "Estimated time: 6-12 hours for 300 designs"
echo ""

