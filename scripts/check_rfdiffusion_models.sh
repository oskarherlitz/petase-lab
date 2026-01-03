#!/bin/bash
# Check RFdiffusion model files

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/data/models/rfdiffusion"

echo "Checking RFdiffusion model files..."
echo ""

# Check if directory exists
if [ ! -d "${MODELS_DIR}" ]; then
    echo "✗ Models directory not found: ${MODELS_DIR}"
    echo "  Run: bash scripts/rfdiffusion_quick_setup.sh"
    exit 1
fi

# Check each required model
REQUIRED_MODELS=(
    "Base_ckpt.pt"
    "ActiveSite_ckpt.pt"
)

for model in "${REQUIRED_MODELS[@]}"; do
    MODEL_PATH="${MODELS_DIR}/${model}"
    if [ -f "${MODEL_PATH}" ]; then
        SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
        # Check if file is valid PyTorch checkpoint (starts with PK or has proper header)
        HEADER=$(head -c 2 "${MODEL_PATH}" 2>/dev/null | od -An -tx1 | tr -d ' \n' || echo "unknown")
        
        # PyTorch checkpoints typically start with PK (ZIP format) or have specific magic bytes
        if [ "${HEADER}" = "504b" ] || [ "${HEADER}" = "8002" ] || [ "${HEADER}" = "8003" ] || [ "${HEADER}" = "8004" ] || [ "${HEADER}" = "8005" ]; then
            echo "✓ ${model}: ${SIZE} (valid header)"
        else
            echo "✗ ${model}: ${SIZE} (INVALID - header: ${HEADER})"
            echo "  This file appears corrupted. Re-download it."
        fi
    else
        echo "✗ ${model}: NOT FOUND"
    fi
done

echo ""
echo "If any files are corrupted, re-download them:"
echo "  bash scripts/rfdiffusion_quick_setup.sh"

