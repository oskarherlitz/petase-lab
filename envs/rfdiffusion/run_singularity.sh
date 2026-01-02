#!/bin/bash
# RFdiffusion Singularity/Apptainer Runner for HPC
#
# Usage:
#   ./run_singularity.sh [RFdiffusion arguments...]
#
# Example:
#   ./run_singularity.sh \
#     inference.output_prefix=outputs/design \
#     inference.model_directory_path=/data/models \
#     inference.input_pdb=inputs/target.pdb \
#     inference.num_designs=10 \
#     'contigmap.contigs=[10-40/A163-181/10-40]'
#
# Environment variables:
#   RFDIFFUSION_MODELS: Path to model weights directory
#   RFDIFFUSION_INPUTS: Path to input files directory
#   RFDIFFUSION_OUTPUTS: Path to output directory
#   RFDIFFUSION_SIF: Path to Singularity image file (default: ./rfdiffusion.sif)
#   SINGULARITY_CMD: Command to use (singularity or apptainer, auto-detected)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Detect Singularity/Apptainer
if [ -z "${SINGULARITY_CMD}" ]; then
    if command -v apptainer &> /dev/null; then
        SINGULARITY_CMD="apptainer"
    elif command -v singularity &> /dev/null; then
        SINGULARITY_CMD="singularity"
    else
        echo "Error: Neither 'singularity' nor 'apptainer' found in PATH"
        exit 1
    fi
fi

# Default paths (adjust for your HPC setup)
MODELS_DIR="${RFDIFFUSION_MODELS:-${PROJECT_ROOT}/data/models/rfdiffusion}"
INPUTS_DIR="${RFDIFFUSION_INPUTS:-${PROJECT_ROOT}/data/raw/structures}"
OUTPUTS_DIR="${RFDIFFUSION_OUTPUTS:-${PROJECT_ROOT}/runs/$(date +%Y-%m-%d)_rfdiffusion/outputs}"
SIF_PATH="${RFDIFFUSION_SIF:-${SCRIPT_DIR}/rfdiffusion.sif}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUTS_DIR}"

# Check if SIF exists
if [ ! -f "${SIF_PATH}" ]; then
    echo "Error: Singularity image not found at ${SIF_PATH}"
    echo ""
    echo "To build the image, run:"
    echo "  ${SINGULARITY_CMD} build ${SIF_PATH} ${SCRIPT_DIR}/rfdiffusion.def"
    exit 1
fi

# Check for GPU support
GPU_FLAG=""
if [ -n "${CUDA_VISIBLE_DEVICES}" ] || command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--nv"
    echo "✓ GPU support enabled"
fi

# Run Singularity container
echo "Running RFdiffusion in ${SINGULARITY_CMD}..."
echo "  Models: ${MODELS_DIR}"
echo "  Inputs: ${INPUTS_DIR}"
echo "  Outputs: ${OUTPUTS_DIR}"
echo "  Image: ${SIF_PATH}"
echo ""

# Mount RFdiffusion and install packages if needed
# Note: We need to install SE3Transformer and RFdiffusion package inside container
# since they're not in the base image (RFdiffusion is mounted at runtime)
${SINGULARITY_CMD} exec ${GPU_FLAG} \
    --bind "${PROJECT_ROOT}/external/rfdiffusion:/app/RFdiffusion:ro" \
    --bind "${MODELS_DIR}:/data/models:ro" \
    --bind "${INPUTS_DIR}:/data/inputs:ro" \
    --bind "${OUTPUTS_DIR}:/data/outputs:rw" \
    --bind "${PROJECT_ROOT}:/workspace:rw" \
    "${SIF_PATH}" \
    bash -c "
        # Install RFdiffusion packages if not already installed
        # Check by trying to import (will fail if not installed)
        if ! python3.9 -c 'import rfdiffusion' 2>/dev/null; then
            echo 'Installing RFdiffusion packages (first run only)...'
            pip install --no-cache-dir /app/RFdiffusion/env/SE3Transformer || true
            pip install --no-cache-dir /app/RFdiffusion --no-deps || true
        fi
        # Run inference
        cd /app/RFdiffusion
        export PYTHONPATH=\"/app/RFdiffusion:\${PYTHONPATH}\"
        python3.9 scripts/run_inference.py \"\$@\"
    " "$@"

echo ""
echo "✓ RFdiffusion run complete!"
echo "Results saved to: ${OUTPUTS_DIR}"

