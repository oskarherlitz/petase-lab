#!/bin/bash
# RFdiffusion Docker Runner for PETase Lab
#
# Usage:
#   ./run_docker.sh [RFdiffusion arguments...]
#
# Example:
#   ./run_docker.sh \
#     inference.output_prefix=outputs/design \
#     inference.model_directory_path=/data/models \
#     inference.input_pdb=inputs/target.pdb \
#     inference.num_designs=10 \
#     'contigmap.contigs=[10-40/A163-181/10-40]'
#
# Environment variables:
#   RFDIFFUSION_MODELS: Path to model weights directory (default: ./models)
#   RFDIFFUSION_INPUTS: Path to input files directory (default: ./inputs)
#   RFDIFFUSION_OUTPUTS: Path to output directory (default: ./outputs)
#   RFDIFFUSION_IMAGE: Docker image name (default: petase-rfdiffusion)

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default paths (relative to project root)
MODELS_DIR="${RFDIFFUSION_MODELS:-${PROJECT_ROOT}/data/models/rfdiffusion}"
INPUTS_DIR="${RFDIFFUSION_INPUTS:-${PROJECT_ROOT}/data/structures/7SH6/raw}"
OUTPUTS_DIR="${RFDIFFUSION_OUTPUTS:-${PROJECT_ROOT}/runs/$(date +%Y-%m-%d)_rfdiffusion/outputs}"
IMAGE_NAME="${RFDIFFUSION_IMAGE:-petase-rfdiffusion}"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUTS_DIR}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if image exists
if ! docker images | grep -q "^${IMAGE_NAME}"; then
    echo "Warning: Docker image '${IMAGE_NAME}' not found."
    echo "Building image from ${SCRIPT_DIR}/Dockerfile..."
    docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${PROJECT_ROOT}"
fi

# Check for GPU support
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    GPU_FLAG="--gpus all"
    echo "✓ GPU support detected"
else
    echo "⚠ Warning: nvidia-smi not found. Running without GPU (may be slow)"
fi

# Run Docker container
echo "Running RFdiffusion in Docker..."
echo "  Models: ${MODELS_DIR}"
echo "  Inputs: ${INPUTS_DIR}"
echo "  Outputs: ${OUTPUTS_DIR}"
echo ""

docker run -it --rm ${GPU_FLAG} \
    -v "${PROJECT_ROOT}/external/rfdiffusion:/app/RFdiffusion:ro" \
    -v "${MODELS_DIR}:/data/models:ro" \
    -v "${INPUTS_DIR}:/data/inputs:ro" \
    -v "${OUTPUTS_DIR}:/data/outputs:rw" \
    -v "${PROJECT_ROOT}:/workspace:rw" \
    -e PYTHONPATH=/app/RFdiffusion \
    -e HYDRA_FULL_ERROR=1 \
    --entrypoint="python3.9" \
    -w /data/outputs \
    "${IMAGE_NAME}" \
    /app/RFdiffusion/scripts/run_inference.py \
    "$@"

echo ""
echo "✓ RFdiffusion run complete!"
echo "Results saved to: ${OUTPUTS_DIR}"

