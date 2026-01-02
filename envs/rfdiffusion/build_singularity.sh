#!/bin/bash
# Build Singularity/Apptainer image for RFdiffusion
#
# Usage:
#   ./build_singularity.sh [output_path]
#
# Example:
#   ./build_singularity.sh ./rfdiffusion.sif
#
# Environment variables:
#   SINGULARITY_CMD: Command to use (singularity or apptainer, auto-detected)
#   SINGULARITY_CACHEDIR: Cache directory (optional)

set -e

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

# Output path
OUTPUT_PATH="${1:-${SCRIPT_DIR}/rfdiffusion.sif}"
DEF_FILE="${SCRIPT_DIR}/rfdiffusion.def"

if [ ! -f "${DEF_FILE}" ]; then
    echo "Error: Definition file not found: ${DEF_FILE}"
    exit 1
fi

echo "Building Singularity image..."
echo "  Definition: ${DEF_FILE}"
echo "  Output: ${OUTPUT_PATH}"
echo "  Command: ${SINGULARITY_CMD}"
echo ""

# Build image
${SINGULARITY_CMD} build "${OUTPUT_PATH}" "${DEF_FILE}"

echo ""
echo "✓ Build complete!"
echo "Image saved to: ${OUTPUT_PATH}"
echo ""
echo "To use the image, run:"
echo "  ${SCRIPT_DIR}/run_singularity.sh [arguments...]"

