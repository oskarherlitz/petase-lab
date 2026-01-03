#!/bin/bash
# RFdiffusion Test Run - Small scale (5 designs) to verify setup
# Run this BEFORE the overnight run!

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuration
INPUT_PDB="${1:-data/structures/7SH6/raw/7SH6.pdb}"
MODELS_DIR="${RFDIFFUSION_MODELS:-${PROJECT_ROOT}/data/models/rfdiffusion}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/$(date +%Y-%m-%d)_rfdiffusion_test"
NUM_DESIGNS=5

# Export for run_docker.sh (use absolute paths)
export RFDIFFUSION_OUTPUTS="${OUTPUT_DIR}"
export RFDIFFUSION_MODELS="${MODELS_DIR}"
export RFDIFFUSION_INPUTS="$(dirname ${PROJECT_ROOT}/${INPUT_PDB})"

# Conservative mask: 13 positions
# Positions: N114, L117, Q119, T140, W159, G165, I168, A180, S188, N205, S214, S269, S282
CONSERVATIVE_MASK="A114/A117/A119/A140/A159/A165/A168/A180/A188/A205/A214/A269/A282"

echo "=========================================="
echo "RFdiffusion Test Run"
echo "=========================================="
echo "Input PDB: ${INPUT_PDB}"
echo "Models: ${MODELS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Designs: ${NUM_DESIGNS}"
echo "Mask: Conservative (13 positions)"
echo ""

# Check prerequisites
if [ ! -f "${INPUT_PDB}" ]; then
    echo "Error: Input PDB not found: ${INPUT_PDB}"
    echo "Run: bash scripts/rfdiffusion_quick_setup.sh"
    exit 1
fi

if [ ! -f "${MODELS_DIR}/Base_ckpt.pt" ]; then
    echo "Error: Model weights not found in ${MODELS_DIR}"
    echo "Run: bash scripts/rfdiffusion_quick_setup.sh"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run RFdiffusion
echo "Starting RFdiffusion test run..."
echo ""

# Check for Docker (RunPod may need sudo or have different setup)
USE_DOCKER=false
USE_SINGULARITY=false

if command -v docker &> /dev/null; then
    # Try docker (may need sudo on some systems)
    if docker ps &> /dev/null 2>&1 || sudo docker ps &> /dev/null 2>&1 || docker info &> /dev/null 2>&1; then
        USE_DOCKER=true
    fi
fi

if [ "$USE_DOCKER" = false ]; then
    if command -v apptainer &> /dev/null || command -v singularity &> /dev/null; then
        USE_SINGULARITY=true
    fi
fi

if [ "$USE_DOCKER" = true ]; then
    echo "Using Docker..."
    # Use absolute paths inside container
    # Hydra creates its own output directory, so we need to set hydra.run.dir
    envs/rfdiffusion/run_docker.sh \
        inference.output_prefix=/data/outputs/designs \
        inference.model_directory_path=/data/models \
        inference.input_pdb=/data/inputs/$(basename ${INPUT_PDB}) \
        inference.num_designs=${NUM_DESIGNS} \
        'contigmap.contigs=[A1-290]' \
        "contigmap.inpaint_seq=[${CONSERVATIVE_MASK}]" \
        inference.ckpt_override_path=/data/models/ActiveSite_ckpt.pt \
        inference.schedule_directory_path=/data/outputs/schedules \
        hydra.run.dir=/data/outputs \
        hydra.job.chdir=False \
        hydra.output_subdir=null
elif [ "$USE_SINGULARITY" = true ]; then
    echo "Using Singularity/Apptainer..."
    # Use absolute paths inside container
    # Hydra creates its own output directory, so we need to set hydra.run.dir
    envs/rfdiffusion/run_singularity.sh \
        inference.output_prefix=/data/outputs/designs \
        inference.model_directory_path=/data/models \
        inference.input_pdb=/data/inputs/$(basename ${INPUT_PDB}) \
        inference.num_designs=${NUM_DESIGNS} \
        'contigmap.contigs=[A1-290]' \
        "contigmap.inpaint_seq=[${CONSERVATIVE_MASK}]" \
        inference.ckpt_override_path=/data/models/ActiveSite_ckpt.pt \
        hydra.run.dir=/data/outputs
else
    echo "Error: Neither Docker nor Singularity/Apptainer found"
    echo ""
    echo "On RunPod, you may need to:"
    echo "  1. Check if Docker is installed: which docker"
    echo "  2. Check if you need sudo: sudo docker ps"
    echo "  3. Or install Docker: curl -fsSL https://get.docker.com | sh"
    echo "  4. Or run RFdiffusion directly (without container) if dependencies are installed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Test Run Complete!"
echo "=========================================="
echo "Check outputs in: ${OUTPUT_DIR}"
echo ""
echo "Verify:"
echo "  ls ${OUTPUT_DIR}/designs/*.pdb"
echo "  ls ${OUTPUT_DIR}/designs/*.trb"
echo ""
echo "If successful, proceed with overnight run:"
echo "  bash scripts/rfdiffusion_conservative.sh"
echo "  bash scripts/rfdiffusion_aggressive.sh"
echo ""

