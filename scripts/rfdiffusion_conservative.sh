#!/bin/bash
# RFdiffusion Conservative Mask Overnight Run
# ~300 designs with conservative mask (13 positions)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuration
INPUT_PDB="${1:-data/structures/7SH6/raw/7SH6.pdb}"
MODELS_DIR="${RFDIFFUSION_MODELS:-${PROJECT_ROOT}/data/models/rfdiffusion}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/$(date +%Y-%m-%d)_rfdiffusion_conservative"
NUM_DESIGNS=300

# Export for run scripts
export RFDIFFUSION_OUTPUTS="${OUTPUT_DIR}"
export RFDIFFUSION_MODELS="${MODELS_DIR}"
export RFDIFFUSION_INPUTS="$(dirname ${PROJECT_ROOT}/${INPUT_PDB})"

# Conservative mask: 13 positions
# Keep FAST-PETase's 5 key mutations fixed: S121E, D186H, R224Q, N233K, R280A
# Allow mutation at: N114, L117, Q119, T140, W159, G165, I168, A180, S188, N205, S214, S269, S282
CONSERVATIVE_MASK="A114/A117/A119/A140/A159/A165/A168/A180/A188/A205/A214/A269/A282"

echo "=========================================="
echo "RFdiffusion Conservative Mask Run"
echo "=========================================="
echo "Input PDB: ${INPUT_PDB}"
echo "Models: ${MODELS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Designs: ${NUM_DESIGNS}"
echo "Mask: Conservative (13 positions)"
echo "Estimated time: 6-12 hours"
echo ""

# Check prerequisites
if [ ! -f "${INPUT_PDB}" ]; then
    echo "Error: Input PDB not found: ${INPUT_PDB}"
    exit 1
fi

if [ ! -f "${MODELS_DIR}/Base_ckpt.pt" ]; then
    echo "Error: Model weights not found in ${MODELS_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run RFdiffusion
echo "Starting RFdiffusion conservative mask run..."
echo "This will generate ${NUM_DESIGNS} designs..."
echo ""

if command -v docker &> /dev/null && docker ps &> /dev/null; then
    echo "Using Docker..."
    # Use absolute paths inside container
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
        hydra.job.chdir=False
elif command -v apptainer &> /dev/null || command -v singularity &> /dev/null; then
    echo "Using Singularity/Apptainer..."
    # Use absolute paths inside container
    envs/rfdiffusion/run_singularity.sh \
        inference.output_prefix=/data/outputs/designs \
        inference.model_directory_path=/data/models \
        inference.input_pdb=/data/inputs/$(basename ${INPUT_PDB}) \
        inference.num_designs=${NUM_DESIGNS} \
        'contigmap.contigs=[A1-290]' \
        "contigmap.inpaint_seq=[${CONSERVATIVE_MASK}]" \
        inference.ckpt_override_path=/data/models/ActiveSite_ckpt.pt \
        inference.schedule_directory_path=/data/outputs/schedules \
        hydra.run.dir=/data/outputs \
        hydra.job.chdir=False
else
    echo "Error: Neither Docker nor Singularity/Apptainer found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Conservative Mask Run Complete!"
echo "=========================================="
echo "Results in: ${OUTPUT_DIR}"
echo ""

