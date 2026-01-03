#!/bin/bash
# RFdiffusion Direct Run (No Container)
# For use when Docker/Singularity aren't available (e.g., RunPod without Docker-in-Docker)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuration
INPUT_PDB="${1:-data/structures/7SH6/raw/7SH6.pdb}"
MODELS_DIR="${RFDIFFUSION_MODELS:-${PROJECT_ROOT}/data/models/rfdiffusion}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/$(date +%Y-%m-%d)_rfdiffusion_test"
NUM_DESIGNS="${2:-5}"

# Conservative mask: 13 positions
CONSERVATIVE_MASK="A114/A117/A119/A140/A159/A165/A168/A180/A188/A205/A214/A269/A282"

echo "=========================================="
echo "RFdiffusion Direct Run (No Container)"
echo "=========================================="
echo "Input PDB: ${INPUT_PDB}"
echo "Models: ${MODELS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Designs: ${NUM_DESIGNS}"
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

# Check if RFdiffusion is available
RFDIFFUSION_DIR="${PROJECT_ROOT}/external/rfdiffusion"
if [ ! -f "${RFDIFFUSION_DIR}/scripts/run_inference.py" ]; then
    echo "Error: RFdiffusion not found at ${RFDIFFUSION_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/schedules"

# Set environment
export PYTHONPATH="${RFDIFFUSION_DIR}:${PYTHONPATH}"
export DGLBACKEND="pytorch"

# Run RFdiffusion directly
echo "Running RFdiffusion directly (no container)..."
echo ""

cd "${RFDIFFUSION_DIR}"

python3 scripts/run_inference.py \
    inference.output_prefix="${OUTPUT_DIR}/designs" \
    inference.model_directory_path="${MODELS_DIR}" \
    inference.input_pdb="${PROJECT_ROOT}/${INPUT_PDB}" \
    inference.num_designs=${NUM_DESIGNS} \
    'contigmap.contigs=[A1-290]' \
    "contigmap.inpaint_seq=[${CONSERVATIVE_MASK}]" \
    inference.ckpt_override_path="${MODELS_DIR}/ActiveSite_ckpt.pt" \
    inference.schedule_directory_path="${OUTPUT_DIR}/schedules" \
    hydra.run.dir="${OUTPUT_DIR}" \
    hydra.job.chdir=False

echo ""
echo "=========================================="
echo "Run Complete!"
echo "=========================================="
echo "Results in: ${OUTPUT_DIR}"
echo ""

