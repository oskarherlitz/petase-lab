#!/bin/bash
# RFdiffusion Aggressive Mask Overnight Run
# ~300 designs with aggressive mask (18-20 positions)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuration
INPUT_PDB="${1:-data/structures/7SH6/raw/7SH6.pdb}"
MODELS_DIR="${RFDIFFUSION_MODELS:-${PROJECT_ROOT}/data/models/rfdiffusion}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/$(date +%Y-%m-%d)_rfdiffusion_aggressive"
NUM_DESIGNS=300

# Export for run scripts
export RFDIFFUSION_OUTPUTS="${OUTPUT_DIR}"
export RFDIFFUSION_MODELS="${MODELS_DIR}"
export RFDIFFUSION_INPUTS="$(dirname ${PROJECT_ROOT}/${INPUT_PDB})"

# Aggressive mask: 18-20 positions
# Includes all conservative positions PLUS FAST-PETase key positions
# Conservative: N114, L117, Q119, T140, W159, G165, I168, A180, S188, N205, S214, S269, S282
# Additional: S121, D186, R224, N233, R280
AGGRESSIVE_MASK="A114/A117/A119/A121/A140/A159/A165/A168/A180/A186/A188/A205/A214/A224/A233/A269/A280/A282"

echo "=========================================="
echo "RFdiffusion Aggressive Mask Run"
echo "=========================================="
echo "Input PDB: ${INPUT_PDB}"
echo "Models: ${MODELS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Designs: ${NUM_DESIGNS}"
echo "Mask: Aggressive (18 positions)"
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

# Fix CUDA library path for DGL
CUDA_LIB_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-11.8/targets/x86_64-linux/lib"
    "/usr/local/cuda-11.6/lib64"
    "/usr/local/cuda-12.4/lib64"
)
for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "${path}" ] && [ -f "${path}/libcudart.so"* ] 2>/dev/null; then
        export LD_LIBRARY_PATH="${path}:${LD_LIBRARY_PATH}"
        break
    fi
done
# Fallback: search for libcudart
if [ -z "${LD_LIBRARY_PATH##*cuda*}" ]; then
    CUDA_LIB=$(find /usr -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -n "${CUDA_LIB}" ]; then
        export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    fi
fi

# Run RFdiffusion directly
echo "Starting RFdiffusion aggressive mask run..."
echo "This will generate ${NUM_DESIGNS} designs..."
echo ""

cd "${RFDIFFUSION_DIR}"

python3 scripts/run_inference.py \
    inference.output_prefix="${OUTPUT_DIR}/designs" \
    inference.model_directory_path="${MODELS_DIR}" \
    inference.input_pdb="${PROJECT_ROOT}/${INPUT_PDB}" \
    inference.num_designs=${NUM_DESIGNS} \
    'contigmap.contigs=[A1-290]' \
    "contigmap.inpaint_seq=[${AGGRESSIVE_MASK}]" \
    inference.ckpt_override_path="${MODELS_DIR}/ActiveSite_ckpt.pt" \
    inference.schedule_directory_path="${OUTPUT_DIR}/schedules" \
    hydra.run.dir="${OUTPUT_DIR}" \
    hydra.job.chdir=False

echo ""
echo "=========================================="
echo "Aggressive Mask Run Complete!"
echo "=========================================="
echo "Results in: ${OUTPUT_DIR}"
echo ""

