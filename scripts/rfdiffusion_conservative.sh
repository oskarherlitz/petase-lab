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
    "/usr/local/nvidia/lib64"
    "/usr/local/nvidia/lib"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-11.8/targets/x86_64-linux/lib"
    "/usr/local/cuda-11.8/lib64"
    "/usr/local/cuda-11.6/lib64"
    "/usr/local/cuda-12.4/lib64"
    "/usr/lib/x86_64-linux-gnu"
)
for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "${path}" ] && find "${path}" -name "libcudart.so*" 2>/dev/null | grep -q .; then
        export LD_LIBRARY_PATH="${path}:${LD_LIBRARY_PATH}"
    fi
done
# Fallback: search for libcudart
if [ -z "${LD_LIBRARY_PATH##*cuda*}" ] && [ -z "${LD_LIBRARY_PATH##*nvidia*}" ]; then
    CUDA_LIB=$(find /usr /usr/local -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -n "${CUDA_LIB}" ]; then
        export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    fi
fi

# Run RFdiffusion directly
echo "Starting RFdiffusion conservative mask run..."
echo "This will generate ${NUM_DESIGNS} designs..."
echo ""

cd "${RFDIFFUSION_DIR}"

# Resolve input PDB path (handle both absolute and relative paths)
if [[ "${INPUT_PDB}" == /* ]]; then
    # Already absolute path
    INPUT_PDB_ABS="${INPUT_PDB}"
else
    # Relative path, make it absolute
    INPUT_PDB_ABS="${PROJECT_ROOT}/${INPUT_PDB}"
fi

python3 scripts/run_inference.py \
    inference.output_prefix="${OUTPUT_DIR}/designs" \
    inference.model_directory_path="${MODELS_DIR}" \
    inference.input_pdb="${INPUT_PDB_ABS}" \
    inference.num_designs=${NUM_DESIGNS} \
    'contigmap.contigs=[A29-289]' \
    "contigmap.inpaint_seq=[${CONSERVATIVE_MASK}]" \
    inference.ckpt_override_path="${MODELS_DIR}/ActiveSite_ckpt.pt" \
    inference.schedule_directory_path="${OUTPUT_DIR}/schedules" \
    hydra.run.dir="${OUTPUT_DIR}" \
    hydra.job.chdir=False

echo ""
echo "=========================================="
echo "Conservative Mask Run Complete!"
echo "=========================================="
echo "Results in: ${OUTPUT_DIR}"
echo ""

