#!/bin/bash
# Batch ColabFold validation for RFdiffusion designs
# Usage: bash scripts/batch_colabfold_validation.sh [results_dir] [output_dir] [max_designs]

set -e

RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
OUTPUT_DIR="${2:-runs/alphafold_validation_conservative}"
MAX_DESIGNS="${3:-50}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

echo "=========================================="
echo "Batch ColabFold Validation"
echo "=========================================="
echo "Input directory: ${RESULTS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Max designs: ${MAX_DESIGNS}"
echo ""

# Check if ColabFold is available
if ! command -v colabfold_batch &> /dev/null; then
    echo "Error: ColabFold not found. Install with:"
    echo "  pip install colabfold[alphafold]"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ GPU detected"
    USE_GPU="--use-gpu-relax"
else
    echo "⚠ No GPU detected - will run on CPU (much slower)"
    USE_GPU=""
fi

# Create temporary directory with selected PDBs
TEMP_DIR=$(mktemp -d)
echo "Creating temporary directory: ${TEMP_DIR}"
echo ""

# Select PDBs (first N, or random sample)
echo "Selecting ${MAX_DESIGNS} designs..."
if [ "${MAX_DESIGNS}" -eq 0 ] || [ "${MAX_DESIGNS}" -ge 300 ]; then
    # Use all designs
    find "${RESULTS_DIR}" -name "designs_*.pdb" -exec cp {} "${TEMP_DIR}/" \;
else
    # Random sample
    find "${RESULTS_DIR}" -name "designs_*.pdb" | shuf -n "${MAX_DESIGNS}" | \
        xargs -I {} cp {} "${TEMP_DIR}/"
fi

PDB_COUNT=$(ls -1 "${TEMP_DIR}"/*.pdb 2>/dev/null | wc -l | tr -d ' ')
echo "Selected ${PDB_COUNT} designs"
echo ""

# Estimate time
if [ -n "${USE_GPU}" ]; then
    ESTIMATED_MIN=$((PDB_COUNT * 3))
    echo "Estimated time: ~${ESTIMATED_MIN} minutes (GPU)"
else
    ESTIMATED_MIN=$((PDB_COUNT * 30))
    echo "Estimated time: ~${ESTIMATED_MIN} minutes (CPU)"
fi
echo ""

# Run ColabFold
echo "Running ColabFold..."
mkdir -p "${OUTPUT_DIR}"

colabfold_batch \
    --num-recycle 3 \
    --num-models 1 \
    --templates \
    ${USE_GPU} \
    "${TEMP_DIR}"/*.pdb \
    "${OUTPUT_DIR}"

# Cleanup
rm -rf "${TEMP_DIR}"

echo ""
echo "=========================================="
echo "ColabFold Validation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Check pLDDT scores (higher is better)"
echo "  2. Compare RMSD to RFdiffusion structures"
echo "  3. Extract sequences from AlphaFold outputs"
echo ""

