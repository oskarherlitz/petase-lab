#!/bin/bash
# Batch Rosetta ΔΔG scoring for RFdiffusion designs
# Usage: bash scripts/batch_rosetta_ddg.sh [results_dir] [num_jobs]

set -e

RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
NUM_JOBS="${2:-8}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

echo "=========================================="
echo "Batch Rosetta ΔΔG Scoring"
echo "=========================================="
echo "Directory: ${RESULTS_DIR}"
echo "Parallel jobs: ${NUM_JOBS}"
echo ""

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel not found. Install with:"
    echo "  brew install parallel  # macOS"
    echo "  apt-get install parallel  # Linux"
    exit 1
fi

echo "Finding PDB files..."
PDB_COUNT=$(find "${RESULTS_DIR}" -name "designs_*.pdb" | wc -l | tr -d ' ')
echo "Found ${PDB_COUNT} PDB files"
echo ""

echo "Starting batch Rosetta scoring..."
echo "This will take approximately $((PDB_COUNT * 15 / NUM_JOBS / 60)) hours"
echo ""

# Run Rosetta in parallel
find "${RESULTS_DIR}" -name "designs_*.pdb" | \
    parallel -j "${NUM_JOBS}" --progress \
    bash scripts/rosetta_ddg.sh {}

echo ""
echo "=========================================="
echo "Rosetta Scoring Complete!"
echo "=========================================="
echo ""
echo "Results should be in: ${RESULTS_DIR}/rosetta_ddg/"
echo ""

