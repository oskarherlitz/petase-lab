#!/bin/bash
# Batch FoldX stability scoring for RFdiffusion designs
# Usage: bash scripts/batch_foldx_stability.sh [results_dir] [num_jobs]

set -e

RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
NUM_JOBS="${2:-16}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

echo "=========================================="
echo "Batch FoldX Stability Scoring"
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

# Check if FoldX script exists
if [ ! -f "scripts/foldx_stability.py" ]; then
    echo "Error: FoldX script not found: scripts/foldx_stability.py"
    echo "Create this script to run FoldX on a single PDB"
    exit 1
fi

echo "Finding PDB files..."
PDB_COUNT=$(find "${RESULTS_DIR}" -name "designs_*.pdb" | wc -l | tr -d ' ')
echo "Found ${PDB_COUNT} PDB files"
echo ""

echo "Starting batch FoldX scoring..."
echo "This will take approximately $((PDB_COUNT * 2 / NUM_JOBS)) minutes"
echo ""

# Run FoldX in parallel
find "${RESULTS_DIR}" -name "designs_*.pdb" | \
    parallel -j "${NUM_JOBS}" --progress \
    python scripts/foldx_stability.py {}

echo ""
echo "=========================================="
echo "FoldX Scoring Complete!"
echo "=========================================="
echo ""
echo "Results should be in: ${RESULTS_DIR}/foldx_scores.csv"
echo ""

