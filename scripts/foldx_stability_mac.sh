#!/bin/bash
# FoldX stability scoring for RFdiffusion designs on Mac
# Runs in parallel using all available CPU cores
# Usage: bash scripts/foldx_stability_mac.sh [results_dir] [num_jobs]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-${PROJECT_ROOT}/runs/2026-01-03_rfdiffusion_conservative}"
NUM_JOBS="${2:-$(sysctl -n hw.ncpu)}"

cd "${PROJECT_ROOT}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

echo "=========================================="
echo "FoldX Stability Scoring (Mac)"
echo "=========================================="
echo "Directory: ${RESULTS_DIR}"
echo "Parallel jobs: ${NUM_JOBS}"
echo ""

# Check if FoldX is available
if ! command -v FoldX &> /dev/null && [ ! -f "${FOLDX_PATH:-/opt/foldx/FoldX}" ]; then
    echo "Error: FoldX not found."
    echo ""
    echo "Install FoldX:"
    echo "  1. Download from: https://foldxsuite.org.eu/"
    echo "  2. Extract to /opt/foldx/"
    echo "  3. Set FOLDX_PATH environment variable:"
    echo "     export FOLDX_PATH=/opt/foldx/FoldX"
    echo ""
    exit 1
fi

FOLDX_CMD="${FOLDX_PATH:-FoldX}"

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    echo "Installing GNU parallel..."
    if command -v brew &> /dev/null; then
        brew install parallel
    else
        echo "Error: GNU parallel not found. Install with:"
        echo "  brew install parallel"
        exit 1
    fi
fi

echo "Finding PDB files..."
PDB_FILES=($(find "${RESULTS_DIR}" -name "designs_*.pdb" | sort))
PDB_COUNT=${#PDB_FILES[@]}

if [ "${PDB_COUNT}" -eq 0 ]; then
    echo "Error: No PDB files found in ${RESULTS_DIR}"
    exit 1
fi

echo "Found ${PDB_COUNT} PDB files"
echo ""

# Create output directory
OUTPUT_DIR="${RESULTS_DIR}/foldx_scores"
mkdir -p "${OUTPUT_DIR}"

# Create temporary directory for FoldX work
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: ${TEMP_DIR}"
echo ""

# Estimate time
ESTIMATED_MIN=$((PDB_COUNT * 2 / NUM_JOBS))
echo "Estimated time: ~${ESTIMATED_MIN} minutes (~$((ESTIMATED_MIN / 60)) hours)"
echo ""

# Function to run FoldX on a single PDB
run_foldx() {
    local pdb_file="$1"
    local design_name=$(basename "${pdb_file}" .pdb)
    local output_file="${OUTPUT_DIR}/${design_name}_foldx.txt"
    
    # Skip if already done
    if [ -f "${output_file}" ]; then
        echo "Skipping ${design_name} (already done)"
        return 0
    fi
    
    # Create temp directory for this design
    local design_temp="${TEMP_DIR}/${design_name}"
    mkdir -p "${design_temp}"
    
    # Copy PDB to temp directory
    cp "${pdb_file}" "${design_temp}/input.pdb"
    
    # Run FoldX Stability
    cd "${design_temp}"
    "${FOLDX_CMD}" --command=Stability \
        --pdb=input.pdb \
        --output-dir="${design_temp}" \
        > "${output_file}.log" 2>&1
    
    # Find FoldX output file
    local foldx_output=$(find "${design_temp}" -name "Stability_*.fxout" 2>/dev/null | head -1)
    
    if [ -f "${foldx_output}" ]; then
        # Copy output to results directory
        cp "${foldx_output}" "${output_file}"
        
        # Extract total energy (usually last column of first data line)
        ENERGY=$(grep -v "^#" "${foldx_output}" | grep -v "^$" | head -1 | awk '{print $NF}' || echo "N/A")
        echo "${design_name}: ${ENERGY}"
    else
        echo "${design_name}: FAILED (check ${output_file}.log)"
    fi
    
    # Cleanup
    rm -rf "${design_temp}"
}

export -f run_foldx
export FOLDX_CMD OUTPUT_DIR TEMP_DIR

echo "Starting FoldX stability scoring..."
echo ""

# Run FoldX in parallel
printf '%s\n' "${PDB_FILES[@]}" | \
    parallel -j "${NUM_JOBS}" --progress run_foldx {}

# Cleanup temp directory
rm -rf "${TEMP_DIR}"

# Compile results into CSV
echo ""
echo "Compiling results..."
CSV_FILE="${OUTPUT_DIR}/foldx_scores.csv"
echo "design,foldx_stability" > "${CSV_FILE}"

for pdb_file in "${PDB_FILES[@]}"; do
    design_name=$(basename "${pdb_file}" .pdb)
    output_file="${OUTPUT_DIR}/${design_name}_foldx.txt"
    
    if [ -f "${output_file}" ]; then
        # Try to extract stability value
        ENERGY=$(grep -E "Total|ΔΔG|stability" "${output_file}" 2>/dev/null | head -1 | awk '{print $NF}' || echo "N/A")
        echo "${design_name},${ENERGY}" >> "${CSV_FILE}"
    else
        echo "${design_name},FAILED" >> "${CSV_FILE}"
    fi
done

echo ""
echo "=========================================="
echo "FoldX Scoring Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${CSV_FILE}"
echo ""
echo "Top 10 most stable designs:"
sort -t',' -k2 -n "${CSV_FILE}" | head -11
echo ""

