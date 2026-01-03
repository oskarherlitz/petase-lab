#!/usr/bin/env bash
# Batch relax ALL ColabFold candidate structures
# Usage: bash scripts/relax_all_candidates.sh [nstruct] [output_dir]
# 
# NOTE: For top N candidates only, use: bash scripts/relax_top_candidates.sh

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

# Configuration
NSTRUCT=${1:-1}  # Number of structures to generate (default: 1 for speed)
OUTPUT_DIR=${2:-runs/colabfold_relaxed}
INPUT_DIR="runs/colabfold_predictions_gpu"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Detect Rosetta binary
RELAX_BIN=""
if [[ -f "$ROSETTA_BIN/relax.static.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/relax.static.macosclangrelease" ]]; then
    RELAX_BIN="relax.static.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/relax.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/relax.macosclangrelease" ]]; then
    RELAX_BIN="relax.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/relax.linuxgccrelease" ]] && [[ -x "$ROSETTA_BIN/relax.linuxgccrelease" ]]; then
    RELAX_BIN="relax.linuxgccrelease"
else
    echo "Error: No executable Rosetta relax binary found in $ROSETTA_BIN" >&2
    exit 1
fi

echo "=========================================="
echo "Batch Relaxation of ColabFold Candidates"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Structures per candidate: $NSTRUCT"
echo "Rosetta binary: $RELAX_BIN"
echo ""

# Find all rank_001 PDB files (best model for each candidate)
PDB_FILES=($(find "$INPUT_DIR" -name "*_rank_001_*.pdb" | sort))

if [ ${#PDB_FILES[@]} -eq 0 ]; then
    echo "Error: No PDB files found in $INPUT_DIR" >&2
    exit 1
fi

TOTAL=${#PDB_FILES[@]}
echo "Found $TOTAL structures to relax"
echo ""

# Estimate time
if [ "$NSTRUCT" -eq 1 ]; then
    EST_MIN=$((TOTAL * 5))
    EST_MAX=$((TOTAL * 15))
    echo "Estimated time: $EST_MIN-$EST_MAX minutes ($((EST_MIN/60))-$((EST_MAX/60)) hours)"
elif [ "$NSTRUCT" -eq 5 ]; then
    EST_MIN=$((TOTAL * 10))
    EST_MAX=$((TOTAL * 30))
    echo "Estimated time: $EST_MIN-$EST_MAX minutes ($((EST_MIN/60))-$((EST_MAX/60)) hours)"
else
    EST_MIN=$((TOTAL * 30))
    EST_MAX=$((TOTAL * 120))
    echo "Estimated time: $EST_MIN-$EST_MAX minutes ($((EST_MIN/60))-$((EST_MAX/60)) hours)"
fi
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting relaxation..."
echo ""

# Process each structure
SUCCESS=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!PDB_FILES[@]}"; do
    PDB_FILE="${PDB_FILES[$i]}"
    CANDIDATE=$(basename "$PDB_FILE" | sed 's/_unrelaxed_rank_001.*//')
    NUM=$((i + 1))
    
    echo "[$NUM/$TOTAL] Relaxing $CANDIDATE..."
    
    # Create candidate-specific output directory
    CAND_OUTPUT="$OUTPUT_DIR/$CANDIDATE"
    mkdir -p "$CAND_OUTPUT"
    
    # Run Rosetta relaxation
    if "$ROSETTA_BIN/$RELAX_BIN" \
        -s "$PDB_FILE" \
        -use_input_sc \
        -nstruct "$NSTRUCT" \
        -relax:cartesian \
        -score:weights ref2015_cart \
        -relax:min_type lbfgs_armijo_nonmonotone \
        -out:path:all "$CAND_OUTPUT" \
        -out:file:scorefile "$CAND_OUTPUT/score.sc" \
        > "$CAND_OUTPUT/relax.log" 2>&1; then
        echo "  ✓ Success"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed (check $CAND_OUTPUT/relax.log)"
        FAILED=$((FAILED + 1))
    fi
    
    # Show progress
    ELAPSED=$(($(date +%s) - START_TIME))
    AVG_TIME=$((ELAPSED / NUM))
    REMAINING=$((AVG_TIME * (TOTAL - NUM)))
    echo "  Progress: $NUM/$TOTAL | Elapsed: $((ELAPSED/60)) min | Est. remaining: $((REMAINING/60)) min"
    echo ""
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "Relaxation Complete!"
echo "=========================================="
echo "Total time: $((TOTAL_TIME/60)) minutes ($((TOTAL_TIME/3600)) hours)"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "To find the best relaxed structure for each candidate:"
echo "  ls -lh $OUTPUT_DIR/*/score.sc"
echo "  (Lowest score = best structure)"

