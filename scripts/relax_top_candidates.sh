#!/usr/bin/env bash
# Relax top N candidates from ColabFold predictions
# Usage: bash scripts/relax_top_candidates.sh [nstruct] [top_n] [output_dir]
# Examples:
#   bash scripts/relax_top_candidates.sh 1 10 runs/colabfold_relaxed_top10
#   bash scripts/relax_top_candidates.sh 1 10  # Uses default output dir
#   bash scripts/relax_top_candidates.sh 1      # Uses top 10, default output dir

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

# Configuration - handle arguments more intelligently
# Check if $2 looks like a number (top_n) or a path (output_dir)
if [[ "$#" -ge 2 ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
    # $2 is a number, so it's top_n
    NSTRUCT=${1:-1}
    TOP_N=${2:-10}
    OUTPUT_DIR=${3:-runs/colabfold_relaxed_top${TOP_N}}
elif [[ "$#" -ge 2 ]] && ([[ "$2" =~ ^runs/ ]] || [[ "$2" =~ ^/ ]] || [[ "$2" =~ \.pdb$ ]]); then
    # $2 looks like a path, so user skipped top_n
    NSTRUCT=${1:-1}
    TOP_N=10  # Default to top 10
    OUTPUT_DIR=${2:-runs/colabfold_relaxed_top10}
else
    # Default case
    NSTRUCT=${1:-1}
    TOP_N=${2:-10}
    OUTPUT_DIR=${3:-runs/colabfold_relaxed_top${TOP_N}}
fi

INPUT_DIR="runs/colabfold_predictions_gpu"
RANKING_FILE="$INPUT_DIR/candidate_ranking.txt"

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
echo "Relaxation of Top $TOP_N ColabFold Candidates"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Structures per candidate: $NSTRUCT"
echo "Rosetta binary: $RELAX_BIN"
echo ""

# Get top N candidates from ranking file
if [ ! -f "$RANKING_FILE" ]; then
    echo "Error: Ranking file not found: $RANKING_FILE" >&2
    echo "Please run the ranking script first." >&2
    exit 1
fi

# Extract top N candidate numbers (skip header lines, get actual data rows)
# The ranking file has a header, then data rows starting with candidate numbers
TOP_CANDIDATES=($(grep -E "^[0-9]+[[:space:]]+candidate_" "$RANKING_FILE" | head -n "$TOP_N" | awk '{print $2}' | sed 's/candidate_//' | sort -n))

if [ ${#TOP_CANDIDATES[@]} -eq 0 ]; then
    echo "Error: Could not extract top candidates from ranking file" >&2
    exit 1
fi

echo "Top $TOP_N candidates to relax:"
for i in "${!TOP_CANDIDATES[@]}"; do
    echo "  $((i+1)). candidate_${TOP_CANDIDATES[$i]}"
done
echo ""

# Find PDB files for these candidates
PDB_FILES=()
for cand_num in "${TOP_CANDIDATES[@]}"; do
    # Find rank_001 PDB file for this candidate
    PDB_FILE=$(find "$INPUT_DIR" -name "candidate_${cand_num}_unrelaxed_rank_001_*.pdb" | head -1)
    if [ -n "$PDB_FILE" ] && [ -f "$PDB_FILE" ]; then
        PDB_FILES+=("$PDB_FILE")
    else
        echo "Warning: PDB file not found for candidate_${cand_num}" >&2
    fi
done

TOTAL=${#PDB_FILES[@]}
if [ $TOTAL -eq 0 ]; then
    echo "Error: No PDB files found for top candidates" >&2
    exit 1
fi

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

