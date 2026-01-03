#!/usr/bin/env bash
# Extract Rosetta scores from relaxed PDB files
# Usage: bash scripts/extract_relaxation_scores.sh [relaxed_dir]

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

RELAXED_DIR=${1:-runs/colabfold_relaxed_top10}
OUTPUT_CSV="${RELAXED_DIR}/relaxation_scores.csv"

echo "=========================================="
echo "Extracting Relaxation Scores"
echo "=========================================="
echo "Input directory: $RELAXED_DIR"
echo "Output: $OUTPUT_CSV"
echo ""

# Detect Rosetta binary
SCORE_BIN=""
if [[ -f "$ROSETTA_BIN/score_jd2.static.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/score_jd2.static.macosclangrelease" ]]; then
    SCORE_BIN="score_jd2.static.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/score_jd2.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/score_jd2.macosclangrelease" ]]; then
    SCORE_BIN="score_jd2.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/score_jd2.linuxgccrelease" ]] && [[ -x "$ROSETTA_BIN/score_jd2.linuxgccrelease" ]]; then
    SCORE_BIN="score_jd2.linuxgccrelease"
else
    echo "Error: No executable Rosetta score_jd2 binary found in $ROSETTA_BIN" >&2
    exit 1
fi

echo "Rosetta binary: $SCORE_BIN"
echo ""

# Find all relaxed PDB files
PDB_FILES=($(find "$RELAXED_DIR" -name "*.pdb" -type f | sort))

if [ ${#PDB_FILES[@]} -eq 0 ]; then
    echo "Error: No PDB files found in $RELAXED_DIR" >&2
    exit 1
fi

echo "Found ${#PDB_FILES[@]} PDB files"
echo ""

# Create CSV header
echo "Candidate,PDB_File,Total_Score" > "$OUTPUT_CSV"

# Score each PDB file
SUCCESS=0
FAILED=0

for PDB_FILE in "${PDB_FILES[@]}"; do
    CANDIDATE=$(basename "$PDB_FILE" | sed 's/_unrelaxed_rank_001.*//' | sed 's/_0001\.pdb$//')
    
    echo "Scoring $CANDIDATE..."
    
    # Run score_jd2 to get scores
    SCORE_OUTPUT=$(mktemp)
    if "$ROSETTA_BIN/$SCORE_BIN" \
        -s "$PDB_FILE" \
        -score:weights ref2015_cart \
        -out:file:scorefile "$SCORE_OUTPUT" \
        > /dev/null 2>&1; then
        
        # Extract total_score from score file
        TOTAL_SCORE=$(grep "^SCORE:" "$SCORE_OUTPUT" | head -1 | awk '{print $2}')
        
        if [ -n "$TOTAL_SCORE" ]; then
            echo "$CANDIDATE,$(basename "$PDB_FILE"),$TOTAL_SCORE" >> "$OUTPUT_CSV"
            echo "  ✓ Score: $TOTAL_SCORE"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  ⚠ Could not extract score"
            FAILED=$((FAILED + 1))
        fi
        
        rm -f "$SCORE_OUTPUT"
    else
        echo "  ✗ Failed to score"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Score Extraction Complete!"
echo "=========================================="
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "Output: $OUTPUT_CSV"
echo ""
echo "To view scores sorted by total_score (lowest = best):"
echo "  sort -t, -k3 -n $OUTPUT_CSV | head -20"

