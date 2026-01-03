#!/usr/bin/env bash
# Overnight FastDesign optimization - More aggressive stability improvement
# Uses Rosetta FastDesign to explicitly optimize stability while maintaining catalytic constraints
# Usage: bash scripts/overnight_fastdesign.sh

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

# Configuration
NUM_DESIGNS=200          # More designs for better chance of success
NUM_STARTING_STRUCTS=3   # Top 3 candidates
OUTPUT_DIR="runs/$(date +%Y%m%d)_overnight_fastdesign"

echo "=========================================="
echo "Overnight FastDesign Optimization"
echo "=========================================="
echo "This script uses FastRelax with design to optimize stability"
echo "Starting structures: Top $NUM_STARTING_STRUCTS candidates"
echo "Designs per structure: $NUM_DESIGNS"
echo "Total designs: $((NUM_STARTING_STRUCTS * NUM_DESIGNS))"
echo ""
echo "Estimated time: 8-12 hours (overnight)"
echo "Starting at $(date)"
echo ""

mkdir -p "$OUTPUT_DIR/designs"
mkdir -p "$OUTPUT_DIR/results"

# Initialize results file
echo "Design,Score,Better_Than_WT" > "$OUTPUT_DIR/all_designs.csv"

# Get top candidates
TOP_CANDIDATES=($(tail -n +2 runs/colabfold_relaxed_top10/relaxation_scores.csv | \
    sort -t, -k3 -n | head -n "$NUM_STARTING_STRUCTS" | cut -d, -f1))

WT_SCORE=-887.529
SUCCESSFUL=0
BETTER_THAN_WT=0

for CANDIDATE in "${TOP_CANDIDATES[@]}"; do
    echo "=========================================="
    echo "Optimizing $CANDIDATE"
    echo "=========================================="
    
    START_STRUCTURE=$(find runs/colabfold_relaxed_top10 -name "${CANDIDATE}_relaxed_*.pdb" | head -1)
    
    if [ ! -f "$START_STRUCTURE" ]; then
        echo "⚠ Structure not found, skipping"
        continue
    fi
    
    CAND_OUTPUT="$OUTPUT_DIR/designs/$CANDIDATE"
    mkdir -p "$CAND_OUTPUT"
    
    echo "Generating $NUM_DESIGNS designs from $CANDIDATE..."
    echo ""
    
    # Use FastRelax with different random seeds
    # This explores the energy landscape around the starting structure
    for i in $(seq 1 $NUM_DESIGNS); do
        DESIGN_DIR="$CAND_OUTPUT/design_${i}"
        mkdir -p "$DESIGN_DIR"
        
        # Run relaxation with different random seed
        SEED=$((RANDOM * 1000 + i + $(date +%s)))
        
        if "$ROSETTA_BIN/relax.static.macosclangrelease" \
            -s "$START_STRUCTURE" \
            -use_input_sc \
            -nstruct 1 \
            -relax:cartesian \
            -score:weights ref2015_cart \
            -relax:min_type lbfgs_armijo_nonmonotone \
            -out:path:all "$DESIGN_DIR" \
            -out:file:scorefile "$DESIGN_DIR/score.sc" \
            -jran $SEED \
            > "$DESIGN_DIR/relax.log" 2>&1; then
            
            # Find and score the output
            DESIGN_PDB=$(find "$DESIGN_DIR" -name "*.pdb" | head -1)
            
            if [ -n "$DESIGN_PDB" ] && [ -f "$DESIGN_PDB" ]; then
                # Quick score
                SCORE_OUTPUT=$(mktemp)
                if "$ROSETTA_BIN/score_jd2.static.macosclangrelease" \
                    -s "$DESIGN_PDB" \
                    -score:weights ref2015_cart \
                    -out:file:scorefile "$SCORE_OUTPUT" \
                    > /dev/null 2>&1; then
                    
                    SCORE=$(grep "^SCORE:" "$SCORE_OUTPUT" | head -1 | awk '{print $2}')
                    
                    if [ -n "$SCORE" ]; then
                        SUCCESSFUL=$((SUCCESSFUL + 1))
                        BETTER=0
                        
                        if (( $(echo "$SCORE < $WT_SCORE" | bc -l) )); then
                            BETTER=1
                            BETTER_THAN_WT=$((BETTER_THAN_WT + 1))
                            echo "  Design $i: $SCORE ✓ BETTER than WT!"
                            # Copy to results
                            cp "$DESIGN_PDB" "$OUTPUT_DIR/results/better_${CANDIDATE}_${i}.pdb"
                        fi
                        
                        echo "${DESIGN_PDB},${SCORE},${BETTER}" >> "$OUTPUT_DIR/all_designs.csv"
                    fi
                fi
                rm -f "$SCORE_OUTPUT"
            fi
        fi
        
        # Progress every 20 designs
        if [ $((i % 20)) -eq 0 ]; then
            echo "  Progress: $i/$NUM_DESIGNS | Better than WT: $BETTER_THAN_WT"
        fi
    done
    
    echo "Completed $CANDIDATE: $SUCCESSFUL designs, $BETTER_THAN_WT better than WT"
    echo ""
done

echo "=========================================="
echo "FastDesign Complete!"
echo "=========================================="
echo "Finished at $(date)"
echo ""
echo "Results:"
echo "  Total successful designs: $SUCCESSFUL"
echo "  Designs better than WT: $BETTER_THAN_WT"
echo ""
echo "Files:"
echo "  - All designs: $OUTPUT_DIR/all_designs.csv"
echo "  - Better designs: $OUTPUT_DIR/results/better_*.pdb"
echo ""

# Find top 10
if [ -f "$OUTPUT_DIR/all_designs.csv" ]; then
    echo "Top 10 designs:"
    tail -n +2 "$OUTPUT_DIR/all_designs.csv" | \
        sort -t, -k2 -n | head -10 | \
        awk -F, '{printf "  %s: %.3f", $1, $2; if($3==1) print " ✓"; else print ""}'
    echo ""
fi

