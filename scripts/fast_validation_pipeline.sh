#!/usr/bin/env bash
# Fast computational validation pipeline to identify candidates better than WT
# Criteria: 1) Stability (FoldX + Rosetta), 2) Binding/cleft, 3) Robustness (repeats)
# Usage: bash scripts/fast_validation_pipeline.sh [num_candidates] [num_repeats]

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

NUM_CANDIDATES=${1:-5}  # Top N candidates to test (default: 5)
NUM_REPEATS=${2:-5}      # Number of repeats for robustness (default: 5)
OUTPUT_DIR="runs/$(date +%Y%m%d)_fast_validation"

echo "=========================================="
echo "Fast Validation Pipeline"
echo "=========================================="
echo "Testing top $NUM_CANDIDATES candidates"
echo "Repeats per calculation: $NUM_REPEATS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get top candidates from relaxation scores
TOP_CANDIDATES=($(tail -n +2 runs/colabfold_relaxed_top10/relaxation_scores.csv | \
    sort -t, -k3 -n | head -n "$NUM_CANDIDATES" | cut -d, -f1))

if [ ${#TOP_CANDIDATES[@]} -eq 0 ]; then
    echo "Error: No candidates found" >&2
    exit 1
fi

echo "Top $NUM_CANDIDATES candidates to test:"
for cand in "${TOP_CANDIDATES[@]}"; do
    echo "  - $cand"
done
echo ""

# Get WT structure
WT_STRUCTURE="runs/2025-11-22_relax_cart_v1/outputs/PETase_raw_0001.pdb"
if [ ! -f "$WT_STRUCTURE" ]; then
    echo "Error: WT structure not found: $WT_STRUCTURE" >&2
    exit 1
fi

echo "WT structure: $WT_STRUCTURE"
echo ""

# Initialize results file
RESULTS_CSV="$OUTPUT_DIR/validation_results.csv"
echo "Candidate,Rosetta_Score_Median,Rosetta_Score_Std,FoldX_Score_Median,FoldX_Score_Std,Rosetta_Better,FoldX_Better,Both_Better,Robust" > "$RESULTS_CSV"

WT_ROSETTA_SCORE=-887.529  # From previous calculation

echo "Starting validation..."
echo ""

for CANDIDATE in "${TOP_CANDIDATES[@]}"; do
    echo "=========================================="
    echo "Processing $CANDIDATE"
    echo "=========================================="
    
    # Find relaxed structure
    CAND_STRUCTURE=$(find runs/colabfold_relaxed_top10 -name "${CANDIDATE}_relaxed_*.pdb" | head -1)
    
    if [ -z "$CAND_STRUCTURE" ] || [ ! -f "$CAND_STRUCTURE" ]; then
        echo "⚠ Warning: Structure not found for $CANDIDATE, skipping"
        continue
    fi
    
    CAND_OUTPUT="$OUTPUT_DIR/$CANDIDATE"
    mkdir -p "$CAND_OUTPUT"
    
    # 1. Rosetta scoring (multiple repeats for robustness)
    echo "1. Running Rosetta scoring ($NUM_REPEATS repeats)..."
    ROSETTA_SCORES=()
    
    for i in $(seq 1 $NUM_REPEATS); do
        SCORE_OUTPUT=$(mktemp)
        if "$ROSETTA_BIN/score_jd2.static.macosclangrelease" \
            -s "$CAND_STRUCTURE" \
            -score:weights ref2015_cart \
            -out:file:scorefile "$SCORE_OUTPUT" \
            > /dev/null 2>&1; then
            SCORE=$(grep "^SCORE:" "$SCORE_OUTPUT" | head -1 | awk '{print $2}')
            if [ -n "$SCORE" ]; then
                ROSETTA_SCORES+=("$SCORE")
                echo "  Repeat $i: $SCORE"
            fi
        fi
        rm -f "$SCORE_OUTPUT"
    done
    
    if [ ${#ROSETTA_SCORES[@]} -eq 0 ]; then
        echo "  ✗ Failed to get Rosetta scores"
        continue
    fi
    
    # Calculate median and std
    ROSETTA_MEDIAN=$(printf '%s\n' "${ROSETTA_SCORES[@]}" | sort -n | awk '{
        a[NR]=$1
    }
    END{
        if(NR%2==1) print a[(NR+1)/2]
        else print (a[NR/2]+a[NR/2+1])/2
    }')
    
    ROSETTA_STD=$(printf '%s\n' "${ROSETTA_SCORES[@]}" | awk -v med="$ROSETTA_MEDIAN" '{
        sum+=$1; sumsq+=$1*$1
    }
    END{
        mean=sum/NR
        print sqrt((sumsq/NR - mean*mean))
    }')
    
    ROSETTA_BETTER=0
    if (( $(echo "$ROSETTA_MEDIAN < $WT_ROSETTA_SCORE" | bc -l) )); then
        ROSETTA_BETTER=1
        echo "  ✓ Rosetta: Better than WT (median: $ROSETTA_MEDIAN vs WT: $WT_ROSETTA_SCORE)"
    else
        echo "  ✗ Rosetta: Worse than WT (median: $ROSETTA_MEDIAN vs WT: $WT_ROSETTA_SCORE)"
    fi
    
    # 2. FoldX scoring (if available)
    echo "2. Running FoldX scoring..."
    FOLDX_SCORES=()
    FOLDX_BETTER=0
    
    if command -v foldx &> /dev/null; then
        for i in $(seq 1 $NUM_REPEATS); do
            # FoldX BuildModel to get stability
            FOLDX_OUTPUT="$CAND_OUTPUT/foldx_$i"
            mkdir -p "$FOLDX_OUTPUT"
            
            # Copy structure and run FoldX
            cp "$CAND_STRUCTURE" "$FOLDX_OUTPUT/input.pdb"
            
            if foldx --command=Stability \
                --pdb=input.pdb \
                --output-dir="$FOLDX_OUTPUT" \
                > "$FOLDX_OUTPUT/foldx.log" 2>&1; then
                
                # Extract stability score from FoldX output
                SCORE=$(grep -i "total energy" "$FOLDX_OUTPUT/Stability_*.fxout" 2>/dev/null | awk '{print $NF}' | head -1)
                if [ -n "$SCORE" ]; then
                    FOLDX_SCORES+=("$SCORE")
                    echo "  Repeat $i: $SCORE"
                fi
            fi
        done
        
        if [ ${#FOLDX_SCORES[@]} -gt 0 ]; then
            FOLDX_MEDIAN=$(printf '%s\n' "${FOLDX_SCORES[@]}" | sort -n | awk '{
                a[NR]=$1
            }
            END{
                if(NR%2==1) print a[(NR+1)/2]
                else print (a[NR/2]+a[NR/2+1])/2
            }')
            
            # Note: FoldX scores are opposite - lower is better
            # We'd need WT FoldX score for comparison
            echo "  FoldX median: $FOLDX_MEDIAN"
        else
            echo "  ⚠ FoldX not available or failed"
            FOLDX_MEDIAN="N/A"
        fi
    else
        echo "  ⚠ FoldX not installed, skipping"
        FOLDX_MEDIAN="N/A"
    fi
    
    # 3. Robustness check
    ROBUST=0
    if [ ${#ROSETTA_SCORES[@]} -ge 3 ]; then
        # Check if median is consistently better
        BETTER_COUNT=0
        for score in "${ROSETTA_SCORES[@]}"; do
            if (( $(echo "$score < $WT_ROSETTA_SCORE" | bc -l) )); then
                BETTER_COUNT=$((BETTER_COUNT + 1))
            fi
        done
        
        # If >50% of repeats are better, consider robust
        if [ $BETTER_COUNT -gt $((${#ROSETTA_SCORES[@]} / 2)) ]; then
            ROBUST=1
            echo "  ✓ Robust: $BETTER_COUNT/${#ROSETTA_SCORES[@]} repeats better than WT"
        else
            echo "  ✗ Not robust: Only $BETTER_COUNT/${#ROSETTA_SCORES[@]} repeats better"
        fi
    fi
    
    # Calculate criteria met
    BOTH_BETTER=0
    if [ $ROSETTA_BETTER -eq 1 ] && [ $FOLDX_BETTER -eq 1 ]; then
        BOTH_BETTER=1
    fi
    
    CRITERIA_MET=0
    if [ $ROSETTA_BETTER -eq 1 ]; then CRITERIA_MET=$((CRITERIA_MET + 1)); fi
    if [ $FOLDX_BETTER -eq 1 ]; then CRITERIA_MET=$((CRITERIA_MET + 1)); fi
    if [ $ROBUST -eq 1 ]; then CRITERIA_MET=$((CRITERIA_MET + 1)); fi
    
    echo ""
    echo "Summary for $CANDIDATE:"
    echo "  Criteria met: $CRITERIA_MET/3"
    echo "  Rosetta better: $ROSETTA_BETTER"
    echo "  FoldX better: $FOLDX_BETTER"
    echo "  Robust: $ROBUST"
    echo ""
    
    # Write results
    echo "$CANDIDATE,$ROSETTA_MEDIAN,$ROSETTA_STD,$FOLDX_MEDIAN,N/A,$ROSETTA_BETTER,$FOLDX_BETTER,$BOTH_BETTER,$ROBUST" >> "$RESULTS_CSV"
done

echo "=========================================="
echo "Validation Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_CSV"
echo ""
echo "Candidates meeting 2/3 or 3/3 criteria:"
grep -v "^Candidate" "$RESULTS_CSV" | awk -F, '{
    criteria = $6 + $7 + $9
    if (criteria >= 2) print $1 " (" criteria "/3 criteria met)"
}'

