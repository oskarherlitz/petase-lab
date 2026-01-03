#!/usr/bin/env bash
# ColabFold run script for RunPod
# Assumes you're in the petase-lab directory

set -euo pipefail

FASTA=${1:-runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta}
OUTPUT_DIR=${2:-runs/colabfold_predictions_gpu}

# Check if FASTA exists
if [ ! -f "$FASTA" ]; then
    echo "Error: FASTA file not found: $FASTA"
    echo ""
    echo "Current directory: $(pwd)"
    echo ""
    echo "Looking for FASTA files..."
    find . -name "*.fasta" -type f 2>/dev/null | head -10
    echo ""
    echo "Please provide correct path to FASTA file"
    exit 1
fi

echo "Running ColabFold on RunPod..."
echo "Input: $FASTA"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run ColabFold (without templates to avoid hhsearch issue)
colabfold_batch \
    --num-recycle 3 \
    --num-models 5 \
    --amber \
    "$FASTA" \
    "$OUTPUT_DIR"

echo ""
echo "✓ Prediction complete!"
echo "Results saved to: $OUTPUT_DIR"

