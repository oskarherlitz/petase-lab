#!/usr/bin/env bash
# Re-run ColabFold without templates (to avoid hhsearch error)

set -euo pipefail

FASTA=${1:-runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta}
OUTPUT_DIR=${2:-runs/colabfold_predictions_no_templates}

if [ ! -f "$FASTA" ]; then
    echo "Error: FASTA file not found: $FASTA"
    exit 1
fi

echo "Re-running ColabFold WITHOUT templates..."
echo "Input: $FASTA"
echo "Output: $OUTPUT_DIR"
echo ""
echo "This will generate structures without template search (MSA-only mode)"
echo ""

mkdir -p "$OUTPUT_DIR"

colabfold_batch \
    --num-recycle 3 \
    --num-models 5 \
    --amber \
    "$FASTA" \
    "$OUTPUT_DIR"

echo ""
echo "✓ Prediction complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "PDB files should now be in: $OUTPUT_DIR/"

