#!/usr/bin/env bash
# ColabFold structure prediction script
# 
# Usage:
#   bash scripts/colabfold_predict.sh <fasta_file> [output_dir]
#
# Example:
#   bash scripts/colabfold_predict.sh data/sequences/design_001.fasta

set -euo pipefail

FASTA=${1:-}
OUTPUT_DIR=${2:-runs/$(date +%F)_colabfold}

if [ -z "$FASTA" ]; then
    echo "Error: Please provide a FASTA file"
    echo "Usage: bash scripts/colabfold_predict.sh <fasta_file> [output_dir]"
    exit 1
fi

if [ ! -f "$FASTA" ]; then
    echo "Error: FASTA file not found: $FASTA"
    exit 1
fi

echo "Running ColabFold prediction..."
echo "Input: $FASTA"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if colabfold_batch is available locally
if command -v colabfold_batch &> /dev/null; then
    echo "Using local ColabFold installation..."
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
    
elif command -v python &> /dev/null; then
    echo "Local ColabFold not found."
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  QUICK SETUP OPTIONS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Option 1: WEB INTERFACE (Easiest - No installation!)"
    echo "  → Go to: https://colabfold.com"
    echo "  → Upload your FASTA file: $FASTA"
    echo "  → Click 'Search' and wait 5-30 minutes"
    echo "  → Download results"
    echo ""
    echo "Option 2: INSTALL LOCALLY (For batch processing)"
    echo "  → Run setup script: bash scripts/setup_colabfold.sh"
    echo "  → Or install manually: pip install colabfold"
    echo "  → Then rerun this script"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Show FASTA file info for web upload
    echo "For web interface, your FASTA file is ready:"
    echo "  File: $FASTA"
    NUM_SEQS=$(grep -c "^>" "$FASTA" 2>/dev/null || echo "?")
    echo "  Sequences: $NUM_SEQS"
    if [ "$NUM_SEQS" -le 5 ]; then
        echo ""
        echo "First sequence preview:"
        grep -v "^>" "$FASTA" | head -1 | cut -c1-80
        echo "..."
    fi
    echo ""
    echo "Upload to: https://colabfold.com"
    
else
    echo "Error: Python not found. Cannot run ColabFold."
    exit 1
fi

