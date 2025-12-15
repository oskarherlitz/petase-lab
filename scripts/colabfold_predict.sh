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
        --num-recycles 3 \
        --num-models 5 \
        --use-amber \
        --use-templates \
        "$FASTA" \
        "$OUTPUT_DIR"
    
    echo ""
    echo "✓ Prediction complete!"
    echo "Results saved to: $OUTPUT_DIR"
    
elif command -v python &> /dev/null; then
    echo "Local ColabFold not found."
    echo ""
    echo "Options:"
    echo "1. Use web interface: https://colabfold.com"
    echo "   - Upload your FASTA file"
    echo "   - Download results"
    echo ""
    echo "2. Install ColabFold locally:"
    echo "   conda install -c conda-forge colabfold"
    echo "   # OR"
    echo "   pip install colabfold"
    echo ""
    echo "3. Use ColabFold via Google Colab:"
    echo "   https://github.com/sokrypton/ColabFold"
    echo ""
    
    # Create a helper script for web interface
    echo "Creating FASTA file info for web upload..."
    echo "File: $FASTA"
    echo "Sequence:"
    grep -v "^>" "$FASTA" | head -1
    echo ""
    echo "Upload this file to: https://colabfold.com"
    
else
    echo "Error: Python not found. Cannot run ColabFold."
    exit 1
fi

