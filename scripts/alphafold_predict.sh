#!/usr/bin/env bash
# AlphaFold structure prediction script for Progen2 sequences
# 
# Usage:
#   bash scripts/alphafold_predict.sh <fasta_file> [output_dir] [options]
#
# Example:
#   bash scripts/alphafold_predict.sh runs/run_20251230_progen2_medium_r1_test/candidates/candidates.ranked.fasta

set -euo pipefail

FASTA=${1:-}
OUTPUT_DIR=${2:-runs/$(date +%F)_alphafold}
MAX_SEQUENCES=${3:-10}  # Limit number of sequences to process

if [ -z "$FASTA" ]; then
    echo "Error: Please provide a FASTA file"
    echo "Usage: bash scripts/alphafold_predict.sh <fasta_file> [output_dir] [max_sequences]"
    echo ""
    echo "Example:"
    echo "  bash scripts/alphafold_predict.sh runs/run_20251230_progen2_medium_r1_test/candidates/candidates.ranked.fasta"
    exit 1
fi

if [ ! -f "$FASTA" ]; then
    echo "Error: FASTA file not found: $FASTA"
    exit 1
fi

echo "AlphaFold Structure Prediction"
echo "=============================="
echo "Input FASTA: $FASTA"
echo "Output directory: $OUTPUT_DIR"
echo "Max sequences: $MAX_SEQUENCES"
echo ""

# Check if AlphaFold is available
ALPHAFOLD_MODE="none"

# Check for AlphaFold Docker
if command -v docker &> /dev/null && docker ps &> /dev/null; then
    if docker images | grep -q alphafold; then
        ALPHAFOLD_MODE="docker"
        echo "✓ Found AlphaFold Docker image"
    fi
fi

# Check for local AlphaFold installation
if [ -d "${ALPHAFOLD_PATH:-}" ] && [ -f "${ALPHAFOLD_PATH}/run_alphafold.py" ]; then
    ALPHAFOLD_MODE="local"
    echo "✓ Found local AlphaFold installation at: $ALPHAFOLD_PATH"
elif [ -d "/opt/alphafold" ] && [ -f "/opt/alphafold/run_alphafold.py" ]; then
    ALPHAFOLD_MODE="local"
    ALPHAFOLD_PATH="/opt/alphafold"
    echo "✓ Found local AlphaFold installation at: $ALPHAFOLD_PATH"
fi

# Check for ColabFold as fallback
if command -v colabfold_batch &> /dev/null; then
    if [ "$ALPHAFOLD_MODE" == "none" ]; then
        echo "⚠ AlphaFold not found, but ColabFold is available"
        echo "  ColabFold is a faster alternative with similar accuracy"
        echo "  Consider using: bash scripts/colabfold_predict.sh $FASTA"
        echo ""
        read -p "Use ColabFold instead? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            bash scripts/colabfold_predict.sh "$FASTA" "$OUTPUT_DIR"
            exit 0
        fi
    fi
fi

if [ "$ALPHAFOLD_MODE" == "none" ]; then
    echo "❌ AlphaFold not found"
    echo ""
    echo "Options:"
    echo ""
    echo "1. Use ColabFold (recommended, easier):"
    echo "   bash scripts/colabfold_predict.sh $FASTA"
    echo ""
    echo "2. Install AlphaFold via Docker:"
    echo "   See: https://github.com/deepmind/alphafold#running-alphafold"
    echo "   Requires: Docker, ~3TB disk space for databases"
    echo ""
    echo "3. Install AlphaFold locally:"
    echo "   See: https://github.com/deepmind/alphafold#installation-and-running-your-first-prediction"
    echo "   Requires: Python, CUDA, ~3TB disk space"
    echo ""
    echo "4. Use AlphaFold via Google Colab:"
    echo "   https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb"
    echo ""
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count sequences in FASTA
NUM_SEQUENCES=$(grep -c "^>" "$FASTA" || echo "0")
echo "Found $NUM_SEQUENCES sequences in FASTA file"

if [ "$NUM_SEQUENCES" -gt "$MAX_SEQUENCES" ]; then
    echo "⚠ Limiting to first $MAX_SEQUENCES sequences (use 3rd argument to change)"
    # Create temporary FASTA with limited sequences
    TEMP_FASTA="${OUTPUT_DIR}/input_limited.fasta"
    awk -v max="$MAX_SEQUENCES" '
        /^>/ { count++; if (count > max) exit }
        { print }
    ' "$FASTA" > "$TEMP_FASTA"
    FASTA="$TEMP_FASTA"
    NUM_SEQUENCES=$MAX_SEQUENCES
fi

echo ""
echo "Processing $NUM_SEQUENCES sequences..."
echo ""

# Run AlphaFold based on mode
if [ "$ALPHAFOLD_MODE" == "docker" ]; then
    echo "Running AlphaFold via Docker..."
    echo ""
    
    # Load AlphaFold config
    CONFIG_FILE="configs/alphafold.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠ Warning: configs/alphafold.yaml not found"
        echo "  Please update it with your database paths"
        echo ""
    fi
    
    # Docker command (adjust based on your AlphaFold Docker setup)
    docker run \
        --rm \
        -v "$(pwd):/data" \
        -v "${ALPHAFOLD_DB_PATH:-/data/alphafold_db}:/alphafold_db" \
        alphafold \
        python /app/run_alphafold.py \
        --fasta_paths="/data/$FASTA" \
        --output_dir="/data/$OUTPUT_DIR" \
        --data_dir="/alphafold_db" \
        --max_template_date=2024-01-01 \
        --model_preset=monomer \
        --db_preset=full_dbs
    
elif [ "$ALPHAFOLD_MODE" == "local" ]; then
    echo "Running AlphaFold locally..."
    echo ""
    
    # Load config
    CONFIG_FILE="configs/alphafold.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        # Extract database paths from config (requires yq or manual parsing)
        echo "Using config: $CONFIG_FILE"
    else
        echo "⚠ Warning: configs/alphafold.yaml not found"
        echo "  Please update it with your database paths"
    fi
    
    # Run AlphaFold
    python "${ALPHAFOLD_PATH}/run_alphafold.py" \
        --fasta_paths="$FASTA" \
        --output_dir="$OUTPUT_DIR" \
        --data_dir="${ALPHAFOLD_DB_PATH:-/data/alphafold_db}" \
        --max_template_date=2024-01-01 \
        --model_preset=monomer \
        --db_preset=full_dbs
fi

echo ""
echo "✓ AlphaFold prediction complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - *_ranked_*.pdb: Ranked structure models (rank 1 is best)"
echo "  - *_plddt.png: Per-residue confidence scores"
echo "  - *_pae.png: Predicted aligned error"
echo "  - *_scores.json: Model confidence scores"
echo ""

