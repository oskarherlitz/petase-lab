#!/usr/bin/env bash
# Quick script to install ColabFold in a conda environment

set -euo pipefail

echo "ColabFold Environment Setup"
echo "============================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Install conda first or use: pip install colabfold"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "petase-colabfold"; then
    echo "Environment 'petase-colabfold' already exists!"
    echo ""
    read -p "Activate and install ColabFold? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Activating environment..."
        eval "$(conda shell.bash hook)"
        conda activate petase-colabfold
        echo "Installing ColabFold..."
        pip install colabfold
        echo ""
        echo "✓ ColabFold installed!"
        echo ""
        echo "To use:"
        echo "  conda activate petase-colabfold"
        echo "  bash scripts/colabfold_predict.sh <fasta_file>"
        exit 0
    else
        echo "Skipping. You can activate manually: conda activate petase-colabfold"
        exit 0
    fi
fi

# Create new environment
echo "Creating conda environment from envs/colabfold.yml..."
conda env create -f envs/colabfold.yml

echo ""
echo "✓ Environment created!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate petase-colabfold"
echo "  2. ColabFold should already be installed (from yml file)"
echo "  3. Verify: colabfold_batch --version"
echo ""
echo "If ColabFold isn't installed, run:"
echo "  conda activate petase-colabfold"
echo "  pip install colabfold"
echo ""

