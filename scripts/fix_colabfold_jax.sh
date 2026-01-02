#!/usr/bin/env bash
# Fix JAX/haiku compatibility issue for ColabFold on macOS

set -euo pipefail

echo "Fixing ColabFold JAX compatibility issue..."
echo ""

# Check if in correct environment
if [[ "$CONDA_DEFAULT_ENV" != "petase-colabfold" ]]; then
    echo "Please activate petase-colabfold environment first:"
    echo "  conda activate petase-colabfold"
    exit 1
fi

echo "Installing compatible JAX and haiku versions..."
echo ""

# Uninstall problematic packages
pip uninstall -y jax jaxlib dm-haiku 2>/dev/null || true

# Install compatible versions
# JAX 0.4.24+ removed linear_util, so we need JAX <0.4.24
# Using JAX 0.4.23 which should work with dm-haiku
echo "Installing JAX <0.4.24 (linear_util was removed in 0.4.24+)..."
pip install "jax[cpu]==0.4.23" "jaxlib==0.4.23"

# Install compatible dm-haiku (0.0.10 should work with JAX 0.4.23)
echo "Installing dm-haiku..."
pip install "dm-haiku==0.0.10"

echo ""
echo "✓ Fixed! Try running ColabFold again:"
echo "  bash scripts/colabfold_predict.sh <fasta_file>"
echo ""

