#!/usr/bin/env bash
# Fix ColabFold JAX compatibility issue on RunPod
# Run this if you get: AttributeError: jax.interpreters.xla.xe was removed

set -euo pipefail

echo "Fixing ColabFold JAX compatibility issue on RunPod..."
echo ""

# Uninstall potentially problematic versions
echo "Uninstalling incompatible JAX versions..."
pip uninstall -y jax jaxlib dm-haiku haiku 2>/dev/null || true

# Install compatible versions
# JAX <0.4.24 is needed because linear_util was removed in 0.4.24+
# dm-haiku 0.0.10 is compatible with jax 0.4.23
echo ""
echo "Installing compatible JAX and haiku versions..."
pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23" "dm-haiku==0.0.10"

echo ""
echo "✓ Fixed! Try running ColabFold again:"
echo "  cd /workspace/petase-lab"
echo "  colabfold_batch --num-recycle 3 --num-models 5 --amber runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta runs/colabfold_predictions_gpu"

