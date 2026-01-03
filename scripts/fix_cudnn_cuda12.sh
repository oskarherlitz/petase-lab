#!/usr/bin/env bash
# Fix cuDNN error by installing JAX with CUDA 12 (matching system CUDA 12.8)

set -euo pipefail

echo "Fixing cuDNN error - installing JAX with CUDA 12..."
echo ""

# Kill any running ColabFold
pkill -f colabfold 2>/dev/null || true
sleep 3

# Uninstall current JAX
echo "Uninstalling current JAX..."
pip uninstall -y jax jaxlib

# Install JAX with CUDA 12 (matching system CUDA 12.8)
echo ""
echo "Installing JAX with CUDA 12..."
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify GPU is detected
echo ""
echo "Verifying GPU detection..."
python3 -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"

echo ""
echo "✓ Done! Try running ColabFold again."

