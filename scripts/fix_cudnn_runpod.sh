#!/usr/bin/env bash
# Fix cuDNN initialization error on RunPod
# This error happens when CUDA/cuDNN versions don't match

set -euo pipefail

echo "Fixing cuDNN initialization error..."
echo ""

# Check CUDA version
echo "Checking CUDA version..."
nvcc --version 2>/dev/null || echo "nvcc not found, checking nvidia-smi..."
nvidia-smi | grep "CUDA Version" || nvidia-smi

echo ""
echo "Checking current JAX installation..."
pip show jax jaxlib | grep -E "Name|Version" || echo "JAX not installed"

echo ""
echo "Uninstalling current JAX..."
pip uninstall -y jax jaxlib

echo ""
echo "Installing JAX with CUDA 11.8 (most compatible)..."
# Try CUDA 11.8 first (most compatible)
pip install "jax[cuda11_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# If that fails, try CUDA 12
if [ $? -ne 0 ]; then
    echo ""
    echo "CUDA 11.8 failed, trying CUDA 12..."
    pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

echo ""
echo "Verifying GPU detection..."
python3 << EOF
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
EOF

echo ""
echo "✓ Done! Try running ColabFold again."

