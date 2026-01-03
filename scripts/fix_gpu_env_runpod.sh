#!/usr/bin/env bash
# Fix GPU detection by setting proper CUDA environment variables

set -euo pipefail

echo "=========================================="
echo "Fixing GPU Detection with Environment Variables"
echo "=========================================="
echo ""

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH

# Set JAX environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

echo "Environment variables set:"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "Testing GPU detection..."
python3 << 'TEST_GPU'
import os
import sys

# Set environment variables before importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Try a simple computation
try:
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"Test computation: {y}")
    print(f"Computation device: {y.device()}")
except Exception as e:
    print(f"Error in computation: {e}")

devices = jax.devices()
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("✓ GPU detected!")
    sys.exit(0)
else:
    print("⚠ Still using CPU")
    sys.exit(1)
TEST_GPU

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ GPU detected! Add these to your shell:"
    echo "=========================================="
    echo ""
    echo "export CUDA_HOME=/usr/local/cuda-12.4"
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH"
    echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false"
    echo "export XLA_PYTHON_CLIENT_ALLOCATOR=platform"
    echo ""
    echo "Or run ColabFold with:"
    echo "CUDA_HOME=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform colabfold_batch ..."
else
    echo ""
    echo "Still not working. Checking JAX installation..."
    python3 -c "import jaxlib; print('JAXlib version:', jaxlib.__version__); print('JAXlib path:', jaxlib.__file__)"
    echo ""
    echo "Trying to reinstall JAX with CUDA 12..."
    pip uninstall -y jax jaxlib
    pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    echo ""
    echo "Testing again with environment variables..."
    python3 << 'TEST_AGAIN'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
import jax
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
TEST_AGAIN
fi

