#!/usr/bin/env bash
# Wrapper script to run ColabFold with proper GPU environment variables

set -euo pipefail

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

# Check if GPU is detected
echo "Checking GPU detection..."
python3 << 'CHECK_GPU'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
devices = jax.devices()
backend = jax.default_backend()

print(f"JAX devices: {devices}")
print(f"JAX backend: {backend}")

if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("✓ GPU detected!")
    exit(0)
else:
    print("⚠ WARNING: GPU not detected, will use CPU (very slow!)")
    print("  This may take many hours. Consider fixing GPU detection first.")
    exit(1)
CHECK_GPU

GPU_STATUS=$?

if [ $GPU_STATUS -ne 0 ]; then
    echo ""
    echo "GPU not detected. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted. Fix GPU detection first."
        exit 1
    fi
fi

echo ""
echo "Running ColabFold with GPU environment..."
echo ""

# Run ColabFold with all the environment variables
exec colabfold_batch "$@"

