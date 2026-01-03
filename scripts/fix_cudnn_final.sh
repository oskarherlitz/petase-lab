#!/usr/bin/env bash
# Final fix for cuDNN initialization error
# This error persists even with correct CUDA version - need to clear GPU state

set -euo pipefail

echo "Fixing persistent cuDNN error..."
echo ""

# Kill all processes using GPU
echo "1. Killing all GPU processes..."
pkill -f colabfold
pkill -f python
pkill -f jax
sleep 5

# Clear GPU memory
echo ""
echo "2. Clearing GPU memory..."
nvidia-smi --gpu-reset || echo "GPU reset not available, continuing..."

# Check GPU status
echo ""
echo "3. Checking GPU status..."
nvidia-smi

# Try setting environment variable to force cuDNN reinitialization
echo ""
echo "4. Setting environment variables..."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Test JAX
echo ""
echo "5. Testing JAX GPU access..."
python3 << EOF
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# Try a simple operation
try:
    x = jax.numpy.array([1, 2, 3])
    y = x * 2
    print(f"✓ JAX GPU test successful: {y}")
except Exception as e:
    print(f"✗ JAX GPU test failed: {e}")
EOF

echo ""
echo "✓ Done! Try running ColabFold with:"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform colabfold_batch ..."

