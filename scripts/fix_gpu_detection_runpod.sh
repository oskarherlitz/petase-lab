#!/usr/bin/env bash
# Comprehensive GPU detection fix for RunPod

set -euo pipefail

echo "=========================================="
echo "Comprehensive GPU Detection Fix"
echo "=========================================="
echo ""

# Step 1: Check GPU hardware
echo "1. Checking GPU hardware..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Step 2: Check CUDA installation
echo "2. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo "nvcc not found, but CUDA runtime should be available"
fi
echo ""

# Step 3: Check current JAX installation
echo "3. Checking current JAX installation..."
python3 << 'CHECK_JAX'
import sys
try:
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    
    # Check if it's CPU-only
    devices = jax.devices()
    if len(devices) == 1 and 'cpu' in str(devices[0]).lower():
        print("⚠ Currently using CPU-only JAX")
        sys.exit(1)
    else:
        print("✓ GPU detected!")
        sys.exit(0)
except ImportError:
    print("JAX not installed")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
CHECK_JAX

JAX_STATUS=$?

if [ $JAX_STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ GPU is already detected!"
    echo "=========================================="
    exit 0
fi

echo ""
echo "4. Fixing JAX installation..."
echo ""

# Step 4: Completely uninstall JAX
echo "Uninstalling all JAX packages..."
pip uninstall -y jax jaxlib jax-cuda12-plugin 2>/dev/null || true
pip cache purge 2>/dev/null || true

# Step 5: Install JAX with CUDA 12
echo ""
echo "Installing JAX with CUDA 12 support..."
pip install --upgrade pip setuptools wheel
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    --no-cache-dir

echo ""
echo "5. Verifying installation..."
python3 << 'VERIFY'
import os
import jax

# Set environment variables that might help
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Try to create a simple computation to force GPU initialization
try:
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"Test computation result: {y}")
    print(f"Computation device: {y.device()}")
except Exception as e:
    print(f"Error during test computation: {e}")

devices = jax.devices()
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("✓ GPU detected and working!")
    exit(0)
else:
    print("⚠ Still using CPU")
    exit(1)
VERIFY

VERIFY_STATUS=$?

echo ""
if [ $VERIFY_STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ GPU detection fixed!"
    echo "=========================================="
    echo ""
    echo "You can now run ColabFold with GPU acceleration."
else
    echo "=========================================="
    echo "⚠ GPU still not detected"
    echo "=========================================="
    echo ""
    echo "Trying alternative: CUDA 11.8..."
    echo ""
    
    pip uninstall -y jax jaxlib 2>/dev/null || true
    pip install "jax[cuda11_local]==0.4.23" "jaxlib==0.4.23" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        --no-cache-dir
    
    echo ""
    echo "Verifying CUDA 11.8 installation..."
    python3 -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
    
    echo ""
    echo "If GPU is still not detected, check:"
    echo "1. CUDA libraries are in PATH: echo \$LD_LIBRARY_PATH"
    echo "2. GPU is accessible: nvidia-smi"
    echo "3. Try setting: export XLA_PYTHON_CLIENT_PREALLOCATE=false"
fi

