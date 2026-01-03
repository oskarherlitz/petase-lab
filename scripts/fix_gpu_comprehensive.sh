#!/usr/bin/env bash
# Comprehensive GPU fix based on research findings

set -euo pipefail

echo "=========================================="
echo "Comprehensive GPU Fix for JAX"
echo "=========================================="
echo ""

# Step 1: Check NVIDIA driver version
echo "1. Checking NVIDIA driver version..."
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "   Driver version: $DRIVER_VERSION"
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 545 ]; then
    echo "   ⚠ Warning: Driver version may be too old for CUDA 12.4 (needs 545+)"
else
    echo "   ✓ Driver version is compatible"
fi

# Step 2: Reload NVIDIA modules (fixes some detection issues)
echo ""
echo "2. Reloading NVIDIA Unified Memory module..."
rmmod nvidia_uvm 2>/dev/null || echo "   Module not loaded (this is OK)"
modprobe nvidia_uvm 2>/dev/null || echo "   Could not load module (may need root)"

# Step 3: Set environment variables
echo ""
echo "3. Setting environment variables..."
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_PLATFORMS=cuda  # Force CUDA platform

echo "   ✓ Environment variables set"

# Step 4: Check cuDNN version
echo ""
echo "4. Checking cuDNN version..."
if [ -f /usr/local/cuda/include/cudnn_version.h ]; then
    CUDNN_MAJOR=$(grep CUDNN_MAJOR /usr/local/cuda/include/cudnn_version.h | awk '{print $3}')
    echo "   cuDNN major version: $CUDNN_MAJOR"
    if [ "$CUDNN_MAJOR" -lt 9 ]; then
        echo "   ⚠ Warning: cuDNN version may be incompatible (JAX 0.4.23 with CUDA 12.4 may need cuDNN 9)"
    fi
else
    echo "   ⚠ cuDNN version file not found"
fi

# Step 5: Reinstall JAX with the correct method
echo ""
echo "5. Reinstalling JAX with CUDA 12 support (using cuda12_pip method)..."
pip uninstall -y jax jaxlib 2>/dev/null || true
pip cache purge 2>/dev/null || true

# Try the recommended installation method from JAX docs
pip install --upgrade pip
pip install --upgrade --no-cache-dir \
    "jax[cuda12_pip]==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# If that doesn't work, try the local method
if ! python3 -c "import jax; devices = jax.devices(); print('GPU' if any('gpu' in str(d).lower() for d in devices) else 'CPU')" 2>/dev/null | grep -q GPU; then
    echo "   cuda12_pip didn't work, trying cuda12_local..."
    pip uninstall -y jax jaxlib 2>/dev/null || true
    pip install --upgrade --no-cache-dir \
        "jax[cuda12_local]==0.4.23" \
        "jaxlib==0.4.23" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# Step 6: Verify installation
echo ""
echo "6. Verifying JAX installation..."
pip show jax jaxlib | grep -E "Name|Version"

# Step 7: Test GPU detection with all fixes
echo ""
echo "7. Testing GPU detection..."
python3 << 'TEST_GPU'
import os
import sys

# Set all environment variables
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_PLATFORMS'] = 'cuda'

# Enable debug logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO)

try:
    import jax
    print(f"   JAX version: {jax.__version__}")
    
    # Force backend initialization
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"   Devices: {devices}")
    print(f"   Backend: {backend}")
    
    # Try a computation
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"   Test computation device: {y.devices()}")
    
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("\n   ✓✓✓ GPU DETECTED! ✓✓✓")
        sys.exit(0)
    else:
        print("\n   ⚠ Still using CPU")
        
        # Try to get more diagnostic info
        try:
            from jax._src import xla_bridge
            backend_obj = xla_bridge.get_backend()
            print(f"   XLA bridge platform: {backend_obj.platform}")
            print(f"   XLA bridge devices: {backend_obj.local_devices()}")
        except Exception as e:
            print(f"   XLA bridge error: {e}")
        
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
TEST_GPU

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ SUCCESS! GPU is now detected!"
    echo "=========================================="
    echo ""
    echo "To use GPU with ColabFold, run:"
    echo "  export CUDA_HOME=/usr/local/cuda-12.4"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH"
    echo "  export JAX_PLATFORMS=cuda"
    echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform colabfold_batch ..."
else
    echo "=========================================="
    echo "⚠ GPU still not detected"
    echo "=========================================="
    echo ""
    echo "Running deep diagnostic..."
    bash scripts/deep_gpu_diagnostic.sh
fi

