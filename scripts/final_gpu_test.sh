#!/usr/bin/env bash
# Final comprehensive GPU test

set -euo pipefail

echo "=========================================="
echo "Final GPU Detection Test"
echo "=========================================="
echo ""

# Set all environment variables
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

echo "1. Environment variables:"
echo "   CUDA_HOME: $CUDA_HOME"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "2. Checking CUDA libraries are accessible:"
python3 << 'CHECK_LIBS'
import os
import ctypes

cuda_path = "/usr/local/cuda-12.4/targets/x86_64-linux/lib"
libcudart = os.path.join(cuda_path, "libcudart.so.12")
if os.path.exists(libcudart):
    print(f"   ✓ Found: {libcudart}")
    try:
        ctypes.CDLL(libcudart)
        print("   ✓ Can load libcudart.so.12")
    except Exception as e:
        print(f"   ✗ Cannot load: {e}")
else:
    print(f"   ✗ Not found: {libcudart}")
CHECK_LIBS

echo ""
echo "3. Checking JAX/jaxlib versions:"
pip show jax jaxlib | grep -E "Name|Version" || echo "Not installed"

echo ""
echo "4. Testing JAX GPU detection:"
python3 << 'TEST_JAX'
import os
import sys

# Set environment before importing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Try to import and check
try:
    import jax
    print(f"   JAX version: {jax.__version__}")
    
    import jaxlib
    print(f"   JAXlib version: {jaxlib.__version__}")
    
    # Force backend initialization
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"   Devices: {devices}")
    print(f"   Backend: {backend}")
    
    # Try a computation to force GPU initialization
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"   Test computation device: {y.devices()}")
    
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("\n   ✓ GPU DETECTED!")
        sys.exit(0)
    else:
        print("\n   ⚠ Still using CPU")
        
        # Check if there are any errors
        try:
            from jaxlib import xla_extension
            print(f"   XLA extension: {xla_extension}")
        except Exception as e:
            print(f"   Error with xla_extension: {e}")
        
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
TEST_JAX

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ SUCCESS! GPU is detected!"
    echo "=========================================="
    echo ""
    echo "You can now run ColabFold with:"
    echo "  export CUDA_HOME=/usr/local/cuda-12.4"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH"
    echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform colabfold_batch ..."
else
    echo "=========================================="
    echo "⚠ GPU still not detected"
    echo "=========================================="
    echo ""
    echo "Trying one more thing - checking if we need to set LD_PRELOAD..."
    echo ""
    
    # Sometimes we need LD_PRELOAD
    export LD_PRELOAD=/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so.12
    
    python3 -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())" || echo "Still not working"
fi

