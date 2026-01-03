#!/usr/bin/env bash
# Deep diagnostic to find the root cause of GPU detection failure

set -euo pipefail

echo "=========================================="
echo "Deep GPU Detection Diagnostic"
echo "=========================================="
echo ""

# Set environment
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "1. System CUDA Information:"
echo "   CUDA Version (from nvidia-smi):"
nvidia-smi | grep "CUDA Version" || echo "   Could not get CUDA version"
echo ""
echo "   CUDA Installation:"
ls -la /usr/local/cuda* 2>/dev/null | head -5 || echo "   No CUDA in /usr/local"
echo ""

echo "2. CUDA Library Check:"
python3 << 'CHECK_CUDA'
import os
import ctypes
import sys

cuda_libs = [
    "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so.12",
    "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libcudart.so",
    "/usr/local/nvidia/lib64/libcudart.so.12",
    "/usr/local/nvidia/lib/libcudart.so.12",
]

found = False
for lib_path in cuda_libs:
    if os.path.exists(lib_path):
        print(f"   ✓ Found: {lib_path}")
        try:
            ctypes.CDLL(lib_path)
            print(f"      ✓ Can load successfully")
            found = True
        except Exception as e:
            print(f"      ✗ Cannot load: {e}")
    else:
        print(f"   ✗ Not found: {lib_path}")

if not found:
    print("   ⚠ No CUDA runtime library found!")
    sys.exit(1)
CHECK_CUDA

echo ""
echo "3. JAX Installation Details:"
python3 << 'CHECK_JAX'
import sys
import os

# Set environment before importing
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

try:
    import jax
    import jaxlib
    
    print(f"   JAX version: {jax.__version__}")
    print(f"   JAXlib version: {jaxlib.__version__}")
    print(f"   JAXlib location: {jaxlib.__file__}")
    
    # Check jaxlib for CUDA files
    import os
    jaxlib_dir = os.path.dirname(jaxlib.__file__)
    print(f"   JAXlib directory: {jaxlib_dir}")
    
    # Look for CUDA-related files
    import glob
    cuda_files = []
    for pattern in ['**/*cuda*', '**/*gpu*', '**/*cublas*', '**/*cudnn*']:
        cuda_files.extend(glob.glob(os.path.join(jaxlib_dir, pattern), recursive=True))
    
    if cuda_files:
        print(f"   ✓ Found {len(cuda_files)} CUDA-related files in jaxlib")
        for f in cuda_files[:5]:
            print(f"      - {os.path.relpath(f, jaxlib_dir)}")
    else:
        print(f"   ⚠ No CUDA files found in jaxlib - this might be CPU-only!")
    
    # Try to import XLA extension
    try:
        from jaxlib import xla_extension
        print(f"   ✓ XLA extension available: {xla_extension}")
        
        # Try to get platforms
        try:
            platforms = xla_extension.get_platforms()
            print(f"   Available platforms: {platforms}")
        except Exception as e:
            print(f"   Could not get platforms: {e}")
            
    except Exception as e:
        print(f"   ✗ Cannot import xla_extension: {e}")
        import traceback
        traceback.print_exc()
    
    # Check what JAX sees
    print(f"\n   JAX devices: {jax.devices()}")
    print(f"   JAX backend: {jax.default_backend()}")
    
    # Try to access internal XLA bridge
    try:
        from jax._src import xla_bridge
        backend = xla_bridge.get_backend()
        print(f"   XLA bridge backend: {backend}")
        print(f"   XLA bridge platform: {backend.platform}")
        print(f"   XLA bridge devices: {backend.local_devices()}")
        
        # Check if there are any errors during initialization
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
    except Exception as e:
        print(f"   Error accessing XLA bridge: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
CHECK_JAX

echo ""
echo "4. Checking for cuDNN:"
find /usr/local/cuda* -name "*cudnn*" 2>/dev/null | head -5 || echo "   No cuDNN found in CUDA directories"
find /usr -name "*cudnn*" 2>/dev/null | head -5 || echo "   No cuDNN found system-wide"

echo ""
echo "5. Environment Variables:"
echo "   CUDA_HOME: ${CUDA_HOME:-not set}"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "   XLA_PYTHON_CLIENT_PREALLOCATE: ${XLA_PYTHON_CLIENT_PREALLOCATE:-not set}"

echo ""
echo "6. Testing with verbose JAX logging:"
XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=true \
python3 << 'VERBOSE_TEST'
import os
import sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force CUDA platform

# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

try:
    import jax
    print(f"JAX version: {jax.__version__}")
    
    # Force backend initialization
    devices = jax.devices()
    print(f"Devices: {devices}")
    print(f"Backend: {jax.default_backend()}")
    
    # Try a computation
    import jax.numpy as jnp
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"Computation device: {y.devices()}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
VERBOSE_TEST

echo ""
echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="

