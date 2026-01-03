#!/usr/bin/env bash
# Force JAX to use GPU by clearing cache and explicitly setting platform

set -euo pipefail

echo "=========================================="
echo "Forcing JAX to Use GPU"
echo "=========================================="
echo ""

# Set environment
CUDNN_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDNN_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "1. Clearing JAX cache..."
rm -rf ~/.cache/jax 2>/dev/null || true
rm -rf /tmp/jax_* 2>/dev/null || true
echo "   ✓ Cache cleared"

echo ""
echo "2. Testing with explicit platform setting..."
python3 << 'TEST_GPU'
import os
import sys

# Set environment BEFORE importing anything
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force CUDA

# Clear any cached imports
if 'jax' in sys.modules:
    del sys.modules['jax']
if 'jaxlib' in sys.modules:
    del sys.modules['jaxlib']

print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:150]}...")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'NOT SET')}")

try:
    import jax
    print(f"\nJAX version: {jax.__version__}")
    
    # Try to access XLA bridge directly and force CUDA
    from jax._src import xla_bridge
    
    # Clear any cached backend
    xla_bridge._backends = {}
    
    # Force CUDA platform
    try:
        backend = xla_bridge.get_backend('gpu')
        print(f"✓ Got GPU backend: {backend}")
        print(f"  Platform: {backend.platform}")
        print(f"  Devices: {backend.local_devices()}")
    except Exception as e:
        print(f"✗ Could not get GPU backend: {e}")
        # Try default
        backend = xla_bridge.get_backend()
        print(f"  Default backend: {backend}")
        print(f"  Default platform: {backend.platform}")
    
    # Now check devices
    devices = jax.devices()
    backend_name = jax.default_backend()
    
    print(f"\nJAX devices: {devices}")
    print(f"JAX backend: {backend_name}")
    
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("\n✓✓✓ GPU DETECTED! ✓✓✓")
        
        # Test a computation
        import jax.numpy as jnp
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print(f"Test computation device: {y.devices()}")
        
        sys.exit(0)
    else:
        print("\n⚠ Still using CPU")
        
        # Try to see what platforms are available
        try:
            from jaxlib import xla_extension
            print(f"\nChecking available platforms...")
            try:
                platforms = xla_extension.get_platforms()
                print(f"Available platforms: {platforms}")
            except:
                print("Could not get platforms list")
        except Exception as e:
            print(f"Error checking platforms: {e}")
        
        sys.exit(1)
        
except RuntimeError as e:
    if 'cuDNN' in str(e):
        print(f"\n✗ cuDNN error: {e}")
        print("\nTrying to verify cuDNN is accessible...")
        import ctypes
        cudnn_lib = "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib/libcudnn.so.9"
        try:
            ctypes.CDLL(cudnn_lib)
            print(f"✓ cuDNN library can be loaded: {cudnn_lib}")
            print("But JAX still can't find it. This might be a jaxlib issue.")
        except Exception as e2:
            print(f"✗ Cannot load cuDNN: {e2}")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
TEST_GPU

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo ""
    echo "3. Checking jaxlib CUDA support..."
    python3 << 'CHECK_JAXLIB'
import jaxlib
import os
import glob

jaxlib_dir = os.path.dirname(jaxlib.__file__)
print(f"JAXlib location: {jaxlib_dir}")

# Check for CUDA files
cuda_files = glob.glob(os.path.join(jaxlib_dir, "**", "*cuda*"), recursive=True)
gpu_files = glob.glob(os.path.join(jaxlib_dir, "**", "*gpu*"), recursive=True)

print(f"\nCUDA-related files: {len(cuda_files)}")
if cuda_files:
    for f in cuda_files[:5]:
        print(f"  {os.path.relpath(f, jaxlib_dir)}")

print(f"\nGPU-related files: {len(gpu_files)}")
if gpu_files:
    for f in gpu_files[:5]:
        print(f"  {os.path.relpath(f, jaxlib_dir)}")

# Check jaxlib version
import jaxlib
print(f"\nJAXlib version: {jaxlib.__version__}")

# Check if it's CUDA-enabled
if '+cuda' in jaxlib.__version__:
    print("✓ JAXlib has CUDA support")
else:
    print("✗ JAXlib does NOT have CUDA support - this is the problem!")
CHECK_JAXLIB
    
    echo ""
    echo "4. If jaxlib doesn't have CUDA, reinstalling..."
    pip show jaxlib | grep Version
    echo ""
    echo "The jaxlib might need to be reinstalled with CUDA 12 support."
fi

