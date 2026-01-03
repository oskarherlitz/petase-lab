#!/usr/bin/env bash
# Thorough cuDNN fix with verification

set -euo pipefail

echo "=========================================="
echo "Thorough cuDNN Fix and Verification"
echo "=========================================="
echo ""

# Step 1: Install cuDNN if not installed
echo "1. Checking/Installing cuDNN..."
if ! python3 -c "import site; import os; any(os.path.exists(os.path.join(d, 'nvidia', 'cudnn')) for d in site.getsitepackages())" 2>/dev/null | grep -q True; then
    echo "   Installing nvidia-cudnn-cu12..."
    pip install nvidia-cudnn-cu12
else
    echo "   cuDNN package already installed"
fi

# Step 2: Find cuDNN path
echo ""
echo "2. Finding cuDNN installation..."
CUDNN_PATH=$(python3 << 'EOF'
import site
import os
for d in site.getsitepackages():
    cudnn_lib = os.path.join(d, "nvidia", "cudnn", "lib")
    if os.path.exists(cudnn_lib):
        print(cudnn_lib)
        break
EOF
)

if [ -z "$CUDNN_PATH" ]; then
    echo "   ERROR: Could not find cuDNN installation"
    echo "   Searching manually..."
    find /usr/local/lib/python* -name "*cudnn*" -type d 2>/dev/null | head -5
    exit 1
fi

echo "   Found cuDNN at: $CUDNN_PATH"

# Step 3: Verify cuDNN libraries exist
echo ""
echo "3. Verifying cuDNN libraries..."
if [ -f "$CUDNN_PATH/libcudnn.so" ] || [ -f "$CUDNN_PATH/libcudnn.so.9" ] || [ -f "$CUDNN_PATH/libcudnn.so.8" ]; then
    echo "   ✓ cuDNN libraries found"
    ls -lh "$CUDNN_PATH"/libcudnn.so* 2>/dev/null | head -3
else
    echo "   ✗ cuDNN libraries not found in $CUDNN_PATH"
    echo "   Listing directory contents:"
    ls -la "$CUDNN_PATH" 2>/dev/null | head -10
    exit 1
fi

# Step 4: Test if libraries can be loaded
echo ""
echo "4. Testing if cuDNN libraries can be loaded..."
python3 << EOF
import ctypes
import os

cudnn_path = "$CUDNN_PATH"
libs = ["libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"]

for lib_name in libs:
    lib_file = os.path.join(cudnn_path, lib_name)
    if os.path.exists(lib_file):
        try:
            ctypes.CDLL(lib_file)
            print(f"   ✓ Can load: {lib_name}")
            break
        except Exception as e:
            print(f"   ✗ Cannot load {lib_name}: {e}")
else:
    print("   ✗ No cuDNN library could be loaded")
EOF

# Step 5: Set environment with cuDNN path FIRST
echo ""
echo "5. Setting up environment with cuDNN path..."
export CUDA_HOME=/usr/local/cuda-12.4
# Put cuDNN path FIRST in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "   LD_LIBRARY_PATH starts with: $(echo $LD_LIBRARY_PATH | cut -d: -f1-3)"

# Step 6: Verify LD_LIBRARY_PATH is set correctly
echo ""
echo "6. Verifying environment..."
echo "   CUDA_HOME: $CUDA_HOME"
echo "   cuDNN path in LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | grep -o "[^:]*cudnn[^:]*" | head -1 || echo "NOT FOUND")"

# Step 7: Test JAX with verbose output
echo ""
echo "7. Testing JAX GPU detection..."
python3 << 'TEST_JAX'
import os
import sys

# Set environment
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Print LD_LIBRARY_PATH for debugging
print(f"   Python sees LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:200]}...")

try:
    import jax
    print(f"   JAX version: {jax.__version__}")
    
    # Try to get devices
    try:
        devices = jax.devices()
        backend = jax.default_backend()
        
        print(f"   Devices: {devices}")
        print(f"   Backend: {backend}")
        
        if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
            print("\n   ✓✓✓ GPU DETECTED! ✓✓✓")
            sys.exit(0)
        else:
            print("\n   ⚠ Still using CPU")
            
            # Try to get more info about why
            try:
                from jax._src import xla_bridge
                backend_obj = xla_bridge.get_backend()
                print(f"   XLA bridge platform: {backend_obj.platform}")
            except Exception as e:
                print(f"   XLA bridge error: {e}")
            
            sys.exit(1)
    except RuntimeError as e:
        if 'cuDNN' in str(e):
            print(f"\n   ✗ cuDNN error: {e}")
            print("   Trying to diagnose...")
            
            # Check if we can import cudnn directly
            try:
                import ctypes
                cudnn_path = "$CUDNN_PATH"
                lib_file = os.path.join(cudnn_path, "libcudnn.so.9")
                if not os.path.exists(lib_file):
                    lib_file = os.path.join(cudnn_path, "libcudnn.so.8")
                if os.path.exists(lib_file):
                    ctypes.CDLL(lib_file)
                    print(f"   ✓ Can load cuDNN directly: {lib_file}")
                    print("   But JAX still can't find it. This might be a JAX/jaxlib issue.")
                else:
                    print(f"   ✗ cuDNN library not found at expected path")
            except Exception as e2:
                print(f"   Error loading cuDNN: {e2}")
            
            sys.exit(1)
        else:
            raise
            
except Exception as e:
    print(f"\n   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
TEST_JAX

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ SUCCESS! GPU is working!"
    echo "=========================================="
    echo ""
    echo "To use GPU with ColabFold, run these commands:"
    echo ""
    echo "export CUDA_HOME=/usr/local/cuda-12.4"
    echo "export LD_LIBRARY_PATH=$CUDNN_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH"
    echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false"
    echo "export XLA_PYTHON_CLIENT_ALLOCATOR=platform"
    echo "colabfold_batch ..."
else
    echo "=========================================="
    echo "Still not working. Checking jaxlib version..."
    echo "=========================================="
    echo ""
    pip show jaxlib | grep Version
    echo ""
    echo "The jaxlib version might need to match the cuDNN version."
    echo "Trying to reinstall jaxlib with cuDNN..."
    echo ""
    
    pip uninstall -y jaxlib
    pip install --no-cache-dir "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    echo ""
    echo "Testing again..."
    python3 -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
fi

