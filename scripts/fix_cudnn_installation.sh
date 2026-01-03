#!/usr/bin/env bash
# Fix cuDNN installation/configuration for JAX

set -euo pipefail

echo "=========================================="
echo "Fixing cuDNN for JAX GPU Support"
echo "=========================================="
echo ""

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

echo "1. Checking for cuDNN installation..."
echo ""

# Check for cuDNN in various locations
CUDNN_FOUND=false

# Check in CUDA directory
if [ -f "$CUDA_HOME/targets/x86_64-linux/lib/libcudnn.so" ] || [ -f "$CUDA_HOME/lib64/libcudnn.so" ]; then
    echo "   ✓ Found cuDNN in CUDA directory"
    CUDNN_FOUND=true
    CUDNN_LIB_DIR="$CUDA_HOME/targets/x86_64-linux/lib"
    if [ ! -f "$CUDNN_LIB_DIR/libcudnn.so" ]; then
        CUDNN_LIB_DIR="$CUDA_HOME/lib64"
    fi
fi

# Check in system locations
for path in /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib64; do
    if [ -f "$path/libcudnn.so" ] || [ -f "$path/libcudnn.so.9" ] || [ -f "$path/libcudnn.so.8" ]; then
        echo "   ✓ Found cuDNN in $path"
        CUDNN_FOUND=true
        CUDNN_LIB_DIR="$path"
        break
    fi
done

# Check in nvidia directories
for path in /usr/local/nvidia/lib64 /usr/local/nvidia/lib; do
    if [ -f "$path/libcudnn.so" ] || [ -f "$path/libcudnn.so.9" ] || [ -f "$path/libcudnn.so.8" ]; then
        echo "   ✓ Found cuDNN in $path"
        CUDNN_FOUND=true
        CUDNN_LIB_DIR="$path"
        break
    fi
done

if [ "$CUDNN_FOUND" = false ]; then
    echo "   ✗ cuDNN not found in standard locations"
    echo ""
    echo "2. Searching for cuDNN libraries..."
    find /usr -name "*cudnn*.so*" 2>/dev/null | head -10 || echo "   No cuDNN libraries found"
    echo ""
    
    echo "3. cuDNN may be bundled with JAX. Checking jaxlib..."
    python3 << 'CHECK_JAXLIB_CUDNN'
import jaxlib
import os
import glob

jaxlib_dir = os.path.dirname(jaxlib.__file__)
print(f"JAXlib directory: {jaxlib_dir}")

# Look for cuDNN in jaxlib
cudnn_files = glob.glob(os.path.join(jaxlib_dir, "**", "*cudnn*"), recursive=True)
if cudnn_files:
    print(f"Found cuDNN files in jaxlib:")
    for f in cudnn_files[:5]:
        print(f"  {f}")
else:
    print("No cuDNN files found in jaxlib")
CHECK_JAXLIB_CUDNN
    
    echo ""
    echo "4. Installing cuDNN via pip (nvidia-cudnn-cu12)..."
    pip install nvidia-cudnn-cu12
    
    echo ""
    echo "5. Finding installed cuDNN location..."
    python3 << 'FIND_CUDNN'
import site
import os
import glob

# Check in site-packages
for site_dir in site.getsitepackages():
    cudnn_path = os.path.join(site_dir, "nvidia", "cudnn", "lib")
    if os.path.exists(cudnn_path):
        print(f"Found nvidia-cudnn-cu12 in: {cudnn_path}")
        libs = glob.glob(os.path.join(cudnn_path, "libcudnn*.so*"))
        for lib in libs[:3]:
            print(f"  {lib}")
        break
FIND_CUDNN
    
    # Add nvidia-cudnn to library path
    CUDNN_PYTHON_PATH=$(python3 -c "import site; import os; print([os.path.join(d, 'nvidia', 'cudnn', 'lib') for d in site.getsitepackages() if os.path.exists(os.path.join(d, 'nvidia', 'cudnn', 'lib'))][0] if any(os.path.exists(os.path.join(d, 'nvidia', 'cudnn', 'lib')) for d in site.getsitepackages()) else '')")
    
    if [ -n "$CUDNN_PYTHON_PATH" ] && [ -d "$CUDNN_PYTHON_PATH" ]; then
        echo ""
        echo "   ✓ Found cuDNN in Python packages"
        export LD_LIBRARY_PATH="$CUDNN_PYTHON_PATH:$LD_LIBRARY_PATH"
        CUDNN_LIB_DIR="$CUDNN_PYTHON_PATH"
        CUDNN_FOUND=true
    fi
fi

if [ "$CUDNN_FOUND" = true ]; then
    echo ""
    echo "6. Adding cuDNN to LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:$LD_LIBRARY_PATH"
    echo "   Updated LD_LIBRARY_PATH includes: $CUDNN_LIB_DIR"
fi

echo ""
echo "7. Testing JAX GPU detection with cuDNN fix..."
python3 << 'TEST_JAX'
import os
import sys

# Set environment
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

try:
    import jax
    print(f"JAX version: {jax.__version__}")
    
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"Devices: {devices}")
    print(f"Backend: {backend}")
    
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("\n✓✓✓ GPU DETECTED! ✓✓✓")
        sys.exit(0)
    else:
        print("\n⚠ Still using CPU")
        sys.exit(1)
        
except RuntimeError as e:
    if 'cuDNN' in str(e):
        print(f"\n✗ cuDNN error: {e}")
        print("\nTrying to set LD_LIBRARY_PATH explicitly...")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
TEST_JAX

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ SUCCESS! GPU is now working!"
    echo "=========================================="
    echo ""
    echo "To use GPU with ColabFold, set these environment variables:"
    echo "  export CUDA_HOME=/usr/local/cuda-12.4"
    echo "  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "  export XLA_PYTHON_CLIENT_PREALLOCATE=false"
    echo "  export XLA_PYTHON_CLIENT_ALLOCATOR=platform"
    echo ""
    echo "Or use the wrapper script:"
    echo "  bash scripts/run_colabfold_gpu.sh ..."
else
    echo "=========================================="
    echo "Still having issues. Creating persistent fix..."
    echo "=========================================="
    echo ""
    echo "Creating .bashrc entries for persistent environment..."
    
    cat >> ~/.bashrc << EOF

# JAX GPU Support
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
EOF
    
    # Add cuDNN path if found
    if [ "$CUDNN_FOUND" = true ] && [ -n "$CUDNN_LIB_DIR" ]; then
        echo "export LD_LIBRARY_PATH=$CUDNN_LIB_DIR:\$LD_LIBRARY_PATH" >> ~/.bashrc
    fi
    
    cat >> ~/.bashrc << EOF
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
EOF
    
    echo "✓ Added to ~/.bashrc"
    echo ""
    echo "Source it with: source ~/.bashrc"
    echo "Or start a new shell session"
fi

