#!/usr/bin/env bash
# Fix cuDNN version mismatch - jaxlib expects cudnn89 but we have cudnn9

set -euo pipefail

echo "=========================================="
echo "Fixing cuDNN Version Mismatch"
echo "=========================================="
echo ""

CUDNN_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib

echo "1. Checking jaxlib cuDNN version requirement..."
JAXLIB_VERSION=$(pip show jaxlib | grep Version | awk '{print $2}')
echo "   JAXlib version: $JAXLIB_VERSION"

if [[ "$JAXLIB_VERSION" == *"cudnn89"* ]]; then
    echo "   JAXlib expects cuDNN 8.9 (cudnn89)"
    EXPECTED_CUDNN="8.9"
elif [[ "$JAXLIB_VERSION" == *"cudnn9"* ]]; then
    echo "   JAXlib expects cuDNN 9"
    EXPECTED_CUDNN="9"
else
    echo "   Could not determine expected cuDNN version"
    EXPECTED_CUDNN="unknown"
fi

echo ""
echo "2. Checking installed cuDNN version..."
if [ -f "$CUDNN_PATH/libcudnn.so.9" ]; then
    echo "   Found cuDNN 9"
    INSTALLED_CUDNN="9"
elif [ -f "$CUDNN_PATH/libcudnn.so.8" ]; then
    echo "   Found cuDNN 8"
    INSTALLED_CUDNN="8"
else
    echo "   Could not determine installed cuDNN version"
    INSTALLED_CUDNN="unknown"
fi

echo ""
echo "3. Creating symlink for compatibility..."
cd "$CUDNN_PATH"

# Create symlink from libcudnn.so.9 to libcudnn.so.8 if needed
if [ -f "libcudnn.so.9" ] && [ ! -f "libcudnn.so.8" ]; then
    echo "   Creating libcudnn.so.8 -> libcudnn.so.9 symlink..."
    ln -sf libcudnn.so.9 libcudnn.so.8
    echo "   ✓ Symlink created"
fi

# Also create generic libcudnn.so symlink
if [ -f "libcudnn.so.9" ] && [ ! -f "libcudnn.so" ]; then
    echo "   Creating libcudnn.so -> libcudnn.so.9 symlink..."
    ln -sf libcudnn.so.9 libcudnn.so
    echo "   ✓ Symlink created"
fi

echo ""
echo "4. Installing cuDNN 8.9 package (if available)..."
# Try to install the specific version jaxlib expects
pip install nvidia-cudnn-cu12==8.9.* 2>/dev/null || {
    echo "   cuDNN 8.9 not available, trying alternative..."
    # Check if we can use the existing one with symlinks
    if [ -f "$CUDNN_PATH/libcudnn.so.8" ]; then
        echo "   ✓ Using existing cuDNN with symlink"
    else
        echo "   ⚠ Could not install cuDNN 8.9, will try with existing version"
    fi
}

echo ""
echo "5. Setting up environment..."
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDNN_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo ""
echo "6. Testing JAX GPU detection..."
python3 << 'TEST_JAX'
import os
import sys

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Try without forcing platform first
try:
    import jax
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"Devices: {devices}")
    print(f"Backend: {backend}")
    
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("\n✓✓✓ GPU DETECTED! ✓✓✓")
        sys.exit(0)
    else:
        print("\n⚠ Still CPU, trying with JAX_PLATFORMS=cuda...")
        os.environ['JAX_PLATFORMS'] = 'cuda'
        
        # Clear cache and try again
        import sys
        if 'jax' in sys.modules:
            del sys.modules['jax']
        if 'jaxlib' in sys.modules:
            del sys.modules['jaxlib']
        
        import jax
        devices = jax.devices()
        backend = jax.default_backend()
        
        print(f"Devices (with JAX_PLATFORMS=cuda): {devices}")
        print(f"Backend: {backend}")
        
        if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
            print("\n✓✓✓ GPU DETECTED! ✓✓✓")
            sys.exit(0)
        else:
            print("\n⚠ Still CPU")
            sys.exit(1)
            
except RuntimeError as e:
    if 'cuDNN' in str(e):
        print(f"\n✗ cuDNN error: {e}")
        print("\nTrying to diagnose...")
        
        # Check what jaxlib is looking for
        try:
            import jaxlib
            print(f"JAXlib location: {jaxlib.__file__}")
            
            # Check if there are bundled cuDNN libraries
            import os
            import glob
            jaxlib_dir = os.path.dirname(jaxlib.__file__)
            bundled_cudnn = glob.glob(os.path.join(jaxlib_dir, "**", "*cudnn*"), recursive=True)
            if bundled_cudnn:
                print(f"Found bundled cuDNN in jaxlib: {bundled_cudnn[:3]}")
        except:
            pass
        
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

if [ $STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS! GPU is working!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Still not working. Trying one more thing..."
    echo "=========================================="
    echo ""
    echo "Checking if jaxlib has bundled cuDNN that needs to be used..."
    
    python3 << 'CHECK_BUNDLED'
import jaxlib
import os
import glob

jaxlib_dir = os.path.dirname(jaxlib.__file__)
print(f"JAXlib directory: {jaxlib_dir}")

# Look for any cuDNN files
cudnn_files = glob.glob(os.path.join(jaxlib_dir, "**", "*cudnn*"), recursive=True)
if cudnn_files:
    print(f"\nFound cuDNN files in jaxlib:")
    for f in cudnn_files:
        print(f"  {f}")
        # If it's a directory, add it to LD_LIBRARY_PATH
        if os.path.isdir(f):
            print(f"    -> This directory should be in LD_LIBRARY_PATH")
CHECK_BUNDLED
fi

