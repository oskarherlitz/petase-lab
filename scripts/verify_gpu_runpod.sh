#!/usr/bin/env bash
# Verify and fix GPU detection on RunPod

set -euo pipefail

echo "=========================================="
echo "GPU Detection Diagnostic"
echo "=========================================="
echo ""

# Check if GPU hardware is available
echo "1. Checking GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    echo ""
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo ""
    # Show full nvidia-smi output to see CUDA version in header
    echo "Full GPU info:"
    nvidia-smi | head -5
    echo ""
    # Try to extract CUDA version from nvidia-smi (usually in header line)
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -i "cuda version" | head -1 | sed 's/.*CUDA Version: \([0-9]\+\).*/\1/' || echo "12")
    if [ -z "$CUDA_VERSION" ] || [ "$CUDA_VERSION" = "12" ]; then
        # Default to trying CUDA 12 first (most common on RunPod)
        CUDA_VERSION="12"
    fi
    echo "Will use CUDA $CUDA_VERSION for JAX installation (if needed)"
else
    echo "✗ nvidia-smi not found - no GPU available"
    exit 1
fi

echo ""
echo "2. Checking JAX installation..."
python3 << EOF
import sys
try:
    import jax
    print(f"✓ JAX version: {jax.__version__}")
    print(f"✓ JAX devices: {jax.devices()}")
    print(f"✓ Default backend: {jax.default_backend()}")
    
    # Check if GPU is detected
    devices = jax.devices()
    if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
        print("✓ GPU detected by JAX!")
    else:
        print("⚠ GPU NOT detected by JAX (using CPU)")
        sys.exit(1)
except ImportError:
    print("✗ JAX not installed")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
EOF

JAX_STATUS=$?

echo ""
if [ $JAX_STATUS -eq 0 ]; then
    echo "=========================================="
    echo "✓ GPU is properly detected!"
    echo "=========================================="
else
    echo "=========================================="
    echo "⚠ GPU not detected - attempting fix..."
    echo "=========================================="
    echo ""
    
    echo "Uninstalling current JAX..."
    pip uninstall -y jax jaxlib 2>/dev/null || true
    
    # Try CUDA 12 first (most common on RunPod)
    echo ""
    echo "Attempting to install JAX with CUDA 12..."
    pip install \
        "jax[cuda12_local]==0.4.23" \
        "jaxlib==0.4.23" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    echo ""
    echo "Verifying GPU detection after CUDA 12 install..."
    python3 << 'VERIFY_EOF'
import jax
devices = jax.devices()
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("✓ GPU detected with CUDA 12!")
    exit(0)
else:
    print("⚠ CUDA 12 didn't work, trying CUDA 11.8...")
    exit(1)
VERIFY_EOF
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "CUDA 12 didn't work, trying CUDA 11.8..."
        pip uninstall -y jax jaxlib 2>/dev/null || true
        pip install \
            "jax[cuda11_local]==0.4.23" \
            "jaxlib==0.4.23" \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        
        echo ""
        echo "Verifying GPU detection after CUDA 11.8 install..."
        python3 << 'VERIFY_EOF2'
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

devices = jax.devices()
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("✓ GPU detected with CUDA 11.8!")
else:
    print("⚠ Still using CPU - GPU may not be properly configured")
VERIFY_EOF2
    fi
    
    echo ""
    echo "=========================================="
    echo "Fix complete!"
    echo "=========================================="
fi

