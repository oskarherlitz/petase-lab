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
    nvidia-smi --query-gpu=name,driver_version,memory.total,cuda_version --format=csv
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1 | tr -d ' ')
    echo ""
    echo "System CUDA version: $CUDA_VERSION"
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
    
    # Determine which CUDA version to use
    if [[ "$CUDA_VERSION" == "12"* ]]; then
        CUDA_OPTION="cuda12_local"
        echo "Installing JAX with CUDA 12..."
    else
        CUDA_OPTION="cuda11_local"
        echo "Installing JAX with CUDA 11.8..."
    fi
    
    echo ""
    echo "Uninstalling current JAX..."
    pip uninstall -y jax jaxlib 2>/dev/null || true
    
    echo ""
    echo "Installing JAX with $CUDA_OPTION..."
    pip install \
        "jax[$CUDA_OPTION]==0.4.23" \
        "jaxlib==0.4.23" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    echo ""
    echo "Verifying fix..."
    python3 << EOF
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

devices = jax.devices()
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("✓ GPU detected!")
else:
    print("⚠ Still using CPU - may need to check CUDA installation")
EOF
    
    echo ""
    echo "=========================================="
    echo "Fix complete!"
    echo "=========================================="
fi

