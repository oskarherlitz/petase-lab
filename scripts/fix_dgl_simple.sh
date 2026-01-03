#!/bin/bash
# Simple fix: Install CUDA runtime via apt

set -e

echo "Installing CUDA runtime libraries..."

# Update package list
apt-get update

# Try to install CUDA runtime (multiple methods)
echo "Method 1: Installing cuda-cudart..."
apt-get install -y cuda-cudart-11-8 2>&1 | grep -v "already the newest" || true

echo "Method 2: Installing libcudart11..."
apt-get install -y libcudart11.0 2>&1 | grep -v "already the newest" || true

echo "Method 3: Installing cuda-runtime..."
apt-get install -y cuda-runtime-11-8 2>&1 | grep -v "already the newest" || true

# Find where it was installed
CUDA_LIB=$(find /usr -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")

if [ -n "${CUDA_LIB}" ]; then
    echo "✓ Found CUDA libs at: ${CUDA_LIB}"
    export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    echo "export LD_LIBRARY_PATH=\"${CUDA_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    echo "✓ Added to ~/.bashrc"
    
    echo ""
    echo "Testing DGL..."
    if python3 -c "import dgl; print('✓ DGL works!')" 2>&1; then
        echo "✓ SUCCESS!"
    else
        echo "✗ Still failing. Try: source ~/.bashrc and test again"
    fi
else
    echo "✗ Could not find CUDA libraries after install"
    echo "Try: apt-get install -y cuda-toolkit-11-8"
fi

