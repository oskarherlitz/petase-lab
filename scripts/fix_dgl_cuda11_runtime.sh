#!/bin/bash
# Install CUDA 11.8 runtime libraries for DGL (without full toolkit)
# This works even if PyTorch uses CUDA 12.4

set -e

echo "=========================================="
echo "Installing CUDA 11.8 Runtime for DGL"
echo "=========================================="
echo ""

# Fix broken dependencies first
echo "1. Fixing broken dependencies..."
apt-get install -f -y 2>&1 | grep -v "already the newest" || true

# Install just the CUDA 11.8 runtime libraries (not the full toolkit)
echo ""
echo "2. Installing CUDA 11.8 runtime libraries..."

# Try installing just the runtime components
apt-get install -y \
    cuda-cudart-11-8 \
    cuda-cudart-dev-11-8 \
    2>&1 | grep -v "already the newest" || true

# Alternative: Try installing via individual packages
if ! find /usr -name "libcudart.so.11*" 2>/dev/null | grep -q .; then
    echo "   Trying alternative method..."
    apt-get install -y \
        libcudart11.0 \
        libcudart11.8 \
        2>&1 | grep -v "already the newest" || true
fi

# Find where CUDA 11.8 libraries were installed
echo ""
echo "3. Locating CUDA 11.8 libraries..."
CUDA_LIB_PATHS=(
    "/usr/local/cuda-11.8/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64"
)

CUDA_LIB=""
for path in "${CUDA_LIB_PATHS[@]}"; do
    if find "${path}" -name "libcudart.so.11*" 2>/dev/null | grep -q .; then
        CUDA_LIB="${path}"
        echo "   ✓ Found at: ${CUDA_LIB}"
        break
    fi
done

# If still not found, search the whole system
if [ -z "${CUDA_LIB}" ]; then
    CUDA_LIB=$(find /usr -name "libcudart.so.11*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -n "${CUDA_LIB}" ]; then
        echo "   ✓ Found at: ${CUDA_LIB}"
    fi
fi

# Set LD_LIBRARY_PATH
if [ -n "${CUDA_LIB}" ]; then
    echo ""
    echo "4. Setting LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    
    # Add to bashrc if not already there
    if ! grep -q "LD_LIBRARY_PATH.*${CUDA_LIB}" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA 11.8 runtime for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${CUDA_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
        echo "   ✓ Added to ~/.bashrc"
    fi
    
    echo "   LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
else
    echo "   ✗ Could not find CUDA 11.8 libraries"
    echo ""
    echo "   Trying to download manually..."
    
    # Create directory
    mkdir -p /usr/local/cuda-11.8/lib64
    cd /tmp
    
    # Try to download libcudart from NVIDIA
    echo "   Downloading libcudart.so.11.8..."
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb || \
    curl -L -o cuda-repo.deb https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb || \
    echo "   ⚠ Download failed"
    
    if [ -f cuda-repo*.deb ]; then
        echo "   Installing from .deb..."
        dpkg -i cuda-repo*.deb || true
        apt-get update
        apt-get install -y cuda-cudart-11-8
        CUDA_LIB="/usr/local/cuda-11.8/lib64"
    fi
fi

# Test DGL
echo ""
echo "5. Testing DGL import..."
if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1; then
    echo "   ✓ SUCCESS! DGL works now."
    echo ""
    echo "   To make it permanent, run: source ~/.bashrc"
    exit 0
else
    echo "   ✗ DGL still fails"
    echo ""
    echo "   Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo ""
    echo "   Manual steps:"
    echo "   1. Find CUDA 11.8 libs: find /usr -name 'libcudart.so.11*'"
    echo "   2. Export: export LD_LIBRARY_PATH=/path/to/lib64:\$LD_LIBRARY_PATH"
    echo "   3. Or reinstall DGL for CUDA 12.x (if available)"
    exit 1
fi

