#!/bin/bash
# Install CUDA 11.8 cuBLAS library for DGL
# This is the missing piece: libcublas.so.11

set -e

echo "=========================================="
echo "Installing CUDA 11.8 cuBLAS for DGL"
echo "=========================================="
echo ""

# Fix broken dependencies
echo "1. Fixing broken dependencies..."
apt-get install -f -y 2>&1 | grep -v "already the newest" || true

# Install cuBLAS (the missing library)
echo ""
echo "2. Installing cuda-cublas-11-8..."
apt-get install -y cuda-cublas-11-8 2>&1 | grep -E "(Setting up|already the newest|E:)" || true

# Also install other commonly needed CUDA 11.8 libraries
echo ""
echo "3. Installing other CUDA 11.8 libraries..."
apt-get install -y \
    cuda-curand-11-8 \
    cuda-cusolver-11-8 \
    cuda-cusparse-11-8 \
    2>&1 | grep -E "(Setting up|already the newest|E:)" || true

# Find where cuBLAS was installed
echo ""
echo "4. Locating libcublas.so.11..."
CUBLAS_PATH=$(find /usr -name "libcublas.so.11*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")

if [ -n "${CUBLAS_PATH}" ]; then
    echo "   ✓ Found at: ${CUBLAS_PATH}"
else
    echo "   ✗ Not found in /usr, checking /usr/local/cuda-11.8..."
    CUBLAS_PATH="/usr/local/cuda-11.8/targets/x86_64-linux/lib"
    if [ -f "${CUBLAS_PATH}/libcublas.so.11"* ] 2>/dev/null; then
        echo "   ✓ Found at: ${CUBLAS_PATH}"
    else
        echo "   ✗ Still not found"
        CUBLAS_PATH=""
    fi
fi

# Set LD_LIBRARY_PATH to include CUDA 11.8 libraries
echo ""
echo "5. Setting LD_LIBRARY_PATH..."

# Primary CUDA 11.8 library location
CUDA_11_8_LIB="/usr/local/cuda-11.8/targets/x86_64-linux/lib"
if [ -d "${CUDA_11_8_LIB}" ]; then
    export LD_LIBRARY_PATH="${CUDA_11_8_LIB}:${LD_LIBRARY_PATH}"
    echo "   Added: ${CUDA_11_8_LIB}"
fi

# Also add system lib directory
SYS_LIB="/usr/lib/x86_64-linux-gnu"
if [ -d "${SYS_LIB}" ]; then
    export LD_LIBRARY_PATH="${SYS_LIB}:${LD_LIBRARY_PATH}"
    echo "   Added: ${SYS_LIB}"
fi

# Add to bashrc permanently
if ! grep -q "LD_LIBRARY_PATH.*cuda-11.8" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# CUDA 11.8 libraries for DGL" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    echo "   ✓ Added to ~/.bashrc"
fi

# Verify all required libraries
echo ""
echo "6. Verifying required libraries..."
REQUIRED=("libcudart.so.11" "libcublas.so.11")
ALL_FOUND=true

for lib in "${REQUIRED[@]}"; do
    if find /usr /usr/local -name "${lib}*" 2>/dev/null | grep -q .; then
        echo "   ✓ ${lib}"
    else
        echo "   ✗ ${lib} - MISSING"
        ALL_FOUND=false
    fi
done

# Test DGL
echo ""
echo "7. Testing DGL import..."
if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1; then
    echo "   ✓ SUCCESS! DGL works now."
    echo ""
    echo "   Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo "   Run 'source ~/.bashrc' to make it permanent in new shells"
    exit 0
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -o "lib[^:]*\.so[^:]*" | head -1 || echo "unknown")
    echo "   ✗ DGL still fails (missing: ${ERROR})"
    echo ""
    echo "   Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo ""
    echo "   Try running the comprehensive fix:"
    echo "   bash scripts/fix_dgl_all_cuda11_libs.sh"
    exit 1
fi

