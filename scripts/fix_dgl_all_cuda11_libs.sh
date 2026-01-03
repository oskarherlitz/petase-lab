#!/bin/bash
# Install ALL CUDA 11.8 libraries needed by DGL
# DGL needs: libcudart, libcublas, libcurand, libcusolver, libcusparse, etc.

set -e

echo "=========================================="
echo "Installing ALL CUDA 11.8 Libraries for DGL"
echo "=========================================="
echo ""

# Fix broken dependencies first
echo "1. Fixing broken dependencies..."
apt-get install -f -y 2>&1 | grep -v "already the newest" || true

# Install all CUDA 11.8 runtime libraries that DGL might need
echo ""
echo "2. Installing CUDA 11.8 libraries..."

# Core libraries DGL needs
LIBRARIES=(
    "cuda-cudart-11-8"
    "cuda-cublas-11-8"
    "cuda-cublas-dev-11-8"
    "cuda-curand-11-8"
    "cuda-curand-dev-11-8"
    "cuda-cusolver-11-8"
    "cuda-cusolver-dev-11-8"
    "cuda-cusparse-11-8"
    "cuda-cusparse-dev-11-8"
    "cuda-cufft-11-8"
    "cuda-cufft-dev-11-8"
    "cuda-nvrtc-11-8"
    "cuda-nvrtc-dev-11-8"
)

for lib in "${LIBRARIES[@]}"; do
    echo "   Installing ${lib}..."
    apt-get install -y "${lib}" 2>&1 | grep -E "(Setting up|already the newest|E:)" || true
done

# Find all CUDA 11.8 library directories
echo ""
echo "3. Locating CUDA 11.8 libraries..."

# Primary location for CUDA 11.8
CUDA_11_8_LIB="/usr/local/cuda-11.8/targets/x86_64-linux/lib"
if [ -d "${CUDA_11_8_LIB}" ]; then
    echo "   ✓ Found: ${CUDA_11_8_LIB}"
    export LD_LIBRARY_PATH="${CUDA_11_8_LIB}:${LD_LIBRARY_PATH}"
fi

# Also check system lib directory
SYS_LIB="/usr/lib/x86_64-linux-gnu"
if find "${SYS_LIB}" -name "libcublas.so.11*" 2>/dev/null | grep -q .; then
    echo "   ✓ Found CUDA libs in: ${SYS_LIB}"
    export LD_LIBRARY_PATH="${SYS_LIB}:${LD_LIBRARY_PATH}"
fi

# Verify key libraries exist
echo ""
echo "4. Verifying required libraries..."
REQUIRED_LIBS=(
    "libcudart.so.11"
    "libcublas.so.11"
    "libcurand.so.11"
    "libcusolver.so.11"
    "libcusparse.so.11"
)

MISSING=()
for lib in "${REQUIRED_LIBS[@]}"; do
    if find /usr -name "${lib}*" 2>/dev/null | grep -q .; then
        echo "   ✓ ${lib}"
    else
        echo "   ✗ ${lib} - MISSING"
        MISSING+=("${lib}")
    fi
done

# Set LD_LIBRARY_PATH permanently
echo ""
echo "5. Setting LD_LIBRARY_PATH..."

# Build the path
NEW_PATH=""
if [ -d "${CUDA_11_8_LIB}" ]; then
    NEW_PATH="${CUDA_11_8_LIB}"
fi
if [ -d "${SYS_LIB}" ]; then
    if [ -n "${NEW_PATH}" ]; then
        NEW_PATH="${NEW_PATH}:${SYS_LIB}"
    else
        NEW_PATH="${SYS_LIB}"
    fi
fi

if [ -n "${NEW_PATH}" ]; then
    export LD_LIBRARY_PATH="${NEW_PATH}:${LD_LIBRARY_PATH}"
    
    # Add to bashrc
    if ! grep -q "LD_LIBRARY_PATH.*cuda-11.8" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA 11.8 libraries for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${NEW_PATH}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
        echo "   ✓ Added to ~/.bashrc"
    fi
    
    echo "   LD_LIBRARY_PATH includes: ${NEW_PATH}"
else
    echo "   ⚠ Could not find CUDA 11.8 library directory"
fi

# Test DGL
echo ""
echo "6. Testing DGL import..."
if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1; then
    echo "   ✓ SUCCESS! DGL works now."
    echo ""
    echo "   To make it permanent, run: source ~/.bashrc"
    exit 0
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -o "lib[^:]*\.so[^:]*" | head -1 || echo "unknown")
    echo "   ✗ DGL still fails (missing: ${ERROR})"
    echo ""
    echo "   Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo ""
    
    if [ ${#MISSING[@]} -gt 0 ]; then
        echo "   Missing libraries: ${MISSING[*]}"
        echo ""
        echo "   Try installing the specific missing library:"
        echo "   apt-get install -y cuda-<library-name>-11-8"
    fi
    
    echo ""
    echo "   Alternative: Use Docker container (includes all CUDA libs)"
    echo "   bash scripts/rfdiffusion_test.sh"
    exit 1
fi

