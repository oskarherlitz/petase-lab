#!/bin/bash
# Fix CUDA library path on RunPod GPU pods
# RunPod GPU pods have CUDA libraries but they may not be in LD_LIBRARY_PATH

set -e

echo "=========================================="
echo "Fixing CUDA Library Path for DGL"
echo "=========================================="
echo ""

# Find CUDA libraries
echo "1. Searching for CUDA libraries..."

# Common locations on RunPod GPU pods
CUDA_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-11.8/lib64"
    "/usr/local/cuda-11.8/targets/x86_64-linux/lib"
    "/usr/local/cuda-12.4/lib64"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/nvidia/lib64"
    "/usr/local/nvidia/lib"
)

FOUND_LIBS=()
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "${path}" ] && find "${path}" -name "libcudart.so*" 2>/dev/null | grep -q .; then
        echo "   ✓ Found at: ${path}"
        FOUND_LIBS+=("${path}")
    fi
done

# Also search system-wide
if [ ${#FOUND_LIBS[@]} -eq 0 ]; then
    echo "   Searching system-wide..."
    CUDA_LIB=$(find /usr /usr/local -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -n "${CUDA_LIB}" ]; then
        echo "   ✓ Found at: ${CUDA_LIB}"
        FOUND_LIBS+=("${CUDA_LIB}")
    fi
fi

# Set LD_LIBRARY_PATH
if [ ${#FOUND_LIBS[@]} -gt 0 ]; then
    echo ""
    echo "2. Setting LD_LIBRARY_PATH..."
    
    # Build the path
    NEW_PATH=""
    for lib_path in "${FOUND_LIBS[@]}"; do
        if [ -z "${NEW_PATH}" ]; then
            NEW_PATH="${lib_path}"
        else
            NEW_PATH="${NEW_PATH}:${lib_path}"
        fi
    done
    
    export LD_LIBRARY_PATH="${NEW_PATH}:${LD_LIBRARY_PATH}"
    echo "   LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    
    # Add to bashrc
    if ! grep -q "LD_LIBRARY_PATH.*cuda" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA library path for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${NEW_PATH}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
        echo "   ✓ Added to ~/.bashrc"
    fi
else
    echo "   ✗ No CUDA libraries found"
    echo ""
    echo "   On RunPod GPU pods, CUDA libraries should be available."
    echo "   Try:"
    echo "     find /usr -name libcudart.so*"
    echo "     find /usr/local -name libcudart.so*"
fi

# Test DGL
echo ""
echo "3. Testing DGL import..."
if python3 -c "import dgl; print('✓ DGL works!')" 2>&1 | grep -v "FutureWarning" | grep -q "✓"; then
    echo "   ✓ SUCCESS! DGL is working."
    echo ""
    echo "   To make it permanent in new shells:"
    echo "     source ~/.bashrc"
    exit 0
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -o "lib[^:]*\.so[^:]*" | head -1 || echo "unknown")
    echo "   ✗ DGL still fails (missing: ${ERROR})"
    echo ""
    echo "   Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    echo ""
    echo "   Try manually:"
    echo "     export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH"
    echo "     python3 -c 'import dgl'"
    exit 1
fi

