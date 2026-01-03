#!/bin/bash
# Fix DGL CUDA 11.8 using conda (for system Python, not conda Python)

set -e

echo "Installing CUDA 11.8 via conda for system Python..."
echo ""

# Accept conda TOS
if command -v conda &> /dev/null; then
    echo "1. Accepting conda Terms of Service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
fi

# Install CUDA 11.8 toolkit
echo ""
echo "2. Installing cudatoolkit=11.8..."
conda install -y -c conda-forge cudatoolkit=11.8 2>&1 | tail -10 || \
conda install -y -c nvidia cudatoolkit=11.8 2>&1 | tail -10 || \
echo "⚠ Conda install may have failed"

# Find conda CUDA libraries
echo ""
echo "3. Locating conda CUDA libraries..."
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
CONDA_LIB="${CONDA_BASE}/lib"

if [ -d "${CONDA_LIB}" ]; then
    CUDA_LIBS=$(find "${CONDA_LIB}" -name "libcudart.so.11*" 2>/dev/null | head -1 || echo "")
    if [ -n "${CUDA_LIBS}" ]; then
        echo "   ✓ Found CUDA 11.8 libs at: ${CONDA_LIB}"
        export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"
        echo "   LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        
        # Add to bashrc
        if ! grep -q "LD_LIBRARY_PATH.*miniconda3" ~/.bashrc 2>/dev/null; then
            echo "" >> ~/.bashrc
            echo "# CUDA 11.8 libraries from conda for DGL" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=\"${CONDA_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
            echo "   ✓ Added to ~/.bashrc"
        fi
    else
        echo "   ✗ CUDA 11.8 libs not found in ${CONDA_LIB}"
    fi
else
    echo "   ✗ Conda lib directory not found: ${CONDA_LIB}"
fi

# Test DGL (using system Python, not conda Python)
echo ""
echo "4. Testing DGL with system Python..."
if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
    echo "   ✓ SUCCESS! DGL is working."
    exit 0
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -o "lib[^:]*\.so[^:]*" | head -1 || echo "unknown")
    echo "   ✗ DGL still failing (missing: ${ERROR})"
    echo ""
    echo "   Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    exit 1
fi

