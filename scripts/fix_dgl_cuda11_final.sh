#!/bin/bash
# Final fix for DGL CUDA 11.8 library issue on RunPod
# Tries multiple methods to get CUDA 11.8 libraries

set -e

echo "=========================================="
echo "Fixing DGL CUDA 11.8 Library Issue"
echo "=========================================="
echo ""

# Step 1: Find existing CUDA 11.8 libraries
echo "1. Searching for CUDA 11.8 libraries..."
bash scripts/find_cuda11_libs.sh

# Step 2: Try installing via conda (most reliable on RunPod)
echo ""
echo "2. Attempting to install CUDA 11.8 via conda..."
if command -v conda &> /dev/null; then
    echo "   Conda found, installing cudatoolkit=11.8..."
    conda install -y -c conda-forge cudatoolkit=11.8 2>&1 | tail -5 || \
    conda install -y -c nvidia cudatoolkit=11.8 2>&1 | tail -5 || \
    echo "   ⚠ Conda install failed"
    
    # Find conda CUDA libs
    if command -v conda &> /dev/null; then
        CONDA_PREFIX=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
        if [ -d "${CONDA_PREFIX}/lib" ]; then
            CUDA_LIB="${CONDA_PREFIX}/lib"
            if find "${CUDA_LIB}" -name "libcudart.so.11*" 2>/dev/null | grep -q .; then
                echo "   ✓ Found CUDA 11.8 libs in conda: ${CUDA_LIB}"
                export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
            fi
        fi
    fi
else
    echo "   Conda not found, skipping..."
fi

# Step 3: Try downloading CUDA 11.8 runtime directly
echo ""
echo "3. Trying to download CUDA 11.8 runtime libraries..."
mkdir -p /tmp/cuda11.8_libs
cd /tmp/cuda11.8_libs

# Try downloading from NVIDIA (this is a workaround)
echo "   Downloading libcudart.so.11.8..."
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb || \
curl -L -o cuda-repo.deb https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb || \
echo "   ⚠ Download failed"

if [ -f cuda-repo*.deb ]; then
    echo "   Extracting CUDA runtime..."
    dpkg-deb -x cuda-repo*.deb /tmp/cuda_extract 2>/dev/null || true
    if [ -d /tmp/cuda_extract/usr/local/cuda-11.8 ]; then
        CUDA_11_8_LIB="/tmp/cuda_extract/usr/local/cuda-11.8/targets/x86_64-linux/lib"
        if [ -d "${CUDA_11_8_LIB}" ]; then
            echo "   ✓ Found CUDA 11.8 libs in extracted package"
            export LD_LIBRARY_PATH="${CUDA_11_8_LIB}:${LD_LIBRARY_PATH}"
        fi
    fi
fi

cd - > /dev/null

# Step 4: Check PyTorch's CUDA libraries (might have CUDA 11.x bundled)
echo ""
echo "4. Checking PyTorch CUDA libraries..."
PYTORCH_CUDA_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
if [ -n "${PYTORCH_CUDA_LIB}" ] && [ -d "${PYTORCH_CUDA_LIB}" ]; then
    if find "${PYTORCH_CUDA_LIB}" -name "libcudart.so.11*" 2>/dev/null | grep -q .; then
        echo "   ✓ Found CUDA 11.x libs in PyTorch: ${PYTORCH_CUDA_LIB}"
        export LD_LIBRARY_PATH="${PYTORCH_CUDA_LIB}:${LD_LIBRARY_PATH}"
    fi
fi

# Step 5: Final check and test
echo ""
echo "5. Final LD_LIBRARY_PATH:"
echo "   ${LD_LIBRARY_PATH}"

# Add to bashrc
if [ -n "${LD_LIBRARY_PATH}" ]; then
    if ! grep -q "LD_LIBRARY_PATH.*cuda.*11" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA 11.8 libraries for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    fi
fi

# Test DGL
echo ""
echo "6. Testing DGL..."
if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
    echo "   ✓ SUCCESS! DGL is working."
    exit 0
else
    echo "   ✗ DGL still failing"
    echo ""
    echo "   Alternative: Use Docker container (includes all CUDA libs)"
    echo "   Or: Install CUDA 11.8 toolkit manually from NVIDIA website"
    exit 1
fi

