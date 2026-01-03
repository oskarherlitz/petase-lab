#!/bin/bash
# Install CUDA 11.8 runtime via conda (often easier than apt on RunPod)

set -e

echo "Installing CUDA 11.8 runtime via conda..."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# Install CUDA 11.8 toolkit via conda
echo "Installing cudatoolkit=11.8..."
conda install -y -c conda-forge cudatoolkit=11.8 || \
conda install -y -c nvidia cudatoolkit=11.8 || \
echo "⚠ Conda install failed, trying alternative..."

# Find where conda installed it
CONDA_PREFIX=$(conda info --base)
CUDA_LIB="${CONDA_PREFIX}/lib"

if [ -f "${CUDA_LIB}/libcudart.so.11"* ] 2>/dev/null; then
    echo "✓ Found CUDA 11.8 libs at: ${CUDA_LIB}"
    export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    echo "export LD_LIBRARY_PATH=\"${CUDA_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    echo "✓ Added to ~/.bashrc"
    
    # Test DGL
    echo ""
    echo "Testing DGL..."
    if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
        echo "✓ SUCCESS!"
    else
        echo "✗ Still failing"
    fi
else
    echo "✗ CUDA 11.8 libs not found after conda install"
fi

