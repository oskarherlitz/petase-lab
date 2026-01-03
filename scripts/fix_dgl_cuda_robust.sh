#!/bin/bash
# Robust fix for DGL CUDA library issues on RunPod
# Installs CUDA toolkit if needed

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Robust DGL CUDA Fix"
echo "=========================================="
echo ""

# Step 1: Find existing CUDA libraries
echo "1. Searching for CUDA libraries..."
CUDA_LIB=$(find /usr -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")

if [ -n "${CUDA_LIB}" ]; then
    echo "   ✓ Found CUDA libs at: ${CUDA_LIB}"
    export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    echo "   Set LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
else
    echo "   ✗ No CUDA libraries found"
    
    # Step 2: Check what CUDA version PyTorch expects
    echo ""
    echo "2. Checking PyTorch CUDA version..."
    PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "   PyTorch CUDA: ${PYTORCH_CUDA}"
    
    # Step 3: Install CUDA toolkit
    echo ""
    echo "3. Installing CUDA runtime libraries..."
    
    # Try to install cuda-toolkit (this will install the runtime)
    # On Ubuntu/Debian systems
    if command -v apt-get &> /dev/null; then
        # Add NVIDIA package repository if not already added
        if ! grep -q "nvidia" /etc/apt/sources.list.d/* 2>/dev/null; then
            echo "   Adding NVIDIA repository..."
            apt-get update
            apt-get install -y software-properties-common
            add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
            apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
        fi
        
        # Install CUDA runtime (smaller than full toolkit)
        echo "   Installing cuda-runtime-11-8..."
        apt-get update
        apt-get install -y cuda-runtime-11-8 || \
        apt-get install -y cuda-cudart-11-8 || \
        apt-get install -y libcudart11.0 || \
        echo "   ⚠ Could not install via apt, trying alternative..."
    fi
    
    # Alternative: Download and install CUDA runtime manually
    if [ -z "${CUDA_LIB}" ]; then
        echo ""
        echo "4. Trying alternative: Install CUDA runtime manually..."
        
        # Create CUDA lib directory
        CUDA_DIR="/usr/local/cuda-11.8"
        mkdir -p "${CUDA_DIR}/lib64"
        
        # Try to download libcudart
        echo "   Downloading CUDA runtime library..."
        cd /tmp
        
        # Try wget or curl
        if command -v wget &> /dev/null; then
            wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run || true
        elif command -v curl &> /dev/null; then
            curl -L -o cuda_11.8.0_520.61.05_linux.run https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run || true
        fi
        
        # If download worked, extract just the runtime
        if [ -f cuda_11.8.0_520.61.05_linux.run ]; then
            echo "   Extracting CUDA runtime..."
            sh cuda_11.8.0_520.61.05_linux.run --extract=/tmp/cuda_extract --silent || true
            # This is complex, let's try a simpler approach
        fi
    fi
fi

# Step 4: Set LD_LIBRARY_PATH permanently
echo ""
echo "5. Setting LD_LIBRARY_PATH permanently..."

# Find CUDA lib again (in case we just installed it)
CUDA_LIB=$(find /usr -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [ -z "${CUDA_LIB}" ]; then
    # Check standard locations
    for path in "/usr/local/cuda/lib64" "/usr/local/cuda-11.8/lib64" "/usr/lib/x86_64-linux-gnu"; do
        if [ -d "${path}" ]; then
            CUDA_LIB="${path}"
            break
        fi
    done
fi

if [ -n "${CUDA_LIB}" ]; then
    export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH}"
    
    # Add to bashrc
    if ! grep -q "LD_LIBRARY_PATH.*cuda" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA library path for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${CUDA_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
        echo "   ✓ Added to ~/.bashrc"
    fi
    
    echo "   LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
else
    echo "   ⚠ Still could not find CUDA libraries"
    echo ""
    echo "   Manual steps:"
    echo "   1. Check if CUDA is installed: ls -la /usr/local/cuda*"
    echo "   2. Install CUDA toolkit: apt-get install -y cuda-toolkit-11-8"
    echo "   3. Or set manually: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi

# Step 5: Test DGL
echo ""
echo "6. Testing DGL import..."
if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1; then
    echo "   ✓ SUCCESS! DGL works now."
    exit 0
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -o "libcudart.so[^:]*" | head -1 || echo "unknown")
    echo "   ✗ DGL still fails (looking for: ${ERROR})"
    echo ""
    echo "   Trying to install matching CUDA runtime..."
    
    # Try installing via conda if available (often easier)
    if command -v conda &> /dev/null; then
        echo "   Installing cudatoolkit via conda..."
        conda install -y -c conda-forge cudatoolkit=11.8 || \
        conda install -y -c nvidia cudatoolkit=11.8 || \
        echo "   ⚠ Conda install failed"
    fi
    
    # Final test
    if python3 -c "import dgl; print('✓ DGL works!')" 2>&1; then
        echo "   ✓ SUCCESS after conda install!"
        exit 0
    else
        echo ""
        echo "   ⚠ Could not automatically fix. Manual steps:"
        echo "   1. Find CUDA version: nvidia-smi"
        echo "   2. Install matching toolkit: apt-get install -y cuda-toolkit-<version>"
        echo "   3. Or use Docker/Singularity instead (see envs/rfdiffusion/)"
        exit 1
    fi
fi

