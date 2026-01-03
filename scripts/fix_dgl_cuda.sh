#!/bin/bash
# Fix DGL CUDA library path issues on RunPod

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Fixing DGL CUDA Library Path"
echo "=========================================="
echo ""

# Check CUDA installation
echo "1. Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "unknown")
    echo "   CUDA Version (from nvidia-smi): ${CUDA_VER}"
else
    echo "   ⚠ nvidia-smi not found"
fi

# Find CUDA libraries
echo ""
echo "2. Finding CUDA libraries..."
CUDA_LIB_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda-11.8/lib64"
    "/usr/local/cuda-11.6/lib64"
    "/usr/local/cuda-12.4/lib64"
    "/usr/lib/x86_64-linux-gnu"
)

FOUND_LIBS=()
for path in "${CUDA_LIB_PATHS[@]}"; do
    if [ -d "${path}" ] && [ -f "${path}/libcudart.so"* ] 2>/dev/null; then
        echo "   ✓ Found CUDA libs at: ${path}"
        FOUND_LIBS+=("${path}")
    fi
done

if [ ${#FOUND_LIBS[@]} -eq 0 ]; then
    echo "   ✗ No CUDA libraries found in standard locations"
    echo "   Searching system..."
    CUDA_LIB=$(find /usr -name "libcudart.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    if [ -n "${CUDA_LIB}" ]; then
        echo "   ✓ Found at: ${CUDA_LIB}"
        FOUND_LIBS+=("${CUDA_LIB}")
    fi
fi

# Set LD_LIBRARY_PATH
if [ ${#FOUND_LIBS[@]} -gt 0 ]; then
    echo ""
    echo "3. Setting LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="${FOUND_LIBS[0]}:${LD_LIBRARY_PATH}"
    echo "   LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    
    # Add to bashrc for persistence
    if ! grep -q "export LD_LIBRARY_PATH.*cuda" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA library path for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${FOUND_LIBS[0]}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
        echo "   ✓ Added to ~/.bashrc"
    fi
else
    echo ""
    echo "3. ⚠ Could not find CUDA libraries automatically"
    echo "   You may need to install CUDA toolkit or set LD_LIBRARY_PATH manually"
fi

# Test DGL import
echo ""
echo "4. Testing DGL import..."
if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1; then
    echo "   ✓ DGL works!"
    exit 0
else
    echo "   ✗ DGL still fails"
    echo ""
    echo "5. Attempting to reinstall DGL..."
    
    # Detect PyTorch CUDA version
    PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
    echo "   PyTorch CUDA version: ${PYTORCH_CUDA}"
    
    # Determine DGL CUDA version to install
    if [[ "${PYTORCH_CUDA}" == "11.6"* ]] || [[ "${PYTORCH_CUDA}" == "11.8"* ]]; then
        DGL_CUDA="cu116"
        echo "   Installing DGL for CUDA 11.6..."
        pip uninstall -y dgl 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html
    elif [[ "${PYTORCH_CUDA}" == "12."* ]]; then
        DGL_CUDA="cu118"  # DGL doesn't have cu12, use cu118
        echo "   Installing DGL for CUDA 11.8 (closest match)..."
        pip uninstall -y dgl 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
    else
        echo "   ⚠ Unknown PyTorch CUDA version, trying cu118..."
        pip uninstall -y dgl 2>/dev/null || true
        pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
    fi
    
    # Test again
    echo ""
    echo "6. Testing DGL import again..."
    if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1; then
        echo "   ✓ DGL works after reinstall!"
        exit 0
    else
        echo "   ✗ DGL still fails after reinstall"
        echo ""
        echo "Manual fix:"
        echo "  1. Find CUDA lib directory: find /usr -name libcudart.so*"
        echo "  2. Export: export LD_LIBRARY_PATH=/path/to/cuda/lib64:\$LD_LIBRARY_PATH"
        echo "  3. Or install CUDA toolkit: apt-get install -y cuda-toolkit-11-8"
        exit 1
    fi
fi

