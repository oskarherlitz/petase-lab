#!/bin/bash
# Fix DGL by reinstalling for CUDA 12.4 compatibility OR finding existing CUDA 11.8 libs
# This is the proper solution - align DGL with PyTorch's CUDA version

set -e

echo "=========================================="
echo "Fixing DGL CUDA Compatibility"
echo "=========================================="
echo ""

# Step 1: Check current state
echo "1. Current State:"
PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "   PyTorch CUDA: ${PYTORCH_CUDA}"

# Step 2: Check if CUDA 12.4 libraries exist
echo ""
echo "2. Checking CUDA 12.4 libraries..."
CUBLAS_12=$(find /usr /usr/local -name "libcublas.so.12*" 2>/dev/null | head -1 || echo "")
if [ -n "${CUBLAS_12}" ]; then
    echo "   ✓ Found CUDA 12.4 cuBLAS: ${CUBLAS_12}"
    CUBLAS_12_DIR=$(dirname "${CUBLAS_12}")
else
    echo "   ✗ No CUDA 12.4 cuBLAS found"
fi

# Step 3: Try to reinstall DGL for CUDA 12.4 (BEST SOLUTION)
echo ""
echo "3. Attempting to reinstall DGL for CUDA 12.4 compatibility..."
echo "   (This is the cleanest solution - aligns DGL with PyTorch)"

# Uninstall current DGL
pip uninstall -y dgl 2>/dev/null || true

# Try installing DGL for CUDA 12 (if available) or CUDA 11.8
echo "   Installing DGL for CUDA 12.4..."
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html 2>&1 | grep -E "(Successfully|ERROR|WARNING)" || \
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html 2>&1 | grep -E "(Successfully|ERROR|WARNING)" || \
pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html 2>&1 | grep -E "(Successfully|ERROR|WARNING)" || \
echo "   ⚠ DGL install may have failed"

# Step 4: Test DGL
echo ""
echo "4. Testing DGL..."
if python3 -c "import dgl; print('✓ DGL imported successfully')" 2>&1 | grep -v "FutureWarning"; then
    echo "   ✓ SUCCESS! DGL works with CUDA 12.4"
    exit 0
fi

# Step 5: If that failed, check for CUDA 11.8 libraries and create symlinks
echo ""
echo "5. DGL still needs CUDA 11.8 libraries. Checking availability..."

CUDA_11_8_LIB="/usr/local/cuda-11.8/targets/x86_64-linux/lib"
if [ -d "${CUDA_11_8_LIB}" ]; then
    echo "   ✓ CUDA 11.8 lib directory exists: ${CUDA_11_8_LIB}"
    
    # Check what's actually there
    echo "   Available libraries:"
    ls -1 "${CUDA_11_8_LIB}"/libcublas.so* 2>/dev/null | head -3 || echo "     No libcublas.so found"
    ls -1 "${CUDA_11_8_LIB}"/libcudart.so* 2>/dev/null | head -3 || echo "     No libcudart.so found"
    
    # Set LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="${CUDA_11_8_LIB}:${LD_LIBRARY_PATH}"
    echo "   Set LD_LIBRARY_PATH to include: ${CUDA_11_8_LIB}"
    
    # Add to bashrc
    if ! grep -q "LD_LIBRARY_PATH.*cuda-11.8" ~/.bashrc 2>/dev/null; then
        echo "" >> ~/.bashrc
        echo "# CUDA 11.8 libraries for DGL" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${CUDA_11_8_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    fi
    
    # Test again
    if python3 -c "import dgl; print('✓ DGL works!')" 2>&1 | grep -v "FutureWarning"; then
        echo "   ✓ SUCCESS! DGL works with CUDA 11.8 libraries"
        exit 0
    fi
else
    echo "   ✗ CUDA 11.8 lib directory not found"
fi

# Step 6: Last resort - check if we can use CUDA 12.4 libraries with symlinks
echo ""
echo "6. Attempting symlink workaround..."

if [ -n "${CUBLAS_12}" ] && [ -d "${CUDA_11_8_LIB}" ]; then
    echo "   Creating symlink from CUDA 12.4 to CUDA 11.8 location..."
    CUBLAS_12_BASE=$(basename "${CUBLAS_12}")
    CUBLAS_11_NAME="libcublas.so.11"
    
    # Create symlink if it doesn't exist
    if [ ! -f "${CUDA_11_8_LIB}/${CUBLAS_11_NAME}" ] && [ -f "${CUBLAS_12}" ]; then
        ln -sf "${CUBLAS_12}" "${CUDA_11_8_LIB}/${CUBLAS_11_NAME}" 2>/dev/null || \
        echo "   ⚠ Could not create symlink (may need sudo)"
    fi
    
    # Test
    if python3 -c "import dgl; print('✓ DGL works!')" 2>&1 | grep -v "FutureWarning"; then
        echo "   ✓ SUCCESS with symlink workaround!"
        exit 0
    fi
fi

# Step 7: Final option - downgrade PyTorch to CUDA 11.8
echo ""
echo "7. Last resort: Downgrading PyTorch to CUDA 11.8..."
echo "   (This will align everything with CUDA 11.8)"
read -p "   Do you want to downgrade PyTorch? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Uninstalling PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    
    echo "   Installing PyTorch for CUDA 11.8..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
    
    # Reinstall DGL for CUDA 11.8
    pip uninstall -y dgl
    pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
    
    # Test
    if python3 -c "import dgl; print('✓ DGL works!')" 2>&1 | grep -v "FutureWarning"; then
        echo "   ✓ SUCCESS after PyTorch downgrade!"
        exit 0
    fi
else
    echo "   Skipped PyTorch downgrade"
fi

echo ""
echo "✗ Could not automatically fix DGL"
echo ""
echo "Manual options:"
echo "  1. Use Docker container (includes all CUDA libs): bash scripts/rfdiffusion_test.sh"
echo "  2. Install CUDA 11.8 toolkit manually from NVIDIA website"
echo "  3. Contact RunPod support about CUDA 11.8 library availability"
exit 1

