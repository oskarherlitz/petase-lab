#!/bin/bash
# Quick fix: Reinstall DGL for CUDA 12.4 (simplest solution)

set -e

echo "Reinstalling DGL for CUDA 12.4 compatibility..."

# Uninstall current DGL
pip uninstall -y dgl 2>/dev/null || true

# Install DGL - try CUDA 12.4 first, fallback to 11.8
echo "Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html || \
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html || \
pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html

# Set LD_LIBRARY_PATH to include CUDA 11.8 if it exists
CUDA_11_8_LIB="/usr/local/cuda-11.8/targets/x86_64-linux/lib"
if [ -d "${CUDA_11_8_LIB}" ]; then
    export LD_LIBRARY_PATH="${CUDA_11_8_LIB}:${LD_LIBRARY_PATH}"
    echo "export LD_LIBRARY_PATH=\"${CUDA_11_8_LIB}:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
fi

# Test
echo ""
echo "Testing DGL..."
python3 -c "import dgl; print('✓ DGL works!')" 2>&1 | grep -v "FutureWarning" && echo "SUCCESS!" || echo "Still failing - run: bash scripts/diagnose_dgl_cuda.sh"

