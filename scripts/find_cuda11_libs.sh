#!/bin/bash
# Find CUDA 11.x libraries on RunPod

echo "Searching for CUDA 11.x libraries (needed by DGL 1.1.3)..."
echo ""

# Search common locations
echo "1. Searching /usr/local..."
find /usr/local -name "libcudart.so.11*" 2>/dev/null | head -10

echo ""
echo "2. Searching /usr..."
find /usr -name "libcudart.so.11*" 2>/dev/null | head -10

echo ""
echo "3. Checking if CUDA 11.8 directory exists..."
if [ -d "/usr/local/cuda-11.8" ]; then
    echo "   ✓ /usr/local/cuda-11.8 exists"
    find /usr/local/cuda-11.8 -name "libcudart.so.11*" 2>/dev/null | head -5
else
    echo "   ✗ /usr/local/cuda-11.8 not found"
fi

echo ""
echo "4. Checking what CUDA libraries ARE available..."
find /usr/local/nvidia -name "libcudart.so*" 2>/dev/null | head -10
find /usr/local/cuda* -name "libcudart.so*" 2>/dev/null 2>/dev/null | head -10

echo ""
echo "5. Checking PyTorch CUDA version..."
python3 -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')" 2>&1 | grep -v "FutureWarning"

