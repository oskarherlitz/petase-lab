#!/bin/bash
# Comprehensive diagnosis of DGL CUDA issues

echo "=========================================="
echo "DGL CUDA Diagnosis"
echo "=========================================="
echo ""

# 1. Check PyTorch CUDA version
echo "1. PyTorch CUDA Version:"
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda}')" 2>&1 | grep -v "FutureWarning" || echo "   ✗ PyTorch not available"

# 2. Check DGL version and what CUDA it was built for
echo ""
echo "2. DGL Installation:"
python3 -c "import dgl; print(f'   DGL version: {dgl.__version__}')" 2>&1 | head -5 || echo "   ✗ DGL not importable"

# 3. Check what CUDA libraries are available
echo ""
echo "3. Available CUDA Libraries:"
echo "   Searching for libcudart..."
find /usr /usr/local -name "libcudart.so*" 2>/dev/null | head -5
echo "   Searching for libcublas..."
find /usr /usr/local -name "libcublas.so*" 2>/dev/null | head -5
echo "   Searching for libcurand..."
find /usr /usr/local -name "libcurand.so*" 2>/dev/null | head -5

# 4. Check CUDA installations
echo ""
echo "4. CUDA Installations:"
ls -la /usr/local/cuda* 2>/dev/null || echo "   No /usr/local/cuda* directories"
ls -la /usr/local/ | grep cuda || echo "   No CUDA in /usr/local"

# 5. Check LD_LIBRARY_PATH
echo ""
echo "5. LD_LIBRARY_PATH:"
echo "   ${LD_LIBRARY_PATH}"

# 6. Check what DGL is actually looking for
echo ""
echo "6. DGL Error Details:"
python3 -c "import dgl" 2>&1 | head -10 || true

# 7. Check apt repositories
echo ""
echo "7. CUDA Packages Available in apt:"
apt-cache search cuda-cublas 2>/dev/null | head -10 || echo "   No cuda-cublas packages found"
apt-cache search cuda-11-8 2>/dev/null | head -10 || echo "   No cuda-11-8 packages found"

# 8. Check if CUDA 12.4 libraries exist
echo ""
echo "8. CUDA 12.4 Libraries:"
find /usr /usr/local -name "libcublas.so.12*" 2>/dev/null | head -5 || echo "   No CUDA 12.4 cuBLAS found"

# 9. Check DGL installation method
echo ""
echo "9. DGL Installation Info:"
pip show dgl 2>/dev/null | grep -E "(Name|Version|Location)" || echo "   DGL not installed via pip"

# 10. Check if we can reinstall DGL for CUDA 12
echo ""
echo "10. DGL CUDA Compatibility:"
echo "   Checking if DGL supports CUDA 12..."
pip index versions dgl 2>/dev/null | head -5 || echo "   Cannot check DGL versions"

