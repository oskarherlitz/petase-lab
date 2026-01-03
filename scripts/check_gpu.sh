#!/bin/bash
# Check GPU availability for RFdiffusion

echo "=========================================="
echo "GPU Availability Check"
echo "=========================================="
echo ""

# Check nvidia-smi
echo "1. NVIDIA GPU (nvidia-smi):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || nvidia-smi
    echo ""
    echo "   GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
else
    echo "   ✗ nvidia-smi not found - no NVIDIA GPU detected"
fi

# Check PyTorch CUDA
echo ""
echo "2. PyTorch CUDA:"
python3 -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'   GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'   GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None" 2>&1 | grep -v "FutureWarning"

# Check DGL backend
echo ""
echo "3. DGL Backend:"
python3 -c "import dgl; print(f'   DGL version: {dgl.__version__}'); print(f'   Backend: {dgl.backend.get_backend()}')" 2>&1 | grep -v "FutureWarning" || echo "   ✗ DGL not available"

# Check CUDA devices
echo ""
echo "4. CUDA Devices:"
if command -v nvidia-smi &> /dev/null; then
    echo "   Available GPUs:"
    nvidia-smi -L
else
    echo "   ✗ No GPUs detected"
fi

echo ""
echo "=========================================="
echo "Summary:"
if command -v nvidia-smi &> /dev/null && python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✓ GPU is available and PyTorch can use it"
    echo "  RFdiffusion will run on GPU (fast)"
else
    echo "✗ GPU not available or PyTorch can't access it"
    echo "  RFdiffusion will run on CPU (VERY SLOW - not recommended)"
    echo ""
    echo "  Troubleshooting:"
    echo "  1. Check if RunPod pod has GPU: nvidia-smi"
    echo "  2. Check PyTorch CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'"
    echo "  3. Restart pod if GPU was just added"
fi
echo "=========================================="

