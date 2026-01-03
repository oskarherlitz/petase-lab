#!/bin/bash
# Install RFdiffusion dependencies directly (no Docker)
# For use on RunPod or systems without Docker

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RFDIFFUSION_DIR="${PROJECT_ROOT}/external/rfdiffusion"
SE3_DIR="${RFDIFFUSION_DIR}/env/SE3Transformer"

echo "=========================================="
echo "Installing RFdiffusion Dependencies"
echo "=========================================="
echo ""

# Check CUDA version (adjust if needed)
CUDA_VERSION="cu118"  # Default to CUDA 11.8, adjust if you have different version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    if [[ "$CUDA_VER" == "11.6" ]]; then
        CUDA_VERSION="cu116"
    elif [[ "$CUDA_VER" == "11.8" ]]; then
        CUDA_VERSION="cu118"
    fi
    echo "Detected CUDA: ${CUDA_VER}, using PyTorch for ${CUDA_VERSION}"
fi

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install DGL
echo "Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/${CUDA_VERSION}/repo.html

# Install other dependencies
echo "Installing other dependencies..."
pip install e3nn wandb pynvml hydra-core pyrsistent decorator "numpy<2.0"

# Install SE3Transformer
echo "Installing SE3Transformer..."
if [ -f "${SE3_DIR}/setup.py" ]; then
    cd "${SE3_DIR}"
    pip install -e .
    cd "${PROJECT_ROOT}"
else
    echo "⚠ Warning: SE3Transformer setup.py not found at ${SE3_DIR}"
    echo "Checking if submodule needs initialization..."
    if [ -d "${SE3_DIR}" ]; then
        echo "Directory exists, trying to install anyway..."
        cd "${SE3_DIR}"
        pip install -e . || echo "SE3Transformer install failed, continuing..."
        cd "${PROJECT_ROOT}"
    else
        echo "Error: SE3Transformer directory not found. May need to initialize submodule:"
        echo "  git submodule update --init --recursive"
        exit 1
    fi
fi

# Install RFdiffusion package
echo "Installing RFdiffusion..."
if [ -f "${RFDIFFUSION_DIR}/setup.py" ]; then
    cd "${RFDIFFUSION_DIR}"
    pip install -e . --no-deps
    cd "${PROJECT_ROOT}"
else
    echo "Error: RFdiffusion setup.py not found at ${RFDIFFUSION_DIR}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
echo "  python3 -c 'import rfdiffusion; print(\"✓ RFdiffusion installed\")'"
echo "  python3 -c 'import torch; print(f\"✓ PyTorch: {torch.__version__}\"); print(f\"✓ CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "Then run:"
echo "  bash scripts/rfdiffusion_direct.sh"
echo ""

