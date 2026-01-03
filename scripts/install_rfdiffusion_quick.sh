#!/bin/bash
# Quick RFdiffusion installation for RunPod
# Assumes PyTorch is already installed

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RFDIFFUSION_DIR="${PROJECT_ROOT}/external/rfdiffusion"
SE3_DIR="${RFDIFFUSION_DIR}/env/SE3Transformer"

echo "Installing RFdiffusion dependencies..."

# Install basic dependencies
pip install e3nn wandb pynvml hydra-core pyrsistent decorator "numpy<2.0" || true

# Install DGL (try cu118 first, fallback to cu116)
echo "Installing DGL..."
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html || \
pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html || \
echo "⚠ DGL install failed, may need manual install"

# Install SE3Transformer if it exists
if [ -f "${SE3_DIR}/setup.py" ]; then
    echo "Installing SE3Transformer..."
    cd "${SE3_DIR}"
    pip install -e .
    cd "${PROJECT_ROOT}"
else
    echo "⚠ SE3Transformer not found, skipping..."
fi

# Install RFdiffusion package
if [ -f "${RFDIFFUSION_DIR}/setup.py" ]; then
    echo "Installing RFdiffusion..."
    cd "${RFDIFFUSION_DIR}"
    pip install -e . --no-deps
    cd "${PROJECT_ROOT}"
    echo "✓ RFdiffusion installed!"
else
    echo "Error: RFdiffusion setup.py not found"
    exit 1
fi

echo ""
echo "Verification:"
python3 -c "import rfdiffusion; print('✓ RFdiffusion imported successfully')" || echo "⚠ Import failed"

