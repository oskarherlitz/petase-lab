#!/bin/bash
# Complete RFdiffusion setup for RunPod GPU pods
# This script consolidates all the fixes that actually worked
# Run this once when you first connect to a new RunPod GPU pod

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "RFdiffusion RunPod Setup"
echo "=========================================="
echo ""

# Step 1: Check GPU
echo "1. Checking GPU..."
if nvidia-smi &> /dev/null; then
    echo "   ✓ GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "   ✗ No GPU detected - this pod may not have GPU!"
    echo "   RFdiffusion requires GPU. Check your RunPod pod configuration."
    exit 1
fi

# Step 2: Install dependencies
echo ""
echo "2. Installing RFdiffusion dependencies..."
echo "   (This may take 5-10 minutes)"
bash scripts/install_rfdiffusion_deps.sh

# Step 3: Download model weights (if needed)
echo ""
echo "3. Checking model weights..."
MODELS_DIR="${PROJECT_ROOT}/data/models/rfdiffusion"
if [ -f "${MODELS_DIR}/Base_ckpt.pt" ] && [ -f "${MODELS_DIR}/ActiveSite_ckpt.pt" ]; then
    echo "   ✓ Model weights already exist"
    ls -lh "${MODELS_DIR}"/*.pt 2>/dev/null | head -2
else
    echo "   ✗ Model weights missing"
    echo "   Downloading model weights (~10GB, will take 15-30 minutes)..."
    bash scripts/fix_rfdiffusion_models.sh
fi

# Step 4: Download input PDB (if needed)
echo ""
echo "4. Checking input PDB..."
INPUT_PDB="${PROJECT_ROOT}/data/structures/7SH6/raw/7SH6.pdb"
if [ -f "${INPUT_PDB}" ]; then
    echo "   ✓ Input PDB exists"
else
    echo "   ✗ Input PDB missing"
    echo "   Downloading 7SH6.pdb..."
    mkdir -p "$(dirname "${INPUT_PDB}")"
    cd "$(dirname "${INPUT_PDB}")"
    wget -q https://files.rcsb.org/view/7SH6.pdb || curl -L -o 7SH6.pdb https://files.rcsb.org/view/7SH6.pdb
    cd "${PROJECT_ROOT}"
    echo "   ✓ Downloaded"
fi

# Step 5: Verify everything
echo ""
echo "5. Verifying setup..."
echo ""

# Check GPU
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "   ✓ PyTorch CUDA: Available"
else
    echo "   ✗ PyTorch CUDA: Not available"
fi

# Check DGL
if python3 -c "import dgl; print('OK')" 2>&1 | grep -q "OK"; then
    echo "   ✓ DGL: Working"
else
    echo "   ✗ DGL: Not working"
fi

# Check RFdiffusion
if python3 -c "import rfdiffusion; print('OK')" 2>&1 | grep -q "OK"; then
    echo "   ✓ RFdiffusion: Installed"
else
    echo "   ✗ RFdiffusion: Not installed"
fi

# Check models
if [ -f "${MODELS_DIR}/Base_ckpt.pt" ] && [ -f "${MODELS_DIR}/ActiveSite_ckpt.pt" ]; then
    echo "   ✓ Model weights: Present"
else
    echo "   ✗ Model weights: Missing"
fi

# Check PDB
if [ -f "${INPUT_PDB}" ]; then
    echo "   ✓ Input PDB: Present"
else
    echo "   ✗ Input PDB: Missing"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify GPU: bash scripts/check_gpu.sh"
echo "  2. Test run: bash scripts/rfdiffusion_tmux.sh test"
echo "  3. Overnight run: bash scripts/rfdiffusion_tmux.sh conservative"
echo ""

