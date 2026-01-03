#!/bin/bash
# Check RFdiffusion prerequisites before running

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Checking RFdiffusion prerequisites..."
echo ""

ERRORS=0

# Check 1: RFdiffusion installation
echo "1. Checking RFdiffusion installation..."
if python3 -c "import rfdiffusion" 2>/dev/null; then
    echo "   ✓ RFdiffusion installed"
else
    echo "   ✗ RFdiffusion NOT installed"
    echo "     Run: bash scripts/install_rfdiffusion_quick.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Input PDB
echo ""
echo "2. Checking input PDB..."
INPUT_PDB="${PROJECT_ROOT}/data/structures/7SH6/raw/7SH6.pdb"
if [ -f "${INPUT_PDB}" ]; then
    echo "   ✓ Input PDB found: ${INPUT_PDB}"
else
    echo "   ✗ Input PDB NOT found: ${INPUT_PDB}"
    echo "     Download it first"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Model weights
echo ""
echo "3. Checking model weights..."
MODELS_DIR="${PROJECT_ROOT}/data/models/rfdiffusion"
if [ -f "${MODELS_DIR}/Base_ckpt.pt" ]; then
    echo "   ✓ Base model found"
else
    echo "   ✗ Base model NOT found: ${MODELS_DIR}/Base_ckpt.pt"
    echo "     Run: bash scripts/rfdiffusion_quick_setup.sh"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "${MODELS_DIR}/ActiveSite_ckpt.pt" ]; then
    echo "   ✓ ActiveSite model found"
else
    echo "   ✗ ActiveSite model NOT found: ${MODELS_DIR}/ActiveSite_ckpt.pt"
    echo "     Run: bash scripts/rfdiffusion_quick_setup.sh"
    ERRORS=$((ERRORS + 1))
fi

# Check 4: PyTorch/CUDA
echo ""
echo "4. Checking PyTorch/CUDA..."
if python3 -c "import torch; print(f'   ✓ PyTorch {torch.__version__}'); print(f'   ✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    :
else
    echo "   ✗ PyTorch/CUDA issue"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All prerequisites met! Ready to run."
    exit 0
else
    echo "✗ Found $ERRORS issue(s). Fix them before running."
    exit 1
fi

