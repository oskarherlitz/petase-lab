#!/bin/bash
# Setup script for ProGen2 virtual environment
# Run this from the repo root: bash scripts/setup_progen2_env.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/venv_progen2"

echo "Setting up ProGen2 virtual environment..."
echo "Repository root: ${REPO_ROOT}"
echo ""

# Create virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment already exists at ${VENV_DIR}"
    echo "Remove it first if you want to recreate: rm -rf ${VENV_DIR}"
else
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Upgrade pip, setuptools, wheel
echo ""
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU/MPS for Apple Silicon)
echo ""
echo "Installing PyTorch (CPU/MPS for Apple Silicon)..."
pip install torch torchvision torchaudio

# Install ProGen2 dependencies
# transformers will pull in a compatible tokenizers version automatically
echo ""
echo "Installing ProGen2 dependencies..."
cd "${REPO_ROOT}/external/progen2"
pip install transformers==4.16.2
# Note: tokenizers is installed automatically as a dependency of transformers
# No need to install separately - transformers 4.16.2 will pull in a compatible version

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
import tokenizers
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Tokenizers:', tokenizers.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if hasattr(torch.backends, 'mps'):
    print('✓ MPS available:', torch.backends.mps.is_available())
else:
    print('✓ MPS: Not available (older PyTorch version)')
"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run the smoke test:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python scripts/progen2_smoke_test.py run_20251229_progen2_small_r1_smoketest --num-samples 10"

