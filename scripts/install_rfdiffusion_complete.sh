#!/bin/bash
# Complete RFdiffusion dependency installation
# Installs all required packages

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Installing Complete RFdiffusion Dependencies"
echo "=========================================="
echo ""

# Core dependencies from RFdiffusion Dockerfile
echo "1. Installing core dependencies..."
pip install \
    e3nn==0.3.3 \
    wandb==0.12.0 \
    pynvml==11.0.0 \
    decorator==5.1.0 \
    hydra-core==1.3.2 \
    pyrsistent==0.19.3 \
    "numpy<2.0" \
    omegaconf \
    || echo "Some packages may have failed"

# Install dllogger
echo ""
echo "2. Installing dllogger..."
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger || echo "dllogger install failed"

# Install DGL (already done, but ensure it's correct)
echo ""
echo "3. Verifying DGL..."
python3 -c "import dgl; print(f'DGL version: {dgl.__version__}')" 2>&1 | grep -v "FutureWarning" || echo "DGL not working"

# Install SE3Transformer
echo ""
echo "4. Installing SE3Transformer..."
SE3_DIR="${PROJECT_ROOT}/external/rfdiffusion/env/SE3Transformer"
if [ -f "${SE3_DIR}/setup.py" ]; then
    cd "${SE3_DIR}"
    pip install -e . || echo "SE3Transformer install failed"
    cd "${PROJECT_ROOT}"
else
    echo "⚠ SE3Transformer not found at ${SE3_DIR}"
fi

# Install RFdiffusion package
echo ""
echo "5. Installing RFdiffusion package..."
RFDIFFUSION_DIR="${PROJECT_ROOT}/external/rfdiffusion"
if [ -f "${RFDIFFUSION_DIR}/setup.py" ]; then
    cd "${RFDIFFUSION_DIR}"
    pip install -e . --no-deps || echo "RFdiffusion install failed"
    cd "${PROJECT_ROOT}"
else
    echo "⚠ RFdiffusion setup.py not found"
fi

# Verify installation
echo ""
echo "6. Verifying installation..."
python3 -c "
import sys
errors = []
try:
    import omegaconf
    print('✓ omegaconf')
except ImportError as e:
    errors.append(f'✗ omegaconf: {e}')

try:
    import hydra
    print('✓ hydra-core')
except ImportError as e:
    errors.append(f'✗ hydra-core: {e}')

try:
    import dgl
    print('✓ dgl')
except ImportError as e:
    errors.append(f'✗ dgl: {e}')

try:
    import rfdiffusion
    print('✓ rfdiffusion')
except ImportError as e:
    errors.append(f'✗ rfdiffusion: {e}')

if errors:
    print('')
    print('Errors:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
" 2>&1 | grep -v "FutureWarning"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="

