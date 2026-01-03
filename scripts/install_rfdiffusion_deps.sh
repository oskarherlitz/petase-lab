#!/bin/bash
# Install RFdiffusion dependencies for RunPod
# This consolidates the working solution:
# - Use system Python (not conda)
# - Install DGL 1.1.3+cu118 with torchdata 0.7.1
# - Install CUDA 11.8 libraries via conda
# - Install all Python dependencies
# - Install SE3Transformer and RFdiffusion as editable packages

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Installing RFdiffusion Dependencies"
echo "=========================================="
echo ""

# Deactivate conda if active (use system Python)
conda deactivate 2>/dev/null || true

# Find system Python (not conda)
SYSTEM_PYTHON=$(which -a python3 | grep -v miniconda | grep -v conda | head -1)
if [ -z "${SYSTEM_PYTHON}" ]; then
    SYSTEM_PYTHON="/usr/bin/python3"
fi

echo "Using system Python: ${SYSTEM_PYTHON}"
${SYSTEM_PYTHON} --version
echo ""

# Step 1: Install CUDA 11.8 libraries via conda (if conda is available)
echo "1. Installing CUDA 11.8 libraries..."
if command -v conda &> /dev/null; then
    echo "   Accepting conda TOS and installing CUDA 11.8..."
    conda install -y -c conda-forge cudatoolkit=11.8 2>&1 | grep -v "already satisfied" || true
    
    # Add conda CUDA libs to LD_LIBRARY_PATH
    if [ -d "${HOME}/miniconda3/lib" ]; then
        export LD_LIBRARY_PATH="${HOME}/miniconda3/lib:${LD_LIBRARY_PATH}"
        echo "   ✓ CUDA libraries from conda added to LD_LIBRARY_PATH"
    fi
else
    echo "   ⚠ Conda not found, skipping CUDA 11.8 installation"
    echo "   Will try to use system CUDA libraries"
fi

# Step 2: Install DGL and torchdata (the working combination)
echo ""
echo "2. Installing DGL and torchdata..."
echo "   (DGL 1.1.3+cu118 with torchdata 0.7.1 - this combination works)"
${SYSTEM_PYTHON} -m pip install --no-cache-dir --upgrade pip

# Uninstall any existing DGL/torchdata
${SYSTEM_PYTHON} -m pip uninstall -y dgl torchdata 2>/dev/null || true

# Install the working versions
${SYSTEM_PYTHON} -m pip install --no-cache-dir \
    "dgl==1.1.3+cu118" \
    -f https://data.dgl.ai/wheels/cu118/repo.html || \
    ${SYSTEM_PYTHON} -m pip install --no-cache-dir \
    "dgl==1.1.3" \
    -f https://data.dgl.ai/wheels/cu118/repo.html

${SYSTEM_PYTHON} -m pip install --no-cache-dir "torchdata==0.7.1"

# Step 3: Install other RFdiffusion Python dependencies
echo ""
echo "3. Installing other Python dependencies..."
${SYSTEM_PYTHON} -m pip install --no-cache-dir \
    e3nn==0.3.3 \
    wandb==0.12.0 \
    pynvml==11.0.0 \
    decorator==5.1.0 \
    hydra-core==1.3.2 \
    pyrsistent==0.19.3 \
    "numpy<2.0" \
    omegaconf \
    git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Step 4: Install SE3Transformer
echo ""
echo "4. Installing SE3Transformer..."
SE3_DIR="${PROJECT_ROOT}/external/rfdiffusion/env/SE3Transformer"
if [ -f "${SE3_DIR}/setup.py" ]; then
    cd "${SE3_DIR}"
    ${SYSTEM_PYTHON} -m pip install --no-cache-dir -e .
    cd "${PROJECT_ROOT}"
    echo "   ✓ SE3Transformer installed"
else
    echo "   ✗ SE3Transformer setup.py not found at ${SE3_DIR}"
    echo "   May need to initialize submodule: git submodule update --init --recursive"
    exit 1
fi

# Step 5: Install RFdiffusion package
echo ""
echo "5. Installing RFdiffusion package..."
RFDIFFUSION_DIR="${PROJECT_ROOT}/external/rfdiffusion"
if [ -f "${RFDIFFUSION_DIR}/setup.py" ]; then
    cd "${RFDIFFUSION_DIR}"
    ${SYSTEM_PYTHON} -m pip install --no-cache-dir -e . --no-deps
    cd "${PROJECT_ROOT}"
    echo "   ✓ RFdiffusion installed"
else
    echo "   ✗ RFdiffusion setup.py not found at ${RFDIFFUSION_DIR}"
    exit 1
fi

# Step 6: Set up CUDA library path permanently
echo ""
echo "6. Setting up CUDA library path..."
# Add to bashrc if not already there
if ! grep -q "CUDA.*libraries.*DGL" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# CUDA 11.8 libraries for DGL (RFdiffusion)" >> ~/.bashrc
    if [ -d "${HOME}/miniconda3/lib" ]; then
        echo "export LD_LIBRARY_PATH=\${HOME}/miniconda3/lib:\${LD_LIBRARY_PATH}" >> ~/.bashrc
    fi
    echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:\${LD_LIBRARY_PATH}" >> ~/.bashrc
    echo "   ✓ Added CUDA library paths to ~/.bashrc"
else
    echo "   ✓ CUDA library paths already in ~/.bashrc"
fi

# Step 7: Verify installation
echo ""
echo "7. Verifying installation..."
${SYSTEM_PYTHON} -c "
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
    print(f'✓ dgl ({dgl.__version__})')
except Exception as e:
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
echo ""
echo "Note: If you open a new terminal, run:"
echo "  source ~/.bashrc"
echo "  (or the LD_LIBRARY_PATH will be set automatically)"
echo ""
