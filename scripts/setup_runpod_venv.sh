#!/usr/bin/env bash
# Set up a clean virtual environment for ColabFold on RunPod
# This avoids dependency conflicts

set -euo pipefail

VENV_DIR=${1:-venv_colabfold}

echo "Setting up virtual environment for ColabFold..."
echo ""

# Create virtual environment
echo "1. Creating virtual environment: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# Activate it
echo ""
echo "2. Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "3. Upgrading pip..."
pip install --upgrade pip

# Install ColabFold with compatible versions
echo ""
echo "4. Installing ColabFold with compatible versions..."
pip install \
  "numpy<2.0.0,>=1.21.6" \
  "jax[cuda12_local]==0.4.23" \
  "jaxlib==0.4.23" \
  "dm-haiku==0.0.11" \
  "colabfold[alphafold]==1.5.4" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installation
echo ""
echo "5. Verifying installation..."
python3 -c "import jax; import colabfold; print('✓ JAX:', jax.__version__); print('✓ Devices:', jax.devices()); print('✓ ColabFold:', colabfold.__version__)"

echo ""
echo "✓ Virtual environment set up!"
echo ""
echo "To use it:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"

