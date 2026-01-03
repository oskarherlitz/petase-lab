#!/usr/bin/env bash
# Force reinstall JAX with CUDA support

set -euo pipefail

echo "=========================================="
echo "Force Reinstalling JAX with CUDA Support"
echo "=========================================="
echo ""

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

echo "1. Completely removing JAX..."
pip uninstall -y jax jaxlib jax-cuda12-plugin 2>/dev/null || true
pip cache purge 2>/dev/null || true

echo ""
echo "2. Installing JAX with explicit CUDA 12 support..."
echo "   This may take a few minutes..."

# Install with explicit CUDA 12 and force reinstall
pip install --force-reinstall --no-cache-dir \
    "jax[cuda12_local]==0.4.23" \
    "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "3. Verifying installation..."
python3 << 'VERIFY'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jaxlib

print(f"JAX version: {jax.__version__}")
print(f"JAXlib version: {jaxlib.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# Check jaxlib location
import os
jaxlib_path = os.path.dirname(jaxlib.__file__)
print(f"\nJAXlib location: {jaxlib_path}")

# Try to import CUDA backend
try:
    from jaxlib import xla_extension
    print(f"XLA extension: {xla_extension}")
except Exception as e:
    print(f"Error: {e}")

devices = jax.devices()
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("\n✓ GPU detected!")
else:
    print("\n⚠ Still CPU - checking jaxlib package...")
    import subprocess
    result = subprocess.run(['pip', 'show', 'jaxlib'], capture_output=True, text=True)
    print(result.stdout)
VERIFY

echo ""
echo "4. If still CPU, the jaxlib package might be wrong..."
echo "   Check: pip show jaxlib | grep Version"
echo "   Should show: Version: 0.4.23+cuda12.cudnn89 (or similar with +cuda)"

