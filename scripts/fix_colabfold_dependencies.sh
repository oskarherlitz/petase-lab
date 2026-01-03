#!/usr/bin/env bash
# Fix ColabFold dependency conflicts
# Resolves conflicts between JAX 0.4.23 and ColabFold requirements

set -euo pipefail

echo "Fixing ColabFold dependency conflicts..."
echo ""

# Fix numpy first (ColabFold wants <2.0.0)
echo "Fixing numpy version..."
pip install "numpy<2.0.0,>=1.21.6"

# Try dm-haiku 0.0.11 (ColabFold requirement) - might work with JAX 0.4.23
echo ""
echo "Installing compatible dm-haiku version..."
pip install "dm-haiku==0.0.11"

# Verify versions
echo ""
echo "Checking installed versions:"
pip show numpy dm-haiku jax jaxlib | grep -E "Name|Version"

echo ""
echo "Testing if it works..."
python3 << EOF
try:
    import jax
    import haiku
    import numpy as np
    print(f"✓ JAX: {jax.__version__}")
    print(f"✓ Haiku: {haiku.__version__}")
    print(f"✓ NumPy: {np.__version__}")
    print(f"✓ Devices: {jax.devices()}")
    print("All imports successful!")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

echo ""
echo "✓ Done! Try running ColabFold now."

