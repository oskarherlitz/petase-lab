#!/usr/bin/env bash
echo "Quick GPU Check:"
echo ""
echo "1. GPU Hardware:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "2. JAX GPU Detection:"
python3 -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
