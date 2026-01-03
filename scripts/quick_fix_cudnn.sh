#!/usr/bin/env bash
# Quick fix for cuDNN issue

set -euo pipefail

echo "Installing cuDNN for CUDA 12..."
pip install nvidia-cudnn-cu12

echo ""
echo "Finding cuDNN installation path..."
CUDNN_PATH=$(python3 << 'EOF'
import site
import os
for d in site.getsitepackages():
    cudnn_lib = os.path.join(d, "nvidia", "cudnn", "lib")
    if os.path.exists(cudnn_lib):
        print(cudnn_lib)
        break
EOF
)

if [ -z "$CUDNN_PATH" ]; then
    echo "ERROR: Could not find cuDNN installation"
    exit 1
fi

echo "Found cuDNN at: $CUDNN_PATH"

echo ""
echo "Setting up environment..."
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDNN_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo ""
echo "Testing GPU detection..."
python3 << 'TEST'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
devices = jax.devices()
backend = jax.default_backend()

print(f"Devices: {devices}")
print(f"Backend: {backend}")

if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print("\n✓✓✓ GPU DETECTED! ✓✓✓")
    exit(0)
else:
    print("\n⚠ Still CPU")
    exit(1)
TEST

if [ $? -eq 0 ]; then
    echo ""
    echo "SUCCESS! To use GPU, run:"
    echo "  export CUDA_HOME=/usr/local/cuda-12.4"
    echo "  export LD_LIBRARY_PATH=$CUDNN_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH"
    echo "  export XLA_PYTHON_CLIENT_PREALLOCATE=false"
    echo "  export XLA_PYTHON_CLIENT_ALLOCATOR=platform"
    echo "  colabfold_batch ..."
fi

