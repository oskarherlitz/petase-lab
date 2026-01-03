#!/usr/bin/env bash
# Run ColabFold with GPU - Final working version

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_fasta> <output_dir>"
    echo ""
    echo "Example:"
    echo "  $0 runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta runs/colabfold_predictions_gpu"
    exit 1
fi

INPUT_FASTA="$1"
OUTPUT_DIR="$2"

# Set all required environment variables for GPU
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "=========================================="
echo "Running ColabFold with GPU"
echo "=========================================="
echo ""
echo "Input: $INPUT_FASTA"
echo "Output: $OUTPUT_DIR"
echo ""

# Verify GPU is detected
echo "Verifying GPU detection..."
python3 << 'VERIFY'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
devices = jax.devices()
backend = jax.default_backend()

if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print(f"✓ GPU detected: {devices}")
    print(f"✓ Backend: {backend}")
    exit(0)
else:
    print(f"✗ GPU not detected: {devices}")
    print(f"✗ Backend: {backend}")
    exit(1)
VERIFY

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: GPU not detected. Please fix GPU setup first."
    exit 1
fi

echo ""
echo "Starting ColabFold prediction..."
echo "This will run in the foreground. Use tmux to run in background."
echo ""

# Run ColabFold
colabfold_batch \
    --num-recycle 2 \
    --num-models 3 \
    --amber \
    "$INPUT_FASTA" \
    "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Prediction complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lh $OUTPUT_DIR/*.pdb"
echo "  ls -lh $OUTPUT_DIR/*.png"

