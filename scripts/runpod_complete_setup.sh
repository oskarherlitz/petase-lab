#!/usr/bin/env bash
# Complete RunPod setup script - Run this after connecting to a fresh pod
# This installs everything needed for ColabFold

set -euo pipefail

echo "=========================================="
echo "RunPod ColabFold Complete Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] && [ ! -d "runs" ]; then
    echo "Error: Not in petase-lab directory"
    echo "Please run: cd /workspace/petase-lab"
    exit 1
fi

# Step 1: Install system tools
echo "Step 1: Installing system tools..."
apt-get update -qq
apt-get install -y python3 python3-pip git tmux > /dev/null 2>&1
echo "✓ System tools installed"
echo ""

# Step 2: Install ColabFold with compatible versions
echo "Step 2: Installing ColabFold with compatible versions..."
echo "This may take 5-10 minutes..."
echo ""

pip install \
  "numpy<2.0.0,>=1.21.6" \
  "jax[cuda12_local]==0.4.23" \
  "jaxlib==0.4.23" \
  "dm-haiku==0.0.11" \
  "colabfold[alphafold]==1.5.4" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "✓ ColabFold installed"
echo ""

# Step 3: Verify installation
echo "Step 3: Verifying installation..."
echo ""

echo "ColabFold version:"
colabfold_batch --version 2>&1 | head -1 || echo "Error getting version"

echo ""
echo "JAX GPU detection:"
python3 << EOF
import jax
try:
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"✓ Backend: {backend}")
    print(f"✓ Devices: {devices}")
    if 'cuda' in str(devices[0]) or 'gpu' in str(backend).lower():
        print("✓ GPU detected and working!")
    else:
        print("⚠ Warning: GPU not detected, will use CPU")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start tmux session:"
echo "   tmux new -s colabfold"
echo ""
echo "2. Inside tmux, run ColabFold:"
echo "   cd /workspace/petase-lab"
echo "   XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \\"
echo "   colabfold_batch --num-recycle 2 --num-models 3 --amber \\"
echo "     runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \\"
echo "     runs/colabfold_predictions_gpu |& tee colabfold.log"
echo ""
echo "3. Detach from tmux: Press Ctrl+B, then D"
echo ""
echo "4. Reattach later: tmux attach -t colabfold"
echo ""

