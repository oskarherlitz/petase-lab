#!/usr/bin/env bash
# Setup script for ColabFold on RunPod
# Run this on your RunPod instance after connecting

set -euo pipefail

echo "ColabFold Setup on RunPod"
echo "========================"
echo ""

# Update system
echo "1. Updating system packages..."
apt-get update -qq
apt-get install -y python3 python3-pip git wget curl > /dev/null 2>&1

# Install ColabFold
echo "2. Installing ColabFold..."
pip install --quiet "colabfold[alphafold]"

# Verify installation
echo "3. Verifying installation..."
colabfold_batch --version

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Clone your repo: git clone git@github.com:oskarherlitz/petase-lab.git"
echo "  2. Or upload your FASTA file"
echo "  3. Run ColabFold: colabfold_batch --num-recycle 3 --num-models 5 --amber your_file.fasta output_dir"
echo ""

