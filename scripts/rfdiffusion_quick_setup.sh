#!/bin/bash
# Quick Setup Script for RFdiffusion Overnight Run
# This script prepares everything needed to run RFdiffusion overnight

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "RFdiffusion Overnight Run Setup"
echo "=========================================="
echo ""

# Step 1: Download model weights
echo "Step 1: Downloading RFdiffusion model weights..."
echo "This will take 15-30 minutes and download ~10GB"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p data/models/rfdiffusion
    if [[ "$OSTYPE" == "darwin"* ]]; then
        bash envs/rfdiffusion/download_models_macos.sh data/models/rfdiffusion
    else
        bash external/rfdiffusion/scripts/download_models.sh data/models/rfdiffusion
    fi
else
    echo "Skipping model download. Run manually later."
fi

# Step 2: Download 7SH6 structure
echo ""
echo "Step 2: Downloading FAST-PETase structure (7SH6)..."
mkdir -p data/structures/7SH6/raw
cd data/structures/7SH6/raw

if [ ! -f "7SH6.pdb" ]; then
    if command -v curl &> /dev/null; then
        curl -O https://files.rcsb.org/view/7SH6.pdb
    elif command -v wget &> /dev/null; then
        wget https://files.rcsb.org/view/7SH6.pdb
    else
        echo "Error: Need curl or wget to download structure"
        exit 1
    fi
    echo "✓ Downloaded 7SH6.pdb"
else
    echo "✓ 7SH6.pdb already exists"
fi

cd "${PROJECT_ROOT}"

# Step 3: Create directory structure
echo ""
echo "Step 3: Creating directory structure..."
mkdir -p configs/rfdiffusion
mkdir -p runs/$(date +%Y-%m-%d)_rfdiffusion_conservative
mkdir -p runs/$(date +%Y-%m-%d)_rfdiffusion_aggressive
echo "✓ Directories created"

# Step 4: Check if structure needs cleaning
echo ""
echo "Step 4: Checking structure..."
if [ -f "data/structures/7SH6/raw/7SH6.pdb" ]; then
    CHAINS=$(grep "^ATOM" data/structures/7SH6/raw/7SH6.pdb | cut -c22 | sort -u | tr -d '\n')
    echo "Found chains: $CHAINS"
    if [[ "$CHAINS" != "A" ]]; then
        echo "⚠ Warning: Structure has multiple chains or non-A chain"
        echo "You may need to extract chain A manually"
        echo "Consider: grep '^ATOM.* A ' 7SH6.pdb > 7SH6_chainA.pdb"
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review and edit configs/rfdiffusion/conservative_mask.json"
echo "2. Review and edit configs/rfdiffusion/aggressive_mask.json"
echo "3. Test with small run (5 designs):"
echo "   bash scripts/rfdiffusion_test.sh"
echo "4. Run overnight:"
echo "   bash scripts/rfdiffusion_conservative.sh"
echo "   bash scripts/rfdiffusion_aggressive.sh"
echo ""

