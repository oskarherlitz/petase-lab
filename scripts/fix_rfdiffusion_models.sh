#!/bin/bash
# Fix corrupted RFdiffusion model files by re-downloading

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/data/models/rfdiffusion"

echo "=========================================="
echo "Fixing RFdiffusion Model Files"
echo "=========================================="
echo ""

# Check current files
echo "1. Checking current model files..."
bash scripts/check_rfdiffusion_models.sh

echo ""
echo "2. Re-downloading corrupted/incomplete models..."
echo ""

# Create models directory
mkdir -p "${MODELS_DIR}"

# Download Base model
echo "Downloading Base_ckpt.pt (~5GB)..."
if [ -f "${MODELS_DIR}/Base_ckpt.pt" ]; then
    echo "  Backing up existing file..."
    mv "${MODELS_DIR}/Base_ckpt.pt" "${MODELS_DIR}/Base_ckpt.pt.backup" || true
fi

cd "${MODELS_DIR}"
wget -c http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt || \
curl -L -o Base_ckpt.pt http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt

# Verify Base model
if [ -f "${MODELS_DIR}/Base_ckpt.pt" ]; then
    SIZE=$(du -h "${MODELS_DIR}/Base_ckpt.pt" | cut -f1)
    echo "  ✓ Base_ckpt.pt downloaded: ${SIZE}"
else
    echo "  ✗ Failed to download Base_ckpt.pt"
    exit 1
fi

# Download ActiveSite model
echo ""
echo "Downloading ActiveSite_ckpt.pt (~5GB)..."
if [ -f "${MODELS_DIR}/ActiveSite_ckpt.pt" ]; then
    echo "  Backing up existing file..."
    mv "${MODELS_DIR}/ActiveSite_ckpt.pt" "${MODELS_DIR}/ActiveSite_ckpt.pt.backup" || true
fi

wget -c http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt || \
curl -L -o ActiveSite_ckpt.pt http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt

# Verify ActiveSite model
if [ -f "${MODELS_DIR}/ActiveSite_ckpt.pt" ] && [ -s "${MODELS_DIR}/ActiveSite_ckpt.pt" ]; then
    SIZE=$(du -h "${MODELS_DIR}/ActiveSite_ckpt.pt" | cut -f1)
    echo "  ✓ ActiveSite_ckpt.pt downloaded: ${SIZE}"
else
    echo "  ✗ Failed to download ActiveSite_ckpt.pt"
    exit 1
fi

cd "${PROJECT_ROOT}"

# Final verification
echo ""
echo "3. Verifying downloaded models..."
bash scripts/check_rfdiffusion_models.sh

echo ""
echo "=========================================="
echo "Model Fix Complete!"
echo "=========================================="
echo ""
echo "If models are valid, try running again:"
echo "  bash scripts/rfdiffusion_tmux.sh test"
