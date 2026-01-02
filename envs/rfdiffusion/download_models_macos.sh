#!/bin/bash
# RFdiffusion Model Download Script for macOS
# Uses curl instead of wget (macOS default)

# Usage: bash download_models_macos.sh /path/to/download/directory
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

DOWNLOAD_DIR="$1"

# Create download directory
mkdir -p "${DOWNLOAD_DIR}"

# Model URLs
MODELS=(
    "http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt"
)

# Optional models
OPTIONAL_MODELS=(
    "http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt"
    "http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt"
)

echo "Downloading RFdiffusion model weights to: ${DOWNLOAD_DIR}"
echo ""

# Download required models
for url in "${MODELS[@]}"; do
    filename=$(basename "${url}")
    filepath="${DOWNLOAD_DIR}/${filename}"
    
    if [ -f "${filepath}" ]; then
        echo "✓ ${filename} already exists, skipping..."
    else
        echo "Downloading ${filename}..."
        curl -L -o "${filepath}" "${url}" || {
            echo "Error: Failed to download ${filename}"
            exit 1
        }
        echo "✓ ${filename} downloaded"
    fi
done

echo ""
echo "✓ All required models downloaded!"
echo ""
echo "Optional models available:"
for url in "${OPTIONAL_MODELS[@]}"; do
    filename=$(basename "${url}")
    echo "  - ${filename}"
done
echo ""
echo "To download optional models, run:"
for url in "${OPTIONAL_MODELS[@]}"; do
    filename=$(basename "${url}")
    echo "  curl -L -o ${DOWNLOAD_DIR}/${filename} ${url}"
done

