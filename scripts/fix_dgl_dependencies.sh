#!/bin/bash
# Fix missing DGL dependencies (torchdata, etc.)

set -e

echo "Installing missing DGL dependencies..."

# Install torchdata (required by newer DGL versions)
pip install torchdata

# Install other potential missing dependencies
pip install pyarrow 2>&1 | grep -E "(Successfully|already satisfied)" || true

# Test DGL
echo ""
echo "Testing DGL..."
if python3 -c "import dgl; print('✓ DGL works!')" 2>&1 | grep -v "FutureWarning" | grep -q "✓"; then
    echo "✓ SUCCESS! DGL is fully working."
else
    ERROR=$(python3 -c "import dgl" 2>&1 | head -5)
    echo "✗ Still failing:"
    echo "${ERROR}"
fi

