#!/bin/bash
# Final fix: Install compatible DGL and torchdata versions

set -e

echo "=========================================="
echo "Installing Compatible DGL and torchdata"
echo "=========================================="
echo ""

# Check current versions
echo "1. Current versions:"
pip show dgl 2>/dev/null | grep Version || echo "   DGL: not installed"
pip show torchdata 2>/dev/null | grep Version || echo "   torchdata: not installed"

# Uninstall incompatible versions
echo ""
echo "2. Uninstalling incompatible versions..."
pip uninstall -y dgl torchdata 2>/dev/null || true

# Install compatible versions
echo ""
echo "3. Installing compatible versions..."

# Option 1: Try DGL 2.1.0 with torchdata 0.8.0 (for CUDA 12.4)
echo "   Trying DGL 2.1.0 + torchdata 0.8.0..."
pip install torchdata==0.8.0
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html || \
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu124/repo.html || \
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu118/repo.html

# Test
echo ""
echo "4. Testing DGL..."
if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
    echo "   ✓ SUCCESS with DGL 2.1.0!"
    exit 0
fi

# Option 2: Try older DGL that doesn't need torchdata
echo ""
echo "5. Trying older DGL (1.1.3) without torchdata requirement..."
pip uninstall -y dgl torchdata
pip install "dgl==1.1.3+cu118" -f https://data.dgl.ai/wheels/cu118/repo.html || \
pip install "dgl==1.1.3" -f https://data.dgl.ai/wheels/cu118/repo.html

# Test
if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
    echo "   ✓ SUCCESS with DGL 1.1.3!"
    exit 0
fi

# Option 3: Try torchdata 0.9.0 (last version with datapipes)
echo ""
echo "6. Trying torchdata 0.9.0 (last version with datapipes)..."
pip uninstall -y dgl torchdata
pip install torchdata==0.9.0
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Final test
echo ""
echo "7. Final test..."
if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
    echo "   ✓ SUCCESS!"
    exit 0
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -E "(Error|ModuleNotFound)" | head -1 || echo "Unknown")
    echo "   ✗ Still failing: ${ERROR}"
    echo ""
    echo "   Manual fix:"
    echo "   pip install torchdata==0.9.0"
    echo "   pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html"
    exit 1
fi

