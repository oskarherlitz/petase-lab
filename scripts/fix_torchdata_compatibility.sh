#!/bin/bash
# Fix torchdata compatibility issue with DGL

set -e

echo "Fixing torchdata compatibility..."

# Check current versions
echo "Current versions:"
python3 -c "import torchdata; print(f'torchdata: {torchdata.__version__}')" 2>/dev/null || echo "torchdata: not importable"
python3 -c "import dgl; print(f'dgl: {dgl.__version__}')" 2>&1 | grep -E "dgl:|Error" | head -1 || echo "dgl: not importable"

# Try to import the specific module
echo ""
echo "Testing torchdata.datapipes..."
python3 -c "from torchdata.datapipes.iter import IterDataPipe" 2>&1 | head -3 || echo "Module not found"

# Option 1: Reinstall torchdata
echo ""
echo "Reinstalling torchdata..."
pip uninstall -y torchdata 2>/dev/null || true
pip install torchdata

# Test again
echo ""
echo "Testing after reinstall..."
if python3 -c "from torchdata.datapipes.iter import IterDataPipe; print('torchdata.datapipes works')" 2>&1 | grep -q "works"; then
    echo "✓ torchdata.datapipes is now available"
else
    echo "✗ Still not working, trying alternative..."
    
    # Option 2: Install specific version
    echo ""
    echo "Trying specific torchdata version..."
    pip install "torchdata>=0.7.0" --upgrade
    
    # Test again
    if python3 -c "from torchdata.datapipes.iter import IterDataPipe; print('works')" 2>&1 | grep -q "works"; then
        echo "✓ Fixed with version upgrade"
    else
        echo "✗ Still failing, checking DGL version compatibility..."
        
        # Option 3: Check if we need to downgrade DGL
        DGL_VERSION=$(pip show dgl 2>/dev/null | grep Version | awk '{print $2}' || echo "unknown")
        echo "DGL version: ${DGL_VERSION}"
        
        # Try installing older DGL that doesn't need torchdata
        echo ""
        echo "Trying DGL 1.1.x (older version without torchdata requirement)..."
        pip uninstall -y dgl
        pip install "dgl==1.1.3+cu118" -f https://data.dgl.ai/wheels/cu118/repo.html || \
        pip install "dgl==1.1.3" -f https://data.dgl.ai/wheels/cu118/repo.html || \
        echo "Could not install older DGL"
    fi
fi

# Final test
echo ""
echo "Final DGL test..."
if python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning" | grep -q "DGL works"; then
    echo "✓ SUCCESS! DGL is working"
else
    ERROR=$(python3 -c "import dgl" 2>&1 | grep -E "(Error|ModuleNotFound)" | head -1 || echo "Unknown error")
    echo "✗ Still failing: ${ERROR}"
    echo ""
    echo "Try manually:"
    echo "  pip install 'torchdata>=0.7.0' --upgrade"
    echo "  pip install 'dgl==1.1.3+cu118' -f https://data.dgl.ai/wheels/cu118/repo.html"
fi

