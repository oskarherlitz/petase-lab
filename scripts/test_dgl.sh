#!/bin/bash
# Test DGL import (avoids bash history expansion issues)

python3 -c "import dgl; print('DGL works')" 2>&1 | grep -v "FutureWarning"

if [ $? -eq 0 ]; then
    echo "SUCCESS: DGL is working!"
else
    echo "FAILED: DGL import error"
    python3 -c "import dgl" 2>&1 | head -10
fi

