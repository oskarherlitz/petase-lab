#!/bin/bash
# Analyze RFdiffusion speed to determine if using GPU or CPU

echo "Analyzing RFdiffusion timestep speed..."
echo ""

# Typical speeds:
echo "Expected speeds:"
echo "  GPU (RTX 4090): ~1-2 seconds per timestep (~1-2 min per design)"
echo "  GPU (RTX 3090): ~2-3 seconds per timestep (~2-3 min per design)"
echo "  CPU: ~30-60+ seconds per timestep (~25-50+ min per design)"
echo ""

# Calculate from your log
echo "Your observed speeds:"
echo "  Timestep 36→35: ~34 seconds"
echo "  Timestep 35→34: ~37 seconds"
echo "  Timestep 34→33: ~33 seconds"
echo "  Timestep 33→32: ~35 seconds"
echo ""
echo "Average: ~35 seconds per timestep"
echo ""

# Estimate total time
TIMESTEPS=50
SEC_PER_STEP=35
TOTAL_SEC=$((TIMESTEPS * SEC_PER_STEP))
TOTAL_MIN=$((TOTAL_SEC / 60))

echo "Estimated time per design:"
echo "  ${TOTAL_SEC} seconds (~${TOTAL_MIN} minutes)"
echo ""

echo "For 5 designs: ~$((TOTAL_MIN * 5)) minutes (~$((TOTAL_MIN * 5 / 60)) hours)"
echo "For 300 designs: ~$((TOTAL_MIN * 300)) minutes (~$((TOTAL_MIN * 300 / 60)) hours = ~$((TOTAL_MIN * 300 / 60 / 24)) days)"
echo ""

if [ $SEC_PER_STEP -lt 5 ]; then
    echo "✓ This is GPU speed (good!)"
elif [ $SEC_PER_STEP -lt 10 ]; then
    echo "⚠ This is slow GPU or fast CPU (check GPU usage)"
else
    echo "✗ This is CPU speed (VERY SLOW - not recommended)"
    echo ""
    echo "  Check GPU:"
    echo "    nvidia-smi"
    echo "    python3 -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
fi

