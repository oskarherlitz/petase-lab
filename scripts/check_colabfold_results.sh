#!/usr/bin/env bash
# Check ColabFold results

RESULTS_DIR="${1:-runs/colabfold_predictions_gpu}"

echo "=========================================="
echo "ColabFold Results Check"
echo "=========================================="
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "✗ Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "Results directory: $RESULTS_DIR"
echo ""

# Count PDB files
PDB_COUNT=$(find "$RESULTS_DIR" -name "*.pdb" -type f 2>/dev/null | wc -l)
echo "PDB files (structures): $PDB_COUNT"

# Count PNG files (confidence plots)
PNG_COUNT=$(find "$RESULTS_DIR" -name "*.png" -type f 2>/dev/null | wc -l)
echo "PNG files (plots): $PNG_COUNT"

echo ""
echo "Sample files:"
ls -lh "$RESULTS_DIR"/*.pdb 2>/dev/null | head -5

echo ""
echo "All result files:"
ls -1 "$RESULTS_DIR" | head -20

echo ""
echo "To view a specific structure:"
echo "  The PDB files are in: $RESULTS_DIR/"
echo "  Best model for each candidate: candidate_X_rank_001_*.pdb"
echo "  All models: candidate_X_rank_*.pdb"

