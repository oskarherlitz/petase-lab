#!/usr/bin/env bash
# Quick script to view ColabFold results

set -euo pipefail

RESULTS_DIR=${1:-runs/colabfold_predictions}

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "ColabFold Results Summary"
echo "========================"
echo ""

# Count completed structures
NUM_PDB=$(find "$RESULTS_DIR" -maxdepth 1 -name "*.pdb" -type f | wc -l | tr -d ' ')
NUM_COMPLETE=$(find "$RESULTS_DIR" -type d -name "candidate_*_env" | wc -l | tr -d ' ')

echo "Completed sequences: $NUM_COMPLETE / 68"
echo "PDB files generated: $NUM_PDB"
echo ""

if [ "$NUM_PDB" -eq 0 ]; then
    echo "⚠ No PDB files found yet. Structure prediction still in progress."
    echo ""
    echo "Check progress:"
    echo "  tail -f $RESULTS_DIR/log.txt"
    echo ""
    exit 0
fi

echo "Top structures (by filename):"
find "$RESULTS_DIR" -maxdepth 1 -name "*_relaxed_rank_1.pdb" -o -name "*_rank_1.pdb" | head -10 | while read pdb; do
    basename "$pdb"
done

echo ""
echo "To visualize a structure:"
echo "  pymol $RESULTS_DIR/candidate_1_relaxed_rank_1.pdb"
echo ""
echo "Or use ChimeraX:"
echo "  chimerax $RESULTS_DIR/candidate_1_relaxed_rank_1.pdb"
echo ""

