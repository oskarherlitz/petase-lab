#!/bin/bash
# Quick sequence check for RFdiffusion designs
# Extracts sequences and checks diversity at mask positions

set -e

RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
NUM_DESIGNS="${2:-10}"

echo "=========================================="
echo "RFdiffusion Sequence Check"
echo "=========================================="
echo "Directory: ${RESULTS_DIR}"
echo "Checking first ${NUM_DESIGNS} designs"
echo ""

# Conservative mask positions (PDB numbering)
CONSERVATIVE_POSITIONS=(114 117 119 140 159 165 168 180 188 205 214 269 282)

echo "Extracting sequences at conservative mask positions..."
echo ""

# Extract residue names at mask positions for each design
for i in $(seq 0 $((NUM_DESIGNS - 1))); do
    PDB_FILE="${RESULTS_DIR}/designs_${i}.pdb"
    if [ ! -f "${PDB_FILE}" ]; then
        echo "⚠ Design ${i} not found, skipping..."
        continue
    fi
    
    # Extract residue names at conservative positions
    # PDB starts at residue 29, so position 114 = line with residue number 114
    MASK_SEQ=""
    for pos in "${CONSERVATIVE_POSITIONS[@]}"; do
        # Get residue name at this position
        RESNAME=$(grep "^ATOM.* A .* ${pos} " "${PDB_FILE}" | head -1 | awk '{print $4}' || echo "XXX")
        if [ -z "${RESNAME}" ] || [ "${RESNAME}" = "XXX" ]; then
            RESNAME="---"
        fi
        MASK_SEQ="${MASK_SEQ}${RESNAME:0:3} "
    done
    
    echo "Design ${i}: ${MASK_SEQ}"
done

echo ""
echo "=========================================="
echo "Analysis"
echo "=========================================="
echo ""
echo "Check if:"
echo "  1. Sequences are different across designs (diversity)"
echo "  2. Mask positions show variation (redesign worked)"
echo "  3. Residue names are valid (not all GLY or XXX)"
echo ""
echo "If all sequences are identical, RFdiffusion may not have"
echo "applied the mask correctly. Check the TRB files for details."
echo ""

