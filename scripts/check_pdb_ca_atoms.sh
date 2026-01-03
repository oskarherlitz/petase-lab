#!/bin/bash
# Check which residues have CA atoms (RFdiffusion requirement)

PDB_FILE="${1:-data/structures/7SH6/raw/7SH6.pdb}"

if [ ! -f "${PDB_FILE}" ]; then
    echo "Error: PDB file not found: ${PDB_FILE}"
    exit 1
fi

echo "Checking CA atoms in PDB (RFdiffusion requirement)..."
echo "File: ${PDB_FILE}"
echo ""

# Extract CA atoms for chain A - use simple grep (atom name " CA " with spaces)
echo "1. CA atoms in chain A (first 5):"
grep "^ATOM" "${PDB_FILE}" | grep " CA " | grep " A " | head -5
echo ""

# Get first and last residue with CA
FIRST_CA_RES=$(grep "^ATOM" "${PDB_FILE}" | grep " CA " | grep " A " | head -1 | cut -c23-26 | tr -d ' ')
LAST_CA_RES=$(grep "^ATOM" "${PDB_FILE}" | grep " CA " | grep " A " | tail -1 | cut -c23-26 | tr -d ' ')

echo "2. Residue range with CA atoms:"
if [ -n "${FIRST_CA_RES}" ] && [ -n "${LAST_CA_RES}" ]; then
    echo "   First CA residue: ${FIRST_CA_RES}"
    echo "   Last CA residue: ${LAST_CA_RES}"
    
    # Check if residue 29 has CA
    if grep "^ATOM" "${PDB_FILE}" | grep " CA " | grep " A " | grep -q " 29 "; then
        echo ""
        echo "✓ Residue 29 has CA atom"
    else
        echo ""
        echo "✗ Residue 29 does NOT have CA atom!"
        echo "RFdiffusion only sees residues with CA atoms."
        echo "Suggested contig: [A${FIRST_CA_RES}-${LAST_CA_RES}]"
    fi
else
    echo "   ✗ No CA atoms found in chain A!"
    exit 1
fi

# Count total CA atoms
CA_COUNT=$(grep "^ATOM" "${PDB_FILE}" | grep " CA " | grep " A " | wc -l | tr -d ' ')
echo ""
echo "3. Total CA atoms in chain A: ${CA_COUNT}"

echo ""
echo "=========================================="
echo "Recommended contig for RFdiffusion:"
echo "  [A${FIRST_CA_RES}-${LAST_CA_RES}]"
echo "=========================================="
echo ""
echo "Note: If RFdiffusion still fails, the PDB file on RunPod"
echo "      might be different from your local file. Check:"
echo "      grep '^ATOM' data/structures/7SH6/raw/7SH6.pdb | grep ' CA ' | head -1"
