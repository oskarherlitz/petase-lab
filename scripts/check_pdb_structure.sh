#!/bin/bash
# Check PDB file structure for RFdiffusion compatibility

PDB_FILE="${1:-data/structures/7SH6/raw/7SH6.pdb}"

if [ ! -f "${PDB_FILE}" ]; then
    echo "Error: PDB file not found: ${PDB_FILE}"
    exit 1
fi

echo "Checking PDB structure: ${PDB_FILE}"
echo ""

# Check chains
echo "1. Chains in PDB:"
CHAINS=$(grep "^ATOM" "${PDB_FILE}" | cut -c22 | sort -u | tr -d '\n' || echo "none")
echo "   Found chains: ${CHAINS}"

# Check first and last residue numbers
echo ""
echo "2. Residue numbering:"
FIRST_RES=$(grep "^ATOM" "${PDB_FILE}" | head -1 | cut -c23-26 | tr -d ' ')
LAST_RES=$(grep "^ATOM" "${PDB_FILE}" | tail -1 | cut -c23-26 | tr -d ' ')
echo "   First residue: ${FIRST_RES}"
echo "   Last residue: ${LAST_RES}"

# Check chain A specifically
echo ""
echo "3. Chain A details:"
if echo "${CHAINS}" | grep -q "A"; then
    CHAIN_A_FIRST=$(grep "^ATOM.* A " "${PDB_FILE}" | head -1 | cut -c23-26 | tr -d ' ')
    CHAIN_A_LAST=$(grep "^ATOM.* A " "${PDB_FILE}" | tail -1 | cut -c23-26 | tr -d ' ')
    CHAIN_A_COUNT=$(grep "^ATOM.* A " "${PDB_FILE}" | wc -l)
    echo "   Chain A first residue: ${CHAIN_A_FIRST}"
    echo "   Chain A last residue: ${CHAIN_A_LAST}"
    echo "   Chain A atom count: ${CHAIN_A_COUNT}"
    
    # Check if it starts at 1
    if [ "${CHAIN_A_FIRST}" != "1" ]; then
        echo "   ⚠ WARNING: Chain A does not start at residue 1!"
        echo "   RFdiffusion expects contig [A1-290] but chain A starts at ${CHAIN_A_FIRST}"
    fi
else
    echo "   ✗ Chain A not found in PDB!"
    echo "   Available chains: ${CHAINS}"
fi

# Check for HETATM or other issues
echo ""
echo "4. Other checks:"
HETATM_COUNT=$(grep "^HETATM" "${PDB_FILE}" | wc -l)
if [ "${HETATM_COUNT}" -gt 0 ]; then
    echo "   Found ${HETATM_COUNT} HETATM records (ligands, water, etc.)"
fi

# Suggest contig fix
echo ""
echo "5. Suggested contig:"
if echo "${CHAINS}" | grep -q "A"; then
    if [ "${CHAIN_A_FIRST}" != "1" ]; then
        echo "   Use: contigmap.contigs=[A${CHAIN_A_FIRST}-${CHAIN_A_LAST}]"
    else
        echo "   Use: contigmap.contigs=[A1-${CHAIN_A_LAST}]"
    fi
else
    FIRST_CHAIN=$(echo "${CHAINS}" | cut -c1)
    FIRST_CHAIN_FIRST=$(grep "^ATOM.* ${FIRST_CHAIN} " "${PDB_FILE}" | head -1 | cut -c23-26 | tr -d ' ')
    FIRST_CHAIN_LAST=$(grep "^ATOM.* ${FIRST_CHAIN} " "${PDB_FILE}" | tail -1 | cut -c23-26 | tr -d ' ')
    echo "   Use: contigmap.contigs=[${FIRST_CHAIN}${FIRST_CHAIN_FIRST}-${FIRST_CHAIN_LAST}]"
fi

