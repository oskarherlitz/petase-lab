#!/bin/bash
# Auto-detect PDB structure and fix contig for RFdiffusion

PDB_FILE="${1:-data/structures/7SH6/raw/7SH6.pdb}"

if [ ! -f "${PDB_FILE}" ]; then
    echo "Error: PDB file not found: ${PDB_FILE}"
    exit 1
fi

echo "Analyzing PDB structure..."
echo ""

# Find chain A first and last residue
CHAIN_A_FIRST=$(grep "^ATOM.* A " "${PDB_FILE}" 2>/dev/null | head -1 | cut -c23-26 | tr -d ' ' || echo "")
CHAIN_A_LAST=$(grep "^ATOM.* A " "${PDB_FILE}" 2>/dev/null | tail -1 | cut -c23-26 | tr -d ' ' || echo "")

if [ -z "${CHAIN_A_FIRST}" ]; then
    echo "Error: Chain A not found in PDB file"
    echo ""
    echo "Available chains:"
    grep "^ATOM" "${PDB_FILE}" | cut -c22 | sort -u
    exit 1
fi

echo "Chain A: residues ${CHAIN_A_FIRST} to ${CHAIN_A_LAST}"
echo ""

# Calculate length
LENGTH=$((CHAIN_A_LAST - CHAIN_A_FIRST + 1))
echo "Chain length: ${LENGTH} residues"
echo ""

# Suggest contig
if [ "${CHAIN_A_FIRST}" = "1" ]; then
    CONTIG="[A1-${CHAIN_A_LAST}]"
    echo "✓ PDB starts at residue 1 - contig is correct: ${CONTIG}"
else
    CONTIG="[A${CHAIN_A_FIRST}-${CHAIN_A_LAST}]"
    echo "⚠ PDB starts at residue ${CHAIN_A_FIRST}"
    echo "  Update contig to: ${CONTIG}"
    echo ""
    echo "To fix, update the scripts to use:"
    echo "  'contigmap.contigs=${CONTIG}'"
fi

