#!/bin/bash
# Clean PDB file for RFdiffusion - extract chain A ATOM records only

INPUT_PDB="${1:-data/structures/7SH6/raw/7SH6.pdb}"
OUTPUT_PDB="${2:-data/structures/7SH6/raw/7SH6_clean.pdb}"

if [ ! -f "${INPUT_PDB}" ]; then
    echo "Error: Input PDB not found: ${INPUT_PDB}"
    exit 1
fi

echo "Cleaning PDB for RFdiffusion..."
echo "Input: ${INPUT_PDB}"
echo "Output: ${OUTPUT_PDB}"
echo ""

# Extract only ATOM records for chain A
# This removes HETATM (ligands, water) and other chains
grep "^ATOM.* A " "${INPUT_PDB}" > "${OUTPUT_PDB}.tmp"

# Check if we got any atoms
if [ ! -s "${OUTPUT_PDB}.tmp" ]; then
    echo "Error: No ATOM records found for chain A"
    exit 1
fi

# Count residues before and after
BEFORE_RES=$(grep "^ATOM" "${INPUT_PDB}" | cut -c23-26 | sort -u | wc -l)
AFTER_RES=$(cut -c23-26 "${OUTPUT_PDB}.tmp" | sort -u | wc -l)

echo "Residues before: ${BEFORE_RES}"
echo "Residues after: ${AFTER_RES}"

# Check first and last residue
FIRST_RES=$(head -1 "${OUTPUT_PDB}.tmp" | cut -c23-26 | tr -d ' ')
LAST_RES=$(tail -1 "${OUTPUT_PDB}.tmp" | cut -c23-26 | tr -d ' ')

echo "First residue: ${FIRST_RES}"
echo "Last residue: ${LAST_RES}"
echo ""

# Check for gaps (missing residues)
echo "Checking for missing residues..."
ALL_RES=$(cut -c23-26 "${OUTPUT_PDB}.tmp" | sort -u | tr -d ' ' | sort -n)
FIRST_NUM=$FIRST_RES
LAST_NUM=$LAST_RES

MISSING=()
for ((i=$FIRST_NUM; i<=$LAST_NUM; i++)); do
    if ! echo "${ALL_RES}" | grep -q "^${i}$"; then
        MISSING+=("${i}")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "⚠ Missing residues: ${MISSING[*]}"
    echo "  (This is normal for crystal structures)"
else
    echo "✓ No missing residues"
fi

# Move temp file to final location
mv "${OUTPUT_PDB}.tmp" "${OUTPUT_PDB}"

echo ""
echo "✓ Cleaned PDB saved to: ${OUTPUT_PDB}"
echo ""
echo "Suggested contig: [A${FIRST_RES}-${LAST_RES}]"
echo ""
echo "To use the cleaned PDB, update scripts to use:"
echo "  data/structures/7SH6/raw/7SH6_clean.pdb"

