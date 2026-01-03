#!/bin/bash
# Validate RFdiffusion results
# Checks: file counts, PDB integrity, TRB metadata, sequence diversity

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${1:-${PROJECT_ROOT}/runs/2026-01-03_rfdiffusion_conservative}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

echo "=========================================="
echo "Validating RFdiffusion Results"
echo "=========================================="
echo "Directory: ${RESULTS_DIR}"
echo ""

# 1. File counts
echo "1. File counts:"
PDB_COUNT=$(ls -1 "${RESULTS_DIR}"/*.pdb 2>/dev/null | wc -l | tr -d ' ')
TRB_COUNT=$(ls -1 "${RESULTS_DIR}"/*.trb 2>/dev/null | wc -l | tr -d ' ')
echo "   PDB files: ${PDB_COUNT}"
echo "   TRB files: ${TRB_COUNT}"

if [ "${PDB_COUNT}" -eq "${TRB_COUNT}" ] && [ "${PDB_COUNT}" -gt 0 ]; then
    echo "   ✓ File counts match"
else
    echo "   ✗ File count mismatch or missing files"
fi
echo ""

# 2. Check PDB integrity
echo "2. Checking PDB integrity (first, middle, last):"
for pdb in "${RESULTS_DIR}/designs_0.pdb" "${RESULTS_DIR}/designs_149.pdb" "${RESULTS_DIR}/designs_299.pdb"; do
    if [ -f "${pdb}" ]; then
        ATOM_COUNT=$(grep -c "^ATOM" "${pdb}" 2>/dev/null || echo "0")
        CHAIN_A_COUNT=$(grep "^ATOM.* A " "${pdb}" | wc -l | tr -d ' ')
        echo "   $(basename ${pdb}): ${ATOM_COUNT} atoms, ${CHAIN_A_COUNT} chain A atoms"
        
        if [ "${ATOM_COUNT}" -lt 1000 ]; then
            echo "     ⚠ Warning: Fewer atoms than expected"
        fi
    fi
done
echo ""

# 3. Check TRB metadata
echo "3. Checking TRB metadata (sample from first design):"
if [ -f "${RESULTS_DIR}/designs_0.trb" ]; then
    python3 << EOF
import pickle
import sys

try:
    with open('${RESULTS_DIR}/designs_0.trb', 'rb') as f:
        trb = pickle.load(f)
    
    print(f"   Keys: {len(trb)} metadata fields")
    print(f"   Contigs: {trb.get('contigs', 'N/A')}")
    print(f"   Inpaint_seq: {trb.get('inpaint_seq', 'N/A')}")
    
    # Check if contig matches expected [A29-289]
    contigs = trb.get('contigs', '')
    if 'A29' in str(contigs) and '289' in str(contigs):
        print("   ✓ Contig matches expected range [A29-289]")
    else:
        print(f"   ⚠ Contig may not match expected: {contigs}")
    
    # Check if inpaint_seq has conservative mask positions
    inpaint = trb.get('inpaint_seq', '')
    conservative_positions = ['114', '117', '119', '140', '159', '165', '168', '180', '188', '205', '214', '269', '282']
    found_positions = [pos for pos in conservative_positions if pos in str(inpaint)]
    print(f"   Conservative mask positions found: {len(found_positions)}/{len(conservative_positions)}")
    
    if len(found_positions) >= 10:
        print("   ✓ Conservative mask appears correct")
    else:
        print("   ⚠ Conservative mask may be incomplete")
        
except Exception as e:
    print(f"   ✗ Error reading TRB: {e}")
    sys.exit(1)
EOF
else
    echo "   ✗ TRB file not found"
fi
echo ""

# 4. Check for sequence diversity
echo "4. Checking sequence diversity (first 5 designs):"
python3 << EOF
import sys
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

parser = PDBParser(QUIET=True)

sequences = []
for i in range(min(5, ${PDB_COUNT})):
    try:
        pdb_file = f'${RESULTS_DIR}/designs_{i}.pdb'
        structure = parser.get_structure('design', pdb_file)
        seq = ''
        for residue in structure.get_residues():
            if residue.id[0] == ' ' and residue.get_resname() != 'GLY':  # Skip HETATM and glycines
                seq += seq1(residue.get_resname())
        sequences.append(seq)
    except Exception as e:
        print(f"   Error reading designs_{i}.pdb: {e}")
        continue

if len(sequences) > 0:
    unique_seqs = len(set(sequences))
    print(f"   Unique sequences: {unique_seqs}/{len(sequences)}")
    if unique_seqs == len(sequences):
        print("   ✓ All sequences are unique (good diversity)")
    else:
        print("   ⚠ Some sequences are identical")
else:
    print("   ✗ Could not read sequences")
EOF

echo ""
echo "=========================================="
echo "Validation Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Visual inspection: Open a few PDBs in PyMOL"
echo "  2. Structural validation: Run AlphaFold on top designs"
echo "  3. Stability scoring: Run Rosetta/FoldX ΔΔG calculations"
echo ""

