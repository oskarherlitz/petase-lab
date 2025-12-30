#!/usr/bin/env bash
# Setup script to prepare initial data for PETase optimization
# This copies the repaired structure and sets up initial files

set -euo pipefail

echo "Setting up initial data for PETase optimization..."

# Check if source file exists
SOURCE_PDB="data/structures/5XJH/foldx/5XJH_Repair.pdb"
TARGET_PDB="data/structures/5XJH/raw/PETase_raw.pdb"

if [ ! -f "$SOURCE_PDB" ]; then
    echo "Error: Source file $SOURCE_PDB not found!"
    echo "Please ensure the FoldX repaired structure exists."
    exit 1
fi

# Copy repaired structure
echo "Copying repaired structure..."
cp "$SOURCE_PDB" "$TARGET_PDB"
echo "✓ Copied to $TARGET_PDB"

# Create results directory if it doesn't exist
mkdir -p results/ddg_scans
mkdir -p runs
echo "✓ Created results directories"

# Verify the structure
echo ""
echo "Verifying structure..."
ATOM_COUNT=$(grep -c "^ATOM" "$TARGET_PDB" || echo "0")
echo "  Found $ATOM_COUNT ATOM records"

# Check for key residues
echo "  Checking for key catalytic residues:"
for res in "SER A 160" "ASP A 206" "HIS A 237" "ASP A 150"; do
    if grep -q "$res" "$TARGET_PDB"; then
        echo "    ✓ Found $res"
    else
        echo "    ✗ Missing $res"
    fi
done

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set ROSETTA_BIN environment variable:"
echo "   export ROSETTA_BIN=/path/to/rosetta/main/source/bin"
echo ""
echo "2. Run Rosetta relaxation:"
echo "   bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb"
echo ""
echo "3. Review mutation list:"
echo "   cat configs/rosetta/mutlist.mut"
echo ""
echo "4. Run ΔΔG calculations:"
echo "   bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut"

