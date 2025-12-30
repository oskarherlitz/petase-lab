#!/usr/bin/env bash
# Rosetta cartesian relaxation script with validation
# Usage: bash scripts/rosetta_relax.sh [input.pdb]

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

# Validate and set input PDB
IN=${1:-data/structures/5XJH/raw/PETase_raw.pdb}

# Validate input file exists
if [ ! -f "$IN" ]; then
    echo "Error: Input PDB file not found: $IN" >&2
    exit 1
fi

if [ ! -r "$IN" ]; then
    echo "Error: Input PDB file is not readable: $IN" >&2
    exit 1
fi

# Set up run directory
TS=$(date +%Y-%m-%d)  # Explicit date format
RUN="runs/${TS}_relax_cart_v1"
mkdir -p "${RUN}/outputs"

# Create manifest
echo "# Run manifest" > "${RUN}/manifest.md"
echo "Tool: Rosetta relax (cartesian)" >> "${RUN}/manifest.md"
echo "Input: $IN" >> "${RUN}/manifest.md"
echo "Date: $(date)" >> "${RUN}/manifest.md"

# Detect and validate Rosetta binary
RELAX_BIN=""
if [[ -f "$ROSETTA_BIN/relax.static.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/relax.static.macosclangrelease" ]]; then
    RELAX_BIN="relax.static.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/relax.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/relax.macosclangrelease" ]]; then
    RELAX_BIN="relax.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/relax.linuxgccrelease" ]] && [[ -x "$ROSETTA_BIN/relax.linuxgccrelease" ]]; then
    RELAX_BIN="relax.linuxgccrelease"
else
    echo "Error: No executable Rosetta relax binary found in $ROSETTA_BIN" >&2
    echo "Expected one of:" >&2
    echo "  - relax.static.macosclangrelease" >&2
    echo "  - relax.macosclangrelease" >&2
    echo "  - relax.linuxgccrelease" >&2
    exit 1
fi

echo "Using Rosetta binary: $RELAX_BIN" >&2
echo "Input PDB: $IN" >&2
echo "Output directory: $RUN/outputs" >&2
echo "" >&2

# Record command in manifest
echo "Command:" >> "${RUN}/manifest.md"
echo "  $RELAX_BIN -s ${IN} -use_input_sc -nstruct 20 -relax:cartesian -score:weights ref2015_cart -relax:min_type lbfgs_armijo_nonmonotone" >> "${RUN}/manifest.md"

# Run Rosetta relax
if ! "$ROSETTA_BIN/$RELAX_BIN" \
    -s "$IN" \
    -use_input_sc \
    -nstruct 20 \
    -relax:cartesian \
    -score:weights ref2015_cart \
    -relax:min_type lbfgs_armijo_nonmonotone \
    -out:path:all "${RUN}/outputs"; then
    echo "" >&2
    echo "Error: Rosetta relax execution failed." >&2
    echo "Check the output above for details." >&2
    exit 1
fi

echo "" >&2
echo "✓ Rosetta relax completed successfully" >&2
echo "Results saved to: $RUN/outputs" >&2
