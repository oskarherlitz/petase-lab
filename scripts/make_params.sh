#!/usr/bin/env bash
set -euo pipefail
: "${ROSETTA_MAIN:?Set ROSETTA_MAIN to Rosetta source root}"
SDF=${1:-data/raw/ligands/BHET.sdf}
NAME=${2:-PETA}
python "$ROSETTA_MAIN/source/scripts/python/public/molfile_to_params.py"       -n "$NAME" -p "$NAME" --conformers-in-one-file "$SDF"
mkdir -p data/params
mv ${NAME}.params data/params/ || true
