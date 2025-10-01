#!/usr/bin/env bash
set -euo pipefail
: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"
MODEL=${1:-runs/*relax*/outputs/*.pdb}
MUT=${2:-configs/rosetta/mutlist.mut}
TS=$(date +%F)
NAME=$(basename "${MUT}" .mut)
RUN="runs/${TS}_ddg_cart_${NAME}"
mkdir -p "${RUN}/outputs"
echo "# Run manifest" > "${RUN}/manifest.md"
echo "Tool: Rosetta cartesian_ddg" >> "${RUN}/manifest.md"
echo "Inputs: $MODEL, $MUT" >> "${RUN}/manifest.md"
"$ROSETTA_BIN/cartesian_ddg.linuxgccrelease"       -s "$MODEL"       -ddg:mut_file "$MUT"       -ddg:iterations 3       -ddg:cartesian       -score:weights ref2015_cart       -fa_max_dis 9.0       -ddg:legacy false       -ddg:json true       -out:path:all "${RUN}/outputs"
