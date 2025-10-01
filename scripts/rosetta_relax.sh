#!/usr/bin/env bash
set -euo pipefail
: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"
IN=${1:-data/raw/PETase_raw.pdb}
TS=$(date +%F)
RUN="runs/${TS}_relax_cart_v1"
mkdir -p "${RUN}/outputs"
echo "# Run manifest" > "${RUN}/manifest.md"
echo "Tool: Rosetta relax (cartesian)" >> "${RUN}/manifest.md"
echo "Command:" >> "${RUN}/manifest.md"
echo "  relax.linuxgccrelease -s ${IN} -use_input_sc -nstruct 20 -relax:cartesian -relax:cartesian-score:weights ref2015_cart -relax:min_type lbfgs_armijo_nonmonotone" >> "${RUN}/manifest.md"
"$ROSETTA_BIN/relax.linuxgccrelease"       -s "$IN" -use_input_sc -nstruct 20       -relax:cartesian       -relax:cartesian-score:weights ref2015_cart       -relax:min_type lbfgs_armijo_nonmonotone       -out:path:all "${RUN}/outputs"
