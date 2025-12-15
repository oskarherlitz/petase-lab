# Run manifest
Tool: Rosetta relax (cartesian)
Command:
  relax.static.macosclangrelease -s data/raw/PETase_raw.pdb -use_input_sc -nstruct 20 -relax:cartesian -score:weights ref2015_cart -relax:min_type lbfgs_armijo_nonmonotone
