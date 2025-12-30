.PHONY: relax ddg
relax:
	bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
ddg:
	bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
