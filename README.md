# PETase Lab
Reproducible, multi-tool pipeline for PETase optimization (Rosetta, FoldX, AlphaFold/ColabFold, RFdiffusion, PyMOL).

## Quickstart
1. Create a new Git repo here and enable Git LFS (see below).
2. Put your PETase PDB in `data/raw/` and ligand SDF in `data/raw/ligands/`.
3. Run Rosetta relax via `scripts/rosetta_relax.sh data/raw/PETase_raw.pdb` (after setting `ROSETTA_BIN`).

## Repo layout
- `envs/`       — small env YAMLs (conda/uv) to avoid dependency clashes.
- `configs/`    — tool configs (Rosetta XML/resfile/cst; FoldX; RFdiffusion; AlphaFold paths).
- `data/`       — immutable, versioned inputs (small files only) + Rosetta params.
- `runs/`       — timestamped, immutable execution folders with `manifest.md`.
- `results/`    — tidy CSV summaries, top-N picks, small figures.
- `scripts/`    — thin wrappers for repeatable commands (bash/python).
- `notebooks/`  — analysis/EDA; write outputs to `results/`.
- `viz/`        — PyMOL sessions (LFS) and exported PNG/MP4.
- `pipelines/`  — optional Makefile/Snakemake automation.
- `cluster/`    — SLURM templates and containers.
- `docs/`       — methodology, glossary, decision records, progress reports.
- `tests/`      — sanity tests for parsers/utilities.

## Initialize Git + LFS
```bash
git init -b main
git lfs install
git add .
git commit -m "chore: scaffold petase-lab"
```

## Push to GitHub (optional, using GitHub CLI)
```bash
gh repo create petase-lab --public --source=. --remote=origin --push
```
# petase-lab
# petase-lab
