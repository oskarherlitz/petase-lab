# PETase Lab
Reproducible, multi-tool pipeline for PETase optimization (Rosetta, FoldX, AlphaFold/ColabFold, RFdiffusion, PyMOL).

## Quickstart

### For New Users
1. **Setup initial data**: `bash scripts/setup_initial_data.sh`
2. **Set Rosetta path**: `export ROSETTA_BIN=/path/to/rosetta/main/source/bin`
3. **Run first relaxation**: `bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb`

### Documentation
- **[Setup Guide](docs/SETUP_GUIDE.md)**  **START HERE** - Environment setup and where to run code
- **[Quick Start Guide](docs/QUICKSTART.md)** - Step-by-step instructions to begin
- **[ColabFold Guide](docs/COLABFOLD_GUIDE.md)** - Structure prediction with ColabFold
- **[Research Plan](docs/RESEARCH_PLAN.md)** - Comprehensive methodology and timeline
- **[Progress Reports](docs/reports/)** - Current status and findings
- **[Methodology](docs/methodology.md)** - Workflow overview
- **[Pipeline plan](docs/NEXT_STEPS_CURSOR.md)** - next steps and associated prompts

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

