# Quick Start Guide: Beginning PETase Optimization

## Prerequisites Checklist

- [ ] Rosetta installed and `ROSETTA_BIN` environment variable set
- [ ] Conda/Mamba installed for environment management
- [ ] Python 3.11+ available
- [ ] PETase structure file ready (5XJH_Repair.pdb)

## Step 1: Environment Setup (5 minutes)

```bash
# Create base environment
conda env create -f envs/base.yml
conda activate petase-lab

# Create Rosetta environment (if using PyRosetta)
conda env create -f envs/pyrosetta.yml

# Set Rosetta path (adjust to your installation)
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
```

## Step 2: Prepare Input Files (10 minutes)

```bash
# Copy your repaired structure
cp data/structures/5XJH/foldx/5XJH_Repair.pdb data/structures/5XJH/raw/PETase_raw.pdb

# Verify the file
head -20 data/structures/5XJH/raw/PETase_raw.pdb
```

## Step 3: First Rosetta Run (30 minutes - 2 hours)

```bash
# Run relaxation
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb

# Check results
ls -lh runs/*relax*/outputs/
cat runs/*relax*/manifest.md
```

## Step 4: Create Initial Mutation List (15 minutes)

Create `configs/rosetta/mutlist.mut`:

```
total 5
1
160 A SER ALA
2
206 A ASP ASN
3
150 A ASP GLY
4
180 A TYR PHE
5
224 A TRP PHE
```

## Step 5: Run ΔΔG Calculations (1-3 hours)

```bash
# Run DDG on best relaxed structure
bash scripts/rosetta_ddg.sh \
  runs/*relax*/outputs/*.pdb \
  configs/rosetta/mutlist.mut

# Parse results
python scripts/parse_ddg.py \
  runs/*ddg*/outputs/*.json \
  results/ddg_scans/initial.csv

# Rank top candidates
python scripts/rank_designs.py \
  results/ddg_scans/initial.csv 10
```

## Step 6: Analyze & Compare (30 minutes)

```bash
# Compare with FoldX results
# Open in PyMOL or use analysis scripts
# Document findings
```

## Next Steps

1. Review results and identify promising mutations
2. Expand mutation list based on findings
3. Set up catalytic constraints (see RESEARCH_PLAN.md)
4. Run FastDesign for active site optimization

## Troubleshooting

**Problem**: `ROSETTA_BIN` not set
```bash
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
```

**Problem**: Structure has issues
- Check for missing residues
- Verify chain assignment
- Use FoldX RepairPDB if needed

**Problem**: Jobs take too long
- Reduce `-nstruct` parameter
- Use cluster resources (see `cluster/slurm_array.template`)
- Run smaller mutation sets

## Getting Help

- Review `docs/RESEARCH_PLAN.md` for detailed methodology
- Check `docs/methodology.md` for workflow overview
- See `docs/glossary.md` for terminology

