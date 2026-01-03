# Quick Start Guide: Beginning PETase Optimization

## Prerequisites Checklist

- [ ] Conda/Mamba installed for environment management
- [ ] Python 3.11+ available
- [ ] (Optional) Rosetta installed and `ROSETTA_BIN` environment variable set
- [ ] (Optional) ProGen2 models downloaded (for sequence generation)
- [ ] (Optional) RunPod account (for GPU-accelerated structure prediction)

---

## Choose Your Workflow

### Option A: Work with Existing Results (Fastest)

The project already has **68 ColabFold-predicted structures** ready for analysis. Start here if you want to:

- Analyze existing predictions
- Relax top candidates
- Calculate stability changes

**Skip to:** [Step 3: Analyze Existing Results](#step-3-analyze-existing-results)

### Option B: Run Full Pipeline from Scratch

Generate new sequences and predict structures. Use this if you want to:

- Generate new sequence variants
- Run your own structure predictions
- Explore different sequence spaces

**Start with:** [Step 1: Environment Setup](#step-1-environment-setup)

---

## Step 1: Environment Setup (5 minutes)

```bash
# Create base environment (required)
conda env create -f envs/base.yml
conda activate petase-lab

# Set Rosetta path (if using Rosetta)
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
```

### Optional: Additional Environments

**For ProGen2 (sequence generation):**
```bash
bash scripts/setup_progen2_env.sh
```

**For ColabFold (structure prediction):**
```bash
conda env create -f envs/colabfold.yml
conda activate petase-colabfold
```

**Note:** For GPU-accelerated ColabFold, use RunPod (see `docs/RUNPOD_COMPLETE_SETUP.md`)

---

## Step 2: Prepare Input Files (10 minutes)

```bash
# Run setup script
bash scripts/setup_initial_data.sh

# Verify structure exists
ls -lh data/structures/5XJH/raw/PETase_raw.pdb
```

---

## Step 3: Analyze Existing Results

### View Top Candidates

```bash
# See ranking
cat runs/colabfold_predictions_gpu/CANDIDATE_RANKING.md

# Or view text file
head -20 runs/colabfold_predictions_gpu/candidate_ranking.txt
```

**Top 5 candidates:**
- candidate_6 (pLDDT: 96.22, pTM: 0.940)
- candidate_9 (pLDDT: 96.11, pTM: 0.940)
- candidate_60 (pLDDT: 96.06, pTM: 0.940)
- candidate_21 (pLDDT: 95.97, pTM: 0.940)
- candidate_66 (pLDDT: 95.73, pTM: 0.940)

### Visualize Structures

```bash
# Visualize top 6 candidates with catalytic triad
pymol scripts/visualize_top6.pml

# Or visualize top 3
pymol scripts/visualize_top3.pml
```

### Analyze Catalytic Triad Geometry

```bash
# View existing analysis
cat runs/colabfold_predictions_gpu/CATALYTIC_TRIAD_ANALYSIS.md

# Or re-run analysis
python scripts/analyze_catalytic_triad.py
```

---

## Step 4: Relax Top Candidates (30 minutes - 2 hours)

Optimize geometry of top candidates using Rosetta:

```bash
# Relax top 10 candidates
bash scripts/relax_top_candidates.sh 1 10 runs/colabfold_relaxed_top10

# Or relax top 5 (faster)
bash scripts/relax_top_candidates.sh 1 5 runs/colabfold_relaxed_top5
```

**What happens:**
- Takes ColabFold-predicted structures
- Optimizes geometry using Rosetta energy minimization
- Generates relaxed structures with improved energy

**Time:** ~5-15 minutes per structure

**Check results:**
```bash
ls -lh runs/colabfold_relaxed_top10/*/
```

---

## Step 5: Calculate Stability Changes (ΔΔG) (1-3 hours)

After relaxation, calculate stability changes for mutations:

```bash
# Create/edit mutation list
nano configs/rosetta/mutlist.mut

# Run ΔΔG calculations
bash scripts/rosetta_ddg.sh \
  runs/colabfold_relaxed_top10/*/best.pdb \
  configs/rosetta/mutlist.mut

# Parse results
python scripts/parse_ddg.py \
  runs/*ddg*/outputs/*.json \
  results/ddg_scans/initial.csv
```

**What ΔΔG means:**
- **Negative ΔΔG** = mutation makes protein more stable (good!)
- **Positive ΔΔG** = mutation makes protein less stable

---

## Alternative: Generate New Sequences (ProGen2)

### Step A: Generate Sequences

```bash
# Activate ProGen2 environment
source venv_progen2/bin/activate

# Run pipeline
python scripts/run_progen2_pipeline.py \
  --baseline data/sequences/PETase_WT.fasta \
  --output-dir runs/run_$(date +%Y%m%d)_progen2_medium \
  --num-samples 200 \
  --prompt-lengths 100,130,150
```

**Output:** `runs/run_*/candidates/candidates.ranked.fasta`

**Time:** ~30-60 minutes depending on sample count

### Step B: Predict Structures (ColabFold)

**Option 1: Local (CPU - slow)**
```bash
conda activate petase-colabfold
bash scripts/colabfold_predict.sh \
  runs/run_*/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions
```

**Time:** ~50+ hours for 68 sequences (not recommended)

**Option 2: RunPod (GPU - recommended)**
```bash
# See docs/RUNPOD_COMPLETE_SETUP.md for setup
# Then on RunPod:
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  candidates.ranked.fasta \
  colabfold_predictions_gpu
```

**Time:** ~2-6 hours for 68 sequences on RTX 4090

### Step C: Analyze New Results

```bash
# Rank candidates
python scripts/rank_candidates.py \
  runs/colabfold_predictions_gpu \
  runs/colabfold_predictions_gpu/candidate_ranking.txt

# Analyze catalytic triad
python scripts/analyze_catalytic_triad.py
```

---

## Step 6: Select Top Designs

Based on your analysis:

1. **Review pLDDT scores** → Higher is better (90+ = high confidence)
2. **Check catalytic triad geometry** → Should maintain functional distances
3. **Review ΔΔG values** → Negative = more stable
4. **Visualize structures** → Use PyMOL to inspect designs

**Select top 5-10 candidates** for experimental validation.

---

## Troubleshooting

### "ROSETTA_BIN: unbound variable"
```bash
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
```

### ColabFold GPU not detected (RunPod)
```bash
# See docs/RUNPOD_TROUBLESHOOTING.md
bash scripts/verify_gpu_runpod.sh
```

### ProGen2 generation fails
- Check models: `ls external/progen2/models/`
- Verify FASTA: `head data/sequences/PETase_WT.fasta`

### Structure has issues
- Check for missing residues
- Verify chain assignment
- Use FoldX RepairPDB if needed

### Jobs take too long
- Reduce `-nstruct` parameter
- Use GPU (RunPod) for ColabFold
- Run smaller mutation sets

---

## Next Steps

1. **Review results** → Identify promising mutations
2. **Expand mutation list** → Test more positions
3. **Set up catalytic constraints** → See `docs/RESEARCH_PLAN.md`
4. **Run FastDesign** → Optimize active site
5. **Cross-validate** → Compare with FoldX results
6. **Select for experiments** → Pick top 5-10 for validation

---

## Getting Help

- **[README.md](../README.md)** - Complete project overview
- **[START_HERE.md](../START_HERE.md)** - Step-by-step guide for new users
- **[Setup Guide](SETUP_GUIDE.md)** - Detailed environment setup
- **[ProGen2 Workflow](PROGEN2_WORKFLOW.md)** - Sequence generation guide
- **[ColabFold Guide](COLABFOLD_GUIDE.md)** - Structure prediction guide
- **[RunPod Setup](RUNPOD_COMPLETE_SETUP.md)** - GPU cloud setup
- **[Research Plan](RESEARCH_PLAN.md)** - Detailed methodology

---

**Ready? Start with Step 1!**

