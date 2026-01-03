# 🚀 START HERE: Your Next Steps

## Quick Summary

This project has **three main workflows**:

1. **Sequence Generation** (ProGen2) - Generate diverse protein sequences
2. **Structure Prediction** (ColabFold) - Predict 3D structures for sequences
3. **Structure Refinement** (Rosetta) - Optimize geometry and calculate stability

**Current Status:** ColabFold predictions completed for 68 candidates. Top candidates identified and ready for Rosetta refinement.

---

## For New Users: Choose Your Path

### Path A: Start with Existing Results (Recommended)
If you want to work with the **already-predicted structures**:

1. **Review top candidates** → See `runs/colabfold_predictions_gpu/CANDIDATE_RANKING.md`
2. **Visualize structures** → Use PyMOL scripts in `scripts/`
3. **Relax top candidates** → Use `scripts/relax_top_candidates.sh`
4. **Calculate stability** → Use `scripts/rosetta_ddg.sh`

### Path B: Run Full Pipeline from Scratch
If you want to **generate new sequences and predict structures**:

1. **Generate sequences** → Use ProGen2 pipeline
2. **Predict structures** → Use ColabFold (local or RunPod)
3. **Analyze results** → Rank and analyze candidates
4. **Refine structures** → Use Rosetta

---

## Step 1: Environment Setup

### Required: Base Python Environment

```bash
# Create base environment
conda env create -f envs/base.yml
conda activate petase-lab
```

### Required: Set Rosetta Path

```bash
# For this session:
export ROSETTA_BIN=/path/to/rosetta/main/source/bin

# To make it permanent (recommended):
echo 'export ROSETTA_BIN=/path/to/rosetta/main/source/bin' >> ~/.zshrc
source ~/.zshrc
```

**Verify it works:**
```bash
$ROSETTA_BIN/relax.static.macosclangrelease -version
```

### Optional: ProGen2 Environment (for sequence generation)

```bash
bash scripts/setup_progen2_env.sh
# Or manually:
python -m venv venv_progen2
source venv_progen2/bin/activate
pip install -r external/progen2/requirements.txt
```

### Optional: ColabFold Environment (for structure prediction)

```bash
conda env create -f envs/colabfold.yml
conda activate petase-colabfold
```

**Note:** For GPU-accelerated predictions, use RunPod (see `docs/RUNPOD_COMPLETE_SETUP.md`)

---

## Step 2: Prepare Initial Data

```bash
bash scripts/setup_initial_data.sh
```

**What it does:**
- Sets up input structure in `data/structures/5XJH/raw/PETase_raw.pdb`
- Creates necessary directories
- Verifies structure integrity

---

## Step 3: Work with Existing ColabFold Results

### View Top Candidates

```bash
# See ranking
cat runs/colabfold_predictions_gpu/CANDIDATE_RANKING.md

# Or view the text file
cat runs/colabfold_predictions_gpu/candidate_ranking.txt
```

**Top 5 candidates:**
- candidate_6 (pLDDT: 96.22)
- candidate_9 (pLDDT: 96.11)
- candidate_60 (pLDDT: 96.06)
- candidate_21 (pLDDT: 95.97)
- candidate_66 (pLDDT: 95.73)

### Visualize Structures

```bash
# Visualize top 6 candidates with catalytic triad highlighted
pymol scripts/visualize_top6.pml

# Or visualize top 3
pymol scripts/visualize_top3.pml
```

### Analyze Catalytic Triad Geometry

```bash
# Already completed - view results:
cat runs/colabfold_predictions_gpu/CATALYTIC_TRIAD_ANALYSIS.md

# Or re-run analysis:
python scripts/analyze_catalytic_triad.py
```

---

## Step 4: Relax Top Candidates (Rosetta)

Relax the top candidates to optimize geometry:

```bash
# Relax top 10 candidates (1 structure each)
bash scripts/relax_top_candidates.sh 1 10 runs/colabfold_relaxed_top10

# Or relax top 5 (faster)
bash scripts/relax_top_candidates.sh 1 5 runs/colabfold_relaxed_top5
```

**What happens:**
- Takes ColabFold-predicted structures
- Optimizes geometry using Rosetta
- Generates relaxed structures with improved energy

**Time:** ~5-15 minutes per structure (50-150 minutes for top 10)

---

## Step 5: Calculate Stability (ΔΔG)

After relaxation, calculate stability changes:

```bash
# First, create a mutation list
# Edit configs/rosetta/mutlist.mut with mutations to test

# Then run ΔΔG calculations
bash scripts/rosetta_ddg.sh \
  runs/colabfold_relaxed_top10/*/best.pdb \
  configs/rosetta/mutlist.mut
```

**What happens:**
- Calculates energy before/after mutations
- Reports ΔΔG (stability change)
- Negative ΔΔG = more stable = good!

**Time:** ~1-3 hours depending on number of mutations

---

## Alternative: Run Full Pipeline from Scratch

### Generate New Sequences (ProGen2)

```bash
# Activate ProGen2 environment
source venv_progen2/bin/activate  # or conda activate petase-progen2

# Run pipeline
python scripts/run_progen2_pipeline.py \
  --baseline data/sequences/PETase_WT.fasta \
  --output-dir runs/run_$(date +%Y%m%d)_progen2_medium \
  --num-samples 200 \
  --prompt-lengths 100,130,150
```

**Output:** `runs/run_*/candidates/candidates.ranked.fasta`

### Predict Structures (ColabFold)

**Option A: Local (CPU - slow, ~50+ hours for 68 sequences)**
```bash
conda activate petase-colabfold
bash scripts/colabfold_predict.sh \
  runs/run_*/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions
```

**Option B: RunPod (GPU - recommended, ~2-6 hours for 68 sequences)**
```bash
# See docs/RUNPOD_COMPLETE_SETUP.md for complete setup
# Then on RunPod instance:
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  candidates.ranked.fasta \
  colabfold_predictions_gpu
```

### Analyze Results

```bash
# Rank candidates
python scripts/rank_candidates.py \
  runs/colabfold_predictions_gpu \
  runs/colabfold_predictions_gpu/candidate_ranking.txt

# Analyze catalytic triad
python scripts/analyze_catalytic_triad.py
```

---

## Troubleshooting

### "ROSETTA_BIN: unbound variable"
→ Set Rosetta path: `export ROSETTA_BIN=/path/to/rosetta/main/source/bin`

### "command not found: relax"
→ Check Rosetta path: `ls $ROSETTA_BIN/relax.*`

### ColabFold GPU not detected (RunPod)
→ See `docs/RUNPOD_TROUBLESHOOTING.md` or run:
```bash
bash scripts/verify_gpu_runpod.sh
```

### ProGen2 generation fails
→ Check models are downloaded: `ls external/progen2/models/`

### Permission denied
→ Make scripts executable: `chmod +x scripts/*.sh`

---

## Next Steps

1. **Review ColabFold results** → Top candidates already identified
2. **Relax top candidates** → Optimize geometry with Rosetta
3. **Calculate ΔΔG** → Predict stability changes
4. **Visualize structures** → Use PyMOL to inspect designs
5. **Select for experiments** → Pick top 5-10 for validation

---

## Documentation

- **[README.md](README.md)** - Complete project overview
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed environment setup
- **[ProGen2 Workflow](docs/PROGEN2_WORKFLOW.md)** - Sequence generation guide
- **[ColabFold Guide](docs/COLABFOLD_GUIDE.md)** - Structure prediction guide
- **[RunPod Setup](docs/RUNPOD_COMPLETE_SETUP.md)** - GPU cloud setup
- **[Research Plan](docs/RESEARCH_PLAN.md)** - Methodology and timeline

---

**Ready? Start with Step 1!**
