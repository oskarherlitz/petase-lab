# PETase Lab

**Reproducible computational pipeline for PETase enzyme optimization using ProGen2 sequence generation, ColabFold structure prediction, and Rosetta energy calculations.**

---

## 🎯 Project Overview

This repository implements a complete computational protein design pipeline to optimize PETase, an enzyme that degrades polyethylene terephthalate (PET) plastic. The workflow combines:

1. **ProGen2** - AI-powered sequence generation for exploring sequence space
2. **ColabFold** - Fast structure prediction using AlphaFold2 models
3. **Rosetta** - Energy-based structure refinement and stability calculations (ΔΔG)
4. **Analysis Tools** - Ranking, catalytic triad geometry analysis, and visualization

### Current Status

✅ **Completed:**
- ProGen2 pipeline for generating diverse protein sequences
- ColabFold structure predictions for 68 candidate sequences (GPU-accelerated on RunPod)
- Candidate ranking by pLDDT confidence scores
- Catalytic triad geometry analysis
- PyMOL visualization scripts

🔄 **In Progress:**
- Rosetta relaxation of top candidates
- Stability calculations (ΔΔG) for top candidates)

📋 **Next Steps:**
- Experimental validation of top designs
- Further sequence optimization based on results

---

## 🚀 Quick Start

### Prerequisites

1. **Conda/Mamba** - For Python environment management
2. **Rosetta** - For structure refinement (requires academic/commercial license)
3. **PyMOL** (optional) - For structure visualization

### Initial Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd petase-lab

# 2. Create base Python environment
conda env create -f envs/base.yml
conda activate petase-lab

# 3. Set Rosetta path (adjust to your installation)
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
# Or add to ~/.zshrc for persistence:
echo 'export ROSETTA_BIN=/path/to/rosetta/main/source/bin' >> ~/.zshrc

# 4. Prepare initial data
bash scripts/setup_initial_data.sh
```

### Run Your First Calculation

```bash
# Relax the wild-type structure
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

---

## 📚 Documentation

### Getting Started
- **[START_HERE.md](START_HERE.md)** - **Begin here!** Step-by-step guide for new users
- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed environment setup instructions
- **[Quick Start Guide](docs/QUICKSTART.md)** - Fast-track to running calculations

### Core Workflows
- **[ProGen2 Pipeline](docs/PROGEN2_WORKFLOW.md)** - Sequence generation workflow
- **[ColabFold Guide](docs/COLABFOLD_GUIDE.md)** - Structure prediction with ColabFold
- **[ColabFold Results Guide](docs/COLABFOLD_RESULTS_GUIDE.md)** - Interpreting prediction results
- **[RunPod Setup](docs/RUNPOD_COMPLETE_SETUP.md)** - Running ColabFold on GPU cloud (RunPod)

### Analysis & Visualization
- **[Candidate Ranking](runs/colabfold_predictions_gpu/CANDIDATE_RANKING.md)** - Top candidates by pLDDT
- **[Catalytic Triad Analysis](runs/colabfold_predictions_gpu/CATALYTIC_TRIAD_ANALYSIS.md)** - Geometry analysis
- **[PyMOL Visualization](scripts/pymol_quick_commands.md)** - Structure visualization commands

### Methodology
- **[Research Plan](docs/RESEARCH_PLAN.md)** - Comprehensive methodology and timeline
- **[Methodology Overview](docs/methodology.md)** - Workflow summary
- **[Next Steps](docs/NEXT_STEPS_CURSOR.md)** - Pipeline roadmap

---

## 🔬 Workflow Overview

### 1. Sequence Generation (ProGen2)

Generate diverse protein sequences using ProGen2 language model:

```bash
# Activate ProGen2 environment
conda activate petase-progen2  # or use venv_progen2

# Run full pipeline
python scripts/run_progen2_pipeline.py \
  --baseline data/sequences/PETase_WT.fasta \
  --output-dir runs/run_$(date +%Y%m%d)_progen2_medium_r1 \
  --num-samples 200 \
  --prompt-lengths 100,130,150
```

**Output:** Filtered candidate sequences in FASTA format (`candidates.ranked.fasta`)

### 2. Structure Prediction (ColabFold)

Predict 3D structures for generated sequences:

**Option A: Local (CPU - slow)**
```bash
conda activate petase-colabfold
bash scripts/colabfold_predict.sh \
  runs/run_*/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions
```

**Option B: RunPod (GPU - recommended)**
```bash
# See docs/RUNPOD_COMPLETE_SETUP.md for full setup
# Then run on RunPod instance:
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  candidates.ranked.fasta \
  colabfold_predictions_gpu
```

**Output:** PDB structures, pLDDT confidence scores, PAE plots

### 3. Analysis & Ranking

```bash
# Rank candidates by pLDDT
python scripts/rank_candidates.py \
  runs/colabfold_predictions_gpu \
  runs/colabfold_predictions_gpu/candidate_ranking.txt

# Analyze catalytic triad geometry
python scripts/analyze_catalytic_triad.py
```

**Output:** Ranked candidate list, geometry analysis CSV

### 4. Structure Refinement (Rosetta)

Relax top candidates to optimize geometry:

```bash
# Relax top 10 candidates
bash scripts/relax_top_candidates.sh 1 10 runs/colabfold_relaxed_top10
```

**Output:** Relaxed PDB structures with improved geometry

### 5. Stability Calculations (Rosetta ΔΔG)

Calculate stability changes for mutations:

```bash
# Run ΔΔG on relaxed structures
bash scripts/rosetta_ddg.sh \
  runs/colabfold_relaxed_top10/*/best.pdb \
  configs/rosetta/mutlist.mut
```

**Output:** ΔΔG values (negative = more stable)

### 6. Visualization (PyMOL)

```bash
# Visualize top candidates with catalytic triad
pymol scripts/visualize_top6.pml
```

---

## 📁 Repository Structure

```
petase-lab/
├── data/                    # Input data (immutable, versioned)
│   ├── structures/         # PDB structures (WT PETase: 5XJH)
│   ├── sequences/          # FASTA sequences
│   └── params/             # Rosetta parameter files
│
├── runs/                    # Execution outputs (timestamped)
│   ├── run_*_progen2_*/    # ProGen2 generation runs
│   ├── colabfold_predictions_gpu/  # ColabFold structures
│   └── colabfold_relaxed_*/ # Rosetta-relaxed structures
│
├── scripts/                 # Executable scripts
│   ├── run_progen2_pipeline.py    # ProGen2 orchestrator
│   ├── colabfold_predict.sh       # ColabFold prediction
│   ├── relax_top_candidates.sh    # Rosetta relaxation
│   ├── analyze_catalytic_triad.py # Geometry analysis
│   └── progen2_pipeline/          # ProGen2 modules
│
├── configs/                 # Tool configurations
│   ├── rosetta/            # Rosetta mutation lists, constraints
│   └── colabfold.yaml      # ColabFold settings
│
├── envs/                    # Conda environment definitions
│   ├── base.yml            # Base Python environment
│   ├── colabfold.yml       # ColabFold dependencies
│   └── design.yml          # Design tools
│
├── docs/                    # Documentation
│   ├── SETUP_GUIDE.md      # Environment setup
│   ├── PROGEN2_WORKFLOW.md # Sequence generation guide
│   ├── COLABFOLD_GUIDE.md  # Structure prediction guide
│   └── RUNPOD_*.md         # Cloud GPU setup guides
│
└── results/                 # Analysis summaries (CSV, figures)
    └── ddg_scans/          # ΔΔG calculation results
```

---

## 🔧 Key Scripts

### Sequence Generation
- `scripts/run_progen2_pipeline.py` - Main ProGen2 orchestrator
- `scripts/apply_catalytic_triad.py` - Mutate catalytic triad positions

### Structure Prediction
- `scripts/colabfold_predict.sh` - Local ColabFold prediction
- `scripts/run_colabfold_runpod.sh` - RunPod GPU execution
- `scripts/setup_runpod_colabfold.sh` - RunPod setup automation

### Analysis
- `scripts/rank_candidates.py` - Rank by pLDDT scores
- `scripts/analyze_catalytic_triad.py` - Catalytic triad geometry
- `scripts/parse_ddg.py` - Parse Rosetta ΔΔG results

### Structure Refinement
- `scripts/rosetta_relax.sh` - Rosetta relaxation
- `scripts/relax_top_candidates.sh` - Batch relax top N candidates
- `scripts/rosetta_ddg.sh` - Stability calculations

### Visualization
- `scripts/visualize_top6.pml` - PyMOL script for top candidates
- `scripts/visualize_catalytic_triad.pml` - Highlight catalytic triad

---

## 📊 Current Results

### Top Candidates (by pLDDT)

| Rank | Candidate | Avg pLDDT | pTM | Status |
|------|-----------|-----------|-----|--------|
| 1 | candidate_6 | 96.22 | 0.940 | ⭐ Excellent |
| 2 | candidate_9 | 96.11 | 0.940 | ⭐ Excellent |
| 3 | candidate_60 | 96.06 | 0.940 | ⭐ Excellent |
| 4 | candidate_21 | 95.97 | 0.940 | ⭐ Excellent |
| 5 | candidate_66 | 95.73 | 0.940 | ⭐ Excellent |

**Full ranking:** See `runs/colabfold_predictions_gpu/CANDIDATE_RANKING.md`

### Catalytic Triad Analysis

All top candidates maintain functional catalytic triad geometry:
- **Ser131 ↔ His208**: ~2.5-3.5 Å (H-bond distance)
- **His208 ↔ Asp177**: ~2.6-3.2 Å (functional triad distance)

**Full analysis:** See `runs/colabfold_predictions_gpu/CATALYTIC_TRIAD_ANALYSIS.md`

---

## 🛠️ Environment Setup

### Base Environment
```bash
conda env create -f envs/base.yml
conda activate petase-lab
```

### ProGen2 Environment
```bash
bash scripts/setup_progen2_env.sh
# Or manually:
python -m venv venv_progen2
source venv_progen2/bin/activate
pip install -r external/progen2/requirements.txt
```

### ColabFold Environment
```bash
conda env create -f envs/colabfold.yml
conda activate petase-colabfold
# Or use setup script:
bash scripts/setup_colabfold.sh
```

### Rosetta
Rosetta requires separate installation and license:
1. Obtain license from [RosettaCommons](https://www.rosettacommons.org/software/license-and-download)
2. Download and install Rosetta
3. Set `ROSETTA_BIN` environment variable

---

## 🐛 Troubleshooting

### Common Issues

**"ROSETTA_BIN: unbound variable"**
```bash
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
```

**ColabFold GPU not detected (RunPod)**
```bash
# See docs/RUNPOD_TROUBLESHOOTING.md
bash scripts/verify_gpu_runpod.sh
```

**ProGen2 generation fails**
- Check that models are downloaded: `ls external/progen2/models/`
- Verify FASTA format: `head data/sequences/PETase_WT.fasta`

**Rosetta relaxation fails**
- Verify structure has no missing residues
- Check Rosetta binary exists: `ls $ROSETTA_BIN/relax.*`

---

## 📖 Additional Resources

- **Rosetta Documentation:** https://www.rosettacommons.org/docs
- **ColabFold:** https://github.com/sokrypton/ColabFold
- **ProGen2:** https://github.com/nvidia/progen2
- **PyMOL:** https://pymol.org/

---

## 📝 Citation

If you use this pipeline in your research, please cite:

- **ColabFold:** Mirdita et al. (2022) ColabFold: making protein folding accessible to all. *Nature Methods*
- **AlphaFold2:** Jumper et al. (2021) Highly accurate protein structure prediction with AlphaFold. *Nature*
- **ProGen2:** Nijkamp et al. (2023) ProGen2: Exploring the Boundaries of Protein Language Models. *arXiv*
- **Rosetta:** Koehler Leman et al. (2020) Macromolecular modeling and design in Rosetta: recent methods and frameworks. *Nature Methods*

---

## 📄 License

[Add your license information here]

---

## 🤝 Contributing

[Add contribution guidelines if applicable]

---

**Last Updated:** January 2025  
**Status:** Active Development
