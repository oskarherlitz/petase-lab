# 🧬 Research Start Guide: Begin Your PETase Optimization

## Current Status Check

✅ **Environment ready**: `petase-lab` conda environment exists  
⚠️ **ROSETTA_BIN**: Not set yet  
⚠️ **Input file**: Exists but may be placeholder  
❌ **No calculations run yet**

---

## Step-by-Step: Begin Research

### Step 1: Set Up Rosetta Path (2 minutes)

**This is REQUIRED before running any calculations.**

```bash
# Set for this session
export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin

# Make it permanent (so you don't have to set it every time)
echo 'export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin' >> ~/.zshrc
source ~/.zshrc
```

**Verify it works:**
```bash
$ROSETTA_BIN/relax.static.macosclangrelease -version
```

**Expected output:** Rosetta version information

**If you get "can't be opened" error (macOS Gatekeeper):**
```bash
# Remove quarantine attribute
xattr -dr com.apple.quarantine ~/Desktop/rosetta.binary.m1.release-408
```

---

### Step 2: Prepare Your Data (1 minute)

**Check if you need to copy the repaired structure:**

```bash
# Check current input file
head -5 data/structures/5XJH/raw/PETase_raw.pdb

# If it says "REPLACE THIS WITH YOUR REAL PDB", run:
bash scripts/setup_initial_data.sh
```

**What this does:**
- Copies `data/structures/5XJH/foldx/5XJH_Repair.pdb` → `data/structures/5XJH/raw/PETase_raw.pdb`
- Creates necessary directories
- Verifies structure has key residues

---

### Step 3: Activate Environment (30 seconds)

```bash
conda activate petase-lab
```

**Verify:**
```bash
python --version  # Should show Python 3.11
```

---

### Step 4: Run Your First Rosetta Calculation (30 min - 2 hours)

**This is your first real research step!**

```bash
# Make sure you're in the project directory
cd ~/Desktop/petase-lab

# Activate environment
conda activate petase-lab

# Set Rosetta path (if not in .zshrc)
export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin

# Run relaxation
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

**What happens:**
- Rosetta takes your structure
- Optimizes geometry (minimizes energy)
- Generates 20 relaxed structures
- Saves to `runs/YYYY-MM-DD_relax_cart_v1/outputs/`

**How long?** 30 minutes to 2 hours (this is normal!)

**While it runs:**
- Check the output directory: `ls runs/*relax*/outputs/`
- Read the manifest: `cat runs/*relax*/manifest.md`

---

### Step 5: Analyze Relaxed Structures (5 minutes)

**After relaxation completes:**

```bash
# List all relaxed structures
ls -lh runs/*relax*/outputs/*.pdb

# Find the best one (lowest score = most stable)
# Rosetta names them with scores - lower is better
```

**What to look for:**
- Files ending in `.pdb` (these are your relaxed structures)
- The one with the lowest number in the filename is usually best
- Check file sizes (should be ~100-200 KB, not tiny)

---

### Step 6: Calculate Stability Changes (ΔΔG) (1-3 hours)

**This predicts which mutations improve stability.**

```bash
# First, check your mutation list
cat configs/rosetta/mutlist.mut

# Run DDG calculations on the best relaxed structure
# Replace *.pdb with the actual best structure filename
bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
```

**What happens:**
- Makes each mutation from your list
- Calculates energy before/after mutation
- Reports ΔΔG (negative = more stable = good!)

**How long?** 1-3 hours depending on number of mutations

---

### Step 7: Analyze Results (10 minutes)

**Parse and rank your results:**

```bash
# Parse JSON output to CSV
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial.csv

# Rank top 10 candidates
python scripts/rank_designs.py results/ddg_scans/initial.csv 10
```

**What you'll see:**
- List of mutations ranked by ΔΔG
- Negative values = stabilizing mutations (good!)
- Positive values = destabilizing mutations (avoid)

---

## Research Workflow Overview

```
1. Setup (Steps 1-3)          → 5 minutes
2. Relaxation (Step 4)         → 30 min - 2 hours
3. DDG Calculations (Step 6)   → 1-3 hours
4. Analysis (Step 7)          → 10 minutes
5. Next: Design & Validation  → Ongoing
```

---

## What to Do While Calculations Run

### Option 1: Review Your Mutation Strategy
```bash
# Edit mutation list
nano configs/rosetta/mutlist.mut

# Key positions to consider:
# - Active site: Ser160, Asp206, His237
# - Substrate binding: Asp150, Tyr180, Trp224
# - Stability hotspots: surface residues
```

### Option 2: Read Documentation
- `docs/RESEARCH_PLAN.md` - Full methodology
- `docs/SETUP_EXPLAINED.md` - Technical details
- `docs/COLABFOLD_GUIDE.md` - Structure prediction

### Option 3: Plan Next Steps
- Which mutations look promising?
- What positions should you test next?
- How will you validate top candidates?

---

## Troubleshooting

### "ROSETTA_BIN: unbound variable"
→ You didn't set ROSETTA_BIN. Go back to Step 1.

### "command not found: relax"
→ ROSETTA_BIN path is wrong. Check: `ls $ROSETTA_BIN/relax.*`

### "Input file not found"
→ Run `bash scripts/setup_initial_data.sh`

### Calculations are slow
→ **This is normal!** Protein calculations are computationally intensive. Be patient.

### Want to run on cluster?
→ See `cluster/slurm_array.template` for SLURM job submission

---

## Next Steps After Initial Results

1. **Expand mutation list** - Test more positions
2. **Run FastDesign** - Optimize active site with constraints
3. **Cross-validate with FoldX** - Double-check predictions
4. **ColabFold prediction** - Validate top designs
5. **Select candidates** - Pick top 5-10 for experimental testing

---

## Quick Reference Commands

```bash
# Setup
export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin
conda activate petase-lab
bash scripts/setup_initial_data.sh

# Run calculations
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut

# Analyze
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial.csv
python scripts/rank_designs.py results/ddg_scans/initial.csv 10
```

---

## Success Criteria

You're ready to begin research when:
- ✅ ROSETTA_BIN is set and verified
- ✅ Input structure is prepared
- ✅ Environment is activated
- ✅ You understand what each step does

**You're ready! Start with Step 1 above.**

---

## Need Help?

- **Setup issues**: See `docs/SETUP_GUIDE.md`
- **Technical questions**: See `docs/SETUP_EXPLAINED.md`
- **Methodology**: See `docs/RESEARCH_PLAN.md`
- **Quick reference**: See `START_HERE.md`

---

**Good luck with your research! 🧬**

