# 🚀 START HERE: Your Next Steps

## Quick Summary
1. Set ROSETTA_BIN (tells scripts where Rosetta is)
2. Run setup script (prepares your data)
3. Run relaxation (optimizes structure)
4. Run DDG calculations (predicts stability)

---

## Step 1: Set ROSETTA_BIN (Required)

### What it does:
Tells your scripts where to find Rosetta binaries.

### Run this:
```bash
# For this session:
export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin

# To make it permanent (recommended):
echo 'export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin' >> ~/.zshrc
source ~/.zshrc
```

### Verify it works:
```bash
$ROSETTA_BIN/relax.static.macosclangrelease -version
```

**Why?** Your scripts need to know where Rosetta programs live. Without this, they'll fail with "command not found".

---

## Step 2: (Optional) Install PyRosetta

### Only needed if:
- You want to write Python scripts using Rosetta
- You plan to use Jupyter notebooks
- You want to automate workflows in Python

### If you want it:
```bash
conda activate petase-lab
pip install pyrosetta-2025.45+release.d79cb06334-cp311-cp311-macosx_12_0_arm64.whl
```

### If you don't need it:
**Skip this step!** Your bash scripts work fine without PyRosetta.

**Why?** PyRosetta is a Python interface. Your scripts use command-line binaries directly, so they don't need it.

---

## Step 3: Prepare Your Data

### Run this:
```bash
bash scripts/setup_initial_data.sh
```

### What it does:
- Copies your FoldX-repaired structure to the right location
- Creates directories for results
- Verifies the structure has key residues

**Why?** Your scripts expect input in `data/structures/5XJH/raw/PETase_raw.pdb`. This script sets that up.

---

## Step 4: Run Your First Rosetta Calculation

### Run this:
```bash
conda activate petase-lab
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

### What happens:
- Takes your structure
- Relaxes it (optimizes geometry, minimizes energy)
- Generates 20 relaxed structures
- Saves to `runs/YYYY-MM-DD_relax_cart_v1/outputs/`

### How long?
- **30 minutes to 2 hours** depending on your computer
- This is normal - protein relaxation is computationally intensive

**Why?** Crystal structures aren't perfect. Relaxation finds the "best" geometry for your sequence.

---

## Step 5: Calculate Stability Changes (ΔΔG)

### First, check your mutation list:
```bash
cat configs/rosetta/mutlist.mut
```

### Then run:
```bash
# Find the best relaxed structure (lowest score)
bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
```

### What happens:
- Makes each mutation from your list
- Calculates energy before/after
- Reports ΔΔG (stability change)
- Saves results as JSON

### How long?
- **1-3 hours** depending on number of mutations

**Why?** This predicts which mutations make the protein more stable. Negative ΔΔG = more stable = good!

---

## Step 6: Analyze Results

### Parse results:
```bash
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial.csv
```

### Rank top candidates:
```bash
python scripts/rank_designs.py results/ddg_scans/initial.csv 10
```

**Why?** Raw Rosetta output is hard to read. This makes it easy to see which mutations are best.

---

## Troubleshooting

### "ROSETTA_BIN: unbound variable"
→ You didn't set ROSETTA_BIN. Go back to Step 1.

### "command not found: relax"
→ ROSETTA_BIN is wrong, or Rosetta isn't installed there. Check the path.

### "Permission denied"
→ Make scripts executable: `chmod +x scripts/*.sh`

### Scripts are slow
→ This is normal! Protein calculations are computationally intensive.

---

## Full Technical Explanation

See `docs/SETUP_EXPLAINED.md` for detailed explanations of:
- Why each step is needed
- How Rosetta works
- What the calculations mean
- The science behind it all

---

## What's Next After This?

1. **Review DDG results** → Find stabilizing mutations
2. **Expand mutation list** → Test more positions
3. **Run FastDesign** → Optimize active site
4. **Cross-validate with FoldX** → Double-check predictions
5. **Select top designs** → Pick 5-10 for experimental testing

---

**Ready? Start with Step 1!**

