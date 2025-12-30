# Setup Steps Explained: The Technical Why

## Overview: What We're Building

You're setting up a **computational protein engineering pipeline** that will:
1. **Relax** protein structures (fix geometry issues)
2. **Calculate ΔΔG** (predict stability changes from mutations)
3. **Design** improved variants (optimize the enzyme)

---

## Step 1: Set ROSETTA_BIN Environment Variable

### What it does:
```bash
export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin
```

### Why it's needed:
- **Rosetta is a collection of command-line tools** (like `relax`, `cartesian_ddg`, etc.)
- These tools live in a specific directory: `main/source/bin/`
- Your scripts need to **find these binaries** to run them
- `ROSETTA_BIN` tells the scripts: "Hey, the Rosetta programs are HERE"

### Technical details:
- **Environment variables** are like global settings your shell remembers
- When a script runs `$ROSETTA_BIN/relax.static.macosclangrelease`, it expands to the full path
- Without this, scripts would fail with "command not found"

### Make it permanent:
```bash
# Add to your shell config so it's always available
echo 'export ROSETTA_BIN=~/Desktop/rosetta.binary.m1.release-408/main/source/bin' >> ~/.zshrc
source ~/.zshrc  # Reload config
```

**Why `.zshrc`?** It runs every time you open a terminal, so `ROSETTA_BIN` is always set.

---

## Step 2: (Optional) Install PyRosetta

### What it does:
```bash
conda activate petase-lab
pip install pyrosetta-2025.45+release.d79cb06334-cp311-cp311-macosx_12_0_arm64.whl
```

### Why it's optional:
- **Your bash scripts DON'T need it** - they call Rosetta binaries directly
- **PyRosetta is ONLY needed if** you want to write Python code that uses Rosetta
- For example: custom analysis scripts, Jupyter notebooks, automated workflows

### Technical details:
- **PyRosetta** = Python bindings to Rosetta (lets you use Rosetta from Python)
- **Standalone Rosetta** = command-line binaries (what your scripts use)
- They're **separate things** - you can use one without the other
- The wheel file is a **pre-compiled package** - no need to build from source

### When you'd use PyRosetta:
```python
# Example: Python script using PyRosetta
import pyrosetta
pyrosetta.init()
pose = pyrosetta.pose_from_pdb("protein.pdb")
# Do stuff in Python...
```

**For now:** Skip this unless you plan to write Python scripts. The bash scripts work fine without it.

---

## Step 3: Prepare Input Data

### What it does:
```bash
bash scripts/setup_initial_data.sh
```

### What happens:
1. **Copies** `data/structures/5XJH/foldx/5XJH_Repair.pdb` → `data/structures/5XJH/raw/PETase_raw.pdb`
2. **Creates** directories for results (`results/`, `runs/`)
3. **Verifies** the structure has key catalytic residues

### Why this matters:
- **FoldX repaired structure** = fixed geometry issues from crystal structure
- **Standardized location** = all scripts know where to find input
- **Verification** = catches problems early (missing residues, wrong chain, etc.)

### Technical details:
- **PDB files** contain 3D coordinates of atoms
- **Crystal structures** sometimes have missing atoms or bad geometry
- **FoldX RepairPDB** fixes these issues
- **Rosetta needs clean input** - bad geometry = bad results

---

## Step 4: Run Rosetta Relaxation

### What it does:
```bash
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

### What happens:
1. **Takes** your PDB structure
2. **Relaxes** it (minimizes energy, fixes clashes, optimizes geometry)
3. **Generates** 20 relaxed structures
4. **Saves** them to `runs/YYYY-MM-DD_relax_cart_v1/outputs/`

### Why relaxation is critical:
- **Crystal structures are snapshots** - not necessarily the lowest energy state
- **Rosetta relaxation** finds the "best" geometry for the sequence
- **Cartesian relaxation** = more accurate (allows bond angle/length changes)
- **Multiple structures (nstruct=20)** = account for flexibility

### Technical details:
- **Energy minimization** = finds geometry with lowest Rosetta score
- **Cartesian vs torsion space**: 
  - Torsion: only rotates bonds (faster, less accurate)
  - Cartesian: moves all atoms (slower, more accurate)
- **ref2015_cart** = scoring function optimized for cartesian space
- **Output**: Best structure = lowest total score

### Why 20 structures?
- Proteins are **flexible** - multiple valid conformations
- **Ensemble** gives you options
- **Best one** = lowest score (most stable predicted structure)

---

## Step 5: Calculate ΔΔG (Stability Changes)

### What it does:
```bash
bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
```

### What happens:
1. **Takes** a relaxed structure
2. **Makes mutations** from your list (e.g., Ser160 → Ala)
3. **Calculates** energy before and after mutation
4. **Reports** ΔΔG = change in folding free energy

### Why ΔΔG matters:
- **ΔΔG < 0** = mutation makes protein MORE stable (good!)
- **ΔΔG > 0** = mutation makes protein LESS stable (bad, might unfold)
- **Goal**: Find mutations that improve stability without breaking function

### Technical details:
- **ΔΔG = ΔG(mutant) - ΔG(wildtype)**
- **Cartesian DDG** = more accurate (allows backbone flexibility)
- **3 iterations** = averages multiple calculations (reduces noise)
- **JSON output** = easy to parse and analyze

### What the mutation file looks like:
```
total 3
1
160 A SER ALA    # Change Serine 160 to Alanine
2
206 A ASP ASN    # Change Aspartate 206 to Asparagine
3
150 A ASP GLY    # Change Aspartate 150 to Glycine
```

---

## Step 6: Analyze Results

### What it does:
```bash
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial.csv
python scripts/rank_designs.py results/ddg_scans/initial.csv 10
```

### What happens:
1. **Parses** JSON output from Rosetta
2. **Extracts** ΔΔG values for each mutation
3. **Ranks** by stability (most negative = most stabilizing)
4. **Shows** top 10 candidates

### Why analysis matters:
- **Raw Rosetta output** = hard to read
- **CSV format** = easy to analyze in Excel/Python
- **Ranking** = identifies best candidates quickly
- **Top candidates** = what you'd test experimentally

---

## The Big Picture: Why This Pipeline?

### Traditional approach:
1. Make random mutations
2. Test experimentally (slow, expensive)
3. Hope something works

### Computational approach (what you're doing):
1. **Predict** which mutations help (fast, cheap)
2. **Filter** to top candidates
3. **Test** only the promising ones (focused, efficient)

### The workflow:
```
Crystal Structure
    ↓
FoldX Repair (fix geometry)
    ↓
Rosetta Relax (optimize structure)
    ↓
Rosetta DDG (predict stability)
    ↓
Analysis (rank candidates)
    ↓
Top Designs → Experimental Testing
```

---

## Key Concepts Explained

### 1. **Energy Minimization**
- Proteins fold to **lowest energy state**
- Rosetta calculates **energy** from:
  - Bond lengths/angles
  - Van der Waals interactions
  - Electrostatics
  - Hydrogen bonds
- **Minimization** = finds geometry that minimizes total energy

### 2. **Scoring Functions**
- **ref2015_cart** = Rosetta's energy function
- Trained on **known protein structures**
- Predicts: "Is this structure realistic?"
- **Lower score = better structure**

### 3. **Cartesian vs Torsion Space**
- **Torsion space**: Only rotates bonds (dihedral angles)
- **Cartesian space**: Moves all atoms independently
- **Cartesian = more accurate** but slower
- **Why use it?** Better for small changes (mutations, ligand binding)

### 4. **Ensemble Generation**
- Generate **multiple structures** (nstruct=20)
- Proteins are **dynamic** - multiple valid conformations
- **Best structure** = lowest score
- **Ensemble** = accounts for flexibility

### 5. **ΔΔG Calculation**
- **ΔG** = folding free energy (how stable is the protein?)
- **ΔΔG** = change in stability from mutation
- **Negative ΔΔG** = more stable (good!)
- **Positive ΔΔG** = less stable (might unfold)

---

## Common Questions

### Q: Why not just use the crystal structure?
**A:** Crystal structures have artifacts (missing atoms, crystal contacts, etc.). Relaxation finds the "real" solution structure.

### Q: Why generate 20 structures?
**A:** Proteins are flexible. Multiple structures capture this - you pick the best one.

### Q: Why cartesian relaxation?
**A:** More accurate for small changes. Torsion-only can miss important geometry changes.

### Q: How accurate is Rosetta ΔΔG?
**A:** ~1-2 kcal/mol error typically. Good enough to rank mutations, but not perfect. That's why you validate experimentally.

### Q: What's the difference between Rosetta and FoldX?
**A:** Both predict stability, but use different methods. Cross-validation (using both) gives more confidence.

---

## Next Steps After Setup

1. **Run relaxation** → Get optimized structure
2. **Run DDG scan** → Find stabilizing mutations
3. **Analyze results** → Identify top candidates
4. **Design active site** → Optimize catalysis (FastDesign)
5. **Cross-validate** → Check with FoldX
6. **Select designs** → Top 5-10 for experimental testing

---

*This pipeline automates the computational part so you can focus on the biology!*

