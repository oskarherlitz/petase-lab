# ColabFold Results Guide

## Understanding Your ColabFold Output

### Current Status

Your ColabFold run is **still in progress**. It's currently:
1. ✅ **MSA Search** - Finding similar sequences (DONE for many candidates)
2. ✅ **Template Search** - Finding structural templates (DONE)
3. ⏳ **Structure Prediction** - Predicting 3D structures (IN PROGRESS)
4. ⏳ **AMBER Relaxation** - Refining structures (WILL HAPPEN NEXT)
5. ⏳ **Final Output** - Generating PDB files and scores (WILL HAPPEN LAST)

---

## Directory Structure

```
runs/colabfold_predictions/
├── config.json                    # Run configuration
├── log.txt                        # Processing log
├── cite.bibtex                    # Citation information
│
├── candidate_1_env/               # Per-sequence directory
│   ├── uniref.a3m                # UniRef MSA alignment
│   ├── bfd.mgnify30.metaeuk30.smag30.a3m  # Combined MSA
│   ├── pdb70.m8                  # Template search results
│   ├── out.tar.gz                # Compressed intermediate files
│   ├── msa.sh                    # MSA search script
│   └── templates_101/            # Template structures (CIF files)
│       ├── *.cif                 # PDB template structures
│       └── *.ffdata, *.ffindex   # Template database files
│
├── candidate_1.pdb               # ⭐ FINAL STRUCTURE (rank 1 - best)
├── candidate_1_relaxed_rank_1.pdb  # ⭐ RELAXED STRUCTURE (most accurate)
├── candidate_1_rank_2.pdb        # Alternative model 2
├── candidate_1_rank_3.pdb        # Alternative model 3
├── candidate_1_rank_4.pdb        # Alternative model 4
├── candidate_1_rank_5.pdb        # Alternative model 5
├── candidate_1_plddt.png         # Confidence plot
├── candidate_1_pae.png           # Error matrix
└── candidate_1_scores.json       # Confidence scores
```

---

## File Types Explained

### Intermediate Files (in `candidate_X_env/`)

#### 1. **MSA Files** (`.a3m`)
- **`uniref.a3m`**: Multiple sequence alignment from UniRef database
- **`bfd.mgnify30.metaeuk30.smag30.a3m`**: Combined MSA from multiple databases
- **What it is**: Alignments of similar sequences used for structure prediction
- **You can ignore**: These are intermediate files, not needed for analysis

#### 2. **Template Files** (`templates_101/`)
- **`*.cif`**: PDB template structures (similar proteins from database)
- **`pdb70.m8`**: Template search results (which templates were found)
- **What it is**: Known structures similar to your sequence
- **You can ignore**: Used during prediction, not needed afterward

#### 3. **`out.tar.gz`**
- Compressed intermediate files
- Contains temporary data from structure prediction
- **You can ignore**: Not needed for analysis

---

### Final Output Files (Main Directory)

These will appear **after structure prediction completes**:

#### 1. **PDB Files** (`.pdb`) ⭐ **MOST IMPORTANT**

**Naming convention:**
- `candidate_X.pdb` or `candidate_X_rank_1.pdb` - **Best model (use this!)**
- `candidate_X_relaxed_rank_1.pdb` - **Best relaxed model (most accurate)**
- `candidate_X_rank_2.pdb` through `rank_5.pdb` - Alternative models

**What to use:**
- **For visualization**: `candidate_X_relaxed_rank_1.pdb` (if available) or `candidate_X_rank_1.pdb`
- **For analysis**: `candidate_X_relaxed_rank_1.pdb` (most accurate)

**What's inside:**
- 3D coordinates of all atoms
- Can be opened in PyMOL, ChimeraX, or any molecular viewer

#### 2. **Confidence Scores** (`*_plddt.png`)

**pLDDT (Predicted LDDT) Score:**
- **>90**: Very high confidence (blue) - Very reliable
- **70-90**: Confident (light blue) - Reliable
- **50-70**: Low confidence (yellow) - Uncertain
- **<50**: Very low confidence (orange/red) - Unreliable, may be disordered

**What to look for:**
- High pLDDT (>70) across most of the structure = Good prediction
- Low pLDDT in specific regions = Those regions are uncertain
- Low pLDDT in catalytic site = Problem! Structure may be wrong there

#### 3. **Error Matrix** (`*_pae.png`)

**PAE (Predicted Aligned Error):**
- Shows confidence in relative positions of different parts
- **Low values (blue)**: High confidence in relative positions
- **High values (red)**: Low confidence, parts may be misaligned

**What to look for:**
- Low PAE overall = Good global structure
- High PAE between domains = Domains may be misoriented
- High PAE in active site = Problem! Active site structure uncertain

#### 4. **Scores JSON** (`*_scores.json`)

Contains numerical confidence scores:
```json
{
  "plddt": 85.2,        # Average confidence
  "ptm": 0.89,          # Predicted TM-score
  "iptm": null,         # Interface TM-score (for multimers)
  "ranking_confidence": 0.92
}
```

**Interpretation:**
- **pLDDT > 70**: Good prediction
- **ptm > 0.7**: Good fold (similar to known structures)
- **ranking_confidence > 0.8**: High confidence in model ranking

---

## How to Interpret Results

### Good Prediction Indicators ✅

1. **High pLDDT** (>70 average, >80 ideal)
2. **Low PAE** (mostly blue in PAE plot)
3. **Consistent models** (rank 1-5 look similar)
4. **Reasonable structure** (no major clashes, proper geometry)
5. **High ptm score** (>0.7)

### Warning Signs ⚠️

1. **Low pLDDT** (<50 in important regions)
2. **High PAE** (red regions in error matrix)
3. **Inconsistent models** (rank 1-5 look very different)
4. **Low confidence in active site** (if catalytic residues have low pLDDT)

### Red Flags 🚩

1. **Very low pLDDT** (<30) across large regions
2. **Disordered structure** (no clear fold)
3. **Major clashes** (atoms overlapping)
4. **Unrealistic geometry** (bonds too long/short)

---

## How to Visualize Structures

### Option 1: PyMOL (Recommended)

**Install:**
- Download: https://pymol.org/
- Or: `conda install -c conda-forge pymol-open-source`

**Open structure:**
```bash
# In PyMOL
File → Open → Select candidate_X_relaxed_rank_1.pdb
```

**Useful commands:**
```python
# Color by confidence (pLDDT)
spectrum b, blue_red, minimum=0, maximum=100

# Show cartoon
show cartoon

# Color by chain
color red, chain A

# Show active site (if you know residues)
select active_site, resi 100-120
show sticks, active_site
```

### Option 2: ChimeraX (Free, User-Friendly)

**Install:**
- Download: https://www.cgl.ucsf.edu/chimerax/

**Open structure:**
```bash
# In ChimeraX
File → Open → Select candidate_X_relaxed_rank_1.pdb
```

**Features:**
- Automatic confidence coloring
- Easy to use interface
- Good for presentations

### Option 3: Online Viewers

**Mol* (Molstar):**
- Go to: https://molstar.org/viewer/
- Upload PDB file
- No installation needed

**3Dmol.js:**
- Go to: https://3dmol.csb.pitt.edu/
- Upload PDB file
- Interactive web viewer

### Option 4: Jupyter Notebook (Python)

```python
import py3Dmol

# Load structure
view = py3Dmol.view(width=800, height=600)
view.addModel(open('candidate_1_relaxed_rank_1.pdb').read(), 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}})
view.zoomTo()
view.show()
```

---

## Comparing with Wild-Type Structure

### Load Both Structures in PyMOL

```python
# Load wild-type
load data/structures/5XJH/5XJH.pdb, wt

# Load ColabFold prediction
load runs/colabfold_predictions/candidate_1_relaxed_rank_1.pdb, pred

# Align structures
align pred, wt

# Calculate RMSD
rms_cur pred, wt

# Color differently
color red, wt
color blue, pred
```

**Interpretation:**
- **RMSD < 2 Å**: Excellent agreement
- **RMSD 2-5 Å**: Good agreement (some differences)
- **RMSD > 5 Å**: Significant differences (may indicate problem)

---

## Next Steps

### 1. **Review Top Candidates**

```bash
# Check which candidates have highest confidence
# Look at pLDDT scores in *_scores.json files
```

**Select criteria:**
- High pLDDT (>75 average)
- Low PAE (good global structure)
- Reasonable fold (looks like an enzyme)
- Preserved active site (if you know catalytic residues)

### 2. **Visual Inspection**

- Open top 5-10 structures in PyMOL/ChimeraX
- Check if they look reasonable
- Compare with wild-type structure
- Look for preserved catalytic site

### 3. **Structural Analysis**

**Check:**
- **Fold preservation**: Does it still look like PETase?
- **Active site**: Are catalytic residues in correct positions?
- **Stability**: Any obvious problems (clashes, bad geometry)?
- **Confidence**: High pLDDT in important regions?

### 4. **Run Stability Calculations**

After selecting promising candidates:

```bash
# Relax with Rosetta (refine structures)
bash scripts/rosetta_relax.sh candidate_X_relaxed_rank_1.pdb

# Calculate ΔΔG (stability)
bash scripts/rosetta_ddg.sh candidate_X_relaxed_rank_1.pdb
```

### 5. **Select Final Candidates**

Combine metrics:
- **ColabFold confidence** (pLDDT, PAE)
- **Structural quality** (visual inspection)
- **Stability** (Rosetta ΔΔG)
- **Sequence properties** (from Progen2 ranking)

---

## Quick Reference: What Files to Use

| Purpose | File to Use |
|---------|-------------|
| **Visualization** | `candidate_X_relaxed_rank_1.pdb` (or `rank_1.pdb`) |
| **Analysis** | `candidate_X_relaxed_rank_1.pdb` |
| **Confidence check** | `candidate_X_plddt.png` |
| **Error assessment** | `candidate_X_pae.png` |
| **Scores** | `candidate_X_scores.json` |
| **Ignore** | Everything in `candidate_X_env/` directories |

---

## Troubleshooting

### "No PDB files found"

**Reason:** Structure prediction still running
**Solution:** Wait for ColabFold to finish (check `log.txt` for progress)

### "Low confidence scores"

**Possible reasons:**
- Novel sequence (no similar structures in database)
- Disordered regions
- Poor MSA coverage

**What to do:**
- Check if MSA has enough sequences
- Look at alternative models (rank 2-5)
- Consider if sequence is realistic

### "Structure looks wrong"

**Check:**
- pLDDT scores (low = unreliable)
- PAE matrix (high errors = problems)
- Compare with wild-type
- Check if sequence is realistic

---

## Summary

1. **Wait for completion** - PDB files appear after structure prediction
2. **Use relaxed rank 1** - Most accurate structure
3. **Check pLDDT** - Should be >70 for good predictions
4. **Visualize in PyMOL** - See how it actually folds
5. **Compare with WT** - Check fold preservation
6. **Select best candidates** - Based on confidence + visual inspection
7. **Run stability calculations** - Rosetta ΔΔG for final selection

---

## Example Workflow

```bash
# 1. Wait for ColabFold to finish
# (Check progress: tail -f runs/colabfold_predictions/log.txt)

# 2. List completed structures
ls runs/colabfold_predictions/*_relaxed_rank_1.pdb

# 3. Open in PyMOL
pymol runs/colabfold_predictions/candidate_1_relaxed_rank_1.pdb

# 4. Check confidence
open runs/colabfold_predictions/candidate_1_plddt.png

# 5. Compare with wild-type
pymol data/structures/5XJH/5XJH.pdb runs/colabfold_predictions/candidate_1_relaxed_rank_1.pdb

# 6. Select best candidates and run stability
bash scripts/rosetta_relax.sh runs/colabfold_predictions/candidate_1_relaxed_rank_1.pdb
```

---

## Questions?

- **Where are my PDB files?** → They'll appear in main directory after prediction completes
- **Which file should I use?** → `*_relaxed_rank_1.pdb` (most accurate)
- **How do I know if it's good?** → Check pLDDT >70, low PAE, reasonable structure
- **Can I visualize it?** → Yes! Use PyMOL, ChimeraX, or online viewers

