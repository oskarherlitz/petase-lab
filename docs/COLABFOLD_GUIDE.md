# ColabFold Setup and Usage Guide

## What is ColabFold?

**ColabFold** is a faster, easier-to-use version of AlphaFold that:
- Predicts protein structures from amino acid sequences
- Runs in your browser
- Or can be installed locally for batch processing
- Uses MMseqs2 for faster database searches

## When to Use ColabFold

Use ColabFold to:
1. **Validate designs** - Check if your designed sequences fold correctly
2. **Predict novel structures** - For sequences not in PDB
3. **Compare predictions** - See if ColabFold agrees with Rosetta designs
4. **Backbone validation** - Verify radical redesigns are realistic

## Two Ways to Use ColabFold

### Option 1: Web Interface (Easiest) Recommended

**No installation needed!**

1. **Go to**: https://colabfold.com
2. **Upload FASTA file** or paste sequence
3. **Click "Search"**
4. **Wait** (usually 5-30 minutes)
5. **Download** PDB files and results

**Pros:**
- No installation
- Free (uses Google Colab)
- Easy to use
- Automatic updates

**Cons:**
- Requires internet
- Limited to one sequence at a time
- Time limits on free tier

### Option 2: Local Installation (Advanced)

For batch processing multiple sequences:

```bash
# Create environment
conda env create -f envs/colabfold.yml
conda activate petase-colabfold

# Install ColabFold
pip install colabfold

# Run predictions
bash scripts/colabfold_predict.sh data/sequences/design_001.fasta
```

**Pros:**
- Batch processing
- No internet needed (after setup)
- Full control

**Cons:**
- Requires ~10GB disk space
- More complex setup
- Slower initial database download

---

## Quick Start: Using ColabFold

### Step 1: Convert PDB to FASTA

If you have a PDB file (e.g., from Rosetta design):

```bash
python scripts/pdb_to_fasta.py runs/*fastdesign*/outputs/design_001.pdb
```

This creates a FASTA file you can upload to ColabFold.

### Step 2: Run Prediction

**Web Interface:**
1. Go to https://colabfold.com
2. Upload the FASTA file
3. Click "Search"
4. Download results

**Local (if installed):**
```bash
bash scripts/colabfold_predict.sh design_001.fasta
```

### Step 3: Analyze Results

Compare ColabFold prediction with your Rosetta design:
- **RMSD < 2 Å** = Good agreement
- **RMSD 2-5 Å** = Some differences (check why)
- **RMSD > 5 Å** = Major differences (may indicate problem)

---

## Workflow Integration

### Typical Workflow:

```
Rosetta Design
    ↓
Extract Sequence (pdb_to_fasta.py)
    ↓
ColabFold Prediction
    ↓
Compare Structures
    ↓
Select Best Designs
```

### Example:

```bash
# 1. Design with Rosetta (already done)
# Results in: runs/2024-11-10_fastdesign/outputs/design_001.pdb

# 2. Convert to FASTA
python scripts/pdb_to_fasta.py \
  runs/2024-11-10_fastdesign/outputs/design_001.pdb \
  data/sequences/design_001.fasta

# 3. Predict with ColabFold (web or local)
# Upload design_001.fasta to https://colabfold.com

# 4. Compare structures in PyMOL
# Load both: design_001.pdb (Rosetta) and design_001_unrelaxed_rank_1.pdb (ColabFold)
```

---

## Configuration

Edit `configs/colabfold.yaml` to adjust:
- Number of models (1-5)
- Number of recycles (1-3)
- AMBER relaxation (on/off)
- Template usage (on/off)

---

## Troubleshooting

### "colabfold_batch: command not found"
→ ColabFold not installed locally. Use web interface instead.

### "Out of memory" (web interface)
→ Sequence too long. Try shorter sequences or use local installation.

### "Database download failed" (local)
→ Need ~10GB free space. Databases download automatically on first use.

### "Slow predictions"
→ Normal! ColabFold takes 5-30 minutes per sequence. Be patient.

---

## Comparison: ColabFold vs AlphaFold

| Feature | ColabFold | AlphaFold |
|---------|-----------|-----------|
| Speed | Faster (MMseqs2) | Slower |
| Setup | Easy (web) or pip install | Complex (requires databases) |
| Accuracy | Similar to AlphaFold | Gold standard |
| Cost | Free (web) | Free (local) |
| Batch | Limited (web) | Yes (local) |

**Recommendation:** Use ColabFold for most cases. It's easier and nearly as accurate.

---

## Next Steps

1. **Try web interface** with a simple sequence
2. **Convert your Rosetta designs** to FASTA
3. **Predict and compare** structures
4. **Select designs** with good agreement between Rosetta and ColabFold

---

## References

- ColabFold: https://github.com/sokrypton/ColabFold
- Web Interface: https://colabfold.com
- Documentation: https://colabfold.readthedocs.io

