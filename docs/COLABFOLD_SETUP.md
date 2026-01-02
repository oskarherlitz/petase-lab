# ColabFold Setup Guide

## Quick Answer: What Do You Need to Do Outside the Repo?

**Short answer:** You have two options:

1. **Nothing!** Use the web interface at https://colabfold.com (no installation)
2. **Install ColabFold** if you want local batch processing (one command)

---

## Option 1: Web Interface (Recommended for Getting Started)

**No installation needed!** This is the easiest way to start.

### Steps:

1. **Run the script** (it will guide you):
   ```bash
   bash scripts/colabfold_predict.sh \
     runs/run_20251230_progen2_medium_r1_test/candidates/candidates.ranked.fasta
   ```

2. **When it says ColabFold isn't installed**, it will show you:
   - A link to https://colabfold.com
   - Your FASTA file location
   - Instructions to upload

3. **Go to https://colabfold.com** and upload your FASTA file

4. **Wait 5-30 minutes** for results

5. **Download** the PDB files

**That's it!** No installation required.

---

## Option 2: Local Installation (For Batch Processing)

If you want to process multiple sequences automatically, install ColabFold locally.

### Quick Setup (Recommended):

```bash
# Run the setup script
bash scripts/setup_colabfold.sh

# Or install directly:
pip install colabfold
```

### Detailed Setup:

#### Method A: Using pip (Simplest)

```bash
# Install ColabFold
pip install colabfold

# Or if using Python 3 specifically:
pip3 install colabfold

# Verify installation
colabfold_batch --version
```

#### Method B: Using conda (If you use conda)

```bash
# Create environment from provided file
conda env create -f envs/colabfold.yml

# Activate environment
conda activate petase-colabfold

# ColabFold should already be installed, but if not:
pip install colabfold
```

### First Run (Database Download)

The first time you run ColabFold locally, it will automatically download databases (~10GB). This happens automatically and only needs to be done once.

```bash
# This will download databases on first run
bash scripts/colabfold_predict.sh your_sequences.fasta
```

**Note:** Database download can take 30-60 minutes depending on your internet speed.

---

## What Gets Installed?

When you install ColabFold locally, it installs:

- **ColabFold Python package** (~500MB)
- **MMseqs2** (for sequence searches)
- **AlphaFold model weights** (~4GB)
- **Databases** (~10GB, downloaded on first use)

**Total disk space needed:** ~15GB

---

## Verification

After installation, verify it works:

```bash
# Check if command is available
colabfold_batch --version

# Or check Python package
python -c "import colabfold; print('ColabFold installed!')"
```

---

## Using ColabFold with Progen2 Sequences

Once set up (or using web interface), here's the workflow:

### Step 1: Generate sequences with Progen2

```bash
python scripts/run_progen2_pipeline.py my_run --num-samples 50
```

### Step 2: Predict structures

**Web interface:**
1. Go to https://colabfold.com
2. Upload `runs/my_run/candidates/candidates.ranked.fasta`
3. Download results

**Local installation:**
```bash
bash scripts/colabfold_predict.sh \
  runs/my_run/candidates/candidates.ranked.fasta \
  runs/my_run/colabfold_predictions
```

### Step 3: Analyze results

Results will be in the output directory:
- `*_ranked_1.pdb` - Best structure (use this!)
- `*_ranked_2.pdb` through `*_ranked_5.pdb` - Alternative models
- `*_plddt.png` - Confidence scores
- `*_pae.png` - Predicted errors

---

## Troubleshooting

### "colabfold_batch: command not found"

**Solution:** ColabFold isn't installed. Either:
- Use web interface: https://colabfold.com
- Install locally: `pip install colabfold`

### "ModuleNotFoundError: No module named 'colabfold'"

**Solution:** Install ColabFold:
```bash
pip install colabfold
```

### "Database download failed"

**Solutions:**
- Check internet connection
- Ensure ~10GB free disk space
- Databases download automatically on first use
- May take 30-60 minutes

### "Out of memory"

**Solutions:**
- Use web interface (handles memory automatically)
- Process fewer sequences at once
- Use shorter sequences
- Increase system RAM

### "Slow predictions"

**This is normal!** ColabFold takes:
- **5-15 minutes** for short sequences (<200 aa)
- **15-30 minutes** for medium sequences (200-400 aa)
- **30-60 minutes** for long sequences (>400 aa)

---

## Comparison: Web vs Local

| Feature | Web Interface | Local Installation |
|---------|---------------|-------------------|
| **Setup** | None | One command (`pip install`) |
| **Speed** | Same | Same |
| **Batch** | One at a time | Multiple sequences |
| **Internet** | Required | Only for first DB download |
| **Disk Space** | None | ~15GB |
| **Best For** | Getting started, occasional use | Batch processing, many sequences |

**Recommendation:**
- **Start with web interface** (no setup)
- **Install locally** if you need to process many sequences

---

## Next Steps

1. **Try web interface first:**
   ```bash
   bash scripts/colabfold_predict.sh your_sequences.fasta
   # Follow the web interface instructions
   ```

2. **If you need batch processing, install locally:**
   ```bash
   bash scripts/setup_colabfold.sh
   ```

3. **Process your Progen2 sequences:**
   ```bash
   bash scripts/colabfold_predict.sh \
     runs/run_*/candidates/candidates.ranked.fasta
   ```

---

## Summary

**What you need to do outside the repo:**

1. **Web interface:** Nothing! Just go to https://colabfold.com
2. **Local installation:** Run `pip install colabfold` (one command)

The scripts in this repo handle everything else!

