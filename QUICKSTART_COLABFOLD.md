# ColabFold Quick Start

## TL;DR: What Do You Need to Do Outside the Repo?

**Answer:** Almost nothing! You have two options:

### Option 1: Web Interface (Easiest - No Installation!)
1. Run: `bash scripts/colabfold_predict.sh your_sequences.fasta`
2. Follow the instructions to go to https://colabfold.com
3. Upload your FASTA file
4. Done!

### Option 2: Local Installation (One Command)
```bash
pip install colabfold
```

That's it! Then use the script as normal.

---

## Quick Start Guide

### For Progen2 Sequences:

```bash
# 1. Generate sequences (if not done already)
python scripts/run_progen2_pipeline.py my_run --num-samples 50

# 2. Predict structures
bash scripts/colabfold_predict.sh \
  runs/my_run/candidates/candidates.ranked.fasta
```

**If ColabFold isn't installed**, the script will:
- Show you the web interface option (easiest!)
- Or guide you to install locally

### Setup Script (Optional):

```bash
# Interactive setup
bash scripts/setup_colabfold.sh
```

---

## What Gets Installed?

**Web Interface:** Nothing! Just use https://colabfold.com

**Local Installation:**
- ColabFold package (~500MB)
- Databases (~10GB, downloaded automatically on first use)
- Total: ~15GB disk space

---

## Full Documentation

- **Setup details:** See `docs/COLABFOLD_SETUP.md`
- **Usage guide:** See `docs/COLABFOLD_GUIDE.md`

