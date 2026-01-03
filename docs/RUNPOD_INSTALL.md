# What to Install on RunPod After Cloning

## Quick Answer

**For ColabFold predictions only:** Just install ColabFold!

```bash
pip install "colabfold[alphafold]"
```

That's it! ColabFold includes all its dependencies automatically.

---

## Detailed Installation

### Step 1: Basic System Tools (Usually Already Installed)

RunPod templates usually have these, but if not:

```bash
apt-get update
apt-get install -y python3 python3-pip git
```

### Step 2: Install ColabFold

```bash
# This installs everything ColabFold needs
pip install "colabfold[alphafold]"
```

**What this installs automatically:**
- ColabFold package
- JAX and JAXlib (for GPU acceleration)
- AlphaFold dependencies
- All Python dependencies (numpy, pandas, biopython, etc.)

**What downloads later (automatic):**
- Model weights (~4GB) - downloaded on first use
- Databases (~10GB) - downloaded on first prediction

### Step 3: Verify Installation

```bash
colabfold_batch --version
```

---

## Do You Need Anything Else?

### For ColabFold Only: ❌ No

ColabFold is self-contained. You don't need:
- ❌ Conda environments
- ❌ Other Python packages from the repo
- ❌ Progen2 dependencies
- ❌ Rosetta
- ❌ Any other tools

### For Using Repo Scripts: ✅ Maybe

If you want to use other scripts from the repo (e.g., analyzing results):

```bash
# Install base dependencies (optional)
pip install pandas numpy biopython
```

But for **just running ColabFold**, you don't need these!

---

## Complete Minimal Setup

```bash
# 1. Clone repo (already done)
git clone https://github.com/oskarherlitz/petase-lab.git
cd petase-lab

# 2. Install ColabFold (this is all you need!)
pip install "colabfold[alphafold]"

# 3. Verify
colabfold_batch --version

# 4. Run prediction
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

**That's it!** No other installations needed.

---

## What About the Repo's Environment Files?

The repo has environment files (`envs/colabfold.yml`, etc.) for **local development**, but on RunPod you don't need them because:

1. **RunPod already has Python** (from the template)
2. **ColabFold installs its own dependencies** via pip
3. **You don't need conda** on RunPod (unless you want it)

**You can ignore:**
- `envs/colabfold.yml`
- `envs/base.yml`
- All other environment files

Just use `pip install` directly!

---

## Optional: Install Repo Dependencies (If Needed Later)

If you want to use other scripts from the repo (e.g., `scripts/parse_ddg.py`, `scripts/summarize_ddg.py`):

```bash
# Install base Python packages
pip install pandas numpy biopython

# That's usually enough for most scripts
```

But again, **not needed for ColabFold predictions!**

---

## Summary

**For ColabFold predictions:**
```bash
pip install "colabfold[alphafold]"
```

**That's all you need!** Everything else is optional.

---

## Troubleshooting

### "pip: command not found"
```bash
apt-get update
apt-get install -y python3-pip
```

### "Permission denied"
```bash
# Use --user flag
pip install --user "colabfold[alphafold]"
```

### "Out of disk space"
- Check: `df -h`
- Free space: `apt-get clean`
- Use network volume for cache (see RUNPOD_NETWORK_VOLUME.md)

