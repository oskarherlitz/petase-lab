# RunPod Network Volume Guide

## Should You Use a Network Volume?

### Short Answer: **Yes, for databases and results**

Network volumes are **worth it** for ColabFold because:
- **Databases are large** (~10GB) and take 30-60 minutes to download
- **Results are important** (PDB files) - you don't want to lose them
- **Cost is minimal** (~$0.10-0.20/month per 100GB)
- **Saves time** on future runs (no re-downloading databases)

---

## What to Store on Network Volume

### ✅ Store These (Recommended):

1. **ColabFold databases** (~10GB)
   - Location: `~/.cache/colabfold/`
   - Why: Takes 30-60 min to download, reused across runs
   - **Saves:** 30-60 minutes per run

2. **Model weights cache** (~4GB)
   - Location: `~/.cache/colabfold/weights/`
   - Why: Downloaded automatically, cached for reuse
   - **Saves:** ~10 minutes per run

3. **Your results** (PDB files, ~3-7GB)
   - Location: `runs/colabfold_predictions_gpu/`
   - Why: You need these files!
   - **Saves:** Your work!

### ❌ Don't Store These:

- **Your repo** - Can be re-cloned quickly (~1 minute)
- **Temporary files** - Will be regenerated
- **Python packages** - Can be reinstalled

---

## Setup Instructions

### Step 1: Create Network Volume

1. **In RunPod dashboard:**
   - Go to "Network Volumes"
   - Click "Create Volume"
   - **Size:** 50GB (minimum) or 100GB (comfortable)
   - **Name:** `colabfold-cache` (or your choice)
   - Click "Create"

### Step 2: Attach to Pod

**When deploying pod:**
- Under "Network Volumes", select your volume
- **Mount path:** `/workspace/cache` (or your choice)

**Or attach to existing pod:**
- Go to pod settings
- Attach network volume
- Mount at `/workspace/cache`

### Step 3: Configure ColabFold to Use It

```bash
# Set environment variable to use network volume for cache
export COLABFOLD_CACHE_DIR=/workspace/cache/colabfold

# Or create symlink
mkdir -p /workspace/cache/colabfold
ln -s /workspace/cache/colabfold ~/.cache/colabfold

# Now ColabFold will use network volume for databases
```

### Step 4: Save Results to Network Volume

```bash
# Run ColabFold with output on network volume
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  candidates.ranked.fasta \
  /workspace/cache/colabfold_results
```

---

## Cost Analysis

### Network Volume Cost:

- **50GB:** ~$0.05-0.10/month
- **100GB:** ~$0.10-0.20/month
- **Very cheap!**

### Time Savings:

- **First run:** Downloads databases (30-60 min)
- **Future runs:** Databases already there (0 min)
- **Saves:** 30-60 minutes per future run

**Verdict:** Network volume pays for itself if you run ColabFold more than once!

---

## Alternative: Download Results Immediately

If you **don't** want to use a network volume:

1. **Download results immediately** after completion
2. **Re-download databases** each time (30-60 min)
3. **Accept the time cost** for simplicity

**This works if:**
- You only run ColabFold once
- You download results immediately
- You don't mind waiting for database downloads

---

## Recommended Setup

### For One-Time Run:

**No network volume needed:**
- Just download results when done
- Accept database download time

### For Multiple Runs:

**Use network volume:**
- Store databases (~10GB)
- Store results (~3-7GB)
- **Total:** 50GB volume is sufficient

---

## Complete Example with Network Volume

```bash
# 1. Create network volume (50GB) in RunPod dashboard
# 2. Attach to pod, mount at /workspace/cache

# 3. On pod, set up cache directory
mkdir -p /workspace/cache/colabfold
export COLABFOLD_CACHE_DIR=/workspace/cache/colabfold

# 4. Clone repo
git clone https://github.com/oskarherlitz/petase-lab.git
cd petase-lab

# 5. Install ColabFold (databases will download to network volume)
pip install "colabfold[alphafold]"

# 6. Run prediction (results saved to network volume)
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  /workspace/cache/colabfold_results

# 7. Results persist even after pod stops!
# 8. Next time: databases already there, just run prediction
```

---

## Summary

**For your use case (68 sequences, potentially multiple runs):**

✅ **Yes, create a network volume!**

**Benefits:**
- Databases persist (save 30-60 min per future run)
- Results are safe (won't lose PDB files)
- Very cheap (~$0.10/month)
- Easy to set up

**Size:** 50GB is sufficient, 100GB gives more room

**When to skip:**
- One-time run only
- Will download results immediately
- Don't mind re-downloading databases

---

## Quick Decision

**Use network volume if:**
- ✅ You might run ColabFold again
- ✅ You want to save time on future runs
- ✅ You want to keep databases cached

**Skip network volume if:**
- ❌ One-time run only
- ❌ Will download results immediately
- ❌ Prefer simplicity over optimization

