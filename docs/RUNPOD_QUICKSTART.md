# RunPod Quick Start Guide

## Step-by-Step Setup

### 1. Create RunPod Account & Deploy Pod

1. Go to: https://www.runpod.io/
2. Sign up / Log in
3. **Deploy Pod:**
   - Click "Pods" → "Deploy"
   - **GPU:** RTX 3090 (or RTX 4090)
   - **Template:** "RunPod PyTorch" or "Ubuntu 22.04"
   - **Storage:** 50GB (minimum)
   - Click "Deploy"
   - Wait ~1-2 minutes for pod to start

### 2. Connect to Pod

1. Click "Connect" on your pod
2. Choose **"HTTP"** (Jupyter) or **"SSH"** (terminal)
   - **HTTP (Jupyter):** Easier, web-based
   - **SSH:** More control, terminal-based

### 3. Install ColabFold

**In terminal (SSH or Jupyter terminal):**

```bash
# Update system
apt-get update
apt-get install -y python3 python3-pip git

# Install ColabFold (this will take a few minutes)
pip install "colabfold[alphafold]"

# Verify it works
colabfold_batch --version
```

**Or use the setup script:**
```bash
# Copy and paste the setup script from scripts/setup_runpod_colabfold.sh
# Or clone your repo first (see step 4)
```

### 4. Get Your FASTA File

**Option A: Clone Your Repo (Recommended)**

```bash
# Clone your repository
git clone git@github.com:oskarherlitz/petase-lab.git
cd petase-lab

# Your FASTA file is here:
# runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta
```

**Option B: Upload FASTA File**

1. Use RunPod's file upload feature (in Jupyter or web interface)
2. Upload: `runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta`

**Option C: Use SCP (from your Mac)**

```bash
# From your Mac terminal
scp runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  root@<runpod-ip>:/workspace/
```

### 5. Run ColabFold

```bash
# Create output directory
mkdir -p colabfold_output

# Run prediction (without templates to avoid hhsearch issue)
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  candidates.ranked.fasta \
  colabfold_output
```

**Expected time:** 6-17 hours for 68 sequences (vs 12+ days on CPU!)

### 6. Monitor Progress

**Check progress:**
```bash
# Watch log file
tail -f colabfold_output/log.txt

# Check for completed PDB files
ls -lh colabfold_output/*.pdb
```

**Or use `screen` to keep session alive:**
```bash
# Start screen session
screen -S colabfold

# Run your command
colabfold_batch ... 

# Detach: Press Ctrl+A, then D
# Reattach later: screen -r colabfold
```

### 7. Download Results

**Option A: Via RunPod Web Interface**
- Use file browser/download feature
- Download entire `colabfold_output/` directory

**Option B: Via SCP (to your Mac)**
```bash
# From your Mac
scp -r root@<runpod-ip>:/workspace/colabfold_output \
  runs/colabfold_predictions_gpu/
```

**Option C: Via Git (if you cloned repo)**
```bash
# On RunPod, commit results (if you want)
# Or just download via web interface
```

### 8. Stop Pod (Save Money!)

**Important:** Stop the pod immediately after completion to avoid charges!

1. Go to RunPod dashboard
2. Click "Stop" on your pod
3. Or it will auto-stop after inactivity (check settings)

---

## Complete Example Workflow

```bash
# 1. On RunPod, after connecting via SSH/HTTP

# 2. Install ColabFold
pip install "colabfold[alphafold]"

# 3. Clone your repo
git clone git@github.com:oskarherlitz/petase-lab.git
cd petase-lab

# 4. Run ColabFold
mkdir -p runs/colabfold_predictions_gpu

colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu

# 5. Wait for completion (6-17 hours)

# 6. Download results via RunPod web interface

# 7. Stop pod!
```

---

## Cost Estimate

**RTX 3090:**
- Setup: ~30 minutes (one-time)
- Processing: ~10 hours
- Cost: $0.35/hour × 10 hours = **$3.50**
- **Total: ~$4-5**

**Much cheaper than waiting 2+ weeks on CPU!**

---

## Troubleshooting

### "Permission denied" when cloning
```bash
# Use HTTPS instead of SSH
git clone https://github.com/oskarherlitz/petase-lab.git
```

### "Out of memory"
- Use RTX 3090/4090 (24GB VRAM)
- Or reduce models: `--num-models 3`

### "Connection lost"
```bash
# Use screen to keep session alive
screen -S colabfold
# Run command
# Detach: Ctrl+A, D
# Reattach: screen -r colabfold
```

### "Database download slow"
- First run downloads ~10GB databases (30-60 min)
- Subsequent runs reuse them
- This is normal!

---

## Summary

**Yes, clone your repo!** It's the easiest way to:
- Get your FASTA file
- Keep everything organized
- Access your scripts

**Then:**
1. Install ColabFold dependencies (`pip install`)
2. Run prediction
3. Download results
4. Stop pod

**Total time:** ~1 day (including setup) vs 2+ weeks on CPU!

