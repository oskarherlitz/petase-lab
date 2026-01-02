# Running ColabFold on RunPod (GPU Cloud)

## Why Use RunPod?

**Current CPU run:**
- ~51 minutes per model
- 5 models per sequence = ~4-5 hours per sequence
- 68 sequences = **~12-17 days total!**

**With GPU (RunPod):**
- ~5-15 minutes per sequence (all 5 models)
- 68 sequences = **~6-17 hours total**
- **Cost: ~$5-15** (much cheaper than waiting 2+ weeks!)

**Verdict: Absolutely worth it!**

---

## Recommended GPU Options

### Option 1: RTX 3090 (Best Value) ⭐ Recommended

**Specs:**
- 24GB VRAM
- Excellent for ColabFold
- Fast and reliable

**Cost:** ~$0.29-0.39/hour
**For 68 sequences:** ~$2-7 total

### Option 2: RTX 4090 (Fastest)

**Specs:**
- 24GB VRAM
- Fastest consumer GPU
- Best performance

**Cost:** ~$0.49-0.69/hour
**For 68 sequences:** ~$3-12 total

### Option 3: A4000/A5000 (Professional)

**Specs:**
- 16GB/24GB VRAM
- Professional grade
- More stable

**Cost:** ~$0.39-0.59/hour
**For 68 sequences:** ~$3-10 total

**Recommendation:** RTX 3090 - best balance of speed and cost

---

## Storage Requirements

### Minimum Storage Needed:

1. **ColabFold installation:** ~500MB
2. **Model weights:** ~4GB
3. **Databases (downloaded on first use):** ~10GB
4. **Input FASTA:** <1MB
5. **Output PDB files:** ~50-100MB per sequence = ~3-7GB for 68 sequences

**Total: ~20GB minimum**

### Recommended Storage:

- **50GB** - Comfortable with room for databases and outputs
- **100GB** - Plenty of space, can keep databases cached

**RunPod default:** Usually 20-50GB, which is sufficient

---

## Setup Steps

### 1. Create RunPod Account

1. Go to: https://www.runpod.io/
2. Sign up (free account)
3. Add payment method

### 2. Create GPU Pod

1. **Go to:** "Pods" → "Deploy"
2. **Select GPU:** RTX 3090 (or RTX 4090)
3. **Template:** Use "RunPod PyTorch" or "Ubuntu 22.04"
4. **Storage:** 50GB (minimum)
5. **Network Volume:** Optional (for persistent storage)
6. **Click "Deploy"**

### 3. Connect to Pod

1. Wait for pod to start (~1-2 minutes)
2. Click "Connect" → "HTTP" or "SSH"
3. You'll get a terminal/notebook interface

### 4. Install ColabFold

```bash
# Update system
apt-get update

# Install Python and pip
apt-get install -y python3 python3-pip

# Install ColabFold
pip install "colabfold[alphafold]"

# Verify installation
colabfold_batch --version
```

### 5. Upload Your FASTA File

**Option A: Via Web Interface**
- Use RunPod's file upload feature
- Upload: `runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta`

**Option B: Via SCP (from your Mac)**
```bash
scp runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  root@<runpod-ip>:/workspace/
```

**Option C: Via Git (if repo is on GitHub)**
```bash
git clone <your-repo-url>
```

### 6. Run ColabFold

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

**Expected time:** 6-17 hours for 68 sequences

### 7. Download Results

**Option A: Via Web Interface**
- Use RunPod's file download feature
- Download entire `colabfold_output/` directory

**Option B: Via SCP (to your Mac)**
```bash
scp -r root@<runpod-ip>:/workspace/colabfold_output \
  runs/colabfold_predictions_gpu/
```

---

## Cost Estimate

### RTX 3090 Example:

- **Setup time:** ~30 minutes (one-time)
- **Processing time:** ~10 hours (for 68 sequences)
- **Cost:** $0.35/hour × 10 hours = **$3.50**
- **Total with setup:** ~**$4-5**

### RTX 4090 Example:

- **Processing time:** ~6 hours
- **Cost:** $0.59/hour × 6 hours = **$3.54**
- **Total:** ~**$4-5**

**Much cheaper than waiting 2+ weeks on CPU!**

---

## Tips for Cost Savings

1. **Use Spot Instances:** 50-70% cheaper (if available)
2. **Stop pod immediately after completion:** Don't leave it running
3. **Download results quickly:** Don't keep pod running just to download
4. **Use network volume:** For persistent storage (optional)

---

## Troubleshooting

### "Out of memory"
- Use RTX 3090/4090 (24GB VRAM)
- Or reduce `--num-models` to 3

### "Database download slow"
- First run downloads databases (~10GB)
- Subsequent runs reuse them
- Consider using network volume for persistence

### "Connection lost"
- RunPod pods can disconnect
- Use `screen` or `tmux` to keep session alive:
  ```bash
  screen -S colabfold
  # Run your command
  # Press Ctrl+A then D to detach
  # Reconnect with: screen -r colabfold
  ```

---

## Alternative: Use Existing MSA Data

If you want to save time, you can:
1. Copy MSA files from your local run
2. Upload to RunPod
3. ColabFold will reuse them (saves ~1-2 hours)

---

## Summary

**Yes, cancel your CPU run and use RunPod!**

- **Time saved:** 12+ days → 6-17 hours
- **Cost:** ~$4-5 (very reasonable)
- **GPU:** RTX 3090 recommended
- **Storage:** 50GB sufficient
- **Setup:** ~30 minutes

**Total time investment:** ~1 day (including setup) vs 2+ weeks on CPU

