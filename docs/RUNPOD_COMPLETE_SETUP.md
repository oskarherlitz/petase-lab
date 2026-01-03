# Complete RunPod Setup Guide - From Scratch

## Step 1: Deploy RunPod

1. Go to https://www.runpod.io/
2. Sign up / Log in
3. **Deploy Pod:**
   - Click "Pods" → "Deploy"
   - **GPU:** RTX 3090 or RTX 4090 (24GB VRAM)
   - **Template:** "RunPod PyTorch" or "Ubuntu 22.04"
   - **Storage:** 50GB minimum (100GB recommended)
   - **Network Volume (Optional but Recommended):**
     - Create 50-100GB network volume
     - Attach to pod, mount at `/workspace/cache`
   - Click "Deploy"
   - Wait ~1-2 minutes

## Step 2: Connect to Pod

1. Click "Connect" on your pod
2. Choose **"SSH"** (terminal) or **"HTTP"** (Jupyter)
3. Copy the SSH command (looks like: `ssh xxxxx@ssh.runpod.io`)
4. Run it in your terminal

## Step 3: Clone Your Repo

```bash
cd /workspace
git clone https://github.com/oskarherlitz/petase-lab.git
cd petase-lab
```

## Step 4: Install ColabFold (Clean Installation)

**Important:** Install everything together with compatible versions to avoid conflicts.

```bash
# Install system tools
apt-get update -qq
apt-get install -y python3 python3-pip git tmux > /dev/null 2>&1

# Install ColabFold with compatible versions (all at once!)
pip install \
  "numpy<2.0.0,>=1.21.6" \
  "jax[cuda12_local]==0.4.23" \
  "jaxlib==0.4.23" \
  "dm-haiku==0.0.11" \
  "colabfold[alphafold]==1.5.4" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify installation
colabfold_batch --version
python3 -c "import jax; print('Devices:', jax.devices())"
```

**Expected output:**
- `colabfold_batch` should show version
- `jax.devices()` should show `[cuda(id=0)]` (GPU detected)

## Step 5: Verify GPU Works

```bash
# Check GPU
nvidia-smi

# Test JAX GPU
python3 -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
```

Should show:
- GPU in nvidia-smi
- `Backend: gpu` or `cuda`
- `Devices: [cuda(id=0)]`

## Step 6: Run ColabFold in tmux

```bash
# Start tmux session
tmux new -s colabfold

# Inside tmux, run ColabFold
cd /workspace/petase-lab

# Run with reduced settings to avoid cuDNN errors
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu |& tee colabfold.log

# Detach: Press Ctrl+B, then D
# (You can now close your computer!)
```

## Step 7: Monitor Progress

**While detached:**

```bash
# Reattach to see live output
tmux attach -t colabfold

# Or check log file
tail -f colabfold.log

# Check for completed files
ls -lh runs/colabfold_predictions_gpu/*.pdb
```

## Step 8: Download Results

**When complete:**

1. **Via RunPod Web Interface:**
   - Use file browser/download feature
   - Download `runs/colabfold_predictions_gpu/` directory

2. **Via SCP (from your Mac):**
   ```bash
   scp -r root@<runpod-ip>:/workspace/petase-lab/runs/colabfold_predictions_gpu \
     ~/Desktop/petase-lab/runs/
   ```

## Step 9: Stop Pod (Save Money!)

**Important:** Stop the pod immediately after completion!

1. Go to RunPod dashboard
2. Click "Stop" on your pod
3. Or it will auto-stop after inactivity

---

## Troubleshooting

### If you get cuDNN errors:

```bash
# Kill processes
pkill -f colabfold
sleep 5

# Try with environment variables
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
colabfold_batch ...
```

### If GPU not detected:

```bash
# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Reinstall JAX with matching CUDA
pip uninstall -y jax jaxlib
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### If dependency conflicts:

```bash
# Uninstall everything
pip uninstall -y colabfold jax jaxlib dm-haiku numpy

# Reinstall all together (from Step 4)
```

---

## Complete Copy-Paste Commands

**After connecting to RunPod, run this entire block:**

```bash
# 1. Clone repo
cd /workspace
git clone https://github.com/oskarherlitz/petase-lab.git
cd petase-lab

# 2. Install system tools
apt-get update -qq && apt-get install -y python3 python3-pip git tmux > /dev/null 2>&1

# 3. Install ColabFold (all at once!)
pip install \
  "numpy<2.0.0,>=1.21.6" \
  "jax[cuda12_local]==0.4.23" \
  "jaxlib==0.4.23" \
  "dm-haiku==0.0.11" \
  "colabfold[alphafold]==1.5.4" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Verify
colabfold_batch --version
python3 -c "import jax; print('Devices:', jax.devices())"

# 5. Start tmux and run
tmux new -s colabfold
# Inside tmux:
cd /workspace/petase-lab
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
colabfold_batch --num-recycle 2 --num-models 3 --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu |& tee colabfold.log

# Detach: Ctrl+B, then D
```

---

## Expected Timeline

- **Setup:** ~10-15 minutes
- **Model download:** ~10 minutes (first run only)
- **Database download:** ~30-60 minutes (first run only)
- **Processing:** ~6-17 hours for 68 sequences (with GPU)
- **Total:** ~1 day (including setup and first-time downloads)

---

## Summary Checklist

- [ ] Deploy RunPod pod (RTX 3090/4090, 50GB+ storage)
- [ ] Connect via SSH
- [ ] Clone repo
- [ ] Install ColabFold with compatible versions (all at once!)
- [ ] Verify GPU detected
- [ ] Start tmux session
- [ ] Run ColabFold with environment variables
- [ ] Detach from tmux
- [ ] Monitor progress later
- [ ] Download results
- [ ] Stop pod

---

## Key Points

1. **Install everything together** - Avoids dependency conflicts
2. **Use CUDA 12** - Matches your system CUDA 12.8
3. **Use tmux** - Keeps session alive when you disconnect
4. **Use environment variables** - Helps with cuDNN initialization
5. **Reduce models/recycles** - Avoids GPU memory issues
6. **Stop pod when done** - Saves money!

