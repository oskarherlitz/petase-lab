# RunPod Troubleshooting Guide

## Common Issues and Fixes

### Issue 1: "OSError: runs/... could not be found"

**Problem:** You're in the wrong directory.

**Fix:**
```bash
# Make sure you're in the repo root
cd /workspace/petase-lab

# Then run ColabFold
colabfold_batch --num-recycle 3 --num-models 5 --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

---

### Issue 2: "AttributeError: jax.interpreters.xla.xe was removed"

**Problem:** JAX version incompatibility (same as on Mac).

**Fix:**
```bash
cd /workspace/petase-lab
bash scripts/fix_colabfold_jax_runpod.sh
```

Or manually:
```bash
pip uninstall -y jax jaxlib dm-haiku
pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23" "dm-haiku==0.0.10"
```

---

### Issue 3: "WARNING: no GPU detected, will be using CPU"

**Problem:** CUDA-enabled JAX not installed.

**Fix:**
```bash
# Uninstall CPU-only JAX
pip uninstall -y jax jaxlib

# Install CUDA-enabled JAX
pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23"

# Verify GPU is detected
python -c "import jax; print(jax.devices())"
```

**Note:** If you still see CPU, check:
```bash
# Check if GPU is available
nvidia-smi

# If nvidia-smi works, GPU is there, just need CUDA JAX
```

---

### Issue 4: "duplicate session: colabfold"

**Problem:** tmux session already exists.

**Fix:**
```bash
# Attach to existing session
tmux attach -t colabfold

# Or kill old session
tmux kill-session -t colabfold
tmux new -s colabfold
```

---

## Complete Setup Checklist

1. **Clone repo:**
   ```bash
   cd /workspace
   git clone https://github.com/oskarherlitz/petase-lab.git
   cd petase-lab
   ```

2. **Install ColabFold:**
   ```bash
   pip install "colabfold[alphafold]"
   ```

3. **Fix JAX compatibility:**
   ```bash
   bash scripts/fix_colabfold_jax_runpod.sh
   ```

4. **Verify GPU (if using GPU pod):**
   ```bash
   nvidia-smi
   python -c "import jax; print(jax.devices())"
   ```

5. **Run ColabFold:**
   ```bash
   # Make sure you're in the right directory!
   cd /workspace/petase-lab
   
   # Start tmux
   tmux new -s colabfold
   
   # Run ColabFold
   colabfold_batch --num-recycle 3 --num-models 5 --amber \
     runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
     runs/colabfold_predictions_gpu |& tee colabfold.log
   
   # Detach: Ctrl+B, then D
   ```

---

## Quick Fix Commands

**If you get the JAX error:**
```bash
cd /workspace/petase-lab
bash scripts/fix_colabfold_jax_runpod.sh
```

**If you're in wrong directory:**
```bash
cd /workspace/petase-lab
pwd  # Should show /workspace/petase-lab
```

**If GPU not detected:**
```bash
pip install "jax[cuda12]==0.4.23" "jaxlib==0.4.23"
```

