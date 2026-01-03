# Fixing cuDNN Errors on RunPod

## Error: "CUDNN_STATUS_INTERNAL_ERROR" or "Could not create cudnn handle"

This means **cuDNN (CUDA Deep Neural Network library) can't initialize**, even though GPU memory is available.

---

## Quick Fix: Reinstall JAX with Correct CUDA Version

```bash
cd /workspace/petase-lab

# Run the fix script
bash scripts/fix_cudnn_runpod.sh
```

Or manually:

```bash
# 1. Check CUDA version
nvidia-smi

# 2. Uninstall current JAX
pip uninstall -y jax jaxlib

# 3. Install JAX with CUDA 11.8 (most compatible)
pip install "jax[cuda11_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 4. Verify GPU is detected
python3 -c "import jax; print(jax.devices())"
```

---

## Alternative: Try CUDA 12

If CUDA 11.8 doesn't work:

```bash
pip uninstall -y jax jaxlib
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Check What's Wrong

```bash
# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Check if GPU is detected
python3 -c "import jax; print(jax.devices())"

# Should show something like:
# [GpuDevice(id=0, process_index=0)]
# If it shows [CpuDevice()], GPU isn't detected
```

---

## Complete Fix Steps

```bash
# 1. Go to repo
cd /workspace/petase-lab

# 2. Kill any running ColabFold
pkill -f colabfold
sleep 5

# 3. Fix cuDNN
bash scripts/fix_cudnn_runpod.sh

# 4. Verify GPU works
python3 -c "import jax; print('Devices:', jax.devices())"

# 5. Run ColabFold again
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu |& tee colabfold.log
```

---

## Why This Happens

- **CUDA version mismatch**: JAX was installed with wrong CUDA version
- **cuDNN not found**: cuDNN library not properly linked
- **Driver mismatch**: GPU driver version incompatible with CUDA

The fix is to reinstall JAX with the correct CUDA version for your RunPod instance.

---

## If Still Not Working

Try installing from conda-forge (sometimes more reliable):

```bash
# Install via conda (if available)
conda install -c conda-forge jax cuda-nvcc cudatoolkit=11.8

# Or use pip with explicit CUDA version
pip install "jax[cuda11_local]==0.4.23" "jaxlib==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Summary

**The issue:** JAX installed with wrong CUDA version → cuDNN can't initialize

**The fix:** Reinstall JAX with correct CUDA version (11.8 or 12)

**Quick command:**
```bash
cd /workspace/petase-lab && bash scripts/fix_cudnn_runpod.sh
```

