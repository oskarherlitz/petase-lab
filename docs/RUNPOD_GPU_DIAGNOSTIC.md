# RunPod GPU Detection Diagnostic Guide

## Quick Diagnostic Steps

When AlphaFold/ColabFold isn't detecting your GPU on RunPod, follow these steps in order:

### Step 1: Verify GPU Hardware is Available

First, check if the GPU hardware is actually present and accessible:

```bash
# Check if nvidia-smi works (this confirms GPU hardware is there)
nvidia-smi
```

**Expected output:** You should see GPU information (name, memory, driver version, CUDA version)

**If `nvidia-smi` fails:**
- Your pod may not have a GPU attached
- Check RunPod dashboard to confirm you deployed a GPU pod (RTX 3090, RTX 4090, etc.)
- You may need to restart or redeploy the pod

**If `nvidia-smi` works:** GPU hardware is present, proceed to Step 2.

---

### Step 2: Check JAX Installation

JAX is the library that AlphaFold/ColabFold uses to access the GPU. Check if JAX can see the GPU:

```bash
python3 -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
```

**Expected output (GPU detected):**
```
Devices: [gpu(id=0)]  # or [cuda(id=0)]
Backend: gpu  # or cuda
```

**If you see CPU instead:**
```
Devices: [cpu(id=0)]
Backend: cpu
```

This means JAX is installed but can't see the GPU. Proceed to Step 3.

**If JAX isn't installed:**
```bash
# Install JAX with CUDA support
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

### Step 3: Check CUDA Version Compatibility

JAX needs to match your CUDA version. Check what CUDA version is available:

```bash
# Check CUDA version from nvidia-smi (shown in header)
nvidia-smi | grep "CUDA Version"

# Or check installed CUDA
nvcc --version 2>/dev/null || echo "nvcc not found (CUDA runtime may still be available)"
```

**Common CUDA versions on RunPod:**
- CUDA 12.x (most common) → Use `jax[cuda12_local]`
- CUDA 11.8 → Use `jax[cuda11_local]`

---

### Step 4: Use Automated Diagnostic Script

The easiest way to diagnose and fix is to use the provided script:

```bash
cd /workspace/petase-lab
bash scripts/verify_gpu_runpod.sh
```

This script will:
1. Check GPU hardware availability
2. Check JAX installation
3. Automatically fix CUDA version mismatches
4. Try CUDA 12 first, then CUDA 11.8 if needed

---

### Step 5: Manual Fix (If Script Doesn't Work)

If the automated script doesn't work, try manual fixes:

#### Option A: Reinstall JAX with Correct CUDA Version

```bash
# Uninstall current JAX
pip uninstall -y jax jaxlib

# Install CUDA 12 version (most common)
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify
python3 -c "import jax; print('Devices:', jax.devices())"
```

If CUDA 12 doesn't work, try CUDA 11.8:

```bash
pip uninstall -y jax jaxlib
pip install "jax[cuda11_local]==0.4.23" "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Option B: Set Environment Variables

Sometimes JAX needs environment variables to find CUDA libraries:

```bash
# Find CUDA installation (common locations)
ls -d /usr/local/cuda* 2>/dev/null
ls -d /usr/local/nvidia* 2>/dev/null

# Set environment variables (adjust paths based on what you find)
export CUDA_HOME=/usr/local/cuda-12.4  # or whatever version you have
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Test again
python3 -c "import jax; print('Devices:', jax.devices())"
```

#### Option C: Use Comprehensive Fix Script

```bash
cd /workspace/petase-lab
bash scripts/fix_gpu_detection_runpod.sh
```

This script does a more thorough fix including environment variable setup.

---

### Step 6: Verify AlphaFold/ColabFold Can See GPU

After fixing JAX, verify that ColabFold can actually use the GPU:

```bash
# Test with a simple ColabFold command
colabfold_batch --help

# Or check if it detects GPU during initialization
python3 << 'EOF'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from colabfold.batch import get_model_runner
print("ColabFold initialized")
EOF
```

---

## Common Issues and Solutions

### Issue: "WARNING: no GPU detected, will be using CPU"

**Cause:** JAX is installed but can't see the GPU (usually wrong CUDA version)

**Solution:**
```bash
pip uninstall -y jax jaxlib
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: "nvidia-smi: command not found"

**Cause:** No GPU attached to pod, or wrong pod type

**Solution:**
1. Check RunPod dashboard - make sure you deployed a GPU pod (RTX 3090, RTX 4090, etc.)
2. If you deployed a CPU-only pod, you need to create a new GPU pod
3. Restart the pod if GPU should be there

### Issue: "JAX devices shows CPU even after installing CUDA JAX"

**Possible causes:**
1. Wrong CUDA version (try both CUDA 12 and CUDA 11.8)
2. CUDA libraries not in PATH
3. Environment variables not set

**Solution:**
```bash
# Use the comprehensive fix script
bash scripts/fix_gpu_detection_runpod.sh

# Or manually set environment variables
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 -c "import jax; print(jax.devices())"
```

### Issue: "CUDA out of memory" or "GPU memory errors"

**Cause:** GPU is detected but running out of memory

**Solution:**
- This is actually good - it means GPU is working!
- Reduce batch size or number of models
- Use `--num-models 3` instead of 5
- Use `--num-recycle 1` instead of 3

---

## Complete Diagnostic Checklist

Run through this checklist to systematically diagnose:

- [ ] **Hardware check:** `nvidia-smi` works and shows GPU
- [ ] **JAX installed:** `python3 -c "import jax"` succeeds
- [ ] **JAX sees GPU:** `jax.devices()` shows `[gpu(id=0)]` or `[cuda(id=0)]`
- [ ] **CUDA version matches:** JAX CUDA version matches system CUDA
- [ ] **Environment variables set:** `LD_LIBRARY_PATH` includes CUDA libraries
- [ ] **ColabFold initialized:** ColabFold can import without errors

---

## Quick Reference Commands

```bash
# Full diagnostic (recommended)
cd /workspace/petase-lab
bash scripts/verify_gpu_runpod.sh

# Comprehensive fix
bash scripts/fix_gpu_detection_runpod.sh

# Environment variable fix
bash scripts/fix_gpu_env_runpod.sh

# Manual checks
nvidia-smi                                    # Check GPU hardware
python3 -c "import jax; print(jax.devices())" # Check JAX GPU detection
nvcc --version                                # Check CUDA version
echo $LD_LIBRARY_PATH                         # Check library paths
```

---

## Still Not Working?

If none of the above works:

1. **Check RunPod pod type:** Make absolutely sure you deployed a GPU pod (not CPU-only)
2. **Try a different pod template:** Use "RunPod PyTorch" template which has CUDA pre-configured
3. **Check RunPod logs:** Look for any GPU-related errors in pod logs
4. **Contact RunPod support:** If hardware is definitely there but not accessible

---

## Expected Behavior When Working

When GPU is properly detected:

```bash
$ nvidia-smi
# Shows GPU info (name, memory, CUDA version)

$ python3 -c "import jax; print(jax.devices())"
[gpu(id=0)]  # or [cuda(id=0)]

$ colabfold_batch --num-models 1 test.fasta output/
# Should run on GPU (much faster than CPU)
# No "WARNING: no GPU detected" message
```

---

## Summary

**Most common issue:** JAX installed without CUDA support (CPU-only version)

**Most common fix:**
```bash
pip uninstall -y jax jaxlib
pip install "jax[cuda12_local]==0.4.23" "jaxlib==0.4.23" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Quickest solution:** Run the diagnostic script:
```bash
bash scripts/verify_gpu_runpod.sh
```

