# Fixing GPU Memory Errors on RunPod

## Error: "CUDNN_STATUS_INTERNAL_ERROR" or "Not Enough GPU memory"

This error usually means:
- GPU memory fragmentation
- Multiple processes using GPU
- Need to reduce memory usage

---

## Quick Fixes

### Fix 1: Reduce Memory Usage (Recommended)

Run with fewer models and recycles:

```bash
# Instead of 5 models, use 3
# Instead of 3 recycles, use 1-2
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

**Memory savings:**
- 3 models instead of 5: ~40% less memory
- 2 recycles instead of 3: ~33% less memory
- **Total: ~60% less memory usage**

---

### Fix 2: Kill Other GPU Processes

Check what's using the GPU:

```bash
# Check GPU usage
nvidia-smi

# Kill any other ColabFold processes
pkill -f colabfold_batch

# Wait a few seconds, then try again
```

---

### Fix 3: Restart and Use Less Memory

```bash
# Kill all ColabFold processes
pkill -f colabfold

# Wait 10 seconds for GPU to clear
sleep 10

# Run with reduced settings
colabfold_batch \
  --num-recycle 1 \
  --num-models 1 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

**Note:** This is slower (1 model, 1 recycle) but uses minimal memory.

---

### Fix 4: Process Sequences One at a Time

If you have many sequences, process them individually:

```bash
# Split FASTA file (if needed)
# Or process candidates one by one

# For a single sequence:
echo ">candidate_1" > single.fasta
echo "YOUR_SEQUENCE_HERE" >> single.fasta

colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  single.fasta \
  runs/colabfold_predictions_gpu
```

---

## Recommended Settings by GPU

### RTX 3090 (24GB VRAM)
```bash
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  input.fasta \
  output_dir
```
**Should work fine** - 24GB is plenty.

### RTX 4090 (24GB VRAM)
```bash
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  input.fasta \
  output_dir
```
**Should work fine** - same as 3090.

### If Still Getting Errors:
```bash
# Reduce to 3 models
colabfold_batch \
  --num-recycle 3 \
  --num-models 3 \
  --amber \
  input.fasta \
  output_dir
```

---

## Check GPU Status

```bash
# See GPU memory usage
nvidia-smi

# See what processes are using GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## Complete Recovery Steps

If you're stuck:

```bash
# 1. Kill all ColabFold processes
pkill -f colabfold
pkill -f python

# 2. Wait for GPU to clear
sleep 15

# 3. Check GPU is free
nvidia-smi

# 4. Run with conservative settings
cd /workspace/petase-lab
colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu |& tee colabfold.log
```

---

## Alternative: Use CPU (Slower but Reliable)

If GPU keeps failing:

```bash
# Uninstall CUDA JAX, install CPU JAX
pip uninstall -y jax jaxlib
pip install "jax[cpu]==0.4.23" "jaxlib==0.4.23"

# Then run (will be slower but won't have GPU memory issues)
colabfold_batch \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

---

## Summary

**Quick fix:** Reduce models and recycles:
```bash
colabfold_batch --num-recycle 2 --num-models 3 --amber input.fasta output_dir
```

**If that doesn't work:** Kill processes, wait, try again.

**Last resort:** Use CPU mode (slower but reliable).

