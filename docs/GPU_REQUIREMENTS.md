# GPU Requirements for RFdiffusion

## ⚠️ CRITICAL: GPU is Required

RFdiffusion **requires a GPU** to run in reasonable time. Running on CPU is **not practical**:

- **GPU (RTX 4090)**: ~1-2 minutes per design
- **GPU (RTX 3090)**: ~2-3 minutes per design  
- **CPU**: ~30-60+ minutes per design (or more)

For 300 designs:
- **GPU**: 5-10 hours
- **CPU**: 150-300+ hours (6-12+ days)

## Checking GPU Availability

On RunPod, run:

```bash
# Check GPU
bash scripts/check_gpu.sh

# Or manually:
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## If GPU is Not Available

1. **Check RunPod Pod Configuration**:
   - Make sure you selected a GPU instance (RTX 4090, RTX 3090, etc.)
   - Not a CPU-only instance

2. **Restart Pod**:
   - If you just added GPU, restart the pod

3. **Check PyTorch CUDA**:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`

4. **Check DGL**:
   ```bash
   python3 -c "import dgl; print(dgl.backend.get_backend())"
   ```
   Should print `pytorch`

## RFdiffusion GPU Usage

RFdiffusion automatically uses GPU if available. You don't need to specify it manually - it will detect and use CUDA automatically.

## Monitoring GPU Usage

While RFdiffusion is running:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

You should see:
- GPU utilization > 0%
- Memory being used
- Process name (python) using GPU

