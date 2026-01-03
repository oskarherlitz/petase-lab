# RunPod GPU Setup - WORKING CONFIGURATION

## ✅ What's Working

After extensive troubleshooting, the GPU is now detected and working! Here's what was fixed:

1. **JAX with CUDA 12**: `jax[cuda12_local]==0.4.23` with `jaxlib==0.4.23+cuda12.cudnn89`
2. **cuDNN installed**: `nvidia-cudnn-cu12` package
3. **cuDNN symlinks**: Created `libcudnn.so.8` -> `libcudnn.so.9` for compatibility
4. **Environment variables**: All CUDA paths properly set

## 🚀 Running ColabFold with GPU

### Quick Start

```bash
cd /workspace/petase-lab

# Set environment variables (required every time!)
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run ColabFold
bash scripts/run_colabfold_gpu_final.sh \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu
```

### Run in Background with tmux

```bash
# Start tmux session
tmux new -s colabfold

# Inside tmux, set environment and run
cd /workspace/petase-lab
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

colabfold_batch \
  --num-recycle 2 \
  --num-models 3 \
  --amber \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions_gpu |& tee colabfold.log

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t colabfold
```

## 📋 Environment Variables (Required)

These must be set every time you run ColabFold:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

## 🔍 Verify GPU is Working

```bash
python3 << 'EOF'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
EOF
```

Should show:
- `Devices: [GpuDevice(id=0)]` or `[cuda(id=0)]`
- `Backend: gpu` or `cuda`

## ⚙️ ColabFold Settings

Recommended settings for GPU (RTX 4090, 24GB):

- `--num-recycle 2`: 2 refinement cycles (balance speed/quality)
- `--num-models 3`: 3 models (good diversity)
- `--amber`: Use AMBER relaxation (improves quality)

For faster runs (lower quality):
- `--num-recycle 1`
- `--num-models 1`
- Remove `--amber`

For higher quality (slower):
- `--num-recycle 3`
- `--num-models 5`
- Keep `--amber`

## 📊 Expected Performance

With RTX 4090 GPU:
- **Per sequence**: ~2-5 minutes (depending on length)
- **68 sequences**: ~2-6 hours total
- **Much faster than CPU**: ~50-100x speedup

## 📁 Output Files

After completion, check:
```bash
# List PDB files (structures)
ls -lh runs/colabfold_predictions_gpu/*.pdb

# List confidence plots
ls -lh runs/colabfold_predictions_gpu/*.png

# View log
tail -f colabfold.log
```

## 🔧 Troubleshooting

If GPU stops working after pod restart:

1. **Recreate cuDNN symlinks**:
   ```bash
   cd /usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib
   ln -sf libcudnn.so.9 libcudnn.so.8
   ln -sf libcudnn.so.9 libcudnn.so
   ```

2. **Verify environment variables are set** (see above)

3. **Check GPU detection**:
   ```bash
   python3 -c "import jax; print(jax.devices())"
   ```

## 💰 Cost Optimization

- **Stop pod immediately** after completion to save money
- Use `tmux` to run in background (can disconnect)
- Monitor progress: `tmux attach -t colabfold`
- Download results before stopping pod

## 📝 Summary

The key fixes that made GPU work:
1. ✅ JAX with CUDA 12: `jax[cuda12_local]==0.4.23`
2. ✅ cuDNN installed: `nvidia-cudnn-cu12`
3. ✅ cuDNN symlinks: `libcudnn.so.8` -> `libcudnn.so.9`
4. ✅ Environment variables: CUDA_HOME, LD_LIBRARY_PATH, XLA settings

Now you can run ColabFold with GPU acceleration! 🚀

