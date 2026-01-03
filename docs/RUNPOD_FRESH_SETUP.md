# RunPod Fresh Pod Setup Guide

When you create a new RunPod GPU pod, follow these steps:

## Quick Setup (Recommended)

```bash
# Run the automated setup script
bash scripts/runpod_fresh_setup.sh
```

This will:
1. Pull latest code (if git repo)
2. Check GPU availability
3. Install RFdiffusion dependencies (DGL, torchdata, etc.)
4. Download model weights (~10GB, optional)
5. Download input PDB file
6. Verify everything is set up

## Manual Setup Steps

If you prefer to do it manually:

### 1. Clone/Pull Repository

```bash
# If you need to clone:
git clone <your-repo-url>
cd petase-lab

# Or if already cloned, pull latest:
git pull
```

### 2. Verify GPU

```bash
nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Should show GPU info and `CUDA: True`

### 3. Install RFdiffusion Dependencies

```bash
# Fix DGL and dependencies
bash scripts/fix_dgl_final.sh
```

### 4. Download Model Weights (~10GB)

```bash
# This takes 15-30 minutes
bash scripts/fix_rfdiffusion_models.sh
```

Or manually:
```bash
mkdir -p data/models/rfdiffusion
cd data/models/rfdiffusion
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
cd /workspace/petase-lab
```

### 5. Download Input PDB

```bash
mkdir -p data/structures/7SH6/raw
cd data/structures/7SH6/raw
wget https://files.rcsb.org/view/7SH6.pdb
cd /workspace/petase-lab
```

### 6. Verify Everything

```bash
# Check GPU
bash scripts/check_gpu.sh

# Check prerequisites
bash scripts/check_rfdiffusion_prereqs.sh
```

## What You Need to Download

**Required:**
- ✅ RFdiffusion dependencies (DGL, torchdata) - ~500MB
- ✅ Model weights (Base_ckpt.pt, ActiveSite_ckpt.pt) - ~10GB total
- ✅ Input PDB (7SH6.pdb) - ~100KB

**Not Required (already in repo):**
- ✅ Code/scripts (git pull)
- ✅ RFdiffusion submodule (git submodule update --init)

## Time Estimates

- Dependencies: 5-10 minutes
- Model weights: 15-30 minutes (depends on connection)
- PDB file: <1 minute
- **Total: ~20-40 minutes**

## After Setup

```bash
# Test run (5 designs, ~5-10 minutes)
bash scripts/rfdiffusion_tmux.sh test

# Monitor
tmux attach -t rfdiffusion_test
```

## Troubleshooting

### GPU Not Detected
- Check RunPod pod configuration
- Make sure you selected a GPU instance
- Restart pod if needed

### DGL Import Errors
```bash
bash scripts/fix_dgl_final.sh
```

### Model Weights Corrupted
```bash
bash scripts/fix_rfdiffusion_models.sh
```

