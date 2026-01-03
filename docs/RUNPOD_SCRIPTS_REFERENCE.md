# RFdiffusion RunPod Scripts Reference

## Quick Start

**For a fresh RunPod GPU pod:**
```bash
bash scripts/setup_rfdiffusion_runpod.sh
```

This single command handles everything: GPU check, dependency installation, model download, PDB download, and verification.

## Script Categories

### Setup Scripts

#### `setup_rfdiffusion_runpod.sh`
**Purpose:** Complete one-command setup for fresh RunPod GPU pods  
**What it does:**
1. Checks GPU availability
2. Installs all dependencies (`install_rfdiffusion_deps.sh`)
3. Downloads model weights (if needed)
4. Downloads input PDB (if needed)
5. Verifies everything is working

**Usage:**
```bash
bash scripts/setup_rfdiffusion_runpod.sh
```

#### `install_rfdiffusion_deps.sh`
**Purpose:** Install all RFdiffusion Python dependencies  
**What it does:**
- Uses system Python (not conda)
- Installs DGL 1.1.3+cu118 with torchdata 0.7.1
- Installs CUDA 11.8 libraries via conda
- Installs all Python packages (omegaconf, hydra, e3nn, etc.)
- Installs SE3Transformer and RFdiffusion as editable packages
- Sets up CUDA library paths

**Usage:**
```bash
bash scripts/install_rfdiffusion_deps.sh
```

### Utility Scripts

#### `check_rfdiffusion_models.sh`
**Purpose:** Verify RFdiffusion model file integrity  
**Usage:**
```bash
bash scripts/check_rfdiffusion_models.sh
```

#### `fix_rfdiffusion_models.sh`
**Purpose:** Re-download corrupted RFdiffusion model files  
**Usage:**
```bash
bash scripts/fix_rfdiffusion_models.sh
```

#### `check_pdb_ca_atoms.sh`
**Purpose:** Check which residues have CA atoms (RFdiffusion requirement)  
**Usage:**
```bash
bash scripts/check_pdb_ca_atoms.sh [pdb_file]
```

#### `check_gpu.sh`
**Purpose:** Verify GPU availability and PyTorch/DGL CUDA status  
**Usage:**
```bash
bash scripts/check_gpu.sh
```

#### `check_rfdiffusion_prereqs.sh`
**Purpose:** Check if all prerequisites are met before running  
**Usage:**
```bash
bash scripts/check_rfdiffusion_prereqs.sh
```

### Run Scripts

#### `rfdiffusion_tmux.sh`
**Purpose:** Run RFdiffusion jobs in tmux (recommended for long runs)  
**Usage:**
```bash
# Test run (5 designs)
bash scripts/rfdiffusion_tmux.sh test

# Conservative mask (300 designs)
bash scripts/rfdiffusion_tmux.sh conservative

# Aggressive mask (300 designs)
bash scripts/rfdiffusion_tmux.sh aggressive
```

**Commands:**
- Attach to session: `tmux attach -t rfdiffusion_test`
- Detach: `Ctrl+B`, then `D`
- View logs: `tail -f runs/rfdiffusion_*.log`
- Kill session: `tmux kill-session -t rfdiffusion_test`

#### `rfdiffusion_test.sh`
**Purpose:** Small test run (5 designs)  
**Usage:**
```bash
bash scripts/rfdiffusion_test.sh
```

#### `rfdiffusion_conservative.sh`
**Purpose:** Conservative mask run (300 designs)  
**Usage:**
```bash
bash scripts/rfdiffusion_conservative.sh
```

#### `rfdiffusion_aggressive.sh`
**Purpose:** Aggressive mask run (300 designs)  
**Usage:**
```bash
bash scripts/rfdiffusion_aggressive.sh
```

#### `rfdiffusion_direct.sh`
**Purpose:** Direct run without tmux (for quick tests)  
**Usage:**
```bash
bash scripts/rfdiffusion_direct.sh [pdb_file] [num_designs]
```

## Troubleshooting

### DGL import fails
```bash
# Reinstall dependencies
bash scripts/install_rfdiffusion_deps.sh
```

### CUDA library errors
```bash
# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Should include conda CUDA libs
export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
```

### Model corruption
```bash
# Re-download models
bash scripts/fix_rfdiffusion_models.sh
```

### PDB parsing errors
```bash
# Check PDB structure
bash scripts/check_pdb_ca_atoms.sh data/structures/7SH6/raw/7SH6.pdb
```

### GPU not detected
```bash
# Check GPU
bash scripts/check_gpu.sh

# If no GPU, you need a GPU-enabled RunPod pod
```

## What Was Removed

All redundant fix scripts have been consolidated. The following scripts were removed:
- All `fix_dgl_*.sh` scripts (replaced by `install_rfdiffusion_deps.sh`)
- All `fix_cudnn_*.sh` scripts (not relevant)
- All `fix_colabfold_*.sh` scripts (not relevant)
- All `fix_cuda_*.sh` scripts (replaced by `install_rfdiffusion_deps.sh`)
- All `fix_gpu_*.sh` scripts (not relevant)
- All `install_dgl_*.sh` scripts (replaced by `install_rfdiffusion_deps.sh`)
- All `install_cuda*.sh` scripts (replaced by `install_rfdiffusion_deps.sh`)

The working solution is now consolidated in `install_rfdiffusion_deps.sh`.

