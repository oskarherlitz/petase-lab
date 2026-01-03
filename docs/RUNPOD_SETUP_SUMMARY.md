# RunPod RFdiffusion Setup - What Worked

This document summarizes what actually worked to get RFdiffusion running on RunPod GPU pods.

## Quick Start

For a fresh RunPod GPU pod, run:
```bash
bash scripts/setup_rfdiffusion_runpod.sh
```

This single command will:
1. Check GPU availability
2. Install all dependencies
3. Download model weights (if needed)
4. Download input PDB (if needed)
5. Verify everything is working

## What Worked

### 1. Dependency Installation
**Script:** `scripts/install_rfdiffusion_deps.sh`

**Key points:**
- Use **system Python** (not conda Python 3.13, which is incompatible)
- Install **DGL 1.1.3+cu118** with **torchdata 0.7.1** (this specific combination works)
- Install **CUDA 11.8 libraries via conda** and add to `LD_LIBRARY_PATH`
- Install all Python dependencies (omegaconf, hydra, e3nn, etc.)
- Install SE3Transformer and RFdiffusion as **editable packages** (`pip install -e`)

### 2. Model Files
**Script:** `scripts/fix_rfdiffusion_models.sh`

**Issue:** Model files can get corrupted during download (pickle errors)
**Solution:** Re-download using verified URLs:
- Base_ckpt.pt: `http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt`
- ActiveSite_ckpt.pt: `http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt`

### 3. PDB Structure
**Script:** `scripts/check_pdb_ca_atoms.sh`

**Issue:** RFdiffusion only sees residues with CA atoms, and PDB residue numbering may not start at 1
**Solution:** Use correct contig string based on actual PDB:
- For 7SH6: `contigmap.contigs=[A29-289]` (not `[A1-290]`)

### 4. CUDA Library Path
**Issue:** DGL 1.1.3+cu118 needs CUDA 11.8 libraries, but RunPod may have CUDA 12.4
**Solution:** 
- Install CUDA 11.8 via conda: `conda install -y -c conda-forge cudatoolkit=11.8`
- Add to `LD_LIBRARY_PATH`: `${HOME}/miniconda3/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib`
- This is automatically handled by `install_rfdiffusion_deps.sh`

## What Didn't Work

1. **Docker** - Not available/working on RunPod (use direct installation instead)
2. **CUDA 12.4 compatibility** - DGL 1.1.3 requires CUDA 11.8 libraries
3. **torchdata 0.9.0** - Missing `torchdata.datapipes` module (use 0.7.1 instead)
4. **DGL 2.x versions** - Compatibility issues with RFdiffusion
5. **Installing CUDA via apt-get** - Unmet dependencies on RunPod
6. **Conda Python 3.13** - Incompatible with DGL 1.1.3 (use system Python 3.11)

## Essential Scripts

### Setup
- `scripts/setup_rfdiffusion_runpod.sh` - Complete one-command setup
- `scripts/install_rfdiffusion_deps.sh` - Install all dependencies

### Utilities
- `scripts/check_rfdiffusion_models.sh` - Verify model file integrity
- `scripts/fix_rfdiffusion_models.sh` - Re-download corrupted models
- `scripts/check_pdb_ca_atoms.sh` - Check PDB structure for RFdiffusion
- `scripts/check_gpu.sh` - Verify GPU availability

### Running Jobs
- `scripts/rfdiffusion_tmux.sh` - Run jobs in tmux (recommended for long runs)
- `scripts/rfdiffusion_test.sh` - Small test run (5 designs)
- `scripts/rfdiffusion_conservative.sh` - Conservative mask (300 designs)
- `scripts/rfdiffusion_aggressive.sh` - Aggressive mask (300 designs)
- `scripts/rfdiffusion_direct.sh` - Direct run without tmux

## Troubleshooting

### DGL import fails
```bash
# Check if using system Python
which python3  # Should be /usr/bin/python3, not ~/miniconda3/bin/python3

# Reinstall DGL
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

## Storage Requirements

For 300 designs (conservative mask):
- **Essential files:** ~50-200MB (PDB + TRB files)
- **With trajectories:** ~2-3GB (optional, can be disabled)

To disable trajectories and save space, add to run command:
```bash
inference.write_trajectory=False
```

