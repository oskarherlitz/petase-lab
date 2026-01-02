# RFdiffusion Environment Setup

Container-based environment for RFdiffusion protein structure generation.

## Overview

This directory contains Docker and Singularity/Apptainer configurations for running RFdiffusion in a reproducible, isolated environment. RFdiffusion requires:
- CUDA 11.6.2
- PyTorch 1.12.1 with CUDA support
- GPU acceleration (recommended)
- Complex dependencies (DGL, e3nn, SE3Transformer)

**Why containers?**
- Reproducible CUDA/PyTorch versions
- Isolated dependencies
- Easy deployment on HPC (Singularity/Apptainer)
- Consistent environment across local and cluster

## Installation

### Installing Docker (macOS)

**Option 1: Docker Desktop (Recommended)**
1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Install and launch Docker Desktop
3. Verify installation:
   ```bash
   docker --version
   docker ps
   ```

**Option 2: Colima (Lightweight alternative)**
```bash
brew install colima docker docker-compose
colima start
```

**⚠️ Important Note for macOS:**
- Docker on macOS does **not** support NVIDIA GPUs
- RFdiffusion will run in CPU mode (very slow)
- For GPU acceleration, use HPC with Singularity/Apptainer
- Or use a Linux machine with NVIDIA GPU

### Installing Singularity/Apptainer (HPC)

Most HPC clusters have Singularity/Apptainer pre-installed. Check with:
```bash
apptainer --version  # or
singularity --version
```

If not available, see: https://apptainer.org/docs/user/main/installation.html

## Quick Start

### Docker (Local Development)

**Prerequisites:** Docker installed and running

1. **Build the image:**
   ```bash
   docker build -f envs/rfdiffusion/Dockerfile -t petase-rfdiffusion .
   ```

2. **Download model weights:**
   
   **On macOS (uses curl):**
   ```bash
   mkdir -p data/models/rfdiffusion
   bash envs/rfdiffusion/download_models_macos.sh data/models/rfdiffusion
   ```
   
   **On Linux (uses wget):**
   ```bash
   mkdir -p data/models/rfdiffusion
   bash external/rfdiffusion/scripts/download_models.sh data/models/rfdiffusion
   ```
   
   **Or install wget on macOS:**
   ```bash
   brew install wget
   bash external/rfdiffusion/scripts/download_models.sh data/models/rfdiffusion
   ```

3. **Run RFdiffusion:**
   ```bash
   envs/rfdiffusion/run_docker.sh \
     inference.output_prefix=/data/outputs/design \
     inference.model_directory_path=/data/models \
     inference.input_pdb=/data/inputs/target.pdb \
     inference.num_designs=10 \
     'contigmap.contigs=[10-40/A163-181/10-40]'
   ```

### Singularity/Apptainer (HPC)

1. **Build the image:**
   ```bash
   envs/rfdiffusion/build_singularity.sh envs/rfdiffusion/rfdiffusion.sif
   ```
   
   Or on HPC:
   ```bash
   apptainer build envs/rfdiffusion/rfdiffusion.sif envs/rfdiffusion/rfdiffusion.def
   ```

2. **Download model weights** (on HPC or transfer):
   ```bash
   mkdir -p data/models/rfdiffusion
   bash external/rfdiffusion/scripts/download_models.sh data/models/rfdiffusion
   ```

3. **Run RFdiffusion:**
   ```bash
   envs/rfdiffusion/run_singularity.sh \
     inference.output_prefix=/data/outputs/design \
     inference.model_directory_path=/data/models \
     inference.input_pdb=/data/inputs/target.pdb \
     inference.num_designs=10 \
     'contigmap.contigs=[10-40/A163-181/10-40]'
   ```

## Files

- **`Dockerfile`** - Docker image definition for local use
- **`rfdiffusion.def`** - Singularity/Apptainer definition file for HPC
- **`run_docker.sh`** - Docker runner script with automatic path mounting
- **`run_singularity.sh`** - Singularity/Apptainer runner script for HPC
- **`build_singularity.sh`** - Helper script to build Singularity image

## Configuration

### Environment Variables

**Docker:**
- `RFDIFFUSION_MODELS` - Path to model weights (default: `./data/models/rfdiffusion`)
- `RFDIFFUSION_INPUTS` - Path to input files (default: `./data/raw/structures`)
- `RFDIFFUSION_OUTPUTS` - Path to outputs (default: `./runs/YYYY-MM-DD_rfdiffusion/outputs`)
- `RFDIFFUSION_IMAGE` - Docker image name (default: `petase-rfdiffusion`)

**Singularity:**
- `RFDIFFUSION_MODELS` - Path to model weights
- `RFDIFFUSION_INPUTS` - Path to input files
- `RFDIFFUSION_OUTPUTS` - Path to outputs
- `RFDIFFUSION_SIF` - Path to .sif image file (default: `./rfdiffusion.sif`)
- `SINGULARITY_CMD` - Command to use (`singularity` or `apptainer`, auto-detected)

## Model Weights

RFdiffusion requires pre-trained model weights. Download them using:

```bash
bash external/rfdiffusion/scripts/download_models.sh /path/to/models
```

Required models (~10GB total):
- Base_ckpt.pt
- Complex_base_ckpt.pt
- Complex_Fold_base_ckpt.pt
- InpaintSeq_ckpt.pt
- InpaintSeq_Fold_ckpt.pt
- ActiveSite_ckpt.pt
- Base_epoch8_ckpt.pt

Optional:
- Complex_beta_ckpt.pt
- RF_structure_prediction_weights.pt

## Usage Examples

### Motif Scaffolding
```bash
envs/rfdiffusion/run_docker.sh \
  inference.output_prefix=/data/outputs/motif \
  inference.model_directory_path=/data/models \
  inference.input_pdb=/data/inputs/motif.pdb \
  inference.num_designs=10 \
  'contigmap.contigs=[10-40/A163-181/10-40]'
```

### Binder Design
```bash
envs/rfdiffusion/run_docker.sh \
  inference.output_prefix=/data/outputs/binder \
  inference.model_directory_path=/data/models \
  inference.input_pdb=/data/inputs/target.pdb \
  inference.num_designs=20 \
  'contigmap.contigs=[12-18 A3-117/0]' \
  ppi.hotspot_res=['A51','A52','A50']
```

### Unconditional Generation
```bash
envs/rfdiffusion/run_docker.sh \
  inference.output_prefix=/data/outputs/unconditional \
  inference.model_directory_path=/data/models \
  inference.num_designs=10 \
  'contigmap.contigs=[100-200]'
```

## HPC Integration

### SLURM Example

```bash
#!/bin/bash
#SBATCH -J rfdiffusion
#SBATCH -o slurm-%j.out
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

module load apptainer

export RFDIFFUSION_MODELS=/scratch/user/models/rfdiffusion
export RFDIFFUSION_OUTPUTS=/scratch/user/runs/rfdiffusion/outputs

envs/rfdiffusion/run_singularity.sh \
  inference.output_prefix=/data/outputs/design \
  inference.model_directory_path=/data/models \
  inference.input_pdb=/data/inputs/target.pdb \
  inference.num_designs=10 \
  'contigmap.contigs=[10-40/A163-181/10-40]'
```

## Troubleshooting

### GPU Not Detected
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- For Docker: Install `nvidia-container-toolkit`
- For Singularity: Use `--nv` flag (automatic in scripts)

### Model Weights Not Found
- Check path: `inference.model_directory_path` should point to directory containing `.pt` files
- Verify models downloaded: `ls data/models/rfdiffusion/*.pt`

### Permission Errors
- Docker: May need `sudo` or add user to `docker` group
- Singularity: Check file permissions on mounted directories

### CUDA Out of Memory
- Reduce `inference.num_designs`
- Use smaller models if available
- Check GPU memory: `nvidia-smi`

## References

- [RFdiffusion GitHub](https://github.com/RosettaCommons/RFdiffusion)
- [RFdiffusion Documentation](https://sites.google.com/omsf.io/rfdiffusion/overview)
- [Official Docker Image](https://hub.docker.com/r/rosettacommons/rfdiffusion)

