# RFdiffusion Installation Guide

## Quick Decision Tree

**Do you have access to an HPC cluster with GPUs?**
- ✅ **Yes** → Use Singularity/Apptainer (see HPC section)
- ❌ **No** → Continue below

**Are you on macOS?**
- ✅ **Yes** → Docker will work but **no GPU support** (CPU only, very slow)
- ❌ **No (Linux with NVIDIA GPU)** → Docker with GPU support

**Do you want to run locally on macOS?**
- ✅ **Yes** → Install Docker Desktop (CPU mode only)
- ❌ **No** → Skip to HPC setup

---

## Option 1: Docker Desktop (macOS - CPU Only)

### Installation

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop/
   - Download for Mac (Apple Silicon or Intel)
   - Install the `.dmg` file

2. **Launch Docker Desktop:**
   - Open Applications → Docker
   - Wait for Docker to start (whale icon in menu bar)
   - Verify: `docker ps` should work without errors

3. **Build RFdiffusion image:**
   ```bash
   cd /Users/oskarherlitz/Desktop/petase-lab
   docker build -f envs/rfdiffusion/Dockerfile -t petase-rfdiffusion .
   ```

**⚠️ Limitations:**
- No GPU acceleration on macOS
- RFdiffusion will be **very slow** (CPU only)
- Consider using HPC for actual runs

---

## Option 2: Colima (macOS - Alternative)

Lightweight Docker alternative:

```bash
# Install via Homebrew
brew install colima docker docker-compose

# Start Colima
colima start

# Verify
docker ps
```

**Same limitations as Docker Desktop** (no GPU on macOS).

---

## Option 3: Singularity/Apptainer (HPC - Recommended)

### On HPC Cluster

Most clusters have Singularity/Apptainer pre-installed:

```bash
# Check if available
apptainer --version
# or
singularity --version
```

### Build Image on HPC

```bash
# Transfer definition file to HPC
scp envs/rfdiffusion/rfdiffusion.def user@cluster:/path/to/project/

# On HPC, build image
apptainer build rfdiffusion.sif rfdiffusion.def
```

### Or Build from Docker Hub (if available)

If you build Docker image and push to registry:
```bash
apptainer build rfdiffusion.sif docker://your-registry/petase-rfdiffusion:latest
```

---

## Option 4: Conda Environment (Alternative - Not Recommended)

While containers are preferred, you can install RFdiffusion directly:

```bash
# Create environment
conda env create -f external/rfdiffusion/env/SE3nv.yml
conda activate SE3nv

# Install additional dependencies
pip install dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install e3nn==0.3.3 wandb==0.12.0 pynvml==11.0.0
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
pip install decorator==5.1.0 hydra-core==1.3.2 pyrsistent==0.19.3

# Install SE3Transformer and RFdiffusion
pip install external/rfdiffusion/env/SE3Transformer
pip install external/rfdiffusion --no-deps
```

**⚠️ Issues:**
- Complex dependency management
- CUDA version conflicts
- Less reproducible
- Still need GPU for reasonable performance

---

## Verification

After installation, verify setup:

### Docker
```bash
docker images | grep petase-rfdiffusion
docker run --rm petase-rfdiffusion python3.9 --version
```

### Singularity
```bash
apptainer exec rfdiffusion.sif python3.9 --version
```

---

## Next Steps

1. **Download model weights:**
   ```bash
   mkdir -p data/models/rfdiffusion
   bash external/rfdiffusion/scripts/download_models.sh data/models/rfdiffusion
   ```

2. **Test run:**
   ```bash
   # Docker
   envs/rfdiffusion/run_docker.sh --help
   
   # Singularity
   envs/rfdiffusion/run_singularity.sh --help
   ```

---

## Troubleshooting

### "docker: command not found"
- Install Docker Desktop (see Option 1)
- Or add Docker to PATH if installed elsewhere

### "Cannot connect to Docker daemon"
- Start Docker Desktop
- Or start Colima: `colima start`

### "No space left on device"
- Docker images can be large (~10GB)
- Clean up: `docker system prune -a`

### GPU not detected (Linux)
- Install NVIDIA drivers: `nvidia-smi` should work
- Install nvidia-container-toolkit:
  ```bash
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

---

## Recommendations

**For development/testing:**
- Use Docker Desktop on macOS (accept CPU-only limitation)

**For production runs:**
- Use HPC with Singularity/Apptainer and GPU nodes
- Or use a Linux machine with NVIDIA GPU and Docker

**Best practice:**
- Develop/test locally with Docker
- Run production jobs on HPC with Singularity

