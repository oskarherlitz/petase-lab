# RFdiffusion Dockerfile for PETase Lab
# Based on official RFdiffusion Dockerfile with adaptations for project structure
#
# Build: docker build -f envs/rfdiffusion/Dockerfile -t petase-rfdiffusion .
# Usage: See envs/rfdiffusion/run_docker.sh

FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get -q update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    git \
    python3.9 \
    python3-pip \
    wget \
    && python3.9 -m pip install -q -U --no-cache-dir pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Copy RFdiffusion from external submodule
# Note: RFdiffusion will be mounted at runtime, but we install dependencies here
COPY external/rfdiffusion /app/RFdiffusion

# Install Python dependencies
# Note: PyTorch 1.12.1 with CUDA 11.6
# Install PyTorch from CUDA 11.6 wheel index (primary index to get CUDA build)
RUN pip install -q --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu116 \
    torch==1.12.1 \
    torchvision==0.13.1 \
    torchaudio==0.12.1 \
    && pip install -q --no-cache-dir \
    --index-url https://pypi.org/simple \
    dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
    e3nn==0.3.3 \
    wandb==0.12.0 \
    pynvml==11.0.0 \
    git+https://github.com/NVIDIA/dllogger#egg=dllogger \
    decorator==5.1.0 \
    hydra-core==1.3.2 \
    pyrsistent==0.19.3 \
    && pip install --no-cache-dir /app/RFdiffusion/env/SE3Transformer \
    && pip install --no-cache-dir /app/RFdiffusion --no-deps

# Set environment variables
ENV DGLBACKEND="pytorch"
ENV PYTHONPATH="/app/RFdiffusion:/app/RFdiffusion"

# Default working directory (will be overridden by mount)
WORKDIR /app/RFdiffusion

# Default entrypoint (can be overridden)
ENTRYPOINT ["python3.9", "scripts/run_inference.py"]

