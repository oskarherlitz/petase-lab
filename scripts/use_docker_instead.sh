#!/bin/bash
# Alternative: Use Docker container which has all CUDA libraries pre-installed
# This avoids the CUDA library path issues

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Using Docker container for RFdiffusion..."
echo "This avoids CUDA library path issues"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker not found"
    echo "Install Docker or use the direct installation method"
    exit 1
fi

# Build Docker image if needed
IMAGE_NAME="petase-rfdiffusion"
if ! docker images | grep -q "${IMAGE_NAME}"; then
    echo "Building Docker image (this may take 10-15 minutes)..."
    docker build -f envs/rfdiffusion/Dockerfile -t "${IMAGE_NAME}" .
fi

# Run test
echo ""
echo "Running RFdiffusion test in Docker..."
bash scripts/rfdiffusion_test.sh

