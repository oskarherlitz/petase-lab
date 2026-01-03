#!/bin/bash
# Start Docker daemon on RunPod (when systemd isn't available)

# Start Docker daemon in background
dockerd > /tmp/dockerd.log 2>&1 &

# Wait for Docker to be ready
echo "Waiting for Docker daemon to start..."
for i in {1..30}; do
    if docker ps &> /dev/null; then
        echo "✓ Docker is ready!"
        docker ps
        exit 0
    fi
    sleep 1
done

echo "⚠ Docker daemon may not have started. Check /tmp/dockerd.log"
echo "You can try running RFdiffusion directly instead:"
echo "  bash scripts/rfdiffusion_direct.sh"

