#!/usr/bin/env bash
# Download ColabFold results from RunPod

set -euo pipefail

# Replace with your RunPod SSH details
RUNPOD_USER="fjcsnjcot8dx1x-64410e48"
RUNPOD_HOST="ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_PATH="/workspace/petase-lab/runs/colabfold_predictions_gpu"
LOCAL_PATH="$HOME/Desktop/petase-lab/runs/colabfold_predictions_gpu"

echo "Downloading ColabFold results from RunPod..."
echo "Remote: $RUNPOD_USER@$RUNPOD_HOST:$REMOTE_PATH"
echo "Local: $LOCAL_PATH"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

# Method 1: Try rsync (may not work if SCP is disabled)
echo "Attempting rsync..."
if rsync -avz -e "ssh -i $SSH_KEY" \
    "$RUNPOD_USER@$RUNPOD_HOST:$REMOTE_PATH/" \
    "$LOCAL_PATH/" 2>&1; then
    echo "✓ Successfully downloaded with rsync!"
    exit 0
fi

echo ""
echo "rsync failed, trying tar+ssh method..."

# Method 2: Use tar over SSH (most reliable)
echo "Creating archive on RunPod..."
ssh -i "$SSH_KEY" "$RUNPOD_USER@$RUNPOD_HOST" \
    "cd /workspace/petase-lab/runs && tar czf - colabfold_predictions_gpu" \
    | tar xzf - -C "$HOME/Desktop/petase-lab/runs/"

echo "✓ Successfully downloaded with tar+ssh!"

