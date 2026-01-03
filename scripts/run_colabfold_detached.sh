#!/usr/bin/env bash
# Run ColabFold in a detached tmux session
# This allows you to disconnect and reconnect later

set -euo pipefail

FASTA=${1:-runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta}
OUTPUT_DIR=${2:-runs/colabfold_predictions_gpu}
SESSION_NAME=${3:-colabfold}

# Check if FASTA exists
if [ ! -f "$FASTA" ]; then
    echo "Error: FASTA file not found: $FASTA"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update -qq
    apt-get install -y tmux > /dev/null 2>&1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create or attach to tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    echo "To detach: Press Ctrl+B, then D"
    tmux attach -t "$SESSION_NAME"
else
    echo "Starting ColabFold in tmux session '$SESSION_NAME'..."
    echo "To detach: Press Ctrl+B, then D"
    echo "To reattach later: tmux attach -t $SESSION_NAME"
    echo ""
    
    # Create new tmux session and run ColabFold
    tmux new-session -d -s "$SESSION_NAME" \
        "colabfold_batch \
            --num-recycle 3 \
            --num-models 5 \
            --amber \
            '$FASTA' \
            '$OUTPUT_DIR' \
        && echo '' && echo '✓ Prediction complete!' && echo 'Results: $OUTPUT_DIR' && echo '' && echo 'Press any key to exit...' && read"
    
    # Attach to the session
    tmux attach -t "$SESSION_NAME"
fi

