#!/usr/bin/env bash
# Run ColabFold in tmux with logging
# Best of both worlds: persistent session + log file

set -euo pipefail

FASTA=${1:-runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta}
OUTPUT_DIR=${2:-runs/colabfold_predictions_gpu}
LOG_FILE=${3:-colabfold.log}
SESSION_NAME=${4:-colabfold}

# Check if FASTA exists
if [ ! -f "$FASTA" ]; then
    echo "Error: FASTA file not found: $FASTA"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Install tmux if needed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update -qq
    apt-get install -y tmux > /dev/null 2>&1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "Attaching to existing session..."
    echo "To detach: Press Ctrl+B, then D"
    tmux attach -t "$SESSION_NAME"
else
    echo "Starting ColabFold in tmux session '$SESSION_NAME'..."
    echo "Log file: $LOG_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Commands:"
    echo "  Detach: Ctrl+B, then D"
    echo "  Reattach: tmux attach -t $SESSION_NAME"
    echo "  View log: tail -f $LOG_FILE"
    echo ""
    
    # Create tmux session and run ColabFold with logging
    tmux new-session -d -s "$SESSION_NAME" \
        "colabfold_batch \
            --num-recycle 3 \
            --num-models 5 \
            --amber \
            '$FASTA' \
            '$OUTPUT_DIR' 2>&1 | tee '$LOG_FILE'"
    
    # Attach to session
    tmux attach -t "$SESSION_NAME"
fi

