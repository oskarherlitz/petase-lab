# Running RFdiffusion on RunPod with tmux

## Why tmux?

- **Persistent sessions**: Job continues if you disconnect
- **Monitor progress**: Check status anytime
- **Background execution**: Detach and let it run
- **Multiple windows**: Run multiple jobs simultaneously

## Quick Start

### 1. Test Run (5 designs)

```bash
# Start test in tmux
bash scripts/rfdiffusion_tmux.sh test

# Attach to see progress
tmux attach -t rfdiffusion_test

# Detach: Ctrl+B, then D
```

### 2. Overnight Run (300 designs)

```bash
# Conservative mask
bash scripts/rfdiffusion_tmux.sh conservative

# Aggressive mask (in separate session)
bash scripts/rfdiffusion_tmux.sh aggressive

# Attach to monitor
tmux attach -t rfdiffusion_conservative
```

## tmux Commands

### Basic Navigation
- **Attach to session**: `tmux attach -t rfdiffusion_test`
- **Detach**: `Ctrl+B`, then `D`
- **List sessions**: `tmux ls`
- **Kill session**: `tmux kill-session -t rfdiffusion_test`

### While Inside tmux
- **Scroll up/down**: `Ctrl+B`, then `[` (enter copy mode), use arrow keys, `q` to exit
- **Split window**: `Ctrl+B`, then `%` (vertical) or `"` (horizontal)
- **Switch windows**: `Ctrl+B`, then arrow keys
- **New window**: `Ctrl+B`, then `c`

## Monitoring Progress

### View Logs (without attaching)
```bash
# Watch log file
tail -f runs/rfdiffusion_rfdiffusion_test.log

# Check how many designs completed
ls runs/*/designs/*.pdb | wc -l

# Check GPU usage
watch -n 1 nvidia-smi
```

### Check Session Status
```bash
# List all sessions
tmux ls

# Check if session is still running
tmux has-session -t rfdiffusion_conservative && echo "Running" || echo "Stopped"
```

## Running Multiple Jobs

### Option 1: Separate Sessions (Recommended)
```bash
# Start conservative mask
bash scripts/rfdiffusion_tmux.sh conservative

# Start aggressive mask (different session)
bash scripts/rfdiffusion_tmux.sh aggressive

# Monitor both
tmux attach -t rfdiffusion_conservative  # In one terminal
tmux attach -t rfdiffusion_aggressive    # In another terminal
```

### Option 2: Same Session, Different Windows
```bash
# Create session manually
tmux new -s rfdiffusion

# In tmux:
# Run conservative (Ctrl+B, then c for new window)
bash scripts/rfdiffusion_conservative.sh

# Switch to new window (Ctrl+B, then n)
# Run aggressive
bash scripts/rfdiffusion_aggressive.sh
```

## Troubleshooting

### Session Disappeared
```bash
# Check if it's still running
tmux ls

# If not listed, check logs
tail -f runs/rfdiffusion_*.log
```

### Can't Attach
```bash
# List all sessions
tmux ls

# Force attach (if session exists but won't attach)
tmux attach -t rfdiffusion_test -d
```

### Job Stopped
```bash
# Check logs for errors
tail -100 runs/rfdiffusion_*.log

# Restart in new session
tmux kill-session -t rfdiffusion_test  # If needed
bash scripts/rfdiffusion_tmux.sh test
```

## Example Workflow

```bash
# 1. Start overnight run
bash scripts/rfdiffusion_tmux.sh conservative

# 2. Detach and check back later
# (Ctrl+B, then D)

# 3. Check progress
tmux attach -t rfdiffusion_conservative

# 4. Or check logs
tail -f runs/rfdiffusion_rfdiffusion_conservative.log

# 5. When done, check outputs
ls runs/*_rfdiffusion_conservative/designs/*.pdb
```

## Tips

1. **Always use tmux for long jobs** - Prevents loss if connection drops
2. **Check logs regularly** - `tail -f runs/rfdiffusion_*.log`
3. **Monitor GPU usage** - `watch -n 1 nvidia-smi` in another terminal
4. **Save session names** - Use descriptive names for multiple runs
5. **Keep RunPod pod running** - Don't stop the pod while jobs are running

