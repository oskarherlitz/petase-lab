# RFdiffusion Quick Start Guide

## Prerequisites Check

Before running, make sure you have:
- ✅ RFdiffusion installed (`python3 -c "import rfdiffusion"`)
- ✅ DGL working (`python3 -c "import dgl"`)
- ✅ Model weights downloaded (`ls data/models/rfdiffusion/Base_ckpt.pt`)
- ✅ Input PDB ready (`ls data/structures/7SH6/raw/7SH6.pdb`)

## Quick Commands

### 1. Test Run (5 designs, ~5-10 minutes)

```bash
# Start in tmux (recommended)
bash scripts/rfdiffusion_tmux.sh test

# Or run directly (without tmux)
bash scripts/rfdiffusion_direct.sh data/structures/7SH6/raw/7SH6.pdb 5
```

**Monitor:**
```bash
# Attach to tmux session
tmux attach -t rfdiffusion_test

# Or watch logs
tail -f runs/rfdiffusion_rfdiffusion_test.log
```

### 2. Conservative Mask Run (300 designs, ~6-12 hours)

```bash
# Start in tmux (recommended for overnight runs)
bash scripts/rfdiffusion_tmux.sh conservative

# Monitor
tmux attach -t rfdiffusion_conservative
# Or: tail -f runs/rfdiffusion_rfdiffusion_conservative.log
```

### 3. Aggressive Mask Run (300 designs, ~6-12 hours)

```bash
# Start in tmux
bash scripts/rfdiffusion_tmux.sh aggressive

# Monitor
tmux attach -t rfdiffusion_aggressive
# Or: tail -f runs/rfdiffusion_rfdiffusion_aggressive.log
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

## Output Locations

- **Test run**: `runs/YYYY-MM-DD_rfdiffusion_test/designs/`
- **Conservative**: `runs/YYYY-MM-DD_rfdiffusion_conservative/designs/`
- **Aggressive**: `runs/YYYY-MM-DD_rfdiffusion_aggressive/designs/`

## Check Progress

```bash
# Count completed designs
ls runs/*/designs/*.pdb 2>/dev/null | wc -l

# Check GPU usage
watch -n 1 nvidia-smi

# View latest log
tail -f runs/rfdiffusion_*.log
```

## Troubleshooting

### Session Exited Immediately
```bash
# Check logs
cat runs/rfdiffusion_*.log

# Check prerequisites
bash scripts/check_rfdiffusion_prereqs.sh
```

### DGL Import Error
```bash
# Fix DGL
bash scripts/fix_dgl_final.sh
```

### CUDA Library Error
```bash
# The scripts should handle this automatically, but if needed:
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

## Running Multiple Jobs

You can run both conservative and aggressive masks simultaneously:

```bash
# Start conservative
bash scripts/rfdiffusion_tmux.sh conservative

# Start aggressive (in separate session)
bash scripts/rfdiffusion_tmux.sh aggressive

# Monitor both
tmux attach -t rfdiffusion_conservative  # Terminal 1
tmux attach -t rfdiffusion_aggressive    # Terminal 2
```

## Expected Runtimes

- **Test (5 designs)**: ~5-10 minutes
- **300 designs (RTX 4090)**: ~3-5 hours
- **300 designs (RTX 3090)**: ~5-8 hours
- **300 designs (Mac)**: Not recommended (no CUDA GPU)

