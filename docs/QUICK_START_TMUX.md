# Quick Start: Run Both Jobs in tmux

## On RunPod (GPU): Aggressive Mask

```bash
# SSH into your RunPod GPU instance
bash scripts/rfdiffusion_aggressive_tmux.sh
```

**This will:**
- Create tmux session `rfdiffusion_aggressive`
- Run 300 designs with aggressive mask (18 positions)
- Estimated time: 6-12 hours
- Results: `runs/2026-01-03_rfdiffusion_aggressive/`

**Commands:**
```bash
# Attach to see progress
tmux attach -t rfdiffusion_aggressive

# Detach: Ctrl+B, then D
# View logs: tail -f runs/rfdiffusion_aggressive.log
```

## On Mac: FoldX Scoring

**First, download results from RunPod:**
```bash
# Download conservative results (delete traj/ and schedules/ first)
rsync -av --exclude='traj' --exclude='schedules' \
  runpod:/workspace/petase-lab/runs/2026-01-03_rfdiffusion_conservative/ \
  ./runs/2026-01-03_rfdiffusion_conservative/
```

**Then run FoldX:**
```bash
# Run FoldX on all 300 designs in tmux
bash scripts/foldx_stability_mac_tmux.sh
```

**This will:**
- Create tmux session `foldx_stability`
- Run FoldX on all 300 designs in parallel (uses all CPU cores)
- Estimated time: ~1-3 hours (depending on CPU cores)
- Results: `runs/2026-01-03_rfdiffusion_conservative/foldx_scores/foldx_scores.csv`

**Commands:**
```bash
# Attach to see progress
tmux attach -t foldx_stability

# Detach: Ctrl+B, then D
# View logs: tail -f runs/foldx_stability.log
```

## Prerequisites

### RunPod:
- RFdiffusion installed (run `bash scripts/setup_rfdiffusion_runpod.sh` if needed)
- Model weights downloaded
- Input PDB (7SH6.pdb) present

### Mac:
- FoldX installed (download from https://foldxsuite.org.eu/)
- Set `FOLDX_PATH` environment variable:
  ```bash
  export FOLDX_PATH=/opt/foldx/FoldX
  ```
- GNU parallel installed:
  ```bash
  brew install parallel
  ```
- tmux installed:
  ```bash
  brew install tmux
  ```

## After Both Complete

1. **Aggressive mask results:** Download from RunPod
2. **FoldX scores:** Check `runs/2026-01-03_rfdiffusion_conservative/foldx_scores/foldx_scores.csv`
3. **Rank designs:** Sort by FoldX stability (lower is better)
4. **Next step:** Run AlphaFold on top 50-100 designs

