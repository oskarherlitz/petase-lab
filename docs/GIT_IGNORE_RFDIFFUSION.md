# Ignoring RFdiffusion Results in Git

RFdiffusion run outputs (PDB files, TRB files, logs) are now ignored by git to prevent conflicts when pulling on RunPod.

## What's Ignored

The following patterns are in `.gitignore`:
- `runs/*rfdiffusion*/` - All RFdiffusion run directories
- `runs/*rfdiffusion*/*.pdb` - All PDB files in RFdiffusion runs
- `runs/*rfdiffusion*/*.trb` - All TRB files in RFdiffusion runs
- `runs/*rfdiffusion*/traj/` - Trajectory folders
- `runs/*rfdiffusion*/schedules/` - Schedule cache folders
- `runs/*rfdiffusion*/run_inference.log` - Log files
- `runs/*rfdiffusion*/foldx_scores/` - FoldX scoring results
- `runs/*rfdiffusion*/alphafold_validation/` - AlphaFold validation results

## If Files Are Already Tracked

If RFdiffusion files were already committed to git before adding to `.gitignore`, you need to remove them from git tracking (but keep the files locally):

```bash
# Remove from git tracking (but keep files locally)
git rm --cached -r runs/2026-01-03_rfdiffusion_conservative/
git rm --cached -r runs/2026-01-03_rfdiffusion_aggressive/
git rm --cached -r runs/*rfdiffusion*/

# Commit the removal
git commit -m "Remove RFdiffusion outputs from git tracking"

# Now you can pull on RunPod without conflicts
```

## On RunPod

After pulling, RFdiffusion results will be ignored:
- No conflicts when pulling
- Results stay on RunPod (not synced to git)
- You can download results manually when needed

## Downloading Results

To get results from RunPod to your Mac:

```bash
# Download conservative results (excluding ignored files)
rsync -av --exclude='traj' --exclude='schedules' \
  runpod:/workspace/petase-lab/runs/2026-01-03_rfdiffusion_conservative/ \
  ./runs/2026-01-03_rfdiffusion_conservative/
```

## What to Keep in Git

If you want to keep summary files (like validation summaries), you can add them explicitly:

```bash
# Add specific summary files (if needed)
git add -f runs/2026-01-03_rfdiffusion_conservative/VALIDATION_SUMMARY.md
```

But generally, all RFdiffusion outputs should be ignored and downloaded manually when needed.

