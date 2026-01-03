# RFdiffusion Output Files Guide

## Essential Files (Keep These)

### `.pdb` files
- **What:** Final predicted structure for each design
- **Size:** ~100-500KB each
- **Needed for:** All downstream analysis (Rosetta, FoldX, AlphaFold)
- **300 designs:** ~30-150MB total

### `.trb` files
- **What:** Metadata for each design (contig mapping, config, inpaint_seq, etc.)
- **Size:** ~10-50KB each
- **Needed for:** Understanding what was designed, mapping residues
- **300 designs:** ~3-15MB total

## Optional Files (Can Delete)

### `traj/` folder
- **What:** Full diffusion trajectories (multi-step PDBs showing the denoising process)
  - `*_Xt-1_traj.pdb`: What went into the model at each timestep
  - `*_pX0_traj.pdb`: What the model predicted at each timestep
- **Size:** ~5-10MB per design × 300 = **~1.5-3GB total**
- **Needed for:** Visualization/debugging in PyMOL (see how structure evolved)
- **Can delete:** Yes - not needed for analysis, only for visualization
- **To disable:** Add `inference.write_trajectory=False` to run command

### `schedules/` folder
- **What:** Cached diffusion schedules (precomputed noise schedules)
- **Size:** Small (~1-10MB total)
- **Needed for:** Speeding up subsequent runs with same parameters
- **Can delete:** Yes - RFdiffusion will regenerate them if needed
- **Note:** These are just cache files, not essential

## Recommendation

**Before downloading from RunPod:**

1. **Delete `traj/` folder** - Saves ~1.5-3GB, not needed for analysis
2. **Delete `schedules/` folder** - Small but not essential, can regenerate
3. **Keep `.pdb` and `.trb` files** - These are essential

**Total space saved:** ~1.5-3GB (mostly from trajectories)

**To delete on RunPod before downloading:**
```bash
# In your output directory
rm -rf traj/
rm -rf schedules/
```

**Or download only essential files:**
```bash
# Download only PDB and TRB files
rsync -av --include='*.pdb' --include='*.trb' --exclude='*' \
  runpod:/workspace/petase-lab/runs/2026-01-03_rfdiffusion_conservative/ \
  ./runs/2026-01-03_rfdiffusion_conservative/
```

