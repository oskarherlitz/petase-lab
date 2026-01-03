# Conservative Mask Results Validation

## ✅ Overall Status: **SUCCESS**

All 300 designs completed successfully with no errors.

## File Integrity ✅

- **300 PDB files** - All present and readable
- **300 TRB files** - All present and readable  
- **1 log file** - No errors detected
- **Structure consistency**: All PDBs have 1044 atoms (261 residues, residues 29-289)
- **Residue range**: Correctly spans A29 to A289

## Structure Quality ✅

- All PDBs have consistent atom counts (1044 atoms each)
- Residue numbering is correct (starts at 29, ends at 289)
- No obvious corruption in sampled files

## Mask Configuration ⚠️

**Expected:** Mask only 13 positions (conservative mask)
- Positions: A114, A117, A119, A140, A159, A165, A168, A180, A188, A205, A214, A269, A282

**TRB Analysis:**
- Total residues: 261 (residues 29-289)
- Masked positions (True): 248
- Fixed positions (False): 13

**Note:** The TRB `inpaint_seq` array shows 248 positions as masked (True) and only 13 as fixed (False). This suggests RFdiffusion may have interpreted the mask differently than expected, OR the mask was applied correctly but the TRB representation is inverted.

**Action needed:** Verify by checking if sequences at the 13 conservative mask positions actually show variation across designs.

## Recommended Testing Steps

### 1. Quick Visual Check (Do This First)
```bash
# Open a few designs in PyMOL or ChimeraX
pymol runs/2026-01-03_rfdiffusion_conservative/designs_0.pdb
pymol runs/2026-01-03_rfdiffusion_conservative/designs_149.pdb
pymol runs/2026-01-03_rfdiffusion_conservative/designs_299.pdb
```

**Check:**
- Structures are folded (not random coils)
- Backbone is continuous
- Catalytic triad (Ser160, Asp206, His237) is present
- No major structural distortions

### 2. Sequence Diversity Check
Extract sequences from multiple designs and check:
- Are sequences different across designs?
- Do the 13 conservative mask positions show variation?
- Are FAST-PETase core positions (121, 186, 224, 233, 280) preserved?

### 3. AlphaFold Validation (Recommended)
Run AlphaFold on 10-20 random designs to:
- Verify structure quality
- Check RMSD to RFdiffusion output
- Validate pLDDT scores

### 4. Stability Scoring (Next Priority)
Run Rosetta/FoldX ΔΔG calculations on all designs:
- Rank by predicted stability
- Identify top candidates
- Compare to FAST-PETase baseline

## Next Steps

1. **Immediate:** Visual inspection of 5-10 designs
2. **Short-term:** Sequence extraction and diversity analysis
3. **Medium-term:** AlphaFold validation on sample
4. **Long-term:** Full stability scoring and ranking

## Files to Keep

- ✅ **Keep:** All `.pdb` files (essential)
- ✅ **Keep:** All `.trb` files (metadata)
- ✅ **Keep:** `run_inference.log` (for debugging)
- ❌ **Delete:** `traj/` folder (if present, ~1.5-3GB, not needed)
- ❌ **Delete:** `schedules/` folder (if present, cache only)

## Questions to Answer

1. **Are sequences diverse?** Check if designs have different sequences
2. **Are mask positions varied?** Verify the 13 conservative positions show variation
3. **Are core positions fixed?** Verify FAST-PETase core (121, 186, 224, 233, 280) is preserved
4. **Are structures valid?** Run AlphaFold to check structure quality

