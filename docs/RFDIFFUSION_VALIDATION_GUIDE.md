# RFdiffusion Results Validation Guide

## Quick Validation Summary

Your conservative mask run completed successfully! ✅

**Results:**
- ✅ **300 PDB files** (all designs generated)
- ✅ **300 TRB files** (metadata for each design)
- ✅ **Consistent structure**: All PDBs have 1044 atoms (residues 29-289)
- ✅ **No errors** in run log
- ✅ **Complete run**: All 300 designs finished

## Detailed Validation

### 1. File Integrity ✅

```bash
# Check file counts
ls -1 runs/2026-01-03_rfdiffusion_conservative/*.pdb | wc -l  # Should be 300
ls -1 runs/2026-01-03_rfdiffusion_conservative/*.trb | wc -l  # Should be 300

# Check PDB structure
grep -c "^ATOM" runs/2026-01-03_rfdiffusion_conservative/designs_0.pdb  # Should be 1044
```

**What to look for:**
- All PDBs should have ~1044 atoms (261 residues × 4 atoms per residue)
- Residue numbering should start at 29 and end at 289
- All files should be readable (no corruption)

### 2. Structure Validation

**Visual inspection:**
```bash
# Open a few designs in PyMOL
pymol runs/2026-01-03_rfdiffusion_conservative/designs_0.pdb
pymol runs/2026-01-03_rfdiffusion_conservative/designs_149.pdb
pymol runs/2026-01-03_rfdiffusion_conservative/designs_299.pdb
```

**What to check:**
- Structures should be folded (not random coils)
- Backbone should be continuous
- No major clashes or distortions
- Catalytic triad (Ser160, Asp206, His237) should be preserved

**AlphaFold validation:**
Run AlphaFold on a sample of designs to verify:
- Predicted structure matches RFdiffusion output
- Low RMSD (< 2 Å) indicates good structure quality
- High pLDDT scores (> 70) indicate confidence

### 3. Sequence Analysis

**Extract sequences:**
```python
# Quick sequence extraction
import pickle
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
sequences = []

for i in range(10):  # Check first 10 designs
    structure = parser.get_structure('design', f'runs/2026-01-03_rfdiffusion_conservative/designs_{i}.pdb')
    seq = ''
    for residue in structure.get_residues():
        if residue.id[0] == ' ':
            seq += residue.get_resname()
    sequences.append(seq)
    print(f"Design {i}: {seq[:50]}...")
```

**What to check:**
- Sequences should be diverse (not all identical)
- Conservative mask positions (114, 117, 119, 140, 159, 165, 168, 180, 188, 205, 214, 269, 282) should show variation
- FAST-PETase core positions (121, 186, 224, 233, 280) should be preserved

### 4. TRB Metadata Analysis

**Check what was designed:**
```python
import pickle

# Load TRB file
with open('runs/2026-01-03_rfdiffusion_conservative/designs_0.trb', 'rb') as f:
    trb = pickle.load(f)

# Check inpaint_seq (which positions were masked)
inpaint_seq = trb['inpaint_seq']
print(f"Positions masked: {sum(inpaint_seq)} out of {len(inpaint_seq)}")

# Check mapping
con_ref_pdb_idx = trb['con_ref_pdb_idx']  # Input PDB indices
con_hal_pdb_idx = trb['con_hal_pdb_idx']  # Output PDB indices
print(f"Residue mapping: {len(con_ref_pdb_idx)} residues")
```

**What to check:**
- `inpaint_seq` should have True values at conservative mask positions
- Mapping should be consistent across designs
- Config should match your run parameters

## Testing Strategy

### Phase 1: Quick Quality Check (Do This First)

1. **Visual inspection** (5-10 designs)
   - Open in PyMOL
   - Check for obvious structural issues
   - Verify catalytic triad is present

2. **Sequence diversity check**
   - Extract sequences from first 20 designs
   - Verify they're not all identical
   - Check that mask positions show variation

3. **Structure quality check**
   - Run AlphaFold on 5-10 random designs
   - Compare RMSD to RFdiffusion output
   - Check pLDDT scores

### Phase 2: Stability Scoring (Next Step)

1. **Rosetta ΔΔG calculation**
   ```bash
   # Run Rosetta on top designs
   bash scripts/rosetta_ddg.sh runs/2026-01-03_rfdiffusion_conservative/designs_0.pdb
   ```

2. **FoldX stability prediction**
   - Run FoldX on all designs
   - Rank by predicted ΔΔG
   - Select top 20-50 designs

### Phase 3: Full Validation (For Top Candidates)

1. **AlphaFold structure prediction**
   - Run on top 20-50 designs
   - Verify structure quality
   - Check catalytic region RMSD

2. **MD simulation** (optional)
   - Short MD runs on top designs
   - Check stability at elevated temperature
   - Verify catalytic geometry

## Recommended Next Steps

1. **Immediate:**
   - ✅ Results validated - all 300 designs complete
   - Run quick visual inspection (5-10 designs)
   - Extract sequences and check diversity

2. **Short-term:**
   - Run AlphaFold on sample (10-20 designs)
   - Run Rosetta/FoldX ΔΔG on all designs
   - Rank by predicted stability

3. **Medium-term:**
   - Select top 20-50 designs based on stability scores
   - Run full AlphaFold validation on top designs
   - Compare to FAST-PETase baseline

4. **Optional:**
   - Run aggressive mask (300 more designs)
   - Combine results from both masks
   - Select final candidates for experimental validation

## Common Issues and Solutions

**Issue: All sequences identical**
- **Cause:** RFdiffusion may have converged to same solution
- **Solution:** Check if mask positions are actually being varied

**Issue: Structures look distorted**
- **Cause:** RFdiffusion may have generated invalid structures
- **Solution:** Run AlphaFold to verify, filter out low-quality designs

**Issue: Missing designs**
- **Cause:** Run may have failed partway through
- **Solution:** Check run log for errors, re-run missing designs

**Issue: TRB files can't be read**
- **Cause:** File corruption or wrong format
- **Solution:** Re-download from RunPod, check file integrity

## Validation Script

Use the validation script:
```bash
bash scripts/validate_rfdiffusion_results.sh [results_directory]
```

This will check:
- File counts
- PDB integrity
- TRB metadata
- Sequence diversity (if BioPython installed)

