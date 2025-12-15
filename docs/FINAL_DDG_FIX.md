# Final DDG Fix (corrected)

## Root cause(s)

### 1) Residue numbering
`cartesian_ddg` is operating on **pose numbering** (sequential residues 1..N after Rosetta loads the PDB), while your relaxed PDB keeps **PDB numbering** starting at 30.  
So e.g. **PDB 160 = pose 131** for your `PETase_raw_0008.pdb`.

### 2) Mutfile parsing strictness
Some builds are picky about mutfiles containing comments/blank lines. To make runs robust, `scripts/rosetta_ddg.sh` now **auto-generates a cleaned mutfile** (no blank lines, no `#` lines) and passes that to Rosetta.

## The working mutfile format (your setup)

Use **4 fields**:

```
<wt-aa> <pose_resnum> <chain-id> <mut-aa>
```

Example:

```
S 131 A A
```

## Output filenames

Rosetta may name outputs after the mutfile stem (e.g. `*.clean.ddg`, `*.clean.json`).  
`scripts/rosetta_ddg.sh` now copies the produced files to **`mutlist.ddg`** and **`mutlist.json`** in the run’s `outputs/` folder for consistency.

## Next step

```bash
bash scripts/rosetta_ddg.sh runs/2025-12-05_relax_cart_v1/outputs/PETase_raw_0008.pdb configs/rosetta/mutlist.mut
```
