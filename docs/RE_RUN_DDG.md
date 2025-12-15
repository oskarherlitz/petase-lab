# Re-running DDG Calculation with Fixed Mutation File

## Status

✅ **Mutation file is now fixed** - includes chain IDs (required by Rosetta)
✅ **Validation is working** - will catch format issues before running
✅ **Parser is working** - correctly detects when mutations weren't processed

## The Issue

The old DDG runs (2025-12-06 and 2025-12-10) were done **before** the mutation file was fixed. They used the old format (missing chain IDs), so Rosetta didn't process mutations - only calculated WT scores.

## Solution: Re-run with Fixed File

The mutation file is now correct. You need to **re-run the DDG calculation**:

```bash
# Use the best relaxed structure
bash scripts/rosetta_ddg.sh runs/2025-12-05_relax_cart_v1/outputs/PETase_raw_0008.pdb configs/rosetta/mutlist.mut
```

## What Will Happen

1. **Validation** will check the mutation file format ✅ (will pass now)
2. **Rosetta** will process all 8 mutations ✅ (with correct format)
3. **Output** will contain mutation-specific DDG values ✅
4. **Parser** will successfully extract mutations ✅

## Expected Output

After re-running, you should see:
- `mutlist.ddg` containing both `WT_` and `MUT_...` entries (often on `COMPLEX:` lines)
- `mutlist.json` containing at least one mutation where `mut != wt`
- Parser will successfully extract all 8 mutations

## Verification

After the run completes, verify mutations were processed:

```bash
# Check for mutant labels in the .ddg file
grep "MUT_" runs/*ddg*/outputs/mutlist.ddg | head

# Parse results
python scripts/parse_ddg.py results/ddg_scans/initial.csv
```

You should see 8 mutations with DDG values!
