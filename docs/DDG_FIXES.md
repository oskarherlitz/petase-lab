# DDG Parsing Issues - Fixes Applied

## Problems Identified

### Problem 1: Mutation File Format Missing Chain IDs
**Issue**: Rosetta `cartesian_ddg` requires chain identifiers even for single-chain proteins. The mutation file was using format:
```
S 160 A
```
But Rosetta requires:
```
S 160 A A
```
Where the first `A` is the chain ID and the second `A` is the mutant amino acid.

**Fix**: Updated `configs/rosetta/mutlist.mut` to include chain IDs for all mutations.

### Problem 1b: Residue numbering mismatch (PDB vs pose numbering)
**Issue**: The relaxed PDB keeps original PDB numbering (starts at 30), but `cartesian_ddg` effectively uses **pose numbering** (sequential 1..N). If the numbers don’t match, Rosetta can fail assertions or produce WT-only output.

**Fix**: Updated `configs/rosetta/mutlist.mut` to use **pose numbering** (e.g. PDB 160 → pose 131 for `PETase_raw_0008.pdb`).

### Problem 2: No Mutation Processing Detection
**Issue**: Rosetta was running but silently failing to process mutations. The output files only contained wildtype (WT) scores with empty mutations arrays, but the script didn't detect this.

**Fix**: Added validation in `scripts/rosetta_ddg.sh` to:
- Check mutation file format before running (validates chain IDs are present)
- Validate output after running (checks if mutations were actually processed)
- Provide clear error messages when mutations aren't processed

### Problem 3: Parser Error Messages Not Helpful
**Issue**: When mutations weren't processed, the parser gave generic warnings that didn't explain the root cause.

**Fix**: Enhanced `scripts/parse_ddg.py` to:
- Detect empty mutations arrays in JSON files
- Identify when .ddg files only have COMPLEX lines (WT) but no MUTATION lines
- Provide specific troubleshooting guidance

## Files Modified

1. **configs/rosetta/mutlist.mut**
   - Added chain IDs to all mutation lines
   - Updated format documentation

2. **scripts/rosetta_ddg.sh**
   - Added mutation file format validation (checks for chain IDs)
   - Added post-run validation to detect when mutations weren't processed
   - Improved error messages
   - Cleans mutfile (strips comments/blank lines) before passing to Rosetta
   - Copies produced `*.ddg/*.json` to `mutlist.ddg/mutlist.json` for consistent downstream greps/parsing

3. **scripts/parse_ddg.py**
   - Enhanced error detection for empty mutations arrays
   - Improved diagnostics for .ddg file parsing
   - Better error messages with troubleshooting guidance

## Testing

To test the fixes:

1. **Verify mutation file format**:
   ```bash
   cat configs/rosetta/mutlist.mut
   ```
   Should show format: `<wt-aa> <residue-number> <chain-id> <mutant-aa>`

2. **Run DDG calculation** (will now validate format):
   ```bash
   bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
   ```

3. **Parse results**:
   ```bash
   python scripts/parse_ddg.py results/ddg_scans/initial.csv
   ```

## Expected Behavior After Fixes

- **Before running**: Script validates mutation file format and exits with error if chain IDs are missing
- **During run**: Rosetta should now process mutations correctly
- **After run**: Script validates that mutations were processed and warns if they weren't
- **Parsing**: Parser provides clear error messages if mutations weren't found

## Next Steps

1. Re-run DDG calculation with the fixed mutation file
2. Verify that mutations are now processed (check for `MUT_` in `.ddg`)
3. Parse results and verify DDG values are extracted

## Mutation File Format Reference

**Correct format**:
```
total N
1
<wt-aa> <residue-number> <chain-id> <mutant-aa>
1
<wt-aa> <residue-number> <chain-id> <mutant-aa>
...
```

**Example**:
```
total 2
1
S 160 A A
1
D 206 A N
```

**Common mistakes**:
- Missing chain ID: `S 160 A` ❌
- Wrong order: `A 160 S A` ❌
- Missing fields: `S 160` ❌
