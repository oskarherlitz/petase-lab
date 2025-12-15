# Sanity Check Report
**Date:** 2025-12-06  
**Agent:** Sanity Checker  
**Scope:** Pipeline components review for logic, structure, reproducibility, clarity, and reliability

---

## Executive Summary

This report identifies **15 critical issues** and **8 recommendations** across the PETase optimization pipeline. Issues are categorized by severity and component.

---

## 🔴 Critical Issues

### 0. **MISMATCH: JSON File is Actually CSV**
**Location:** `runs/2025-12-06_ddg_cart_mutlist/outputs/mutlist.json`  
**Issue:** The file `mutlist.json` contains CSV data (header: `run,mutation,ddg_reu`), not JSON. This suggests:
- Rosetta may not have produced JSON output
- JSON output may have a different filename pattern
- Previous `parse_ddg.py` run may have overwritten the file

**Impact:** 
- `parse_ddg.py` fails to parse (tries to `json.load()` on CSV)
- Pipeline appears broken
- No way to know if Rosetta actually produced JSON

**Investigation Needed:**
- Check Rosetta documentation for actual JSON output filename
- Verify if `-ddg:json true` actually produces JSON
- Check if JSON output requires different Rosetta version or flags
- Verify expected JSON structure from Rosetta cartesian_ddg

**Fix:** 
1. Determine correct JSON filename pattern from Rosetta
2. Add validation to detect CSV vs JSON files
3. Add check to prevent overwriting Rosetta output files

---

### 1. **Fragile Path Parsing in `parse_ddg.py`**
**Location:** `scripts/parse_ddg.py:11`  
**Issue:** Hardcoded path split assumes specific directory structure
```python
run = jpath.split("/")[1]  # ❌ Breaks with different path structures
```
**Impact:** 
- Fails on Windows paths (backslashes)
- Fails with absolute paths
- Fails if run directory is nested differently
- Silent failure produces incorrect run names

**Example failure:**
- Path: `/Users/oskarherlitz/Desktop/petase-lab/runs/2025-12-06_ddg_cart_mutlist/outputs/mutlist.json`
- Current: `run = "Users"` (wrong!)
- Expected: `run = "2025-12-06_ddg_cart_mutlist"`

**Fix:** Use `pathlib` or extract run name from path more robustly:
```python
from pathlib import Path
run = Path(jpath).parent.parent.name  # Gets "2025-12-06_ddg_cart_mutlist"
```

---

### 2. **No Input Validation in `parse_ddg.py`**
**Location:** `scripts/parse_ddg.py:8-13`  
**Issues:**
- No check if JSON files exist before opening
- No validation that JSON is well-formed
- No handling of missing `"mutations"` key
- No handling of missing `"ddg"` key in mutation info
- No error messages for debugging

**Impact:** Silent failures, empty CSV output, no indication of what went wrong

**Current behavior:**
```python
for jpath in inputs:
    with open(jpath) as jf:  # ❌ No try/except, no file existence check
        j = json.load(jf)     # ❌ No JSON validation
    for mut, info in j.get("mutations", {}).items():  # ✅ Good use of .get()
        w.writerow([run, mut, info.get("ddg")])  # ❌ Missing ddg = None written
```

**Fix:** Add validation and error handling:
```python
for jpath in inputs:
    if not os.path.exists(jpath):
        print(f"Warning: File not found: {jpath}", file=sys.stderr)
        continue
    try:
        with open(jpath) as jf:
            j = json.load(jf)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {jpath}: {e}", file=sys.stderr)
        continue
    except Exception as e:
        print(f"Error reading {jpath}: {e}", file=sys.stderr)
        continue
    
    mutations = j.get("mutations", {})
    if not mutations:
        print(f"Warning: No mutations found in {jpath}", file=sys.stderr)
        continue
    
    for mut, info in mutations.items():
        ddg = info.get("ddg")
        if ddg is None:
            print(f"Warning: Missing ddg for mutation {mut} in {jpath}", file=sys.stderr)
            continue
        w.writerow([run, mut, ddg])
```

---

### 3. **No Empty Input Check in `parse_ddg.py`**
**Location:** `scripts/parse_ddg.py:3`  
**Issue:** No validation that any input files were found
```python
inputs = sorted([p for pat in sys.argv[1:-1] for p in glob.glob(pat)])
# ❌ No check if inputs is empty
```

**Impact:** Script runs successfully but produces empty CSV, causing confusion (as seen in terminal output: "No data found in CSV file")

**Fix:**
```python
inputs = sorted([p for pat in sys.argv[1:-1] for p in glob.glob(pat)])
if not inputs:
    print(f"Error: No input files found matching patterns: {sys.argv[1:-1]}", file=sys.stderr)
    sys.exit(1)
```

---

### 4. **Silent Failures in `rank_designs.py`**
**Location:** `scripts/rank_designs.py:8-12`  
**Issue:** Exceptions are caught and silently ignored
```python
try:
    row['ddg_reu'] = float(row['ddg_reu']) if row['ddg_reu'] else float('inf')
    rows.append(row)
except (ValueError, KeyError):
    continue  # ❌ Silent failure - no indication of what went wrong
```

**Impact:** 
- Invalid data rows are silently skipped
- No way to know if data was corrupted
- No debugging information

**Fix:** Log warnings:
```python
except (ValueError, KeyError) as e:
    print(f"Warning: Skipping invalid row: {row} ({e})", file=sys.stderr)
    continue
```

---

### 5. **No Input Validation in `rank_designs.py`**
**Location:** `scripts/rank_designs.py:5-6`  
**Issues:**
- No check if CSV file exists
- No check if CSV has required columns
- No argument count validation

**Impact:** Cryptic errors when file missing or malformed

**Fix:**
```python
if len(sys.argv) < 2:
    print("Usage: python scripts/rank_designs.py <csv_file> [top_k]", file=sys.stderr)
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: CSV file not found: {csv_file}", file=sys.stderr)
    sys.exit(1)

with open(csv_file) as f:
    reader = csv.DictReader(f)
    required_cols = ['run', 'mutation', 'ddg_reu']
    if not all(col in reader.fieldnames for col in required_cols):
        print(f"Error: CSV missing required columns. Found: {reader.fieldnames}, Required: {required_cols}", file=sys.stderr)
        sys.exit(1)
```

---

### 6. **Unvalidated Glob Pattern in `rosetta_ddg.sh`**
**Location:** `scripts/rosetta_ddg.sh:4`  
**Issue:** Default MODEL parameter uses glob that may match 0 or multiple files
```bash
MODEL=${1:-runs/*relax*/outputs/*.pdb}  # ❌ Could match 0, 1, or many files
```

**Impact:**
- If no files match: Rosetta fails with unclear error
- If multiple files match: Only first is used (unpredictable)
- No indication which file was actually used

**Fix:** Validate that exactly one file matches:
```bash
MODEL=${1:-}
if [ -z "$MODEL" ]; then
    # Find most recent relaxed structure
    MODEL=$(ls -t runs/*relax*/outputs/*.pdb 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then
        echo "Error: No relaxed PDB files found. Run relaxation first." >&2
        exit 1
    fi
    echo "Using most recent relaxed structure: $MODEL"
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL" >&2
    exit 1
fi
```

---

### 7. **No Rosetta Binary Validation**
**Location:** `scripts/rosetta_ddg.sh:14-20`, `scripts/rosetta_relax.sh:12-18`  
**Issue:** Binary detection doesn't verify the binary is executable
```bash
if [[ -f "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]]; then
    DDG_BIN="cartesian_ddg.static.macosclangrelease"
# ❌ Checks existence but not executability
```

**Impact:** Script may fail at runtime with "Permission denied" or other execution errors

**Fix:**
```bash
if [[ -f "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]]; then
    DDG_BIN="cartesian_ddg.static.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/cartesian_ddg.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.macosclangrelease" ]]; then
    DDG_BIN="cartesian_ddg.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/cartesian_ddg.linuxgccrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.linuxgccrelease" ]]; then
    DDG_BIN="cartesian_ddg.linuxgccrelease"
else
    echo "Error: No executable Rosetta binary found in $ROSETTA_BIN" >&2
    echo "Expected one of: cartesian_ddg.static.macosclangrelease, cartesian_ddg.macosclangrelease, cartesian_ddg.linuxgccrelease" >&2
    exit 1
fi
```

---

### 8. **No Mutation File Validation**
**Location:** `scripts/rosetta_ddg.sh:5`  
**Issue:** No validation that mutation file exists or has correct format
```bash
MUT=${2:-configs/rosetta/mutlist.mut}
# ❌ No check if file exists or is valid
```

**Impact:** Rosetta fails with cryptic error if file missing or malformed

**Fix:**
```bash
MUT=${2:-configs/rosetta/mutlist.mut}
if [ ! -f "$MUT" ]; then
    echo "Error: Mutation file not found: $MUT" >&2
    exit 1
fi

# Basic format validation
if ! grep -q "^total [0-9]" "$MUT"; then
    echo "Warning: Mutation file may be malformed. Expected 'total N' on first line." >&2
fi
```

---

### 9. **No Input PDB Validation in `rosetta_relax.sh`**
**Location:** `scripts/rosetta_relax.sh:4`  
**Issue:** No check if input PDB exists
```bash
IN=${1:-data/raw/PETase_raw.pdb}
# ❌ No validation
```

**Impact:** Rosetta fails with unclear error if file missing

**Fix:**
```bash
IN=${1:-data/raw/PETase_raw.pdb}
if [ ! -f "$IN" ]; then
    echo "Error: Input PDB file not found: $IN" >&2
    exit 1
fi
```

---

### 10. **Inconsistent Output Redirection Usage**
**Location:** Documentation vs `parse_ddg.py`  
**Issue:** Documentation shows output redirection, but script expects file argument
```bash
# Documentation says:
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial.csv

# But script expects:
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial.csv
```

**Impact:** Confusion, potential data loss if user redirects to existing file

**Fix:** Update documentation to match script usage, or make script support both patterns

---

## 🟡 Medium Issues

### 11. **No Error Handling for Rosetta Failures**
**Location:** `scripts/rosetta_ddg.sh:22`, `scripts/rosetta_relax.sh:21`  
**Issue:** Scripts use `set -euo pipefail` but don't provide helpful error messages if Rosetta fails

**Impact:** User sees Rosetta error but no context about what went wrong

**Fix:** Add error checking after Rosetta execution:
```bash
if [ $? -ne 0 ]; then
    echo "Error: Rosetta execution failed. Check output above for details." >&2
    exit 1
fi
```

---

### 12. **No Validation of JSON Structure**
**Location:** `scripts/parse_ddg.py:10-13`  
**Issue:** Assumes specific JSON structure without validation

**Impact:** Fails silently if Rosetta changes output format

**Fix:** Add structure validation:
```python
if not isinstance(j, dict):
    print(f"Error: Expected JSON object, got {type(j).__name__} in {jpath}", file=sys.stderr)
    continue
```

---

### 13. **Missing Chain ID in Mutation File Format**
**Location:** `configs/rosetta/mutlist.mut`  
**Issue:** Mutation file format comment says "No chain ID needed" but Rosetta cartesian_ddg may require it for multi-chain proteins

**Impact:** Potential failures with multi-chain structures

**Recommendation:** Document when chain ID is required vs optional

---

### 14. **No Check for Empty Results**
**Location:** `scripts/parse_ddg.py`  
**Issue:** Script doesn't warn if no mutations were parsed

**Impact:** Empty CSV files are created without indication of failure

**Fix:** Add check after parsing loop:
```python
if rows_written == 0:
    print("Warning: No mutations were parsed from input files.", file=sys.stderr)
```

---

### 15. **Inconsistent Date Format in Run Names**
**Location:** `scripts/rosetta_ddg.sh:6`, `scripts/rosetta_relax.sh:5`  
**Issue:** Uses `date +%F` which may produce different formats on different systems

**Impact:** Inconsistent run directory names across systems

**Fix:** Use explicit format:
```bash
TS=$(date +%Y-%m-%d)  # Explicit format
```

---

## 🟢 Recommendations

### R1. **Add Logging to Scripts**
Add structured logging to track pipeline execution:
- Which files were processed
- How many mutations were found
- Any warnings or errors

### R2. **Create Validation Script**
Add `scripts/validate_pipeline.sh` to check:
- Required files exist
- Rosetta binaries are available
- Configuration files are valid
- Output directories are writable

### R3. **Add Unit Tests**
The test file `tests/test_parse_ddg.py` only has a placeholder. Add real tests for:
- Path parsing edge cases
- JSON parsing with various structures
- Error handling

### R4. **Document Error Cases**
Add troubleshooting section to documentation covering:
- What to do when JSON files are missing
- How to interpret empty CSV output
- Common Rosetta error messages

### R5. **Standardize Error Messages**
Use consistent error message format:
- `Error:` for fatal errors (exit 1)
- `Warning:` for recoverable issues (continue)
- Include file paths and line numbers where relevant

### R6. **Add Progress Indicators**
For long-running scripts, add progress output:
- Number of files processed
- Estimated time remaining
- Current mutation being calculated

### R7. **Validate Mutation File Format**
Add script to validate `mutlist.mut` format before running Rosetta:
- Check "total N" matches actual mutations
- Validate amino acid codes
- Check residue numbers are valid

### R8. **Add Configuration File**
Create `config.yaml` or similar to centralize:
- Default paths
- Rosetta parameters
- Output directories
- Mutation file locations

---

## Summary Statistics

- **Critical Issues:** 11 (including JSON/CSV mismatch)
- **Medium Issues:** 5
- **Recommendations:** 8
- **Files Affected:** 4 scripts, 1 config file, multiple docs, output files

---

## Priority Fix Order

1. **Immediate (Critical):**
   - Fix path parsing in `parse_ddg.py` (#1)
   - Add input validation to all scripts (#2, #3, #5, #6, #8, #9)
   - Fix silent failures (#4)

2. **Short-term (Medium):**
   - Add error handling for Rosetta failures (#11)
   - Validate JSON structure (#12)
   - Fix output redirection inconsistency (#10)

3. **Long-term (Recommendations):**
   - Add comprehensive testing (R3)
   - Create validation scripts (R2)
   - Improve documentation (R4)

---

## Notes

- Terminal output shows `parse_ddg.py` produced empty CSV, confirming issue #3
- **CRITICAL:** The `mutlist.json` file is actually CSV (contains `run,mutation,ddg_reu` header), not JSON. This is the root cause of the parsing failure.
- Rosetta may not have produced JSON output, or JSON has different filename/format than expected
- Scripts follow good practices (use of `set -euo pipefail`, `.get()` for safe dict access) but need more validation
- Need to verify Rosetta cartesian_ddg JSON output format and filename pattern

---

**Report Generated:** 2025-12-06  
**Next Review:** After fixes applied
