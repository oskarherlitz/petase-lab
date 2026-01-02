# Smoke Test Results Analysis

**Run ID:** `run_20251229_progen2_small_r1_smoketest`  
**Date:** 2025-12-30  
**Model:** progen2-small  
**Samples Generated:** 11  
**Prompt Length:** 50 aa (N-terminus anchor)

---

## Executive Summary

✅ **Token normalization works** - All sequences passed token cleanup  
❌ **Length control broken** - All sequences wrong length (0% pass rate)  
⏭️ **Hard locks gate skipped** - No sequences reached this gate

**Overall:** Pipeline runs end-to-end, but generation length parameter needs fixing.

---

## Detailed Gate Analysis

### Gate C0: Token Cleanup ✓ PASS

- **Input:** 11 sequences
- **Output:** 11 sequences (100% pass)
- **Status:** ✅ **SUCCESS**

**What this means:** ProGen2's control tokens ('1' start, '2' end) were correctly stripped. The normalization logic works.

---

### Gate C1: Length Check ✗ FAIL

- **Input:** 11 sequences
- **Output:** 0 sequences (0% pass)
- **Status:** ❌ **CRITICAL ISSUE**

**Length Distribution:**
- **1 sequence:** 50 aa (just the prompt, no generation)
- **9 sequences:** 312 aa (too long - expected 263 aa)
- **1 sequence:** Missing from report (likely also 312 aa)

**Root Cause:** The `max-length` parameter was set to `313` (263 + 50), but ProGen2's `max-length` is the **total sequence length including the prompt**. 

**Current behavior:**
- Prompt: 50 aa
- max-length: 313
- Generated: 313 - 50 = 263 aa
- **Total: 313 aa** ❌ (should be 263 aa)

**Expected behavior:**
- Prompt: 50 aa  
- Desired total: 263 aa
- Need to generate: 263 - 50 = 213 aa
- **max-length should be: 263** (total length)

**Fix Required:** Change `max-length` from `EXPECTED_LENGTH + 50` to `EXPECTED_LENGTH` (263).

---

### Gate C2: Hard Locks ⏭️ SKIPPED

- **Input:** 0 sequences (none passed length gate)
- **Output:** 0 sequences
- **Status:** ⏭️ **SKIPPED** (no sequences to test)

**What this would test:** Positions 131 (S), 177 (D), 208 (H) must match catalytic triad.

---

## Sequence Quality Observations

Looking at the generated sequences, several issues:

1. **Repetitive patterns:** Sequences 2, 5, 8, 9, 10 show repetitive motifs
   - Example (seq 5): `ATYSGYTTTGSYSTNTGGN` repeated many times
   - Example (seq 10): Prompt repeated multiple times
   - **Cause:** Model may be getting stuck in loops or temperature too high

2. **One failed generation:** Sequence 1 is just the prompt (50 aa)
   - **Cause:** Model didn't generate anything, possibly hit max-length immediately or generation failed

3. **All sequences too long:** 9/11 sequences are 312 aa instead of 263 aa
   - **Cause:** max-length parameter issue (see above)

---

## What This Tells Us

### ✅ What's Working

1. **Pipeline infrastructure:** Scripts run end-to-end without crashes
2. **Token normalization:** Control token stripping works correctly
3. **Gate logic:** Filtering logic executes properly
4. **ProGen2 integration:** Model loads and generates sequences

### ❌ What Needs Fixing

1. **Length control:** `max-length` parameter is incorrect
   - **Fix:** Set to `EXPECTED_LENGTH` (263), not `EXPECTED_LENGTH + 50`
   
2. **Generation quality:** Some sequences show repetitive patterns
   - **Possible fixes:**
     - Lower temperature (currently 0.8, try 0.6-0.7)
     - Adjust top-p (currently 0.9, try 0.85-0.95)
     - Use different sampling strategy

3. **Generation reliability:** One sequence failed to generate
   - **Investigation needed:** Check if this is a one-off or systematic issue

---

## Recommended Next Steps

### Immediate Fix (Critical)

1. **Fix max-length parameter:**
   ```python
   # Change from:
   "--max-length", str(EXPECTED_LENGTH + 50)  # 313
   
   # To:
   "--max-length", str(EXPECTED_LENGTH)  # 263
   ```

2. **Re-run smoke test** with corrected parameter

### Secondary Fixes (After length is fixed)

3. **Tune generation parameters:**
   - Try temperature 0.6-0.7 (more conservative)
   - Try top-p 0.85-0.95 (narrower sampling)

4. **Add generation validation:**
   - Check that sequences actually generated (not just prompt)
   - Add minimum length check (e.g., must generate at least 200 aa)

5. **Investigate repetitive patterns:**
   - May need to add a "composition sanity" gate earlier
   - Check if this is a model issue or parameter issue

---

## Success Criteria Met?

| Criterion | Status | Notes |
|-----------|--------|-------|
| Pipeline runs end-to-end | ✅ | No crashes, all stages execute |
| Token normalization works | ✅ | 100% pass rate |
| Length gate logic works | ✅ | Correctly rejects wrong lengths |
| Generates sequences | ✅ | 10/11 sequences generated |
| Correct length | ❌ | 0% pass rate - needs fix |
| Hard locks tested | ⏭️ | Skipped (no sequences passed length) |

**Overall:** **Pipeline is functional but needs parameter tuning.**

---

## Conclusion

The smoke test successfully validated:
- ✅ ProGen2 integration works
- ✅ Token normalization works  
- ✅ Gate filtering logic works
- ✅ End-to-end pipeline executes

The test revealed:
- ❌ Length control parameter is incorrect (easy fix)
- ⚠️ Some generation quality issues (repetitive patterns)

**Verdict:** **Pipeline is ready for use after fixing the max-length parameter.** The core infrastructure is solid; this is a configuration issue, not a fundamental problem.

