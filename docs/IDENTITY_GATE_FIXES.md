# Identity Gate Fixes - Ideal Solutions

## Problem Summary

Sequences are failing the identity gate because they have only **11-21% identity** to the baseline, well below the **55% minimum** threshold for the Explore bucket.

**Root Cause:** ProGen2-medium is generating repetitive, low-quality sequences that are genuinely out-of-family.

---

## ✅ Implemented Fixes (Applied)

### 1. **Use Only 80aa Prompts** ⭐ **HIGHEST IMPACT**
- **Change:** Default prompt lengths changed from `[20, 50, 80]` to `[80]`
- **Why:** Longer prompts provide more conditioning, keeping sequences closer to PETase family
- **Impact:** Should significantly improve identity scores
- **Trade-off:** Less diversity, but better quality

### 2. **Stricter Composition Gate**
- **Changes:**
  - Max single-AA run: `10 → 8` consecutive residues
  - Low-complexity threshold: `50% → 40%` (top 3 AAs)
- **Why:** Filters out more repetitive sequences before identity check
- **Impact:** Removes pathological sequences earlier

### 3. **More Conservative Sampling**
- **Changes:**
  - Conservative lane: temp `0.4 → 0.3`, top_p `0.95 → 0.90`
  - Exploratory lane: temp `0.8 → 0.6`
- **Why:** Lower temperature = more conservative, in-family sequences
- **Impact:** Should generate sequences closer to baseline

### 4. **Relaxed Identity Thresholds (Optional)**
- **New flag:** `--relax-identity`
- **Changes when enabled:**
  - Explore bucket: `55-75% → 30-50%`
  - Near bucket: `75-95% → 50-95%`
- **Why:** For testing/debugging when sequences are genuinely low-identity
- **Usage:** `python scripts/run_progen2_pipeline.py ... --relax-identity`

---

## 🎯 Recommended Usage

### For Production Runs (Best Quality)
```bash
# Use 80aa prompts (default now) + conservative sampling
python scripts/run_progen2_pipeline.py run_20251231_progen2_medium_r1_prod \
  --num-samples 100 \
  --model progen2-medium
```

### For Testing/Debugging (If Still Getting 0 Passes)
```bash
# Use relaxed identity thresholds
python scripts/run_progen2_pipeline.py run_20251231_progen2_medium_r1_test \
  --num-samples 100 \
  --model progen2-medium \
  --relax-identity
```

---

## 📊 Expected Improvements

With these fixes, you should see:

1. **Higher identity scores:** 80aa prompts + conservative sampling should push sequences from ~16% to 30-50%+ identity
2. **Better sequence quality:** Stricter composition gate removes more repetitive junk
3. **More candidates passing:** Relaxed thresholds (if used) allow testing with lower-quality sequences

---

## 🔄 Additional Options (If Needed)

### Option A: Use ProGen2-Large
- **Best quality** but **slowest** (may timeout on CPU)
- Should generate sequences with 40-60%+ identity
- Use if medium model still produces low-identity sequences

### Option B: Use Only Conservative Lane
- Skip Exploratory lane (remove diversity, focus on quality)
- Modify `SAMPLING_LANES` to only include Conservative
- Faster (half the combinations)

### Option C: Increase Sample Size
- Generate more sequences to find rare high-identity ones
- `--num-samples 200` or `300`
- Trade-off: longer runtime

### Option D: Fine-tune Identity Thresholds
- Manually adjust `IDENTITY_BUCKETS` in `filters.py`
- Current: Explore 55-75%, Near 75-95%
- Could try: Explore 40-60%, Near 60-95%

---

## 🧪 Testing Strategy

1. **First:** Run with default settings (80aa prompts, conservative sampling)
2. **If still 0 passes:** Use `--relax-identity` to see what identity scores you're getting
3. **If scores are 30-50%:** Consider permanently lowering thresholds
4. **If scores are still <30%:** Consider ProGen2-large or different approach

---

## 📝 Notes

- **80aa prompts** are now the default - this is the most impactful change
- **Composition gate** runs FIRST (before identity) to filter junk early
- **Identity thresholds** are strict by design - they ensure sequences are PETase-like
- **Relaxed thresholds** are for testing only - don't use in production

---

## Next Steps

1. Run a test with the new defaults (80aa prompts)
2. Check identity distribution in results
3. Adjust thresholds if needed based on actual identity scores
4. Consider ProGen2-large if medium still doesn't work

