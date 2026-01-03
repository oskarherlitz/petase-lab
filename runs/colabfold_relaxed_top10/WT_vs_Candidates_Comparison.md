# WT vs Candidates: Relaxation Score Comparison

## Summary

**Wild-Type (WT) Relaxed Score:** -887.529 REU

**Best Candidate Score:** -867.984 REU (candidate_9)

**Result:** ❌ **None of the candidates have better scores than WT**

---

## Detailed Comparison

| Rank | Candidate | Score (REU) | vs WT (Δ) | Status |
|------|-----------|-------------|-----------|--------|
| **WT** | **PETase_raw** | **-887.529** | **0.0** | **Baseline** |
| 1 | candidate_9 | -867.984 | +19.545 | ⚠️ Less stable |
| 2 | candidate_8 | -849.510 | +38.019 | ⚠️ Less stable |
| 3 | candidate_4 | -848.714 | +38.815 | ⚠️ Less stable |
| 4 | candidate_6 | -846.138 | +41.391 | ⚠️ Less stable |
| 5 | candidate_60 | -841.366 | +46.163 | ⚠️ Less stable |
| 6 | candidate_28 | -839.957 | +47.572 | ⚠️ Less stable |
| 7 | candidate_25 | -838.038 | +49.491 | ⚠️ Less stable |
| 8 | candidate_56 | -833.971 | +53.558 | ⚠️ Less stable |
| 9 | candidate_21 | -824.709 | +62.820 | ⚠️ Less stable |
| 10 | candidate_66 | -768.266 | +119.263 | ⚠️ Less stable |

**Note:** In Rosetta scoring, **lower (more negative) = better**. Positive Δ means less stable than WT.

---

## Interpretation

### Why Are Candidates Less Stable?

1. **Different Starting Structures**
   - WT: Relaxed from crystal structure (experimentally determined)
   - Candidates: Relaxed from ColabFold predictions (computational models)

2. **Sequence Differences**
   - Candidates have mutations that may affect stability
   - Some mutations may be destabilizing but improve function

3. **Structure Quality**
   - ColabFold predictions may have minor structural issues
   - Relaxation can't fully correct all prediction errors

### Is This a Problem?

**Not necessarily!** Here's why:

1. **Small Differences**
   - The best candidate (candidate_9) is only ~20 REU worse
   - This is a relatively small energy difference
   - Many functional proteins tolerate small stability losses

2. **Function vs Stability Trade-off**
   - Candidates may have **better catalytic activity** despite lower stability
   - Improved activity can compensate for slight stability loss
   - This is common in enzyme engineering

3. **Experimental Validation Needed**
   - Computational scores are predictions
   - Real stability depends on many factors (temperature, pH, etc.)
   - Activity assays will show if candidates are actually better

---

## Recommendations

1. **Focus on Top Candidates**
   - candidate_9, candidate_8, candidate_4, candidate_6 are closest to WT
   - These have the best stability scores

2. **Consider Activity**
   - Lower stability might be acceptable if activity is improved
   - Need to test catalytic activity experimentally

3. **Further Optimization**
   - Could try additional mutations to improve stability
   - Could use Rosetta FastDesign to optimize stability while maintaining function

4. **Experimental Testing**
   - Test top 3-5 candidates experimentally
   - Measure both stability (Tm, half-life) and activity (kcat, Km)

---

## Next Steps

1. **Calculate ΔΔG** for specific mutations to understand which changes are destabilizing
2. **Analyze catalytic triad geometry** - ensure function is maintained
3. **Select top 3-5 candidates** for experimental validation
4. **Consider stability optimization** if needed (FastDesign, additional mutations)

---

**Last Updated:** January 2025

