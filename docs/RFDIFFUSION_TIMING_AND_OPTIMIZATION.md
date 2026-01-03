# RFdiffusion Validation Timing & Optimization Guide

## Time Estimates

### 1. AlphaFold Validation (10-20 designs)

**Per design:**
- **ColabFold (GPU):** ~2-5 minutes per design
- **ColabFold (CPU):** ~30-60 minutes per design
- **Full AlphaFold (GPU):** ~10-20 minutes per design
- **Full AlphaFold (CPU):** ~2-4 hours per design

**Total time:**
- **10 designs on GPU:** ~20-50 minutes (ColabFold) or ~2-3 hours (Full AlphaFold)
- **20 designs on GPU:** ~40-100 minutes (ColabFold) or ~4-6 hours (Full AlphaFold)
- **10 designs on CPU:** ~5-10 hours (ColabFold) or ~20-40 hours (Full AlphaFold) ⚠️

**Recommendation:** Use ColabFold on GPU for speed.

### 2. Rosetta ΔΔG Scoring (300 designs)

**Per design:**
- **cartesian_ddg (CPU):** ~10-30 minutes per design
- **FastRelax + scoring (CPU):** ~5-15 minutes per design
- **Parallel (8 cores):** ~1-4 minutes per design

**Total time:**
- **Sequential (1 core):** ~50-150 hours (2-6 days) ⚠️
- **Parallel (8 cores):** ~6-20 hours
- **Parallel (16 cores):** ~3-10 hours
- **Cluster (100+ cores):** ~1-3 hours

**Recommendation:** Use parallel processing or cluster.

### 3. FoldX Stability Scoring (300 designs)

**Per design:**
- **FoldX (CPU):** ~1-5 minutes per design
- **Parallel (8 cores):** ~10-30 seconds per design

**Total time:**
- **Sequential (1 core):** ~5-25 hours
- **Parallel (8 cores):** ~1-3 hours
- **Parallel (16 cores):** ~30-90 minutes

**Recommendation:** FoldX is faster than Rosetta - use for initial screening.

## Optimization Strategies

### Strategy 1: Two-Stage Approach (Recommended)

**Stage 1: Quick Screening (Fast)**
1. **FoldX on all 300 designs** → ~1-3 hours (parallel)
2. **Rank by FoldX ΔΔG** → Select top 50-100
3. **AlphaFold on top 50-100** → ~2-8 hours (GPU, ColabFold)

**Stage 2: Detailed Validation (Slower)**
4. **Rosetta ΔΔG on top 20-50** → ~3-25 hours (parallel)
5. **Full AlphaFold on top 10-20** → ~2-6 hours (GPU, Full AlphaFold)

**Total time:** ~8-42 hours (1-2 days) vs. weeks for full validation

### Strategy 2: Parallel Processing

**For Rosetta:**
```bash
# Run 8 designs in parallel
parallel -j 8 bash scripts/rosetta_ddg.sh ::: runs/2026-01-03_rfdiffusion_conservative/designs_{0..299}.pdb
```

**For FoldX:**
```bash
# Run 16 designs in parallel
parallel -j 16 python scripts/foldx_stability.py ::: runs/2026-01-03_rfdiffusion_conservative/designs_{0..299}.pdb
```

**For AlphaFold/ColabFold:**
```bash
# Batch process with ColabFold
colabfold_batch --num-recycle 3 --num-models 1 --templates --use-gpu-relax \
  runs/2026-01-03_rfdiffusion_conservative/designs_*.pdb \
  runs/alphafold_validation/
```

### Strategy 3: Use GPU for AlphaFold

**ColabFold (Fastest):**
- Use ColabFold instead of full AlphaFold
- ~5x faster than full AlphaFold
- Good enough for validation
- Can run on RunPod GPU

**Setup:**
```bash
# Install ColabFold
pip install colabfold[alphafold]

# Run batch prediction
colabfold_batch --num-recycle 3 --num-models 1 --templates \
  --use-gpu-relax runs/2026-01-03_rfdiffusion_conservative/designs_*.pdb \
  runs/alphafold_validation/
```

### Strategy 4: Sampling Strategy

**Don't validate all 300 designs!**

1. **Random sample:** 50 designs (~17%)
2. **Or stratified sample:** Top/bottom/middle by visual inspection
3. **Or smart sample:** Based on sequence diversity

**Time savings:** 5-6x faster (50 designs vs. 300)

### Strategy 5: Use Faster Scoring Methods

**FoldX vs. Rosetta:**
- FoldX: ~1-5 min/design (fast, less accurate)
- Rosetta: ~10-30 min/design (slow, more accurate)

**Recommendation:** Use FoldX for initial screening, Rosetta for top candidates.

**FastRelax + scoring:**
- Faster than full cartesian_ddg
- Good enough for ranking
- ~5-15 min/design vs. ~10-30 min/design

## Recommended Workflow (Optimized)

### Phase 1: Quick Screening (Day 1)

**Morning (2-3 hours):**
1. Visual inspection: 10-20 designs in PyMOL
2. Extract sequences: Check diversity
3. FoldX on all 300 designs (parallel, 1-3 hours)

**Afternoon (2-4 hours):**
4. Rank by FoldX ΔΔG
5. Select top 50-100 designs
6. AlphaFold (ColabFold) on top 50-100 (GPU, 2-4 hours)

**Evening:**
7. Review AlphaFold results
8. Select top 20-30 candidates

### Phase 2: Detailed Validation (Day 2)

**Morning (3-6 hours):**
9. Rosetta ΔΔG on top 20-30 (parallel, 3-6 hours)
10. Full AlphaFold on top 10-20 (GPU, 2-4 hours)

**Afternoon:**
11. Final ranking and selection
12. Prepare top 5-10 candidates for experimental validation

**Total time:** ~2 days (vs. weeks for full validation)

## Hardware Recommendations

### Minimum (Slow but works):
- **CPU:** 8+ cores
- **RAM:** 16GB+
- **Time:** ~1-2 weeks for full validation

### Recommended (Fast):
- **CPU:** 16+ cores (for parallel Rosetta)
- **GPU:** RTX 3090/4090 or better (for AlphaFold)
- **RAM:** 32GB+
- **Time:** ~1-2 days for optimized workflow

### Optimal (Very Fast):
- **CPU:** 32+ cores or cluster
- **GPU:** A100 or multiple GPUs
- **RAM:** 64GB+
- **Time:** ~6-12 hours for optimized workflow

## Scripts for Optimization

### Batch Rosetta Scoring
```bash
# scripts/batch_rosetta_ddg.sh
#!/bin/bash
RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
NUM_JOBS="${2:-8}"

find "${RESULTS_DIR}" -name "designs_*.pdb" | \
  parallel -j "${NUM_JOBS}" bash scripts/rosetta_ddg.sh {}
```

### Batch FoldX Scoring
```bash
# scripts/batch_foldx_stability.sh
#!/bin/bash
RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
NUM_JOBS="${2:-16}"

find "${RESULTS_DIR}" -name "designs_*.pdb" | \
  parallel -j "${NUM_JOBS}" python scripts/foldx_stability.py {}
```

### Batch ColabFold
```bash
# scripts/batch_colabfold.sh
#!/bin/bash
RESULTS_DIR="${1:-runs/2026-01-03_rfdiffusion_conservative}"
OUTPUT_DIR="${2:-runs/alphafold_validation}"

colabfold_batch \
  --num-recycle 3 \
  --num-models 1 \
  --templates \
  --use-gpu-relax \
  "${RESULTS_DIR}"/designs_*.pdb \
  "${OUTPUT_DIR}"
```

## Cost-Benefit Analysis

### Full Validation (All 300 designs):
- **Time:** 2-6 weeks
- **Cost:** High (compute time)
- **Benefit:** Complete dataset
- **Recommendation:** ❌ Not worth it

### Optimized Workflow (Top 50-100):
- **Time:** 1-2 days
- **Cost:** Low (targeted compute)
- **Benefit:** Top candidates identified
- **Recommendation:** ✅ Best approach

### Minimal Validation (Top 20):
- **Time:** 6-12 hours
- **Cost:** Very low
- **Benefit:** Quick screening
- **Recommendation:** ✅ Good for initial pass

## Summary

**Fastest path to results:**
1. FoldX on all 300 (1-3 hours, parallel)
2. AlphaFold on top 50 (2-4 hours, GPU, ColabFold)
3. Rosetta on top 20 (3-6 hours, parallel)
4. **Total: ~1 day**

**Most thorough (still optimized):**
1. FoldX on all 300 (1-3 hours)
2. AlphaFold on top 100 (4-8 hours, GPU)
3. Rosetta on top 50 (6-12 hours, parallel)
4. **Total: ~2 days**

**Key takeaway:** Don't validate everything! Use two-stage screening (FoldX → AlphaFold → Rosetta) to focus compute on top candidates.

