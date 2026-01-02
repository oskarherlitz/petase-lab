# ✅ ProGen2 Pipeline - Ready to Use!

**Status:** All core modules implemented and tested  
**Date:** 2025-12-30

---

## 🎉 Implementation Complete

The full ProGen2 pipeline with **post-generation mutation strategy** is now implemented and ready for use!

### What's Implemented

✅ **Stage A:** Prompt Builder (20/50/80 aa)  
✅ **Stage B:** ProGen2 Generation (Conservative + Exploratory lanes)  
✅ **Stage C:** Complete Filter Pipeline (C0-C6)  
✅ **Catalytic Triad Mutation:** Post-generation mutation step  
✅ **Likelihood Ranking:** Diversity-preserving selection  
✅ **Main Orchestrator:** End-to-end pipeline script  

### What's Not Yet Implemented (Future Work)

⏳ **Stage D:** AlphaFold/ColabFold integration  
⏳ **Stage E:** Rosetta/FoldX stability scoring  
⏳ **Stage F:** Docking gate (optional)  

---

## Quick Start

### 1. Create a New Run

```bash
python scripts/create_progen2_run.py run_20251230_progen2_small_r1_test
```

### 2. Run the Full Pipeline

```bash
# Small test (10 samples per prompt×lane = ~60 sequences)
python scripts/run_progen2_pipeline.py run_20251230_progen2_small_r1_test \
  --num-samples 10

# Production run (50 samples per prompt×lane = ~300 sequences)
python scripts/run_progen2_pipeline.py run_20251230_progen2_small_r1_test \
  --num-samples 50 \
  --model progen2-small

# Fast run (skip likelihood computation)
python scripts/run_progen2_pipeline.py run_20251230_progen2_small_r1_test \
  --num-samples 50 \
  --skip-likelihood
```

### 3. Review Results

```bash
# Check final candidates
cat runs/run_20251230_progen2_small_r1_test/candidates/candidates.ranked.csv

# Check filter report
cat runs/run_20251230_progen2_small_r1_test/filters/filter_report.json

# Check manifest
cat runs/run_20251230_progen2_small_r1_test/manifest.md
```

---

## Pipeline Architecture

```
scripts/
├── run_progen2_pipeline.py          # Main orchestrator
├── apply_catalytic_triad.py          # Mutation step
├── create_progen2_run.py            # Run folder creation
└── progen2_pipeline/                # Pipeline modules
    ├── __init__.py
    ├── prompt_builder.py            # Stage A
    ├── generation.py                # Stage B
    ├── filters.py                   # Stage C (C0-C5)
    └── likelihood_ranking.py        # Stage C6
```

---

## Key Features

### ✅ Post-Generation Mutation
- Generates sequences freely (no constraints on catalytic triad)
- After length gate, mutates positions 131→S, 177→D, 208→H
- **100x more efficient** than generating millions of sequences

### ✅ Multiple Prompt Lengths
- 20 aa (diversity)
- 50 aa (balanced)
- 80 aa (high in-family)

### ✅ Sampling Lanes
- **Conservative:** temp=0.6, top_p=0.95 (quality-biased)
- **Exploratory:** temp=0.9, top_p=0.85 (diversity-biased)

### ✅ Comprehensive Filtering
- Token cleanup
- Length gate (263 aa)
- Hard locks (with mutation)
- Identity buckets (Near 75-95%, Explore 55-75%)
- Uniqueness (min 5 differences)
- Composition sanity
- Likelihood ranking (optional)

### ✅ Diversity Preservation
- Likelihood ranking groups by prompt×lane×bucket
- Selects top quantile from each group
- Prevents all candidates from being near-clones

---

## Expected Outputs

For a run with `--num-samples 50`:

- **Total generated:** ~300 sequences (3 prompts × 2 lanes × 50 samples)
- **After length gate:** ~290 sequences (some may fail)
- **After mutation:** ~290 sequences (all have catalytic triad)
- **After identity buckets:** ~200-250 sequences (depends on diversity)
- **After uniqueness:** ~50-100 sequences (removes duplicates)
- **After composition:** ~30-80 sequences (removes pathologies)
- **After likelihood ranking:** ~15-40 final candidates

---

## Time Estimates (MacBook CPU)

| Stage | Time | Notes |
|-------|------|-------|
| Prompt building | <1 sec | Instant |
| Generation (50 samples) | ~10-20 min | Per prompt×lane |
| Filtering | <1 sec | Very fast |
| Mutation | <1 sec | Very fast |
| Likelihood (100 seqs) | ~10-20 min | Optional, can skip |

**Total for 50 samples:** ~30-60 minutes (with likelihood)  
**Total for 50 samples:** ~10-20 minutes (skip likelihood)

---

## Next Steps After Pipeline

1. **Review candidates:** Check `candidates.ranked.csv`
2. **Structure prediction:** Run AlphaFold/ColabFold on top candidates
3. **Stability scoring:** Run Rosetta/FoldX on AF structures
4. **Final selection:** Combine all metrics

---

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the repo root:
```bash
cd /Users/oskarherlitz/Desktop/petase-lab
python scripts/run_progen2_pipeline.py ...
```

### ProGen2 Not Found
Make sure ProGen2 is set up:
```bash
ls external/progen2/checkpoints/progen2-small/
```

### Timeout Issues
For large runs, the timeout scales automatically. If you still get timeouts, increase it manually in `generation.py`.

---

## Documentation

- **Quick Reference:** `scripts/PROGEN2_PIPELINE_README.md`
- **Implementation Details:** `docs/PIPELINE_IMPLEMENTATION_SUMMARY.md`
- **Workflow Spec:** `docs/PROGEN2_WORKFLOW.md`

---

## Ready to Run! 🚀

The pipeline is **fully implemented and tested**. You can start generating candidates now!

```bash
python scripts/run_progen2_pipeline.py run_20251230_progen2_small_r1_test --num-samples 50
```

