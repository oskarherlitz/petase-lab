# ProGen2 Pipeline - Quick Reference

## Overview

Complete pipeline for ProGen2 sequence generation with post-generation mutation strategy.

## Quick Start

```bash
# 1. Create a new run
python scripts/create_progen2_run.py run_20251230_progen2_small_r1_test

# 2. Run the full pipeline
python scripts/run_progen2_pipeline.py run_20251230_progen2_small_r1_test \
  --num-samples 50 \
  --model progen2-small
```

## Pipeline Stages

### Stage A: Prompt Building
- Builds N-terminus anchor prompts at 20/50/80 aa
- Saves prompts and manifest

### Stage B: Generation
- Generates sequences using Conservative + Exploratory lanes
- Multiple prompt lengths × lanes
- Saves raw and normalized sequences

### Stage C: Filtering
- C0: Token cleanup
- C1: Length gate (263 aa)
- C2: Hard locks (S131, D177, H208) - **applies mutations here**
- C3: Identity buckets (Near 75-95%, Explore 55-75%)
- C4: Uniqueness (min 5 residue differences)
- C5: Composition sanity
- C6: Likelihood ranking (optional)

### Stage D: Structure Gate (TODO)
- AlphaFold/ColabFold prediction
- Confidence metrics (pLDDT, PAE)

### Stage E: Stability Gate (TODO)
- Rosetta/FoldX scoring
- Stability ranking

## Key Features

### Post-Generation Mutation Strategy
- Sequences are generated freely
- After length gate, catalytic triad is applied (131→S, 177→D, 208→H)
- Much more efficient than generating millions of sequences

### Sampling Lanes
- **Conservative**: temp=0.6, top_p=0.95 (quality-biased)
- **Exploratory**: temp=0.9, top_p=0.85 (diversity-biased)

### Diversity Preservation
- Likelihood ranking preserves diversity across prompt×lane×bucket groups
- Selects top quantile from each group

## Output Structure

```
runs/run_YYYYMMDD_progen2_<model>_r<round>_<tag>/
├── prompts/
│   ├── prompt_20aa.txt
│   ├── prompt_50aa.txt
│   ├── prompt_80aa.txt
│   └── prompt_manifest.json
├── generated/
│   ├── raw_generations.fasta
│   └── normalized.fasta
├── filters/
│   └── filter_report.json
├── candidates/
│   ├── candidates.length_pass.fasta  (before mutation)
│   ├── candidates.filtered.fasta     (after mutation)
│   ├── candidates.ranked.fasta       (after likelihood ranking)
│   └── candidates.ranked.csv         (ranking table)
├── manifest.json
└── manifest.md
```

## Command-Line Options

```bash
python scripts/run_progen2_pipeline.py <run_id> [options]

Required:
  run_id              Run ID (e.g., run_20251230_progen2_small_r1_test)

Options:
  --baseline-fasta    Path to baseline FASTA (default: data/sequences/wt/baseline_5XJH_30-292.fasta)
  --model             ProGen2 model (default: progen2-small)
  --num-samples       Samples per (prompt, lane) (default: 50)
  --max-length        Max sequence length (default: 264 to get 263 aa)
  --rng-seed          Random seed (default: 42)
  --device            Device: cpu, cuda, mps (default: cpu)
  --skip-likelihood   Skip likelihood computation (faster)
  --top-quantile      Top quantile to keep per group (default: 0.5)
```

## Example Workflow

```bash
# Small test run
python scripts/run_progen2_pipeline.py run_test --num-samples 10

# Production run
python scripts/run_progen2_pipeline.py run_prod --num-samples 100 --model progen2-base

# Fast run (skip likelihood)
python scripts/run_progen2_pipeline.py run_fast --num-samples 50 --skip-likelihood
```

## Module Structure

```
scripts/progen2_pipeline/
├── __init__.py
├── prompt_builder.py    # Stage A: Build prompts
├── generation.py         # Stage B: Generate sequences
├── filters.py            # Stage C: Filter sequences
└── likelihood_ranking.py # Stage C6: Rank by likelihood
```

Each module can be run independently or as part of the full pipeline.

## Next Steps After Pipeline

1. **Review candidates**: Check `candidates/candidates.ranked.csv`
2. **Structure prediction**: Run AlphaFold/ColabFold on top candidates
3. **Stability scoring**: Run Rosetta/FoldX on AF structures
4. **Final selection**: Combine all metrics for final ranking

