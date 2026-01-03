# Relaxation Timing Estimates for ColabFold Structures

## Summary

**Total structures to relax:** 69 (68 candidates + WT)  
**Structure size:** ~263 residues (PETase)  
**Recommended method:** Rosetta cartesian relaxation

---

## Time Estimates Per Structure

### Option 1: Single Relaxed Structure (nstruct 1)
**Time per structure:** 5-15 minutes  
**Total for 69 structures:** 6-17 hours

**Command:**
```bash
# Modify rosetta_relax.sh to use -nstruct 1
$ROSETTA_BIN/relax.* -s input.pdb -nstruct 1 -relax:cartesian ...
```

**Pros:**
- Fastest option
- Good for screening many structures
- Sufficient for geometry optimization

**Cons:**
- Only one relaxed structure (no ensemble)
- May miss better local minima

---

### Option 2: Standard Relaxation (nstruct 20) - Current Default
**Time per structure:** 30 minutes - 2 hours  
**Total for 69 structures:** 35-138 hours (1.5-6 days)

**Command:**
```bash
# Current script default
bash scripts/rosetta_relax.sh input.pdb
# Uses -nstruct 20
```

**Pros:**
- Generates ensemble of 20 structures
- Better chance of finding optimal geometry
- Standard for publication-quality work

**Cons:**
- Much slower
- May be overkill for initial screening

---

### Option 3: Moderate Ensemble (nstruct 5)
**Time per structure:** 10-30 minutes  
**Total for 69 structures:** 12-35 hours (0.5-1.5 days)

**Command:**
```bash
# Modify to use -nstruct 5
$ROSETTA_BIN/relax.* -s input.pdb -nstruct 5 -relax:cartesian ...
```

**Pros:**
- Good balance of speed and quality
- Multiple structures for comparison
- Reasonable for screening

**Cons:**
- Still slower than nstruct 1
- Fewer structures than standard

---

## Recommended Strategy

### Phase 1: Quick Screening (Top Candidates Only)
**Relax top 10 candidates with nstruct 1:**
- **Time:** 50-150 minutes (1-2.5 hours)
- **Purpose:** Quick geometry check, identify which need more work

### Phase 2: Detailed Relaxation (Top 5-10)
**Relax top candidates with nstruct 5-20:**
- **Time:** 2.5-20 hours (depending on nstruct)
- **Purpose:** High-quality structures for analysis

### Phase 3: Full Relaxation (All Candidates - Optional)
**Only if needed, relax all 69 with nstruct 1:**
- **Time:** 6-17 hours
- **Purpose:** Complete dataset

---

## Parallelization Options

### Option A: Sequential (One at a Time)
**Time:** As listed above  
**Setup:** Run script in loop

```bash
for pdb in runs/colabfold_predictions_gpu/candidate_*_rank_001_*.pdb; do
    bash scripts/rosetta_relax.sh "$pdb"
done
```

### Option B: Parallel (Multiple Simultaneous)
**Time:** Divide by number of cores  
**Setup:** Use GNU parallel or background jobs

```bash
# Using GNU parallel (if installed)
parallel -j 4 bash scripts/rosetta_relax.sh ::: runs/colabfold_predictions_gpu/candidate_*_rank_001_*.pdb

# Or manually in background
for pdb in runs/colabfold_predictions_gpu/candidate_*_rank_001_*.pdb; do
    bash scripts/rosetta_relax.sh "$pdb" &
done
wait
```

**With 4 cores:** 1.5-4 hours (nstruct 1) or 9-35 hours (nstruct 20)

### Option C: Cluster/Cloud (Recommended for Full Dataset)
**Time:** Depends on cluster size  
**Setup:** Submit as array job

```bash
# Example SLURM array job
#SBATCH --array=1-69
#SBATCH --cpus-per-task=1

bash scripts/rosetta_relax.sh candidate_${SLURM_ARRAY_TASK_ID}_rank_001_*.pdb
```

**With 10 nodes:** ~1-3 hours (nstruct 1) or ~3.5-14 hours (nstruct 20)

---

## Time Breakdown by Method

| Method | nstruct | Time/Structure | Total (69) | Parallel (4 cores) |
|--------|---------|----------------|------------|-------------------|
| Quick | 1 | 5-15 min | 6-17 hours | 1.5-4 hours |
| Moderate | 5 | 10-30 min | 12-35 hours | 3-9 hours |
| Standard | 20 | 30-120 min | 35-138 hours | 9-35 hours |

---

## Factors Affecting Speed

1. **CPU speed:** Faster CPU = faster relaxation
2. **Structure quality:** Well-folded structures relax faster
3. **Number of structures (nstruct):** Linear scaling
4. **Cartesian vs. backbone:** Cartesian is slower but more accurate
5. **System load:** Other processes slow down relaxation

---

## Practical Recommendations

### For Your 69 Structures:

**Best approach:**
1. **Relax top 10 candidates first** (nstruct 1): ~1-2.5 hours
2. **Analyze catalytic triad geometry** after relaxation
3. **Relax top 5 with nstruct 5-20**: ~2.5-20 hours
4. **Relax remaining if needed**: Additional 5-15 hours

**Total time:** 8-38 hours (can be done in phases)

### Alternative: Use AMBER Relaxation (Faster)

If you have AMBER tools available:
- **Time per structure:** 2-5 minutes
- **Total for 69:** 2-6 hours
- **Quality:** Good for geometry optimization, but Rosetta is more thorough

**Note:** ColabFold's `--amber` flag attempted this but failed. You could try:
- Using AMBER tools directly
- Or using Rosetta (more reliable, but slower)

---

## Script to Relax All Candidates

I can create a batch script to relax all candidates. Would you like:
1. Quick version (nstruct 1) - fastest
2. Standard version (nstruct 20) - best quality
3. Configurable version - you choose nstruct

Let me know which you prefer!

