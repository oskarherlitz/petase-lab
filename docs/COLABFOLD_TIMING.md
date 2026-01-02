# ColabFold Timing Estimates

## For 68 Sequences (~263 amino acids each)

### Web Interface (https://colabfold.com)

**Time per sequence:** 15-30 minutes (medium length sequences)

**Total time:** 
- **17-34 hours** if processing one at a time
- **Manual work required:** You'd need to upload each sequence individually
- **Not recommended** for 68 sequences

### Local Installation (Batch Processing)

**Time per sequence:** 15-30 minutes on CPU, 5-15 minutes with GPU

**Total time (CPU only - typical Mac):**
- **17-34 hours** total runtime
- Can run unattended (batch processing)
- Your Mac will be busy during this time

**Total time (with GPU - if available):**
- **6-17 hours** total runtime
- Much faster, but requires GPU support

### Breakdown by Sequence Length

| Sequence Length | Time per Sequence | 68 Sequences |
|----------------|-------------------|---------------|
| <200 aa (short) | 5-15 minutes | 6-17 hours |
| 200-400 aa (medium) | 15-30 minutes | **17-34 hours** |
| >400 aa (long) | 30-60 minutes | 34-68 hours |

**Your sequences:** ~263 aa = **medium category**

---

## Recommendations for 68 Sequences

### Option 1: Process in Batches (Recommended)

**Process top 10-20 first:**
```bash
# Extract top 10 sequences
head -20 runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta > top10.fasta

# Process them
bash scripts/colabfold_predict.sh top10.fasta
# Time: ~2.5-5 hours
```

**Then process more as needed:**
- Review results from first batch
- Select best candidates
- Process additional sequences if needed

### Option 2: Run Overnight/Weekend

**Full batch (68 sequences):**
```bash
# Start before leaving (e.g., Friday evening)
bash scripts/colabfold_predict.sh \
  runs/run_20251231.2_progen2_medium_r1_test/candidates/candidates.ranked.fasta \
  runs/colabfold_predictions
```

**Let it run:**
- **17-34 hours** total
- Can check progress periodically
- Results will be ready when complete

### Option 3: Use Cloud/Cluster (If Available)

If you have access to:
- **Google Colab Pro** (faster, GPU access)
- **Cloud computing** (AWS, GCP, Azure)
- **HPC cluster** (university/supercomputer)

These can significantly speed up processing.

---

## Mac-Specific Considerations

### CPU-Only (Most Macs)

- **M1/M2/M3 Macs:** Good performance, but still CPU-only for ColabFold
- **Intel Macs:** Slower, may take longer
- **No GPU acceleration** (ColabFold on Mac typically uses CPU)

**Expected time:** 20-30 minutes per sequence = **23-34 hours for 68 sequences**

### With GPU (Rare on Mac)

- **External GPU (eGPU):** Possible but complex setup
- **Cloud GPU:** Better option (Google Colab Pro, etc.)

**Expected time:** 10-15 minutes per sequence = **11-17 hours for 68 sequences**

---

## Optimization Tips

### 1. Reduce Number of Models

Default: 5 models per sequence
Faster: 1-2 models per sequence

```bash
# Edit configs/colabfold.yaml or use command line
colabfold_batch --num-models 2 --num-recycles 3 your_sequences.fasta output_dir
```

**Time savings:** ~40% faster (10-18 hours instead of 17-34 hours)

### 2. Reduce Recycles

Default: 3 recycles
Faster: 1-2 recycles (slightly less accurate)

**Time savings:** ~20-30% faster

### 3. Process in Priority Order

```bash
# Process top candidates first
head -20 candidates.ranked.fasta > top20.fasta
bash scripts/colabfold_predict.sh top20.fasta

# Review results, then process more if needed
```

### 4. Use Web Interface for Quick Tests

- Test 1-2 sequences on web interface first
- Verify results look good
- Then run full batch locally

---

## Realistic Timeline

### Conservative Estimate (CPU-only Mac)

- **Per sequence:** 25 minutes average
- **68 sequences:** 28.3 hours (~1.2 days)
- **With breaks/checks:** ~1.5-2 days total

### Optimistic Estimate (with optimizations)

- **Per sequence:** 15 minutes (2 models, 2 recycles)
- **68 sequences:** 17 hours (~0.7 days)
- **With breaks/checks:** ~1 day total

### Best Case (GPU available)

- **Per sequence:** 10 minutes
- **68 sequences:** 11.3 hours (~0.5 days)
- **With breaks/checks:** ~12-15 hours total

---

## What to Expect

### During Processing

1. **First run:** Downloads databases (~10GB, 30-60 minutes, one-time)
2. **Each sequence:**
   - MMseqs2 search: 2-5 minutes
   - Structure prediction: 10-25 minutes
   - AMBER relaxation (if enabled): 2-5 minutes
3. **Progress:** ColabFold shows progress for each sequence

### Resource Usage

- **CPU:** High usage (80-100%) during prediction
- **Memory:** 4-8 GB RAM per sequence
- **Disk:** ~50-100 MB per sequence output
- **Network:** Only for initial database download

---

## Practical Recommendation

**For 68 sequences on a Mac:**

1. **Start with top 10-20 sequences** (2.5-5 hours)
   ```bash
   head -20 candidates.ranked.fasta > top20.fasta
   bash scripts/colabfold_predict.sh top20.fasta
   ```

2. **Review results** (check structures, pLDDT scores)

3. **If results look good, process the rest:**
   ```bash
   # Run overnight/weekend
   bash scripts/colabfold_predict.sh candidates.ranked.fasta
   ```

4. **Total time:** 
   - Initial batch: 2.5-5 hours
   - Full batch: Additional 15-29 hours
   - **Total: ~1-1.5 days** (can be unattended)

---

## Alternative: Use Web Interface Selectively

If you don't want to tie up your Mac:

1. **Process top 5-10 on web interface** (manual, but free)
2. **Use local batch for the rest** (overnight)

This gives you quick results for top candidates while processing the rest in background.

---

## Summary

**68 sequences × ~263 aa each on Mac (CPU-only):**

- **Best case:** 17 hours (with optimizations)
- **Typical:** 23-30 hours
- **Worst case:** 34 hours

**Recommendation:** 
- Process top 10-20 first (2.5-5 hours)
- Then run full batch overnight/weekend
- Or use cloud/GPU resources if available

