# RFdiffusion Overnight Run Setup - Action Plan

**Goal:** Run ~600 RFdiffusion designs (300 conservative + 300 aggressive) overnight

**Timeline:** Complete setup in 1-2 hours, then start overnight run

---

## ✅ Prerequisites Checklist

- [ ] RFdiffusion Docker/Singularity environment built
- [ ] Model weights downloaded (~10GB)
- [ ] FAST-PETase structure (7SH6) downloaded and prepared
- [ ] RFdiffusion configs created for both masks
- [ ] Run scripts created and tested
- [ ] Small test run (5-10 designs) completed successfully

---

## Step 1: Download Model Weights (15-30 min)

**On macOS:**
```bash
mkdir -p data/models/rfdiffusion
bash envs/rfdiffusion/download_models_macos.sh data/models/rfdiffusion
```

**On Linux/HPC:**
```bash
mkdir -p data/models/rfdiffusion
bash external/rfdiffusion/scripts/download_models.sh data/models/rfdiffusion
```

**Expected:** 7 model files (~10GB total):
- Base_ckpt.pt
- Complex_base_ckpt.pt
- Complex_Fold_base_ckpt.pt
- InpaintSeq_ckpt.pt
- InpaintSeq_Fold_ckpt.pt
- ActiveSite_ckpt.pt
- Base_epoch8_ckpt.pt

---

## Step 2: Download and Prepare FAST-PETase Structure (5 min)

```bash
# Create directory
mkdir -p data/structures/7SH6/raw

# Download 7SH6 from PDB
cd data/structures/7SH6/raw
curl -O https://files.rcsb.org/view/7SH6.pdb

# Or use wget
# wget https://files.rcsb.org/view/7SH6.pdb

# Clean/prepare structure (extract chain A, remove waters/ligands if needed)
# You may need to manually edit or use a script to extract chain A
cd ../../../../..
```

**Note:** RFdiffusion needs a clean PDB with chain A only. You may need to:
- Extract chain A
- Remove waters, ligands, heteroatoms
- Ensure proper numbering

**Quick check:**
```bash
grep "^ATOM" data/structures/7SH6/raw/7SH6.pdb | head -20
```

---

## Step 3: Create RFdiffusion Configs (10 min)

### 3.1 Conservative Mask Config

**File:** `configs/rfdiffusion/conservative_mask.json`

Conservative mask allows mutation at 13 positions while keeping FAST-PETase's 5 key mutations fixed:
- Fixed: S121E, D186H, R224Q, N233K, R280A
- Mutable: N114, L117, Q119, T140, W159, G165, I168, A180, S188, N205, S214, S269, S282

**Strategy:** Use `inpaint_seq` to mask only the 13 mutable positions, keeping backbone fixed.

### 3.2 Aggressive Mask Config

**File:** `configs/rfdiffusion/aggressive_mask.json`

Aggressive mask allows mutation at 18-20 positions, including FAST-PETase's key positions:
- All conservative positions PLUS
- S121, D186, R224, N233, R280

---

## Step 4: Create Run Scripts (15 min)

### 4.1 Conservative Mask Run Script

**File:** `scripts/rfdiffusion_conservative.sh`

```bash
#!/bin/bash
# RFdiffusion Conservative Mask Run
# ~300 designs overnight

# Set paths
INPUT_PDB="data/structures/7SH6/raw/7SH6_chainA.pdb"
MODELS_DIR="data/models/rfdiffusion"
OUTPUT_DIR="runs/$(date +%Y-%m-%d)_rfdiffusion_conservative"
NUM_DESIGNS=300

# Conservative mask: 13 positions
# Using inpaint_seq to mask only mutable positions
# Format: 'contigmap.inpaint_seq=[A114/A117/A119/A140/A159/A165/A168/A180/A188/A205/A214/A269/A282]'

# Run via Docker or Singularity
if command -v docker &> /dev/null; then
    envs/rfdiffusion/run_docker.sh \
        inference.output_prefix=${OUTPUT_DIR}/designs \
        inference.model_directory_path=${MODELS_DIR} \
        inference.input_pdb=${INPUT_PDB} \
        inference.num_designs=${NUM_DESIGNS} \
        'contigmap.contigs=[A1-290]' \
        'contigmap.inpaint_seq=[A114/A117/A119/A140/A159/A165/A168/A180/A188/A205/A214/A269/A282]' \
        inference.ckpt_override_path=${MODELS_DIR}/ActiveSite_ckpt.pt
else
    envs/rfdiffusion/run_singularity.sh \
        inference.output_prefix=${OUTPUT_DIR}/designs \
        inference.model_directory_path=${MODELS_DIR} \
        inference.input_pdb=${INPUT_PDB} \
        inference.num_designs=${NUM_DESIGNS} \
        'contigmap.contigs=[A1-290]' \
        'contigmap.inpaint_seq=[A114/A117/A119/A140/A159/A165/A168/A180/A188/A205/A214/A269/A282]' \
        inference.ckpt_override_path=${MODELS_DIR}/ActiveSite_ckpt.pt
fi
```

### 4.2 Aggressive Mask Run Script

**File:** `scripts/rfdiffusion_aggressive.sh`

Similar to conservative but with more positions in `inpaint_seq`.

---

## Step 5: Test Run (30 min)

**CRITICAL:** Test with 5-10 designs before overnight run!

```bash
# Quick test - modify scripts to use num_designs=5
bash scripts/rfdiffusion_conservative.sh  # But first edit to use 5 designs

# Check outputs
ls runs/*_rfdiffusion_conservative/designs/
```

**Verify:**
- [ ] PDB files generated
- [ ] Sequences look reasonable
- [ ] No errors in logs
- [ ] Catalytic triad preserved (check manually)

---

## Step 6: Overnight Run Setup

### Option A: Local Docker (if on Linux with GPU)

```bash
# Run conservative mask
nohup bash scripts/rfdiffusion_conservative.sh > runs/rfdiffusion_conservative.log 2>&1 &

# Run aggressive mask (in parallel if you have 2 GPUs, or sequentially)
nohup bash scripts/rfdiffusion_aggressive.sh > runs/rfdiffusion_aggressive.log 2>&1 &
```

### Option B: HPC with Singularity

**SLURM script:** `cluster/rfdiffusion_overnight.sh`

```bash
#!/bin/bash
#SBATCH -J rfdiffusion_overnight
#SBATCH -o slurm-%j.out
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1

module load apptainer

# Run conservative mask
bash scripts/rfdiffusion_conservative.sh

# Run aggressive mask (or submit as separate job)
bash scripts/rfdiffusion_aggressive.sh
```

---

## Critical Notes

### RFdiffusion Strategy for Fixed Backbone

The plan calls for **fixed backbone** sequence redesign. RFdiffusion's `inpaint_seq` mode is designed for this:
- `contigmap.contigs=[A1-290]` - Keep entire backbone fixed
- `contigmap.inpaint_seq=[A114/A117/...]` - Mask only mutable positions
- RFdiffusion will redesign sequence at masked positions only

### Model Choice

- **ActiveSite_ckpt.pt** recommended for small motifs (catalytic triad)
- **Base_ckpt.pt** for general design
- Consider testing both

### Numbering

**CRITICAL:** Ensure PDB numbering matches your mask positions!
- FAST-PETase uses PDB numbering (1-290)
- Verify positions in your PDB match the mask

---

## Expected Outputs

After overnight run:
- `runs/YYYY-MM-DD_rfdiffusion_conservative/designs/` - 300 PDB files
- `runs/YYYY-MM-DD_rfdiffusion_aggressive/designs/` - 300 PDB files
- Each PDB has redesigned sequence at masked positions
- `.trb` files with metadata

---

## Next Steps (After Overnight Run)

1. **Extract sequences** from PDB files
2. **Filter** by basic criteria (length, catalytic triad preservation)
3. **Score** with Rosetta/FoldX (ΔΔG)
4. **Predict** with AlphaFold/ColabFold
5. **Rank** by stability + RMSD
6. **Select** top 5-20 candidates

---

## Troubleshooting

**"Model not found"**
- Check model weights path
- Verify all 7 models downloaded

**"PDB parsing error"**
- Clean PDB (chain A only, no waters)
- Check numbering matches mask positions

**"CUDA out of memory"**
- Reduce `num_designs` per batch
- Use smaller model if available

**"No designs generated"**
- Check contig format
- Verify inpaint_seq positions exist in PDB
- Test with simpler mask first

