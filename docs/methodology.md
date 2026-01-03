# Methodology Overview

## Complete Workflow

This project implements a multi-stage computational pipeline for PETase enzyme optimization:

### Stage 1: Sequence Generation (ProGen2)

**Goal:** Generate diverse protein sequences that explore sequence space while maintaining catalytic function.

**Method:**
- Use ProGen2 language model to generate sequences
- Apply N-terminus anchor prompts at multiple lengths (100, 130, 150 aa)
- Use dual sampling lanes:
  - **Conservative**: temp=0.6, top_p=0.95 (quality-biased)
  - **Exploratory**: temp=0.9, top_p=0.85 (diversity-biased)

**Filtering Pipeline:**
- **C0**: Token cleanup
- **C1**: Length gate (263 aa)
- **C2**: Hard locks (catalytic triad: S131, D177, H208)
- **C3**: Identity buckets (Near 75-95%, Explore 55-75%)
- **C4**: Uniqueness (min 5 residue differences)
- **C5**: Composition sanity
- **C6**: Likelihood ranking (diversity-preserving selection)

**Output:** Ranked candidate sequences in FASTA format

**See:** `docs/PROGEN2_WORKFLOW.md` for detailed guide

---

### Stage 2: Structure Prediction (ColabFold)

**Goal:** Predict 3D structures for generated sequences to assess fold quality.

**Method:**
- Use ColabFold (AlphaFold2-based) for fast structure prediction
- Run with 2-3 recycles, 3-5 models per sequence
- Optional: AMBER relaxation for geometry refinement

**Metrics:**
- **pLDDT**: Per-residue confidence (0-100, higher = better)
- **pTM**: Predicted Template Modeling score (0-1, higher = better)
- **PAE**: Predicted Aligned Error matrix (inter-residue confidence)

**Output:** PDB structures, confidence scores, plots

**See:** `docs/COLABFOLD_GUIDE.md` and `docs/COLABFOLD_RESULTS_GUIDE.md`

**Note:** GPU-accelerated on RunPod cloud for faster predictions (~2-6 hours for 68 sequences vs. 50+ hours on CPU)

---

### Stage 3: Analysis & Ranking

**Goal:** Identify top candidates based on structure quality and catalytic geometry.

**Methods:**

1. **pLDDT Ranking**
   - Rank by average pLDDT score
   - Filter candidates with pLDDT > 90 (high confidence)

2. **Catalytic Triad Analysis**
   - Measure Ser131 ↔ His208 distance (~2.5-3.5 Å for H-bond)
   - Measure His208 ↔ Asp177 distance (~2.6-3.2 Å for functional triad)
   - Assess geometry angles
   - Flag non-functional geometries

**Output:** Ranked candidate list, geometry analysis CSV

**See:** `runs/colabfold_predictions_gpu/CANDIDATE_RANKING.md` and `CATALYTIC_TRIAD_ANALYSIS.md`

---

### Stage 4: Structure Refinement (Rosetta)

**Goal:** Optimize geometry of top candidates using physics-based energy minimization.

**Method:**
- Rosetta cartesian relaxation
- Energy minimization with ref2015_cart scoring function
- Generate multiple relaxed structures (typically 1-5 per candidate)
- Select best structure by lowest energy score

**Output:** Relaxed PDB structures with optimized geometry

**Time:** ~5-15 minutes per structure

**See:** `docs/WHAT_IS_RELAXATION.md`

---

### Stage 5: Stability Calculations (Rosetta ΔΔG)

**Goal:** Predict stability changes from mutations.

**Method:**
- Rosetta cartesian ΔΔG calculations
- Calculate energy before/after mutation
- Report ΔΔG = ΔG_mutant - ΔG_wildtype
- Negative ΔΔG = more stable (good!)

**Output:** ΔΔG values for each mutation

**Time:** ~1-3 hours depending on number of mutations

---

### Stage 6: Visualization & Selection

**Goal:** Visual inspection of top candidates.

**Method:**
- PyMOL visualization with catalytic triad highlighting
- Structural alignment to wild-type
- Inspection of active site geometry
- Assessment of surface properties

**Output:** PyMOL sessions, rendered images

**See:** `scripts/visualize_top6.pml` and `scripts/pymol_quick_commands.md`

---

## Current Status

✅ **Completed:**
- ProGen2 sequence generation pipeline
- ColabFold structure predictions (68 candidates)
- Candidate ranking by pLDDT
- Catalytic triad geometry analysis
- PyMOL visualization scripts

🔄 **In Progress:**
- Rosetta relaxation of top candidates
- Stability calculations (ΔΔG)

📋 **Next Steps:**
- Complete relaxation of top 10 candidates
- Calculate ΔΔG for key mutations
- Select top 5-10 designs for experimental validation

---

## Key Design Constraints

### Catalytic Triad (Must Maintain)

- **Ser131** (OG) - Nucleophile
- **Asp177** (OD1/OD2) - Base
- **His208** (NE2) - Acid

**Geometry Requirements:**
- Ser OG ↔ His NE2: ~2.5-3.5 Å (H-bond)
- His NE2 ↔ Asp OD: ~2.6-3.2 Å (functional triad)

### Sequence Constraints

- **Length**: 263 amino acids (must match wild-type)
- **Identity**: 55-95% to wild-type (depending on exploration strategy)
- **Hard locks**: Positions 131, 177, 208 must remain S, D, H

---

## Tools & Software

| Tool | Purpose | Status |
|------|---------|--------|
| **ProGen2** | Sequence generation | ✅ Active |
| **ColabFold** | Structure prediction | ✅ Active |
| **Rosetta** | Structure refinement, ΔΔG | ✅ Active |
| **PyMOL** | Visualization | ✅ Active |
| **FoldX** | Alternative stability prediction | Optional |
| **RFdiffusion** | Backbone generation | Optional (future) |

---

## References

- **ColabFold:** Mirdita et al. (2022) *Nature Methods*
- **AlphaFold2:** Jumper et al. (2021) *Nature*
- **ProGen2:** Nijkamp et al. (2023) *arXiv*
- **Rosetta:** Koehler Leman et al. (2020) *Nature Methods*

---

**Last Updated:** January 2025
