# PETase Thermostability Sprint Plan  
_RFdiffusion + Rosetta/FoldX + AlphaFold, scaffolded on state-of-the-art PETase mutants_

---

## 0. High-level goal (what you’re trying to achieve in a few days)

Generate a **small, high-confidence panel of PETase variants** with:

- **Improved predicted thermostability** relative to a strong baseline (FAST-PETase)  
- **Preserved catalytic geometry** around the Ser160–Asp206–His237 triad :contentReference[oaicite:0]{index=0}  
- **Minimal structural deviation** from the baseline backbone

Using:

- **RFdiffusion** – constrained, fixed-backbone sequence redesign at selected hotspots  
- **Rosetta and/or FoldX** – ∆∆G stability scoring  
- **AlphaFold/ColabFold** – structure sanity + local RMSD checks

Deliverables after a few days:

1. A **ranked list of 5–20 PETase variants** with:
   - Mutation list vs FAST-PETase  
   - Predicted ∆∆G (stability)  
   - Optional interface/binding scores (if you add PET fragments later)  
   - AF-predicted structural RMSD in catalytic region
2. A **reproducible directory layout + script entry points**, so you (or someone else) can rerun or extend the pipeline.

---

## 1. Design choices (frozen decisions)

### 1.1 Scaffold choice

We **do not** start from wild-type PETase. Instead we build on **FAST-PETase**:

- FAST-PETase = WT PETase + mutations **S121E, D186H, R224Q, N233K, R280A**. :contentReference[oaicite:1]{index=1}  
- It shows **much higher activity and thermal tolerance** between 30–50 °C than WT and earlier engineered variants. :contentReference[oaicite:2]{index=2}  
- A crystal structure exists: **PDB 7SH6**. :contentReference[oaicite:3]{index=3}  

> **Decision:** Use **7SH6** as the baseline backbone and sequence for all design work.

You could swap to TS-PETase (S121E/D186H/N233C/R280A/S282C, Tm ≈ 69 °C) or DuraPETase (10 mutations, Tm ≈ 77 °C) later, but FAST-PETase offers a good **activity/thermostability balance** and is heavily analyzed in recent literature. :contentReference[oaicite:4]{index=4}  

---

### 1.2 Optimization objective

Single primary axis:

> **Maximize predicted thermostability** with minimal disruption of catalytic geometry and overall fold.

Operational proxies:

- More negative ∆∆G for folding (Rosetta/FoldX) vs FAST-PETase  
- Preserved:
  - Catalytic triad: **Ser160, Asp206, His237** :contentReference[oaicite:5]{index=5}  
  - Oxyanion hole backbone (e.g., Tyr87, Met161) :contentReference[oaicite:6]{index=6}  
- Local RMSD around catalytic residues < ~1 Å in AlphaFold predictions

---

### 1.3 RFdiffusion strategy

Given limited time + one GPU:

- **Core mode:**  
  - **Fixed backbone** (7SH6 coordinates)  
  - **Local sequence redesign** at curated hotspots only  
  - No large-scale backbone regeneration or de novo scaffolds

- **Two masks (two “risk tiers”):**
  1. **Conservative mask:** mutate loops and second-shell residues with strong literature support, but **keep FAST-PETase’s five key mutations fixed**.
  2. **Aggressive mask:** additionally allow RFdiffusion to modify some of those key positions (e.g., D186, N233) to explore alternatives like D186N/V or N233C/K that have shown large ∆Tm gains in other contexts. :contentReference[oaicite:7]{index=7}  

Sampling:

- ~**300 designs** per mask (≈ 600 total) is realistic for an overnight GPU run.

---

## 2. Literature-guided hotspot selection

### 2.1 Catalytic and structural “no-touch” core

These residues are treated as **frozen**:

- **Catalytic triad:** Ser160, Asp206, His237 :contentReference[oaicite:8]{index=8}  
- **Oxyanion hole:** Tyr87, backbone near Met161 :contentReference[oaicite:9]{index=9}  
- **Disulfides:** C203–C239 and C237–C289 (structural; leaving them alone avoids misfolding). :contentReference[oaicite:10]{index=10}  

You can also mark nearby backbone residues as “discouraged” for mutation if your RFdiffusion interface allows graded penalties.

---

### 2.2 Canonical thermostability clusters from literature

#### 2.2.1 ThermoPETase & TS-PETase cluster

- **ThermoPETase**: S121E/D186H/R280A  
  - Stabilizes the β6–β7 loop and improves activity; ∆Tm ≈ +8–9 °C vs WT. :contentReference[oaicite:11]{index=11}  
- **TS-PETase**: S121E/D186H/N233C/R280A/S282C  
  - Adds a **disulfide (N233C/S282C)**, pushing Tm to ≈ 69 °C. :contentReference[oaicite:12]{index=12}  

**Key residues:** S121, D186, N233, R280, S282.

#### 2.2.2 DuraPETase cluster (large ∆Tm, solvent tolerance)

DuraPETase has **10 mutations**:  
L117F, Q119Y, T140D, W159H, G165A, I168R, A180I, S188Q, S214H, R280A, giving ≈ +31 °C Tm vs WT. :contentReference[oaicite:13]{index=13}  

These cluster into:

- **Substrate-binding loops** and regions near subsite II  
- Surface/core positions modulating packing and hydrogen-bond networks

#### 2.2.3 FAST-PETase cluster

FAST-PETase = ThermoPETase plus **R224Q, N233K**. :contentReference[oaicite:14]{index=14}  

- Shows **dramatic activity** gains at 50 °C and high tolerance to pH variation. :contentReference[oaicite:15]{index=15}  
- Computational analysis (CNAnalysis + MD) indicates the 5 mutations reduce flexibility in **thermolabile sequence stretches** at higher temperature. :contentReference[oaicite:16]{index=16}  

**Key residues:** R224 and N233, in addition to S121, D186, R280.

#### 2.2.4 New loop mutations: QM-PETase-2

Recent work identified **N114I, N205K, N233K, S269V** as a quadruple mutant **QM-PETase-2**, with ∆Tm ≈ +12.4 °C and ~5-fold catalytic efficiency increase; all four positions are in loop regions. :contentReference[oaicite:17]{index=17}  

These loop mutations were successfully combined with multiple high-performance scaffolds, including FAST-PETase, suggesting they are **modular, transferable stabilizing motifs**. :contentReference[oaicite:18]{index=18}  

#### 2.2.5 Second-shell D186 variants

Focused mutagenesis on **D186** (a non-active-site second-shell residue) found:

- **D186N**: ∆Tm ≈ +8.9 °C  
- **D186V**: ∆Tm ≈ +12.9 °C  
with improved PET degradation performance vs WT. :contentReference[oaicite:19]{index=19}  

This reinforces **D186 as a critical thermostability knob** worth sampling.

---

### 2.3 Final hotspot sets for RFdiffusion

Numbering assumed to match FAST-PETase / WT IsPETase (PDB 7SH6 vs 6EQE/5XJH).

#### 2.3.1 Conservative mask (13 positions)

Keep FAST-PETase’s five signature mutations **fixed**:
- S121E, D186H, R224Q, N233K, R280A

Allow RFdiffusion sequence changes at:

| Position | WT → notable variants                 | Rationale |
|----------|---------------------------------------|-----------|
| N114     | N114I (QM-PETase-2) :contentReference[oaicite:20]{index=20} | Loop mutation; increases Tm and activity in multiple scaffolds |
| L117     | L117F (DuraPETase) :contentReference[oaicite:21]{index=21} | Hydrophobic packing near substrate-binding region |
| Q119     | Q119Y (DuraPETase) :contentReference[oaicite:22]{index=22} | Aromatic side chain improves packing / π interactions |
| T140     | T140D (DuraPETase, FAST-PETase follow-up) :contentReference[oaicite:23]{index=23} | Introduces stabilizing electrostatics in α3–β5 loop |
| W159     | W159H (DuraPETase) :contentReference[oaicite:24]{index=24} | Modulates local interactions near subsite II |
| G165     | G165A (DuraPETase) :contentReference[oaicite:25]{index=25} | Restricts flexibility; stabilizes local backbone |
| I168     | I168R (DuraPETase) :contentReference[oaicite:26]{index=26} | Introduces salt bridges / H-bonding on surface |
| A180     | A180I (DuraPETase) :contentReference[oaicite:27]{index=27} | Core hydrophobic packing |
| S188     | S188Q (DuraPETase) :contentReference[oaicite:28]{index=28} | Hydrogen-bond network near loop |
| N205     | N205K (QM-PETase-2) :contentReference[oaicite:29]{index=29} | Loop charge modulation in vicinity of Asp206 |
| S214     | S214H (DuraPETase) :contentReference[oaicite:30]{index=30} | Stabilizes surface loop; correlated with increased Tm |
| S269     | S269V (QM-PETase-2) :contentReference[oaicite:31]{index=31} | Loop mutation correlated with ∆Tm +12.4 °C when combined with N114I/N205K/N233K |
| S282     | S282C in TS-PETase :contentReference[oaicite:32]{index=32} | Forms disulfide with N233C; candidate for alternative disulfide engineering |

This conservative mask targets **loop/second-shell residues repeatedly implicated in thermostability**, while preserving the strongly validated FAST-PETase core.

#### 2.3.2 Aggressive mask (18–20 positions)

Starts from the conservative set and additionally allows mutation at:

- **S121** (currently E in FAST-PETase) – to explore S121D/N or revert in combination with other stabilizing clusters.
- **D186** (currently H in FAST-PETase) – to explore D186N/V/A, which have strong thermostabilizing evidence. :contentReference[oaicite:33]{index=33}  
- **R224** – explore Q/E/K; may affect surface electrostatics, binding and stability. :contentReference[oaicite:34]{index=34}  
- **N233** – allow K/C; both N233K (FAST-PETase/QM-PETase-2) and N233C (TS-PETase disulfide) are stabilizing in different contexts. :contentReference[oaicite:35]{index=35}  
- **R280** – R280A is known to relieve steric/electrostatic issues in the PET-binding pocket; small hydrophobics here are beneficial. :contentReference[oaicite:36]{index=36}  

You can decide whether to give these “aggressive” positions a **restricted alphabet** (e.g., only {H,N,V,A,D} at 186; {K,Q,E} at 224) or let RFdiffusion propose anything and rely on scoring filters.

---

## 3. Project layout (what Cursor should see)

Proposed repo structure:

```text
petase_thermostability/
  README.md                  # This document (or a refined version)
  env/                       # environment + dependency notes
  data/
    structures/
      FAST_PETase_7SH6_raw.pdb
      FAST_PETase_7SH6_clean.pdb
    sequences/
      FAST_PETase.fasta
  design/
    rfdiffusion/
      configs/
        conservative_mask.yaml
        aggressive_mask.yaml
      masks/
        conservative_positions.txt
        aggressive_positions.txt
      runs/
        conservative/
          samples/
        aggressive/
          samples/
  scoring/
    foldx/
      WT/                    # FAST-PETase baseline
      conservative/
      aggressive/
    rosetta/
      WT/
      conservative/
      aggressive/
  alphafold/
    configs/
    runs/
      conservative_top/
      aggressive_top/
  analysis/
    notebooks/
      01_score_summary.ipynb
      02_af_rmsd.ipynb
    results/
      ranked_variants.tsv
      top_variants_summary.md
