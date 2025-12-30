# Session summary: mutation candidates + PyMOL workflow (technical, no code details)

Date: 2025-12-23  
Repo: `petase-lab`

## Goal

- Identify the **top mutation candidates** from the current ddG results.
- Provide a **minimal, reliable PyMOL workflow** to visualize those candidates and overlay mutant vs WT structures.
- Propose a **Phase 2** mutation list that expands around the best non-catalytic hits, informed by literature and repo-specific numbering.

## What we inspected / relied on

- **Phase 1 ranking source**: `analysis/ddg/phase1.csv` (and `analysis/ddg/phase1.md`)
  - We used this because it already contains aggregated ddG statistics (mean/std across replicates) and tiers.
  - We noted that `analysis/ddg/ddg_cart_mutlist_2025-12-12_summary.csv` existed but was effectively empty (header only), so it was not usable as the primary ranking table.

- **Mutation list used in the ddG scan**: `configs/rosetta/mutlist.mut`
  - This file documents the key numbering convention used in the repo.

- **ddG run outputs containing structures**:
  - Latest: `runs/2025-12-14_ddg_cart_mutlist/outputs/`
  - Prior equivalent run: `runs/2025-12-13_ddg_cart_mutlist_172303/outputs/`
  - These contain **WT replicate structures** (`WT__bj1.pdb`, `WT__bj2.pdb`, `WT__bj3.pdb`) and **mutant replicate structures** (`MUT_<pos><AAA>_bj*.pdb`).

## Key decisions and why

### 1) Use the repo’s aggregated ddG summary to define “top candidates”

- We treated “top” as **most stabilizing** in the repo’s sign convention:
  - **More negative ΔΔG** → more stabilizing relative to WT.
- This matches the repo’s stated intent in docs and scripts (ranking ascending by ddG).

### 2) Use the repo’s numbering convention (POSE vs PDB) consistently

- The ddG mutation files and Rosetta outputs are in **Rosetta pose numbering**.
- In this repo, PETase pose numbering is mapped from PDB residue numbering using:
  - **pose_index = pdb_resnum − 29**
  - Example: PDB residue 238 corresponds to pose 209 (used for S238 ↔ S209 mapping).
- This matters for:
  - selecting residues in PyMOL (`resi` values),
  - interpreting `MUT_209LEU_*.pdb` filenames,
  - writing Phase 2 mutation lists.

### 3) Treat `bj1/bj2/bj3` as replicates, not separate “different mutants”

- The `bj1/bj2/bj3` structures are **independent stochastic samples** from Rosetta’s protocol (packing/minimization sampling).
- Decision: when visualizing, **pick one replicate** (often `bj2`) for clarity, and only compare all three if geometry looks suspicious.
- When ranking, rely on **mean + std** across replicates (where available).

### 4) Exclude catalytic-triad positions from “best mutation” interpretation

- Phase 1 contained strong stabilization scores at a catalytic residue position (e.g., **S131A**).
- Decision: **do not promote catalytic residue mutations** as “best candidates” for an active enzyme, even if ddG is favorable.
- Repo context indicates catalytic-triad positions (pose indices in this project):
  - pose **131** (PDB 160): catalytic Ser
  - pose **177** (PDB 206): catalytic Asp
  - pose **208** (PDB 237): catalytic His
- Rationale: stability scoring alone does not preserve function; mutating the catalytic triad is typically incompatible with activity.

### 5) Include literature-informed substitutions when building Phase 2

To expand around the best non-catalytic positions, we incorporated well-known PETase engineering motifs:

- **S238F** (Austin et al., 2018) is a canonical improved PETase mutation.
  - Under this repo’s mapping, **PDB S238F ↔ pose S209F**.
  - Decision: include **S209F** explicitly in Phase 2, even if not scanned (or if scanned under different conditions).

- Variants such as FAST-PETase include substitutions around **R224** (e.g., **R224Q**, **R224E**).
  - Under this repo’s mapping, **PDB R224 ↔ pose R195**.
  - Decision: include **R195Q** and **R195E** as plausible functional/stability variants to test, alongside the Phase 1 hit **R195F**.

## Results: “top” candidates identified from the current Phase 1 table

From `analysis/ddg/phase1.csv` (sorted by mean ΔΔG ascending):

- **S209L** — Tier A (most stabilizing in this table)
- **S131A** — Tier A (but excluded from functional candidate list; catalytic Ser position)
- **R195F** — Tier B (moderately stabilizing/near-neutral depending on thresholding)

After excluding catalytic positions, the **top non-catalytic positions** we focused on were:

- **Pose 209** (PDB 238): S → L was best in the scan; expand around this site.
- **Pose 195** (PDB 224): R → F was next-best non-catalytic hit; expand around this site.

## Practical PyMOL workflow decisions

### 1) Fixing file loading: relative vs absolute paths

- PyMOL failed to load `data/...` paths initially because PyMOL’s `pwd` was not the repo root.
- Decision: use either:
  - `cd /Users/oskarherlitz/Desktop/petase-lab` inside PyMOL, then load with relative paths; or
  - load with absolute paths.

### 2) Overlay mutant onto WT using matching replicate IDs

- Decision: overlay a mutant replicate against the **WT replicate with the same `bj` index**:
  - Example pairing: `WT__bj2.pdb` with `MUT_209LEU_bj2.pdb`.
- Rationale: both structures come from the same run context/replicate sampling.

### 3) “Which file is the mutation?”

- Mutation PDBs are stored directly under the ddG run outputs directory:
  - `runs/2025-12-14_ddg_cart_mutlist/outputs/`
    - WT: `WT__bj1.pdb`, `WT__bj2.pdb`, `WT__bj3.pdb`
    - Mutants: `MUT_121GLY_bj*.pdb`, `MUT_131ALA_bj*.pdb`, …, `MUT_209LEU_bj*.pdb`
    - Tables: `mutlist.ddg`, `mutlist.json` (and `.clean.*` counterparts)

## Phase 2 plan implemented (targeted expansion)

### What “Phase 2” means here

- Keep the protocol the same (Rosetta cartesian_ddg), but run a **new mutation list** that:
  - focuses only on positions that look promising,
  - adds 2–4 rational alternatives at those positions,
  - avoids catalytic-triad positions.

### Phase 2 mutation list produced

- Output file: `configs/rosetta/mutlist_phase2.mut`
- Contents (high level):
  - At **pose 209 / PDB 238**: try S→L plus nearby hydrophobic/aromatic alternatives, including **S→F** (literature).
  - At **pose 195 / PDB 224**: try R→F plus literature- and chemistry-motivated alternatives (e.g., **R→Q**, **R→E**).

### How Phase 2 is intended to be run

- Use the existing ddG pipeline, pointing it at the new mutation list:
  - Input structure example: `runs/2025-12-05_relax_cart_v1/outputs/PETase_raw_0008.pdb`
  - Mutation list: `configs/rosetta/mutlist_phase2.mut`

## Notes / caveats

- **Catalytic constraints file** (`configs/rosetta/catalytic.cst`) currently appears to be a stub/template, so we did not treat it as an authoritative source of catalytic residue identities; instead we used the repo’s established pose/PDB mapping and known PETase catalytic triad positions.
- ddG alone does not guarantee improved activity; Phase 2 expansions were chosen to balance **stability signals** with **known functional engineering precedents** (especially at the S238/209 hotspot).

