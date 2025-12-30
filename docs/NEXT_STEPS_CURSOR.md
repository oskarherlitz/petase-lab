# PETase Optimization – Next Steps Playbook (for Cursor)

This document is your task list + “prompts for Cursor” guide for the next phases of your PETase computational design project.  
You can paste sections into Cursor and use the embedded prompts to generate scripts, analysis code, or automation.

---

## 0. Context & Goals

**Current status**

- You have:
  - A WT PETase structure from the PDB.
  - A Rosetta **relax** step implemented (multiple relaxed models like `PETase_raw_0001.pdb` … `PETase_raw_0020.pdb`).
  - A Rosetta **ddG** scan implemented using a curated mutation list (`mutlist.mut`).
  - Outputs: `mutlist.ddg`, `mutlist.json`, `WT__bj*.pdb`, `MUT_*.pdb`.

**High-level goals**

1. Turn current ddG outputs into a **clear ranking of mutations** and structural sanity checks.
2. Use those insights to design a **Phase 2 mutation set** and run another ddG scan.
3. Incorporate **ProGen2 sequence generation** for broader sequence exploration.
4. Run **AlphaFold/ColabFold** on selected variants.
5. Run **relax and stability scoring** on AlphaFold structures.
6. Maintain strong **directory organization** and **reproducibility logs**.

---

## 1. Repository Organization & Logging

**Goal:** Make the repo clean and predictable so each protocol has clear inputs/outputs and is easy to reproduce.

### Tasks

- Set up a consistent directory structure (adapt names as needed):

  - `data/`
    - `structures/`
      - `5XJH/`
        - `raw/` (downloaded PDB)
        - `relax/` (all relaxed WT models + score.sc)
        - `ddg_phase1/`
          - `inputs/` (mutation lists, configs)
          - `outputs/` (ddG results, mutant PDBs)
    - `sequences/`
      - `wt/`
      - `progen2_candidates/`
      - `selected_variants/`

  - `configs/`
    - `rosetta/` (relax flags, ddG flags, mutation lists, enzdes XML, etc.)
    - `foldx/`
    - `alphafold/`

  - `analysis/`
    - `ddg/`
    - `structures/`
    - `ml/` (if used later)

  - `scripts/`
    - One script per protocol type (relax, ddG, analysis helpers, etc.).

- Create a top-level `PROJECT_LOG.md` describing:
  - Date, tool versions.
  - Which script was run.
  - Inputs and outputs for each run.
  - Short conclusions (e.g., “ddG_phase1 suggests mutations X, Y are stabilizing”).

### Suggested Cursor prompt

> You are helping me organize a computational protein design repo for PETase optimization.  
> I want a **proposed directory tree** (no actual code) and a short explanation of what goes into each folder.  
> Constraints:  
> - Tools used: Rosetta, FoldX, AlphaFold/ColabFold, RFdiffusion (later), ProGen2.  
> - I want `data/`, `configs/`, `analysis/`, and `scripts/` at minimum.  
> - Under `data/structures/`, include subfolders for the WT PDB, relaxed models, ddG outputs, and future designed variants.  
> Give me the tree and brief descriptions for each folder, suitable for adding to a README.

(You’ve mostly designed this already; you can use Cursor to refine and keep it synced with the actual repo.)

---

## 2. Analyze Existing ddG Results (Phase 1)

**Goal:** Convert `mutlist.ddg` and `mutlist.json` into a ranked table of mutations with statistics and structural sanity categories.

### Tasks

- Parse `mutlist.json` to extract:
  - Each mutation (e.g., D121G).
  - Replicate ddG values (bj1, bj2, bj3, etc.).
- For each mutation:
  - Calculate:
    - Mean ΔΔG.
    - Standard deviation.
  - Decide sign convention:
    - Negative ΔΔG = stabilizing relative to WT (in Rosetta score units).
- Create a summary table (e.g. CSV or markdown) with columns:
  - `mutation`
  - `mean_ddG`
  - `std_ddG`
  - `n_replicates`
  - `tier` (A/B/C – see below)
  - `notes` (space for structural comments later).
- Define heuristic tiers:
  - **Tier A**: clearly favorable (e.g., mean ΔΔG ≤ –1).
  - **Tier B**: roughly neutral (–1 < mean ΔΔG < +1).
  - **Tier C**: clearly unfavorable (mean ΔΔG ≥ +1).

### Suggested Cursor prompt

> I have Rosetta ddG results in a JSON file (`mutlist.json`) and possibly a text summary (`mutlist.ddg`).  
> I want a small analysis script that:  
> 1. Loads `mutlist.json`.  
> 2. Groups entries by mutation identity (e.g., D121G).  
> 3. For each mutation, computes mean ΔΔG and standard deviation across replicates.  
> 4. Assigns each mutation to a tier:  
>    - Tier A: mean ΔΔG ≤ -1.0  
>    - Tier B: -1.0 < mean ΔΔG < +1.0  
>    - Tier C: mean ΔΔG ≥ +1.0 
> 5. Outputs a sorted CSV or markdown table with columns: mutation, mean_ddG, std_ddG, n_replicates, tier.  
>  
> Please infer the JSON schema from a few example entries that I will paste, then write the analysis script. Add clear comments so I can adapt thresholds later.

(You’ll paste a few snippets from `mutlist.json` into Cursor when running this.)

---

## 3. Structural Sanity Check Using PyMOL

**Goal:** Visually inspect top Tier A/B mutations to make sure they look physically reasonable.

### Tasks

For each top candidate mutation (Tier A and the best Tier B):

1. Load in PyMOL:
   - WT structure from ddG (e.g., `WT__bj2.pdb`).
   - Corresponding mutant (e.g., `MUT_195PHE_bj2.pdb`).
2. Align mutant to WT (if not already).
3. Inspect:
   - Local side-chain environment.
   - Catalytic residues and any key active-site residues nearby.
   - Any obvious clashes or weird geometry.
4. Add short comments to the `notes` column in your ddG summary table:
   - Examples:
     - “Looks good; fills hydrophobic pocket.”
     - “Introduces buried charge with no partner; risky.”
     - “Perturbs catalytic His orientation; likely bad despite score.”

### PyMOL sanity checklist (for your reference)

When looking at a mutant vs WT in PyMOL, quickly ask:

- Are there any **obvious steric clashes**? (Atoms overlapping, interpenetrating.)
- Did the mutation:
  - Improve packing?
  - Or create a cavity?
- Are **catalytic residues** still in reasonable positions?
- Did any region appear unnaturally distorted relative to WT?

You don’t need Cursor for this part; it’s mostly manual visualization and short notes.

---

## 4. Design Phase 2 Mutation List (Targeted Expansion)

**Goal:** Use Phase 1 results to design a **Phase 2 mutation set** focusing on promising positions and rational alternatives.

### Tasks

- From the ddG summary:
  - Identify residue positions that:
    - Have at least one Tier A mutation.
    - Or multiple Tier B mutations that seem structurally okay.
- For each such position:
  - Propose 2–4 additional rational substitutions:
    - Swap among hydrophobics for pocket residues.
    - Swap charges for surface residues to build salt bridges.
    - Use literature knowledge about PETase for good candidates.
- Assemble a new mutation list (call it `mutlist_phase2.mut`) in the same format as your first run.
- Plan to run the same Rosetta ddG protocol but with this expanded list.

### Suggested Cursor prompt

> I have an existing CSV/markdown table summarizing single-point PETase mutations with mean ddG, standard deviation, and a tier label (A/B/C).  
> I want a helper script that:  
> 1. Reads this table.  
> 2. Identifies residue positions where at least one Tier A or Tier B mutation exists.  
> 3. For each such position, proposes additional rational amino-acid substitutions based on simple rules, such as:  
>    - If residue is hydrophobic (A, V, L, I, M, F, W, Y), propose other hydrophobics.  
>    - If residue is charged (D, E, K, R), propose opposite-charge partners if it is solvent-exposed.  
> 4. Produces a structured text output that I can convert into a Rosetta mutation list (`mutlist_phase2.mut`).  
>  
> Implement the rules as clearly commented, editable logic so I can change which amino acids are allowed per residue type.

---

## 5. Plan and Integrate ProGen2 Sequence Generation

**Goal:** Incorporate a generative model (ProGen2) to design a broader set of PETase variants beyond single-point mutations.

*(You might be running ProGen2 via a notebook, API, or external service; Cursor can help you write the helper scripts, not the model itself.)*

### Tasks

- Decide:
  - How many sequences you want to generate (e.g., 50–200).
  - Acceptable sequence identity range relative to WT (e.g., 60–95% identity).
  - Must-keep residues (e.g., catalytic residues, key motifs).
- Prepare:
  - WT PETase sequence in FASTA format.
  - Optional: a small set of related PETase or cutinase sequences for context.
- After obtaining ProGen2 outputs:
  - Filter sequences to:
    - Remove ones with stop codons or big insertions/deletions (unless you want them).
    - Enforce identity and conservation constraints.
  - Assign each candidate a unique ID (e.g., `PG2_001`, `PG2_002`).
  - Save:
    - `progen2_all.fasta`
    - `progen2_selected.fasta`

### Suggested Cursor prompt

> I will have a FASTA file containing candidate PETase-like sequences generated by ProGen2, plus another FASTA file containing the WT PETase sequence.  
> I want a script that:  
> 1. Reads all candidate sequences.  
> 2. Computes sequence identity to WT.  
> 3. Filters candidates to a user-defined identity window (e.g., between 60% and 95%).  
> 4. Filters out sequences with internal stop codons or non-standard amino acids.  
> 5. Optionally checks that specific positions (catalytic residues I will provide) are conserved.  
> 6. Writes two FASTA files: `progen2_filtered.fasta` (all passing filters) and `progen2_selected.fasta` (top N sequences by identity or some diversity rule).  
>  
> Please include clear comments and parameters I can easily change (identity thresholds, list of catalytic residues to enforce, N for the final selection).

---

## 6. AlphaFold / ColabFold Prediction for Selected Variants

**Goal:** Predict 3D structures for a subset of promising sequences (from rational design and ProGen2) and prepare them for Rosetta.

### Tasks

- Choose:
  - All top rational mutants (e.g., a handful of best single-point or combined mutations).
  - A selected subset of ProGen2 variants (`progen2_selected.fasta`).
- For each variant:
  - Run AlphaFold/ColabFold with:
    - The full sequence.
    - Standard settings (unless you have specific preferences).
  - Save:
    - Ranked models.
    - pLDDT scores and PAE matrices.
  - Put results into:
    - `data/structures/<variant_id>/alphafold/`
- Extract:
  - The best-ranked model (or best pLDDT) for each variant.
  - A small summary of pLDDT (e.g., mean, minimum around catalytic residues).

### Suggested Cursor prompt

> I will be running AlphaFold/ColabFold externally and saving predicted structures (PDB files) into `data/structures/<variant_id>/alphafold/`.  
> I want a script that:  
> 1. Walks through `data/structures/` and finds all variant folders that contain AlphaFold outputs.  
> 2. For each variant, identifies the best-ranked PDB (e.g., by a ranking file or filename convention).  
> 3. Computes simple metrics from the PDB plus AlphaFold metadata, such as mean pLDDT, minimum pLDDT near catalytic residues (I will provide their residue indices), and model length.  
> 4. Writes a summary table (CSV or markdown) across all variants with columns like: variant_id, best_model_filename, mean_pLDDT, catalytic_region_min_pLDDT.  
>  
> Please assume I will give you an example AlphaFold output folder layout, and make the script flexible to minor naming differences.

---

## 7. Relax + Stability Scoring for AlphaFold Models

**Goal:** Make all AlphaFold-predicted variants comparable under Rosetta’s energy function.

### Tasks

For each variant with an AlphaFold structure:

- Run Rosetta **relax** on the best AlphaFold model:
  - Generate a small ensemble (e.g., 5–10 relaxed structures).
  - Save in `data/structures/<variant_id>/relax/`.
- Run a **stability scoring** protocol:
  - Use Rosetta `score_jd2` or your existing ddG-style protocol adapted for whole-protein scoring.
  - Record per-model scores in a `score.sc` file.
- Create a **variant-level summary**:
  - Best (lowest) score.
  - Average score across relaxed models.
  - Compare to WT PETase relaxed models.

### Suggested Cursor prompt

> I have multiple variant directories under `data/structures/`, each containing a best AlphaFold PDB in an `alphafold/` subfolder.  
> I already have a Rosetta relax script that can take an input PDB and produce multiple relaxed models plus a `score.sc` file.  
> I want an orchestration script that:  
> 1. Iterates over all variants with AlphaFold models.  
> 2. For each, calls my existing relax script with appropriate arguments to generate a specified number of relaxed models (e.g., nstruct = 10).  
> 3. After relax finishes, calls a scoring script (or uses Rosetta’s scoring executable) to produce a `score.sc` in each variant’s `relax/` folder.  
> 4. Once all variants are processed, parses all `score.sc` files and produces a summary table with per-variant best score and average score, plus comparison back to WT.  
>  
> Please focus on directory traversal, orchestration, and aggregation of results. Assume I will provide the exact commands for calling the Rosetta binaries.

(Important: do **not** ask Cursor to embed Rosetta executables; you just pass in the command lines you already use.)

---

## 8. Integrating Stability, Structure, and Sequence Data

**Goal:** Combine ddG results, AlphaFold metrics, and sequence information into a single ranked view of variant quality.

### Tasks

- Combine data sources into one master table:
  - From ddG analyses:
    - Per-mutation or per-variant ΔΔG.
  - From AlphaFold:
    - mean pLDDT, catalytic region pLDDT.
  - From relax scoring:
    - Best/average Rosetta score per variant.
  - From sequence analysis:
    - Identity to WT.
    - Number of mutations and their types/positions.
- Define a simple scoring rubric:
  - For example, rank variants by:
    - High stability (low energy).
    - Good structural confidence (pLDDT).
    - Preserved catalytic geometry (based on your notes).
- Tag top 3–5 variants as **“final candidates”** to carry into any docking or RFdiffusion work.

### Suggested Cursor prompt

> I will have several CSV or markdown tables:  
> - ddG summary per mutation or per variant.  
> - Structural summary per variant (mean pLDDT, catalytic region pLDDT).  
> - Rosetta relax scores per variant (best and average).  
> - Sequence identity per variant relative to WT.  
>  
> I want a script that merges these into one master table keyed by variant ID, handling missing entries gracefully.  
> Then, I want a simple composite ranking score, where I can assign weights to:  
> - stability (Rosetta score)  
> - structural confidence (mean pLDDT)  
> - catalytic region confidence  
> - sequence identity  
>  
> The script should:  
> 1. Merge all input tables.  
> 2. Compute a composite “design_score” using configurable weights.  
> 3. Sort variants by this design_score.  
> 4. Output both a CSV and a markdown summary.  
>  
> Please implement this with clear, adjustable weighting parameters and comments.

---

## 9. Ongoing Reproducibility Practices

**Goal:** Make the entire pipeline easy to explain and reproduce in your final report.

### Tasks

- Maintain `PROJECT_LOG.md`:
  - Append a short entry after each major run (relax, ddG, AlphaFold, ProGen2 filtering).
- For each protocol type, create a small “recipe” markdown:
  - Inputs (e.g., PDBs, mutation list).
  - Key parameters (nstruct, scoring function, etc.).
  - Expected outputs.
- Keep a **variant dictionary**:
  - `variant_id → description (mutations, origin: “rational” vs “ProGen2”, etc.)`.

### Suggested Cursor prompt

> Help me draft a set of short “protocol recipes” in markdown for my PETase optimization project.  
> For each of these steps:  
> - Rosetta relax (WT or variant)  
> - Rosetta ddG scan  
> - ProGen2 candidate filtering  
> - AlphaFold/ColabFold prediction  
> - Rosetta stability scoring of AlphaFold models  
>  
> I want a brief section with:  
> - Purpose  
> - Inputs  
> - Key parameters (as bullet points)  
> - Outputs  
> - Notes / common pitfalls  
>  
> The output should be concise but clear enough that someone else in the lab could follow it without seeing my scripts.

---

## 10. How to Use This Document in Cursor

- Keep this file in your repo as something like `NEXT_STEPS_CURSOR.md`.
- When you’re ready to implement a step:
  - Copy the relevant “Suggested Cursor prompt” section.
  - Paste into Cursor’s chat.
  - Add any concrete file paths, snippets, or JSON samples as needed.
  - Let Cursor generate the script or helper code.
- Link back to this document from your main `README.md` so Future You knows where the pipeline plan lives.

