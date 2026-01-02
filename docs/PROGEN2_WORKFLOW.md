# ProGen2 Workflow & Pipeline (PETase 5XJH Construct)

## Operator Playbook

This is the minimal, end-to-end “do a full round” checklist. It assumes ProGen2 is available under `external/progen2/` and you are running everything from this repo as the **driver**.

### Round recipe (one run folder = one experiment)

1. **Create a new run folder**

   * Name it `runs/run_YYYYMMDD_progen2_<model>_r<round>_<shorttag>/`.
   * Initialize the run manifest with:

     * SHIS repo git commit
     * ProGen2 repo commit (or submodule commit)
     * checkpoint/model name
     * hardware + environment versions

2. **Build prompts from the frozen baseline**

   * Generate N-terminus anchored prompts at **20 / 50 / 80 aa**.
   * Save prompts + a prompt manifest inside the run folder.

3. **Generate with ProGen2 (use lanes, not one setting)**

   * Run two sampling lanes:

     * **Conservative** (quality-biased)
     * **Exploratory** (diversity-biased)
   * Generate enough raw sequences for strict filtering (Round 1 target: hundreds+).
   * Save:

     * `generated/raw_generations.fasta` (verbatim output)
     * `generated/normalized.fasta` (control tokens removed)

4. **Sequence-only gates (the big compute saver)**
   Apply in this order and log attrition reasons:

   * Token + alphabet cleanup
   * **Length = 263 aa exact**
   * **Hard locks:** pos131=S, pos177=D, pos208=H
   * Identity buckets vs baseline (Near + Explore)
   * Uniqueness/decloning
   * Composition sanity
   * Likelihood ranking (diversity-preserving selection across prompt×lane×bucket)

   Output:

   * `filters/filter_report.json`
   * `candidates/candidates.filtered.fasta` (AF input)
   * `candidates/candidates.ranked.csv`

5. **Structure gate (ColabFold/AlphaFold)**

   * Predict structures for the filtered candidates.
   * Gate on confidence + fold sanity vs 5XJH.
   * Save:

     * `af/af_summary.csv`
     * `af/af_gate_pass/` (only pass structures)

6. **Stability gate (Rosetta/FoldX, consistent protocol)**

   * Run the repo’s standard relax/score on AF-pass designs.
   * Rank and flag red signals (clashes/packing pathologies).
   * Save:

     * `stability/stability_summary.csv`

7. **(Optional) Function proxy gate**

   * Dock a consistent PET fragment set as a failure filter.
   * Save:

     * `docking/docking_summary.csv`

8. **Finalize run outputs**

   * Write `final/final_rank.csv` (the shortlist you actually consider).
   * Update the run manifest with **counts at every gate** and the final selected candidates.

### What to hand off to downstream design/validation

For each run, the “handoff bundle” is:

* `final/final_rank.csv`
* `candidates/candidates.filtered.fasta`
* `af/af_gate_pass/` structures
* `stability/stability_summary.csv`
* the run manifest + filter report (for traceability)

---

This document defines a **high-quality, reproducible** ProGen2-driven sequence generation and evaluation pipeline for PETase designs in this SHIS repo.

It is written as an implementation spec for Cursor: **what to build, what to log, what to gate on, and how components interface**.

---

## 1) Goal

Generate **PETase-like** sequences for the **5XJH chain A crystallized construct** (PDB 30–292, **263 aa**) using **ProGen2**, then filter and evaluate them with **structure prediction + stability scoring**. Output a ranked shortlist of candidates that:

* obey hard constraints (catalytic triad locks)
* maintain plausible PETase fold
* improve stability/packing (as measured by the repo’s standard Rosetta/FoldX procedures)
* are reproducible (every run traceable to config + seeds + checkpoint + commit)

---

## 2) Baseline construct (frozen)

**Baseline sequence definition** (do not change unless explicitly creating “Design Spec v2”):

* Construct: **PDB 5XJH chain A**, crystallized/mature construct (not secreted precursor)
* Residues: **PDB 30–292**
* Length: **263 aa**
* Mapping: sequence position **1 ↔ PDB 30**, …, **263 ↔ PDB 292**

**Catalytic triad**

* PDB: **S160 / D206 / H237**
* Repo pose indices (pose = pdb − 29): **S131 / D177 / H208**

**Hard-locked residues (non-negotiable)**

* **Pos131 = S**
* **Pos177 = D**
* **Pos208 = H**

Everything else is mutable.

---

## 3) Dependency strategy (keep your repo as the “driver”)

Preferred: add the official ProGen2 repo as a **dependency** under your repo (not as a separate working project).

Two acceptable patterns:

* **Best (reproducible):** git submodule under `external/progen2/` pinned to a known commit.
* **Simplest:** clone into `external/progen2/` and record the commit hash in every run manifest.

**Rule:** ProGen2 code must be treated as read-only; your repo writes all outputs into `runs/<run_id>/…` only.

---

## 4) Run-centric reproducibility contract

All work happens inside a **run folder**. No “loose outputs” anywhere else.

### 4.1 Run folder naming

Use:

* `runs/run_YYYYMMDD_progen2_<model>_r<round>_<shorttag>/`

Examples:

* `runs/run_20251229_progen2_small_r1_anchor50/`

### 4.2 Required run manifest fields

Every run must write a manifest (human-readable + machine-readable) that includes:

* **git commit hash** of your SHIS repo
* ProGen2 repo commit hash (or submodule commit)
* ProGen2 checkpoint identifier (model size)
* hardware/context (CPU/GPU, framework versions)
* prompt set definition (prompt lengths and sequences used)
* sampling settings grid (temperature/top-p and any other sampling controls)
* RNG seeds used
* counts at every gate:

  * generated raw
  * passed token/AA cleanup
  * passed length gate
  * passed hard locks
  * passed identity buckets
  * passed uniqueness gate
  * passed AF gate
  * passed Rosetta/FoldX gate

---

## 5) Pipeline stages (stage-gated)

### Stage A — Prompt construction (conditioning)

**Objective:** keep generations PETase-like while permitting diversity.

Prompting mode: **N-terminus anchoring**

* Create multiple prompt lengths:

  * **20 aa** (diversity)
  * **50 aa** (balanced default)
  * **80 aa** (high in-family)
  * Optional: 10 aa as an “exploration lane”

Prompts are derived from the frozen baseline construct sequence.

**Deliverables (per run):**

* `prompts/` containing the prompt set used (with explicit prompt length labels)
* `prompt_manifest` listing: prompt length, raw prompt string, and the baseline reference

### Stage B — ProGen2 generation

**Objective:** generate a large enough pool so filters can be strict.

Model/size plan:

* Round 1: **progen2-small** (fast) to debug filters and pipeline
* Round 2: **progen2-medium/base** for higher quality
* Round 3 (optional): **progen2-large** once gates are stable

Sampling plan: **use lanes**, not one setting.

* Lane “Conservative” (quality-biased)
* Lane “Exploratory” (diversity-biased)

Generation targets:

* Round 1: 200–1000 raw sequences total across prompt-length × lane
* Later rounds: fewer sequences, more selective prompts (elite prompting)

**Critical:** ProGen2 uses control tokens (e.g., start/end tokens). Your pipeline must:

* preserve raw outputs for provenance
* normalize to pure AA sequences for downstream tools

**Deliverables:**

* `generated/raw_generations.fasta` (exact model outputs)
* `generated/normalized.fasta` (control tokens removed, only AAs)

### Stage C — Sequence-only gates (highest ROI)

**Objective:** reduce raw sequences to 10–30 AF-ready candidates.

Apply gates in the order below and log failures by reason.

#### Gate C0: Token + alphabet cleanup

Reject sequences that:

* contain non-standard amino acids
* contain internal control tokens
* contain invalid characters/whitespace artifacts

#### Gate C1: Length gate

* Required length after normalization: **263 aa exactly**

#### Gate C2: Hard-lock gate

* Enforce hard-locked triad by **position**:

  * 131 must be **S**
  * 177 must be **D**
  * 208 must be **H**

#### Gate C3: Family plausibility (identity buckets)

Compute identity to baseline and keep two buckets to preserve diversity:

* **Near bucket:** ~75–95% identity
* **Explore bucket:** ~55–75% identity

Reject:

* near-clones above the near bucket upper bound
* too-far sequences below the explore bucket lower bound (unless explicitly doing an “out-of-family” experiment)

#### Gate C4: Uniqueness gate

Prevent wasting AF compute on duplicates:

* enforce a minimum pairwise difference threshold (e.g., must differ by ≥5–10 residues)
* optionally cluster by identity and select representatives across prompts/lanes

#### Gate C5: Composition sanity gate

Reject obvious pathologies:

* long single-AA runs
* extreme hydrophobicity spikes or low-complexity stretches

#### Gate C6: ProGen2 likelihood ranking (recommended)

Use ProGen2 likelihood scoring to rank survivors as a “naturalness prior.”

Selection rule (diversity-preserving):

* rank **within each prompt length × lane × identity bucket**
* take top quantile from each group

**Deliverables:**

* `filters/filter_report.json` with counts removed per rule and examples
* `candidates/candidates.filtered.fasta` (AF input)
* `candidates/candidates.ranked.csv` (features: identity, bucket, lane, prompt length, likelihood score)

### Stage D — Structure gate (ColabFold/AlphaFold)

**Objective:** ensure candidates fold into a PETase-like structure before Rosetta.

For each candidate:

* run structure prediction under a consistent recipe
* record confidence metrics (pLDDT/PAE)
* compare to baseline fold (global similarity + local active-site region)

AF gate criteria (define exact thresholds in Design Spec v1.0):

* minimum mean pLDDT (global)
* local pLDDT around catalytic region
* no large-scale fold collapse or domain swaps
* triad spatial plausibility (basic geometry sanity check)

**Deliverables:**

* `af/` containing predicted structures for all attempted candidates
* `af/af_gate_pass/` containing only pass structures
* `af/af_summary.csv` containing metrics + pass/fail + reason

### Stage E — Rosetta/FoldX stability gate (consistent scoring)

**Objective:** rank AF-pass structures by stability/packing under the repo’s standardized protocols.

Rules:

* Use the **same** relaxation/scoring protocol across all candidates.
* Score relative to the baseline construct using matched settings.

Record:

* post-relax energy metrics
* packing/clash indicators used by your SHIS
* FoldX ΔΔG (secondary corroboration)

Gate logic:

* reject obvious clash/packing disasters
* rank remaining by a composite scorecard (pre-defined weights)

**Deliverables:**

* `stability/` outputs
* `stability/stability_summary.csv` with rank, deltas vs baseline, and flags

### Stage F — Function proxy gate (optional)

Use only as a **failure filter**, not a truth oracle.

* Dock a consistent PET fragment set
* Look for plausible binding near catalytic machinery
* Require basic consistency across docking replicates

Deliverables:

* `docking/docking_summary.csv`
* `final/final_rank.csv`

---

## 6) Iteration strategy (how rounds evolve)

### Round 1: Pipeline validation

* Primary goal: gates work and produce 1–2 credible candidates
* Model: progen2-small
* High diversity; wide sampling grid; strict filters

### Round 2: Quality refinement

* Model: progen2-medium/base
* Use **elite prompting** from best Round 1 sequences:

  * reuse their N-termini as prompts
* Narrow sampling grid; keep identity buckets

### Round 3 (optional): Final push

* Model: progen2-large
* Tight prompts; small candidate count; stronger AF/Rosetta gates

---

## 7) Interfaces Cursor should implement (no hidden coupling)

Cursor should implement the pipeline as **modular steps** with explicit I/O contracts.

### 7.1 Modules (conceptual)

1. **Prompt Builder**

* Input: baseline FASTA
* Output: prompt set files + prompt manifest

2. **ProGen2 Runner Wrapper**

* Input: prompt set + generation config (model, sampling lanes, seeds)
* Output: raw generations + normalized generations

3. **Sequence Filter + Feature Extractor**

* Input: normalized generations + baseline
* Output: filter report + filtered FASTA + ranked CSV

4. **AF Orchestrator**

* Input: filtered FASTA
* Output: AF structures + AF summary + pass subset

5. **Rosetta/FoldX Orchestrator**

* Input: AF pass structures
* Output: stability metrics + ranked shortlist

6. **(Optional) Docking Orchestrator**

* Input: shortlisted structures
* Output: docking summary + final rank

### 7.2 Design principles

* Every module must be runnable independently using run folder paths.
* No module reads from “current working directory assumptions.”
* Every module appends to the run manifest and writes a summary table.

---

## 8) Quality controls and failure modes

### Common failure modes

* **Token mishandling** → downstream tools see invalid residues
* **Length drift** → numbering constraints no longer map to triad locks
* **Duplicate sequences** → wasted AF compute
* **Over-conditioning** (very long prompts) → near-clones and low novelty
* **Under-conditioning** (very short prompts) → junk or out-of-family outputs

### Built-in QC checks

* Per-run histograms: identity distribution, mutation counts, composition
* Uniqueness stats: cluster sizes and representative selection coverage
* AF confidence scatter: pLDDT vs identity bucket
* “Gate accounting”: counts at each stage must reconcile

---

## 9) Acceptance criteria (what “done” means)

A run is considered successful if it produces:

* a complete run manifest with all required fields
* a filter report explaining sequence attrition
* at least one candidate that passes:

  * hard locks + length
  * AF confidence + fold sanity
  * Rosetta/FoldX stability gate (no major red flags)

---

## 10) Quick-start checklist (for Cursor)

* [ ] Add ProGen2 under `external/progen2/` (submodule preferred)
* [ ] Implement run folder creation + manifest writing
* [ ] Implement prompt builder for 20/50/80-aa anchoring prompts
* [ ] Implement ProGen2 generation wrapper that outputs raw + normalized FASTA
* [ ] Implement strict Stage C filters (length + triad locks + identity buckets + uniqueness)
* [ ] Implement likelihood ranking and diversity-preserving selection
* [ ] Wire candidates into ColabFold/AF stage with consistent metrics + gate
* [ ] Wire AF-pass into Rosetta/FoldX stage with a fixed scorecard
* [ ] Produce `final_rank.csv` and a short human-readable summary

---

## 11) Notes on future enhancements

* Add a motif-level soft constraint near the catalytic serine neighborhood (derived from baseline) to reduce “folds but nonsense active site” candidates.
* Consider two-objective selection: stability (Rosetta/FoldX) + “naturalness” (ProGen2 likelihood) to avoid over-optimized but unrealistic sequences.
* If later needed: fine-tune progen2-small on PETase/cutinase homologs, but only after the base pipeline is stable.
