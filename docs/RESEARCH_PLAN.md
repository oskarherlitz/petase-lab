# PETase Optimization Research Plan

## Current State Assessment

### ✅ Completed
- Repository infrastructure set up
- PETase crystal structure obtained (5XJH.pdb from PDB)
- FoldX structure repair completed (5XJH_Repair.pdb)
- Initial FoldX analysis performed
- Mutation testing started (DA150G)

### ⚠️ In Progress
- Integration of FoldX results with Rosetta pipeline
- Systematic mutation screening

### ❌ Not Started
- Rosetta cartesian relaxation
- Rosetta ΔΔG calculations
- Active site design with catalytic constraints
- Comprehensive mutation library
- Cross-validation between FoldX and Rosetta

---

## Phase 1: Data Integration & Setup (Week 1)

### 1.1 Prepare Input Files
**Goal**: Integrate existing FoldX work with Rosetta pipeline

**Tasks**:
1. **Copy repaired structure to raw directory**
   ```bash
   cp data/structures/5XJH/foldx/5XJH_Repair.pdb data/structures/5XJH/raw/PETase_raw.pdb
   ```

2. **Verify structure quality**
   - Check for missing residues
   - Verify chain assignment
   - Ensure proper atom naming

3. **Prepare ligand parameters** (if using BHET substrate)
   ```bash
   bash scripts/make_params.sh data/raw/ligands/BHET.sdf
   ```

4. **Update data registry**
   - Add file metadata to `data/registry.csv`
   - Document source and processing steps

### 1.2 Define Catalytic Constraints
**Goal**: Set up proper constraints for PETase active site

**PETase Catalytic Triad**:
- **Ser160** (OG) - Nucleophile
- **Asp206** (OD1/OD2) - Base
- **His237** (NE2) - Acid

**Tasks**:
1. Identify exact atom names in structure
2. Measure distances and angles from crystal structure
3. Update `configs/rosetta/catalytic.cst` with real values
4. Test constraints don't break structure

**Example constraint format**:
```
CST::BEGIN
TEMPLATE:: ATOM_MAP: 1 residue: 160 atom_name: OG
TEMPLATE:: ATOM_MAP: 2 residue: 206 atom_name: OD1
CONSTRAINT:: distanceAB: 2.8 0.2 100.0
CONSTRAINT:: angle_ABX: 107.0 5.0 50.0
CST::END
```

### 1.3 Create Mutation Library
**Goal**: Define systematic mutation strategy

**Key Positions to Target** (based on literature):
1. **Active site residues** (160, 206, 237) - catalytic efficiency
2. **Substrate binding pocket** (150, 180, 224) - substrate affinity
3. **Stability hotspots** - thermal stability
4. **Surface residues** - solubility/expression

**Tasks**:
1. Create `configs/rosetta/mutlist.mut` with systematic mutations
2. Start with single-point mutations
3. Expand to double/triple mutations for top candidates

**Example mutation file**:
```
total 3
1
160 A SER ALA
2
206 A ASP ASN
3
150 A ASP GLY
```

---

## Phase 2: Initial Structure Optimization (Week 1-2)

### 2.1 Rosetta Cartesian Relaxation
**Goal**: Generate high-quality starting structure for design

**Command**:
```bash
export ROSETTA_BIN=/path/to/rosetta/main/source/bin
make relax
# OR
bash scripts/rosetta_relax.sh data/structures/5XJH/raw/PETase_raw.pdb
```

**Expected Output**:
- `runs/YYYY-MM-DD_relax_cart_v1/outputs/` - 20 relaxed structures
- Best structure selected by total score

**Analysis**:
- Compare Rosetta scores with FoldX results
- Verify structure quality (Ramachandran, clash score)
- Select best relaxed structure for downstream work

### 2.2 ΔΔG Calculations
**Goal**: Identify stability bottlenecks and validate mutations

**Command**:
```bash
# First, create mutation list
# Then run:
make ddg
# OR
bash scripts/rosetta_ddg.sh runs/*relax*/outputs/*.pdb configs/rosetta/mutlist.mut
```

**Analysis**:
```bash
# Parse and rank results
python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/initial_scan.csv
python scripts/rank_designs.py results/ddg_scans/initial_scan.csv 20
```

**Compare with FoldX**:
- Cross-validate ΔΔG predictions
- Identify discrepancies
- Build consensus model

---

## Phase 3: Active Site Design (Week 2-3)

### 3.1 FastDesign with Catalytic Constraints
**Goal**: Optimize active site while preserving catalysis

**Setup**:
1. Update `configs/rosetta/design.res` with target positions
2. Ensure catalytic constraints are active
3. Run FastDesign protocol

**Command**:
```bash
$ROSETTA_BIN/rosetta_scripts.linuxgccrelease \
  -s runs/*relax*/outputs/*.pdb \
  -parser:protocol configs/rosetta/design.xml \
  -nstruct 50 \
  -out:path:all runs/$(date +%F)_fastdesign
```

**Design Strategy**:
- **Conservative**: Allow only similar amino acids at active site
- **Moderate**: Allow broader set, but maintain charge/polarity
- **Aggressive**: Full design space (use with caution)

### 3.2 Filter and Rank Designs
**Goal**: Identify promising candidates

**Criteria**:
1. **Stability**: Total score improvement
2. **Catalytic geometry**: Constraint satisfaction
3. **Substrate binding**: Interface score (if ligand docked)
4. **FoldX validation**: Cross-check top candidates

**Workflow**:
```bash
# Score all designs
# Filter by constraints
# Rank by total score
# Select top 10-20 for FoldX validation
```

---

## Phase 4: Cross-Validation & Selection (Week 3-4)

### 4.1 FoldX BuildModel Validation
**Goal**: Validate top Rosetta designs with FoldX

**Command**:
```bash
# For each top design
foldx --command=BuildModel \
  --pdb=design.pdb \
  --mutant-file=mutations.txt \
  --output-dir=foldx_validation/
```

**Analysis**:
- Compare Rosetta vs FoldX ΔΔG
- Identify consensus predictions
- Flag designs with large discrepancies

### 4.2 Structure Quality Assessment
**Goal**: Ensure designs are physically reasonable

**Checks**:
- Ramachandran plot (should be >95% favored)
- Clash score (<10)
- Packing score
- Secondary structure preservation

**Tools**:
- PyMOL for visualization
- MolProbity for validation
- Rosetta score breakdown

---

## Phase 5: Advanced Optimization (Week 4-6, Optional)

### 5.1 Structure Prediction
**Goal**: Validate novel designs with AlphaFold/ColabFold

**When to use**:
- Radical redesigns
- Large backbone changes
- Novel mutations not in training data

**Workflow**:
1. Generate sequences from top designs
2. Run AlphaFold/ColabFold prediction
3. Compare predicted vs designed structures
4. Select designs with good agreement

### 5.2 Backbone Generation (RFdiffusion)
**Goal**: Explore novel backbone conformations

**When to use**:
- Current active site geometry is limiting
- Need to explore alternative conformations
- Seeking radical improvements

**Workflow**:
1. Define design goals (binding site, stability)
2. Run RFdiffusion with constraints
3. Post-relax with Rosetta
4. Score and rank generated structures

---

## Phase 6: Experimental Validation Planning (Week 6+)

### 6.1 Prioritize Candidates
**Goal**: Select top 5-10 designs for experimental testing

**Selection Criteria**:
1. **Computational consensus**: Good scores in both Rosetta and FoldX
2. **Stability**: Predicted ΔΔG < -1.0 kcal/mol
3. **Catalytic preservation**: Constraints satisfied
4. **Novelty**: Different from known variants
5. **Feasibility**: Mutations are experimentally accessible

### 6.2 Generate Experimental Protocols
**Tasks**:
- Create mutation primers/sequences
- Design expression constructs
- Plan activity assays
- Define success metrics

---

## Research Workflow Summary

### Daily Workflow
1. **Morning**: Review previous day's results
2. **Run computations**: Submit jobs (local or cluster)
3. **Afternoon**: Analyze results, update mutation lists
4. **Evening**: Document findings, plan next steps

### Weekly Milestones
- **Week 1**: Complete Phase 1 & 2.1
- **Week 2**: Complete Phase 2.2 & 3.1
- **Week 3**: Complete Phase 3.2 & 4.1
- **Week 4**: Complete Phase 4.2, start Phase 5 if needed
- **Week 5-6**: Complete Phase 5, begin Phase 6

### Key Metrics to Track
- Number of designs generated
- Average ΔΔG improvement
- Constraint satisfaction rate
- Rosetta-FoldX correlation
- Top candidate scores

---

## Immediate Next Steps (This Week)

### Day 1-2: Setup
1. ✅ Copy 5XJH_Repair.pdb to data/structures/5XJH/raw/PETase_raw.pdb
2. ✅ Set up Rosetta environment
3. ✅ Define catalytic constraints based on 5XJH structure
4. ✅ Create initial mutation list (start with 10-20 positions)

### Day 3-4: Initial Runs
1. ✅ Run Rosetta relaxation
2. ✅ Analyze relaxed structures
3. ✅ Run initial ΔΔG scan on key positions
4. ✅ Compare with existing FoldX results

### Day 5: Analysis & Planning
1. ✅ Parse and rank ΔΔG results
2. ✅ Identify top mutation candidates
3. ✅ Plan FastDesign runs
4. ✅ Update research plan based on findings

---

## Resources & References

### Key Papers
- PETase structure: Joo et al. (2018) Nature Communications
- Rosetta methodology: Kortemme & Baker (2004)
- FoldX validation: Schymkowitz et al. (2005)

### Tools Documentation
- Rosetta: https://www.rosettacommons.org/docs
- FoldX: http://foldxsuite.org/
- AlphaFold: https://alphafold.ebi.ac.uk/

### Internal Documentation
- Methodology: `docs/methodology.md`
- Glossary: `docs/glossary.md`
- Tooling decisions: `docs/decisions/ADR-0001-tooling.md`

---

## Notes & Considerations

### Computational Resources
- Rosetta relaxation: ~1-2 hours per structure (20 structures)
- ΔΔG calculations: ~30 min per mutation
- FastDesign: ~2-4 hours per design (50 designs)

### Quality Control
- Always validate structures before design
- Cross-check predictions with multiple tools
- Document all parameter choices
- Keep detailed run manifests

### Troubleshooting
- If constraints break structure: relax constraint weights
- If designs have poor scores: adjust design positions
- If Rosetta-FoldX disagree: investigate specific mutations
- If convergence issues: reduce design space

---

*Last updated: [Current Date]*
*Next review: Weekly*

