# PyMOL Quick Commands for ColabFold Results

## Quick Start

### Option 1: Visualize top 5 candidates together
```bash
pymol scripts/visualize_top_candidates.pml
```

### Option 2: Visualize a single candidate
```bash
pymol scripts/visualize_single_candidate.pml candidate_6
```

### Option 3: Interactive PyMOL session
```bash
pymol
```

Then in PyMOL:
```python
# Load a single candidate
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, candidate_6

# Color by pLDDT confidence (B-factor column)
spectrum b, blue_red, candidate_6, minimum=50, maximum=100

# Show as cartoon
show cartoon, candidate_6
```

## Top 10 Candidates (by pLDDT score)

1. **candidate_6** - pLDDT: 96.22
2. **candidate_9** - pLDDT: 96.11
3. **candidate_60** - pLDDT: 96.06
4. **candidate_21** - pLDDT: 95.97
5. **candidate_66** - pLDDT: 95.73
6. **candidate_25** - pLDDT: 94.89
7. **candidate_28** - pLDDT: 94.48
8. **candidate_8** - pLDDT: 94.39
9. **candidate_56** - pLDDT: 94.06
10. **candidate_4** - pLDDT: 94.05

## Useful PyMOL Commands

### Loading structures
```python
# Load a single structure
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, candidate_6

# Load multiple candidates for comparison
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_*.pdb, candidate_6
load runs/colabfold_predictions_gpu/candidate_9_unrelaxed_rank_001_*.pdb, candidate_9
load runs/colabfold_predictions_gpu/candidate_60_unrelaxed_rank_001_*.pdb, candidate_60
```

### Coloring by confidence
```python
# Color by pLDDT (stored in B-factor column)
spectrum b, blue_red, all, minimum=50, maximum=100

# Alternative: Color by chain
color red, candidate_6
color blue, candidate_9
```

### Display styles
```python
# Show as cartoon (default)
show cartoon

# Show as surface
show surface

# Show as sticks (for active site)
show sticks

# Show both cartoon and sticks
show cartoon
show sticks, resi 105-110
```

### Alignment and comparison
```python
# Align two structures
align candidate_9, candidate_6

# Superimpose multiple structures
super candidate_9, candidate_6
super candidate_60, candidate_6
```

### Selecting regions
```python
# Select by residue number
select active_site, resi 105-110

# Select by confidence (low confidence regions)
select low_conf, b < 70

# Select by chain
select chain_a, chain A
```

### Viewing and rendering
```python
# Set background
bg_color white

# Remove shadows for cleaner look
set ray_shadows, 0

# Zoom to selection
zoom active_site

# Zoom to all
zoom all

# Rotate view
rotate y, 90
rotate x, 45
```

### Saving images
```python
# Save as PNG
png candidate_6_view.png, width=2000, height=2000, dpi=300

# Save as ray-traced image (higher quality)
ray 2000, 2000
png candidate_6_raytraced.png
```

### Comparing multiple models
```python
# Load all 3 models for a candidate
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_*.pdb, model_1
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_002_*.pdb, model_2
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_003_*.pdb, model_3

# Color each differently
color red, model_1
color blue, model_2
color green, model_3

# Align them
align model_2, model_1
align model_3, model_1
```

## Example: Visualize top 3 candidates side-by-side

```python
# Load top 3
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, top1
load runs/colabfold_predictions_gpu/candidate_9_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, top2
load runs/colabfold_predictions_gpu/candidate_60_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top3

# Color by confidence
spectrum b, blue_red, all, minimum=50, maximum=100

# Show as cartoon
show cartoon

# Align for comparison
align top2, top1
align top3, top1

# Arrange in view
split_states top1
split_states top2
split_states top3
```

