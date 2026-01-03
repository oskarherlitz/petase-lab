# PyMOL script to visualize WT enzyme and top 3 candidates side-by-side
# Usage: pymol scripts/visualize_wt_and_top3.pml

# Clear any existing structures
reinitialize

# Set working directory
cd /Users/oskarherlitz/Desktop/petase-lab

# Load WT enzyme first (as reference)
load data/structures/5XJH/raw/PETase_raw.pdb, WT

# Load top 3 candidates (by pLDDT score)
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb, top1
load runs/colabfold_predictions_gpu/candidate_9_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top2
load runs/colabfold_predictions_gpu/candidate_60_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top3

# Color each structure distinctly
color gray, WT
color red, top1
color blue, top2
color green, top3

# Show all as cartoon
show cartoon, WT
show cartoon, top1
show cartoon, top2
show cartoon, top3

# Set transparency
set cartoon_transparency, 0.3, WT
set cartoon_transparency, 0.3, top1
set cartoon_transparency, 0.3, top2
set cartoon_transparency, 0.3, top3

# Align all candidates to WT for comparison
align top1, WT
align top2, WT
align top3, WT

# Set nice view
set ray_shadows, 0
bg_color white
zoom all

# Print info
print "=" * 60
print "WT Enzyme and Top 3 ColabFold Candidates:"
print "  WT (gray): Wild-type PETase reference"
print "  top1 (red, candidate_6):  pLDDT = 96.22"
print "  top2 (blue, candidate_9):  pLDDT = 96.11"
print "  top3 (green, candidate_60): pLDDT = 96.06"
print "=" * 60
print ""
print "All structures aligned to WT. Use 'hide all; show cartoon, WT' to see only WT."
print "Or 'hide all; show cartoon, top1' to see only candidate_6."

