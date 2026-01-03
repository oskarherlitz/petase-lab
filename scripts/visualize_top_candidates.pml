# PyMOL script to visualize top ColabFold candidates
# Usage: pymol scripts/visualize_top_candidates.pml

# Clear any existing structures
reinitialize

# Set working directory to repo root
cd /Users/oskarherlitz/Desktop/petase-lab

# Top 5 candidates by pLDDT score
# candidate_6: 96.22
# candidate_9: 96.11
# candidate_60: 96.06
# candidate_21: 95.97
# candidate_66: 95.73

# Load the top 5 candidates
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, candidate_6
load runs/colabfold_predictions_gpu/candidate_9_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, candidate_9
load runs/colabfold_predictions_gpu/candidate_60_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, candidate_60
load runs/colabfold_predictions_gpu/candidate_21_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb, candidate_21
load runs/colabfold_predictions_gpu/candidate_66_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, candidate_66

# Color by pLDDT confidence (using B-factor column which contains pLDDT)
# High confidence (pLDDT > 90): blue
# Medium confidence (70-90): cyan
# Low confidence (< 70): red
spectrum b, blue_red, all, minimum=50, maximum=100

# Show as cartoon with transparency
show cartoon
set cartoon_transparency, 0.3

# Color each candidate differently for comparison
color red, candidate_6
color orange, candidate_9
color yellow, candidate_60
color green, candidate_21
color blue, candidate_66

# Arrange in a grid for comparison
# You can manually rotate/translate each structure, or use:
# align candidate_9, candidate_6
# align candidate_60, candidate_6
# align candidate_21, candidate_6
# align candidate_66, candidate_6

# Set nice view
set ray_shadows, 0
bg_color white

# Zoom to show all structures
zoom all

# Print information
print "=" * 60
print "Top 5 ColabFold Candidates (by pLDDT score):"
print "  candidate_6:  pLDDT = 96.22"
print "  candidate_9:  pLDDT = 96.11"
print "  candidate_60: pLDDT = 96.06"
print "  candidate_21: pLDDT = 95.97"
print "  candidate_66: pLDDT = 95.73"
print "=" * 60
print ""
print "Commands:"
print "  'hide all; show cartoon, candidate_6' - Show only candidate_6"
print "  'align candidate_9, candidate_6' - Align candidate_9 to candidate_6"
print "  'select active_site, resi 105-110' - Select specific residues"
print "  'show sticks, active_site' - Show selected residues as sticks"

