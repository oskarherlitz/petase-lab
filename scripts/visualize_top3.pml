# PyMOL script to visualize WT enzyme and top 3 candidates
# Usage: pymol scripts/visualize_top3.pml

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

# Color WT in gray (reference structure)
color gray, WT

# Color candidates by pLDDT confidence (B-factor column contains pLDDT)
spectrum b, blue_red, top1, minimum=50, maximum=100
spectrum b, blue_red, top2, minimum=50, maximum=100
spectrum b, blue_red, top3, minimum=50, maximum=100

# Show all as cartoon
show cartoon, WT
show cartoon, top1
show cartoon, top2
show cartoon, top3

# Set transparency so we can see all structures
set cartoon_transparency, 0.5, WT
set cartoon_transparency, 0.4, top1
set cartoon_transparency, 0.4, top2
set cartoon_transparency, 0.4, top3

# Align all candidates to WT for comparison
align top1, WT
align top2, WT
align top3, WT

# ============================================================================
# HIGHLIGHT CATALYTIC TRIAD
# ============================================================================
# Catalytic triad: Ser160, Asp206, His237 (PDB numbering in WT)
# In ColabFold structures (numbered from 1): Ser131, Asp177, His208

# WT structure (PDB numbering: 160, 206, 237)
select triad_WT, (WT and resi 160+206+237)
show sticks, triad_WT
color yellow, triad_WT
set stick_radius, 0.3, triad_WT

# Highlight key catalytic atoms as spheres
select ser160_WT, (WT and resi 160 and name OG)
select asp206_WT, (WT and resi 206 and name OD1+OD2)
select his237_WT, (WT and resi 237 and name NE2)
show spheres, ser160_WT+asp206_WT+his237_WT
color yellow, ser160_WT
color orange, asp206_WT
color red, his237_WT
set sphere_scale, 1.2, ser160_WT+asp206_WT+his237_WT

# Top1 (candidate_6) - ColabFold numbering: 131, 177, 208
select triad_top1, (top1 and resi 131+177+208)
show sticks, triad_top1
color yellow, triad_top1
set stick_radius, 0.3, triad_top1

select ser131_top1, (top1 and resi 131 and name OG)
select asp177_top1, (top1 and resi 177 and name OD1+OD2)
select his208_top1, (top1 and resi 208 and name NE2)
show spheres, ser131_top1+asp177_top1+his208_top1
color yellow, ser131_top1
color orange, asp177_top1
color red, his208_top1
set sphere_scale, 1.2, ser131_top1+asp177_top1+his208_top1

# Top2 (candidate_9) - ColabFold numbering: 131, 177, 208
select triad_top2, (top2 and resi 131+177+208)
show sticks, triad_top2
color yellow, triad_top2
set stick_radius, 0.3, triad_top2

select ser131_top2, (top2 and resi 131 and name OG)
select asp177_top2, (top2 and resi 177 and name OD1+OD2)
select his208_top2, (top2 and resi 208 and name NE2)
show spheres, ser131_top2+asp177_top2+his208_top2
color yellow, ser131_top2
color orange, asp177_top2
color red, his208_top2
set sphere_scale, 1.2, ser131_top2+asp177_top2+his208_top2

# Top3 (candidate_60) - ColabFold numbering: 131, 177, 208
select triad_top3, (top3 and resi 131+177+208)
show sticks, triad_top3
color yellow, triad_top3
set stick_radius, 0.3, triad_top3

select ser131_top3, (top3 and resi 131 and name OG)
select asp177_top3, (top3 and resi 177 and name OD1+OD2)
select his208_top3, (top3 and resi 208 and name NE2)
show spheres, ser131_top3+asp177_top3+his208_top3
color yellow, ser131_top3
color orange, asp177_top3
color red, his208_top3
set sphere_scale, 1.2, ser131_top3+asp177_top3+his208_top3

# Set nice view
set ray_shadows, 0
bg_color white
zoom all

# Print info
print "=" * 60
print "WT Enzyme and Top 3 ColabFold Candidates:"
print "  WT (wild-type PETase): Reference structure"
print "  top1 (candidate_6):  pLDDT = 96.22"
print "  top2 (candidate_9):  pLDDT = 96.11"
print "  top3 (candidate_60): pLDDT = 96.06"
print ""
print "Catalytic Triad Highlighted:"
print "  - Ser160 (WT) / Ser131 (ColabFold) - Yellow spheres (OG)"
print "  - Asp206 (WT) / Asp177 (ColabFold) - Orange spheres (OD1/OD2)"
print "  - His237 (WT) / His208 (ColabFold) - Red spheres (NE2)"
print "=" * 60
print ""
print "All structures are aligned to WT for comparison."
print "WT is shown in gray, candidates colored by pLDDT confidence."
print "Use 'zoom triad_WT' to focus on the catalytic triad."

