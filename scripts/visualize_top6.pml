# PyMOL script to visualize WT enzyme and top 6 candidates with catalytic triad
# Usage: pymol scripts/visualize_top6.pml

# Clear any existing structures
reinitialize

# Set working directory
cd /Users/oskarherlitz/Desktop/petase-lab

# Load WT enzyme (reference) - uses PDB numbering
load data/structures/5XJH/raw/PETase_raw.pdb, WT

# Load top 6 candidates (by pLDDT score) - uses ColabFold numbering (starts at 1)
load runs/colabfold_predictions_gpu/candidate_6_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb, top1
load runs/colabfold_predictions_gpu/candidate_9_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top2
load runs/colabfold_predictions_gpu/candidate_60_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top3
load runs/colabfold_predictions_gpu/candidate_21_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top4
load runs/colabfold_predictions_gpu/candidate_66_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top5
load runs/colabfold_predictions_gpu/candidate_25_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb, top6

# Color structures distinctly
color gray, WT
color red, top1
color blue, top2
color green, top3
color yellow, top4
color orange, top5
color purple, top6

# Show all as cartoon
show cartoon
set cartoon_transparency, 0.3

# Align all candidates to WT for comparison
align top1, WT
align top2, WT
align top3, WT
align top4, WT
align top5, WT
align top6, WT

# ============================================================================
# HIGHLIGHT CATALYTIC TRIAD
# ============================================================================
# Catalytic triad: Ser160, Asp206, His237 (PDB numbering in WT)
# In ColabFold structures (numbered from 1): Ser131, Asp177, His208

# Function to highlight catalytic triad for a structure
# Usage: highlight_triad object_name, resi_ser, resi_asp, resi_his, prefix
python
def highlight_triad(obj_name, ser_res, asp_res, his_res, prefix):
    # Select catalytic triad residues
    cmd.select(f"triad_{prefix}", f"({obj_name} and resi {ser_res}+{asp_res}+{his_res})")
    cmd.show("sticks", f"triad_{prefix}")
    cmd.color("yellow", f"triad_{prefix}")
    cmd.set("stick_radius", 0.3, f"triad_{prefix}")
    
    # Highlight key catalytic atoms
    cmd.select(f"ser_{prefix}", f"({obj_name} and resi {ser_res} and name OG)")
    cmd.select(f"asp_{prefix}", f"({obj_name} and resi {asp_res} and name OD1+OD2)")
    cmd.select(f"his_{prefix}", f"({obj_name} and resi {his_res} and name NE2)")
    cmd.show("spheres", f"ser_{prefix}+asp_{prefix}+his_{prefix}")
    cmd.color("yellow", f"ser_{prefix}")
    cmd.color("orange", f"asp_{prefix}")
    cmd.color("red", f"his_{prefix}")
    cmd.set("sphere_scale", 1.2, f"ser_{prefix}+asp_{prefix}+his_{prefix}")
python end

# Highlight catalytic triad for WT (PDB numbering: 160, 206, 237)
python
highlight_triad("WT", 160, 206, 237, "WT")
python end

# Highlight catalytic triad for all candidates (ColabFold numbering: 131, 177, 208)
python
for i in range(1, 7):
    highlight_triad(f"top{i}", 131, 177, 208, f"top{i}")
python end

# Set nice view
set ray_shadows, 0
bg_color white
zoom all

# Print info
print "=" * 60
print "WT Enzyme and Top 6 ColabFold Candidates:"
print "  WT (gray): Wild-type PETase reference"
print "  top1 (red, candidate_6):  pLDDT = 96.22"
print "  top2 (blue, candidate_9):  pLDDT = 96.11"
print "  top3 (green, candidate_60): pLDDT = 96.06"
print "  top4 (yellow, candidate_21): pLDDT = 95.97"
print "  top5 (orange, candidate_66): pLDDT = 95.73"
print "  top6 (purple, candidate_25): pLDDT = 94.89"
print ""
print "Catalytic Triad Highlighted:"
print "  - Ser160 (WT) / Ser131 (ColabFold) - Yellow spheres (OG)"
print "  - Asp206 (WT) / Asp177 (ColabFold) - Orange spheres (OD1/OD2)"
print "  - His237 (WT) / His208 (ColabFold) - Red spheres (NE2)"
print "=" * 60
print ""
print "Use 'zoom triad_WT' to focus on the catalytic triad."
print "Use 'hide all; show cartoon, top1' to view individual candidates."

