# PyMOL script to visualize a single candidate with pLDDT coloring
# Usage: pymol scripts/visualize_single_candidate.pml
# Or: pymol -c "run scripts/visualize_single_candidate.pml; load_candidate candidate_6"

# Clear any existing structures
reinitialize

# Set working directory to repo root
cd /Users/oskarherlitz/Desktop/petase-lab

# Function to load a candidate (call from PyMOL command line)
# Usage: load_candidate candidate_6
python
def load_candidate(candidate_name):
    import glob
    pdb_files = glob.glob(f"runs/colabfold_predictions_gpu/{candidate_name}_unrelaxed_rank_001_*.pdb")
    if pdb_files:
        cmd.load(pdb_files[0], candidate_name)
        print(f"Loaded: {pdb_files[0]}")
        return True
    else:
        print(f"Error: No PDB file found for {candidate_name}")
        return False
python end

# Load candidate_6 by default (top candidate)
python
load_candidate("candidate_6")
python end

# Color by pLDDT confidence (B-factor column contains pLDDT)
# High confidence (pLDDT > 90): blue
# Medium confidence (70-90): cyan/yellow
# Low confidence (< 70): red/orange
spectrum b, blue_red, all, minimum=50, maximum=100

# Show as cartoon
show cartoon
set cartoon_transparency, 0.2

# Show side chains for low confidence regions
select low_conf, b < 70
show sticks, low_conf
color red, low_conf

# Set nice view
set ray_shadows, 0
bg_color white
zoom all

print(f"\nLoaded {candidate}")
print("Structure colored by pLDDT confidence:")
print("  Blue = High confidence (pLDDT > 90)")
print("  Yellow/Orange = Medium confidence (70-90)")
print("  Red = Low confidence (pLDDT < 70)")

