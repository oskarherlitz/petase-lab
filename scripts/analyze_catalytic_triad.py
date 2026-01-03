#!/usr/bin/env python3
"""
Analyze catalytic triad geometry for PETase structures.

For each structure (WT and candidates), computes:
- Ser OG ↔ His NE2/ND1 distance
- His ND1/NE2 ↔ Asp OD1/OD2 distance
- Key angles
- H-bond geometry assessment
"""

import sys
import json
import glob
import re
from pathlib import Path
import numpy as np
from collections import defaultdict

# Residue numbering
# WT uses PDB numbering: Ser160, Asp206, His237
# ColabFold structures use numbering from 1: Ser131, Asp177, His208
WT_TRIAD = {
    'Ser': 160,
    'Asp': 206,
    'His': 237
}

COLABFOLD_TRIAD = {
    'Ser': 131,
    'Asp': 177,
    'His': 208
}


def parse_pdb(pdb_file):
    """Parse PDB file and extract atom coordinates."""
    atoms = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                key = (res_num, res_name, atom_name)
                atoms[key] = np.array([x, y, z])
    return atoms


def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates."""
    return np.linalg.norm(coord1 - coord2)


def calculate_angle(coord1, coord2, coord3):
    """Calculate angle at coord2 (in degrees)."""
    v1 = coord1 - coord2
    v2 = coord3 - coord2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    return np.degrees(np.arccos(cos_angle))


def find_closest_atom(atoms, res_num, res_name, atom_names):
    """Find the closest atom from a list of possible atom names."""
    candidates = []
    for atom_name in atom_names:
        key = (res_num, res_name, atom_name)
        if key in atoms:
            candidates.append((atom_name, atoms[key]))
    
    if not candidates:
        return None, None
    
    # Return the first found (usually there's only one)
    return candidates[0]


def analyze_triad(atoms, triad_residues, is_wt=False):
    """Analyze catalytic triad geometry."""
    results = {}
    
    ser_res = triad_residues['Ser']
    asp_res = triad_residues['Asp']
    his_res = triad_residues['His']
    
    # Get Ser OG
    ser_og_key = (ser_res, 'SER', 'OG')
    if ser_og_key not in atoms:
        return None
    ser_og = atoms[ser_og_key]
    
    # Get His NE2 and ND1
    his_ne2_key = (his_res, 'HIS', 'NE2')
    his_nd1_key = (his_res, 'HIS', 'ND1')
    
    his_ne2 = atoms.get(his_ne2_key)
    his_nd1 = atoms.get(his_nd1_key)
    
    if his_ne2 is None and his_nd1 is None:
        return None
    
    # Get Asp OD1 and OD2
    asp_od1_key = (asp_res, 'ASP', 'OD1')
    asp_od2_key = (asp_res, 'ASP', 'OD2')
    
    asp_od1 = atoms.get(asp_od1_key)
    asp_od2 = atoms.get(asp_od2_key)
    
    if asp_od1 is None and asp_od2 is None:
        return None
    
    # Calculate Ser OG ↔ His NE2 distance (preferred)
    if his_ne2 is not None:
        ser_his_dist = calculate_distance(ser_og, his_ne2)
        results['Ser_OG_His_NE2_dist'] = ser_his_dist
        results['His_active_N'] = 'NE2'
        his_active = his_ne2
    else:
        # Fallback to ND1
        ser_his_dist = calculate_distance(ser_og, his_nd1)
        results['Ser_OG_His_ND1_dist'] = ser_his_dist
        results['His_active_N'] = 'ND1'
        his_active = his_nd1
    
    # Calculate His ↔ Asp distance (try both OD1 and OD2, use closest)
    if asp_od1 is not None and asp_od2 is not None:
        dist_od1 = calculate_distance(his_active, asp_od1)
        dist_od2 = calculate_distance(his_active, asp_od2)
        
        if dist_od1 < dist_od2:
            results['His_Asp_dist'] = dist_od1
            results['Asp_active_O'] = 'OD1'
            asp_active = asp_od1
        else:
            results['His_Asp_dist'] = dist_od2
            results['Asp_active_O'] = 'OD2'
            asp_active = asp_od2
    elif asp_od1 is not None:
        results['His_Asp_dist'] = calculate_distance(his_active, asp_od1)
        results['Asp_active_O'] = 'OD1'
        asp_active = asp_od1
    elif asp_od2 is not None:
        results['His_Asp_dist'] = calculate_distance(his_active, asp_od2)
        results['Asp_active_O'] = 'OD2'
        asp_active = asp_od2
    else:
        return None
    
    # Calculate angles
    # Get Ser CA for angle calculation
    ser_ca_key = (ser_res, 'SER', 'CA')
    if ser_ca_key in atoms:
        ser_ca = atoms[ser_ca_key]
        # Angle: Ser CA - Ser OG - His N
        angle_ser = calculate_angle(ser_ca, ser_og, his_active)
        results['Angle_Ser_CA_OG_His'] = angle_ser
    
    # Get His CA for angle calculation
    his_ca_key = (his_res, 'HIS', 'CA')
    if his_ca_key in atoms:
        his_ca = atoms[his_ca_key]
        # Angle: His CA - His N - Asp O
        angle_his = calculate_angle(his_ca, his_active, asp_active)
        results['Angle_His_CA_N_Asp'] = angle_his
    
    # Assess H-bond geometry
    # Typical H-bond distances: 2.5-3.5 Å
    # Typical H-bond angles: 120-180° (for donor-H-acceptor)
    
    ser_his_ok = 2.5 <= ser_his_dist <= 3.5
    his_asp_ok = 2.5 <= results['His_Asp_dist'] <= 3.5
    
    results['Ser_His_Hbond_plausible'] = ser_his_ok
    results['His_Asp_Hbond_plausible'] = his_asp_ok
    
    # Overall assessment
    if results['His_Asp_dist'] > 4.0:
        results['Assessment'] = 'Non-functional (Asp too far)'
    elif 2.6 <= results['His_Asp_dist'] <= 3.2:
        results['Assessment'] = 'Functional (good distance)'
    elif 3.2 < results['His_Asp_dist'] <= 4.0:
        results['Assessment'] = 'Marginal (may need optimization)'
    else:
        results['Assessment'] = 'Too close (unlikely)'
    
    return results


def main():
    results_dir = Path('runs/colabfold_predictions_gpu')
    
    all_results = []
    
    # Analyze WT
    wt_pdb = Path('data/structures/5XJH/raw/PETase_raw.pdb')
    if wt_pdb.exists():
        print(f"Analyzing WT: {wt_pdb}")
        atoms = parse_pdb(wt_pdb)
        wt_results = analyze_triad(atoms, WT_TRIAD, is_wt=True)
        if wt_results:
            wt_results['Candidate'] = 'WT'
            wt_results['pLDDT'] = None  # WT doesn't have pLDDT
            wt_results['pTM'] = None
            all_results.append(wt_results)
            print(f"  Ser-OG ↔ His-NE2: {wt_results.get('Ser_OG_His_NE2_dist', wt_results.get('Ser_OG_His_ND1_dist', 'N/A')):.3f} Å")
            print(f"  His-N ↔ Asp-O: {wt_results['His_Asp_dist']:.3f} Å")
            print(f"  Assessment: {wt_results['Assessment']}")
    
    # Analyze all candidates
    candidate_files = sorted(glob.glob(str(results_dir / 'candidate_*_unrelaxed_rank_001_*.pdb')))
    
    # Get pLDDT scores from JSON files
    plddt_scores = {}
    for json_file in glob.glob(str(results_dir / '*.json')):
        if 'rank_001' in json_file:
            match = re.search(r'candidate_(\d+)', json_file)
            if match:
                candidate_num = int(match.group(1))
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'plddt' in data:
                        plddt = data['plddt']
                        if isinstance(plddt, list):
                            avg_plddt = sum(plddt) / len(plddt) if plddt else 0
                        else:
                            avg_plddt = plddt
                        plddt_scores[candidate_num] = avg_plddt
                    
                    ptm = data.get('ptm', 0)
                    if isinstance(ptm, list):
                        ptm = sum(ptm) / len(ptm) if ptm else 0
                    plddt_scores[candidate_num] = (avg_plddt, ptm)
    
    print(f"\nAnalyzing {len(candidate_files)} candidates...")
    
    for pdb_file in candidate_files:
        match = re.search(r'candidate_(\d+)', pdb_file)
        if not match:
            continue
        
        candidate_num = int(match.group(1))
        print(f"Analyzing candidate_{candidate_num}...", end=' ')
        
        atoms = parse_pdb(pdb_file)
        candidate_results = analyze_triad(atoms, COLABFOLD_TRIAD, is_wt=False)
        
        if candidate_results:
            candidate_results['Candidate'] = f'candidate_{candidate_num}'
            if candidate_num in plddt_scores:
                candidate_results['pLDDT'], candidate_results['pTM'] = plddt_scores[candidate_num]
            else:
                candidate_results['pLDDT'] = None
                candidate_results['pTM'] = None
            
            all_results.append(candidate_results)
            print(f"OK - Ser-His: {candidate_results.get('Ser_OG_His_NE2_dist', candidate_results.get('Ser_OG_His_ND1_dist', 'N/A')):.3f} Å, "
                  f"His-Asp: {candidate_results['His_Asp_dist']:.3f} Å")
        else:
            print("FAILED - Missing atoms")
    
    # Sort by pLDDT (WT first, then by pLDDT descending)
    def sort_key(r):
        if r['Candidate'] == 'WT':
            return (0, 0)
        return (1, -(r.get('pLDDT') or 0))
    
    all_results.sort(key=sort_key)
    
    # Write results to CSV
    output_csv = results_dir / 'catalytic_triad_analysis.csv'
    with open(output_csv, 'w') as f:
        # Header
        f.write('Candidate,pLDDT,pTM,Ser_OG_His_N_dist,His_N_Asp_O_dist,His_active_N,Asp_active_O,')
        f.write('Angle_Ser_CA_OG_His,Angle_His_CA_N_Asp,')
        f.write('Ser_His_Hbond_plausible,His_Asp_Hbond_plausible,Assessment\n')
        
        # Data
        for r in all_results:
            ser_his_dist = r.get('Ser_OG_His_NE2_dist') or r.get('Ser_OG_His_ND1_dist')
            if ser_his_dist is None:
                ser_his_str = 'N/A'
            else:
                ser_his_str = f"{ser_his_dist:.3f}"
            
            plddt = r.get('pLDDT')
            plddt_str = f"{plddt:.2f}" if plddt is not None else 'N/A'
            
            ptm = r.get('pTM')
            ptm_str = f"{ptm:.3f}" if ptm is not None else 'N/A'
            
            angle_ser = r.get('Angle_Ser_CA_OG_His')
            angle_ser_str = f"{angle_ser:.1f}" if angle_ser is not None else 'N/A'
            
            angle_his = r.get('Angle_His_CA_N_Asp')
            angle_his_str = f"{angle_his:.1f}" if angle_his is not None else 'N/A'
            
            f.write(f"{r['Candidate']},{plddt_str},{ptm_str},")
            f.write(f"{ser_his_str},")
            f.write(f"{r['His_Asp_dist']:.3f},")
            f.write(f"{r.get('His_active_N', 'N/A')},")
            f.write(f"{r.get('Asp_active_O', 'N/A')},")
            f.write(f"{angle_ser_str},")
            f.write(f"{angle_his_str},")
            f.write(f"{r.get('Ser_His_Hbond_plausible', False)},")
            f.write(f"{r.get('His_Asp_Hbond_plausible', False)},")
            f.write(f"{r.get('Assessment', 'N/A')}\n")
    
    # Write summary markdown
    output_md = results_dir / 'CATALYTIC_TRIAD_ANALYSIS.md'
    with open(output_md, 'w') as f:
        f.write("# Catalytic Triad Geometry Analysis\n\n")
        f.write("Quantitative analysis of Ser-Asp-His catalytic triad geometry for WT and all candidates.\n\n")
        f.write("## Key Metrics\n\n")
        f.write("- **Ser OG ↔ His N distance**: Should be ~2.5-3.5 Å for H-bond\n")
        f.write("- **His N ↔ Asp O distance**: Should be ~2.6-3.2 Å for functional triad\n")
        f.write("  - **> 4.0 Å**: Non-functional (Asp too far)\n")
        f.write("  - **2.6-3.2 Å**: Functional (good distance)\n")
        f.write("  - **3.2-4.0 Å**: Marginal (may need optimization)\n")
        f.write("  - **< 2.6 Å**: Too close (unlikely)\n\n")
        f.write("## Results\n\n")
        f.write("| Candidate | pLDDT | Ser↔His (Å) | His↔Asp (Å) | Assessment |\n")
        f.write("|-----------|-------|-------------|-------------|------------|\n")
        
        for r in all_results:
            ser_his_dist = r.get('Ser_OG_His_NE2_dist') or r.get('Ser_OG_His_ND1_dist') or 'N/A'
            if isinstance(ser_his_dist, float):
                ser_his_str = f"{ser_his_dist:.3f}"
            else:
                ser_his_str = 'N/A'
            
            plddt_str = f"{r.get('pLDDT', 0):.2f}" if r.get('pLDDT') else 'N/A'
            
            f.write(f"| {r['Candidate']} | {plddt_str} | {ser_his_str} | "
                   f"{r['His_Asp_dist']:.3f} | {r.get('Assessment', 'N/A')} |\n")
        
        f.write("\n## Detailed Results\n\n")
        f.write("See `catalytic_triad_analysis.csv` for complete data including angles.\n")
    
    print(f"\n✓ Analysis complete!")
    print(f"  CSV output: {output_csv}")
    print(f"  Markdown output: {output_md}")
    print(f"  Total structures analyzed: {len(all_results)}")


if __name__ == '__main__':
    main()

