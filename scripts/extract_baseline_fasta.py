#!/usr/bin/env python3
"""
Extract and freeze the baseline PETase sequence (5XJH chain A, PDB 30-292, 263 aa).

This creates the canonical baseline FASTA that all ProGen2 modules will consume.
The sequence is treated as immutable for "Design Spec v1.0".

Usage:
    python scripts/extract_baseline_fasta.py [pdb_file] [output_fasta]

Default:
    pdb_file: data/structures/5XJH/raw/PETase_raw.pdb
    output_fasta: data/sequences/wt/baseline_5XJH_30-292.fasta
"""

import sys
import hashlib
from pathlib import Path
from collections import OrderedDict

# Amino acid three-letter to one-letter mapping
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

# Hard constraints from PROGEN2_WORKFLOW.md
PDB_START = 30
PDB_END = 292
EXPECTED_LENGTH = 263
CHAIN_ID = 'A'

# Catalytic triad mapping: PDB position → Repo pose position
# Repo pose = PDB - 29
CATALYTIC_TRIAD = {
    'S': (160, 131),  # PDB 160 → pose 131
    'D': (206, 177),  # PDB 206 → pose 177
    'H': (237, 208),  # PDB 237 → pose 208
}


def extract_baseline_sequence(pdb_file, pdb_start=PDB_START, pdb_end=PDB_END, chain_id=CHAIN_ID):
    """Extract sequence from PDB file for specified chain and residue range."""
    
    pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Parse PDB file directly
    # We'll track residues by their CA atoms (most reliable)
    residues = OrderedDict()  # resnum -> (resname, chain)
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # Parse ATOM line
                # Format: ATOM  serial  name  altLoc  resName  chainID  resSeq  x  y  z  ...
                # Columns: 0-5, 6-11, 12-15, 16, 17-19, 20, 22-25, 30-37, 38-45, 46-53
                try:
                    atom_name = line[12:16].strip()
                    resname = line[17:20].strip()
                    chain = line[21:22].strip()
                    resnum = int(line[22:26].strip())
                    
                    # Only process CA atoms from the target chain
                    if atom_name == 'CA' and chain == chain_id:
                        if pdb_start <= resnum <= pdb_end:
                            if resnum not in residues:
                                residues[resnum] = resname
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
    
    if not residues:
        raise ValueError(f"No residues found for chain {chain_id} in range {pdb_start}-{pdb_end}")
    
    # Convert to sequence
    sequence = []
    residue_positions = []
    
    for resnum in sorted(residues.keys()):
        resname = residues[resnum]
        if resname in AA_3_TO_1:
            aa = AA_3_TO_1[resname]
            sequence.append(aa)
            residue_positions.append((resnum, aa))
        else:
            raise ValueError(f"Unknown residue type at PDB position {resnum}: {resname}")
    
    seq_string = ''.join(sequence)
    
    # Verify length
    if len(seq_string) != EXPECTED_LENGTH:
        raise ValueError(
            f"Sequence length mismatch: expected {EXPECTED_LENGTH} aa, got {len(seq_string)} aa. "
            f"Found residues: {residue_positions[0] if residue_positions else 'none'} to {residue_positions[-1] if residue_positions else 'none'}"
        )
    
    # Verify catalytic triad positions
    # Map PDB positions to sequence indices (0-based)
    pdb_to_seq_idx = {pos: idx for idx, (pos, _) in enumerate(residue_positions)}
    
    errors = []
    for aa_name, (pdb_pos, pose_pos) in CATALYTIC_TRIAD.items():
        if pdb_pos not in pdb_to_seq_idx:
            errors.append(f"Catalytic {aa_name} at PDB {pdb_pos} (pose {pose_pos}) not found in extracted sequence")
        else:
            seq_idx = pdb_to_seq_idx[pdb_pos]
            actual_aa = seq_string[seq_idx]
            if actual_aa != aa_name:
                errors.append(
                    f"Catalytic triad mismatch at PDB {pdb_pos} (pose {pose_pos}): "
                    f"expected {aa_name}, got {actual_aa}"
                )
    
    if errors:
        raise ValueError("Catalytic triad verification failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return seq_string, residue_positions


def write_baseline_fasta(sequence, output_file, pdb_file=None):
    """Write baseline FASTA with metadata header."""
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute SHA256 hash for reproducibility
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()
    
    # Write FASTA with metadata in header
    with open(output_file, 'w') as f:
        header = (
            f">baseline_5XJH_chainA_PDB30-292 "
            f"length={len(sequence)} "
            f"sha256={seq_hash[:16]} "
            f"source=PDB_5XJH_chainA_residues_{PDB_START}-{PDB_END}"
        )
        f.write(f"{header}\n")
        f.write(f"{sequence}\n")
    
    return seq_hash


def main():
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
    else:
        pdb_file = "data/structures/5XJH/raw/PETase_raw.pdb"
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = "data/sequences/wt/baseline_5XJH_30-292.fasta"
    
    print(f"Extracting baseline sequence from {pdb_file}")
    print(f"  Chain: {CHAIN_ID}")
    print(f"  PDB range: {PDB_START}-{PDB_END}")
    print(f"  Expected length: {EXPECTED_LENGTH} aa")
    print()
    
    try:
        sequence, residue_positions = extract_baseline_sequence(pdb_file)
        
        print(f"✓ Extracted sequence: {len(sequence)} aa")
        print(f"  PDB range covered: {residue_positions[0][0]} to {residue_positions[-1][0]}")
        print()
        
        # Verify catalytic triad
        print("Verifying catalytic triad:")
        pdb_to_seq_idx = {pos: idx for idx, (pos, _) in enumerate(residue_positions)}
        for aa_name, (pdb_pos, pose_pos) in CATALYTIC_TRIAD.items():
            seq_idx = pdb_to_seq_idx[pdb_pos]
            actual_aa = sequence[seq_idx]
            status = "✓" if actual_aa == aa_name else "✗"
            print(f"  {status} PDB {pdb_pos} (pose {pose_pos}): {actual_aa} (expected {aa_name})")
        print()
        
        # Write FASTA
        seq_hash = write_baseline_fasta(sequence, output_file, pdb_file)
        
        print(f"✓ Saved baseline FASTA: {output_file}")
        print(f"  Sequence hash (SHA256): {seq_hash}")
        print(f"  First 20 aa: {sequence[:20]}...")
        print(f"  Last 20 aa: ...{sequence[-20:]}")
        print()
        print("This baseline is now frozen for Design Spec v1.0")
        print("All ProGen2 modules should consume this file as the canonical baseline.")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

