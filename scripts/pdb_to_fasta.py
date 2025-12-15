#!/usr/bin/env python3
"""
Convert PDB file to FASTA sequence for ColabFold/AlphaFold prediction.

Usage:
    python scripts/pdb_to_fasta.py <pdb_file> [output_fasta]

Example:
    python scripts/pdb_to_fasta.py runs/2024-11-10_fastdesign/outputs/design_001.pdb
"""

import sys
from pathlib import Path
from Bio import PDB
from Bio.SeqUtils import seq1

def pdb_to_fasta(pdb_file, output_file=None):
    """Extract sequence from PDB file and write to FASTA."""
    
    pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = pdb_file.with_suffix('.fasta')
    else:
        output_file = Path(output_file)
    
    # Parse PDB
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Extract sequences for each chain
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residues = []
            
            for residue in chain:
                # Only include standard amino acids
                if residue.id[0] == ' ':  # Standard residue (not heteroatom)
                    try:
                        aa = seq1(residue.get_resname())
                        if aa:  # Valid amino acid
                            residues.append(aa)
                    except KeyError:
                        continue  # Skip non-standard residues
            
            if residues:
                sequences[chain_id] = ''.join(residues)
    
    # Write FASTA
    with open(output_file, 'w') as f:
        for chain_id, seq in sequences.items():
            # Use PDB filename as sequence name
            seq_name = f"{pdb_file.stem}_chain{chain_id}"
            f.write(f">{seq_name}\n")
            f.write(f"{seq}\n")
    
    print(f"✓ Converted {pdb_file} to {output_file}")
    print(f"  Chains found: {', '.join(sequences.keys())}")
    print(f"  Sequence lengths: {', '.join(f'{c}:{len(s)}' for c, s in sequences.items())}")
    
    return output_file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        pdb_to_fasta(pdb_file, output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

