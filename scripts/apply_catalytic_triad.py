#!/usr/bin/env python3
"""
Apply catalytic triad mutations to sequences that pass other gates.

This mutates positions 131, 177, 208 to S, D, H respectively for sequences
that have passed length and other gates but failed the hard locks gate.

Usage:
    python scripts/apply_catalytic_triad.py <input_fasta> <output_fasta> [--dry-run]
"""

import sys
from pathlib import Path
import argparse

# Hard locks: positions that must be mutated
HARD_LOCKS = {
    131: 'S',  # PDB 160
    177: 'D',  # PDB 206
    208: 'H',  # PDB 237
}


def mutate_catalytic_triad(sequence, positions=HARD_LOCKS):
    """
    Mutate sequence to have catalytic triad at specified positions.
    
    Args:
        sequence: Amino acid sequence (string)
        positions: Dict of {position: required_aa}
    
    Returns:
        Tuple of (mutated_sequence, mutations_applied)
        mutations_applied is a list of (pos, old_aa, new_aa)
    """
    if len(sequence) != 263:
        raise ValueError(f"Sequence length must be 263 aa, got {len(sequence)}")
    
    seq_list = list(sequence)
    mutations = []
    
    for pos, required_aa in positions.items():
        # Convert to 0-based index
        idx = pos - 1
        
        if idx >= len(seq_list):
            raise ValueError(f"Position {pos} is out of range for sequence of length {len(sequence)}")
        
        old_aa = seq_list[idx]
        if old_aa != required_aa:
            seq_list[idx] = required_aa
            mutations.append((pos, old_aa, required_aa))
    
    return ''.join(seq_list), mutations


def main():
    parser = argparse.ArgumentParser(
        description="Apply catalytic triad mutations to sequences"
    )
    parser.add_argument("input_fasta", help="Input FASTA file (sequences that passed length gate)")
    parser.add_argument("output_fasta", help="Output FASTA file (with catalytic triad applied)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be mutated without writing")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_fasta)
    output_path = Path(args.output_fasta)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Read sequences (simple FASTA parser)
    sequences = []
    with open(input_path) as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:].split()[0]  # Get ID (first word after >)
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))
    
    print(f"Reading {len(sequences)} sequences from {input_path}")
    print()
    
    # Apply mutations
    mutated_sequences = []
    total_mutations = 0
    
    for seq_id, sequence in sequences:
        try:
            mutated_seq, mutations = mutate_catalytic_triad(sequence)
            
            if mutations:
                print(f"{seq_id}:")
                for pos, old_aa, new_aa in mutations:
                    print(f"  Position {pos}: {old_aa} → {new_aa}")
                total_mutations += len(mutations)
            else:
                print(f"{seq_id}: Already has catalytic triad (no mutations needed)")
            
            mutated_sequences.append((seq_id, mutated_seq, mutations))
            
        except ValueError as e:
            print(f"{seq_id}: ERROR - {e}", file=sys.stderr)
            continue
    
    print()
    print(f"Total mutations applied: {total_mutations}")
    print(f"Sequences processed: {len(mutated_sequences)}")
    print()
    
    if args.dry_run:
        print("DRY RUN: No output file written")
        return
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for seq_id, mutated_seq, mutations in mutated_sequences:
            # Add mutation info to header
            if mutations:
                mut_str = ";".join(f"{pos}{old}→{new}" for pos, old, new in mutations)
                header = f">{seq_id} [mutated: {mut_str}]"
            else:
                header = f">{seq_id} [catalytic_triad_present]"
            
            f.write(f"{header}\n")
            f.write(f"{mutated_seq}\n")
    
    print(f"✓ Written {len(mutated_sequences)} sequences to {output_path}")
    print()
    print("These sequences now have the catalytic triad and will pass the hard locks gate.")


if __name__ == '__main__':
    main()

