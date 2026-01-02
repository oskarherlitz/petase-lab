#!/usr/bin/env python3
"""
Sequence Filter Pipeline Module.

Implements all sequence-only gates (C0-C6) from PROGEN2_WORKFLOW.md.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set
import json
from collections import defaultdict


# Hard constraints
EXPECTED_LENGTH = 263
HARD_LOCKS = {
    131: 'S',  # PDB 160
    177: 'D',  # PDB 206
    208: 'H',  # PDB 237
}

# Identity bucket definitions
IDENTITY_BUCKETS = {
    "Near": (0.75, 0.95),      # 75-95% identity
    "Explore": (0.55, 0.75),   # 55-75% identity
}

# Uniqueness threshold
MIN_PAIRWISE_DIFF = 5  # Minimum number of different residues


def gate_token_cleanup(sequences: List[str]) -> Tuple[List[str], List[str]]:
    """Gate C0: Token + alphabet cleanup."""
    passed = []
    removed = []
    
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    for seq in sequences:
        if all(c.upper() in valid_aa for c in seq):
            passed.append(seq)
        else:
            invalid_chars = set(c for c in seq if c.upper() not in valid_aa)
            removed.append(f"Invalid chars: {invalid_chars}")
    
    return passed, removed


def gate_length(sequences: List[str]) -> Tuple[List[str], List[str]]:
    """Gate C1: Length check (must be exactly 263 aa)."""
    passed = []
    removed = []
    
    for seq in sequences:
        if len(seq) == EXPECTED_LENGTH:
            passed.append(seq)
        else:
            removed.append(f"Length {len(seq)} (expected {EXPECTED_LENGTH})")
    
    return passed, removed


def gate_hard_locks(sequences: List[str]) -> Tuple[List[str], List[str]]:
    """Gate C2: Hard-lock gate (positions 131, 177, 208 must be S, D, H)."""
    passed = []
    removed = []
    
    for seq in sequences:
        if len(seq) != EXPECTED_LENGTH:
            removed.append("Length mismatch")
            continue
        
        errors = []
        for pos, required_aa in HARD_LOCKS.items():
            idx = pos - 1
            if idx >= len(seq):
                errors.append(f"Pos {pos} out of range")
            elif seq[idx] != required_aa:
                errors.append(f"Pos {pos}: expected {required_aa}, got {seq[idx]}")
        
        if errors:
            removed.append("; ".join(errors))
        else:
            passed.append(seq)
    
    return passed, removed


def compute_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def gate_identity_buckets(
    sequences: List[str],
    baseline_seq: str,
    buckets: Dict[str, Tuple[float, float]] = IDENTITY_BUCKETS
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Gate C3: Family plausibility (identity buckets).
    
    Returns:
        (passed_sequences, removed_reasons, bucket_assignments)
        bucket_assignments: {bucket_name: [sequence_indices]}
    """
    passed = []
    removed = []
    bucket_assignments = defaultdict(list)
    
    for i, seq in enumerate(sequences):
        identity = compute_identity(seq, baseline_seq)
        
        # Assign to bucket
        assigned = False
        for bucket_name, (min_id, max_id) in buckets.items():
            if min_id <= identity <= max_id:
                passed.append(seq)
                bucket_assignments[bucket_name].append(i)
                assigned = True
                break
        
        if not assigned:
            if identity > buckets["Near"][1]:
                removed.append(f"Identity {identity:.2%} (too high, near-clone)")
            elif identity < buckets["Explore"][0]:
                removed.append(f"Identity {identity:.2%} (too low, out-of-family)")
            else:
                removed.append(f"Identity {identity:.2%} (no bucket match)")
    
    return passed, removed, dict(bucket_assignments)


def gate_uniqueness(
    sequences: List[str],
    min_diff: int = MIN_PAIRWISE_DIFF
) -> Tuple[List[str], List[str]]:
    """
    Gate C4: Uniqueness gate.
    
    Removes sequences that are too similar to already-passed sequences.
    """
    passed = []
    removed = []
    
    for seq in sequences:
        is_unique = True
        for existing in passed:
            # Count differences
            if len(seq) != len(existing):
                continue
            
            diff_count = sum(1 for a, b in zip(seq, existing) if a != b)
            if diff_count < min_diff:
                removed.append(f"Too similar to existing (diff: {diff_count} < {min_diff})")
                is_unique = False
                break
        
        if is_unique:
            passed.append(seq)
    
    return passed, removed


def gate_composition(sequences: List[str]) -> Tuple[List[str], List[str]]:
    """
    Gate C5: Composition sanity gate.
    
    Rejects sequences with:
    - Long single-AA runs (>10 consecutive)
    - Extreme low-complexity stretches
    """
    passed = []
    removed = []
    
    for seq in sequences:
        # Check for long single-AA runs
        max_run = 1
        current_run = 1
        current_aa = seq[0] if seq else None
        
        for aa in seq[1:]:
            if aa == current_aa:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
                current_aa = aa
        
        if max_run > 8:  # Stricter: was 10, now 8
            removed.append(f"Long single-AA run: {max_run} consecutive {current_aa}")
            continue
        
        # Check for low-complexity (stricter: >40% of sequence is top 3 AAs, was 50%)
        aa_counts = {}
        for aa in seq:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        sorted_counts = sorted(aa_counts.values(), reverse=True)
        top_3_fraction = sum(sorted_counts[:3]) / len(seq)
        
        if top_3_fraction > 0.40:  # Stricter: was 0.5, now 0.4
            removed.append(f"Low complexity: top 3 AAs = {top_3_fraction:.1%}")
            continue
        
        passed.append(seq)
    
    return passed, removed


def run_filter_pipeline(
    sequences: List[str],
    baseline_seq: str,
    metadata: List[Dict] = None
) -> Tuple[List[str], Dict, List[Dict]]:
    """
    Run complete filter pipeline (gates C0-C5).
    
    Args:
        sequences: List of sequences to filter
        baseline_seq: Baseline sequence for identity computation
        metadata: Optional list of metadata dicts (one per sequence)
    
    Returns:
        (filtered_sequences, filter_report, filtered_metadata)
    """
    if metadata is None:
        metadata = [{}] * len(sequences)
    
    filter_report = {
        "gate_C0_token_cleanup": {"count_in": len(sequences), "count_out": 0, "removed": []},
        "gate_C1_length": {"count_in": 0, "count_out": 0, "removed": []},
        "gate_C2_hard_locks": {"count_in": 0, "count_out": 0, "removed": []},
        "gate_C3_identity_buckets": {"count_in": 0, "count_out": 0, "removed": []},
        "gate_C4_uniqueness": {"count_in": 0, "count_out": 0, "removed": []},
        "gate_C5_composition": {"count_in": 0, "count_out": 0, "removed": []},
    }
    
    current_seqs = sequences
    current_meta = metadata
    
    # Gate C0: Token cleanup
    passed, removed = gate_token_cleanup(current_seqs)
    filter_report["gate_C0_token_cleanup"]["count_out"] = len(passed)
    filter_report["gate_C0_token_cleanup"]["removed"] = removed[:10]  # First 10 examples
    current_seqs = passed
    current_meta = [current_meta[i] for i in range(len(current_seqs)) if i < len(current_meta)]
    
    # Gate C1: Length
    passed, removed = gate_length(current_seqs)
    filter_report["gate_C1_length"]["count_in"] = len(current_seqs)
    filter_report["gate_C1_length"]["count_out"] = len(passed)
    filter_report["gate_C1_length"]["removed"] = removed[:10]
    current_seqs = passed
    current_meta = [current_meta[i] for i in range(len(current_seqs)) if i < len(current_meta)]
    
    # Gate C2: Hard locks (will fail for most - that's expected)
    # Note: We keep sequences that passed length for mutation, but also track natural passes
    passed, removed = gate_hard_locks(current_seqs)
    filter_report["gate_C2_hard_locks"]["count_in"] = len(current_seqs)
    filter_report["gate_C2_hard_locks"]["count_out"] = len(passed)
    filter_report["gate_C2_hard_locks"]["removed"] = removed[:10]
    # Keep length-pass sequences for mutation
    length_pass_seqs = current_seqs.copy()
    length_pass_meta = current_meta.copy()
    # Continue with sequences that naturally passed hard locks
    current_seqs = passed
    current_meta = [current_meta[i] for i in range(len(current_seqs)) if i < len(current_meta)]
    
    # Gate C3: Identity buckets
    passed, removed, bucket_assignments = gate_identity_buckets(current_seqs, baseline_seq)
    filter_report["gate_C3_identity_buckets"]["count_in"] = len(current_seqs)
    filter_report["gate_C3_identity_buckets"]["count_out"] = len(passed)
    filter_report["gate_C3_identity_buckets"]["removed"] = removed[:10]
    filter_report["gate_C3_identity_buckets"]["bucket_assignments"] = {
        bucket: len(indices) for bucket, indices in bucket_assignments.items()
    }
    current_seqs = passed
    current_meta = [current_meta[i] for i in range(len(current_seqs)) if i < len(current_meta)]
    
    # Gate C4: Uniqueness
    passed, removed = gate_uniqueness(current_seqs)
    filter_report["gate_C4_uniqueness"]["count_in"] = len(current_seqs)
    filter_report["gate_C4_uniqueness"]["count_out"] = len(passed)
    filter_report["gate_C4_uniqueness"]["removed"] = removed[:10]
    current_seqs = passed
    current_meta = [current_meta[i] for i in range(len(current_seqs)) if i < len(current_meta)]
    
    # Gate C5: Composition
    passed, removed = gate_composition(current_seqs)
    filter_report["gate_C5_composition"]["count_in"] = len(current_seqs)
    filter_report["gate_C5_composition"]["count_out"] = len(passed)
    filter_report["gate_C5_composition"]["removed"] = removed[:10]
    current_seqs = passed
    current_meta = [current_meta[i] for i in range(len(current_seqs)) if i < len(current_meta)]
    
    return current_seqs, filter_report, current_meta, length_pass_seqs, length_pass_meta


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sequence filter pipeline")
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("baseline_fasta", help="Baseline FASTA file")
    parser.add_argument("output_dir", help="Output directory")
    
    args = parser.parse_args()
    
    # Load sequences
    sequences = []
    with open(args.input_fasta) as f:
        for line in f:
            if not line.startswith('>'):
                seq = line.strip()
                if seq:
                    sequences.append(seq)
    
    # Load baseline
    with open(args.baseline_fasta) as f:
        baseline_lines = f.readlines()
        baseline_seq = ''.join(line.strip() for line in baseline_lines[1:])
    
    # Run filters
    filtered, report, meta, length_pass, length_pass_meta = run_filter_pipeline(
        sequences, baseline_seq
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save filter report
    with open(output_dir / "filter_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save length-pass sequences (for mutation)
    with open(output_dir / "length_pass.fasta", 'w') as f:
        for i, seq in enumerate(length_pass):
            f.write(f">length_pass_{i+1}\n")
            f.write(f"{seq}\n")
    
    # Save filtered sequences
    with open(output_dir / "filtered.fasta", 'w') as f:
        for i, seq in enumerate(filtered):
            f.write(f">filtered_{i+1}\n")
            f.write(f"{seq}\n")
    
    print(f"Filtered {len(sequences)} → {len(filtered)} sequences")
    print(f"Length-pass sequences (for mutation): {len(length_pass)}")

