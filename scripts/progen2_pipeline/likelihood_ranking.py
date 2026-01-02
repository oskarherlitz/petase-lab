#!/usr/bin/env python3
"""
ProGen2 Likelihood Ranking Module.

Uses ProGen2 likelihood scoring to rank sequences as a "naturalness prior."
Implements diversity-preserving selection across prompt×lane×bucket groups.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import json
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def compute_likelihood(
    sequence: str,
    model: str,
    progen2_dir: Path,
    device: str = "cpu"
) -> float:
    """
    Compute ProGen2 log-likelihood for a sequence.
    
    Args:
        sequence: Amino acid sequence
        model: ProGen2 model name
        progen2_dir: Path to ProGen2 directory
        device: Device to use
    
    Returns:
        Log-likelihood score (higher = more natural)
    """
    # Check ProGen2 setup
    likelihood_script = progen2_dir / "likelihood.py"
    if not likelihood_script.exists():
        raise FileNotFoundError(f"ProGen2 likelihood.py not found at {likelihood_script}")
    
    checkpoint_dir = progen2_dir / "checkpoints" / model
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    # Ensure models/progen structure exists
    models_dir = progen2_dir / "models"
    if not models_dir.exists():
        models_dir.mkdir()
        try:
            os.symlink(progen2_dir / "progen", models_dir / "progen")
        except OSError:
            pass
    
    # Format sequence for ProGen2 (add control tokens)
    context = f"1{sequence}2"
    
    # Run likelihood script
    cmd = [
        sys.executable,
        str(likelihood_script),
        "--model", model,
        "--context", context,
        "--device", device,
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(progen2_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=60  # Should be fast for single sequence
        )
        
        # Parse output - look for "ll_mean="
        for line in result.stdout.split('\n'):
            if 'll_mean=' in line:
                try:
                    ll_str = line.split('ll_mean=')[1].strip()
                    return float(ll_str)
                except (ValueError, IndexError):
                    pass
        
        # Fallback: try to parse from stderr
        for line in result.stderr.split('\n'):
            if 'll_mean=' in line:
                try:
                    ll_str = line.split('ll_mean=')[1].strip()
                    return float(ll_str)
                except (ValueError, IndexError):
                    pass
        
        raise ValueError(f"Could not parse likelihood from output: {result.stdout[:500]}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Likelihood computation timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Likelihood computation failed: {e.stderr[:500]}"
        )


def rank_with_diversity_preservation(
    sequences: List[str],
    metadata: List[Dict],
    baseline_seq: str,
    top_quantile: float = 0.5
) -> Tuple[List[str], List[Dict], List[Dict]]:
    """
    Rank sequences with diversity-preserving selection.
    
    Groups sequences by (prompt_length, lane, identity_bucket) and selects
    top quantile from each group.
    
    Args:
        sequences: List of sequences
        metadata: List of metadata dicts (must include prompt_length, lane, identity_bucket)
        baseline_seq: Baseline sequence for identity computation
        top_quantile: Fraction to keep from each group (0.5 = top 50%)
    
    Returns:
        (selected_sequences, selected_metadata, ranking_df)
    """
    if len(sequences) != len(metadata):
        raise ValueError(f"Sequences and metadata must have same length: {len(sequences)} vs {len(metadata)}")
    
    # Group by (prompt_length, lane, identity_bucket)
    groups = defaultdict(list)
    
    for i, (seq, meta) in enumerate(zip(sequences, metadata)):
        # Compute identity if not in metadata
        if "identity" not in meta:
            # Import here to avoid circular import
            from progen2_pipeline.filters import compute_identity
            meta["identity"] = compute_identity(seq, baseline_seq)
        
        # Assign identity bucket
        if "identity_bucket" not in meta:
            identity = meta["identity"]
            if 0.75 <= identity <= 0.95:
                meta["identity_bucket"] = "Near"
            elif 0.55 <= identity <= 0.75:
                meta["identity_bucket"] = "Explore"
            else:
                meta["identity_bucket"] = "Other"
        
        # Create group key
        prompt_len = meta.get("prompt_length", "unknown")
        lane = meta.get("lane", "unknown")
        bucket = meta.get("identity_bucket", "unknown")
        group_key = (prompt_len, lane, bucket)
        
        groups[group_key].append((i, seq, meta))
    
    # Select top quantile from each group
    selected_indices = set()
    
    for group_key, group_items in groups.items():
        # Sort by likelihood (if available) or identity
        group_items.sort(
            key=lambda x: x[2].get("likelihood", x[2].get("identity", 0)),
            reverse=True
        )
        
        # Take top quantile
        n_keep = max(1, int(len(group_items) * top_quantile))
        for idx, seq, meta in group_items[:n_keep]:
            selected_indices.add(idx)
    
    # Build selected lists
    selected_sequences = [sequences[i] for i in sorted(selected_indices)]
    selected_metadata = [metadata[i] for i in sorted(selected_indices)]
    
    # Create ranking data
    ranking_data = []
    for i, (seq, meta) in enumerate(zip(selected_sequences, selected_metadata)):
        ranking_data.append({
            "sequence_id": f"candidate_{i+1}",
            "sequence": seq,
            "prompt_length": meta.get("prompt_length", "unknown"),
            "lane": meta.get("lane", "unknown"),
            "identity_bucket": meta.get("identity_bucket", "unknown"),
            "identity": meta.get("identity", 0.0),
            "likelihood": meta.get("likelihood", None),
        })
    
    # Sort by likelihood, then identity
    ranking_data.sort(
        key=lambda x: (x["likelihood"] if x["likelihood"] is not None else float('-inf'), x["identity"]),
        reverse=True
    )
    
    # Return ranking data as list of dicts (pandas optional for CSV writing)
    return selected_sequences, selected_metadata, ranking_data


def _write_csv_simple(path: Path, data: List[Dict]):
    """Simple CSV writer without pandas."""
    with open(path, 'w') as f:
        if not data:
            return
        # Write header
        headers = list(data[0].keys())
        f.write(",".join(headers) + "\n")
        # Write rows
        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            # Escape commas in values
            values = [f'"{v}"' if ',' in str(v) else str(v) for v in values]
            f.write(",".join(values) + "\n")


def compute_likelihoods_batch(
    sequences: List[str],
    model: str,
    progen2_dir: Path,
    device: str = "cpu",
    verbose: bool = True
) -> List[float]:
    """
    Compute likelihoods for a batch of sequences.
    
    Args:
        sequences: List of sequences
        model: ProGen2 model name
        progen2_dir: Path to ProGen2 directory
        device: Device to use
        verbose: Print progress
    
    Returns:
        List of likelihood scores
    """
    likelihoods = []
    
    for i, seq in enumerate(sequences):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Computing likelihood {i+1}/{len(sequences)}...")
        
        try:
            ll = compute_likelihood(seq, model, progen2_dir, device)
            likelihoods.append(ll)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to compute likelihood for sequence {i+1}: {e}")
            likelihoods.append(None)
    
    return likelihoods


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute ProGen2 likelihoods and rank sequences")
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("baseline_fasta", help="Baseline FASTA file")
    parser.add_argument("metadata_json", help="Metadata JSON file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--model", default="progen2-small", help="ProGen2 model")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--top-quantile", type=float, default=0.5, help="Top quantile to keep per group")
    
    args = parser.parse_args()
    
    # Load sequences
    sequences = []
    seq_ids = []
    with open(args.input_fasta) as f:
        current_id = None
        for line in f:
            if line.startswith('>'):
                current_id = line[1:].strip().split()[0]
                seq_ids.append(current_id)
            else:
                seq = line.strip()
                if seq and current_id:
                    sequences.append(seq)
    
    # Load metadata
    with open(args.metadata_json) as f:
        metadata = json.load(f)
    
    # Load baseline
    with open(args.baseline_fasta) as f:
        baseline_lines = f.readlines()
        baseline_seq = ''.join(line.strip() for line in baseline_lines[1:])
    
    # Compute likelihoods
    print(f"Computing likelihoods for {len(sequences)} sequences...")
    repo_root = Path(__file__).parent.parent.parent
    progen2_dir = repo_root / "external" / "progen2"
    
    likelihoods = compute_likelihoods_batch(sequences, args.model, progen2_dir, args.device)
    
    # Add likelihoods to metadata
    for i, ll in enumerate(likelihoods):
        if i < len(metadata):
            metadata[i]["likelihood"] = ll
    
    # Rank with diversity preservation
    print("Ranking sequences with diversity preservation...")
    selected_seqs, selected_meta, ranking_df = rank_with_diversity_preservation(
        sequences, metadata, baseline_seq, args.top_quantile
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ranking CSV
    ranking_csv = output_dir / "candidates.ranked.csv"
    _write_csv_simple(ranking_csv, ranking_df)
    
    with open(output_dir / "candidates.ranked.fasta", 'w') as f:
        for i, seq in enumerate(selected_seqs):
            f.write(f">candidate_{i+1}\n")
            f.write(f"{seq}\n")
    
    with open(output_dir / "metadata.ranked.json", 'w') as f:
        json.dump(selected_meta, f, indent=2)
    
    print(f"✓ Ranked and selected {len(selected_seqs)} sequences")
    print(f"✓ Saved ranking to {output_dir / 'candidates.ranked.csv'}")

