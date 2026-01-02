#!/usr/bin/env python3
"""
Main ProGen2 Pipeline Orchestrator.

Runs the complete pipeline from prompt building through filtering and ranking.
Implements the workflow from PROGEN2_WORKFLOW.md with post-generation mutation.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import pipeline modules
sys.path.insert(0, str(Path(__file__).parent))
from progen2_pipeline import prompt_builder
from progen2_pipeline import generation
from progen2_pipeline import filters
from progen2_pipeline import likelihood_ranking

# Import mutation script (will call as subprocess)
# import apply_catalytic_triad  # Not needed, we call it as subprocess


# Configuration
EXPECTED_LENGTH = 263
# Default prompt schedule.
# You can override per run with --prompt-lengths 100,130,150 (comma-separated).
PROMPT_LENGTHS = [100, 130, 150]
SAMPLING_LANES = ["Conservative", "Exploratory"]


def load_baseline_fasta(baseline_path: Path) -> str:
    """Load baseline FASTA sequence."""
    with open(baseline_path, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])
    return sequence


def parse_sequence_metadata(normalized_fasta: Path) -> list:
    """
    Parse metadata from normalized FASTA headers.
    
    Headers are like: >seq_1_lane_Conservative_prompt_50aa
    """
    metadata = []
    
    with open(normalized_fasta) as f:
        for line in f:
            if line.startswith('>'):
                header = line[1:].strip()
                parts = header.split('_')
                
                # Parse: seq_1_lane_Conservative_prompt_50aa
                meta = {}
                for i, part in enumerate(parts):
                    if part == "lane" and i + 1 < len(parts):
                        meta["lane"] = parts[i + 1]
                    elif part == "prompt" and i + 1 < len(parts):
                        prompt_str = parts[i + 1]
                        if prompt_str.endswith("aa"):
                            meta["prompt_length"] = int(prompt_str[:-2])
                
                metadata.append(meta)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ProGen2 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/run_progen2_pipeline.py run_20251230_progen2_small_r1_test \\
    --num-samples 50 \\
    --model progen2-small
        """
    )
    parser.add_argument("run_id", help="Run ID (e.g., run_20251230_progen2_small_r1_test)")
    parser.add_argument("--baseline-fasta", default="data/sequences/wt/baseline_5XJH_30-292.fasta",
                        help="Path to baseline FASTA")
    parser.add_argument("--model", default="progen2-small", help="ProGen2 model")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples per (prompt, lane) combination")
    parser.add_argument("--max-length", type=int, default=264,
                        help="Max sequence length (264 to get 263 aa)")
    parser.add_argument("--rng-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--skip-likelihood", action="store_true",
                        help="Skip likelihood computation (faster, but no ranking)")
    parser.add_argument("--top-quantile", type=float, default=0.5,
                        help="Top quantile to keep per group in likelihood ranking")
    parser.add_argument("--relax-identity", action="store_true",
                        help="Relax identity thresholds for testing (Explore: 30-50%%, Near: 50-95%%)")
    parser.add_argument(
        "--prompt-lengths",
        default="",
        help="Comma-separated prompt lengths in amino acids (e.g. 100,130,150). Empty = use defaults.",
    )
    
    args = parser.parse_args()

    prompt_lengths = PROMPT_LENGTHS
    if args.prompt_lengths.strip():
        try:
            prompt_lengths = [int(x.strip()) for x in args.prompt_lengths.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"Invalid --prompt-lengths value: {args.prompt_lengths!r} (expected comma-separated ints)")
        if not prompt_lengths:
            raise SystemExit("Invalid --prompt-lengths: no lengths parsed")
        if any(x <= 0 for x in prompt_lengths):
            raise SystemExit(f"Invalid --prompt-lengths: all lengths must be positive, got {prompt_lengths}")
    
    repo_root = Path(__file__).parent.parent
    run_dir = repo_root / "runs" / args.run_id
    
    # Check if run exists, if not create it
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        print("Creating run directory...")
        from create_progen2_run import create_run_folder
        create_run_folder(args.run_id, args.model, round_num=1, tag="")
        run_dir = repo_root / "runs" / args.run_id
    
    baseline_path = repo_root / args.baseline_fasta
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline FASTA not found: {baseline_path}")
    
    progen2_dir = repo_root / "external" / "progen2"
    
    print("=" * 70)
    print("ProGen2 Full Pipeline")
    print("=" * 70)
    print(f"Run ID: {args.run_id}")
    print(f"Model: {args.model}")
    print(f"Prompt lengths: {prompt_lengths} aa")
    print(f"Sampling lanes: {SAMPLING_LANES}")
    print(f"Samples per (prompt, lane): {args.num_samples}")
    print(f"Total expected sequences: {len(prompt_lengths) * len(SAMPLING_LANES) * args.num_samples}")
    print()
    
    # Stage A: Build prompts
    print("Stage A: Building prompts...")
    baseline_seq = load_baseline_fasta(baseline_path)
    prompts = prompt_builder.build_prompts(baseline_seq, prompt_lengths)
    prompt_builder.save_prompts(prompts, run_dir / "prompts", baseline_path)
    print(f"✓ Built {len(prompts)} prompts")
    print()
    
    # Stage B: Generate sequences
    print("Stage B: Running ProGen2 generation...")
    all_sequences = generation.generate_all_lanes(
        prompts=prompts,
        num_samples_per_prompt_lane=args.num_samples,
        model=args.model,
        max_length=args.max_length,
        rng_seed=args.rng_seed,
        progen2_dir=progen2_dir,
        output_dir=run_dir / "generated",
        device=args.device
    )
    
    total_generated = sum(
        len(seqs) for lane_seqs in all_sequences.values()
        for seqs in lane_seqs.values()
    )
    print(f"✓ Generated {total_generated} total sequences")
    print()
    
    # Stage C: Filter sequences
    print("Stage C: Running sequence filters...")
    
    # Load normalized sequences
    normalized_fasta = run_dir / "generated" / "normalized.fasta"
    sequences = []
    with open(normalized_fasta) as f:
        for line in f:
            if not line.startswith('>'):
                seq = line.strip()
                if seq:
                    sequences.append(seq)
    
    # Parse metadata
    metadata = parse_sequence_metadata(normalized_fasta)
    # Pad metadata if needed
    while len(metadata) < len(sequences):
        metadata.append({})
    
    # Run filter pipeline
    filtered_seqs, filter_report, filtered_meta, length_pass_seqs, length_pass_meta = \
        filters.run_filter_pipeline(sequences, baseline_seq, metadata)
    
    # Save length-pass sequences (for mutation)
    length_pass_fasta = run_dir / "candidates" / "candidates.length_pass.fasta"
    length_pass_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(length_pass_fasta, 'w') as f:
        for i, seq in enumerate(length_pass_seqs):
            f.write(f">length_pass_{i+1}\n")
            f.write(f"{seq}\n")
    
    # Save filter report
    with open(run_dir / "filters" / "filter_report.json", 'w') as f:
        json.dump(filter_report, f, indent=2)
    
    print(f"✓ Filtered {len(sequences)} → {len(filtered_seqs)} sequences (natural hard locks)")
    print(f"✓ {len(length_pass_seqs)} sequences passed length gate (ready for mutation)")
    print()
    
    # Apply catalytic triad mutations
    print("Stage C (mutation): Applying catalytic triad mutations...")
    mutated_fasta = run_dir / "candidates" / "candidates.filtered.fasta"
    
    # Run mutation script
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "apply_catalytic_triad.py"),
            str(length_pass_fasta),
            str(mutated_fasta),
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Warning: Mutation script had issues: {result.stderr[:500]}")
    
    # Load mutated sequences
    mutated_sequences = []
    with open(mutated_fasta) as f:
        for line in f:
            if not line.startswith('>'):
                seq = line.strip()
                if seq and len(seq) == EXPECTED_LENGTH:
                    mutated_sequences.append(seq)
    
    print(f"✓ Applied mutations to {len(mutated_sequences)} sequences")
    print()
    
    # Continue filtering mutated sequences (composition FIRST to filter junk, then identity, uniqueness)
    print("Stage C (continued): Filtering mutated sequences...")
    
    # Run composition gate FIRST to filter out repetitive junk early
    # (They already pass length and hard locks)
    if len(mutated_sequences) > 0:
        # Gate C5: Composition (run FIRST to filter repetitive junk)
        composition_pass, _ = filters.gate_composition(mutated_sequences)
        print(f"  After composition (filtered repetitive junk): {len(composition_pass)}")
        
        # Gate C3: Identity buckets (on composition-passed sequences)
        # Use relaxed thresholds if requested
        if args.relax_identity:
            relaxed_buckets = {
                "Near": (0.50, 0.95),      # Lowered from 0.75
                "Explore": (0.30, 0.50),   # Lowered from 0.55
            }
            identity_pass, _, bucket_assignments = filters.gate_identity_buckets(
                composition_pass, baseline_seq, buckets=relaxed_buckets
            )
        else:
            identity_pass, _, bucket_assignments = filters.gate_identity_buckets(
                composition_pass, baseline_seq
            )
        print(f"  After identity buckets: {len(identity_pass)}")
        
        # Gate C4: Uniqueness
        uniqueness_pass, _ = filters.gate_uniqueness(identity_pass)
    else:
        identity_pass = []
        uniqueness_pass = []
        composition_pass = []
        bucket_assignments = {}
    
        print(f"  After uniqueness: {len(uniqueness_pass)}")
    print()
    
    # Stage C6: Likelihood ranking (optional)
    # Use uniqueness_pass (sequences that passed ALL gates), not composition_pass
    if not args.skip_likelihood and len(uniqueness_pass) > 0:
        print("Stage C6: Computing likelihoods and ranking...")
        
        # Create metadata for mutated sequences
        mutated_metadata = []
        for seq in uniqueness_pass:
            # Compute identity
            from progen2_pipeline.filters import compute_identity
            identity = compute_identity(seq, baseline_seq)
            
            # Assign bucket
            if 0.75 <= identity <= 0.95:
                bucket = "Near"
            elif 0.55 <= identity <= 0.75:
                bucket = "Explore"
            else:
                bucket = "Other"
            
            mutated_metadata.append({
                "identity": identity,
                "identity_bucket": bucket,
                # Note: prompt_length and lane info is lost after mutation
                # Could be preserved if needed
            })
        
        # Compute likelihoods
        likelihoods = likelihood_ranking.compute_likelihoods_batch(
            uniqueness_pass, args.model, progen2_dir, args.device, verbose=True
        )
        
        # Add likelihoods to metadata
        for i, ll in enumerate(likelihoods):
            if i < len(mutated_metadata):
                mutated_metadata[i]["likelihood"] = ll
        
        # Rank with diversity preservation
        ranked_seqs, ranked_meta, ranking_df = likelihood_ranking.rank_with_diversity_preservation(
            uniqueness_pass, mutated_metadata, baseline_seq, args.top_quantile
        )
        
        # Save ranking CSV
        ranking_csv = run_dir / "candidates" / "candidates.ranked.csv"
        if isinstance(ranking_df, list):
            # Simple CSV writer without pandas
            with open(ranking_csv, 'w') as f:
                if ranking_df:
                    headers = list(ranking_df[0].keys())
                    f.write(",".join(headers) + "\n")
                    for row in ranking_df:
                        values = [str(row.get(h, "")) for h in headers]
                        f.write(",".join(values) + "\n")
        else:
            ranking_df.to_csv(ranking_csv, index=False)
        
        print(f"✓ Ranked and selected {len(ranked_seqs)} sequences")
        print()
        
        final_candidates = ranked_seqs
        ranked_fasta = run_dir / "candidates" / "candidates.ranked.fasta"
    else:
        # No likelihood ranking - use uniqueness_pass as final candidates
        if args.skip_likelihood:
            print("Stage C6: Skipped (--skip-likelihood)")
        else:
            print("Stage C6: Skipped (no sequences to rank)")
        print()
        
        final_candidates = uniqueness_pass
        ranked_fasta = run_dir / "candidates" / "candidates.filtered.fasta"
    
    # Save final candidates
    with open(ranked_fasta, 'w') as f:
        for i, seq in enumerate(final_candidates):
            f.write(f">candidate_{i+1}\n")
            f.write(f"{seq}\n")
    
    # Update manifest
    manifest_file = run_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        manifest["generation"]["num_samples_per_prompt_lane"] = args.num_samples
        manifest["generation"]["total_generated"] = total_generated
        manifest["gates"]["C0_token_cleanup"]["count_in"] = filter_report["gate_C0_token_cleanup"]["count_in"]
        manifest["gates"]["C0_token_cleanup"]["count_out"] = filter_report["gate_C0_token_cleanup"]["count_out"]
        manifest["gates"]["C1_length"]["count_in"] = filter_report["gate_C1_length"]["count_in"]
        manifest["gates"]["C1_length"]["count_out"] = filter_report["gate_C1_length"]["count_out"]
        manifest["gates"]["C2_hard_locks"]["count_in"] = len(length_pass_seqs)
        manifest["gates"]["C2_hard_locks"]["count_out"] = len(mutated_sequences)
        # Note: Gate order is C5 (composition) FIRST, then C3 (identity), then C4 (uniqueness)
        manifest["gates"]["C5_composition"]["count_in"] = len(mutated_sequences)
        manifest["gates"]["C5_composition"]["count_out"] = len(composition_pass)
        manifest["gates"]["C3_identity_buckets"]["count_in"] = len(composition_pass)
        manifest["gates"]["C3_identity_buckets"]["count_out"] = len(identity_pass)
        manifest["gates"]["C4_uniqueness"]["count_in"] = len(identity_pass)
        manifest["gates"]["C4_uniqueness"]["count_out"] = len(uniqueness_pass)
        manifest["final"]["candidates_selected"] = len(final_candidates)
        manifest["final"]["final_rank_file"] = str(ranked_fasta.relative_to(run_dir)) if 'ranked_fasta' in locals() else None
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    # Final summary
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"Total sequences generated: {total_generated}")
    print(f"Passed token cleanup: {filter_report['gate_C0_token_cleanup']['count_out']}")
    print(f"Passed length gate: {filter_report['gate_C1_length']['count_out']}")
    print(f"After mutation: {len(mutated_sequences)}")
    print(f"Passed composition: {len(composition_pass)}")
    print(f"Passed identity buckets: {len(identity_pass)}")
    print(f"Passed uniqueness: {len(uniqueness_pass)}")
    print(f"Final candidates: {len(final_candidates)}")
    print()
    print("✓ Pipeline complete!")
    print(f"  Final candidates: {run_dir / 'candidates' / 'candidates.ranked.fasta'}")
    print()
    print("Next steps:")
    print("  1. Review candidates.ranked.csv")
    print("  2. Run AlphaFold/ColabFold on candidates (Stage D)")
    print("  3. Run Rosetta/FoldX stability scoring (Stage E)")


if __name__ == '__main__':
    main()

