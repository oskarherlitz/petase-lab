#!/usr/bin/env python3
"""
Minimal smoke test for ProGen2 pipeline.

This validates:
1. Baseline sequence extraction and prompt building
2. ProGen2 generation and token normalization
3. First 2-3 gates (token cleanup, length, hard locks)

Usage:
    python scripts/progen2_smoke_test.py <run_id> [--num-samples N]
"""

import sys
import json
import subprocess
import os
from pathlib import Path
import argparse


# Hard constraints
EXPECTED_LENGTH = 263
HARD_LOCKS = {
    131: 'S',  # PDB 160
    177: 'D',  # PDB 206
    208: 'H',  # PDB 237
}


def load_baseline_fasta(baseline_path):
    """Load baseline FASTA sequence."""
    with open(baseline_path, 'r') as f:
        lines = f.readlines()
        # Skip header, get sequence
        sequence = ''.join(line.strip() for line in lines[1:])
    return sequence


def build_prompt(baseline_seq, prompt_length=50):
    """Build N-terminus anchor prompt."""
    prompt = baseline_seq[:prompt_length]
    # ProGen2 format: start with '1', prompt sequence, then model generates
    return f"1{prompt}"


def run_progen2_generation(prompt, num_samples=10, model="progen2-small", run_dir=None):
    """Run ProGen2 generation and return raw outputs."""
    
    # Check if ProGen2 is available
    progen2_dir = Path(__file__).parent.parent / "external" / "progen2"
    sample_script = progen2_dir / "sample.py"
    
    if not sample_script.exists():
        raise FileNotFoundError(f"ProGen2 sample.py not found at {sample_script}")
    
    # Check if checkpoint exists
    checkpoint_dir = progen2_dir / "checkpoints" / model
    if not checkpoint_dir.exists():
        print(f"Warning: Checkpoint {model} not found at {checkpoint_dir}")
        print("  You may need to download it first.")
        print(f"  See: {progen2_dir / 'README.md'}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    # Run ProGen2
    # Conservative settings for smoke test
    # ProGen2 expects 'models.progen' import, so we need to add the parent directory to PYTHONPATH
    # or create the expected structure. We'll set PYTHONPATH to include progen2_dir.
    cmd = [
        sys.executable,
        str(sample_script),
        "--model", model,
        "--context", prompt,
        "--num-samples", str(num_samples),
        "--max-length", str(EXPECTED_LENGTH + 1),  # Total length including prompt (264 to get 263 aa - off-by-one fix)
        "--t", "0.8",  # Temperature
        "--p", "0.9",  # Top-p
        "--rng-seed", "42",
    ]
    
    print(f"Running ProGen2 generation...")
    print(f"  Command: {' '.join(cmd)}")
    
    # Ensure models/progen structure exists (ProGen2 expects 'from models.progen.modeling_progen')
    models_dir = progen2_dir / "models"
    if not models_dir.exists():
        models_dir.mkdir()
        try:
            os.symlink(progen2_dir / "progen", models_dir / "progen")
        except OSError:
            # Symlink might already exist, that's fine
            pass
    
    # Calculate timeout based on number of samples
    # Rough estimate: ~2-5 seconds per sequence on CPU, add buffer
    timeout_seconds = max(300, num_samples * 5)  # At least 5 min, or 5 sec per sample
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(progen2_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds
        )
        
        # Parse output - ProGen2 prints sequences after the context
        output_lines = result.stdout.split('\n')
        raw_sequences = []
        
        # Look for sequences (they come after the context line)
        in_sequences = False
        for line in output_lines:
            line = line.strip()
            if not line:
                continue
            
            # Sequences are printed as numbered items
            if line.isdigit() or (line.startswith('1') and len(line) > 50):
                # This might be a sequence
                if line.startswith('1') and len(line) > 50:
                    raw_sequences.append(line)
                elif line.isdigit():
                    in_sequences = True
            elif in_sequences and len(line) > 50:
                raw_sequences.append(line)
        
        # If we didn't parse well, try to extract from stderr or look for sequences
        if not raw_sequences:
            # Fallback: look for any line that looks like a sequence
            for line in output_lines:
                if len(line) > 50 and all(c in 'ACDEFGHIKLMNPQRSTVWYX12' for c in line.upper()):
                    raw_sequences.append(line)
        
        if not raw_sequences:
            print("Warning: Could not parse sequences from ProGen2 output")
            print("STDOUT:", result.stdout[:500])
            print("STDERR:", result.stderr[:500])
            raise ValueError("No sequences found in ProGen2 output")
        
        return raw_sequences
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("ProGen2 generation timed out")
    except subprocess.CalledProcessError as e:
        print(f"ProGen2 failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout[:500])
        print("STDERR:", e.stderr[:500])
        raise


def normalize_sequence(raw_seq):
    """Normalize ProGen2 output: remove control tokens, extract AA sequence."""
    # ProGen2 uses '1' as start token, '2' as end token
    # Remove everything before first '1' and after first '2'
    seq = raw_seq
    
    # Find start (first '1' that's followed by amino acids)
    start_idx = seq.find('1')
    if start_idx != -1:
        seq = seq[start_idx + 1:]  # Remove the '1'
    
    # Find end (first '2' or end of string)
    end_idx = seq.find('2')
    if end_idx != -1:
        seq = seq[:end_idx]
    
    # Remove any remaining control tokens or non-AA characters
    # Keep only standard amino acids
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    normalized = ''.join(c for c in seq if c.upper() in valid_aa)
    
    return normalized


def gate_token_cleanup(sequences):
    """Gate C0: Token + alphabet cleanup."""
    passed = []
    removed = []
    
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    for seq in sequences:
        # Check for invalid characters
        if all(c.upper() in valid_aa for c in seq):
            passed.append(seq)
        else:
            invalid_chars = set(c for c in seq if c.upper() not in valid_aa)
            removed.append(f"Invalid chars: {invalid_chars}")
    
    return passed, removed


def gate_length(sequences):
    """Gate C1: Length check (must be exactly 263 aa)."""
    passed = []
    removed = []
    
    for seq in sequences:
        if len(seq) == EXPECTED_LENGTH:
            passed.append(seq)
        else:
            removed.append(f"Length {len(seq)} (expected {EXPECTED_LENGTH})")
    
    return passed, removed


def gate_hard_locks(sequences):
    """Gate C2: Hard-lock gate (positions 131, 177, 208 must be S, D, H)."""
    passed = []
    removed = []
    
    for seq in sequences:
        if len(seq) != EXPECTED_LENGTH:
            removed.append("Length mismatch")
            continue
        
        errors = []
        for pos, required_aa in HARD_LOCKS.items():
            # Convert to 0-based index
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


def main():
    parser = argparse.ArgumentParser(description="Run ProGen2 smoke test")
    parser.add_argument("run_id", help="Run ID")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--model", default="progen2-small", help="ProGen2 model")
    parser.add_argument("--prompt-length", type=int, default=50, help="Prompt length (aa)")
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    run_dir = repo_root / "runs" / args.run_id
    
    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")
    
    baseline_path = repo_root / "data" / "sequences" / "wt" / "baseline_5XJH_30-292.fasta"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline FASTA not found: {baseline_path}")
    
    print("=" * 60)
    print("ProGen2 Smoke Test")
    print("=" * 60)
    print(f"Run ID: {args.run_id}")
    print(f"Model: {args.model}")
    print(f"Prompt length: {args.prompt_length} aa")
    print(f"Num samples: {args.num_samples}")
    print()
    
    # Step 1: Load baseline and build prompt
    print("Step 1: Building prompt...")
    baseline_seq = load_baseline_fasta(baseline_path)
    print(f"  Baseline length: {len(baseline_seq)} aa")
    
    prompt = build_prompt(baseline_seq, args.prompt_length)
    print(f"  Prompt: {prompt[:20]}...{prompt[-20:]} (length: {len(prompt)} chars)")
    print()
    
    # Save prompt
    prompt_file = run_dir / "prompts" / f"prompt_{args.prompt_length}aa.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_file, 'w') as f:
        f.write(f">prompt_{args.prompt_length}aa\n")
        f.write(f"{prompt}\n")
    print(f"  Saved prompt to: {prompt_file}")
    print()
    
    # Step 2: Run ProGen2 generation
    print("Step 2: Running ProGen2 generation...")
    try:
        raw_sequences = run_progen2_generation(
            prompt,
            num_samples=args.num_samples,
            model=args.model,
            run_dir=run_dir
        )
        print(f"  Generated {len(raw_sequences)} raw sequences")
        print()
    except Exception as e:
        print(f"  ERROR: {e}")
        print()
        print("Note: If checkpoint is missing, you need to download it first.")
        print("  See: external/progen2/README.md")
        sys.exit(1)
    
    # Save raw sequences
    raw_fasta = run_dir / "generated" / "raw_generations.fasta"
    with open(raw_fasta, 'w') as f:
        for i, seq in enumerate(raw_sequences):
            f.write(f">raw_{i+1}\n")
            f.write(f"{seq}\n")
    print(f"  Saved raw sequences to: {raw_fasta}")
    print()
    
    # Step 3: Normalize sequences
    print("Step 3: Normalizing sequences...")
    normalized_sequences = []
    for i, raw_seq in enumerate(raw_sequences):
        norm_seq = normalize_sequence(raw_seq)
        normalized_sequences.append(norm_seq)
        print(f"  Seq {i+1}: {len(raw_seq)} chars -> {len(norm_seq)} aa")
    
    # Save normalized sequences
    norm_fasta = run_dir / "generated" / "normalized.fasta"
    with open(norm_fasta, 'w') as f:
        for i, seq in enumerate(normalized_sequences):
            f.write(f">normalized_{i+1}\n")
            f.write(f"{seq}\n")
    print(f"  Saved normalized sequences to: {norm_fasta}")
    print()
    
    # Step 4: Run gates
    print("Step 4: Running gates...")
    
    # Gate C0: Token cleanup
    print("  Gate C0: Token cleanup...")
    passed_c0, removed_c0 = gate_token_cleanup(normalized_sequences)
    print(f"    Input: {len(normalized_sequences)}, Output: {len(passed_c0)}, Removed: {len(removed_c0)}")
    
    # Gate C1: Length
    print("  Gate C1: Length check (263 aa)...")
    passed_c1, removed_c1 = gate_length(passed_c0)
    print(f"    Input: {len(passed_c0)}, Output: {len(passed_c1)}, Removed: {len(removed_c1)}")
    
    # Gate C2: Hard locks
    print("  Gate C2: Hard locks (S131, D177, H208)...")
    passed_c2, removed_c2 = gate_hard_locks(passed_c1)
    print(f"    Input: {len(passed_c1)}, Output: {len(passed_c2)}, Removed: {len(removed_c2)}")
    print()
    
    # Save sequences that passed length gate (before hard locks)
    # These can be mutated to add catalytic triad
    length_pass_fasta = run_dir / "candidates" / "candidates.length_pass.fasta"
    with open(length_pass_fasta, 'w') as f:
        for i, seq in enumerate(passed_c1):
            f.write(f">length_pass_{i+1}\n")
            f.write(f"{seq}\n")
    print(f"  Saved length-pass sequences to: {length_pass_fasta}")
    print(f"    (These can be mutated to add catalytic triad)")
    print()
    
    # Save filtered sequences (that passed all gates including hard locks)
    if passed_c2:
        filtered_fasta = run_dir / "candidates" / "candidates.filtered.fasta"
        with open(filtered_fasta, 'w') as f:
            for i, seq in enumerate(passed_c2):
                f.write(f">candidate_{i+1}\n")
                f.write(f"{seq}\n")
        print(f"  Saved filtered candidates to: {filtered_fasta}")
        print()
    
    # Create filter report
    filter_report = {
        "gate_C0_token_cleanup": {
            "count_in": len(normalized_sequences),
            "count_out": len(passed_c0),
            "removed": removed_c0[:10],  # First 10 examples
        },
        "gate_C1_length": {
            "count_in": len(passed_c0),
            "count_out": len(passed_c1),
            "removed": removed_c1[:10],
        },
        "gate_C2_hard_locks": {
            "count_in": len(passed_c1),
            "count_out": len(passed_c2),
            "removed": removed_c2[:10],
        },
    }
    
    report_file = run_dir / "filters" / "filter_report.json"
    with open(report_file, 'w') as f:
        json.dump(filter_report, f, indent=2)
    print(f"  Saved filter report to: {report_file}")
    print()
    
    # Summary
    print("=" * 60)
    print("Smoke Test Summary")
    print("=" * 60)
    print(f"Raw sequences generated: {len(raw_sequences)}")
    print(f"After normalization: {len(normalized_sequences)}")
    print(f"After token cleanup: {len(passed_c0)}")
    print(f"After length gate: {len(passed_c1)}")
    print(f"After hard locks gate: {len(passed_c2)}")
    print()
    
    if len(passed_c2) > 0:
        print("✓ Smoke test PASSED!")
        print(f"  {len(passed_c2)} candidates passed all gates")
    else:
        print("⚠ Smoke test completed but no candidates passed all gates")
        print("  This may be normal for a small sample size.")
        print("  Check filter_report.json for details.")
    
    print()
    print("Next steps:")
    print("  1. Review filter_report.json")
    print("  2. If passing, expand to full pipeline")
    print("  3. If failing, debug token normalization or gate logic")


if __name__ == '__main__':
    main()

