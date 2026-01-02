#!/usr/bin/env python3
"""
ProGen2 Generation Module with Sampling Lanes.

Generates sequences using multiple sampling strategies (Conservative + Exploratory lanes).
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import json


# Sampling lane definitions
SAMPLING_LANES = {
    "Conservative": {
        "temperature": 0.3,  # More conservative: was 0.4, now 0.3 for better in-family sequences
        "top_p": 0.90,  # Slightly tighter: was 0.95, now 0.90
        "description": "Quality-biased, lower temperature for more conservative sequences"
    },
    "Exploratory": {
        "temperature": 0.6,  # More conservative: was 0.8, now 0.6 to reduce out-of-family drift
        "top_p": 0.85,
        "description": "Diversity-biased, higher temperature for more exploration"
    }
}


def run_progen2_generation(
    prompt: str,
    num_samples: int,
    model: str,
    lane: str,
    max_length: int,
    rng_seed: int,
    progen2_dir: Path,
    device: str = "cpu"
) -> List[str]:
    """
    Run ProGen2 generation for a single prompt and lane.
    
    Args:
        prompt: Prompt string (with '1' prefix)
        num_samples: Number of sequences to generate
        model: ProGen2 model name
        lane: Lane name (must be in SAMPLING_LANES)
        max_length: Maximum sequence length (total, including prompt)
        rng_seed: Random seed
        progen2_dir: Path to ProGen2 directory
        device: Device to use (cpu, cuda, mps)
    
    Returns:
        List of raw generated sequences (with control tokens)
    """
    if lane not in SAMPLING_LANES:
        raise ValueError(f"Unknown lane: {lane}. Must be one of {list(SAMPLING_LANES.keys())}")
    
    lane_config = SAMPLING_LANES[lane]
    
    # Check ProGen2 setup
    sample_script = progen2_dir / "sample.py"
    if not sample_script.exists():
        raise FileNotFoundError(f"ProGen2 sample.py not found at {sample_script}")
    
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
            pass  # Symlink might already exist
    
    # Build command
    cmd = [
        sys.executable,
        str(sample_script),
        "--model", model,
        "--context", prompt,
        "--num-samples", str(num_samples),
        "--max-length", str(max_length),
        "--t", str(lane_config["temperature"]),
        "--p", str(lane_config["top_p"]),
        "--rng-seed", str(rng_seed),
        "--device", device,
    ]
    
    # Run ProGen2
    # Timeout calculation: scale with model size and number of samples
    # Small models: ~2-5 sec/sample, Medium: ~5-10 sec/sample, Large: ~30-60 sec/sample
    # Be very generous for overnight runs
    if "large" in model.lower() or "xlarge" in model.lower():
        # Large models are extremely slow on CPU - may take hours per batch
        # For overnight runs, disable timeout or set very high
        seconds_per_sample = 120  # 2 minutes per sample (very conservative)
        min_timeout = 7200  # At least 2 hours
        # For overnight: allow up to 6 hours per combination
        max_timeout = 21600  # 6 hours max
    elif "medium" in model.lower() or "base" in model.lower():
        seconds_per_sample = 15  # 15 seconds per sample
        min_timeout = 600  # At least 10 minutes
        max_timeout = 7200  # 2 hours max (should be plenty)
    else:
        # Small model
        seconds_per_sample = 5
        min_timeout = 300  # At least 5 minutes
        max_timeout = 1800  # 30 minutes max
    
    # Calculate timeout with safety margin, but cap at max
    timeout_seconds = min(max_timeout, int(max(min_timeout, num_samples * seconds_per_sample) * 1.5))
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(progen2_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds
        )
        
        # Parse output - extract sequences
        output_lines = result.stdout.split('\n')
        raw_sequences = []
        
        # Look for sequences (they come after the context line)
        for line in output_lines:
            line = line.strip()
            if not line:
                continue
            
            # Sequences are typically printed as lines starting with '1' and containing amino acids
            if line.startswith('1') and len(line) > 50:
                # This looks like a sequence
                raw_sequences.append(line)
            elif len(line) > 50 and all(c in 'ACDEFGHIKLMNPQRSTVWYX12' for c in line.upper()):
                # Fallback: any long line that looks like a sequence
                raw_sequences.append(line)
        
        if not raw_sequences:
            # Try stderr as fallback
            for line in result.stderr.split('\n'):
                line = line.strip()
                if len(line) > 50 and all(c in 'ACDEFGHIKLMNPQRSTVWYX12' for c in line.upper()):
                    raw_sequences.append(line)
        
        if not raw_sequences:
            raise ValueError(f"No sequences found in ProGen2 output. STDOUT: {result.stdout[:500]}")
        
        return raw_sequences
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ProGen2 generation timed out after {timeout_seconds} seconds")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ProGen2 failed with exit code {e.returncode}\n"
            f"STDOUT: {e.stdout[:500]}\n"
            f"STDERR: {e.stderr[:500]}"
        )


def normalize_sequence(raw_seq: str) -> str:
    """
    Normalize ProGen2 output: remove control tokens, extract AA sequence.
    
    Args:
        raw_seq: Raw sequence with control tokens
    
    Returns:
        Normalized amino acid sequence
    """
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
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    normalized = ''.join(c for c in seq if c.upper() in valid_aa)
    
    return normalized


def generate_all_lanes(
    prompts: Dict[int, str],
    num_samples_per_prompt_lane: int,
    model: str,
    max_length: int,
    rng_seed: int,
    progen2_dir: Path,
    output_dir: Path,
    device: str = "cpu"
) -> Dict[str, Dict[int, List[Tuple[str, str]]]]:
    """
    Generate sequences for all prompt lengths × lanes.
    
    Args:
        prompts: Dict of {prompt_length: prompt_string}
        num_samples_per_prompt_lane: Number of samples per (prompt, lane) combination
        model: ProGen2 model name
        max_length: Maximum sequence length
        rng_seed: Base random seed (will be incremented per combination)
        progen2_dir: Path to ProGen2 directory
        output_dir: Output directory for raw and normalized sequences
        device: Device to use
    
    Returns:
        Dict of {lane: {prompt_length: [(raw_seq, normalized_seq), ...]}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_sequences = {lane: {} for lane in SAMPLING_LANES.keys()}
    seed_counter = rng_seed
    
    # Generate for each prompt length × lane combination
    for prompt_length, prompt in prompts.items():
        for lane in SAMPLING_LANES.keys():
            print(f"Generating: prompt {prompt_length}aa, lane {lane}...")
            
            try:
                raw_seqs = run_progen2_generation(
                    prompt=prompt,
                    num_samples=num_samples_per_prompt_lane,
                    model=model,
                    lane=lane,
                    max_length=max_length,
                    rng_seed=seed_counter,
                    progen2_dir=progen2_dir,
                    device=device
                )
                
                # Normalize sequences
                normalized_seqs = [normalize_sequence(raw) for raw in raw_seqs]
                
                # Store with metadata
                all_sequences[lane][prompt_length] = list(zip(raw_seqs, normalized_seqs))
                
                print(f"  Generated {len(raw_seqs)} sequences")
                seed_counter += 1  # Increment seed for next combination
                
            except Exception as e:
                print(f"  ERROR: {e}")
                all_sequences[lane][prompt_length] = []
    
    # Save raw and normalized sequences
    raw_fasta = output_dir / "raw_generations.fasta"
    norm_fasta = output_dir / "normalized.fasta"
    
    with open(raw_fasta, 'w') as f_raw, open(norm_fasta, 'w') as f_norm:
        seq_counter = 1
        for lane in SAMPLING_LANES.keys():
            for prompt_length in sorted(prompts.keys()):
                if prompt_length in all_sequences[lane]:
                    for raw_seq, norm_seq in all_sequences[lane][prompt_length]:
                        f_raw.write(f">seq_{seq_counter}_lane_{lane}_prompt_{prompt_length}aa\n")
                        f_raw.write(f"{raw_seq}\n")
                        f_norm.write(f">seq_{seq_counter}_lane_{lane}_prompt_{prompt_length}aa\n")
                        f_norm.write(f"{norm_seq}\n")
                        seq_counter += 1
    
    print(f"\n✓ Saved raw sequences to: {raw_fasta}")
    print(f"✓ Saved normalized sequences to: {norm_fasta}")
    
    return all_sequences


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ProGen2 generation with lanes")
    parser.add_argument("prompt_manifest", help="Path to prompt manifest JSON")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--model", default="progen2-small", help="ProGen2 model")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per (prompt, lane)")
    parser.add_argument("--max-length", type=int, default=264, help="Max sequence length")
    parser.add_argument("--rng-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Load prompts from manifest
    with open(args.prompt_manifest) as f:
        manifest = json.load(f)
    
    prompts = {}
    for length_str, prompt_info in manifest["prompts"].items():
        length = int(length_str)
        prompts[length] = prompt_info["prompt_string"]
    
    # Find ProGen2 directory
    repo_root = Path(__file__).parent.parent.parent
    progen2_dir = repo_root / "external" / "progen2"
    
    # Generate
    generate_all_lanes(
        prompts=prompts,
        num_samples_per_prompt_lane=args.num_samples,
        model=args.model,
        max_length=args.max_length,
        rng_seed=args.rng_seed,
        progen2_dir=progen2_dir,
        output_dir=Path(args.output_dir),
        device=args.device
    )

