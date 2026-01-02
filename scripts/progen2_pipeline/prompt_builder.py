#!/usr/bin/env python3
"""
Prompt Builder Module for ProGen2 Pipeline.

Builds N-terminus anchor prompts at multiple lengths from baseline sequence.
"""

from pathlib import Path
import json
from typing import Dict, List


def load_baseline_fasta(baseline_path: Path) -> str:
    """Load baseline FASTA sequence."""
    with open(baseline_path, 'r') as f:
        lines = f.readlines()
        # Skip header, get sequence
        sequence = ''.join(line.strip() for line in lines[1:])
    return sequence


def build_prompts(baseline_seq: str, prompt_lengths: List[int] = [20, 50, 80]) -> Dict[int, str]:
    """
    Build N-terminus anchor prompts at specified lengths.
    
    Args:
        baseline_seq: Baseline sequence (263 aa)
        prompt_lengths: List of prompt lengths in amino acids
    
    Returns:
        Dict mapping prompt_length -> prompt_string (with '1' prefix for ProGen2)
    """
    prompts = {}
    
    for length in prompt_lengths:
        if length > len(baseline_seq):
            raise ValueError(f"Prompt length {length} exceeds baseline length {len(baseline_seq)}")
        
        prompt_aa = baseline_seq[:length]
        # ProGen2 format: start with '1', then sequence
        prompt = f"1{prompt_aa}"
        prompts[length] = prompt
    
    return prompts


def save_prompts(prompts: Dict[int, str], output_dir: Path, baseline_path: Path) -> Path:
    """
    Save prompts to files and create prompt manifest.
    
    Args:
        prompts: Dict of {length: prompt_string}
        output_dir: Directory to save prompts
        baseline_path: Path to baseline FASTA (for manifest)
    
    Returns:
        Path to prompt manifest file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual prompt files
    for length, prompt in prompts.items():
        prompt_file = output_dir / f"prompt_{length}aa.txt"
        with open(prompt_file, 'w') as f:
            f.write(f">prompt_{length}aa\n")
            f.write(f"{prompt}\n")
    
    # Create prompt manifest
    manifest = {
        "baseline_fasta": str(baseline_path),
        "prompts": {}
    }
    
    for length, prompt in prompts.items():
        prompt_aa_only = prompt[1:]  # Remove '1' prefix
        manifest["prompts"][length] = {
            "prompt_string": prompt,
            "prompt_aa_only": prompt_aa_only,
            "prompt_length_aa": length,
            "file": f"prompt_{length}aa.txt"
        }
    
    manifest_file = output_dir / "prompt_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Also create human-readable markdown version
    manifest_md = output_dir / "prompt_manifest.md"
    with open(manifest_md, 'w') as f:
        f.write("# Prompt Manifest\n\n")
        f.write(f"**Baseline:** `{baseline_path}`\n\n")
        f.write("## Prompts\n\n")
        f.write("| Length (aa) | File | Preview |\n")
        f.write("|-------------|------|---------|\n")
        for length in sorted(prompts.keys()):
            prompt_aa = prompts[length][1:]  # Remove '1'
            preview = prompt_aa[:20] + "..." + prompt_aa[-10:] if len(prompt_aa) > 30 else prompt_aa
            f.write(f"| {length} | `prompt_{length}aa.txt` | `{preview}` |\n")
    
    return manifest_file


def main():
    """CLI interface for prompt builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build ProGen2 prompts from baseline")
    parser.add_argument("baseline_fasta", help="Path to baseline FASTA file")
    parser.add_argument("output_dir", help="Output directory for prompts")
    parser.add_argument("--lengths", nargs="+", type=int, default=[20, 50, 80],
                        help="Prompt lengths in aa (default: 20 50 80)")
    
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline_fasta)
    output_dir = Path(args.output_dir)
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline FASTA not found: {baseline_path}")
    
    # Load baseline and build prompts
    baseline_seq = load_baseline_fasta(baseline_path)
    print(f"Loaded baseline: {len(baseline_seq)} aa")
    
    prompts = build_prompts(baseline_seq, args.lengths)
    print(f"Built {len(prompts)} prompts at lengths: {sorted(prompts.keys())}")
    
    # Save prompts
    manifest_file = save_prompts(prompts, output_dir, baseline_path)
    print(f"✓ Saved prompts to {output_dir}")
    print(f"✓ Created manifest: {manifest_file}")


if __name__ == '__main__':
    main()

