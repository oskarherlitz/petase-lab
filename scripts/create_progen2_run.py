#!/usr/bin/env python3
"""
Create a new ProGen2 run folder with manifest system.

This creates the run directory skeleton and initializes the manifest
with repo commits, environment info, and placeholder fields for all gates.

Usage:
    python scripts/create_progen2_run.py <run_id> [--model MODEL] [--round ROUND] [--tag TAG]

Example:
    python scripts/create_progen2_run.py run_20251229_progen2_small_r1_anchor50
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


def get_git_commit(repo_path):
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_submodule_commit(submodule_path):
    """Get git commit hash of a submodule."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_python_version():
    """Get Python version."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def create_run_folder(run_id, model="progen2-small", round_num=1, tag=""):
    """Create run folder structure and manifest."""
    
    repo_root = Path(__file__).parent.parent
    runs_dir = repo_root / "runs"
    run_dir = runs_dir / run_id
    
    if run_dir.exists():
        raise ValueError(f"Run directory already exists: {run_dir}")
    
    # Create directory structure
    subdirs = [
        "prompts",
        "generated",
        "filters",
        "candidates",
        "af",
        "af/af_gate_pass",
        "stability",
        "docking",
        "final",
    ]
    
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Get git commits
    shis_commit = get_git_commit(repo_root)
    progen2_path = repo_root / "external" / "progen2"
    progen2_commit = get_git_submodule_commit(progen2_path) if progen2_path.exists() else "unknown"
    
    # Create manifest
    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "design_spec": "v1.0",
        "baseline": {
            "pdb": "5XJH",
            "chain": "A",
            "residues": "30-292",
            "length": 263,
            "baseline_fasta": "data/sequences/wt/baseline_5XJH_30-292.fasta",
        },
        "reproducibility": {
            "shis_repo_commit": shis_commit,
            "progen2_repo_commit": progen2_commit,
            "progen2_checkpoint": model,
            "python_version": get_python_version(),
        },
        "hardware": {
            "platform": sys.platform,
            # Will be filled in during actual runs
        },
        "prompt_set": {
            "prompt_lengths": [],
            "prompts": {},
            "prompt_manifest": None,
        },
        "generation": {
            "model": model,
            "sampling_lanes": {},
            "seeds": [],
            "settings": {},
        },
        "gates": {
            "C0_token_cleanup": {"count_in": 0, "count_out": 0, "removed": []},
            "C1_length": {"count_in": 0, "count_out": 0, "removed": []},
            "C2_hard_locks": {"count_in": 0, "count_out": 0, "removed": []},
            "C3_identity_buckets": {"count_in": 0, "count_out": 0, "removed": []},
            "C4_uniqueness": {"count_in": 0, "count_out": 0, "removed": []},
            "C5_composition": {"count_in": 0, "count_out": 0, "removed": []},
            "C6_likelihood_ranking": {"count_in": 0, "count_out": 0, "removed": []},
            "D_af_gate": {"count_in": 0, "count_out": 0, "removed": []},
            "E_stability_gate": {"count_in": 0, "count_out": 0, "removed": []},
            "F_docking_gate": {"count_in": 0, "count_out": 0, "removed": []},
        },
        "final": {
            "candidates_selected": 0,
            "final_rank_file": None,
        },
    }
    
    # Write manifest as both JSON and markdown
    manifest_json = run_dir / "manifest.json"
    manifest_md = run_dir / "manifest.md"
    
    with open(manifest_json, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Write human-readable markdown version
    with open(manifest_md, 'w') as f:
        f.write(f"# Run Manifest: {run_id}\n\n")
        f.write(f"**Created:** {manifest['created']}\n")
        f.write(f"**Design Spec:** {manifest['design_spec']}\n\n")
        
        f.write("## Baseline\n\n")
        f.write(f"- PDB: {manifest['baseline']['pdb']} chain {manifest['baseline']['chain']}\n")
        f.write(f"- Residues: {manifest['baseline']['residues']} ({manifest['baseline']['length']} aa)\n")
        f.write(f"- FASTA: `{manifest['baseline']['baseline_fasta']}`\n\n")
        
        f.write("## Reproducibility\n\n")
        f.write(f"- SHIS repo commit: `{manifest['reproducibility']['shis_repo_commit']}`\n")
        f.write(f"- ProGen2 repo commit: `{manifest['reproducibility']['progen2_repo_commit']}`\n")
        f.write(f"- ProGen2 checkpoint: `{manifest['reproducibility']['progen2_checkpoint']}`\n")
        f.write(f"- Python version: {manifest['reproducibility']['python_version']}\n\n")
        
        f.write("## Gate Counts\n\n")
        f.write("| Gate | Input | Output | Status |\n")
        f.write("|------|-------|--------|--------|\n")
        for gate_name, gate_data in manifest['gates'].items():
            status = "✓" if gate_data['count_out'] > 0 else "⏳"
            f.write(f"| {gate_name} | {gate_data['count_in']} | {gate_data['count_out']} | {status} |\n")
        f.write("\n")
        
        f.write("## Notes\n\n")
        f.write("_This manifest will be updated as the pipeline progresses._\n")
    
    print(f"✓ Created run folder: {run_dir}")
    print(f"  Manifest: {manifest_json}")
    print(f"  Manifest (markdown): {manifest_md}")
    print()
    print("Run folder structure:")
    for subdir in subdirs:
        print(f"  {run_dir / subdir}/")
    
    return run_dir, manifest


def main():
    parser = argparse.ArgumentParser(description="Create a new ProGen2 run folder")
    parser.add_argument("run_id", help="Run ID (e.g., run_20251229_progen2_small_r1_anchor50)")
    parser.add_argument("--model", default="progen2-small", help="ProGen2 model name")
    parser.add_argument("--round", type=int, default=1, help="Round number")
    parser.add_argument("--tag", default="", help="Short tag for the run")
    
    args = parser.parse_args()
    
    try:
        run_dir, manifest = create_run_folder(args.run_id, args.model, args.round, args.tag)
        print()
        print("Run folder initialized successfully!")
        print(f"Next steps:")
        print(f"  1. Build prompts: populate {run_dir / 'prompts'}/")
        print(f"  2. Run generation: populate {run_dir / 'generated'}/")
        print(f"  3. Run filters: populate {run_dir / 'filters'}/")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

