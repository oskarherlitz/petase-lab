#!/usr/bin/env python3
"""
Parse Rosetta cartesian_ddg output files (JSON or DDG format) into CSV.

Usage:
    python scripts/parse_ddg.py runs/*ddg*/outputs/*.json results/ddg_scans/out.csv
    python scripts/parse_ddg.py results/ddg_scans/out.csv  # Auto-search for files
    python scripts/parse_ddg.py 'runs/*ddg*/outputs/*.json' results/ddg_scans/out.csv  # Quote globs for zsh

The script will:
1. Search for JSON files in outputs/ directories AND root directory
2. Try to parse JSON files first (if -ddg:json was used)
3. Fall back to parsing .ddg files if JSON is not available
4. Extract mutation and ΔΔG values
5. Write results to CSV
"""

import json
import sys
import csv
import glob
import os
from pathlib import Path
import re

def extract_run_name(filepath):
    """Extract run directory name from file path robustly."""
    path = Path(filepath)
    # Navigate up from outputs/ to get run directory name
    # Path structure: runs/YYYY-MM-DD_ddg_cart_NAME/outputs/file.ext
    parts = path.parts
    if 'outputs' in parts:
        idx = parts.index('outputs')
        if idx > 0:
            return parts[idx - 1]  # Get directory before 'outputs'
    # Fallback: use parent directory name or filename stem
    if path.parent.name:
        return path.parent.name
    return path.stem

_AA3_TO_AA1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

def _avg(values):
    return sum(values) / len(values) if values else None

def _mutation_signature_from_json_mutations(mutations):
    """
    Convert Rosetta cartesian_ddg JSON 'mutations' list to a stable signature.

    For WT entries, Rosetta often reports mutations with mut==wt (no change),
    or an empty list. We treat those as WT.
    """
    if not isinstance(mutations, list) or not mutations:
        return "WT"

    pieces = []
    for m in mutations:
        if not isinstance(m, dict):
            continue
        wt = m.get("wt")
        mut = m.get("mut")
        pos = m.get("pos")
        if wt and mut and pos is not None:
            # Skip no-op entries (wt==mut) which represent WT scoring
            if wt == mut:
                continue
            pieces.append(f"{wt}{pos}{mut}")

    return ";".join(pieces) if pieces else "WT"

def _ddg_from_cartesian_ddg_json(j):
    """
    Newer cartesian_ddg JSON often stores per-state scores but not ddg.
    We compute ddg = avg(mut_total) - avg(wt_total).
    """
    if not isinstance(j, list):
        return []

    totals_by_sig = {}
    for item in j:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        total = scores.get("total")
        if total is None:
            continue
        try:
            total = float(total)
        except (ValueError, TypeError):
            continue

        sig = _mutation_signature_from_json_mutations(item.get("mutations"))
        totals_by_sig.setdefault(sig, []).append(total)

    wt_avg = _avg(totals_by_sig.get("WT", []))
    if wt_avg is None:
        return []

    ddgs = []
    for sig, totals in totals_by_sig.items():
        if sig == "WT":
            continue
        mut_avg = _avg(totals)
        if mut_avg is None:
            continue
        ddgs.append((sig, mut_avg - wt_avg))

    return ddgs

def parse_json_file(jpath, run_name):
    """Parse Rosetta JSON output file."""
    mutations_data = []
    
    try:
        with open(jpath, 'r') as jf:
            content = jf.read().strip()
            
            # Check if file is actually CSV (common mistake)
            if content.startswith('run,mutation') or content.startswith('run, mutation'):
                print(f"Warning: {jpath} appears to be CSV, not JSON. Skipping.", file=sys.stderr)
                return mutations_data
            
            # Try to parse as JSON
            try:
                j = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {jpath}: {e}", file=sys.stderr)
                return mutations_data
            
            # Handle both array and dict formats
            if isinstance(j, list):
                # Rosetta may output array of results
                for idx, item in enumerate(j):
                    if not isinstance(item, dict):
                        continue
                    
                    mutations = item.get("mutations", [])
                    if isinstance(mutations, list):
                        # Array format: mutations is a list
                        for mut in mutations:
                            if isinstance(mut, dict):
                                # Mutation is a dict with keys like "name", "ddg", etc.
                                mut_name = mut.get("name") or mut.get("mutation") or mut.get("mut")
                                ddg = mut.get("ddg") or mut.get("ddg_reu") or mut.get("delta_delta_g")
                                if mut_name and ddg is not None:
                                    try:
                                        ddg_float = float(ddg)
                                        mutations_data.append((run_name, mut_name, ddg_float))
                                    except (ValueError, TypeError):
                                        print(f"Warning: Invalid ddg value '{ddg}' for mutation {mut_name} in {jpath}", file=sys.stderr)
                    elif isinstance(mutations, dict):
                        # Dict format: mutations is a dict
                        for mut_name, mut_info in mutations.items():
                            if isinstance(mut_info, dict):
                                ddg = mut_info.get("ddg") or mut_info.get("ddg_reu")
                            else:
                                ddg = mut_info
                            
                            if ddg is not None:
                                try:
                                    ddg_float = float(ddg)
                                    mutations_data.append((run_name, mut_name, ddg_float))
                                except (ValueError, TypeError):
                                    print(f"Warning: Invalid ddg value '{ddg}' for mutation {mut_name} in {jpath}", file=sys.stderr)
                    
                    # Check for ddg field at top level
                    ddg = item.get("ddg") or item.get("ddg_reu")
                    if ddg is not None and not mutations_data:
                        print(f"Warning: Found ddg at top level but no mutation name in {jpath}", file=sys.stderr)
            
            elif isinstance(j, dict):
                # Dict format: { "mutations": {...} }
                mutations = j.get("mutations", {})
                
                if isinstance(mutations, dict):
                    for mut_name, mut_info in mutations.items():
                        if isinstance(mut_info, dict):
                            ddg = mut_info.get("ddg") or mut_info.get("ddg_reu")
                        else:
                            ddg = mut_info
                        
                        if ddg is not None:
                            try:
                                ddg_float = float(ddg)
                                mutations_data.append((run_name, mut_name, ddg_float))
                            except (ValueError, TypeError):
                                print(f"Warning: Invalid ddg value '{ddg}' for mutation {mut_name} in {jpath}", file=sys.stderr)
                elif isinstance(mutations, list):
                    # Array of mutation dicts
                    for mut in mutations:
                        if isinstance(mut, dict):
                            mut_name = mut.get("name") or mut.get("mutation")
                            ddg = mut.get("ddg") or mut.get("ddg_reu")
                            if mut_name and ddg is not None:
                                try:
                                    ddg_float = float(ddg)
                                    mutations_data.append((run_name, mut_name, ddg_float))
                                except (ValueError, TypeError):
                                    print(f"Warning: Invalid ddg value '{ddg}' for mutation {mut_name} in {jpath}", file=sys.stderr)
            
            # If we didn't find explicit ddg fields, try computing ddg from scores totals.
            if not mutations_data:
                ddgs = _ddg_from_cartesian_ddg_json(j)
                for sig, ddg_val in ddgs:
                    mutations_data.append((run_name, sig, ddg_val))

            if not mutations_data:
                print(f"Warning: No mutations found in {jpath}. JSON structure may be different than expected.", file=sys.stderr)
                print(f"  JSON type: {type(j).__name__}", file=sys.stderr)
                if isinstance(j, (list, dict)) and len(j) > 0:
                    sample = j[0] if isinstance(j, list) else j
                    if isinstance(sample, dict):
                        print(f"  Sample keys: {list(sample.keys())[:5]}", file=sys.stderr)
                        # Check if mutations array is empty
                        mutations = sample.get("mutations", [])
                        if isinstance(mutations, list) and len(mutations) == 0:
                            print(f"  ⚠ Mutations array is empty - Rosetta may not have processed mutations!", file=sys.stderr)
                            print(f"  Possible causes:", file=sys.stderr)
                            print(f"    1. Mutation file format issue (missing chain IDs?)", file=sys.stderr)
                            print(f"    2. Residue numbers/names don't match PDB file", file=sys.stderr)
                            print(f"    3. Check ROSETTA_CRASH.log for errors", file=sys.stderr)
            
    except FileNotFoundError:
        print(f"Error: File not found: {jpath}", file=sys.stderr)
    except Exception as e:
        print(f"Error reading {jpath}: {e}", file=sys.stderr)
    
    return mutations_data

def parse_ddg_file(ddg_path, mut_file_path, run_name):
    """
    Parse Rosetta .ddg output file.
    This is a fallback when JSON is not available.
    """
    mutations_data = []
    
    try:
        # Read mutation file to get mutation list
        mutations_list = []
        wt_by_pos_mut = {}  # (pos, mut1) -> wt1
        if os.path.exists(mut_file_path):
            with open(mut_file_path, 'r') as f:
                lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
                total = None
                for line in lines:
                    if line.startswith('total'):
                        total = int(line.split()[-1])
                        break
                
                # Extract mutations
                # Common formats seen in this repo:
                # - "S 131 A A"  -> wt, pose_resnum, chain, mut
                i = 0
                while i < len(lines):
                    if lines[i].isdigit():  # Mutation number
                        i += 1
                        if i < len(lines):
                            parts = lines[i].split()
                            if len(parts) >= 4:
                                wt_aa, resnum, _chain, mut_aa = parts[0], parts[1], parts[2], parts[3]
                                mutations_list.append(f"{wt_aa}{resnum}{mut_aa}")
                                try:
                                    wt_by_pos_mut[(int(resnum), mut_aa)] = wt_aa
                                except ValueError:
                                    pass
                            elif len(parts) >= 3:
                                # Fallback (legacy): wt, resnum, mut
                                wt_aa, resnum, mut_aa = parts[0], parts[1], parts[2]
                                mutations_list.append(f"{wt_aa}{resnum}{mut_aa}")
                                try:
                                    wt_by_pos_mut[(int(resnum), mut_aa)] = wt_aa
                                except ValueError:
                                    pass
                        i += 1
                    else:
                        i += 1
        
        # Try to parse .ddg file for DDG values
        # Rosetta .ddg format: MUTATION: Round1: <mut_name>: <ddg_value> ...
        with open(ddg_path, 'r') as f:
            content = f.read()
            
            # Look for MUTATION lines with DDG values
            # Pattern: MUTATION: RoundN: <name>: <ddg> ...
            mutation_pattern = r'MUTATION:\s+Round\d+:\s+(\S+):\s+([-\d.]+)'
            matches = re.findall(mutation_pattern, content)
            
            if matches:
                # Group by mutation name and average DDG values
                mut_ddg = {}
                for mut_name, ddg_str in matches:
                    try:
                        ddg_val = float(ddg_str)
                        if mut_name not in mut_ddg:
                            mut_ddg[mut_name] = []
                        mut_ddg[mut_name].append(ddg_val)
                    except ValueError:
                        continue
                
                # Average DDG values across rounds
                for mut_name, ddg_values in mut_ddg.items():
                    avg_ddg = sum(ddg_values) / len(ddg_values)
                    mutations_data.append((run_name, mut_name, avg_ddg))

            # Newer cartesian_ddg output: lines like
            #   COMPLEX:   Round1: WT_: -1292.488 ...
            #   COMPLEX:   Round1: MUT_131ALA: -1294.187 ...
            if not mutations_data:
                complex_pattern = r'^COMPLEX:\s+Round(\d+):\s+(\S+):\s+([-\d.]+)'
                complex_matches = re.findall(complex_pattern, content, flags=re.MULTILINE)
                if complex_matches:
                    by_round = {}
                    for round_s, label, score_s in complex_matches:
                        try:
                            rnum = int(round_s)
                            score = float(score_s)
                        except ValueError:
                            continue
                        by_round.setdefault(rnum, {})[label] = score

                    ddg_by_label = {}
                    for rnum, scores in by_round.items():
                        wt = scores.get("WT_")
                        if wt is None:
                            continue
                        for label, score in scores.items():
                            if label == "WT_":
                                continue
                            ddg_by_label.setdefault(label, []).append(score - wt)

                    for label, ddgs in ddg_by_label.items():
                        avg_ddg = _avg(ddgs)
                        if avg_ddg is None:
                            continue

                        # Try to convert MUT_131ALA -> S131A using mut_file info if available
                        mut_name = label
                        m = re.match(r'^MUT_(\d+)([A-Z]{3})$', label)
                        if m:
                            pos = int(m.group(1))
                            mut3 = m.group(2)
                            mut1 = _AA3_TO_AA1.get(mut3, mut3[0])
                            wt1 = wt_by_pos_mut.get((pos, mut1))
                            if wt1:
                                mut_name = f"{wt1}{pos}{mut1}"
                            else:
                                mut_name = f"{pos}{mut1}"

                        mutations_data.append((run_name, mut_name, avg_ddg))
            
            if not mutations_data:
                print(f"Warning: Could not extract DDG values from {ddg_path}", file=sys.stderr)
                print(f"  Found {len(mutations_list)} mutations in mutation file", file=sys.stderr)
                
                # Check what's actually in the file
                has_complex = "COMPLEX:" in content
                has_mutation = "MUTATION:" in content
                
                if has_complex and not has_mutation:
                    print(f"  ⚠ File contains COMPLEX lines but no MUTATION: lines.", file=sys.stderr)
                    print(f"  This is OK for newer cartesian_ddg; parse_ddg will attempt to compute ddG from WT_ vs MUT_* totals.", file=sys.stderr)
                elif not has_complex and not has_mutation:
                    print(f"  .ddg file appears empty or in unexpected format", file=sys.stderr)
                else:
                    print(f"  .ddg file format may be different than expected", file=sys.stderr)
            
    except Exception as e:
        print(f"Error parsing .ddg file {ddg_path}: {e}", file=sys.stderr)
    
    return mutations_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/parse_ddg.py <input_patterns...> <output_csv>", file=sys.stderr)
        print("   OR: python scripts/parse_ddg.py <output_csv>  # Auto-search for files", file=sys.stderr)
        print("Example: python scripts/parse_ddg.py 'runs/*ddg*/outputs/*.json' results/ddg_scans/out.csv", file=sys.stderr)
        print("Note: Quote glob patterns in zsh to prevent shell expansion issues", file=sys.stderr)
        sys.exit(1)
    
    # Collect input files
    if len(sys.argv) < 3:
        # If only one argument, treat as output and search for files automatically
        output_csv = sys.argv[1]
        print("No input patterns provided. Searching for DDG output files...", file=sys.stderr)
        # Search for JSON files in multiple locations
        json_patterns = [
            'runs/*ddg*/outputs/*.json',
            'runs/*/outputs/*.json',
            '*.json',  # Root directory
            'mutlist.json',  # Common name in root
        ]
        ddg_patterns = [
            'runs/*ddg*/outputs/*.ddg',
            'runs/*/outputs/*.ddg',
            '*.ddg',  # Root directory
            'mutlist.ddg',  # Common name in root
        ]
        inputs = []
        for pattern in json_patterns + ddg_patterns:
            found = glob.glob(pattern)
            inputs.extend(found)
        inputs = sorted(set(inputs))  # Remove duplicates
    else:
        input_patterns = sys.argv[1:-1]
        output_csv = sys.argv[-1]
        inputs = []
        for pat in input_patterns:
            # Handle both quoted and unquoted patterns
            found = glob.glob(pat)
            inputs.extend(found)
        inputs = sorted(set(inputs))
    
    if not inputs:
        if len(sys.argv) < 3:
            print("Error: No DDG output files found.", file=sys.stderr)
            print("Searched in:", file=sys.stderr)
            print("  - runs/*ddg*/outputs/", file=sys.stderr)
            print("  - runs/*/outputs/", file=sys.stderr)
            print("  - Current directory (*.json, *.ddg)", file=sys.stderr)
        else:
            print(f"Error: No input files found matching patterns: {sys.argv[1:-1]}", file=sys.stderr)
        print("Hint: Check that Rosetta DDG run completed and produced output files.", file=sys.stderr)
        print("      Rosetta may write files to current directory instead of outputs/", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(inputs)} input file(s)", file=sys.stderr)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    # Collect all mutations
    all_mutations = []
    json_files = [f for f in inputs if f.endswith('.json')]
    ddg_files = [f for f in inputs if f.endswith('.ddg')]
    
    # Try JSON files first
    if json_files:
        print(f"Processing {len(json_files)} JSON file(s)...", file=sys.stderr)
        for jpath in json_files:
            run_name = extract_run_name(jpath)
            mutations = parse_json_file(jpath, run_name)
            all_mutations.extend(mutations)
            if mutations:
                print(f"  ✓ {jpath}: {len(mutations)} mutations", file=sys.stderr)
            else:
                print(f"  ✗ {jpath}: No valid mutations found", file=sys.stderr)
    
    # Fallback to .ddg files if no JSON data found
    if not all_mutations and ddg_files:
        print(f"No JSON data found. Attempting to parse {len(ddg_files)} .ddg file(s)...", file=sys.stderr)
        # Try to find mutation file in configs
        mut_file = "configs/rosetta/mutlist.mut"
        for ddg_path in ddg_files:
            run_name = extract_run_name(ddg_path)
            mutations = parse_ddg_file(ddg_path, mut_file, run_name)
            all_mutations.extend(mutations)
            if mutations:
                print(f"  ✓ {ddg_path}: {len(mutations)} mutations", file=sys.stderr)
            else:
                print(f"  ✗ {ddg_path}: No mutations extracted", file=sys.stderr)
    
    # Write CSV output
    with open(output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["run", "mutation", "ddg_reu"])
        
        if all_mutations:
            for run, mut, ddg in all_mutations:
                w.writerow([run, mut, ddg])
            print(f"\n✓ Wrote {len(all_mutations)} mutations to {output_csv}", file=sys.stderr)
        else:
            print(f"\n✗ No mutations found in any input files.", file=sys.stderr)
            print("Troubleshooting:", file=sys.stderr)
            print("  1. Verify Rosetta DDG run completed successfully", file=sys.stderr)
            print("  2. Check that -ddg:json true flag was used in rosetta_ddg.sh", file=sys.stderr)
            print("  3. Check if files are in root directory (not outputs/)", file=sys.stderr)
            print("  4. Verify JSON files contain 'mutations' key with 'ddg' values", file=sys.stderr)
            print("  5. Check file permissions and paths", file=sys.stderr)

if __name__ == '__main__':
    main()
