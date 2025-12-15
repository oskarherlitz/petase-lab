#!/usr/bin/env python3
"""
Rank mutations by ΔΔG (stability change).

Usage:
    python scripts/rank_designs.py <csv_file> [top_k]

Example:
    python scripts/rank_designs.py results/ddg_scans/out.csv 20

Reads CSV with columns: run, mutation, ddg_reu
Sorts by ddg_reu (ascending - most negative = most stabilizing)
Prints top k results
"""

import sys
import csv
import os

def main():
    # Validate arguments
    if len(sys.argv) < 2:
        print("Usage: python scripts/rank_designs.py <csv_file> [top_k]", file=sys.stderr)
        print("Example: python scripts/rank_designs.py results/ddg_scans/out.csv 20", file=sys.stderr)
        sys.exit(1)
    
    csv_file = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    # Validate CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}", file=sys.stderr)
        sys.exit(1)
    
    # Validate k is positive
    if k <= 0:
        print(f"Error: top_k must be positive, got: {k}", file=sys.stderr)
        sys.exit(1)
    
    # Read and parse CSV
rows = []
    invalid_rows = 0
    
    try:
        with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
            
            # Validate required columns
            required_cols = ['run', 'mutation', 'ddg_reu']
            if not reader.fieldnames:
                print("Error: CSV file appears to be empty or has no header row.", file=sys.stderr)
                sys.exit(1)
            
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: CSV missing required columns: {missing_cols}", file=sys.stderr)
                print(f"Found columns: {list(reader.fieldnames)}", file=sys.stderr)
                print(f"Required columns: {required_cols}", file=sys.stderr)
                sys.exit(1)
            
            # Parse rows
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    ddg_str = row.get('ddg_reu', '').strip()
                    if not ddg_str:
                        print(f"Warning: Row {row_num}: Missing ddg_reu value, skipping", file=sys.stderr)
                        invalid_rows += 1
                        continue
                    
                    ddg_value = float(ddg_str)
                    row['ddg_reu'] = ddg_value
            rows.append(row)
                    
                except ValueError as e:
                    print(f"Warning: Row {row_num}: Invalid ddg_reu value '{row.get('ddg_reu', '')}': {e}, skipping", file=sys.stderr)
                    invalid_rows += 1
                    continue
                except KeyError as e:
                    print(f"Warning: Row {row_num}: Missing required column: {e}, skipping", file=sys.stderr)
                    invalid_rows += 1
            continue

    except IOError as e:
        print(f"Error: Cannot read CSV file {csv_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Report parsing results
    if invalid_rows > 0:
        print(f"Warning: Skipped {invalid_rows} invalid row(s)", file=sys.stderr)
    
    if not rows:
        print("No data found in CSV file.", file=sys.stderr)
        print("Troubleshooting:", file=sys.stderr)
        print("  1. Verify CSV file has data rows (not just header)", file=sys.stderr)
        print("  2. Check that ddg_reu column contains numeric values", file=sys.stderr)
        print("  3. Verify CSV format: run,mutation,ddg_reu", file=sys.stderr)
        sys.exit(1)
    
# Sort by ddg_reu (ascending - most negative = most stabilizing)
rows.sort(key=lambda x: x['ddg_reu'])

    # Print results
    print(f"Found {len(rows)} valid mutation(s), showing top {min(k, len(rows))}:", file=sys.stderr)
    print(file=sys.stderr)  # Blank line
    
    # Print header
    print(','.join(rows[0].keys()))
    
    # Print top k rows
    for row in rows[:k]:
        print(','.join(str(row[col]) for col in rows[0].keys()))

if __name__ == '__main__':
    main()
