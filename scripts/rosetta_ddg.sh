#!/usr/bin/env bash
# Rosetta cartesian_ddg script with validation
# Usage: bash scripts/rosetta_ddg.sh [model.pdb] [mutlist.mut]

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

# Validate and set MODEL parameter
MODEL=${1:-}
if [ -z "$MODEL" ]; then
    # Find most recent relaxed structure
    MODEL=$(ls -t runs/*relax*/outputs/*.pdb 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then
        echo "Error: No relaxed PDB files found. Run relaxation first." >&2
        echo "Expected: runs/*relax*/outputs/*.pdb" >&2
        exit 1
    fi
    echo "Using most recent relaxed structure: $MODEL" >&2
fi

# Validate MODEL file exists and is readable
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL" >&2
    exit 1
fi

if [ ! -r "$MODEL" ]; then
    echo "Error: Model file is not readable: $MODEL" >&2
    exit 1
fi

# Validate and set MUT parameter
MUT=${2:-configs/rosetta/mutlist.mut}

# Validate mutation file exists
if [ ! -f "$MUT" ]; then
    echo "Error: Mutation file not found: $MUT" >&2
    exit 1
fi

if [ ! -r "$MUT" ]; then
    echo "Error: Mutation file is not readable: $MUT" >&2
    exit 1
fi

# Basic format validation of mutation file
if ! grep -q "^total [0-9]" "$MUT" 2>/dev/null; then
    echo "Warning: Mutation file may be malformed. Expected 'total N' on first non-comment line." >&2
    echo "File: $MUT" >&2
fi

# Validate mutation file format.
# Rosetta cartesian_ddg commonly accepts mutation lines as either:
#   - 3 fields: <wt-aa> <pose-resnum> <mut-aa>
#   - 4 fields: <wt-aa> <pose-resnum> <chain-id> <mut-aa>  (chain is ignored by this script when cleaned)
# Check that mutation lines (non-comment, non-empty, non-number lines) have 3 or 4 fields.
FORMAT_ISSUES=0
while IFS= read -r line; do
    # Skip comments, empty lines, "total" lines, and number-only lines (mutation counts)
    if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]] || [[ "$line" =~ ^total ]] || [[ "$line" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
        continue
    fi
    # Check if this looks like a mutation line (starts with amino acid code)
    if [[ "$line" =~ ^[A-Z][[:space:]]+[0-9]+ ]]; then
        FIELD_COUNT=$(echo "$line" | awk '{print NF}')
        if [ "$FIELD_COUNT" -ne 3 ] && [ "$FIELD_COUNT" -ne 4 ]; then
            echo "Error: Mutation line has unexpected format (expected 3 or 4 fields, found $FIELD_COUNT): $line" >&2
            FORMAT_ISSUES=$((FORMAT_ISSUES + 1))
        fi
    fi
done < "$MUT"

if [ "$FORMAT_ISSUES" -gt 0 ]; then
    echo "" >&2
    echo "Error: Mutation file format issues detected. Please fix before running." >&2
    echo "Expected mutation line format:" >&2
    echo "  - 3 fields: <wt-aa> <pose-resnum> <mut-aa>         (example: S 131 A)" >&2
    echo "  - 4 fields: <wt-aa> <pose-resnum> <chain> <mut-aa> (example: S 131 A A)" >&2
    exit 1
fi

# Set up run directory
TS=$(date +%Y-%m-%d)  # Explicit date format
NAME=$(basename "${MUT}" .mut)
RUN="runs/${TS}_ddg_cart_${NAME}"
if [ -d "$RUN" ]; then
    RUN="runs/${TS}_ddg_cart_${NAME}_$(date +%H%M%S)"
fi
mkdir -p "${RUN}/outputs"
RUN_OUT_ABS="$(cd "${RUN}/outputs" && pwd)"

# Create manifest
echo "# Run manifest" > "${RUN}/manifest.md"
echo "Tool: Rosetta cartesian_ddg" >> "${RUN}/manifest.md"
echo "Inputs: $MODEL, $MUT" >> "${RUN}/manifest.md"
echo "Date: $(date)" >> "${RUN}/manifest.md"

# Detect and validate Rosetta binary
DDG_BIN=""
if [[ -f "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]]; then
    DDG_BIN="cartesian_ddg.static.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/cartesian_ddg.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.macosclangrelease" ]]; then
    DDG_BIN="cartesian_ddg.macosclangrelease"
elif [[ -f "$ROSETTA_BIN/cartesian_ddg.linuxgccrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.linuxgccrelease" ]]; then
    DDG_BIN="cartesian_ddg.linuxgccrelease"
else
    echo "Error: No executable Rosetta cartesian_ddg binary found in $ROSETTA_BIN" >&2
    echo "Expected one of:" >&2
    echo "  - cartesian_ddg.static.macosclangrelease" >&2
    echo "  - cartesian_ddg.macosclangrelease" >&2
    echo "  - cartesian_ddg.linuxgccrelease" >&2
    exit 1
fi

echo "Using Rosetta binary: $DDG_BIN" >&2
echo "Output directory: $RUN/outputs" >&2
echo "" >&2

# Save original working directory
ORIGINAL_PWD=$(pwd)

# Get absolute paths BEFORE changing directory
# Convert relative paths to absolute paths
if [[ "$MODEL" = /* ]]; then
    # Already absolute
    MODEL_ABS="$MODEL"
else
    # Relative path - make absolute
    MODEL_ABS="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
fi

if [[ "$MUT" = /* ]]; then
    # Already absolute
    MUT_ABS="$MUT"
else
    # Relative path - make absolute
    MUT_ABS="$(cd "$(dirname "$MUT")" && pwd)/$(basename "$MUT")"
fi

# IMPORTANT:
# cartesian_ddg mut_file parsing is strict and does NOT tolerate comments/extra text
# in some builds. We therefore generate a cleaned mut_file (no blank lines, no lines
# starting with '#') and pass that to Rosetta.
MUT_CLEAN="${RUN_OUT_ABS}/$(basename "${MUT}" .mut).clean.mut"
awk '
  NF == 0 { next }
  $1 ~ /^#/ { next }
  $1 == "total" { print; next }
  (NF == 1 && $1 ~ /^[0-9]+$/) { print; next }   # per-set mutation count
  (NF == 3) { print; next }                      # wt resnum mut
  (NF == 4) { print $1, $2, $4; next }           # wt resnum chain mut -> wt resnum mut
  { bad = 1; print "Bad line in mutfile:", $0 > "/dev/stderr" }
  END { exit bad }
' "$MUT_ABS" > "$MUT_CLEAN"

# Pre-flight validation: ensure mutations actually match the input PDB.
# This catches the most common failure mode: using the wrong numbering scheme (pose vs PDB),
# wrong chain IDs, or WT amino-acid mismatches. If everything is invalid, cartesian_ddg
# can crash with `Assertion num_mutations>0 failed`.
python3 - "$MODEL_ABS" "$MUT_CLEAN" <<'PY'
import sys

MODEL_ABS = sys.argv[1]
MUT_CLEAN = sys.argv[2]

aa3_to_aa1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}

def load_pose_order_from_pdb(pdb_path: str):
    # Approximate Rosetta pose numbering by residue order in the PDB.
    # Returns:
    #   pose_aa1: list[str]  (1-based pose index -> aa1 at that residue)
    #   pose_pdb: list[tuple] (1-based pose index -> (chain, resseq, icode, resname3))
    pose_aa1 = []
    pose_pdb = []
    seen = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = (line[21].strip() or "")
            resseq_s = line[22:26].strip()
            icode = (line[26].strip() or "")
            if not resseq_s:
                continue
            try:
                resseq = int(resseq_s)
            except ValueError:
                continue
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            resname3 = line[17:20].strip().upper()
            aa1 = aa3_to_aa1.get(resname3)
            if aa1 is None:
                # Skip non-canonical residues for this validation.
                continue
            pose_aa1.append(aa1)
            pose_pdb.append((chain, resseq, icode, resname3))
    return pose_aa1, pose_pdb

def parse_clean_mutfile(mut_path: str):
    # Clean file should contain:
    #   total N
    #   <count>
    #   <wt> <pose_resnum> <mut>
    muts = []
    with open(mut_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("total"):
                continue
            if line.isdigit():
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            wt, resnum_s, mut = parts
            try:
                resnum = int(resnum_s)
            except ValueError:
                continue
            muts.append((wt.upper(), resnum, mut.upper()))
    return muts

pose_aa1, pose_pdb = load_pose_order_from_pdb(MODEL_ABS)
muts = parse_clean_mutfile(MUT_CLEAN)

if not muts:
    print(f"Error: No mutations parsed from cleaned mut_file: {MUT_CLEAN}", file=sys.stderr)
    sys.exit(2)
if not pose_aa1:
    print(f"Error: Could not parse any residues from PDB: {MODEL_ABS}", file=sys.stderr)
    sys.exit(2)

bad = 0
checked = 0
nres = len(pose_aa1)
for wt, resnum, mut in muts:
    checked += 1
    if resnum < 1 or resnum > nres:
        bad += 1
        print(f"Error: Pose residue index out of range: {wt} {resnum} {mut} (pose has {nres} residues)", file=sys.stderr)
        continue
    pdb_chain, pdb_resseq, pdb_icode, pdb_resname3 = pose_pdb[resnum - 1]
    pdb_wt = pose_aa1[resnum - 1]
    if pdb_wt != wt:
        bad += 1
        lbl = f"{pdb_chain}{pdb_resseq}{pdb_icode}" if pdb_icode else f"{pdb_chain}{pdb_resseq}"
        print(
            f"Error: WT mismatch at pose {resnum} (PDB {lbl} {pdb_resname3}): mutfile says {wt}→{mut}, but pose has {pdb_wt}",
            file=sys.stderr,
        )

if bad:
    print("", file=sys.stderr)
    print(f"Error: {bad}/{checked} mutation lines do not match the input structure.", file=sys.stderr)
    print("Tip: cartesian_ddg uses POSE numbering (1..N) for the residue index.", file=sys.stderr)
    sys.exit(2)
PY

# Change to outputs directory so Rosetta writes files there
# Rosetta may write JSON/ddg files to current working directory
cd "${RUN}/outputs" || exit 1

# Run Rosetta cartesian_ddg
if ! "$ROSETTA_BIN/$DDG_BIN" \
    -s "$MODEL_ABS" \
    -ddg:mut_file "$MUT_CLEAN" \
    -ddg:iterations "${DDG_ITERS:-3}" \
    -ddg:cartesian \
    -score:weights ref2015_cart \
    -fa_max_dis 9.0 \
    -ddg:legacy false \
    -ddg:json true \
    -out:path:all .; then
    echo "" >&2
    echo "Error: Rosetta cartesian_ddg execution failed." >&2
    echo "Check the output above for details." >&2
    exit 1
fi

# Move any files from root directory to outputs if they exist
# (Rosetta sometimes writes to original working directory)
cd "$ORIGINAL_PWD" || exit 1
if [ -f "mutlist.json" ] || [ -f "mutlist.ddg" ]; then
    echo "Moving output files from root to outputs directory..." >&2
    [ -f "mutlist.json" ] && mv "mutlist.json" "${RUN}/outputs/" 2>/dev/null || true
    [ -f "mutlist.ddg" ] && mv "mutlist.ddg" "${RUN}/outputs/" 2>/dev/null || true
fi

# Normalize output names for downstream tools/docs.
# cartesian_ddg often names outputs after the mut_file stem (e.g. *.clean.json/*.clean.ddg).
# We copy the first produced JSON/DDG to mutlist.{json,ddg} if those standard filenames
# are missing, so greps and parse scripts work consistently.
shopt -s nullglob
_JSONS=( "${RUN_OUT_ABS}"/*.json )
_DDGS=( "${RUN_OUT_ABS}"/*.ddg )
shopt -u nullglob

if [ ! -f "${RUN_OUT_ABS}/mutlist.json" ] && [ -n "${_JSONS[0]:-}" ]; then
    cp "${_JSONS[0]}" "${RUN_OUT_ABS}/mutlist.json" 2>/dev/null || true
fi
if [ ! -f "${RUN_OUT_ABS}/mutlist.ddg" ] && [ -n "${_DDGS[0]:-}" ]; then
    cp "${_DDGS[0]}" "${RUN_OUT_ABS}/mutlist.ddg" 2>/dev/null || true
fi

# Validate that mutations were actually processed
echo "" >&2
echo "Validating output..." >&2

MUTATIONS_PROCESSED=0
shopt -s nullglob
JSON_FILES=( "${RUN_OUT_ABS}"/*.json )
DDG_FILES=( "${RUN_OUT_ABS}"/*.ddg )
shopt -u nullglob

# Prefer standard names if present; otherwise fall back to first match.
JSON_FILE="${RUN_OUT_ABS}/mutlist.json"
DDG_FILE="${RUN_OUT_ABS}/mutlist.ddg"
if [ ! -f "${JSON_FILE}" ]; then
    JSON_FILE="${JSON_FILES[0]:-}"
fi
if [ ! -f "${DDG_FILE}" ]; then
    DDG_FILE="${DDG_FILES[0]:-}"
fi

if [ -n "${JSON_FILE}" ] && [ -f "${JSON_FILE}" ]; then
    echo "  Found JSON: $(basename "${JSON_FILE}")" >&2
    # Check JSON file for at least one REAL mutation (mut != wt)
    MUTATIONS_IN_JSON=$(python3 -c "
import json
import sys
try:
    with open('${JSON_FILE}', 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    mutations = item.get('mutations', [])
                    if isinstance(mutations, list):
                        for m in mutations:
                            if isinstance(m, dict):
                                wt = m.get('wt')
                                mut = m.get('mut')
                                if wt is not None and mut is not None and wt != mut:
                                    sys.exit(0)
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null && echo "yes" || echo "no")
    
    if [ "$MUTATIONS_IN_JSON" = "yes" ]; then
        MUTATIONS_PROCESSED=1
    fi
fi

if [ -n "${DDG_FILE}" ] && [ -f "${DDG_FILE}" ]; then
    echo "  Found DDG:  $(basename "${DDG_FILE}")" >&2
    # Newer cartesian_ddg writes MUT_* labels on COMPLEX lines (not necessarily MUTATION:)
    if grep -q "MUT_" "${DDG_FILE}" 2>/dev/null; then
        MUTATIONS_PROCESSED=1
    fi
fi

if [ "$MUTATIONS_PROCESSED" -eq 0 ]; then
    echo "" >&2
    echo "⚠ WARNING: No mutations found in output files!" >&2
    echo "  This suggests Rosetta did not process the mutations." >&2
    echo "  Possible causes:" >&2
    echo "    1. Mutation file format issue (check chain IDs)" >&2
    echo "    2. Residue numbers don't match PDB file" >&2
    echo "    3. Wild-type amino acids don't match PDB file" >&2
    echo "  Check ROSETTA_CRASH.log for details." >&2
    echo "" >&2
    echo "  Expected mutation file format:" >&2
    echo "    <wt-aa> <residue-number> <chain-id> <mutant-aa>" >&2
    echo "    Example: S 160 A A" >&2
    echo "" >&2
    echo "Error: No mutations were processed. Failing the run." >&2
    exit 1
fi

echo "✓ Rosetta cartesian_ddg completed" >&2
echo "Results saved to: $RUN/outputs" >&2
echo "Files in outputs:" >&2
ls -lh "${RUN}/outputs/" | tail -n +2 | awk '{print "  " $0}' >&2
