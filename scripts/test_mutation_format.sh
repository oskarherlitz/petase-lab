#!/usr/bin/env bash
# Test mutation file format with Rosetta

set -euo pipefail

: "${ROSETTA_BIN:?Set ROSETTA_BIN to Rosetta bin dir}"

MODEL="runs/2025-12-05_relax_cart_v1/outputs/PETase_raw_0008.pdb"

# Test with a single mutation in the format Rosetta expects
cat > /tmp/test_mut_format.mut << 'EOF'
total 1
1
S 131 A A
EOF

echo "Testing mutation file format..."
echo "Mutation file contents:"
cat /tmp/test_mut_format.mut
echo ""

# Get absolute path
MUT_ABS="$(cd "$(dirname /tmp/test_mut_format.mut)" && pwd)/$(basename /tmp/test_mut_format.mut)"
MODEL_ABS="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

echo "Using mutation file: $MUT_ABS"
echo "Using model: $MODEL_ABS"
echo ""

# Find Rosetta binary
DDG_BIN=""
if [[ -f "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]] && [[ -x "$ROSETTA_BIN/cartesian_ddg.static.macosclangrelease" ]]; then
    DDG_BIN="cartesian_ddg.static.macosclangrelease"
else
    echo "Error: Rosetta binary not found" >&2
    exit 1
fi

# Run with verbose output
echo "Running Rosetta cartesian_ddg (this may take a minute)..."
echo "Command: $ROSETTA_BIN/$DDG_BIN -s $MODEL_ABS -ddg:mut_file $MUT_ABS -ddg:iterations 1 -ddg:cartesian -score:weights ref2015_cart -fa_max_dis 9.0 -ddg:legacy false -ddg:json true"
echo ""

mkdir -p /tmp/test_ddg_output
cd /tmp/test_ddg_output

"$ROSETTA_BIN/$DDG_BIN" \
    -s "$MODEL_ABS" \
    -ddg:mut_file "$MUT_ABS" \
    -ddg:iterations 1 \
    -ddg:cartesian \
    -score:weights ref2015_cart \
    -fa_max_dis 9.0 \
    -ddg:legacy false \
    -ddg:json true \
    -out:path:all . 2>&1 | tee /tmp/rosetta_test_output.log

echo ""
echo "Checking output..."
json_files=( *.json )
ddg_files=( *.ddg )

if [ -e "${json_files[0]}" ]; then
    JSON_FILE="${json_files[0]}"
    echo "JSON file created: ${JSON_FILE}"
    python3 -c "
import json
with open('${JSON_FILE}', 'r') as f:
    data = json.load(f)
    print(f'Entries: {len(data)}')
    if len(data) > 0:
        first = data[0]
        mutations = first.get('mutations', [])
        print(f'Mutations in first entry: {len(mutations)}')
        if mutations:
            print(f'First mutation: {mutations[0]}')
        else:
            print('⚠ Mutations array is empty!')
"
else
    echo "⚠ No JSON file created"
fi

if [ -e "${ddg_files[0]}" ]; then
    DDG_FILE="${ddg_files[0]}"
    echo ""
    echo "DDG file created: ${DDG_FILE}"
    if grep -q "^MUTATION:" "${DDG_FILE}"; then
        echo "✓ Found MUTATION lines!"
        grep "^MUTATION:" "${DDG_FILE}" | head -3
    else
        echo "⚠ No MUTATION lines found (only COMPLEX lines)"
    fi
else
    echo "⚠ No DDG file created"
fi
