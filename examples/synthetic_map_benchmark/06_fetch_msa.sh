#!/bin/bash
# Step 6: Fetch MSAs for all proteins via the Protenix MSA server.
# Run 05_make_minimal_json.py first to produce json_minimal/*.json.
# Output: msa/ and json_minimal/<pdbid>_<chain>_minimal-add-msa.json
#
# Usage:
#   bash 06_fetch_msa.sh          # all 77 systems
#   bash 06_fetch_msa.sh --test   # only 3 systems (8AHU_A, 8ANE_A, 8AU1_A)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LAYERNORM_TYPE=torch_layernorm

json_dir="$SCRIPT_DIR/json_minimal"
msa_dir="$SCRIPT_DIR/msa"
mkdir -p "$msa_dir"

TEST_MODE=false
if [[ "${1:-}" == "--test" ]]; then
    TEST_MODE=true
fi

if $TEST_MODE; then
    json_files=("$json_dir/8AHU_A_minimal.json" "$json_dir/8ANE_A_minimal.json" "$json_dir/8AU1_A_minimal.json")
    echo "Test mode: running on ${#json_files[@]} systems"
else
    mapfile -t json_files < <(ls "$json_dir"/*.json 2>/dev/null)
fi

total=${#json_files[@]}
echo "Found $total JSON files to process"
echo ""

count=0
failed=0
for json_file in "${json_files[@]}"; do
    if [ -f "$json_file" ]; then
        count=$((count + 1))
        base_name=$(basename "$json_file")
        echo "[$count/$total] Processing: $base_name"

        if pixi run protenix msa \
            --input "$json_file" \
            --out_dir "$msa_dir" \
            --msa_server_mode protenix; then
            echo "  Done"
        else
            echo "  Failed"
            failed=$((failed + 1))
        fi
        echo ""
    fi
done

echo "Done! Processed $count files, failed $failed"
echo "MSA results saved to: $msa_dir"
