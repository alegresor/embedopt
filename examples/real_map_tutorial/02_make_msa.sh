#!/bin/bash
# Step 2: Generate MSA for 9UGC_A via ColabFold server.
# Run 01_make_minimal_json.py first to produce 9UGC_A_minimal.json.
# Output: 9UGC_A_msa/9UGC_A/msa/0/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LAYERNORM_TYPE=torch_layernorm

pixi run protenix msa \
    --input 9UGC_A_minimal.json \
    --out_dir 9UGC_A_msa \
    --msa_server_mode protenix

echo "MSA ready at $SCRIPT_DIR/9UGC_A_msa/9UGC_A/msa/0"
