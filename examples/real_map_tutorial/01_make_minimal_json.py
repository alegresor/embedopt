#!/usr/bin/env python3
"""
Step 1: Read a FASTA file and produce a minimal input JSON for MSA generation.

The minimal JSON contains only the sequence — no MSA path, no reward, no algorithm.
It is used as input to 02_make_msa.sh (protenix msa).

Usage:
    pixi run python examples/real_map_tutorial/01_make_minimal_json.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
FASTA = HERE / "9UGC_A.fasta"
OUTPUT = HERE / "9UGC_A_minimal.json"


def read_fasta(path: Path) -> tuple[str, str]:
    """Return (name, sequence) from a single-record FASTA file."""
    lines = path.read_text().splitlines()
    name = lines[0].lstrip(">").strip()
    sequence = "".join(line.strip() for line in lines[1:] if line.strip())
    return name, sequence


def main():
    name, sequence = read_fasta(FASTA)
    minimal = [{"name": name, "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}]}]
    with open(OUTPUT, "w") as f:
        json.dump(minimal, f, indent=4)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
