#!/usr/bin/env python3
"""
Adds an explicit `algorithm: {type: base}` block to each post-MSA minimal JSON
in `json_minimal/`, in place.

Input:  json_minimal/{system}_minimal-update-msa.json (output of step 03)
Output: same files, with `"algorithm": {"type": "base"}` injected
"""

import json
from pathlib import Path
from tqdm import tqdm


JSON_DIR = Path("json_minimal")


def main():
    json_files = sorted(JSON_DIR.glob("*_minimal-update-msa.json"))
    if not json_files:
        raise SystemExit(f"No *_minimal-update-msa.json files found in {JSON_DIR}/. Run step 03 first.")

    print(f"Found {len(json_files)} post-MSA JSON files. Adding `algorithm: {{type: base}}` to each.")
    n_updated = 0
    n_already_set = 0
    for path in tqdm(json_files):
        with open(path) as f:
            data = json.load(f)

        # Per Protenix conventions each minimal JSON is a list with one job dict.
        # Iterate so we don't break if the convention ever changes.
        for job in data:
            existing = job.get("algorithm", {}).get("type")
            if existing == "base":
                n_already_set += 1
                continue
            if existing is not None and existing != "base":
                raise SystemExit(
                    f"{path.name}: refusing to overwrite existing algorithm '{existing}'. "
                    "This script only sets the algorithm when it is unset or already 'base'."
                )
            job["algorithm"] = {"type": "base"}
            n_updated += 1

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    print(f"Done. Updated {n_updated} jobs; {n_already_set} already had algorithm.type='base'.")


if __name__ == "__main__":
    main()
