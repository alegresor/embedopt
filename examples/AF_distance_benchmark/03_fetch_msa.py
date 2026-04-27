#!/usr/bin/env python3
"""
Cache-aware MSA fetcher for the distance benchmark.

For each bare minimal JSON in `json_minimal/`, ensures both:
  - the MSA `.a3m` files exist in the cache directory
  - a per-system `*_minimal-update-msa.json` exists with paths pointing at them

Cache layout:
    {MSA_CACHE_DIR}/{system}_minimal/msa/0/{pairing.a3m, non_pairing.a3m}

Per-system state machine (cache? × json?):

  cache ✓ + json ✓ (paths valid)  → [skip ] nothing to do
  cache ✓ + json ✗                → [write] write json ourselves (no server work)
  cache ✗ + (json anything)       → [fetch] invoke `protenix msa` (CLI writes both
                                            the a3m files and the *-update-msa.json)

Defaults:
- MSA_CACHE_DIR env var, falls back to ./msa_cache (created on first run)
- Backend = "protenix"

Usage:
    python 03_fetch_msa.py                       # all systems
    python 03_fetch_msa.py 1IXC_A 2BJ7_A         # specific systems
    MSA_CACHE_DIR=/path/to/cache python 03_fetch_msa.py 1IXC_A   # custom cache location
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CACHE_DIR = "./msa_cache"
JSON_DIR = Path("json_minimal")
MSA_BACKEND = "protenix"


def cache_paths_for(cache_dir: Path, system: str) -> tuple[Path, Path]:
    """Return (paired, non_paired) paths in the cache for a given system."""
    base = cache_dir / f"{system}_minimal" / "msa" / "0"
    return base / "pairing.a3m", base / "non_pairing.a3m"


def cached(cache_dir: Path, system: str) -> bool:
    paired, non_paired = cache_paths_for(cache_dir, system)
    return paired.exists() and non_paired.exists()


def json_paths_match_cache(update_json: Path) -> bool:
    """True iff every pairedMsaPath/unpairedMsaPath in the JSON points at an existing file."""
    try:
        with open(update_json) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    for job in data:
        for seq in job.get("sequences", []):
            if "proteinChain" not in seq:
                continue
            for key in ("pairedMsaPath", "unpairedMsaPath"):
                p = seq["proteinChain"].get(key)
                if p is None or not Path(p).exists():
                    return False
    return True


def download_via_protenix(minimal_json: Path, cache_dir: Path) -> None:
    """Invoke `protenix msa` to populate the cache and write *-update-msa.json.

    The CLI writes a3m files to `{cache_dir}/{json.name}/msa/0/...` and a JSON
    `{minimal_json.stem}-update-msa.json` next to the input. Because step 02 sets
    name = "{system}_minimal", both land where we expect.
    """
    env = {**os.environ, "LAYERNORM_TYPE": "torch_layernorm"}
    subprocess.run(
        [
            "protenix", "msa",
            "-i", str(minimal_json),
            "-o", str(cache_dir),
            "-m", MSA_BACKEND,
        ],
        check=True,
        env=env,
    )


def write_update_msa_json(minimal_json: Path, cache_dir: Path, output_json: Path) -> None:
    """Write a *-update-msa.json by injecting cache paths into the bare minimal JSON.

    Mirrors what the protenix CLI would write, without doing server work.
    Called when the cache is populated but the JSON is missing or stale.
    """
    with open(minimal_json) as f:
        data = json.load(f)
    for job in data:
        system_name = job.get("name")
        paired, non_paired = cache_paths_for(cache_dir, system_name.removesuffix("_minimal"))
        for seq in job.get("sequences", []):
            if "proteinChain" in seq:
                seq["proteinChain"]["pairedMsaPath"] = str(paired)
                seq["proteinChain"]["unpairedMsaPath"] = str(non_paired)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("systems", nargs="*", help="Specific systems to process (default: all minimal JSONs)")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("MSA_CACHE_DIR", DEFAULT_CACHE_DIR),
        help="MSA cache root (default: $MSA_CACHE_DIR or ./msa_cache)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover systems
    if args.systems:
        minimal_jsons = []
        for s in args.systems:
            p = JSON_DIR / f"{s}_minimal.json"
            if not p.exists():
                sys.exit(f"Minimal JSON not found: {p}. Run step 02 first.")
            minimal_jsons.append(p)
    else:
        minimal_jsons = sorted(JSON_DIR.glob("*_minimal.json"))
        # Skip previously-written *-update-msa.json files when globbing
        minimal_jsons = [p for p in minimal_jsons if "-update-msa" not in p.name]
        if not minimal_jsons:
            sys.exit(f"No *_minimal.json files in {JSON_DIR}/. Run step 02 first.")

    print(f"MSA cache root: {cache_dir}")
    print(f"Backend: {MSA_BACKEND}")
    print(f"Systems to process: {len(minimal_jsons)}")
    print()

    n_skip = 0
    n_write = 0
    n_fetch = 0
    for minimal_json in minimal_jsons:
        system = minimal_json.stem.removesuffix("_minimal")
        update_json = JSON_DIR / f"{system}_minimal-update-msa.json"

        cache_ok = cached(cache_dir, system)
        json_ok = update_json.exists() and json_paths_match_cache(update_json)

        if cache_ok and json_ok:
            print(f"[skip ] {system}")
            n_skip += 1
        elif cache_ok and not json_ok:
            print(f"[write] {system}")
            write_update_msa_json(minimal_json, cache_dir, update_json)
            n_write += 1
        else:
            print(f"[fetch] {system} — calling protenix msa")
            download_via_protenix(minimal_json, cache_dir)
            if not cached(cache_dir, system) or not update_json.exists():
                sys.exit(f"protenix msa finished but expected outputs missing for {system}")
            n_fetch += 1

    print()
    print(f"Done. {n_skip} skipped, {n_write} rewritten, {n_fetch} fetched. Total: {len(minimal_jsons)} systems.")


if __name__ == "__main__":
    main()
