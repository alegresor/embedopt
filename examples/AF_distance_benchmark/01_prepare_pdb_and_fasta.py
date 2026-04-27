#!/usr/bin/env python3
"""
Download CIF + FASTA from RCSB and extract per-chain PDB + FASTA in one pass.

For each PDBID_CHAIN in benchmark_systems.txt:
  1. Download the full CIF and FASTA from RCSB into a tempdir.
  2. Extract the target chain's polymer atoms → pdb_extracted/{system}.pdb.
  3. Extract the target chain's sequence       → fasta_extracted/{system}.fasta.
  4. Validate the PDB sequence is a subsequence of the FASTA sequence.
  5. Clean up the tempdir.

Skips systems whose extracted files already exist.

Outputs (the only artifacts that persist):
  pdb_extracted/{system}.pdb
  fasta_extracted/{system}.fasta
"""

import re
import sys
import time
import tempfile
import urllib.request
from pathlib import Path

from SFC_Torch import PDBParser
from tqdm import tqdm


SYSTEMS_FILE = "benchmark_systems.txt"
PDB_DIR = Path("pdb_extracted")
FASTA_DIR = Path("fasta_extracted")


# ---------------------------------------------------------------------------
# RCSB downloads
# ---------------------------------------------------------------------------

def download_rcsb(pdb_id: str, cif_path: Path, fasta_path: Path) -> None:
    """Fetch the CIF + FASTA for one PDB entry into the given paths."""
    valid = pdb_id.upper()
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{valid}.cif", str(cif_path))
    urllib.request.urlretrieve(f"https://www.rcsb.org/fasta/entry/{valid}", str(fasta_path))


# ---------------------------------------------------------------------------
# Chain extraction from CIF → PDB
# ---------------------------------------------------------------------------

def extract_chain_pdb(cif_path: Path, chain_id: str, out_pdb: Path) -> None:
    """Write a per-chain polymer PDB. Falls back to label_asym_id if needed."""
    st = PDBParser(str(cif_path))
    st_chain = st.selection(f"//{chain_id};polymer")

    # Fallback: auth_asym_id selection failed; try the corresponding label_asym_id.
    if st_chain is None:
        label_chain_id = None
        with open(cif_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and chain_id in line:
                    parts = line.split()
                    if len(parts) >= 14 and parts[13] == chain_id:
                        label_chain_id = parts[6]
                        break
        if label_chain_id and label_chain_id != chain_id:
            st_chain = st.selection(f"//{label_chain_id};polymer")

    if st_chain is None:
        raise RuntimeError(f"could not select chain '{chain_id}' from {cif_path.name}")

    st_chain.savePDB(str(out_pdb))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Chain extraction from FASTA
# ---------------------------------------------------------------------------

def _check_auth_match(header: str, target_chain: str) -> bool:
    parts = header.split("|")
    if len(parts) < 2:
        return False
    return bool(re.search(rf"\[auth\s+{target_chain}\]", parts[1]))


def _check_label_match(header: str, target_chain: str) -> bool:
    parts = header.split("|")
    if len(parts) < 2:
        return False
    chains_part = parts[1]
    patterns = [
        rf"\bChain\s+{target_chain}\b",
        rf"\bChains\s+{target_chain}(?![^\[]*\[auth)",
        rf",\s*{target_chain}(?:\s|,|$)",
    ]
    return any(re.search(p, chains_part) for p in patterns)


def extract_chain_fasta(fasta_path: Path, chain_id: str, out_fasta: Path, system: str) -> None:
    """Find the FASTA entry for the target chain and write it. Prefers [auth X] over label matches."""
    matches = []
    current_header, current_seq = None, []

    def _flush():
        if current_header and current_seq:
            is_auth = _check_auth_match(current_header, chain_id)
            is_label = _check_label_match(current_header, chain_id)
            if is_auth or is_label:
                matches.append({
                    "header": current_header,
                    "sequence": "".join(current_seq),
                    "is_auth": is_auth,
                })

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                current_header, current_seq = line, []
            elif line:
                current_seq.append(line)
    _flush()

    if not matches:
        raise RuntimeError(f"chain '{chain_id}' not found in {fasta_path.name}")

    # Prefer entries that use the [auth X] annotation
    auth_matches = [m for m in matches if m["is_auth"]]
    target = auth_matches[0] if auth_matches else matches[0]

    with open(out_fasta, "w") as f:
        f.write(f">{system}|Chain {chain_id}\n")
        seq = target["sequence"]
        for i in range(0, len(seq), 80):
            f.write(seq[i : i + 80] + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_subsequence(subseq: str, seq: str) -> bool:
    """True if subseq's residues appear in order (not necessarily contiguously) in seq."""
    seq_iter = iter(seq)
    return all(res in seq_iter for res in subseq)


def validate_extraction(pdb_path: Path, fasta_path: Path) -> None:
    pdb_seq = PDBParser(str(pdb_path)).sequence.strip().upper()
    fasta_seq = ""
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                fasta_seq += line.strip()
    fasta_seq = fasta_seq.upper()
    if not is_subsequence(pdb_seq, fasta_seq):
        raise RuntimeError(
            f"PDB sequence (len {len(pdb_seq)}) is not a subsequence of FASTA (len {len(fasta_seq)})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PDB_DIR.mkdir(exist_ok=True)
    FASTA_DIR.mkdir(exist_ok=True)

    if not Path(SYSTEMS_FILE).exists():
        sys.exit(f"{SYSTEMS_FILE} not found in current directory.")

    with open(SYSTEMS_FILE) as f:
        systems = [line.strip() for line in f if line.strip()]
    print(f"Found {len(systems)} systems in {SYSTEMS_FILE}")
    print("=" * 80)

    n_skipped = 0
    n_done = 0
    failures = []

    for system in tqdm(systems, desc="Processing systems"):
        out_pdb = PDB_DIR / f"{system}.pdb"
        out_fasta = FASTA_DIR / f"{system}.fasta"

        if out_pdb.exists() and out_fasta.exists():
            n_skipped += 1
            continue

        try:
            pdb_id, chain_id = system.split("_", 1)
        except ValueError:
            failures.append((system, "system name must be PDBID_CHAIN"))
            continue

        try:
            with tempfile.TemporaryDirectory(prefix=f"rcsb_{system}_") as td:
                tmp = Path(td)
                cif_tmp = tmp / f"{system}.cif"
                fasta_tmp = tmp / f"{system}.fasta"

                download_rcsb(pdb_id, cif_tmp, fasta_tmp)
                extract_chain_pdb(cif_tmp, chain_id, out_pdb)
                extract_chain_fasta(fasta_tmp, chain_id, out_fasta, system)
                validate_extraction(out_pdb, out_fasta)

            n_done += 1
            time.sleep(0.1)  # be polite to RCSB
        except Exception as e:
            failures.append((system, str(e)))
            for p in (out_pdb, out_fasta):
                if p.exists():
                    p.unlink()
            tqdm.write(f"✗ {system}: {e}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total systems:  {len(systems)}")
    print(f"Already done:   {n_skipped}")
    print(f"Newly written:  {n_done}")
    print(f"Failed:         {len(failures)}")
    if failures:
        print("\nFailures:")
        for system, reason in failures:
            print(f"  {system}: {reason}")


if __name__ == "__main__":
    main()
