"""Benchmark inputs for a (multi-chain) PDB entry, in the format of the synthetic-map benchmark.

    python make_target.py 7N6N          # before MSA: pdb_clean + json_minimal (+ cfg without MSA paths)
    protenix msa --input $B/json_minimal/7N6N_minimal.json --out_dir $B/msa --msa_server_mode protenix
    python make_target.py 7N6N          # after MSA: cfg with MSA paths

Input : targets/<ID>/<ID>.cif
Output: $B/pdb_clean/<ID>.pdb, $B/json_minimal/<ID>_minimal.json, $B/stage3a_cfgs/<ID>_benchmark.json

x* is cleaned as 03_clean_cif_to_pdb.py (polymer only, B = 50, no ANISOU, P1 cell = bbox + 10 A, centered), with all
polymer chains kept. moments.invariants.match_atoms pairs atoms by global residue ordinal, so x* and the prediction
must have identical residue lists: each entity is trimmed to the label_seq range modeled in every copy, the Protenix
sequence is that range, and an unmodeled residue inside it is an error.
"""
import json, sys
from pathlib import Path
import gemmi
import numpy as np

ID = sys.argv[1]
B = Path("examples/synthetic_map_benchmark").resolve()
PAD = 10.0

st = gemmi.read_structure(str(Path("targets") / ID / f"{ID}.cif"))
st.setup_entities(); st.remove_hydrogens(); st.remove_alternative_conformations(); st.remove_ligands_and_waters()
st.remove_empty_chains()
model = st[0]

# Group chains by entity, in entity order (the order of the Protenix sequences).
ents = [e for e in st.entities if e.entity_type == gemmi.EntityType.Polymer]
groups = []
for e in ents:
    chains = [ch.name for ch in model if len(ch.get_polymer()) and ch.get_polymer().subchain_id() in e.subchains]
    if not chains: continue
    full = gemmi.one_letter_code([gemmi.Entity.first_mon(m) for m in e.full_sequence])
    seqs = {c: [r.label_seq for r in model[c].get_polymer()] for c in chains}
    # copies modeled over different ranges (e.g. a protease chain and a peptide of the same entity) become
    # separate sequences; copies with the same range share one (count = number of copies)
    byrange = {}
    for c in chains:
        byrange.setdefault((min(seqs[c]), max(seqs[c])), []).append(c)
    if len(byrange) > 1:
        # merge ranges that nearly coincide (termini differing by a few residues) by intersecting them
        merged = []
        for (lo, hi), cs in sorted(byrange.items(), key=lambda kv: -(kv[0][1] - kv[0][0])):
            for m in merged:
                if min(hi, m[1]) - max(lo, m[0]) >= 0.9 * max(hi - lo, m[1] - m[0]):
                    m[0], m[1] = max(lo, m[0]), min(hi, m[1]); m[2] += cs; break
            else:
                merged.append([lo, hi, list(cs)])
        byrange = {(lo, hi): cs for lo, hi, cs in merged}
    for (lo, hi), cs in byrange.items():
        for c in cs:
            gap = sorted(set(range(lo, hi + 1)) - set(seqs[c]))
            if gap: sys.exit(f"entity {e.name} chain {c}: unmodeled residues {gap} inside {lo}-{hi}")
        if "X" in full[lo - 1:hi]: sys.exit(f"entity {e.name}: non-standard residue in modeled range: {full[lo - 1:hi]}")
        groups.append(dict(entity=e.name, chains=cs, lo=lo, hi=hi, seq=full[lo - 1:hi], full=full))
        print(f"entity {e.name}: chains {cs}, modeled {lo}-{hi} of {len(full)} "
              f"(dropped {lo - 1} N-term, {len(full) - hi} C-term): {full[lo - 1:hi]}")

left = [ch.name for ch in model if len(ch.get_polymer()) and ch.name not in {c for g in groups for c in g["chains"]}]
if left: sys.exit(f"polymer chains {left} not assigned to any entity")

# Trim to the common range, reorder chains by entity, uniform B, center in the P1 box.
order = [c for g in groups for c in g["chains"]]
rng = {c: (g["lo"], g["hi"]) for g in groups for c in g["chains"]}
for ch in model:
    for i in reversed(range(len(ch))):
        if ch[i].label_seq is None or not rng[ch.name][0] <= ch[i].label_seq <= rng[ch.name][1]: del ch[i]
for name in [ch.name for ch in model if ch.name not in order]: model.remove_chain(name)
new = gemmi.Model("1")
for c in order: new.add_chain(model[c])
st.add_model(new); del st[0]
st.renumber_models()

atoms = [a for ch in st[0] for r in ch for a in r]
X = np.array([a.pos.tolist() for a in atoms])
size = X.max(0) - X.min(0) + 2 * PAD
shift = size / 2 - (X.max(0) + X.min(0)) / 2
for a, x in zip(atoms, X + shift):
    a.pos = gemmi.Position(*x); a.b_iso = 50.0; a.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)
st.cell = gemmi.UnitCell(*size, 90, 90, 90); st.spacegroup_hm = "P 1"
for d in ("pdb_clean", "json_minimal"): (B / d).mkdir(exist_ok=True)
st.write_pdb(str(B / "pdb_clean" / f"{ID}.pdb"))
print(f"x*: {len(order)} chains {order}, {sum(len(ch) for ch in st[0])} residues, {len(atoms)} atoms, "
      f"cell {np.round(size, 1).tolist()} A")

# Minimal JSON (MSA input) and benchmark config (copy of an existing one with this target's sequences/paths).
chains = [{"proteinChain": {"sequence": g["seq"], "count": len(g["chains"])}} for g in groups]
json.dump([{"name": f"{ID}_minimal", "sequences": chains}],
          open(B / "json_minimal" / f"{ID}_minimal.json", "w"), indent=2)
cfg = json.load(open(B / "stage3a_cfgs" / "8AHU_A_benchmark.json"))[0]
cfg["name"] = f"{ID}_embedopt_0.1_5A"
def a3m_query(p):
    """First sequence of an a3m (the query), insertions (lowercase) removed."""
    lines = Path(p).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(">"))
    return "".join(c for c in lines[i + 1].strip() if not c.islower())

# Pick, for each sequence, the MSA directory whose query equals it (the server's numbering of msa/<k> need not
# follow the input order).
dirs = sorted((B / "msa" / f"{ID}_minimal" / "msa").glob("*/non_pairing.a3m"))
missing = []
for k, c in enumerate(chains):
    hit = [d.parent for d in dirs if a3m_query(d) == c["proteinChain"]["sequence"]]
    if hit:
        d = hit[0]
        c["proteinChain"].update(pairedMsaPath=str(d / "pairing.a3m"), unpairedMsaPath=str(d / "non_pairing.a3m"))
        print(f"sequence {k} ({len(c['proteinChain']['sequence'])} aa): MSA {d.relative_to(B)}")
    else:
        missing.append(k)
cfg["sequences"] = chains
cfg["reward"]["params"].update(ref_pdb_path=str(B / "pdb_clean" / f"{ID}.pdb"),
                               ref_map_path=str(B / "synthetic_maps" / "5.0A" / f"{ID}.map"))
json.dump([cfg], open(B / "stage3a_cfgs" / f"{ID}_benchmark.json", "w"), indent=2)
print(f"wrote {ID}_minimal.json and {ID}_benchmark.json; tokens = {sum(len(g['seq']) * len(g['chains']) for g in groups)}")
if missing: print(f"no MSA whose query matches sequence(s) {missing}: run `protenix msa` (or check "
                  f"{B.name}/msa/{ID}_minimal/msa/*/non_pairing.a3m) and rerun this script")
