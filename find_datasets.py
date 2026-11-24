"""Shortlist real cryo-EM datasets for pose-free moment guidance.

EMPIAR (raw data, imageset categories) -> EMDB (resolution, symmetry, sample) -> PDB (residues, chains).
Usage:  pip install requests pandas
        python find_datasets.py [--max-id 13700] [--workers 4] [--rps 5]
Output: candidates.csv (filtered + ranked), all_entries.csv (everything), cache/*.json (re-runs are free).
"""
import argparse, json, re, threading, time, concurrent.futures as cf
from pathlib import Path
import requests, pandas as pd

S, CACHE = requests.Session(), Path("cache")
S.headers["User-Agent"] = "cryoem-dataset-survey (sorokin@uchicago.edu)"
CACHE.mkdir(exist_ok=True)
# Sample-text markers of membrane environments / non-protein density in every image.
MEMBRANE = re.compile(r"nanodisc|detergent|micelle|\bddm\b|lmng|\bgdn\b|digitonin|amphipol|saposin|salipro|"
                      r"liposome|\bmsp1|lipid|membrane|transporter|channel|receptor", re.I)


class RateLimit:
    """Global cap on request rate across all threads."""
    def __init__(self, rps): self.dt, self.t, self.lock = 1.0 / rps, 0.0, threading.Lock()
    def wait(self):
        with self.lock:
            self.t = max(self.t + self.dt, time.monotonic()); d = self.t - time.monotonic()
        if d > 0: time.sleep(d)


RL = RateLimit(5)


def get(url, key, payload=None):
    """Cached JSON fetch. Only successful or 404 responses are cached; 429/5xx back off and retry."""
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    for k in range(5):
        RL.wait()
        try:
            r = S.post(url, json=payload, timeout=60) if payload else S.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** k); continue
        if r.status_code == 404:
            f.write_text("null"); return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 2 ** k))); continue
        r.raise_for_status()
        j = r.json(); f.write_text(json.dumps(j)); return j
    print(f"giving up on {url} (not cached; rerun to retry)")
    return None


def walk(o, key):
    """All values stored under `key` anywhere in nested JSON (schemas are deep and change)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o: yield from walk(v, key)


def num(v):
    v = v.get("valueOf_", v.get("value")) if isinstance(v, dict) else v
    try: return float(v)
    except (TypeError, ValueError): return None


def empiar(i):
    j = get(f"https://www.ebi.ac.uk/empiar/api/entry/EMPIAR-{i}/", f"empiar_{i}")
    if not j: return None
    e = next(iter(j.values()))
    cats = [str(s.get("category", "")) for s in e.get("imagesets", [])]
    return dict(empiar=i, title=e.get("title", ""), imagesets="; ".join(cats),
                has_particles=any("particle" in c.lower() for c in cats),
                emd=sorted(set(re.findall(r"EMD-\d+", json.dumps(e.get("cross_references", e))))))


def emdb(emd):
    j = get(f"https://www.ebi.ac.uk/emdb/api/entry/{emd}", emd) or {}
    res = [r for r in map(num, walk(j, "resolution")) if r]
    # Membrane markers live in admin.keywords, the sample, and the specimen buffer (detergents).
    adm = j.get("admin", {})
    text = json.dumps([adm.get("title", ""), adm.get("keywords", ""), j.get("sample", {}), list(walk(j, "buffer"))])
    return dict(emd=emd, method=",".join(sorted({str(m) for m in walk(j, "method")}))[:40],
                resolution=min(res) if res else None,
                symmetry=",".join(sorted({str(p) for p in walk(j, "point_group")})) or "unreported",
                membrane_flag=bool(MEMBRANE.search(text)),
                pdb=sorted({str(p).upper() for p in walk(j, "pdb_id")}))


GQL = """query($ids: [String!]!) { entries(entry_ids: $ids) { rcsb_id
  struct_keywords { pdbx_keywords }
  rcsb_entry_info { deposited_polymer_monomer_count deposited_modeled_polymer_monomer_count
    deposited_polymer_entity_instance_count polymer_entity_count_protein
    polymer_entity_count_nucleic_acid molecular_weight } } }"""


def pdb_batch(ids):
    """One RCSB GraphQL request per 100 PDB IDs."""
    j = get("https://data.rcsb.org/graphql", f"pdb_{ids[0]}_{ids[-1]}_{len(ids)}",
            {"query": GQL, "variables": {"ids": ids}}) or {}
    out = []
    for e in (j.get("data") or {}).get("entries") or []:
        info = e.get("rcsb_entry_info") or {}
        n, m = info.get("deposited_polymer_monomer_count"), info.get("deposited_modeled_polymer_monomer_count")
        out.append(dict(pdb=e["rcsb_id"], n_res=n, model_cov=(m / n if n and m else None),
                        n_chains=info.get("deposited_polymer_entity_instance_count"),
                        n_protein_ent=info.get("polymer_entity_count_protein"),
                        n_na_ent=info.get("polymer_entity_count_nucleic_acid"),
                        mw_kda=info.get("molecular_weight"),
                        keywords=(e.get("struct_keywords") or {}).get("pdbx_keywords", "")))
    return out


def pmap(f, xs, w):
    with cf.ThreadPoolExecutor(w) as ex:
        return [r for r in ex.map(f, xs) if r]


def main(a):
    RL.dt = 1.0 / a.rps
    E = pd.DataFrame(pmap(empiar, range(10001, a.max_id + 1), a.workers)).explode("emd").dropna(subset=["emd"])
    M = pd.DataFrame(pmap(emdb, E.emd.unique(), a.workers)).explode("pdb")
    ids = sorted(M.pdb.dropna().unique())
    P = pd.DataFrame([r for b in pmap(pdb_batch, [ids[i:i + 100] for i in range(0, len(ids), 100)], a.workers) for r in b])
    df = E.merge(M, on="emd", how="left").merge(P, on="pdb", how="left")
    df.to_csv("all_entries.csv", index=False)

    keep = (df.method.str.contains("singleParticle", na=False) & (df.resolution <= a.max_res)
            & ~df.membrane_flag & ~df.keywords.str.contains("MEMBRANE|TRANSPORT", na=False)
            & (df.n_na_ent.fillna(0) == 0) & df.n_res.between(a.min_res_count, a.max_res_count)
            & (df.model_cov >= a.min_cov))
    c = df[keep].assign(c1=lambda d: d.symmetry.str.fullmatch("C1"))
    c = c.sort_values(["has_particles", "c1", "resolution"], ascending=[False, False, True])
    c.drop_duplicates(["empiar", "emd"]).to_csv("candidates.csv", index=False)
    print(f"{len(df)} EMPIAR-EMDB-PDB rows, {len(c)} candidates -> candidates.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-id", type=int, default=13700)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rps", type=float, default=5.0)            # global requests/second
    p.add_argument("--max-res", type=float, default=3.5)          # Å
    p.add_argument("--min-res-count", type=int, default=250)      # total residues in the model
    p.add_argument("--max-res-count", type=int, default=1500)     # Protenix memory on 16 GB
    p.add_argument("--min-cov", type=float, default=0.85)         # modeled / deposited residues
    main(p.parse_args())