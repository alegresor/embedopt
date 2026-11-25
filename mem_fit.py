import numpy as np, pandas as pd, gemmi, torch
B = "examples/synthetic_map_benchmark/pdb_clean"
def rmax(s):
    X = np.array([c.atom.pos.tolist() for c in gemmi.read_structure(f"{B}/{s}.pdb")[0].all()])
    return 1.2 * np.linalg.norm(X - X.mean(0), axis=1).max()

d = pd.read_csv("mem_probe.csv").groupby(["system", "tokens"], as_index=False).peak_mib.max()
d["r_max"] = d.system.map(rmax)
print(d)

N, R, M = (torch.tensor(d[c].values, dtype=torch.float64) for c in ("tokens", "r_max", "peak_mib"))
A = torch.stack([torch.ones_like(N), N**2, R**3], 1)
coef = torch.linalg.lstsq(A, (M / 1024)[:, None]).solution.squeeze()
print("m0, a, b =", coef.tolist(), "  residuals (GiB):", (A @ coef - M / 1024).tolist())