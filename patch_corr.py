"""Adds M12=corr / M3=corr to moments/image_moments.py (ImageLoss) and lets load_data read the masks of the data file.

    python patch_corr.py            # edits moments/image_moments.py in place (backup: image_moments.py.bak)

corr: per-entry normalized correlation with the data, scale-invariant per shell pair (c2) and shell triple (c3), on the
bias-free entries the data reproduce (masks M2, M3t and sign3 from `python -m moments.real_data export`):
    m12 = 1 - mean_{(i,j) in M2} Re<c2m_.ij, c2d_.ij>_q / (|c2m_.ij| |c2d_.ij|)
    m3  = 1 - sign3 * mean_{(a,b,c) in M3t} Re<c3m_..abc, c3d_..abc>_{q1,q2,q3 != 0} / (|.| |.|)
"""
import shutil
from pathlib import Path

P = Path("moments/image_moments.py")
s = P.read_text()
if "def _corr_init" in s:
    raise SystemExit("already patched")


def sub(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"patch point not found exactly once (your file differs from the snapshot):\n{old}")
    s = s.replace(old, new)


sub('''    M12, M3 = ("plain", "white"), ("block", "perq", "white", "wblock")''',
    '''    M12, M3 = ("plain", "white", "corr"), ("block", "perq", "white", "wblock", "corr")''')

sub('''        self.keep, self.n3 = n > 1e-12 * n.max(), n
        self.info = ""
''', '''        self.keep, self.n3 = n > 1e-12 * n.max(), n
        self.info = ""
        if "corr" in (m12, m3):
            self._corr_init(ref)
''')

sub('''    def m12(self, m, mode=None):
        mode = mode or self.mode12
''', '''    def _corr_init(self, ref):
        """Masks and data norms of the correlation loss. Defaults without masks in the data: c2 pairs |i-j| >= 2,
        all c3 triples, sign +1."""
        c2, c3 = ref["c2"], ref["c3"]
        NK, dev = c2.shape[1], c2.device
        k = torch.arange(NK, device=dev)
        self.M2 = ref["M2"].bool() if "M2" in ref else (k[:, None] - k[None]).abs() >= 2
        Q = c3.shape[0]
        qs = torch.arange(Q, device=dev) - (Q - 1) // 2
        self.B3 = (qs[:, None] != 0) & (qs[None] != 0) & ((qs[:, None] + qs[None]) != 0)        # q1, q2, q3 != 0
        self.T3 = ref["M3t"].bool() if "M3t" in ref else torch.ones(c3.shape[2:], dtype=torch.bool, device=dev)
        self.sign3 = float(ref.get("sign3", 1.0))
        self.n2d = c2.to(torch.complex128).abs().pow(2).sum(0)                                   # (NK, NK)
        self.n3d = c3.to(torch.complex128)[self.B3].abs().pow(2).sum(0)                          # (S, S, S)
        self.info += (f"corr: {int(self.M2.sum())} c2 shell pairs, {int(self.T3.sum())} c3 triples x "
                      f"{int(self.B3.sum())} blocks, sign3 {self.sign3:+.0f}; ")

    @staticmethod
    def _rho(x, d, nd):
        """Re<x, d> / (|x| |d|) summed over the leading axis (float64)."""
        x = x.to(torch.complex128)
        num = (x.conj() * d.to(torch.complex128)).real.sum(0)
        return num / (x.abs().pow(2).sum(0) * nd).clamp_min(1e-300).sqrt()

    def m12(self, m, mode=None):
        mode = mode or self.mode12
        if mode == "corr":
            rho = self._rho(m["c2"], self.ref["c2"], self.n2d)
            return (1 - rho[self.M2].mean()).to(m["c2"].real.dtype)
''')

sub('''    def m3(self, m, mode=None):
        mode = mode or self.mode3
''', '''    def m3(self, m, mode=None):
        mode = mode or self.mode3
        if mode == "corr":
            rho = self._rho(m["c3"][self.B3], self.ref["c3"][self.B3], self.n3d)
            return (1 - self.sign3 * rho[self.T3].mean()).to(m["c3"].real.dtype)
''')

sub('''    dev = im.beta.device
    return {k: torch.as_tensor(d[k], dtype=im.cdt, device=dev) for k in ("a", "c2", "c3")}''',
    '''    dev = im.beta.device
    out = {k: torch.as_tensor(d[k], dtype=im.cdt, device=dev) for k in ("a", "c2", "c3")}
    for k in ("M2", "M3t"):                                      # masks of the correlation loss (real_data export)
        if k in d:
            out[k] = torch.as_tensor(d[k], dtype=torch.bool, device=dev)
    if "sign3" in d:
        out["sign3"] = float(d["sign3"])
    return out''')

shutil.copy(P, str(P) + ".bak")
P.write_text(s)
print(f"patched {P} (backup {P}.bak)")
