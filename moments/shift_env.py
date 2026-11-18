"""Random in-plane shifts in the image moments: an exact envelope model.

If every image is shifted by an independent, isotropic Gaussian t ~ N(0, sigma^2 I_2) (independent of the pose), each
Fourier sample is multiplied by exp(2 pi i xi . t). Averaging over t multiplies the angle-domain moments by
    a        : E e^{2 pi i xi.t}                     = exp(-2 pi^2 sigma^2 kappa^2)
    second   : E e^{2 pi i (xi1 - xi2).t}            = exp(-2 pi^2 sigma^2 (k1^2 + k2^2)) exp(+4 pi^2 sigma^2 k1 k2 cos(phi1 - phi2))
    third    : E e^{2 pi i (xia + xib + xic).t}      = exp(-2 pi^2 sigma^2 (ka^2 + kb^2 + kc^2)) prod_{i<j} exp(-4 pi^2 sigma^2 ki kj cos(phi_i - phi_j)).
With e^{z cos th} = sum_n I_n(z) e^{i n th} (modified Bessel functions), these become convolutions in the in-plane
indices:
    c2'_q          = e^{-2 pi^2 s^2 (k1^2 + k2^2)} sum_n I_n(z12) c2_{q-n},                      z12 = 4 pi^2 s^2 k1 k2,
    c3'_{q1 q2}    = e^{-2 pi^2 s^2 (ka^2+kb^2+kc^2)} sum_{n1,n2,n3} I_n1(zab) I_n2(zac) I_n3(zbc) c3_{q1-n1-n2, q2+n1-n3},
                     zij = -4 pi^2 s^2 ki kj,
using c2_{-q}(k1, k2) = c2_q(k2, k1) for real images. The model's c3 is therefore evaluated on a window wider by
2 NB3 than the data's and cropped after the convolution. The whitened m12 compares C' = B^{-1} c2'.
sigma is a single nuisance parameter: fixed, or fitted at every reward evaluation by a 1-D search on m12.
"""

import numpy as np
import torch
from scipy.special import iv

from moments.image_moments import tables


class ShiftEnvelope:
    def __init__(self, im, im_wide, nb2=16, nb3=6):
        """im: ImageMoments of the data grid; im_wide: the same grid with Q3 + 2 nb3 (for the model's c3)."""
        assert im_wide.Q3 == im.Q3 + 2 * nb3
        self.im, self.nb2, self.nb3 = im, nb2, nb3
        self.k = np.asarray(im.kappa, float)
        self.k3 = self.k[im.shells3]
        beta, *_ = tables(im.L, im.LB, im.nphi, im.Q3)
        self.Binv = torch.as_tensor(np.linalg.inv(beta.T / (4 * np.pi)), dtype=im.cdt, device=im.beta.device)
        self.dev, self.cdt = im.beta.device, im.cdt
        self._cache = {}

    def _kernels(self, s):
        key = round(float(s), 6)
        if key in self._cache:
            return self._cache[key]
        k, k3, n2, n3 = self.k, self.k3, np.arange(-self.nb2, self.nb2 + 1), np.arange(-self.nb3, self.nb3 + 1)
        f = 2 * np.pi**2 * s**2
        z12 = 2 * f * np.outer(k, k)
        K2 = iv(n2[:, None, None], z12[None]) * np.exp(-f * (k[:, None] ** 2 + k[None, :] ** 2))[None]  # (n, k1, k2)
        S, D = len(k3), 2 * self.nb3
        zp = -2 * f * np.outer(k3, k3)
        Ib = iv(n3[:, None, None], zp[None])                                              # (n, i, j)
        K3 = np.zeros((2 * D + 1, 2 * D + 1, S, S, S))                                    # (d1, d2, a, b, c)
        for i1, m1 in enumerate(n3):
            for i2, m2 in enumerate(n3):
                for i3, m3 in enumerate(n3):
                    d1, d2 = m1 + m2, -m1 + m3
                    K3[d1 + D, d2 + D] += (Ib[i1][:, :, None] * Ib[i2][:, None, :] * Ib[i3][None, :, :])
        e3 = np.exp(-f * (k3[:, None, None] ** 2 + k3[None, :, None] ** 2 + k3[None, None, :] ** 2))
        K3 *= e3[None, None]
        out = (torch.as_tensor(np.exp(-f * k**2), dtype=self.cdt, device=self.dev),
               torch.as_tensor(K2, dtype=self.cdt, device=self.dev), torch.as_tensor(K3, dtype=self.cdt, device=self.dev))
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = out
        return out

    def apply12(self, a, c2, s):
        """a (NK,), c2 (L+1, NK, NK) -> a', c2', C' (C' = B^{-1} c2', for the whitened m12)."""
        if s <= 0:
            return a, c2, (self.Binv @ c2.reshape(c2.shape[0], -1)).reshape(c2.shape)
        ea, K2, _ = self._kernels(s)
        L = c2.shape[0] - 1
        full = torch.cat([c2[1:].flip(0).transpose(-1, -2), c2], 0)                         # q = -L..L
        out = torch.zeros_like(c2)
        for i, n in enumerate(range(-self.nb2, self.nb2 + 1)):
            q = torch.arange(max(0, n - L), min(L, n + L) + 1, device=c2.device)
            if len(q):
                out[q] = out[q] + K2[i] * full[q - n + L]
        C = (self.Binv @ out.reshape(L + 1, -1)).reshape(out.shape)
        return a * ea, out, C

    def apply3(self, c3w, s):
        """c3w: model c3 on the wide window (Qw, Qw, S, S, S) -> c3' on the data window (Q, Q, S, S, S)."""
        Q3, D = self.im.Q3, 2 * self.nb3
        if s <= 0:
            return c3w[D:D + 2 * Q3 + 1, D:D + 2 * Q3 + 1]
        _, _, K3 = self._kernels(s)
        Q = 2 * Q3 + 1
        out = 0
        for d1 in range(-D, D + 1):
            for d2 in range(-D, D + 1):
                # c3'(q1, q2) += K(d1, d2) c3(q1 - d1, q2 - d2): data index q -> wide index q + Q3w, Q3w = Q3 + D
                sl = c3w[D - d1:D - d1 + Q, D - d2:D - d2 + Q]
                out = out + K3[d1 + D, d2 + D] * sl
        return out

    def fit_sigma(self, a, c2, loss12, smax=3.0, iters=14):
        """sigma in [0, smax] minimizing loss12(a', c2', C') (golden-section search, no gradients)."""
        g = (np.sqrt(5) - 1) / 2
        with torch.no_grad():
            f = lambda s: float(loss12(*self.apply12(a, c2, s)))
            lo, hi = 0.0, smax
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            f1, f2 = f(x1), f(x2)
            for _ in range(iters):
                if f1 < f2:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - g * (hi - lo)
                    f1 = f(x1)
                else:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + g * (hi - lo)
                    f2 = f(x2)
            s = (lo + hi) / 2
            return s if f(s) <= f(0.0) else 0.0
