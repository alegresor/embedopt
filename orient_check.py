"""How non-uniform are the viewing directions of a RELION particle set (diagnostic: uses the depositors' poses)?

    python orient_check.py stars/10810_shiny.star

Viewing direction d = (cos rot sin tilt, sin rot sin tilt, cos tilt) (RELION ZYZ: rot, tilt fix the projection
direction, psi the in-plane angle). Reports for l = 1..8 the power P_l = sum_m |mean_j Y_lm(d_j)|^2 of the empirical
density, relative to its expectation (2l+1)/(4 pi N) for N uniform directions (~1 = uniform), and max/mean over
equal-area bins (20 in cos tilt x 10 in rot).
"""
import sys
import numpy as np
import starfile
try:                                                     # scipy >= 1.15: sph_harm_y(l, m, polar, azimuth)
    from scipy.special import sph_harm_y as _Y

    Y = lambda l, m, az, pol: _Y(l, m, pol, az)
except ImportError:                                      # older: sph_harm(m, l, azimuth, polar)
    from scipy.special import sph_harm as _Y

    Y = lambda l, m, az, pol: _Y(m, l, az, pol)

p = starfile.read(sys.argv[1])["particles"]
rot, tilt = np.deg2rad(p["rlnAngleRot"].to_numpy(float)), np.deg2rad(p["rlnAngleTilt"].to_numpy(float))
N = len(rot)
print(f"{N} particles; tilt percentiles (deg) 5/50/95: {np.percentile(np.rad2deg(tilt), [5, 50, 95]).round(1)}")
print("   l   P_l / uniform expectation")
for l in range(1, 9):
    P = sum(abs(Y(l, m, rot, tilt).mean()) ** 2 for m in range(-l, l + 1))
    print(f"   {l}   {P / ((2 * l + 1) / (4 * np.pi * N)):10.1f}")
H, _, _ = np.histogram2d(np.cos(tilt), rot % (2 * np.pi), bins=(20, 10), range=((-1, 1), (0, 2 * np.pi)))
print(f"equal-area bins: max/mean {H.max() / H.mean():.2f}, min/mean {H.min() / H.mean():.2f}, "
      f"fraction of bins below 0.25x mean {(H < 0.25 * H.mean()).mean():.2f}")
print("density in cos(tilt) bands (rows, -1..1) x rot (cols), relative to mean:")
for row in (H / H.mean()):
    print("   " + " ".join(f"{v:4.1f}" for v in row))
