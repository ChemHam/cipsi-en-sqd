"""C2H4, a closed-shell singlet, twisted about the C-C bond.

The coordinate is the HCCH dihedral in degrees. Nothing relaxes along it:
the bond lengths and HCC angles are idealized sp2 values held fixed, so the
curve is a rigid torsional profile. C1's hydrogens stay in the xz plane;
C2's are rotated about z.
"""

import math

SPIN = 0
COORDINATE = "theta"
UNITS = "degrees"

D_CC = 1.34
D_CH = 1.08
ANGLE_HCC = 121.5


def geometry(theta, cfg):
    kw = cfg.geometry_kwargs
    d_cc = kw.get("d_cc", D_CC)
    d_ch = kw.get("d_ch", D_CH)
    a = math.radians(kw.get("angle_hcc", ANGLE_HCC))

    z1, z2 = -d_cc / 2.0, +d_cc / 2.0
    radial = d_ch * math.sin(a)
    h1_z = z1 + d_ch * math.cos(a)        # cos < 0 above 90 degrees
    h2_z = z2 - d_ch * math.cos(a)

    phi = math.radians(theta)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    return (f"C 0.0 0.0 {z1};"
            f"C 0.0 0.0 {z2};"
            f"H {+radial} {0.0} {h1_z};"
            f"H {-radial} {0.0} {h1_z};"
            f"H {+radial * cos_p} {+radial * sin_p} {h2_z};"
            f"H {-radial * cos_p} {-radial * sin_p} {h2_z}")
