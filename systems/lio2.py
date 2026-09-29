"""LiO2, a doublet in C2v.

The scan coordinate is the Li to O-O midpoint distance, not a bond length:
the O-O separation is held at the bound superoxide value throughout, so the
dissociation limit is Li + O2 at that fixed O-O distance.
"""

SPIN = 1
COORDINATE = "R"
UNITS = "Angstrom"

OO_DIST = 1.34


def geometry(R, cfg):
    d = cfg.geometry_kwargs.get("oo_dist", OO_DIST) / 2.0
    return f"Li {R} 0 0; O 0 {d} 0; O 0 {-d} 0"
