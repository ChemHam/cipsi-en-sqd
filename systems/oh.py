"""OH, a doublet, scanned along the O-H separation."""

SPIN = 1
COORDINATE = "R"
UNITS = "Angstrom"


def geometry(R, cfg):
    return f"O 0 0 0; H 0 0 {R}"
