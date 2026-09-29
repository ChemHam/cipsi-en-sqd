"""CN, a doublet, scanned along the C-N separation."""

SPIN = 1
COORDINATE = "R"
UNITS = "Angstrom"


def geometry(R, cfg):
    return f"C 0 0 0; N 0 0 {R}"
