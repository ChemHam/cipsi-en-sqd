"""N2, a closed-shell singlet, scanned along the N-N separation."""

SPIN = 0
COORDINATE = "R"
UNITS = "Angstrom"


def geometry(R, cfg):
    return f"N 0 0 {-R/2}; N 0 0 {R/2}"
