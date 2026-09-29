"""Bitstring and determinant-string helpers.

Extracted verbatim from the production notebook pipeline. The function bodies
are byte-identical to the versions that produced the published scans; nothing
in this file has been retyped. Deliberate changes are marked FIX and listed in
CHANGES.md.
"""

import numpy as np
from pyscf.fci import cistring

def safe_array(x):
    return np.nan_to_num(np.asarray(x, float))


def det_strings(norb, nelec):
    na, nb = nelec
    return (np.array(cistring.make_strings(range(norb), na), np.int64),
            np.array(cistring.make_strings(range(norb), nb), np.int64))


def reverse_bits(s, norb):
    result = 0
    for i in range(norb):
        if (s >> i) & 1:
            result |= (1 << (norb - 1 - i))
    return result


def fix_ci_strs(strs, norb):
    return np.array(sorted(set(reverse_bits(int(s), norb) for s in strs)), dtype=np.int64)
